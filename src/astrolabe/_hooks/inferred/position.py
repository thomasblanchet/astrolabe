"""Report an inferred return type that a user writes outside the return type of a function.

`InferredExpr`, `InferredDataFrame`, `InferredLazyFrame` and
`InferredRemainder` mean something only as the return type of a function:
the checker replaces them at each call with the types of the values that
the body returns. Anywhere else, for example on a parameter or a
variable, nothing replaces them. The first three are empty classes, so
mypy refuses such a use with errors that do not name the cause.
`InferredRemainder` sits inside a real frame type, so mypy accepts the
use, and each operation on the frame meets a remainder that it cannot
read. The type-analysis step cannot see where a type stands, so the
checker reports these uses where it meets them: at a call of a function
whose signature holds one, and at a method call on a frame whose
remainder is `InferredRemainder`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy.types import AnyType, Instance, TupleType, TypeOfAny, UnionType, get_proper_type

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.errors import InferMisuseError
from astrolabe._hooks.inferred.families import InferredRemainderReturn, family_for
from astrolabe.declare import InferredDataFrame, InferredExpr, InferredLazyFrame, InferredRemainder

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.plugin import FunctionSigContext, MethodContext
    from mypy.types import CallableType
    from mypy.types import Type as MypyType


class InferredMarkers:
    """Find the inferred return types that a type holds, at any depth.

    A marker can stand at the top of a type, as in `df: InferredDataFrame`,
    or inside it, as in `list[InferredDataFrame]` or
    `TypedDataFrame[Sales, InferredRemainder]`. The walk reads the type
    arguments of each class, and the items of each union and each tuple.

    Example:
        >>> fixture = new_type_fixture()
        >>> InferredMarkers().found(int_dtype(fixture))
        []
    """

    _NAMES: ClassVar[dict[str, str]] = {
        Target.of(marker).fullname: marker.__name__
        for marker in (InferredExpr, InferredDataFrame, InferredLazyFrame, InferredRemainder)
    }

    def misplaced(self, signature: CallableType) -> list[str]:
        """Give the name of each marker that a signature holds outside its own return type.

        A return type that is a marker, or a frame type whose remainder is
        `InferredRemainder`, is the one place where a marker belongs. Any
        other marker in the return type, and any marker in a parameter
        type, is misplaced.

        Args:
            signature: The declared signature of the called function.

        Returns:
            The short name of each misplaced marker, once each, in order.
        """
        found: list[str] = []
        for declared in signature.arg_types:
            found.extend(self.found(declared))
        if family_for(signature.ret_type) is None:
            found.extend(self.found(signature.ret_type))
        return list(dict.fromkeys(found))

    def found(self, typ: MypyType) -> list[str]:
        """Give the short name of each marker inside one type.

        Args:
            typ: The type to read.

        Returns:
            The name of each marker, in the order of the walk.
        """
        names: list[str] = []
        seen: set[str] = set()
        pending = [typ]
        while pending:
            item = pending.pop()
            names.extend(self._read_once(item, seen, pending))
        return names

    def _read_once(self, item: MypyType, seen: set[str], pending: list[MypyType]) -> list[str]:
        """Read one type of the walk, unless the walk has read it before.

        A recursive type alias, for example `Json = int | list[Json]`, holds
        itself. Each expansion gives the alias again, so a walk with no
        memory never ends. The walk keeps the printed form of each type that
        it has read, and skips a type that it meets again.

        Args:
            item: The type to read.
            seen: The printed form of each type read so far. This step adds
                to it.
            pending: The types still to read. This step adds the parts of
                `item` to it: the type arguments of a class, or the items
                of a union or a tuple.

        Returns:
            The name of the marker that `item` is, or nothing.

        Example:
            >>> fixture = new_type_fixture()
            >>> InferredMarkers()._read_once(int_dtype(fixture), {str(int_dtype(fixture))}, [])
            []
        """
        key = str(item)
        if key in seen:
            return []
        seen.add(key)
        proper = get_proper_type(item)
        if isinstance(proper, (UnionType, TupleType)):
            pending.extend(proper.items)
        if not isinstance(proper, Instance):
            return []
        pending.extend(proper.args)
        name = self._NAMES.get(proper.type.fullname)
        return [] if name is None else [name]


class MisplacedInferredHook:
    """Report, at a call, each inferred return type that the signature holds out of place.

    The fit of a call runs through a signature hook, so this check takes
    the same form. A parameter type that holds a marker accepts no real
    argument, so mypy would add an error of its own for each argument.
    The check gives such a parameter the type `Any`, so that the call
    reports one error for its one fault.
    """

    def __init__(self, names: list[str]) -> None:
        """Keep the names of the misplaced markers of the function.

        Args:
            names: The short name of each misplaced marker.
        """
        self._names = names

    def __call__(self, ctx: FunctionSigContext) -> CallableType:
        """Report each misplaced marker at the call.

        Args:
            ctx: The mypy context of the signature at the call.

        Returns:
            The signature, with each parameter type that holds a marker set
            to `Any`.
        """
        for name in self._names:
            InferMisuseError.wrong_position(name).fail(ctx)
        signature = ctx.default_signature
        markers = InferredMarkers()
        anything = AnyType(TypeOfAny.from_error)
        arg_types = [anything if markers.found(typ) else typ for typ in signature.arg_types]
        return signature.copy_modified(arg_types=arg_types)


class InferredRemainderGuard:
    """Report a method call on a frame whose remainder is `InferredRemainder`.

    The hook of a frame method reads the remainder of its receiver.
    `InferredRemainder` is no remainder: it marks a return type that the
    checker fills in at each call. A frame with it is a type written out
    of place, so the guard reports that in place of the result of the
    method, and the call keeps the type that mypy gives it.
    """

    def __init__(self, hook: Callable[[MethodContext], MypyType]) -> None:
        """Keep the hook of the method.

        Args:
            hook: The hook that types the method call.
        """
        self._hook = hook

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Report a receiver with an inferred remainder, or run the hook of the method.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The type that mypy gives the call after a report, else the type
            that the hook of the method gives.
        """
        if InferredRemainderReturn().declared(ctx.type) is None:
            return self._hook(ctx)
        InferMisuseError.wrong_position(InferredRemainder.__name__).fail(ctx)
        return ctx.default_return_type
