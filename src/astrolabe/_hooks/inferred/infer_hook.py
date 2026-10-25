"""Check each `infer(...)` call, and record the type of the value that it marks.

`infer(x)` marks one value that a function returns. The function
declares its return type as `InferredExpr`, `InferredDataFrame`,
`InferredLazyFrame`, or a frame type whose remainder is
`InferredRemainder`, and gets the union of the marked types at each
call. The hook here is the half of that work that runs while mypy
checks the body of the function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mypy.nodes import FuncDef, TypeInfo
from mypy.types import CallableType, Instance

from mypy.types import get_proper_type

from astrolabe._hooks.domain.declared_columns import DeclaredColumnsFit
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.errors import InferMisuseError
from astrolabe._hooks.inferred.cached_variants import CachedVariants
from astrolabe._hooks.inferred.families import InferredRemainderReturn, family_for
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext, FunctionSigContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup
    from astrolabe._hooks.domain.unproven_gate import UnprovenGate
    from astrolabe._hooks.inferred.accumulator import InferAccumulator
    from astrolabe._hooks.inferred.families import PlaceholderFamily


@dataclass(frozen=True)
class _ResolvedCall:
    """Hold the function around one `infer(...)` call, and the placeholder that it returns.

    Attributes:
        enclosing: The function whose body holds the call.
        family: The placeholder that the function declares as its return
            type, with the classes that go with it.
        ret_type: The declared return type of the function.
    """

    enclosing: FuncDef
    family: PlaceholderFamily
    ret_type: MypyType


class InferHook:
    """Record the type of the value that one `infer(...)` call marks, and hide it from mypy.

    mypy calls this hook for each call of `infer` while it checks the
    body of a function. The hook writes the type of the argument into
    an `InferAccumulator`, keyed by that function. Code that types a call
    of the function reads it from there.

    The hook then gives the call the type of the placeholder, and not
    the type of its argument. mypy checks each `return` statement
    against the declared return type. A `TypedExpr[...]` is not a
    subtype of `InferredExpr`, so `return infer(x)` with the real type
    fails with `[return-value]`. With the placeholder type, the check
    passes.

    A return type such as `TypedDataFrame[WithTotal, InferredRemainder]`
    is a frame type, so the call gets that type itself. Its declared
    columns are a promise to each caller, so the hook also checks that the
    value holds each of them, with a data type that agrees.

    If the call is not inside a function that returns a placeholder, the
    hook reports an `[infer-misuse]` error. This covers a call at module
    level, where no function holds it.
    """

    def __init__(
        self,
        accumulator: InferAccumulator,
        plugin: PluginLookup,
        gate: UnprovenGate,
    ) -> None:
        """Keep the shared record, the plugin that looks up the placeholder classes, and the gate.

        Args:
            accumulator: The record that keeps the type of each marked
                value until a call of its function reads it.
            plugin: The mypy plugin instance. It finds the `TypeInfo` of
                the placeholder class that the call gets as its type.
            gate: The gate that decides whether a declared column that
                only the remainder of the value can hold gets an error.
        """
        self._accumulator = accumulator
        self._plugin = plugin
        self._columns = DeclaredColumnsFit(gate, "fit")

    def __call__(self, ctx: FunctionContext) -> MypyType:
        """Record the marked type and give the call the placeholder type, or report a misuse.

        A call with no argument records nothing, and still gets the
        placeholder type.

        Args:
            ctx: The mypy context of the `infer(...)` call.

        Returns:
            An instance of the placeholder that the enclosing function
            returns. After a misuse, the type that mypy worked out for
            the call.
        """
        resolved = self._enclosing_placeholder(ctx)
        if resolved is None:
            InferMisuseError.outside_inferred_function().fail(ctx)
            return ctx.default_return_type
        if ctx.arg_types and ctx.arg_types[0]:
            arg_type = ctx.arg_types[0][0]
            self._accumulator.record(resolved.enclosing, id(ctx.context), arg_type)
            self._cache(ctx, resolved.enclosing)
            self._check_declared(ctx, resolved.ret_type, arg_type)
        return self._placeholder_type(ctx, resolved)

    @propagates(None)
    def _check_declared(self, ctx: FunctionContext, ret_type: MypyType, arg_type: MypyType) -> None:
        """Check a returned frame against the declared columns of the return type.

        Only a return type whose remainder is `InferredRemainder` declares
        columns. A value that is not a typed frame is checked at each call
        of the function, with the other variants.

        Args:
            ctx: The mypy context of the `infer(...)` call.
            ret_type: The declared return type of the function.
            arg_type: The type of the returned value.
        """
        declared = not_none(InferredRemainderReturn().declared(ret_type))
        actual = not_none(Frame.from_proper_type(get_proper_type(arg_type)))
        columns = not_none(declared.known_columns).columns
        self._columns.check(ctx, columns, not_none(actual.known_columns))

    def _placeholder_type(self, ctx: FunctionContext, resolved: _ResolvedCall) -> MypyType:
        """Give the type of the call: the declared frame type, or the placeholder class.

        Args:
            ctx: The mypy context of the `infer(...)` call.
            resolved: The function around the call, and its placeholder.

        Returns:
            The declared return type if it is a frame type with an
            inferred remainder, else an instance of the placeholder class.
            The type that mypy worked out for the call if the placeholder
            class is not loaded.
        """
        if InferredRemainderReturn().declared(resolved.ret_type) is not None:
            return resolved.ret_type
        info = self._plugin.lookup_fully_qualified(resolved.family.placeholder_target.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # The enclosing function names the placeholder class in its
            # return type, so mypy has loaded that class before it checks
            # any `infer()` call in the body.
            return ctx.default_return_type
        return Instance(info.node, [])

    def _cache(self, ctx: FunctionContext, enclosing: FuncDef) -> None:
        """Write the live variants of the function beside it, for a later run of mypy.

        Args:
            ctx: The mypy context of the `infer(...)` call.
            enclosing: The function whose body holds the call.
        """
        variants = self._accumulator.live_variants(enclosing)
        module_names = ctx.api.tree.names  # type: ignore[attr-defined]
        CachedVariants().write(module_names, enclosing, variants, ctx.api)

    def _enclosing_placeholder(self, ctx: FunctionContext) -> _ResolvedCall | None:
        """Find the function around the call, and the placeholder that it returns.

        The function is the outermost function whose body mypy checks
        now. The placeholder is its declared return type, with the
        classes that go with it.

        Args:
            ctx: The mypy context of the `infer(...)` call.

        Returns:
            The function and its placeholder, or None if no function
            holds the call, for example at module level, or the function
            returns another type.
        """
        enclosing = ctx.api.scope.top_level_function()  # type: ignore[attr-defined]
        if not isinstance(enclosing, FuncDef):
            return None
        signature = enclosing.type
        if not isinstance(signature, CallableType):  # pragma: no cover
            # mypy checks the body of one concrete function at a time,
            # and that function always has a callable type. The type
            # `Overloaded` belongs to a whole `@overload` group, and the
            # scope never gives the group.
            return None
        family = family_for(signature.ret_type)
        if family is None:
            return None
        return _ResolvedCall(enclosing, family, signature.ret_type)


class InferSignatureHook:
    """Give each `infer(...)` call the signature of its own argument, with no type variable.

    `infer` is declared as `infer[T](x: T) -> T`. mypy solves `T` from the
    type that the `return` statement expects as well as from the
    argument. With a return type such as `TypedDataFrame[WithTotal,
    InferredRemainder]`, the expected type is a frame type of the same
    class as the value, so mypy compares the value with it and reports a
    second error beside the check of the declared columns. With this
    signature the call has nothing to solve, and its return type is the
    type of its argument until `InferHook` gives the call its own type.
    """

    def __call__(self, ctx: FunctionSigContext) -> CallableType:
        """Give the signature of one `infer(...)` call.

        Args:
            ctx: The mypy context of the signature at the call.

        Returns:
            The signature `(x: A) -> A`, where `A` is the type of the
            argument, or the declared signature for a call with no single
            argument.
        """
        if len(ctx.args) != 1 or len(ctx.args[0]) != 1:
            return ctx.default_signature
        actual = ctx.api.get_expression_type(ctx.args[0][0])
        return ctx.default_signature.copy_modified(
            arg_types=[actual],
            ret_type=actual,
            variables=[],
        )
