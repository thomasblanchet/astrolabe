"""Find the hook for a call from the full name of the class or function that it calls.

mypy asks a plugin for a hook by full name, for example
`polars.dataframe.frame.DataFrame.select`, once for each call that it
checks. The plugin answers with the hook registered for that name, or
with None. This module holds the two parts of that lookup. `Target`
names what a hook handles. `HookDispatcher[T, R]` maps each target to
its hook.

Each kind of mypy hook has its own class in this package, and each
class owns one dispatcher. `T` is the context type that the kind of
hook receives, for example `MethodContext` for a method hook. `R` is
what the hook returns: a mypy type for each kind of hook that computes
a type, and `None` for a dynamic-class hook, which declares a symbol
instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

from mypy.nodes import CallExpr, RefExpr
from mypy.types import Instance

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeGuard

    from mypy.nodes import Expression
    from mypy.types import ProperType


class Hook[T, R](Protocol):
    """The call shape that every hook class of the plugin has.

    A hook is an object that mypy calls with the context of one call,
    and that returns what that kind of hook returns. A dispatcher
    stores hooks of one kind by this shape.
    """

    def __call__(self, ctx: T, /) -> R: ...


class Target:
    """Name the class, function or method that a hook handles, by its mypy full name.

    mypy finds a hook by a full name such as `polars.functions.col`. A
    name typed by hand does not follow a rename in polars, and the hook
    then stops firing with no error. Build a `Target` from the real
    Python object with `Target.of` instead, so that the name always
    matches the definition.

    The plain constructor takes a full name directly. It serves the one
    case with no object to build from: the full name that mypy passes
    when it asks for a hook. Two targets compare and hash equal when
    they name the same full name. A dispatch table can therefore
    register with `Target.of(SomeClass)` and look up with
    `Target(observed_fullname)`.

    Code that reads the abstract syntax tree (AST) or a mypy type also
    uses a target, to ask whether a call or a type names this class or
    function.

    Example:
        >>> Target.of(int).fullname
        'builtins.int'
        >>> Target.of(int, "bit_length").fullname
        'builtins.int.bit_length'
        >>> Target("builtins.int").fullname
        'builtins.int'
        >>> {Target.of(int): "found"}[Target("builtins.int")]
        'found'
        >>> Target.of(int) == Target.of(int)
        True
    """

    def __init__(self, fullname: str) -> None:
        """Name a target by a full name that mypy gives.

        Args:
            fullname: The full name that this target names.
        """
        self.fullname: Final = fullname

    @classmethod
    def of(cls, owner: type | Callable[..., object], attr: str | None = None) -> Target:
        """Build a target from a real Python object, not from a name typed by hand.

        Args:
            owner: The class, function or method that the target names.
            attr: A method or attribute of `owner`, if the target names
                it and not `owner` itself.

        Returns:
            A target that names the full name of `owner`, or of
            `owner.attr`.

        Example: see class docstring.
        """
        base = f"{owner.__module__}.{owner.__qualname__}"
        return cls(f"{base}.{attr}" if attr else base)

    def __eq__(self, other: object) -> bool:
        """Compare two targets by full name.

        Returns:
            True if `other` is a `Target` with the same full name.

        Example:
            >>> Target.of(int) == Target.of(int)
            True
            >>> Target.of(int) == "builtins.int"
            False
        """
        if not isinstance(other, Target):
            return NotImplemented
        return self.fullname == other.fullname

    def __hash__(self) -> int:
        """Hash by full name, so that equal targets have equal hashes.

        Returns:
            `hash(self.fullname)`.
        """
        return hash(self.fullname)

    def matches(self, proper: ProperType) -> TypeGuard[Instance]:
        """Answer whether a type is an instance of exactly this class.

        Args:
            proper: The type to read, with its aliases expanded.

        Returns:
            True if `proper` is an `Instance` of the class with this
            full name. A subclass does not count, see `is_base_of`.

        Example:
            >>> import polars as pl
            >>> fixture = new_type_fixture()
            >>> Target.of(pl.Int64).matches(int_dtype(fixture))
            True
            >>> Target.of(int).matches(int_dtype(fixture))
            False
        """
        if not isinstance(proper, Instance):
            return False
        return proper.type.fullname == self.fullname

    def is_base_of(self, proper: ProperType) -> TypeGuard[Instance]:
        """Answer whether a type is an instance of this class or of a subclass.

        Args:
            proper: The type to read, with its aliases expanded.

        Returns:
            True if `proper` is an `Instance` whose class is this class
            or has it as a base.

        Example:
            >>> fixture = new_type_fixture()
            >>> Target.of(object).is_base_of(int_dtype(fixture))
            True
            >>> Target.of(str).is_base_of(int_dtype(fixture))
            False
            >>> from mypy.types import UninhabitedType
            >>> Target.of(object).is_base_of(UninhabitedType())
            False
        """
        if not isinstance(proper, Instance):
            return False
        return proper.type.has_base(self.fullname)

    def matches_call(self, node: Expression) -> TypeGuard[CallExpr]:
        """Answer whether an AST node is a call of this target.

        Args:
            node: The AST node to read.

        Returns:
            True if `node` is a call of the function or class with this
            full name. The call can use an attribute access, as in
            `pl.Series(...)`, or a name imported directly, as in
            `Series(...)` after `from polars import Series`.

        Example:
            >>> import polars as pl
            >>> Target.of(pl.Series).matches_call(series_call_node("price"))
            True
            >>> Target.of(pl.Int64).matches_call(series_call_node("price"))
            False
            >>> Target.of(pl.Series).matches_call(int_literal_node(5))
            False
        """
        if not (isinstance(node, CallExpr) and isinstance(node.callee, RefExpr)):
            return False
        return Target(node.callee.fullname) == self


class HookDispatcher[T, R]:
    """Hold the hooks of one kind, and find the one for a full name.

    mypy asks the plugin for a hook once for each call, by full name.
    Each class that registers the hooks of one kind fills one
    dispatcher, and the plugin answers mypy through it. A name with no
    hook gives None, and mypy then uses its own type.

    Example:
        >>> dispatcher = HookDispatcher[int, int]()
        >>> dispatcher.register(Target.of(int), lambda ctx: ctx)
        >>> dispatcher.get_hook("builtins.int") is not None
        True
        >>> dispatcher.get_hook("builtins.str") is None
        True
    """

    def __init__(self) -> None:
        """Start with no hook registered."""
        self._hooks: dict[Target, Hook[T, R]] = {}

    def register(self, target: Target, hook: Hook[T, R]) -> None:
        """Register a hook for one target.

        Args:
            target: The class, function, method or attribute that the
                hook handles.
            hook: The hook that mypy calls for `target`.
        """
        self._hooks[target] = hook

    def get_hook(self, fullname: str) -> Hook[T, R] | None:
        """Find the hook registered for a full name.

        Args:
            fullname: The full name that mypy gives for the call.

        Returns:
            The registered hook, or None if no hook is registered for
            the name.
        """
        return self._hooks.get(Target(fullname))
