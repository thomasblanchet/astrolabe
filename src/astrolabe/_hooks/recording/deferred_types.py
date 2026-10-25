"""Read the code that a deferred type holds, and write a code as a deferred type.

A deferred type is a head class of `astrolabe.expr` with one argument,
`Deferred[Literal["<code>"], *leaves]`. The head says which methods the value
offers, as the class of polars that it stands for does: `TypedExpr` for an
expression, `TypedWhen` and `TypedChainedWhen` for a `when` chain that waits
for a `.then(...)`, `TypedThen` for a chain that has a branch, and
`TypedSelector` for a selector. A namespace such as `.str` wraps the type of
its expression, and the wrapper gives that code too.

A declared type, such as the type of a parameter `price: Price`, is the same
head with the arguments of a table. The recorder writes it as a leaf of the
code, by reference, and does not read its table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl
import polars.selectors as cs
from mypy.types import Instance, LiteralType, UnionType, get_proper_type
from polars.expr.whenthen import ChainedThen, ChainedWhen, Then, When

from astrolabe._hooks.deferred.code import Code
from astrolabe._hooks.dispatch.generic import Target
from astrolabe.expr import (
    Deferred,
    TypedAgg,
    TypedAggSelection,
    TypedChainedWhen,
    TypedColumn,
    TypedExpr,
    TypedExprArrNamespace,
    TypedExprBinNamespace,
    TypedExprCatNamespace,
    TypedExprDtNamespace,
    TypedExprListNamespace,
    TypedExprNameNamespace,
    TypedExprStrNamespace,
    TypedExprStructNamespace,
    TypedLit,
    TypedSelection,
    TypedSelector,
    TypedThen,
    TypedWhen,
)

if TYPE_CHECKING:
    from astrolabe._hooks.encoding.marker_minter import MarkerMinter
    from mypy.types import Type as MypyType


class DeferredHeads:
    """Name the head classes of a deferred type, and the head for each class of polars.

    A call of polars returns an `Expr`, a `Selector`, or a step of a `when`
    chain. The deferred type of the call takes the head of that class. The
    more specific classes come first, because a `Selector` and a `Then` are
    also an `Expr`.

    Example:
        >>> DeferredHeads().namespace_of("builtins.str") is None
        True
    """

    BY_CLASS: ClassVar[tuple[tuple[type, type], ...]] = (
        (cs.Selector, TypedSelector),
        (ChainedWhen, TypedChainedWhen),
        (When, TypedWhen),
        (ChainedThen, TypedThen),
        (Then, TypedThen),
        (pl.Expr, TypedExpr),
    )
    DECLARED: ClassVar[tuple[type, ...]] = (
        TypedExpr,
        TypedThen,
        TypedAgg,
        TypedLit,
        TypedColumn,
        TypedSelector,
        TypedSelection,
        TypedAggSelection,
    )
    NAMESPACES: ClassVar[dict[str, type]] = {
        "str": TypedExprStrNamespace,
        "dt": TypedExprDtNamespace,
        "bin": TypedExprBinNamespace,
        "list": TypedExprListNamespace,
        "arr": TypedExprArrNamespace,
        "cat": TypedExprCatNamespace,
        "name": TypedExprNameNamespace,
        "struct": TypedExprStructNamespace,
    }

    def head_for(self, typ: MypyType) -> type | None:
        """Give the head of the deferred type of a call, from the type that polars gives it.

        Args:
            typ: The type that the stub of polars gives the call.

        Returns:
            The head, or `None` where the call gives no expression.

        Example:
            >>> DeferredHeads().head_for(Instance(new_type_fixture().std_tuplei, [])) is None
            True
        """
        proper = get_proper_type(typ)
        if not isinstance(proper, Instance):
            return None
        for polars_class, head in self.BY_CLASS:
            fullname = Target.of(polars_class).fullname
            if any(info.fullname == fullname for info in proper.type.mro):
                return head
        return None

    def namespace_of(self, fullname: str) -> str | None:
        """Give the namespace that a wrapper class stands for, by the full name of the class.

        Args:
            fullname: The full name of a class.

        Returns:
            The namespace, such as `str`, or `None`.

        Example:
            >>> DeferredHeads().namespace_of(Target.of(TypedExprStrNamespace).fullname)
            'str'
        """
        for namespace, wrapper in self.NAMESPACES.items():
            if Target.of(wrapper).fullname == fullname:
                return namespace
        return None


class DeferredReader:
    """Read the code that a type holds, or the declared type that stands as a leaf.

    Example:
        >>> DeferredReader().code(Instance(new_type_fixture().std_tuplei, [])) is None
        True
    """

    def code(self, typ: MypyType) -> Code[MypyType] | None:
        """Give the code that a deferred type holds.

        Args:
            typ: The type.

        Returns:
            The code, or `None` where the type holds no code.

        Example:
            >>> DeferredReader().code(new_type_fixture().anyt) is None
            True
        """
        proper = get_proper_type(typ)
        if not isinstance(proper, Instance) or len(proper.args) != 1:
            return None
        deferred = get_proper_type(proper.args[0])
        if (
            not isinstance(deferred, Instance)
            or deferred.type.fullname != Target.of(Deferred).fullname
        ):
            return None
        text = get_proper_type(deferred.args[0]) if deferred.args else None
        if not isinstance(text, LiteralType) or not isinstance(text.value, str):
            return None
        return Code(text.value, tuple(deferred.args[1:]))

    def declared(self, typ: MypyType) -> bool:
        """Say whether a type is a declared type of an expression, which a code reads as a leaf.

        Args:
            typ: The type.

        Returns:
            Whether it is a head of `astrolabe.expr` that holds no code.

        Example:
            >>> DeferredReader().declared(new_type_fixture().anyt)
            False
        """
        proper = get_proper_type(typ)
        if not isinstance(proper, Instance) or self.code(proper) is not None:
            return False
        names = {Target.of(head).fullname for head in DeferredHeads.DECLARED}
        return proper.type.fullname in names

    def unwrapped(self, typ: MypyType) -> MypyType:
        """Give the type that a wrapper of a namespace wraps, or the type itself.

        Args:
            typ: The type.

        Returns:
            The type of the expression.

        Example:
            >>> fixture = new_type_fixture()
            >>> DeferredReader().unwrapped(fixture.anyt) is fixture.anyt
            True
        """
        proper = get_proper_type(typ)
        if isinstance(proper, Instance) and DeferredHeads().namespace_of(proper.type.fullname):
            return proper.args[0]
        return typ

    def members(self, typ: MypyType) -> list[MypyType]:
        """Give the members of a union, or the type itself.

        Args:
            typ: The type.

        Returns:
            The members.

        Example:
            >>> fixture = new_type_fixture()
            >>> DeferredReader().members(fixture.anyt) == [fixture.anyt]
            True
        """
        proper = get_proper_type(typ)
        if isinstance(proper, UnionType):
            return list(proper.items)
        return [typ]


class DeferredMinter:
    """Write a code as a deferred type of one head.

    Example:
        >>> fixture = new_type_fixture()
        >>> minter = DeferredMinter(new_marker_minter(fixture))
        >>> minter.minted(TypedExpr, Code("pl.col('a')"))
        astrolabe.expr.TypedExpr[astrolabe.expr.Deferred[Literal["pl.col('a')"]]]
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the minter of marker types for the phase of the check.

        Args:
            minter: The minter.

        Example:
            >>> DeferredMinter(new_marker_minter(new_type_fixture()))._minter is not None
            True
        """
        self._minter = minter

    def minted(self, head: type, code: Code[MypyType]) -> Instance | None:
        """Give the deferred type of a code.

        Args:
            head: The head class.
            code: The code.

        Returns:
            The type, or `None` where mypy did not load the module of the head.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = DeferredMinter(new_marker_minter(fixture))
            >>> minter.minted(TypedSelector, Code("cs.numeric()"))
            astrolabe.expr.TypedSelector[astrolabe.expr.Deferred[Literal['cs.numeric()']]]
        """
        deferred = self._minter.marker(Deferred, [self._minter.string(code.text), *code.leaves])
        if deferred is None:  # pragma: no cover
            # The plugin adds the module of the markers to every build.
            return None
        return self._minter.marker(head, [deferred])

    def wrapped(self, namespace: str, typ: MypyType) -> Instance | None:
        """Give the wrapper of a namespace over the type of an expression.

        Args:
            namespace: The namespace, such as `str`.
            typ: The type of the expression.

        Returns:
            The wrapper, or `None` where mypy did not load its module.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = DeferredMinter(new_marker_minter(fixture))
            >>> minter.wrapped("str", fixture.anyt)
            astrolabe.expr.TypedExprStrNamespace[Any]
        """
        return self._minter.marker(DeferredHeads.NAMESPACES[namespace], [typ])
