"""Read the region of a declared read: the data types at which the column is read.

In `column("price", pl.Float64)` the second argument is a region. The
checker compares expressions by kind, that is, by data type with the
parameters left out: `Datetime` for `Datetime("us")`. A region is
therefore a set of kinds. The user can write it as one data type, a
polars family such as `NumericType`, a set of data types, `EveryDType`,
or `Unchecked`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mypy.nodes import SetExpr

from astrolabe._hooks.calls.dtype_constructor_readers import (
    SemanticAnalyzerLiteralTypeBuilder,
)
from astrolabe._hooks.calls.dtype_node import DtypeNodeReader
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeReads
from astrolabe._hooks.expression.kind import KindRoster
from astrolabe._hooks.expression.lifting import UNCHECKED
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe.expr import EveryDType

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import DynamicClassDefContext
    from mypy.types import Instance


@dataclass(frozen=True)
class ParsedRegion:
    """Hold one region as the user wrote it.

    Most code needs only the set of kinds. The declaration of a single
    column also needs the exact data type, with its parameters, so that
    its type can describe a field of a `Struct`. This value keeps both.

    Attributes:
        kinds: The kinds that the region holds. Empty for `Unchecked`.
        unchecked: True for `Unchecked`, the region that asks for no
            check of the data type at all.
        dtype: The data type with its parameters, if the user wrote the
            region as one data type. Otherwise None.

    Example:
        >>> ParsedRegion(frozenset({"Int64"})).kinds
        frozenset({'Int64'})
    """

    kinds: frozenset[str]
    unchecked: bool = False
    dtype: Instance | None = None


class RegionParser:
    """Read the node of a region into the set of kinds that it stands for.

    Every reader of a declaration that takes a region uses this class,
    so that all of them accept the same five forms. A data type with
    parameters, for example `pl.Datetime("us")`, stands for its kind. A
    check reads the parameters from the column of the frame, and not
    from the region.

    The code reads the syntax during semantic analysis, the mypy phase
    that resolves names before any expression has a type.
    """

    def __init__(self, ctx: DynamicClassDefContext) -> None:
        """Keep the context of the declaration, and build the readers that this class uses.

        Args:
            ctx: The mypy context of the declaration. The data type
                reader resolves names through it.
        """
        self._ctx = ctx
        self._dtypes = DtypeNodeReader(SemanticAnalyzerLiteralTypeBuilder())
        self._roster = KindRoster()

    def parsed(self, node: Expression) -> ParsedRegion | None:
        """Read one region, in whichever of the five forms the user wrote it.

        Args:
            node: The node of the region.

        Returns:
            The region, or None if the node has none of the five forms.
        """
        if isinstance(node, SetExpr):
            return self._set(node)
        named = self._named(self._dtypes.fullname(node))
        if named is not None:
            return named
        return self._one_dtype(node)

    def _named(self, fullname: str | None) -> ParsedRegion | None:
        """Read a class that stands for several kinds: `EveryDType` or a polars family.

        A polars family class, for example `NumericType`, stands for the kinds in it.

        Args:
            fullname: The fullname of the class that the node names, or
                None if the node names no class.

        Returns:
            Every kind for `EveryDType`, the kinds of a family, or None
            for any other class.

        Example:
            >>> parser = RegionParser.__new__(RegionParser)
            >>> parser._roster = KindRoster()
            >>> len(parser._named("astrolabe.expr.EveryDType").kinds) > 10
            True
            >>> "Float64" in parser._named("polars.datatypes.classes.NumericType").kinds
            True
            >>> parser._named("polars.datatypes.classes.Int64") is None
            True
        """
        if fullname == Target.of(EveryDType).fullname:
            return ParsedRegion(frozenset(self._roster.names()))
        for base in self._roster.group_bases():
            if Target.of(base).fullname == fullname:
                return ParsedRegion(frozenset(self._roster.family(base)))
        return None

    @propagates(None)
    def _set(self, node: SetExpr) -> ParsedRegion | None:
        """Read a set of data types, for example `{pl.Int64, pl.Float64}`, as one region.

        Args:
            node: The set written in the code.

        Returns:
            The region with the kind of every item, or None if an item
            is not a data type.
        """
        kinds: frozenset[str] = frozenset()
        for item in node.items:
            kinds |= not_none(self._one_dtype(item)).kinds
        return ParsedRegion(kinds)

    @propagates(None)
    def _one_dtype(self, node: Expression) -> ParsedRegion | None:
        """Read one data type as its kind, and keep the data type with its parameters.

        Args:
            node: The node of the data type.

        Returns:
            The region, with `unchecked` set for `Unchecked`, or None if
            the node names no data type.
        """
        instance = not_none(self._dtypes.instance(self._ctx, node))
        rendering = not_none(ConstDtypeReads().read(instance))
        kind = rendering.split("(", maxsplit=1)[0]
        if kind == UNCHECKED:
            return ParsedRegion(frozenset(), unchecked=True, dtype=instance)
        return ParsedRegion(frozenset({kind}), dtype=instance)
