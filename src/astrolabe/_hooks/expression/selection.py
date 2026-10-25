"""The declared type of a selection, and what it gives when a frame applies it.

A selection is a typed selector expression, for example
`cs.numeric()` or `cs.numeric().abs().name.suffix("_abs")`. It starts
from a selector, not from named columns, so it can give many output
columns: one for each column that the selector picks. Its type holds
the selector, one table that the checker reads once for each picked
column, a rename (a prefix, a suffix, or a fixed name), and the
character, which says how the expression counts rows.

A frame applies a selection in two parts. For the declared columns
(the columns that the frame's type names), the checker keeps each
column that the selector picks, reads the table at its data type and
renames it. An open frame also has a remainder: the columns that its
type does not name. For the remainder, the checker gives one effect:
keep what the selector picks, give the kept columns a new data type
where the table says so, then rename them. This module is part of the
engine, which knows nothing about mypy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.expression.character import ELEMENTWISE
from astrolabe._hooks.expression.context import ContextRule
from astrolabe._hooks.expression.kind import KindRoster
from astrolabe._hooks.expression.lifting import UNCHECKED, Collapse
from astrolabe._hooks.expression.materialize import Materializer
from astrolabe._hooks.expression.outcome import Axis, Outcome
from astrolabe._hooks.expression.rendering import Rendered
from astrolabe._hooks.expression.table import Table
from astrolabe._hooks.remainder.type_set import TypeSet

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astrolabe._hooks.expression.character import Character
    from astrolabe._hooks.remainder.record import Record
    from astrolabe._hooks.remainder.selector import Selector


@dataclass(frozen=True)
class Affix:
    """The rename that a selection applies to each picked column.

    `.name.prefix(...)` and `.name.suffix(...)` add text to each name.
    Either part can be empty. A fixed name replaces every name. Polars
    gives a fixed name where an expression is the left operand of a
    selection, for example `pl.col("a") + cs.numeric()`: every output
    then takes the name `a`. Two picked columns then have the same
    output name, and polars raises.

    Example:
        >>> Affix("p_", "_s").applied("x")
        'p_x_s'
        >>> Affix().is_identity()
        True
    """

    prefix: str = ""
    suffix: str = ""
    fixed: str | None = None

    def applied(self, name: str) -> str:
        """Give the output name of one picked column.

        Args:
            name: The name of the picked column.

        Returns:
            The output name.

        Example:
            >>> Affix(suffix="_r").applied("bid")
            'bid_r'
            >>> Affix(fixed="a").applied("bid")
            'a'
        """
        if self.fixed is not None:
            return self.fixed
        return self.prefix + name + self.suffix

    def is_identity(self) -> bool:
        """Answer whether the rename leaves every name as it is.

        Returns:
            True where the prefix and the suffix are empty and no name
            is fixed.

        Example:
            >>> Affix(prefix="a").is_identity()
            False
        """
        return not self.prefix and not self.suffix and self.fixed is None


@dataclass(frozen=True)
class SelectionType:
    """The type of a selection expression.

    A declaration such as `typed_selection(cs.numeric(), gives=same)`
    gives this value, and code that applies a declared selection to a
    frame reads it. The map has one read under the empty name `""`,
    which stands for each picked column in turn. Its cells cover the
    kinds of the picked column.

    Attributes:
        selector: The rule that says which columns the selection picks.
        table: The map for one picked column, over the read `""`.
        affix: The rename.
        character: How the selection counts rows.

    Example:
        >>> from astrolabe._hooks.remainder.selector import OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> numeric = SelectionType.identity(OfType(Family.numeric()))
        >>> numeric.is_identity()
        True
    """

    AXIS: ClassVar[str] = ""

    selector: Selector
    table: Table
    affix: Affix
    character: Character

    @classmethod
    def identity(cls, selector: Selector) -> SelectionType:
        """Give the type of a selector with no operation: each picked column unchanged.

        The table covers only the kinds that the selector can pick.
        `cs.numeric()` never picks a `String` column, so its table has
        no cell at `String`.

        Args:
            selector: The selector.

        Returns:
            The selection that gives each picked column back, with no
            rename.

        Example:
            >>> from astrolabe._hooks.remainder.absence import Always
            >>> from astrolabe._hooks.remainder.selector import NameTest, OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> SelectionType.identity(NameTest(Always())).affix
            Affix(prefix='', suffix='', fixed=None)
            >>> numeric = SelectionType.identity(OfType(Family.numeric()))
            >>> "String" in numeric.table.projection("")
            False
        """
        kinds = cls.matched_kinds(selector)
        cells: dict[tuple[str, ...], Outcome] = {(kind,): Axis(cls.AXIS) for kind in kinds}
        table = Table.from_cells((cls.AXIS,), cells)
        return cls(selector, table, Affix(), ELEMENTWISE)

    @classmethod
    def matched_kinds(cls, selector: Selector) -> frozenset[str]:
        """Give the kinds of column that a selector can pick.

        Args:
            selector: The selector.

        Returns:
            Every kind that polars defines and that the selector can
            pick.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> "Int64" in SelectionType.matched_kinds(OfType(Family.numeric()))
            True
        """
        return selector.surviving_types(TypeSet.anything()).kinds()

    def is_identity(self) -> bool:
        """Answer whether this is a selector with no operation applied yet.

        Returns:
            True where the selection gives each picked column back,
            with no rename, row by row.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> pure = SelectionType.identity(OfType(Family.numeric()))
            >>> pure.is_identity()
            True
            >>> SelectionType(pure.selector, pure.table, Affix("p"), ELEMENTWISE).is_identity()
            False
        """
        identity = SelectionType.identity(self.selector)
        return self == identity


@dataclass(frozen=True)
class RemainderEffect:
    """What a selection does to the remainder of an open frame.

    The remainder is the set of columns that the frame's type does not
    name, so the checker cannot list them. It describes the result as
    three steps instead. Code that types the frame operation applies
    them to the remainder of the result type.

    Attributes:
        selector: The rule that says which columns stay.
        retype: The data type that every kept column takes, except the
            columns of the kinds that `retypes` lists, or None where each of
            those columns keeps its own.
        affix: The rename, which applies last.
        retypes: Where the kinds of the kept columns take different data
            types: each other data type with the kinds whose columns take it.

    Example:
        >>> from astrolabe._hooks.remainder.selector import OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> RemainderEffect(OfType(Family.numeric()), "String", Affix()).retype
        'String'
    """

    selector: Selector
    retype: str | None
    affix: Affix
    retypes: tuple[tuple[str, frozenset[str]], ...] = ()

    def changes(self) -> bool:
        """Say whether the effect gives a kept column another data type.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> RemainderEffect(OfType(Family.numeric()), None, Affix()).changes()
            False
        """
        return self.retype is not None or bool(self.retypes)

    def answers(self) -> frozenset[str]:
        """Give each data type that the effect gives a kept column.

        Returns:
            The data types.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> retypes = (("Int16", frozenset({"Int8"})),)
            >>> RemainderEffect(OfType(Family.numeric()), None, Affix(), retypes).answers()
            frozenset({'Int16'})
        """
        answers = {answer for answer, _ in self.retypes}
        if self.retype is not None:
            answers.add(self.retype)
        return frozenset(answers)


@dataclass(frozen=True)
class SelectionMaterializer:
    """Apply a selection to a frame: to its declared columns and to its remainder.

    Code that types a frame operation such as `select(cs.numeric())`
    uses this class. For the declared columns, it gives each output
    column with its data type. For the remainder, it gives a
    `RemainderEffect`. It also finds two picked columns that get the
    same output name, which polars rejects.

    The kinds in `HELD` are the kinds that the data of a column can hold:
    every kind of polars but `Unknown`, which only the planner gives.

    Attributes:
        members: The kinds that a column declared `Unchecked` can have.

    Example:
        >>> from astrolabe._hooks.remainder.selector import OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> numeric = SelectionType.identity(OfType(Family.numeric()))
        >>> probe = SelectionMaterializer(("Int64", "String"))
        >>> probe.outputs({"a": "Int64", "s": "String"}, numeric, "row")
        {'a': 'Int64'}
    """

    HELD: ClassVar[tuple[str, ...]] = tuple(
        kind for kind in KindRoster().names() if kind != "Unknown"
    )

    members: tuple[str, ...]

    def outputs(
        self,
        schema: Mapping[str, str],
        selection: SelectionType,
        context: str,
    ) -> dict[str, str] | str:
        """Give the output columns for every declared column that the selection picks.

        The table of the context gives the output data type of each
        picked column. The context rule then applies: in an aggregation
        context, a selection that works row by row gives a `List` of its
        values.

        Args:
            schema: The frame's declared columns, each name mapped to
                its rendering.
            selection: The type of the selection.
            context: "row" for `select` and `with_columns`, or "agg"
                for `agg`.

        Returns:
            Each output name mapped to its rendering, or the name of a
            picked column where polars raises.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = SelectionType.identity(OfType(Family.numeric()))
            >>> probe = SelectionMaterializer(("Int64",))
            >>> probe.outputs({"a": "Int64"}, numeric, "agg")
            {'a': 'List(Int64)'}
            >>> probe.outputs({"a": "Int64"}, numeric, "row")
            {'a': 'Int64'}
        """
        outputs: dict[str, str] = {}
        for name, rendering in self._picked(schema, selection):
            at = {SelectionType.AXIS: rendering}
            answer = Materializer(self.members).read(at, selection.table)
            if answer is None:
                return name
            applied = ContextRule().applied(answer, selection.character, context)
            outputs[selection.affix.applied(name)] = applied
        return outputs

    def _picked(self, schema: Mapping[str, str], selection: SelectionType) -> list[tuple[str, str]]:
        """Give the declared columns that the selector picks.

        Args:
            schema: The frame's declared columns, each name mapped to
                its rendering.
            selection: The type of the selection.

        Returns:
            The name and rendering of each picked column, in the order
            of `schema`.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = SelectionType.identity(OfType(Family.numeric()))
            >>> SelectionMaterializer(())._picked({"a": "Int64", "s": "String"}, numeric)
            [('a', 'Int64')]
        """
        return [
            (name, rendering)
            for name, rendering in schema.items()
            if self._picks(selection.selector, name, rendering)
        ]

    def partial(self, schema: Mapping[str, str], selection: SelectionType) -> list[str]:
        """Give the declared columns that the selector picks for some data and not for other data.

        A column declared `Unchecked` can hold any kind. A selector that
        tests the data type picks it at some kinds only, so the data decides
        whether the selection gives the column.

        Args:
            schema: The frame's declared columns, each name mapped to
                its rendering.
            selection: The type of the selection.

        Returns:
            The names, in the order of `schema`.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = SelectionType.identity(OfType(Family.numeric()))
            >>> probe = SelectionMaterializer(("Int64",))
            >>> probe.partial({"u": "Unchecked", "n": "Int64"}, numeric)
            ['u']
        """
        partial = []
        for name, rendering in schema.items():
            picks = {selection.selector.picks(name, kind) for kind in self.HELD}
            if rendering == UNCHECKED and len(picks) > 1:
                partial.append(name)
        return partial

    def _picks(self, selector: Selector, name: str, rendering: str) -> bool:
        """Answer whether a selector picks one declared column.

        A column declared `Unchecked` can hold any kind. The selector
        picks it where it picks the column at one or more of those kinds.

        Args:
            selector: The selector of the selection.
            name: The name of the column.
            rendering: The declared rendering of the column.

        Returns:
            True where the selector picks the column.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> probe = SelectionMaterializer(("Int64", "String"))
            >>> probe._picks(OfType(Family.numeric()), "x", "Unchecked")
            True
            >>> probe._picks(OfType(Family.numeric()), "x", "String")
            False
        """
        if rendering == UNCHECKED:
            return any(selector.picks(name, kind) for kind in self.HELD)
        return selector.picks(name, rendering)

    def remainder(
        self,
        record: Record,
        selection: SelectionType,
        context: str = "row",
    ) -> RemainderEffect | str:
        """Give what the selection does to the remainder of an open frame.

        The checker does not know the columns of the remainder, but it
        knows which kinds they can have. For each kind that the
        selector can pick, it reads the table. The answers then merge.
        If every kind gives its own type back, the kept columns keep
        their data types. If every kind gives one shared data type,
        the kept columns take it. If the answers differ, the kept
        columns take `Unchecked`.

        In an aggregation context, a selection that works row by row
        gives a `List` of each column's values, so a column given back
        becomes a `List` of its own type.

        Args:
            record: What the checker knows about the remainder,
                including the kinds its columns can have.
            selection: The type of the selection.
            context: "row" or "agg".

        Returns:
            The effect, or a message for the user where polars raises
            at a kind that the remainder can hold.

        Example:
            >>> from astrolabe._hooks.remainder.record import Record
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = SelectionType.identity(OfType(Family.numeric()))
            >>> effect = SelectionMaterializer(("Int64",)).remainder(
            ...     Record.unconstrained(), numeric
            ... )
            >>> effect.retype is None
            True
        """
        kinds = selection.selector.surviving_types(record.types).kinds()
        answers = {self._in_context(kind, selection, context) for kind in sorted(kinds)}
        if None in answers:
            return "the selection has no result type at a type the remainder can hold"
        return RemainderEffect(selection.selector, self._collapsed(answers), selection.affix)

    def _in_context(self, kind: str, selection: SelectionType, context: str) -> str | None:
        """Give what the selection does to one kind of remainder column, in one context.

        The answer of the map is that of `_retype_at`.

        Args:
            kind: One kind that the remainder can hold.
            selection: The type of the selection.
            context: "row" or "agg".

        Returns:
            In a row context, the answer of the table. In an aggregation
            context, that answer with the context rule applied. There, an empty
            answer (the kind given back) counts as the kind itself.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = SelectionType.identity(OfType(Family.numeric()))
            >>> SelectionMaterializer(("Int64",))._in_context("Int64", numeric, "agg")
            'List(Int64)'
            >>> SelectionMaterializer(("Int64",))._in_context("Int64", numeric, "row")
            ''
        """
        answer = self._retype_at(kind, selection.table)
        if context != "agg" or answer is None:
            return answer
        return ContextRule().applied(answer or kind, selection.character, context)

    def _retype_at(self, kind: str, table: Table) -> str | None:
        """Give what the table does to one kind of remainder column.

        Args:
            kind: One kind that the remainder can hold.
            table: The table of the selection in the context.

        Returns:
            The empty text where the column keeps its own type, the
            output data type otherwise, `Unchecked` for a kind that the
            checker does not know, or None where polars raises.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = SelectionType.identity(OfType(Family.numeric()))
            >>> SelectionMaterializer(("Int64",))._retype_at("Int64", numeric.table)
            ''
            >>> cast = Table.constant(("",), ("Int64",), "String")
            >>> SelectionMaterializer(("Int64",))._retype_at("Int8", cast)
            'Unchecked'
            >>> SelectionMaterializer(("Int64", "String"))._retype_at("String", cast) is None
            True
        """
        outcome = table.outcome_at((kind,))
        if outcome == Axis(SelectionType.AXIS):
            return ""
        if outcome is None and kind not in self.members:
            return UNCHECKED
        return table.read_at({SelectionType.AXIS: Rendered.bare(kind)})

    def _collapsed(self, answers: set[str | None]) -> str | None:
        """Merge the answers for each kind into the one data type of the kept columns.

        A kind that the checker does not know gives `Unchecked`. That counts as an
        answer like any other, because the table does not say what the
        operation gives at that kind.

        Args:
            answers: The answers, with the empty text for a kind that
                keeps its own type.

        Returns:
            None where every kind keeps its own type or no kind is
            left, the one shared data type, or `Unchecked` where the
            answers differ.

        Example:
            >>> SelectionMaterializer(("Int64",))._collapsed({"", ""})
            >>> SelectionMaterializer(("Int64",))._collapsed({"String"})
            'String'
            >>> SelectionMaterializer(("Int64",))._collapsed({"", "String"})
            'Unchecked'
            >>> SelectionMaterializer(("Int64",))._collapsed({"", UNCHECKED})
            'Unchecked'
            >>> SelectionMaterializer(("Int64",))._collapsed({UNCHECKED})
            'Unchecked'
        """
        texts = Collapse().defined(answers)
        if texts <= {""}:
            return None
        return Collapse().collapsed(texts)
