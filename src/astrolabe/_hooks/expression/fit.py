"""Expression declarations, their rules, and the fit of an argument to one.

A user can declare the type of an expression parameter, for example
`Price = column("price", pl.Float64)` or
`typed_expr(column("q", pl.Int64), gives=pl.Boolean, named="is_buy")`.
Such a declaration answers four questions. Which columns does the
expression read, and at which data types? What type does it give at each
of those data types? What is its output name? How does it count rows?

This module holds three things. `SliceDecl` is the declaration as plain
data. `WellFormedness` checks the rules that a declaration must obey on
its own, where the user writes it. `Fit` checks an argument against a
declaration, where a call passes it. The argument fits if it reads only
declared columns, gives the declared output name, gives the declared
type at every declared data type, and counts rows in the same way. A
character never stands in for another: a fixed value does not fit a
declaration that works row by row.

Everything here works on plain values. Code that parses a declaration,
and code that checks a call, supplies them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from astrolabe._hooks.expression.character import CharacterRoster
from astrolabe._hooks.expression.lifting import UNCHECKED
from astrolabe._hooks.expression.outcome import Axis, Outcome, Wrap
from astrolabe._hooks.expression.rendering import Rendered
from astrolabe._hooks.expression.quadruple import ExpressionType
from astrolabe._hooks.expression.table import Table

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping


@dataclass(frozen=True)
class Read:
    """One column that a declaration reads, with the kinds that it accepts there.

    The region is the set of kinds that the declaration accepts for the
    column. A kind is a data type with its parameters erased, for example
    `Datetime`. `EveryDType` declares every kind. The `unchecked` flag
    marks a read declared as `Unchecked`: it declares no region at all,
    and it accepts the column at any data type.

    A read with no name is a blank, which `some_column(...)` declares.
    The call binds a blank to the name of the column that the argument
    reads.

    Attributes:
        name: The column name, or None for a blank.
        region: The kinds that the read accepts. Empty for an unchecked
            read.
        unchecked: True for a read declared as `Unchecked`.

    Example:
        >>> Read("quantity", frozenset({"Int64"})).name
        'quantity'
    """

    name: str | None
    region: frozenset[str]
    unchecked: bool = False


@dataclass(frozen=True)
class Row:
    """One area of a declared table: a region for each read, and one outcome for all of it.

    A declaration can give different types at different data types, with
    `row(...)`. Each row covers the product of its regions, one region
    for each read in the order of the reads. The outcome is the type that
    the declaration promises at every cell of that area, for example a
    fixed type, or `same` for the type of a read.

    Example:
        >>> from astrolabe._hooks.expression.outcome import Const
        >>> Row((frozenset({"Int64"}),), Const("Boolean")).outcome
        Const(dtype='Boolean')
    """

    regions: tuple[frozenset[str], ...]
    outcome: Outcome


@dataclass(frozen=True)
class SliceDecl:
    """One expression declaration as plain data: reads, rows, output name and character.

    The declaration covers only the cells that its rows name, which is
    why it is a slice of a full table. Code that parses `typed_expr`,
    `typed_agg`, `typed_lit` or `column` builds one. Code that checks a
    call reads one back from the mypy type of a parameter.

    Attributes:
        reads: The columns that the declaration reads.
        rows: The areas of the table and their outcomes.
        named: The `named=` argument, or None if the user did not
            write one.
        character: "elementwise", "aggregating" or "literal".

    Example:
        >>> from astrolabe._hooks.expression.outcome import Const
        >>> decl = SliceDecl(
        ...     (Read("q", frozenset({"Int64"})),),
        ...     (Row((frozenset({"Int64"}),), Const("Boolean")),),
        ...     "is_buy", "elementwise",
        ... )
        >>> decl.output_name()
        'is_buy'
    """

    reads: tuple[Read, ...]
    rows: tuple[Row, ...]
    named: str | None
    character: str

    @classmethod
    def from_expression(cls, expression: ExpressionType) -> SliceDecl:
        """Build the declaration back from the expression type that the checker minted for it.

        The type of a declaration is an expression type whose table
        holds only the declared cells. So the axes of the table are the
        reads, the kinds on each axis are its region, and each cell is a
        row whose regions each hold one kind. An axis with the empty name
        is the blank. An axis at `Unchecked` is a read that declares no
        region, and the rows skip its kind with an empty region.

        Args:
            expression: The expression type of the declaration, as the
                encoding layer read it from a signature.

        Returns:
            The declaration, ready for the fit.

        Example:
            >>> from astrolabe._hooks.expression.character import ELEMENTWISE
            >>> from astrolabe._hooks.expression.outcome import Const
            >>> from astrolabe._hooks.expression.quadruple import (
            ...     ExpressionType,
            ... )
            >>> from astrolabe._hooks.expression.table import Table
            >>> flag = Table.constant(("q",), (UNCHECKED,), "Boolean")
            >>> expr = ExpressionType(frozenset({"q"}), flag, "is_buy", ELEMENTWISE)
            >>> decl = SliceDecl.from_expression(expr)
            >>> decl.reads
            (Read(name='q', region=frozenset(), unchecked=True),)
            >>> decl.rows[0].outcome
            Const(dtype='Boolean')
            >>> decl.rows[0].regions
            (frozenset(),)
            >>> identity = Table.from_cells(("a",), {("Int64",): Axis("a")})
            >>> read = ExpressionType(frozenset({"a"}), identity, "a", ELEMENTWISE)
            >>> SliceDecl.from_expression(read).rows[0].regions
            (frozenset({'Int64'}),)
        """
        table = expression.table
        reads = tuple(cls._read_of(table, name) for name in table.columns)
        rows: list[Row] = []
        for cell, outcome in table.cells:
            regions = tuple(frozenset({kind} - {UNCHECKED}) for kind in cell)
            rows.append(Row(regions, outcome))
        return cls(reads, tuple(rows), expression.name or None, expression.character.label)

    @classmethod
    def _read_of(cls, table: Table, name: str) -> Read:
        """Read one axis as a declared read.

        Args:
            table: The table of the declaration.
            name: The column of the axis.

        Returns:
            The read. It is a blank for the empty name, and unchecked if
            the axis holds only `Unchecked`.

        Example:
            >>> from astrolabe._hooks.expression.table import Table
            >>> blank = Table.from_cells(("",), {("Float64",): Axis("")})
            >>> SliceDecl._read_of(blank, "")
            Read(name=None, region=frozenset({'Float64'}), unchecked=False)
        """
        region = table.projection(name)
        if region == {UNCHECKED}:
            return Read(name or None, frozenset(), unchecked=True)
        return Read(name or None, region)

    def as_expression(self) -> ExpressionType:
        """Build the expression type that holds this declaration.

        The reads are the axes of the table. The blank has the empty
        name, and an unchecked read holds the one kind `Unchecked`. The
        cells of each row are the product of its regions, each with the
        outcome of the row. Each outcome then takes the one form that a
        minted table uses. For example, a declared fixed type that equals
        the kind of its own cell becomes "the type of that column". So a
        declaration always spells the same way as the expression that it
        describes. A declared outcome is a fixed type, the type of a
        read, or a `List` around one of those. So it always has an answer
        at a kind, and a missing answer is a bug.

        Returns:
            The expression type of the declaration.

        Example:
            >>> from astrolabe._hooks.expression.outcome import Const
            >>> decl = SliceDecl(
            ...     (Read("q", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), Const("Boolean")),),
            ...     "is_buy", "elementwise",
            ... )
            >>> expr = decl.as_expression()
            >>> expr.name, sorted(expr.needs), expr.table.cells
            ('is_buy', ['q'], ((('Int64',), Const(dtype='Boolean')),))
            >>> same = SliceDecl(
            ...     (Read("q", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), Const("Int64")),),
            ...     None, "elementwise",
            ... )
            >>> same.as_expression().table.cells
            ((('Int64',), Axis(column='q')),)
            >>> listed = SliceDecl(
            ...     (Read(None, frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), Wrap("List", Axis(""))),),
            ...     None, "elementwise",
            ... )
            >>> table = listed.as_expression().table
            >>> table.columns, table.read_at({"": Rendered.bare("Int64")})
            (('',), 'List(Int64)')
        """
        columns = tuple(read.name or "" for read in self.reads)
        outputs: dict[tuple[str, ...], str] = {}
        for row in self.rows:
            for cell in product(*map(self._coordinates, row.regions)):
                kinds = zip(columns, cell, strict=True)
                rendered = {name: Rendered.bare(kind) for name, kind in kinds}
                outputs[cell] = row.outcome.evaluated(rendered)
        table = Table.from_outputs(columns, outputs)
        character = CharacterRoster().named(self.character)
        return ExpressionType(frozenset(columns), table, self.output_name() or "", character)

    def _coordinates(self, region: frozenset[str]) -> tuple[str, ...]:
        """Give the kinds that the region of one read adds to the cells of a row.

        Args:
            region: The region of the read in that row.

        Returns:
            The kinds, sorted, or `Unchecked` alone for a read that
            declares no region.

        Example:
            >>> decl = SliceDecl((), (), "len", "aggregating")
            >>> decl._coordinates(frozenset())
            ('Unchecked',)
            >>> decl._coordinates(frozenset({"Int64", "Float64"}))
            ('Float64', 'Int64')
        """
        if not region:
            return (UNCHECKED,)
        return tuple(sorted(region))

    def output_name(self) -> str | None:
        """Give the output name of the declaration, with the default applied.

        Returns:
            `named=` if the user wrote it. In other cases, the name of the
            only read. None if there is no such name, which a rule of
            `WellFormedness` rejects.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),), (), None, "elementwise",
            ... )
            >>> decl.output_name()
            'a'
            >>> two = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),
            ...      Read("b", frozenset({"Int64"}))),
            ...     (), None, "elementwise",
            ... )
            >>> two.output_name() is None
            True
        """
        if self.named is not None:
            return self.named
        if len(self.reads) == 1:
            return self.reads[0].name
        return None  # pragma: no cover -- a declaration of two reads names its result

    def read_names(self) -> frozenset[str]:
        """Give the names of the columns that the declaration reads.

        Returns:
            The name of every read that is not a blank.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),), (), None, "elementwise",
            ... )
            >>> decl.read_names()
            frozenset({'a'})
        """
        return frozenset(read.name for read in self.reads if read.name is not None)

    def blank_count(self) -> int:
        """Count the blank reads.

        Returns:
            The number of reads that have no name.

        Example:
            >>> SliceDecl((), (), "len", "aggregating").blank_count()
            0
        """
        return sum(1 for read in self.reads if read.name is None)

    def declared_cells(self, row: Row) -> Iterator[dict[str, str]]:
        """List the cells of one row, each as a map from column to kind.

        The cells of a row are the product of the regions of its reads,
        one kind for each read. An unchecked read adds no kind, because
        it declares no region to check. The kind of a blank read stays
        under the empty name until a call binds the blank.

        Args:
            row: The row whose cells to list.

        Yields:
            One map for each cell.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), Axis("a")),),
            ...     None, "elementwise",
            ... )
            >>> list(decl.declared_cells(decl.rows[0]))
            [{'a': 'Int64'}]
        """
        checked = [
            (read, region)
            for read, region in zip(self.reads, row.regions, strict=True)
            if not read.unchecked
        ]
        for kinds in product(*(sorted(region) for _, region in checked)):
            yield {read.name or "": kind for (read, _), kind in zip(checked, kinds, strict=True)}

    def outcome_at(self, row: Row, assignment: Mapping[str, str]) -> str:
        """Give the type that one row promises at one cell, or at one data type of a cell.

        Args:
            row: The row that holds the outcome.
            assignment: The cell, as a map from column to kind, or to a
                full data type of that kind.

        Returns:
            The output type that the declaration promises there. The type
            of an unchecked read gives `Unchecked`, because that read
            stands for every kind.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Datetime"})),),
            ...     (Row((frozenset({"Datetime"}),), Axis("a")),),
            ...     None, "elementwise",
            ... )
            >>> decl.outcome_at(decl.rows[0], {"a": "Datetime"})
            'Datetime'
            >>> decl.outcome_at(decl.rows[0], {"a": "Datetime(time_unit='ms', time_zone=None)"})
            "Datetime(time_unit='ms', time_zone=None)"
        """
        rendered = {
            name: Rendered(text.partition("(")[0], text) for name, text in assignment.items()
        }
        return row.outcome.evaluated(rendered)


class WellFormedness:
    """Check the rules that a declaration must obey on its own.

    Code that parses a declaration runs these rules where the user wrote
    it, and reports each broken rule there. A broken declaration can
    never fit an argument in a useful way, so the user must fix it
    first. Each rule gives a message that says what is wrong and how to
    fix it, or None.

    Example:
        >>> decl = SliceDecl((), (), None, "elementwise")
        >>> "no column is read" in " ".join(WellFormedness().problems(decl))
        True
    """

    def problems(self, decl: SliceDecl) -> list[str]:
        """List every rule that the declaration breaks.

        The rules about the shape run first. If the declaration breaks
        one, the rules about its cells do not run, because they need the
        shape that the first rules make sure of.

        Args:
            decl: The declaration to check.

        Returns:
            One message for each broken rule. An empty list means that
            the declaration is correct.

        Example:
            >>> WellFormedness().problems(SliceDecl((), (), "len", "aggregating"))
            []
        """
        structural = self._found(decl, (self._rule_one, self._rule_two))
        if structural:
            return structural
        return self._found(
            decl,
            (self._rule_three, self._rule_four, self._rule_five, self._rule_six),
        )

    def _found(
        self,
        decl: SliceDecl,
        checks: tuple[Callable[[SliceDecl], str | None], ...],
    ) -> list[str]:
        """Run a group of rules and keep the messages.

        Args:
            decl: The declaration to check.
            checks: The rules to run.

        Returns:
            Each message that the group gives.

        Example:
            >>> probe = WellFormedness()
            >>> decl = SliceDecl((), (), "len", "aggregating")
            >>> probe._found(decl, (probe._rule_one,))
            []
        """
        found = [check(decl) for check in checks]
        return [problem for problem in found if problem is not None]

    def _rule_one(self, decl: SliceDecl) -> str | None:
        """Check that a declaration reads a column, unless it is a literal or a named aggregation.

        An aggregation that reads no column, for example `pl.len()`,
        has no read to take its name from, so it must give `named=`.

        Args:
            decl: The declaration to check.

        Returns:
            A message, or None.

        Example:
            >>> probe = WellFormedness()
            >>> probe._rule_one(SliceDecl((), (), "len", "aggregating")) is None
            True
            >>> probe._rule_one(SliceDecl((), (), None, "literal")) is None
            True
            >>> probe._rule_one(SliceDecl((), (), None, "aggregating"))
            'no column is read. Read one, name a typed_agg, or use typed_lit'
        """
        if decl.reads or decl.character == "literal":
            return None
        if decl.character == "aggregating" and decl.named is not None:
            return None
        return "no column is read. Read one, name a typed_agg, or use typed_lit"

    def _rule_two(self, decl: SliceDecl) -> str | None:
        """Check that every row has one region for each read.

        Args:
            decl: The declaration to check.

        Returns:
            A message, or None.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((), Axis("a")),), None, "elementwise",
            ... )
            >>> WellFormedness()._rule_two(decl)
            'a row reads a different number of columns. Give each row the same reads'
        """
        for row in decl.rows:
            if len(row.regions) != len(decl.reads):
                return "a row reads a different number of columns. Give each row the same reads"
        return None

    def _rule_three(self, decl: SliceDecl) -> str | None:
        """Check that no two rows cover the same cell.

        A cell that two rows cover gets two promised types, and the
        declaration then says two things at once.

        Args:
            decl: The declaration to check.

        Returns:
            A message, or None.

        Example:
            >>> from astrolabe._hooks.expression.outcome import Const
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (
            ...         Row((frozenset({"Int64"}),), Axis("a")),
            ...         Row((frozenset({"Int64"}),), Const("String")),
            ...     ),
            ...     None, "elementwise",
            ... )
            >>> WellFormedness()._rule_three(decl)
            'two rows cover the same cell. Make the rows disjoint'
        """
        seen: set[tuple[tuple[str, str], ...]] = set()
        for row in decl.rows:
            for assignment in decl.declared_cells(row):
                key = tuple(sorted(assignment.items()))
                if key in seen:
                    return "two rows cover the same cell. Make the rows disjoint"
                seen.add(key)
        return None

    def _rule_four(self, decl: SliceDecl) -> str | None:
        """Check that each `same(name)` names a column that the declaration reads.

        The rule applies to `same(name)` alone and inside `wrap(...)`.

        Args:
            decl: The declaration to check.

        Returns:
            A message, or None.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), Axis("zz")),),
            ...     None, "elementwise",
            ... )
            >>> WellFormedness()._rule_four(decl)
            'same(name) names a column the declaration does not read'
        """
        allowed = decl.read_names() | {""}
        for row in decl.rows:
            if self._addressed(row.outcome) - allowed:
                return "same(name) names a column the declaration does not read"
        return None

    def _addressed(self, outcome: Outcome) -> frozenset[str]:
        """Give the names of the reads whose type an outcome gives back.

        Args:
            outcome: The outcome that the row promises.

        Returns:
            The names. A fixed type names none.

        Example:
            >>> WellFormedness()._addressed(Wrap("List", Axis("a")))
            frozenset({'a'})
        """
        if isinstance(outcome, Axis):
            return frozenset({outcome.column})
        if isinstance(outcome, Wrap):
            return self._addressed(outcome.inner)
        return frozenset()

    def _rule_five(self, decl: SliceDecl) -> str | None:
        """Check that the declaration has at most one blank read.

        Args:
            decl: The declaration to check.

        Returns:
            A message, or None.

        Example:
            >>> blank = Read(None, frozenset({"Float64"}))
            >>> WellFormedness()._rule_five(
            ...     SliceDecl((blank, blank), (), None, "elementwise")
            ... )
            'the declaration has more than one some_column read. Keep one'
        """
        if decl.blank_count() > 1:
            return "the declaration has more than one some_column read. Keep one"
        return None

    def _rule_six(self, decl: SliceDecl) -> str | None:
        """Check the reads: no column twice, no empty region, and the rules for a blank.

        A read must declare at least one kind, unless it is declared as
        `Unchecked`. The rules for a blank run here too. A blank read must
        stand alone, and `named=` must not rename it, because the call
        binds the blank to the name of the argument and gives that name
        back.

        Args:
            decl: The declaration to check.

        Returns:
            A message, or None.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset()),), (), None, "elementwise",
            ... )
            >>> WellFormedness()._rule_six(decl)
            'a read has an empty region. Give it at least one data type'
            >>> twice = Read("a", frozenset({"Int64"}))
            >>> WellFormedness()._rule_six(SliceDecl((twice, twice), (), None, "elementwise"))
            'a column is read twice. Read each column once'
            >>> unchecked = Read("a", frozenset(), unchecked=True)
            >>> WellFormedness()._rule_six(SliceDecl((unchecked,), (), None, "elementwise")) is None
            True
        """
        names = [read.name for read in decl.reads if read.name is not None]
        if len(names) != len(set(names)):
            return "a column is read twice. Read each column once"
        if any(not read.region and not read.unchecked for read in decl.reads):
            return "a read has an empty region. Give it at least one data type"
        return self._rule_eight(decl)

    def _rule_eight(self, decl: SliceDecl) -> str | None:
        """Check that a blank read stands alone and keeps its own name.

        Args:
            decl: The declaration to check.

        Returns:
            A message, or None.

        Example:
            >>> decl = SliceDecl(
            ...     (Read(None, frozenset({"Float64"})),
            ...      Read("q", frozenset({"Int64"}))),
            ...     (), None, "elementwise",
            ... )
            >>> WellFormedness()._rule_eight(decl)
            'a some_column read stands beside another read. Declare it alone'
            >>> renamed = SliceDecl(
            ...     (Read(None, frozenset({"Float64"})),),
            ...     (), "renamed", "elementwise",
            ... )
            >>> WellFormedness()._rule_eight(renamed)
            'named= renames a some_column read. Delete named='
        """
        if not decl.blank_count():
            return None
        if len(decl.reads) > 1:
            return "a some_column read stands beside another read. Declare it alone"
        if decl.named is not None:
            return "named= renames a some_column read. Delete named="
        return None


class CellRefusal:
    """Say how an expression and its declared type differ at one data type of a column.

    Code that checks an argument against a declaration uses this sentence
    in its error. An expression can have no result for a data type, where
    polars refuses it. The sentence states that as an absence, and never
    as a type named `None`.

    Example:
        >>> CellRefusal().wording("Float64", "Boolean")
        'it makes a Float64 column, but the declared type is Boolean'
    """

    def wording(self, answered: str | None, declared: str) -> str:
        """Say how the two sides differ at one data type.

        Args:
            answered: The type that the expression makes there, or None
                if it has no result.
            declared: The type that the declaration promises there.

        Returns:
            The difference, in one sentence.

        Example:
            >>> CellRefusal().wording(None, "UInt32")
            'it has no result where the declared type is UInt32'
            >>> CellRefusal().wording("Int64", "String")
            'it makes an Int64 column, but the declared type is String'
        """
        if answered is None:
            return f"it has no result where the declared type is {declared}"
        made = f"it makes {self._article(answered)} {answered} column"
        return f"{made}, but the declared type is {declared}"

    def _article(self, dtype: str) -> str:
        """Give the article that reads correctly before one type name.

        The article follows the sound, not the letter. `UInt32` starts
        with the sound of "you" and takes "a". `Int32` and `Unknown`
        take "an".

        Args:
            dtype: The name of the type.

        Returns:
            "a" or "an".

        Example:
            >>> CellRefusal()._article("Int64")
            'an'
            >>> CellRefusal()._article("UInt32")
            'a'
            >>> CellRefusal()._article("String")
            'a'
        """
        if dtype.startswith("UInt"):
            return "a"
        return "an" if dtype[0] in "AEIOU" else "a"


class Fit:
    """Decide whether the answer of an argument agrees with a declaration at one data type.

    Code that checks an argument against a declaration evaluates the argument
    at each declared cell, and asks this class whether its answer there breaks
    the declared outcome. `Unchecked` on either side agrees with anything.

    Example:
        >>> Fit().disagree("Int64", "Int64")
        False
    """

    def disagree(self, answered: str | None, declared: str) -> bool:
        """Answer whether the answer of the argument at one cell breaks the declaration.

        `Unchecked` agrees with any type, on either side, because it
        stands for a type that nobody knows.

        Args:
            answered: The answer of the argument at the cell, with an
                unchecked read covered.
            declared: The type that the declaration promises.

        Returns:
            True if the two differ and neither is `Unchecked`.

        Example:
            >>> Fit().disagree("Int64", "Boolean")
            True
            >>> Fit().disagree("Unchecked", "Boolean")
            False
        """
        return answered != declared and UNCHECKED not in {answered, declared}
