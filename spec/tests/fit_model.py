"""Executable expression types, slice declarations, fit, and materialization.

This module is the reference implementation of the expression half's
decision rules: the quadruple type, the slice declaration with its
well-formedness rules, the fit relation, and the four-step
materialization procedure with the lifting folded in. The theorem
suite runs the soundness claims against it by exhaustion, and the
grammar-corner sweep feeds it every pathological declaration.

Outcomes reuse the one-axis rule spellings: `same`, `same:<read>`,
`const:T`, and `wrap:C` around the sole read.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, ClassVar

from spec.tests.table_model import QUOTIENT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from spec.tests.axis_model import Grid

CHARACTERS = ("elementwise", "aggregating", "literal")


class Abstraction:
    """The collapse from a set of possible data types to one displayed type.

    This is the display half of the lifting: a computation is carried
    out over a set of data types, and a minted type shows a singleton
    as its member and anything wider as `Unchecked`.

    Example:
        >>> Abstraction().collapsed(frozenset({"Int64"}))
        'Int64'
        >>> Abstraction().collapsed(frozenset({"Int64", "Float64"}))
        'Unchecked'
    """

    # The data type a position of several possible types shows. The
    # collapse is what produces it, so the collapse owns its name.
    MARKER: ClassVar[str] = "Unchecked"

    def collapsed(self, members: frozenset[str]) -> str | None:
        """Collapse a result set to the data type a minted type shows.

        Args:
            members: The possible result data types.

        Returns:
            The one member, `List(Unchecked)` for several lists,
            `Unchecked` for several other members, or None for none,
            which is the error case: no possible input succeeds.

        Example:
            >>> Abstraction().collapsed(frozenset()) is None
            True
            >>> Abstraction().collapsed(frozenset({"List(Int8)", "List(String)"}))
            'List(Unchecked)'
        """
        if not members:
            return None
        if len(members) == 1:
            return next(iter(members))
        return self.widened(members)

    def widened(self, members: frozenset[str]) -> str:
        """Give the one data type that holds two or more members.

        Args:
            members: The possible result data types, two or more.

        Returns:
            `List(Unchecked)` where every member is a list, else
            `Unchecked`.

        Example:
            >>> Abstraction().widened(frozenset({"List(Int8)", "Int8"}))
            'Unchecked'
        """
        if all(member.startswith("List(") for member in members):
            return f"List({Abstraction.MARKER})"
        return Abstraction.MARKER


@dataclass(frozen=True)
class ExprType:
    """One inferred expression type: the quadruple, executable.

    Example:
        >>> from spec.tests.axis_model import Grid
        >>> identity = ExprType(
        ...     frozenset({"a"}), Grid.constant(("a",), "Boolean"),
        ...     "a", "elementwise",
        ... )
        >>> identity.name
        'a'
    """

    needs: frozenset[str]
    grid: Grid
    name: str
    character: str


@dataclass(frozen=True)
class Read:
    """One declared read: a column name, or a blank, over a region.

    A region is a set of kinds; `EveryDType` is the whole set, and the
    `unchecked` flag is the other end, the region position that
    declares no region at all.

    Example:
        >>> Read("quantity", frozenset({"Int64"})).name
        'quantity'
    """

    name: str | None
    region: frozenset[str]
    unchecked: bool = False


@dataclass(frozen=True)
class Row:
    """One area of a declared map: per-read regions, one outcome.

    Example:
        >>> Row((frozenset({"Int64"}),), "same").outcome
        'same'
    """

    regions: tuple[frozenset[str], ...]
    outcome: str


@dataclass(frozen=True)
class SliceDecl:
    """One slice declaration: reads, rows, output name, character.

    Example:
        >>> decl = SliceDecl(
        ...     (Read("quantity", frozenset({"Int64"})),),
        ...     (Row((frozenset({"Int64"}),), "const:Boolean"),),
        ...     "is_buy", "elementwise",
        ... )
        >>> decl.output_name()
        'is_buy'
    """

    reads: tuple[Read, ...]
    rows: tuple[Row, ...]
    named: str | None
    character: str

    def output_name(self) -> str | None:
        """Give the declared output name, defaults applied.

        Returns:
            `named=` when written, else the sole read's name, else
            None, which rule 1 or the defaults reject elsewhere.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),), (), None, "elementwise",
            ... )
            >>> decl.output_name()
            'a'
        """
        if self.named is not None:
            return self.named
        if len(self.reads) == 1:
            return self.reads[0].name
        return None

    def read_names(self) -> frozenset[str]:
        """Give the named columns the declaration reads.

        Returns:
            Every non-blank read's name.

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
            How many reads have no name.

        Example:
            >>> SliceDecl((), (), "len", "aggregating").blank_count()
            0
        """
        return sum(1 for read in self.reads if read.name is None)

    def declared_cells(self, row: Row) -> Iterator[dict[str, str]]:
        """Enumerate one row's cells as column-to-kind assignments.

        A row's cells are the product of its reads' regions, one
        coordinate per read; an unchecked read contributes no
        coordinate, because it declares no region to check.

        Args:
            row: The row whose cells are wanted.

        Yields:
            One assignment per cell, blanks under the empty name.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), "same"),), None, "elementwise",
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
        """Evaluate one row's outcome at one assignment.

        A wrap over an `Unchecked` read is `Unchecked` whole. Wrapping
        every kind at once gives many types, and many collapse to the
        marker.

        Args:
            row: The row holding the outcome.
            assignment: The cell, as column-to-kind.

        Returns:
            The output type the declaration promises there.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), "same"),), None, "elementwise",
            ... )
            >>> decl.outcome_at(decl.rows[0], {"a": "Int64"})
            'Int64'
            >>> wrap = Row((frozenset({"Int64"}),), "wrap:List")
            >>> decl.outcome_at(wrap, {"a": "Int64"})
            'List(Int64)'
            >>> decl.outcome_at(wrap, {})
            'Unchecked'
        """
        kind, _, argument = row.outcome.partition(":")
        if kind == "const":
            return argument
        coordinate = self._same_coordinate(assignment, argument or None)
        if kind != "wrap":
            return coordinate
        sole = self._same_coordinate(assignment, None)
        if sole == Abstraction.MARKER:
            return Abstraction.MARKER
        return f"{argument}({sole})"

    def _same_coordinate(self, assignment: Mapping[str, str], read_name: str | None) -> str:
        """Give the coordinate a `same` outcome returns.

        Args:
            assignment: The cell, as column-to-kind.
            read_name: The read `same(n)` addresses, or None for the
                sole read.

        Returns:
            That read's coordinate.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),), (), None, "elementwise",
            ... )
            >>> decl._same_coordinate({"a": "Int64"}, None)
            'Int64'
        """
        key = read_name if read_name is not None else (self.reads[0].name or "")
        return assignment.get(key, Abstraction.MARKER)


class WellFormedness:
    """The declaration-site rules, each answering with a problem or None.

    Example:
        >>> decl = SliceDecl((), (), None, "elementwise")
        >>> "no column is read" in " ".join(WellFormedness().problems(decl))
        True
    """

    def problems(self, decl: SliceDecl) -> list[str]:
        """List every rule the declaration breaks.

        The structural rules run first, and a declaration failing one
        skips the cell-level rules, whose sweeps assume the shape the
        structural rules guarantee.

        Args:
            decl: The declaration to check.

        Returns:
            One message per broken rule; empty means well formed.

        Example:
            >>> decl = SliceDecl((), (), "len", "aggregating")
            >>> WellFormedness().problems(decl)
            []
        """
        structural = self._found(decl, (self._rule_one, self._rule_two))
        if structural:
            return structural
        return self._found(
            decl,
            (
                self._rule_three,
                self._rule_four,
                self._rule_five,
                self._rule_six,
                self._rule_eight,
            ),
        )

    def _found(
        self,
        decl: SliceDecl,
        checks: tuple[Callable[[SliceDecl], str | None], ...],
    ) -> list[str]:
        """Run a group of rules and keep the problems.

        Args:
            decl: The declaration to check.
            checks: The rules to run.

        Returns:
            Each problem the group reports.

        Example:
            >>> probe = WellFormedness()
            >>> decl = SliceDecl((), (), "len", "aggregating")
            >>> probe._found(decl, (probe._rule_one,))
            []
        """
        found = [check(decl) for check in checks]
        return [problem for problem in found if problem is not None]

    def _rule_one(self, decl: SliceDecl) -> str | None:
        """`typed_expr` reads; a zero-read aggregation needs its name.

        A literal stands without reads.

        Args:
            decl: The declaration to check.

        Returns:
            A problem, or None.

        Example:
            >>> WellFormedness()._rule_one(
            ...     SliceDecl((), (), "len", "aggregating"),
            ... ) is None
            True
            >>> WellFormedness()._rule_one(
            ...     SliceDecl((), (), None, "literal"),
            ... ) is None
            True
        """
        if decl.reads or decl.character == "literal":
            return None
        if decl.character == "aggregating" and decl.named is not None:
            return None
        return "no column is read. Read one, name a typed_agg, or use typed_lit"

    def _rule_two(self, decl: SliceDecl) -> str | None:
        """Every row spans the same reads.

        Args:
            decl: The declaration to check.

        Returns:
            A problem, or None.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((), "same"),), None, "elementwise",
            ... )
            >>> WellFormedness()._rule_two(decl)
            'a row reads a different number of columns. Give each row the same reads'
        """
        for row in decl.rows:
            if len(row.regions) != len(decl.reads):
                return "a row reads a different number of columns. Give each row the same reads"
        return None

    def _rule_three(self, decl: SliceDecl) -> str | None:
        """Rows are disjoint over their cells.

        Args:
            decl: The declaration to check.

        Returns:
            A problem, or None.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (
            ...         Row((frozenset({"Int64"}),), "same"),
            ...         Row((frozenset({"Int64"}),), "const:String"),
            ...     ),
            ...     None, "elementwise",
            ... )
            >>> WellFormedness()._rule_three(decl)
            'two rows cover the same cell. Make the rows disjoint'
            >>> one_row = SliceDecl(decl.reads, decl.rows[:1], None, "elementwise")
            >>> WellFormedness()._rule_three(one_row) is None
            True
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
        """`same` needs one read; `same(n)` names a read.

        A bare `same`, or a `wrap`, beside several reads does not say
        which read it addresses.

        Args:
            decl: The declaration to check.

        Returns:
            A problem, or None.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),
            ...      Read("b", frozenset({"Int64"}))),
            ...     (Row((frozenset({"Int64"}), frozenset({"Int64"})), "same"),),
            ...     "o", "elementwise",
            ... )
            >>> WellFormedness()._rule_four(decl)
            'same with no argument needs one read. Write same(name)'
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), "same:b"),),
            ...     None, "elementwise",
            ... )
            >>> WellFormedness()._rule_four(decl)
            'same(name) names a column the declaration does not read'
        """
        several = len(decl.reads) != 1
        for row in decl.rows:
            kind, _, argument = row.outcome.partition(":")
            unaddressed = kind == "wrap" or (kind == "same" and not argument)
            if several and unaddressed:
                return "same with no argument needs one read. Write same(name)"
            if kind == "same" and argument and argument not in decl.read_names():
                return "same(name) names a column the declaration does not read"
        return None

    def _rule_five(self, decl: SliceDecl) -> str | None:
        """At most one blank read.

        Args:
            decl: The declaration to check.

        Returns:
            A problem, or None.

        Example:
            >>> blank = Read(None, frozenset({"Float64"}))
            >>> decl = SliceDecl(
            ...     (blank, blank), (), None, "elementwise",
            ... )
            >>> WellFormedness()._rule_five(decl)
            'the declaration has more than one some_column read. Keep one'
        """
        if decl.blank_count() > 1:
            return "the declaration has more than one some_column read. Keep one"
        return None

    def _rule_six(self, decl: SliceDecl) -> str | None:
        """Regions are never empty; no column is read twice.

        Args:
            decl: The declaration to check.

        Returns:
            A problem, or None.

        Example:
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset()),), (), None, "elementwise",
            ... )
            >>> WellFormedness()._rule_six(decl)
            'a read has an empty region. Give it at least one data type'
            >>> unchecked = Read("a", frozenset(), unchecked=True)
            >>> decl = SliceDecl((unchecked,), (), None, "elementwise")
            >>> WellFormedness()._rule_six(decl) is None
            True
        """
        names = [read.name for read in decl.reads if read.name is not None]
        if len(names) != len(set(names)):
            return "a column is read twice. Read each column once"
        for read in decl.reads:
            if not read.region and not read.unchecked:
                return "a read has an empty region. Give it at least one data type"
        return None

    def _rule_eight(self, decl: SliceDecl) -> str | None:
        """A blank stands alone, and no axis outcome reads the unpinned.

        A blank read also takes no explicit `named=`.

        Args:
            decl: The declaration to check.

        Returns:
            A problem, or None.

        Example:
            >>> decl = SliceDecl(
            ...     (Read(None, frozenset({"Float64"})),
            ...      Read("q", frozenset({"Int64"}))),
            ...     (), None, "elementwise",
            ... )
            >>> WellFormedness()._rule_eight(decl)
            'a some_column read stands beside another read. Declare it alone'
            >>> decl = SliceDecl(
            ...     (Read(None, frozenset({"Float64"})),),
            ...     (), "renamed", "elementwise",
            ... )
            >>> WellFormedness()._rule_eight(decl)
            'named= renames a some_column read. Delete named='
        """
        if not decl.blank_count():
            return None
        if len(decl.reads) > 1:
            return "a some_column read stands beside another read. Declare it alone"
        if decl.named is not None:
            return "named= renames a some_column read. Delete named="
        return None


class Fit:
    """The fit relation, executable clause by clause.

    Example:
        >>> from spec.tests.axis_model import Grid
        >>> decl = SliceDecl(
        ...     (Read("a", frozenset({"Int64"})),),
        ...     (Row((frozenset({"Int64"}),), "same"),), None, "elementwise",
        ... )
        >>> argument = ExprType(
        ...     frozenset({"a"}), Grid.constant(("a",), "Int64"),
        ...     "a", "elementwise",
        ... )
        >>> Fit().verdict(argument, decl) is None
        True
    """

    def verdict(self, argument: ExprType, decl: SliceDecl) -> str | None:
        """Judge one argument against one declaration.

        Args:
            argument: The argument's inferred type.
            decl: The declared slice.

        Returns:
            None when the argument fits, else the first refusal.

        Example:
            >>> from spec.tests.axis_model import Grid
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), "same"),), None, "elementwise",
            ... )
            >>> aggregated = ExprType(
            ...     frozenset({"a"}), Grid.constant(("a",), "Int64"),
            ...     "a", "aggregating",
            ... )
            >>> Fit().verdict(aggregated, decl)
            'character aggregating where elementwise is declared'
        """
        checks = (self._needs, self._name, self._character, self._cells)
        for check in checks:
            refusal = check(argument, decl)
            if refusal is not None:
                return refusal
        return None

    def _needs(self, argument: ExprType, decl: SliceDecl) -> str | None:
        """Clause one: the needs sit inside the declared reads.

        Args:
            argument: The argument's inferred type.
            decl: The declared slice.

        Returns:
            A refusal, or None.

        Example:
            >>> from spec.tests.axis_model import Grid
            >>> decl = SliceDecl((), (), "len", "aggregating")
            >>> reader = ExprType(
            ...     frozenset({"z"}), Grid.constant(("z",), "Int64"),
            ...     "len", "aggregating",
            ... )
            >>> Fit()._needs(reader, decl)
            "reads {'z'} outside the declared reads"
        """
        allowed = decl.read_names()
        if decl.blank_count():
            allowed |= argument.needs
        outside = argument.needs - allowed
        if outside:
            return f"reads {set(sorted(outside))!r} outside the declared reads"
        return None

    def _name(self, argument: ExprType, decl: SliceDecl) -> str | None:
        """Clause two, the name half: the output name is the declared one.

        A blank declaration binds the argument's own name, and rule 5
        demands the argument read exactly one column and give it back.

        Args:
            argument: The argument's inferred type.
            decl: The declared slice.

        Returns:
            A refusal, or None.

        Example:
            >>> from spec.tests.axis_model import Grid
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), "same"),), None, "elementwise",
            ... )
            >>> renamed = ExprType(
            ...     frozenset({"a"}), Grid.constant(("a",), "Int64"),
            ...     "b", "elementwise",
            ... )
            >>> Fit()._name(renamed, decl)
            'gives b, where a is declared'
            >>> blank = SliceDecl(
            ...     (Read(None, frozenset({"Int64"})),), (), None, "elementwise",
            ... )
            >>> Fit()._name(renamed, blank)
            'a blank binds only an expression giving its one read back'
        """
        if decl.blank_count():
            if argument.needs == {argument.name}:
                return None
            return "a blank binds only an expression giving its one read back"
        declared = decl.output_name()
        if argument.name != declared:
            return f"gives {argument.name}, where {declared} is declared"
        return None

    def _character(self, argument: ExprType, decl: SliceDecl) -> str | None:
        """Clause three: the characters are equal, no substitution.

        Args:
            argument: The argument's inferred type.
            decl: The declared slice.

        Returns:
            A refusal, or None.

        Example:
            >>> from spec.tests.axis_model import Grid
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), "same"),), None, "elementwise",
            ... )
            >>> constant = ExprType(
            ...     frozenset(), Grid((), ((("",) [:0], "Boolean"),)),
            ...     "a", "literal",
            ... )
            >>> Fit()._character(constant, decl)
            'character literal where elementwise is declared'
        """
        if argument.character != decl.character:
            return f"character {argument.character} where {decl.character} is declared"
        return None

    def _cells(self, argument: ExprType, decl: SliceDecl) -> str | None:
        """Clause two, the table half: agreement at every declared cell.

        The marker agrees with anything, on either side.

        Args:
            argument: The argument's inferred type.
            decl: The declared slice.

        Returns:
            A refusal, or None.

        Example:
            >>> from spec.tests.axis_model import Grid
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), "const:Boolean"),),
            ...     "o", "elementwise",
            ... )
            >>> casting = ExprType(
            ...     frozenset({"a"}), Grid.constant(("a",), "String"),
            ...     "o", "elementwise",
            ... )
            >>> Fit()._cells(casting, decl)
            'the table answers String at a cell declared Boolean'
            >>> unchecked = SliceDecl(
            ...     (Read("a", frozenset(), unchecked=True),),
            ...     (Row((frozenset(),), "const:Boolean"),),
            ...     "o", "elementwise",
            ... )
            >>> varying = Grid.from_outputs(
            ...     ("a",),
            ...     {("Int64",): "Int64", ("String",): "String"},
            ... )
            >>> unknown = ExprType(frozenset({"a"}), varying, "o", "elementwise")
            >>> Fit()._cells(unknown, unchecked) is None
            True
        """
        for row in decl.rows:
            for assignment in decl.declared_cells(row):
                answered = self._lifted_answer(argument, decl, assignment)
                declared = decl.outcome_at(row, assignment)
                marked = Abstraction.MARKER in {answered, declared}
                if answered != declared and not marked:
                    return f"the table answers {answered} at a cell declared {declared}"
        return None

    def _lifted_answer(
        self,
        argument: ExprType,
        decl: SliceDecl,
        assignment: Mapping[str, str],
    ) -> str | None:
        """Read the argument's table at one declared cell, lifted.

        A declared `Unchecked` read pins no coordinate, so the
        argument's table is read over every member along that axis
        and the answers collapse, the lifting's own two rules. A
        blank's coordinate sits under the empty name, so it is moved
        to the argument's bound name, which the argument's own axes
        read.

        Args:
            argument: The argument's inferred type.
            decl: The declared slice.
            assignment: The declared cell's pinned coordinates.

        Returns:
            The collapsed answer, or None where nothing is defined.

        Example:
            >>> from spec.tests.axis_model import Grid
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), "same"),), None, "elementwise",
            ... )
            >>> expr = ExprType(
            ...     frozenset({"a"}), Grid.constant(("a",), "Int64"),
            ...     "a", "elementwise",
            ... )
            >>> Fit()._lifted_answer(expr, decl, {"a": "Int64"})
            'Int64'
            >>> blank = SliceDecl(
            ...     (Read(None, frozenset({"Float64"})),), (), None, "elementwise",
            ... )
            >>> Fit()._lifted_answer(expr, blank, {"": "Float64"})
            'Int64'
        """
        frame = {(column or argument.name): kind for column, kind in assignment.items()}
        for read in decl.reads:
            if read.unchecked and read.name is not None:
                frame[read.name] = Abstraction.MARKER
        return Materializer().read(frame, argument.grid)


class Materializer:
    """The four-step procedure of 17.1, with the lifting folded in.

    Example:
        >>> from spec.tests.axis_model import Grid
        >>> expr = ExprType(
        ...     frozenset({"a"}), Grid.constant(("a",), "Boolean"),
        ...     "a", "elementwise",
        ... )
        >>> Materializer().result({"a": "Int64"}, expr, "row")
        {'a': 'Boolean'}
    """

    def result(
        self,
        frame: Mapping[str, str],
        expr: ExprType,
        context: str,
    ) -> dict[str, str] | str:
        """Materialize one expression against one frame.

        Args:
            frame: The frame's schema, name to declared type.
            expr: The expression's type.
            context: "row" or "agg".

        Returns:
            The output column as a one-entry schema, or an error text.

        Example:
            >>> from spec.tests.axis_model import Grid
            >>> expr = ExprType(
            ...     frozenset({"a"}), Grid.constant(("a",), "Boolean"),
            ...     "a", "elementwise",
            ... )
            >>> Materializer().result({}, expr, "row")
            "the frame declares no column 'a'"
        """
        missing = expr.needs - set(frame)
        if missing:
            return f"the frame declares no column '{min(missing)}'"
        answer = self.read(frame, expr.grid)
        if answer is None:
            return "the declaration gives no result at the data types of this frame"
        return {expr.name: self.in_context(answer, expr.character, context)}

    def read(self, frame: Mapping[str, str], grid: Grid) -> str | None:
        """Read a grid at a frame, the lifting covering `Unchecked`.

        A frame column declared `Unchecked` stands for every kind, so
        the grid is read at every member and the defined answers
        collapse, exactly the lifting's two rules.

        Args:
            frame: The frame's schema.
            grid: The expression's table.

        Returns:
            The answer, or None where every member fails.

        Example:
            >>> from spec.tests.axis_model import Grid
            >>> grid = Grid.constant(("a",), "UInt32")
            >>> Materializer().read({"a": "Unchecked"}, grid)
            'UInt32'
            >>> Materializer().read({"a": "Int64"}, grid)
            'UInt32'
        """
        members: list[tuple[str, ...]] = []
        for name in grid.columns:
            if frame[name] == Abstraction.MARKER:
                members.append(QUOTIENT)
            else:
                members.append((frame[name],))
        answers: set[str] = set()
        for kinds in product(*members):
            answer = grid.read_at(dict(zip(grid.columns, kinds, strict=True)))
            if answer is not None:
                answers.add(answer)
        return Abstraction().collapsed(frozenset(answers))

    def in_context(self, answer: str, character: str, context: str) -> str:
        """Apply the context rule to one answer.

        Args:
            answer: The row-context output type.
            character: The expression's character.
            context: "row" or "agg".

        Returns:
            The type the context produces.

        Example:
            >>> Materializer().in_context("Int64", "elementwise", "agg")
            'List(Int64)'
        """
        if context == "agg" and character == "elementwise":
            return f"List({answer})"
        return answer
