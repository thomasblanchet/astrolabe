"""Read a `typed_expr`, `typed_agg` or `typed_lit` call into the value that the engine checks.

A user declares an expression type with a call such as
`typed_expr(Quantity, gives=pl.Boolean, named="is_buy")`. The call
states four things: the columns that the expression reads, the data
type that it gives, its output name, and how it counts rows. The engine
holds these four facts in a `SliceDecl` and checks its rules there. The
code here reads the call into that value, and reports at the call each
problem that it finds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astrolabe._hooks.declaring.outcomes import OutcomeParser
from astrolabe._hooks.declaring.reads import ReadParser
from astrolabe._hooks.declaring.site import CallShape, DeclarationCall, DeclarationSite
from astrolabe._hooks.errors import MalformedDeclarationError
from astrolabe._hooks.expression.fit import Read, Row, SliceDecl, WellFormedness
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe.declare import row, same

if TYPE_CHECKING:
    from mypy.nodes import CallExpr, Expression
    from mypy.plugin import DynamicClassDefContext

    from astrolabe._hooks.expression.outcome import Outcome


class Malformed(Exception):
    """Stop the reading of a declaration after its first problem.

    A parser reports the problem before it raises this. The code that
    starts the reading catches it and gives None, and the hook then
    registers nothing.
    """


@dataclass(frozen=True)
class ParsedRow:
    """Hold the reads of one row of a declaration, before its outcome is read.

    A declaration can split the data types that it reads into rows, each
    with its own result: `row(column("q", pl.Int64), gives=pl.Float64)`.
    The flat form, with the reads and `gives=` on the declaration itself,
    is one row. The parser cannot read `gives=` until it has every read,
    because a bare `same` refers to the one read of the whole
    declaration. So this class keeps the `gives=` node unread.

    Example:
        >>> ParsedRow((Read("q", frozenset({"Int64"})),), None).names()
        ('q',)
    """

    reads: tuple[Read, ...]
    gives: Expression | None

    def names(self) -> tuple[str | None, ...]:
        """Give the column names of the reads, so that two rows can be compared.

        Returns:
            The names, in order. A read from `some_column(...)`, whose
            name the argument fills later, gives None.

        Example:
            >>> ParsedRow((Read(None, frozenset({"Int64"})),), None).names()
            (None,)
        """
        return tuple(read.name for read in self.reads)


class DeclarationParser:
    """Read one `typed_expr`, `typed_agg` or `typed_lit` call into a `SliceDecl`.

    The parser reads the call in two steps. The first step reads its
    shape: the keywords, the rows, the reads and the output name. A
    problem there stops the reading at once, because the later parts
    depend on it. The second step checks the finished value with
    `WellFormedness`, and reports every rule that it breaks.

    Each problem becomes a `[malformed-declaration]` error at the call.
    If the declaration has a problem, the parser gives None, and the
    name gets no type.
    """

    KEYWORDS: frozenset[str] = frozenset({"gives", "named"})

    def __init__(self, ctx: DynamicClassDefContext, character: str) -> None:
        """Keep the context of the declaration, and build the readers for its parts.

        Args:
            ctx: The mypy context of the declaration. The readers resolve
                names through it, and errors point at its call.
            character: How the declared expression counts rows:
                `elementwise`, `aggregating` or `literal`. The called
                function decides it.
        """
        self._ctx = ctx
        self._character = character
        self._reads = ReadParser(ctx)
        self._outcomes = OutcomeParser(ctx)
        self._shape = CallShape()

    def parsed(self, call: CallExpr) -> SliceDecl | None:
        """Read the declaration call, and check it.

        Args:
            call: The call node of the declaration.

        Returns:
            The declaration, or None if it has a problem. The parser
            has then reported the problem.
        """
        try:
            return self._checked(self._slice(DeclarationCall(call)))
        except Malformed:
            return None

    def _checked(self, decl: SliceDecl) -> SliceDecl | None:
        """Check the rules that a declaration must obey, and report each broken one.

        Args:
            decl: The declaration as the parser read it.

        Returns:
            The declaration, or None if it breaks a rule.
        """
        problems = WellFormedness().problems(decl)
        for problem in problems:
            MalformedDeclarationError(problem).fail(DeclarationSite(self._ctx))
        if problems:
            return None
        return decl

    def _slice(self, call: DeclarationCall) -> SliceDecl:
        """Read the parts of the call, and put them together.

        The reads of the first row are the reads of the declaration. A
        later row reads the same columns, at other data types. Each row
        of the engine holds the region of each read, that is, the set of
        data types at which the row reads its column, and the outcome.

        The output name comes from `named=`. If the call has no `named=`,
        the name gets its default.

        Args:
            call: The declaration call.

        Returns:
            The declaration, not yet checked against its rules.
        """
        self._check_keywords(call)
        rows = self._rows(call)
        reads = rows[0].reads
        node = call.keyword("named")
        named = self._literal_name(node) if node is not None else self._default_name(reads)
        built: list[Row] = []
        for parsed in rows:
            regions = tuple(read.region for read in parsed.reads)
            built.append(Row(regions, self._outcome(parsed, reads)))
        return SliceDecl(reads, tuple(built), named, self._character)

    def _check_keywords(self, call: DeclarationCall) -> None:
        """Report a keyword that the declaration functions do not take.

        Args:
            call: The declaration call.
        """
        for keyword in call.keywords():
            if keyword not in self.KEYWORDS:
                self._fail(f"unexpected keyword {keyword}=")

    def _rows(self, call: DeclarationCall) -> list[ParsedRow]:
        """Read the rows of the call, in the rows form or in the flat form.

        The call uses the rows form if a part passed by position is a
        `row(...)` call. A call that mixes `row(...)` parts with bare
        reads is a problem.

        Args:
            call: The declaration call.

        Returns:
            At least one row. Every row reads the same column names.
        """
        found = [self._shape.call(part, row) for part in call.positional()]
        row_calls = list(filter(None, found))
        if not row_calls:
            return [self._row(call)]
        if len(row_calls) != len(found):
            self._fail("rows and bare reads cannot mix")
        return self._same_names([self._row(DeclarationCall(one)) for one in row_calls])

    def _row(self, call: DeclarationCall) -> ParsedRow:
        """Read the reads of one row, and keep its `gives=` node for later.

        A `Columns` class or `subset_of(...)` gives several reads.

        Args:
            call: A `row(...)` call, or the declaration itself in the
                flat form.

        Returns:
            The row, with its outcome still unread.
        """
        reads: list[Read] = []
        for part in call.positional():
            part_reads = self._reads.parsed(part)
            if part_reads is None:
                self._fail(
                    "a read is column(name, region), some_column(region), "
                    "a declared column, a Columns class, column_of(...) or subset_of(...)"
                )
            reads.extend(not_none(part_reads))
        return ParsedRow(tuple(reads), call.keyword("gives"))

    def _same_names(self, rows: list[ParsedRow]) -> list[ParsedRow]:
        """Report a row that reads other columns than the first row.

        The rows of a declaration split the data types of one set of
        columns. A row over other columns has no meaning.

        Args:
            rows: The rows as the parser read them.

        Returns:
            The rows, unchanged.
        """
        names = rows[0].names()
        for parsed in rows[1:]:
            if parsed.names() != names:
                self._fail("the rows read different columns. Give every row the same reads")
        return rows

    def _default_name(self, reads: tuple[Read, ...]) -> str | None:
        """Give the output name of a declaration that has no `named=`.

        polars names a literal `literal`. Any other expression keeps the
        name of the column that it reads. With two or more reads that
        name is not clear, so `named=` is then required.

        Args:
            reads: The reads of the declaration.

        Returns:
            `literal` for a literal, or None for the name of the one
            read.

        Example:
            >>> parser = DeclarationParser.__new__(DeclarationParser)
            >>> parser._character = "literal"
            >>> parser._default_name(())
            'literal'
        """
        if self._character == "literal":
            return "literal"
        if len(reads) > 1:
            self._fail("named= is required with two or more reads")
        return None

    @propagates(None)
    def _literal_name(self, node: Expression) -> str | None:
        """Read the value of `named=`, which must be a string literal.

        Args:
            node: The node passed as `named=`.

        Returns:
            The name. None after a report for a value that is not a string
            literal, or for the empty string, which names no column that a
            declaration can mean.
        """
        name = self._shape.literal(node)
        if name is None:
            self._fail("named= must be a string literal")
        if not name:
            self._fail("named= must be a non-empty string")
        return not_falsy(name)

    def _outcome(self, parsed: ParsedRow, reads: tuple[Read, ...]) -> Outcome:
        """Read the outcome of one row: what the expression gives there.

        Args:
            parsed: The row, with its `gives=` node.
            reads: The reads of the declaration. A bare `same` refers to
                the one read among them.

        Returns:
            The outcome.
        """
        if parsed.gives is None:
            self._fail("gives= is required")
        node = not_none(parsed.gives)
        sole_read = self._sole_read(reads)
        self._check_bare_same(node, sole_read)
        outcome = self._outcomes.parsed(node, sole_read)
        if outcome is None:
            self._fail("gives= must be a data type, same, same(name) or wrap(...)")
        return not_none(outcome)

    def _check_bare_same(self, node: Expression, sole_read: str | None) -> None:
        """Report a bare `same` in a declaration that does not read exactly one column.

        A bare `same` means "the data type of the one read". With no read
        or with two, it names nothing, and the user must write
        `same(name)`.

        Args:
            node: The node passed as `gives=`.
            sole_read: The name of the one read, or None if the
                declaration has another count of reads.
        """
        if sole_read is None and self._shape.refers_to(node, same, "same"):
            self._fail("same with no argument needs one read. Write same(name)")

    def _sole_read(self, reads: tuple[Read, ...]) -> str | None:
        """Give the name of the one read, which a bare `same` refers to.

        Args:
            reads: The reads of the declaration.

        Returns:
            The name. A read from `some_column(...)` gives the empty
            name. Any count of reads other than one gives None.

        Example:
            >>> parser = DeclarationParser.__new__(DeclarationParser)
            >>> parser._sole_read((Read(None, frozenset({"Int64"})),))
            ''
            >>> parser._sole_read(()) is None
            True
        """
        if len(reads) != 1:
            return None
        return reads[0].name or ""

    def _fail(self, problem: str) -> None:
        """Report one problem at the declaration, and stop the reading.

        Args:
            problem: The text of the error.

        Raises:
            Malformed: Always, so that the reading stops.
        """
        MalformedDeclarationError(problem).fail(DeclarationSite(self._ctx))
        raise Malformed
