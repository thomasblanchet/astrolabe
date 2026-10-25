"""Read a `typed_selection` or `typed_agg_selection` call into the type that it declares.

A selection is a selector expression, for example `cs.numeric()`, with
operations applied to it. It gives one output column for each column
that the selector picks. A user declares its type with a call such as
`typed_selection(cs.numeric(), gives=pl.Float64, suffix="_f")`. The code
here reads that call into a `SelectionType`, the value that the engine
uses for a selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.declaring.outcomes import OutcomeParser
from astrolabe._hooks.declaring.parser import Malformed
from astrolabe._hooks.declaring.regions import RegionParser
from astrolabe._hooks.declaring.selectors import SelectorSyntax
from astrolabe._hooks.declaring.site import CallShape, DeclarationCall, DeclarationSite
from astrolabe._hooks.errors import MalformedDeclarationError
from astrolabe._hooks.expression.character import CharacterRoster
from astrolabe._hooks.expression.fit import Read, Row, SliceDecl, WellFormedness
from astrolabe._hooks.expression.selection import Affix, SelectionType
from astrolabe._hooks.propagate import not_none
from astrolabe.declare import row

if TYPE_CHECKING:
    from mypy.nodes import CallExpr, Expression
    from mypy.plugin import DynamicClassDefContext

    from astrolabe._hooks.expression.outcome import Outcome
    from astrolabe._hooks.remainder.selector import Selector


class SelectionDeclarationParser:
    """Read one selection declaration, and check it against the rules of a declaration.

    The first argument is the selector. The rest of the call says what
    each picked column gives. The flat form gives one outcome, through
    `gives=`, for every data type that the selector can pick. The rows
    form splits those data types: each `row(region, gives=...)` names a
    region, that is, a set of kinds, and the outcome there. A kind is a
    data type without its parameters.

    An expression declaration already has rules for rows, for example
    that two rows do not overlap. To reuse them, the parser views the
    rows as an expression declaration with one read. That read has no
    name, because the selector fills it, and its region joins the
    regions of every row. Each broken rule becomes a
    `[malformed-declaration]` error at the call.
    """

    KEYWORDS: frozenset[str] = frozenset({"gives", "suffix", "prefix"})

    def __init__(self, ctx: DynamicClassDefContext, character: str) -> None:
        """Keep the context of the declaration, and build the readers for its parts.

        Args:
            ctx: The mypy context of the declaration. The readers resolve
                names through it, and errors point at its call.
            character: How the declared selection counts rows:
                `elementwise` or `aggregating`. The called function
                decides it.
        """
        self._ctx = ctx
        self._character = character
        self._regions = RegionParser(ctx)
        self._outcomes = OutcomeParser(ctx)
        self._shape = CallShape()

    def parsed(self, call: CallExpr) -> SelectionType | None:
        """Read the declaration call, and check it.

        Args:
            call: The call node of the declaration.

        Returns:
            The selection type, or None if the declaration has a
            problem. The parser has then reported the problem.
        """
        try:
            return self._checked(DeclarationCall(call))
        except Malformed:
            return None

    def _checked(self, call: DeclarationCall) -> SelectionType | None:
        """Read the parts of the call, check the rows, and build the selection type.

        The rename of every output column comes from `prefix=` and `suffix=`. An empty prefix
        and an empty suffix mean that no output column changes its name.

        Args:
            call: The declaration call.

        Returns:
            The selection type, or None if the rows break a rule.
        """
        self._check_keywords(call)
        parts = call.positional()
        if not parts:
            self._fail("a selector comes first")
        selector = self._selector(parts[0])
        matched = SelectionType.matched_kinds(selector)
        decl = self._well_formed(self._slice(self._rows(call, parts[1:], matched)))
        if decl is None:
            return None
        prefix = self._literal(call.keyword("prefix"))
        suffix = self._literal(call.keyword("suffix"))
        affix = Affix(prefix or "", suffix or "")
        character = CharacterRoster().named(self._character)
        return SelectionType(selector, decl.as_expression().table, affix, character)

    def _well_formed(self, decl: SliceDecl) -> SliceDecl | None:
        """Check the rows with the rules of an expression declaration, and report each broken one.

        Args:
            decl: The rows, viewed as a declaration with one read that
                has no name.

        Returns:
            The declaration, or None if it breaks a rule.
        """
        problems = WellFormedness().problems(decl)
        for problem in problems:
            MalformedDeclarationError(problem).fail(DeclarationSite(self._ctx))
        if problems:
            return None
        return decl

    def _check_keywords(self, call: DeclarationCall) -> None:
        """Report a keyword that the called function does not take.

        An elementwise selection takes `gives=`, `suffix=` and `prefix=`. Only `typed_selection`
        takes a rename, so an aggregating selection takes `gives=` alone.

        Args:
            call: The declaration call.
        """
        allowed = self.KEYWORDS if self._character == "elementwise" else frozenset({"gives"})
        for keyword in call.keywords():
            if keyword not in allowed:
                self._fail(f"unexpected keyword {keyword}=")

    def _selector(self, node: Expression) -> Selector:
        """Read the first argument as the selector.

        Args:
            node: The first argument passed by position.

        Returns:
            The selector.
        """
        selector = SelectorSyntax().parsed(node)
        if selector is None:
            self._fail("a selector is written with cs.*, pl.all() or pl.col(names...)")
        return not_none(selector)

    def _rows(
        self,
        call: DeclarationCall,
        parts: list[Expression],
        matched: frozenset[str],
    ) -> list[Row]:
        """Read the rows, in the rows form or in the flat form.

        Each region keeps only the kinds that the selector can pick. A
        row cannot describe a data type that the selection never sees.

        Args:
            call: The declaration call.
            parts: The arguments passed by position after the selector.
            matched: The kinds that the selector can pick.

        Returns:
            At least one row, each over one region.
        """
        if not parts:
            return [self._row(matched, call.keyword("gives"))]
        return [self._row_part(part, matched) for part in parts]

    def _row_part(self, part: Expression, matched: frozenset[str]) -> Row:
        """Read one `row(region, gives=...)` argument.

        A selection reads one column at a time, so a row has exactly one region, that is, a set
        of kinds. The rows split the kinds that the selector matches into parts, and each part
        gives one outcome. That region cannot be `Unchecked`, because `Unchecked` is not a set
        of kinds. It asks for no check of the data type, so it cannot be one part of that split.

        Args:
            part: The node of the argument.
            matched: The kinds that the selector can pick.

        Returns:
            The row, its region cut to those kinds.
        """
        inner = self._shape.call(part, row)
        if inner is None:
            self._fail("after the selector, every part is row(region, gives=...)")
        inner_call = DeclarationCall(not_none(inner))
        nodes = inner_call.positional()
        if len(nodes) != 1:
            self._fail("a row of a selection takes one region")
        region = self._regions.parsed(nodes[0])
        if region is None or region.unchecked:
            self._fail("a selection's row takes a region, never Unchecked")
        kinds = not_none(region).kinds & matched
        return self._row(kinds, inner_call.keyword("gives"))

    def _row(self, region: frozenset[str], gives: Expression | None) -> Row:
        """Read the outcome of one row, and build the row.

        A bare `same` in the outcome refers to the picked column.

        Args:
            region: The region of the row.
            gives: The node passed as `gives=`, or None if it is missing.

        Returns:
            The row.
        """
        if gives is None:
            self._fail("gives= is required")
        outcome: Outcome | None = self._outcomes.parsed(not_none(gives), SelectionType.AXIS)
        if outcome is None:
            self._fail("gives= must be a data type, same or wrap(...)")
        return Row((region,), not_none(outcome))

    def _slice(self, rows: list[Row]) -> SliceDecl:
        """View the rows as an expression declaration with one read that has no name.

        Args:
            rows: The rows.

        Returns:
            The declaration that the rules check. Its one read covers
            the region of every row.

        Example:
            >>> from astrolabe._hooks.expression.outcome import Axis
            >>> parser = SelectionDeclarationParser.__new__(SelectionDeclarationParser)
            >>> parser._character = "elementwise"
            >>> decl = parser._slice([Row((frozenset({"Int64"}),), Axis(""))])
            >>> decl.reads
            (Read(name=None, region=frozenset({'Int64'}), unchecked=False),)
        """
        united: frozenset[str] = frozenset()
        for one in rows:
            united |= one.regions[0]
        return SliceDecl((Read(None, united),), tuple(rows), None, self._character)

    def _literal(self, node: Expression | None) -> str | None:
        """Read the value of `prefix=` or `suffix=`, which must be a string literal.

        Args:
            node: The node passed under the keyword, or None.

        Returns:
            The string, or None if the call does not pass the keyword.
        """
        if node is None:
            return None
        text = self._shape.literal(node)
        if text is None:
            self._fail("suffix= and prefix= must be string literals")
        return text

    def _fail(self, problem: str) -> None:
        """Report one problem at the declaration, and stop the reading.

        Args:
            problem: The text of the error.

        Raises:
            Malformed: Always, so that the reading stops.
        """
        MalformedDeclarationError(problem).fail(DeclarationSite(self._ctx))
        raise Malformed
