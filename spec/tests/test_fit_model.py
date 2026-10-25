"""Fit is sound, executable: what fits materializes identically.

The fit theorem says a declared slice can stand in for every argument
that fits it. These tests run the claim by exhaustion over a battery
of declarations, fitting arguments, and every frame inside each
declared slice, with the falsifiability controls beside it: an
argument fit refuses really does materialize differently, and the
materializer's lifting really recovers and really loses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spec.tests.axis_model import Grid
from spec.tests.fit_model import (
    Abstraction,
    ExprType,
    Fit,
    Materializer,
    Read,
    Row,
    SliceDecl,
    WellFormedness,
)
from spec.tests.table_model import QUOTIENT

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

NUMERIC = frozenset({"Int32", "Int64", "Float32", "Float64"})

POINT = SliceDecl(
    (Read("price", frozenset({"Float64"})),),
    (Row((frozenset({"Float64"}),), "same"),),
    None,
    "elementwise",
)

REGION = SliceDecl(
    (Read("quantity", NUMERIC),),
    (Row((NUMERIC,), "const:Boolean"),),
    "is_buy",
    "elementwise",
)

TWO_ROWS = SliceDecl(
    (Read("quantity", NUMERIC),),
    (
        Row((frozenset({"Int32", "Float32"}),), "const:Int64"),
        Row((frozenset({"Int64", "Float64"}),), "same"),
    ),
    "total",
    "aggregating",
)

BLANK = SliceDecl(
    (Read(None, frozenset({"Float32", "Float64"})),),
    (Row((frozenset({"Float32", "Float64"}),), "same"),),
    None,
    "elementwise",
)

TRUSTED = SliceDecl(
    (Read("payload", frozenset(), unchecked=True),),
    (Row((frozenset(),), "const:String"),),
    "payload_text",
    "elementwise",
)

ROOTLESS = SliceDecl((), (Row((), "const:UInt32"),), "len", "aggregating")


def fitting_pairs() -> Iterator[tuple[SliceDecl, ExprType]]:
    """Pair each battery declaration with an argument that fits it.

    Yields:
        One (declaration, fitting argument) pair per battery entry.
    """
    yield POINT, ExprType(frozenset({"price"}), identity_grid("price"), "price", "elementwise")
    yield (
        REGION,
        ExprType(
            frozenset({"quantity"}),
            Grid.constant(("quantity",), "Boolean"),
            "is_buy",
            "elementwise",
        ),
    )
    yield (
        TWO_ROWS,
        ExprType(frozenset({"quantity"}), summing_grid("quantity"), "total", "aggregating"),
    )
    yield BLANK, ExprType(frozenset({"temp"}), identity_grid("temp"), "temp", "elementwise")
    yield (
        TRUSTED,
        ExprType(
            frozenset({"payload"}),
            Grid.constant(("payload",), "String"),
            "payload_text",
            "elementwise",
        ),
    )
    yield ROOTLESS, ExprType(frozenset(), Grid.constant((), "UInt32"), "len", "aggregating")


def identity_grid(column: str) -> Grid:
    """Build the identity grid over one column.

    Args:
        column: The read column's name.

    Returns:
        The grid answering each cell's own kind.

    Example:
        >>> identity_grid("a").output_at(("Date",))
        'Date'
    """
    return Grid.from_outputs((column,), {(kind,): kind for kind in QUOTIENT})


def summing_grid(column: str) -> Grid:
    """Build the two-row grid `TWO_ROWS` promises.

    Args:
        column: The read column's name.

    Returns:
        Int64 on the narrow kinds, identity on the wide ones.

    Example:
        >>> summing_grid("q").output_at(("Int32",))
        'Int64'
    """
    outputs: dict[tuple[str, ...], str] = {("Int32",): "Int64", ("Float32",): "Int64"}
    outputs |= {(kind,): kind for kind in ("Int64", "Float64")}
    return Grid.from_outputs((column,), outputs)


def declared_type(decl: SliceDecl, bound_name: str) -> ExprType:
    """Read a declaration as the expression type it promises.

    Args:
        decl: The declared slice.
        bound_name: The name a blank binds, ignored otherwise.

    Returns:
        The quadruple over exactly the declared cells.

    Example:
        >>> declared_type(POINT, "price").name
        'price'
    """
    columns = _checked_columns(decl, bound_name)
    outputs = _declared_outputs(decl)
    name = decl.output_name() or bound_name
    needs = _declared_needs(decl, bound_name)
    return ExprType(needs, Grid.from_outputs(columns, outputs), name, decl.character)


def _checked_columns(decl: SliceDecl, bound_name: str) -> tuple[str, ...]:
    """Give the checked reads' column names, blanks bound.

    Args:
        decl: The declared slice.
        bound_name: The name a blank binds.

    Returns:
        One name per checked read.

    Example:
        >>> _checked_columns(POINT, "price")
        ('price',)
    """
    return tuple(read.name or bound_name for read in decl.reads if not read.unchecked)


def _declared_needs(decl: SliceDecl, bound_name: str) -> frozenset[str]:
    """Give every column the declaration reads, blanks bound.

    Args:
        decl: The declared slice.
        bound_name: The name a blank binds.

    Returns:
        The needs the promised quadruple carries.

    Example:
        >>> sorted(_declared_needs(POINT, "price"))
        ['price']
    """
    return frozenset(read.name or bound_name for read in decl.reads)


def _declared_outputs(decl: SliceDecl) -> dict[tuple[str, ...], str]:
    """Evaluate every declared cell's promised outcome.

    Args:
        decl: The declared slice.

    Returns:
        Each declared cell mapped to its promised output type.

    Example:
        >>> _declared_outputs(POINT)
        {('Float64',): 'Float64'}
    """
    outputs: dict[tuple[str, ...], str] = {}
    for row in decl.rows:
        _row_outputs(outputs, decl, row)
    return outputs


def _row_outputs(outputs: dict[tuple[str, ...], str], decl: SliceDecl, row: Row) -> None:
    """Record one row's cells and outcomes.

    Args:
        outputs: The cells recorded so far, updated in place.
        decl: The declared slice.
        row: The row whose cells are recorded.

    Example:
        >>> box: dict[tuple[str, ...], str] = {}
        >>> _row_outputs(box, POINT, POINT.rows[0])
        >>> box
        {('Float64',): 'Float64'}
    """
    for assignment in decl.declared_cells(row):
        outputs[_cell_of(decl, assignment)] = decl.outcome_at(row, assignment)


def _cell_of(decl: SliceDecl, assignment: Mapping[str, str]) -> tuple[str, ...]:
    """Give one assignment's coordinates in declared read order.

    Args:
        decl: The declared slice.
        assignment: The cell, as column-to-kind.

    Returns:
        One kind per checked read.

    Example:
        >>> _cell_of(POINT, {"price": "Float64"})
        ('Float64',)
    """
    return tuple(assignment[read or ""] for read in _keys(decl))


def _keys(decl: SliceDecl) -> tuple[str | None, ...]:
    """Give the checked reads' assignment keys, blanks as None.

    Args:
        decl: The declared slice.

    Returns:
        One key per checked read.

    Example:
        >>> _keys(POINT)
        ('price',)
    """
    return tuple(read.name for read in decl.reads if not read.unchecked)


class TestTheBatteryIsWellFormed:
    """Every battery declaration passes the declaration-site rules."""

    def test_no_battery_declaration_breaks_a_rule(self) -> None:
        """The soundness sweep below runs over lawful declarations."""
        for decl, _ in fitting_pairs():
            assert WellFormedness().problems(decl) == []

    def test_every_battery_argument_fits(self) -> None:
        """Each paired argument passes all four fit clauses."""
        for decl, argument in fitting_pairs():
            assert Fit().verdict(argument, decl) is None, decl


class TestFitIsSound:
    """What fits materializes as the declaration, frame for frame."""

    def test_fitting_arguments_materialize_as_the_declaration(self) -> None:
        """The executable fit theorem, by exhaustion.

        For every battery pair, every frame inside the declared slice,
        and both contexts: the argument's materialization equals the
        declaration's own.
        """
        for decl, argument in fitting_pairs():
            promised = declared_type(decl, argument.name)
            for frame in self.frames_in(decl, argument.name):
                self.assert_equal_results(frame, argument, promised)

    def frames_in(self, decl: SliceDecl, bound: str) -> list[dict[str, str]]:
        """Enumerate every frame schema inside one declared slice.

        Each cell of each row becomes one schema, its blank name bound
        to `bound`. A slice with no row gives its unchecked reads
        alone.

        Args:
            decl: The declared slice.
            bound: The name the blank binds, ignored otherwise.

        Returns:
            One schema per declared cell, unchecked reads at the
            marker.

        Example:
            >>> TestFitIsSound().frames_in(POINT, "price")
            [{'price': 'Float64'}]
        """
        frames = []
        for row in decl.rows:
            for assignment in decl.declared_cells(row):
                frame = {(name or bound): kind for name, kind in assignment.items()}
                frames.append(frame | self.unchecked_columns(decl))
        return frames or [dict(self.unchecked_columns(decl))]

    def unchecked_columns(self, decl: SliceDecl) -> dict[str, str]:
        """Give the frame entries for a declaration's unchecked reads.

        Args:
            decl: The declared slice.

        Returns:
            Each unchecked read's column at the marker.

        Example:
            >>> TestFitIsSound().unchecked_columns(TRUSTED)
            {'payload': 'Unchecked'}
        """
        return {
            read.name: Abstraction.MARKER
            for read in decl.reads
            if read.unchecked and read.name is not None
        }

    def assert_equal_results(
        self,
        frame: dict[str, str],
        argument: ExprType,
        promised: ExprType,
    ) -> None:
        """Assert one frame answers alike for argument and declaration.

        Args:
            frame: The frame schema inside the slice.
            argument: The fitting argument's type.
            promised: The declaration read as a type.
        """
        for context in ("row", "agg"):
            actual = Materializer().result(frame, argument, context)
            declared = Materializer().result(frame, promised, context)
            assert actual == declared, (frame, context, actual, declared)

    def test_a_refused_argument_really_differs(self) -> None:
        """The control: fit's refusal marks a real divergence.

        A casting argument is refused against `POINT`, and its
        materialization differs from the declaration's at the
        declared cell, so the refusal is not pedantry.
        """
        casting = ExprType(
            frozenset({"price"}),
            Grid.constant(("price",), "String"),
            "price",
            "elementwise",
        )
        assert Fit().verdict(casting, POINT) is not None
        mine = Materializer().result({"price": "Float64"}, casting, "row")
        promised = Materializer().result({"price": "Float64"}, declared_type(POINT, "price"), "row")
        assert mine != promised

    def test_the_dropped_literal_exception_stays_dropped(self) -> None:
        """A literal argument is refused against an elementwise slice.

        This is the fit decision that closed the soundness hole: in
        the aggregation context an elementwise result implodes where
        a literal stays unwrapped, so the two characters cannot stand
        in for each other.
        """
        constant = ExprType(frozenset(), Grid.constant((), "Boolean"), "is_buy", "literal")
        verdict = Fit().verdict(constant, REGION)
        assert verdict == "character literal where elementwise is declared"


class TestTheLiftingInsideMaterialization:
    """Reading a frame's `Unchecked` column runs the lifting, not a rule."""

    def test_a_single_answer_map_recovers(self) -> None:
        """A constant grid answers its constant at the marker."""
        grid = Grid.from_outputs(("a",), {("String",): "UInt32"})
        expr = ExprType(frozenset({"a"}), grid, "a", "elementwise")
        assert Materializer().result({"a": Abstraction.MARKER}, expr, "row") == {"a": "UInt32"}

    def test_a_varying_map_propagates(self) -> None:
        """An identity grid answers the marker at the marker."""
        expr = ExprType(frozenset({"a"}), identity_grid("a"), "a", "elementwise")
        result = Materializer().result({"a": Abstraction.MARKER}, expr, "row")
        assert result == {"a": Abstraction.MARKER}

    def test_an_everywhere_empty_map_errs(self) -> None:
        """A grid defined nowhere errs even at the marker."""
        expr = ExprType(frozenset({"a"}), Grid.from_outputs(("a",), {}), "a", "elementwise")
        result = Materializer().result({"a": Abstraction.MARKER}, expr, "row")
        assert result == "the declaration gives no result at the data types of this frame"
