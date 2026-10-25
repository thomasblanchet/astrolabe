"""Check that the expression engine gives the same answers as the Python models in `spec/tests`.

The models in `spec/tests` are the executable form of the design. They
are small and slow, and they write each answer as text. The engine is
the fast code that the plugin runs. These tests give both sides the
same inputs and compare the answers.

The inputs are a fixed set of pairs, each a declaration and an
expression that fits it. A declaration is a type that the user writes
for an expression, for example `column("price", pl.Int64)`. To fit is
to match that declaration. The tests compare four things: the table of
answers, the fit check, the rules that a declaration must obey, and
the column that an expression gives on a frame. They also compare how
a set of possible data types collapses to one shown answer.

The functions at the top of the module turn a model value into the
matching engine value.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from spec.tests import fit_model
from spec.tests.table_model import QUOTIENT
from spec.tests.test_fit_model import fitting_pairs

from astrolabe._hooks.expression.character import CharacterRoster
from astrolabe._hooks.expression.fit import Read, Row, SliceDecl, WellFormedness
from astrolabe._hooks.expression.lifting import Collapse
from astrolabe._hooks.expression.materialize import Materializer
from astrolabe._hooks.expression.outcome import Axis, Const, Outcome, Wrap
from astrolabe._hooks.expression.rendering import Rendered

from astrolabe._hooks.expression.quadruple import ExpressionType
from astrolabe._hooks.expression.table import Table

if TYPE_CHECKING:
    from spec.tests.axis_model import Grid


def engine_decl(m_decl: fit_model.SliceDecl) -> SliceDecl:
    """Turn a model declaration into the engine declaration that promises the same answers.

    Args:
        m_decl: The declaration of the model.

    Returns:
        The engine declaration.

    Example:
        >>> from spec.tests.test_fit_model import POINT
        >>> engine_decl(POINT).output_name()
        'price'
    """
    reads = tuple(Read(r.name, r.region, r.unchecked) for r in m_decl.reads)
    rows = tuple(Row(r.regions, engine_outcome(m_decl, r.outcome)) for r in m_decl.rows)
    return SliceDecl(reads, rows, m_decl.named, m_decl.character)


def engine_outcome(m_decl: fit_model.SliceDecl, text: str) -> Outcome:
    """Turn the text of one model answer into the engine value that means the same.

    The model writes an answer as text: `const:Boolean` for a fixed data
    type, `wrap:List` for the data type of the column inside a `List`,
    and `same` or `same:name` for the data type of a column that the
    expression reads.

    Args:
        m_decl: The declaration that holds the answer.
        text: The text of the answer.

    Returns:
        The engine answer.

    Example:
        >>> from spec.tests.test_fit_model import POINT
        >>> engine_outcome(POINT, "const:Boolean")
        Const(dtype='Boolean')
    """
    kind, _, argument = text.partition(":")
    if kind == "const":
        return Const(argument)
    if kind == "wrap":
        return Wrap(argument, Axis(_sole_key(m_decl)))
    return Axis(argument or _sole_key(m_decl))


def _sole_key(m_decl: fit_model.SliceDecl) -> str:
    """Give the column that a `same` with no name refers to.

    A `same` with no name means the one column that the declaration
    reads with a checked data type. A blank read, a read with no name
    that takes the name of the argument, is the empty text.

    Args:
        m_decl: The declaration that holds the answer.

    Returns:
        The name of that column, or the empty text.

    Example:
        >>> from spec.tests.test_fit_model import BLANK
        >>> _sole_key(BLANK)
        ''
    """
    names = _checked_names(m_decl)
    if names:
        return names[0]
    return ""


def _checked_names(m_decl: fit_model.SliceDecl) -> list[str]:
    """List the names of the columns that a declaration reads with a checked data type.

    A read marked `Unchecked` accepts any data type, so it has no axis in
    the table and is left out.

    Args:
        m_decl: The declaration.

    Returns:
        One name for each checked read, with the empty text for a blank read.

    Example:
        >>> from spec.tests.test_fit_model import POINT
        >>> _checked_names(POINT)
        ['price']
    """
    return [read.name or "" for read in m_decl.reads if not read.unchecked]


def engine_expr(m_expr: fit_model.ExprType) -> ExpressionType:
    """Turn a model expression type into the engine expression type.

    Both sides describe an expression with four parts: the columns it
    reads, a table of answers, the name of its output, and how it counts
    rows.

    Args:
        m_expr: The expression type of the model.

    Returns:
        The engine expression type, over the same table.

    Example:
        >>> from spec.tests.test_fit_model import identity_grid
        >>> point = fit_model.ExprType(
        ...     frozenset({"p"}), identity_grid("p"), "p", "elementwise",
        ... )
        >>> engine_expr(point).name
        'p'
    """
    table = Table.from_outputs(m_expr.grid.columns, dict(m_expr.grid.outputs))
    character = CharacterRoster().named(m_expr.character)
    return ExpressionType(m_expr.needs, table, m_expr.name, character)


class TestTableParity:
    """The engine table and the model table give the same answer at every cell."""

    def test_reads_agree_at_every_cell(self) -> None:
        """Each table of the fixed pairs answers alike on both sides."""
        for _, m_expr in fitting_pairs():
            self.assert_same_reads(m_expr.grid, engine_expr(m_expr).table)

    def assert_same_reads(self, grid: Grid, table: Table) -> None:
        """Read both tables at every combination of data types, and compare the answers.

        The data types here have their parameters taken away, so the set
        is finite.

        Args:
            grid: The table of the model.
            table: The engine table over the same answers.
        """
        for cell in product(QUOTIENT, repeat=len(grid.columns)):
            assignment = dict(zip(grid.columns, cell, strict=True))
            rendered = {name: Rendered.bare(kind) for name, kind in assignment.items()}
            assert table.read_at(rendered) == grid.read_at(assignment)


class TestWellFormednessParity:
    """The rules that a declaration must obey find the same faults on both sides."""

    def test_the_battery_is_well_formed_on_both_sides(self) -> None:
        """Each declaration of the fixed pairs obeys the rules on both sides."""
        for m_decl, _ in fitting_pairs():
            assert fit_model.WellFormedness().problems(m_decl) == []
            assert WellFormedness().problems(engine_decl(m_decl)) == []

    def test_broken_declarations_report_the_same_rules(self) -> None:
        """Each broken declaration breaks the same rule on both sides."""
        for m_decl in self.broken_declarations():
            expected = fit_model.WellFormedness().problems(m_decl)
            assert WellFormedness().problems(engine_decl(m_decl)) == expected

    def broken_declarations(self) -> list[fit_model.SliceDecl]:
        """List one broken declaration for each rule that the engine can express.

        One model case is left out: a `same` with no name in a
        declaration that reads several columns. The engine cannot write
        that answer, so its parser refuses it before these rules run.

        Returns:
            The broken declarations.
        """
        blank = fit_model.Read(None, frozenset({"Float64"}))
        named = fit_model.Read("a", frozenset({"Int64"}))
        int_row = fit_model.Row((frozenset({"Int64"}),), "same")
        return [
            fit_model.SliceDecl((), (), None, "elementwise"),
            fit_model.SliceDecl((named,), (fit_model.Row((), "same"),), None, "elementwise"),
            fit_model.SliceDecl((named,), (int_row, int_row), None, "elementwise"),
            fit_model.SliceDecl(
                (named,),
                (fit_model.Row((frozenset({"Int64"}),), "same:zz"),),
                None,
                "elementwise",
            ),
            fit_model.SliceDecl((blank, blank), (), None, "elementwise"),
            fit_model.SliceDecl((named, named), (), None, "elementwise"),
            fit_model.SliceDecl((fit_model.Read("a", frozenset()),), (), None, "elementwise"),
            fit_model.SliceDecl((blank, named), (), None, "elementwise"),
            fit_model.SliceDecl((blank,), (), "renamed", "elementwise"),
        ]


class TestMaterializerParity:
    """An expression on a frame gives the same column, or the same refusal, on both sides."""

    def test_results_agree_over_the_battery(self) -> None:
        """Each frame of the fixed pairs gives the same result in both contexts."""
        for m_decl, m_expr in fitting_pairs():
            self.assert_frames_agree(m_decl, m_expr)

    def assert_frames_agree(self, m_decl: fit_model.SliceDecl, m_expr: fit_model.ExprType) -> None:
        """Apply one fitting argument to each frame that its declaration allows, in both contexts.

        The two contexts are a row context such as `select` and an
        aggregation context such as `group_by(...).agg`.

        Args:
            m_decl: The declaration, whose cells give the frames.
            m_expr: The argument that fits the declaration.
        """
        for frame in self.frames_of(m_decl, m_expr.name):
            for context in ("row", "agg"):
                expected = fit_model.Materializer().result(frame, m_expr, context)
                result = Materializer(QUOTIENT).result(frame, engine_expr(m_expr), context)
                assert result == expected, (m_decl, frame, context)

    def frames_of(self, m_decl: fit_model.SliceDecl, bound: str) -> list[dict[str, str]]:
        """List the schemas of the frames that a declaration allows.

        Each cell of the declaration, that is, each combination of data
        types it answers for, gives one schema. A read marked `Unchecked`
        adds its column with the data type `Unchecked`.

        Args:
            m_decl: The declaration.
            bound: The name that a blank read takes.

        Returns:
            One schema for each cell. If the declaration has no cell, one
            schema with only the `Unchecked` columns.
        """
        overlay = self.overlay_of(m_decl)
        frames = [
            self.frame_of(assignment, bound, overlay)
            for row in m_decl.rows
            for assignment in m_decl.declared_cells(row)
        ]
        return frames or [dict(overlay)]

    def overlay_of(self, m_decl: fit_model.SliceDecl) -> dict[str, str]:
        """Give the columns that a declaration reads with the data type `Unchecked`.

        Args:
            m_decl: The declaration.

        Returns:
            Each such column, mapped to `Unchecked`.
        """
        return {
            read.name: "Unchecked"
            for read in m_decl.reads
            if read.unchecked and read.name is not None
        }

    def frame_of(
        self,
        assignment: dict[str, str],
        bound: str,
        overlay: dict[str, str],
    ) -> dict[str, str]:
        """Turn one cell of a declaration into the schema of a frame.

        Args:
            assignment: The data type of each read. A blank read is under
                the empty name.
            bound: The name that a blank read takes.
            overlay: The columns that the declaration reads as `Unchecked`.

        Returns:
            The schema.
        """
        frame = {(name or bound): kind for name, kind in assignment.items()}
        return frame | overlay

    def test_a_missing_column_is_refused_alike(self) -> None:
        """Both sides refuse a frame that lacks a column the expression reads."""
        m_decl, m_expr = next(iter(fitting_pairs()))
        del m_decl
        expected = fit_model.Materializer().result({}, m_expr, "row")
        assert Materializer(QUOTIENT).result({}, engine_expr(m_expr), "row") == (expected)


class TestLiftingParity:
    """Both sides collapse each set of possible data types to the same shown answer."""

    def test_collapse_agrees_on_every_subset(self) -> None:
        """Each subset of three data types collapses alike on both sides."""
        for subset in self.subsets(("Int64", "Float64", "String")):
            assert Collapse().collapsed(subset) == fit_model.Abstraction().collapsed(subset)

    def subsets(self, kinds: tuple[str, ...]) -> list[frozenset[str]]:
        """List every subset of some data types.

        Args:
            kinds: The data types, with no parameters.

        Returns:
            Every subset, the empty one included.
        """
        found: list[frozenset[str]] = [frozenset()]
        for kind in kinds:
            found += [subset | {kind} for subset in found]
        return found
