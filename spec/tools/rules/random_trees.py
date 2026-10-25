"""Build random trees of the census nodes, and run polars on them.

A random tree has a census node at each branch and an input outcome of the
census at each leaf: a column of a census kind in a row form, with its values
all null or not, or a literal of a class. The check of whole trees builds
such trees, and compares the answers of polars with those of the rules.

The truth of polars comes from the sample frames of the census, typical
first. A run type comes from the first sample on which the run gives a column,
so a run that raises for some values and not for others keeps its type.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Protocol

import polars as pl

from spec.tools.node_census.kinds import CensusKinds, Sample, SampleFrames
from spec.tools.node_census.outcomes import OutcomeSpace
from spec.tools.node_census.surface import CensusSurface, NodeTemplate
from astrolabe._rules.definitions.registry import Definitions
from astrolabe._rules.options import RowForm
from astrolabe._rules.tree import ExpressionTree

if TYPE_CHECKING:
    import random
    from collections.abc import Callable

    from spec.tools.node_census.outcomes import Outcome


class Tree(Protocol):
    """Stand for a node of a random tree: a leaf or a branch."""

    def expression(self) -> pl.Expr:
        """Give the polars expression of the tree."""

    def text(self) -> str:
        """Give a short text of the tree, for a report."""

    def nulled(self, chosen: frozenset[int] | None = None, start: int = 0) -> Tree:
        """Give the same tree with the values of the chosen column leaves all null.

        Args:
            chosen: The numbers of the leaves to null, in the order of the
                tree, or `None` for every leaf.
            start: The number of the first leaf of this tree.
        """

    def leaves(self) -> int:
        """Give the number of leaves of the tree."""


@dataclass(frozen=True)
class Leaf:
    """Hold one input outcome of the census as a leaf, with its values all null or not.

    Example:
        >>> from spec.tools.node_census.outcomes import ColumnOutcome
        >>> Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=True).text()
        'k0/per row null'
    """

    outcome: Outcome
    null: bool

    def expression(self) -> pl.Expr:
        """Give the expression of the outcome.

        Returns:
            The expression.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> str(Leaf(ColumnOutcome(3, RowForm.PER_ROW), null=False).expression())
            'col("k3")'
        """
        return self.outcome.expression(null=self.null)

    def text(self) -> str:
        """Give the label of the outcome.

        Returns:
            The text.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> Leaf(ColumnOutcome(0, RowForm.ONE_VALUE), null=False).text()
            'k0/one value'
        """
        return self.outcome.label() + (" null" if self.null else "")

    def nulled(self, chosen: frozenset[int] | None = None, start: int = 0) -> Leaf:
        """Give the leaf with all values null, if it is chosen and the outcome has an all-null twin.

        Args:
            chosen: The numbers of the leaves to null, or `None` for every leaf.
            start: The number of this leaf.

        Returns:
            The leaf.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False).nulled().text()
            'k0/per row null'
        """
        picked = chosen is None or start in chosen
        return replace(self, null=self.null or (picked and self.outcome.can_be_null()))

    def leaves(self) -> int:
        """Give one: a leaf is one leaf.

        Returns:
            One.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False).leaves()
            1
        """
        return 1


@dataclass(frozen=True)
class Branch:
    """Hold one census node with a tree at each input.

    Example:
        >>> from spec.tools.node_census.outcomes import ColumnOutcome
        >>> node = NodeTemplate("root.abs", "pl.col('x').abs()")
        >>> Branch(node, (Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False),)).text()
        'root.abs(k0/per row)'
    """

    template: NodeTemplate
    inputs: tuple[Tree, ...]

    def expression(self) -> pl.Expr:
        """Give the node, built on the expressions of its inputs.

        Returns:
            The expression.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> node = NodeTemplate("root.abs", "pl.col('x').abs()")
            >>> int8 = Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False)
            >>> str(Branch(node, (int8,)).expression())
            'col("k0").abs()'
        """
        return self.template.build([tree.expression() for tree in self.inputs])

    def text(self) -> str:
        """Give the name of the node and the texts of its inputs.

        Returns:
            The text.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> node = NodeTemplate("root.abs", "pl.col('x').abs()")
            >>> int8 = Leaf(ColumnOutcome(0, RowForm.ONE_ROW), null=False)
            >>> Branch(node, (int8,)).text()
            'root.abs(k0/one row)'
        """
        inner = ", ".join(tree.text() for tree in self.inputs)
        return f"{self.template.key}({inner})"

    def nulled(self, chosen: frozenset[int] | None = None, start: int = 0) -> Branch:
        """Give the same tree with the values of the chosen column leaves all null.

        Args:
            chosen: The numbers of the leaves to null, in the order of the
                tree, or `None` for every leaf.
            start: The number of the first leaf of this tree.

        Returns:
            The tree.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> node = NodeTemplate("root.add", "pl.col('x').add(pl.col('x'))")
            >>> int8 = Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False)
            >>> Branch(node, (int8, int8)).nulled(frozenset({1})).text()
            'root.add(k0/per row, k0/per row null)'
        """
        inputs: list[Tree] = []
        for tree in self.inputs:
            inputs.append(tree.nulled(chosen, start))
            start += tree.leaves()
        return replace(self, inputs=tuple(inputs))

    def leaves(self) -> int:
        """Give the number of leaves of the tree.

        Returns:
            The number.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> node = NodeTemplate("root.add", "pl.col('x').add(pl.col('x'))")
            >>> int8 = Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False)
            >>> Branch(node, (int8, int8)).leaves()
            2
        """
        return sum(tree.leaves() for tree in self.inputs)


@dataclass(frozen=True)
class SampleColumn:
    """Hold the column that one run of polars gives, and the height of the frame it ran on.

    Example:
        >>> SampleColumn(pl.Series([1, 2]), 3).height
        3
    """

    column: pl.Series
    height: int


class TreeTruth:
    """Run polars on a tree over the sample frames of the census, and give its types.

    A run type comes from the first sample on which the run gives a column,
    typical first: a run that raises for some values and not for others meets
    a value that does not fit, and keeps its type.

    Example:
        >>> from spec.tools.node_census.outcomes import ColumnOutcome
        >>> node = NodeTemplate("root.abs", "pl.col('x').abs()")
        >>> int8 = Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False)
        >>> TreeTruth.example().tree(Branch(node, (int8,)))
        ['Int8', 'Int8', 'per row', 'List(Int8)', 'List(Int8)']
    """

    RAISES: ClassVar[str] = "raises"

    def __init__(self, frames: list[pl.DataFrame]) -> None:
        """Keep the frames of the samples.

        Args:
            frames: The frame of each sample of the census, typical first.

        Example:
            >>> TreeTruth.example()._frame.height
            3
        """
        self._frames = frames
        self._frame = frames[0]

    @classmethod
    def example(cls) -> TreeTruth:
        """Give the truth on the census frames.

        Returns:
            The truth.

        Example:
            >>> TreeTruth.example()._frame.width > 50
            True
        """
        frames = SampleFrames(CensusKinds())
        return cls([frames.frame(sample) for sample in Sample])

    def tree(self, tree: Tree) -> list[object]:
        """Give the types of a tree in `select`, then in `agg`.

        Args:
            tree: The tree.

        Returns:
            The planner type, the run type and the row form in `select`, then
            the planner type and the run type in `agg`. A run that raises has
            no row form.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> node = NodeTemplate("root.sum", "pl.col('x').sum()")
            >>> int8 = Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False)
            >>> TreeTruth.example().tree(Branch(node, (int8,)))[2]
            'one value'
        """
        named = tree.expression().alias("o")
        lazy = self._frame.lazy()
        grouped = lazy.group_by("g").agg(named)
        selected = self._columns(lambda frame: frame.select(named)["o"])
        heights = [*selected, *self._nulled(tree)] if selected else []
        planner = self._dtype(lambda: lazy.select(named).collect_schema()["o"])
        planner_agg = self._dtype(lambda: grouped.collect_schema()["o"])
        aggregated = self._first(lambda frame: frame.group_by("g").agg(named)["o"])
        return [
            planner,
            self._rendered(selected[0].column if selected else self.RAISES),
            self._form(heights, scalar=planner_agg == planner),
            planner_agg,
            self._rendered(aggregated),
        ]

    def runs(self, tree: Tree) -> bool:
        """Say whether polars runs a tree in `select`.

        Args:
            tree: The tree.

        Returns:
            Whether the run gives a column.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> TreeTruth.example().runs(Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False))
            True
        """
        named = tree.expression().alias("o")
        return isinstance(self._column(lambda: self._frame.select(named)), pl.DataFrame)

    def _nulled(self, tree: Tree) -> list[SampleColumn]:
        """Run a tree on the typical sample with its column leaves all null, together and alone.

        A node such as `drop_nulls` changes its length over the nulls of an
        input. Nulling every leaf at once alone can make another node fail
        first.

        Args:
            tree: The tree.

        Returns:
            The column of each run that gives one.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> node = NodeTemplate("root.drop_nulls", "pl.col('x').drop_nulls()")
            >>> int8 = Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False)
            >>> [run.column.len() for run in TreeTruth.example()._nulled(Branch(node, (int8,)))]
            [0, 0]
        """
        alone = [tree.nulled(frozenset({leaf})) for leaf in range(tree.leaves())]
        runs: list[SampleColumn] = []
        for variant in [tree.nulled(), *alone]:
            column = self._selected(variant)
            if isinstance(column, pl.Series):
                runs.append(SampleColumn(column, self._frame.height))
        return runs

    def _selected(self, tree: Tree) -> object:
        """Run a tree in `select` on the typical sample, and give its column, or `raises`.

        Args:
            tree: The tree.

        Returns:
            The column, or `raises`.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> int8 = Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False)
            >>> TreeTruth.example()._selected(int8).len()
            3
        """
        named = tree.expression().alias("o")
        return self._column(lambda: self._frame.select(named)["o"])

    def _form(self, selected: list[SampleColumn], *, scalar: bool) -> str | None:
        """Give the row form of the runs in `select`.

        A node gives one value when `agg` does not wrap its type in a list.
        Otherwise it gives one value per row when every run has the height of
        its frame, on every sample and with the column leaves all null, one
        row when every run has one row, and another length otherwise.

        Args:
            selected: The column of each run that gives one, with the height of its frame.
            scalar: Whether the planner types in `select` and in `agg` are equal.

        Returns:
            The row form, or `None` if the run raises on every sample.

        Example:
            >>> runs = [SampleColumn(pl.Series([1, 2, 3]), 3), SampleColumn(pl.Series([1]), 400)]
            >>> TreeTruth.example()._form(runs, scalar=False)
            'other length'
            >>> TreeTruth.example()._form([], scalar=False) is None
            True
        """
        if not selected:
            return None
        if scalar:
            return RowForm.ONE_VALUE.value
        if all(run.column.len() == run.height for run in selected):
            return RowForm.PER_ROW.value
        if all(run.column.len() == 1 for run in selected):
            return RowForm.ONE_ROW.value
        return RowForm.OTHER_LENGTH.value

    def _first(self, run: Callable[[pl.DataFrame], object]) -> object:
        """Run a query on each sample in turn, and give the first column it gives, or `raises`.

        Args:
            run: The query, on the frame of one sample.

        Returns:
            The column, or `raises` if the query raises on every sample.

        Example:
            >>> truth = TreeTruth.example()
            >>> truth._first(lambda frame: frame.select(pl.col("k0") // 0)["k0"]).dtype
            Int8
            >>> truth._first(lambda frame: 1 // 0)
            'raises'
        """
        for frame in self._frames:
            result = self._column(functools.partial(run, frame))
            if isinstance(result, pl.Series):
                return result
        return self.RAISES

    def _columns(self, run: Callable[[pl.DataFrame], object]) -> list[SampleColumn]:
        """Run a query on every sample, and give each column it gives, with the height of its frame.

        Args:
            run: The query, on the frame of one sample.

        Returns:
            The columns, in the order of the samples, typical first.

        Example:
            >>> columns = TreeTruth.example()._columns(lambda frame: frame["k0"].unique())
            >>> sorted(run.column.len() for run in columns)
            [1, 1, 3, 3, 3]
        """
        columns: list[SampleColumn] = []
        for frame in self._frames:
            result = self._column(functools.partial(run, frame))
            if isinstance(result, pl.Series):
                columns.append(SampleColumn(result, frame.height))
        return columns

    def _rendered(self, result: object) -> str:
        """Give the rendering of the data type of a column, or `raises`.

        Args:
            result: The column, or `raises`.

        Returns:
            The rendering.

        Example:
            >>> TreeTruth.example()._rendered(pl.Series([1]))
            'Int64'
        """
        return str(result.dtype) if isinstance(result, pl.Series) else self.RAISES

    def _column(self, run: Callable[[], object]) -> object:
        """Run a query, and give its result, or `raises`.

        Args:
            run: The query.

        Returns:
            The result, or `raises`.

        Example:
            >>> TreeTruth.example()._column(lambda: 1 // 0)
            'raises'
        """
        try:
            return run()
        except BaseException as error:  # a panic of polars is a BaseException
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return self.RAISES

    def _dtype(self, run: Callable[[], object]) -> str:
        """Run a query that gives a data type, and give its rendering, or `raises`.

        Args:
            run: The query.

        Returns:
            The rendering.

        Example:
            >>> TreeTruth.example()._dtype(lambda: pl.Int8())
            'Int8'
        """
        return str(self._column(run))


class CensusNodes:
    """Give the census nodes that a random tree can hold: those of a definition and few inputs.

    A census node is a call of polars on placeholder inputs (`NodeTemplate`).
    The check of whole trees needs the nodes that the rules state, and the
    census holds the further inputs of a node beyond the third only beside its
    receiver.

    Example:
        >>> nodes = CensusNodes().defined()
        >>> all(template.inputs() <= 3 for template in nodes)
        True
    """

    MAX_INPUTS: ClassVar[int] = 3

    def defined(self) -> list[NodeTemplate]:
        """Give the census nodes of at most three inputs whose node has a definition.

        Returns:
            The nodes, in the order of the census.

        Example:
            >>> keys = [template.key for template in CensusNodes().defined()]
            >>> "root.abs" in keys
            True
        """
        templates = CensusSurface().templates()
        few = [template for template in templates if template.inputs() <= self.MAX_INPUTS]
        return [template for template in few if self._stated(template)]

    def _stated(self, template: NodeTemplate) -> bool:
        """Say whether a definition states the node of a census call.

        Args:
            template: The census call.

        Returns:
            Whether a definition states it.

        Example:
            >>> CensusNodes()._stated(NodeTemplate("root.abs", "pl.col('x').abs()"))
            True
        """
        placeholders: list[pl.Expr] = [pl.col(f"__{index}") for index in range(template.inputs())]
        identity = ExpressionTree().of(template.build(placeholders)).identity
        return identity in Definitions.BY_IDENTITY


@dataclass(frozen=True)
class TreeSpace:
    """Hold what a random tree is made of: the nodes, the input outcomes and the truth of polars.

    Example:
        >>> space = TreeSpace.census()
        >>> len(space.outcomes)
        124
    """

    templates: tuple[NodeTemplate, ...]
    outcomes: tuple[Outcome, ...]
    truth: TreeTruth

    @classmethod
    def census(cls) -> TreeSpace:
        """Give the space of the census: its nodes with a definition, its outcomes and its frames.

        Returns:
            The space.

        Example:
            >>> from spec.tools.node_census.outcomes import ColumnOutcome
            >>> TreeSpace.census().truth.runs(Leaf(ColumnOutcome(0, RowForm.PER_ROW), null=False))
            True
        """
        kinds = CensusKinds()
        templates = tuple(CensusNodes().defined())
        outcomes = tuple(OutcomeSpace(kinds).all())
        return cls(templates, outcomes, TreeTruth.example())


class TreeMaker:
    """Build random trees over the census nodes, from one seeded random source.

    A tree of depth $d$ has a node at its root, and a tree of depth $d - 1$
    at its first input. Each other input is a smaller tree or a leaf. Most
    random trees raise, so each inner tree is built again until polars runs
    it, at most `TRIES` times. The root is not filtered, so the check also
    covers the trees that polars refuses.

    Example:
        >>> import random
        >>> maker = TreeMaker(TreeSpace.census(), random.Random(1))
        >>> maker.tree(2).text().count("(") >= 2
        True
    """

    TRIES: ClassVar[int] = 20
    INNER: ClassVar[float] = 0.3
    NULL: ClassVar[float] = 0.1

    def __init__(self, space: TreeSpace, rng: random.Random) -> None:
        """Keep the nodes, the outcomes, the truth of polars and the random source.

        Args:
            space: The nodes, the outcomes and the truth of polars.
            rng: The random source.

        Example:
            >>> import random
            >>> TreeMaker(TreeSpace.census(), random.Random(1))._rng.random() < 1
            True
        """
        self._space = space
        self._rng = rng

    def tree(self, depth: int) -> Branch:
        """Give a random tree of a depth, whose inner trees polars runs.

        Args:
            depth: The depth, at least 1.

        Returns:
            The tree.

        Example:
            >>> import random
            >>> maker = TreeMaker(TreeSpace.census(), random.Random(2))
            >>> isinstance(maker.tree(1), Branch)
            True
        """
        template = self._rng.choice(self._space.templates)
        inputs = [self._input(depth, position) for position in range(template.inputs())]
        return Branch(template, tuple(inputs))

    def _until(self, depth: int, accept: Callable[[Tree], bool]) -> Branch:
        """Give a random tree of a depth that a test accepts, if one comes in `TRIES` tries.

        Args:
            depth: The depth.
            accept: The test, for example that polars runs the tree.

        Returns:
            The tree. After the last try, the last tree, which the test refuses.

        Example:
            >>> import random
            >>> maker = TreeMaker(TreeSpace.census(), random.Random(4))
            >>> maker._until(1, lambda tree: False).leaves() >= 0
            True
        """
        tree = self.tree(depth)
        for _ in range(self.TRIES - 1):
            if accept(tree):
                break
            tree = self.tree(depth)
        return tree

    def _input(self, depth: int, position: int) -> Tree:
        """Give the tree at one input of a node of a depth.

        Args:
            depth: The depth of the node.
            position: The position of the input.

        Returns:
            A smaller tree, always at the first input, or a leaf.

        Example:
            >>> import random
            >>> maker = TreeMaker(TreeSpace.census(), random.Random(5))
            >>> isinstance(maker._input(1, 0), Leaf)
            True
            >>> isinstance(maker._input(2, 0), Branch)
            True
        """
        if depth > 1 and (position == 0 or self._rng.random() < self.INNER):
            return self._until(depth - 1, self._space.truth.runs)
        outcome = self._rng.choice(self._space.outcomes)
        null = outcome.can_be_null() and self._rng.random() < self.NULL
        return Leaf(outcome, null=null)
