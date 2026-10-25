"""Check the options and the shapes in `astrolabe._rules.options` against polars itself.

Each node of the tests has a definition in the vocabulary of the rules, which
states the options that polars gives it. The tests check what those options
imply: the kind of the node in the plan, the list wrap that `agg` gives it,
whether it works value by value, and the row form of its output in a `select`.

The row form of an output can depend on its values, as the length of
`drop_nulls` does. So the tests run each node on two frames of one shape
and different values, and read the row form from the lengths of both runs.
Each frame has two groups of different sizes.
"""

from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import PolarsAnswer, Stopped
from astrolabe._rules.execution import (
    InSelect,
    KernelSelect,
    MinMaxBySelect,
    Selector,
    SliceSelect,
)
from astrolabe._rules.failures import PolarsRefusal, Stop
from astrolabe._rules.options import (
    CastingRules,
    FunctionFlag,
    FunctionOptions,
    Height,
    KernelLength,
    NodeKind,
    RowForm,
    Shape,
)
from astrolabe._rules.supertype import SupertypeFlag

if TYPE_CHECKING:
    from collections.abc import Callable


class Samples:
    """Give two frames of one shape and their inputs of each form, for polars and for the rules.

    The values differ between the two frames, so that a kernel whose length
    the values decide gives two lengths. A literal also takes another value
    in each frame, and another length is `head(2)` in the first frame and
    `head(3)` in the second.

    Example:
        >>> [frame.height for frame in Samples().frames()]
        [5, 5]
    """

    FORMS: ClassVar[dict[str, RowForm]] = {
        "per row": RowForm.PER_ROW,
        "one value": RowForm.ONE_VALUE,
        "other length": RowForm.OTHER_LENGTH,
        "one row": RowForm.ONE_ROW,
        "literal": RowForm.ONE_VALUE,
    }
    LITERALS: ClassVar[tuple[dict[str, object], ...]] = (
        {"x": 1, "y": 1, "b": True, "l": [9, 9]},
        {"x": None, "y": None, "b": False, "l": []},
    )

    def frames(self) -> list[pl.DataFrame]:
        """Give the two frames.

        Returns:
            The frames, each of two groups of sizes 3 and 2.

        Example:
            >>> Samples().frames()[0]["x"].to_list()
            [3, 1, 3, 1, None]
        """
        groups = [1, 1, 1, 2, 2]
        return [
            pl.DataFrame({
                "g": groups,
                "x": [3, 1, 3, 1, None],
                "y": [2, None, 1, 1, 3],
                "b": [True, True, False, True, False],
                "l": [[1, 2], [3], [], [4], [5, 6, 7]],
            }),
            pl.DataFrame({
                "g": groups,
                "x": [None, 2, 2, 5, 5],
                "y": [1, 1, None, 4, 4],
                "b": [False, True, False, False, True],
                "l": [[1], [2, 3], [4], [5], []],
            }),
        ]

    def polars(self, form: str, column: str, sample: int) -> pl.Expr:
        """Give the polars expression of an input of a form, for one frame.

        Args:
            form: The form.
            column: The column that the input reads.
            sample: The number of the frame.

        Returns:
            The expression.

        Example:
            >>> str(Samples().polars("literal", "x", 1))
            'null'
        """
        base = pl.col(column)
        if form == "literal":
            value = self.LITERALS[sample][column]
            return pl.lit(value, dtype=pl.Int64()) if value is None else pl.lit(value)
        builds: dict[str, Callable[[], pl.Expr]] = {
            "per row": lambda: base,
            "one value": base.first,
            "other length": lambda: base.head(2 + sample),
            "one row": lambda: base.head(1),
        }
        return builds[form]()


@dataclass(frozen=True)
class Node:
    """Hold a real node of polars with the shape that its definition gives.

    `columns` holds the column that each input of a varying form reads.
    `extra` holds the heights of the inputs that polars adds after them,
    such as the literal count of `shift(1)`. `kind` is the name of the node
    in the plan, where it is not the value of the kind of the shape.

    Example:
        >>> Nodes().all()["abs"].heights([Height.SCALAR])
        [<Height.SCALAR: 'scalar'>]
    """

    name: str
    shape: Shape
    build: Callable[..., pl.Expr]
    columns: tuple[str, ...] = ("x",)
    extra: tuple[Height, ...] = ()
    kind: str = ""

    def heights(self, inputs: list[Height]) -> list[Height]:
        """Give the heights of all the inputs of the node in the plan.

        Args:
            inputs: The heights of the inputs of a varying form.

        Returns:
            The heights, with those of the inputs that polars adds.

        Example:
            >>> len(Nodes().all()["head 2"].heights([Height.COLUMN]))
            3
        """
        return [*inputs, *self.extra]

    def plan_name(self) -> str:
        """Give the name that the plan of polars gives the node.

        Returns:
            The name.

        Example:
            >>> Nodes().all()["over"].plan_name()
            'Window'
        """
        return self.kind or self.shape.kind.value


class Nodes:
    """Declare the nodes of the tests: for each class of options, a few real nodes of polars.

    Each definition states the options that polars gives the function, or the
    kind of its node in the plan.

    Example:
        >>> Nodes().all()["fill_null"].shape.options.cast_options.supertype_flags is not None
        True
    """

    LITERAL: ClassVar[tuple[Height, ...]] = (Height.SCALAR,)
    SLICE: ClassVar[tuple[Height, ...]] = (Height.SCALAR, Height.SCALAR)

    def all(self) -> dict[str, Node]:
        """Give the nodes, by name.

        Returns:
            The nodes.

        Example:
            >>> Nodes().all()["min_by"].shape.options.returns_scalar()
            True
        """
        nodes = [*self.single(), *self.multiple()]
        return {node.name: node for node in nodes}

    def single(self) -> list[Node]:
        """Give the nodes of one input of a varying form.

        Returns:
            The nodes.

        Example:
            >>> Nodes().single()[0].name
            'abs'
        """
        elementwise = FunctionOptions.elementwise()
        preserving = FunctionOptions.length_preserving()
        order = FunctionFlag.NON_ORDER_PRODUCING
        unordered = order | FunctionFlag.NON_ORDER_OBSERVING | FunctionFlag.TERMINATES_INPUT_ORDER
        unique = FunctionOptions.groupwise().with_flags(lambda flags: flags | unordered)
        drop = FunctionFlag.ALLOW_EMPTY_INPUTS | order
        drop_nulls = FunctionOptions.row_separable().flag(drop)
        null_count = FunctionOptions.aggregation().flag(FunctionFlag.NON_ORDER_OBSERVING)
        shrinks = KernelLength.shrinks()
        return [
            Node("abs", Shape.function(elementwise), lambda x: x.abs()),
            Node("is_null", Shape.function(elementwise), lambda x: x.is_null()),
            Node("cast", Shape.of(NodeKind.CAST), lambda x: x.cast(pl.Int8)),
            Node("sum", Shape.of(NodeKind.AGG), lambda x: x.sum()),
            Node("first", Shape.of(NodeKind.AGG), lambda x: x.first()),
            Node("implode", Shape.of(NodeKind.AGG), lambda x: x.implode()),
            Node("null_count", Shape.function(null_count), lambda x: x.null_count()),
            Node("cum_sum", Shape.function(preserving), lambda x: x.cum_sum()),
            Node("rank", Shape.function(preserving), lambda x: x.rank()),
            Node("shift", Shape.function(preserving), lambda x: x.shift(1), extra=self.LITERAL),
            Node("unique", Shape.function(unique, shrinks), lambda x: x.unique()),
            Node("mode", Shape.function(unique, shrinks), lambda x: x.mode()),
            Node(
                "gather_every",
                Shape.function(FunctionOptions.groupwise(), shrinks),
                lambda x: x.gather_every(2),
            ),
            Node("drop_nulls", Shape.function(drop_nulls), lambda x: x.drop_nulls()),
            Node(
                "head 2",
                Shape(NodeKind.SLICE, length=KernelLength.slice(0, 2)),
                lambda x: x.head(2),
                extra=self.SLICE,
            ),
            Node(
                "head 1",
                Shape(NodeKind.SLICE, length=KernelLength.slice(0, 1)),
                lambda x: x.head(1),
                extra=self.SLICE,
            ),
            Node(
                "tail 1",
                Shape(NodeKind.SLICE, length=KernelLength.slice(-1, 1)),
                lambda x: x.tail(1),
                extra=self.SLICE,
            ),
            Node("sort", Shape.of(NodeKind.SORT), lambda x: x.sort()),
            Node(
                "explode",
                Shape.of(NodeKind.EXPLODE),
                lambda x: x.explode(empty_as_null=True),
                columns=("l",),
            ),
            Node("get", Shape.gather(returns_scalar=True), lambda x: x.get(0), extra=self.LITERAL),
            Node("over", Shape.over(), lambda x: x.sum().over("g"), kind="Window"),
        ]

    def multiple(self) -> list[Node]:
        """Give the nodes of several inputs of a varying form.

        Returns:
            The nodes.

        Example:
            >>> Nodes().multiple()[0].columns
            ('x', 'y')
        """
        elementwise = FunctionOptions.elementwise()
        fill = elementwise.with_supertyping(SupertypeFlag.default())
        search = FunctionOptions.groupwise().with_supertyping(SupertypeFlag(0))
        pair = ("x", "y")
        return [
            Node("add", Shape.of(NodeKind.BINARY), operator.add, columns=pair),
            Node("fill_null", Shape.function(fill), lambda x, y: x.fill_null(y), columns=pair),
            Node(
                "clip",
                Shape.function(elementwise, KernelLength.first()),
                lambda x, y: x.clip(y),
                columns=pair,
            ),
            Node(
                "min_by",
                Shape.function(FunctionOptions.aggregation()),
                lambda x, y: x.min_by(y),
                columns=pair,
            ),
            Node(
                "search_sorted",
                Shape.function(search, KernelLength.of_input(1)),
                lambda x, y: x.search_sorted(y),
                columns=pair,
            ),
            Node(
                "sort_by",
                Shape.of(NodeKind.SORT_BY),
                lambda x, y: x.sort_by(y),
                columns=pair,
            ),
            Node(
                "filter",
                Shape.of(NodeKind.FILTER),
                lambda x, b: x.filter(b),
                columns=("x", "b"),
            ),
            Node(
                "gather",
                Shape.gather(returns_scalar=False),
                lambda x, y: x.gather(y.fill_null(0) * 0),
                columns=pair,
            ),
            Node(
                "when",
                Shape.of(NodeKind.TERNARY),
                lambda b, x, y: pl.when(b).then(x).otherwise(y),
                columns=("b", "x", "y"),
            ),
        ]


class Cases:
    """Give each node with each combination of the forms of its inputs.

    Example:
        >>> Cases().ident(Cases().all()[0])
        'abs(per row)'
    """

    def all(self) -> list[tuple[Node, tuple[str, ...]]]:
        """Give the cases.

        Returns:
            The cases: a node and the form of each input.

        Example:
            >>> len(Cases().all()) > 300
            True
        """
        cases: list[tuple[Node, tuple[str, ...]]] = []
        for node in Nodes().all().values():
            for forms in itertools.product(Samples.FORMS, repeat=len(node.columns)):
                cases.append((node, forms))
        return cases

    def single(self) -> list[tuple[Node, tuple[str, ...]]]:
        """Give the cases of the nodes of one input of a varying form.

        Returns:
            The cases.

        Example:
            >>> Cases().single()[-1][0].name
            'over'
        """
        return [case for case in self.all() if len(case[0].columns) == 1]

    def ident(self, case: tuple[Node, tuple[str, ...]]) -> str:
        """Name a case for the report of pytest.

        Args:
            case: The node and the forms of its inputs.

        Returns:
            The name.

        Example:
            >>> Cases().ident((Nodes().all()["add"], ("literal", "one row")))
            'add(literal, one row)'
        """
        node, forms = case
        return f"{node.name}({', '.join(forms)})"


class Plans:
    """Read what the planner of polars gives a case.

    Example:
        >>> Plans().wrapped(Nodes().all()["sum"], ("per row",))
        False
    """

    def expression(self, node: Node, forms: tuple[str, ...], sample: int = 0) -> pl.Expr:
        """Build the polars expression of a case for one frame.

        Args:
            node: The node.
            forms: The form of each input.
            sample: The number of the frame.

        Returns:
            The expression, named `o`.

        Example:
            >>> str(Plans().expression(Nodes().all()["abs"], ("one row",)))
            'col("x").slice(offset=dyn int: 0, length=dyn int: 1).abs().alias("o")'
        """
        inputs = []
        for form, column in zip(forms, node.columns, strict=True):
            inputs.append(Samples().polars(form, column, sample))
        return node.build(*inputs).alias("o")

    def wrapped(self, node: Node, forms: tuple[str, ...]) -> bool:
        """Say whether the planner of `agg` wraps the values of each group in a list.

        The planner wraps where the type of `agg` is the list of the type
        of one row in a `select`.

        Args:
            node: The node.
            forms: The form of each input.

        Returns:
            Whether it wraps.

        Example:
            >>> Plans().wrapped(Nodes().all()["implode"], ("per row",))
            False
        """
        expression = self.expression(node, forms)
        lazy = Samples().frames()[0].lazy()
        row = lazy.select(expression).collect_schema()["o"]
        agg = lazy.group_by("g").agg(expression).collect_schema()["o"]
        return agg == pl.List(row)

    def kind(self, node: Node) -> str:
        """Give the name of the root of the plan of a node over a column.

        Args:
            node: The node.

        Returns:
            The name.

        Example:
            >>> Plans().kind(Nodes().all()["head 1"])
            'Slice'
        """
        forms = ("per row",) * len(node.columns)
        query = Samples().frames()[0].lazy().select(self.expression(node, forms))
        flags = pl.QueryOptFlags.none()
        plan = query._ldf.with_optimizations(flags._pyoptflags)  # ruff: ignore[private-member-access]
        traverser = plan.visit()
        root = traverser.view_expression(traverser.view_current_node().expr[0].node)
        return type(root).__name__

    def form(self, node: Node, forms: tuple[str, ...]) -> object:
        """Read the row form of a case in a `select` from the runs on both frames.

        One value is a value that the planner knows to be one value. Any
        other output has one value per row where each run has the height of
        its frame, one row where each run has one row, and another length
        otherwise.

        Args:
            node: The node.
            forms: The form of each input.

        Returns:
            The row form, or how a run stops where it does not run.

        Example:
            >>> Plans().form(Nodes().all()["drop_nulls"], ("one value",))
            <RowForm.OTHER_LENGTH: 'other length'>
        """
        heights: list[int] = []
        for sample, frame in enumerate(Samples().frames()):
            query = frame.lazy().select(self.expression(node, forms, sample))
            outcome = PolarsAnswer().result(query.collect)
            if isinstance(outcome, Stopped):
                return outcome
            heights.append(outcome.height)
        if not self.wrapped(node, forms):
            return RowForm.ONE_VALUE
        if all(height == 5 for height in heights):
            return RowForm.PER_ROW
        return RowForm.ONE_ROW if set(heights) == {1} else RowForm.OTHER_LENGTH


@pytest.mark.parametrize("case", Cases().all(), ids=Cases().ident)
def test_agg_wraps_a_node_that_the_planner_does_not_know_to_be_one_value(
    case: tuple[Node, tuple[str, ...]],
) -> None:
    """Check the list wrap of `agg`, which the projection height decides, against polars.

    Args:
        case: The node and the form of each input.
    """
    node, forms = case
    heights = [Samples.FORMS[form].height() for form in forms]
    assert node.shape.wraps_in_agg(node.heights(heights)) == Plans().wrapped(node, forms)


@pytest.mark.parametrize("case", Cases().all(), ids=Cases().ident)
def test_the_output_of_a_select_has_the_row_form_of_the_rules(
    case: tuple[Node, tuple[str, ...]],
) -> None:
    """Check the row form of the output of a node in a `select` against polars.

    The row form assumes that inputs of other lengths agree, so a case
    where polars raises must raise in the run of the rules too, on the
    lengths of the frame of the tests.

    Args:
        case: The node and the form of each input.
    """
    node, forms = case
    observed = Plans().form(node, forms)
    if isinstance(observed, Stopped):
        assert Lengths().raises(node, forms), observed
        return
    rows = [Samples.FORMS[form] for form in forms]
    extra = [RowForm.ONE_VALUE] * len(node.extra)
    assert node.shape.form([*rows, *extra]) == observed


class Lengths:
    """Say whether the run of the rules raises for a case in a `select` of the first frame.

    Example:
        >>> Lengths().raises(Nodes().all()["add"], ("per row", "other length"))
        True
    """

    INPUTS: ClassVar[dict[str, InSelect]] = {
        "per row": InSelect(5, Height.COLUMN),
        "one value": InSelect(1, Height.SCALAR),
        "other length": InSelect(2, Height.UNKNOWN),
        "one row": InSelect(1, Height.UNKNOWN),
        "literal": InSelect(1, Height.SCALAR),
    }

    def raises(self, node: Node, forms: tuple[str, ...]) -> bool:
        """Say whether the rules raise.

        A node whose run this vocabulary does not translate in `agg`, such
        as a filter, still has the rule of its kernel in a `select`.

        Args:
            node: The node.
            forms: The form of each input.

        Returns:
            Whether they raise.

        Example:
            >>> Lengths().raises(Nodes().all()["add"], ("per row", "one row"))
            False
        """
        inputs = [self.INPUTS[form] for form in forms]
        inputs.extend([InSelect.literal()] * len(node.extra))
        selects: dict[str, Callable[[Shape], Selector]] = {
            "min_by": MinMaxBySelect,
            "head 2": SliceSelect,
            "head 1": SliceSelect,
            "tail 1": SliceSelect,
        }
        select = selects.get(node.name, KernelSelect)(node.shape)
        try:
            select.select(inputs)
        except PolarsRefusal:
            return True
        return False


@pytest.mark.parametrize("node", Nodes().all().values(), ids=operator.attrgetter("name"))
def test_a_declared_node_has_the_kind_of_its_node_in_the_plan(node: Node) -> None:
    """Check the kind of each declared node against the root of its plan in polars.

    Args:
        node: The node.
    """
    assert Plans().kind(node) == node.plan_name()


class Windows:
    """Read whether a node works value by value, from the `over` of polars in `agg`.

    An `over` in `agg` needs a partition key whose every node is elementwise.
    So the node is elementwise exactly where `pl.col("x").over(node)` runs in
    `agg`.

    Example:
        >>> Windows().elementwise(pl.col("y").abs())
        True
    """

    def elementwise(self, key: pl.Expr) -> bool:
        """Say whether polars takes the key of an `over` in `agg`.

        Args:
            key: The partition key.

        Returns:
            Whether the `over` runs.

        Raises:
            AssertionError: The run raises for another cause.

        Example:
            >>> Windows().elementwise(pl.col("y").cum_sum())
            False
        """
        frame = Samples().frames()[0].lazy()
        query = frame.group_by("g").agg(pl.col("x").over(key).alias("o"))
        outcome = PolarsAnswer().result(lambda: query.collect().height)
        if not isinstance(outcome, Stopped):
            return True
        if outcome != Stopped(Stop.RAISES, "InvalidOperationError"):
            raise AssertionError(outcome)
        return False


@pytest.mark.parametrize("node", Nodes().all().values(), ids=operator.attrgetter("name"))
def test_an_over_in_agg_takes_a_key_that_the_rules_say_is_elementwise(node: Node) -> None:
    """Check the flag `elementwise` of each declared node against the `over` of polars.

    The key reads plain columns, which are elementwise, so the key is
    elementwise exactly where its root is.

    Args:
        node: The node.
    """
    forms = ("per row",) * len(node.columns)
    key = Plans().expression(node, forms)
    assert node.shape.is_elementwise() == Windows().elementwise(key)


def test_a_tree_is_elementwise_where_each_of_its_nodes_is() -> None:
    """Check against polars that a chain of two nodes is elementwise where each node is."""
    x = pl.col("y")
    windows = Windows()
    assert windows.elementwise((x + 1).abs())
    assert not windows.elementwise(x.abs().cum_sum())
    assert not windows.elementwise(x.cum_sum().abs())
    assert windows.elementwise(pl.lit(1) + x)


def test_the_constructors_set_the_flags_of_the_source() -> None:
    """Check that the constructors and builders of `FunctionOptions` set the flags of polars."""
    both = FunctionFlag.ROW_SEPARABLE | FunctionFlag.LENGTH_PRESERVING
    assert FunctionOptions.elementwise().flags == both
    assert FunctionOptions.row_separable().flags == FunctionFlag.ROW_SEPARABLE
    assert FunctionOptions.length_preserving().flags == FunctionFlag.LENGTH_PRESERVING
    assert FunctionOptions.groupwise().flags == FunctionFlag(0)
    assert FunctionOptions.aggregation().flags == FunctionFlag.RETURNS_SCALAR
    supertype = FunctionOptions.groupwise().with_supertyping(SupertypeFlag.default())
    assert supertype.cast_options == CastingRules.cast_to_supertypes()
    assert FunctionOptions().check_lengths
