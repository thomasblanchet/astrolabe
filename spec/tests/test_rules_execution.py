"""Check the rules of the run of polars in `astrolabe._rules.execution` against polars itself.

The rules give the number of rows of a node in a `select`, and the shape of
its column in `group_by().agg()`: a list for each group, with its length,
or one value for each group. They also say where polars raises, and with
which error. Each test here runs a real node of polars over inputs of each
row form, and compares what polars gives with what the rules give.

The frame has two groups of different sizes, 3 and 2, because one group
hides the checks of polars on the lengths of the groups. Its values make
each kernel give the most rows that it can give: the values of a group
differ, and each list holds one value. A `filter` keeps the rows of its
mask, so it gives at most the rows of the rules, and each index of a
`gather` is zero, which every input holds.

A kernel that can fail on a value runs once for each group where the groups
drop some values, so a dropped value cannot raise. The column `f` holds a
value that a strict cast to `UInt8` and a `replace_strict` refuse, in a row
that every input form but a whole column drops, and those kernels read only
those forms.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import multiprocessing
import multiprocessing.queues
import operator
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import pytest
from polars.io.plugins import register_io_source

from spec.tests.rule_probes import PolarsAnswer, RuleAnswer, Stopped
from astrolabe._rules.definitions.aggregations import Aggregations
from astrolabe._rules.definitions.length_preserving import LengthPreserving
from astrolabe._rules.definitions.many_inputs import ManyInputs
from astrolabe._rules.definitions.node_kinds import NodeKinds
from astrolabe._rules.definitions.node_options import NodeOptions
from astrolabe._rules.definitions.other_lengths import OtherLengths
from astrolabe._rules.dtypes import DType
from astrolabe._rules.execution import (
    AggOutput,
    BinaryFold,
    GroupBy,
    InGroups,
    InSelect,
    Kernel,
    PhysicalExpr,
    Pushdown,
    QueryFlags,
    Run,
    SliceInput,
    SlicePushdown,
)
from astrolabe._rules.failures import PolarsRefusal, Stop
from astrolabe._rules.operators import Operator
from astrolabe._rules.releases import PolarsRelease
from astrolabe._rules.options import (
    RowForm,
    FunctionFlag,
    FunctionOptions,
    Height,
    KernelLength,
    NodeKind,
    Shape,
)
from astrolabe._rules.node_evaluation import ColumnLeaf, Refused
from astrolabe._rules.supertype import SupertypeFlag
from astrolabe._rules.tree import ExpressionTree
from astrolabe._rules.tree_evaluation import TreeEvaluation
from astrolabe._rules.type_functions import Operand

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from astrolabe._rules.definitions.definition import Definition


class Slices:
    """Hold the shapes of `head(2)` and `head(1)`, which give the forms other length and one row.

    Example:
        >>> inputs = [InSelect.column(5), InSelect.literal(), InSelect.literal()]
        >>> Run(Slices.HEAD_1).select(inputs).length
        1
    """

    HEAD_2: ClassVar[Shape] = Shape(NodeKind.SLICE, length=KernelLength.slice(0, 2))
    HEAD_1: ClassVar[Shape] = Shape(NodeKind.SLICE, length=KernelLength.slice(0, 1))


class Frames:
    """Give the frame of the tests and the inputs of each form, for polars and for the rules.

    The six forms are the four row forms of the checker, a literal, and a
    group of one value: `.first().cum_sum()` holds one value in each group,
    as `.first()` does, but a node rebuilt its groups.

    Example:
        >>> str(Frames().polars("one row", "x"))
        'col("x").slice(offset=dyn int: 0, length=dyn int: 1)'
    """

    FORMS: ClassVar[tuple[str, ...]] = (
        "per row",
        "one value",
        "other length",
        "one row",
        "literal",
        "group",
    )
    KEPT: ClassVar[tuple[str, ...]] = FORMS[1:]
    SIZES: ClassVar[tuple[int, ...]] = (3, 2)
    LITERALS: ClassVar[dict[str, object]] = {"b": True, "l": [9], "i": 0}

    def __init__(self) -> None:
        """Build the frame and its groups.

        Example:
            >>> Frames().frame.height
            5
        """
        self.frame = pl.DataFrame({
            "g": [1, 1, 1, 2, 2],
            "x": [3, 1, 2, 5, 4],
            "y": [1, 2, 3, 4, 5],
            "b": [True, False, True, True, False],
            "l": [[1], [2], [3], [4], [5]],
            "i": [0, 0, 0, 0, 0],
            "f": [3, 1, -1, 5, 4],
        })
        self.groups = GroupBy(self.SIZES)

    def polars(self, form: str, column: str) -> pl.Expr:
        """Give the polars expression of an input of a form.

        A group of one value of a `Boolean` uses `cum_max`, which keeps the type.

        Args:
            form: The form.
            column: The column that the input reads.

        Returns:
            The expression.

        Example:
            >>> str(Frames().polars("group", "x"))
            'col("x").first().cum_sum()'
        """
        base = pl.col(column)
        builds: dict[str, Callable[[], pl.Expr]] = {
            "per row": lambda: base,
            "one value": base.first,
            "other length": lambda: base.head(2),
            "one row": lambda: base.head(1),
            "literal": lambda: pl.lit(self.LITERALS.get(column, 1)),
            "group": lambda: base.first().cum_max() if column == "b" else base.first().cum_sum(),
        }
        return builds[form]()

    def selected(self, form: str) -> InSelect:
        """Give the input of a form in a `select`, from the rules.

        Args:
            form: The form.

        Returns:
            The input.

        Example:
            >>> Frames().selected("other length")
            InSelect(length=2, height=<Height.UNKNOWN: 'unknown'>, decided=False, bounded=True)
        """
        column = InSelect.column(self.frame.height)
        literal = InSelect.literal()
        first = Run(Shape.of(NodeKind.AGG)).select([column])
        cum_sum = Run(Shape.function(FunctionOptions.length_preserving()))
        builds: dict[str, Callable[[], InSelect]] = {
            "per row": lambda: column,
            "one value": lambda: first,
            "other length": lambda: Run(Slices.HEAD_2).select([column, literal, literal]),
            "one row": lambda: Run(Slices.HEAD_1).select([column, literal, literal]),
            "literal": lambda: literal,
            "group": lambda: cum_sum.select([first]),
        }
        return builds[form]()

    def grouped(self, form: str) -> InGroups:
        """Give the input of a form over the groups of `agg`, from the rules.

        Args:
            form: The form.

        Returns:
            The input.

        Example:
            >>> Frames().grouped("group").groups
            (1, 1)
        """
        frame = self.groups
        column = frame.column()
        literal = frame.literal()
        first = Run(Shape.of(NodeKind.AGG)).agg([column], frame)
        cum_sum = Run(Shape.function(FunctionOptions.length_preserving()))
        builds: dict[str, Callable[[], InGroups]] = {
            "per row": lambda: column,
            "one value": lambda: first,
            "other length": lambda: Run(Slices.HEAD_2).agg([column, literal, literal], frame),
            "one row": lambda: Run(Slices.HEAD_1).agg([column, literal, literal], frame),
            "literal": lambda: literal,
            "group": lambda: cum_sum.agg([first], frame),
        }
        return builds[form]()


@dataclass(frozen=True)
class Node:
    """Hold a real node of polars with its definition in the vocabulary of the rules.

    `columns` holds the column that each input of a varying form reads.
    `extra` is the number of literal inputs that polars adds after them,
    such as the count of `shift(1)`. `exact` says that the kernel gives an
    exact length on the frame of the tests. Where it does not, polars gives
    at most the length of the rules. `forms` holds the forms that the inputs
    take: a list column has no group of one value that keeps its type.

    Example:
        >>> Cases().nodes()["abs"].columns
        ('x',)
    """

    name: str
    shape: Shape
    build: Callable[..., pl.Expr]
    columns: tuple[str, ...] = ("x",)
    extra: int = 0
    exact: bool = True
    kernel: Kernel = field(default_factory=Kernel)
    forms: tuple[str, ...] = Frames.FORMS

    def run(self) -> Run:
        """Give the run of the node in the rules.

        Returns:
            The run.

        Example:
            >>> Cases().nodes()["sum"].run().select([InSelect.column(5)]).length
            1
        """
        return Run(self.shape, self.kernel)


class Cases:
    """Build the declared nodes and the cases of the tests: each node with a form for each input.

    Each definition states the options that polars gives the function, or the
    kind of its node in the plan.

    Example:
        >>> len(Cases().all()) > 400
        True
    """

    def nodes(self) -> dict[str, Node]:
        """Give the declared nodes, by name.

        Returns:
            The nodes.

        Example:
            >>> Cases().nodes()["head 2"].extra
            2
        """
        elementwise = FunctionOptions.elementwise()
        preserving = FunctionOptions.length_preserving()
        fill = elementwise.with_supertyping(SupertypeFlag.default())
        order = FunctionFlag.NON_ORDER_PRODUCING | FunctionFlag.NON_ORDER_OBSERVING
        mode = FunctionOptions.groupwise().with_flags(lambda flags: flags | order)
        search = FunctionOptions.groupwise().with_supertyping(SupertypeFlag(0))
        tail = Shape(NodeKind.SLICE, length=KernelLength.slice(-1, 1))
        implode = Kernel(physical=PhysicalExpr.IMPLODE)
        min_by = Kernel(physical=PhysicalExpr.MIN_MAX_BY)
        pair = ("x", "y")
        nodes = [
            Node("abs", Shape.function(elementwise), lambda x: x.abs()),
            Node("cast", Shape.of(NodeKind.CAST), lambda x: x.cast(pl.Int8)),
            Node("sum", Shape.of(NodeKind.AGG), lambda x: x.sum()),
            Node("first", Shape.of(NodeKind.AGG), lambda x: x.first()),
            Node("product", Shape.function(FunctionOptions.aggregation()), lambda x: x.product()),
            Node("cum_sum", Shape.function(preserving), lambda x: x.cum_sum()),
            Node("rank", Shape.function(preserving), lambda x: x.rank()),
            Node("shift", Shape.function(preserving), lambda x: x.shift(1), extra=1),
            Node("rolling_mean", Shape.function(preserving), lambda x: x.rolling_mean(2)),
            Node(
                "gather_every",
                Shape.function(FunctionOptions.groupwise(), KernelLength.shrinks()),
                lambda x: x.gather_every(2),
                exact=False,
            ),
            Node("mode", Shape.function(mode, KernelLength.shrinks()), lambda x: x.mode()),
            Node(
                "explode",
                Shape.of(NodeKind.EXPLODE),
                lambda x: x.explode(empty_as_null=True),
                columns=("l",),
                forms=Frames.FORMS[:-1],
            ),
            Node("head 2", Slices.HEAD_2, lambda x: x.head(2), extra=2),
            Node("head 1", Slices.HEAD_1, lambda x: x.head(1), extra=2),
            Node("tail 1", tail, lambda x: x.tail(1), extra=2),
            Node("sort", Shape.of(NodeKind.SORT), lambda x: x.sort()),
            Node("implode", Shape.of(NodeKind.AGG), lambda x: x.implode(), kernel=implode),
            Node("add", Shape.of(NodeKind.BINARY), operator.add, columns=pair),
            Node("fill_null", Shape.function(fill), lambda x, y: x.fill_null(y), columns=pair),
            Node(
                "clip",
                Shape.function(elementwise, KernelLength.first()),
                lambda x, y: x.clip(y),
                columns=pair,
            ),
            Node(
                "repeat_by",
                Shape.function(elementwise),
                lambda x, y: x.repeat_by(y),
                columns=pair,
            ),
            Node(
                "search_sorted",
                Shape.function(search, KernelLength.of_input(1)),
                lambda x, y: x.search_sorted(y),
                columns=pair,
            ),
            Node(
                "min_by",
                Shape.function(FunctionOptions.aggregation()),
                lambda x, y: x.min_by(y),
                columns=pair,
                kernel=min_by,
            ),
            Node(
                "when",
                Shape.of(NodeKind.TERNARY),
                lambda b, x, y: pl.when(b).then(x).otherwise(y),
                columns=("b", "x", "y"),
            ),
            self._defined(
                "filter",
                NodeKinds.FILTER,
                lambda x, b: x.filter(b),
                ("x", "b"),
                exact=False,
            ),
            self._defined("sort_by", NodeKinds.SORT_BY, lambda x, y: x.sort_by(y), pair),
            self._defined("gather", NodeKinds.GATHER, lambda x, i: x.gather(i), ("x", "i")),
            self._defined("unique", OtherLengths.UNIQUE, lambda x: x.unique(), exact=False),
            self._defined("drop_nulls", OtherLengths.DROP_NULLS, lambda x: x.drop_nulls()),
            self._defined("reverse", LengthPreserving.REVERSE, lambda x: x.reverse()),
            self._defined(
                "forward fill",
                LengthPreserving.FILL_NULL_WITH_STRATEGY,
                lambda x: x.fill_null(strategy="forward"),
            ),
            self._defined("null_count", Aggregations.NULL_COUNT, lambda x: x.null_count()),
            self._defined("skew", Aggregations.SKEW, lambda x: x.skew()),
            self._defined("any", Aggregations.ANY, lambda b: b.any(), ("b",)),
            self._defined("bitwise_and", Aggregations.BITWISE_AND, lambda x: x.bitwise_and()),
            self._defined("kurtosis", Aggregations.KURTOSIS, lambda x: x.kurtosis()),
            self._defined("has_nulls", Aggregations.HAS_NULLS, lambda x: x.has_nulls()),
            self._defined("all", Aggregations.ALL, lambda b: b.all(), ("b",)),
            self._defined(
                "backward fill",
                LengthPreserving.FILL_NULL_WITH_STRATEGY,
                lambda x: x.fill_null(strategy="backward"),
            ),
            self._defined("drop_nans", OtherLengths.DROP_NANS, lambda x: x.drop_nans()),
            replace(
                self._defined("quantile", Aggregations.QUANTILE, lambda x: x.quantile(0.5)),
                extra=1,
            ),
            Node(
                "strict cast",
                Shape.of(NodeKind.CAST),
                lambda f: f.cast(pl.UInt8, strict=True),
                columns=("f",),
                kernel=Kernel(fallible=True, strict=True),
                forms=Frames.KEPT,
            ),
            replace(
                self._defined(
                    "replace_strict",
                    ManyInputs.REPLACE_STRICT,
                    lambda f: f.replace_strict([3, 1, 5, 4], [30, 10, 50, 40]),
                    ("f",),
                ),
                extra=2,
                kernel=Kernel(fallible=True),
                forms=Frames.KEPT,
            ),
        ]
        return {node.name: node for node in nodes}

    def _defined(
        self,
        name: str,
        definition: Definition,
        build: Callable[..., pl.Expr],
        columns: tuple[str, ...] = ("x",),
        *,
        exact: bool = True,
    ) -> Node:
        """Give a node with the shape and the physical expression that its definition reads.

        Args:
            name: The name of the node.
            definition: The definition of the node in the rules.
            build: The call of polars, with one expression for each input.
            columns: The column that each input reads.
            exact: Whether the kernel gives an exact length on the frame of the tests.

        Returns:
            The node.

        Example:
            >>> from astrolabe._rules.definitions.length_preserving import LengthPreserving
            >>> node = Cases()._defined("reverse", LengthPreserving.REVERSE, lambda x: x.reverse())
            >>> node.kernel.physical
            <PhysicalExpr.REGROUP: 'regroup'>
        """
        expression = build(*(pl.col(column) for column in columns))
        tree = json.loads(expression.meta.serialize(format="json"))
        resolved = definition.at(NodeOptions(tree))
        kernel = Kernel(physical=resolved.physical_expr())
        return Node(name, resolved.shape(), build, columns=columns, exact=exact, kernel=kernel)

    def all(self) -> list[tuple[Node, tuple[str, ...]]]:
        """Give every node with every combination of the forms of its inputs.

        Returns:
            The cases.

        Example:
            >>> Cases().all()[0][1]
            ('per row',)
        """
        cases: list[tuple[Node, tuple[str, ...]]] = []
        for node in self.nodes().values():
            for forms in itertools.product(node.forms, repeat=len(node.columns)):
                cases.append((node, forms))
        return cases

    def ident(self, case: tuple[Node, tuple[str, ...]]) -> str:
        """Name a case for the report of pytest.

        Args:
            case: The case: a node and the forms of its inputs.

        Returns:
            The name.

        Example:
            >>> Cases().ident((Cases().nodes()["add"], ("per row", "literal")))
            'add(per row, literal)'
        """
        node, forms = case
        return f"{node.name}({', '.join(forms)})"


class Rules:
    """Give what the rules say a case gives, in a `select` and in `agg`.

    Example:
        >>> Rules(Frames()).select(Cases().nodes()["add"], ("per row", "other length"))
        raises ShapeError
    """

    def __init__(self, frames: Frames) -> None:
        """Keep the frame of the tests.

        Args:
            frames: The frame and its inputs.

        Example:
            >>> Rules(Frames())._frames.SIZES
            (3, 2)
        """
        self._frames = frames

    def select(self, node: Node, forms: tuple[str, ...]) -> object:
        """Give the number of rows of the node in a `select`, or how the rules say it stops.

        Args:
            node: The node.
            forms: The form of each input.

        Returns:
            The number of rows, or a `Stopped`.

        Example:
            >>> Rules(Frames()).select(Cases().nodes()["head 2"], ("per row",))
            2
        """
        inputs = self._selected(node, forms)
        return RuleAnswer().result(lambda: node.run().select(inputs).length)

    def decided(self, node: Node, forms: tuple[str, ...]) -> bool:
        """Say whether the rules let the values decide the number of rows of the node in a `select`.

        Such a number is the most rows that polars gives, and the data can give fewer.

        Args:
            node: The node.
            forms: The form of each input.

        Returns:
            Whether they do. A node that the rules refuse gives no number of rows.

        Example:
            >>> Rules(Frames()).decided(Cases().nodes()["head 2"], ("per row",))
            False
        """
        inputs = self._selected(node, forms)
        answer = RuleAnswer().result(lambda: node.run().select(inputs).decided)
        return answer is True

    def _selected(self, node: Node, forms: tuple[str, ...]) -> list[InSelect]:
        """Give the inputs of the node in a `select`, from the rules.

        Args:
            node: The node.
            forms: The form of each input.

        Returns:
            The inputs, with the literal inputs that polars adds after them.

        Example:
            >>> len(Rules(Frames())._selected(Cases().nodes()["head 2"], ("per row",)))
            3
        """
        inputs = [self._frames.selected(form) for form in forms]
        inputs.extend([InSelect.literal()] * node.extra)
        return inputs

    def agg(self, node: Node, forms: tuple[str, ...]) -> object:
        """Give the column of the node in `agg`, or how the rules say it stops.

        Args:
            node: The node.
            forms: The form of each input.

        Returns:
            The column, or a `Stopped`.

        Example:
            >>> Rules(Frames()).agg(Cases().nodes()["sum"], ("per row",))
            AggOutput(lists=None)
        """
        frame = self._frames.groups
        inputs = [self._frames.grouped(form) for form in forms]
        inputs.extend([frame.literal()] * node.extra)
        heights = [self._frames.selected(form).height for form in forms]
        heights.extend([Height.SCALAR] * node.extra)
        wrapped = node.shape.wraps_in_agg(heights)
        return RuleAnswer().result(
            lambda: frame.output(node.run().agg(inputs, frame), wrapped=wrapped),
        )


class Polars:
    """Give what polars gives for a case, in a `select` and in `agg`.

    Example:
        >>> Polars(Frames()).select(Cases().nodes()["add"], ("per row", "other length"))
        raises ShapeError
    """

    def __init__(self, frames: Frames) -> None:
        """Keep the frame of the tests.

        Args:
            frames: The frame and its inputs.

        Example:
            >>> Polars(Frames())._frames.frame.width
            7
        """
        self._frames = frames

    def expression(self, node: Node, forms: tuple[str, ...]) -> pl.Expr:
        """Build the polars expression of a case.

        Args:
            node: The node.
            forms: The form of each input.

        Returns:
            The expression, named `o`.

        Example:
            >>> str(Polars(Frames()).expression(Cases().nodes()["abs"], ("one value",)))
            'col("x").first().abs().alias("o")'
        """
        pairs = zip(forms, node.columns, strict=True)
        inputs = list(itertools.starmap(self._frames.polars, pairs))
        return node.build(*inputs).alias("o")

    def select(self, node: Node, forms: tuple[str, ...]) -> object:
        """Give the number of rows of the node in a `select`, or how polars stops.

        Args:
            node: The node.
            forms: The form of each input.

        Returns:
            The number of rows, or a `Stopped`.

        Example:
            >>> Polars(Frames()).select(Cases().nodes()["head 2"], ("per row",))
            2
        """
        lazy = self._frames.frame.lazy().select(self.expression(node, forms))
        return PolarsAnswer().result(lambda: lazy.collect(engine="in-memory").height)

    def agg(self, node: Node, forms: tuple[str, ...]) -> object:
        """Give the column of the node in `agg`, or how polars stops.

        The column holds a list for each group where its data type is the
        list of the planned type of one row.

        Args:
            node: The node.
            forms: The form of each input.

        Returns:
            The column, or a `Stopped`.

        Example:
            >>> Polars(Frames()).agg(Cases().nodes()["head 2"], ("per row",))
            AggOutput(lists=(2, 2))
        """
        expression = self.expression(node, forms)
        lazy = self._frames.frame.lazy()
        row = lazy.select(expression).collect_schema()["o"]
        query = lazy.group_by("g", maintain_order=True).agg(expression)
        return PolarsAnswer().result(
            lambda: self._column(query.collect(engine="in-memory")["o"], row)
        )

    def _column(self, column: pl.Series, row: pl.DataType) -> AggOutput:
        """Read the shape of the column of `agg`.

        Args:
            column: The column.
            row: The planned type of one row of the expression.

        Returns:
            The shape.

        Example:
            >>> Polars(Frames())._column(pl.Series([[1], [2, 3]]), pl.Int64())
            AggOutput(lists=(1, 2))
        """
        if column.dtype != pl.List(row):
            return AggOutput(None)
        return AggOutput(tuple(column.list.len().to_list()))


class Agreement:
    """Compare what the rules give with what polars gives.

    An exact kernel must give the same rows. A kernel whose length the
    values decide gives at most the rows of the rules.

    Example:
        >>> Agreement(exact=False).holds(AggOutput((3, 2)), AggOutput((2, 1)))
        True
    """

    def __init__(self, *, exact: bool) -> None:
        """Keep whether the kernel gives an exact length.

        Args:
            exact: Whether it does.

        Example:
            >>> Agreement(exact=True).holds(2, 2)
            True
        """
        self._exact = exact

    def holds(self, rules: object, polars: object) -> bool:
        """Say whether the two answers agree.

        Args:
            rules: The answer of the rules.
            polars: The answer of polars.

        Returns:
            Whether they agree.

        Example:
            >>> refused = Stopped(Stop.RAISES, "ShapeError")
            >>> Agreement(exact=False).holds(5, 3), Agreement(exact=True).holds(refused, 3)
            (True, False)
        """
        if self._exact or isinstance(rules, Stopped) or isinstance(polars, Stopped):
            return rules == polars
        if isinstance(rules, int) and isinstance(polars, int):
            return polars <= rules
        return self._within(rules, polars)

    def _within(self, rules: object, polars: object) -> bool:
        """Say whether a column of polars fits within the column of the rules.

        Args:
            rules: The column of the rules.
            polars: The column of polars.

        Returns:
            Whether both hold one value for each group, or both hold lists
            and each list of polars is at most as long.

        Example:
            >>> Agreement(exact=False)._within(AggOutput(None), AggOutput((1,)))
            False
        """
        if not isinstance(rules, AggOutput) or not isinstance(polars, AggOutput):
            return False
        if rules.lists is None or polars.lists is None:
            return rules == polars
        pairs = zip(polars.lists, rules.lists, strict=True)
        return all(itertools.starmap(operator.le, pairs))


@pytest.mark.parametrize("case", Cases().all(), ids=Cases().ident)
def test_select_gives_the_rows_of_polars(case: tuple[Node, tuple[str, ...]]) -> None:
    """Check the rows of a node in a `select`, or the error, against polars.

    Args:
        case: The node and the form of each input.
    """
    node, forms = case
    frames = Frames()
    rules = Rules(frames)
    answer = rules.select(node, forms)
    polars = Polars(frames).select(node, forms)
    exact = node.exact and not rules.decided(node, forms)
    assert Agreement(exact=exact).holds(answer, polars), (answer, polars)


@pytest.mark.parametrize("case", Cases().all(), ids=Cases().ident)
def test_agg_gives_the_column_of_polars(case: tuple[Node, tuple[str, ...]]) -> None:
    """Check the column of a node in `agg`, or the error, against polars, over two groups.

    Args:
        case: The node and the form of each input.
    """
    node, forms = case
    frames = Frames()
    rules = Rules(frames).agg(node, forms)
    if rules == Stopped(Stop.CRASHES):
        # A read past the end of an input can end the process, so it does not run here.
        assert node.name == "min_by"
        assert "one value" in forms
        return
    polars = Polars(frames).agg(node, forms)
    assert Agreement(exact=node.exact).holds(rules, polars), (rules, polars)


def test_min_by_reads_past_the_end_of_one_value_for_each_group() -> None:
    """Check that `min_by` in `agg` reads an input of one value for each group past its end.

    The value and the key hold one value in each group, so the least key of
    a group picks the one value of that group. Polars reads both inputs at the
    rows of the groups of three and two rows, past the end of their two rows,
    and gives other values.
    """
    frames = Frames()
    node = Cases().nodes()["min_by"]
    forms = ("one value", "one value")
    grouped = frames.frame.group_by("g", maintain_order=True)
    firsts = grouped.agg(pl.col("x").first())["x"].to_list()
    read = grouped.agg(Polars(frames).expression(node, forms))["o"].to_list()
    assert read != firsts
    assert Rules(frames).agg(node, forms) == Stopped(Stop.CRASHES)


@pytest.mark.skipif(
    PolarsRelease.installed() >= PolarsRelease.parse("2.0.0"),
    reason="polars 2.0.0 runs a literal key of min_by in agg group by group",
)
def test_a_literal_key_of_min_by_gives_one_row_for_two_groups() -> None:
    """Check that a literal key of `min_by` raises in `agg` with two groups, and runs with one."""
    frames = Frames()
    node = Cases().nodes()["min_by"]
    error = Stopped(Stop.RAISES, "ComputeError")
    assert Polars(frames).agg(node, ("per row", "literal")) == error
    assert Rules(frames).agg(node, ("per row", "literal")) == error
    frame = GroupBy((5,))
    value = node.run().agg([frame.column(), frame.literal()], frame)
    assert frame.output(value, wrapped=False) == AggOutput(None)
    one_group = frames.frame.lazy().with_columns(g=pl.lit(1))
    # The checker evaluates `agg` over two groups, so it refuses this call, which runs with one.
    query = one_group.group_by("g").agg(pl.col("x").min_by(pl.lit(1)).alias("o"))  # type: ignore[invalid-call-shape]
    assert query.collect(engine="in-memory").height == 1


@pytest.mark.skipif(
    PolarsRelease.installed() < PolarsRelease.parse("2.0.0"),
    reason="polars 2.0.0 runs a literal key of min_by in agg group by group",
)
def test_a_literal_key_of_min_by_gives_one_value_for_each_group() -> None:
    """Check that a literal key of `min_by` picks one value in each group, from polars 2.0.0."""
    frames = Frames()
    node = Cases().nodes()["min_by"]
    assert Polars(frames).agg(node, ("per row", "literal")) == AggOutput(None)
    assert Rules(frames).agg(node, ("per row", "literal")) == AggOutput(None)


class SortedValues:
    """Run nodes of `agg` over a `sort` of one value for each group, in a process of its own.

    Polars sorts one value for each group with the indices of the rows of the
    groups of the frame, so it reads past the end of the values
    (`spec/POLARS_BEHAVIORS.md`), and the process can crash. The sort then holds the
    groups of the frame. Each later node gives the answer that it gives with
    no sort, so the rules keep the groups of one value. Only `min_by` and
    `max_by` read those groups, and they read past the end with no sort too.

    Example:
        >>> sorted(SortedValues.NODES)[:3]
        ['+ y', 'arg_max', 'cum_count']
    """

    NODES: ClassVar[dict[str, Callable[[pl.Expr], pl.Expr]]] = {
        "cum_sum": lambda x: x.cum_sum(),
        "sort_by": lambda x: x.sort_by(pl.col("y").first()),
        "+ y": lambda x: x + pl.col("y"),
        "len": lambda x: x.len(),
        "implode": lambda x: x.implode(),
        "gather": lambda x: x.gather(0),
        "head": lambda x: x.head(2),
        "shift": lambda x: x.shift(1),
        "rank": lambda x: x.rank(),
        "reverse": lambda x: x.reverse(),
        "filter": lambda x: x.filter(pl.col("y") > 0),
        "unique": lambda x: x.unique(),
        "max": lambda x: x.max(),
        "last": lambda x: x.last(),
        "arg_max": lambda x: x.arg_max(),
        "sort": lambda x: x.sort(),
        "cum_count": lambda x: x.cum_count(),
        "explode": lambda x: x.explode(),
        "mode": lambda x: x.mode(),
        "diff": lambda x: x.diff(),
    }

    @staticmethod
    def answer(
        name: str,
        sort: bool,
        results: multiprocessing.queues.Queue[str],
    ) -> None:
        """Give the column of a node in `agg`, as text, from a process of its own.

        Args:
            name: The name of the node.
            sort: Whether the input of the node is sorted first.
            results: The queue that takes the answer.
        """
        first = pl.col("x").first()
        node = SortedValues.NODES[name](first.sort() if sort else first).alias("o")
        query = Frames().frame.lazy().group_by("g", maintain_order=True).agg(node)
        answer = PolarsAnswer().result(lambda: query.collect(engine="in-memory")["o"])
        results.put(repr(answer.to_list()) if isinstance(answer, pl.Series) else repr(answer))

    def differences(self) -> list[str]:
        """Give each node whose answer over a sorted input differs from its answer with no sort.

        Returns:
            The names of the nodes, with the two answers.
        """
        context = multiprocessing.get_context("spawn")
        problems = []
        for name in self.NODES:
            answers = []
            for sort in (True, False):
                results = context.Queue()
                process = context.Process(target=SortedValues.answer, args=(name, sort, results))
                process.start()
                process.join(timeout=60)
                answers.append(results.get() if process.exitcode == 0 else "crashed")
            if answers[0] != answers[1]:
                problems.append(f"{name}: sorted {answers[0]}, not sorted {answers[1]}")
        return problems


def test_a_sort_of_one_value_for_each_group_changes_no_later_node() -> None:
    """The rules keep a sorted value for each group, and no later node of polars sees the sort."""
    frame = GroupBy(Frames.SIZES)
    first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
    ordered = Run(Shape.of(NodeKind.SORT)).agg([first], frame)
    assert (ordered.state, ordered.groups) == (first.state, first.groups)
    problems = SortedValues().differences()
    assert not problems, "\n".join(problems)


@dataclass(frozen=True)
class Split:
    """Hold a column that a frame declares at one type while its data has another type.

    Polars plans a query once, at the declared types, and runs it on the
    data with no new check. A frame from `register_io_source` declares a
    schema and holds data of other types, so it shows the rule of the list
    builder with no cast, and a cast of a `Struct` crashes some kernels.

    Example:
        >>> Split("ints", pl.Int64(), pl.Series([1.0])).data.dtype
        Float64
    """

    name: str
    declared: pl.DataType
    data: pl.Series

    def lazy(self) -> pl.LazyFrame:
        """Give a lazy frame of two groups whose column `x` declares one type and holds the data.

        Returns:
            The lazy frame.

        Example:
            >>> Split("ints", pl.Int64(), pl.Series([1.0] * 5)).lazy().collect_schema()["x"]
            Int64
        """
        frame = pl.DataFrame([
            self.data.alias("x"),
            pl.Series("g", [1, 1, 1, 2, 2]),
            pl.Series("k", [2, 2, 2, 2, 2]),
        ])

        def source(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            """Give the frame, with the columns that the query reads.

            Args:
                with_columns: The columns that the query reads, or `None` for all.
                predicate: A filter that the query pushes down, which the frame ignores.
                n_rows: A limit that the query pushes down, which the frame ignores.
                batch_size: The size of a batch, which the frame ignores.

            Yields:
                The frame.
            """
            del predicate, n_rows, batch_size
            yield frame if with_columns is None else frame.select(with_columns)

        schema = {"x": self.declared, "g": pl.Int64(), "k": pl.Int64()}
        return register_io_source(io_source=source, schema=schema)


@dataclass(frozen=True)
class BuiltNode:
    """Hold a node for the test of the list builder, with its inputs in the rules.

    `refuses` holds the kinds of data that the kernel of the node refuses
    for a cause of its own. `unwraps` says that the node reads the values
    inside a list, as `explode` does.

    Example:
        >>> BuildCases().nodes()[0].name
        'cum_max'
    """

    name: str
    shape: Shape
    build: Callable[[pl.Expr], pl.Expr]
    inputs: Callable[[GroupBy], list[InGroups]]
    refuses: frozenset[str] = frozenset()
    unwraps: bool = False


class BuildCases:
    """Give the splits and the nodes of the test of the list builder.

    Example:
        >>> len(BuildCases().all()) > 50
        True
    """

    def splits(self) -> list[Split]:
        """Give the columns whose data type differs from the declared type.

        Returns:
            The columns.

        Example:
            >>> BuildCases().splits()[0].name
            'Int64 holds Float64'
        """
        floats = [1.0, 2.0, 3.0, 4.0, 5.0]
        ints = [1, 2, 3, 4, 5]
        times = [dt.datetime(2020, 1, day) for day in ints]
        return [
            Split("Int64 holds Float64", pl.Int64(), pl.Series(floats)),
            Split("Int64 holds Datetime", pl.Int64(), pl.Series(times, dtype=pl.Datetime("us"))),
            Split("Null holds Float64", pl.Null(), pl.Series(floats)),
            Split("Int64 holds Null", pl.Int64(), pl.Series([None] * 5, dtype=pl.Null())),
            Split("Int64 holds Int32", pl.Int64(), pl.Series(ints, dtype=pl.Int32())),
            Split(
                "Decimal holds Decimal",
                pl.Decimal(10, 2),
                pl.Series(ints, dtype=pl.Decimal(38, 2)),
            ),
            Split("String holds Int64", pl.String(), pl.Series(ints)),
            Split("Boolean holds Int64", pl.Boolean(), pl.Series(ints)),
            Split("Struct holds Struct", pl.Struct({"a": pl.Int64()}), pl.Series([{"b": 1}] * 5)),
            Split("List holds List", pl.List(pl.Int64()), pl.Series([[1.0]] * 5)),
            Split("Float64 holds Int64", pl.Float64(), pl.Series(ints)),
            Split("Date holds Int32", pl.Date(), pl.Series(ints, dtype=pl.Int32())),
        ]

    def nodes(self) -> list[BuiltNode]:
        """Give the nodes, with the options that polars gives them.

        Returns:
            The nodes.

        Example:
            >>> [node.name for node in BuildCases().nodes()][-1]
            'abs'
        """
        preserving = Shape.function(FunctionOptions.length_preserving())
        top_k = Shape.function(FunctionOptions.groupwise(), KernelLength.at_most(2))
        fill = FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default())
        nested = frozenset({"Null", "Struct", "List"})
        first = Run(Shape.of(NodeKind.AGG))
        return [
            BuiltNode("cum_max", preserving, lambda x: x.cum_max(), self._alone, nested),
            BuiltNode("shift", preserving, lambda x: x.shift(1), self._with_literal),
            BuiltNode(
                "explode",
                Shape.of(NodeKind.EXPLODE),
                lambda x: x.explode(empty_as_null=True),
                self._alone,
                unwraps=True,
            ),
            BuiltNode(
                "top_k",
                top_k,
                lambda x: x.top_k(pl.col("k").first()),
                lambda frame: [frame.column(), first.agg([frame.column()], frame)],
            ),
            BuiltNode(
                "fill_null by its first",
                Shape.function(fill),
                lambda x: x.fill_null(x.first()),
                lambda frame: [frame.column(), first.agg([frame.column()], frame)],
            ),
            BuiltNode(
                "fill_null by a literal",
                Shape.function(fill),
                lambda x: x.fill_null(pl.lit(None)),
                self._with_literal,
                frozenset({"Null"}),
            ),
            BuiltNode(
                "abs",
                Shape.function(FunctionOptions.elementwise()),
                lambda x: x.abs(),
                self._alone,
                nested | {"Datetime"},
            ),
        ]

    def _alone(self, frame: GroupBy) -> list[InGroups]:
        """Give the inputs of a node of one column.

        Args:
            frame: The groups.

        Returns:
            The column.

        Example:
            >>> len(BuildCases()._alone(GroupBy((3, 2))))
            1
        """
        return [frame.column()]

    def _with_literal(self, frame: GroupBy) -> list[InGroups]:
        """Give the inputs of a node of one column and one literal.

        Args:
            frame: The groups.

        Returns:
            The column and the literal.

        Example:
            >>> [value.state.value for value in BuildCases()._with_literal(GroupBy((3, 2)))]
            ['not aggregated', 'literal scalar']
        """
        return [frame.column(), frame.literal()]

    def all(self) -> list[tuple[BuiltNode, Split]]:
        """Give each node with each split whose data its kernel takes.

        Returns:
            The cases.

        Example:
            >>> BuildCases().all()[0][1].name
            'Int64 holds Float64'
        """
        cases: list[tuple[BuiltNode, Split]] = []
        for node in self.nodes():
            for split in self.splits():
                if type(split.data.dtype).__name__ not in node.refuses:
                    cases.append((node, split))
        return cases

    def ident(self, case: tuple[BuiltNode, Split]) -> str:
        """Name a case for the report of pytest.

        Args:
            case: The node and the split.

        Returns:
            The name.

        Example:
            >>> BuildCases().ident(BuildCases().all()[0])
            'cum_max over Int64 holds Float64'
        """
        node, split = case
        return f"{node.name} over {split.name}"


class Builds:
    """Give the type of the lists of a node in `agg`, from the rules and from polars.

    Example:
        >>> node, split = BuildCases().all()[0]
        >>> Builds().rules(node, split), Builds().polars(node, split)
        (panics, panics)
    """

    def rules(self, node: BuiltNode, split: Split) -> object:
        """Give the type of the values of each list, from the rules, or how polars stops.

        The planned type of each node of the test is the type of its input,
        and its kernel gives the type of the data.

        Args:
            node: The node.
            split: The column.

        Returns:
            The type, or how the run stops.

        Example:
            >>> node, split = BuildCases().all()[1]
            >>> Builds().rules(node, split)
            Int64
        """
        planned = DType.of(split.declared)
        data = DType.of(split.data.dtype)
        if node.unwraps and planned.name == "List":
            planned, data = planned.inner(), data.inner()
        frame = GroupBy((3, 2))
        run = Run(node.shape, Kernel(planned=planned, data=data))
        try:
            return run.agg(node.inputs(frame), frame).dtype
        except PolarsRefusal as refusal:
            return Stopped.of(refusal)

    def polars(self, node: BuiltNode, split: Split) -> object:
        """Give the type of the values of each list, from polars, or how polars stops.

        Args:
            node: The node.
            split: The column.

        Returns:
            The type, or how the run stops.

        Example:
            >>> node, split = BuildCases().all()[1]
            >>> Builds().polars(node, split)
            Int64
        """
        grouped = split.lazy().group_by("g", maintain_order=True)
        query = grouped.agg(node.build(pl.col("x")).alias("o"))
        outcome = PolarsAnswer().result(lambda: query.collect(engine="in-memory")["o"].dtype)
        if isinstance(outcome, pl.List):
            return DType.of(outcome.inner)
        return outcome


@pytest.mark.parametrize("case", BuildCases().all(), ids=BuildCases().ident)
def test_a_group_aware_node_builds_its_lists_at_its_planned_type(
    case: tuple[BuiltNode, Split],
) -> None:
    """Check the rule of the list builder against polars, over data that has not the planned type.

    Args:
        case: The node and the column.
    """
    node, split = case
    rules = Builds().rules(node, split)
    polars = Builds().polars(node, split)
    assert rules == polars, (rules, polars)


class Folds:
    """Give the literals of the test of the fold, and build them in polars.

    Example:
        >>> Folds().polars(Operand.typed_literal(1, DType("Int8"))).meta.output_name()
        'literal'
    """

    OPERATORS: ClassVar[dict[Operator, Callable[[pl.Expr, pl.Expr], object]]] = {
        Operator.PLUS: operator.add,
        Operator.MINUS: operator.sub,
        Operator.MULTIPLY: operator.mul,
        Operator.TRUE_DIVIDE: operator.truediv,
        Operator.FLOOR_DIVIDE: operator.floordiv,
        Operator.MODULUS: operator.mod,
        Operator.GT: operator.gt,
        Operator.LT_EQ: operator.le,
        Operator.EQ: operator.eq,
        Operator.AND: operator.and_,
        Operator.XOR: operator.xor,
    }

    def cases(self) -> list[tuple[Operator, Operand, Operand]]:
        """Give the binary nodes of two literals.

        Returns:
            The nodes: the operator, the left literal and the right literal.

        Example:
            >>> Folds().cases()[0][0]
            <Operator.PLUS: 'Plus'>
        """
        dynamic = Operand.of_value
        int8, uint8 = DType("Int8"), DType("UInt8")
        true, false = Operand.of_value(True), Operand.of_value(False)
        return [
            (Operator.PLUS, dynamic(200), dynamic(100)),
            (Operator.MULTIPLY, dynamic(2**31), dynamic(2)),
            (Operator.MINUS, dynamic(1), dynamic(2)),
            (Operator.FLOOR_DIVIDE, dynamic(-7), dynamic(2)),
            (Operator.MODULUS, dynamic(-7), dynamic(2)),
            (Operator.TRUE_DIVIDE, dynamic(7), dynamic(2)),
            (Operator.PLUS, dynamic(1.5), dynamic(1)),
            (Operator.PLUS, Operand.typed_literal(100, int8), Operand.typed_literal(100, int8)),
            (Operator.MULTIPLY, Operand.typed_literal(200, uint8), Operand.typed_literal(2, uint8)),
            (Operator.GT, dynamic(300), dynamic(1)),
            (Operator.LT_EQ, dynamic(1.5), dynamic(0.5)),
            (Operator.EQ, Operand.typed_literal(1, int8), Operand.typed_literal(1, int8)),
            (Operator.AND, true, false),
            (Operator.XOR, true, true),
        ]

    def polars(self, literal: Operand) -> pl.Expr:
        """Build a literal in polars.

        Args:
            literal: The literal.

        Returns:
            The expression: a literal with no data type for a dynamic literal.

        Example:
            >>> str(Folds().polars(Operand.of_value(1)))
            'dyn int: 1'
        """
        if literal.dtype.name == "Unknown":
            return pl.lit(literal.value)
        dtype: pl.DataType = getattr(pl, literal.dtype.name)()
        return pl.lit(literal.value, dtype=dtype)

    def node(self, case: tuple[Operator, Operand, Operand]) -> pl.Expr:
        """Build the binary node of two literals in polars.

        Args:
            case: The operator and the two literals.

        Returns:
            The expression.

        Example:
            >>> str(Folds().node(Folds().cases()[0]))
            '[(dyn int: 200) + (dyn int: 100)]'
        """
        operator_, left, right = case
        build = self.OPERATORS[operator_]
        node = build(self.polars(left), self.polars(right))
        return node if isinstance(node, pl.Expr) else pl.lit(node)

    def ident(self, case: tuple[Operator, Operand, Operand]) -> str:
        """Name a case for the report of pytest.

        Args:
            case: The operator and the two literals.

        Returns:
            The name.

        Example:
            >>> Folds().ident(Folds().cases()[0])
            'Plus(200, 100)'
        """
        operator_, left, right = case
        return f"{operator_.value}({left.value}, {right.value})"


@pytest.mark.parametrize("case", Folds().cases(), ids=Folds().ident)
def test_the_fold_gives_the_literal_that_polars_computes(
    case: tuple[Operator, Operand, Operand],
) -> None:
    """Check the value and the type of a folded literal against the select of the node in polars.

    Args:
        case: The operator and the two literals.
    """
    folded = BinaryFold().binary(*case)
    assert folded is not None
    series = pl.select(Folds().node(case).alias("o"))["o"]
    assert DType.of(series.dtype).same(folded.dtype.materialized())
    assert series.to_list() == [pytest.approx(folded.value)]


def test_no_fold_of_other_types_of_a_division_by_zero_and_of_strings() -> None:
    """Check the nodes of two literals that the conversion pass does not fold."""
    fold = BinaryFold()
    int8, int16 = Operand.typed_literal(1, DType("Int8")), Operand.typed_literal(1, DType("Int16"))
    text = Operand.typed_literal("a", DType.unknown("str"))
    assert fold.binary(Operator.PLUS, int8, int16) is None
    assert fold.binary(Operator.FLOOR_DIVIDE, Operand.of_value(1), Operand.of_value(0)) is None
    assert fold.binary(Operator.MODULUS, Operand.of_value(1), Operand.of_value(0)) is None
    assert fold.binary(Operator.PLUS, text, text) is None
    assert fold.binary(Operator.AND, Operand.of_value(1), Operand.of_value(1)) is None


class FoldContexts:
    """Give the places where a node meets a literal subtree, and what polars gives there.

    In each context, the node `u + (left op right)` meets the subtree at its
    cast. Where the fold is visible, polars gives the type of the node with
    the folded literal in place of the subtree. Otherwise it gives the
    planned type, which reads the subtree as written.

    Example:
        >>> FoldContexts().visible("nested in select")
        False
    """

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({
        "g": [1, 1, 2],
        "u": pl.Series([1, 2, 3], dtype=pl.UInt8()),
    })
    NAMES: ClassVar[tuple[str, ...]] = (
        "root of select",
        "nested in select",
        "wrapped in agg",
        "root of agg",
        "root of an eager select",
        "root without the simplifier",
    )

    def visible(self, name: str) -> bool:
        """Give whether the cast of the node reads the folded literal.

        The conversion pass folds where the simplifier runs, and only the root
        of the pass reads a folded input at its cast. Before polars 1.40.0,
        `agg` puts no node above the expression, so the node that it wraps is
        the root too.

        Args:
            name: The context.

        Returns:
            Whether it does.

        Example:
            >>> FoldContexts().visible("root of agg")
            True
        """
        implodes = PolarsRelease.installed() >= PolarsRelease.parse("1.40.0")
        below = {"nested in select", "wrapped in agg"} if implodes else {"nested in select"}
        root = name not in below
        simplify = name != "root without the simplifier"
        return root and simplify

    def ran(self, name: str, subtree: pl.Expr) -> object:
        """Give the type that polars runs in a context, or how it stops.

        Args:
            name: The context.
            subtree: The literal subtree.

        Returns:
            The type.

        Example:
            >>> FoldContexts().ran("root of select", pl.lit(1))
            UInt8
        """
        node = self._node(name, subtree)
        lazy = self.FRAME.lazy()
        off = pl.QueryOptFlags(simplify_expression=False)
        runs: dict[str, Callable[[], pl.DataFrame]] = {
            "root of select": lambda: lazy.select(node).collect(engine="in-memory"),
            "nested in select": lambda: lazy.select(node).collect(engine="in-memory"),
            "wrapped in agg": lambda: lazy.group_by("g").agg(node).collect(engine="in-memory"),
            "root of agg": lambda: lazy.group_by("g").agg(node).collect(engine="in-memory"),
            "root of an eager select": lambda: self.FRAME.select(node),
            "root without the simplifier": lambda: lazy.select(node).collect(
                optimizations=off, engine="in-memory"
            ),
        }
        return PolarsAnswer().result(lambda: runs[name]()["o"].dtype)

    def planned(self, name: str, subtree: pl.Expr) -> object:
        """Give the planned type in a context: the type of `collect_schema()`.

        Args:
            name: The context.
            subtree: The literal subtree.

        Returns:
            The type.

        Example:
            >>> FoldContexts().planned("wrapped in agg", pl.lit(1))
            List(UInt8)
        """
        node = self._node(name, subtree)
        lazy = self.FRAME.lazy()
        agg = name in {"wrapped in agg", "root of agg"}
        query = lazy.group_by("g").agg(node) if agg else lazy.select(node)
        return PolarsAnswer().result(lambda: query.collect_schema()["o"])

    def _node(self, name: str, subtree: pl.Expr) -> pl.Expr:
        """Build the node that meets the subtree in a context.

        Args:
            name: The context.
            subtree: The literal subtree.

        Returns:
            The node, named `o`.

        Example:
            >>> str(FoldContexts()._node("root of agg", pl.lit(1)))
            '[(col("u").first()) + (dyn int: 1)].alias("o")'
        """
        u = pl.col("u")
        if name == "nested in select":
            return (u + subtree).abs().alias("o")
        if name == "root of agg":
            return (u.first() + subtree).alias("o")
        return (u + subtree).alias("o")


@pytest.mark.parametrize("context", FoldContexts.NAMES)
@pytest.mark.parametrize("case", Folds().cases()[:3], ids=Folds().ident)
def test_only_the_root_reads_a_folded_literal(
    case: tuple[Operator, Operand, Operand],
    context: str,
) -> None:
    """Check where the cast of a node reads its literal subtree folded, against polars.

    Args:
        case: The operator and the two literals of the subtree.
        context: The place of the node.
    """
    contexts = FoldContexts()
    folded = BinaryFold().binary(*case)
    assert folded is not None
    ran = contexts.ran(context, Folds().node(case))
    if contexts.visible(context):
        assert ran == contexts.ran(context, Folds().polars(folded))
    else:
        assert ran == contexts.planned(context, Folds().node(case))


@dataclass(frozen=True)
class SliceCase:
    """Hold a slice node of the test of the pushdown: its three inputs in polars and in the rules.

    Example:
        >>> SliceCases().cases()[0].name
        'x.slice(o, n)'
    """

    name: str
    value: tuple[pl.Expr, SliceInput]
    offset: tuple[pl.Expr, SliceInput]
    length: tuple[pl.Expr, SliceInput]

    def expression(self) -> pl.Expr:
        """Build the slice in polars.

        Returns:
            The expression, named `o`.

        Example:
            >>> SliceCases().cases()[0].expression().meta.output_name()
            'o'
        """
        return self.value[0].slice(self.offset[0], self.length[0]).alias("o")

    def outcome(self) -> Pushdown | Stopped:
        """Give what the rules say that the pushdown does, or that the query never returns.

        Returns:
            What the pushdown does, or a `Stopped` that hangs.

        Example:
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     SliceCases().cases()[0].outcome()
            <Pushdown.DELETED: 'deleted'>
        """
        inputs = (self.value[1], self.offset[1], self.length[1])
        return RuleAnswer().result(lambda: SlicePushdown().outcome(*inputs))


class SliceCases:
    """Give the slice nodes of the test of the pushdown, over a frame of two groups.

    Example:
        >>> SliceCases().ident(SliceCases().cases()[1])
        'x.slice(0, n)'
    """

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({
        "g": [1, 1, 1, 2, 2],
        "x": [3, 1, 2, 5, 4],
        "o": [0, 1, 0, 1, 0],
        "n": [2, 2, 2, 2, 2],
    })

    def inputs(self) -> dict[str, tuple[pl.Expr, SliceInput]]:
        """Give the inputs of the slices, in polars and in the rules.

        Returns:
            The inputs, by name.

        Example:
            >>> SliceCases().inputs()["(x + 1)"][1].passes
            True
        """
        x, o, n = pl.col("x"), pl.col("o"), pl.col("n")
        column = Shape.of(NodeKind.COLUMN)
        agg = Shape.of(NodeKind.AGG)
        cum_sum = Shape.function(FunctionOptions.length_preserving())
        abs_ = Shape.function(FunctionOptions.elementwise())
        literal = SliceInput.of(Shape.literal(), [], literal=True)
        cols = [Height.COLUMN]
        heads = [Height.COLUMN, Height.SCALAR, Height.SCALAR]
        return {
            "x": (x, SliceInput.of(column, [])),
            "(x + 1)": (
                x + 1,
                SliceInput.of(Shape.of(NodeKind.BINARY), [Height.COLUMN, Height.SCALAR]),
            ),
            "x.abs()": (x.abs(), SliceInput.of(abs_, cols)),
            "x.last()": (x.last(), SliceInput.of(agg, cols)),
            "x.cum_sum()": (x.cum_sum(), SliceInput.of(cum_sum, cols)),
            "x.head(2)": (x.head(2), SliceInput.of(Slices.HEAD_2, heads)),
            "o": (o, SliceInput.of(column, [])),
            "n": (n, SliceInput.of(column, [])),
            "o.first()": (o.first(), SliceInput.of(agg, cols)),
            "n.first()": (n.first(), SliceInput.of(agg, cols)),
            "o.head(2)": (o.head(2), SliceInput.of(Slices.HEAD_2, heads)),
            "0": (pl.lit(0), literal),
            "2": (pl.lit(2), literal),
        }

    def cases(self) -> list[SliceCase]:
        """Give the slices.

        Returns:
            The slices.

        Example:
            >>> len(SliceCases().cases())
            16
        """
        names = [
            ("x", "o", "n"),
            ("x", "0", "n"),
            ("x", "o", "2"),
            ("x", "o.first()", "n.first()"),
            ("x", "0", "2"),
            ("x.last()", "o", "n"),
            ("x.cum_sum()", "o", "n"),
            ("x.head(2)", "o", "n"),
            ("x", "o.head(2)", "n"),
            ("x.head(2)", "0", "2"),
            ("(x + 1)", "0", "2"),
            ("(x + 1)", "o.first()", "n.first()"),
            ("x.abs()", "o.first()", "2"),
            ("(x + 1)", "o", "n"),
            ("x.abs()", "o", "n"),
            ("(x + 1)", "0", "n"),
        ]
        inputs = self.inputs()
        cases: list[SliceCase] = []
        for value, offset, length in names:
            name = f"{value}.slice({offset}, {length})"
            cases.append(SliceCase(name, inputs[value], inputs[offset], inputs[length]))
        return cases

    def ident(self, case: SliceCase) -> str:
        """Name a case for the report of pytest.

        Args:
            case: The slice.

        Returns:
            The name.

        Example:
            >>> SliceCases().ident(SliceCases().cases()[0])
            'x.slice(o, n)'
        """
        return case.name

    def runs(self, case: SliceCase) -> list[object]:
        """Give what a lazy frame and an eager frame give, in a `select` and in `agg`.

        Args:
            case: The slice.

        Returns:
            The four outcomes: the lazy select, the eager select, the lazy agg and the eager agg.

        Example:
            >>> SliceCases().runs(SliceCases().cases()[4])[0]
            [3, 1]
        """
        expression = case.expression()
        lazy = self.FRAME.lazy()
        eager = self.FRAME
        return [
            PolarsAnswer().result(lambda: self._listed(lazy.select(expression))),
            PolarsAnswer().result(lambda: self._listed(eager.select(expression))),
            PolarsAnswer().result(
                lambda: self._listed(lazy.group_by("g", maintain_order=True).agg(expression)),
            ),
            PolarsAnswer().result(
                lambda: self._listed(eager.group_by("g", maintain_order=True).agg(expression)),
            ),
        ]

    def values(self, case: SliceCase) -> list[object]:
        """Give what the value of the slice gives alone, in a `select` and in `agg`.

        In `agg`, the planner wraps the slice in a list for each group
        before the pushdown deletes it, so the value comes as the list of
        its values in each group.

        Args:
            case: The slice.

        Returns:
            The two outcomes: the select and the agg, both on a lazy frame.

        Example:
            >>> SliceCases().values(SliceCases().cases()[0])[0]
            [3, 1, 2, 5, 4]
        """
        expression = case.value[0].alias("o")
        imploded = expression.implode()
        lazy = self.FRAME.lazy()
        return [
            PolarsAnswer().result(lambda: self._listed(lazy.select(expression))),
            PolarsAnswer().result(
                lambda: self._listed(lazy.group_by("g", maintain_order=True).agg(imploded)),
            ),
        ]

    def _listed(self, frame: pl.DataFrame | pl.LazyFrame) -> list[object]:
        """Give the column `o` of a frame as a list, after the collect of a lazy frame.

        Args:
            frame: The frame.

        Returns:
            The values of the column.

        Example:
            >>> SliceCases()._listed(SliceCases.FRAME.lazy().select(pl.col("x").sum().alias("o")))
            [15]
        """
        collected = frame.collect(engine="in-memory") if isinstance(frame, pl.LazyFrame) else frame
        return collected["o"].to_list()


class HangProbe:
    """Run a slice in a process of its own, where the optimizer can hang.

    Example:
        >>> HangProbe.QUERIES["(x + 1).slice(o, n)"].meta.output_name()
        'o'
    """

    QUERIES: ClassVar[dict[str, pl.Expr]] = {
        "(x + 1).slice(o, n)": (pl.col("x") + 1).slice(pl.col("o"), pl.col("n")).alias("o"),
        "x.abs().slice(o, n)": pl.col("x").abs().slice(pl.col("o"), pl.col("n")).alias("o"),
        "(x + 1).slice(0, n)": (pl.col("x") + 1).slice(0, pl.col("n")).alias("o"),
        "x.abs().slice(o, n).len()": pl.col("x").abs().slice(pl.col("o"), pl.col("n")).len(),
    }
    WAIT: ClassVar[float] = 3.0

    @staticmethod
    def collect(name: str, started: multiprocessing.synchronize.Event) -> None:
        """Collect a query of the test in a process of its own.

        Args:
            name: The name of the query.
            started: The event to set before the query runs.
        """
        started.set()
        SliceCases.FRAME.lazy().select(HangProbe.QUERIES[name]).collect(engine="in-memory")

    def hangs(self, names: list[str]) -> list[bool]:
        """Say for each query whether it runs longer than the wait.

        Args:
            names: The names of the queries.

        Returns:
            Whether each query is still running after the wait.
        """
        context = multiprocessing.get_context("spawn")
        processes = []
        for name in names:
            started = context.Event()
            process = context.Process(target=HangProbe.collect, args=(name, started), daemon=True)
            process.start()
            processes.append((process, started))
        for _, started in processes:
            started.wait(timeout=60)
        deadline = time.monotonic() + self.WAIT
        for process, _ in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [process.is_alive() for process, _ in processes]
        for process, _ in processes:
            process.kill()
            process.join()
        return alive


@pytest.mark.parametrize("case", SliceCases().cases(), ids=SliceCases().ident)
def test_the_slice_pushdown_keeps_moves_or_deletes_a_slice_as_polars_does(case: SliceCase) -> None:
    """Check what the slice pushdown of a lazy frame does to a slice, against polars.

    A slice that the pushdown keeps or moves gives on a lazy frame what it
    gives on an eager frame, which does not push slices. A slice that the
    pushdown deletes gives its value unchanged on a lazy frame.

    Args:
        case: The slice.
    """
    outcome = case.outcome()
    if isinstance(outcome, Stopped):
        assert outcome == Stopped(Stop.HANGS)
        assert case.name in HangProbe.QUERIES
        return
    lazy_select, eager_select, lazy_agg, eager_agg = SliceCases().runs(case)
    if outcome is Pushdown.DELETED:
        assert [lazy_select, lazy_agg] == SliceCases().values(case)
        assert eager_select != lazy_select
    else:
        assert (lazy_select, lazy_agg) == (eager_select, eager_agg)


@pytest.mark.skipif(
    PolarsRelease.installed() < PolarsRelease.parse("1.40.0"),
    reason="polars 1.40.0 makes an offset or a length that is not a literal a place of a slice",
)
def test_the_slice_pushdown_hangs_where_a_cycle_reaches_the_value() -> None:
    """Check that the slices that the rules say hang do not return, in processes of their own.

    The rules give the hang for the slice node, and for a lazy `select` that
    holds the slice alone, which translates the pushdown step by step. A
    `len` over the slice hangs too, before the projection pushdown counts
    the rows.
    """
    names = list(HangProbe.QUERIES)
    cases = {case.name: case for case in SliceCases().cases()}
    nodes = [name for name in names if name in cases]
    assert [cases[name].outcome() for name in nodes] == [Stopped(Stop.HANGS)] * len(nodes)
    leaf = ColumnLeaf(DType("Int64"), RowForm.PER_ROW)
    lone = TreeEvaluation(
        dict.fromkeys(SliceCases.FRAME.columns, leaf), GroupBy((5,), QueryFlags.lazy(lone=True))
    )
    runs = [lone.answers(ExpressionTree().of(HangProbe.QUERIES[name])).run_select for name in names]
    assert runs == [Refused(Stop.HANGS)] * len(names)
    assert HangProbe().hangs(names) == [True] * len(names)


@pytest.mark.skipif(
    PolarsRelease.installed() >= PolarsRelease.parse("1.40.0"),
    reason="polars 1.40.0 makes an offset or a length that is not a literal a place of a slice",
)
def test_the_slice_pushdown_of_an_older_release_ends_where_a_later_one_hangs() -> None:
    """Check that the slices that hang from polars 1.40.0 return before it, in the rules and polars.

    The older releases read an offset or a length as it is, so the pushdown
    builds no cycle.
    """
    names = list(HangProbe.QUERIES)
    cases = {case.name: case for case in SliceCases().cases()}
    outcomes = [cases[name].outcome() for name in names if name in cases]
    assert Stopped(Stop.HANGS) not in outcomes
    assert HangProbe().hangs(names) == [False] * len(names)


class LoneCounts:
    """Give each `len` that stands alone in a lazy `select`, and its run from polars and the rules.

    From polars 1.41.0, the projection pushdown counts the rows in place of
    such a `len` where its input has the height of the frame, also after the
    slice pushdown moves the slices of the input into the plan (`RowCount`).
    The input `s` is text, which `abs` refuses at the run, so a run that keeps
    its input raises.

    Example:
        >>> LoneCounts().polars("abs") == LoneCounts().rules("abs")
        True
    """

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({"s": ["a", "b", "c"], "g": [1, 1, 2]})
    TREES: ClassVar[dict[str, pl.Expr]] = {
        "abs": pl.col("s").abs().len(),
        "abs under an alias": pl.col("s").abs().len().alias("n"),
        "count": pl.col("s").abs().count(),
        "plus one": pl.col("s").abs().len() + 1,
        "sum": pl.col("s").abs().sum().len(),
        "first": pl.col("s").abs().first().len(),
        "filter": pl.col("s").abs().filter(pl.col("g") > 1).len(),
        "sort": pl.col("s").abs().sort().len(),
        "cum_sum": pl.col("s").abs().cum_sum().len(),
        "head": pl.col("s").abs().head(2).len(),
        "head below abs": pl.col("s").head(2).abs().len(),
        "tail": pl.col("s").abs().tail(2).len(),
        "head after sort": pl.col("s").abs().sort().head(2).len(),
        "two equal heads": (pl.col("s").abs().head(2) + pl.col("s").abs().head(2)).len(),
        "two heads of two lengths": (pl.col("s").abs().head(2) + pl.col("s").abs().head(1)).len(),
        "a head and a column": (pl.col("s").abs().head(2) + pl.col("s").abs()).len(),
        "a slice with a column bound": pl.col("s").abs().slice(0, pl.col("g").first()).len(),
        "a literal": pl.lit("a").abs().len(),  # type: ignore[invalid-call-shape]
        "a head beside first": (pl.col("s").abs().head(2) + pl.col("s").abs().first()).len(),
        "a head inside a head": pl.col("s").abs().head(3).head(2).len(),
        "last": pl.col("s").abs().last().len(),
        "an input that the planner refuses": pl.col("s").list.len().len(),
        "a slice of a literal beside a column": (
            pl.lit("ab").head(1) + pl.col("s").abs().head(1)
        ).len(),
        "an offset of a series literal": pl.col("s").abs().slice(pl.lit(pl.Series([0])), 2).len(),
        "an eval of lists": pl.col("s").str.split("a").list.eval(pl.element().abs()).len(),
        "an eval of lists after a head": (
            pl.col("s").head(2).str.split("a").list.eval(pl.element().abs()).len()
        ),
        "an eval of one list": (
            pl.col("s").str.split("a").implode().list.eval(pl.element()).abs().len()
        ),
        "an eval of one list of a head": (
            pl.col("s").head(2).abs().implode().list.eval(pl.element()).len()
        ),
        "an eval after a refused shift": (
            pl.col("s").shift(pl.col("g")).str.split("a").list.eval(pl.element().abs()).len()
        ),
        "a cumulative eval": pl.col("s").abs().cumulative_eval(pl.element().first()).len(),
        "a struct eval": pl.col("s").abs().struct.with_fields(pl.field("a")).len(),
    }

    def polars(self, name: str) -> str:
        """Give the type of the lazy `select` of the tree alone, or how polars stops.

        Args:
            name: The name of the tree.

        Returns:
            The rendering of the type of the run, or `raises`.

        Example:
            >>> LoneCounts().polars("count")
            'raises'
        """
        query = self.FRAME.lazy().select(self.TREES[name])
        return RunText().polars(lambda: query.collect(engine="in-memory").dtypes[0])

    def rules(self, name: str) -> str:
        """Give the run of the tree from the rules, in a lazy `select` that holds it alone.

        Args:
            name: The name of the tree.

        Returns:
            The rendering of the type, `raises`, or `unchecked` where the rules do not know.

        Example:
            >>> LoneCounts().rules("count")
            'raises'
        """
        columns = {
            "s": ColumnLeaf(DType("String"), RowForm.PER_ROW),
            "g": ColumnLeaf(DType("Int64"), RowForm.PER_ROW),
        }
        frame = GroupBy((3,), QueryFlags.lazy(lone=True))
        tree = ExpressionTree().of(self.TREES[name])
        return RunText().rules(TreeEvaluation(columns, frame).answers(tree).run_select)


@pytest.mark.parametrize("name", sorted(LoneCounts.TREES))
def test_a_lone_len_in_a_lazy_select_counts_the_rows(name: str) -> None:
    """Check the run of a `len` that stands alone in a lazy `select` against polars."""
    assert LoneCounts().rules(name) == LoneCounts().polars(name)


class RunText:
    """Give an answer of the rules and a run of polars in one text, so that a test compares them.

    Example:
        >>> RunText().rules(DType("Int8")), RunText().polars(lambda: pl.Int8)
        ('Int8', 'Int8')
    """

    def rules(self, answer: object) -> str:
        """Give the text of an answer of the rules.

        Args:
            answer: The answer: a type, a refusal, or a question that the rules cannot answer.

        Returns:
            The rendering of the type, `raises`, or `unchecked` where the rules do not know.

        Example:
            >>> RunText().rules(Refused(Stop.PANICS))
            'raises'
        """
        if isinstance(answer, DType):
            return answer.render()
        return "raises" if isinstance(answer, Refused) else "unchecked"

    def polars(self, run: Callable[[], object]) -> str:
        """Give the text of a run of polars.

        Args:
            run: The run, which gives the data type of the output.

        Returns:
            The rendering of the type, or `raises`.

        Example:
            >>> frame = pl.DataFrame({"s": ["a"]})
            >>> RunText().polars(lambda: frame.select(pl.col("s").abs()).dtypes[0])
            'raises'
        """
        answer = PolarsAnswer().result(run)
        return "raises" if isinstance(answer, Stopped) else str(answer)


class OverFrame:
    """Hold the frame of the tests of `over`: two groups of `g`, of three rows and two.

    Example:
        >>> OverFrame.FRAME.height, sorted(OverFrame().columns())
        (5, ['g', 't', 'x'])
    """

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({
        "g": [1, 1, 1, 2, 2],
        "x": [1, -2, 3, -4, 5],
        "t": ["a", "b", "c", "d", "e"],
    })

    def columns(self) -> dict[str, ColumnLeaf]:
        """Give the leaf of each column of the frame, for the rules.

        Returns:
            The leaves, by the name of the column.

        Example:
            >>> OverFrame().columns()["t"].dtype
            String
        """
        return {
            name: ColumnLeaf(DType.of(dtype), RowForm.PER_ROW)
            for name, dtype in self.FRAME.schema.items()
        }


class OverOfOneGroup:
    """Give each `over` whose keys each hold one row, and its run from polars and the rules.

    Such keys make one group of the whole frame, so the function must keep
    every row of the frame, unless it gives one value. In `agg`, each group
    is the window. The mapping `join` gives each row a list of its group, and
    checks no rows.

    Example:
        >>> OverOfOneGroup().polars("head of three", "select")
        'raises'
        >>> OverOfOneGroup().rules("head of three", "select")
        'raises'
    """

    TREES: ClassVar[dict[str, pl.Expr]] = {
        "head of three": pl.col("x").head(3).over(pl.lit(0)),
        "head of two": pl.col("x").head(2).over(pl.lit(0)),
        "head of one": pl.col("x").head(1).over(pl.lit(0)),
        "head past the rows": pl.col("x").head(9).over(pl.lit(0)),
        "tail of three": pl.col("x").tail(3).over(pl.lit(0)),
        "sum": pl.col("x").sum().over(pl.lit(0)),
        "two literal keys": pl.col("x").head(3).over(pl.lit(0), pl.lit(1)),
        "a key of one value": pl.col("x").head(3).over(pl.col("g").first()),
        "join": pl.col("x").head(3).over(pl.lit(0), mapping_strategy="join"),
        "join of one row": pl.col("x").head(1).over(pl.lit(0), mapping_strategy="join"),
        "explode": pl.col("x").head(3).over(pl.lit(0), mapping_strategy="explode"),
        "explode beside the column": (
            pl.col("x").head(3).over(pl.lit(0), mapping_strategy="explode") + pl.col("x")
        ),
    }

    def polars(self, name: str, context: str) -> str:
        """Give the type of the eager run of the tree, or how polars stops.

        Args:
            name: The name of the tree.
            context: `select`, or `agg` over the groups of `g`.

        Returns:
            The rendering of the type of the run, or `raises`.

        Example:
            >>> OverOfOneGroup().polars("head of three", "agg")
            'List(Int64)'
        """
        tree = self.TREES[name]
        frame = OverFrame.FRAME
        if context == "select":
            return RunText().polars(lambda: frame.select(tree).dtypes[0])
        return RunText().polars(lambda: frame.group_by("g").agg(tree).dtypes[1])

    def rules(self, name: str, context: str) -> str:
        """Give the run of the tree from the rules, over the groups of the frame.

        Args:
            name: The name of the tree.
            context: `select`, or `agg` over the groups of `g`.

        Returns:
            The rendering of the type, `raises`, or `unchecked` where the rules do not know.

        Example:
            >>> OverOfOneGroup().rules("head of three", "agg")
            'List(Int64)'
        """
        frame = GroupBy((3, 2), QueryFlags.eager())
        tree = ExpressionTree().of(self.TREES[name])
        answers = TreeEvaluation(OverFrame().columns(), frame).answers(tree)
        return RunText().rules(answers.run_select if context == "select" else answers.run_agg)


@pytest.mark.parametrize("context", ["select", "agg"])
@pytest.mark.parametrize("name", sorted(OverOfOneGroup.TREES))
def test_an_over_of_keys_of_one_row_runs_over_one_group_of_the_frame(
    name: str,
    context: str,
) -> None:
    """Check an `over` whose keys make one group of the frame against polars."""
    rules = OverOfOneGroup().rules(name, context)
    assert rules in {OverOfOneGroup().polars(name, context), "unchecked"}


class LoneSlices:
    """Give each sliced expression alone in a lazy `select`, and its run in polars and the rules.

    From polars 1.40.0, the slice pushdown moves the slices of such an
    expression into the plan, where each slice cuts a column and the slices
    join into one set of rows, `first` and `last` and a slice inside a slice
    included (`SlicePushdown`). The expression then runs over the rows of the
    plan, so a window over a literal key fits them. Each window raises in an
    eager `select`, where it has the rows of the frame, and in the releases
    before 1.40.0.

    Example:
        >>> LoneSlices().polars("head") == LoneSlices().rules("head")
        True
    """

    TREES: ClassVar[dict[str, pl.Expr]] = {
        "head": pl.col("x").head(2).over(pl.lit(0)),
        "head of three": pl.col("x").head(3).over(pl.lit(0)),
        "slice": pl.col("x").slice(1, 3).over(pl.lit(0)),
        "tail": pl.col("x").tail(2).over(pl.lit(0)),
        "slice to the end": pl.col("x").slice(1).over(pl.lit(0)),
        "head of no row": pl.col("x").head(0).over(pl.lit(0)).median(),
        "head under an alias": pl.col("x").head(2).over(pl.lit(0)).alias("y"),
        "head below plus one": (pl.col("x").head(2) + 1).over(pl.lit(0)),
        "head below a sort": pl.col("x").head(2).sort().over(pl.lit(0)),
        "a key cut by the same slice": pl.col("x").head(2).over(pl.col("g").head(2)),
        "two columns cut by one slice": (pl.col("x").head(2) + pl.col("g").head(2)).over(pl.lit(0)),
        "a head beside a column": (pl.col("x").head(2) + pl.col("x")).over(pl.lit(0)),
        "a head beside a slice that a sort stops": (
            pl.col("x").head(2) + pl.int_range(0, 5).sort().head(2)
        ).over(pl.lit(0)),
        "two heads of two lengths": (pl.col("x").head(2) + pl.col("g").head(3)).over(pl.lit(0)),
        "a head beside first": (pl.col("x").head(2) + pl.col("x").first()).over(pl.lit(0)),
        "a head inside a head": pl.col("x").head(2).head(3).over(pl.lit(0)),
        "a type error beside first": pl.col("t").first() + 1,
        "first after a window over a head": pl.col("x").head(2).over(pl.lit(0)).first(),
        "last after a window over a tail": pl.col("x").tail(2).over(pl.lit(0)).last(),
        "a tail beside a tail": (pl.col("x").tail(2) * pl.col("x").tail(1)).over(pl.lit(0)),
        "two slices of two offsets": (pl.col("x").slice(1, 2) + pl.col("x").head(2)).over(
            pl.lit(0),
        ),
        "a slice from the end inside a head": pl.col("x").slice(-3, 2).head(1).over(pl.lit(0)),  # type: ignore[invalid-call-shape]
        "a head inside a slice from the end": pl.col("x").head(3).slice(-2, 1).over(pl.lit(0)),  # type: ignore[invalid-call-shape]
        "a slice by a first inside a head": pl
        .col("x")
        .slice(pl.col("g").first(), 3)
        .head(2)
        .over(
            pl.lit(0),
        ),
    }

    def polars(self, name: str) -> str:
        """Give the type of the lazy `select` of the tree alone, or how polars stops.

        Args:
            name: The name of the tree.

        Returns:
            The rendering of the type of the run, or `raises`.

        Example:
            >>> LoneSlices().polars("a head beside a column")
            'raises'
        """
        query = OverFrame.FRAME.lazy().select(self.TREES[name])
        return RunText().polars(lambda: query.collect(engine="in-memory").dtypes[0])

    def rules(self, name: str) -> str:
        """Give the run of the tree from the rules, in a lazy `select` that holds it alone.

        Args:
            name: The name of the tree.

        Returns:
            The rendering of the type, `raises`, or `unchecked` where the rules do not know.

        Example:
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     LoneSlices().rules("a head inside a head")
            'Int64'
        """
        frame = GroupBy((3, 2), QueryFlags.lazy(lone=True))
        tree = ExpressionTree().of(self.TREES[name])
        answer = TreeEvaluation(OverFrame().columns(), frame).answers(tree).run_select
        return RunText().rules(answer)


@pytest.mark.parametrize("name", sorted(LoneSlices.TREES))
def test_a_lone_select_runs_over_the_rows_of_its_moved_slices(name: str) -> None:
    """Check an expression that stands alone in a lazy `select` against polars."""
    assert LoneSlices().rules(name) == LoneSlices().polars(name)


def test_an_eager_select_of_a_lone_slice_keeps_the_rows_of_the_frame() -> None:
    """An eager frame runs no slice pushdown, so its window has the rows of the frame."""
    frame = GroupBy((3, 2), QueryFlags.eager())
    tree = ExpressionTree().of(LoneSlices.TREES["head"])
    answer = TreeEvaluation(OverFrame().columns(), frame).answers(tree).run_select
    tree = LoneSlices.TREES["head"]
    polars = RunText().polars(lambda: OverFrame.FRAME.select(tree).dtypes[0])
    assert RunText().rules(answer) == polars == "raises"


class LiteralFirst:
    """Give each elementwise function with a literal first input, and its run in polars and rules.

    `clip` gives the length of its first input. In `agg`, a literal beside a
    list of a group then gives one value where the list holds more, and
    polars refuses the output. A list first, or a column that `agg` does
    not aggregate, keeps the length of each group.

    Example:
        >>> LiteralFirst().polars("by a filtered column", "agg")
        'raises'
        >>> LiteralFirst().rules("by a filtered column", "agg")
        'raises'
    """

    TREES: ClassVar[dict[str, pl.Expr]] = {
        "by a filtered column": pl.lit(1).clip(pl.col("x").filter(pl.col("x") > 0)),
        "by a cumulative sum": pl.lit(1).clip(pl.col("x").cum_sum()),
        "by a column": pl.lit(1).clip(pl.col("x")),
        "a filtered column by a literal": pl.col("x").filter(pl.col("x") > 0).clip(pl.lit(1)),
    }

    def polars(self, name: str, context: str) -> str:
        """Give the type of the eager run of the tree, or how polars stops.

        Args:
            name: The name of the tree.
            context: `select`, or `agg` over the groups of `g`.

        Returns:
            The rendering of the type of the run, or `raises`.

        Example:
            >>> LiteralFirst().polars("by a column", "select")
            'Int32'
        """
        tree = self.TREES[name]
        frame = OverFrame.FRAME
        if context == "select":
            return RunText().polars(lambda: frame.select(tree).dtypes[0])
        return RunText().polars(lambda: frame.group_by("g").agg(tree).dtypes[1])

    def rules(self, name: str, context: str) -> str:
        """Give the run of the tree from the rules, over the groups of the frame.

        Args:
            name: The name of the tree.
            context: `select`, or `agg` over the groups of `g`.

        Returns:
            The rendering of the type, `raises`, or `unchecked` where the rules do not know.

        Example:
            >>> LiteralFirst().rules("a filtered column by a literal", "agg")
            'List(Int64)'
        """
        frame = GroupBy((3, 2), QueryFlags.eager())
        tree = ExpressionTree().of(self.TREES[name])
        answers = TreeEvaluation(OverFrame().columns(), frame).answers(tree)
        return RunText().rules(answers.run_select if context == "select" else answers.run_agg)


@pytest.mark.parametrize("context", ["select", "agg"])
@pytest.mark.parametrize("name", sorted(LiteralFirst.TREES))
def test_a_literal_first_gives_its_length_beside_the_list_of_a_group(
    name: str,
    context: str,
) -> None:
    """Check an elementwise function whose first input is a literal against polars."""
    polars = LiteralFirst().polars(name, context)
    assert LiteralFirst().rules(name, context) in {polars, "unchecked"}


class SortByInputs:
    """Give each `sort_by` in `agg` over inputs of other states, and its run in polars and rules.

    From polars 2.0.0, `sort_by` reads each input as a list for each group
    where the inputs do not share their groups: an input of one value for
    each group beside lists, or a value whose lists a node rebuilt. One value
    for each group then gives no list, and polars panics. Before that
    release, only a rebuilt value reads its key so, and a key of lists then
    runs.

    Example:
        >>> SortByInputs().rules("one value by two values")
        'Int64'
    """

    TREES: ClassVar[dict[str, pl.Expr]] = {
        "a rebuilt value by an imploded key": pl.col("x").cum_sum().sort_by(pl.col("x").implode()),
        "one value by one row": pl.col("x").first().sort_by(pl.col("x").head(1)),
        "one row by one value": pl.col("x").head(1).sort_by(pl.col("t").first()),
        "one value by one value and a literal": pl
        .col("x")
        .first()
        .sort_by([pl.col("t").first(), pl.lit(1)]),
        "one value by two values": pl
        .col("x")
        .first()
        .sort_by([pl.col("t").first(), pl.col("x").last()]),
        "one row by one row": pl.col("x").head(1).sort_by(pl.col("t").head(1)),
    }

    def polars(self, name: str) -> str:
        """Give the type of the eager run of the tree in `agg`, or how polars stops.

        Args:
            name: The name of the tree.

        Returns:
            The rendering of the type of the run, or `raises`.

        Example:
            >>> SortByInputs().polars("one row by one row")
            'List(Int64)'
        """
        tree = self.TREES[name]
        return RunText().polars(lambda: OverFrame.FRAME.group_by("g").agg(tree).dtypes[1])

    def rules(self, name: str) -> str:
        """Give the run of the tree in `agg` from the rules, over the groups of the frame.

        Args:
            name: The name of the tree.

        Returns:
            The rendering of the type, `raises`, or `unchecked` where the rules do not know.

        Example:
            >>> SortByInputs().rules("one row by one row")
            'List(Int64)'
        """
        frame = GroupBy((3, 2), QueryFlags.eager())
        tree = ExpressionTree().of(self.TREES[name])
        return RunText().rules(TreeEvaluation(OverFrame().columns(), frame).answers(tree).run_agg)


@pytest.mark.parametrize("name", sorted(SortByInputs.TREES))
def test_a_sort_by_in_agg_reads_its_inputs_as_polars_does(name: str) -> None:
    """Check a `sort_by` in `agg` over inputs of other states against polars."""
    assert SortByInputs().rules(name) == SortByInputs().polars(name)
