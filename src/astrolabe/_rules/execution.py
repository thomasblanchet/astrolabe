"""How polars runs a node, apart from the type function that types it.

The type function of a node gives its planned type. The run decides the rest:
whether the inputs have lengths that fit, how many rows the output has, and, in
`group_by().agg()`, the state of the values of each group and the list that
each group builds. This module translates those mechanisms of the physical
expressions, the executors and the optimizer of polars. Later code then
computes the shape of a node in a `select` and in `agg` from the shapes of its
inputs.

A value of a `select` is its number of rows. A value of `agg` is the state that
polars keeps for it: its values are not aggregated yet, are one list for each
group, are one value for each group, or are a literal. The rules read concrete
lengths, as polars does: the groups of a frame have sizes, and a check compares
them. A kernel whose length the values decide, such as `unique`, gives the most
rows that it can give. A run that polars stops raises `PolarsRefusal`, and a
node whose run this module does not translate raises `UndecidedError`.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Protocol

from astrolabe._rules.dtypes import DType, IntegerRange, Is
from astrolabe._rules.failures import (
    ErrorClass,
    PolarsRefusal,
    Stop,
    UnequalLengths,
    Undecided,
    UndecidedError,
)
from astrolabe._rules.operators import Operator
from astrolabe._rules.options import FunctionOptions, Height, KernelLength, NodeKind, Shape
from astrolabe._rules.type_functions import Operand
from astrolabe._rules.releases import polars_releases

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class AggState(enum.Enum):
    """Name the state that polars keeps for the values of an expression in `agg`.

    Polars gives such a value one of four states:

    - not aggregated: the flat values, with the groups that index them, as a column gives;
    - aggregated list: one list for each group, as a function that runs once for each group gives;
    - aggregated scalar: one value for each group, as `sum()` or `.first()` gives;
    - literal scalar: one value for all the groups, as `pl.lit(1)` gives.

    Example:
        >>> AggState.LITERAL_SCALAR.value
        'literal scalar'
    """

    NOT_AGGREGATED = "not aggregated"
    AGGREGATED_LIST = "aggregated list"
    AGGREGATED_SCALAR = "aggregated scalar"
    LITERAL_SCALAR = "literal scalar"


@dataclass(frozen=True)
class InSelect:
    """Hold a value of a `select` as its run holds it: its number of rows and its height.

    `height` is the projection height that the planner gives the value. A
    check of the frame reads it: a value of one row spreads to the height
    of the frame only where the planner knows it to be one value. `decided`
    says that the values decide the length, as for a `filter`: `length` is
    then the most rows, and the data can give any other number of rows.
    `bounded` says that the code bounds the length, as for a literal or a
    `head(2)`: no frame gives more than `length` rows.

    Example:
        >>> InSelect.column(5)
        InSelect(length=5, height=<Height.COLUMN: 'column'>, decided=False, bounded=False)
    """

    length: int
    height: Height
    decided: bool = False
    bounded: bool = False

    @classmethod
    def column(cls, rows: int) -> InSelect:
        """Give a column of a frame of `rows` rows.

        Args:
            rows: The height of the frame.

        Returns:
            The value.

        Example:
            >>> InSelect.column(3).length
            3
        """
        return cls(rows, Height.COLUMN)

    @classmethod
    def literal(cls) -> InSelect:
        """Give a literal of one value.

        Returns:
            The value.

        Example:
            >>> InSelect.literal().height
            <Height.SCALAR: 'scalar'>
        """
        return cls(1, Height.SCALAR, bounded=True)


@dataclass(frozen=True)
class InGroups:
    """Hold a value of `agg` as polars holds it, in an aggregation context.

    An aggregation context has a state, the groups that index its values, and a
    flag that says whether those groups are the groups of the `group_by`.
    `groups` holds the size of each group as the groups of the context give it.
    `rows` is the number of flat values: the whole column for values that are
    not aggregated, and the values of all the lists for an aggregated list.
    `height` is the projection height that the planner gives the expression,
    after the cast pass. `dtype` is the data type of the values, where it is
    known. `decided` says that the values decide the size of each group, as
    for a `filter`: `groups` then holds the most rows of each group.
    `bounded` says that the code bounds the size of each group, as for a
    `head(2)`: no frame gives more rows. One value for each group is bounded.
    `rebuilt` says that a node built the list of each group anew, as a
    function that runs group by group, such as `cum_sum`, does: the values
    then follow other groups than the groups of the context. `overlapping`
    says that the groups are windows that share values, as the groups of a
    `rolling` do, so a value that is not aggregated belongs to several groups.

    Example:
        >>> value = GroupBy((3, 2)).column()
        >>> value.state, value.values()
        (<AggState.NOT_AGGREGATED: 'not aggregated'>, (3, 2))
    """

    state: AggState
    groups: tuple[int, ...]
    height: Height
    rows: int
    original: bool = True
    dtype: DType | None = None
    decided: bool = False
    bounded: bool = False
    rebuilt: bool = False
    overlapping: bool = False

    SINGLE: ClassVar[frozenset[AggState]] = frozenset({
        AggState.AGGREGATED_SCALAR,
        AggState.LITERAL_SCALAR,
    })

    def values(self) -> tuple[int, ...]:
        """Give the number of values that each group holds for a kernel.

        An aggregated scalar and a literal hold one value in each group: a node
        that runs once for each group reads that one value.

        Returns:
            The number of values of each group.

        Example:
            >>> GroupBy((3, 2)).literal().values()
            (1, 1)
        """
        if self.state in self.SINGLE:
            return (1,) * len(self.groups)
        return self.groups

    def flat(self) -> int:
        """Give the number of flat values that a kernel reads where it ignores the groups.

        Values that are not aggregated give the whole column, and an aggregated
        list gives the values of all its lists. An aggregated scalar gives one
        value for each group, and a literal gives its one value.

        Returns:
            The number of values.

        Example:
            >>> GroupBy((3, 2)).literal().flat()
            1
        """
        if self.state is AggState.LITERAL_SCALAR:
            return 1
        if self.state is AggState.AGGREGATED_SCALAR:
            return len(self.groups)
        return self.rows

    def covers(self) -> bool:
        """Say whether the groups index every flat value.

        A slice drops values from its groups, and the column keeps them.
        A fallible node then runs once for each group, so that a dropped
        value cannot raise.

        Returns:
            Whether the sizes of the groups add up to the flat values.

        Example:
            >>> GroupBy((3, 2)).column().covers()
            True
        """
        return self.state in self.SINGLE or sum(self.groups) == self.rows

    def is_aggregated(self) -> bool:
        """Say whether the state is an aggregated list or an aggregated scalar.

        Returns:
            Whether the values are aggregated.

        Example:
            >>> GroupBy((3, 2)).literal().is_aggregated()
            False
        """
        return self.state in {AggState.AGGREGATED_LIST, AggState.AGGREGATED_SCALAR}

    def aggregated(self) -> InGroups:
        """Give the values aggregated into one list for each group.

        Values that are not aggregated become the list of each group. A
        literal becomes a list of its one value for each group.

        Returns:
            The values, in the state aggregated list where they were not aggregated.

        Example:
            >>> GroupBy((3, 2)).literal().aggregated().groups
            (1, 1)
        """
        if self.state is AggState.NOT_AGGREGATED:
            aggregated = AggState.AGGREGATED_LIST
            return replace(self, state=aggregated, rows=sum(self.groups), overlapping=False)
        if self.state is AggState.LITERAL_SCALAR:
            ones = (1,) * len(self.groups)
            return replace(self, state=AggState.AGGREGATED_LIST, groups=ones, rows=len(ones))
        return self

    def with_unit_groups(self) -> InGroups:
        """Give each value one group of its own.

        An aggregated scalar and a literal have no groups of their own. A node
        that runs once for each group first gives each group one value. The new
        groups are not those of the `group_by`.

        Returns:
            The values with one group of size 1 for each group.

        Example:
            >>> GroupBy((3, 2)).literal().with_unit_groups().groups
            (1, 1)
        """
        if self.state not in self.SINGLE:
            return self
        return replace(self, groups=(1,) * len(self.groups), original=False)


@dataclass(frozen=True)
class AggOutput:
    """Hold the column that `agg` gives for one expression.

    `lists` holds the length of the list of each group where the column
    holds a list for each group. It is `None` where the column holds one
    value for each group.

    Example:
        >>> AggOutput((2, 1)).listed()
        True
    """

    lists: tuple[int, ...] | None

    def listed(self) -> bool:
        """Say whether the column holds a list for each group.

        Returns:
            Whether it does.

        Example:
            >>> AggOutput(None).listed()
            False
        """
        return self.lists is not None


@dataclass(frozen=True)
class QueryFlags:
    """Say which passes of the optimizer of polars run where a frame runs an expression.

    The optimizer can put an input or a literal in place of a node
    (`simplify`), and a `collect()` of a lazy frame moves a slice down the
    tree (`slice_pushdown`). An eager `select` or `with_columns` simplifies
    and moves no slice. An eager `group_by().agg()` runs no pass at all. A
    lazy frame runs both passes, in a `select` and in `agg`. The default is
    the eager `select` in both contexts. Where an expression stands alone in
    a lazy `select` (`lone`), the projection pushdown can put a count of the
    rows in place of a `len` (`RowCount`).

    Attributes:
        simplify_select: Whether the optimizer simplifies a `select`.
        simplify_agg: Whether the optimizer simplifies `agg`.
        slice_pushdown: Whether the optimizer moves a slice down the tree.
        lone: Whether the expression is the only expression of a lazy `select`.

    Example:
        >>> QueryFlags.eager().simplify_agg, QueryFlags.lazy().slice_pushdown
        (False, True)
    """

    simplify_select: bool = True
    simplify_agg: bool = True
    slice_pushdown: bool = False
    lone: bool = False

    @classmethod
    def eager(cls) -> QueryFlags:
        """Give the passes of an eager frame: a `select` simplifies, and `agg` runs none.

        Returns:
            The flags.

        Example:
            >>> QueryFlags.eager().simplify_select
            True
        """
        return cls(simplify_agg=False)

    @classmethod
    def lazy(cls, *, lone: bool = False) -> QueryFlags:
        """Give the passes of a lazy frame that collects: every pass runs.

        Args:
            lone: Whether the expression is the only expression of a `select`.

        Returns:
            The flags.

        Example:
            >>> QueryFlags.lazy().simplify_agg, QueryFlags.lazy(lone=True).lone
            (True, True)
        """
        return cls(slice_pushdown=True, lone=lone)


@dataclass(frozen=True)
class GroupBy:
    """Hold the groups of a `group_by`, and give the inputs and the output of `agg`.

    `sizes` holds the number of rows of each group. A column of the frame and a
    literal each start in a state of their own. At the end, the planner wraps
    an expression that is not one value in an `implode`, and the run gives one
    value or one list for each group. `flags` says which passes of the
    optimizer run over the frame. The groups of a `rolling` or of a
    `group_by_dynamic` are windows of the frame: `height` gives the rows of
    the frame where the sizes do not add up to them, and `overlapping` says
    that the windows share rows.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> frame.output(frame.column(), wrapped=True)
        AggOutput(lists=(3, 2))
        >>> GroupBy((1, 2, 2, 2, 2), height=5, overlapping=True).column().rows
        5
    """

    sizes: tuple[int, ...]
    flags: QueryFlags = QueryFlags()
    height: int | None = None
    overlapping: bool = False

    def column(self, dtype: DType | None = None) -> InGroups:
        """Give a column of the frame.

        Args:
            dtype: The data type of the column, where it matters.

        Returns:
            The values in the state not aggregated, with the groups of the `group_by`.

        Example:
            >>> GroupBy((3, 2)).column().rows
            5
        """
        rows = sum(self.sizes) if self.height is None else self.height
        state = AggState.NOT_AGGREGATED
        value = InGroups(state, self.sizes, Height.COLUMN, rows, dtype=dtype)
        return replace(value, overlapping=self.overlapping)

    def literal(self, dtype: DType | None = None) -> InGroups:
        """Give a literal of one value.

        Args:
            dtype: The data type of the literal, where it matters.

        Returns:
            The value in the state literal scalar.

        Example:
            >>> GroupBy((3, 2)).literal().state
            <AggState.LITERAL_SCALAR: 'literal scalar'>
        """
        return InGroups(AggState.LITERAL_SCALAR, self.sizes, Height.SCALAR, 1, dtype=dtype)

    def output(self, value: InGroups, *, wrapped: bool) -> AggOutput:
        """Give the column that `agg` gives for an expression whose run gives `value`.

        Where the planner wraps the expression, the `implode` gives each group
        the list of its values. Otherwise the run gives the values as they are,
        and a literal spreads to each group. The group by then checks that the
        column has one row for each group.

        It raises `PolarsRefusal` where the column does not have one row for each group.

        Args:
            value: The values that the run of the expression gives.
            wrapped: Whether the planner wraps the expression (`Shape.wraps_in_agg`).

        Returns:
            The column.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> frame.output(frame.literal(), wrapped=False)
            AggOutput(lists=None)
            >>> frame.output(frame.column(), wrapped=False)
            AggOutput(lists=(3, 2))
        """
        if wrapped:
            return AggOutput(value.values())
        if value.state is AggState.LITERAL_SCALAR:
            return AggOutput(None)
        if value.state is AggState.AGGREGATED_SCALAR:
            self._check_rows(value.rows)
            return AggOutput(None)
        return AggOutput(value.aggregated().groups)

    def _check_rows(self, rows: int) -> None:
        """Check that an aggregated scalar has one value for each group.

        Args:
            rows: The number of values.

        Raises:
            PolarsRefusal: The number differs from the number of groups.

        Example:
            >>> GroupBy((3, 2))._check_rows(2)
        """
        if rows != len(self.sizes):
            message = "the aggregation does not give one value for each group"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)


class PhysicalExpr(enum.Enum):
    """Name the physical expressions of polars that this module translates.

    The planner of the run picks one for each node. A function runs as `APPLY`,
    except `min_by` and `max_by`, which run as `MIN_MAX_BY`, and `arg_min` and
    `arg_max`, which run as `AGGREGATION`. A few functions run as `APPLY` with
    a kernel of their own for the groups of `agg`: `GROUPS_FUNCTION` keeps an
    input of one value as it is, `COUNT_GROUPS` counts the values of each
    group first, and `REGROUP` moves the values of each group with no kernel,
    as `reverse` does. An `explode` runs as `APPLY` with the options of
    `groupwise()`. `implode()` is an aggregation with a path of its own. A
    literal runs as `LITERAL`.

    Example:
        >>> PhysicalExpr.of(NodeKind.EXPLODE)
        <PhysicalExpr.APPLY: 'apply'>
    """

    APPLY = "apply"
    AGGREGATION = "aggregation"
    IMPLODE = "implode"
    BINARY = "binary"
    TERNARY = "ternary"
    CAST = "cast"
    SORT = "sort"
    SLICE = "slice"
    FILTER = "filter"
    SORT_BY = "sort_by"
    GATHER = "gather"
    MIN_MAX_BY = "min_max_by"
    GROUPS_FUNCTION = "groups function"
    COUNT_GROUPS = "count groups"
    LITERAL = "literal"
    REGROUP = "regroup"

    @classmethod
    def of(cls, kind: NodeKind) -> PhysicalExpr:
        """Give the physical expression of a kind of node.

        Args:
            kind: The kind of node.

        Returns:
            The physical expression.

        Raises:
            UndecidedError: This module does not translate the run of the kind.

        Example:
            >>> PhysicalExpr.of(NodeKind.BINARY)
            <PhysicalExpr.BINARY: 'binary'>
            >>> PhysicalExpr.of(NodeKind.OVER)
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not translated: the run of Over
        """
        kinds = {
            NodeKind.FUNCTION: cls.APPLY,
            NodeKind.EXPLODE: cls.APPLY,
            NodeKind.AGG: cls.AGGREGATION,
            NodeKind.BINARY: cls.BINARY,
            NodeKind.TERNARY: cls.TERNARY,
            NodeKind.CAST: cls.CAST,
            NodeKind.SORT: cls.SORT,
            NodeKind.SLICE: cls.SLICE,
            NodeKind.FILTER: cls.FILTER,
            NodeKind.SORT_BY: cls.SORT_BY,
            NodeKind.GATHER: cls.GATHER,
            NodeKind.LITERAL: cls.LITERAL,
        }
        if kind not in kinds:  # pragma: no cover -- a leaf, over and rolling run elsewhere
            raise UndecidedError(Undecided.NOT_TRANSLATED, f"the run of {kind.value}")
        return kinds[kind]


@dataclass(frozen=True)
class Kernel:
    """Hold the facts of the kernel of a node that the run reads besides its shape.

    `planned` is the type of the output in the plan that runs, after the cast
    pass: the physical planner reads it from that plan. `collect_schema()`
    gives the type before the cast pass, which can differ. `data` is the data
    type that the kernel gives at the data types of the inputs. A node that
    runs once for each group builds the list of each group at the planned type
    (`ListBuilder`), so the two can disagree. `fallible` says that the kernel
    can raise on a value: a strict cast, `list.get`, `replace_strict` and a
    strict `strptime` can, and so can arithmetic on a `Decimal`. `strict` says
    that a cast raises on a value that does not fit. `physical` names the
    physical expression of polars where it is not the default of the kind of
    the node. `before` says that the run stops before the kernel: it makes
    the checks that polars makes before it calls the kernel, and leaves the
    checks of the output of the kernel to a run after the kernel.
    `own_lengths` says that, where the kernel runs once for each group, it
    checks the lengths of the sides of a group itself, after its checks of
    the types. The run then checks those lengths only after the kernel.

    Example:
        >>> Kernel(physical=PhysicalExpr.IMPLODE).physical
        <PhysicalExpr.IMPLODE: 'implode'>
    """

    planned: DType | None = None
    data: DType | None = None
    fallible: bool = False
    strict: bool = False
    physical: PhysicalExpr | None = None
    before: bool = False
    own_lengths: bool = False


class GroupLengths:
    """Check that values aggregated into lists have groups of one size.

    A value whose group sizes the values decide passes: the data can give
    the sizes of the others.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> GroupLengths().check([frame.column(), frame.column()])
    """

    def check(self, values: Sequence[InGroups]) -> None:
        """Check the sizes of the groups of the values.

        Args:
            values: The values.

        Raises:
            PolarsRefusal: Two values have groups of other sizes.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> GroupLengths().check([frame.column(), frame.literal().aggregated()])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
        """
        sizes = {value.groups for value in values if not value.decided}
        if len(sizes) > 1:
            message = "the groups have other sizes"
            raise PolarsRefusal(ErrorClass.SHAPE, message)


class Selector(Protocol):
    """Give the output of one kind of node in a `select`."""

    def select(self, inputs: Sequence[InSelect]) -> InSelect:
        """Give the output.

        Args:
            inputs: The inputs.
        """


class GroupRun(Protocol):
    """Give the output of one kind of node over the groups of `agg`."""

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output.

        Args:
            inputs: The inputs.
            frame: The groups of the `group_by`.
        """


class KernelCalls(Protocol):
    """Give the number of rows of each input at each call of the kernel of a node in `agg`."""

    def calls(self, inputs: Sequence[InGroups]) -> tuple[tuple[int, ...], ...]:
        """Give one tuple for each call of the kernel, with the rows of each input.

        Args:
            inputs: The inputs.
        """


class KernelPath(enum.Enum):
    """Name the path on which a node of several inputs calls its kernel in `agg`.

    - literals: once, on the literals;
    - groups: once for each group, on the values of the group;
    - flat: once, on the flat values of all the groups.

    The path decides the rows that the kernel sees. On the flat path, the
    values of a column that a slice cut keep all the rows of the column.

    Example:
        >>> KernelPath.FLAT.value
        'flat'
    """

    LITERALS = "literals"
    GROUPS = "groups"
    FLAT = "flat"


class GroupCalls:
    """Give the calls of a kernel that runs once for each group, on the values of the group.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> GroupCalls(Shape.of(NodeKind.SORT), Kernel()).calls([frame.column()])
        ((3,), (2,))
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Take the shape and the kernel of the node, which this rule does not read.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> GroupCalls(Shape.of(NodeKind.CAST), Kernel()).calls([])
            ()
        """
        del shape, kernel

    def calls(self, inputs: Sequence[InGroups]) -> tuple[tuple[int, ...], ...]:
        """Give one call for each group.

        Args:
            inputs: The inputs.

        Returns:
            The rows of each input in each group.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> GroupCalls(Shape.of(NodeKind.AGG), Kernel()).calls([frame.literal()])
            ((1,), (1,))
        """
        counts = [value.values() for value in inputs]
        return tuple(zip(*counts, strict=True))


class Run:
    """Run one node of a plan of polars over the shapes of its inputs.

    The shape of the node (`Shape`) gives its projection height and the rule of
    the length of its kernel. The kernel (`Kernel`) gives the facts that only
    the run reads. The run then follows the physical expression that polars
    plans for the node: `PhysicalExpr.APPLY` for a function,
    `PhysicalExpr.BINARY` for `a + b`, and so on. In a `select`, the run gives
    the number of rows of the output. In `agg`, it gives the state of the
    output and the size of each group.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
        >>> Run(Shape.of(NodeKind.BINARY)).agg([frame.column(), first], frame).state
        <AggState.AGGREGATED_LIST: 'aggregated list'>
    """

    def __init__(self, shape: Shape, kernel: Kernel | None = None) -> None:
        """Pick the physical expression of the node.

        It raises `UndecidedError` where this module does not translate the
        run of the kind of the node (`PhysicalExpr.of`).

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel, or `None` where the run needs none.

        Example:
            >>> Run(Shape.of(NodeKind.SORT)).select([InSelect.column(4)]).length
            4
        """
        facts = Kernel() if kernel is None else kernel
        physical = facts.physical or PhysicalExpr.of(shape.kind)
        runs: dict[PhysicalExpr, Callable[[Shape, Kernel], GroupRun]] = {
            PhysicalExpr.APPLY: ApplyRun,
            PhysicalExpr.AGGREGATION: AggregationRun,
            PhysicalExpr.IMPLODE: ImplodeRun,
            PhysicalExpr.BINARY: BinaryRun,
            PhysicalExpr.TERNARY: TernaryRun,
            PhysicalExpr.CAST: CastRun,
            PhysicalExpr.SORT: SortRun,
            PhysicalExpr.SLICE: SliceRun,
            PhysicalExpr.FILTER: FilterRun,
            PhysicalExpr.SORT_BY: SortByRun,
            PhysicalExpr.GATHER: GatherRun,
            PhysicalExpr.MIN_MAX_BY: MinMaxByRun,
            PhysicalExpr.GROUPS_FUNCTION: GroupsFunctionRun,
            PhysicalExpr.COUNT_GROUPS: CountGroupsRun,
            PhysicalExpr.LITERAL: LiteralRun,
            PhysicalExpr.REGROUP: RegroupRun,
        }
        selects: dict[PhysicalExpr, Callable[[Shape], Selector]] = {
            PhysicalExpr.SLICE: SliceSelect,
            PhysicalExpr.MIN_MAX_BY: MinMaxBySelect,
        }
        callers: dict[PhysicalExpr, Callable[[Shape, Kernel], KernelCalls]] = {
            PhysicalExpr.APPLY: ApplyRun,
            PhysicalExpr.BINARY: BinaryRun,
            PhysicalExpr.TERNARY: TernaryRun,
            PhysicalExpr.REGROUP: RegroupRun,
        }
        self._select = selects.get(physical, KernelSelect)(shape)
        self._groups = runs[physical](shape, facts)
        self._calls = callers.get(physical, GroupCalls)(shape, facts)

    def select(self, inputs: Sequence[InSelect]) -> InSelect:
        """Give the output of the node in a `select`.

        Args:
            inputs: The inputs, in the order of polars.

        Returns:
            The output.

        Example:
            >>> Run(Shape.of(NodeKind.AGG)).select([InSelect.column(5)])
            InSelect(length=1, height=<Height.SCALAR: 'scalar'>, decided=False, bounded=True)
        """
        return self._select.select(inputs)

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output of the node over the groups of `agg`.

        Args:
            inputs: The inputs, in the order of polars.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame).values()
            (1, 1)
        """
        return self._groups.agg(inputs, frame)

    def calls(self, inputs: Sequence[InGroups]) -> tuple[tuple[int, ...], ...]:
        """Give the rows of each input at each call of the kernel of the node in `agg`.

        A kernel that reads the lengths of its inputs, such as one that spreads
        an input of one row, sees these rows. Most nodes call their kernel once
        for each group. A function, a binary node and a `when` can call it once
        on the flat values of all the groups (`KernelPath`).

        Args:
            inputs: The inputs, in the order of polars.

        Returns:
            One tuple for each call, with the rows of each input.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> sliced = Run(Shape(NodeKind.SLICE, length=KernelLength.slice(0, 1)))
            >>> one_row = sliced.agg([frame.column(), frame.literal(), frame.literal()], frame)
            >>> Run(Shape.of(NodeKind.BINARY)).calls([frame.literal(), one_row])
            ((1, 5),)
        """
        return self._calls.calls(inputs)


class KernelSelect:
    """Give the output of a node in a `select`: the kernel over the whole inputs.

    In a `select`, a node reads each input whole, and its kernel gives the
    length of the output by its rule (`KernelLength`). The kinds of node
    that check their inputs in another way have a class of their own:
    `SliceSelect` and `MinMaxBySelect`.

    Example:
        >>> KernelSelect(Shape.of(NodeKind.AGG)).select([InSelect.column(5)]).height
        <Height.SCALAR: 'scalar'>
    """

    def __init__(self, shape: Shape) -> None:
        """Keep the shape of the node.

        Args:
            shape: The shape.

        Example:
            >>> KernelSelect(Shape.of(NodeKind.SORT)).select([InSelect.column(4)]).length
            4
        """
        self._shape = shape

    def select(self, inputs: Sequence[InSelect]) -> InSelect:
        """Give the output.

        It raises `PolarsRefusal` where the kernel does not accept the lengths of the inputs.

        Args:
            inputs: The inputs.

        Returns:
            The output.

        Example:
            >>> inputs = [InSelect.column(3), InSelect.column(2)]
            >>> KernelSelect(Shape.of(NodeKind.BINARY)).select(inputs)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
        """
        rule = self._shape.kernel_length()
        lengths = [value.length for value in inputs]
        decided = [value.decided for value in inputs]
        length = rule.length(lengths, decided)
        height = self._shape.height([value.height for value in inputs])
        bounded = rule.bounded(lengths, [value.bounded for value in inputs], length)
        return InSelect(length, height, rule.decided(lengths, decided), bounded)


class ListBuilder:
    """Collect the result of each group of a group-aware node into one list for each group.

    A node that runs once for each group builds the list of each group with a
    builder of its planned output type, where that type holds no `Unknown`. The
    builder follows the physical type of the plan:

    - a number, a temporal type, a `Decimal` or a categorical type takes data
      of the same physical type, and gives the planned type;
    - `Boolean`, `String` and `Binary` take only data of that type;
    - a `List`, an `Array` or a `Struct` takes data of exactly the planned
      type, or `Null`;
    - `Null` takes data of any type, and holds only nulls.

    A builder that refuses the data of a group raises. The builder of a nested
    type gives a `ComputeError`, and the other builders give a `SchemaError`. A
    node of several inputs gives that error, and a node of one input panics
    there.

    Example:
        >>> ListBuilder(DType("Int64")).build(DType("Datetime", ("us", None)))
        Int64
        >>> ListBuilder(DType("Null")).build(DType("Float64"))
        Null
    """

    BINARIES: ClassVar[frozenset[str]] = frozenset({"Boolean", "String", "Binary"})
    ANONYMOUS: ClassVar[frozenset[str]] = frozenset({"List", "Array", "Struct"})

    def __init__(self, planned: DType) -> None:
        """Keep the planned type of the output of the node.

        Args:
            planned: The planned type.

        Example:
            >>> ListBuilder(DType("Boolean")).accepts(DType("Int64"))
            False
        """
        self._planned = planned

    def accepts(self, data: DType) -> bool:
        """Say whether the builder takes the data of a group.

        Args:
            data: The data type that the kernel gives for the group.

        Returns:
            Whether the builder takes it.

        Example:
            >>> ListBuilder(DType("Decimal", (10, 2))).accepts(DType("Decimal", (38, 2)))
            True
        """
        planned = self._planned
        if planned.name == "Null":
            return True
        if planned.name in self.BINARIES:
            return data.name == planned.name
        if planned.name in self.ANONYMOUS:
            return data.name == "Null" or data.same(planned)
        return data.physical().same(planned.physical())

    def build(self, data: DType) -> DType:
        """Give the type of the values of each list that the builder builds.

        Args:
            data: The data type that the kernel gives for each group.

        Returns:
            The planned type, which the builder keeps whatever the data.

        Raises:
            PolarsRefusal: The builder refuses the data.

        Example:
            >>> ListBuilder(DType("Int64")).build(DType("Float64"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
            >>> planned = DType("Struct", ((("a", DType("Int64")),),))
            >>> data = DType("Struct", ((("b", DType("Int64")),),))
            >>> ListBuilder(planned).build(data)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        if self.accepts(data):
            return self._planned
        if self._planned.name in self.ANONYMOUS:
            raise PolarsRefusal(ErrorClass.COMPUTE, "dtypes don't match")
        raise PolarsRefusal(ErrorClass.SCHEMA, "cannot build list with different dtypes")


class GroupAware:
    """Run a function once for each group of `agg`, and build the list of each group.

    Each group gives the kernel the values of the group of each input, and the
    kernel gives the values of the output of the group. The run builds the list
    of each group at the planned type (`ListBuilder`). A node that the planner
    knows to give one value then keeps the one value of each group, and raises
    where a group gives more.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> cum_sum = Shape.function(FunctionOptions.length_preserving())
        >>> GroupAware(cum_sum, Kernel()).run([frame.column()]).groups
        (3, 2)
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> GroupAware(Shape.of(NodeKind.EXPLODE), Kernel())._lengths.length([2])
            2
        """
        self._shape = shape
        self._kernel = kernel
        self._lengths = shape.kernel_length()

    def run(self, inputs: Sequence[InGroups]) -> InGroups:
        """Give the output over the groups.

        A node of one input first gives each value of one value a group of its
        own. A node of several inputs keeps the groups of its first input.
        Polars calls the kernel of each group with no check of the lengths
        before it. So a kernel that checks the lengths of a group itself
        (`Kernel.own_lengths`) meets that check only in the run after it.

        It raises `PolarsRefusal` where the inputs of a group do not fit, the builder refuses
        the data, or a group gives more than one value where the planner knows the node to give
        one value.

        Args:
            inputs: The inputs.

        Returns:
            The output: one value for each group where the planner knows the
            node to give one value, and one list for each group otherwise.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> mode = Shape.function(FunctionOptions.groupwise(), KernelLength.shrinks())
            >>> GroupAware(mode, Kernel()).run([frame.literal()]).state
            <AggState.AGGREGATED_LIST: 'aggregated list'>
            >>> by = Shape.function(FunctionOptions.length_preserving())
            >>> state, unknown = AggState.NOT_AGGREGATED, Height.UNKNOWN
            >>> sliced = InGroups(state, (2, 2), unknown, 5, original=False)
            >>> own = Kernel(before=True, own_lengths=True)
            >>> GroupAware(by, own).run([frame.column(), sliced]).groups
            (3, 2)
        """
        base = inputs[0].with_unit_groups() if len(inputs) == 1 else inputs[0]
        calls = self.calls(inputs)
        decided = [value.decided for value in inputs]
        sizes = self._sizes(calls, decided)
        dtype = self._dtype(inputs)
        height = self._shape.height([value.height for value in inputs])
        if height is Height.SCALAR:
            return self._scalar(base, sizes, dtype)
        rows = sum(sizes)
        free = any(self._lengths.decided(call, decided) for call in calls)
        out = InGroups(AggState.AGGREGATED_LIST, sizes, height, rows, dtype=dtype, decided=free)
        return replace(out, original=base.original, rebuilt=True)

    def _sizes(
        self,
        calls: Sequence[Sequence[int]],
        decided: Sequence[bool],
    ) -> tuple[int, ...]:
        """Give the size of each group of the output: the rows that each call of the kernel gives.

        A run that stops before a kernel that checks its own lengths keeps the
        rows of the first input of each group.

        Args:
            calls: The rows of each input at each call.
            decided: Whether the values decide the rows of each input.

        Returns:
            The sizes.

        Example:
            >>> cum_sum = Shape.function(FunctionOptions.length_preserving())
            >>> GroupAware(cum_sum, Kernel())._sizes([(3,), (2,)], [False])
            (3, 2)
        """
        if self._kernel.before and self._kernel.own_lengths:
            return tuple(call[0] for call in calls)
        return tuple(self._lengths.length(call, decided) for call in calls)

    def calls(self, inputs: Sequence[InGroups]) -> tuple[tuple[int, ...], ...]:
        """Give the rows of each input at each call of the kernel: one call for each group.

        Args:
            inputs: The inputs.

        Returns:
            The rows of each input in each group.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> cum_sum = Shape.function(FunctionOptions.length_preserving())
            >>> GroupAware(cum_sum, Kernel()).calls([frame.column(), frame.literal()])
            ((3, 1), (2, 1))
        """
        return GroupCalls(self._shape, self._kernel).calls(inputs)

    def _dtype(self, inputs: Sequence[InGroups]) -> DType | None:
        """Give the type of the values that the builder of each group gives.

        It raises `PolarsRefusal` where the builder refuses the data. A node of
        several inputs gives the error of the builder. A node of one input
        collects its groups in parallel, and that collection turns the error
        into a panic. A node of one input over values that are not aggregated
        in windows that overlap builds each list from its data, with no
        builder, so the lists take the type of the data.

        Args:
            inputs: The inputs of the node.

        Returns:
            The planned type where the builder takes the data, the data type
            where the planned type holds an `Unknown` or where no builder runs,
            or `None` where the kernel gives no type.

        Raises:
            PolarsRefusal: The builder refuses the data.

        Example:
            >>> column = GroupBy((3, 2)).column()
            >>> kernel = Kernel(DType("Null"), DType("Int8"))
            >>> GroupAware(Shape.of(NodeKind.EXPLODE), kernel)._dtype([column])
            Null
            >>> kernel = Kernel(DType("Int64"), DType("Float64"))
            >>> GroupAware(Shape.of(NodeKind.EXPLODE), kernel)._dtype([column])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
            >>> windows = GroupBy((1, 2, 2), height=3, overlapping=True).column()
            >>> GroupAware(Shape.of(NodeKind.EXPLODE), kernel)._dtype([windows])
            Float64
        """
        planned, data = self._kernel.planned, self._kernel.data
        if planned is None or data is None or not Is.known(planned):
            return data
        if len(inputs) == 1 and self._unbuilt(inputs[0]):
            return data
        try:
            return ListBuilder(planned).build(data)
        except PolarsRefusal as refusal:
            if len(inputs) > 1:
                raise
            raise PolarsRefusal(Stop.PANICS, refusal.message) from refusal

    def _unbuilt(self, value: InGroups) -> bool:
        """Say whether polars builds the lists of one input from their data, with no builder.

        Over windows that overlap, polars runs the node on each window as it
        reads it, and collects the outputs into a list with the type of their
        data.

        Args:
            value: The input.

        Returns:
            Whether the input is not aggregated, in windows that overlap.

        Example:
            >>> GroupAware(Shape.of(NodeKind.EXPLODE), Kernel())._unbuilt(GroupBy((3, 2)).column())
            False
        """
        return value.state is AggState.NOT_AGGREGATED and value.overlapping

    def _scalar(self, base: InGroups, sizes: tuple[int, ...], dtype: DType | None) -> InGroups:
        """Keep the one value of each group.

        The explode of the lists counts an empty list as one null.

        Args:
            base: The input whose groups the output keeps.
            sizes: The number of values that each group gives.
            dtype: The type of the values.

        Returns:
            The output in the state aggregated scalar.

        Raises:
            PolarsRefusal: A group gives more than one value.

        Example:
            >>> base = GroupBy((3, 2)).column()
            >>> GroupAware(Shape.of(NodeKind.EXPLODE), Kernel())._scalar(base, (1, 0), None).rows
            2
            >>> GroupAware(Shape.of(NodeKind.EXPLODE), Kernel())._scalar(base, (2, 1), None)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: expected scalar ...
        """
        if any(size > 1 for size in sizes):  # pragma: no cover -- its kernel gives one value
            message = "expected scalar for expr"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return InGroups(
            AggState.AGGREGATED_SCALAR,
            base.groups,
            Height.SCALAR,
            len(sizes),
            original=base.original,
            dtype=dtype,
        )


class ApplyRun:
    """Run a function as polars runs it, with the physical expression `PhysicalExpr.APPLY`.

    In a `select`, the kernel reads the whole inputs. In `agg`, an elementwise
    function of one input works on the flat values and keeps their state. An
    elementwise function of several inputs picks its path from the states of
    its inputs (`ElementwiseDispatch`). Any other function runs once for each
    group (`GroupAware`).

    Example:
        >>> frame = GroupBy((3, 2))
        >>> abs_ = ApplyRun(Shape.function(FunctionOptions.elementwise()), Kernel())
        >>> abs_.agg([frame.literal()], frame).state
        <AggState.LITERAL_SCALAR: 'literal scalar'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the function.

        An `explode` runs with the options of `groupwise()`.

        Args:
            shape: The shape of the function.
            kernel: The facts of its kernel.

        Example:
            >>> ApplyRun(Shape.of(NodeKind.EXPLODE), Kernel())._options.is_elementwise()
            False
        """
        self._shape = shape
        self._kernel = kernel
        self._options = shape.options if shape.kind is NodeKind.FUNCTION else FunctionOptions()
        self._group_aware = GroupAware(shape, kernel)

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output over the groups of `agg`.

        Args:
            inputs: The inputs.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        A function with no input, as `pl.format` of a text with no
        placeholder, gives one literal value.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> rank = ApplyRun(Shape.function(FunctionOptions.length_preserving()), Kernel())
            >>> rank.agg([frame.literal()], frame).state
            <AggState.AGGREGATED_SCALAR: 'aggregated scalar'>
            >>> elementwise = ApplyRun(Shape.function(FunctionOptions.elementwise()), Kernel())
            >>> elementwise.agg([], frame).state
            <AggState.LITERAL_SCALAR: 'literal scalar'>
        """
        if not inputs:
            return frame.literal(self._kernel.data)
        if not self._options.is_elementwise():
            return self._group_aware.run(inputs)
        if len(inputs) == 1:
            return self._single(inputs[0])
        return ElementwiseDispatch(self._shape, self._kernel, inputs).run()

    def calls(self, inputs: Sequence[InGroups]) -> tuple[tuple[int, ...], ...]:
        """Give the rows of each input at each call of the kernel, on the path of `agg`.

        Args:
            inputs: The inputs.

        Returns:
            One tuple for each call, with the rows of each input.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> abs_ = ApplyRun(Shape.function(FunctionOptions.elementwise()), Kernel())
            >>> abs_.calls([frame.column()])
            ((5,),)
            >>> kept = InGroups(AggState.AGGREGATED_LIST, (2, 1), Height.UNKNOWN, 5, original=False)
            >>> elementwise = Shape.function(FunctionOptions.elementwise())
            >>> fallible = ApplyRun(elementwise, Kernel(fallible=True))
            >>> fallible.calls([kept])
            ((2,), (1,))
        """
        if not self._options.is_elementwise():
            return self._group_aware.calls(inputs)
        if (  # pragma: no cover -- every fallible kernel takes two inputs
            len(inputs) == 1 and self._kernel.fallible and not inputs[0].covers()
        ):
            return self._group_aware.calls(inputs)
        if len(inputs) == 1:
            return ((inputs[0].flat(),),)
        return ElementwiseDispatch(self._shape, self._kernel, inputs).calls()

    def _single(self, value: InGroups) -> InGroups:
        """Run an elementwise function of one input.

        A fallible function runs once for each group where the groups do
        not index every value. Otherwise the function keeps the state and
        the groups of its input.

        Args:
            value: The input.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> abs_ = ApplyRun(Shape.function(FunctionOptions.elementwise()), Kernel())
            >>> abs_._single(frame.column()).state
            <AggState.NOT_AGGREGATED: 'not aggregated'>
            >>> kept = InGroups(AggState.AGGREGATED_LIST, (2, 1), Height.UNKNOWN, 5, original=False)
            >>> elementwise = Shape.function(FunctionOptions.elementwise())
            >>> fallible = ApplyRun(elementwise, Kernel(fallible=True))
            >>> fallible._single(kept).groups
            (2, 1)
        """
        if (  # pragma: no cover -- every fallible kernel takes two inputs
            self._kernel.fallible and not value.covers()
        ):
            return self._group_aware.run([value])
        height = self._shape.height([value.height])
        return replace(value, height=height, dtype=self._kernel.data)


class ElementwiseDispatch:
    """Pick the path of an elementwise function of several inputs in `agg`, and run it.

    Polars reads the states of the inputs:

    - every input a literal: the kernel once, on the literals;
    - an aggregated scalar beside values of several rows: once for each group;
    - a fallible function whose groups miss some values: once for each group;
    - one value in a group beside more values in the same group: once for each group;
    - otherwise the kernel on the flat values.

    A group by gives groups that do not overlap, so the path for groups
    that overlap does not occur here.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> shape = Shape.function(FunctionOptions.elementwise())
        >>> ElementwiseDispatch(shape, Kernel(), [frame.literal(), frame.literal()]).run().state
        <AggState.LITERAL_SCALAR: 'literal scalar'>
    """

    def __init__(self, shape: Shape, kernel: Kernel, inputs: Sequence[InGroups]) -> None:
        """Keep the function and its inputs.

        Args:
            shape: The shape of the function.
            kernel: The facts of its kernel.
            inputs: The inputs.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> shape = Shape.function(FunctionOptions.elementwise())
            >>> len(ElementwiseDispatch(shape, Kernel(), [frame.column(), frame.literal()])._others)
            1
        """
        self._shape = shape
        self._kernel = kernel
        self._inputs = list(inputs)
        self._others = [value for value in inputs if value.state is not AggState.LITERAL_SCALAR]
        self._lengths = shape.kernel_length()
        self._height = shape.height([value.height for value in inputs])

    def run(self) -> InGroups:
        """Run the path that the states of the inputs pick.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> shape = Shape.function(FunctionOptions.elementwise())
            >>> ElementwiseDispatch(shape, Kernel(), [frame.column(), first]).run().groups
            (3, 2)
        """
        if not self._others:
            return self._all_literal()
        if self._group_aware():
            return GroupAware(self._shape, self._kernel).run(self._inputs)
        return self._flat()

    def calls(self) -> tuple[tuple[int, ...], ...]:
        """Give the rows of each input at each call of the kernel, on the path that `run` takes.

        Returns:
            One tuple for each call, with the rows of each input.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> shape = Shape.function(FunctionOptions.elementwise())
            >>> ElementwiseDispatch(shape, Kernel(), [frame.column(), frame.literal()]).calls()
            ((5, 1),)
        """
        if not self._others:
            return ((1,) * len(self._inputs),)
        if self._group_aware():
            return GroupAware(self._shape, self._kernel).calls(self._inputs)
        return (tuple(value.flat() for value in self._aggregated()),)

    def _group_aware(self) -> bool:
        """Say whether the states of the inputs make polars run the function once for each group.

        Returns:
            Whether they do.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> shape = Shape.function(FunctionOptions.elementwise())
            >>> ElementwiseDispatch(shape, Kernel(), [frame.column(), first])._group_aware()
            True
        """
        states = {value.state for value in self._others}
        if AggState.AGGREGATED_SCALAR in states and len(states) > 1:
            return True
        if self._kernel.fallible and not all(value.covers() for value in self._others):
            return True
        return Broadcast().between(self._others)

    def _all_literal(self) -> InGroups:
        """Run the kernel once on the literals.

        Returns:
            The output in the state literal scalar.

        Raises:
            PolarsRefusal: The kernel does not give exactly one value.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> shape = Shape.function(FunctionOptions.elementwise())
            >>> ElementwiseDispatch(shape, Kernel(), [frame.literal()])._all_literal().rows
            1
            >>> joined = Shape.function(FunctionOptions.elementwise(), KernelLength.joined())
            >>> literals = [frame.literal(), frame.literal()]
            >>> ElementwiseDispatch(joined, Kernel(), literals)._all_literal()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: an elementwise expression ...
        """
        if (  # pragma: no cover -- each elementwise length gives 1 on literals
            self._lengths.length([1] * len(self._inputs)) != 1
        ):
            message = "an elementwise expression must give one value on literals"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return replace(self._inputs[-1], height=self._height, dtype=self._kernel.data)

    def _flat(self) -> InGroups:
        """Run the kernel on the flat values of the inputs.

        Values that are not aggregated become lists where they meet a list, or
        where their groups can differ. Lists must have groups of one size. The
        kernel then reads the flat values, and its output must have the length
        of the first input that is not a literal. Polars checks that length
        after the kernel, so a run before the kernel leaves it out
        (`Kernel.before`).

        It raises `PolarsRefusal` where two lists have groups of other sizes,
        or where the kernel gives another length.

        Returns:
            The output, in the state of that first input.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> shape = Shape.function(FunctionOptions.elementwise())
            >>> dispatch = ElementwiseDispatch(shape, Kernel(), [frame.column(), frame.literal()])
            >>> dispatch._flat().state
            <AggState.NOT_AGGREGATED: 'not aggregated'>
        """
        inputs = self._aggregated()
        lists = [value for value in inputs if value.state is AggState.AGGREGATED_LIST]
        GroupLengths().check(lists)
        base = next(value for value in inputs if value.state is not AggState.LITERAL_SCALAR)
        length = self._lengths.length(
            [value.flat() for value in inputs], [value.decided for value in inputs]
        )
        listed = base.state is AggState.AGGREGATED_LIST
        if not self._kernel.before:
            self._check_output(length, base)
        aggregated = any(value.is_aggregated() for value in inputs)
        state = AggState.AGGREGATED_SCALAR if aggregated and not listed else base.state
        return replace(base, state=state, height=self._height, dtype=self._kernel.data)

    def _check_output(self, length: int, base: InGroups) -> None:
        """Check that the kernel gives as many values as the first input that is not a literal.

        Over lists, the kernel must keep each value. Otherwise the function
        checks the length where its options say so.

        Args:
            length: The number of values that the kernel gives.
            base: The first input that is not a literal.

        Raises:
            PolarsRefusal: The kernel gives another number of values.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> shape = Shape.function(FunctionOptions.elementwise())
            >>> dispatch = ElementwiseDispatch(shape, Kernel(), [frame.column()])
            >>> dispatch._check_output(1, frame.column())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> listed = InGroups(AggState.AGGREGATED_LIST, (3, 2), Height.UNKNOWN, 5)
            >>> dispatch._check_output(1, listed)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: the function must apply ...
            >>> from dataclasses import replace
            >>> loose = replace(FunctionOptions.elementwise(), check_lengths=False)
            >>> column, loose_shape = frame.column(), Shape.function(loose)
            >>> ElementwiseDispatch(loose_shape, Kernel(), [column])._check_output(1, column)
        """
        if length == base.flat():
            return
        if base.state is AggState.AGGREGATED_LIST:
            message = "the function must apply element-wise"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if self._shape.options.check_lengths:  # pragma: no branch -- every definition checks
            message = "the output length differs"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def _aggregated(self) -> list[InGroups]:
        """Aggregate the inputs that are not aggregated, where polars must.

        Polars aggregates them where a list is among the inputs, or where
        two inputs that are not aggregated can have other groups.

        Returns:
            The inputs.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> sorted_ = replace(frame.column(), original=False)
            >>> shape = Shape.function(FunctionOptions.elementwise())
            >>> dispatch = ElementwiseDispatch(shape, Kernel(), [frame.column(), sorted_])
            >>> [value.state.value for value in dispatch._aggregated()]
            ['aggregated list', 'aggregated list']
        """
        flat = [value for value in self._others if value.state is AggState.NOT_AGGREGATED]
        lists = any(value.state is AggState.AGGREGATED_LIST for value in self._others)
        if not flat or not (lists or Divergence().of(flat)):
            return self._inputs
        aggregated: list[InGroups] = []
        for value in self._inputs:
            flat_value = value.state is AggState.NOT_AGGREGATED
            aggregated.append(value.aggregated() if flat_value else value)
        return aggregated


class Divergence:
    """Say whether values that are not aggregated can have other groups.

    Polars compares the groups of two such values by identity. Two columns of
    the frame share the groups of the `group_by`. A value whose groups a node
    rebuilt has groups of its own.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> Divergence().of([frame.column(), frame.column()])
        False
    """

    def of(self, values: Sequence[InGroups]) -> bool:
        """Say whether the groups of the values can differ.

        Args:
            values: The values that are not aggregated.

        Returns:
            Whether two values or more meet, and one has groups of its own.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> Divergence().of([frame.column(), replace(frame.column(), original=False)])
            True
        """
        return len(values) > 1 and not all(value.original for value in values)


class Broadcast:
    """Say whether one value of a group meets more values in the same group.

    Polars then runs the node once for each group, where the kernel spreads the
    one value.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> one_row = InGroups(AggState.NOT_AGGREGATED, (1, 1), Height.UNKNOWN, 5)
        >>> Broadcast().between([frame.column(), one_row])
        True
    """

    def between(self, values: Sequence[InGroups]) -> bool:
        """Compare the groups of each value with the groups of the first value.

        Args:
            values: The values that are not literals.

        Returns:
            Whether a group has one value in one input and more in another.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> Broadcast().between([frame.column(), frame.column()])
            False
        """
        base = values[0].groups
        for value in values[1:]:
            for left, right in zip(base, value.groups, strict=True):
                if left != right and 1 in {left, right}:
                    return True
        return False


class AggregationRun:
    """Run an aggregation, such as `sum` or `first`, as `PhysicalExpr.AGGREGATION`.

    In a `select`, an aggregation gives one value. In `agg`, it gives one value
    for each group, with the groups of the `group_by`. Over a literal, it stays
    a literal.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> AggregationRun(Shape.of(NodeKind.AGG), Kernel()).agg([frame.literal()], frame).state
        <AggState.LITERAL_SCALAR: 'literal scalar'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> AggregationRun(Shape.of(NodeKind.AGG), Kernel())._shape.kind
            <NodeKind.AGG: 'Agg'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give one value for each group.

        Args:
            inputs: The input.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> AggregationRun(Shape.of(NodeKind.AGG), Kernel()).agg([frame.column()], frame).groups
            (3, 2)
        """
        value = inputs[0]
        if value.state is AggState.LITERAL_SCALAR:
            return replace(value, height=Height.SCALAR, dtype=self._kernel.data)
        state = AggState.AGGREGATED_SCALAR
        rows = len(frame.sizes)
        return InGroups(state, frame.sizes, Height.SCALAR, rows, dtype=self._kernel.data)


class ImplodeRun:
    """Run `implode()`, which gives the list of the values of each group.

    In `agg`, the list of each group is one value, and each value gets a group
    of its own. Over a literal, `implode` stays a literal of one list.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> ImplodeRun(Shape.of(NodeKind.AGG), Kernel()).agg([frame.column()], frame).groups
        (1, 1)
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> ImplodeRun(Shape.of(NodeKind.AGG), Kernel())._shape.kind
            <NodeKind.AGG: 'Agg'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give one list for each group, as one value.

        Args:
            inputs: The input.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> ImplodeRun(Shape.of(NodeKind.AGG), Kernel()).agg([frame.literal()], frame).state
            <AggState.LITERAL_SCALAR: 'literal scalar'>
        """
        del frame
        value = inputs[0]
        if value.state is AggState.LITERAL_SCALAR:
            return replace(value, height=Height.SCALAR, dtype=self._kernel.data)
        units = (1,) * len(value.groups)
        state = AggState.AGGREGATED_SCALAR
        out = InGroups(state, units, Height.SCALAR, len(units), dtype=self._kernel.data)
        return replace(out, original=False)


class GroupsFunctionRun:
    """Run a function whose kernel over the groups of `agg` keeps an input of one value as it is.

    Polars gives `null_count`, `has_nulls`, `any`, `all`, the bitwise
    aggregations, `skew` and `kurtosis` a kernel of their own for the groups.
    Over an input of one value, a literal or one value for each group, the
    kernel maps that value and keeps its state. Over any other input, it gives
    one value for each group of the input.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> null_count = Shape.function(FunctionOptions.aggregation())
        >>> GroupsFunctionRun(null_count, Kernel()).agg([frame.literal()], frame).state
        <AggState.LITERAL_SCALAR: 'literal scalar'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> any_ = Shape.function(FunctionOptions.aggregation())
            >>> GroupsFunctionRun(any_, Kernel())._shape.kind
            <NodeKind.FUNCTION: 'Function'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give one value for each group, or keep an input of one value.

        Args:
            inputs: The input.
            frame: The groups of the `group_by`, which the output does not read.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> any_ = GroupsFunctionRun(Shape.function(FunctionOptions.aggregation()), Kernel())
            >>> any_.agg([frame.column()], frame).state
            <AggState.AGGREGATED_SCALAR: 'aggregated scalar'>
        """
        del frame
        value = inputs[0]
        if value.state in InGroups.SINGLE:
            return replace(value, height=Height.SCALAR, dtype=self._kernel.data)
        state = AggState.AGGREGATED_SCALAR
        out = InGroups(
            state, value.groups, Height.SCALAR, len(value.groups), dtype=self._kernel.data
        )
        return replace(out, original=value.original)


class RegroupRun:
    """Run a function whose kernel over the groups of `agg` moves the values of each group.

    Polars gives `reverse`, and the fill of nulls forward and backward, a
    kernel of their own for the groups. Over an input of one value, a
    literal or one value for each group, the kernel keeps the input as it is.
    Over any other input, it gives each group a new order of the indices of
    its values, and keeps the values themselves. So it runs no kernel of
    the data, and the values keep their type.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> reverse = RegroupRun(Shape.function(FunctionOptions.length_preserving()), Kernel())
        >>> reverse.agg([frame.column(DType("Date"))], frame).dtype
        Date
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel, which give the type of a value with none.

        Example:
            >>> shape = Shape.function(FunctionOptions.length_preserving())
            >>> RegroupRun(shape, Kernel())._shape.kind
            <NodeKind.FUNCTION: 'Function'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Keep an input of one value, and give the values of any other input with new groups.

        New groups make polars put the values of an aggregated list back in a
        flat column, which the groups index.

        Args:
            inputs: The input.
            frame: The groups of the `group_by`, which the output does not read.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> reverse = RegroupRun(Shape.function(FunctionOptions.length_preserving()), Kernel())
            >>> reverse.agg([frame.literal()], frame).state
            <AggState.LITERAL_SCALAR: 'literal scalar'>
            >>> reverse.agg([frame.column()], frame).original
            False
        """
        del frame
        value = inputs[0]
        dtype = self._kernel.data if value.dtype is None else value.dtype
        if value.state in InGroups.SINGLE:
            return replace(value, dtype=dtype)
        state = AggState.NOT_AGGREGATED
        return replace(value, state=state, original=False, dtype=dtype)

    def calls(self, inputs: Sequence[InGroups]) -> tuple[tuple[int, ...], ...]:
        """Give no call: the run over the groups calls no kernel of the data.

        Args:
            inputs: The input, which the answer does not read.

        Returns:
            No call.

        Example:
            >>> shape = Shape.function(FunctionOptions.length_preserving())
            >>> RegroupRun(shape, Kernel()).calls([GroupBy((3, 2)).column()])
            ()
        """
        del inputs
        return ()


class CountGroupsRun:
    """Run a function whose kernel over the groups of `agg` counts the values of each group first.

    `is_empty` counts the values of each group of its input, and compares each
    count with zero. So it gives one value for each group of the `group_by`,
    also over a literal.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> is_empty = Shape.function(FunctionOptions.aggregation())
        >>> CountGroupsRun(is_empty, Kernel()).agg([frame.literal()], frame).state
        <AggState.AGGREGATED_SCALAR: 'aggregated scalar'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> CountGroupsRun(Shape.function(FunctionOptions.aggregation()), Kernel())._shape.kind
            <NodeKind.FUNCTION: 'Function'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give one value for each group of the `group_by`.

        Args:
            inputs: The input, which the output does not read.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> is_empty = Shape.function(FunctionOptions.aggregation())
            >>> CountGroupsRun(is_empty, Kernel()).agg([frame.column()], frame).groups
            (3, 2)
        """
        del inputs
        state = AggState.AGGREGATED_SCALAR
        rows = len(frame.sizes)
        return InGroups(state, frame.sizes, Height.SCALAR, rows, dtype=self._kernel.data)


class LiteralRun:
    """Run a literal, as `PhysicalExpr.LITERAL`: one value for all the groups of `agg`.

    The conversion of polars puts a literal in place of some nodes before the
    cast pass, for example the bound of `lower_bound`. The node then has the
    state of a literal, and its inputs do not run.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> LiteralRun(Shape.literal(), Kernel()).agg([frame.column()], frame).state
        <AggState.LITERAL_SCALAR: 'literal scalar'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> LiteralRun(Shape.literal(), Kernel())._shape.kind
            <NodeKind.LITERAL: 'Literal'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give one value for all the groups.

        Args:
            inputs: The inputs, which do not run.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> LiteralRun(Shape.literal(), Kernel()).agg([], frame).rows
            1
        """
        del inputs
        return frame.literal(self._kernel.data)


class BinaryRun:
    """Run a binary operation, such as `a + b`, as `PhysicalExpr.BINARY`.

    In a `select`, the two sides must have one length, or one side must have
    length 1. In `agg`, polars picks a path from the states of the two sides.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> BinaryRun(Shape.of(NodeKind.BINARY), Kernel()).agg([frame.literal()] * 2, frame).state
        <AggState.LITERAL_SCALAR: 'literal scalar'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> BinaryRun(Shape.of(NodeKind.BINARY), Kernel())._shape.kind
            <NodeKind.BINARY: 'BinaryExpr'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output over the groups.

        Args:
            inputs: The two sides.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> add = BinaryRun(Shape.of(NodeKind.BINARY), Kernel())
            >>> add.agg([frame.column(), frame.literal()], frame).state
            <AggState.NOT_AGGREGATED: 'not aggregated'>
        """
        del frame
        return BinaryDispatch(self._shape, self._kernel, inputs).run()

    def calls(self, inputs: Sequence[InGroups]) -> tuple[tuple[int, ...], ...]:
        """Give the rows of each side at each call of the operator, on the path of `agg`.

        Args:
            inputs: The two sides.

        Returns:
            One tuple for each call, with the rows of each side.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> BinaryRun(Shape.of(NodeKind.BINARY), Kernel()).calls([first, frame.column()])
            ((1, 3), (1, 2))
        """
        return BinaryDispatch(self._shape, self._kernel, inputs).calls()


class BinaryDispatch:
    """Pick the path of a binary operation in `agg`, and run it.

    Polars reads the states of the two sides:

    - both literals: the operation once;
    - an aggregated scalar beside values of several rows: once for each group;
    - a fallible operation, such as arithmetic on a `Decimal`, whose groups
      miss some values: once for each group;
    - one value in a group beside more values in the same group: once for each group;
    - otherwise the operation on the flat values.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> dispatch = BinaryDispatch(Shape.of(NodeKind.BINARY), Kernel(), [frame.column()] * 2)
        >>> dispatch.run().state
        <AggState.NOT_AGGREGATED: 'not aggregated'>
    """

    def __init__(self, shape: Shape, kernel: Kernel, inputs: Sequence[InGroups]) -> None:
        """Keep the operation and its two sides.

        Args:
            shape: The shape of the operation.
            kernel: The facts of its kernel.
            inputs: The two sides.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> BinaryDispatch(Shape.of(NodeKind.BINARY), Kernel(), [frame.column()] * 2)._height
            <Height.COLUMN: 'column'>
        """
        self._shape = shape
        self._kernel = kernel
        self._inputs = list(inputs)
        self._height = shape.height([value.height for value in inputs])
        self._lengths = KernelLength.broadcast()

    def run(self) -> InGroups:
        """Run the path that the states of the two sides pick.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> sides = [first, frame.column()]
            >>> dispatch = BinaryDispatch(Shape.of(NodeKind.BINARY), Kernel(), sides)
            >>> dispatch.run().groups
            (3, 2)
        """
        path = self.path()
        if path is KernelPath.LITERALS:
            return replace(self._inputs[0], height=self._height, dtype=self._kernel.data)
        if path is KernelPath.GROUPS:
            return self._group_aware()
        return self._elementwise()

    def path(self) -> KernelPath:
        """Give the path that the states of the two sides pick.

        Returns:
            The path.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> sides = [frame.literal(), frame.column()]
            >>> BinaryDispatch(Shape.of(NodeKind.BINARY), Kernel(), sides).path()
            <KernelPath.FLAT: 'flat'>
        """
        states = {value.state for value in self._inputs}
        if states == {AggState.LITERAL_SCALAR}:
            return KernelPath.LITERALS
        others = [value for value in self._inputs if value.state is not AggState.LITERAL_SCALAR]
        several = states & {AggState.AGGREGATED_LIST, AggState.NOT_AGGREGATED}
        if AggState.AGGREGATED_SCALAR in states and several:
            return KernelPath.GROUPS
        if self._kernel.fallible and not all(value.covers() for value in others):
            return KernelPath.GROUPS
        lists = {AggState.AGGREGATED_LIST, AggState.NOT_AGGREGATED}
        if states <= lists and Broadcast().between(self._inputs):
            return KernelPath.GROUPS
        return KernelPath.FLAT

    def calls(self) -> tuple[tuple[int, ...], ...]:
        """Give the rows of each side at each call of the operator, on the path that `run` takes.

        On the flat path, the operator reads the flat values of the two sides,
        after the sides that polars must aggregate become lists.

        Returns:
            One tuple for each call, with the rows of each side.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> sides = [frame.literal(), frame.literal()]
            >>> BinaryDispatch(Shape.of(NodeKind.BINARY), Kernel(), sides).calls()
            ((1, 1),)
        """
        path = self.path()
        if path is KernelPath.LITERALS:
            return ((1, 1),)
        if path is KernelPath.GROUPS:
            return GroupCalls(self._shape, self._kernel).calls(self._inputs)
        return (tuple(side.flat() for side in self._aggregated()),)

    def _group_aware(self) -> InGroups:
        """Run the operation once for each group.

        In each group, the operation of two columns spreads a side of one value
        to the length of the other side. It checks the lengths itself, and it
        raises an `InvalidOperationError` where the two sides of a group have
        other lengths and neither has one value (`_group_length`). A kernel
        that checks the lengths itself after its checks of the types
        (`Kernel.own_lengths`) meets the check only in the run after it. A
        `select` checks the lengths before the operation, with a `ShapeError`.

        Returns:
            The output in the state aggregated list.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> sides = [first, frame.column()]
            >>> dispatch = BinaryDispatch(Shape.of(NodeKind.BINARY), Kernel(), sides)
            >>> dispatch._group_aware().rows
            5
            >>> state, unknown = AggState.NOT_AGGREGATED, Height.UNKNOWN
            >>> sliced = InGroups(state, (2, 2), unknown, 5, original=False)
            >>> sides = [frame.column(), sliced]
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.44.0"):
            ...     BinaryDispatch(Shape.of(NodeKind.BINARY), Kernel(), sides)._group_aware()
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: can't compute broadcast length, ...
        """
        left, right = self._inputs
        pairs = zip(left.values(), right.values(), strict=True)
        free = left.decided or right.decided
        if (self._kernel.before and self._kernel.own_lengths) or free:
            sizes = tuple(itertools.starmap(max, pairs))
        else:
            sizes = tuple(itertools.starmap(self._group_length, pairs))
        out = InGroups(AggState.AGGREGATED_LIST, sizes, self._height, sum(sizes), decided=free)
        return replace(out, original=left.original, dtype=self._kernel.data)

    def _group_length(self, left: int, right: int) -> int:
        """Give the length of the operation of two columns in one group.

        It raises `PolarsRefusal` where the two sides have other lengths, and
        neither has one value (`UnequalLengths`).

        Args:
            left: The values of the left side in the group.
            right: The values of the right side in the group.

        Returns:
            The length of the longer side, where a side has one value or the two agree.

        Example:
            >>> sides = [GroupBy((3, 2)).column()] * 2
            >>> BinaryDispatch(Shape.of(NodeKind.BINARY), Kernel(), sides)._group_length(1, 0)
            0
        """
        if left == 1:
            return right
        if right in {1, left}:
            return left
        message = (
            f"cannot do a binary operation on columns of different lengths: got {left} and {right}"
        )
        raise UnequalLengths().refusal([left, right], message)

    def _elementwise(self) -> InGroups:
        """Run the operation on the flat values of the two sides.

        Values that are not aggregated become lists where they meet a list,
        or where the two sides can have other groups. Two lists must have
        groups of one size. The operator over two lists checks the types of
        their values first, so the run before the kernel leaves that check to
        the run after it. The output keeps the state and the groups of the
        side that is not a literal.

        It raises `PolarsRefusal` where two lists have groups of other sizes, or the flat values do
        not fit.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> sides = [frame.literal(), frame.column()]
            >>> add = BinaryDispatch(Shape.of(NodeKind.BINARY), Kernel(), sides)
            >>> add._elementwise().state
            <AggState.NOT_AGGREGATED: 'not aggregated'>
        """
        sides = self._aggregated()
        lists = [side for side in sides if side.state is AggState.AGGREGATED_LIST]
        if len(lists) == 2 and not self._kernel.before:
            GroupLengths().check(lists)
        self._lengths.length([side.flat() for side in sides], [side.decided for side in sides])
        base = next(side for side in sides if side.state is not AggState.LITERAL_SCALAR)
        return replace(base, height=self._height, dtype=self._kernel.data)

    def _aggregated(self) -> list[InGroups]:
        """Aggregate the sides that are not aggregated, where polars must.

        Returns:
            The two sides.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> sorted_ = replace(frame.column(), original=False)
            >>> sides = [frame.column(), sorted_]
            >>> dispatch = BinaryDispatch(Shape.of(NodeKind.BINARY), Kernel(), sides)
            >>> [side.state.value for side in dispatch._aggregated()]
            ['aggregated list', 'aggregated list']
        """
        flat = [side for side in self._inputs if side.state is AggState.NOT_AGGREGATED]
        lists = any(side.state is AggState.AGGREGATED_LIST for side in self._inputs)
        if not flat or not (lists or Divergence().of(flat)):
            return self._inputs
        sides: list[InGroups] = []
        for side in self._inputs:
            flat_side = side.state is AggState.NOT_AGGREGATED
            sides.append(side.aggregated() if flat_side else side)
        return sides


class TernaryRun:
    """Run `when(predicate).then(truthy).otherwise(falsy)` as `PhysicalExpr.TERNARY`.

    The inputs come in the order of polars: the predicate, then the value where
    it holds, then the value where it does not. In a `select`, the three must
    have one length, and each of length 1 spreads. In `agg`, polars picks a
    path from the states of the three. Unlike a binary operation, a ternary
    spreads no group of one value beside a group of more values: it compares
    the groups of two lists, and raises where they differ.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> when = TernaryRun(Shape.of(NodeKind.TERNARY), Kernel())
        >>> when.agg([frame.column(), frame.literal(), frame.literal()], frame).state
        <AggState.NOT_AGGREGATED: 'not aggregated'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> TernaryRun(Shape.of(NodeKind.TERNARY), Kernel())._shape.kind
            <NodeKind.TERNARY: 'Ternary'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output over the groups.

        Values that are not aggregated and keep the groups of the `group_by`
        meet on their flat values. Otherwise each value becomes aggregated.
        Lists and values of one value for each group then meet once for each
        group. Lists alone must have groups of one size.

        It raises `PolarsRefusal` where the values do not fit.

        Args:
            inputs: The predicate, the value where it holds, and the value where it does not.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> state = AggState.NOT_AGGREGATED
            >>> one_row = InGroups(state, (1, 1), Height.UNKNOWN, 5, original=False)
            >>> when = TernaryRun(Shape.of(NodeKind.TERNARY), Kernel())
            >>> when.agg([one_row, frame.column(), frame.column()], frame)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
        """
        del frame
        return TernaryDispatch(self._shape, self._kernel, inputs).run()

    def calls(self, inputs: Sequence[InGroups]) -> tuple[tuple[int, ...], ...]:
        """Give the rows of each input at each call of the kernel, on the path of `agg`.

        Args:
            inputs: The predicate, the value where it holds, and the value where it does not.

        Returns:
            One tuple for each call, with the rows of each input.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> when = TernaryRun(Shape.of(NodeKind.TERNARY), Kernel())
            >>> when.calls([frame.column(), frame.column(), frame.literal()])
            ((5, 5, 1),)
        """
        return TernaryDispatch(self._shape, self._kernel, inputs).calls()


class TernaryDispatch:
    """Pick the path of a ternary in `agg` from the states of its three inputs, and run it.

    The inputs are the predicate, the value where it holds, and the value where
    it does not.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> inputs = [frame.column(), frame.column(), frame.literal()]
        >>> TernaryDispatch(Shape.of(NodeKind.TERNARY), Kernel(), inputs).run().groups
        (3, 2)
    """

    def __init__(self, shape: Shape, kernel: Kernel, inputs: Sequence[InGroups]) -> None:
        """Keep the ternary and its three inputs.

        Args:
            shape: The shape of the ternary.
            kernel: The facts of its kernel.
            inputs: The predicate, the value where it holds, and the value where it does not.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> inputs = [frame.literal()] * 3
            >>> TernaryDispatch(Shape.of(NodeKind.TERNARY), Kernel(), inputs)._height
            <Height.SCALAR: 'scalar'>
        """
        self._shape = shape
        self._kernel = kernel
        self._inputs = list(inputs)
        self._height = shape.height([value.height for value in inputs])
        self._lengths = KernelLength.broadcast()

    def run(self) -> InGroups:
        """Run the path that the states of the inputs pick.

        Values that are not aggregated and keep the groups of the `group_by`
        meet on their flat values. Otherwise each such value becomes
        aggregated. Lists beside values of one value for each group then meet
        once for each group. Lists alone must have groups of one size.

        It raises `PolarsRefusal` where the values do not fit.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> inputs = [first, frame.column(), frame.literal()]
            >>> TernaryDispatch(Shape.of(NodeKind.TERNARY), Kernel(), inputs).run().state
            <AggState.AGGREGATED_LIST: 'aggregated list'>
        """
        if self._as_is():
            flat = [value for value in self._inputs if value.state is AggState.NOT_AGGREGATED]
            return self._flat(flat)
        values = self._aggregated()
        if self._mixed(values):
            return self._per_group(values)
        others = [value for value in values if value.state is not AggState.LITERAL_SCALAR]
        if others[0].state is AggState.AGGREGATED_LIST:
            GroupLengths().check(others)
        self._lengths.length(
            [value.flat() for value in values], [value.decided for value in values]
        )
        return replace(others[0], height=self._height, dtype=self._kernel.data)

    def calls(self) -> tuple[tuple[int, ...], ...]:
        """Give the rows of each input at each call of the kernel, on the path that `run` takes.

        Returns:
            One tuple for each call, with the rows of each input.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> inputs = [first, frame.column(), frame.literal()]
            >>> TernaryDispatch(Shape.of(NodeKind.TERNARY), Kernel(), inputs).calls()
            ((1, 3, 1), (1, 2, 1))
        """
        if self._as_is():
            return (tuple(value.flat() for value in self._inputs),)
        values = self._aggregated()
        if self._mixed(values):
            return GroupCalls(self._shape, self._kernel).calls(values)
        return (tuple(value.flat() for value in values),)

    def _as_is(self) -> bool:
        """Say whether the three inputs meet on their flat values as they are.

        They do where no input is aggregated, and no node rebuilt the groups of
        an input that is not aggregated.

        Returns:
            Whether they do.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> inputs = [frame.column(), frame.column(), frame.literal()]
            >>> TernaryDispatch(Shape.of(NodeKind.TERNARY), Kernel(), inputs)._as_is()
            True
        """
        flat = [value for value in self._inputs if value.state is AggState.NOT_AGGREGATED]
        rebuilt = not all(value.original for value in flat)
        return not any(value.is_aggregated() for value in self._inputs) and not rebuilt

    def _mixed(self, values: Sequence[InGroups]) -> bool:
        """Say whether lists meet values of one value for each group, so that they meet per group.

        Args:
            values: The three inputs, aggregated.

        Returns:
            Whether the inputs that are not literals have two states.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> inputs = [first, frame.column().aggregated(), frame.literal()]
            >>> TernaryDispatch(Shape.of(NodeKind.TERNARY), Kernel(), inputs)._mixed(inputs)
            True
        """
        others = [value for value in values if value.state is not AggState.LITERAL_SCALAR]
        return len({value.state for value in others}) > 1

    def _aggregated(self) -> list[InGroups]:
        """Aggregate each input that is not aggregated.

        Returns:
            The three inputs.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> inputs = [frame.column(), frame.literal(), frame.literal()]
            >>> dispatch = TernaryDispatch(Shape.of(NodeKind.TERNARY), Kernel(), inputs)
            >>> [value.state.value for value in dispatch._aggregated()][:2]
            ['aggregated list', 'literal scalar']
        """
        values: list[InGroups] = []
        for value in self._inputs:
            flat = value.state is AggState.NOT_AGGREGATED
            values.append(value.aggregated() if flat else value)
        return values

    def _flat(self, flat: Sequence[InGroups]) -> InGroups:
        """Meet the three inputs on their flat values.

        Args:
            flat: The inputs that are not aggregated.

        Returns:
            The output, with the state and the groups of the first input that
            is not aggregated, or a literal where every input is a literal.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> inputs = [frame.literal()] * 3
            >>> TernaryDispatch(Shape.of(NodeKind.TERNARY), Kernel(), inputs)._flat([]).state
            <AggState.LITERAL_SCALAR: 'literal scalar'>
        """
        self._lengths.length(
            [value.flat() for value in self._inputs], [value.decided for value in self._inputs]
        )
        base = flat[0] if flat else self._inputs[1]
        return replace(base, height=self._height, dtype=self._kernel.data)

    def _per_group(self, values: Sequence[InGroups]) -> InGroups:
        """Meet the three inputs once for each group.

        Args:
            values: The three inputs, aggregated.

        Returns:
            The output in the state aggregated list.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> inputs = [frame.literal().aggregated()] * 3
            >>> dispatch = TernaryDispatch(Shape.of(NodeKind.TERNARY), Kernel(), inputs)
            >>> dispatch._per_group(inputs).groups
            (1, 1)
        """
        groups = zip(*(value.values() for value in values), strict=True)
        decided = [value.decided for value in values]
        sizes = tuple(self._lengths.length(group, decided) for group in groups)
        free = any(decided)
        out = InGroups(AggState.AGGREGATED_LIST, sizes, self._height, sum(sizes), decided=free)
        return replace(out, original=values[1].original, dtype=self._kernel.data)


class CastRun:
    """Run a cast as `PhysicalExpr.CAST`.

    A cast keeps the state of its input. A strict cast of values that are not
    aggregated, whose groups a node rebuilt, first aggregates them, so that a
    value outside the groups cannot raise.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> CastRun(Shape.of(NodeKind.CAST), Kernel()).agg([frame.literal()], frame).state
        <AggState.LITERAL_SCALAR: 'literal scalar'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> CastRun(Shape.of(NodeKind.CAST), Kernel())._shape.kind
            <NodeKind.CAST: 'Cast'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output over the groups.

        Args:
            inputs: The input.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> state = AggState.NOT_AGGREGATED
            >>> sliced = InGroups(state, (2, 2), Height.UNKNOWN, 5, original=False)
            >>> CastRun(Shape.of(NodeKind.CAST), Kernel(strict=True)).agg([sliced], frame).state
            <AggState.AGGREGATED_LIST: 'aggregated list'>
        """
        del frame
        value = inputs[0]
        flat = value.state is AggState.NOT_AGGREGATED
        if flat and self._kernel.strict and not value.original:
            value = value.aggregated()
        return replace(value, dtype=self._kernel.data)


class SortRun:
    """Run `sort()` as `PhysicalExpr.SORT`.

    A sort of lists sorts each list. A sort of any other value builds new
    groups, and keeps the sizes of the groups. The new groups are not those of
    the `group_by`. For an aggregated scalar and a literal, polars builds the
    new groups from groups that do not index their values. The rule keeps the
    old sizes, since these values hold one value for each group.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> sort = SortRun(Shape.of(NodeKind.SORT), Kernel())
        >>> sort.agg([InGroups(AggState.AGGREGATED_LIST, (3, 2), Height.UNKNOWN, 5)], frame).state
        <AggState.AGGREGATED_LIST: 'aggregated list'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> SortRun(Shape.of(NodeKind.SORT), Kernel())._shape.kind
            <NodeKind.SORT: 'Sort'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output over the groups.

        Args:
            inputs: The input.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> SortRun(Shape.of(NodeKind.SORT), Kernel()).agg([frame.literal()], frame).state
            <AggState.LITERAL_SCALAR: 'literal scalar'>
        """
        del frame
        value = inputs[0]
        if value.state is AggState.AGGREGATED_LIST:
            return replace(value, dtype=self._kernel.data)
        return replace(value, original=False, dtype=self._kernel.data)


class SliceSelect:
    """Give the output of `slice`, `head` or `tail` in a `select`.

    The inputs come in the order of polars: the value, the offset and the
    length. The offset and the length must each have at most one value. The
    rule of the kernel of the shape holds the offset and the length
    (`KernelLength.slice`).

    Example:
        >>> head = SliceSelect(Shape(NodeKind.SLICE, length=KernelLength.slice(0, 2)))
        >>> head.select([InSelect.column(5), InSelect.literal(), InSelect.literal()]).length
        2
    """

    def __init__(self, shape: Shape) -> None:
        """Keep the shape of the slice.

        Args:
            shape: The shape.

        Example:
            >>> SliceSelect(Shape.of(NodeKind.SLICE))._shape.kind
            <NodeKind.SLICE: 'Slice'>
        """
        self._shape = shape

    def select(self, inputs: Sequence[InSelect]) -> InSelect:
        """Give the output.

        Args:
            inputs: The value, the offset and the length.

        Returns:
            The output.

        Raises:
            PolarsRefusal: The offset or the length has more than one value.

        Example:
            >>> head = SliceSelect(Shape(NodeKind.SLICE, length=KernelLength.slice(0, 2)))
            >>> head.select([InSelect.column(5), InSelect.literal(), InSelect.column(5)])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        value, *arguments = inputs
        if any(argument.length > 1 for argument in arguments):
            message = "an offset of several values"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        rule = self._shape.kernel_length()
        length = rule.length([value.length])
        height = self._shape.height([entry.height for entry in inputs])
        bounded = rule.bounded([value.length], [value.bounded], length)
        return InSelect(length, height, value.decided, bounded)


class SliceRun:
    """Run `slice`, `head` or `tail` over the groups of `agg`.

    The inputs come in the order of polars: the value, the offset and the
    length. The rule of the kernel of the shape holds the offset and the length
    (`KernelLength.slice`). An offset or a length that is not a literal must
    have one value for each group, and the rule gives each group the same
    offset and length. The slice cuts the groups of its value, and its values
    are not aggregated. A slice of a literal by literals gives one list for
    each group.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> head = SliceRun(Shape(NodeKind.SLICE, length=KernelLength.slice(0, 2)), Kernel())
        >>> head.agg([frame.column(), frame.literal(), frame.literal()], frame).groups
        (2, 2)
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the slice.

        Args:
            shape: The shape of the slice.
            kernel: The facts of its kernel.

        Example:
            >>> SliceRun(Shape.of(NodeKind.SLICE), Kernel())._shape.kind
            <NodeKind.SLICE: 'Slice'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output over the groups.

        An aggregated scalar and a literal first become one list of one value
        for each group. It raises `PolarsRefusal` where an offset or a length
        is not one value for each group (`_check_arguments`).

        Args:
            inputs: The value, the offset and the length.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> head = SliceRun(Shape(NodeKind.SLICE, length=KernelLength.slice(0, 2)), Kernel())
            >>> head.agg([frame.literal(), frame.literal(), frame.literal()], frame).state
            <AggState.AGGREGATED_LIST: 'aggregated list'>
        """
        del frame
        value, *arguments = inputs
        rule = self._shape.kernel_length()
        self._check_arguments(arguments)
        units = (1,) * len(value.groups)
        literals = all(argument.state is AggState.LITERAL_SCALAR for argument in arguments)
        bounded = rule.bounded([1], [value.bounded or value.state in InGroups.SINGLE], 1)
        if literals and value.state is AggState.LITERAL_SCALAR:
            kept = tuple(rule.length([size]) for size in units)
            out = InGroups(AggState.AGGREGATED_LIST, kept, Height.UNKNOWN, sum(kept))
            return replace(out, dtype=self._kernel.data, bounded=bounded)
        if value.state in InGroups.SINGLE:
            value = InGroups(AggState.AGGREGATED_LIST, units, value.height, len(units))
        sizes = tuple(rule.length([size]) for size in value.groups)
        state = AggState.NOT_AGGREGATED
        out = InGroups(state, sizes, Height.UNKNOWN, value.rows, original=False)
        overlapping = value.overlapping
        return replace(out, dtype=self._kernel.data, bounded=bounded, overlapping=overlapping)

    def _check_arguments(self, arguments: Sequence[InGroups]) -> None:
        """Refuse an offset or a length that is not one value for each group.

        Args:
            arguments: The offset and the length.

        Raises:
            PolarsRefusal: An argument holds more than one value in a group.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> run = SliceRun(Shape(NodeKind.SLICE, length=KernelLength.slice(0, 1)), Kernel())
            >>> run._check_arguments([frame.literal(), frame.literal()])
        """
        for argument in arguments:
            if argument.state not in InGroups.SINGLE:
                message = "invalid slice argument: cannot use an array as argument"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)


class MinMaxBySelect:
    """Give the output of `min_by` or `max_by` in a `select`: one value.

    The value and the key must have one length.

    Example:
        >>> min_by = MinMaxBySelect(Shape.function(FunctionOptions.aggregation()))
        >>> min_by.select([InSelect.column(5)] * 2)
        InSelect(length=1, height=<Height.SCALAR: 'scalar'>, decided=False, bounded=True)
    """

    def __init__(self, shape: Shape) -> None:
        """Keep the shape of the node.

        Args:
            shape: The shape.

        Example:
            >>> MinMaxBySelect(Shape.function(FunctionOptions.aggregation()))._shape.kind
            <NodeKind.FUNCTION: 'Function'>
        """
        self._shape = shape

    def select(self, inputs: Sequence[InSelect]) -> InSelect:
        """Give the output.

        Args:
            inputs: The value and the key.

        Returns:
            The output.

        Raises:
            PolarsRefusal: The key does not have the length of the value.

        Example:
            >>> min_by = MinMaxBySelect(Shape.function(FunctionOptions.aggregation()))
            >>> min_by.select([InSelect.column(5), InSelect.literal()])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
        """
        value, key = inputs
        if value.length != key.length:
            message = "the key has another length"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        return InSelect(1, self._shape.height([value.height, key.height]), bounded=True)


class RowRuns:
    """Read the rows of each group that a node of rows sees, as polars sets them before the run.

    An aggregated scalar and a literal have no groups of their own. Polars
    gives each of their groups one row before a `filter` or a `sort_by` reads
    the groups.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> RowRuns().sizes(frame.literal()), RowRuns().sizes(frame.column())
        ((1, 1), (3, 2))
    """

    def sizes(self, value: InGroups) -> tuple[int, ...]:
        """Give the rows of each group of a value.

        Args:
            value: The value.

        Returns:
            The rows of each group.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> RowRuns().sizes(first)
            (1, 1)
        """
        if value.state in {AggState.AGGREGATED_SCALAR, AggState.LITERAL_SCALAR}:
            return (1,) * len(value.groups)
        return value.groups


class FilterRun:
    """Run `filter` over the groups of `agg`.

    The value and the mask must have the same rows in each group, where a
    group of one row spreads to the rows of the other side. The mask then
    keeps rows of each group, so the output is a list of the type of the
    value in each group, whose rows the values decide.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> run = FilterRun(Shape.of(NodeKind.FILTER), Kernel(data=DType("Int8")))
        >>> run.agg([frame.column(), frame.literal()], frame).state
        <AggState.AGGREGATED_LIST: 'aggregated list'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> FilterRun(Shape.of(NodeKind.FILTER), Kernel())._shape.kind
            <NodeKind.FILTER: 'Filter'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output over the groups.

        Args:
            inputs: The value and the mask.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Raises:
            PolarsRefusal: A group of the value and of the mask have other rows,
                and neither has one row.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> run = FilterRun(Shape.of(NodeKind.FILTER), Kernel())
            >>> run.agg([frame.column(), frame.column().with_unit_groups()], frame).rows
            5
        """
        value, mask = inputs
        sizes = list(zip(RowRuns().sizes(value), RowRuns().sizes(mask), strict=True))
        free = value.decided or mask.decided
        if not free and any(left != right and 1 not in {left, right} for left, right in sizes):
            raise PolarsRefusal(ErrorClass.SHAPE, "filter: the lengths of the groups differ")
        groups = tuple(itertools.starmap(max, sizes))
        rows = sum(groups)
        return InGroups(
            AggState.AGGREGATED_LIST,
            groups,
            Height.UNKNOWN,
            rows,
            original=False,
            dtype=self._kernel.data,
            decided=True,
        )


class SortByRun:
    """Run `sort_by` over the groups of `agg`.

    Where every key is a literal, the run gives the value as it is. A literal
    value beside a key that is not a literal sorts over the rows of the keys:
    a key of more than one row in a group does not match the one row of the
    literal, and a key of one row gives its own state to the output.
    Otherwise the value and each key must have the same rows in each group,
    and the output keeps the rows of the value, in another order.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> run = SortByRun(Shape.of(NodeKind.SORT_BY), Kernel(data=DType("Int8")))
        >>> run.agg([frame.column(), frame.column()], frame).original
        False
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> SortByRun(Shape.of(NodeKind.SORT_BY), Kernel())._shape.kind
            <NodeKind.SORT_BY: 'SortBy'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output over the groups.

        Args:
            inputs: The value, then each key.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Raises:
            PolarsRefusal: A key has other rows than the value in a group.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> run = SortByRun(Shape.of(NodeKind.SORT_BY), Kernel())
            >>> run.agg([frame.column(), frame.literal()], frame).state
            <AggState.NOT_AGGREGATED: 'not aggregated'>
        """
        del frame
        value, keys = inputs[0], inputs[1:]
        literal = AggState.LITERAL_SCALAR
        if all(key.state is literal for key in keys):
            return replace(value, dtype=self._kernel.data or value.dtype)
        if value.state is literal:
            return self._literal_value(keys)
        if value.rebuilt and len(keys) == 1:
            return self._rebuilt(value, keys[0])
        for key in keys:
            if not self._matched(value, key.aggregated() if key.state is literal else key):
                message = "expressions in 'sort_by' must have matching group lengths"
                raise PolarsRefusal(ErrorClass.SHAPE, message)
        return replace(value, original=False, dtype=self._kernel.data)

    def _matched(self, value: InGroups, key: InGroups) -> bool:
        """Say whether a key passes the check of the group lengths beside the value.

        A key or a value whose group sizes the values decide passes: the data
        can give the sizes of the other.

        Args:
            value: The value.
            key: The key, as a list for each group.

        Returns:
            Whether it passes.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> run = SortByRun(Shape.of(NodeKind.SORT_BY), Kernel())
            >>> run._matched(frame.column(), frame.literal().aggregated())
            False
            >>> run._matched(frame.column(), replace(frame.literal().aggregated(), decided=True))
            True
        """
        if value.decided or key.decided:
            return True
        return RowRuns().sizes(key) == RowRuns().sizes(value)

    def _rebuilt(self, value: InGroups, key: InGroups) -> InGroups:
        """Give the output over a value whose groups a node rebuilt, beside one key.

        Polars then sorts the list of each group by the list of the key in
        that group. A key of one value for each group gives no list there,
        and polars panics. A key of other rows in a group raises a
        `ComputeError`.

        Args:
            value: The value, whose lists a node built group by group.
            key: The key.

        Returns:
            The output: the value, in another order.

        Raises:
            PolarsRefusal: The key holds one value for each group, or other
                rows than the value in a group where the values decide neither.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> run = SortByRun(Shape.of(NodeKind.SORT_BY), Kernel())
            >>> rebuilt = replace(frame.column().aggregated(), rebuilt=True)
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> run._rebuilt(rebuilt, first)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        if key.state is AggState.AGGREGATED_SCALAR:
            message = "sort_by reads the key of one value for each group as a list"
            raise PolarsRefusal(Stop.PANICS, message)
        listed = key.aggregated() if key.state is AggState.LITERAL_SCALAR else key
        if not self._matched(value, listed):
            message = "series lengths don't match in 'sort_by' expression"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return replace(value, dtype=self._kernel.data)

    def _literal_value(self, keys: Sequence[InGroups]) -> InGroups:
        """Give the output of a literal value beside keys that are not all literals.

        Args:
            keys: The keys.

        Returns:
            The state of the first key that is not a literal, with the type of the value.

        Raises:
            PolarsRefusal: A key has more than one row in a group.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> run = SortByRun(Shape.of(NodeKind.SORT_BY), Kernel(data=DType("Int32")))
            >>> run._literal_value([first]).state
            <AggState.AGGREGATED_SCALAR: 'aggregated scalar'>
        """
        columns = [key for key in keys if key.state is not AggState.LITERAL_SCALAR]
        if any(size != 1 for key in columns for size in RowRuns().sizes(key)):
            message = "`sort_by` produced another length than the Series to sort"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        return replace(columns[0], original=False, dtype=self._kernel.data)


class GatherRun:
    """Run a gather over the groups of `agg`: `get`, which gives one value, or `gather`.

    `get` needs indices of one value for each group, and gives one value for
    each group. `gather` gives a list for each group, with one row for each
    index of the group.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> run = GatherRun(Shape.gather(returns_scalar=True), Kernel(data=DType("Int8")))
        >>> run.agg([frame.column(), frame.literal()], frame).state
        <AggState.AGGREGATED_SCALAR: 'aggregated scalar'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> GatherRun(Shape.gather(returns_scalar=False), Kernel())._shape.kind
            <NodeKind.GATHER: 'Gather'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give the output over the groups.

        Args:
            inputs: The value and the indices.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Raises:
            PolarsRefusal: `get` meets indices of more than one value for each group.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> run = GatherRun(Shape.gather(returns_scalar=True), Kernel())
            >>> run.agg([frame.column(), frame.column()], frame)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        indices = inputs[1]
        if self._shape.returns_scalar:
            if indices.state in {AggState.AGGREGATED_LIST, AggState.NOT_AGGREGATED}:
                message = "expected an aggregated scalar for the indices of `get`"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)
            groups = len(frame.sizes)
            return InGroups(
                AggState.AGGREGATED_SCALAR,
                frame.sizes,
                Height.SCALAR,
                groups,
                dtype=self._kernel.data,
            )
        listed = indices.aggregated()
        return InGroups(
            AggState.AGGREGATED_LIST,
            RowRuns().sizes(listed),
            Height.UNKNOWN,
            listed.rows,
            original=False,
            dtype=self._kernel.data,
        )


class OverSelect:
    """Run an `over` in a `select`: its function over the groups of its keys, back on the rows.

    Each key must have the rows of the frame, or one row, which spreads. The
    function runs over the groups of the keys. With the mapping
    `GroupsToRows`, a value that does not spread must keep the rows of each
    group. The values of a key of more than one row decide those rows:
    `head(2)` fits groups of two rows or fewer. So the rule checks the rows
    only where every key has one row, which makes one group of the whole
    frame. The output has the rows of the frame. With the mapping `join`,
    each row gets the values of its group in a list, so the rows of the
    function take no part. With the mapping `explode`, the output has the
    rows of the groups, one group after the other.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
        >>> OverSelect(explode=False).select(first, [InSelect.column(5)], frame).length
        5
        >>> OverSelect(explode=True).select(first, [InSelect.column(5)], frame).height
        <Height.UNKNOWN: 'unknown'>
    """

    def __init__(self, *, explode: bool, joined: bool = False) -> None:
        """Keep the mapping, where it is `explode` or `join` and not `GroupsToRows`.

        Args:
            explode: Whether the mapping is `explode`, which gives the rows of each group in turn.
            joined: Whether the mapping is `join`, which gives each row a list of its group.

        Example:
            >>> OverSelect(explode=True)._explode
            True
        """
        self._explode = explode
        self._joined = joined

    def select(self, function: InGroups, keys: Sequence[InSelect], frame: GroupBy) -> InSelect:
        """Give the output of the `over`.

        Args:
            function: The value of the function over the groups of the keys.
            keys: The value of each key in the `select`.
            frame: The groups of the keys.

        Returns:
            The output.

        Raises:
            PolarsRefusal: A key does not fit the rows, or the function does
                not keep the rows of the one group of the frame.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> OverSelect(explode=False).select(frame.column(), [InSelect.column(2)], frame)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
            >>> joined = OverSelect(explode=False, joined=True)
            >>> joined.select(frame.column(), [InSelect.literal()], GroupBy((5,))).length
            5
        """
        rows = sum(frame.sizes)
        for key in keys:
            if key.length not in {rows, 1}:
                message = (
                    "columns used as `partition_by` must have the same length as the DataFrame"
                )
                raise PolarsRefusal(ErrorClass.SHAPE, message)
        if self._explode:
            return InSelect(function.rows, Height.UNKNOWN)
        spread = {AggState.AGGREGATED_SCALAR, AggState.LITERAL_SCALAR}
        checked = not self._joined and function.state not in spread
        one_group = all(key.length == 1 for key in keys)
        if one_group and checked and function.groups != frame.sizes:
            message = "the length of the window expression did not match that of the group"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        return InSelect.column(rows)


class RollingRun:
    """Run a `rolling`: its function over a window for each row of the frame, or of each group.

    The index places the windows. The function runs over each window, which
    can hold any number of rows, so its own length takes no part. In a
    `select`, a plain column has the rows of the frame, and an index of one
    row spreads to them. In `agg`, each group of the index must keep the rows
    of its group, so the output has the rows of each group.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> RollingRun().select(InSelect.column(5), frame).length
        5
        >>> RollingRun().agg(frame.column(), frame).groups
        (3, 2)
    """

    def select(self, index: InSelect, frame: GroupBy) -> InSelect:
        """Give the output of the `rolling` in a `select`.

        Args:
            index: The value of the index in the `select`.
            frame: The rows of the frame.

        Returns:
            The output.

        Raises:
            PolarsRefusal: The index does not fit the rows.

        Example:
            >>> RollingRun().select(InSelect.column(2), GroupBy((3,)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
        """
        rows = sum(frame.sizes)
        if index.length not in {rows, 1}:
            message = "the index of `rolling` must have the length of the DataFrame"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        return InSelect.column(rows)

    def agg(self, index: InGroups, frame: GroupBy) -> InGroups:
        """Give the output of the `rolling` over the groups of `agg`.

        Args:
            index: The value of the index over the groups.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Raises:
            PolarsRefusal: A group of the index does not have the rows of its
                group. An index of one value for each group, or a literal, has
                one value where the run reads the rows of the group, so polars
                panics.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> RollingRun().agg(frame.literal(), frame)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        if RowRuns().sizes(index) == frame.sizes:
            return replace(frame.column(), original=False)
        if index.state in InGroups.SINGLE:
            raise PolarsRefusal(Stop.PANICS, "rolling: range end index out of range for slice")
        raise PolarsRefusal(ErrorClass.SHAPE, "rolling: the lengths of the groups differ")


class MinMaxByRun:
    """Run `min_by` or `max_by` over the groups of `agg`.

    The groups of the value and of the key must have one size. Values of one
    value for each group keep the groups of the `group_by`, and a group of one
    value that a node rebuilt has groups of size 1. So the two do not meet,
    though each holds one value in each group. A literal key is a column of one
    value. So the search for its least or its greatest value gives one index,
    and not one index for each group. The output then has one row, and the
    group by raises where the frame has two groups or more.

    Beside any other key, the run reads each input at the places of the
    groups of the `group_by`, with no check of bounds. An input of one value
    for each group that keeps those groups holds fewer rows than they index.
    So the run reads past the end of the input (`_past_end`). Such a read
    gives values of no meaning, or it ends the process, at every type. The
    rules refuse it as a crash.

    Example:
        >>> frame = GroupBy((3, 2))
        >>> min_by = MinMaxByRun(Shape.function(FunctionOptions.aggregation()), Kernel())
        >>> min_by.agg([frame.column(), frame.column()], frame).state
        <AggState.AGGREGATED_SCALAR: 'aggregated scalar'>
    """

    def __init__(self, shape: Shape, kernel: Kernel) -> None:
        """Keep the shape and the kernel of the node.

        Args:
            shape: The shape of the node.
            kernel: The facts of its kernel.

        Example:
            >>> MinMaxByRun(Shape.function(FunctionOptions.aggregation()), Kernel())._shape.kind
            <NodeKind.FUNCTION: 'Function'>
        """
        self._shape = shape
        self._kernel = kernel

    def agg(self, inputs: Sequence[InGroups], frame: GroupBy) -> InGroups:
        """Give one value for each group.

        It raises `PolarsRefusal` where the groups of the two inputs have
        other sizes (`GroupLengths`).

        Args:
            inputs: The value and the key.
            frame: The groups of the `group_by`.

        Returns:
            The output.

        Raises:
            PolarsRefusal: The run reads past the end of an input.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> min_by = MinMaxByRun(Shape.function(FunctionOptions.aggregation()), Kernel())
            >>> min_by.agg([frame.column(), frame.literal()], frame).rows
            1
        """
        value, key = inputs
        GroupLengths().check([value, key])
        literal = key.state is AggState.LITERAL_SCALAR
        if not literal and any(self._past_end(entry) for entry in inputs):
            message = "min_by reads past the end of an input"
            raise PolarsRefusal(Stop.CRASHES, message)
        rows = 1 if literal else len(frame.sizes)
        state = AggState.AGGREGATED_SCALAR
        return InGroups(state, frame.sizes, Height.SCALAR, rows, dtype=self._kernel.data)

    def _past_end(self, value: InGroups) -> bool:
        """Say whether an input holds one value for each group, at groups larger than one value.

        Args:
            value: The input.

        Returns:
            Whether the groups of the input index more rows than it holds.

        Example:
            >>> frame = GroupBy((3, 2))
            >>> min_by = MinMaxByRun(Shape.function(FunctionOptions.aggregation()), Kernel())
            >>> first = Run(Shape.of(NodeKind.AGG)).agg([frame.column()], frame)
            >>> min_by._past_end(first), min_by._past_end(first.with_unit_groups())
            (True, False)
        """
        scalar = value.state is AggState.AGGREGATED_SCALAR
        return scalar and value.rows < sum(value.groups)


class BinaryFold:
    """Fold a binary node of two literals into one literal, as the conversion pass of polars does.

    The conversion pass runs the simplifier and the cast pass at each node. It
    visits the inputs of the root before the root, and every other node before
    its inputs. So a node folds only where its two inputs are literals of the
    tree as written, and only the root reads a folded input at its cast.
    The negation of a literal folds later, in the optimizer, after
    the cast pass: a parent reads the class of the literal before the negation.
    The sum of two strings becomes a `concat_str`, not a literal. The logical
    `and` and `or` that `all_horizontal` and `any_horizontal` become do not
    fold (`UNFOLDED`). Each literal is an `Operand`: its planner type and its
    value.

    Example:
        >>> fold = BinaryFold()
        >>> fold.binary(Operator.PLUS, Operand.of_value(200), Operand.of_value(100))
        Operand(dtype=Unknown(int, 300), literal=True, value=300)
    """

    COMPARISONS: ClassVar[dict[Operator, str]] = {
        Operator.LT: "__lt__",
        Operator.GT: "__gt__",
        Operator.EQ: "__eq__",
        Operator.EQ_VALIDITY: "__eq__",
        Operator.NOT_EQ: "__ne__",
        Operator.NOT_EQ_VALIDITY: "__ne__",
        Operator.GT_EQ: "__ge__",
        Operator.LT_EQ: "__le__",
    }
    ARITHMETIC: ClassVar[dict[Operator, str]] = {
        Operator.PLUS: "__add__",
        Operator.MINUS: "__sub__",
        Operator.MULTIPLY: "__mul__",
        Operator.FLOOR_DIVIDE: "__floordiv__",
        Operator.MODULUS: "__mod__",
    }
    UNFOLDED: ClassVar[frozenset[Operator]] = frozenset({Operator.LOGICAL_AND, Operator.LOGICAL_OR})
    LOGICAL: ClassVar[dict[Operator, str]] = {
        Operator.AND: "__and__",
        Operator.OR: "__or__",
        Operator.XOR: "__xor__",
    }

    def binary(self, operator: Operator, left: Operand, right: Operand) -> Operand | None:
        """Fold a binary node of two literals.

        Two literals with no data type fold where they are both integers or
        both floats, and an integer beside a float first becomes a float. Two
        literals with a data type fold where they have one type, with the
        arithmetic of that type: an integer wraps around. A division `/` gives
        a `Float64` with a data type. A comparison gives a `Boolean`. A
        division `//` or a `%` by zero does not fold.

        Args:
            operator: The operator.
            left: The left literal.
            right: The right literal.

        Returns:
            The folded literal, or `None` where the node does not fold.

        Example:
            >>> int8 = Operand.typed_literal(100, DType("Int8"))
            >>> BinaryFold().binary(Operator.PLUS, int8, int8)
            Operand(dtype=Int8, literal=True, value=-56)
            >>> BinaryFold().binary(Operator.TRUE_DIVIDE, Operand.of_value(7), Operand.of_value(2))
            Operand(dtype=Float64, literal=True, value=3.5)
        """
        left, right = self.promoted(left, right)
        if operator in self.LOGICAL:
            return self._logical(operator, left, right)
        if operator in self.UNFOLDED or not self._same_kind(left, right):
            return None
        if operator in self.COMPARISONS:
            result = getattr(left.value, self.COMPARISONS[operator])(right.value)
            return Operand.typed_literal(bool(result), DType("Boolean"))
        if operator is Operator.TRUE_DIVIDE:
            quotient = self._number(left.value) / self._number(right.value)
            return Operand.typed_literal(quotient, DType("Float64"))
        zero = right.value == 0
        if operator in {Operator.FLOOR_DIVIDE, Operator.MODULUS} and zero:
            return None
        value = getattr(left.value, self.ARITHMETIC[operator])(right.value)
        return self._typed(left.dtype, value)

    def promoted(self, left: Operand, right: Operand) -> tuple[Operand, Operand]:
        """Turn an integer with no data type into a float beside a float with no data type.

        The conversion does this at a binary node before it tries the fold,
        so the node keeps the float where the two literals do not fold, as
        for `^`. The two literals are one fact, so they come as a pair.

        Args:
            left: The left literal.
            right: The right literal.

        Returns:
            The two literals.

        Example:
            >>> BinaryFold().promoted(Operand.of_value(1), Operand.of_value(1.5))[0]
            Operand(dtype=Unknown, literal=True, value=1.0)
        """
        pair = (left, right)
        kinds = {Is.dynamic_int(side.dtype) for side in pair}
        floats = [Is.float(side.dtype) and Is.unknown(side.dtype) for side in pair]
        if kinds != {True, False} or not any(floats):
            return pair
        promoted = [Operand.of_value(self._number(side.value)) for side in pair]
        return promoted[0], promoted[1]

    def _same_kind(self, left: Operand, right: Operand) -> bool:
        """Say whether two literals fold together.

        They fold where both are dynamic integers, both are dynamic floats,
        or both have one number type.

        Args:
            left: The left literal.
            right: The right literal.

        Returns:
            Whether they fold.

        Example:
            >>> int8 = Operand.typed_literal(1, DType("Int8"))
            >>> BinaryFold()._same_kind(int8, Operand.typed_literal(1, DType("Int16")))
            False
        """
        if Is.unknown(left.dtype) or Is.unknown(right.dtype):
            return left.dtype.same(right.dtype) and left.dtype.args[0] != "str"
        return left.dtype.same(right.dtype) and Is.primitive_numeric(left.dtype)

    def _logical(self, operator: Operator, left: Operand, right: Operand) -> Operand | None:
        """Fold `&`, `|` or `^` of two `Boolean` literals.

        Args:
            operator: The operator.
            left: The left literal.
            right: The right literal.

        Returns:
            The folded literal, or `None` where a side is not a `Boolean`.

        Example:
            >>> true = Operand.of_value(True)
            >>> BinaryFold()._logical(Operator.XOR, true, true)
            Operand(dtype=Boolean, literal=True, value=False)
        """
        if left.dtype.name != "Boolean" or right.dtype.name != "Boolean":
            return None
        value = getattr(left.value, self.LOGICAL[operator])(right.value)
        return Operand.typed_literal(bool(value), DType("Boolean"))

    def _typed(self, dtype: DType, value: object) -> Operand:
        """Give the result of the arithmetic of a type, where an integer wraps around.

        A dynamic integer takes the type of its new value. A typed integer
        wraps around the range of its type (`IntegerRange.wrapped`).

        Args:
            dtype: The type of the two literals.
            value: The result of the arithmetic in Python.

        Returns:
            The literal.

        Example:
            >>> BinaryFold()._typed(DType("UInt8"), 256)
            Operand(dtype=UInt8, literal=True, value=0)
        """
        if Is.dynamic_int(dtype) and isinstance(value, int):
            return Operand.of_value(value)
        if not (Is.integer(dtype) and isinstance(value, int)):
            return Operand.typed_literal(value, dtype)
        return Operand.typed_literal(IntegerRange().wrapped(dtype.name, value), dtype)

    def _number(self, value: object) -> float:
        """Read the value of a number literal as a float.

        Args:
            value: The value of the literal.

        Returns:
            The value as a float.

        Raises:
            TypeError: The value is not a number, which is a bug in the caller.

        Example:
            >>> BinaryFold()._number(7)
            7.0
        """
        if not isinstance(value, int | float):  # pragma: no cover -- the fold reads number literals
            raise TypeError(value)  # only, so another value is a bug in the caller.
        return float(value)


class Pushdown(enum.Enum):
    """Name what the slice pushdown of the optimizer does to a slice node.

    - kept: the slice runs as written;
    - moved: the slice moves below an elementwise input, and the output does not change;
    - deleted: the slice gives its input unchanged, with its type and its row form.

    Where the optimizer builds a cycle that the value reads, the query never
    returns, and `SlicePushdown` raises `PolarsRefusal`.

    Example:
        >>> Pushdown.DELETED.value
        'deleted'
    """

    KEPT = "kept"
    MOVED = "moved"
    DELETED = "deleted"


@dataclass(frozen=True)
class SliceInput:
    """Hold an input of a slice node as the slice pushdown reads it.

    `height` is the projection height of the input. `literal` says that the
    input is a literal of one value, which the pushdown reads as a number.
    `passes` says that the root of the input is elementwise over inputs of one
    height, so that the places to push a slice sit below it.

    Example:
        >>> plus_one = SliceInput.of(Shape.of(NodeKind.BINARY), [Height.COLUMN, Height.SCALAR])
        >>> plus_one.passes
        True
    """

    height: Height
    literal: bool = False
    passes: bool = False

    @classmethod
    def of(cls, shape: Shape, inputs: Sequence[Height], *, literal: bool = False) -> SliceInput:
        """Give the input from the shape of its root node and the heights of its inputs.

        The pushdown carries up the places of the last nodes that are not
        elementwise. An elementwise node passes the places of its inputs where
        the inputs that are not one value all have the height of the frame, or
        where one input is not one value. Otherwise the node is the place
        itself.

        Args:
            shape: The shape of the root node of the input.
            inputs: The heights of the inputs of that node.
            literal: Whether the input is a literal of one value.

        Returns:
            The input.

        Example:
            >>> SliceInput.of(Shape.of(NodeKind.COLUMN), []).passes
            False
        """
        rows = [height for height in inputs if height is not Height.SCALAR]
        aligned = len(rows) == 1 or all(height is Height.COLUMN for height in rows)
        passes = shape.is_elementwise() and bool(rows) and aligned
        return cls(shape.height(inputs), literal=literal, passes=passes)


class RowCount:
    """Say where the projection pushdown of a lazy `select` counts the rows in place of a `len`.

    A lazy `select` that holds one expression, a `len` that counts the nulls,
    over an input with the height of the frame, gives the number of rows of the
    frame. The input never runs, so a run that it refuses does not fail. The
    planner and the cast pass still read the whole tree, so a refusal of either
    stands. `count`, which leaves out the nulls, keeps its input. Before
    polars 1.41.0, the pushdown counts the rows only in place of `pl.len()`,
    which has no input, so every `len` of an input keeps its input.

    Example:
        >>> from astrolabe._rules.releases import PolarsRelease
        >>> lone = QueryFlags.lazy(lone=True)
        >>> with PolarsRelease.pretend("1.41.0"):
        ...     RowCount().counts(lone, nulls=True, height=Height.COLUMN)
        True
        >>> RowCount().counts(QueryFlags.lazy(), nulls=True, height=Height.COLUMN)
        False
    """

    @polars_releases
    def counts(self, flags: QueryFlags, *, nulls: bool, height: Height) -> bool:
        """Say whether the pushdown counts the rows in place of a `len`.

        Args:
            flags: The passes of the frame.
            nulls: Whether the `len` counts the nulls of its input.
            height: The projection height of the input, after the slice pushdown.

        Returns:
            Whether it does.

        Example:
            >>> RowCount().counts(QueryFlags.lazy(lone=True), nulls=False, height=Height.COLUMN)
            False
        """
        return flags.lone and nulls and height is Height.COLUMN

    @counts.before("1.41.0")
    def _keeps_inputs(self, flags: QueryFlags, *, nulls: bool, height: Height) -> bool:
        """Say that the pushdown of the older releases counts the rows in place of a `len`: never.

        Args:
            flags: The passes of the frame.
            nulls: Whether the `len` counts the nulls of its input.
            height: The projection height of the input, after the slice pushdown.

        Returns:
            Always `False`.

        Example:
            >>> lone = QueryFlags.lazy(lone=True)
            >>> RowCount()._keeps_inputs(lone, nulls=True, height=Height.COLUMN)
            False
        """
        del flags, nulls, height
        return False


class SlicePushdown:
    """Say what the slice pushdown of the optimizer does to a slice node.

    A `collect()` of a lazy frame pushes a slice down the tree, in a `select`
    and in `agg`. An eager frame and `collect_schema()` do not. The pushdown
    collects the places below the slice where it can push it: the places of the
    first input that is not one value, and of each later input that is not one
    value where both have the height of the frame. Where two such inputs have
    other heights, there is no place.

    A slice moves to each place that is not its own value, and the slice node
    then gives its value. An offset or a length that is not a literal is a
    place too. The slice that moves there reads that same node as its offset or
    its length, so the plan holds a cycle. Where a place below the value reads
    that cycle, the query never returns. Otherwise the cycle is out of the
    tree, and the slice node gives its value unchanged. In `agg`, the planner
    wraps the slice in a list for each group before the pushdown runs, so the
    column then holds the values of the value in a list for each group.

    Example:
        >>> column = SliceInput(Height.COLUMN)
        >>> SlicePushdown().outcome(column, column, column)
        <Pushdown.DELETED: 'deleted'>
    """

    def outcome(self, value: SliceInput, offset: SliceInput, length: SliceInput) -> Pushdown:
        """Give what the pushdown does to a slice of `value`.

        Args:
            value: The value that the slice cuts.
            offset: The offset.
            length: The length.

        Returns:
            What the pushdown does.

        Raises:
            PolarsRefusal: A place below the value reads the cycle, so the query never returns.

        Example:
            >>> literal = SliceInput(Height.SCALAR, literal=True)
            >>> plus_one = SliceInput(Height.COLUMN, passes=True)
            >>> SlicePushdown().outcome(plus_one, literal, literal)
            <Pushdown.MOVED: 'moved'>
            >>> column = SliceInput(Height.COLUMN)
            >>> SlicePushdown().outcome(plus_one, column, column)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: hangs: ...
        """
        inputs = [value, offset, length]
        rows = [entry for entry in inputs if entry.height is not Height.SCALAR]
        if not rows or not self._aligned(rows):
            return Pushdown.KEPT
        below = value in rows and value.passes
        arguments = any(entry in rows for entry in (offset, length))
        if below and arguments:
            message = "a place below the value reads the cycle of the offset or the length"
            raise PolarsRefusal(Stop.HANGS, message)
        outcomes = {
            (False, False): Pushdown.KEPT,
            (True, False): Pushdown.MOVED,
            (False, True): Pushdown.DELETED,
        }
        return outcomes[below, arguments]

    def _aligned(self, rows: Sequence[SliceInput]) -> bool:
        """Say whether the inputs that are not one value keep their places.

        Args:
            rows: The inputs that are not one value, in order.

        Returns:
            Whether there is one, or all have the height of the frame.

        Example:
            >>> SlicePushdown()._aligned([SliceInput(Height.UNKNOWN), SliceInput(Height.COLUMN)])
            False
        """
        return len(rows) == 1 or all(entry.height is Height.COLUMN for entry in rows)


@dataclass(frozen=True)
class InputInPlace:
    """Hold the input that the optimizer of polars puts in place of a node, by its position.

    Example:
        >>> InputInPlace(0).index
        0
    """

    index: int


@dataclass(frozen=True)
class LiteralInPlace:
    """Hold the literal that the optimizer of polars puts in place of a node.

    Example:
        >>> LiteralInPlace(Operand.typed_literal(True, DType("Boolean"))).literal.value
        True
    """

    literal: Operand


type Replacement = InputInPlace | LiteralInPlace


class BooleanSimplify:
    """Replace `&` and `|` beside a `Boolean` literal, as the optimizer of polars does.

    A `collect()` runs the optimizer after the cast pass, so the optimizer
    reads each input with the type that the cast pass gives it. `x & True`
    gives `x`, and `x | False` gives `x`. Where the left input is one value,
    `x & False` gives `False`, and `x | True` gives `True`. `True | x` gives
    `True` where the right input is one value. A literal that the cast pass
    gave another type, such as `True` beside an integer column, is not a
    `Boolean` there, so the node stays.

    The kernel of the node then never runs. So the node gives the type of its
    input, even where the kernel refuses that type, as for `x & True` over a
    `List` column. `collect_schema()` does not run the optimizer, so the
    planner type does not change. Inside a filter, the optimizer has more
    rules, which do not apply to a projection.

    Example:
        >>> true = Operand.typed_literal(True, DType("Boolean"))
        >>> listed = Operand.column(DType("List", (DType("Int8"),)))
        >>> heights = [Height.COLUMN, Height.SCALAR]
        >>> BooleanSimplify().replaced(Operator.AND, [listed, true], heights)
        InputInPlace(index=0)
    """

    def replaced(
        self,
        operator: Operator,
        inputs: Sequence[Operand],
        heights: Sequence[Height],
    ) -> Replacement | None:
        """Give what the optimizer puts in place of a binary node, or `None` where the node stays.

        Args:
            operator: The operator of the node.
            inputs: The two inputs, with the types that the cast pass gives them.
            heights: The projection height of each input.

        Returns:
            The replacement, or `None`.

        Example:
            >>> false = Operand.typed_literal(False, DType("Boolean"))
            >>> one = [Height.SCALAR, Height.SCALAR]
            >>> BooleanSimplify().replaced(Operator.AND, [false, false], one)
            LiteralInPlace(literal=Operand(dtype=Boolean, literal=True, value=False))
            >>> column = Operand.column(DType("Boolean"))
            >>> rows = [Height.COLUMN, Height.SCALAR]
            >>> BooleanSimplify().replaced(Operator.AND, [column, false], rows) is None
            True
            >>> true = Operand.typed_literal(True, DType("Boolean"))
            >>> BooleanSimplify().replaced(Operator.OR, [true, column], one)
            LiteralInPlace(literal=Operand(dtype=Boolean, literal=True, value=True))
        """
        if operator not in {Operator.AND, Operator.OR}:
            return None
        neutral = operator is Operator.AND
        left, right = inputs
        if self._is(right, value=neutral):
            return InputInPlace(0)
        scalar = [height is Height.SCALAR for height in heights]
        absorbing = LiteralInPlace(Operand.typed_literal(not neutral, DType("Boolean")))
        if scalar[0] and self._is(right, value=not neutral):
            return absorbing
        if operator is Operator.OR and scalar[1] and self._is(left, value=True):
            return absorbing
        return None

    def _is(self, operand: Operand, *, value: bool) -> bool:
        """Say whether an input is a `Boolean` literal of one value.

        Args:
            operand: The input.
            value: The value.

        Returns:
            Whether it is.

        Example:
            >>> true = Operand.typed_literal(True, DType("Boolean"))
            >>> BooleanSimplify()._is(true, value=True)
            True
            >>> BooleanSimplify()._is(Operand.typed_literal(True, DType("Int64")), value=True)
            False
        """
        boolean = operand.dtype.name == "Boolean"
        return operand.literal and boolean and operand.value is value
