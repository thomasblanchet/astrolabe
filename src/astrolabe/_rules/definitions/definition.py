"""State the rules of one node of polars as the parts of the vocabulary that it uses.

A node of polars has rules at five places: its type function, its options,
its cast pass, its kernel, and the rewrites of the simplifier. A definition
names, at each place, the part of the vocabulary that polars names there, for
example `FieldsMapper.with_same_dtype` as the type function of `abs`. It
computes nothing of its own. Code that evaluates a node reads the definition
and calls the parts that it names.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from astrolabe._rules.definitions.node_options import (
    NodeIdentity,
    NodeOptions,
    OptionValue,
    OptionValues,
)
from astrolabe._rules.definitions.simplify_adapters import NoSimplify, SimplifyRule
from astrolabe._rules.kernels.dispatch import Data
from astrolabe._rules.releases import polars_releases

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.definitions.cast_adapters import CastRule
    from astrolabe._rules.definitions.kernel_adapters import KernelRule
    from astrolabe._rules.definitions.type_function_adapters import TypeFunction
    from astrolabe._rules.definitions.value_adapters import ValueRule
    from astrolabe._rules.dtypes import DType
    from astrolabe._rules.execution import PhysicalExpr
    from astrolabe._rules.options import Shape
    from astrolabe._rules.type_functions import Operand


class NullRule(enum.Enum):
    """Say when the output of a node holds only nulls, from the inputs that hold only nulls.

    A later node reads that fact: polars casts a series of only nulls to any
    type, with no kernel. `NONE` says that the output can hold a value
    whatever the inputs hold, as `fill_null` or `len` do. `FIRST` says that
    the output holds only nulls where the first input does, as a cast or a
    slice does. `ANY` says that it holds only nulls where any input does, as
    `+` does. A struct beside a struct of only nulls gives a struct of null
    fields, and not null structs, so `ANY` claims nothing for a struct.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> int8 = DType("Int8")
        >>> NullRule.ANY.output([Data(int8), Data(int8, nulls=True)], int8)
        True
    """

    NONE = "none"
    FIRST = "first"
    ANY = "any"

    def output(self, inputs: Sequence[Data], dtype: DType) -> bool:
        """Say whether the output holds only nulls.

        Args:
            inputs: The data of each input.
            dtype: The type of the output.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> int8 = DType("Int8")
            >>> NullRule.FIRST.output([Data(int8), Data(int8, nulls=True)], int8)
            False
        """
        if self is NullRule.FIRST:
            return bool(inputs) and inputs[0].nulls
        if self is NullRule.ANY and dtype.name != "Struct":
            return any(data.nulls for data in inputs)
        return False

    def held(self, inputs: Sequence[Data], dtype: DType) -> bool:
        """Say whether the output holds a null wherever it holds a row, value by value.

        A node that works value by value gives a null in each row where the
        rule meets a null, so it holds a null where an input that the rule
        reads holds one.

        Args:
            inputs: The data of each input.
            dtype: The type of the output.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> int8 = DType("Int8")
            >>> NullRule.ANY.held([Data(int8), Data(int8, some_null=True)], int8)
            True
        """
        held = [Data(data.dtype, nulls=data.holds_null()) for data in inputs]
        return self.output(held, dtype)


class ShortNulls(Protocol):
    """Say when a node gives nulls because of its first input: only nulls over few rows, or some."""

    def all_null(self, rows: int, computed: Sequence[Operand]) -> bool:
        """Say whether the output holds only nulls over an input of at most so many rows.

        Args:
            rows: The most rows of the first input.
            computed: Each input as the run computes it.
        """

    def some_null(self, computed: Sequence[Operand]) -> bool:
        """Say whether the output holds a null wherever it holds a row, as the code decides it.

        Args:
            computed: Each input as the run computes it.
        """

    def at(self, node: NodeOptions) -> ShortNulls:
        """Give the rule of one node, where it reads an option of the node.

        Args:
            node: The serialized root of the node.
        """


@dataclass(frozen=True)
class PeriodNulls:
    """Say that a node that moves its values by a count gives only nulls over few rows.

    `shift`, `diff` and `pct_change` give a null for each of the first rows
    that the count moves past, so an input of no more rows than the count
    gives only nulls. The count is the second input, and the code holds it
    where it is a literal.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> PeriodNulls().all_null(1, [Operand.of_value(1), Operand.of_value(1)])
        True
    """

    def all_null(self, rows: int, computed: Sequence[Operand]) -> bool:
        """Say whether the count moves past every row.

        Args:
            rows: The most rows of the first input.
            computed: Each input as the run computes it.

        Returns:
            Whether the count is a literal of at least so many rows, and not zero.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> PeriodNulls().all_null(2, [Operand.of_value(1), Operand.of_value(1)])
            False
        """
        count = self._count(computed)
        return count is not None and count != 0 and rows <= abs(count)

    def some_null(self, computed: Sequence[Operand]) -> bool:
        """Say whether the count moves past a row, which leaves a null in its place.

        Args:
            computed: Each input as the run computes it.

        Returns:
            Whether the count is a literal other than zero.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> PeriodNulls().some_null([Operand.of_value(1), Operand.of_value(-2)])
            True
        """
        count = self._count(computed)
        return count is not None and count != 0

    def _count(self, computed: Sequence[Operand]) -> int | None:
        """Give the count where the code holds it as a literal integer.

        Args:
            computed: Each input as the run computes it.

        Returns:
            The count, or `None`.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> PeriodNulls()._count([Operand.of_value(1)]) is None
            True
        """
        count = computed[1] if len(computed) > 1 else None
        if count is None or not count.literal or not isinstance(count.value, int):
            return None
        return count.value

    def at(self, node: NodeOptions) -> PeriodNulls:
        """Give the rule of one node: the same rule, which reads no option.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> PeriodNulls().at(NodeOptions({"Function": {}}))
            PeriodNulls()
        """
        del node
        return self


@dataclass(frozen=True)
class WindowNulls:
    """Say that a rolling function gives only nulls over fewer rows than the samples that it needs.

    A window gives a null until it holds `min_samples` values, so an input
    of fewer rows gives only nulls.

    Example:
        >>> WindowNulls(2).all_null(1, [])
        True
    """

    samples: int | OptionValue[int]

    def all_null(self, rows: int, computed: Sequence[Operand]) -> bool:
        """Say whether the input holds fewer rows than the samples.

        Args:
            rows: The most rows of the first input.
            computed: Each input as the run computes it, which the rule does not read.

        Returns:
            Whether it does.

        Example:
            >>> WindowNulls(2).all_null(2, [])
            False
        """
        del computed
        return rows < OptionValues().known(self.samples, "the samples of the window")

    def some_null(self, computed: Sequence[Operand]) -> bool:
        """Say whether the window of the first row holds fewer values than the samples.

        Args:
            computed: Each input as the run computes it, which the rule does not read.

        Returns:
            Whether the window needs two samples or more.

        Example:
            >>> WindowNulls(1).some_null([])
            False
        """
        return self.all_null(1, computed)

    def at(self, node: NodeOptions) -> WindowNulls:
        """Give the rule of one node, with the samples that its options give.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> WindowNulls(3).at(NodeOptions({"Function": {}}))
            WindowNulls(samples=3)
        """
        return WindowNulls(OptionValues().at(self.samples, node))


@dataclass(frozen=True)
class DdofNulls:
    """Say that a deviation gives a null over no more rows than its degrees of freedom.

    `std` and `var` divide by the number of values less `ddof`, and give a
    null where that divisor is not more than zero.

    Example:
        >>> DdofNulls(1).all_null(1, [])
        True
    """

    ddof: int | OptionValue[int]

    def all_null(self, rows: int, computed: Sequence[Operand]) -> bool:
        """Say whether the input holds no more rows than the degrees of freedom.

        Args:
            rows: The most rows of the first input.
            computed: Each input as the run computes it, which the rule does not read.

        Returns:
            Whether it does.

        Example:
            >>> DdofNulls(1).all_null(2, [])
            False
        """
        del computed
        return rows <= OptionValues().known(self.ddof, "the degrees of freedom")

    def some_null(self, computed: Sequence[Operand]) -> bool:
        """Say that one value holds a null only where every value does, which `all_null` says.

        Args:
            computed: Each input as the run computes it, which the rule does not read.

        Returns:
            False.

        Example:
            >>> DdofNulls(1).some_null([])
            False
        """
        del computed
        return False

    def at(self, node: NodeOptions) -> DdofNulls:
        """Give the rule of one node, with the degrees of freedom that its options give.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> DdofNulls(2).at(NodeOptions({"Agg": {}}))
            DdofNulls(ddof=2)
        """
        return DdofNulls(OptionValues().at(self.ddof, node))


@dataclass(frozen=True)
class WeightedNulls:
    """Say that an exponentially weighted node gives only nulls over fewer rows than it needs.

    The node gives a null until it has read `min_samples` values. From
    polars 1.43.0, an unbiased deviation, that is `ewm_std` or `ewm_var` with
    `bias=False`, also gives a null until it has read two values. Older
    releases give 0 for the deviation of one value.

    Example:
        >>> from astrolabe._rules.releases import PolarsRelease
        >>> with PolarsRelease.pretend("1.43.0"):
        ...     WeightedNulls(1, bias=False).all_null(1, [])
        True
    """

    samples: int | OptionValue[int]
    bias: bool | OptionValue[bool] = True

    def all_null(self, rows: int, computed: Sequence[Operand]) -> bool:
        """Say whether the input holds fewer rows than the node needs.

        Args:
            rows: The most rows of the first input.
            computed: Each input as the run computes it, which the rule does not read.

        Returns:
            Whether it does.

        Example:
            >>> WeightedNulls(1).all_null(1, [])
            False
        """
        del computed
        samples = OptionValues().known(self.samples, "the samples of the weights")
        biased = OptionValues().known(self.bias, "the bias of the deviation")
        return rows < (samples if biased else max(samples, self._unbiased_values()))

    @polars_releases
    def _unbiased_values(self) -> int:
        """Give the values that an unbiased deviation reads before it gives a value: two.

        Returns:
            The number of values.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.43.0"):
            ...     WeightedNulls(1)._unbiased_values()
            2
        """
        return 2

    @_unbiased_values.before("1.43.0")
    def _one_value(self) -> int:
        """Give the values that an unbiased deviation of the older releases reads: one.

        Returns:
            The number of values.

        Example:
            >>> WeightedNulls(1)._one_value()
            1
        """
        return 1

    def some_null(self, computed: Sequence[Operand]) -> bool:
        """Say whether the first row holds fewer values than the node needs.

        Args:
            computed: Each input as the run computes it, which the rule does not read.

        Returns:
            Whether the node needs two values or more.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.43.0"):
            ...     WeightedNulls(1, bias=False).some_null([])
            True
        """
        return self.all_null(1, computed)

    def at(self, node: NodeOptions) -> WeightedNulls:
        """Give the rule of one node, with the samples and the bias that its options give.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> WeightedNulls(3).at(NodeOptions({"Function": {}}))
            WeightedNulls(samples=3, bias=True)
        """
        samples = OptionValues().at(self.samples, node)
        return WeightedNulls(samples, bias=OptionValues().at(self.bias, node))


@dataclass(frozen=True)
class Definition:
    """Hold the rules of one node of polars, stated in the vocabulary.

    `identity` names the node. `type_function` gives its planner type,
    `options` its row form and its wrap in `agg`, `casts` the changes of the
    cast pass to its inputs, and `kernel` the type of the data that it gives.
    `simplify` gives the rewrites of the simplifier of polars, which can take
    the place of the node before it runs. Most nodes have none. `physical`
    names the physical expression that polars runs the node with, where it is
    not the default of its kind of node: `min_by`, for example, runs as an
    expression of its own, and not as a function. It can be an option of the
    node, as for `fill_null`, whose strategy decides it. `names` says that a
    part reads the output names of the inputs of the node, as `pl.struct`
    names its fields. `nulls` says when the output holds only nulls
    (`NullRule`), and `short` says when it does because its first input
    holds few rows (`ShortNulls`). `computes` gives the value that the node
    computes over a literal, where a later rule reads it (`ValueRule`). Each
    field names a part of the vocabulary, and the definition computes nothing
    of its own. A part can
    read an option of the node, so code that evaluates a node first reads its
    options (`at`).

    Example:
        >>> from astrolabe._rules.definitions.elementwise import Elementwise
        >>> Elementwise.ABS.identity
        NodeIdentity(shape='Function', function='Abs')
        >>> Elementwise.ABS.type_function
        Mapper(with_same_dtype)
    """

    identity: NodeIdentity
    type_function: TypeFunction
    options: Shape | OptionValue[Shape]
    casts: CastRule
    kernel: KernelRule
    simplify: SimplifyRule = field(default_factory=NoSimplify)
    physical: PhysicalExpr | OptionValue[PhysicalExpr | None] | None = None
    names: bool = False
    nulls: NullRule = NullRule.NONE
    short: ShortNulls | None = None
    computes: ValueRule | None = None

    def at(self, node: NodeOptions) -> Definition:
        """Give the rules of one node: each part that is an option read at that node.

        Args:
            node: The serialized root of the node.

        Returns:
            The rules.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> cast = {"Cast": {"expr": {}, "dtype": {"Literal": "Date"}, "options": "Strict"}}
            >>> Elementwise.CAST.at(NodeOptions(cast)).casts
            CastNode(target=Date, option=<CastOption.STRICT: 0>)
        """
        return Definition(
            identity=self.identity,
            type_function=self.type_function.at(node),
            options=OptionValues().at(self.options, node),
            casts=self.casts.at(node),
            kernel=self.kernel.at(node),
            simplify=self.simplify.at(node),
            physical=OptionValues().at(self.physical, node),
            names=self.names,
            nulls=self.nulls,
            short=None if self.short is None else self.short.at(node),
            computes=self.computes,
        )

    def shape(self) -> Shape:
        """Give the options of the node in polars, which decide its row form and its run.

        It raises `UnreadOptionError` where the shape is an option of the
        node, and nobody read the options of the node.

        Returns:
            The shape.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> Aggregations.SUM.shape().kind
            <NodeKind.AGG: 'Agg'>
        """
        return OptionValues().known(self.options, "the shape")

    def physical_expr(self) -> PhysicalExpr | None:
        """Give the physical expression that polars runs the node with, where it is not the default.

        It raises `UnreadOptionError` where the physical expression is an
        option of the node, and nobody read the options of the node.

        Returns:
            The physical expression, or `None` for the default of its kind of node.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> Elementwise.ABS.physical_expr() is None
            True
        """
        return OptionValues().known(self.physical, "the physical expression")
