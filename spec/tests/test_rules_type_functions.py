"""Check the type functions of `astrolabe._rules.type_functions` against the planner of polars.

Each rule gives the planner type of a node: the type that `collect_schema()`
shows. Polars types each node with one method of its `FieldsMapper`, or with
the rule of its kind of node (`ExprFields`). Each test here runs one or two
real nodes on every type of `RuleKinds`, in `select` and in `group_by().agg()`.
In `agg`, a node shows a list of its type unless it gives one value for each
group, as `ProjectionSchema` says. Polars refuses a plan where the type
function refuses it, and also where the cast pass refuses it. So the rule of a
node whose cast pass can refuse runs that pass too, with the rules of
`astrolabe._rules.cast_pass`.

Where polars refuses a plan, the rules must refuse it with the class of the
error that polars raises. Each test takes one node, and its assertion names
every type that fails.
"""

from __future__ import annotations

import datetime
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import (
    PolarsAnswer,
    RuleAnswer,
    RuleKinds,
    Stopped,
    TreeColumns,
    TreeFrame,
)
from astrolabe._rules.cast_pass import (
    BinaryCasts,
    EwmCasts,
    FunctionCasts,
    SettledCasts,
    TernaryCasts,
)
from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.operators import Operator
from astrolabe._rules.options import CastingRules
from astrolabe._rules.supertype import SupertypeFlag
from astrolabe._rules.type_functions import (
    AggFields,
    ArrayClosures,
    ExprFields,
    FieldsMapper,
    FunctionClosures,
    ListClosures,
    Operand,
    ProjectionSchema,
    StructClosures,
    TemporalClosures,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class Planner:
    """Give the planner type of an expression over a frame, in `select` and in `group_by().agg()`.

    The frame has the columns of a schema and a key `g`. A plan that polars
    refuses gives how polars stops.

    Example:
        >>> Planner().select({"x": pl.Int8()}, pl.col("x").abs())
        Int8
        >>> Planner().agg({"x": pl.Int8()}, pl.col("x").abs())
        List(Int8)
    """

    def select(self, schema: Mapping[str, pl.DataType], expression: pl.Expr) -> DType | Stopped:
        """Give the type in `select`.

        Args:
            schema: The columns of the frame.
            expression: The expression.

        Returns:
            The type, or how polars stops where it refuses the plan.

        Example:
            >>> Planner().select({"x": pl.String()}, pl.col("x").sum())
            raises InvalidOperationError
        """
        frame = pl.LazyFrame(schema=schema)
        return PolarsAnswer().answer(
            lambda: frame.select(expression.alias("o")).collect_schema()["o"],
        )

    def agg(self, schema: Mapping[str, pl.DataType], expression: pl.Expr) -> DType | Stopped:
        """Give the type in `group_by("g").agg()`.

        Args:
            schema: The columns of the frame.
            expression: The expression.

        Returns:
            The type, or how polars stops where it refuses the plan.

        Example:
            >>> Planner().agg({"x": pl.Int8()}, pl.col("x").sum())
            Int64
        """
        frame = pl.LazyFrame(schema={**schema, "g": pl.Int8()})
        grouped = frame.group_by("g").agg(expression.alias("o"))
        return PolarsAnswer().answer(lambda: grouped.collect_schema()["o"])


class RuleSchema:
    """Run a rule on its inputs, and give its type in `select` and `agg`, or how it refuses.

    Example:
        >>> RuleSchema().select(ExprFields().literal, Operand.of_value(1))
        Int32
        >>> RuleSchema().agg(AggFields().same, DType("Int8"), scalar=False)
        List(Int8)
    """

    def select(self, rule: Callable[..., DType], *inputs: object) -> DType | Stopped:
        """Give the type of the rule in the schema of `select`.

        Args:
            rule: The rule.
            inputs: The inputs of the rule.

        Returns:
            The type, or how the rule says that polars stops.

        Example:
            >>> RuleSchema().select(AggFields().sum, DType("String"))
            raises InvalidOperationError
        """
        dtype = RuleAnswer().result(lambda: rule(*inputs))
        return dtype if isinstance(dtype, Stopped) else ProjectionSchema().select(dtype)

    def agg(self, rule: Callable[..., DType], *inputs: object, scalar: bool) -> DType | Stopped:
        """Give the type of the rule in the schema of `agg`.

        Args:
            rule: The rule.
            inputs: The inputs of the rule.
            scalar: Whether the node gives one value for each group.

        Returns:
            The type, or how the rule says that polars stops.

        Example:
            >>> RuleSchema().agg(AggFields().sum, DType("Int8"), scalar=True)
            Int64
        """
        dtype = RuleAnswer().result(lambda: rule(*inputs))
        if isinstance(dtype, Stopped):
            return dtype
        return ProjectionSchema().agg(dtype, scalar=scalar)


@dataclass(frozen=True)
class UnaryNode:
    """Hold a node of one column, the rule that types it, and whether it gives one value per group.

    Example:
        >>> node = UnaryNode("abs", lambda x: x.abs(), lambda d: FieldsMapper(d).with_same_dtype())
        >>> node.rule(DType("Int8"))
        Int8
    """

    name: str
    build: Callable[[pl.Expr], pl.Expr]
    rule: Callable[[DType], DType]
    scalar: bool = False


class EwmPlanned:
    """Give the planner type of an exponentially weighted function, after its cast pass.

    The cast pass of these functions casts the input to `Float64`, and it
    refuses some inputs (`EwmCasts`). Polars types the node with its field
    at the input before the cast.

    Example:
        >>> EwmPlanned(lambda d: FieldsMapper(d).var_dtype())(DType("Int8"))
        Float64
    """

    def __init__(self, field: Callable[[DType], DType]) -> None:
        """Keep the field of the function.

        Args:
            field: The rule of the field, at the type of the input.

        Example:
            >>> EwmPlanned(lambda d: d)(DType("Float32"))
            Float32
        """
        self._field = field

    def __call__(self, dtype: DType) -> DType:
        """Give the planner type, or raise the refusal of the cast pass.

        Args:
            dtype: The type of the input.

        Returns:
            The type of the field.

        Example:
            >>> EwmPlanned(lambda d: d)(DType("Int8"))
            Int8
        """
        EwmCasts().visit([Operand.column(dtype)])
        return self._field(dtype)


@dataclass(frozen=True)
class PairNode:
    """Hold a node of two columns, the rule that types it, and whether it gives one value per group.

    Example:
        >>> node = PairNode("coalesce", pl.coalesce, PlannedPair().coalesce)
        >>> node.rule(Operand.column(DType("Int8")), Operand.column(DType("UInt8")))
        Int16
    """

    name: str
    build: Callable[[pl.Expr, pl.Expr], pl.Expr]
    rule: Callable[[Operand, Operand], DType]
    scalar: bool = False


class PlannedPair:
    """Give the rules of nodes of two inputs that also run the cast pass where it can refuse.

    Example:
        >>> int8, text = Operand.column(DType("Int8")), Operand.column(DType("String"))
        >>> PlannedPair().coalesce(int8, text)
        String
    """

    def binary(self, op: Operator) -> Callable[[Operand, Operand], DType]:
        """Give the rule of a binary operator: its field, then a visit of the cast pass.

        Args:
            op: The operator.

        Returns:
            The rule.

        Example:
            >>> int8, text = Operand.column(DType("Int8")), Operand.column(DType("String"))
            >>> PlannedPair().binary(Operator.PLUS)(int8, text)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """

        def rule(left: Operand, right: Operand) -> DType:
            dtype = ExprFields().binary(op, left, right)
            visit = BinaryCasts().visit(op, left, right)
            SettledCasts().of(visit, [left, right])
            return dtype

        return rule

    def ternary(self, truthy: Operand, falsy: Operand) -> DType:
        """Give the rule of `when(True).then(truthy).otherwise(falsy)`.

        Args:
            truthy: The `then` input.
            falsy: The `otherwise` input.

        Returns:
            The type.

        Example:
            >>> PlannedPair().ternary(Operand.column(DType("Int8")), Operand.of_value(300))
            Int16
        """
        dtype = ExprFields().ternary(truthy.dtype, falsy.dtype)
        visit = TernaryCasts().visit(truthy, falsy)
        SettledCasts().of(visit, [truthy, falsy])
        return dtype

    def coalesce(self, first: Operand, second: Operand) -> DType:
        """Give the rule of `coalesce`: the supertype, then the cast pass with the default join.

        Args:
            first: The first input.
            second: The second input.

        Returns:
            The type.

        Example:
            >>> int8, uint8 = Operand.column(DType("Int8")), Operand.column(DType("UInt8"))
            >>> PlannedPair().coalesce(int8, uint8)
            Int16
        """
        return self._supertyped(CastingRules.cast_to_supertypes(), [first, second])

    def max_horizontal(self, first: Operand, second: Operand) -> DType:
        """Give the rule of `max_horizontal`: the cast pass joins with no cast of a number to text.

        Args:
            first: The first input.
            second: The second input.

        Returns:
            The type.

        Example:
            >>> int8, text = Operand.column(DType("Int8")), Operand.column(DType("String"))
            >>> PlannedPair().max_horizontal(int8, text)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        return self._supertyped(CastingRules.supertype(SupertypeFlag(0)), [first, second])

    def sum_horizontal(self, first: Operand, second: Operand) -> DType:
        """Give the rule of `sum_horizontal`: the supertype, with `UInt32` for `Boolean`.

        The cast pass casts the inputs to the supertype.

        Args:
            first: The first input.
            second: The second input.

        Returns:
            The type.

        Example:
            >>> boolean = Operand.column(DType("Boolean"))
            >>> PlannedPair().sum_horizontal(boolean, boolean)
            UInt32
        """
        joined = self._supertyped(CastingRules.cast_to_supertypes(), [first, second])
        return FunctionClosures().sum_horizontal(joined)

    def _supertyped(self, rules: CastingRules, operands: list[Operand]) -> DType:
        """Give `map_to_supertype`, then run the cast pass of the function.

        Args:
            rules: The casting rule of the function, with the flags of its join.
            operands: The inputs.

        Returns:
            The type.

        Example:
            >>> join = CastingRules.cast_to_supertypes()
            >>> PlannedPair()._supertyped(join, [Operand.column(DType("Int8"))] * 2)
            Int8
        """
        dtype = FieldsMapper(*(operand.dtype for operand in operands)).map_to_supertype()
        visit = FunctionCasts(rules).visit(operands)
        SettledCasts().of(visit, operands)
        return dtype


class Nodes:
    """Give the nodes that the tests run, each with the rule of polars that types it.

    The nodes of one column run on every type of `RuleKinds`, and the nodes of
    two columns on every pair.

    Example:
        >>> len(Nodes.UNARY) > 40, len(Nodes.PAIRS) > 10
        (True, True)
    """

    UNARY: ClassVar[tuple[UnaryNode, ...]] = (
        UnaryNode("abs", lambda x: x.abs(), lambda d: FieldsMapper(d).with_same_dtype()),
        UnaryNode("reverse", lambda x: x.reverse(), lambda d: FieldsMapper(d).with_same_dtype()),
        UnaryNode(
            "null_count",
            lambda x: x.null_count(),
            lambda d: FieldsMapper(d).with_dtype(DType.idx()),
            scalar=True,
        ),
        UnaryNode(
            "is_null",
            lambda x: x.is_null(),
            lambda d: FieldsMapper(d).with_dtype(DType("Boolean")),
        ),
        UnaryNode(
            "hash",
            lambda x: x.hash(),
            lambda d: FieldsMapper(d).with_dtype(DType("UInt64")),
        ),
        UnaryNode(
            "cum_sum",
            lambda x: x.cum_sum(),
            lambda d: FieldsMapper(d).map_dtype(FunctionClosures().cum_sum),
        ),
        UnaryNode(
            "cum_prod",
            lambda x: x.cum_prod(),
            lambda d: FieldsMapper(d).map_dtype(FunctionClosures().cum_prod),
        ),
        UnaryNode(
            "product",
            lambda x: x.product(),
            lambda d: FieldsMapper(d).map_dtype(FunctionClosures().product),
            scalar=True,
        ),
        UnaryNode(
            "diff",
            lambda x: x.diff(),
            lambda d: FieldsMapper(d).map_dtype(FunctionClosures().diff),
        ),
        UnaryNode(
            "pct_change",
            lambda x: x.pct_change(),
            lambda d: FieldsMapper(d).map_dtype(FunctionClosures().pct_change),
        ),
        UnaryNode(
            "rle",
            lambda x: x.rle(),
            lambda d: FieldsMapper(d).map_dtype(FunctionClosures().rle),
        ),
        UnaryNode(
            "value_counts",
            lambda x: x.value_counts(),
            lambda d: FieldsMapper(d).map_dtype(
                FunctionClosures().value_counts("x", "count", normalize=False),
            ),
        ),
        UnaryNode(
            "repeat_by",
            lambda x: x.repeat_by(2),
            lambda d: FieldsMapper(d).map_dtype(FunctionClosures().repeat_by),
        ),
        UnaryNode(
            "reshape",
            lambda x: x.reshape((-1, 1)),
            lambda d: FieldsMapper(d).try_map_dtype(FunctionClosures().reshape((-1, 1))),
        ),
        UnaryNode(
            "struct.field",
            lambda x: x.struct.field("a"),
            lambda d: FieldsMapper(d).try_map_field(StructClosures().field_by_name("a")),
        ),
        UnaryNode(
            "struct.prefix_fields",
            lambda x: x.name.prefix_fields("p_"),
            lambda d: FieldsMapper(d).try_map_dtype(StructClosures().prefix_fields("p_")),
        ),
        UnaryNode(
            "sqrt",
            lambda x: x.sqrt(),
            lambda d: FieldsMapper(d).map_numeric_to_float_dtype(coerce_decimal=True),
        ),
        UnaryNode(
            "interpolate",
            lambda x: x.interpolate(),
            lambda d: FieldsMapper(d).map_numeric_to_float_dtype(coerce_decimal=False),
        ),
        UnaryNode("sin", lambda x: x.sin(), lambda d: FieldsMapper(d).map_to_float_dtype()),
        UnaryNode("exp", lambda x: x.exp(), lambda d: FieldsMapper(d).map_to_float_dtype()),
        UnaryNode(
            "to_physical",
            lambda x: x.to_physical(),
            lambda d: FieldsMapper(d).to_physical_type(),
        ),
        UnaryNode(
            "quantile",
            lambda x: x.quantile(0.5),
            lambda d: FieldsMapper(d).moment_dtype(),
            scalar=True,
        ),
        UnaryNode(
            "rolling_mean",
            lambda x: x.rolling_mean(2),
            lambda d: FieldsMapper(d).moment_dtype(),
        ),
        UnaryNode("rolling_var", lambda x: x.rolling_var(2), lambda d: FieldsMapper(d).var_dtype()),
        UnaryNode("rolling_sum", lambda x: x.rolling_sum(2), lambda d: FieldsMapper(d).sum_dtype()),
        UnaryNode(
            "rolling_min",
            lambda x: x.rolling_min(2),
            lambda d: FieldsMapper(d).with_same_dtype(),
        ),
        UnaryNode(
            "ewm_var",
            lambda x: x.ewm_var(alpha=0.5),
            EwmPlanned(lambda d: FieldsMapper(d).var_dtype()),
        ),
        UnaryNode(
            "ewm_mean",
            lambda x: x.ewm_mean(alpha=0.5),
            EwmPlanned(lambda d: FieldsMapper(d).map_numeric_to_float_dtype(coerce_decimal=True)),
        ),
        UnaryNode(
            "sign",
            lambda x: x.sign(),
            lambda d: (
                FieldsMapper(d)
                .ensure_satisfies(lambda _, t: Is.numeric(t), "sign")
                .with_same_dtype()
            ),
        ),
        UnaryNode(
            "not",
            lambda x: x.not_(),
            lambda d: FieldsMapper(d).try_map_dtype(FunctionClosures().not_),
        ),
        UnaryNode(
            "bitwise_count_ones",
            lambda x: x.bitwise_count_ones(),
            lambda d: FieldsMapper(d).try_map_dtype(FunctionClosures().bitwise("count_ones")),
        ),
        UnaryNode(
            "list.len",
            lambda x: x.list.len(),
            lambda d: FieldsMapper(d).ensure_is_list().with_dtype(DType.idx()),
        ),
        UnaryNode(
            "list.first",
            lambda x: x.list.first(),
            lambda d: FieldsMapper(d).ensure_is_list().map_to_list_and_array_inner_dtype(),
        ),
        UnaryNode("list.sum", lambda x: x.list.sum(), lambda d: FieldsMapper(d).nested_sum_type()),
        UnaryNode(
            "list.mean",
            lambda x: x.list.mean(),
            lambda d: FieldsMapper(d).nested_mean_median_type(),
        ),
        UnaryNode(
            "list.std",
            lambda x: x.list.std(),
            lambda d: FieldsMapper(d).ensure_is_list().moment_dtype(),
        ),
        UnaryNode(
            "list.var",
            lambda x: x.list.var(),
            lambda d: FieldsMapper(d).ensure_is_list().var_dtype(),
        ),
        UnaryNode(
            "list.diff",
            lambda x: x.list.diff(),
            lambda d: FieldsMapper(d).try_map_dtype(ListClosures().diff),
        ),
        UnaryNode(
            "list.join",
            lambda x: x.list.join(","),
            lambda d: FieldsMapper(d).try_map_dtype(ListClosures().join),
        ),
        UnaryNode(
            "list.to_array",
            lambda x: x.list.to_array(2),
            lambda d: FieldsMapper(d).ensure_is_list().try_map_dtype(ListClosures().to_array(2)),
        ),
        UnaryNode(
            "arr.len",
            lambda x: x.arr.len(),
            lambda d: FieldsMapper(d).ensure_is_array().with_dtype(DType.idx()),
        ),
        UnaryNode(
            "arr.sum",
            lambda x: x.arr.sum(),
            lambda d: FieldsMapper(d).ensure_is_array().nested_sum_type(),
        ),
        UnaryNode(
            "arr.mean",
            lambda x: x.arr.mean(),
            lambda d: FieldsMapper(d).ensure_is_array().moment_dtype(),
        ),
        UnaryNode(
            "arr.explode",
            lambda x: x.arr.explode(),
            lambda d: FieldsMapper(d).ensure_is_array().try_map_to_array_inner_dtype(),
        ),
        UnaryNode(
            "arr.to_list",
            lambda x: x.arr.to_list(),
            lambda d: FieldsMapper(d).ensure_is_array().try_map_dtype(ArrayClosures().to_list),
        ),
        UnaryNode(
            "dt.cast_time_unit",
            lambda x: x.dt.cast_time_unit("ms"),
            lambda d: FieldsMapper(d).try_map_dtype(TemporalClosures().with_time_unit("ms")),
        ),
        UnaryNode(
            "dt.replace_time_zone",
            lambda x: x.dt.replace_time_zone("UTC"),
            lambda d: FieldsMapper(d).map_datetime_dtype_timezone("UTC"),
        ),
        UnaryNode(
            "dt.year",
            lambda x: x.dt.year(),
            lambda d: FieldsMapper(d).with_dtype(DType("Int32")),
        ),
        UnaryNode(
            "dt.datetime",
            lambda x: x.dt.datetime(),
            lambda d: FieldsMapper(d).try_map_dtype(TemporalClosures().naive),
        ),
        UnaryNode(
            "dt.convert_time_zone",
            lambda x: x.dt.convert_time_zone("UTC"),
            lambda d: FieldsMapper(d).try_map_dtype(TemporalClosures().convert_time_zone("UTC")),
        ),
        UnaryNode(
            "dt.combine",
            lambda x: x.dt.combine(datetime.time(1), "ms"),
            lambda d: FieldsMapper(d).try_map_dtype(TemporalClosures().combine("ms")),
        ),
        UnaryNode(
            "struct.rename_fields",
            lambda x: x.struct.rename_fields(["p", "q"]),
            lambda d: FieldsMapper(d).map_dtype(StructClosures().rename_fields(("p", "q"))),
        ),
        UnaryNode(
            "struct.suffix_fields",
            lambda x: x.name.suffix_fields("_s"),
            lambda d: FieldsMapper(d).try_map_dtype(StructClosures().suffix_fields("_s")),
        ),
        UnaryNode(
            "list.to_struct",
            lambda x: x.list.to_struct(fields=["a", "b"]),
            lambda d: FieldsMapper(d).try_map_dtype(ListClosures().to_struct(("a", "b"))),
        ),
        UnaryNode(
            "arr.slice",
            lambda x: x.arr.slice(1, 5, as_array=True),
            lambda d: FieldsMapper(d).ensure_is_array().try_map_dtype(ArrayClosures().slice(1, 5)),
        ),
        UnaryNode(
            "arr.to_struct",
            lambda x: x.arr.to_struct(),
            lambda d: (
                FieldsMapper(d).ensure_is_array().try_map_dtype(ArrayClosures().to_struct(None))
            ),
        ),
        UnaryNode("sum", lambda x: x.sum(), AggFields().sum, scalar=True),
        UnaryNode("mean", lambda x: x.mean(), AggFields().mean, scalar=True),
        UnaryNode("median", lambda x: x.median(), AggFields().mean, scalar=True),
        UnaryNode("std", lambda x: x.std(), AggFields().mean, scalar=True),
        UnaryNode("var", lambda x: x.var(), AggFields().var, scalar=True),
        UnaryNode("max", lambda x: x.max(), AggFields().same, scalar=True),
        UnaryNode("first", lambda x: x.first(), AggFields().same, scalar=True),
        UnaryNode("n_unique", lambda x: x.n_unique(), AggFields().count, scalar=True),
        UnaryNode("count", lambda x: x.count(), AggFields().count, scalar=True),
        UnaryNode("implode", lambda x: x.implode(), AggFields().implode, scalar=True),
        UnaryNode("explode", lambda x: x.explode(), ExprFields().explode),
        UnaryNode("sort", lambda x: x.sort(), lambda d: FieldsMapper(d).with_same_dtype()),
        UnaryNode("head", lambda x: x.head(2), lambda d: FieldsMapper(d).with_same_dtype()),
        UnaryNode(
            "filter", lambda x: x.filter(pl.lit(True)), lambda d: FieldsMapper(d).with_same_dtype()
        ),
        UnaryNode("gather", lambda x: x.gather(0), lambda d: FieldsMapper(d).with_same_dtype()),
        UnaryNode(
            "cast",
            lambda x: x.cast(pl.Int16, strict=False),
            lambda d: ExprFields().cast(d, DType("Int16")),
        ),
    )
    PAIRS: ClassVar[tuple[PairNode, ...]] = (
        PairNode("coalesce", pl.coalesce, PlannedPair().coalesce),
        PairNode("max_horizontal", pl.max_horizontal, PlannedPair().max_horizontal),
        PairNode("sum_horizontal", pl.sum_horizontal, PlannedPair().sum_horizontal),
        PairNode(
            "mean_horizontal",
            pl.mean_horizontal,
            lambda a, b: FunctionClosures().mean_horizontal(
                FieldsMapper(a.dtype, b.dtype).map_to_supertype(),
            ),
        ),
        PairNode(
            "concat_list",
            pl.concat_list,
            lambda a, b: FieldsMapper(a.dtype, b.dtype).map_to_list_supertype(),
        ),
        PairNode(
            "pow",
            lambda a, b: a.pow(b),
            lambda a, b: FieldsMapper(a.dtype, b.dtype).pow_dtype(),
        ),
        PairNode(
            "log",
            lambda a, b: a.log(b),
            lambda a, b: FieldsMapper(a.dtype, b.dtype).log_dtype(),
        ),
        PairNode(
            "rolling_cov",
            lambda a, b: pl.rolling_cov(a, b, window_size=2),
            lambda a, b: FieldsMapper(a.dtype, b.dtype).try_map_dtypes(FunctionClosures().corr_cov),
        ),
        PairNode(
            "replace_strict",
            lambda a, b: a.replace_strict(b, b, default=a),
            lambda a, b: FieldsMapper(a.dtype, b.dtype, b.dtype, a.dtype).replace_dtype(None),
        ),
        PairNode(
            "linear_spaces",
            lambda a, b: pl.linear_spaces(a, b, 3),
            lambda a, b: DType(
                "List",
                (FieldsMapper(a.dtype, b.dtype).try_map_dtypes(FunctionClosures().linear_space),),
            ),
        ),
        PairNode(
            "when",
            lambda a, b: pl.when(pl.lit(True)).then(a).otherwise(b),
            PlannedPair().ternary,
        ),
        *(
            PairNode(f"binary {op.value}", build, PlannedPair().binary(op))
            for op, build in (
                (Operator.PLUS, lambda a, b: a + b),
                (Operator.MINUS, lambda a, b: a - b),
                (Operator.MULTIPLY, lambda a, b: a * b),
                (Operator.TRUE_DIVIDE, lambda a, b: a / b),
                (Operator.FLOOR_DIVIDE, lambda a, b: a // b),
                (Operator.MODULUS, lambda a, b: a % b),
                (Operator.AND, lambda a, b: a & b),
                (Operator.OR, lambda a, b: a | b),
                (Operator.XOR, lambda a, b: a ^ b),
                (Operator.EQ, lambda a, b: a == b),
                (Operator.LT, lambda a, b: a < b),
                (Operator.EQ_VALIDITY, lambda a, b: a.eq_missing(b)),
            )
        ),
    )

    @classmethod
    def literal_pairs(cls) -> list[PairNode]:
        """Give the nodes of two inputs that a test runs with a literal.

        They are the binary operators, `when` and `coalesce`.

        Returns:
            The nodes.

        Example:
            >>> [node.name for node in Nodes.literal_pairs()][:2]
            ['coalesce', 'when']
        """
        names = ("binary", "when", "coalesce")
        return [node for node in cls.PAIRS if node.name.startswith(names)]


class TestTypeFunctions:
    """Replay the type functions against the planner of polars, node by node."""

    @pytest.mark.parametrize("node", Nodes.UNARY, ids=lambda node: node.name)
    def test_node_of_one_column_matches_the_planner(self, node: UnaryNode) -> None:
        """The node over a column of each type has the type of its rule, in `select` and `agg`."""
        problems = []
        for dtype in RuleKinds().dtypes():
            schema = {"x": dtype}
            expression = node.build(pl.col("x"))
            column = DType.of(dtype)
            answers = {
                "select": (
                    Planner().select(schema, expression),
                    RuleSchema().select(node.rule, column),
                ),
                "agg": (
                    Planner().agg(schema, expression),
                    RuleSchema().agg(node.rule, column, scalar=node.scalar),
                ),
            }
            for context, (answer, rule) in answers.items():
                if answer != rule:
                    problems.append(f"{context} {dtype!r}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("node", Nodes.PAIRS, ids=lambda node: node.name)
    def test_node_of_two_columns_matches_the_planner(self, node: PairNode) -> None:
        """The node over two columns of each pair of types has the planner type of its rule."""
        problems = []
        kinds = RuleKinds().dtypes()
        for left in kinds:
            for right in kinds:
                schema = {"x0": left, "x1": right}
                expression = node.build(pl.col("x0"), pl.col("x1"))
                inputs = (Operand.column(DType.of(left)), Operand.column(DType.of(right)))
                answer = Planner().select(schema, expression)
                rule = RuleSchema().select(node.rule, *inputs)
                if answer != rule:
                    problems.append(f"{left!r} | {right!r}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("node", Nodes.PAIRS, ids=lambda node: node.name)
    def test_node_of_two_columns_in_agg_is_a_list_of_its_type(self, node: PairNode) -> None:
        """In `agg`, the node over two columns of one type shows a list of its type in `select`."""
        problems = []
        for dtype in RuleKinds().dtypes():
            schema = {"x0": dtype, "x1": dtype}
            expression = node.build(pl.col("x0"), pl.col("x1"))
            inputs = (Operand.column(DType.of(dtype)), Operand.column(DType.of(dtype)))
            answer = Planner().agg(schema, expression)
            rule = RuleSchema().agg(node.rule, *inputs, scalar=node.scalar)
            if answer != rule:
                problems.append(f"{dtype!r}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("value", [*RuleKinds.LITERALS, True, None], ids=repr)
    @pytest.mark.parametrize("node", Nodes.literal_pairs(), ids=lambda node: node.name)
    def test_node_with_a_literal_matches_the_planner(
        self,
        node: PairNode,
        value: int | float | str | None,
    ) -> None:
        """A binary operator, `when` and `coalesce` of a column and a literal have the rule type.

        The literal is on either side.
        """
        literal = Operand.of_value(value)
        problems = []
        for dtype in RuleKinds().dtypes():
            column = Operand.column(DType.of(dtype))
            orders = {
                "column first": (node.build(pl.col("x"), pl.lit(value)), (column, literal)),
                "literal first": (node.build(pl.lit(value), pl.col("x")), (literal, column)),
            }
            for order, (expression, inputs) in orders.items():
                answer = Planner().select({"x": dtype}, expression)
                rule = RuleSchema().select(node.rule, *inputs)
                if answer != rule:
                    problems.append(f"{order} {dtype!r}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("node", Nodes.literal_pairs(), ids=lambda node: node.name)
    def test_node_with_an_unknown_input_matches_the_planner(self, node: PairNode) -> None:
        """A node beside an input of the planner type `Unknown(Any)` has the type of its rule.

        `when` of a `Date` and an integer literal has that planner type. The
        other input is a column of each type, or a literal.
        """
        unknown = pl.when(pl.col("c")).then(pl.col("d")).otherwise(1)
        any_input = Operand.column(DType.unknown("any"))
        values = [*RuleKinds.LITERALS, True, None]
        others = [(pl.lit(value), Operand.of_value(value), pl.Int8()) for value in values]
        kinds = RuleKinds().dtypes()
        columns = [(pl.col("x"), Operand.column(DType.of(dtype)), dtype) for dtype in kinds]
        problems = []
        for expression, operand, dtype in others + columns:
            schema = {"c": pl.Boolean(), "d": pl.Date(), "x": dtype}
            orders = {
                "unknown first": (node.build(unknown, expression), (any_input, operand)),
                "unknown second": (node.build(expression, unknown), (operand, any_input)),
            }
            for order, (built, inputs) in orders.items():
                answer = Planner().select(schema, built)
                rule = RuleSchema().select(node.rule, *inputs)
                if answer != rule:
                    case = f"{order} {expression} {dtype!r}"
                    problems.append(f"{case}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("value", [*RuleKinds.LITERALS, True, None], ids=repr)
    def test_literal_matches_the_planner(self, value: int | float | str | None) -> None:
        """A literal alone has the type that it takes alone, in `select` and in `agg`."""
        literal = Operand.of_value(value)
        expression = pl.lit(value)
        answers = {
            "select": (
                Planner().select({"x": pl.Int8()}, expression),
                RuleSchema().select(ExprFields().literal, literal),
            ),
            "agg": (
                Planner().agg({"x": pl.Int8()}, expression),
                RuleSchema().agg(ExprFields().literal, literal, scalar=True),
            ),
        }
        problems = [
            f"{context}: polars {answer!r}, rule {rule!r}"
            for context, (answer, rule) in answers.items()
            if answer != rule
        ]
        assert not problems, "\n".join(problems)

    def test_length_and_over_match_the_planner(self) -> None:
        """`pl.len()` gives `UInt32`, and `over` and `rolling` wrap a node that is not one value."""
        int8 = DType("Int8")
        cases = {
            "len": (pl.len(), ExprFields().length()),
            "sum over": (
                pl.col("x").sum().over("g"),
                ExprFields().over(AggFields().sum(int8), join=False, scalar=True),
            ),
            "abs over join": (
                pl.col("x").abs().over("g", mapping_strategy="join"),
                ExprFields().over(int8, join=True, scalar=False),
            ),
            "sum over join": (
                pl.col("x").sum().over("g", mapping_strategy="join"),
                ExprFields().over(AggFields().sum(int8), join=True, scalar=True),
            ),
            "sum rolling": (
                pl.col("x").sum().rolling(index_column="g", period="2i"),
                ExprFields().rolling(AggFields().sum(int8), scalar=True),
            ),
            "abs rolling": (
                pl.col("x").abs().rolling(index_column="g", period="2i"),
                ExprFields().rolling(int8, scalar=False),
            ),
        }
        frame = {"x": pl.Int8(), "g": pl.Int8()}
        problems = []
        for name, (expression, rule) in cases.items():
            answer = Planner().select(frame, expression)
            if answer != ProjectionSchema().select(rule):
                problems.append(f"{name}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    def test_datetime_ranges_match_the_planner(self) -> None:
        """`datetime_ranges` of two moments gives a list of the `Datetime` of their supertype.

        The cast pass of a range of moments refuses a start or an end that is
        not a `Date` or a `Datetime`, so the pairs here are moments.
        """
        dtypes = RuleKinds().dtypes()
        moments = [dtype for dtype in dtypes if isinstance(dtype, pl.Date | pl.Datetime)]
        problems = []
        for left in moments:
            for right in moments:
                schema = {"x0": left, "x1": right}
                expression = pl.datetime_ranges(pl.col("x0"), pl.col("x1"), "1d")
                answer = Planner().select(schema, expression)
                mapper = FieldsMapper(DType.of(left), DType.of(right))
                rule = RuleSchema().select(
                    lambda mapper=mapper: DType(
                        "List",
                        (mapper.map_to_datetime_range_dtype(None, None),),
                    ),
                )
                if answer != rule:
                    problems.append(f"{left!r} | {right!r}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)


class TypeFunctionTrees:
    """Hold trees that reach paths of the type functions which the node samples do not reach.

    These trees reach an option of a node, as the fields that `hist` adds or a
    dimension of `reshape`, an input of a nested or a narrow type, and a
    selector of fields by place. Polars writes each leaf of fields as strict
    from Python, so a tree read from its serialized form reaches a leaf that
    is not strict. Each tree runs over the frame of `TreeFrame`, with the
    columns that it adds.

    Example:
        >>> "hist with a breakpoint" in TypeFunctionTrees.TREES
        True
    """

    PAIR: ClassVar[pl.Series] = pl.Series([{"a": 1, "b": "x"}] * 5)
    FLOATS: ClassVar[pl.Series] = pl.Series([[1.5]] * 5, dtype=pl.List(pl.Float32))
    SPANS: ClassVar[pl.Series] = pl.Series([[datetime.timedelta(hours=1)]] * 5)
    NESTED_SPANS: ClassVar[pl.Series] = pl.Series([[[datetime.timedelta(hours=1)]]] * 5)
    TREES: ClassVar[dict[str, tuple[dict[str, pl.Series], pl.Expr]]] = {
        "linear_space of Float32": (
            {},
            pl.linear_space(pl.lit(0.0, dtype=pl.Float32), pl.lit(1.0, dtype=pl.Float32), 3),
        ),
        "linear_spaces as arrays": ({}, pl.linear_spaces(0, 1, 3, as_array=True)),
        "list.mean of Float32": ({"f": FLOATS}, pl.col("f").list.mean()),
        "list.median of durations": ({"d": SPANS}, pl.col("d").list.median()),
        "list.var of lists of durations": ({"ll": NESTED_SPANS}, pl.col("ll").list.var()),
        "rolling_var of lists of durations": ({"ll": NESTED_SPANS}, pl.col("ll").rolling_var(2)),
        "reshape with -1 in a later place": ({}, pl.col("x").reshape((5, -1))),  # type: ignore[invalid-call-shape]
        "reshape into no row": ({}, pl.col("x").reshape((0, -1))),  # type: ignore[invalid-call-shape]
        "hist with a breakpoint and a category": (
            {},
            pl.col("x").hist(bin_count=2, include_breakpoint=True, include_category=True),
        ),
        "hist with a breakpoint": ({}, pl.col("x").hist(bin_count=2, include_breakpoint=True)),
        "hist with a category": ({}, pl.col("x").hist(bin_count=2, include_category=True)),
        "reinterpret of bytes as text": (
            {"b": TreeColumns.BYTES},
            pl.col("b").bin.reinterpret(dtype=pl.String),  # type: ignore[invalid-call-shape]
        ),
        "field by place": ({"p": PAIR}, pl.col("p").struct[1]),
        "field by a negative place": ({"p": PAIR}, pl.col("p").struct[-1]),
        "field by a missing place": ({"p": PAIR}, pl.col("p").struct[5]),
        "fields by places that are not strict": (
            {"p": PAIR},
            pl.Expr.deserialize(
                io.StringIO(
                    '{"Function":{"input":[{"Column":"p"}],"function":{"StructExpr":'
                    '{"SelectFields":{"ByIndex":{"indices":[0,5],"strict":false}}}}}}'
                ),
                format="json",
            ),
        ),
        "fields by one place twice": (
            {"p": PAIR},
            pl.Expr.deserialize(
                io.StringIO(
                    '{"Function":{"input":[{"Column":"p"}],"function":{"StructExpr":'
                    '{"SelectFields":{"ByIndex":{"indices":[0,0],"strict":true}}}}}}'
                ),
                format="json",
            ),
        ),
        "fields by names that are not strict": (
            {"p": PAIR},
            pl.Expr.deserialize(
                io.StringIO(
                    '{"Function":{"input":[{"Column":"p"}],"function":{"StructExpr":'
                    '{"SelectFields":{"ByName":{"names":["a","zz"],"strict":false}}}}}}'
                ),
                format="json",
            ),
        ),
    }


@pytest.mark.parametrize("name", sorted(TypeFunctionTrees.TREES))
def test_a_tree_reaches_a_path_of_the_type_functions(name: str) -> None:
    """The four answers of the rules match polars, for a tree that reaches a type function."""
    columns, expression = TypeFunctionTrees.TREES[name]
    frame = TreeFrame(columns)
    assert frame.rules(expression) == frame.polars(expression)
