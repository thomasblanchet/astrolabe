"""Check the cast pass rules of `astrolabe._rules.cast_pass` against the plans of polars.

The rules give the changes that one visit of the cast pass makes to the
inputs of a node, then the result of the visits of the cast nodes that it
puts in (`SettledCasts`). The plan of polars after the cast pass shows the
same changes: the plan traverser `LazyFrame._ldf.visit()` gives each node,
each cast with its option, each literal with its type, and each `implode`.
The traverser runs with no optimization, so the plan is the plan that the
conversion leaves. Each test compares the plan of the root node with the
rules:

- The binary operators, on every pair of the types of `RuleKinds`, on a
  column with a literal in either order, and on an input of the planner
  type `Unknown`. A comparison with a literal can become a new node.
- `when`, `fill_null`, `coalesce` with two and three inputs,
  `max_horizontal` and `index_of`, which cast their inputs to one type, and
  `clip`, which has no casting rule.
- `is_in` and `list.contains`, with the `implode` of a collection of the
  same level, and with literals with no data type. From polars 2.0.0, the
  node can hold a cast of its values that runs with the kernel. The
  traverser cannot show such a node, and its error names the target of the
  cast, which the test compares (`NeedleCast`).
- The visit of a strict cast, which removes or relaxes it, and the fold of
  a cast of a literal.

A refusal of the rules is a refusal of `collect_schema()`, so the plan does
not exist there, and the rules must name the class of the error that polars
raises. A null literal in `is_in` of a literal with no data type
makes polars loop with no end, so those plans run in a subprocess.

Each test takes one operator or one function, and its assertion names every
case that fails.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import itertools
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- a hang of polars needs a process
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

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
    CastNodeVisit,
    CastOption,
    FunctionCasts,
    IsInCasts,
    Namespace,
    RootKind,
    SettledCasts,
    Step,
    StepKind,
    TernaryCasts,
    Visit,
)
from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import Stop, UndecidedError
from astrolabe._rules.operators import Operator
from astrolabe._rules.options import CastingRules
from astrolabe._rules.supertype import SupertypeFlag
from astrolabe._rules.type_functions import ExprFields, FieldsMapper, Operand
from astrolabe._rules.releases import PolarsRelease

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class InputShape:
    """Hold the changes above one input of a node: its steps from the leaf up, and its literal type.

    A step is its kind, its type and its option. An `implode` shows no type
    in the plan, so its type is `None`. `literal` is the type of a literal
    input, and `None` for any other input.

    Example:
        >>> InputShape((("cast", DType("Int16"), 1),), None)
        InputShape(steps=(('cast', Int16, 1),), literal=None)
    """

    steps: tuple[tuple[str, DType | None, int | None], ...]
    literal: DType | None


@dataclass(frozen=True)
class PlanShape:
    """Hold the root node of a plan: what it is, and the shape of each of its inputs.

    Example:
        >>> PlanShape("binary Plus", ())
        PlanShape(root='binary Plus', inputs=())
    """

    root: str
    inputs: tuple[InputShape, ...]


class NeedleCast:
    """Name the cast of its values that an `is_in` or a `contains` node holds, as polars shows it.

    From polars 2.0.0, the cast pass of `is_in` can leave a cast of the values
    in the function, which runs with the kernel. The plan traverser cannot
    show that function, and its error names the target, for example
    `is_in[needle: i8]`. The shape of the node is then the target alone.

    Example:
        >>> NeedleCast().rule(DType("Int8"))
        PlanShape(root='needle i8', inputs=())
    """

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:is_in|list\.contains|arr\.contains)\[needle: (.+)\]"
    )

    def read(self, error: NotImplementedError) -> PlanShape | None:
        """Give the shape that the error of the traverser names, or `None` for another error.

        Args:
            error: The error of the traverser.

        Returns:
            The shape.

        Example:
            >>> NeedleCast().read(NotImplementedError("is_in[needle: decimal[38,0]]"))
            PlanShape(root='needle decimal[38,0]', inputs=())
        """
        match = self.PATTERN.fullmatch(str(error))
        return None if match is None else PlanShape(f"needle {match.group(1)}", ())

    def rule(self, dtype: DType) -> PlanShape:
        """Give the shape of a node whose rules hold a cast of its values to a type.

        Args:
            dtype: The target of the cast.

        Returns:
            The shape.

        Example:
            >>> NeedleCast().rule(DType.dynamic_int(1))
            PlanShape(root='needle dyn int', inputs=())
        """
        if Is.unknown(dtype):
            return PlanShape(f"needle dyn {dtype.args[0]}", ())
        text = dtype.polars()._string_repr()  # ruff: ignore[private-member-access] -- the name polars shows
        return PlanShape(f"needle {text}", ())


class PlanReader:
    """Read the root node of the plan of a query, after the cast pass, with the plan traverser.

    The query is a `select` of one expression over a frame of given columns.
    The traverser `LazyFrame._ldf.visit()` builds the plan with the given
    optimizations, and none run here. A query that polars refuses gives how
    polars stops.

    Example:
        >>> schema = {"x0": pl.Int8(), "x1": pl.Int32()}
        >>> plan = PlanReader().read(schema, pl.col("x0") + pl.col("x1"))
        >>> plan.root, plan.inputs[0].steps
        ('binary Plus', (('cast', Int32, 1),))
    """

    FLAGS: ClassVar[pl.QueryOptFlags] = pl.QueryOptFlags.none()

    def read(self, schema: dict[str, pl.DataType], expression: pl.Expr) -> PlanShape | Stopped:
        """Give the shape of the root node of the plan.

        Args:
            schema: The columns of the frame.
            expression: The expression.

        Returns:
            The shape, or how polars stops where it refuses the query.

        Example:
            >>> PlanReader().read({"x": pl.String()}, pl.col("x") + 1)
            raises InvalidOperationError
        """
        view = self._view(schema, expression)
        if isinstance(view, Stopped):
            return view
        try:
            root = view.view_expression(view.view_current_node().expr[0].node)
        except NotImplementedError as error:
            needle = NeedleCast().read(error)
            if needle is None:
                raise
            return needle
        label, inputs = self._root(view, root)
        return PlanShape(label, tuple(self.input(view, index) for index in inputs))

    def root_input(
        self,
        schema: dict[str, pl.DataType],
        expression: pl.Expr,
    ) -> InputShape | Stopped:
        """Give the shape of the root node read as one input: the casts above its leaf.

        Args:
            schema: The columns of the frame.
            expression: The expression, for example a cast of a column.

        Returns:
            The shape, or how polars stops where it refuses the query.

        Example:
            >>> PlanReader().root_input({"x": pl.Int8()}, pl.col("x").cast(pl.Int8)).steps
            ()
        """
        view = self._view(schema, expression)
        if isinstance(view, Stopped):
            return view
        return self.input(view, view.view_current_node().expr[0].node)

    def _view(self, schema: dict[str, pl.DataType], expression: pl.Expr) -> Any:
        """Build the plan traverser of the query.

        Args:
            schema: The columns of the frame.
            expression: The expression.

        Returns:
            The traverser, or how polars stops where it refuses the query.

        Example:
            >>> PlanReader()._view({"x": pl.Int8()}, pl.col("x")) is not None
            True
        """
        query = pl.LazyFrame(schema=schema).select(expression.alias("o"))
        flags = self.FLAGS._pyoptflags  # ruff: ignore[private-member-access] -- for the traverser
        plan = query._ldf.with_optimizations(flags)  # ruff: ignore[private-member-access]
        return PolarsAnswer().result(plan.visit)

    def _root(self, view: Any, node: Any) -> tuple[str, tuple[int, ...]]:
        """Name the root node and give its inputs.

        Args:
            view: The traverser.
            node: The root node.

        Returns:
            The name, and the indices of the inputs that the comparison reads.

        Example:
            >>> plan = PlanReader()._view({"x": pl.Int8()}, pl.col("x"))
            >>> PlanReader()._root(plan, plan.view_expression(0))
            ('column', ())
        """
        kind = type(node).__name__
        if kind == "BinaryExpr":
            op = str(node.op).rsplit(".", 1)[-1]
            return f"binary {op}", (node.left, node.right)
        if kind == "Ternary":
            return self._ternary(view, node)
        if kind == "Function":
            return self._function(view, node)
        return kind.lower(), ()

    def _ternary(self, view: Any, node: Any) -> tuple[str, tuple[int, ...]]:
        """Name a `when`: the rewrite `when(x.is_not_null()).then(value)`, or a plain `when`.

        Args:
            view: The traverser.
            node: The `Ternary` node.

        Returns:
            The name, and the inputs `then` and `otherwise` of a plain `when`.

        Example:
            >>> plan = PlanReader()._view({"x": pl.UInt8()}, pl.col("x") < 300)
            >>> root = plan.view_expression(plan.view_current_node().expr[0].node)
            >>> PlanReader()._ternary(plan, root)
            ('when_not_null(True)', ())
        """
        predicate = view.view_expression(node.predicate)
        function = predicate.function_data[0] if type(predicate).__name__ == "Function" else ""
        if str(function).endswith("IsNotNull"):
            truthy = view.view_expression(node.truthy)
            return f"when_not_null({truthy.value})", ()
        return "ternary", (node.truthy, node.falsy)

    def _function(self, view: Any, node: Any) -> tuple[str, tuple[int, ...]]:
        """Name a function: a rewrite of a comparison, or a function of the query.

        Args:
            view: The traverser.
            node: The `Function` node.

        Returns:
            The name, and the inputs of a function of the query.

        Example:
            >>> plan = PlanReader()._view({"x": pl.Int8()}, pl.col("x").eq_missing(None))
            >>> root = plan.view_expression(plan.view_current_node().expr[0].node)
            >>> PlanReader()._function(plan, root)
            ('is_null', ())
        """
        name = str(node.function_data[0]).rsplit(".", 1)[-1]
        rewrites = {"IsNull": "is_null", "IsNotNull": "is_not_null", "IsBetween": "is_between"}
        if name in rewrites:
            return rewrites[name], ()
        if name == "repeat":
            value = view.view_expression(node.input[0]).value
            return ("null_boolean" if value is None else f"constant({value})"), ()
        return "function", tuple(node.input)

    def input(self, view: Any, index: int) -> InputShape:
        """Give the shape of one input: the casts and the `implode` above its leaf.

        Args:
            view: The traverser.
            index: The node of the input.

        Returns:
            The shape, with the steps from the leaf up.

        Example:
            >>> plan = PlanReader()._view({"x": pl.Int8()}, pl.col("x").cast(pl.Float32))
            >>> PlanReader().input(plan, plan.view_current_node().expr[0].node)
            InputShape(steps=(('cast', Float32, 0),), literal=None)
        """
        steps: list[tuple[str, DType | None, int | None]] = []
        node = view.view_expression(index)
        while True:
            kind = type(node).__name__
            if kind == "Cast":
                steps.append(("cast", DType.of(node.dtype).materialized(), node.options))
                node = view.view_expression(node.expr)
            elif kind == "Agg" and node.name == "implode":
                steps.append(("implode", None, None))
                node = view.view_expression(node.arguments[0])
            else:
                break
        literal = DType.of(node.dtype).materialized() if kind == "Literal" else None
        return InputShape(tuple(reversed(steps)), literal)


class RuleShape:
    """Give the shape that the rules give a node, in the form of `PlanShape`.

    Example:
        >>> visit = Visit(((Step.cast(DType("Int32"), CastOption.NON_STRICT),), ()))
        >>> RuleShape().node("binary Plus", visit, [Operand.column(DType("Int8"))] * 2).root
        'binary Plus'
    """

    def node(self, label: str, visit: Visit, operands: Sequence[Operand]) -> PlanShape:
        """Give the shape of a node whose root stays, or the new root of a rewrite.

        Args:
            label: The name of the node where it stays.
            visit: The settled visit.
            operands: The inputs before the visit.

        Returns:
            The shape.

        Example:
            >>> from astrolabe._rules.cast_pass import NewRoot
            >>> RuleShape().node("binary Lt", Visit(((), ()), NewRoot(RootKind.IS_NULL)), []).root
            'is_null'
        """
        if visit.root is not None:
            return PlanShape(repr(visit.root), ())
        pairs = zip(operands, visit.steps, strict=True)
        return PlanShape(label, tuple(itertools.starmap(self.input, pairs)))

    def binary(self, op: Operator, visit: Visit, operands: Sequence[Operand]) -> PlanShape:
        """Give the shape of a binary operator, with its inputs swapped by a rewrite.

        Args:
            op: The operator.
            visit: The settled visit.
            operands: The inputs before the visit.

        Returns:
            The shape.

        Example:
            >>> from astrolabe._rules.cast_pass import NewRoot
            >>> visit = Visit(((Step.literal(DType("UInt8")),), ()), NewRoot(RootKind.SWAPPED))
            >>> inputs = [Operand.of_value(1), Operand.column(DType("UInt8"))]
            >>> RuleShape().binary(Operator.LT, visit, inputs).root
            'binary Gt'
        """
        if visit.root is None or visit.root.kind is not RootKind.SWAPPED:
            return self.node(f"binary {op.value}", visit, operands)
        swapped = op.swapped()
        label = f"binary {swapped.value if swapped is not None else op.value}"
        inputs = (self.input(operands[1], visit.steps[1]), self.input(operands[0], visit.steps[0]))
        return PlanShape(label, inputs)

    def input(self, operand: Operand, steps: Sequence[Step]) -> InputShape:
        """Give the shape of one input: its casts and its `implode`, and the type of a literal.

        Args:
            operand: The input before the visit.
            steps: Its settled steps.

        Returns:
            The shape.

        Example:
            >>> RuleShape().input(Operand.of_value(300), ())
            InputShape(steps=(), literal=Int32)
        """
        shown: list[tuple[str, DType | None, int | None]] = []
        literal = operand.dtype.materialized() if operand.literal else None
        for step in steps:
            if step.kind is StepKind.LITERAL:
                literal = step.dtype.materialized()
            elif step.kind is StepKind.IMPLODE:
                shown.append(("implode", None, None))
            else:
                option = None if step.option is None else step.option.value
                shown.append(("cast", step.dtype.materialized(), option))
        return InputShape(tuple(shown), literal)


class Rules:
    """Run the rules of one node: the type function first, then the visit of the cast pass.

    A refusal of either gives how the rule says that polars stops, as polars gives no plan there.

    Example:
        >>> inputs = [Operand.column(DType("UInt8")), Operand.of_value(300)]
        >>> shape = Rules().binary(Operator.PLUS, inputs)
        >>> shape.inputs[0].steps, shape.inputs[1].literal
        ((('cast', UInt16, 1),), UInt16)
    """

    def binary(self, op: Operator, operands: Sequence[Operand]) -> PlanShape | Stopped:
        """Give the shape of a binary operator.

        Args:
            op: The operator.
            operands: The two inputs.

        Returns:
            The shape, or how the rules say that polars stops.

        Example:
            >>> inputs = [Operand.column(DType("Date")), Operand.of_value("a")]
            >>> Rules().binary(Operator.EQ, inputs)
            raises InvalidOperationError
        """
        left, right = operands[0], operands[1]
        return self._shape(
            lambda: ExprFields().binary(op, left, right),
            lambda: BinaryCasts().visit(op, left, right),
            lambda visit: RuleShape().binary(op, visit, operands),
            operands,
        )

    def ternary(self, operands: Sequence[Operand]) -> PlanShape | Stopped:
        """Give the shape of `when(c).then(a).otherwise(b)` for the inputs `a` and `b`.

        Args:
            operands: The inputs `then` and `otherwise`.

        Returns:
            The shape, or how the rules say that polars stops.

        Example:
            >>> inputs = [Operand.column(DType("Int8")), Operand.column(DType("Int16"))]
            >>> Rules().ternary(inputs).inputs[0]
            InputShape(steps=(('cast', Int16, 2),), literal=None)
        """
        truthy, falsy = operands[0], operands[1]
        return self._shape(
            lambda: ExprFields().ternary(truthy.dtype, falsy.dtype),
            lambda: TernaryCasts().visit(truthy, falsy),
            lambda visit: RuleShape().node("ternary", visit, operands),
            operands,
        )

    def function(
        self,
        field: Callable[[], DType],
        casts: Callable[[], Visit],
        operands: Sequence[Operand],
    ) -> PlanShape | Stopped:
        """Give the shape of a function.

        Args:
            field: The type function.
            casts: The visit of the cast pass.
            operands: The inputs.

        Returns:
            The shape, or how the rules say that polars stops.

        Example:
            >>> inputs = [Operand.column(DType("Int8")), Operand.column(DType("Int8"))]
            >>> Rules().function(lambda: DType("Int8"), lambda: Visit.unchanged(2), inputs).root
            'function'
        """
        return self._shape(
            field,
            casts,
            lambda visit: RuleShape().node("function", visit, operands),
            operands,
        )

    def _shape(
        self,
        field: Callable[[], DType],
        casts: Callable[[], Visit],
        shape: Callable[[Visit], PlanShape],
        operands: Sequence[Operand],
    ) -> PlanShape | Stopped:
        """Run the type function and the visit, and settle the visit.

        Args:
            field: The type function.
            casts: The visit of the cast pass.
            shape: The shape of the settled visit.
            operands: The inputs.

        Returns:
            The shape, or how a rule says that polars stops.

        Example:
            >>> shape = lambda visit: PlanShape("x", ())
            >>> visit = lambda: Visit.unchanged(1)
            >>> Rules()._shape(lambda: DType("Int8"), visit, shape, [Operand.of_value(1)])
            PlanShape(root='x', inputs=())
        """

        def settled() -> Visit:
            field()
            return SettledCasts().of(casts(), operands)

        visit = RuleAnswer().result(settled)
        return visit if isinstance(visit, Stopped) else shape(visit)


@dataclass(frozen=True)
class FunctionCase:
    """Hold a function of two inputs: how to build it from two expressions, and its rules.

    Example:
        >>> rules = FunctionRules().supertyping(CastingRules.cast_to_supertypes())
        >>> case = FunctionCase(lambda e: pl.coalesce(*e), rules)
        >>> case.rules([Operand.column(DType("Int8"))] * 2).root
        'function'
    """

    build: Callable[[Sequence[pl.Expr]], pl.Expr]
    rules: Callable[[Sequence[Operand]], PlanShape | Stopped]


class FunctionRules:
    """Give the rules of a function by its casting rule: a supertype, lossless, or none.

    Example:
        >>> FunctionRules().no_rule([Operand.column(DType("Int8")), Operand.of_value(1)]).inputs[1]
        InputShape(steps=(), literal=Int32)
    """

    def supertyping(
        self,
        casting: CastingRules,
    ) -> Callable[[Sequence[Operand]], PlanShape | Stopped]:
        """Give the rules of a function that casts its inputs to their supertype.

        The type function is `map_to_supertype`.

        Args:
            casting: The casting rule of the function, with the flags of its join.

        Returns:
            The rules.

        Example:
            >>> rules = FunctionRules().supertyping(CastingRules.cast_to_supertypes())
            >>> rules([Operand.column(DType("Date")), Operand.column(DType("Boolean"))])
            raises SchemaError
        """

        def rules(inputs: Sequence[Operand]) -> PlanShape | Stopped:
            mapper = FieldsMapper(*(operand.dtype for operand in inputs))
            casts = FunctionCasts(casting)
            return Rules().function(mapper.map_to_supertype, lambda: casts.visit(inputs), inputs)

        return rules

    def find(self, inputs: Sequence[Operand]) -> PlanShape | Stopped:
        """Give the rules of `str.find`, which checks that its first input is `String`.

        The type function is `with_dtype(UInt32)`, and the options join with
        the default flags.

        Args:
            inputs: The text and the pattern.

        Returns:
            The shape, or how the rules say that polars stops.

        Example:
            >>> FunctionRules().find([Operand.column(DType("Int8")), Operand.of_value("a")])
            raises InvalidOperationError
        """
        casts = FunctionCasts(CastingRules.cast_to_supertypes(), Namespace.STRING)
        return Rules().function(
            lambda: FieldsMapper(inputs[0].dtype).with_dtype(DType("UInt32")),
            lambda: casts.visit(inputs),
            inputs,
        )

    def lossless(self, inputs: Sequence[Operand]) -> PlanShape | Stopped:
        """Give the rules of `index_of`, which casts its element to the type of its column.

        The type function is `with_dtype(DType.idx())`.

        Args:
            inputs: The column and the element.

        Returns:
            The shape, or how the rules say that polars stops.

        Example:
            >>> inputs = [Operand.column(DType("Int8")), Operand.column(DType("Int16"))]
            >>> FunctionRules().lossless(inputs)
            raises InvalidOperationError
        """
        casts = FunctionCasts(CastingRules.first_arg_lossless())
        return Rules().function(
            lambda: FieldsMapper(inputs[0].dtype).with_dtype(DType.idx()),
            lambda: casts.visit(inputs),
            inputs,
        )

    def supertype_only(self, inputs: Sequence[Operand]) -> PlanShape | Stopped:
        """Give the rules of a function that types its output with the supertype and casts nothing.

        `max_horizontal` has these rules before polars 1.40.0.

        Args:
            inputs: The inputs.

        Returns:
            The shape, or how the rules say that polars stops.

        Example:
            >>> inputs = [Operand.column(DType("Int8")), Operand.column(DType("UInt8"))]
            >>> FunctionRules().supertype_only(inputs).inputs[0].steps
            ()
        """
        mapper = FieldsMapper(*(operand.dtype for operand in inputs))
        return Rules().function(
            mapper.map_to_supertype,
            lambda: Visit.unchanged(len(inputs)),
            inputs,
        )

    def no_rule(self, inputs: Sequence[Operand]) -> PlanShape | Stopped:
        """Give the rules of `clip`, which has no casting rule.

        The type function is `with_same_dtype`.

        Args:
            inputs: The inputs.

        Returns:
            The shape.

        Example:
            >>> FunctionRules().no_rule([Operand.column(DType("Int8"))] * 2).inputs[0].steps
            ()
        """
        return Rules().function(
            FieldsMapper(inputs[0].dtype).with_same_dtype,
            lambda: Visit.unchanged(len(inputs)),
            inputs,
        )


class Cases:
    """Give the operators, the functions and the inputs that the tests run.

    Example:
        >>> len(Cases.OPERATORS), len(Cases.VALUES)
        (17, 9)
    """

    OPERATORS: ClassVar[list[tuple[Operator, Callable[[pl.Expr, pl.Expr], pl.Expr]]]] = [
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
        (Operator.NOT_EQ, lambda a, b: a != b),
        (Operator.LT, lambda a, b: a < b),
        (Operator.LT_EQ, lambda a, b: a <= b),
        (Operator.GT, lambda a, b: a > b),
        (Operator.GT_EQ, lambda a, b: a >= b),
        (Operator.EQ_VALIDITY, lambda a, b: a.eq_missing(b)),
        (Operator.NOT_EQ_VALIDITY, lambda a, b: a.ne_missing(b)),
    ]
    OPERATOR_IDS: ClassVar[list[str]] = [op.value for op, _ in OPERATORS]
    VALUES: ClassVar[list[bool | int | float | str | None]] = [*RuleKinds.LITERALS, True, None]
    TEMPORAL: ClassVar[list[tuple[object, DType]]] = [
        (datetime.datetime(2020, 1, 1, 12), DType("Datetime", ("us", None))),
        (datetime.date(2020, 1, 1), DType("Date")),
        (datetime.timedelta(days=1), DType("Duration", ("us",))),
    ]
    DEFAULT: ClassVar[CastingRules] = CastingRules.cast_to_supertypes()
    NO_TEXT: ClassVar[CastingRules] = CastingRules.supertype(SupertypeFlag(0))

    def functions(self) -> dict[str, FunctionCase]:
        """Give each function of two inputs, by name.

        `fill_null` and `coalesce` join with the default flags, and
        `max_horizontal` joins with no cast of a number to text, from polars
        1.40.0 on, and casts nothing before. `index_of`
        casts its element to the type of its column, and refuses a cast that
        can lose a value. `clip` has no casting rule. `str.find` checks its
        namespace. `index_of` refuses an element that is not one value.

        Returns:
            The functions.

        Example:
            >>> sorted(Cases().functions())[:2]
            ['clip', 'coalesce']
        """
        rules = FunctionRules()
        joins = PolarsRelease.installed() >= PolarsRelease.parse("1.40.0")
        return {
            "fill_null": FunctionCase(
                lambda e: e[0].fill_null(e[1]),
                rules.supertyping(self.DEFAULT),
            ),
            "coalesce": FunctionCase(lambda e: pl.coalesce(*e), rules.supertyping(self.DEFAULT)),
            "max_horizontal": FunctionCase(
                lambda e: pl.max_horizontal(*e),
                rules.supertyping(self.NO_TEXT) if joins else rules.supertype_only,
            ),
            "index_of": FunctionCase(
                lambda e: e[0].index_of(self._one_value(e[1])),
                rules.lossless,
            ),
            "clip": FunctionCase(lambda e: e[0].clip(e[1]), rules.no_rule),
            "str.find": FunctionCase(lambda e: e[0].str.find(e[1]), rules.find),
        }

    def numbers(self) -> list[tuple[str, pl.Expr, Operand]]:
        """Give each number of a literal type, as a literal node and as a node that does not fold.

        `abs` keeps the literal type of its input, so the cast pass meets
        that type on a node that is not a literal.

        Returns:
            The name, the expression and the input of the rules of each number.

        Example:
            >>> name, expression, operand = Cases().numbers()[1]
            >>> name, operand.literal
            ('abs(1)', False)
        """
        numbers: list[tuple[str, pl.Expr, Operand]] = []
        for value in RuleKinds.LITERALS:
            if isinstance(value, str):
                continue
            literal = Operand.of_value(value)
            kept = Operand.column(literal.dtype)
            numbers.extend(
                [
                    (repr(value), pl.lit(value), literal),
                    (f"abs({value!r})", pl.lit(value).abs(), kept),
                ],
            )
        return numbers

    def _one_value(self, expression: pl.Expr) -> pl.Expr:
        """Give an expression of one value: a literal as it is, and the first value of a column.

        Args:
            expression: The expression.

        Returns:
            The expression of one value.

        Example:
            >>> Cases()._one_value(pl.col("x")).meta.is_literal()
            False
        """
        return expression if expression.meta.is_literal() else expression.first()


class TestBinaryCasts:
    """Replay the cast pass of the binary operators against the plans of polars."""

    @pytest.mark.parametrize(("op", "build"), Cases.OPERATORS, ids=Cases.OPERATOR_IDS)
    def test_columns_match_the_plan(
        self,
        op: Operator,
        build: Callable[[pl.Expr, pl.Expr], pl.Expr],
    ) -> None:
        """Each pair of columns casts as the rules say, and they refuse where polars refuses."""
        problems = []
        kinds = RuleKinds().dtypes()
        for left in kinds:
            for right in kinds:
                schema = {"x0": left, "x1": right}
                plan = PlanReader().read(schema, build(pl.col("x0"), pl.col("x1")))
                inputs = [Operand.column(DType.of(left)), Operand.column(DType.of(right))]
                rule = Rules().binary(op, inputs)
                if plan != rule:
                    problems.append(f"{left!r} | {right!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize(("op", "build"), Cases.OPERATORS, ids=Cases.OPERATOR_IDS)
    def test_literals_match_the_plan(
        self,
        op: Operator,
        build: Callable[[pl.Expr, pl.Expr], pl.Expr],
    ) -> None:
        """A column beside a literal, in either order, casts and rewrites as the rules say."""
        problems = []
        for dtype in RuleKinds().dtypes():
            column = Operand.column(DType.of(dtype))
            for value in Cases.VALUES:
                literal = Operand.of_value(value)
                orders = {
                    "column first": (build(pl.col("x"), pl.lit(value)), [column, literal]),
                    "literal first": (build(pl.lit(value), pl.col("x")), [literal, column]),
                }
                for order, (expression, inputs) in orders.items():
                    plan = PlanReader().read({"x": dtype}, expression)
                    rule = Rules().binary(op, inputs)
                    if plan != rule:
                        case = f"{order} {dtype!r} | {value!r}"
                        problems.append(f"{case}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize(("op", "build"), Cases.OPERATORS, ids=Cases.OPERATOR_IDS)
    def test_temporal_literals_match_the_plan(
        self,
        op: Operator,
        build: Callable[[pl.Expr, pl.Expr], pl.Expr],
    ) -> None:
        """A column beside a moment or a duration literal casts and rewrites as the rules say.

        A Python `datetime` is a literal `Datetime` in microseconds, a `date`
        a `Date`, and a `timedelta` a `Duration` in microseconds. A
        comparison of a finer `Datetime` with a coarser literal becomes
        `is_between` for `==`.
        """
        problems = []
        for dtype in RuleKinds().dtypes():
            column = Operand.column(DType.of(dtype))
            for value, literal_type in Cases.TEMPORAL:
                literal = Operand.typed_literal(value, literal_type)
                orders = {
                    "column first": (build(pl.col("x"), pl.lit(value)), [column, literal]),
                    "literal first": (build(pl.lit(value), pl.col("x")), [literal, column]),
                }
                for order, (expression, inputs) in orders.items():
                    plan = PlanReader().read({"x": dtype}, expression)
                    rule = Rules().binary(op, inputs)
                    if plan != rule:
                        case = f"{order} {dtype!r} | {value!r}"
                        problems.append(f"{case}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize(("op", "build"), Cases.OPERATORS, ids=Cases.OPERATOR_IDS)
    def test_numbers_of_a_literal_type_match_the_plan(
        self,
        op: Operator,
        build: Callable[[pl.Expr, pl.Expr], pl.Expr],
    ) -> None:
        """Two numbers of a literal type cast as the rules say, where a side does not fold.

        An integer literal type beside a float literal type casts to the
        float literal type, before the rewrite of a comparison with a literal.
        """
        numbers = Cases().numbers()
        problems = []
        for left_name, left, left_operand in numbers:
            for right_name, right, right_operand in numbers:
                if left_operand.literal and right_operand.literal:
                    continue
                plan = PlanReader().read({}, build(left, right))
                rule = Rules().binary(op, [left_operand, right_operand])
                if plan != rule:
                    problems.append(f"{left_name} | {right_name}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize(("op", "build"), Cases.OPERATORS, ids=Cases.OPERATOR_IDS)
    def test_input_of_an_unknown_type_matches_the_plan(
        self,
        op: Operator,
        build: Callable[[pl.Expr, pl.Expr], pl.Expr],
    ) -> None:
        """An input of the planner type `Unknown(Any)` casts as the rules say.

        The other input is a literal or a column. `when` of a `Date` and an
        integer literal has that planner type.
        """
        unknown = pl.when(pl.col("c")).then(pl.col("d")).otherwise(1)
        any_input = Operand.column(DType.unknown("any"))
        others = [(pl.lit(value), Operand.of_value(value), pl.Int8()) for value in Cases.VALUES]
        dtypes = RuleKinds().dtypes()
        columns = [(pl.col("x"), Operand.column(DType.of(dtype)), dtype) for dtype in dtypes]
        problems = []
        for expression, operand, dtype in others + columns:
            schema = {"c": pl.Boolean(), "d": pl.Date(), "x": dtype}
            orders = {
                "unknown first": (build(unknown, expression), [any_input, operand]),
                "unknown second": (build(expression, unknown), [operand, any_input]),
            }
            for order, (built, inputs) in orders.items():
                plan = PlanReader().read(schema, built)
                rule = Rules().binary(op, inputs)
                if plan != rule:
                    problems.append(f"{order} {expression} {dtype!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)


class TestFunctionCasts:
    """Replay the cast pass of `when` and of the functions that cast their inputs to one type."""

    def test_when_of_columns_matches_the_plan(self) -> None:
        """`when` casts `then` and `otherwise` to their supertype, and the policy relaxes them."""
        problems = []
        kinds = RuleKinds().dtypes()
        for left in kinds:
            for right in kinds:
                schema = {"c": pl.Boolean(), "x0": left, "x1": right}
                expression = pl.when(pl.col("c")).then(pl.col("x0")).otherwise(pl.col("x1"))
                plan = PlanReader().read(schema, expression)
                inputs = [Operand.column(DType.of(left)), Operand.column(DType.of(right))]
                rule = Rules().ternary(inputs)
                if plan != rule:
                    problems.append(f"{left!r} | {right!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    def test_when_with_literals_matches_the_plan(self) -> None:
        """`when` of a column and a literal, in either order, casts and folds as the rules say."""
        problems = []
        for dtype in RuleKinds().dtypes():
            column = Operand.column(DType.of(dtype))
            for value in Cases.VALUES:
                literal = Operand.of_value(value)
                orders = {
                    "column first": (pl.col("x"), pl.lit(value), [column, literal]),
                    "literal first": (pl.lit(value), pl.col("x"), [literal, column]),
                }
                for order, (then, otherwise, inputs) in orders.items():
                    expression = pl.when(pl.col("c")).then(then).otherwise(otherwise)
                    plan = PlanReader().read({"c": pl.Boolean(), "x": dtype}, expression)
                    rule = Rules().ternary(inputs)
                    if plan != rule:
                        case = f"{order} {dtype!r} | {value!r}"
                        problems.append(f"{case}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("name", sorted(Cases().functions()))
    def test_function_of_columns_matches_the_plan(self, name: str) -> None:
        """The function of two columns casts its inputs as its casting rule says."""
        case = Cases().functions()[name]
        problems = []
        kinds = RuleKinds().dtypes()
        for left in kinds:
            for right in kinds:
                schema = {"x0": left, "x1": right}
                plan = PlanReader().read(schema, case.build([pl.col("x0"), pl.col("x1")]))
                rule = case.rules([Operand.column(DType.of(left)), Operand.column(DType.of(right))])
                if plan != rule:
                    problems.append(f"{left!r} | {right!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("name", sorted(Cases().functions()))
    def test_function_with_literals_matches_the_plan(self, name: str) -> None:
        """The function of a column and a literal, either way round, casts as its rule says."""
        case = Cases().functions()[name]
        problems = []
        for dtype in RuleKinds().dtypes():
            column = Operand.column(DType.of(dtype))
            for value in Cases.VALUES:
                literal = Operand.of_value(value)
                orders = {
                    "column first": ([pl.col("x"), pl.lit(value)], [column, literal]),
                    "literal first": ([pl.lit(value), pl.col("x")], [literal, column]),
                }
                for order, (expressions, inputs) in orders.items():
                    plan = PlanReader().read({"x": dtype}, case.build(expressions))
                    rule = case.rules(inputs)
                    if plan != rule:
                        problems.append(
                            f"{order} {dtype!r} | {value!r}: polars {plan}, rule {rule}",
                        )
        assert not problems, "\n".join(problems)

    def test_coalesce_of_three_columns_matches_the_plan(self) -> None:
        """`coalesce` of three inputs folds the join with no change by a literal."""
        rules = FunctionRules().supertyping(Cases.DEFAULT)
        problems = []
        kinds = RuleKinds().dtypes()
        for left in kinds:
            for right in kinds:
                schema = {"x0": left, "x1": right}
                plan = PlanReader().read(schema, pl.coalesce("x0", "x1", "x0"))
                rule = rules([Operand.column(DType.of(dtype)) for dtype in (left, right, left)])
                if plan != rule:
                    problems.append(f"{left!r} | {right!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)


class IsolatedIsIn:
    """Plan `pl.lit(None).is_in(pl.lit(other))` in a subprocess: it plans, refuses or hangs.

    Example:
        >>> IsolatedIsIn().answer("a")
        'plans'
    """

    TIMEOUT: ClassVar[int] = 8
    CODE: ClassVar[str] = (
        "import polars as pl\n"
        "frame = pl.LazyFrame({{'x': [1]}})\n"
        "try:\n"
        "    frame.select(pl.lit(None).is_in(pl.lit({other!r}))).collect_schema()\n"
        "except BaseException:\n"
        "    print('refuses')\n"
        "else:\n"
        "    print('plans')\n"
    )

    def answer(self, other: bool | int | float | str | None) -> str:
        """Give the answer of polars.

        Args:
            other: The value of the collection literal.

        Returns:
            `plans`, `refuses` or `hangs`.

        Example:
            >>> IsolatedIsIn().answer(None)
            'plans'
        """
        code = self.CODE.format(other=other)
        try:
            run = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed
                [sys.executable, "-W", "ignore", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "hangs"
        return run.stdout.strip()


class TestIsInCasts:
    """Replay the cast pass of `is_in` and `list.contains` against the plans of polars."""

    @pytest.mark.parametrize("left", RuleKinds().dtypes(), ids=repr)
    def test_is_in_of_columns_matches_the_plan(self, left: pl.DataType) -> None:
        """`is_in` of two columns implodes a collection of the same level, and casts the sides."""
        problems = []
        for right in RuleKinds().dtypes():
            for collection in (right, pl.List(right)):
                schema = {"x0": left, "x1": collection}
                plan = PlanReader().read(schema, pl.col("x0").is_in(pl.col("x1")))
                inputs = [Operand.column(DType.of(left)), Operand.column(DType.of(collection))]
                rule = self._rule(IsInCasts.is_in(), inputs)
                if plan != rule:
                    problems.append(f"{left!r} | {collection!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("right", RuleKinds().dtypes(), ids=repr)
    def test_list_contains_of_columns_matches_the_plan(self, right: pl.DataType) -> None:
        """`list.contains` of a list and a column casts as its rule says, and never implodes.

        Polars 1.42 and older cannot show the plan of a list expression, so
        the test does not run there.
        """
        probe = pl.col("x0").list.contains(pl.col("x1"))
        try:
            PlanReader().read({"x0": pl.List(pl.Int8()), "x1": pl.Int8()}, probe)
        except NotImplementedError:
            pytest.skip("this release of polars cannot show the plan of a list expression")
        problems = []
        for inner in RuleKinds().dtypes():
            for collection in (pl.List(inner), inner):
                schema = {"x0": collection, "x1": right}
                plan = PlanReader().read(schema, pl.col("x0").list.contains(pl.col("x1")))
                inputs = [Operand.column(DType.of(collection)), Operand.column(DType.of(right))]
                rule = self._rule(IsInCasts.list_contains(), inputs, contains=True)
                if plan != rule:
                    problems.append(f"{collection!r} | {right!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("value", Cases.VALUES, ids=repr)
    def test_is_in_with_a_literal_matches_the_plan(
        self,
        value: bool | int | float | str | None,
    ) -> None:
        """`is_in` of a column and a literal, in either order, folds and casts as its rule says."""
        literal = Operand.of_value(value)
        problems = []
        for dtype in RuleKinds().dtypes():
            for collection in (dtype, pl.List(dtype)):
                column = Operand.column(DType.of(collection))
                cases = {
                    "literal in column": (pl.lit(value).is_in(pl.col("x")), [literal, column]),
                    "column in literal": (pl.col("x").is_in(pl.lit(value)), [column, literal]),
                }
                for order, (expression, inputs) in cases.items():
                    plan = PlanReader().read({"x": collection}, expression)
                    rule = self._rule(IsInCasts.is_in(), inputs)
                    if plan != rule:
                        problems.append(f"{order} {collection!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize(
        "value",
        [value for value in Cases.VALUES if value is not None],
        ids=repr,
    )
    def test_is_in_of_two_literals_matches_the_plan(self, value: bool | int | float | str) -> None:
        """`is_in` of a literal in a literal folds the implode and the casts as its rule says."""
        problems = []
        for other in Cases.VALUES:
            expression = pl.lit(value).is_in(pl.lit(other))
            plan = PlanReader().read({"x": pl.Int8()}, expression)
            rule = self._rule(IsInCasts.is_in(), [Operand.of_value(value), Operand.of_value(other)])
            if plan != rule:
                problems.append(f"{value!r} in {other!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    def test_is_in_of_a_null_literal_in_a_literal_hangs_where_the_rule_says(self) -> None:
        """A null literal in a literal hangs where the rule says, and plans as it says elsewhere.

        Each query runs in its own subprocess, and a query that does not end
        in a few seconds hangs.
        """
        isolated = IsolatedIsIn()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(Cases.VALUES)) as pool:
            answers = list(zip(Cases.VALUES, pool.map(isolated.answer, Cases.VALUES), strict=True))
        problems = []
        for other, answer in answers:
            rule = self._null_in(other)
            if answer != rule:
                problems.append(f"None in {other!r}: polars {answer}, rule {rule}")
        assert not problems, "\n".join(problems)

    def _null_in(self, other: bool | int | float | str | None) -> str:
        """Give what the rules say that `is_in` of a null literal in a literal does.

        Args:
            other: The value of the collection literal.

        Returns:
            `plans`, `refuses` or `hangs`.

        Example:
            >>> TestIsInCasts()._null_in(1), TestIsInCasts()._null_in("a")
            ('hangs', 'plans')
        """
        inputs = [Operand.of_value(None), Operand.of_value(other)]
        visit = RuleAnswer().result(lambda: IsInCasts.is_in().visit(inputs))
        if isinstance(visit, Stopped):
            return "hangs" if visit.stop is Stop.HANGS else "refuses"
        return "plans"

    def _rule(
        self,
        casts: IsInCasts,
        inputs: Sequence[Operand],
        *,
        contains: bool = False,
    ) -> PlanShape | Stopped:
        """Give the shape that the rules give `is_in` or `contains`.

        Args:
            casts: The rule of the cast pass.
            inputs: The inputs.
            contains: Whether the function is `list.contains`, which needs a list.

        Returns:
            The shape, or how the rules say that polars stops.

        Example:
            >>> inputs = [Operand.column(DType("Int8")), Operand.column(DType("Int8"))]
            >>> TestIsInCasts()._rule(IsInCasts.is_in(), inputs).inputs[1].steps
            (('implode', None, None),)
        """
        mapper = FieldsMapper(*(operand.dtype for operand in inputs))

        def field() -> DType:
            checked = mapper.ensure_is_list() if contains else mapper
            return checked.with_dtype(DType("Boolean"))

        shape = Rules().function(field, lambda: casts.visit(inputs), inputs)
        needle = None if isinstance(shape, Stopped) else casts.needle_cast(inputs)
        return shape if needle is None else NeedleCast().rule(needle)


class TestCastNodes:
    """Replay the visit of a cast node: the policy of a strict cast, and the fold of a literal."""

    @pytest.mark.parametrize("source", RuleKinds().dtypes(), ids=repr)
    def test_strict_cast_of_a_column_matches_the_plan(self, source: pl.DataType) -> None:
        """A strict cast of a column goes away, relaxes or stays strict, as the policy says."""
        column = Operand.column(DType.of(source))
        problems = []
        for target in RuleKinds().dtypes():
            plan = PlanReader().root_input({"x": source}, pl.col("x").cast(target))
            step = CastNodeVisit().visit(column, DType.of(target), CastOption.STRICT)
            rule = RuleShape().input(column, () if step is None else (step,))
            if plan != rule:
                problems.append(f"{source!r} to {target!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("strict", [True, False])
    @pytest.mark.parametrize("value", Cases.VALUES, ids=repr)
    def test_cast_of_a_literal_matches_the_plan(
        self,
        value: bool | int | float | str | None,
        strict: bool,
    ) -> None:
        """A cast of a literal folds into a literal of the type of the cast, or the planner refuses.

        A strict cast of an integer to `Categorical` keeps the value only
        where the process holds a category with that id, so the test skips
        the cases that the rule cannot decide.
        """
        literal = Operand.of_value(value)
        option = CastOption.STRICT if strict else CastOption.NON_STRICT
        problems = []
        for target in RuleKinds().dtypes():
            expression = pl.lit(value).cast(target, strict=strict)
            plan = PlanReader().root_input({"x": pl.Int8()}, expression)
            try:
                rule = self._literal_rule(literal, DType.of(target), option)
            except UndecidedError:
                continue
            if plan != rule:
                problems.append(f"{value!r} to {target!r}: polars {plan}, rule {rule}")
        assert not problems, "\n".join(problems)

    def _literal_rule(
        self,
        literal: Operand,
        target: DType,
        option: CastOption,
    ) -> InputShape | Stopped:
        """Give the shape that the visit of a cast of a literal gives, or how it refuses.

        Args:
            literal: The literal.
            target: The target of the cast.
            option: The option of the cast.

        Returns:
            The shape of the root, or how the rule says that polars stops.

        Example:
            >>> literal = Operand.of_value(300)
            >>> TestCastNodes()._literal_rule(literal, DType("Int8"), CastOption.STRICT)
            raises InvalidOperationError
        """
        step = RuleAnswer().result(lambda: CastNodeVisit().visit(literal, target, option))
        if isinstance(step, Stopped):
            return step
        return RuleShape().input(literal, () if step is None else (step,))


class CastPassTrees:
    """Hold trees that reach paths of the cast pass which the pairs of kinds do not reach.

    A list literal beside lists of another inner type keeps the type of the
    lists. A comparison rewrites a null literal or a literal out of range
    only where the cast of the input to the supertype keeps every value. An
    input of the planner type `Unknown` beside another changes nothing. A
    struct of no field beside a number, and `int_range` of a float type,
    reach their own arms. Each tree runs over the frame of `TreeFrame`, with
    the columns that it adds.

    Example:
        >>> "a far date beside nanoseconds" in CastPassTrees.TREES
        True
    """

    TREES: ClassVar[dict[str, tuple[dict[str, pl.Series], pl.Expr]]] = {
        "a list literal beside lists of another inner type": (
            {},
            pl.lit([1, 2]) == pl.col("l").cast(pl.List(pl.Int8)),
        ),
        "eq_missing of a null of a type that the input does not cast to": (
            {},
            pl.col("x").eq_missing(pl.lit(None, dtype=pl.UInt64)),
        ),
        "a far date beside nanoseconds": (
            {"m": TreeColumns.MOMENTS},
            pl.col("m").cast(pl.Datetime("ns")) > pl.lit(datetime.date(3000, 1, 1)),
        ),
        "two inputs of the type Unknown": (
            {},
            (pl.lit(1).cast(pl.Unknown) + pl.lit(2).cast(pl.Unknown)).abs(),
        ),
        "a node of a literal beside an input of the type Unknown": (
            {},
            pl.lit(1).abs() + pl.lit(2).cast(pl.Unknown),
        ),
        "a struct of no field beside a number": (
            {},
            pl.col("x") + pl.lit({}, dtype=pl.Struct({})),
        ),
        "int_range of a float type": ({}, pl.int_range(0, 3, dtype=pl.Float64)),  # type: ignore[arg-type]
    }


@pytest.mark.parametrize("name", sorted(CastPassTrees.TREES))
def test_a_tree_reaches_a_path_of_the_cast_pass(name: str) -> None:
    """The four answers of the rules match polars, for a tree that reaches one path of the pass."""
    columns, expression = CastPassTrees.TREES[name]
    frame = TreeFrame(columns)
    assert frame.rules(expression) == frame.polars(expression)
