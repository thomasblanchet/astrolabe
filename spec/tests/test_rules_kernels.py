"""Check the kernel rules of `astrolabe._rules.kernels` against the kernels of polars at run time.

A kernel rule takes the data that reaches a kernel and gives a data type or
a failure. Each test here reaches the kernel of polars with no cast pass in
front of it, and compares its answer with the rule:

- The arithmetic, the comparison and the bitwise operators of two series are
  methods of the native series of polars. These methods call the Rust kernels
  at once. The operators of a Python `Series` call the same methods, after
  checks and casts of their own in Python, so the tests call the methods.
- The other kernels run only inside an expression: the true and the floor
  division, `pow`, the comparisons that count nulls, the logical operators,
  the in-place path of `+`, `-` and `*`, and every kernel of one input. The
  planner checks an expression first, and the cast pass can cast its inputs.
  So each test reads the plan of `explain()`: where the planner refuses or
  the plan holds a cast, the kernel is not alone, and the test skips the
  pair. Every other pair reaches the kernel at the types of its columns.
- `sqrt`, `cbrt`, `exp`, `log1p` and `log` end the process over a struct
  with values. Those cells run in a subprocess.

Each pair runs on a sample of typical values and on series of only nulls,
on each side. The bitwise tests and the nested tests of two series also
run with one row on one side. A series of no row holds no value, and each
kernel of two series reads it as it reads a series of only nulls: those
tests also run with no row on a side, beside no row or one row. The
in-place path of `+`, `-` and `*` also runs with the cast pass off, as a
query can ask, where it meets two columns of other types (`UncastCheck`).
Where polars raises, the rule must name the class of the error that polars
raises.
A list in a sample holds two values: a list of one row whose only value is
null panics where it broadcasts, a fact of the values that `KernelFacts`
lists. Each test takes one operator and one left type, and its assertion
names every case that fails.
"""

from __future__ import annotations

import functools
import io
import itertools
import json
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

import polars as pl
import pytest

from spec.tests.rule_probes import (
    ApiGate,
    KernelFlags,
    KernelSamples,
    NestedKinds,
    PolarsAnswer,
    RuleAnswer,
    RuleKinds,
    Stopped,
    TemplateCall,
    TreeColumns,
    TreeFrame,
)
from astrolabe._rules.dtypes import DType
from astrolabe._rules.failures import Stop, UndecidedError
from astrolabe._rules.kernels.arithmetic import Pow, SeriesArithmetic
from astrolabe._rules.kernels.binary import OperatorKernel
from astrolabe._rules.kernels.comparison import CompareOp, Comparison
from astrolabe._rules.kernels.dispatch import Data, Dispatch, KernelFacts
from astrolabe._rules.kernels.fill import FillNull
from astrolabe._rules.releases import PolarsRelease
from astrolabe._rules.kernels.floats import FloatKernels, Log
from astrolabe._rules.kernels.layout import LayoutKernels
from astrolabe._rules.kernels.logic import BitOp, BitOps, BitwiseKernels
from astrolabe._rules.kernels.namespaces import FixedWidthParse, FormatFault, TextFormat
from astrolabe._rules.kernels.numeric import NumericKernels
from astrolabe._rules.operators import NumericOp, Operator
from astrolabe._rules.type_functions import FieldsMapper

if TYPE_CHECKING:
    from collections.abc import Callable

    from astrolabe._rules.kernels.dispatch import BinaryKernel, KernelFact


@dataclass(frozen=True)
class Case:
    """Hold the shape of the data on each side of a kernel of two inputs.

    Example:
        >>> Case(left_nulls=True, right_nulls=False).text()
        'left nulls, right typical'
    """

    left_nulls: bool
    right_nulls: bool
    left_one_row: bool = False
    right_one_row: bool = False

    CASES: ClassVar[tuple[tuple[bool, bool], ...]] = (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    )

    @classmethod
    def equal_lengths(cls) -> list[Case]:
        """Give each pair of samples, with three rows on both sides.

        Returns:
            The four cases.

        Example:
            >>> len(Case.equal_lengths())
            4
        """
        return list(itertools.starmap(cls, cls.CASES))

    @classmethod
    def every_length(cls) -> list[Case]:
        """Give each pair of samples, with three rows or one row on each side.

        Returns:
            The sixteen cases.

        Example:
            >>> Case.every_length()[-1].text()
            'left nulls in one row, right nulls in one row'
        """
        cases = cls.equal_lengths()
        cases += [cls(left, right, left_one_row=True) for left, right in cls.CASES]
        cases += [cls(left, right, right_one_row=True) for left, right in cls.CASES]
        both = {"left_one_row": True, "right_one_row": True}
        cases += [cls(left, right, **both) for left, right in cls.CASES]
        return cases

    def text(self) -> str:
        """Give a description of the case.

        Returns:
            The description.

        Example:
            >>> Case(False, True, left_one_row=True).text()
            'left typical in one row, right nulls'
        """
        left = "nulls" if self.left_nulls else "typical"
        left_rows = " in one row" if self.left_one_row else ""
        right = "nulls" if self.right_nulls else "typical"
        right_rows = " in one row" if self.right_one_row else ""
        return f"left {left}{left_rows}, right {right}{right_rows}"

    def data(self, left: pl.DataType, right: pl.DataType) -> tuple[Data, Data]:
        """Give the data of the rule for each side.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The data of the left side and of the right side, which a kernel rule reads as a pair.

        Example:
            >>> Case(True, False).data(pl.Int8(), pl.Int8())[0]
            Data(dtype=Int8, nulls=True, one_row=False)
        """
        return (
            Data(DType.of(left), nulls=self.left_nulls, one_row=self.left_one_row),
            Data(DType.of(right), nulls=self.right_nulls, one_row=self.right_one_row),
        )


class SeriesCalls:
    """Call a method of the native series of polars, which runs a Rust kernel of polars at once.

    `Series` keeps its native series, a `PySeries` of the Rust binding, in
    `_s`. Its methods `add`, `sub`, `mul`, `div`, `rem`, `eq`, `neq`, `lt`,
    `lt_eq`, `gt`, `gt_eq`, `bitand`, `bitor` and `bitxor` call the Rust
    operator of two series.

    Example:
        >>> SeriesCalls().dtype("add", pl.Series([1], dtype=pl.Int8), pl.Series([1.0]))
        Float64
    """

    def dtype(self, method: str, left: pl.Series, right: pl.Series) -> pl.DataType:
        """Give the data type of the output of the method.

        Args:
            method: The name of the method.
            left: The left series.
            right: The right series.

        Returns:
            The data type.

        Example:
            >>> SeriesCalls().dtype("lt", pl.Series([1]), pl.Series([2]))
            Boolean
        """
        native = getattr(left._s, method)(right._s)  # ruff: ignore[private-member-access]
        dtype: pl.DataType = native.dtype()
        return dtype


class PairCheck:
    """Compare a kernel of two series with its rule, for one left type and many right types.

    Example:
        >>> check = PairCheck("add", SeriesArithmetic(NumericOp.ADD))
        >>> check.problems(pl.Int8(), [pl.Float32()], Case.equal_lengths())
        []
    """

    EMPTY: ClassVar[tuple[tuple[int, int], ...]] = ((0, 0), (0, 1), (1, 0))

    def __init__(self, method: str, kernel: BinaryKernel) -> None:
        """Keep the method of polars and the rule.

        Args:
            method: The method of the native series.
            kernel: The rule.

        Example:
            >>> PairCheck("eq", Comparison(CompareOp.EQUAL))._method
            'eq'
        """
        self._method = method
        self._kernel = kernel
        self._samples = KernelSamples()

    def problems(
        self,
        left: pl.DataType,
        rights: list[pl.DataType],
        cases: list[Case],
    ) -> list[str]:
        """Give each case where polars and the rule differ.

        Args:
            left: The left type.
            rights: The right types.
            cases: The shapes of the data.

        Returns:
            A description of each case that differs.

        Example:
            >>> check = PairCheck("bitand", BitOps(BitOp.AND))
            >>> check.problems(pl.Int8(), [pl.Int16()], Case.every_length())
            []
        """
        problems = []
        for right in rights:
            for case in cases:
                answer = self._answer(left, right, case)
                rule = self._rule(left, right, case)
                if answer != rule:
                    cell = f"{left!r} {self._method} {right!r}, {case.text()}"
                    problems.append(f"{cell}: polars {answer}, rule {rule}")
        return problems

    def empty_problems(self, left: pl.DataType, rights: list[pl.DataType]) -> list[str]:
        """Give each case with a side of no row where polars and the rule differ.

        A series of no row holds no value, so the rule reads it as data of
        only nulls. The other side has no row or one row of typical values.

        Args:
            left: The left type.
            rights: The right types.

        Returns:
            A description of each case that differs.

        Example:
            >>> check = PairCheck("eq", Comparison(CompareOp.EQUAL))
            >>> check.empty_problems(pl.List(pl.Int64()), [pl.String()])
            []
        """
        problems = []
        for right in rights:
            for rows in self.EMPTY:
                problem = self._empty_problem(left, right, rows)
                if problem is not None:
                    problems.append(problem)
        return problems

    def _empty_problem(
        self,
        left: pl.DataType,
        right: pl.DataType,
        rows: tuple[int, int],
    ) -> str | None:
        """Give what differs for one pair of types with no row or one row on each side.

        Args:
            left: The left type.
            right: The right type.
            rows: The rows of each side.

        Returns:
            A description of the difference, or `None`.

        Example:
            >>> check = PairCheck("bitand", BitOps(BitOp.AND))
            >>> check._empty_problem(pl.Int64(), pl.String(), (0, 1))
        """
        sides = []
        datas = []
        for dtype, count in zip((left, right), rows, strict=True):
            sides.append(self._samples.series(dtype, nulls=False, one_row=True).head(count))
            datas.append(Data(DType.of(dtype), nulls=count == 0, one_row=count == 1))
        answer = PolarsAnswer().answer(lambda: SeriesCalls().dtype(self._method, *sides))
        rule = RuleAnswer().result(lambda: self._kernel.output(*datas))
        if answer == rule:
            return None
        return f"{left!r} {self._method} {right!r}, rows {rows}: polars {answer}, rule {rule}"

    def _answer(self, left: pl.DataType, right: pl.DataType, case: Case) -> DType | Stopped:
        """Give the answer of polars for one case.

        Args:
            left: The left type.
            right: The right type.
            case: The shape of the data.

        Returns:
            The answer.

        Example:
            >>> check = PairCheck("add", SeriesArithmetic(NumericOp.ADD))
            >>> check._answer(pl.Int8(), pl.Int16(), Case(False, False))
            Int16
        """
        one = self._samples.series(left, nulls=case.left_nulls, one_row=case.left_one_row)
        other = self._samples.series(right, nulls=case.right_nulls, one_row=case.right_one_row)
        return PolarsAnswer().answer(lambda: SeriesCalls().dtype(self._method, one, other))

    def _rule(self, left: pl.DataType, right: pl.DataType, case: Case) -> DType | Stopped:
        """Give the answer of the rule for one case.

        Args:
            left: The left type.
            right: The right type.
            case: The shape of the data.

        Returns:
            The answer.

        Example:
            >>> check = PairCheck("add", SeriesArithmetic(NumericOp.ADD))
            >>> check._rule(pl.Int8(), pl.Int8(), Case(False, False))
            Int8
        """
        one, other = case.data(left, right)
        return RuleAnswer().result(lambda: self._kernel.output(one, other))


class PlanCheck:
    """Say whether a kernel inside an expression sees its columns as they are.

    The planner checks an expression before it runs, and the cast pass can
    cast its inputs. Where the planner refuses the expression, the kernel
    never runs. Where the plan of `explain()` holds a cast, the kernel sees
    the cast data. In both cases a test cannot reach the kernel alone.

    Example:
        >>> PlanCheck().alone(pl.col("l") + pl.col("r"), {"l": pl.Int8(), "r": pl.Int16()})
        False
        >>> PlanCheck().alone(pl.col("l") + pl.col("r"), {"l": pl.Int8(), "r": pl.Int8()})
        True
    """

    def alone(self, expression: pl.Expr, schema: dict[str, pl.DataType]) -> bool:
        """Say whether the planner accepts the expression and casts no column.

        Args:
            expression: The expression.
            schema: The types of the columns.

        Returns:
            Whether the kernel sees the columns as they are.

        Example:
            >>> PlanCheck().alone(pl.col("x").sum(), {"x": pl.String()})
            False
        """
        frame = pl.LazyFrame(schema=schema)
        plan = PolarsAnswer().text(lambda: frame.select(expression).explain())
        return plan is not None and "cast(" not in plan


class ExpressionRule(Protocol):
    """Give the rule of the kernel that an expression of two columns runs."""

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The data of the column `l`.
            right: The data of the column `r`.
        """


@dataclass(frozen=True)
class OperatorRule:
    """Give the rule of a binary operator of an expression over two columns of rows.

    Example:
        >>> OperatorRule(Operator.TRUE_DIVIDE).output(Data(DType("Int8")), Data(DType("Int8")))
        Float64
    """

    operator: Operator

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The data of the left column.
            right: The data of the right column.

        Returns:
            The type of the output.

        Example:
            >>> OperatorRule(Operator.PLUS).output(Data(DType("Int32")), Data(DType("Date")))
            Int32
        """
        return OperatorKernel(owned=True).output(self.operator, left, right)


class ExpressionCheck:
    """Compare a kernel that an expression of two columns `l` and `r` runs with its rule.

    The check reads the plan of the expression first. Where the planner
    refuses the expression, or where the plan holds a cast, the kernel does
    not see the columns as they are, and the check skips the pair. A case
    where the rule says that the process ends does not run here.

    Example:
        >>> check = ExpressionCheck(pl.col("l") / pl.col("r"), OperatorRule(Operator.TRUE_DIVIDE))
        >>> check.problems(pl.Int8(), [pl.Int8(), pl.Float32()])
        []
    """

    def __init__(self, expression: pl.Expr, rule: ExpressionRule) -> None:
        """Keep the expression and the rule.

        Args:
            expression: The expression over the columns `l` and `r`.
            rule: The rule of its kernel.

        Example:
            >>> ExpressionCheck(pl.col("l"), OperatorRule(Operator.PLUS))._samples is not None
            True
        """
        self._expression = expression.alias("o")
        self._rule = rule
        self._samples = KernelSamples()

    def problems(self, left: pl.DataType, rights: list[pl.DataType]) -> list[str]:
        """Give each case where polars and the rule differ.

        Args:
            left: The type of `l`.
            rights: The types of `r`.

        Returns:
            A description of each case that differs.

        Example:
            >>> rule = OperatorRule(Operator.FLOOR_DIVIDE)
            >>> check = ExpressionCheck(pl.col("l") // pl.col("r"), rule)
            >>> check.problems(pl.Int64(), [pl.Duration("us")])
            []
        """
        problems = []
        for right in rights:
            if PlanCheck().alone(self._expression, {"l": left, "r": right}):
                problems += self._pair_problems(left, right)
        return problems

    def _pair_problems(self, left: pl.DataType, right: pl.DataType) -> list[str]:
        """Give each case of one pair of types where polars and the rule differ.

        A case where the rule says that the process ends does not run.

        Args:
            left: The type of `l`.
            right: The type of `r`.

        Returns:
            A description of each case that differs.

        Example:
            >>> check = ExpressionCheck(pl.col("l").log(pl.col("r")), LogRule())
            >>> check._pair_problems(pl.Struct({"a": pl.Int8()}), pl.Float64())
            []
        """
        problems = []
        for case in Case.equal_lengths():
            rule = self._rule_of(left, right, case)
            if isinstance(rule, Stopped) and rule.stop in OutOfProcess.STOPS:
                continue
            answer = self._answer(left, right, case)
            if answer != rule:
                cell = f"{left!r} and {right!r}, {case.text()}"
                problems.append(f"{cell}: polars {answer}, rule {rule}")
        return problems

    def _rule_of(self, left: pl.DataType, right: pl.DataType, case: Case) -> DType | Stopped:
        """Give the answer of the rule for one case.

        Args:
            left: The type of `l`.
            right: The type of `r`.
            case: The shape of the data.

        Returns:
            The answer.

        Example:
            >>> check = ExpressionCheck(pl.col("l"), OperatorRule(Operator.MODULUS))
            >>> check._rule_of(pl.Int32(), pl.Date(), Case(False, False))
            Int32
        """
        one, other = case.data(left, right)
        return RuleAnswer().result(lambda: self._rule.output(one, other))

    def _answer(self, left: pl.DataType, right: pl.DataType, case: Case) -> DType | Stopped:
        """Give the answer of polars for one case.

        Args:
            left: The type of `l`.
            right: The type of `r`.
            case: The shape of the data.

        Returns:
            The answer.

        Example:
            >>> check = ExpressionCheck(pl.col("l").pow(pl.col("r")), OperatorRule(Operator.PLUS))
            >>> check._answer(pl.Int8(), pl.Float32(), Case(False, False))
            Float32
        """
        one = self._samples.series(left, nulls=case.left_nulls, one_row=False).alias("l")
        other = self._samples.series(right, nulls=case.right_nulls, one_row=False).alias("r")
        frame = pl.DataFrame([one, other])
        return PolarsAnswer().answer(lambda: frame.select(self._expression).to_series().dtype)


class UncastCheck:
    """Compare the in-place path of `+`, `-` and `*` with its rule, with the cast pass off.

    With the cast pass on, polars casts two columns of other types to their
    supertype, so the in-place path meets two columns of one type. A query
    that turns the cast pass off gives the path the columns as they are, and
    the path then joins the two types itself, with no check: a missing
    supertype panics, and so does a supertype that is not a number. The check
    skips a pair that the planner refuses.

    Example:
        >>> UncastCheck("+").problems(pl.Int8(), [pl.UInt8()])
        []
    """

    OPERATORS: ClassVar[dict[str, Operator]] = {
        "+": Operator.PLUS,
        "-": Operator.MINUS,
        "*": Operator.MULTIPLY,
    }

    def __init__(self, operator: str) -> None:
        """Keep the operator.

        Args:
            operator: The text of the operator.

        Example:
            >>> UncastCheck("*")._operator
            <Operator.MULTIPLY: 'Multiply'>
        """
        self._operator = self.OPERATORS[operator]
        build = Kernels.OPERATORS[operator][0]
        self._expression = build(pl.col("l"), pl.col("r")).alias("o")
        self._samples = KernelSamples()

    def problems(self, left: pl.DataType, rights: list[pl.DataType]) -> list[str]:
        """Give each pair where polars and the rule differ.

        Args:
            left: The type of `l`.
            rights: The types of `r`.

        Returns:
            A description of each pair that differs.

        Example:
            >>> UncastCheck("-").problems(pl.Date(), [pl.Int8()])
            []
        """
        problems = []
        for right in rights:
            problem = self._problem(left, right)
            if problem is not None:
                problems.append(problem)
        return problems

    def _problem(self, left: pl.DataType, right: pl.DataType) -> str | None:
        """Give what differs for one pair of types, or `None` where they agree or polars plans none.

        Args:
            left: The type of `l`.
            right: The type of `r`.

        Returns:
            A description of the difference, or `None`.

        Example:
            >>> UncastCheck("+")._problem(pl.Int8(), pl.UInt8()) is None
            True
        """
        frame = pl.LazyFrame(schema={"l": left, "r": right}).select(self._expression)
        if isinstance(PolarsAnswer().answer(lambda: frame.collect_schema()["o"]), Stopped):
            return None
        answer = self._answer(left, right)
        data = (Data(DType.of(left)), Data(DType.of(right)))
        rule = RuleAnswer().result(lambda: OperatorKernel().output(self._operator, *data))
        return None if answer == rule else f"{left!r} and {right!r}: polars {answer}, rule {rule}"

    def _answer(self, left: pl.DataType, right: pl.DataType) -> DType | Stopped:
        """Give the answer of polars over typical values, with the cast pass off.

        Args:
            left: The type of `l`.
            right: The type of `r`.

        Returns:
            The answer.

        Example:
            >>> UncastCheck("+")._answer(pl.Int8(), pl.Decimal(10, 2))
            panics
        """
        one = self._samples.series(left, nulls=False, one_row=False).alias("l")
        other = self._samples.series(right, nulls=False, one_row=False).alias("r")
        query = pl.DataFrame([one, other]).lazy().select(self._expression)
        flags = KernelFlags().uncast()
        return PolarsAnswer().answer(lambda: query.collect(optimizations=flags)["o"].dtype)


@dataclass(frozen=True)
class PowRule:
    """Give the rule of `pow` of two columns.

    Example:
        >>> PowRule().output(Data(DType("UInt8")), Data(DType("Float16")))
        Float16
    """

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The data of the base.
            right: The data of the exponent.

        Returns:
            The type of the output.

        Example:
            >>> PowRule().output(Data(DType("Float64")), Data(DType("Int8")))
            Float64
        """
        return Pow().output(left, right)


class UnaryRule(Protocol):
    """Give the rule of the kernel of a function of one column: a dispatch, or a view of one."""

    @property
    def name(self) -> str:
        """Give the name of the kernel."""

    def output(self, data: Data) -> DType:
        """Give the type of the output.

        Args:
            data: The data of the column.
        """


@dataclass(frozen=True)
class LogOfBase:
    """Give the rule of `log` over one column, with the float literal of `log()` as its base.

    The literal reaches the kernel as one `Float64` value.

    Example:
        >>> LogOfBase().output(Data(DType("Int8")))
        Float64
    """

    name: str = "log"

    def output(self, data: Data) -> DType:
        """Give the type of the output.

        Args:
            data: The data of the column.

        Returns:
            The type.

        Example:
            >>> LogOfBase().output(Data(DType("Float32")))
            Float32
        """
        return Log().output([data, Data(DType("Float64"))], [])


@dataclass(frozen=True)
class LogRule:
    """Give the rule of `log` of two columns: a value and a base of the same length.

    Example:
        >>> LogRule().output(Data(DType("Int8")), Data(DType("Float16")))
        Float16
    """

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The data of the value.
            right: The data of the base.

        Returns:
            The type of the output.

        Example:
            >>> LogRule().output(Data(DType("Date")), Data(DType("Int8")))
            Float64
        """
        return Log().output([left, right], [])


class UnaryCheck:
    """Compare a kernel of one input that an expression runs with its dispatch.

    The check reads the plan of the expression first, and skips a type
    that the planner refuses or casts. A cell where the dispatch says that
    the process ends does not run here.

    Example:
        >>> UnaryCheck(pl.col("x").abs(), NumericKernels.ABS).problems([pl.Int8(), pl.String()])
        []
    """

    def __init__(self, expression: pl.Expr, dispatch: UnaryRule, *, grouped: bool = False) -> None:
        """Keep the expression and the dispatch.

        Args:
            expression: The expression over the column `x`.
            dispatch: The dispatch of its kernel.
            grouped: Whether the expression runs in `group_by().agg()`, over two
                groups, and not in a `select`.

        Example:
            >>> UnaryCheck(pl.col("x").sqrt(), FloatKernels.SQRT)._dispatch.name
            'sqrt'
        """
        self._expression = expression.alias("o")
        self._dispatch = dispatch
        self._samples = KernelSamples()
        self._grouped = grouped

    def problems(self, dtypes: list[pl.DataType]) -> list[str]:
        """Give each cell where polars and the dispatch differ.

        Args:
            dtypes: The types of `x`.

        Returns:
            A description of each cell that differs.

        Example:
            >>> check = UnaryCheck(pl.col("x").cum_sum(), NumericKernels.CUM_SUM)
            >>> check.problems([pl.Decimal(10, 2)])
            []
        """
        problems = []
        for dtype in dtypes:
            if PlanCheck().alone(self._expression, {"x": dtype}):
                problems += self._cell_problems(dtype)
        return problems

    def _cell_problems(self, dtype: pl.DataType) -> list[str]:
        """Give each sample of one type where polars and the dispatch differ.

        A sample where the dispatch says that the process ends does not run.

        Args:
            dtype: The type of `x`.

        Returns:
            A description of each sample that differs.

        Example:
            >>> check = UnaryCheck(pl.col("x").sqrt(), FloatKernels.SQRT)
            >>> check._cell_problems(pl.Struct({"a": pl.Int8()}))
            []
        """
        problems = []
        for nulls in (False, True):
            rule = self._rule_of(dtype, nulls=nulls)
            if isinstance(rule, Stopped) and rule.stop in OutOfProcess.STOPS:
                continue
            answer = self._answer(dtype, nulls=nulls)
            if answer != rule:
                problems.append(f"{dtype!r} with nulls={nulls}: polars {answer}, rule {rule}")
        return problems

    def _rule_of(self, dtype: pl.DataType, *, nulls: bool) -> DType | Stopped:
        """Give the answer of the dispatch for one cell.

        Args:
            dtype: The type of `x`.
            nulls: Whether every value is null.

        Returns:
            The answer.

        Example:
            >>> UnaryCheck(pl.col("x"), FloatKernels.EXP)._rule_of(pl.Date(), nulls=False)
            Float64
        """
        data = Data(DType.of(dtype), nulls=nulls)
        return RuleAnswer().result(lambda: self._dispatch.output(data))

    def _answer(self, dtype: pl.DataType, *, nulls: bool) -> DType | Stopped:
        """Give the answer of polars for one cell.

        Args:
            dtype: The type of `x`.
            nulls: Whether every value is null.

        Returns:
            The answer.

        Example:
            >>> UnaryCheck(pl.col("x").exp(), FloatKernels.EXP)._answer(pl.Date(), nulls=False)
            Float64
            >>> grouped = UnaryCheck(pl.col("x").sum(), NumericKernels.SUM_GROUPS, grouped=True)
            >>> grouped._answer(pl.Date(), nulls=False)
            Date
        """
        frame = self._samples.series(dtype, nulls=nulls, one_row=False).to_frame()
        if not self._grouped:
            return PolarsAnswer().answer(lambda: frame.select(self._expression).to_series().dtype)
        groups = frame.with_columns(g=pl.Series([1, 1, 2])).group_by("g", maintain_order=True)
        return PolarsAnswer().answer(lambda: groups.agg(self._expression)["o"].dtype)


class IsolatedSelect:
    """Run a function of one column in a subprocess, and give its answer or the way that it fails.

    A kernel that ends the process cannot run in the process of the tests.
    The subprocess builds the column, runs the function and shows the type.
    A return code that a signal sets means that the process ended, and a
    timeout means that it hangs.

    Example:
        >>> IsolatedSelect("abs").answer(pl.Int8(), [1, 2])
        Int8
    """

    TIMEOUT: ClassVar[int] = 60
    CODE: ClassVar[str] = (
        "import datetime, decimal\n"
        "import polars as pl\n"
        "dtype = eval({dtype!r}, vars(pl))\n"
        "values = eval({values!r}, {{'datetime': datetime, 'Decimal': decimal.Decimal}})\n"
        "frame = pl.Series('x', values, dtype=dtype).to_frame()\n"
        "try:\n"
        "    output = frame.select(pl.col('x').{function}()).to_series().dtype\n"
        "except BaseException as error:\n"
        "    name = type(error).__name__\n"
        "    print('panics' if name == 'PanicException' else 'raises ' + name)\n"
        "else:\n"
        "    print('dtype', repr(output))\n"
    )

    def __init__(self, function: str) -> None:
        """Keep the name of the function.

        Args:
            function: The name of a method of an expression with no argument, such as `sqrt`.

        Example:
            >>> IsolatedSelect("exp")._function
            'exp'
        """
        self._function = function

    def answer(self, dtype: pl.DataType, values: list[object]) -> DType | Stopped:
        """Give the answer of the function on a column.

        Args:
            dtype: The type of the column.
            values: The values of the column.

        Returns:
            The data type, or how the run stops.

        Example:
            >>> IsolatedSelect("sqrt").answer(pl.Struct({"a": pl.Int8()}), [None])
            Float64
        """
        code = self.CODE.format(dtype=repr(dtype), values=repr(values), function=self._function)
        try:
            run = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Stopped(Stop.HANGS)
        if run.returncode < 0:
            return Stopped(Stop.CRASHES)
        return self._read(run.stdout.strip())

    def _read(self, text: str) -> DType | Stopped:
        """Read the answer that the subprocess shows.

        Args:
            text: The text of the subprocess.

        Returns:
            The data type, or how the run stops.

        Example:
            >>> isolated = IsolatedSelect("sqrt")
            >>> isolated._read("dtype Float64"), isolated._read("raises ComputeError")
            (Float64, raises ComputeError)
        """
        if text == "panics":
            return Stopped(Stop.PANICS)
        if text.startswith("raises "):
            return Stopped(Stop.RAISES, text.removeprefix("raises "))
        dtype = eval(text.removeprefix("dtype "), vars(pl))  # ruff: ignore[suspicious-eval-usage]
        return DType.of(dtype)


class CrashCheck:
    """Compare the cells where a dispatch says that the process ends with a run in a subprocess.

    Each cell runs on typical values, in its own subprocess, and the
    subprocesses run in parallel.

    Example:
        >>> CrashCheck("abs", NumericKernels.ABS).cells([pl.Struct({"a": pl.Int8()})])
        []
    """

    WORKERS: ClassVar[int] = 8

    def __init__(self, function: str, dispatch: UnaryRule) -> None:
        """Keep the function and its dispatch.

        Args:
            function: The name of a method of an expression with no argument.
            dispatch: The dispatch of its kernel.

        Example:
            >>> CrashCheck("sqrt", FloatKernels.SQRT)._dispatch.name
            'sqrt'
        """
        self._isolated = IsolatedSelect(function)
        self._dispatch = dispatch
        self._samples = KernelSamples()

    def cells(self, dtypes: list[pl.DataType]) -> list[pl.DataType]:
        """Give the types over which the dispatch says that the process ends or hangs.

        Args:
            dtypes: The types to read.

        Returns:
            The types of those cells.

        Example:
            >>> struct = pl.Struct({"a": pl.Int8()})
            >>> CrashCheck("exp", FloatKernels.EXP).cells([pl.Int8(), struct])
            [Struct({'a': Int8})]
        """
        cells = []
        for dtype in dtypes:
            rule = self._rule_of(dtype)
            if isinstance(rule, Stopped) and rule.stop in OutOfProcess.STOPS:
                cells.append(dtype)
        return cells

    def problems(self, dtypes: list[pl.DataType]) -> list[str]:
        """Give each cell where the subprocess and the dispatch differ.

        Args:
            dtypes: The types to read.

        Returns:
            A description of each cell that differs.

        Example:
            >>> CrashCheck("abs", NumericKernels.ABS).problems([pl.Int8()])
            []
        """
        cells = self.cells(dtypes)
        with ThreadPoolExecutor(max_workers=self.WORKERS) as pool:
            answers = list(pool.map(self._answer, cells))
        problems = []
        for dtype, answer in zip(cells, answers, strict=True):
            rule = self._rule_of(dtype)
            if answer != rule:
                problems.append(f"{dtype!r}: polars {answer}, rule {rule}")
        return problems

    def _rule_of(self, dtype: pl.DataType) -> DType | Stopped:
        """Give the answer of the dispatch for a column of typical values.

        Args:
            dtype: The type of the column.

        Returns:
            The answer.

        Example:
            >>> CrashCheck("exp", FloatKernels.EXP)._rule_of(pl.Struct({"a": pl.Int8()}))
            crashes
        """
        return RuleAnswer().result(lambda: self._dispatch.output(Data(DType.of(dtype))))

    def _answer(self, dtype: pl.DataType) -> DType | Stopped:
        """Give the answer of the subprocess for a column of typical values.

        Args:
            dtype: The type of the column.

        Returns:
            The answer.

        Example:
            >>> CrashCheck("abs", NumericKernels.ABS)._answer(pl.UInt8())
            UInt8
        """
        return self._isolated.answer(dtype, self._samples.typical(dtype))


class FillCheck:
    """Compare the kernel of `fill_null` with its rule, with the cast pass off.

    The cast pass gives the value and the fill one type, so the kernel alone
    meets two types only where the pass is off. The run turns the pass off as
    the census does for its kernels. Polars still types the node, so its type
    function refuses a pair with no supertype before the kernel runs. The
    value holds values or only nulls, and has three rows or one row beside a
    fill of three rows. A cell where the rule cannot answer does not run.

    Example:
        >>> FillCheck().problems(pl.Int8(), [pl.Int16(), pl.Null()])
        []
    """

    def __init__(self) -> None:
        """Build the series of the samples and the flags of a run with the cast pass off.

        Example:
            >>> FillCheck()._flags._pyoptflags.type_coercion
            False
        """
        self._samples = KernelSamples()
        self._flags = KernelFlags().uncast()

    def problems(self, value: pl.DataType, fills: list[pl.DataType]) -> list[str]:
        """Give each cell where polars and the rule differ.

        Args:
            value: The type of the value.
            fills: The types of the fill.

        Returns:
            A description of each cell that differs.

        Example:
            >>> FillCheck().problems(pl.Null(), [pl.Date()])
            []
        """
        problems = []
        for fill in fills:
            for nulls, one_row in itertools.product((False, True), repeat=2):
                rule = self._rule(value, fill, nulls=nulls, one_row=one_row)
                if rule is None:
                    continue
                answer = self._answer(value, fill, nulls=nulls, one_row=one_row)
                if answer != rule:
                    cell = f"{value!r} with {fill!r} {nulls=} {one_row=}"
                    problems.append(f"{cell}: polars {answer}, rule {rule}")
        return problems

    def _rule(
        self,
        value: pl.DataType,
        fill: pl.DataType,
        *,
        nulls: bool,
        one_row: bool,
    ) -> DType | Stopped | None:
        """Give the answer of the rule, or `None` where it cannot answer.

        Args:
            value: The type of the value.
            fill: The type of the fill.
            nulls: Whether every value is null.
            one_row: Whether the value has one row beside the fill.

        Returns:
            The answer, or `None`.

        Example:
            >>> FillCheck()._rule(pl.Int8(), pl.Int16(), nulls=True, one_row=False)
            panics
            >>> FillCheck()._rule(pl.Int8(), pl.Date(), nulls=False, one_row=False)
            raises SchemaError
        """
        mapper = FieldsMapper(DType.of(value), DType.of(fill))
        planned = RuleAnswer().result(mapper.map_to_supertype)
        if isinstance(planned, Stopped):
            return planned
        data = Data(DType.of(value), nulls=nulls, one_row=one_row)
        try:
            return RuleAnswer().result(lambda: FillNull().output(data, Data(DType.of(fill))))
        except UndecidedError:
            return None

    def _answer(
        self,
        value: pl.DataType,
        fill: pl.DataType,
        *,
        nulls: bool,
        one_row: bool,
    ) -> DType | Stopped:
        """Give the answer of polars.

        Args:
            value: The type of the value.
            fill: The type of the fill.
            nulls: Whether every value is null.
            one_row: Whether the value has one row beside the fill.

        Returns:
            The answer.

        Example:
            >>> FillCheck()._answer(pl.Int8(), pl.Null(), nulls=True, one_row=True)
            Null
        """
        values = self._samples.series(value, nulls=nulls, one_row=False).alias("v")
        filled = self._samples.series(fill, nulls=False, one_row=False).alias("f")
        frame = pl.DataFrame([values, filled]).lazy()
        column: pl.Expr = pl.col("v").head(1) if one_row else pl.col("v")
        filler: pl.Expr = pl.col("f")
        query = frame.select(column.fill_null(filler).alias("o"))
        return PolarsAnswer().answer(lambda: query.collect(optimizations=self._flags)["o"].dtype)


class OutOfProcess:
    """Name the ways of stopping that a test cannot run in its own process.

    Example:
        >>> Stop.HANGS in OutOfProcess.STOPS
        True
    """

    STOPS: ClassVar[frozenset[Stop]] = frozenset({Stop.CRASHES, Stop.HANGS})


class Kernels:
    """Hold the operators and the functions that the tests check, with the rule of each.

    Example:
        >>> sorted(Kernels.UNARY)[:3]
        ['abs', 'arccos', 'bitwise_and']
    """

    ARITHMETIC: ClassVar[dict[str, NumericOp]] = {
        "add": NumericOp.ADD,
        "sub": NumericOp.SUB,
        "mul": NumericOp.MUL,
        "div": NumericOp.DIV,
        "rem": NumericOp.REM,
    }
    COMPARISONS: ClassVar[dict[str, CompareOp]] = {
        "eq": CompareOp.EQUAL,
        "neq": CompareOp.NOT_EQUAL,
        "lt": CompareOp.LT,
        "lt_eq": CompareOp.LT_EQ,
        "gt": CompareOp.GT,
        "gt_eq": CompareOp.GT_EQ,
    }
    BITWISE: ClassVar[dict[str, BitOp]] = {
        "bitand": BitOp.AND,
        "bitor": BitOp.OR,
        "bitxor": BitOp.XOR,
    }

    def pair(self, method: str) -> BinaryKernel:
        """Give the rule of a method of two native series.

        Args:
            method: The name of the method.

        Returns:
            The rule.

        Example:
            >>> type(Kernels().pair("bitxor")).__name__
            'BitOps'
        """
        if method in self.ARITHMETIC:
            return SeriesArithmetic(self.ARITHMETIC[method])
        if method in self.COMPARISONS:
            return Comparison(self.COMPARISONS[method])
        return BitOps(self.BITWISE[method])

    OPERATORS: ClassVar[dict[str, tuple[Callable[[pl.Expr, pl.Expr], pl.Expr], ExpressionRule]]] = {
        "+": (lambda left, right: left + right, OperatorRule(Operator.PLUS)),
        "-": (lambda left, right: left - right, OperatorRule(Operator.MINUS)),
        "*": (lambda left, right: left * right, OperatorRule(Operator.MULTIPLY)),
        "/": (lambda left, right: left / right, OperatorRule(Operator.TRUE_DIVIDE)),
        "//": (lambda left, right: left // right, OperatorRule(Operator.FLOOR_DIVIDE)),
        "%": (lambda left, right: left % right, OperatorRule(Operator.MODULUS)),
        "eq_missing": (
            lambda left, right: left.eq_missing(right),
            OperatorRule(Operator.EQ_VALIDITY),
        ),
        "ne_missing": (
            lambda left, right: left.ne_missing(right),
            OperatorRule(Operator.NOT_EQ_VALIDITY),
        ),
        "all_horizontal": (
            pl.all_horizontal,
            OperatorRule(Operator.LOGICAL_AND),
        ),
        "any_horizontal": (
            pl.any_horizontal,
            OperatorRule(Operator.LOGICAL_OR),
        ),
        "pow": (lambda left, right: left.pow(right), PowRule()),
        "log": (lambda left, right: left.log(right), LogRule()),
    }
    UNARY: ClassVar[dict[str, tuple[Callable[[pl.Expr], pl.Expr], UnaryRule]]] = {
        "abs": (lambda column: column.abs(), NumericKernels.ABS),
        "cum_sum": (lambda column: column.cum_sum(), NumericKernels.CUM_SUM),
        "rolling_mean": (lambda column: column.rolling_mean(2), NumericKernels.ROLLING_MEAN),
        "sum": (lambda column: column.sum(), NumericKernels.SUM),
        "mean": (lambda column: column.mean(), NumericKernels.MEAN),
        "sqrt": (lambda column: column.sqrt(), FloatKernels.SQRT),
        "cbrt": (lambda column: column.cbrt(), FloatKernels.CBRT),
        "exp": (lambda column: column.exp(), FloatKernels.EXP),
        "log1p": (lambda column: column.log1p(), FloatKernels.LOG1P),
        "sin": (lambda column: column.sin(), FloatKernels.TRIGONOMETRY),
        "arccos": (lambda column: column.arccos(), FloatKernels.TRIGONOMETRY),
        "tanh": (lambda column: column.tanh(), FloatKernels.TRIGONOMETRY),
        "degrees": (lambda column: column.degrees(), FloatKernels.TRIGONOMETRY),
        "not_": (lambda column: column.not_(), BitwiseKernels.NOT),
        "bitwise_count_ones": (lambda column: column.bitwise_count_ones(), BitwiseKernels.COUNT),
        "bitwise_leading_zeros": (
            lambda column: column.bitwise_leading_zeros(),
            BitwiseKernels.COUNT,
        ),
        "bitwise_and": (lambda column: column.bitwise_and(), BitwiseKernels.REDUCE),
        "bitwise_xor": (lambda column: column.bitwise_xor(), BitwiseKernels.REDUCE),
        "log": (lambda column: column.log(), LogOfBase()),
        "neg": (lambda column: column.neg(), NumericKernels.NEGATE),
        "round": (lambda column: column.round(), NumericKernels.KEEP_NUMBER),
        "floor": (lambda column: column.floor(), NumericKernels.KEEP_NUMBER),
        "ceil": (lambda column: column.ceil(), NumericKernels.KEEP_NUMBER),
        "truncate": (lambda column: column.truncate(), NumericKernels.KEEP_NUMBER),
        "round_sig_figs": (
            lambda column: column.round_sig_figs(2),
            NumericKernels().round_sig_figs(2),
        ),
        "round_sig_figs 0": (
            lambda column: column.round_sig_figs(0),
            NumericKernels().round_sig_figs(0),
        ),
        "cut": (
            lambda column: column.cut([0, 2]),
            NumericKernels().cut([0.0, 2.0], None, left_closed=False, include_breaks=False),
        ),
        "is_nan": (lambda column: column.is_nan(), FloatKernels.IS_NAN),
        "is_not_nan": (lambda column: column.is_not_nan(), FloatKernels.IS_NOT_NAN),
        "is_finite": (lambda column: column.is_finite(), FloatKernels.IS_FINITE),
        "is_infinite": (lambda column: column.is_infinite(), FloatKernels.IS_INFINITE),
        "is_null": (lambda column: column.is_null(), LayoutKernels.IS_NULL),
        "hash": (lambda column: column.hash(), LayoutKernels.HASH),
        "set_sorted": (lambda column: column.set_sorted(), LayoutKernels.SET_SORTED),
        "to_physical": (lambda column: column.to_physical(), LayoutKernels.TO_PHYSICAL),
    }
    GROUPED: ClassVar[dict[str, tuple[Callable[[pl.Expr], pl.Expr], Dispatch]]] = {
        "sum": (lambda column: column.sum(), NumericKernels.SUM_GROUPS),
    }
    CRASHING: ClassVar[tuple[str, ...]] = ("exp", "log1p", "log")
    CHECKED: ClassVar[tuple[str, ...]] = ("sqrt", "cbrt")

    @classmethod
    def crashing(cls) -> tuple[str, ...]:
        """Give the float kernels that end the process over a struct, in the installed release.

        `sqrt` and `cbrt` check the type that their cast gives from polars
        1.44.0, so a struct raises there, and the other tests of the kernels
        run it in this process.

        Returns:
            The names of the kernels.

        Example:
            >>> "exp" in Kernels.crashing()
            True
        """
        if PolarsRelease.installed() < PolarsRelease.parse("1.44.0"):
            return cls.CHECKED + cls.CRASHING
        return cls.CRASHING

    def every_dtype(self) -> list[pl.DataType]:
        """Give the types of the rules and of the nested kernels, each once.

        Returns:
            The types.

        Example:
            >>> len(Kernels().every_dtype())
            75
        """
        dtypes: list[pl.DataType] = []
        for dtype in [*RuleKinds().dtypes(), *NestedKinds().dtypes()]:
            if dtype not in dtypes:
                dtypes.append(dtype)
        return dtypes


class TestSeriesKernels:
    """Replay the kernels of two series against the native series, on every pair of types."""

    @pytest.mark.parametrize("method", list(Kernels.ARITHMETIC))
    @pytest.mark.parametrize("left", RuleKinds().dtypes(), ids=repr)
    def test_arithmetic_of_two_series_matches_the_rule(
        self,
        method: str,
        left: pl.DataType,
    ) -> None:
        """`+`, `-`, `*`, `/` and `%` of two series give the type of the rule or fail as it says."""
        check = PairCheck(method, SeriesArithmetic(Kernels.ARITHMETIC[method]))
        problems = check.problems(left, RuleKinds().dtypes(), Case.equal_lengths())
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("method", list(Kernels.COMPARISONS))
    @pytest.mark.parametrize("left", RuleKinds().dtypes(), ids=repr)
    def test_comparison_of_two_series_matches_the_rule(
        self,
        method: str,
        left: pl.DataType,
    ) -> None:
        """A comparison of two series gives `Boolean` where the rule does, or fails as it says."""
        check = PairCheck(method, Comparison(Kernels.COMPARISONS[method]))
        problems = check.problems(left, RuleKinds().dtypes(), Case.equal_lengths())
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("method", list(Kernels.BITWISE))
    @pytest.mark.parametrize("left", RuleKinds().dtypes(), ids=repr)
    def test_bitwise_operator_of_two_series_matches_the_rule(
        self,
        method: str,
        left: pl.DataType,
    ) -> None:
        """`&`, `|` and `^` of two series match the rule, with one row on a side too."""
        check = PairCheck(method, BitOps(Kernels.BITWISE[method]))
        problems = check.problems(left, RuleKinds().dtypes(), Case.every_length())
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize(
        "method", [*Kernels.ARITHMETIC, *Kernels.COMPARISONS, *Kernels.BITWISE]
    )
    @pytest.mark.parametrize("left", RuleKinds().dtypes(), ids=repr)
    def test_kernel_of_a_series_of_no_row_reads_it_as_nulls(
        self,
        method: str,
        left: pl.DataType,
    ) -> None:
        """A kernel of two series reads a side of no row as a side of only nulls."""
        check = PairCheck(method, Kernels().pair(method))
        problems = check.empty_problems(left, RuleKinds().dtypes())
        assert not problems, "\n".join(problems)


class TestNestedKernels:
    """Replay the nested kernels against polars: lists, arrays and structs two levels deep."""

    @pytest.mark.parametrize("method", list(Kernels.ARITHMETIC))
    @pytest.mark.parametrize("left", NestedKinds().dtypes(), ids=repr)
    def test_nested_arithmetic_of_two_series_matches_the_rule(
        self,
        method: str,
        left: pl.DataType,
    ) -> None:
        """Arithmetic on nested values matches the rule, with one row on a side too."""
        check = PairCheck(method, SeriesArithmetic(Kernels.ARITHMETIC[method]))
        problems = check.problems(left, NestedKinds().dtypes(), Case.every_length())
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("method", list(Kernels.COMPARISONS))
    @pytest.mark.parametrize("left", NestedKinds().dtypes(), ids=repr)
    def test_nested_comparison_of_two_series_matches_the_rule(
        self,
        method: str,
        left: pl.DataType,
    ) -> None:
        """A comparison of nested values matches the rule, with one row on a side too."""
        check = PairCheck(method, Comparison(Kernels.COMPARISONS[method]))
        problems = check.problems(left, NestedKinds().dtypes(), Case.every_length())
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("operator", list(Kernels.OPERATORS))
    @pytest.mark.parametrize("left", NestedKinds().dtypes(), ids=repr)
    def test_nested_operator_of_an_expression_matches_the_rule(
        self,
        operator: str,
        left: pl.DataType,
    ) -> None:
        """An operator on nested values matches the rule where the kernel sees the columns."""
        build, rule = Kernels.OPERATORS[operator]
        check = ExpressionCheck(build(pl.col("l"), pl.col("r")), rule)
        problems = check.problems(left, NestedKinds().dtypes())
        assert not problems, "\n".join(problems)


class TestOperatorKernels:
    """Replay the kernels that only an expression reaches against polars, on every pair of types."""

    @pytest.mark.parametrize("operator", list(Kernels.OPERATORS))
    @pytest.mark.parametrize("left", RuleKinds().dtypes(), ids=repr)
    def test_operator_of_an_expression_matches_the_rule(
        self,
        operator: str,
        left: pl.DataType,
    ) -> None:
        """An operator of two columns matches the rule where the kernel sees the columns."""
        build, rule = Kernels.OPERATORS[operator]
        check = ExpressionCheck(build(pl.col("l"), pl.col("r")), rule)
        problems = check.problems(left, RuleKinds().dtypes())
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("operator", list(UncastCheck.OPERATORS))
    @pytest.mark.parametrize("left", RuleKinds().dtypes(), ids=repr)
    def test_in_place_arithmetic_with_the_cast_pass_off_matches_the_rule(
        self,
        operator: str,
        left: pl.DataType,
    ) -> None:
        """The in-place path of an operator matches the rule on columns of two types."""
        problems = UncastCheck(operator).problems(left, RuleKinds().dtypes())
        assert not problems, "\n".join(problems)


class TestUnaryKernels:
    """Replay the kernels of one input against polars, on every type."""

    @pytest.mark.parametrize("function", list(Kernels.UNARY))
    def test_kernel_of_one_input_matches_its_dispatch(self, function: str) -> None:
        """A function of one column gives the type of its dispatch, or fails as it says."""
        build, dispatch = Kernels.UNARY[function]
        problems = UnaryCheck(build(pl.col("x")), dispatch).problems(Kernels().every_dtype())
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("function", list(Kernels.GROUPED))
    def test_kernel_of_each_group_matches_its_dispatch(self, function: str) -> None:
        """A kernel of the groups of `agg` gives the type of its dispatch, or fails as it says."""
        build, dispatch = Kernels.GROUPED[function]
        check = UnaryCheck(build(pl.col("x")), dispatch, grouped=True)
        problems = check.problems(Kernels().every_dtype())
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("function", Kernels.crashing())
    def test_float_kernel_over_a_struct_ends_the_process(self, function: str) -> None:
        """Each cell where the dispatch says that the process ends does end it, in a subprocess."""
        _, dispatch = Kernels.UNARY[function]
        check = CrashCheck(function, dispatch)
        assert check.cells(Kernels().every_dtype()), "no cell ends the process"
        problems = check.problems(Kernels().every_dtype())
        assert not problems, "\n".join(problems)


class TestFillKernel:
    """Replay the kernel of `fill_null` against polars with no cast pass, on each pair of types."""

    @pytest.mark.parametrize("value", RuleKinds().dtypes(), ids=repr)
    def test_fill_of_a_value_matches_the_rule(self, value: pl.DataType) -> None:
        """The fill of a value gives the type of the rule, or fails as it says, beside each fill."""
        problems = FillCheck().problems(value, RuleKinds().dtypes())
        assert not problems, "\n".join(problems)


class TestKernelFacts:
    """Check that the probe of each fact of the kernels still shows the fact."""

    @pytest.mark.parametrize("fact", KernelFacts.ALL, ids=lambda fact: fact.name)
    def test_probe_of_a_fact_panics(self, fact: KernelFact) -> None:
        """The probe of each fact panics in polars, as its reason says.

        In a release older than the first one that shows the fact, the probe runs.
        """
        namespace = {"pl": pl}
        probe = fact.probe
        answer = PolarsAnswer().answer(
            lambda: eval(probe, namespace).dtype,  # ruff: ignore[suspicious-eval-usage],
        )
        first = PolarsRelease.parse(fact.since) if fact.since is not None else None
        older = first is not None and PolarsRelease.installed() < first
        assert (answer != Stopped(Stop.PANICS)) if older else (answer == Stopped(Stop.PANICS))


TEXTS: tuple[str, ...] = (
    "a",
    "2020-01-01",
    "2020-1-1",
    "20-01-01",
    " 2020-01-01",
    "2020/01/01",
    "01/02/2020",
    "2020-01-01 10:20:30",
    "2020-01-01T10:20:30",
    "10:20:30",
    "1:2:3",
    "",
    "Jan 5 2020",
    "+2020-01-01",
    "12 PM",
)
FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%H:%M:%S",
    "%F %T",
    "%b %d %Y",
    "%I %p",
    "%y-%m-%d",
)


@pytest.mark.parametrize("form", FORMATS)
def test_a_text_that_cannot_fit_a_format_fails_every_parse(form: str) -> None:
    """Where `TextFormat` refuses a text, a strict parse of polars raises at every target type.

    Polars refuses some formats for a `Time` or a `Datetime` before it reads
    the text, with a `ComputeError`.
    """
    for text in TEXTS:
        if not TextFormat().refuses(text, form):
            continue
        for target in (pl.Date, pl.Datetime("us"), pl.Time):
            parse = pl.lit(text).str.strptime(target, form, strict=True)
            refusals = (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError)
            with pytest.raises(refusals):
                pl.select(parse)


def parsed_type(frame: pl.DataFrame, parse: pl.Expr) -> pl.DataType:
    """Give the type of a parse over a frame.

    Args:
        frame: The frame.
        parse: The parse.

    Returns:
        The type.

    Example:
        >>> parsed_type(pl.DataFrame(), pl.lit("2024").str.to_date("%Y"))
        Date
    """
    return frame.select(parse).dtypes[0]


MONTH_TEXTS: tuple[str, ...] = (
    "2024-01",
    "2024-1",
    "202401",
    "20241",
    "01/2024",
    "1/2024",
    "2024-Jan",
    "2024-jan",
    "2024 January",
    "2024 Janu",
    "24",
    "2x",
    "2024",
    "-2024",
    "+2024",
    "2024%01",
)
MONTH_FORMATS: tuple[str, ...] = ("%Y-%m", "%Y%m", "%m/%Y", "%Y-%b", "%Y %B", "%y", "%Y")


@pytest.mark.parametrize("form", MONTH_FORMATS)
def test_a_format_without_a_day_parses_only_what_the_fixed_width_parser_reads(form: str) -> None:
    """Polars parses a text by a format with no day exactly where its fixed-width parser reads it.

    `chrono` needs the day, so a text that the fixed-width parser does not
    read fails, for a date and for a moment.
    """
    assert FixedWidthParse().supports(form)
    assert TextFormat().lacks_date(form)
    for text in MONTH_TEXTS:
        for target in (pl.Date, pl.Datetime("us")):
            parse = pl.lit(text).str.strptime(target, form, strict=True)
            runs = PolarsAnswer().answer(functools.partial(parsed_type, pl.DataFrame(), parse))
            assert (not isinstance(runs, Stopped)) == FixedWidthParse().reads(text, form), text


@pytest.mark.parametrize("form", ["%Y%%%m", "%Y-%m%%", "%m-%d"])
def test_a_format_without_a_whole_date_that_chrono_reads_fails_every_text(form: str) -> None:
    """A format that lacks a whole date fails every text where `chrono` reads it.

    `chrono` reads a format that the fixed-width parser does not read, as one
    with `%%`, and every format of a parse that is not exact.
    """
    assert TextFormat().lacks_date(form)
    exacts = [False] if FixedWidthParse().supports(form) else [True, False]
    targets = (pl.Date(), pl.Datetime("us"))
    for text, target, exact in itertools.product((*MONTH_TEXTS, "01-02"), targets, exacts):
        parse = pl.col("s").str.strptime(target, form, strict=True, exact=exact)
        with pytest.raises(pl.exceptions.InvalidOperationError):
            pl.DataFrame({"s": [text]}).select(parse)


@pytest.mark.parametrize(
    "form",
    [
        "%Y-%m-%d %H",
        "%Y-%m-%d %M",
        "%Y-%m-%d %S",
        "%Y %I:%M",
        "%Y %H:%M %p",
        "%F %T",
        "%F %R",
        "%F",
    ],
)
def test_polars_refuses_a_format_at_every_frame_where_the_fault_says_so(form: str) -> None:
    """An exact parse into a date or a moment raises over no row where the format is faulty."""
    fault = FormatFault().of(form)
    for target in (pl.Date, pl.Datetime("us")):
        parse = pl.col("s").str.strptime(target, form, strict=True)
        empty = pl.DataFrame({"s": []}, schema={"s": pl.String})
        answer = PolarsAnswer().answer(functools.partial(parsed_type, empty, parse))
        assert isinstance(answer, Stopped) == (fault is not None), target


class KernelTrees:
    """Hold trees that reach paths of the kernels which the samples of the nodes do not reach.

    The samples of the definitions run each node over its census inputs. These
    trees reach the rest: an option of a node, as the fraction of `sample` or a
    week with no business day, an input of a nested type, and an input whose
    rows the code bounds. Each tree runs over the frame of `TreeFrame`, with the
    columns that it adds.

    Example:
        >>> "sample by a fraction" in KernelTrees.TREES
        True
    """

    TREES: ClassVar[dict[str, tuple[dict[str, pl.Series], pl.Expr]]] = ApiGate().kept({
        "business days of a week with no business day": (
            {"d": TreeColumns.DATES},
            pl.col("d").dt.add_business_days(1, week_mask=[False] * 7),
        ),
        "is_close with a negative tolerance": ({}, pl.col("x").is_close(1, abs_tol=-1.0)),
        "sum_horizontal that keeps the nulls": (
            {},
            pl.sum_horizontal(pl.col("x"), pl.col("g"), ignore_nulls=False),
        ),
        "sample by a fraction": ({}, pl.col("x").sample(fraction=0.5)),
        "replace of the year of a moment in nanoseconds": (
            {"n": TreeColumns.MOMENTS.cast(pl.Datetime("ns"))},
            pl.col("n").dt.replace(year=2021),
        ),
        "fill by one of a decimal of nulls": (
            {},
            pl.lit(None, dtype=pl.Decimal(10, 2)).fill_null(strategy="one"),
        ),
        "linear spaces of Float32": (
            {},
            pl.linear_spaces(pl.lit(0.0, dtype=pl.Float32), pl.lit(1.0, dtype=pl.Float32), 3),
        ),
        "replace of two occurrences by a column": (
            {},
            pl.col("t").str.replace("a", pl.col("t"), n=2),
        ),
        "sample of more rows than a head holds": ({}, pl.col("x").head(2).sample(5)),  # type: ignore[invalid-call-shape]
        "reshape of a list": ({}, pl.col("l").reshape((-1,))),
        "reshape into rows of five": ({}, pl.col("x").reshape((-1, 5))),
        "extend_constant of no row": ({}, pl.col("x").head(0).extend_constant(1, 2)),
        "search_sorted of nulls": ({"n": TreeColumns.NULLS}, pl.col("n").search_sorted(None)),
        "set_union of lists of booleans": (
            {"b": TreeColumns.CHOICES},
            pl.col("b").list.set_union(pl.col("b")),
        ),
        "slice by a null offset": ({}, pl.col("x").slice(pl.lit(None, dtype=pl.Int64), 2)),  # type: ignore[invalid-call-shape]
        "int_range of a step of zero": ({}, pl.int_range(0, 5, 0)),  # type: ignore[invalid-call-shape]
        "int_ranges of a step of zero": ({}, pl.int_ranges(0, pl.col("x"), 0)),
        "quantile by a null list": (
            {},
            pl.col("x").quantile(pl.lit(None, dtype=pl.List(pl.Float64))),  # type: ignore[invalid-call-shape]
        ),
        "replace by a value of another length": (
            {},
            pl.col("t").str.replace("a", pl.col("t").head(2)),
        ),
        "to_integer into a float type": (
            {"n": TreeColumns.NUMBERS},
            pl.col("n").str.to_integer(dtype=pl.Float32),  # type: ignore[arg-type]
        ),
        "rolling_var of weights that sum to zero": (
            {},
            pl.col("x").rolling_var(2, weights=[0.0, 0.0]),
        ),
        "rolling_mean of more weights than the window": (
            {},
            pl.col("x").rolling_mean(2, weights=[1.0, 1.0, 1.0]),
        ),
        "rolling_var of a list literal, which the cast to a float refuses": (
            {},
            pl.lit([1, 2]).rolling_var(2),  # type: ignore[invalid-call-shape]
        ),
        "list.slice by a text of a float": ({}, pl.col("l").list.slice(pl.lit("1.5"))),
        "list.slice of one list by a null offset beside a column length": (
            {},
            pl.lit([1, 2]).list.slice(pl.lit(None, dtype=pl.Int64), pl.col("x")),
        ),
        "list.slice by a text of a float beside a column length": (
            {},
            pl.col("l").list.slice(pl.lit("1.5"), pl.col("x")),
        ),
        "reshape to no element": ({}, pl.col("x").reshape((0,))),
        "replace of no occurrence by a column": (
            {},
            pl.col("t").str.replace("a", pl.col("t"), n=0),
        ),
        "rolling over windows of no length": (
            {"m": TreeColumns.MOMENTS},
            pl.col("x").sum().rolling(index_column="m", period="0d"),
        ),
        "append of a list whose data the plan does not name": (
            {"ts": TreeColumns.TEXTS},
            pl.col("ts").append(pl.col("ts").list.eval(pl.element().sqrt())),
        ),
        "append of a struct whose data the plan does not name": (
            {},
            pl.struct(pl.col("t").alias("a")).append(pl.struct(pl.col("t").sqrt().alias("a"))),
        ),
        "append of a cumulative sum of decimals": (
            {},
            pl
            .col("x")
            .cast(pl.Decimal(10, 2))
            .append(pl.col("x").cast(pl.Decimal(10, 2)).cum_sum()),
        ),
        "a struct of one field of a struct plus a struct of two": (
            {},
            pl.struct(pl.struct("x").alias("a"))
            + pl.struct(pl.struct("x").alias("p"), pl.struct("g").alias("q")),
        ),
        "a struct of one field of a struct floordiv a struct of two": (
            {},
            pl.struct(pl.struct("x").alias("a"))
            // pl.struct(pl.struct("x").alias("p"), pl.struct("g").alias("q")),
        ),
        "linear_space by a count of text": ({}, pl.linear_space(0, 1, pl.lit("3"))),
        "linear_space by a count of text with an underscore": (
            {},
            pl.linear_space(0, 1, pl.lit("1_0")),  # type: ignore[invalid-call-shape]
        ),
        "linear_spaces of Float16 bounds": (
            {},
            pl.linear_spaces(pl.lit(0.0, dtype=pl.Float16), pl.lit(1.0, dtype=pl.Float16), 3),
        ),
        "list.slice by a list offset beside a categorical length": (
            {"c": TreeColumns.CATEGORIES},
            pl.lit([1, 2]).list.slice(pl.lit([1, 2]), pl.col("c")),
        ),
        "list.slice by a bytes offset beside a column length": (
            {},
            pl.lit([1, 2]).list.slice(pl.lit(b"1"), pl.col("x")),
        ),
        "list.slice of one list of nulls by a column and a length of no row": (
            {"n": TreeColumns.NULLS, "d": TreeColumns.DATES, "ts": TreeColumns.TEXTS},
            TemplateCall("pl.col('x').list.slice(0, 1)")(
                pl.col("n").implode(),
                pl.col("d"),
                pl.col("ts").head(0),
            ),
        ),
        "replace_strict over no row by a list of nulls": (
            {
                "sp": TreeColumns.SPANS,
                "ts": TreeColumns.TEXTS,
                "n": TreeColumns.NULLS,
                "b": TreeColumns.FLAGS,
            },
            TemplateCall("pl.col('x').replace_strict({1: 2}, default=0)")(
                pl.col("sp").head(0),
                pl.col("ts"),
                pl.col("n").implode(),
                pl.col("b"),
            ),
        ),
        "replace_strict by bytes with a default that the values do not hold": (
            {"u8": TreeColumns.UNSIGNED, "n": TreeColumns.NULLS},
            TemplateCall("pl.col('x').replace_strict({1: 2}, default=0)")(
                pl.col("u8"),
                pl.col("n"),
                pl.lit(b"\xff"),
                pl.lit(300),
            ),
        ),
        "offset_by a text of a number with no unit": (
            {"d": TreeColumns.DATES},
            pl.col("d").dt.offset_by(pl.lit("1d2")),
        ),
        "replace of a null date": ({}, pl.lit(None, dtype=pl.Date).dt.replace(year=2024)),
        "a comparison of objects": (
            {"o": pl.Series([object() for _ in range(5)], dtype=pl.Object)},
            pl.col("o") == pl.col("o"),
        ),
        "a comparison of structs of objects": (
            {"o": pl.Series([object() for _ in range(5)], dtype=pl.Object)},
            pl.struct(pl.col("o")) == pl.struct(pl.col("o")),
        ),
        "a date format with a percent sign": ({}, pl.lit("2024%0102").str.to_date("%Y%%%m%d")),
        "a time format with no minute": ({}, pl.lit("01").str.strptime(pl.Time, "%H")),  # type: ignore[invalid-call-shape]
        "a time format of a 12-hour clock with no half of the day": (
            {},
            pl.lit("01:02").str.strptime(pl.Time, "%I:%M"),  # type: ignore[invalid-call-shape]
        ),
        "a time parse that is not exact, which only a serialized tree writes": (
            {},
            pl.Expr.deserialize(
                io.StringIO(
                    json.dumps({
                        "Function": {
                            "input": [
                                {"Literal": {"Scalar": {"String": "01:02"}}},
                                {"Literal": {"Scalar": {"String": "raise"}}},
                            ],
                            "function": {
                                "StringExpr": {
                                    "Strptime": [
                                        {"Literal": "Time"},
                                        {
                                            "format": "%H:%M",
                                            "strict": True,
                                            "exact": False,
                                            "cache": True,
                                        },
                                    ],
                                },
                            },
                        },
                    }),
                ),
                format="json",
            ),
        ),
        "a time format with a directive that the rules do not read": (
            {},
            pl.lit("01:02:03.5").str.strptime(pl.Time, "%H:%M:%S%.f"),
        ),
        "a time format of a 12-hour clock with its half of the day": (
            {},
            pl.lit("01:02 am").str.strptime(pl.Time, "%I:%M %p"),
        ),
        **ApiGate().built(
            "ewm_sum_by",
            lambda: {
                "ewm_sum_by of a correlation of two literals": (
                    {},
                    TemplateCall("pl.col('x').ewm_sum_by(by='t', half_life='1d')")(
                        TemplateCall("pl.corr('x', 'y')")(pl.lit(300), pl.lit({"a": 1})),
                        pl.col("l").head(0),
                    ),
                ),
            },
        ),
        "fill by one of a null literal of a narrow decimal": (
            {},
            pl.lit(None, dtype=pl.Decimal(2, 2)).fill_null(strategy="one"),  # type: ignore[invalid-call-shape]
        ),
    })
    UNDECIDED: ClassVar[dict[str, tuple[dict[str, pl.Series], pl.Expr]]] = {
        "to_integer by a categorical base": (
            {"n": TreeColumns.NUMBERS, "c": TreeColumns.CATEGORIES},
            pl.col("n").str.to_integer(base=pl.col("c")),
        ),
    }


@pytest.mark.parametrize("name", sorted(KernelTrees.TREES))
def test_a_tree_reaches_a_path_of_the_kernels(name: str) -> None:
    """The four answers of the rules match polars, for a tree that reaches one path of a kernel."""
    columns, expression = KernelTrees.TREES[name]
    frame = TreeFrame(columns)
    assert frame.rules(expression) == frame.polars(expression)


@pytest.mark.parametrize("name", sorted(KernelTrees.UNDECIDED))
def test_a_tree_of_a_kernel_that_the_rules_do_not_decide_runs_unchecked(name: str) -> None:
    """The planner answers match polars, and the rules leave each run to the process."""
    columns, expression = KernelTrees.UNDECIDED[name]
    frame = TreeFrame(columns)
    rules, polars = frame.rules(expression), frame.polars(expression)
    assert (rules[:2], rules[2:]) == (polars[:2], ["unchecked", "unchecked"])
