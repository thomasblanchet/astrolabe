"""The types, the values and the calls of polars that the tests of the rules read.

The modules of `astrolabe._rules` translate the source of polars. Their tests
compare each rule with what polars itself gives. These classes hold the parts
that several tests share: the types and the values to check, a safe way to
read an answer from polars and from a rule in one form, and the public calls
that compute one join or one cast pass.

The tests of the definitions of the nodes share the sample of each node: the
inputs of every type in every row form and the literals (`Inputs`), the frames
of two groups with samples of values that fit and that do not (`LiveFrames`),
the four answers of polars for one cell (`PolarsCell`), their comparison with
the rules (`CellCheck`), and the cells of each node (`NodeSample`).
"""

from __future__ import annotations

import datetime as dt
import enum
import functools
import json
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from astrolabe._rules.casts import CastCheck
from astrolabe._rules.definitions.node_options import NodeOptions
from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.execution import GroupBy, QueryFlags
from astrolabe._rules.failures import PolarsRefusal, Stop, Undecided
from astrolabe._rules.node_evaluation import (
    ColumnLeaf,
    LiteralLeaf,
    NodeEvaluation,
    Refused,
    Unchecked,
)
from astrolabe._rules.operators import Operator
from astrolabe._rules.options import RowForm
from astrolabe._rules.supertype import Supertype, SupertypeFlag
from astrolabe._rules.tree import ExpressionTree
from astrolabe._rules.tree_evaluation import TreeEvaluation
from spec.tools.node_census.kinds import AnyValues, CensusKinds, Sample
from spec.tools.node_census.outcomes import LiteralOutcome, OutcomeSpace
from spec.tools.node_census.surface import NodeTemplate
from spec.tools.rules.census_text import CensusLeaves, CensusText

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from astrolabe._rules.definitions.definition import Definition
    from astrolabe._rules.node_evaluation import Answer, CellAnswers, Leaf


class RuleKinds:
    """Give the data types that the tests of the rules pair, and the literals beside them.

    They are the 27 kinds of the census, then the variants that the rules
    treat apart: the units and the zones of a `Datetime`, the units of a
    `Duration`, the precisions and the scales of a `Decimal`, the widths of an
    `Array`, the sets and the orders of the fields of a `Struct`, a `List` and
    an `Array` two deep, a list and a struct of a logical type, and the
    categories of an `Enum`.

    A literal with no data type is an integer or a float. A Python string is a
    literal of type `String`.

    Example:
        >>> len(RuleKinds().dtypes())
        55
        >>> RuleKinds().literal_type(300)
        Unknown(int, 300)
    """

    VARIANTS: ClassVar[tuple[pl.DataType, ...]] = (
        pl.Datetime("us"),
        pl.Datetime("ns"),
        pl.Datetime("ms", "UTC"),
        pl.Datetime("ns", "Europe/Paris"),
        pl.Duration("ms"),
        pl.Duration("ns"),
        pl.Decimal(38, 2),
        pl.Decimal(5, 0),
        pl.Decimal(20, 10),
        pl.Decimal(38, 0),
        pl.Decimal(3, 3),
        pl.Array(pl.Int64(), 3),
        pl.Array(pl.Int8(), 2),
        pl.Struct({"b": pl.Int8()}),
        pl.Struct({"a": pl.Int8(), "b": pl.String()}),
        pl.Struct({"b": pl.String(), "a": pl.Int8()}),
        pl.Struct({"a": pl.Int64()}),
        pl.List(pl.List(pl.Int64())),
        pl.List(pl.Array(pl.Int64(), 2)),
        pl.Array(pl.List(pl.Int64()), 2),
        pl.Array(pl.Array(pl.Int64(), 2), 2),
        pl.List(pl.Struct({"a": pl.Int8()})),
        pl.List(pl.String()),
        pl.List(pl.Int8()),
        pl.Enum(["a", "b"]),
        pl.Enum(["c", "b", "a"]),
        pl.List(pl.Date()),
        pl.Struct({"a": pl.Date()}),
    )
    LITERALS: ClassVar[tuple[int | float | str, ...]] = (1, -1, 300, 2**40, 2**63, 1.5, "a")

    def dtypes(self) -> list[pl.DataType]:
        """Give the data types: the census kinds, then the variants.

        Returns:
            The data types.

        Example:
            >>> RuleKinds().dtypes()[27]
            Datetime(time_unit='us', time_zone=None)
        """
        return CensusKinds().dtypes() + list(self.VARIANTS)

    def lists(self) -> list[pl.DataType]:
        """Give the data types that are a `List`.

        Returns:
            The list types.

        Example:
            >>> RuleKinds().lists()[0]
            List(Int64)
        """
        return [dtype for dtype in self.dtypes() if isinstance(dtype, pl.List)]

    def literal_type(self, value: int | float | str) -> DType:
        """Give the planner type of a literal.

        Args:
            value: The Python value of `pl.lit(value)`.

        Returns:
            `Unknown` for a number, and `String` for a string.

        Example:
            >>> RuleKinds().literal_type(1.5).args
            ('float', None)
        """
        if isinstance(value, str):
            return DType("String")
        if isinstance(value, int):
            return DType.dynamic_int(value)
        return DType.unknown("float")


class NestedKinds:
    """Give the data types of the tests of the nested kernels.

    They are lists, arrays and structs whose leaves are of every kind that
    the nested kernels tell apart, two levels deep, and flat types to put
    beside them: numbers, `Boolean`, `Null`, text and temporal types.

    Example:
        >>> len(NestedKinds().dtypes())
        37
    """

    FLAT: ClassVar[tuple[pl.DataType, ...]] = (
        pl.Int8(),
        pl.UInt64(),
        pl.Float32(),
        pl.Float64(),
        pl.Boolean(),
        pl.Null(),
        pl.String(),
        pl.Decimal(10, 2),
        pl.Date(),
        pl.Duration("us"),
    )
    LISTS: ClassVar[tuple[pl.DataType, ...]] = (
        pl.List(pl.Int8()),
        pl.List(pl.Float32()),
        pl.List(pl.Boolean()),
        pl.List(pl.Null()),
        pl.List(pl.String()),
        pl.List(pl.Decimal(10, 2)),
        pl.List(pl.Date()),
        pl.List(pl.List(pl.Int16())),
        pl.List(pl.List(pl.Boolean())),
        pl.List(pl.Array(pl.Int8(), 2)),
        pl.List(pl.Struct({"a": pl.Int8()})),
    )
    ARRAYS: ClassVar[tuple[pl.DataType, ...]] = (
        pl.Array(pl.Int8(), 2),
        pl.Array(pl.Float64(), 2),
        pl.Array(pl.Boolean(), 2),
        pl.Array(pl.Null(), 2),
        pl.Array(pl.Int8(), 3),
        pl.Array(pl.Array(pl.Int16(), 2), 2),
        pl.Array(pl.List(pl.Int8()), 2),
        pl.Array(pl.String(), 2),
    )
    STRUCTS: ClassVar[tuple[pl.DataType, ...]] = (
        pl.Struct({"a": pl.Int8()}),
        pl.Struct({"a": pl.Int8(), "b": pl.Float64()}),
        pl.Struct({"x": pl.Float32(), "y": pl.Int8(), "z": pl.Boolean()}),
        pl.Struct({"a": pl.List(pl.Int8())}),
        pl.Struct({"a": pl.Struct({"b": pl.Int8()})}),
        pl.Struct({"a": pl.Date()}),
        pl.Struct({"a": pl.Boolean()}),
        pl.Struct({"a": pl.String()}),
    )

    def dtypes(self) -> list[pl.DataType]:
        """Give the data types.

        Returns:
            The flat types, then the lists, the arrays and the structs.

        Example:
            >>> NestedKinds().dtypes()[10]
            List(Int8)
        """
        return [*self.FLAT, *self.LISTS, *self.ARRAYS, *self.STRUCTS]


class KernelSamples:
    """Give the values of the series that the tests of the kernels build.

    The values are the typical sample of the census, with two changes. A
    `String` holds `a` and `b`, which are categories of every `Enum` of the
    tests, so that an ordering of an `Enum` and a text runs. A `List` holds
    two values in each row, so that a list of one row never holds one null
    value only (`KernelFacts.SHARED_NULL_LEAF`).

    Example:
        >>> KernelSamples().typical(pl.List(pl.Int8()))
        [[1, 1], [2, 2], [3, 3]]
        >>> KernelSamples().series(pl.Int8(), nulls=True, one_row=True).to_list()
        [None]
    """

    def typical(self, dtype: pl.DataType) -> list[object]:
        """Give three typical values of a data type.

        Args:
            dtype: The data type.

        Returns:
            The values.

        Example:
            >>> KernelSamples().typical(pl.Struct({"a": pl.String()}))
            [{'a': 'a'}, {'a': 'b'}, {'a': 'a'}]
        """
        if isinstance(dtype, pl.String):
            return ["a", "b", "a"]
        if isinstance(dtype, pl.List):
            return [[value, value] for value in self.typical(self._instance(dtype.inner))]
        if isinstance(dtype, pl.Array):
            return [[value] * dtype.size for value in self.typical(self._instance(dtype.inner))]
        if isinstance(dtype, pl.Struct):
            return self._struct(dtype)
        return list(AnyValues().of(dtype, Sample.TYPICAL))

    def series(self, dtype: pl.DataType, *, nulls: bool, one_row: bool) -> pl.Series:
        """Give a series of a data type.

        Args:
            dtype: The data type.
            nulls: Whether every value is null.
            one_row: Whether the series has one row, and not three.

        Returns:
            The series.

        Example:
            >>> KernelSamples().series(pl.Date(), nulls=False, one_row=True).to_list()
            [datetime.date(2020, 1, 1)]
        """
        values = [None, None, None] if nulls else self.typical(dtype)
        return pl.Series("x", values[:1] if one_row else values, dtype=dtype)

    def _struct(self, dtype: pl.Struct) -> list[object]:
        """Give three typical values of a struct, one row of field values each.

        Args:
            dtype: The struct type.

        Returns:
            The values.

        Example:
            >>> KernelSamples()._struct(pl.Struct({"a": pl.Int8()}))
            [{'a': 1}, {'a': 2}, {'a': 3}]
        """
        fields = {field.name: self.typical(self._instance(field.dtype)) for field in dtype.fields}
        return [{name: values[row] for name, values in fields.items()} for row in range(3)]

    def _instance(self, dtype: object) -> pl.DataType:
        """Give an instance of a data type that polars can also give as a class.

        Args:
            dtype: The data type, as an instance or as a class.

        Returns:
            The instance.

        Example:
            >>> KernelSamples()._instance(pl.Int8)
            Int8
        """
        if isinstance(dtype, pl.DataType):
            return dtype
        instance: pl.DataType = dtype()  # type: ignore[operator]
        return instance


@dataclass(frozen=True)
class Stopped:
    """Hold how a run stopped: the way, and the name of the class of its error where it raises.

    A test runs the same query in polars and in the rules, and compares the
    two answers. Where both stop, they stop in the same way only where both
    give the same `Stopped`. A panic of polars has no class of its own.

    Example:
        >>> Stopped(Stop.RAISES, "ShapeError"), Stopped(Stop.PANICS)
        (raises ShapeError, panics)
    """

    stop: Stop
    error: str | None = None

    def __repr__(self) -> str:
        """Give the way and the class of the error.

        Returns:
            The text.

        Example:
            >>> Stopped(Stop.HANGS)
            hangs
        """
        return self.stop.value if self.error is None else f"{self.stop.value} {self.error}"

    @classmethod
    def of(cls, refusal: PolarsRefusal) -> Stopped:
        """Give how a rule says that polars stops.

        Args:
            refusal: The refusal of the rule.

        Returns:
            The way, and the class of the error where polars raises.

        Example:
            >>> from astrolabe._rules.failures import ErrorClass
            >>> Stopped.of(PolarsRefusal(ErrorClass.SHAPE, "lengths"))
            raises ShapeError
        """
        return cls(refusal.stop, None if refusal.error is None else refusal.error.value)

    def way(self) -> Stopped:
        """Give the way alone, for a test that does not compare the classes of the errors.

        Returns:
            The way, with no class.

        Example:
            >>> Stopped(Stop.RAISES, "ComputeError").way()
            raises
        """
        return Stopped(self.stop)


class KernelFlags:
    """Give the optimization flags of an eager run with the cast pass and the type check off.

    A kernel test runs a node on data whose types the cast pass would change.
    With these flags, polars calls the kernel on the data as it is, and the
    test reads the type that the kernel gives with no cast.

    Example:
        >>> KernelFlags().uncast()._pyoptflags.type_coercion
        False
    """

    def uncast(self) -> pl.QueryOptFlags:
        """Give the flags.

        Returns:
            The flags.

        Example:
            >>> KernelFlags().uncast()._pyoptflags.type_check
            False
        """
        flags = pl.QueryOptFlags._eager()  # ruff: ignore[private-member-access]
        inner = flags._pyoptflags  # ruff: ignore[private-member-access]
        inner.type_coercion = False
        inner.type_check = False
        return flags


class PolarsAnswer:
    """Run one call of polars, and give what it gives or how it stops.

    A panic of polars reaches Python as a `PanicException`, which derives
    from `BaseException`, so the catch is that wide, and it lets an
    interrupt through. Polars warns for a deprecated cast, and the warning
    does not change the answer.

    Example:
        >>> PolarsAnswer().dtype(lambda: pl.Int8())
        Int8
        >>> PolarsAnswer().dtype(lambda: pl.Series(["a"]).cast(pl.Date).dtype) is None
        True
        >>> PolarsAnswer().answer(lambda: pl.Series(["a"]).cast(pl.Date).dtype)
        raises InvalidOperationError
    """

    STOPS: ClassVar[tuple[type[BaseException], ...]] = (
        KeyboardInterrupt,
        SystemExit,
        GeneratorExit,
    )

    def result[T](self, call: Callable[[], T]) -> T | Stopped:
        """Give the result of a call, or how it stops.

        Args:
            call: The call.

        Returns:
            The result, or a `Stopped` with the name of the class of the error.

        Example:
            >>> PolarsAnswer().result(lambda: 1 // 0)
            raises ZeroDivisionError
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return call()
        except BaseException as error:
            if isinstance(error, self.STOPS):
                raise
            if type(error).__name__ == "PanicException":
                return Stopped(Stop.PANICS)
            return Stopped(Stop.RAISES, type(error).__name__)

    def answer(self, call: Callable[[], pl.DataType]) -> DType | Stopped:
        """Give the data type that a call gives, or how it stops.

        Args:
            call: The call.

        Returns:
            The data type, or how the call stops.

        Example:
            >>> PolarsAnswer().answer(lambda: pl.Series([[True]]).list.eval(pl.element() + 1).dtype)
            List(Int32)
        """
        result = self.result(call)
        return result if isinstance(result, Stopped) else DType.of(result)

    def dtype(self, call: Callable[[], pl.DataType]) -> DType | None:
        """Give the data type that a call gives, or `None` where it fails.

        The planner type `Unknown` of polars holds no more, so it reads as
        `Unknown(Any)` (`DType.of`).

        Args:
            call: The call.

        Returns:
            The data type, or `None`.

        Example:
            >>> PolarsAnswer().dtype(lambda: pl.List(pl.Unknown()))
            List(Unknown)
        """
        answer = self.answer(call)
        return None if isinstance(answer, Stopped) else answer

    def text(self, call: Callable[[], str]) -> str | None:
        """Give the text that a call gives, or `None` where it fails.

        Args:
            call: The call.

        Returns:
            The text, or `None`.

        Example:
            >>> PolarsAnswer().text(lambda: "plan")
            'plan'
        """
        text = self.result(call)
        return None if isinstance(text, Stopped) else text


class RuleAnswer:
    """Run one rule, and give what it gives or how it says that polars stops.

    A rule says that polars stops with `PolarsRefusal`. A rule that cannot
    answer raises `UndecidedError`, which passes through, so that a test
    decides what to do with it.

    Example:
        >>> RuleAnswer().dtype(lambda: Supertype().require(DType("Date"), DType("Boolean"))) is None
        True
        >>> RuleAnswer().result(lambda: Supertype().require(DType("Date"), DType("Boolean")))
        raises SchemaError
    """

    def result[T](self, call: Callable[[], T]) -> T | Stopped:
        """Give the result of a rule, or how it says that polars stops.

        Args:
            call: The rule.

        Returns:
            The result, or a `Stopped`.

        Example:
            >>> RuleAnswer().result(lambda: Supertype().require(DType("Int8"), DType("UInt8")))
            Int16
        """
        try:
            return call()
        except PolarsRefusal as refusal:
            return Stopped.of(refusal)

    def dtype(self, call: Callable[[], DType]) -> DType | None:
        """Give the data type of a rule, or `None` where it says that polars refuses.

        Args:
            call: The rule.

        Returns:
            The data type, or `None`.

        Example:
            >>> RuleAnswer().dtype(lambda: DType("Int8"))
            Int8
        """
        answer = self.result(call)
        return None if isinstance(answer, Stopped) else answer


class DebugRendering:
    """Give the text that polars prints for a data type in `explain()`.

    The text is the debug form of the data type in Rust. It differs from the
    rendering of Python: a decimal is `Decimal(10, 2)`, a microsecond is `μs`,
    an array is `Array(Int64, 2)`, and an `Enum` hides its categories.

    Example:
        >>> DebugRendering().of(DType.of(pl.Datetime("us", "UTC")))
        "Datetime('μs', 'UTC')"
    """

    UNITS: ClassVar[dict[object, str]] = {"ns": "ns", "us": "μs", "ms": "ms"}

    def of(self, dtype: DType) -> str:
        """Give the text.

        Args:
            dtype: The data type.

        Returns:
            The text.

        Example:
            >>> DebugRendering().of(DType.of(pl.List(pl.Array(pl.Decimal(10, 2), 3))))
            'List(Array(Decimal(10, 2), 3))'
        """
        args = dtype.args
        renders: dict[str, Callable[[], str]] = {
            "Duration": lambda: f"Duration('{self.UNITS[args[0]]}')",
            "Datetime": lambda: self._datetime(dtype),
            "Decimal": lambda: f"Decimal({args[0]}, {args[1]})",
            "Array": lambda: f"Array({self.of(dtype.inner())}, {args[1]})",
            "List": lambda: f"List({self.of(dtype.inner())})",
            "Struct": lambda: "Struct({" + self._fields(dtype) + "})",
            "Enum": lambda: "Enum([...])",
        }
        render = renders.get(dtype.name)
        return dtype.name if render is None else render()

    def _datetime(self, dtype: DType) -> str:
        """Give the text of a `Datetime`, with its zone where it has one.

        Args:
            dtype: The `Datetime` type.

        Returns:
            The text.

        Example:
            >>> DebugRendering()._datetime(DType("Datetime", ("ms", None)))
            "Datetime('ms')"
        """
        unit, zone = dtype.args
        if zone is None:
            return f"Datetime('{self.UNITS[unit]}')"
        return f"Datetime('{self.UNITS[unit]}', '{zone}')"

    def _fields(self, dtype: DType) -> str:
        """Give the text of the fields of a struct.

        Args:
            dtype: The struct type.

        Returns:
            The fields, separated by commas.

        Example:
            >>> DebugRendering()._fields(DType.of(pl.Struct({"a": pl.Int8()})))
            "'a': Int8"
        """
        return ", ".join(f"'{name}': {self.of(field)}" for name, field in dtype.fields())


class JoinCalls:
    """Run the calls of polars that compute each variant of the join, and give their answers.

    Each call gives a data type, or nothing where polars refuses it.

    Example:
        >>> JoinCalls().relaxed_concat([pl.Int8(), pl.UInt8()])
        Int16
    """

    def relaxed_concat(self, dtypes: list[pl.DataType]) -> DType | None:
        """Give the type of a column in the relaxed concat of one frame of each type.

        Args:
            dtypes: The type of the column in each frame.

        Returns:
            The type, or `None`.

        Example:
            >>> JoinCalls().relaxed_concat([pl.Date(), pl.Boolean()]) is None
            True
        """
        frames = [pl.LazyFrame(schema={"x": dtype}) for dtype in dtypes]
        return PolarsAnswer().dtype(
            lambda: pl.concat(frames, how="vertical_relaxed").collect_schema()["x"],
        )

    def planned(self, dtype: pl.DataType, expression: pl.Expr) -> DType | None:
        """Give the planner type of an expression over a frame with one column `x`.

        Args:
            dtype: The type of `x`.
            expression: The expression.

        Returns:
            The type, or `None`.

        Example:
            >>> when = pl.when(True).then(pl.col("x")).otherwise(pl.lit(300))
            >>> JoinCalls().planned(pl.Int8(), when)
            Int16
        """
        frame = pl.LazyFrame(schema={"x": dtype})
        return PolarsAnswer().dtype(
            lambda: frame.select(expression.alias("o")).collect_schema()["o"],
        )

    def explained(self, dtypes: tuple[pl.DataType, pl.DataType], expression: pl.Expr) -> str | None:
        """Give the text of `explain()` for an expression over two columns `x0` and `x1`.

        Args:
            dtypes: The types of `x0` and `x1`.
            expression: The expression.

        Returns:
            The text, or `None` where polars refuses the query.

        Example:
            >>> plan = JoinCalls().explained((pl.Int8(), pl.Int8()), pl.max_horizontal("x0", "x1"))
            >>> "cast" in plan
            False
        """
        frame = pl.LazyFrame(schema={"x0": dtypes[0], "x1": dtypes[1]})
        return PolarsAnswer().text(lambda: frame.select(expression).explain())

    def full_join(self, left: pl.DataType, right: pl.DataType) -> DType | None:
        """Give the type of the key of a full join that coalesces its keys, from a run on no rows.

        Args:
            left: The type of the left key.
            right: The type of the right key.

        Returns:
            The type, or `None`.

        Example:
            >>> JoinCalls().full_join(pl.UInt8(), pl.Int8())
            Int16
        """
        joined = pl.LazyFrame(schema={"k": left}).join(
            pl.LazyFrame(schema={"k": right}),
            on="k",
            how="full",
            coalesce=True,
        )
        return PolarsAnswer().dtype(lambda: joined.collect().schema["k"])


class CastPassCheck:
    """Compare the casts that the cast pass of a function inserts with the casts of the rules.

    The cast pass of a function with supertyping folds the supertype of its
    inputs with the flags of the function. It refuses the query where there is
    no supertype, or where `CastCheck` refuses the cast of an input to it. It
    casts each input whose type differs, and leaves two inputs of the same type
    alone.

    Example:
        >>> text = 'SELECT [col("x0").strict_cast(Int16).max_horizontal([col("x1")])]'
        >>> check = CastPassCheck(SupertypeFlag.default())
        >>> check.problem((DType("Int8"), DType("Int16")), text) is None
        True
    """

    def __init__(self, flags: SupertypeFlag) -> None:
        """Keep the flags of the function.

        Args:
            flags: The flags of the join.

        Example:
            >>> pair = (DType("Int8"), DType("String"))
            >>> no_text = CastPassCheck(SupertypeFlag(0))
            >>> no_text.problem(pair, None) is None
            True
        """
        self._supertype = Supertype(flags)

    def problem(self, inputs: tuple[DType, DType], text: str | None) -> str | None:
        """Give what differs between the plan of polars and the rules, or `None`.

        Args:
            inputs: The types of the inputs `x0` and `x1`.
            text: The text of `explain()`, or `None` where polars refuses the query.

        Returns:
            A description of the difference, or `None`.

        Example:
            >>> CastPassCheck(SupertypeFlag.default()).problem((DType("Int8"), DType("Int8")), None)
            'polars refuses, the rules give Int8'
        """
        joined = self._joined(inputs)
        if joined is None and text is None:
            return None
        if joined is None or text is None:
            return f"polars refuses, the rules give {joined!r}" if text is None else "polars plans"
        for index, dtype in enumerate(inputs):
            if not self._cast_as_planned(f'col("x{index}")', dtype, joined, text):
                return f"the rules cast x{index} to {joined!r}, polars plans {text.splitlines()[0]}"
        return None

    def _cast_as_planned(self, column: str, dtype: DType, joined: DType, text: str) -> bool:
        """Say whether the plan casts one input exactly where the rules cast it, to the same type.

        Args:
            column: The column in the text of the plan, for example `col("x0")`.
            dtype: The type of the input.
            joined: The type that the rules cast the inputs to.
            text: The text of `explain()`.

        Returns:
            Whether the plan agrees.

        Example:
            >>> check = CastPassCheck(SupertypeFlag.default())
            >>> check._cast_as_planned('col("x0")', DType("Int8"), DType("Int8"), "")
            True
        """
        prefixes = (f"{column}.strict_cast(", f"{column}.cast(")
        casts = any(prefix in text for prefix in prefixes)
        if dtype.same(joined):
            return not casts
        rendering = DebugRendering().of(joined)
        return any(f"{prefix}{rendering})" in text for prefix in prefixes)

    def _joined(self, inputs: tuple[DType, DType]) -> DType | None:
        """Give the type that the rules cast the inputs to, or `None` where the cast pass refuses.

        Args:
            inputs: The types of the two inputs.

        Returns:
            The supertype, or `None`.

        Example:
            >>> one = DType.of(pl.Struct({"a": pl.Int8()}))
            >>> two = DType.of(pl.Struct({"b": pl.Int8()}))
            >>> CastPassCheck(SupertypeFlag.default())._joined((one, two)) is None
            True
        """
        left, right = inputs
        if left.same(right):
            return left
        joined = self._supertype.of(left, right)
        if joined is None or Is.unknown_any(joined):
            return None
        refused = any(CastCheck().of(dtype, joined) is False for dtype in inputs)
        return None if refused else joined


@dataclass(frozen=True)
class Input:
    """Hold one input of a node: its name, its leaf for the rules, and its expression for polars.

    Example:
        >>> column = Inputs().all()[1]
        >>> column.label, str(column.expression)
        ('Int8/one value', 'col("c0").first()')
    """

    label: str
    leaf: Leaf
    expression: pl.Expr


class Inputs:
    """Give the inputs of the tests: each type of `RuleKinds` in each row form, and the literals.

    The column of the type at place `i` of `RuleKinds` is `c{i}`. The literals
    are the classes of the census, each direct and derived.

    Example:
        >>> len(Inputs().all())
        236
        >>> Inputs().all()[-1].label
        'lit null/derived'
    """

    FORMS: ClassVar[dict[RowForm, Callable[[pl.Expr], pl.Expr]]] = {
        RowForm.PER_ROW: lambda column: column,
        RowForm.ONE_VALUE: lambda column: column.first(),
        RowForm.OTHER_LENGTH: lambda column: column.head(2),
        RowForm.ONE_ROW: lambda column: column.head(1),
    }

    def all(self) -> list[Input]:
        """Give every input: the columns, then the literals.

        Returns:
            The inputs.

        Example:
            >>> Inputs().all()[0].label
            'Int8/per row'
        """
        return self.columns() + self.literals()

    def columns(self) -> list[Input]:
        """Give each type as a column in each row form.

        Returns:
            The inputs.

        Example:
            >>> Inputs().columns()[3].leaf
            ColumnLeaf(dtype=Int8, form=<RowForm.ONE_ROW: 'one row'>, nulls=False, planned=None)
        """
        inputs: list[Input] = []
        for index, polars_dtype in enumerate(RuleKinds().dtypes()):
            dtype = DType.of(polars_dtype)
            column = pl.col(f"c{index}")
            for form, build in self.FORMS.items():
                label = f"{dtype!r}/{form.value}"
                inputs.append(Input(label, ColumnLeaf(dtype, form), build(column)))
        return inputs

    def with_text(self, text: str) -> list[Input]:
        """Give every input, with a literal of one text in place of the text literals of the census.

        A node that parses text, such as `str.to_date`, raises on a literal
        text that its format does not parse, at every run. The rules read the
        type of a text and not its value, so a test of such a node takes a
        text that the format parses.

        Args:
            text: The text.

        Returns:
            The inputs.

        Example:
            >>> [entry.label for entry in Inputs().with_text("12:00")][-6:-4]
            ['lit text/direct', 'lit text/derived']
        """
        direct = Input("lit text/direct", LiteralLeaf(text), pl.lit(text))
        leaf = LiteralLeaf(text, Operator.PLUS, "")
        derived = Input("lit text/derived", leaf, pl.lit(text) + pl.lit(""))
        replaced = {"lit str/direct": direct, "lit str/derived": derived}
        return [replaced.get(entry.label, entry) for entry in self.all()]

    def labelled(self, label: str) -> Input:
        """Give the input of a label, such as a typical input of a node.

        Args:
            label: The label, for example `Float64/per row` or `lit float/direct`.

        Returns:
            The input.

        Raises:
            KeyError: No input has the label.

        Example:
            >>> str(Inputs().labelled("lit float/direct").expression)
            'dyn float: 1.5'
            >>> Inputs().labelled("Int8/twice")
            Traceback (most recent call last):
            ...
            KeyError: 'no input has the label Int8/twice'
        """
        for entry in self.all():
            if entry.label == label:
                return entry
        message = f"no input has the label {label}"
        raise KeyError(message)

    def literals(self) -> list[Input]:
        """Give each literal class of the census, direct and derived.

        Returns:
            The inputs.

        Example:
            >>> Inputs().literals()[1].label
            'lit int 1/derived'
        """
        inputs: list[Input] = []
        for literal in OutcomeSpace.CLASSES:
            for derived in (False, True):
                outcome = LiteralOutcome(literal, derived=derived)
                leaf = CensusLeaves().leaf(outcome)
                inputs.append(Input(outcome.label(), leaf, outcome.expression(null=False)))
        return inputs


class Values(enum.Enum):
    """Name the samples of values of the frames of the tests.

    The first five are the samples of the census. The next six hold one text
    in every row: text that parses as a number, as a date, as a time, as a
    datetime with the offset of a zone, or as a duration, and the text `raise`
    that `ambiguous` of a zone accepts. No sample of the census holds such
    text in every row, so the census never sees a cast of text to a number,
    or a parse of text, that runs. `FIRST` holds the first typical value of
    each type in every row, so a cast that only small values fit, such as a
    cast to an `Enum` of fewer categories, runs. `NAN_FIRST` holds a NaN as
    the first value of each float, so a kernel that reads the first float
    meets a NaN. `TEN` holds the value ten in every row, so a kernel that
    reads a number as an index or a base finds one that fits (`TenValues`).

    Example:
        >>> Values.SMALL.census(), Values.NUMBER_TEXT.census() is None
        (<Sample.SMALL: 'small'>, True)
    """

    TYPICAL = "typical"
    SMALL = "small"
    LARGE = "large"
    TEXT = "text"
    SHAPES = "shapes"
    NUMBER_TEXT = "number text"
    DATE_TEXT = "date text"
    TIME_TEXT = "time text"
    ZONED_TEXT = "zoned text"
    DURATION_TEXT = "duration text"
    AMBIGUOUS_TEXT = "ambiguous text"
    FIRST = "first"
    NAN_FIRST = "nan first"
    TEN = "ten"

    def text(self) -> str | None:
        """Give the text of every row of a sample of the tests alone.

        Returns:
            The text, or `None` for a sample of the census.

        Example:
            >>> Values.DATE_TEXT.text(), Values.TEXT.text() is None
            ('2020-01-01', True)
        """
        texts = {
            Values.NUMBER_TEXT: "1",
            Values.DATE_TEXT: "2020-01-01",
            Values.TIME_TEXT: "12:00",
            Values.ZONED_TEXT: "2020-01-01T00:00:00+00:00",
            Values.DURATION_TEXT: "1d",
            Values.AMBIGUOUS_TEXT: "raise",
        }
        return texts.get(self)

    def census(self) -> Sample | None:
        """Give the sample of the census of the same name.

        Returns:
            The sample, or `None` for a sample of the tests alone.

        Example:
            >>> Values.TEXT.census()
            <Sample.TEXT: 'text'>
        """
        names = {sample.value: sample for sample in Sample}
        return names.get(self.value)


class LiveFrames:
    """Build the frame of each sample of values: a column of each type, in two groups.

    The group key `g` gives a group of 3 rows and a group of 2 rows. Each
    column repeats the three values of its sample. The typical values are those
    of the tests of the kernels (`KernelSamples`), whose text fits each `Enum`
    of the tests. A sample that has no values for a type takes the typical
    values.

    Example:
        >>> frame = LiveFrames().frame(Values.TYPICAL)
        >>> frame.height, frame["g"].to_list()
        (5, [1, 1, 1, 2, 2])
    """

    GROUPS: ClassVar[list[int]] = [1, 1, 1, 2, 2]
    BUILT: ClassVar[dict[Values, pl.DataFrame]] = {}

    def frame(self, values: Values) -> pl.DataFrame:
        """Give the frame of a sample, built once for all the tests.

        Args:
            values: The sample.

        Returns:
            The frame.

        Example:
            >>> LiveFrames().frame(Values.SMALL)["c0"].to_list()
            [0, 0, 0, 0, 0]
        """
        if values not in self.BUILT:
            columns: list[pl.Series] = [pl.Series("g", self.GROUPS)]
            for index, dtype in enumerate(RuleKinds().dtypes()):
                columns.append(self._column(f"c{index}", dtype, values))
            self.BUILT[values] = pl.DataFrame(columns)
        return self.BUILT[values]

    def _column(self, name: str, dtype: pl.DataType, values: Values) -> pl.Series:
        """Give the column of one type for a sample, or of its typical values where polars refuses.

        Args:
            name: The name of the column.
            dtype: The type.
            values: The sample.

        Returns:
            The column of 5 rows.

        Example:
            >>> LiveFrames()._column("x", pl.Enum(["a", "b"]), Values.NUMBER_TEXT).to_list()
            ['a', 'b', 'a', 'a', 'b']
        """
        if values is Values.TEN:
            ten = TenValues().column(name, dtype, len(self.GROUPS))
            return self._column(name, dtype, Values.TYPICAL) if ten is None else ten
        sample = self._values(dtype, values)
        rows = [sample[row % len(sample)] for row in range(len(self.GROUPS))]
        answer = PolarsAnswer().result(lambda: pl.Series(name, rows, dtype=dtype))
        if isinstance(answer, pl.Series):
            return answer
        return self._column(name, dtype, Values.TYPICAL)

    def _values(self, dtype: pl.DataType, values: Values) -> list[object]:
        """Give the three values of a type in a sample.

        Args:
            dtype: The type.
            values: The sample.

        Returns:
            The values.

        Example:
            >>> LiveFrames()._values(pl.Struct({"a": pl.String()}), Values.NUMBER_TEXT)
            [{'a': '1'}, {'a': '1'}, {'a': '1'}]
            >>> LiveFrames()._values(pl.Float32(), Values.NAN_FIRST)
            [nan, 2.0, 3.0]
        """
        typical = KernelSamples().typical(dtype)
        census = values.census()
        if census is not None and census is not Sample.TYPICAL:
            return list(AnyValues().of(dtype, census))
        text = values.text()
        if text is not None:
            return [self._with_text(value, text) for value in typical]
        if values is Values.FIRST:
            return typical[:1]
        if values is Values.NAN_FIRST and Is.float(DType.of(dtype)):
            return [math.nan, *typical[1:]]
        return typical

    def _with_text(self, value: object, text: str) -> object:
        """Give a value with each text in it replaced by one text.

        Args:
            value: The value, which can be a list or a struct.
            text: The text.

        Returns:
            The value.

        Example:
            >>> LiveFrames()._with_text([{"a": "x", "b": 2}], "1")
            [{'a': '1', 'b': 2}]
        """
        if isinstance(value, str):
            return text
        if isinstance(value, list):
            return [self._with_text(item, text) for item in value]
        if isinstance(value, dict):
            return {key: self._with_text(item, text) for key, item in value.items()}
        return value


class TenValues:
    """Give a column of the sample `TEN`: the value ten in every row, in the unit of its type.

    A number holds 10, a temporal type holds 10 of its unit, and a `Boolean`
    holds `true`, so each of them reads as the index or the base 10. The text
    is `10`, which parses in the base 10, and the bytes are twelve long, so
    that the index 10 fits. An `Enum` holds its last category. A struct holds
    a null in each field, so its text is null. Every other type keeps its
    typical values, and so does a `Decimal` that cannot hold 10.

    Example:
        >>> TenValues().column("x", pl.Date(), 2).to_list()
        [datetime.date(1970, 1, 11), datetime.date(1970, 1, 11)]
        >>> TenValues().column("x", pl.Struct({"a": pl.Int8()}), 1).to_list()
        [{'a': None}]
    """

    BYTES: ClassVar[bytes] = b"0123456789ab"

    def column(self, name: str, dtype: pl.DataType, rows: int) -> pl.Series | None:
        """Give the column, or `None` where the type keeps its typical values.

        Args:
            name: The name of the column.
            dtype: The type.
            rows: The number of rows.

        Returns:
            The column.

        Example:
            >>> TenValues().column("x", pl.Enum(["a", "b"]), 1).to_list()
            ['b']
            >>> TenValues().column("x", pl.List(pl.Int8()), 1) is None
            True
        """
        if dtype.is_numeric() or dtype.is_temporal() or isinstance(dtype, pl.Boolean):
            cast = pl.Series(name, [10] * rows).cast(dtype, strict=False)
            return None if cast.has_nulls() else cast
        values: dict[type[pl.DataType], object] = {pl.String: "10", pl.Binary: self.BYTES}
        if type(dtype) in values:
            return pl.Series(name, [values[type(dtype)]] * rows, dtype=dtype)
        if isinstance(dtype, pl.Enum):
            return pl.Series(name, [dtype.categories[-1]] * rows, dtype=dtype)
        if isinstance(dtype, pl.Struct):
            empty = {field.name: None for field in dtype.fields}
            return pl.Series(name, [empty] * rows, dtype=dtype)
        return None


class PolarsCell:
    """Give the answers of polars for one node, in a `select` and in `agg` over two groups.

    A run gives the type of the first sample that runs, as the census does.
    It runs in the flags of an eager frame, which keep the fold of literals
    and leave out the other optimizations. The groups of `agg` keep their
    order, so that a run that fails in more than one group stops at the same
    group each time.

    Example:
        >>> cell = PolarsCell((pl.col("c0") + 1).alias("o"))
        >>> cell.planner(agg=True), cell.run(agg=False, every=True)
        (List(Int8), Int8)
    """

    def __init__(self, expression: pl.Expr) -> None:
        """Keep the node, named `o`.

        Args:
            expression: The node.

        Example:
            >>> PolarsCell(pl.col("c0").alias("o"))._expression.meta.output_name()
            'o'
        """
        self._expression = expression
        self._frames = LiveFrames()

    def planner(self, *, agg: bool) -> DType | Stopped:
        """Give the planner type of the node in a context.

        Args:
            agg: Whether to read `agg`.

        Returns:
            The type, or how polars stops.

        Example:
            >>> PolarsCell(pl.col("c13").abs().alias("o")).planner(agg=False)
            String
        """
        query = self._query(self._frames.frame(Values.TYPICAL), agg=agg)
        return PolarsAnswer().answer(lambda: query.collect_schema()["o"])

    def run(
        self,
        *,
        agg: bool,
        every: bool,
        first: Values = Values.TYPICAL,
        skip: frozenset[Values] = frozenset(),
    ) -> DType | Stopped:
        """Give the run type of the node in a context, on the first sample that runs.

        Args:
            agg: Whether to run `agg`.
            every: Whether to try every sample, and not only the first one.
            first: The sample that the run tries first.
            skip: The samples that the run does not try.

        Returns:
            The type, or how the run on the first sample stops.

        Example:
            >>> PolarsCell(pl.col("c13").abs().alias("o")).run(agg=False, every=False)
            raises InvalidOperationError
        """
        eager = pl.QueryOptFlags._eager()  # ruff: ignore[private-member-access]
        answers: list[DType | Stopped] = []
        others = [values for values in Values if values is not first and values not in skip]
        for values in [first, *others] if every else [first]:
            query = self._query(self._frames.frame(values), agg=agg)
            answer = PolarsAnswer().answer(functools.partial(self._dtype, query, eager))
            if isinstance(answer, DType):
                return answer
            answers.append(answer)
        return answers[0]

    def types(self, *, agg: bool, skip: frozenset[Values] = frozenset()) -> set[DType]:
        """Give the run type of the node in a context, on each sample that runs.

        Args:
            agg: Whether to run `agg`.
            skip: The samples that the run does not try.

        Returns:
            The types, one for each sample that runs.

        Example:
            >>> PolarsCell(pl.col("c0").abs().alias("o")).types(agg=False)
            {Int8}
        """
        eager = pl.QueryOptFlags._eager()  # ruff: ignore[private-member-access]
        types: set[DType] = set()
        for values in [values for values in Values if values not in skip]:
            query = self._query(self._frames.frame(values), agg=agg)
            answer = PolarsAnswer().answer(functools.partial(self._dtype, query, eager))
            if isinstance(answer, DType):
                types.add(answer)
        return types

    def _dtype(self, query: pl.LazyFrame, flags: pl.QueryOptFlags) -> pl.DataType:
        """Run a query, and give the type of its column `o`.

        Args:
            query: The query.
            flags: The optimization flags of the run.

        Returns:
            The type.

        Example:
            >>> query = pl.LazyFrame({"o": [1]})
            >>> PolarsCell(pl.col("o"))._dtype(query, pl.QueryOptFlags())
            Int64
        """
        return query.collect(optimizations=flags)["o"].dtype

    def _query(self, frame: pl.DataFrame, *, agg: bool) -> pl.LazyFrame:
        """Give the query of the node in a context.

        Args:
            frame: The frame.
            agg: Whether to put the node in `agg`.

        Returns:
            The query.

        Example:
            >>> cell = PolarsCell(pl.col("c0").sum().alias("o"))
            >>> cell._query(LiveFrames().frame(Values.TYPICAL), agg=True).collect_schema()["o"]
            Int64
        """
        if agg:
            return frame.lazy().group_by("g", maintain_order=True).agg(self._expression)
        return frame.lazy().select(self._expression)


class CellCheck:
    """Compare the four answers of the rules with the four answers of polars, for one cell.

    The rules read the options of the node from the tree that polars builds
    for the call, so one definition serves every value of an option. Where
    the rules refuse a run, the run of polars reads one sample, `refused`.
    A node that casts text to a number with a strict cast before its kernel
    reads text that parses there: other text fails the cast first, and the
    class of that error is a fact of the values. A node whose length a count
    decides skips the samples of large values (`skip`).

    Example:
        >>> from astrolabe._rules.definitions.elementwise import Elementwise
        >>> left = Inputs().all()[0]
        >>> CellCheck(Elementwise.ABS, lambda x: x.abs()).problems([left])
        []
    """

    OUT_OF_PROCESS: ClassVar[frozenset[Stop]] = frozenset({Stop.CRASHES, Stop.HANGS})
    GROUPS: ClassVar[GroupBy] = GroupBy((3, 2))

    def __init__(
        self,
        definition: Definition,
        build: Callable[..., pl.Expr],
        refused: Values = Values.TYPICAL,
        skip: frozenset[Values] = frozenset(),
    ) -> None:
        """Keep the definition of the node, the call of polars that builds it, and the samples.

        Args:
            definition: The definition.
            build: The call, with one expression for each input.
            refused: The sample that a run reads where the rules refuse it.
            skip: The samples that a run does not try, such as large counts
                that take more memory than the machine holds.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> CellCheck(Aggregations.SUM, lambda x: x.sum())._definition.identity.function
            'Sum'
        """
        self._definition = definition
        self._build = build
        self._refused = refused
        self._skip = skip

    @classmethod
    def label(cls, cell: tuple[Input, ...]) -> str:
        """Give the name of a cell, which names it in a report and in the identifier of a test.

        Args:
            cell: One input for each input of a node.

        Returns:
            The labels of the inputs.

        Example:
            >>> CellCheck.label(tuple(Inputs().all()[:2]))
            'Int8/per row, Int8/one value'
        """
        return ", ".join(entry.label for entry in cell)

    def problems(self, inputs: list[Input]) -> list[str]:
        """Give each answer where polars and the rules differ.

        Args:
            inputs: One input for each input of the node.

        Returns:
            A description of each answer that differs.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> text = [Inputs().all()[52]]
            >>> CellCheck(Aggregations.SUM, lambda x: x.sum()).problems(text)
            []
        """
        leaves = [entry.leaf for entry in inputs]
        node = self._build(*(entry.expression for entry in inputs))
        tree = json.loads(node.meta.serialize(format="json"))
        definition = self._definition.at(NodeOptions(tree))
        rules = NodeEvaluation(definition, leaves, self.GROUPS).answers()
        polars = self._polars(PolarsCell(node.alias("o")), rules)
        expected = [self._expected(answer) for answer in self._ordered(rules)]
        names = ("planner select", "planner agg", "run select", "run agg")
        cell = self.label(tuple(inputs))
        problems = []
        for name, given, wanted in zip(names, polars, expected, strict=True):
            if given is not None and given != wanted:
                problems.append(f"{cell}: {name}: polars {given}, rules {wanted}")
        return problems

    def _polars(self, cell: PolarsCell, rules: CellAnswers) -> list[DType | Stopped | None]:
        """Give the four answers of polars, or `None` for a run that does not run here.

        A context whose planner refuses refuses its run too, with no run. A
        run where the rules give a type tries each sample, and a run where they
        refuse tries the typical sample.

        Args:
            cell: The node in polars.
            rules: The answers of the rules.

        Returns:
            The answers, in the order of the census.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> text = Inputs().all()[52]
            >>> rules = NodeEvaluation(Elementwise.ABS, [text.leaf], CellCheck.GROUPS).answers()
            >>> CellCheck(Elementwise.ABS, lambda x: x.abs())._polars(
            ...     PolarsCell(text.expression.abs().alias("o")),
            ...     rules,
            ... )
            [String, List(String), raises InvalidOperationError, raises InvalidOperationError]
        """
        planners = [cell.planner(agg=False), cell.planner(agg=True)]
        runs: list[DType | Stopped | None] = []
        for agg, planner, wanted in zip(
            (False, True), planners, self._ordered(rules)[2:], strict=True
        ):
            runs.append(self._run(cell, planner, wanted, agg=agg))
        return [*planners, *runs]

    def _run(
        self,
        cell: PolarsCell,
        planner: DType | Stopped,
        wanted: Answer,
        *,
        agg: bool,
    ) -> DType | Stopped | None:
        """Give the run answer of polars in one context, or `None` where it does not run here.

        Args:
            cell: The node in polars.
            planner: The planner answer of polars in that context.
            wanted: The run answer of the rules.
            agg: Whether the context is `agg`.

        Returns:
            The answer, or `None`. Where the rules say that the values decide
            the type, the answer is `None` if polars gives two types or more
            over the samples, and the one type otherwise.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> cell = PolarsCell(pl.col("c0").alias("o"))
            >>> crash = Refused(Stop.CRASHES)
            >>> CellCheck(Elementwise.ABS, lambda x: x)._run(cell, DType("Int8"), crash, agg=False)
        """
        if isinstance(planner, Stopped):
            return planner
        if isinstance(wanted, Refused) and wanted.stop in self.OUT_OF_PROCESS:
            return None
        if isinstance(wanted, Unchecked) and wanted.cause is Undecided.VALUES:
            types = cell.types(agg=agg, skip=self._skip)
            return None if len(types) > 1 else cell.run(agg=agg, every=True, skip=self._skip)
        if isinstance(wanted, DType):
            return cell.run(agg=agg, every=True, skip=self._skip)
        return cell.run(agg=agg, every=False, first=self._refused)

    def _ordered(self, answers: CellAnswers) -> list[Answer]:
        """Give the four answers of the rules in the order of the census.

        Args:
            answers: The answers.

        Returns:
            The planner types, then the run types, each in a `select` and then in `agg`.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> from astrolabe._rules.node_evaluation import CellAnswers
            >>> int8 = DType("Int8")
            >>> check = CellCheck(Elementwise.ABS, lambda x: x)
            >>> len(check._ordered(CellAnswers(int8, int8, int8, int8)))
            4
        """
        return [answers.planner_select, answers.planner_agg, answers.run_select, answers.run_agg]

    def _expected(self, answer: Answer) -> DType | Stopped | str:
        """Give an answer of the rules in the form of an answer of polars.

        Args:
            answer: The answer of the rules.

        Returns:
            The type, how polars stops, or the question that the rules cannot answer.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> from astrolabe._rules.failures import ErrorClass
            >>> refused = Refused(Stop.RAISES, ErrorClass.SHAPE)
            >>> CellCheck(Elementwise.ABS, lambda x: x)._expected(refused)
            raises ShapeError
        """
        if isinstance(answer, DType):
            return answer
        if isinstance(answer, Unchecked):
            return repr(answer)
        error = None if answer.error is None else answer.error.value
        return Stopped(answer.stop, error)


class TemplateCall:
    """Build a node of polars from the tree of a call, with any input at each place.

    Some calls of polars fix an input: `str.to_date` always passes the
    literal `raise` as its second input. The tree of the call takes any input
    at that place (`NodeTemplate`), so a test can vary every input of the
    node.

    Example:
        >>> call = TemplateCall("pl.col('x').str.to_date('%Y-%m-%d')")
        >>> str(call(pl.col("a"), pl.col("b")))
        'col("a").str.strptime([col("b")])'
    """

    def __init__(self, code: str) -> None:
        """Read the tree of the call.

        Args:
            code: The call, over the column `x`.

        Example:
            >>> TemplateCall("pl.col('x').abs()")._template.inputs()
            1
        """
        self._template = NodeTemplate("test", code)

    def __call__(self, *inputs: pl.Expr) -> pl.Expr:
        """Build the node with the inputs.

        Args:
            *inputs: One expression for each input, in the order of polars.

        Returns:
            The node.

        Example:
            >>> str(TemplateCall("pl.col('x').abs()")(pl.lit(1)))
            'dyn int: 1.abs()'
        """
        return self._template.build(list(inputs))


class PairSample:
    """Give a sample of about 500 pairs of inputs of a node of two inputs.

    The full product of the inputs holds 55,696 pairs. The sample puts each
    input at the left beside one typical input at the right, and each input
    at the right beside one typical input at the left. Then it adds pairs
    spread over the whole product, at fixed places so that each run checks
    the same pairs.

    Example:
        >>> inputs = Inputs().all()
        >>> len(PairSample(inputs, inputs[52], inputs[0]).pairs())
        500
    """

    SPREAD: ClassVar[int] = 28

    def __init__(self, inputs: list[Input], left: Input, right: Input) -> None:
        """Keep the inputs and the typical input at each place.

        Args:
            inputs: The inputs that each place takes.
            left: The typical input at the left.
            right: The typical input at the right.

        Example:
            >>> inputs = Inputs().all()
            >>> PairSample(inputs, inputs[0], inputs[1])._left.label
            'Int8/per row'
        """
        self._inputs = inputs
        self._left = left
        self._right = right

    def pairs(self) -> list[tuple[Input, Input]]:
        """Give the pairs.

        Returns:
            The pairs, each a left input and a right input.

        Example:
            >>> inputs = Inputs().all()
            >>> pairs = PairSample(inputs, inputs[52], inputs[0]).pairs()
            >>> pairs[0][0].label, pairs[0][1].label
            ('Int8/per row', 'Int8/per row')
        """
        inputs = self._inputs
        count = len(inputs)
        pairs = [(entry, self._right) for entry in inputs]
        pairs.extend((self._left, entry) for entry in inputs)
        for index in range(self.SPREAD):
            left = inputs[(index * 37 + 5) % count]
            right = inputs[(index * 101 + 11) % count]
            pairs.append((left, right))
        return pairs


class NodeSample:
    """Give the sample of the cells of one node: about 500, each in `select` and in `agg`.

    The full product of the inputs of a node grows with the power of its
    number of inputs, so a test takes a sample. A node of one input takes
    every input (`Inputs`): each type of `RuleKinds` in each of the four row
    forms, and each literal class of the census, direct and derived. That is
    236 cells, each in `select` and in `agg`. A node of two inputs takes
    about 500 pairs (`PairSample`), beside a typical input at each place.
    The frames hold two groups of other sizes and samples of values that fit
    and that do not fit (`LiveFrames`).

    A call that differs from the main call of its node only by an option
    takes a smaller sample (`options`): each type in one value per row, and
    each literal class direct.

    Example:
        >>> len(NodeSample().cells(1)), len(NodeSample().options())
        (236, 63)
    """

    def __init__(self, inputs: list[Input] | None = None) -> None:
        """Keep the inputs that each place of a node takes.

        Args:
            inputs: The inputs, or `None` for every input of `Inputs`.

        Example:
            >>> len(NodeSample(Inputs().literals())._inputs)
            16
        """
        self._inputs = Inputs().all() if inputs is None else inputs

    def cells(self, arity: int, typical: tuple[Input, ...] = ()) -> list[tuple[Input, ...]]:
        """Give the cells of a node: one input for each of its places.

        Args:
            arity: The number of inputs of the node: one, two or three.
            typical: For a node of two or three inputs, the typical input at each place.

        Returns:
            The cells.

        Raises:
            ValueError: The node has more than three inputs, or a node of two or
                three inputs has no typical input at each place.

        Example:
            >>> inputs = Inputs().all()
            >>> len(NodeSample().cells(2, (inputs[40], inputs[220])))
            500
            >>> NodeSample().cells(3)
            Traceback (most recent call last):
            ...
            ValueError: no sample of 3 inputs with 0 typical inputs
        """
        if arity == 1:
            return [(entry,) for entry in self._inputs]
        if arity == 2 and len(typical) == 2:
            return list(PairSample(self._inputs, typical[0], typical[1]).pairs())
        if arity == 3 and len(typical) == 3:
            triple = (typical[0], typical[1], typical[2])
            return list(TripleSample(self._inputs, triple).cells())
        message = f"no sample of {arity} inputs with {len(typical)} typical inputs"
        raise ValueError(message)

    def options(self) -> list[tuple[Input]]:
        """Give the smaller sample of a call that differs from the main call by an option.

        Returns:
            The cells: each column in one value per row, then each literal class direct.

        Example:
            >>> NodeSample().options()[-1][0].label
            'lit null/direct'
        """
        per_row = [entry for entry in self._inputs if entry.label.endswith("/per row")]
        direct = [entry for entry in self._inputs if entry.label.endswith("/direct")]
        return [(entry,) for entry in per_row + direct]


class TripleSample:
    """Give a sample of about 500 cells of a node of three inputs.

    The full product of the inputs holds 13,144,256 cells. The sample puts
    each input at the first place beside the typical inputs at the two other
    places. At the second and at the third place, it puts each column of one
    value per row and each direct literal beside the typical inputs at the
    other places (`NodeSample.options`). Then it adds cells spread over the
    whole product, at fixed places so that each run checks the same cells.

    Example:
        >>> inputs = Inputs().all()
        >>> typical = (inputs[52], inputs[220], inputs[220])
        >>> len(TripleSample(inputs, typical).cells())
        500
    """

    SPREAD: ClassVar[int] = 138

    def __init__(self, inputs: list[Input], typical: tuple[Input, Input, Input]) -> None:
        """Keep the inputs and the typical input at each place.

        Args:
            inputs: The inputs that each place takes.
            typical: The typical input at each place.

        Example:
            >>> inputs = Inputs().all()
            >>> TripleSample(inputs, (inputs[0], inputs[1], inputs[2]))._typical[2].label
            'Int8/other length'
        """
        self._inputs = inputs
        self._typical = typical

    def cells(self) -> list[tuple[Input, Input, Input]]:
        """Give the cells.

        Returns:
            The cells, each one input for each place.

        Example:
            >>> inputs = Inputs().all()
            >>> cells = TripleSample(inputs, (inputs[52], inputs[220], inputs[220])).cells()
            >>> [entry.label for entry in cells[236]]
            ['String/per row', 'Int8/per row', 'lit int 1/direct']
        """
        first, second, third = self._typical
        cells = [(entry, second, third) for entry in self._inputs]
        options = [cell[0] for cell in NodeSample(self._inputs).options()]
        cells.extend((first, entry, third) for entry in options)
        cells.extend((first, second, entry) for entry in options)
        count = len(self._inputs)
        for index in range(self.SPREAD):
            left = self._inputs[(index * 37 + 5) % count]
            middle = self._inputs[(index * 101 + 11) % count]
            right = self._inputs[(index * 53 + 17) % count]
            cells.append((left, middle, right))
        return cells


class TreeColumns:
    """Hold columns of five rows, of the data types that the frame of `TreeFrame` does not hold.

    A tree that reaches a path of the rules for another data type adds one of
    these columns to the frame. Their values are ordinary values of the type.

    Example:
        >>> TreeColumns.DATES.dtype, TreeColumns.DATES.len()
        (Date, 5)
    """

    DATES: ClassVar[pl.Series] = pl.Series([dt.date(2024, 1, day) for day in range(8, 13)])
    MOMENTS: ClassVar[pl.Series] = pl.Series([dt.datetime(2024, 1, day, 6) for day in range(8, 13)])
    ZONED: ClassVar[pl.Series] = MOMENTS.dt.replace_time_zone("UTC")
    TIMES: ClassVar[pl.Series] = pl.Series([dt.time(hour) for hour in range(1, 6)])
    SPANS: ClassVar[pl.Series] = pl.Series([dt.timedelta(hours=hour) for hour in range(1, 6)])
    DECIMALS: ClassVar[pl.Series] = pl.Series([1, 2, 3, 4, 5], dtype=pl.Decimal(10, 2))
    CATEGORIES: ClassVar[pl.Series] = pl.Series(list("abcab"), dtype=pl.Categorical)
    ENUMS: ClassVar[pl.Series] = pl.Series(list("abcab"), dtype=pl.Enum(["a", "b", "c"]))
    BYTES: ClassVar[pl.Series] = pl.Series([b"a", b"b", b"c", b"d", b"e"])
    FLAGS: ClassVar[pl.Series] = pl.Series([True, False, True, False, True])
    SMALL: ClassVar[pl.Series] = pl.Series([1, 2, 3, 4, 5], dtype=pl.Int8)
    UNSIGNED: ClassVar[pl.Series] = pl.Series([1, 2, 3, 4, 5], dtype=pl.UInt8)
    FLOATS: ClassVar[pl.Series] = pl.Series([1.5, 2.5, 3.5, 4.5, 5.5], dtype=pl.Float32)
    NULLS: ClassVar[pl.Series] = pl.Series([None] * 5)
    NESTED: ClassVar[pl.Series] = pl.Series([[[1]], [[2]], [[3]], [[4]], [[5]]])
    TEXTS: ClassVar[pl.Series] = pl.Series([["a"], ["b"], ["c"], ["d"], ["e"]])
    CHOICES: ClassVar[pl.Series] = pl.Series([[True], [False], [True], [False], [True]])
    NUMBERS: ClassVar[pl.Series] = pl.Series(["1", "2", "3", "4", "5"])


class TreeFrame:
    """Hold the frame of the tests, and give the four answers of polars and of the evaluator.

    The frame holds two groups of three and two rows. A test can add columns
    of five rows, for a tree that reads a data type that the frame does not
    hold.

    Example:
        >>> TreeFrame().polars(pl.col("x").abs())
        ['Int64', 'List(Int64)', 'Int64', 'List(Int64)']
        >>> dates = pl.Series([None] * 5, dtype=pl.Date)
        >>> TreeFrame({"d": dates}).polars(pl.col("d").dt.year())
        ['Int32', 'List(Int32)', 'Int32', 'List(Int32)']
    """

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({
        "g": [1, 1, 1, 2, 2],
        "x": [1, -2, 3, -4, 5],
        "t": ["a", "b", "c", "d", "e"],
        "l": [[1], [2, 3], [4], [5], [6, 7]],
        "a": pl.Series([[1, 2]] * 5, dtype=pl.Array(pl.Int64, 2)),
        "s": [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}],
    })

    def __init__(self, columns: Mapping[str, pl.Series] | None = None) -> None:
        """Keep the frame, with the added columns.

        Args:
            columns: Columns of five rows to add to the frame, by their name.
        """
        self._frame = self.FRAME.with_columns(**(columns or {}))

    def polars(self, expression: pl.Expr) -> list[str]:
        """Give the four answers of polars: a rendering, or `raises`.

        Args:
            expression: The tree.

        Returns:
            The answers, in the order of the census.

        Example:
            >>> TreeFrame().polars(pl.col("t") + 1)
            ['raises', 'raises', 'raises', 'raises']
        """
        node = expression.alias("o")
        lazy = self._frame.lazy()
        runs: list[Callable[[], pl.DataType]] = [
            lambda: lazy.select(node).collect_schema()["o"],
            lambda: lazy.group_by("g").agg(node).collect_schema()["o"],
            lambda: lazy.select(node).collect().schema["o"],
            lambda: lazy.group_by("g").agg(node).collect().schema["o"],
        ]
        return [self.answer(run) for run in runs]

    def rules(self, expression: pl.Expr) -> list[str]:
        """Give the four answers of the evaluator, in the text that the census stores.

        Args:
            expression: The tree.

        Returns:
            The answers, in the order of the census.

        Example:
            >>> TreeFrame().rules(pl.col("x").abs())
            ['Int64', 'List(Int64)', 'Int64', 'List(Int64)']
        """
        columns = {
            name: ColumnLeaf(DType.of(dtype), RowForm.PER_ROW)
            for name, dtype in self._frame.schema.items()
        }
        evaluation = TreeEvaluation(columns, GroupBy((3, 2), QueryFlags.lazy(lone=True)))
        return list(CensusText().answers(evaluation.answers(ExpressionTree().of(expression))))

    def polars_message(self, expression: pl.Expr) -> str:
        """Give the first line of the message of polars where it refuses a tree in a `select`.

        Args:
            expression: The tree.

        Returns:
            The line, or nothing where polars runs the tree.

        Raises:
            KeyboardInterrupt: The run stops by a key, which is not an answer.
            SystemExit: The run exits, which is not an answer.
            GeneratorExit: The run closes a generator, which is not an answer.

        Example:
            >>> TreeFrame().polars_message(pl.col("x").shift(pl.col("g")))
            "'n' must be a scalar value"
        """
        try:
            self._frame.lazy().select(expression.alias("o")).collect()
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as error:
            return str(error).splitlines()[0]
        return ""

    def rules_message(self, expression: pl.Expr) -> str:
        """Give the message of the refusal of the evaluator for the run of a tree in a `select`.

        Args:
            expression: The tree.

        Returns:
            The message, or nothing where the evaluator gives no refusal.

        Example:
            >>> TreeFrame().rules_message(pl.col("x").shift(pl.col("g")))
            "'n' must be a scalar value"
        """
        columns = {
            name: ColumnLeaf(DType.of(dtype), RowForm.PER_ROW)
            for name, dtype in self._frame.schema.items()
        }
        evaluation = TreeEvaluation(columns, GroupBy((3, 2)))
        answer = evaluation.answers(ExpressionTree().of(expression)).run_select
        return answer.message if isinstance(answer, Refused) else ""

    def answer(self, run: Callable[[], pl.DataType]) -> str:
        """Give the answer of one run: its type, or `raises`.

        Args:
            run: The run.

        Returns:
            The answer.

        Raises:
            KeyboardInterrupt: The run stops by a key, which is not an answer.
            SystemExit: The run exits, which is not an answer.
            GeneratorExit: The run closes a generator, which is not an answer.

        Example:
            >>> TreeFrame().answer(lambda: pl.Int8())
            'Int8'
        """
        try:
            return str(run())
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException:
            return "raises"


class WindowFrame:
    """Hold the frame of the tests with an index, and give the answers in windows of its rows.

    `rolling` and `group_by_dynamic` group the rows of a frame into windows
    along an index column, and `agg` runs over each window as over a group. A
    rolling window of two units gives the sizes `(1, 2, 2, 2, 2)` over the
    index `0` to `4`, and its windows overlap. A dynamic window every two units
    and of three gives `(3, 3, 1)`, which overlap too, and a dynamic window of
    two gives `(2, 2, 1)`, which do not. The rules evaluate each context with
    the same windows (`GroupBy`).

    Example:
        >>> WindowFrame().polars(pl.col("x").sum())
        ['Int64', 'Int64', 'Int64', 'Int64', 'Int64', 'Int64']
    """

    FRAME: ClassVar[pl.DataFrame] = TreeFrame.FRAME.with_columns(i=pl.int_range(5))
    CONTEXTS: ClassVar[dict[str, tuple[Callable[[pl.LazyFrame], Any], GroupBy]]] = {
        "rolling": (
            lambda lazy: lazy.rolling("i", period="2i"),
            GroupBy((1, 2, 2, 2, 2), height=5, overlapping=True),
        ),
        "dynamic": (
            lambda lazy: lazy.group_by_dynamic("i", every="2i", period="3i"),
            GroupBy((3, 3, 1), height=5, overlapping=True),
        ),
        "disjoint": (
            lambda lazy: lazy.group_by_dynamic("i", every="2i"),
            GroupBy((2, 2, 1), height=5),
        ),
    }

    def polars(self, expression: pl.Expr) -> list[str]:
        """Give the planner type and the run type of polars in each context, in order.

        Args:
            expression: The tree.

        Returns:
            The answers: a rendering, or `raises`.

        Example:
            >>> WindowFrame().polars(pl.col("t") + 1)
            ['raises', 'raises', 'raises', 'raises', 'raises', 'raises']
        """
        node = expression.alias("o")
        lazy = self.FRAME.lazy()
        answers: list[str] = []
        for windows, _ in self.CONTEXTS.values():
            answers.extend(self._answers(windows(lazy).agg(node)))
        return answers

    def _answers(self, grouped: pl.LazyFrame) -> list[str]:
        """Give the planner type and the run type of one query: each a rendering, or `raises`.

        Args:
            grouped: The query.

        Returns:
            The two answers.

        Example:
            >>> grouped = WindowFrame.FRAME.lazy().rolling("i", period="2i").agg(o=pl.col("x"))
            >>> WindowFrame()._answers(grouped)
            ['List(Int64)', 'List(Int64)']
        """
        runs: list[Callable[[], pl.DataType]] = [
            lambda: grouped.collect_schema()["o"],
            lambda: grouped.collect().schema["o"],
        ]
        return [TreeFrame().answer(run) for run in runs]

    def rules(self, expression: pl.Expr) -> list[str]:
        """Give the planner type and the run type of the rules in each context, in order.

        Args:
            expression: The tree.

        Returns:
            The answers, in the text of the census.

        Example:
            >>> WindowFrame().rules(pl.col("x").rolling_mean(2).alias("y"))[:2]
            ['List(Float64)', 'List(Float64)']
        """
        columns = {
            name: ColumnLeaf(DType.of(dtype), RowForm.PER_ROW)
            for name, dtype in self.FRAME.schema.items()
        }
        tree = ExpressionTree().of(expression)
        answers: list[str] = []
        for _, groups in self.CONTEXTS.values():
            cell = TreeEvaluation(columns, groups).answers(tree)
            answers.extend(CensusText().answers(cell)[1::2])
        return answers
