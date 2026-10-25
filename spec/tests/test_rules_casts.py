"""Check the cast rules of `astrolabe._rules.casts` against polars, on every pair of types.

The rules translate the casts of polars from its source, in its own split:
the plan checks, the kernel of each source kind, and the fate of each value.
Each test here replays one part against the call of polars that runs it, on
every ordered pair of the 27 census kinds and of the variants of their
parameters that the rules treat apart:

- The planner type: `collect_schema()` of `pl.col("x").cast(target)`, and of
  the cast of a literal, which the cast pass runs while it plans.
- The run: `Series.cast(target, strict=...)` on each sample of values of the
  census, and on a series of only nulls, which casts with no kernel. A
  nested series also runs with nulls inside its values, because a nested
  cast casts its inner values as a series, and inner values that are all
  null cast with no kernel too.
- The lossless check: the cast pass of `index_of` refuses an element that
  does not cast to the type of the column with no loss.
- The cast check (`CastCheck`) with a `Boolean` target: the cast pass of
  `any()` refuses an input that it refuses.

A strict cast of an integer to `Categorical` keeps a value only where the
process holds a category with that id (`ProcessCategories`). No rule decides
those cells, and the rules raise `UndecidedError` there, so the tests accept
either answer of polars: the type of the kernel, or a failure.

Where polars refuses, the rules must refuse with the class of the error that
polars raises. Each test takes one source type or one literal, and its
assertion names every pair that fails.
"""

from __future__ import annotations

import datetime as dt
import functools
from typing import ClassVar

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
from spec.tools.node_census.kinds import AnyValues, Sample
from astrolabe._rules.cast_pass import CastFold, CastOption
from astrolabe._rules.casts import CastCheck, CastKernel, LosslessCast, PhysicalValue, SeriesCast
from astrolabe._rules.dtypes import DType
from astrolabe._rules.failures import UndecidedError
from astrolabe._rules.type_functions import ExprFields, Operand


class CastSamples:
    """Give the values that the run of a cast reads for one data type.

    They are the five samples of the census, with values that fit and values
    that do not, and a series of only nulls. A nested type also has a value
    with a null at each inner place, and a list has an empty list. Text also has one series for each
    form that a parse rule of a cast tells apart: a sign, a space, an
    exponent, `inf`, a date with short parts, a moment with and without an
    offset, and a value that rounds out of a precision.

    Example:
        >>> CastSamples().of(pl.Int8())["large"]
        [127, 127, 127]
        >>> CastSamples().of(pl.String())["text '-0'"]
        ['-0']
    """

    TEXTS: ClassVar[tuple[str, ...]] = (
        "+1",
        "-1",
        "-0",
        " 1",
        "1e3",
        "1e",
        ".5",
        "1.",
        "-1.5e2",
        "inf",
        "-inf",
        "nan",
        "255",
        "256",
        "86400000000000",
        "2020-1-1",
        "2020-02-30",
        "2020-01-01T00:00",
        "2020-01-01T25:00",
        "2020-01-01 00:00:00",
        "2020-01-01T00:00:00+00:00",
        "2020-01-01T00:00:00+01:00",
        "1.255",
        "999.5",
        "é",
    )

    def of(self, dtype: pl.DataType) -> dict[str, list[object]]:
        """Give the values of each sample, by the name of the sample.

        Args:
            dtype: The data type.

        Returns:
            The values of each sample, three nulls under `nulls`, and for
            text one value of each form.

        Example:
            >>> CastSamples().of(pl.String())["text"]
            ['1', '2020-01-01', '1.5']
        """
        samples = {sample.value: list(AnyValues().of(dtype, sample)) for sample in Sample}
        samples["nulls"] = [None, None, None]
        if isinstance(dtype, pl.String):
            samples |= {f"text {text!r}": [text] for text in self.TEXTS}
        if dtype.is_nested():
            samples["inner nulls"] = [self._nulls_inside(dtype)]
        if isinstance(dtype, pl.List):
            samples["empty list"] = [[]]
        return samples

    def _nulls_inside(self, dtype: object) -> object:
        """Give a value of a type with a null at each place that holds no nested value.

        Args:
            dtype: The data type.

        Returns:
            The value: `None` for a flat type, a list of one inner value, an
            array of inner values, or a struct of inner values.

        Example:
            >>> CastSamples()._nulls_inside(pl.Struct({"a": pl.List(pl.Int8)}))
            {'a': [None]}
        """
        if isinstance(dtype, pl.Struct):
            return {field.name: self._nulls_inside(field.dtype) for field in dtype.fields}
        if isinstance(dtype, pl.Array):
            return [self._nulls_inside(dtype.inner)] * dtype.size
        if isinstance(dtype, pl.List):
            return [self._nulls_inside(dtype.inner)]
        return None


class CastCalls:
    """Run the calls of polars that plan and run a cast, and give their answers.

    Each call gives a data type, or how polars stops where it refuses the call.

    Example:
        >>> CastCalls().run([300], pl.Int64(), pl.Int8(), strict=True)
        raises InvalidOperationError
    """

    def planned(self, source: pl.DataType, target: pl.DataType, *, strict: bool) -> DType | Stopped:
        """Give the planner type of a cast of a column.

        Args:
            source: The type of the column.
            target: The type of the cast.
            strict: Whether the cast is strict.

        Returns:
            The type, or how polars stops.

        Example:
            >>> CastCalls().planned(pl.Binary(), pl.Date(), strict=True)
            Date
        """
        frame = pl.LazyFrame(schema={"x": source})
        cast = pl.col("x").cast(target, strict=strict)
        return PolarsAnswer().answer(lambda: frame.select(cast).collect_schema()["x"])

    def literal(
        self,
        value: int | float | str,
        target: pl.DataType,
        *,
        strict: bool,
    ) -> DType | Stopped:
        """Give the planner type of a cast of a literal.

        Args:
            value: The Python value of the literal.
            target: The type of the cast.
            strict: Whether the cast is strict.

        Returns:
            The type, or how polars stops.

        Example:
            >>> CastCalls().literal(300, pl.Int8(), strict=True)
            raises InvalidOperationError
        """
        frame = pl.LazyFrame(schema={"y": pl.Int8()})
        cast = pl.lit(value).cast(target, strict=strict).alias("x")
        return PolarsAnswer().answer(lambda: frame.select(cast).collect_schema()["x"])

    def keeps(self, value: int | float | str, target: pl.DataType) -> bool | None:
        """Say whether the fold of a cast that is not strict keeps the value of a literal.

        The value is lost where the result holds a null, also inside a list or
        a struct.

        Args:
            value: The Python value of the literal.
            target: The type of the cast.

        Returns:
            Whether the result holds no null, or `None` where polars refuses the cast.

        Example:
            >>> CastCalls().keeps(300, pl.Int8()), CastCalls().keeps(1, pl.Struct({"a": pl.Int8()}))
            (False, True)
        """
        cast = pl.lit(value).cast(target, strict=False).alias("x")
        result = PolarsAnswer().result(lambda: pl.select(cast).to_series().to_list())
        if isinstance(result, Stopped):
            return None
        return not self._holds_null(result)

    def _holds_null(self, value: object) -> bool:
        """Say whether a value holds a null, also inside a list or a struct.

        Args:
            value: The value.

        Returns:
            Whether it holds one.

        Example:
            >>> CastCalls()._holds_null([{"a": None}])
            True
        """
        if value is None:
            return True
        if isinstance(value, list):
            return any(self._holds_null(item) for item in value)
        if isinstance(value, dict):
            return any(self._holds_null(item) for item in value.values())
        return False

    def run(
        self,
        values: list[object],
        source: pl.DataType,
        target: pl.DataType,
        *,
        strict: bool,
    ) -> DType | Stopped:
        """Give the type of a run of `Series.cast` on given values.

        Args:
            values: The values of the series.
            source: The type of the series.
            target: The type of the cast.
            strict: Whether the cast is strict.

        Returns:
            The type, or how polars stops.

        Example:
            >>> CastCalls().run([None], pl.Date(), pl.Binary(), strict=True)
            Binary
        """
        series = pl.Series("x", values, dtype=source)
        return PolarsAnswer().answer(lambda: series.cast(target, strict=strict).dtype)

    def finds_index(self, column: pl.DataType, element: pl.DataType | int | float | str) -> bool:
        """Say whether the planner accepts `index_of` of an element in a column.

        Args:
            column: The type of the column.
            element: The type of a scalar element, or the Python value of a literal.

        Returns:
            Whether `collect_schema()` gives a type.

        Example:
            >>> CastCalls().finds_index(pl.Int8(), 300)
            False
        """
        if isinstance(element, pl.DataType):
            frame = pl.LazyFrame(schema={"x": column, "e": element})
            return self._accepts(frame, pl.col("x").index_of(pl.col("e").first()))
        return self._accepts(pl.LazyFrame(schema={"x": column}), pl.col("x").index_of(element))

    def _accepts(self, frame: pl.LazyFrame, expression: pl.Expr) -> bool:
        """Say whether the planner accepts an expression over a frame.

        Args:
            frame: The frame.
            expression: The expression, which keeps the name `x`.

        Returns:
            Whether `collect_schema()` gives a type.

        Example:
            >>> CastCalls()._accepts(pl.LazyFrame(schema={"x": pl.Date()}), pl.col("x").any())
            False
        """
        answer = PolarsAnswer().dtype(lambda: frame.select(expression).collect_schema()["x"])
        return answer is not None

    def any_value(self, dtype: pl.DataType) -> bool:
        """Say whether the planner accepts `any()` of a column.

        Args:
            dtype: The type of the column.

        Returns:
            Whether `collect_schema()` gives a type.

        Example:
            >>> CastCalls().any_value(pl.Date())
            False
        """
        return self._accepts(pl.LazyFrame(schema={"x": dtype}), pl.col("x").any())


class RunCheck:
    """Compare one run of a cast with the rules.

    Where the answer depends on the categories that the process holds, the
    rules give no answer, and the check accepts the type of the kernel or a
    refusal of any class.

    Example:
        >>> RunCheck().problem([1, 300], pl.Int64(), pl.Int8(), strict=True) is None
        True
    """

    def problem(
        self,
        values: list[object],
        source: pl.DataType,
        target: pl.DataType,
        *,
        strict: bool,
    ) -> str | None:
        """Give what differs between the run of polars and the rules, or `None`.

        Args:
            values: The values of the series.
            source: The type of the series.
            target: The type of the cast.
            strict: Whether the cast is strict.

        Returns:
            A description of the difference, or `None`.

        Example:
            >>> RunCheck().problem([1], pl.Int8(), pl.Categorical(), strict=True) is None
            True
        """
        answer = CastCalls().run(values, source, target, strict=strict)
        ours, theirs = DType.of(source), DType.of(target)
        try:
            run = SeriesCast(ours, theirs, strict=strict)
            rule = RuleAnswer().result(lambda: run.output(values))
        except UndecidedError:
            kernel = RuleAnswer().dtype(lambda: CastKernel().output(ours, theirs))
            if isinstance(answer, Stopped) or answer == kernel:
                return None
            return f"polars {answer!r}, the kernel gives {kernel!r} or a refusal"
        return None if answer == rule else f"polars {answer!r}, rule {rule!r}"


class LiteralCast:
    """Give the planner type of a cast of a literal by the rules, or how they say polars stops.

    The planner gives a cast its target type (`ExprFields.cast`). The cast
    pass then folds the cast into the literal, and refuses the plan where the
    run of the cast on its one value fails (`CastFold`).

    Example:
        >>> LiteralCast().planned(300, DType("Int8"), strict=True)
        raises InvalidOperationError
        >>> LiteralCast().planned(-1, DType("Time"), strict=False)
        Time
    """

    def planned(self, value: int | float | str, target: DType, *, strict: bool) -> DType | Stopped:
        """Give the planner type.

        It raises `UndecidedError` where the answer depends on the categories
        of the process.

        Args:
            value: The Python value of the literal.
            target: The type of the cast.
            strict: Whether the cast is strict.

        Returns:
            The target, or how polars stops.

        Example:
            >>> LiteralCast().planned("a", DType("String"), strict=True)
            String
        """
        literal = Operand.of_value(value)
        option = CastOption.STRICT if strict else CastOption.NON_STRICT
        folded = RuleAnswer().result(lambda: CastFold().of(literal, target, option))
        if isinstance(folded, Stopped):
            return folded
        return ExprFields().cast(literal.dtype, target)


class TestCasts:
    """Replay the cast rules against the calls of polars that plan and run each cast."""

    @pytest.mark.parametrize("source", RuleKinds().dtypes(), ids=repr)
    def test_planner_types_a_cast_of_a_column_as_its_target(self, source: pl.DataType) -> None:
        """The planner gives a cast of a column its target type, strict or not."""
        problems = []
        for target in RuleKinds().dtypes():
            rule = ExprFields().cast(DType.of(source), DType.of(target))
            for strict in (False, True):
                answer = CastCalls().planned(source, target, strict=strict)
                if answer != rule:
                    problems.append(
                        f"{source!r} -> {target!r} {strict=}: polars {answer!r}, rule {rule!r}",
                    )
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("source", RuleKinds().dtypes(), ids=repr)
    def test_run_of_a_cast_matches_the_rules(self, source: pl.DataType) -> None:
        """A run of `Series.cast` on each sample gives the type of the rules, or fails with them."""
        samples = CastSamples().of(source)
        problems = []
        for target in RuleKinds().dtypes():
            for name, values in samples.items():
                for strict in (False, True):
                    problem = RunCheck().problem(values, source, target, strict=strict)
                    if problem is not None:
                        cell = f"{source!r} -> {target!r} on {name} {strict=}"
                        problems.append(f"{cell}: {problem}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("source", RuleKinds().dtypes(), ids=repr)
    def test_physical_value_is_the_value_that_a_series_holds(self, source: pl.DataType) -> None:
        """The fit of a value reads the physical value that a series of the type holds.

        A series reads a datetime with no zone as UTC, also for a type with a
        zone. A literal reads it in the zone, and the tree of polars holds its
        physical value, so the rules read no datetime of a literal.
        """
        dtype = DType.of(source)
        problems = []
        for name, values in CastSamples().of(source).items():
            series = pl.Series("x", values, dtype=source)
            if source.is_nested() or not series.dtype.is_temporal():
                continue
            physical = series.to_physical().to_list()
            rule = [PhysicalValue().of(value, dtype) for value in values]
            if physical != rule:
                problems.append(f"{name}: polars {physical}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("strict", [False, True])
    def test_arrow_cast_to_categories_keeps_each_integer_as_an_id(self, *, strict: bool) -> None:
        """A list cast to an array of categories, or an array to a list, runs on any integer id.

        The arrow kernel casts an integer to the physical id of a category, and
        it does not check that the process holds the id. Only a later read of
        an id that the process lacks crashes, so the test reads the type alone.
        """
        cells: list[tuple[list[object], pl.DataType, pl.DataType]] = [
            ([[5], [10**6]], pl.List(pl.Int64()), pl.Array(pl.Categorical(), 1)),
            ([[5]], pl.Array(pl.Int64(), 1), pl.List(pl.Categorical())),
            ([[-1]], pl.List(pl.Int64()), pl.Array(pl.Categorical(), 1)),
            ([["a"]], pl.List(pl.String()), pl.Array(pl.Categorical(), 1)),
            ([[5]], pl.List(pl.Int64()), pl.Array(pl.Enum(["a"]), 1)),
        ]
        problems = []
        for values, source, target in cells:
            problem = RunCheck().problem(values, source, target, strict=strict)
            if problem is not None:
                problems.append(f"{source!r} -> {target!r} on {values}: {problem}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("value", RuleKinds().LITERALS, ids=repr)
    def test_planner_runs_the_cast_of_a_literal(self, value: int | float | str) -> None:
        """The planner raises where the cast of a literal fails, as a run on its one value does."""
        problems = []
        for target in RuleKinds().dtypes():
            for strict in (False, True):
                answer = CastCalls().literal(value, target, strict=strict)
                try:
                    rule = LiteralCast().planned(value, DType.of(target), strict=strict)
                except UndecidedError:
                    continue
                if answer != rule:
                    problems.append(
                        f"{value!r} -> {target!r} {strict=}: polars {answer!r}, rule {rule!r}",
                    )
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("value", RuleKinds().LITERALS, ids=repr)
    def test_fold_of_a_literal_keeps_the_value_that_polars_keeps(
        self,
        value: int | float | str,
    ) -> None:
        """A fold that is not strict keeps the value of a literal exactly where polars keeps it.

        A cast to `Categorical` keeps a value only where the process holds a
        category of that id, so no rule decides it, and it does not run here.
        """
        problems = []
        for target in RuleKinds().dtypes():
            answer = CastCalls().keeps(value, target)
            if answer is None or isinstance(target, pl.Categorical):
                continue
            keeps = functools.partial(CastFold().keeps, Operand.of_value(value), DType.of(target))
            rule = RuleAnswer().result(keeps)
            if answer != rule:
                problems.append(f"{value!r} -> {target!r}: polars keeps {answer}, rule {rule}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("column", RuleKinds().dtypes(), ids=repr)
    def test_lossless_check_matches_the_planner_of_index_of(self, column: pl.DataType) -> None:
        """The planner of `index_of` refuses an element that cannot cast to the column with no loss.

        Two elements of the type of the column need no cast and no check.
        """
        kinds = RuleKinds()
        target = DType.of(column)
        problems = []
        elements: list[pl.DataType | int | float | str] = [*kinds.dtypes(), *kinds.LITERALS]
        for element in elements:
            if isinstance(element, pl.DataType):
                source = DType.of(element)
            else:
                source = kinds.literal_type(element)
            rule = source.same(target) or LosslessCast().of(source, target)
            answer = CastCalls().finds_index(column, element)
            if answer != rule:
                problems.append(f"{element!r} in {column!r}: polars {answer}, rule {rule}")
        assert not problems, "\n".join(problems)

    def test_check_of_a_cast_to_boolean_matches_the_planner_of_any(self) -> None:
        """The planner of `any()` refuses an input whose cast to `Boolean` `CastCheck` refuses."""
        problems = []
        for dtype in RuleKinds().dtypes():
            rule = CastCheck().of(DType.of(dtype), DType("Boolean")) is not False
            answer = CastCalls().any_value(dtype)
            if answer != rule:
                problems.append(f"{dtype!r}: polars {answer}, rule {rule}")
        assert not problems, "\n".join(problems)


class CastTrees:
    """Hold trees that reach paths of the casts which the samples of the casts do not reach.

    These trees cast a list of `UInt8` to bytes, bytes that are no text to
    text, an array to categories, and a list of structs to an array, whose
    structs cast field by field. `replace_strict` checks its old values
    against the table of the casts that polars marks as possible
    (`CastAbility`), so a column of categories, of booleans, of lists or of
    structs, with old values of another type, reaches each arm of that table.
    Each tree maps every value of its column, so that the run does not depend
    on the values. Each tree runs over the frame of `TreeFrame`, with the
    columns that it adds.

    Example:
        >>> "a list of UInt8 to bytes" in CastTrees.TREES
        True
    """

    BYTES: ClassVar[pl.Series] = pl.Series([[1], [2], [3], [4], [5]], dtype=pl.List(pl.UInt8))
    LISTS: ClassVar[pl.Series] = pl.Series([[1], [2], [3], [4], [5]], dtype=pl.List(pl.Int8))
    PAIRS: ClassVar[pl.Series] = pl.Series(
        [{"a": 1, "b": 2}] * 5,
        dtype=pl.Struct({"a": pl.Int8, "b": pl.Int8}),
    )
    NULL_STRUCTS: ClassVar[pl.Series] = pl.Series([None] * 5, dtype=pl.Struct({"a": pl.Int8}))
    ARRAYS: ClassVar[dict[str, pl.Series]] = {
        "wider": pl.Series([[1, 2, 3]], dtype=pl.Array(pl.Int64, 3)),
        "narrower inner": pl.Series([[1, 2]], dtype=pl.Array(pl.Int8, 2)),
    }
    STRUCTS: ClassVar[pl.Series] = pl.Series([{"a": 1}] * 5, dtype=pl.Struct({"a": pl.Int8}))
    TREES: ClassVar[dict[str, tuple[dict[str, pl.Series], pl.Expr]]] = {
        "a list of structs to an array of structs whose field the value does not fit": (
            {},
            pl.lit([{"a": 300}]).cast(pl.Array(pl.Struct({"a": pl.Int8}), 1)),  # type: ignore[invalid-call-shape]
        ),
        "a list of structs to an array of structs, with a cast that is not strict": (
            {},
            pl.lit([{"a": 300}]).cast(pl.Array(pl.Struct({"a": pl.Int8}), 1), strict=False),
        ),
        "a list of structs of arrays to an array": (
            {},
            pl.lit([{"a": [1, 2]}], dtype=pl.List(pl.Struct({"a": pl.Array(pl.Int8, 2)}))).cast(  # type: ignore[invalid-call-shape]
                pl.Array(pl.Struct({"a": pl.Array(pl.Int16, 2)}), 1),
            ),
        ),
        "a list of a zoned moment to a list of integers": (
            {},
            pl.lit([dt.datetime(2024, 1, 2, tzinfo=dt.UTC)]).cast(pl.List(pl.Int64)),
        ),
        "text of a huge exponent to a decimal": (
            {},
            pl.lit("1e9999999999999999999999").cast(pl.Decimal(10, 2), strict=False),
        ),
        "a struct of no field filled by a struct": (
            {},
            pl.coalesce(pl.lit({}, dtype=pl.Struct({})), pl.col("s")),
        ),
        "text replaced by categories": (
            {},
            pl.col("t").replace_strict(
                pl.Series(["a", "b", "c", "d", "e"], dtype=pl.Categorical),
                [1, 2, 3, 4, 5],
            ),
        ),
        "arrays replaced by wider arrays": ({}, pl.col("a").replace_strict(ARRAYS["wider"], [0])),
        "arrays replaced by arrays of a narrower inner type": (
            {},
            pl.col("a").replace_strict(ARRAYS["narrower inner"], [0]),
        ),
        "structs replaced by structs of more fields": (
            {"p": PAIRS},
            pl.col("p").replace_strict([{"a": 1, "b": 2, "c": 3}], [0]),
        ),
        "structs replaced by structs of bytes": (
            {"p": PAIRS},
            pl.col("p").replace_strict([{"a": b"x", "b": b"y"}], [0]),
        ),
        "null structs replaced by structs of no field": (
            {"q": NULL_STRUCTS},
            pl.col("q").replace_strict(pl.Series([{}], dtype=pl.Struct({})), [0]),
        ),
        "a list literal of UInt8 to bytes": (
            {},
            pl.lit([1, 2], dtype=pl.List(pl.UInt8)).cast(pl.Binary),
        ),
        "a list of UInt8 to bytes": ({"u": BYTES}, pl.col("u").cast(pl.Binary)),
        "bytes that are no text to text": ({}, pl.lit(b"\xff").cast(pl.String)),  # type: ignore[invalid-call-shape]
        "an array to categories": ({}, pl.col("a").cast(pl.Array(pl.Categorical, 2))),
        "categories replaced by bytes": (
            {"c": TreeColumns.CATEGORIES},
            pl.col("c").replace_strict([b"a"], ["x"]),
        ),
        "categories replaced by numbers": (
            {"c": TreeColumns.CATEGORIES},
            pl.col("c").replace_strict([1], ["x"]),
        ),
        "booleans replaced by numbers": (
            {"b": TreeColumns.FLAGS},
            pl.col("b").replace_strict([1, 0], [2, 3]),
        ),
        "lists of Int8 replaced by lists of Int64": (
            {"l": LISTS},
            pl.col("l").replace_strict([[1], [2], [3], [4], [5]], [[2]] * 5),
        ),
        "structs of Int8 replaced by structs of Int64": (
            {"s": STRUCTS},
            pl.col("s").replace_strict([{"a": 1}], [{"a": 2}]),
        ),
    }


class TestObjectCasts:
    """Replay the casts of `Object`, a kind outside the census, against polars.

    The planner gives the cast its target. The kernel casts a series that
    holds a value only to `Object`, strict or not, and a series of only nulls
    casts to any type with no kernel.
    """

    TARGETS: ClassVar[tuple[pl.DataType, ...]] = (*RuleKinds().dtypes(), pl.Object())
    SAMPLES: ClassVar[dict[str, list[object]]] = {"value": [object()], "nulls": [None]}

    def test_planner_types_a_cast_of_an_object_column_as_its_target(self) -> None:
        """The plan of the cast gives the target, also where the run raises."""
        problems = []
        for target in self.TARGETS:
            rule = ExprFields().cast(DType("Object"), DType.of(target))
            for strict in (False, True):
                answer = CastCalls().planned(pl.Object(), target, strict=strict)
                if answer != rule:
                    problems.append(f"{target!r} {strict=}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    def test_run_of_a_cast_of_an_object_series_matches_the_rules(self) -> None:
        """A series that holds a value casts only to `Object`, and only nulls cast to any type."""
        problems = []
        for target in self.TARGETS:
            for name, values in self.SAMPLES.items():
                for strict in (False, True):
                    problem = RunCheck().problem(values, pl.Object(), target, strict=strict)
                    if problem is not None:
                        problems.append(f"{target!r} on {name} {strict=}: {problem}")
        assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", sorted(CastTrees.TREES))
def test_a_tree_reaches_a_path_of_the_casts(name: str) -> None:
    """The four answers of the rules match polars, for a tree that reaches one path of a cast."""
    columns, expression = CastTrees.TREES[name]
    frame = TreeFrame(columns)
    assert frame.rules(expression) == frame.polars(expression)
