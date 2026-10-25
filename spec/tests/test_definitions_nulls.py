"""Check the null rule of each definition against polars, over inputs of only nulls.

A definition says when the output of its node holds only nulls (`NullRule`):
never, where its first input does, or where any input does. A later node
reads that fact, because polars casts a series of only nulls to any type with
no kernel. Each test runs one node of polars with an input of only nulls of
each census kind at each place, beside typical values of that kind at the
other place, in a `select` and in `agg`. Wherever polars runs the node and
the rule claims only nulls for the type of its output, the output must hold
only nulls: in each row of a `select`, and in the list or the value of each
group in `agg`. Every definition that states a rule has a call here.
"""

from __future__ import annotations

import functools
import importlib
import operator
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import ApiGate, KernelSamples, PolarsAnswer, RuleKinds, TreeFrame
from astrolabe._rules.definitions.aggregations import Aggregations
from astrolabe._rules.definitions.binary_operators import BinaryOperators
from astrolabe._rules.definitions.definition import NullRule
from astrolabe._rules.definitions.elementwise import Elementwise
from astrolabe._rules.definitions.length_preserving import LengthPreserving
from astrolabe._rules.definitions.node_kinds import NodeKinds
from astrolabe._rules.definitions.ranges import Ranges
from astrolabe._rules.definitions.registry import Definitions
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.dispatch import Data

if TYPE_CHECKING:
    from collections.abc import Callable

    from astrolabe._rules.definitions.definition import Definition


class NullCalls:
    """Hold a call of polars for each definition that states a null rule.

    Each call takes the input that can hold only nulls, and the typical
    input of the same kind at the other place, which a node of one input
    does not read.

    Example:
        >>> NullCalls().of(Elementwise.ABS)(pl.col("n"), pl.col("t")).meta.root_names()
        ['n']
    """

    CALLS: ClassVar[dict[str, Callable[[pl.Expr, pl.Expr], pl.Expr]]] = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "truediv": operator.truediv,
        "floordiv": operator.floordiv,
        "mod": operator.mod,
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
        "abs": lambda x, _: x.abs(),
        "sqrt": lambda x, _: x.sqrt(),
        "cast": lambda x, _: x.cast(pl.String),
        "cbrt": lambda x, _: x.cbrt(),
        "exp": lambda x, _: x.exp(),
        "log1p": lambda x, _: x.log1p(),
        "log": lambda x, _: x.log(),
        "sin": lambda x, _: x.sin(),
        "cos": lambda x, _: x.cos(),
        "tan": lambda x, _: x.tan(),
        "cot": lambda x, _: x.cot(),
        "arcsin": lambda x, _: x.arcsin(),
        "arccos": lambda x, _: x.arccos(),
        "arctan": lambda x, _: x.arctan(),
        "sinh": lambda x, _: x.sinh(),
        "cosh": lambda x, _: x.cosh(),
        "tanh": lambda x, _: x.tanh(),
        "arcsinh": lambda x, _: x.arcsinh(),
        "arccosh": lambda x, _: x.arccosh(),
        "arctanh": lambda x, _: x.arctanh(),
        "degrees": lambda x, _: x.degrees(),
        "radians": lambda x, _: x.radians(),
        "neg": lambda x, _: x.neg(),
        "sign": lambda x, _: x.sign(),
        "floor": lambda x, _: x.floor(),
        "ceil": lambda x, _: x.ceil(),
        "round": lambda x, _: x.round(),
        "truncate": lambda x, _: x.truncate(),
        "round_sig_figs": lambda x, _: x.round_sig_figs(2),
        "is_nan": lambda x, _: x.is_nan(),
        "is_not_nan": lambda x, _: x.is_not_nan(),
        "is_finite": lambda x, _: x.is_finite(),
        "is_infinite": lambda x, _: x.is_infinite(),
        "not_": lambda x, _: x.not_(),
        "bitwise_count_ones": lambda x, _: x.bitwise_count_ones(),
        "bitwise_count_zeros": lambda x, _: x.bitwise_count_zeros(),
        "bitwise_leading_ones": lambda x, _: x.bitwise_leading_ones(),
        "bitwise_leading_zeros": lambda x, _: x.bitwise_leading_zeros(),
        "bitwise_trailing_ones": lambda x, _: x.bitwise_trailing_ones(),
        "bitwise_trailing_zeros": lambda x, _: x.bitwise_trailing_zeros(),
        "to_physical": lambda x, _: x.to_physical(),
        "set_sorted": lambda x, _: x.set_sorted(),
        "reinterpret": lambda x, _: x.reinterpret(signed=True),
        "cut": lambda x, _: x.cut([0]),
        "filter": lambda x, _: x.filter(pl.col("b")),
        "sort": lambda x, _: x.sort(),
        "sort_by": lambda x, _: x.sort_by(pl.col("i")),
        "slice": lambda x, _: x.head(2),
        "gather": lambda x, y: x.gather(y.cast(pl.Int64) % 3),
        "explode": lambda x, _: x.explode(),
        "alias": lambda x, _: x.alias("z"),
        "name.keep": lambda x, _: x.name.keep(),
        "name.suffix": lambda x, _: x.name.suffix("_s"),
        "min": lambda x, _: x.min(),
        "max": lambda x, _: x.max(),
        "first": lambda x, _: x.first(),
        "last": lambda x, _: x.last(),
        "mean": lambda x, _: x.mean(),
        "median": lambda x, _: x.median(),
        "std": lambda x, _: x.std(),
        "var": lambda x, _: x.var(),
        "quantile": lambda x, _: x.quantile(0.5),
        "skew": lambda x, _: x.skew(),
        "kurtosis": lambda x, _: x.kurtosis(),
        "cum_sum": lambda x, _: x.cum_sum(),
        "cum_max": lambda x, _: x.cum_max(),
        "cum_min": lambda x, _: x.cum_min(),
        "cum_prod": lambda x, _: x.cum_prod(),
        "reverse": lambda x, _: x.reverse(),
        "shift": lambda x, _: x.shift(1),
        "diff": lambda x, _: x.diff(),
        "pct_change": lambda x, _: x.pct_change(),
        "rank": lambda x, _: x.rank(),
        "interpolate": lambda x, _: x.interpolate(),
        "rechunk": lambda x, _: x.rechunk(),
        "ewm_mean": lambda x, _: x.ewm_mean(alpha=0.5),
        "ewm_sum": lambda x, _: x.ewm_sum(alpha=0.5),
        "ewm_std": lambda x, _: x.ewm_std(alpha=0.5),
        "ewm_var": lambda x, _: x.ewm_var(alpha=0.5),
        "rolling_min": lambda x, _: x.rolling_min(2),
        "rolling_max": lambda x, _: x.rolling_max(2),
        "rolling_sum": lambda x, _: x.rolling_sum(2),
        "rolling_mean": lambda x, _: x.rolling_mean(2),
        "rolling_std": lambda x, _: x.rolling_std(2),
        "rolling_var": lambda x, _: x.rolling_var(2),
        "int_ranges": pl.int_ranges,
        "linear_spaces": lambda x, y: pl.linear_spaces(x, y, 3),
        "date_ranges": pl.date_ranges,
        "datetime_ranges": pl.datetime_ranges,
        "time_ranges": pl.time_ranges,
    }
    DEFINITIONS: ClassVar[dict[str, Definition]] = {
        "add": BinaryOperators.ADD,
        "sub": BinaryOperators.SUBTRACT,
        "mul": BinaryOperators.MULTIPLY,
        "truediv": BinaryOperators.TRUE_DIVIDE,
        "floordiv": BinaryOperators.FLOOR_DIVIDE,
        "mod": BinaryOperators.MODULUS,
        "eq": BinaryOperators.EQ,
        "ne": BinaryOperators.NOT_EQ,
        "lt": BinaryOperators.LT,
        "le": BinaryOperators.LT_EQ,
        "gt": BinaryOperators.GT,
        "ge": BinaryOperators.GT_EQ,
        "abs": Elementwise.ABS,
        "sqrt": Elementwise.SQRT,
        "cast": Elementwise.CAST,
        "cbrt": Elementwise.CBRT,
        "exp": Elementwise.EXP,
        "log1p": Elementwise.LOG1P,
        "log": Elementwise.LOG,
        "sin": Elementwise.SIN,
        "cos": Elementwise.COS,
        "tan": Elementwise.TAN,
        "cot": Elementwise.COT,
        "arcsin": Elementwise.ARCSIN,
        "arccos": Elementwise.ARCCOS,
        "arctan": Elementwise.ARCTAN,
        "sinh": Elementwise.SINH,
        "cosh": Elementwise.COSH,
        "tanh": Elementwise.TANH,
        "arcsinh": Elementwise.ARCSINH,
        "arccosh": Elementwise.ARCCOSH,
        "arctanh": Elementwise.ARCTANH,
        "degrees": Elementwise.DEGREES,
        "radians": Elementwise.RADIANS,
        "neg": Elementwise.NEGATE,
        "sign": Elementwise.SIGN,
        "floor": Elementwise.FLOOR,
        "ceil": Elementwise.CEIL,
        "round": Elementwise.ROUND,
        "truncate": Elementwise.TRUNCATE,
        "round_sig_figs": Elementwise.ROUND_SIG_FIGS,
        "is_nan": Elementwise.IS_NAN,
        "is_not_nan": Elementwise.IS_NOT_NAN,
        "is_finite": Elementwise.IS_FINITE,
        "is_infinite": Elementwise.IS_INFINITE,
        "not_": Elementwise.NOT,
        "bitwise_count_ones": Elementwise.BITWISE_COUNT_ONES,
        "bitwise_count_zeros": Elementwise.BITWISE_COUNT_ZEROS,
        "bitwise_leading_ones": Elementwise.BITWISE_LEADING_ONES,
        "bitwise_leading_zeros": Elementwise.BITWISE_LEADING_ZEROS,
        "bitwise_trailing_ones": Elementwise.BITWISE_TRAILING_ONES,
        "bitwise_trailing_zeros": Elementwise.BITWISE_TRAILING_ZEROS,
        "to_physical": Elementwise.TO_PHYSICAL,
        "set_sorted": Elementwise.SET_SORTED,
        "reinterpret": Elementwise.REINTERPRET,
        "cut": Elementwise.CUT,
        "filter": NodeKinds.FILTER,
        "sort": NodeKinds.SORT,
        "sort_by": NodeKinds.SORT_BY,
        "slice": NodeKinds.SLICE,
        "gather": NodeKinds.GATHER,
        "explode": NodeKinds.EXPLODE,
        "alias": NodeKinds.ALIAS,
        "name.keep": NodeKinds.KEEP_NAME,
        "name.suffix": NodeKinds.RENAME_ALIAS,
        "min": Aggregations.MIN,
        "max": Aggregations.MAX,
        "first": Aggregations.FIRST,
        "last": Aggregations.LAST,
        "mean": Aggregations.MEAN,
        "median": Aggregations.MEDIAN,
        "std": Aggregations.STD,
        "var": Aggregations.VAR,
        "quantile": Aggregations.QUANTILE,
        "skew": Aggregations.SKEW,
        "kurtosis": Aggregations.KURTOSIS,
        "cum_sum": LengthPreserving.CUM_SUM,
        "cum_max": LengthPreserving.CUM_MAX,
        "cum_min": LengthPreserving.CUM_MIN,
        "cum_prod": LengthPreserving.CUM_PROD,
        "reverse": LengthPreserving.REVERSE,
        "shift": LengthPreserving.SHIFT,
        "diff": LengthPreserving.DIFF,
        "pct_change": LengthPreserving.PCT_CHANGE,
        "rank": LengthPreserving.RANK,
        "interpolate": LengthPreserving.INTERPOLATE,
        "rechunk": LengthPreserving.RECHUNK,
        "ewm_mean": LengthPreserving.EWM_MEAN,
        "ewm_sum": LengthPreserving.EWM_SUM,
        "ewm_std": LengthPreserving.EWM_STD,
        "ewm_var": LengthPreserving.EWM_VAR,
        "rolling_min": LengthPreserving.ROLLING_MIN,
        "rolling_max": LengthPreserving.ROLLING_MAX,
        "rolling_sum": LengthPreserving.ROLLING_SUM,
        "rolling_mean": LengthPreserving.ROLLING_MEAN,
        "rolling_std": LengthPreserving.ROLLING_STD,
        "rolling_var": LengthPreserving.ROLLING_VAR,
        "int_ranges": Ranges.INT_RANGES,
        "linear_spaces": Ranges.LINEAR_SPACES,
        "date_ranges": Ranges.DATE_RANGES,
        "datetime_ranges": Ranges.DATETIME_RANGES,
        "time_ranges": Ranges.TIME_RANGES,
    }

    def of(self, definition: Definition) -> Callable[[pl.Expr, pl.Expr], pl.Expr]:
        """Give the call of a definition.

        Args:
            definition: The definition.

        Returns:
            The call.

        Example:
            >>> str(NullCalls().of(NodeKinds.SORT)(pl.col("n"), pl.col("t")))
            'col("n").sort(asc)'
        """
        names = [name for name, known in self.DEFINITIONS.items() if known is definition]
        return self.CALLS[names[0]]


class NullFrame:
    """Run a call of polars over a column of only nulls, and check that its output holds only nulls.

    Example:
        >>> NullFrame(pl.Int8()).problems(lambda x, _: x.abs(), NullRule.FIRST)
        []
    """

    GROUPS: ClassVar[list[int]] = [1, 1, 2]

    def __init__(self, dtype: pl.DataType) -> None:
        """Build the frame of one kind: a column of only nulls, typical values, an index and a mask.

        Args:
            dtype: The kind.

        Example:
            >>> NullFrame(pl.String()).frame["n"].null_count()
            3
        """
        self.frame = pl.DataFrame([
            pl.Series("g", self.GROUPS),
            pl.Series("n", [None] * 3, dtype=dtype),
            pl.Series("t", KernelSamples().typical(dtype), dtype=dtype),
            pl.Series("i", [2, 0, 1]),
            pl.Series("b", [True, False, True]),
        ])

    def problems(self, call: Callable[[pl.Expr, pl.Expr], pl.Expr], rule: NullRule) -> list[str]:
        """Give each place and context where the rule claims only nulls and polars gives a value.

        The rule reads the type of the output that polars gives: the type of
        the values of each group in `agg`. Only a place where the rule can
        claim only nulls runs, because a float function ends the process over
        a struct with values.

        Args:
            call: The call of polars, over the two inputs.
            rule: The null rule of its definition.

        Returns:
            A description of each failure.

        Example:
            >>> NullFrame(pl.Int8()).problems(operator.add, NullRule.ANY)
            []
        """
        problems = []
        places = {NullRule.FIRST: (0,), NullRule.ANY: (0, 1)}.get(rule, ())
        for place in places:
            problems += self._place_problems(call, rule, place)
        return problems

    def _place_problems(
        self,
        call: Callable[[pl.Expr, pl.Expr], pl.Expr],
        rule: NullRule,
        place: int,
    ) -> list[str]:
        """Give each context where the rule claims only nulls and polars gives a value, at a place.

        Args:
            call: The call of polars, over the two inputs.
            rule: The null rule of its definition.
            place: The place of the input of only nulls.

        Returns:
            A description of each failure.

        Example:
            >>> NullFrame(pl.Int8())._place_problems(operator.add, NullRule.ANY, 1)
            []
        """
        dtype = DType.of(self.frame.schema["n"])
        sides: list[pl.Expr] = [pl.col("t"), pl.col("t")]
        sides[place] = pl.col("n")
        inputs = [Data(dtype, nulls=index == place) for index in range(len(sides))]
        expression = call(*sides).alias("o")
        problems = []
        for context, values in self._outputs(expression).items():
            if values is None or not rule.output(inputs, DType.of(values.dtype)):
                continue
            if values.null_count() != values.len():
                problems.append(f"{expression} in {context}: {values.to_list()}")
        return problems

    def _outputs(self, expression: pl.Expr) -> dict[str, pl.Series | None]:
        """Give the values of the output in each context, or `None` where polars refuses the call.

        In `agg`, the values of every group: the values of its list, or its value.

        Args:
            expression: The expression.

        Returns:
            The values, by context.

        Example:
            >>> NullFrame(pl.Int8())._outputs(pl.col("n").abs().alias("o"))["select"].to_list()
            [None, None, None]
        """
        selected = PolarsAnswer().result(lambda: self.frame.select(expression)["o"])
        grouped = PolarsAnswer().result(lambda: self.frame.group_by("g").agg(expression)["o"])
        outputs: dict[str, pl.Series | None] = {}
        for context, column in (("select", selected), ("agg", grouped)):
            if not isinstance(column, pl.Series):
                outputs[context] = None
            elif isinstance(column.dtype, pl.List) and context == "agg":
                outputs[context] = column.explode()
            else:
                outputs[context] = column
        return outputs


class CensusCalls:
    """Find the calls of the census for each node of one input, by the function of its definition.

    The tests of the definitions hold a call of polars for each node of one
    input, which the test of a null rule reuses.

    Example:
        >>> "TemporalExpr.Year" in CensusCalls().of()
        True
    """

    MODULES: ClassVar[tuple[str, ...]] = (
        "aggregations",
        "length_preserving",
        "other_lengths",
        "strings",
        "temporal",
        "nested",
    )
    TABLES: ClassVar[tuple[str, ...]] = ("ONE_INPUT", "ONE_INPUT_OPTIONS", "OPTIONS")

    @functools.cache  # ruff: ignore[cached-instance-method] -- the tables never change.
    def of(self) -> dict[str, list[Callable[[pl.Expr], pl.Expr]]]:
        """Give the calls of each node of one input.

        Returns:
            The calls, by the function of the definition of the node.

        Example:
            >>> len(CensusCalls().of()["TemporalExpr.Year"])
            1
        """
        calls: dict[str, list[Callable[[pl.Expr], pl.Expr]]] = {}
        for check in self._checks():
            function = check._definition.identity.function  # ruff: ignore[private-member-access]
            calls.setdefault(function, []).append(check._build)  # ruff: ignore[private-member-access]
        return calls

    def _checks(self) -> list[Any]:
        """Give the check of each node of one input of the census modules.

        Returns:
            The checks: each holds the definition and the call of its node.

        Example:
            >>> len(CensusCalls()._checks()) > 100
            True
        """
        checks = []
        for name in self.MODULES:
            module = importlib.import_module(f"spec.tests.test_definitions_{name}")
            for holder in vars(module).values():
                for table in self._tables(holder):
                    checks += [self._check(entry) for entry in table.values()]
        return checks

    def _check(self, entry: Any) -> Any:
        """Give the check of an entry of a table: the entry, or the check that it builds.

        Args:
            entry: The entry.

        Returns:
            The check.

        Example:
            >>> CensusCalls()._check(1)
            1
        """
        return entry.check() if hasattr(entry, "check") else entry

    def _tables(self, holder: object) -> list[dict[str, Any]]:
        """Give the tables of calls of one input of a class of a census module.

        Args:
            holder: A value of the module.

        Returns:
            The tables.

        Example:
            >>> CensusCalls()._tables(1)
            []
        """
        if not isinstance(holder, type):
            return []
        tables = [getattr(holder, table, None) for table in self.TABLES]
        return [table for table in tables if isinstance(table, dict)]


class NullCases:
    """Give each definition that states a null rule, with the calls that check it.

    A node of the table of `NullCalls` takes its call there, and a node of one
    input of the census takes each call of the census.

    Example:
        >>> "Plus" in NullCases().all()
        True
    """

    def all(self) -> dict[str, tuple[NullRule, list[Callable[[pl.Expr, pl.Expr], pl.Expr]]]]:
        """Give the rule and the calls of each definition that states a rule.

        Returns:
            The rule and the calls, by the function of the definition.

        Example:
            >>> rule, calls = NullCases().all()["Abs"]
            >>> rule, len(calls)
            (<NullRule.FIRST: 'first'>, 1)
        """
        gate = ApiGate()
        absent = gate.absent()
        explicit = {
            definition.identity: NullCalls.CALLS[name]
            for name, definition in NullCalls.DEFINITIONS.items()
            if gate.has(name)
        }
        census = CensusCalls().of()
        stated = [
            definition
            for definition in Definitions.BY_IDENTITY.values()
            if definition.nulls is not NullRule.NONE and definition.identity not in absent
        ]
        cases: dict[str, tuple[NullRule, list[Callable[[pl.Expr, pl.Expr], pl.Expr]]]] = {}
        for definition in stated:
            function = definition.identity.function or definition.identity.shape
            builds = census.get(function, [])
            paired = [self._paired(build) for build in builds]
            known = explicit.get(definition.identity)
            cases[function] = (definition.nulls, paired if known is None else [known])
        return cases

    def _paired(self, build: Callable[[pl.Expr], pl.Expr]) -> Callable[[pl.Expr, pl.Expr], pl.Expr]:
        """Give a call of one input as a call of two, which reads the first.

        Args:
            build: The call of one input.

        Returns:
            The call of two inputs.

        Example:
            >>> str(NullCases()._paired(lambda x: x.abs())(pl.col("n"), pl.col("t")))
            'col("n").abs()'
        """
        return lambda first, _: build(first)


def test_each_definition_that_states_a_null_rule_has_a_call() -> None:
    """A rule that no test checks is a claim with no proof."""
    missing = [function for function, (_, calls) in NullCases().all().items() if not calls]
    assert not missing


@pytest.mark.parametrize("function", sorted(NullCases().all()))
def test_an_input_of_only_nulls_gives_only_nulls(function: str) -> None:
    """The output holds only nulls where the rule of the definition says so, for each kind."""
    rule, calls = NullCases().all()[function]
    problems = []
    for call in calls:
        for dtype in RuleKinds().dtypes():
            found = NullFrame(dtype).problems(call, rule)
            problems += [f"{dtype!r}: {problem}" for problem in found]
    assert not problems, "\n".join(problems)


class ShortCalls:
    """Hold a call of each node that gives only nulls over few rows, with the rows of that input.

    A cast to `List(Date)` shows the fact: polars wraps a series of values
    at the physical type, `List(Int32)`, and a series of only nulls at the
    type itself.

    Example:
        >>> sorted(ShortCalls.CALLS)[:2]
        ['diff', 'ewm_mean']
    """

    CALLS: ClassVar[dict[str, tuple[Definition, Callable[[pl.Expr], pl.Expr], int]]] = {
        "shift": (LengthPreserving.SHIFT, lambda x: x.shift(1), 1),
        "diff": (LengthPreserving.DIFF, lambda x: x.diff(2), 2),
        "pct_change": (LengthPreserving.PCT_CHANGE, lambda x: x.pct_change(1), 1),
        "rolling_min": (LengthPreserving.ROLLING_MIN, lambda x: x.rolling_min(2), 1),
        "rolling_max": (LengthPreserving.ROLLING_MAX, lambda x: x.rolling_max(2), 1),
        "rolling_sum": (LengthPreserving.ROLLING_SUM, lambda x: x.rolling_sum(3), 2),
        "rolling_mean": (LengthPreserving.ROLLING_MEAN, lambda x: x.rolling_mean(2), 1),
        "rolling_quantile": (
            LengthPreserving.ROLLING_QUANTILE,
            lambda x: x.rolling_quantile(0.5, window_size=2),
            1,
        ),
        "rolling_std": (LengthPreserving.ROLLING_STD, lambda x: x.rolling_std(2), 1),
        "rolling_var": (LengthPreserving.ROLLING_VAR, lambda x: x.rolling_var(2), 1),
        "rolling_rank": (LengthPreserving.ROLLING_RANK, lambda x: x.rolling_rank(2), 1),
        "rolling_skew": (LengthPreserving.ROLLING_SKEW, lambda x: x.rolling_skew(2), 1),
        "rolling_kurtosis": (LengthPreserving.ROLLING_KURTOSIS, lambda x: x.rolling_kurtosis(2), 1),
        "rolling_corr": (
            LengthPreserving.ROLLING_CORR_COV,
            lambda x: pl.rolling_corr(x, x, window_size=2),
            1,
        ),
        "std": (Aggregations.STD, lambda x: x.std(), 1),
        "std ddof 2": (Aggregations.STD, lambda x: x.std(ddof=2), 2),
        "var": (Aggregations.VAR, lambda x: x.var(), 1),
        "ewm_mean": (LengthPreserving.EWM_MEAN, lambda x: x.ewm_mean(alpha=0.5, min_samples=2), 1),
        "ewm_sum": (LengthPreserving.EWM_SUM, lambda x: x.ewm_sum(alpha=0.5, min_samples=3), 2),
        "ewm_std": (LengthPreserving.EWM_STD, lambda x: x.ewm_std(alpha=0.5), 1),
        "ewm_var": (
            LengthPreserving.EWM_VAR,
            lambda x: x.ewm_var(alpha=0.5, bias=True, min_samples=2),
            1,
        ),
    }


def test_each_definition_that_states_a_short_rule_has_a_call() -> None:
    """A rule of a short input that no test checks is a claim with no proof.

    A function that the installed release of polars does not have keeps its
    call, which the tests below leave out.
    """
    stated = {
        definition.identity
        for definition in Definitions.BY_IDENTITY.values()
        if definition.short is not None
    }
    assert stated == {definition.identity for definition, _, _ in ShortCalls.CALLS.values()}


@pytest.mark.parametrize("name", sorted(ApiGate().kept(ShortCalls.CALLS)))
@pytest.mark.parametrize("short", [True, False])
def test_a_short_input_gives_only_nulls(name: str, *, short: bool) -> None:
    """Over a short input the node gives only nulls, and over a longer input it gives values."""
    _, call, rows = ShortCalls.CALLS[name]
    column = pl.col("x").head(rows) if short else pl.col("x")
    expression = call(column).cast(pl.List(pl.Date))
    assert TreeFrame().rules(expression) == TreeFrame().polars(expression)


@pytest.mark.parametrize("name", sorted(ApiGate().kept(ShortCalls.CALLS)))
def test_a_null_that_the_code_decides_meets_one_row(name: str) -> None:
    """`rolling_corr` panics beside one row where an input holds a null, which the rules know.

    The node gives a null where its rule of few rows says so, and `+` keeps
    it, value by value.
    """
    _, call, _ = ShortCalls.CALLS[name]
    column = call(pl.col("x")).cast(pl.Float64) + 1
    expression = pl.rolling_corr(column, pl.col("x").first(), window_size=2)
    assert TreeFrame().rules(expression) == TreeFrame().polars(expression)
