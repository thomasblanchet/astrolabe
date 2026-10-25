"""Check the definitions of the nodes that keep the length of their input against polars.

Each such node gives one value for each row of its input, and a row can read
other rows: the cumulative, exponentially weighted and rolling functions,
`shift`, `diff`, `pct_change`, the interpolations, the ranks, the fills of
nulls by a strategy, and the tests of duplicates. Their definitions are in
`astrolabe._rules.definitions.length_preserving`, and `NodeEvaluation` gives
the four answers of a node from a definition: the planner type and the run
type, in a `select` and in `group_by().agg()`. Each test builds the node of
polars over inputs of every form, and compares the four answers of polars
with the four answers of the rules (`CellCheck`).

A node of one input takes every input of `Inputs`: each type of `RuleKinds`
in each of the four row forms, and each literal class of the census, direct
or derived. A node of two inputs takes about 500 pairs of inputs, and a node
of three inputs about 500 triples, beside a typical input at each place
(`NodeSample`). A call that differs from the main
call of its node only by an option takes a smaller sample: one value per row
of each type, and each literal class direct, at the first place of the node.
The frames have two groups of 3 and 2 rows, and samples of values that fit
and that do not (`LiveFrames`). A failure names the node and the cell.

`diff` casts its count to `Int64` with a strict cast, so a struct count
with a text field fails that cast on text that does not parse, a fact of the
values. Where the rules refuse a run of `diff`, polars reads text that parses
as a number, and then gives the refusal of the rules. `qcut` casts its input
to a float with a cast that turns text that does not parse into null, and
skips its checks over only nulls. Where the rules refuse `qcut` for its
labels or its quantiles, polars also reads text that parses.

A cell where the rules say that the run ends the process or never returns
does not run in the process of the tests. Those cells run in a subprocess
(`IsolatedCell`): `ewm_mean` and `ewm_sum` over a struct never return,
and `ewm_std`, `ewm_var` and `pct_change` over a struct end the process.
"""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import (
    CellCheck,
    Input,
    Inputs,
    NodeSample,
    Stopped,
    TemplateCall,
    Values,
)
from astrolabe._rules.definitions.length_preserving import LengthPreserving
from astrolabe._rules.dtypes import DType
from astrolabe._rules.failures import Stop
from astrolabe._rules.node_evaluation import ColumnLeaf, NodeEvaluation, Refused
from astrolabe._rules.options import RowForm

if TYPE_CHECKING:
    from astrolabe._rules.definitions.definition import Definition


class Nodes:
    """Hold the nodes of the tests: each definition with the call of polars that builds its node.

    `ONE_INPUT` holds the main call of each node of one input, and
    `ONE_INPUT_OPTIONS` the calls at other values of an option. `MANY_INPUTS`
    holds each node of two inputs or more with its typical input at each
    place: the values and the count of `shift`, the values, the count and the
    fill of `shift` with a fill value, or the values and the key of
    `rolling_mean_by`. `TWO_INPUT_OPTIONS` holds the calls of such a node at
    other values of an option.

    Example:
        >>> sorted(Nodes.ONE_INPUT)[:3]
        ['arg_sort', 'backward_fill', 'cum_count']
        >>> Nodes.KEY.label
        "Datetime(time_unit='ms', time_zone=None)/per row"
    """

    FLOAT: ClassVar[Input] = Inputs().labelled("Float64/per row")
    INTEGER: ClassVar[Input] = Inputs().labelled("Int64/per row")
    COUNT: ClassVar[Input] = Inputs().labelled("lit int 1/direct")
    KEY: ClassVar[Input] = Inputs().labelled("Datetime(time_unit='ms', time_zone=None)/per row")
    ONE_INPUT: ClassVar[dict[str, CellCheck]] = {
        "arg_sort": CellCheck(LengthPreserving.ARG_SORT, lambda x: x.arg_sort()),
        "is_duplicated": CellCheck(LengthPreserving.IS_DUPLICATED, lambda x: x.is_duplicated()),
        "is_first_distinct": CellCheck(
            LengthPreserving.IS_FIRST_DISTINCT,
            lambda x: x.is_first_distinct(),
        ),
        "is_last_distinct": CellCheck(
            LengthPreserving.IS_LAST_DISTINCT,
            lambda x: x.is_last_distinct(),
        ),
        "is_unique": CellCheck(LengthPreserving.IS_UNIQUE, lambda x: x.is_unique()),
        "cum_count": CellCheck(LengthPreserving.CUM_COUNT, lambda x: x.cum_count()),
        "cum_max": CellCheck(LengthPreserving.CUM_MAX, lambda x: x.cum_max()),
        "cum_min": CellCheck(LengthPreserving.CUM_MIN, lambda x: x.cum_min()),
        "cum_prod": CellCheck(LengthPreserving.CUM_PROD, lambda x: x.cum_prod()),
        "rle_id": CellCheck(LengthPreserving.RLE_ID, lambda x: x.rle_id()),
        "rank": CellCheck(LengthPreserving.RANK, lambda x: x.rank()),
        "rechunk": CellCheck(LengthPreserving.RECHUNK, lambda x: x.rechunk()),
        "reverse": CellCheck(LengthPreserving.REVERSE, lambda x: x.reverse()),
        "peak_max": CellCheck(LengthPreserving.PEAK_MAX, lambda x: x.peak_max()),
        "peak_min": CellCheck(LengthPreserving.PEAK_MIN, lambda x: x.peak_min()),
        "forward_fill": CellCheck(
            LengthPreserving.FILL_NULL_WITH_STRATEGY,
            lambda x: x.forward_fill(),
        ),
        "backward_fill": CellCheck(
            LengthPreserving.FILL_NULL_WITH_STRATEGY,
            lambda x: x.backward_fill(),
        ),
        "interpolate": CellCheck(LengthPreserving.INTERPOLATE, lambda x: x.interpolate()),
        "qcut": CellCheck(LengthPreserving.QCUT, lambda x: x.qcut([0.5])),
        "ewm_mean": CellCheck(LengthPreserving.EWM_MEAN, lambda x: x.ewm_mean(alpha=0.5)),
        "ewm_sum": CellCheck(LengthPreserving.EWM_SUM, lambda x: x.ewm_sum(alpha=0.5)),
        "ewm_std": CellCheck(LengthPreserving.EWM_STD, lambda x: x.ewm_std(alpha=0.5)),
        "ewm_var": CellCheck(LengthPreserving.EWM_VAR, lambda x: x.ewm_var(alpha=0.5)),
        "rolling_kurtosis": CellCheck(
            LengthPreserving.ROLLING_KURTOSIS,
            lambda x: x.rolling_kurtosis(window_size=2),
        ),
        "rolling_max": CellCheck(
            LengthPreserving.ROLLING_MAX,
            lambda x: x.rolling_max(window_size=2),
        ),
        "rolling_mean": CellCheck(
            LengthPreserving.ROLLING_MEAN,
            lambda x: x.rolling_mean(window_size=2),
        ),
        "rolling_median": CellCheck(
            LengthPreserving.ROLLING_QUANTILE,
            lambda x: x.rolling_median(window_size=2),
        ),
        "rolling_min": CellCheck(
            LengthPreserving.ROLLING_MIN,
            lambda x: x.rolling_min(window_size=2),
        ),
        "rolling_quantile": CellCheck(
            LengthPreserving.ROLLING_QUANTILE,
            lambda x: x.rolling_quantile(quantile=0.5, window_size=2),
        ),
        "rolling_rank": CellCheck(
            LengthPreserving.ROLLING_RANK,
            lambda x: x.rolling_rank(window_size=2),
        ),
        "rolling_skew": CellCheck(
            LengthPreserving.ROLLING_SKEW,
            lambda x: x.rolling_skew(window_size=2),
        ),
        "rolling_std": CellCheck(
            LengthPreserving.ROLLING_STD,
            lambda x: x.rolling_std(window_size=2),
        ),
        "rolling_sum": CellCheck(
            LengthPreserving.ROLLING_SUM,
            lambda x: x.rolling_sum(window_size=2),
        ),
        "rolling_var": CellCheck(
            LengthPreserving.ROLLING_VAR,
            lambda x: x.rolling_var(window_size=2),
        ),
    }
    ONE_INPUT_OPTIONS: ClassVar[dict[str, CellCheck]] = {
        "pl.cum_count": CellCheck(LengthPreserving.CUM_COUNT, TemplateCall("pl.cum_count('x')")),
        "cum_count reverse": CellCheck(
            LengthPreserving.CUM_COUNT,
            lambda x: x.cum_count(reverse=True),
        ),
        "cum_max reverse": CellCheck(LengthPreserving.CUM_MAX, lambda x: x.cum_max(reverse=True)),
        "cum_prod reverse": CellCheck(
            LengthPreserving.CUM_PROD,
            lambda x: x.cum_prod(reverse=True),
        ),
        "arg_sort descending": CellCheck(
            LengthPreserving.ARG_SORT,
            lambda x: x.arg_sort(descending=True, nulls_last=True),
        ),
        "rank min": CellCheck(LengthPreserving.RANK, lambda x: x.rank("min")),
        "rank dense descending": CellCheck(
            LengthPreserving.RANK,
            lambda x: x.rank("dense", descending=True),
        ),
        "rank ordinal": CellCheck(LengthPreserving.RANK, lambda x: x.rank("ordinal")),
        "rank random": CellCheck(LengthPreserving.RANK, lambda x: x.rank("random", seed=1)),
        "interpolate nearest": CellCheck(
            LengthPreserving.INTERPOLATE,
            lambda x: x.interpolate("nearest"),
        ),
        "qcut breaks": CellCheck(
            LengthPreserving.QCUT,
            lambda x: x.qcut([0.25, 0.75], include_breaks=True),
        ),
        "qcut labels": CellCheck(
            LengthPreserving.QCUT,
            lambda x: x.qcut(2, labels=["low", "high"], left_closed=True),
        ),
        "qcut few labels": CellCheck(
            LengthPreserving.QCUT,
            lambda x: x.qcut([0.5], labels=["a"]),
            refused=Values.NUMBER_TEXT,
        ),
        "qcut outside": CellCheck(
            LengthPreserving.QCUT,
            lambda x: x.qcut([1.5]),
            refused=Values.NUMBER_TEXT,
        ),
        "qcut duplicates": CellCheck(
            LengthPreserving.QCUT,
            lambda x: x.qcut([0.5, 0.5], allow_duplicates=True),
        ),
        "fill min": CellCheck(
            LengthPreserving.FILL_NULL_WITH_STRATEGY,
            lambda x: x.fill_null(strategy="min"),
        ),
        "fill max": CellCheck(
            LengthPreserving.FILL_NULL_WITH_STRATEGY,
            lambda x: x.fill_null(strategy="max"),
        ),
        "fill mean": CellCheck(
            LengthPreserving.FILL_NULL_WITH_STRATEGY,
            lambda x: x.fill_null(strategy="mean"),
        ),
        "fill zero": CellCheck(
            LengthPreserving.FILL_NULL_WITH_STRATEGY,
            lambda x: x.fill_null(strategy="zero"),
        ),
        "fill one": CellCheck(
            LengthPreserving.FILL_NULL_WITH_STRATEGY,
            lambda x: x.fill_null(strategy="one"),
        ),
        "forward_fill limit": CellCheck(
            LengthPreserving.FILL_NULL_WITH_STRATEGY,
            lambda x: x.forward_fill(limit=1),
        ),
        "ewm_mean com": CellCheck(
            LengthPreserving.EWM_MEAN,
            lambda x: x.ewm_mean(com=1, adjust=False),
        ),
        "ewm_var bias": CellCheck(
            LengthPreserving.EWM_VAR,
            lambda x: x.ewm_var(half_life=2, bias=True),
        ),
        "rolling_max weights": CellCheck(
            LengthPreserving.ROLLING_MAX,
            lambda x: x.rolling_max(window_size=2, weights=[1.0, 2.0]),
        ),
        "rolling_min weights": CellCheck(
            LengthPreserving.ROLLING_MIN,
            lambda x: x.rolling_min(window_size=2, weights=[1.0, 2.0]),
        ),
        "rolling_sum weights": CellCheck(
            LengthPreserving.ROLLING_SUM,
            lambda x: x.rolling_sum(window_size=2, weights=[1.0, 2.0]),
        ),
        "rolling_mean weights": CellCheck(
            LengthPreserving.ROLLING_MEAN,
            lambda x: x.rolling_mean(window_size=2, weights=[1.0, 2.0]),
        ),
        "rolling_mean min_samples": CellCheck(
            LengthPreserving.ROLLING_MEAN,
            lambda x: x.rolling_mean(window_size=2, min_samples=3),
        ),
        "rolling_sum min_samples": CellCheck(
            LengthPreserving.ROLLING_SUM,
            lambda x: x.rolling_sum(window_size=2, min_samples=3),
        ),
        "rolling_var center": CellCheck(
            LengthPreserving.ROLLING_VAR,
            lambda x: x.rolling_var(window_size=3, center=True, ddof=0),
        ),
        "rolling_rank min": CellCheck(
            LengthPreserving.ROLLING_RANK,
            lambda x: x.rolling_rank(window_size=2, method="min"),
        ),
        "rolling_rank dense": CellCheck(
            LengthPreserving.ROLLING_RANK,
            lambda x: x.rolling_rank(window_size=2, method="dense"),
        ),
        "rolling_quantile lower": CellCheck(
            LengthPreserving.ROLLING_QUANTILE,
            lambda x: x.rolling_quantile(quantile=0.3, interpolation="lower", window_size=2),
        ),
        "rolling_skew unbiased": CellCheck(
            LengthPreserving.ROLLING_SKEW,
            lambda x: x.rolling_skew(window_size=3, bias=False),
        ),
    }
    MANY_INPUTS: ClassVar[dict[str, tuple[CellCheck, tuple[Input, ...]]]] = {
        "shift": (
            CellCheck(LengthPreserving.SHIFT, TemplateCall("pl.col('x').shift()")),
            (INTEGER, COUNT),
        ),
        "shift with a fill": (
            CellCheck(LengthPreserving.SHIFT_AND_FILL, lambda x, n, f: x.shift(n, fill_value=f)),
            (INTEGER, COUNT, COUNT),
        ),
        "diff": (
            CellCheck(
                LengthPreserving.DIFF,
                TemplateCall("pl.col('x').diff()"),
                refused=Values.NUMBER_TEXT,
            ),
            (INTEGER, COUNT),
        ),
        "pct_change": (
            CellCheck(LengthPreserving.PCT_CHANGE, TemplateCall("pl.col('x').pct_change()")),
            (INTEGER, COUNT),
        ),
        "interpolate_by": (
            CellCheck(
                LengthPreserving.INTERPOLATE_BY,
                TemplateCall("pl.col('x').interpolate_by(pl.col('x'))"),
            ),
            (FLOAT, INTEGER),
        ),
        "ewm_mean_by": (
            CellCheck(
                LengthPreserving.EWM_MEAN_BY,
                TemplateCall("pl.col('x').ewm_mean_by(by='t', half_life='1d')"),
            ),
            (FLOAT, KEY),
        ),
        "ewm_sum_by": (
            CellCheck(
                LengthPreserving.EWM_SUM_BY,
                TemplateCall("pl.col('x').ewm_sum_by(by='t', half_life='1d')"),
            ),
            (FLOAT, KEY),
        ),
        "rolling_corr": (
            CellCheck(
                LengthPreserving.ROLLING_CORR_COV,
                TemplateCall("pl.rolling_corr('x', 'y', window_size=2)"),
            ),
            (FLOAT, FLOAT),
        ),
        "rolling_cov": (
            CellCheck(
                LengthPreserving.ROLLING_CORR_COV,
                TemplateCall("pl.rolling_cov('x', 'y', window_size=2)"),
            ),
            (FLOAT, FLOAT),
        ),
        "rolling_min_by": (
            CellCheck(
                LengthPreserving.ROLLING_MIN_BY,
                TemplateCall("pl.col('x').rolling_min_by(by='t', window_size='1d')"),
            ),
            (FLOAT, KEY),
        ),
        "rolling_max_by": (
            CellCheck(
                LengthPreserving.ROLLING_MAX_BY,
                TemplateCall("pl.col('x').rolling_max_by(by='t', window_size='1d')"),
            ),
            (FLOAT, KEY),
        ),
        "rolling_sum_by": (
            CellCheck(
                LengthPreserving.ROLLING_SUM_BY,
                TemplateCall("pl.col('x').rolling_sum_by(by='t', window_size='1d')"),
            ),
            (FLOAT, KEY),
        ),
        "rolling_mean_by": (
            CellCheck(
                LengthPreserving.ROLLING_MEAN_BY,
                TemplateCall("pl.col('x').rolling_mean_by(by='t', window_size='1d')"),
            ),
            (FLOAT, KEY),
        ),
        "rolling_median_by": (
            CellCheck(
                LengthPreserving.ROLLING_QUANTILE_BY,
                TemplateCall("pl.col('x').rolling_median_by(by='t', window_size='1d')"),
            ),
            (FLOAT, KEY),
        ),
        "rolling_quantile_by": (
            CellCheck(
                LengthPreserving.ROLLING_QUANTILE_BY,
                TemplateCall(
                    "pl.col('x').rolling_quantile_by(by='t', window_size='1d', quantile=0.5)"
                ),
            ),
            (FLOAT, KEY),
        ),
        "rolling_std_by": (
            CellCheck(
                LengthPreserving.ROLLING_STD_BY,
                TemplateCall("pl.col('x').rolling_std_by(by='t', window_size='1d')"),
            ),
            (FLOAT, KEY),
        ),
        "rolling_var_by": (
            CellCheck(
                LengthPreserving.ROLLING_VAR_BY,
                TemplateCall("pl.col('x').rolling_var_by(by='t', window_size='1d')"),
            ),
            (FLOAT, KEY),
        ),
        "rolling_rank_by": (
            CellCheck(
                LengthPreserving.ROLLING_RANK_BY,
                TemplateCall("pl.col('x').rolling_rank_by(by='t', window_size='1d')"),
            ),
            (FLOAT, KEY),
        ),
    }
    TWO_INPUT_OPTIONS: ClassVar[dict[str, tuple[CellCheck, tuple[Input, Input]]]] = {
        "shift 2": (
            CellCheck(LengthPreserving.SHIFT, TemplateCall("pl.col('x').shift(-2)")),
            (INTEGER, COUNT),
        ),
        "diff drop": (
            CellCheck(
                LengthPreserving.DIFF,
                TemplateCall("pl.col('x').diff(null_behavior='drop')"),
                refused=Values.NUMBER_TEXT,
            ),
            (INTEGER, COUNT),
        ),
        "rolling_mean_by integers": (
            CellCheck(
                LengthPreserving.ROLLING_MEAN_BY,
                TemplateCall("pl.col('x').rolling_mean_by(by='t', window_size='2i')"),
            ),
            (FLOAT, INTEGER),
        ),
        "rolling_sum_by zero": (
            CellCheck(
                LengthPreserving.ROLLING_SUM_BY,
                TemplateCall("pl.col('x').rolling_sum_by(by='t', window_size='0i')"),
            ),
            (FLOAT, INTEGER),
        ),
        "rolling_max_by month": (
            CellCheck(
                LengthPreserving.ROLLING_MAX_BY,
                TemplateCall(
                    "pl.col('x').rolling_max_by(by='t', window_size='1mo', closed='both')"
                ),
            ),
            (FLOAT, KEY),
        ),
        "rolling_rank_by left": (
            CellCheck(
                LengthPreserving.ROLLING_RANK_BY,
                TemplateCall(
                    "pl.col('x').rolling_rank_by(by='t', window_size='1d', closed='left')"
                ),
            ),
            (FLOAT, KEY),
        ),
        "rolling_rank_by dense": (
            CellCheck(
                LengthPreserving.ROLLING_RANK_BY,
                TemplateCall(
                    "pl.col('x').rolling_rank_by(by='t', window_size='1d', method='dense')"
                ),
            ),
            (FLOAT, KEY),
        ),
        "ewm_mean_by negative": (
            CellCheck(
                LengthPreserving.EWM_MEAN_BY,
                TemplateCall("pl.col('x').ewm_mean_by(by='t', half_life='-1d')"),
            ),
            (FLOAT, KEY),
        ),
        "ewm_sum_by month": (
            CellCheck(
                LengthPreserving.EWM_SUM_BY,
                TemplateCall("pl.col('x').ewm_sum_by(by='t', half_life='1mo')"),
            ),
            (FLOAT, KEY),
        ),
        "rolling_cov min_samples": (
            CellCheck(
                LengthPreserving.ROLLING_CORR_COV,
                TemplateCall("pl.rolling_cov('x', 'y', window_size=2, min_samples=3, ddof=0)"),
            ),
            (FLOAT, FLOAT),
        ),
    }
    OUT_OF_PROCESS: ClassVar[dict[str, tuple[Definition, str, int]]] = {
        "ewm_mean": (LengthPreserving.EWM_MEAN, "pl.col('x').ewm_mean(alpha=0.5)", 1),
        "ewm_sum": (LengthPreserving.EWM_SUM, "pl.col('x').ewm_sum(alpha=0.5)", 1),
        "ewm_std": (LengthPreserving.EWM_STD, "pl.col('x').ewm_std(alpha=0.5)", 1),
        "ewm_var": (LengthPreserving.EWM_VAR, "pl.col('x').ewm_var(alpha=0.5)", 1),
        "pct_change": (LengthPreserving.PCT_CHANGE, "pl.col('x').pct_change()", 2),
    }

    @classmethod
    def many_input_cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give the cells of the nodes of several inputs: each node beside each cell of its sample.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> cells = Nodes.many_input_cells()
            >>> len(cells) > 8500, cells[0][0]
            (True, 'shift')
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (_, typical) in cls.MANY_INPUTS.items():
            cells.extend((name, cell) for cell in NodeSample().cells(len(typical), typical))
        return cells

    @classmethod
    def two_input_option_cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give the cells of the calls of two inputs at other values of an option.

        Each call takes the smaller sample at each place, beside the typical
        input at the other place.

        Returns:
            The name of the call and the cell, for each cell.

        Example:
            >>> cells = Nodes.two_input_option_cells()
            >>> len(cells), cells[0][1][1].label
            (1260, 'lit int 1/direct')
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (_, (left, right)) in cls.TWO_INPUT_OPTIONS.items():
            for (entry,) in NodeSample().options():
                cells.extend([(name, (entry, right)), (name, (left, entry))])
        return cells


class IsolatedCell:
    """Run a node over a struct column in a subprocess, and give its answer or how it stops.

    A node that ends the process or never returns cannot run in the process
    of the tests. The subprocess builds a frame of two groups with a struct
    column, runs the node in a `select` or in `agg`, and shows the type. A
    return code that a signal sets means that the process ended, and a
    timeout means that it never returns.

    Example:
        >>> IsolatedCell("pl.col('x').reverse()").answer(agg=False)
        Struct({'a': Int8})
    """

    TIMEOUT: ClassVar[int] = 20
    CODE: ClassVar[str] = (
        "import polars as pl\n"
        "values = [{{'a': 1}}, {{'a': 2}}, {{'a': 3}}, {{'a': 1}}, {{'a': 2}}]\n"
        "x = pl.Series('x', values, dtype=pl.Struct({{'a': pl.Int8}}))\n"
        "frame = pl.DataFrame([pl.Series('g', [1, 1, 1, 2, 2]), x])\n"
        "node = ({node}).alias('o')\n"
        "groups = frame.group_by('g', maintain_order=True)\n"
        "try:\n"
        "    output = groups.agg(node) if {agg} else frame.select(node)\n"
        "except BaseException as error:\n"
        "    name = type(error).__name__\n"
        "    print('panics' if name == 'PanicException' else 'raises ' + name)\n"
        "else:\n"
        "    dtype = output['o'].dtype\n"
        "    print('dtype', repr(dtype.inner if {agg} else dtype))\n"
    )

    def __init__(self, node: str) -> None:
        """Keep the code of the node, over the column `x`.

        Args:
            node: The code, for example `pl.col('x').ewm_mean(alpha=0.5)`.

        Example:
            >>> IsolatedCell("pl.col('x')")._node
            "pl.col('x')"
        """
        self._node = node

    def answer(self, *, agg: bool) -> DType | Stopped:
        """Give the answer of the node in a context: the type of the values, or how the run stops.

        Args:
            agg: Whether to run the node in `agg`, where the answer is the type inside the list.

        Returns:
            The type, or how the run stops.

        Example:
            >>> IsolatedCell("pl.col('x').cum_max()").answer(agg=True)
            raises InvalidOperationError
        """
        code = self.CODE.format(node=self._node, agg=agg)
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
            >>> IsolatedCell("x")._read("dtype Float64"), IsolatedCell("x")._read("panics")
            (Float64, panics)
        """
        if text == "panics":
            return Stopped(Stop.PANICS)
        if text.startswith("raises "):
            return Stopped(Stop.RAISES, text.removeprefix("raises "))
        dtype = eval(text.removeprefix("dtype "), vars(pl))  # ruff: ignore[suspicious-eval-usage]
        return DType.of(dtype)


class StructRule:
    """Give how the rules say that a node stops over a struct column, in a context.

    The struct column has one value per row. A node of two inputs takes the
    literal count `1` at its second place.

    Example:
        >>> StructRule(LengthPreserving.EWM_MEAN, 1).answer(agg=False)
        hangs
    """

    STRUCT: ClassVar[DType] = DType("Struct", ((("a", DType("Int8")),),))

    def __init__(self, definition: Definition, arity: int) -> None:
        """Keep the definition of the node and its number of inputs.

        Args:
            definition: The definition.
            arity: The number of inputs: one, or two for a node with a count.

        Example:
            >>> StructRule(LengthPreserving.EWM_SUM, 1)._leaves[0].form
            <RowForm.PER_ROW: 'per row'>
        """
        self._definition = definition
        leaves = [ColumnLeaf(self.STRUCT, RowForm.PER_ROW), Nodes.COUNT.leaf]
        self._leaves = leaves[:arity]

    def answer(self, *, agg: bool) -> Stopped | None:
        """Give how the rules say that the run stops, or `None` where they say that it runs.

        Args:
            agg: Whether to read the answer in `agg`.

        Returns:
            How the run stops, or `None`.

        Example:
            >>> StructRule(LengthPreserving.PCT_CHANGE, 2).answer(agg=True)
            crashes
        """
        answers = NodeEvaluation(self._definition, self._leaves, CellCheck.GROUPS).answers()
        answer = answers.run_agg if agg else answers.run_select
        if not isinstance(answer, Refused):
            return None
        return Stopped(answer.stop, None if answer.error is None else answer.error.value)


@pytest.mark.parametrize("node", list(Nodes.ONE_INPUT))
@pytest.mark.parametrize("cell", NodeSample().cells(1), ids=CellCheck.label)
def test_node_of_one_input_gives_the_answers_of_polars(node: str, cell: tuple[Input, ...]) -> None:
    """The four answers of a node of one input match polars, at one input."""
    problems = Nodes.ONE_INPUT[node].problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("node", list(Nodes.ONE_INPUT_OPTIONS))
@pytest.mark.parametrize("cell", NodeSample().options(), ids=CellCheck.label)
def test_option_of_one_input_gives_the_answers_of_polars(
    node: str, cell: tuple[Input, ...]
) -> None:
    """The four answers of a node of one input at another value of an option match polars."""
    problems = Nodes.ONE_INPUT_OPTIONS[node].problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("cell", NodeSample().options(), ids=CellCheck.label)
def test_qcut_over_a_nan_quantile_is_unchecked_only_over_nulls(cell: tuple[Input, ...]) -> None:
    """`qcut` with a NaN quantile matches polars, but its run over only nulls is `Unchecked`.

    The JSON form of the tree writes a quantile that is NaN or infinite as
    null. Over only nulls, polars raises for NaN and runs for an infinite
    quantile, so the rules cannot decide the run there.
    """
    check = CellCheck(LengthPreserving.QCUT, lambda x: x.qcut([float("nan")]))
    problems = check.problems(list(cell))
    if cell[0].label not in {"Null/per row", "lit null/direct"}:
        assert not problems, "\n".join(problems)
        return
    undecided = "rules unchecked: the error of qcut over a quantile that is NaN or infinite"
    assert len(problems) == 2, problems
    assert all(undecided in problem for problem in problems), problems


@pytest.mark.parametrize(
    ("node", "cell"),
    Nodes.many_input_cells(),
    ids=[f"{node}-{CellCheck.label(cell)}" for node, cell in Nodes.many_input_cells()],
)
def test_node_of_many_inputs_gives_the_answers_of_polars(
    node: str, cell: tuple[Input, ...]
) -> None:
    """The four answers of a node of two inputs or more match polars, at one cell of its sample."""
    check, _ = Nodes.MANY_INPUTS[node]
    problems = check.problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("node", "cell"),
    Nodes.two_input_option_cells(),
    ids=[f"{node}-{CellCheck.label(cell)}" for node, cell in Nodes.two_input_option_cells()],
)
def test_option_of_two_inputs_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a node of two inputs at another value of an option match polars."""
    check, _ = Nodes.TWO_INPUT_OPTIONS[node]
    problems = check.problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("node", list(Nodes.OUT_OF_PROCESS))
@pytest.mark.parametrize("agg", [False, True], ids=["select", "agg"])
def test_node_over_a_struct_ends_or_hangs_the_process(node: str, *, agg: bool) -> None:
    """Over a struct, the run of the node ends the process or never returns, as the rules say."""
    definition, code, arity = Nodes.OUT_OF_PROCESS[node]
    rule = StructRule(definition, arity).answer(agg=agg)
    assert rule is not None
    assert rule.stop in CellCheck.OUT_OF_PROCESS, rule
    assert IsolatedCell(code).answer(agg=agg) == rule
