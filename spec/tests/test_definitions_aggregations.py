"""Check the definitions of the aggregations against polars, node by node.

The family holds the nodes that give one value from their input, or one value
for each group in `agg`: the `Agg` nodes of polars, such as `min`, `std` and
`implode`, and the functions whose options give one value, such as `any`,
`product`, `quantile`, `corr` and `min_by`. Each test builds the node of polars
over the inputs of one cell, and compares the four answers of polars with the
four answers of the rules (`CellCheck`): the planner type and the run type, in
a `select` and in `agg` over two groups of other sizes.

A node of one input takes every input (`NodeSample`). A node of two inputs
takes about 500 pairs beside a typical input at each place (`PairSample`). A
call that differs from the main call of its node only by an option takes a
smaller sample: each column of one value per row and each direct literal, at
one place, beside the typical input at the other place. The frames hold text
that parses as a number, and a NaN as the first float (`LiveFrames`). `skew`
and `kurtosis` cast their input to `Float64` with a strict cast, so where the
rules refuse their run, polars runs them over text that parses: other text
fails the cast first, with another error, a fact of the values.

Each test names the node and the cell, so a failure names both.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import (
    ApiGate,
    CellCheck,
    Input,
    Inputs,
    NodeSample,
    PairSample,
    TemplateCall,
    Values,
)
from astrolabe._rules.definitions.aggregations import Aggregations
from astrolabe._rules.node_evaluation import LiteralLeaf


class AggregationNodes:
    """Hold the nodes of the tests, each with its call and the typical inputs of its sample.

    The typical value of `corr` and `cov` is a `Float64` column at each place.
    The typical value and key of `min_by` and `max_by` are `Int64` columns.
    `index_of` takes an `Int64` column and the literal `1`. `quantile` takes a
    `Float64` column and the literal `0.5`, a quantile between 0 and 1, where
    the literal float of the census, 1.5, is outside of them.

    Example:
        >>> "all" in AggregationNodes.ONE_INPUT
        True
        >>> AggregationNodes.HALF.label
        'lit float 0.5/direct'
    """

    FLOATS: ClassVar[Input] = Inputs().labelled("Float64/per row")
    INTEGERS: ClassVar[Input] = Inputs().labelled("Int64/per row")
    ONE: ClassVar[Input] = Inputs().labelled("lit int 1/direct")
    HALF: ClassVar[Input] = Input("lit float 0.5/direct", LiteralLeaf(0.5), pl.lit(0.5))

    ONE_INPUT: ClassVar[dict[str, CellCheck]] = ApiGate().kept({
        "agg_groups": CellCheck(Aggregations.AGG_GROUPS, lambda x: x.agg_groups()),
        "count": CellCheck(Aggregations.COUNT, lambda x: x.count()),
        "len": CellCheck(Aggregations.COUNT, lambda x: x.len()),
        "first": CellCheck(Aggregations.FIRST, lambda x: x.first()),
        "first non-null": CellCheck(
            Aggregations.FIRST_NON_NULL,
            lambda x: x.first(ignore_nulls=True),
        ),
        "last": CellCheck(Aggregations.LAST, lambda x: x.last()),
        "last non-null": CellCheck(Aggregations.LAST_NON_NULL, lambda x: x.last(ignore_nulls=True)),
        "implode": CellCheck(Aggregations.IMPLODE, lambda x: x.implode()),
        "item": CellCheck(Aggregations.ITEM, lambda x: x.item()),
        "max": CellCheck(Aggregations.MAX, lambda x: x.max()),
        "nan_max": CellCheck(Aggregations.MAX, lambda x: x.nan_max()),
        "min": CellCheck(Aggregations.MIN, lambda x: x.min()),
        "nan_min": CellCheck(Aggregations.MIN, lambda x: x.nan_min()),
        "mean": CellCheck(Aggregations.MEAN, lambda x: x.mean()),
        "median": CellCheck(Aggregations.MEDIAN, lambda x: x.median()),
        "n_unique": CellCheck(Aggregations.N_UNIQUE, lambda x: x.n_unique()),
        "std": CellCheck(Aggregations.STD, lambda x: x.std()),
        "var": CellCheck(Aggregations.VAR, lambda x: x.var()),
        "approx_n_unique": CellCheck(Aggregations.APPROX_N_UNIQUE, lambda x: x.approx_n_unique()),
        "arg_max": CellCheck(Aggregations.ARG_MAX, lambda x: x.arg_max()),
        "arg_min": CellCheck(Aggregations.ARG_MIN, lambda x: x.arg_min()),
        "bitwise_and": CellCheck(Aggregations.BITWISE_AND, lambda x: x.bitwise_and()),
        "bitwise_or": CellCheck(Aggregations.BITWISE_OR, lambda x: x.bitwise_or()),
        "bitwise_xor": CellCheck(Aggregations.BITWISE_XOR, lambda x: x.bitwise_xor()),
        "all": CellCheck(Aggregations.ALL, lambda x: x.all()),
        "any": CellCheck(Aggregations.ANY, lambda x: x.any()),
        "has_nulls": CellCheck(Aggregations.HAS_NULLS, lambda x: x.has_nulls()),
        "is_empty": CellCheck(Aggregations.IS_EMPTY, lambda x: x.is_empty()),
        "is_sorted": CellCheck(Aggregations.IS_SORTED, lambda x: x.is_sorted()),
        "null_count": CellCheck(Aggregations.NULL_COUNT, lambda x: x.null_count()),
        "entropy": CellCheck(Aggregations.ENTROPY, lambda x: x.entropy()),
        "product": CellCheck(Aggregations.PRODUCT, lambda x: x.product()),
        "skew": CellCheck(Aggregations.SKEW, lambda x: x.skew(), Values.NUMBER_TEXT),
        "kurtosis": CellCheck(Aggregations.KURTOSIS, lambda x: x.kurtosis(), Values.NUMBER_TEXT),
        "lower_bound": CellCheck(Aggregations.LOWER_BOUND, lambda x: x.lower_bound()),
        "upper_bound": CellCheck(Aggregations.UPPER_BOUND, lambda x: x.upper_bound()),
    })
    ONE_INPUT_OPTIONS: ClassVar[dict[str, CellCheck]] = ApiGate().kept({
        "item allow_empty": CellCheck(Aggregations.ITEM, lambda x: x.item(allow_empty=True)),
        "implode unordered": CellCheck(
            Aggregations.IMPLODE,
            lambda x: x.implode(maintain_order=False),
        ),
        "std ddof 0": CellCheck(Aggregations.STD, lambda x: x.std(ddof=0)),
        "var ddof 0": CellCheck(Aggregations.VAR, lambda x: x.var(ddof=0)),
        "all with nulls": CellCheck(Aggregations.ALL, lambda x: x.all(ignore_nulls=False)),
        "any with nulls": CellCheck(Aggregations.ANY, lambda x: x.any(ignore_nulls=False)),
        "is_empty ignore_nulls": CellCheck(
            Aggregations.IS_EMPTY,
            lambda x: x.is_empty(ignore_nulls=True),
        ),
        "is_sorted descending": CellCheck(
            Aggregations.IS_SORTED,
            lambda x: x.is_sorted(descending=True, nulls_last=True),
        ),
        "is_sorted inferred": CellCheck(
            Aggregations.IS_SORTED,
            lambda x: x.is_sorted(descending=None, nulls_last=None),
        ),
        "entropy base 2": CellCheck(
            Aggregations.ENTROPY,
            lambda x: x.entropy(base=2, normalize=False),
        ),
        "skew unbiased": CellCheck(
            Aggregations.SKEW,
            lambda x: x.skew(bias=False),
            Values.NUMBER_TEXT,
        ),
        "kurtosis pearson": CellCheck(
            Aggregations.KURTOSIS,
            lambda x: x.kurtosis(fisher=False, bias=False),
            Values.NUMBER_TEXT,
        ),
    })
    TWO_INPUTS: ClassVar[dict[str, tuple[CellCheck, tuple[Input, Input]]]] = {
        "corr": (
            CellCheck(Aggregations.CORRELATION, TemplateCall("pl.corr('x', 'y')")),
            (FLOATS, FLOATS),
        ),
        "cov": (
            CellCheck(Aggregations.CORRELATION, TemplateCall("pl.cov('x', 'y')")),
            (FLOATS, FLOATS),
        ),
        "max_by": (
            CellCheck(Aggregations.MAX_BY, lambda x, y: x.max_by(y)),
            (INTEGERS, INTEGERS),
        ),
        "min_by": (
            CellCheck(Aggregations.MIN_BY, lambda x, y: x.min_by(y)),
            (INTEGERS, INTEGERS),
        ),
        "index_of": (
            CellCheck(Aggregations.INDEX_OF, lambda x, y: x.index_of(y)),
            (INTEGERS, ONE),
        ),
        "quantile": (
            CellCheck(Aggregations.QUANTILE, lambda x, y: x.quantile(y)),
            (FLOATS, HALF),
        ),
    }
    TWO_INPUT_OPTIONS: ClassVar[dict[str, tuple[CellCheck, tuple[Input, Input]]]] = {
        "corr spearman": (
            CellCheck(
                Aggregations.CORRELATION,
                TemplateCall("pl.corr('x', 'y', method='spearman')"),
            ),
            (FLOATS, FLOATS),
        ),
        "corr spearman with nans": (
            CellCheck(
                Aggregations.CORRELATION,
                TemplateCall("pl.corr('x', 'y', method='spearman', propagate_nans=True)"),
            ),
            (FLOATS, FLOATS),
        ),
        "cov ddof 0": (
            CellCheck(Aggregations.CORRELATION, TemplateCall("pl.cov('x', 'y', ddof=0)")),
            (FLOATS, FLOATS),
        ),
        "quantile linear": (
            CellCheck(
                Aggregations.QUANTILE,
                lambda x, y: x.quantile(y, interpolation="linear"),
            ),
            (FLOATS, HALF),
        ),
    }

    @classmethod
    def one_input_cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node of one input beside each cell of its sample.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> len(AggregationNodes.one_input_cells()) > 8000
            True
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name in cls.ONE_INPUT:
            cells.extend((name, cell) for cell in NodeSample().cells(1))
        for name in cls.ONE_INPUT_OPTIONS:
            cells.extend((name, cell) for cell in NodeSample().options())
        return cells

    @classmethod
    def two_input_cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node of two inputs beside each cell of its sample.

        A call at another value of an option takes the typical input at one
        place beside each input of the smaller sample at the other place.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> len(AggregationNodes.two_input_cells())
            3504
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (_, typical) in cls.TWO_INPUTS.items():
            cells.extend((name, cell) for cell in PairSample(Inputs().all(), *typical).pairs())
        options = [cell[0] for cell in NodeSample().options()]
        for name, (_, (left, right)) in cls.TWO_INPUT_OPTIONS.items():
            cells.extend((name, (entry, right)) for entry in options)
            cells.extend((name, (left, entry)) for entry in options)
        return cells

    @classmethod
    def check(cls, name: str) -> CellCheck:
        """Give the comparison of the rules with polars for the node of a name.

        Args:
            name: The name of the node.

        Returns:
            The comparison.

        Example:
            >>> AggregationNodes.check("quantile linear").label(())
            ''
        """
        checks = {**cls.ONE_INPUT, **cls.ONE_INPUT_OPTIONS}
        for nodes in (cls.TWO_INPUTS, cls.TWO_INPUT_OPTIONS):
            checks.update({key: check for key, (check, _) in nodes.items()})
        return checks[name]

    @classmethod
    def ids(cls, cells: list[tuple[str, tuple[Input, ...]]]) -> list[str]:
        """Give the identifier of each test: the node and the cell.

        Args:
            cells: The name of the node and the cell, for each cell.

        Returns:
            The identifiers.

        Example:
            >>> cell = (AggregationNodes.FLOATS, AggregationNodes.HALF)
            >>> AggregationNodes.ids([("quantile", cell)])
            ['quantile-Float64/per row, lit float 0.5/direct']
        """
        return [f"{name}-{CellCheck.label(cell)}" for name, cell in cells]


@pytest.mark.parametrize(
    ("node", "cell"),
    AggregationNodes.one_input_cells(),
    ids=AggregationNodes.ids(AggregationNodes.one_input_cells()),
)
def test_aggregation_of_one_input_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of an aggregation of one input match polars, at one input."""
    problems = AggregationNodes.check(node).problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("node", "cell"),
    AggregationNodes.two_input_cells(),
    ids=AggregationNodes.ids(AggregationNodes.two_input_cells()),
)
def test_aggregation_of_two_inputs_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of an aggregation of two inputs match polars, at one pair of its sample."""
    problems = AggregationNodes.check(node).problems(list(cell))
    assert not problems, "\n".join(problems)
