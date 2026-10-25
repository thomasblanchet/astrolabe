"""The value that a node computes over a literal, as a later rule reads it.

A rule can read the value of a literal input: the count of `diff`, or the
text that a strict cast parses. Where the input is a node over a literal, the
run meets the value that the node computes, and the definition of the node
states it (`ValueRule`). Each test here runs the node over literals of
several values in polars, and compares what polars computes with the literal
that the rules give. A rule that gives no literal claims nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import TreeFrame
from astrolabe._rules.definitions.aggregations import Aggregations
from astrolabe._rules.definitions.elementwise import Elementwise
from astrolabe._rules.definitions.length_preserving import LengthPreserving
from astrolabe._rules.definitions.node_kinds import NodeKinds
from astrolabe._rules.definitions.other_lengths import OtherLengths
from astrolabe._rules.definitions.registry import Definitions
from astrolabe._rules.definitions.strings import Strings
from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.execution import GroupBy
from astrolabe._rules.failures import PolarsRefusal, UndecidedError
from astrolabe._rules.node_evaluation import ColumnLeaf
from astrolabe._rules.options import RowForm
from astrolabe._rules.tree import ExpressionTree
from astrolabe._rules.tree_evaluation import TreeEvaluation

if TYPE_CHECKING:
    from collections.abc import Callable

    from astrolabe._rules.definitions.definition import Definition
    from astrolabe._rules.type_functions import Operand


class ValueCalls:
    """Hold a call of each node whose definition states the value that it computes.

    Example:
        >>> sorted(ValueCalls.CALLS)[:2]
        ['cast Boolean', 'cast Float64']
    """

    CALLS: ClassVar[dict[str, tuple[Definition, Callable[[pl.Expr], pl.Expr]]]] = {
        "min": (Aggregations.MIN, lambda x: x.min()),
        "max": (Aggregations.MAX, lambda x: x.max()),
        "first": (Aggregations.FIRST, lambda x: x.first()),
        "last": (Aggregations.LAST, lambda x: x.last()),
        "first non null": (Aggregations.FIRST_NON_NULL, lambda x: x.first(ignore_nulls=True)),
        "last non null": (Aggregations.LAST_NON_NULL, lambda x: x.last(ignore_nulls=True)),
        "item": (Aggregations.ITEM, lambda x: x.item()),
        "unique": (OtherLengths.UNIQUE, lambda x: x.unique()),
        "mode": (OtherLengths.MODE, lambda x: x.mode()),
        "drop_nulls": (OtherLengths.DROP_NULLS, lambda x: x.drop_nulls()),
        "sort": (NodeKinds.SORT, lambda x: x.sort()),
        "cum_max": (LengthPreserving.CUM_MAX, lambda x: x.cum_max()),
        "cum_min": (LengthPreserving.CUM_MIN, lambda x: x.cum_min()),
        "reverse": (LengthPreserving.REVERSE, lambda x: x.reverse()),
        "rechunk": (LengthPreserving.RECHUNK, lambda x: x.rechunk()),
        "str.join": (Strings.CONCAT_VERTICAL, lambda x: x.str.join()),
        "cast Int16": (Elementwise.CAST, lambda x: x.cast(pl.Int16)),
        "cast Float64": (Elementwise.CAST, lambda x: x.cast(pl.Float64)),
        "cast String": (Elementwise.CAST, lambda x: x.cast(pl.String)),
        "cast Boolean": (Elementwise.CAST, lambda x: x.cast(pl.Boolean)),
    }
    VALUES: ClassVar[list[object]] = [1, -2, 300, 2**40, 1.5, "a", "12", True]


class ComputedOperand:
    """Give the input that a node over literals is to a later node, as the rules compute it.

    Example:
        >>> ComputedOperand().of(pl.lit("a").first()).value
        'a'
    """

    def of(self, expression: pl.Expr) -> Operand | None:
        """Give the operand.

        Args:
            expression: The node over literals.

        Returns:
            The operand as the run computes it, or `None` where the rules
            refuse the node.

        Example:
            >>> ComputedOperand().of(pl.lit(1).abs()).literal
            False
            >>> ComputedOperand().of(pl.lit("a").cast(pl.Boolean)) is None
            True
        """
        columns = {
            name: ColumnLeaf(DType.of(dtype), RowForm.PER_ROW)
            for name, dtype in TreeFrame.FRAME.schema.items()
        }
        evaluation = TreeEvaluation(columns, GroupBy((3,)))
        try:
            return evaluation.computed(ExpressionTree().of(expression))
        except (PolarsRefusal, UndecidedError):
            return None


class PolarsColumn:
    """Give the column that polars computes for a node over literals, or `None` where it stops.

    Example:
        >>> PolarsColumn().of(pl.lit("a").first()).to_list()
        ['a']
        >>> PolarsColumn().of(pl.lit("a").cast(pl.Int8)) is None
        True
    """

    def of(self, expression: pl.Expr) -> pl.Series | None:
        """Give the column.

        A panic of polars is a `BaseException`, so the run catches that too.

        Args:
            expression: The node over literals.

        Returns:
            The column, or `None` where polars raises or panics.

        Example:
            >>> PolarsColumn().of(pl.lit(1).cast(pl.Int16)).dtype
            Int16
        """
        try:
            return pl.select(expression.alias("o"))["o"]
        except BaseException as error:  # a panic of polars is a BaseException
            if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            return None


def test_each_definition_that_states_a_value_has_a_call() -> None:
    """A rule of values that no test checks is a claim with no proof."""
    stated = {
        definition.identity
        for definition in Definitions.BY_IDENTITY.values()
        if definition.computes is not None
    }
    assert stated == {definition.identity for definition, _ in ValueCalls.CALLS.values()}


@pytest.mark.parametrize("name", sorted(ValueCalls.CALLS))
@pytest.mark.parametrize("value", ValueCalls.VALUES, ids=repr)
def test_the_value_of_a_node_over_a_literal_is_the_value_of_polars(
    name: str, value: object
) -> None:
    """Where the rules give a literal, polars computes that one value at that type."""
    _, call = ValueCalls.CALLS[name]
    expression = call(pl.lit(value))
    computed = ComputedOperand().of(expression)
    column = PolarsColumn().of(expression)
    if computed is None or not computed.literal or column is None:
        return
    dtype = computed.dtype.materialized() if Is.unknown(computed.dtype) else computed.dtype
    assert (column.to_list(), str(column.dtype)) == ([computed.value], dtype.render())


def test_a_node_over_a_literal_gives_its_value_to_a_later_rule() -> None:
    """A strict cast reads the text that `str.join` keeps, and raises at every run."""
    expression = pl.lit("a").str.join().cast(pl.Time)  # type: ignore[invalid-call-shape]
    assert TreeFrame().rules(expression) == TreeFrame().polars(expression)
