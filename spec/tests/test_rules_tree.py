"""Check the reader of the tree of polars against every node of the census.

The census finds the inputs of each node from polars' own tree, and builds
the node again with a placeholder column at each input. The tree reader must
find the same inputs, in the same order, and the identity that the census
node has. A failure names the census call.
"""

from __future__ import annotations


import polars as pl
import pytest

from spec.tools.node_census.surface import CensusSurface, NodeTemplate
from astrolabe._rules.tree import ExpressionTree


@pytest.mark.parametrize("template", CensusSurface().templates(), ids=lambda template: template.key)
def test_the_tree_holds_the_inputs_of_each_census_node(template: NodeTemplate) -> None:
    """Find the placeholder inputs of a census node in order."""
    placeholders: list[pl.Expr] = [pl.col(f"__{index}") for index in range(template.inputs())]
    tree = ExpressionTree().of(template.build(placeholders))
    names = [node.serialized for node in tree.inputs]
    assert names == [{"Column": f"__{index}"} for index in range(template.inputs())]


def test_an_evaluation_stands_apart_from_the_inputs() -> None:
    """Keep the evaluation of `list.eval` out of the inputs of the node."""
    tree = ExpressionTree().of(pl.col("l").list.eval(pl.element() + 1))
    assert [node.identity.shape for node in tree.inputs] == ["Column"]
    assert [node.identity.shape for node in tree.evaluation] == ["BinaryExpr"]


def test_the_sort_key_of_an_over_is_an_input_and_its_options_are_not() -> None:
    """Read the sort key of `over(order_by=...)` after the keys, and leave its options out."""
    tree = ExpressionTree().of(pl.col("x").cum_sum().over("g", order_by="t"))
    assert [node.serialized for node in tree.inputs[1:]] == [{"Column": "g"}, {"Column": "t"}]
