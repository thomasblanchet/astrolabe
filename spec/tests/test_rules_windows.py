"""The rules of `agg` over the windows of `rolling` and `group_by_dynamic`, against polars.

`rolling` and `group_by_dynamic` group the rows of a frame into windows along
an index column, and `agg` runs over each window as over a group. The rules
take the windows of the test frame in place of the groups (`WindowFrame`):
windows that share rows, and windows that do not. Over windows that share
rows, a function of one input that runs window by window builds each list
at the type of its data, and a slice keeps the windows.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import WindowFrame


class WindowTrees:
    """Hold the trees that the tests run in each context of windows.

    Example:
        >>> "a group-aware function of text" in WindowTrees.TREES
        True
    """

    TREES: ClassVar[dict[str, pl.Expr]] = {
        "a group-aware function of text": pl.col("t").rolling_mean(2),
        "a group-aware function after a slice": pl.col("t").head(2).rolling_mean(2),
        "a group-aware function of numbers": pl.col("x").rolling_sum(2),
        "a group-aware function of one value": pl.col("t").first().rolling_mean(2),
        "an interpolation of text": pl.col("t").interpolate(),
        "an aggregation": pl.col("x").sum(),
        "a function after an aggregation": pl.col("x").sum().rolling_mean(2),
        "a column": pl.col("x"),
        "an elementwise node": pl.col("x").abs() + 1,
        "a cumulative sum": pl.col("x").cum_sum(),
        "a value beside one value for each window": pl.col("x") + pl.col("x").first(),
        "a list function": pl.col("l").list.len(),
        "a sort of text": pl.col("t").sort(),
        "two group-aware inputs": pl.rolling_corr(pl.col("x"), pl.col("x"), window_size=2),
    }


@pytest.mark.parametrize("name", sorted(WindowTrees.TREES))
def test_the_rules_give_the_answers_of_polars_over_windows(name: str) -> None:
    """The planner type and the run type of each context of windows are those of polars."""
    expression = WindowTrees.TREES[name]
    assert WindowFrame().rules(expression) == WindowFrame().polars(expression)


def test_a_group_aware_function_over_shared_rows_takes_the_type_of_its_data() -> None:
    """Over windows that share rows, `rolling_mean` of text runs. Over a `group_by`, it panics."""
    frame = WindowFrame.FRAME
    expression = pl.col("t").rolling_mean(2)
    rolled = frame.rolling("i", period="2i").agg(expression)
    assert rolled.schema["t"] == pl.List(pl.Float64)
    with pytest.raises(BaseException, match="invalid series dtype"):
        frame.group_by("g").agg(expression)
