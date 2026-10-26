"""Every Part I catalogue row, replayed against real polars.

The catalogue states, per operation, what happens to the schema and
to the columns a signature does not declare. This module replays each
Part I operation over one frame holding declared columns and
remainder stand-ins, and asserts the full output schema, so a wrong
effect assignment fails the row that carries it. The census counts
were always computed; this is the census *content*, executed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest

from spec.tests.documents import Document
from astrolabe._rules.releases import PolarsRelease

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

DT = "Datetime(time_unit='us', time_zone=None)"
STRUCT = "Struct({'a': Int64})"

FULL: dict[str, str] = {
    "k": "Int64",
    "t": DT,
    "v": "Float64",
    "s": "String",
    "l": "List(Int64)",
    "st": STRUCT,
    "xn": "Int32",
    "xs": "String",
    "x_p": "Float64",
}


def replay_frame() -> pl.DataFrame:
    """Build the replay frame: declared columns plus remainder stand-ins.

    `k`, `t`, `v`, `s`, `l` and `st` play the declared schema; `xn`,
    `xs` and `x_p` stand in for a remainder's unknown columns, so what
    an operation does to them is what its effect row claims.

    Returns:
        The nine-column frame, `k` and `t` sorted.

    Example:
        >>> str(replay_frame().schema["xn"])
        'Int32'
    """
    built = pl.DataFrame({
        "k": pl.Series([1, 2], dtype=pl.Int64),
        "t": pl.Series(
            [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            dtype=pl.Datetime("us"),
        ),
        "v": pl.Series([1.5, 2.5], dtype=pl.Float64),
        "s": pl.Series(["x", "y"], dtype=pl.String),
        "l": pl.Series([[1], [2, 3]], dtype=pl.List(pl.Int64)),
        "st": pl.Series([{"a": 1}, {"a": 2}], dtype=pl.Struct({"a": pl.Int64})),
        "xn": pl.Series([1, 2], dtype=pl.Int32),
        "xs": pl.Series(["a", "b"], dtype=pl.String),
        "x_p": pl.Series([0.5, 0.7], dtype=pl.Float64),
    })
    return built.set_sorted("k").with_columns(pl.col("t").set_sorted())


def other_frame() -> pl.DataFrame:
    """Build the second frame the joining rows need.

    Returns:
        A frame sharing `k`, bringing `w`, and colliding on `xs`.

    Example:
        >>> other_frame().columns
        ['k', 'w', 'xs']
    """
    built = pl.DataFrame({
        "k": pl.Series([1, 2], dtype=pl.Int64),
        "w": pl.Series([9.0, 8.0], dtype=pl.Float64),
        "xs": pl.Series(["c", "d"], dtype=pl.String),
    })
    return built.set_sorted("k")


def full(**changes: str) -> dict[str, str]:
    """Give the replay frame's schema with named types replaced.

    Args:
        **changes: Each column name mapped to its new type text.

    Returns:
        The full schema with the changes applied.

    Example:
        >>> full(v="Int64")["v"]
        'Int64'
    """
    return {**FULL, **changes}


def minus(*names: str, **changes: str) -> dict[str, str]:
    """Give the replay frame's schema without some columns.

    Args:
        *names: The columns the operation removes.
        **changes: Remaining columns mapped to new type texts.

    Returns:
        The reduced schema with the changes applied.

    Example:
        >>> "v" in minus("v")
        False
    """
    kept = {name: dtype for name, dtype in FULL.items() if name not in names}
    return {**kept, **changes}


def everywhere(dtype: str) -> dict[str, str]:
    """Give the replay frame's schema with one type at every column.

    Args:
        dtype: The type text every column becomes.

    Returns:
        The retyped schema.

    Example:
        >>> everywhere("UInt32")["s"]
        'UInt32'
    """
    return {name: dtype for name in FULL}


@dataclass(frozen=True)
class Replay:
    """One catalogue row's replay: the call, and the schema it must give.

    `since` is the first release of polars where the call runs over the
    replay frame, where an older release that the checker supports refuses it.

    Example:
        >>> Replay("sort", "sort", lambda df: df.sort("k"), FULL).label
        'sort'
    """

    label: str
    method: str
    build: Callable[[pl.DataFrame], pl.DataFrame]
    expects: Mapping[str, str]
    since: str | None = None

    def runs(self) -> bool:
        """Say whether the call runs over the replay frame in the installed release of polars.

        Returns:
            Whether the installed release is `since` or a later one.

        Example:
            >>> Replay("sort", "sort", lambda df: df.sort("k"), FULL).runs()
            True
        """
        return self.since is None or PolarsRelease.installed() >= PolarsRelease.parse(self.since)

    def observed(self) -> dict[str, str]:
        """Run the call and give the output schema as text.

        Returns:
            Each output column mapped to its type's text.

        Example:
            >>> probe = Replay("sort", "sort", lambda df: df.sort("k"), FULL)
            >>> probe.observed() == dict(FULL)
            True
        """
        out = self.build(replay_frame())
        return {name: str(dtype) for name, dtype in out.schema.items()}


ROW_ONLY: tuple[Replay, ...] = (
    Replay("bottom_k", "bottom_k", lambda df: df.bottom_k(1, by="k"), FULL),
    Replay("clear", "clear", lambda df: df.clear(), FULL),
    Replay("clone", "clone", lambda df: df.clone(), FULL),
    Replay("drop_nans", "drop_nans", lambda df: df.drop_nans(), FULL),
    Replay("drop_nulls", "drop_nulls", lambda df: df.drop_nulls(), FULL),
    Replay("filter", "filter", lambda df: df.filter(pl.col("k") > 0), FULL),
    Replay("first", "first", lambda df: df.lazy().first().collect(), FULL),
    Replay("gather", "gather", lambda df: df.gather([0]), FULL, since="1.41.0"),
    Replay("gather_every", "gather_every", lambda df: df.gather_every(1), FULL),
    Replay("head", "head", lambda df: df.head(1), FULL),
    Replay("last", "last", lambda df: df.lazy().last().collect(), FULL),
    Replay("limit", "limit", lambda df: df.limit(1), FULL),
    Replay("remove", "remove", lambda df: df.remove(pl.col("k") > 99), FULL),
    Replay("reverse", "reverse", lambda df: df.reverse(), FULL),
    Replay("sample", "sample", lambda df: df.sample(1), FULL),
    Replay("set_sorted", "set_sorted", lambda df: df.set_sorted("k"), FULL),
    Replay("shift", "shift", lambda df: df.shift(1), FULL),
    Replay("slice", "slice", lambda df: df.slice(0, 1), FULL),
    Replay("sort", "sort", lambda df: df.sort("k"), FULL),
    Replay("tail", "tail", lambda df: df.tail(1), FULL),
    Replay("top_k", "top_k", lambda df: df.top_k(1, by="k"), FULL),
    Replay("unique", "unique", lambda df: df.unique(), FULL),
    Replay(
        "upsample",
        "upsample",
        lambda df: df.upsample(time_column="t", every="1d"),
        {"t": DT, **minus("t")},
    ),
    Replay(
        "merge_sorted",
        "merge_sorted",
        lambda df: df.merge_sorted(replay_frame(), key="k"),
        FULL,
        since="1.41.0",
    ),
    Replay("vstack", "vstack", lambda df: df.vstack(replay_frame()), FULL),
    Replay("extend", "extend", lambda df: df.extend(replay_frame()), FULL),
    Replay(
        "match_to_schema",
        "match_to_schema",
        lambda df: df.match_to_schema(replay_frame().schema),
        FULL,
    ),
)

COLUMN_SHAPING: tuple[Replay, ...] = (
    Replay("drop", "drop", lambda df: df.drop("v"), minus("v")),
    Replay(
        "drop_in_place",
        "drop_in_place",
        lambda df: (df.drop_in_place("v"), df)[1],
        minus("v"),
    ),
    Replay(
        "select",
        "select",
        lambda df: df.select(["k", "v"]),
        {"k": "Int64", "v": "Float64"},
    ),
    Replay(
        "select_seq",
        "select_seq",
        lambda df: df.select_seq(["k", "v"]),
        {"k": "Int64", "v": "Float64"},
    ),
    Replay(
        "rename",
        "rename",
        lambda df: df.rename({"v": "v9"}),
        {**{("v9" if n == "v" else n): t for n, t in FULL.items()}},
    ),
    Replay(
        "with_columns",
        "with_columns",
        lambda df: df.with_columns(z=pl.col("v") * 2),
        full(z="Float64"),
    ),
    Replay(
        "with_columns_seq",
        "with_columns_seq",
        lambda df: df.with_columns_seq(z=pl.col("v") * 2),
        full(z="Float64"),
    ),
    Replay(
        "with_row_index",
        "with_row_index",
        lambda df: df.with_row_index(),
        {"index": "UInt32", **FULL},
    ),
    Replay(
        "with_row_count",
        "with_row_count",
        lambda df: df.with_row_count(),
        {"row_nr": "UInt32", **FULL},
    ),
    Replay(
        "insert_column",
        "insert_column",
        lambda df: df.insert_column(1, pl.Series("v2", [7.0, 8.0], dtype=pl.Float64)),
        full(v2="Float64"),
    ),
    Replay(
        "replace_column",
        "replace_column",
        lambda df: df.replace_column(2, pl.Series("v2", [7.0, 8.0], dtype=pl.Float64)),
        {**{("v2" if n == "v" else n): t for n, t in FULL.items()}},
    ),
    Replay(
        "map_columns",
        "map_columns",
        lambda df: df.map_columns(["xn"], lambda series: series),
        FULL,
    ),
    Replay("cast_partial", "cast", lambda df: df.cast({"v": pl.Int64}), full(v="Int64")),
    Replay(
        "cast_whole",
        "cast",
        lambda df: df.drop("l", "st").cast(pl.String),
        {n: "String" for n in minus("l", "st")},
    ),
    Replay("fill_nan", "fill_nan", lambda df: df.fill_nan(0), FULL),
    Replay(
        "fill_nan_text",
        "fill_nan",
        lambda df: df.fill_nan("a"),  # type: ignore[arg-type]
        full(v="String", x_p="String"),
    ),
    Replay(
        "fill_null",
        "fill_null",
        lambda df: df.fill_null(strategy="forward"),
        FULL,
    ),
    Replay(
        "interpolate",
        "interpolate",
        lambda df: df.interpolate(),
        full(k="Float64", xn="Float64"),
    ),
    Replay(
        "to_dummies",
        "to_dummies",
        lambda df: df.to_dummies(["s"]),
        {**minus("s", "l", "st"), "s_x": "UInt8", "s_y": "UInt8", "l": "List(Int64)", "st": STRUCT},
    ),
)

RESHAPING: tuple[Replay, ...] = (
    Replay("explode", "explode", lambda df: df.explode("l"), full(l="Int64")),
    Replay(
        "unnest",
        "unnest",
        lambda df: df.unnest("st"),
        {
            ("a" if name == "st" else name): ("Int64" if name == "st" else t)
            for name, t in FULL.items()
        },
    ),
    Replay(
        "unpivot",
        "unpivot",
        lambda df: df.unpivot(on=["v", "x_p"], index=["k"]),
        {"k": "Int64", "variable": "String", "value": "Float64"},
    ),
    Replay(
        "melt",
        "melt",
        lambda df: df.melt(id_vars=["k"], value_vars=["v"]),
        {"k": "Int64", "variable": "String", "value": "Float64"},
    ),
    Replay(
        "pivot",
        "pivot",
        lambda df: df.pivot(on="s", index="k", values="v"),
        {"k": "Int64", "x": "Float64", "y": "Float64"},
    ),
    Replay(
        "pivot_on_columns",
        "pivot",
        lambda df: df.pivot(on="s", on_columns=["x", "y", "z"], index="k", values="v"),
        {"k": "Int64", "x": "Float64", "y": "Float64", "z": "Float64"},
    ),
    Replay(
        "transpose",
        "transpose",
        lambda df: df.select(["k", "xn"]).transpose(),
        {"column_0": "Int64", "column_1": "Int64"},
    ),
    Replay(
        "unstack",
        "unstack",
        lambda df: df.select(["k", "v"]).unstack(step=1),
        {"k_0": "Int64", "k_1": "Int64", "v_0": "Float64", "v_1": "Float64"},
    ),
    Replay(
        "partition_by",
        "partition_by",
        lambda df: df.partition_by("k")[0],
        FULL,
    ),
)

JOINS: tuple[Replay, ...] = (
    Replay(
        "join",
        "join",
        lambda df: df.join(other_frame(), on="k"),
        full(w="Float64", xs_right="String"),
    ),
    Replay(
        "join_asof",
        "join_asof",
        lambda df: df.join_asof(other_frame(), on="k"),
        full(w="Float64", xs_right="String"),
    ),
    Replay(
        "join_where",
        "join_where",
        lambda df: df.join_where(
            other_frame().rename({"k": "k2", "xs": "xs2"}),
            pl.col("k") < pl.col("k2"),
        ),
        full(k2="Int64", w="Float64", xs2="String"),
    ),
    Replay(
        "update",
        "update",
        lambda df: df.update(other_frame().drop("xs"), on="k"),
        FULL,
    ),
    Replay(
        "hstack",
        "hstack",
        lambda df: df.hstack(other_frame().rename({"k": "k2", "xs": "xs2"})),
        full(k2="Int64", w="Float64", xs2="String"),
    ),
)

REDUCERS: tuple[Replay, ...] = (
    Replay("max", "max", lambda df: df.max(), FULL),
    Replay("min", "min", lambda df: df.min(), FULL),
    Replay("sum", "sum", lambda df: df.sum(), FULL),
    Replay(
        "mean",
        "mean",
        lambda df: df.mean(),
        full(k="Float64", xn="Float64"),
    ),
    Replay(
        "median",
        "median",
        lambda df: df.median(),
        full(k="Float64", xn="Float64"),
    ),
    Replay(
        "quantile",
        "quantile",
        lambda df: df.quantile(0.5),
        full(k="Float64", xn="Float64"),
    ),
    Replay("std", "std", lambda df: df.std(), full(k="Float64", xn="Float64")),
    Replay("var", "var", lambda df: df.var(), full(k="Float64", xn="Float64")),
    Replay(
        "product",
        "product",
        lambda df: df.product(),
        full(t="Null", s="Null", l="Null", st="Null", xn="Int64", xs="Null"),
    ),
    Replay("count", "count", lambda df: df.count(), everywhere("UInt32")),
    Replay("null_count", "null_count", lambda df: df.null_count(), everywhere("UInt32")),
    Replay(
        "describe",
        "describe",
        lambda df: df.describe(),
        {
            "statistic": "String",
            "k": "Float64",
            "t": "String",
            "v": "Float64",
            "s": "String",
            "l": "Float64",
            "st": "Float64",
            "xn": "Float64",
            "xs": "String",
            "x_p": "Float64",
        },
    ),
)

GROUPING: tuple[Replay, ...] = (
    Replay(
        "group_by",
        "group_by",
        lambda df: df.group_by("k").agg(pl.col("v").sum()),
        {"k": "Int64", "v": "Float64"},
    ),
    Replay(
        "group_by_dynamic",
        "group_by_dynamic",
        lambda df: df.group_by_dynamic("t", every="1d").agg(pl.col("v").sum()),
        {"t": DT, "v": "Float64"},
    ),
    Replay(
        "rolling",
        "rolling",
        lambda df: df.rolling("t", period="1d").agg(pl.col("v").sum()),
        {"t": DT, "v": "Float64"},
    ),
    Replay(
        "group_by_all",
        "group_by",
        lambda df: df.select("k", "v", "s").group_by("k").all(),
        {"k": "Int64", "v": "List(Float64)", "s": "List(String)"},
    ),
    Replay(
        "group_by_len",
        "group_by",
        lambda df: df.group_by("k").len(),
        {"k": "Int64", "len": "UInt32"},
    ),
    Replay(
        "group_by_n_unique",
        "group_by",
        lambda df: df.select("k", "v").group_by("k").n_unique(),
        {"k": "Int64", "v": "UInt32"},
    ),
)

ENTRIES: tuple[Replay, ...] = ROW_ONLY + COLUMN_SHAPING + RESHAPING + JOINS + REDUCERS + GROUPING

SCALAR_OR_SIDE: frozenset[str] = frozenset({"n_unique"})

# `corr` needs numpy, which this project does not install, so
# `TestCorrReplays` replays it where numpy is present.
NEEDS_NUMPY: frozenset[str] = frozenset({"corr"})


class TestEveryRowReplays:
    """Each replayed row's full output schema equals the expectation."""

    def test_every_entry_matches(self) -> None:
        """Run all replays; a schema drift names its row.

        A call that the installed release of polars refuses over the replay
        frame has no replay there: `gather` before polars 1.41.0, which adds
        it, and `merge_sorted` before that release, which panics over the
        list column `l`.
        """
        for entry in ENTRIES:
            if entry.runs():
                assert entry.observed() == dict(entry.expects), entry.label


class TestCorrReplays:
    """`.corr()` keeps every name, and every column becomes `Float64`."""

    def test_every_column_becomes_a_float(self) -> None:
        """One row per column, every column `Float64`, a label column first."""
        pytest.importorskip("numpy")
        frame = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 0.5, 2.0], "c": [True, False, True]})
        assert dict(frame.corr().schema) == dict.fromkeys(("a", "b", "c"), pl.Float64())
        labelled = frame.corr(label="name").schema
        assert next(iter(labelled.items())) == ("name", pl.String())


class TestTheReplayCoversPartOne:
    """Every operation Part I rows describe has a replay entry."""

    def part_one_methods(self) -> set[str]:
        """Give the frame methods Part I's text names as calls.

        Returns:
            Each method named as a call in Part I.

        Example:
            >>> "join" in TestTheReplayCoversPartOne().part_one_methods()
            True
        """
        text = Document("OPERATION_CATALOG.md").text
        part_one = text[: text.index("# Part II.")]
        found = set(re.findall(r"`\.?([a-z_]+)\(", part_one))
        return {name for name in found if hasattr(pl.DataFrame, name)}

    def test_every_part_one_method_is_replayed_or_excluded(self) -> None:
        """The replay's coverage is computed, never remembered.

        `n_unique` returns a number on a frame, so its frame form has
        no schema to replay; its group form is replayed above.
        """
        replayed = {entry.method for entry in ENTRIES}
        missing = self.part_one_methods() - replayed - SCALAR_OR_SIDE - NEEDS_NUMPY
        assert not missing, sorted(missing)
