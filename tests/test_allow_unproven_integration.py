"""Check the `allow_unproven` setting from start to end, with a real mypy run.

When the checker cannot prove a result, it names a reason and either
reports an error or lets the result pass. The `allow_unproven` setting
under `[tool.astrolabe]` in `pyproject.toml` decides which reasons
pass. These tests write a real `pyproject.toml` and a program into a
temporary directory, and call `mypy.api.run` there. They read the exit
code and the count of the errors, which a `.yml` case does not state.

Each program holds one kind of unproven result. A test runs it at the
level that reports it, at the level that lets it pass, or with a
setting for one operation. It then checks the error code, the number
of errors, and the note that names the setting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mypy.api

if TYPE_CHECKING:
    from pathlib import Path

# A function of polars whose tree polars cannot write, so the rules cannot read
# it. This is a gap in the checker, not in the code of the user.
_CHECKER_GAP = """\
import polars as pl
from astrolabe import Columns, TypedDataFrame
class Vw(Columns): v: pl.Int64; w: pl.Int64

def f(
    df: TypedDataFrame[Vw],
) -> None:
    df.select(pl.cum_sum_horizontal("v", "w").alias("bad"))
"""

# The same gap in two operations: select on line 12, and agg on line 13.
_SELECT_AND_AGG_GAPS = """\
import polars as pl
from astrolabe import Columns, TypedDataFrame

class Idvw(Columns):
    id: pl.Int64
    v: pl.Int64
    w: pl.Int64

def f(
    df: TypedDataFrame[Idvw],
) -> None:
    df.select(pl.cum_sum_horizontal("v", "w").alias("bad"))
    df.group_by("id").agg(pl.cum_sum_horizontal("v", "w").alias("bad"))
"""

# A data type rule that the checker does not model, in two operations: pivot on
# line 10, and unpivot on line 11.
_PIVOT_AND_UNPIVOT_GAPS = """\
import polars as pl
from astrolabe import Columns, TypedDataFrame

class Sales[D: pl.DataType](Columns):
    id: pl.Int64
    cat: pl.String
    v: D

def f[D: pl.DataType](df: TypedDataFrame[Sales[D]]) -> None:
    df.pivot(on="cat", on_columns=["a"], index="id", values="v", aggregate_function="sum")
    df.unpivot(on=["cat", "v"], index="id")
"""

# An integer variable, whose data type polars decides from its value, in one
# operation.
_BARE_INT = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
A = columns("A", {"a": pl.Int64})

Frame = TypedDataFrame[A]

df: Frame
value: int = 1
df.select(value)
"""

# The same integer variable in four operations. Each operation has its own path
# to the fallback.
_BARE_INT_EVERY_OPERATION = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
Ka = columns("Ka", {"k": pl.String, "a": pl.Int64})

Frame = TypedDataFrame[Ka]

df: Frame
plain_df: pl.DataFrame
value: int = 1
df.select(value)
df.with_columns(b=value)
df.group_by("k").agg(total=value)
plain_df.select(value)
"""

# map_elements with no return_dtype= on line 8 (the level omitted), and with it
# on line 9 (proven). map_columns on line 10 (the level stated, because polars
# has no keyword that names the data type).
_OPAQUE_MEMBERS = """\
import polars as pl
from astrolabe import Columns, TypedDataFrame

class Quotes(Columns):
    bid: pl.Float64

quotes: TypedDataFrame[Quotes]
quotes.select(pl.col("bid").map_elements(str))
quotes.select(pl.col("bid").map_elements(str, return_dtype=pl.String))
quotes.map_columns("bid", lambda s: s)
"""

# Two calls whose result schema the checker cannot know: select on line 11, and
# join on line 12.
_UNRESOLVED_CALLS = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
Ids = columns("Ids", {"id": pl.Int64})

Other = TypedDataFrame[Ids]

df: TypedDataFrame[Ids]
other: Other
kwargs: dict[str, pl.Expr] = {}
key: str = "id"
df.select(**kwargs)
df.join(other, on=key)
"""

_MISSING_COLUMN = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
Ids = columns("Ids", {"id": pl.Int64})

df: TypedDataFrame[Ids]
df.select("missing")
"""

_SCHEMALESS_READ_CSV = """\
import polars as pl

reveal_type(pl.read_csv("data.csv"))
"""

# A column name in a variable on lines 9, 10 (through **kwargs) and 12 (agg). A
# plain expression, which no hook typed, on line 11, and one that keeps its
# alias as the name on line 13.
_DYNAMIC_AND_PLAIN = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
Ak = columns("Ak", {"a": pl.Int64, "k": pl.String})

df: TypedDataFrame[Ak]
name: str = "a"
kwargs: dict[str, pl.Expr] = {}
plain: pl.Expr = pl.col("a")
df.select(name)
df.select(**kwargs)
df.with_columns(plain)
df.group_by("k").agg(**kwargs)
df.select(plain.alias("b"))
"""

# Two outputs with one name, in select on line 6, with_columns on line 7, and
# agg on line 8, where the group key and the aggregation have the same name.
_DUPLICATE_OUTPUTS = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
A = columns("A", {"a": pl.Int64})

df: TypedDataFrame[A]
df.select(pl.col("a"), pl.col("a"))
df.with_columns(pl.col("a"), pl.col("a"))
df.group_by("a").agg(pl.col("a"))
"""

# A column name in a variable, in six frame operations, one on each line from
# 8 to 13: drop, rename, with_row_index, join, pivot and unpivot.
_FRAME_DYNAMIC_NAMES = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
Ak = columns("Ak", {"a": pl.Int64, "k": pl.String})

df: TypedDataFrame[Ak]
other: TypedDataFrame[Ak]
name: str = "a"
df.drop(name)
df.rename({name: "b"})
df.with_row_index(name)
df.join(other, on=name)
df.pivot(on=name, on_columns=["x"], index="k", values="a")
df.unpivot(on=name, index="k")
"""

# Calls that polars refuses, on lines 7 to 10, and a join key that it cannot
# find, on line 11. mypy reports the call with no mapping on line 7 from the
# parameters of polars, and the checker reports the others.
_REPORTED_ONCE = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
A = columns("A", {"a": pl.Int64})

df: TypedDataFrame[A]
other: TypedDataFrame[A]
df.rename()
df.join(other, on="a", left_on="a", right_on="a")
df.explode([])
pl.concat([])
df.join(other, on="zz")
"""

# A join key whose data type on one side is a free type variable, on line 9.
_UNRESOLVED_KEY_DTYPE = """\
import polars as pl
from astrolabe import Columns, TypedDataFrame
class A[T: pl.DataType](Columns): a: T

def f[T: pl.DataType](
    df: TypedDataFrame[A[T]],
    other: TypedDataFrame[A[pl.Int64]],
) -> None:
    df.join(other, on="a")
"""

# Two group-by calls that report a hard error: an unknown key on line 8, and a
# counting shorthand over a key of the same name on line 9. Each must give one
# error, and no second error from the gate.
_GROUP_BY_ERRORS = """\
import polars as pl
from astrolabe import Columns, TypedDataFrame

class K(Columns):
    k: pl.Int64

def f(df: TypedDataFrame[K]) -> None:
    df.group_by("zz").agg(pl.col("k").sum())
    df.group_by("k").len().group_by("len").len()
"""

_UNCHECKED = "[implicitly-unchecked]"
_UNTYPED = "[implicitly-untyped]"


def _run_mypy(tmp_path: Path, astrolabe_config: str, scenario: str) -> tuple[str, int]:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.mypy]\nplugins = ["astrolabe.plugin"]\nallow_redefinition = true\n\n'
        f"{astrolabe_config}"
    )
    main_file = tmp_path / "main.py"
    main_file.write_text(scenario)
    stdout, _stderr, exit_code = mypy.api.run([
        "--config-file",
        str(pyproject),
        "--cache-dir",
        str(tmp_path / ".mypy_cache"),
        str(main_file),
    ])
    return stdout, exit_code


def _at_level(level: str, *extra: str) -> str:
    return "\n".join(["[tool.astrolabe]", f'allow_unproven = "{level}"', *extra, ""])


def _lines_at(stdout: str, marker: str) -> list[str]:
    return [line for line in stdout.splitlines() if marker in line]


def _reported_at(stdout: str, marker: str, code: str) -> bool:
    return any(code in line for line in _lines_at(stdout, marker))


def _revealed_at(stdout: str, marker: str, *parts: str) -> bool:
    return any(_holds_all(line, parts) for line in _lines_at(stdout, marker))


def _holds_all(line: str, parts: tuple[str, ...]) -> bool:
    return all(part in line for part in parts)


def test_the_default_reports_no_gap_and_no_omission(tmp_path: Path) -> None:
    for scenario in (_CHECKER_GAP, _BARE_INT, _OPAQUE_MEMBERS, _UNRESOLVED_CALLS):
        stdout, exit_code = _run_mypy(tmp_path, "", scenario)
        assert _UNCHECKED not in stdout
        assert _UNTYPED not in stdout
        assert exit_code == 0


def test_never_reports_a_checker_gap_with_its_hint(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("never"), _CHECKER_GAP)
    assert _UNCHECKED in stdout
    assert 'note: set allow_unproven = "none" in [tool.astrolabe]' in stdout
    assert exit_code == 1


def test_none_does_not_report_a_checker_gap(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("none"), _CHECKER_GAP)
    assert _UNCHECKED not in stdout
    assert exit_code == 0


def test_report_checker_gaps_reports_a_gap_at_the_default_level(tmp_path: Path) -> None:
    config = "[tool.astrolabe]\nreport_checker_gaps = true\n"
    stdout, exit_code = _run_mypy(tmp_path, config, _CHECKER_GAP)
    assert _UNCHECKED in stdout
    assert "note: set report_checker_gaps = false in [tool.astrolabe]" in stdout
    assert exit_code == 1

    stdout, exit_code = _run_mypy(tmp_path, config, _BARE_INT)
    assert _UNCHECKED not in stdout
    assert exit_code == 0


def test_never_reports_the_same_gap_in_select_and_agg(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("never"), _SELECT_AND_AGG_GAPS)
    assert stdout.count(_UNCHECKED) == 2
    assert exit_code == 1


def test_an_override_silences_one_operation_under_never(tmp_path: Path) -> None:
    config = _at_level("never", "", "[tool.astrolabe.allow_unproven_overrides]", 'select = "none"')
    stdout, exit_code = _run_mypy(tmp_path, config, _SELECT_AND_AGG_GAPS)
    assert stdout.count(_UNCHECKED) == 1
    assert not _reported_at(stdout, "main.py:12", _UNCHECKED)
    assert _reported_at(stdout, "main.py:13", _UNCHECKED)
    assert exit_code == 1


def test_an_override_to_never_reports_gaps_in_that_operation_only(tmp_path: Path) -> None:
    config = '[tool.astrolabe.allow_unproven_overrides]\nagg = "never"\n'
    stdout, exit_code = _run_mypy(tmp_path, config, _SELECT_AND_AGG_GAPS)
    assert stdout.count(_UNCHECKED) == 1
    assert not _reported_at(stdout, "main.py:12", _UNCHECKED)
    assert _reported_at(stdout, "main.py:13", _UNCHECKED)
    assert exit_code == 1


def test_never_reports_the_pivot_and_unpivot_gaps(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("never"), _PIVOT_AND_UNPIVOT_GAPS)
    assert stdout.count(_UNCHECKED) == 2
    assert exit_code == 1


def test_an_override_silences_pivot_while_unpivot_still_reports(tmp_path: Path) -> None:
    config = _at_level("never", "", "[tool.astrolabe.allow_unproven_overrides]", 'pivot = "none"')
    stdout, exit_code = _run_mypy(tmp_path, config, _PIVOT_AND_UNPIVOT_GAPS)
    assert stdout.count(_UNCHECKED) == 1
    assert not _reported_at(stdout, "main.py:10", _UNCHECKED)
    assert _reported_at(stdout, "main.py:11", _UNCHECKED)
    assert exit_code == 1


def test_stated_reports_a_bare_int_and_names_the_omitted_level(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("stated"), _BARE_INT)
    assert _UNCHECKED in stdout
    assert "a number variable has no fixed width" in stdout
    assert 'note: set allow_unproven = "omitted" in [tool.astrolabe]' in stdout
    assert exit_code == 1


def test_omitted_lets_a_bare_int_pass(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("omitted"), _BARE_INT)
    assert _UNCHECKED not in stdout
    assert exit_code == 0


def test_stated_reports_a_bare_int_in_every_operation(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("stated"), _BARE_INT_EVERY_OPERATION)
    assert stdout.count(_UNCHECKED) == 4
    assert exit_code == 1


def test_stated_reports_map_elements_but_not_map_columns(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("stated"), _OPAQUE_MEMBERS)
    assert _reported_at(stdout, "main.py:8", _UNCHECKED)
    assert "no return_dtype= names it" in stdout
    assert "main.py:9" not in stdout
    assert not _reported_at(stdout, "main.py:10", _UNCHECKED)
    assert exit_code == 1


def test_none_reports_both_opaque_members_once_each(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("none"), _OPAQUE_MEMBERS)
    assert stdout.count(_UNCHECKED) == 2
    assert "main.py:9" not in stdout
    assert exit_code == 1


def test_an_override_raises_one_opaque_member_over_the_global_level(tmp_path: Path) -> None:
    config = _at_level(
        "none",
        "",
        "[tool.astrolabe.allow_unproven_overrides]",
        'map_elements = "omitted"',
    )
    stdout, exit_code = _run_mypy(tmp_path, config, _OPAQUE_MEMBERS)
    assert stdout.count(_UNCHECKED) == 1
    assert not _reported_at(stdout, "main.py:8", _UNCHECKED)
    assert _reported_at(stdout, "main.py:10", _UNCHECKED)
    assert exit_code == 1


def test_never_reports_both_unresolved_calls(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("never"), _UNRESOLVED_CALLS)
    assert stdout.count(_UNTYPED) == 2
    assert exit_code == 1


def test_never_reports_a_missing_column_once_and_not_as_untyped(tmp_path: Path) -> None:
    # The checker reports the missing column and keeps it under Unchecked. The
    # call therefore does not fall back to the plain frame, so no untyped report.
    stdout, exit_code = _run_mypy(tmp_path, _at_level("never"), _MISSING_COLUMN)
    assert _reported_at(stdout, "main.py:6", "[unknown-column]")
    assert not _reported_at(stdout, "main.py:6", _UNTYPED)
    assert exit_code == 1


def test_the_default_lets_a_schemaless_read_csv_fall_back_to_plain(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, "", _SCHEMALESS_READ_CSV)
    assert _UNTYPED not in stdout
    assert 'Revealed type is "polars.dataframe.frame.DataFrame"' in stdout
    assert exit_code == 0


def test_none_reports_a_schemaless_read_csv_and_stated_lets_it_pass(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("none"), _SCHEMALESS_READ_CSV)
    assert _UNTYPED in stdout
    assert "no schema gives the column types" in stdout
    assert exit_code == 1

    stdout, exit_code = _run_mypy(tmp_path, _at_level("stated"), _SCHEMALESS_READ_CSV)
    assert _UNTYPED not in stdout
    assert exit_code == 0


def test_an_override_lowers_read_csv_while_the_default_stays(tmp_path: Path) -> None:
    config = '[tool.astrolabe.allow_unproven_overrides]\nread_csv = "none"\n'
    stdout, exit_code = _run_mypy(tmp_path, config, _SCHEMALESS_READ_CSV)
    assert _UNTYPED in stdout
    assert (
        'note: set [tool.astrolabe.allow_unproven_overrides] read_csv = "stated" to allow this'
        in stdout
    )
    assert exit_code == 1

    stdout, exit_code = _run_mypy(tmp_path, config, _UNRESOLVED_CALLS)
    assert _UNTYPED not in stdout
    assert exit_code == 0


def test_none_reports_a_dynamic_name_and_a_plain_expression_with_their_reasons(
    tmp_path: Path,
) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("none"), _DYNAMIC_AND_PLAIN)
    for marker in ("main.py:9", "main.py:10", "main.py:11", "main.py:12"):
        assert _reported_at(stdout, marker, _UNTYPED)
    assert _reported_at(stdout, "main.py:13", _UNCHECKED)
    assert "a column name is not a literal string" in stdout
    assert "an expression is not typed" in stdout
    assert 'note: set allow_unproven = "stated" in [tool.astrolabe]' in stdout
    assert exit_code == 1


def test_stated_lets_a_dynamic_name_and_a_plain_expression_pass(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("stated"), _DYNAMIC_AND_PLAIN)
    assert _UNTYPED not in stdout
    assert _UNCHECKED not in stdout
    assert exit_code == 0


def test_never_reports_a_duplicate_output_once(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("never"), _DUPLICATE_OUTPUTS)
    assert _reported_at(stdout, "main.py:6", "[duplicate-column]")
    assert _reported_at(stdout, "main.py:7", "[duplicate-column]")
    assert _reported_at(stdout, "main.py:8", "[duplicate-column]")
    assert _UNTYPED not in stdout
    assert exit_code == 1


def test_none_reports_a_dynamic_name_in_every_frame_operation(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("none"), _FRAME_DYNAMIC_NAMES)
    for line in range(8, 14):
        assert _reported_at(stdout, f"main.py:{line}:", _UNTYPED), line
    assert "a column name is not a literal string" in stdout
    assert exit_code == 1


def test_stated_lets_a_dynamic_name_pass_in_every_frame_operation(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("stated"), _FRAME_DYNAMIC_NAMES)
    assert _UNTYPED not in stdout
    assert exit_code == 0


def test_never_reports_a_refused_call_shape_once(tmp_path: Path) -> None:
    stdout, _exit_code = _run_mypy(tmp_path, _at_level("never"), _REPORTED_ONCE)
    assert _reported_at(stdout, "main.py:7:", "[call-arg]")
    for line in (8, 9, 10):
        assert _reported_at(stdout, f"main.py:{line}:", "[invalid-call-shape]"), line
    assert _reported_at(stdout, "main.py:11:", "[unknown-column]")
    assert _UNTYPED not in stdout


def test_never_reports_an_unresolved_join_key_dtype_as_a_gap(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("never"), _UNRESOLVED_KEY_DTYPE)
    assert _reported_at(stdout, "main.py:9:", _UNTYPED)
    assert "the checker does not model this case" in stdout
    assert exit_code == 1


# An open frame, whose type has a remainder of columns it does not name. A name
# it does not declare, in select on line 9 and drop on line 10. A join on line
# 11, whose renamed columns depend on the columns of the caller.
_REMAINDER_MAY_HOLD = """\
import polars as pl
from astrolabe import TypedDataFrame, columns, other_columns
Symbols = columns("Symbols", {"symbol": pl.String})
SectorsSchema = columns("SectorsSchema", {"symbol": pl.String, "sector": pl.String})
Rest = other_columns()
Sectors = TypedDataFrame[SectorsSchema]

def f(df: TypedDataFrame[Symbols, Rest], ref: Sectors) -> None:
    reveal_type(df.select(pl.col("zz")))
    reveal_type(df.drop("zz"))
    reveal_type(df.join(ref, on="symbol"))
"""

# Two claims that the checker cannot prove. On line 19, the remainder of the
# argument lacks the requirement of the parameter. On line 20, a plain
# expression is passed for a declared one.
_CLAIMS = """\
import polars as pl
from astrolabe import Columns, TypedDataFrame, column, other_columns

class Trades(Columns):
    symbol: pl.String

NoSector = other_columns(without=["sector"])
Weaker = other_columns()
Price = column("price", pl.Float64)

def add_sector(df: TypedDataFrame[Trades, NoSector]) -> TypedDataFrame[Trades, NoSector]:
    raise NotImplementedError

def discounted(price: Price) -> Price:
    return price * 0.9

weaker: TypedDataFrame[Trades, Weaker]
plain: pl.Expr
add_sector(weaker)
discounted(plain)
"""


def test_the_default_reports_what_the_remainder_may_hold(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, "", _REMAINDER_MAY_HOLD)
    assert _reported_at(stdout, "main.py:9:", "[undeclared-column]")
    assert _reported_at(stdout, "main.py:10:", "[undeclared-column]")
    assert _reported_at(stdout, "main.py:11:", "[join-rename]")
    assert 'set allow_unproven = "possible"' in stdout
    assert exit_code == 1


def test_possible_keeps_what_the_remainder_may_hold(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("possible"), _REMAINDER_MAY_HOLD)
    assert "error:" not in stdout
    assert _revealed_at(stdout, "main.py:9:", "'zz'", "Unchecked")
    assert _revealed_at(stdout, "main.py:10:", "Minus")
    assert _revealed_at(stdout, "main.py:11:", "polars.dataframe.frame.DataFrame")
    assert exit_code == 0


def test_an_override_lets_one_undeclared_drop_pass(tmp_path: Path) -> None:
    config = '[tool.astrolabe]\n\n[tool.astrolabe.allow_unproven_overrides]\ndrop = "possible"\n'
    stdout, _exit_code = _run_mypy(tmp_path, config, _REMAINDER_MAY_HOLD)
    assert _reported_at(stdout, "main.py:9:", "[undeclared-column]")
    assert not _reported_at(stdout, "main.py:10:", "[undeclared-column]")
    assert _reported_at(stdout, "main.py:11:", "[join-rename]")


def test_possible_reports_both_claims_and_names_the_asserted_level(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("possible"), _CLAIMS)
    assert _reported_at(stdout, "main.py:19:", "[forbidden-column]")
    assert _reported_at(stdout, "main.py:20:", "[expression-fit]")
    assert 'fit = "asserted"' in stdout
    assert exit_code == 1


def test_asserted_lets_both_claims_pass(tmp_path: Path) -> None:
    stdout, exit_code = _run_mypy(tmp_path, _at_level("asserted"), _CLAIMS)
    assert "error:" not in stdout
    assert exit_code == 0


def test_a_group_by_error_is_reported_once_under_never(tmp_path: Path) -> None:
    config = _at_level("never", "report_checker_gaps = true")
    stdout, exit_code = _run_mypy(tmp_path, config, _GROUP_BY_ERRORS)
    assert len(_lines_at(stdout, "main.py:8:")) == 1
    assert _reported_at(stdout, "main.py:8:", "[unknown-column]")
    assert len(_lines_at(stdout, "main.py:9:")) == 1
    assert _reported_at(stdout, "main.py:9:", "[duplicate-column]")
    assert exit_code == 1
