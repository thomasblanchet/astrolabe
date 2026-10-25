"""Check the `disallow_vertical_concat` setting from start to end, with a real mypy run.

A vertical concat stacks frames that must hold the same columns. With
`disallow_vertical_concat = true` under `[tool.astrolabe]`, the checker
reports each `pl.concat(...)` with `how="vertical"` or
`how="vertical_relaxed"` as a `[vertical-concat-disallowed]` error.

These tests write a real `pyproject.toml` and a program into a
temporary directory, and call `mypy.api.run` there. They read the exit
code and the count of the errors, which a `.yml` case does not state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mypy.api

if TYPE_CHECKING:
    from pathlib import Path

_IDENTICAL_SCHEMAS_CONCAT = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
LeftSchema = columns("LeftSchema", {"a": pl.Int64})
RightSchema = columns("RightSchema", {"a": pl.Int64})
Left = TypedDataFrame[LeftSchema]
Right = TypedDataFrame[RightSchema]

left: Left
right: Right
reveal_type(pl.concat([left, right]))
"""

_MISMATCHED_SCHEMAS_CONCAT = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
LeftSchema = columns("LeftSchema", {"a": pl.Int64})
RightSchema = columns("RightSchema", {"a": pl.Float64})
Left = TypedDataFrame[LeftSchema]
Right = TypedDataFrame[RightSchema]

left: Left
right: Right
pl.concat([left, right])
"""

_IDENTICAL_SCHEMAS_CONCAT_RELAXED = """\
import polars as pl
from astrolabe import TypedDataFrame, columns
LeftSchema = columns("LeftSchema", {"a": pl.Int64})
RightSchema = columns("RightSchema", {"a": pl.Int64})
Left = TypedDataFrame[LeftSchema]
Right = TypedDataFrame[RightSchema]

left: Left
right: Right
reveal_type(pl.concat([left, right], how="vertical_relaxed"))
"""

_DISALLOWED_CODE = "[vertical-concat-disallowed]"
_SCHEMA_MISMATCH_CODE = "[schema-mismatch]"
_IMPLICITLY_UNTYPED_CODE = "[implicitly-untyped]"


def _run_mypy(tmp_path: Path, astrolabe_config: str, scenario: str) -> tuple[str, str, int]:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[tool.mypy]\nplugins = ["astrolabe.plugin"]\n\n{astrolabe_config}')
    main_file = tmp_path / "main.py"
    main_file.write_text(scenario)
    stdout, stderr, exit_code = mypy.api.run([
        "--config-file",
        str(pyproject),
        "--cache-dir",
        str(tmp_path / ".mypy_cache"),
        str(main_file),
    ])
    return stdout, stderr, exit_code


def test_default_does_not_disallow_vertical_concat(tmp_path: Path) -> None:
    stdout, _stderr, exit_code = _run_mypy(tmp_path, "", _IDENTICAL_SCHEMAS_CONCAT)
    assert _DISALLOWED_CODE not in stdout
    assert 'Revealed type is "polars.dataframe.frame.DataFrame"' in stdout
    assert exit_code == 0


def test_explicit_false_does_not_disallow_vertical_concat(tmp_path: Path) -> None:
    stdout, _stderr, exit_code = _run_mypy(
        tmp_path,
        "[tool.astrolabe]\ndisallow_vertical_concat = false\n",
        _IDENTICAL_SCHEMAS_CONCAT,
    )
    assert _DISALLOWED_CODE not in stdout
    assert exit_code == 0


def test_default_still_reports_a_genuine_schema_mismatch(tmp_path: Path) -> None:
    stdout, _stderr, exit_code = _run_mypy(tmp_path, "", _MISMATCHED_SCHEMAS_CONCAT)
    assert _SCHEMA_MISMATCH_CODE in stdout
    assert _DISALLOWED_CODE not in stdout
    assert exit_code == 1


def test_disallowed_reports_even_identical_schemas(tmp_path: Path) -> None:
    stdout, _stderr, exit_code = _run_mypy(
        tmp_path,
        "[tool.astrolabe]\ndisallow_vertical_concat = true\n",
        _IDENTICAL_SCHEMAS_CONCAT,
    )
    assert _DISALLOWED_CODE in stdout
    assert exit_code == 1


def test_disallowed_also_disallows_vertical_relaxed(tmp_path: Path) -> None:
    stdout, _stderr, exit_code = _run_mypy(
        tmp_path,
        "[tool.astrolabe]\ndisallow_vertical_concat = true\n",
        _IDENTICAL_SCHEMAS_CONCAT_RELAXED,
    )
    assert _DISALLOWED_CODE in stdout
    assert exit_code == 1


def test_disallowed_skips_the_schema_mismatch_check(tmp_path: Path) -> None:
    stdout, _stderr, exit_code = _run_mypy(
        tmp_path,
        "[tool.astrolabe]\ndisallow_vertical_concat = true\n",
        _MISMATCHED_SCHEMAS_CONCAT,
    )
    assert _DISALLOWED_CODE in stdout
    assert _SCHEMA_MISMATCH_CODE not in stdout
    assert exit_code == 1


def test_disallowed_reports_only_the_disallowed_concat_under_never(tmp_path: Path) -> None:
    stdout, _stderr, exit_code = _run_mypy(
        tmp_path,
        '[tool.astrolabe]\ndisallow_vertical_concat = true\nallow_unproven = "never"\n',
        _IDENTICAL_SCHEMAS_CONCAT,
    )
    assert _DISALLOWED_CODE in stdout
    assert _IMPLICITLY_UNTYPED_CODE not in stdout
    assert exit_code == 1
