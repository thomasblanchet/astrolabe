"""Check, with a real mypy run, how settings in `pyproject.toml` change the checker.

The programs here use an open frame, a frame whose type has a
remainder: columns that the type does not name. A name that the type
does not declare can then be one of those columns, or not. The tests
check that such a name is an error with the default settings, and
that an unknown key in `[tool.astrolabe]` changes nothing. They also
check the settings `elevate_untyped_select` and
`disallow_implicit_inferred_fallback`.

These tests write a real `pyproject.toml` and a program into a
temporary directory, and call `mypy.api.run` there. They read the exit
code and the count of the errors, which a `.yml` case does not state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mypy.api

if TYPE_CHECKING:
    from pathlib import Path

_SCENARIO = """\
import polars as pl
from astrolabe import TypedDataFrame, columns, other_columns
Foo = columns("Foo", {"foo": pl.Int64})

Rest = other_columns()


def func(df: TypedDataFrame[Foo, Rest]) -> None:
    df.select("baz")
"""

_DROP_SCENARIO = """\
import polars as pl
from astrolabe import TypedDataFrame, columns, other_columns
Foo = columns("Foo", {"foo": pl.Int64})

Rest = other_columns()


def func(df: TypedDataFrame[Foo, Rest]) -> None:
    df.drop("baz")
"""

_RENAME_SCENARIO = """\
import polars as pl
from astrolabe import TypedDataFrame, columns, other_columns
Foo = columns("Foo", {"foo": pl.Int64})

Rest = other_columns()


def func(df: TypedDataFrame[Foo, Rest]) -> None:
    df.rename({"baz": "qux"})
"""

_RENAME_DESTINATION_SCENARIO = """\
import polars as pl
from astrolabe import TypedDataFrame, columns, other_columns
Foo = columns("Foo", {"foo": pl.Int64})

Rest = other_columns()


def func(df: TypedDataFrame[Foo, Rest]) -> None:
    df.rename({"foo": "bar"})
"""

_TYPED_EXPR_SCENARIO = """\
import polars as pl
from astrolabe import TypedDataFrame, columns, other_columns
Foo = columns("Foo", {"foo": pl.Int64})

Rest = other_columns()


def func(df: TypedDataFrame[Foo, Rest]) -> None:
    df.select(pl.col("baz").alias("qux"))
"""

_UNTYPED_SELECT_SCENARIO = """\
import polars as pl

df: pl.DataFrame
reveal_type(df.select(pl.col("a")))
"""

_INFERRED_UNTYPED_FALLBACK_SCENARIO = """\
import polars as pl
from astrolabe import InferredExpr, infer

def untyped_helper() -> pl.Expr:
    return pl.col("x")

def maybe_untyped(flag: bool) -> InferredExpr:
    if flag:
        return infer(pl.col("a"))
    return infer(untyped_helper())

reveal_type(maybe_untyped(True))
"""

_AMBIGUOUS_COLUMN_MESSAGE = "is not declared and cannot be proven to exist"
_RENAME_COLLISION_MESSAGE = "Rename would produce duplicate column"
_UNKNOWN_STRICT_SETTING = "[tool.astrolabe]\nstrict_undeclared_columns = false\n"
_TYPED_DATAFRAME_REVEAL = 'Revealed type is "astrolabe.frame.TypedDataFrame['
_PLAIN_DATAFRAME_REVEAL = 'Revealed type is "polars.dataframe.frame.DataFrame"'
_INFERRED_FALLBACK_MESSAGE = "falls back to the plain, erased type instead of InferredExpr"


def _run_mypy(
    tmp_path: Path,
    astrolabe_config: str,
    scenario: str = _SCENARIO,
) -> tuple[str, str, int]:
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


def test_an_undeclared_select_is_an_error_at_the_default_level(tmp_path: Path) -> None:
    """With no `[tool.astrolabe]` section, the plugin uses its default settings."""
    stdout, _stderr, exit_code = _run_mypy(tmp_path, "")
    assert _AMBIGUOUS_COLUMN_MESSAGE in stdout
    assert exit_code == 1


def test_an_unknown_strict_setting_does_not_suppress_the_error(tmp_path: Path) -> None:
    """`strict_undeclared_columns` is not a setting, so the plugin ignores the key."""
    stdout, _stderr, exit_code = _run_mypy(tmp_path, _UNKNOWN_STRICT_SETTING)
    assert _AMBIGUOUS_COLUMN_MESSAGE in stdout
    assert exit_code == 1


def test_an_undeclared_drop_is_an_error_at_the_default_level(tmp_path: Path) -> None:
    stdout, _stderr, exit_code = _run_mypy(tmp_path, _UNKNOWN_STRICT_SETTING, _DROP_SCENARIO)
    assert _AMBIGUOUS_COLUMN_MESSAGE in stdout
    assert exit_code == 1


def test_an_undeclared_rename_source_is_an_error_at_the_default_level(tmp_path: Path) -> None:
    stdout, _stderr, exit_code = _run_mypy(tmp_path, _UNKNOWN_STRICT_SETTING, _RENAME_SCENARIO)
    assert _AMBIGUOUS_COLUMN_MESSAGE in stdout
    assert exit_code == 1


def test_an_ambiguous_rename_destination_is_always_a_duplicate_error(tmp_path: Path) -> None:
    """The new name in a `rename` can match an undeclared column, so it is an error too."""
    stdout, _stderr, exit_code = _run_mypy(
        tmp_path,
        _UNKNOWN_STRICT_SETTING,
        _RENAME_DESTINATION_SCENARIO,
    )
    assert _RENAME_COLLISION_MESSAGE in stdout
    assert exit_code == 1


def test_an_undeclared_typed_expr_column_is_an_error_at_the_default_level(tmp_path: Path) -> None:
    """An expression that reads an undeclared column is an error, as a string name is."""
    stdout, _stderr, exit_code = _run_mypy(tmp_path, _UNKNOWN_STRICT_SETTING, _TYPED_EXPR_SCENARIO)
    assert _AMBIGUOUS_COLUMN_MESSAGE in stdout
    assert exit_code == 1


def test_elevate_untyped_select_default_narrows_a_bare_dataframe(tmp_path: Path) -> None:
    """With no `[tool.astrolabe]` section, the plugin uses its default settings."""
    stdout, _stderr, _exit_code = _run_mypy(tmp_path, "", _UNTYPED_SELECT_SCENARIO)
    assert _TYPED_DATAFRAME_REVEAL in stdout


def test_explicit_elevate_untyped_select_true_narrows_a_bare_dataframe(tmp_path: Path) -> None:
    stdout, _stderr, _exit_code = _run_mypy(
        tmp_path,
        "[tool.astrolabe]\nelevate_untyped_select = true\n",
        _UNTYPED_SELECT_SCENARIO,
    )
    assert _TYPED_DATAFRAME_REVEAL in stdout


def test_elevate_untyped_select_false_leaves_a_bare_dataframe_plain(tmp_path: Path) -> None:
    stdout, _stderr, _exit_code = _run_mypy(
        tmp_path,
        "[tool.astrolabe]\nelevate_untyped_select = false\n",
        _UNTYPED_SELECT_SCENARIO,
    )
    assert _PLAIN_DATAFRAME_REVEAL in stdout


def test_inferred_fallback_default_is_permissive(tmp_path: Path) -> None:
    """With no `[tool.astrolabe]` section, the plugin uses its default settings."""
    stdout, _stderr, exit_code = _run_mypy(tmp_path, "", _INFERRED_UNTYPED_FALLBACK_SCENARIO)
    assert _INFERRED_FALLBACK_MESSAGE not in stdout
    assert exit_code == 0


def test_explicit_disallow_implicit_inferred_fallback_false_is_permissive(
    tmp_path: Path,
) -> None:
    stdout, _stderr, exit_code = _run_mypy(
        tmp_path,
        "[tool.astrolabe]\ndisallow_implicit_inferred_fallback = false\n",
        _INFERRED_UNTYPED_FALLBACK_SCENARIO,
    )
    assert _INFERRED_FALLBACK_MESSAGE not in stdout
    assert exit_code == 0


def test_disallow_implicit_inferred_fallback_true_errors(tmp_path: Path) -> None:
    stdout, _stderr, exit_code = _run_mypy(
        tmp_path,
        "[tool.astrolabe]\ndisallow_implicit_inferred_fallback = true\n",
        _INFERRED_UNTYPED_FALLBACK_SCENARIO,
    )
    assert _INFERRED_FALLBACK_MESSAGE in stdout
    assert exit_code == 1
