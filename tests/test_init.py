"""Check the `astrolabe init` command, which adds the plugin to the `pyproject.toml` of a project.

Each test runs the command in a real directory under `tmp_path`. It
then reads the file that the command wrote with `tomllib`. The test
therefore checks the file that a user gets, and not the `tomlkit`
objects that the command edits.
"""

from __future__ import annotations

import io
import logging
import sys
import tomllib
from typing import TYPE_CHECKING

import mypy.api
import pytest

from astrolabe._hooks.config import AstrolabeConfig, AstrolabeConfigLoader
from astrolabe._hooks.unproven import Level, UnprovenPolicy
from astrolabe._cli import AstrolabeSection, InitCommand

if TYPE_CHECKING:
    from pathlib import Path

_EXISTING = """\
# The demo project.
[project]
name = "demo"

[tool.mypy]
strict = true
plugins = [
    "other.plugin",  # keep this comment
]

[tool.ruff]
line-length = 100
"""


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = InitCommand(out, err).run(["init", *argv])
    return code, out.getvalue(), err.getvalue()


def _read(root: Path) -> dict[str, object]:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def _section(root: Path, name: str) -> dict[str, object]:
    tool = _read(root)["tool"]
    assert isinstance(tool, dict)
    section = tool[name]
    assert isinstance(section, dict)
    return section


def test_a_fresh_directory_gets_a_minimal_pyproject(tmp_path: Path) -> None:
    code, out, _err = _run("--path", str(tmp_path))
    assert code == 0
    assert out.startswith("create pyproject.toml\n")
    assert _read(tmp_path).keys() == {"tool"}
    assert _section(tmp_path, "mypy") == {
        "plugins": ["astrolabe.plugin"],
        "allow_redefinition": True,
    }


def test_the_written_settings_load_as_the_defaults_with_no_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _run("--path", str(tmp_path))
    with caplog.at_level(logging.WARNING):
        config = AstrolabeConfigLoader().load(tmp_path / "pyproject.toml")
    assert config == AstrolabeConfig()
    assert not caplog.records


def test_every_setting_except_the_override_tables_is_written() -> None:
    keys = {entry.key for entry in AstrolabeSection(None).entries()}
    overrides = {key for key in AstrolabeConfig.KNOWN_KEYS if key.endswith("_overrides")}
    assert keys == AstrolabeConfig.KNOWN_KEYS - overrides


def test_an_existing_file_keeps_its_comments_tables_and_plugins(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(_EXISTING, encoding="utf-8")
    code, out, _err = _run("--path", str(tmp_path))
    text = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert code == 0
    assert "create" not in out
    assert text.startswith(_EXISTING.partition("[tool.mypy]")[0])
    assert "# keep this comment" in text
    assert _section(tmp_path, "mypy")["plugins"] == ["other.plugin", "astrolabe.plugin"]
    assert _section(tmp_path, "ruff") == {"line-length": 100}
    assert _section(tmp_path, "astrolabe")["allow_unproven"] == "omitted"


def test_a_string_plugin_list_becomes_an_array(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mypy]\nplugins = "a.plugin, b.plugin"\n',
        encoding="utf-8",
    )
    _run("--path", str(tmp_path))
    assert _section(tmp_path, "mypy")["plugins"] == ["a.plugin", "b.plugin", "astrolabe.plugin"]


def test_allow_redefinition_false_is_set_to_true_and_reported(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.mypy]\nallow_redefinition = false\n",
        encoding="utf-8",
    )
    _code, out, _err = _run("--path", str(tmp_path))
    assert "set tool.mypy.allow_redefinition = true (it was false)" in out
    assert _section(tmp_path, "mypy")["allow_redefinition"] is True


def test_a_second_run_changes_nothing(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(_EXISTING, encoding="utf-8")
    _run("--path", str(tmp_path))
    before = (tmp_path / "pyproject.toml").read_bytes()
    code, out, _err = _run("--path", str(tmp_path))
    assert code == 0
    assert out == "pyproject.toml is already set up.\n"
    assert (tmp_path / "pyproject.toml").read_bytes() == before


def test_an_existing_setting_is_kept_and_missing_ones_are_added(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.astrolabe]\nallow_unproven = "none"\ndisallow_vertical_concat = true\n',
        encoding="utf-8",
    )
    _code, out, _err = _run("--path", str(tmp_path))
    section = _section(tmp_path, "astrolabe")
    assert section["allow_unproven"] == "none"
    assert section["disallow_vertical_concat"] is True
    assert section["report_checker_gaps"] is False
    assert "allow_unproven" not in out


def test_the_level_flag_overrides_an_existing_level(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.astrolabe]\nallow_unproven = "none"\n',
        encoding="utf-8",
    )
    _code, out, _err = _run("--path", str(tmp_path), "--allow-unproven", "never")
    assert 'set tool.astrolabe.allow_unproven = "never" (it was "none")' in out
    assert _section(tmp_path, "astrolabe")["allow_unproven"] == "never"


def test_the_level_flag_sets_the_level_of_a_new_table(tmp_path: Path) -> None:
    _run("--path", str(tmp_path), "--allow-unproven", "possible")
    config = AstrolabeConfigLoader().load(tmp_path / "pyproject.toml")
    assert config.allow_unproven == UnprovenPolicy(Level.POSSIBLE)


def test_an_inline_astrolabe_table_gets_the_missing_keys(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool]\nastrolabe = {allow_unproven = "stated"}\n',
        encoding="utf-8",
    )
    code, _out, _err = _run("--path", str(tmp_path))
    section = _section(tmp_path, "astrolabe")
    assert code == 0
    assert section["allow_unproven"] == "stated"
    assert section["elevate_untyped_select"] is True


def test_a_dry_run_prints_the_changes_and_writes_nothing(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(_EXISTING, encoding="utf-8")
    code, out, _err = _run("--path", str(tmp_path), "--dry-run")
    assert code == 0
    assert 'add "astrolabe.plugin" to tool.mypy.plugins' in out
    assert out.endswith("Dry run: pyproject.toml is not changed.\n")
    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == _EXISTING


def test_a_dry_run_in_a_fresh_directory_creates_nothing(tmp_path: Path) -> None:
    _run("--path", str(tmp_path), "--dry-run")
    assert not (tmp_path / "pyproject.toml").exists()


@pytest.mark.parametrize("name", ["mypy.ini", ".mypy.ini"])
def test_an_ini_mypy_config_is_refused(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("[mypy]\nstrict = True\n", encoding="utf-8")
    code, out, err = _run("--path", str(tmp_path))
    assert code == 1
    assert not out
    assert f"mypy reads its configuration from {name}" in err
    assert "plugins = astrolabe.plugin" in err
    assert not (tmp_path / "pyproject.toml").exists()


def test_a_setup_cfg_with_a_mypy_section_is_refused(tmp_path: Path) -> None:
    (tmp_path / "setup.cfg").write_text("[mypy]\nstrict = True\n", encoding="utf-8")
    code, _out, err = _run("--path", str(tmp_path))
    assert code == 1
    assert "setup.cfg" in err


def test_a_setup_cfg_beside_a_tool_mypy_table_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "setup.cfg").write_text("[mypy]\nstrict = True\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.mypy]\nstrict = true\n", encoding="utf-8")
    code, out, _err = _run("--path", str(tmp_path))
    assert code == 0
    assert 'add "astrolabe.plugin" to tool.mypy.plugins' in out


@pytest.mark.parametrize("text", ["[tool\n", "tool = 1\n"])
def test_a_setup_cfg_beside_a_pyproject_without_a_mypy_table_is_refused(
    tmp_path: Path,
    text: str,
) -> None:
    (tmp_path / "setup.cfg").write_text("[mypy]\nstrict = True\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(text, encoding="utf-8")
    code, _out, err = _run("--path", str(tmp_path))
    assert code == 1
    assert "setup.cfg" in err


def test_a_setup_cfg_without_a_mypy_section_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "setup.cfg").write_text("[flake8]\nmax-line-length = 100\n", encoding="utf-8")
    code, _out, _err = _run("--path", str(tmp_path))
    assert code == 0
    assert (tmp_path / "pyproject.toml").exists()


def test_invalid_toml_is_refused(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool\n", encoding="utf-8")
    code, _out, err = _run("--path", str(tmp_path))
    assert code == 1
    assert "is not valid TOML" in err


def test_a_tool_value_that_is_not_a_table_is_refused(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("tool = 1\n", encoding="utf-8")
    code, _out, err = _run("--path", str(tmp_path))
    assert code == 1
    assert '"tool" in pyproject.toml is not a table' in err


def test_a_missing_directory_is_refused(tmp_path: Path) -> None:
    code, _out, err = _run("--path", str(tmp_path / "missing"))
    assert code == 1
    assert "does not exist" in err


def test_an_unknown_level_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _run("--path", str(tmp_path), "--allow-unproven", "loose")
    assert exit_info.value.code == 2


def test_main_reads_the_process_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["astrolabe", "init", "--path", str(tmp_path)])
    assert InitCommand.main() == 0
    assert "create pyproject.toml" in capsys.readouterr().out


def test_mypy_runs_the_plugin_after_init(tmp_path: Path) -> None:
    _run("--path", str(tmp_path))
    main_file = tmp_path / "main.py"
    main_file.write_text(
        "import polars as pl\n"
        "from astrolabe import TypedDataFrame, columns\n"
        'A = columns("A", {"a": pl.Int64})\n'
        "\n"
        "df: TypedDataFrame[A]\n"
        'df.select("missing")\n',
        encoding="utf-8",
    )
    stdout, _stderr, exit_code = mypy.api.run([
        "--config-file",
        str(tmp_path / "pyproject.toml"),
        "--cache-dir",
        str(tmp_path / ".mypy_cache"),
        str(main_file),
    ])
    assert "[unknown-column]" in stdout
    assert exit_code == 1
