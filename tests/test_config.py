"""Check how the plugin reads its settings from `[tool.astrolabe]` in `pyproject.toml`.

`AstrolabeConfigLoader.load` reads a file path and gives the settings,
so these tests need no mypy run. They cover a missing file, a missing
or broken section, each setting, and the warning for a bad value, which
falls back to the default. They also check how the plugin finds the
file, and the key that tells mypy when to throw away its cache.
`test_allow_unproven_integration.py` checks that a setting takes effect
in a real mypy run.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
from mypy.options import Options
from mypy.plugin import ReportConfigContext

import astrolabe
from astrolabe._hooks.code_fingerprint import CodeFingerprint
from astrolabe._hooks.config import (
    AstrolabeConfig,
    AstrolabeConfigLoader,
    ConfigPathResolver,
    ImplicitInferredFallbackPolicy,
)
from astrolabe._hooks.unproven import Level, UnprovenPolicy
from astrolabe.plugin import AstrolabePlugin


def _load(tmp_path: Path, content: str) -> AstrolabeConfig:
    path = tmp_path / "pyproject.toml"
    path.write_text(content)
    return AstrolabeConfigLoader().load(path)


def test_a_file_that_is_not_utf8_uses_the_default_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_bytes(b"# \xff\n[tool.astrolabe]\n")
    with caplog.at_level(logging.WARNING):
        assert AstrolabeConfigLoader().load(path) == AstrolabeConfig()
    assert "could not read" in caplog.text


def test_missing_path_uses_default(tmp_path: Path) -> None:
    assert AstrolabeConfigLoader().load(None) == AstrolabeConfig()
    assert AstrolabeConfigLoader().load(tmp_path / "does_not_exist.toml") == AstrolabeConfig()


def test_absent_section_uses_default(tmp_path: Path) -> None:
    assert _load(tmp_path, "[tool.mypy]\nplugins = []\n") == AstrolabeConfig()


def test_malformed_toml_uses_default(tmp_path: Path) -> None:
    assert _load(tmp_path, "this is not [ valid toml") == AstrolabeConfig()


def test_non_table_tool_section_uses_default(tmp_path: Path) -> None:
    assert _load(tmp_path, 'tool = "not a table"\n') == AstrolabeConfig()


def test_non_table_astrolabe_section_uses_default(tmp_path: Path) -> None:
    assert _load(tmp_path, '[tool]\nastrolabe = "not a table"\n') == AstrolabeConfig()


@pytest.mark.parametrize(
    "setting",
    ["strict_undeclared_columns = false", "disallow_implicit_untyped = true"],
)
def test_a_removed_setting_is_ignored_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    setting: str,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert _load(tmp_path, f"[tool.astrolabe]\n{setting}\n") == AstrolabeConfig()
    key = setting.split(" = ", maxsplit=1)[0]
    assert f"unknown setting {key!r}" in caplog.text


def test_every_known_setting_is_read_with_no_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    content = (
        "[tool.astrolabe]\n"
        'allow_unproven = "stated"\n'
        "report_checker_gaps = true\n"
        "disallow_vertical_concat = true\n"
        "elevate_untyped_select = false\n"
        "disallow_implicit_inferred_fallback = true\n"
        "[tool.astrolabe.allow_unproven_overrides]\n"
        'select = "none"\n'
        "[tool.astrolabe.disallow_implicit_inferred_fallback_overrides]\n"
        "expr = false\n"
    )
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, content)
    assert not caplog.text
    assert config == AstrolabeConfig(
        allow_unproven=UnprovenPolicy(Level.STATED, {"select": Level.NONE}, report_gaps=True),
        disallow_vertical_concat=True,
        elevate_untyped_select=False,
        disallow_implicit_inferred_fallback=ImplicitInferredFallbackPolicy(True, {"expr": False}),
    )


def test_absent_allow_unproven_uses_the_default_policy(tmp_path: Path) -> None:
    assert _load(tmp_path, "[tool.astrolabe]\n").allow_unproven == UnprovenPolicy()
    assert UnprovenPolicy() == UnprovenPolicy(Level.OMITTED, {}, report_gaps=False)


@pytest.mark.parametrize("level", list(Level))
def test_allow_unproven_reads_every_level(tmp_path: Path, level: Level) -> None:
    config = _load(tmp_path, f'[tool.astrolabe]\nallow_unproven = "{level.spelling}"\n')
    assert config.allow_unproven == UnprovenPolicy(level)


@pytest.mark.parametrize("value", ['"lenient"', '"NONE"', "true", "3"])
def test_a_malformed_level_uses_the_default_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    value: str,
) -> None:
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, f"[tool.astrolabe]\nallow_unproven = {value}\n")
    assert config.allow_unproven == UnprovenPolicy()
    expected = "allow_unproven must be one of never, none, stated, omitted, possible, asserted"
    assert expected in caplog.text


def test_report_checker_gaps_is_read(tmp_path: Path) -> None:
    config = _load(tmp_path, "[tool.astrolabe]\nreport_checker_gaps = true\n")
    assert config.allow_unproven == UnprovenPolicy(report_gaps=True)


def test_report_checker_gaps_of_the_wrong_type_uses_the_default_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, '[tool.astrolabe]\nreport_checker_gaps = "yes"\n')
    assert config.allow_unproven == UnprovenPolicy()
    assert "report_checker_gaps must be true or false" in caplog.text


def test_overrides_are_read_as_levels(tmp_path: Path) -> None:
    content = (
        '[tool.astrolabe]\nallow_unproven = "possible"\n'
        '[tool.astrolabe.allow_unproven_overrides]\nselect = "none"\nmap_elements = "stated"\n'
    )
    assert _load(tmp_path, content).allow_unproven == UnprovenPolicy(
        Level.POSSIBLE,
        {"select": Level.NONE, "map_elements": Level.STATED},
    )


@pytest.mark.parametrize("value", ['"lenient"', "false"])
def test_a_malformed_override_is_ignored_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    value: str,
) -> None:
    content = f'[tool.astrolabe.allow_unproven_overrides]\nselect = {value}\njoin = "never"\n'
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, content)
    assert config.allow_unproven == UnprovenPolicy(overrides={"join": Level.NEVER})
    assert "[tool.astrolabe.allow_unproven_overrides] select must be one of never" in caplog.text


def test_an_unknown_override_key_is_ignored_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    content = '[tool.astrolabe.allow_unproven_overrides]\nboolean_op = "none"\n'
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, content)
    assert config.allow_unproven == UnprovenPolicy()
    assert "unknown operation 'boolean_op'" in caplog.text


def test_an_overrides_value_that_is_not_a_table_is_ignored_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, '[tool.astrolabe]\nallow_unproven_overrides = "none"\n')
    assert config.allow_unproven == UnprovenPolicy()
    assert "allow_unproven_overrides must be a table" in caplog.text


def test_an_inferred_fallback_override_is_read(tmp_path: Path) -> None:
    content = "[tool.astrolabe.disallow_implicit_inferred_fallback_overrides]\nexpr = true\n"
    expected = ImplicitInferredFallbackPolicy(False, {"expr": True})
    assert _load(tmp_path, content).disallow_implicit_inferred_fallback == expected


def test_a_malformed_inferred_fallback_override_is_ignored_with_a_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    content = '[tool.astrolabe.disallow_implicit_inferred_fallback_overrides]\nexpr = "yes"\n'
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, content)
    assert config.disallow_implicit_inferred_fallback == ImplicitInferredFallbackPolicy()
    assert "expr must be true or false" in caplog.text


# Every operation name that a place in the plugin passes to the policy when it
# reports. The names come from a literal string, from `function.__name__` for
# the file readers and the constructors from other libraries, from
# `UnpivotConfig.operation` for `unpivot` and `melt`, from the filter hook for
# `filter` and `remove`, from the name of an opaque member, and `fit` for the
# fit of a declared parameter. The test asks for equality, not a subset: a name
# in the roster with no place that reports it is a setting that does nothing.
_REAL_OPERATIONS = frozenset({
    "select",
    "with_columns",
    "agg",
    "group_by",
    "rolling",
    "group_by_dynamic",
    "filter",
    "remove",
    "drop",
    "rename",
    "join",
    "join_where",
    "join_asof",
    "explode",
    "unnest",
    "hstack",
    "pivot",
    "corr",
    "unpivot",
    "melt",
    "with_row_index",
    "update",
    "map_columns",
    "concat",
    "Series",
    "read_csv",
    "scan_csv",
    "read_parquet",
    "scan_parquet",
    "read_json",
    "read_ndjson",
    "scan_ndjson",
    "from_numpy",
    "from_dict",
    "from_dicts",
    "from_records",
    "map_elements",
    "map_batches",
    "rolling_map",
    "fit",
    "get_column",
    "fill_null",
    "fill_nan",
})


def test_the_roster_is_exactly_the_operations_with_a_report_site() -> None:
    assert UnprovenPolicy.KNOWN_OPERATIONS == _REAL_OPERATIONS


def test_absent_disallow_vertical_concat_key_uses_default(tmp_path: Path) -> None:
    config = _load(tmp_path, "[tool.astrolabe]\n")
    assert config == AstrolabeConfig(disallow_vertical_concat=False)


def test_disallow_vertical_concat_true_is_honored(tmp_path: Path) -> None:
    config = _load(tmp_path, "[tool.astrolabe]\ndisallow_vertical_concat = true\n")
    assert config == AstrolabeConfig(disallow_vertical_concat=True)


def test_disallow_vertical_concat_false_is_honored(tmp_path: Path) -> None:
    config = _load(tmp_path, "[tool.astrolabe]\ndisallow_vertical_concat = false\n")
    assert config == AstrolabeConfig(disallow_vertical_concat=False)


def test_disallow_vertical_concat_wrong_type_uses_default(tmp_path: Path) -> None:
    config = _load(tmp_path, '[tool.astrolabe]\ndisallow_vertical_concat = "yes"\n')
    assert config == AstrolabeConfig(disallow_vertical_concat=False)


def test_absent_elevate_untyped_select_key_uses_default(tmp_path: Path) -> None:
    config = _load(tmp_path, "[tool.astrolabe]\n")
    assert config == AstrolabeConfig(elevate_untyped_select=True)


def test_elevate_untyped_select_true_is_honored(tmp_path: Path) -> None:
    config = _load(tmp_path, "[tool.astrolabe]\nelevate_untyped_select = true\n")
    assert config == AstrolabeConfig(elevate_untyped_select=True)


def test_elevate_untyped_select_false_is_honored(tmp_path: Path) -> None:
    config = _load(tmp_path, "[tool.astrolabe]\nelevate_untyped_select = false\n")
    assert config == AstrolabeConfig(elevate_untyped_select=False)


def test_elevate_untyped_select_wrong_type_uses_default(tmp_path: Path) -> None:
    config = _load(tmp_path, '[tool.astrolabe]\nelevate_untyped_select = "yes"\n')
    assert config == AstrolabeConfig(elevate_untyped_select=True)


def test_settings_present_together_are_each_honored_independently(tmp_path: Path) -> None:
    content = (
        "[tool.astrolabe]\n"
        "elevate_untyped_select = false\n"
        'allow_unproven = "never"\n'
        "disallow_vertical_concat = true\n"
    )
    assert _load(tmp_path, content) == AstrolabeConfig(
        elevate_untyped_select=False,
        allow_unproven=UnprovenPolicy(Level.NEVER),
        disallow_vertical_concat=True,
    )


def test_one_setting_present_does_not_reset_the_others(tmp_path: Path) -> None:
    assert _load(tmp_path, '[tool.astrolabe]\nallow_unproven = "none"\n') == AstrolabeConfig(
        elevate_untyped_select=True,
        allow_unproven=UnprovenPolicy(Level.NONE),
    )


def test_the_cache_key_carries_every_setting() -> None:
    key = AstrolabeConfig(allow_unproven=UnprovenPolicy(Level.NEVER)).cache_key()
    assert isinstance(key, dict)
    assert set(key) == {
        "allow_unproven",
        "disallow_vertical_concat",
        "elevate_untyped_select",
        "disallow_implicit_inferred_fallback",
    }
    assert key["allow_unproven"] == UnprovenPolicy(Level.NEVER).cache_key()


def test_resolve_prefers_mypy_config_file_when_it_is_pyproject_toml(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text("[tool.astrolabe]\ndisallow_vertical_concat = true\n")
    options = Options()
    options.config_file = str(path)
    assert ConfigPathResolver().resolve(options) == path


def test_resolve_falls_back_to_cwd_pyproject_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text("[tool.astrolabe]\ndisallow_vertical_concat = true\n")
    monkeypatch.chdir(tmp_path)
    options = Options()
    options.config_file = str(tmp_path / "mypy.ini")  # an ini file, with no [tool.astrolabe]
    assert ConfigPathResolver().resolve(options) == path


def test_resolve_returns_none_when_nothing_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    options = Options()
    options.config_file = None
    assert ConfigPathResolver().resolve(options) is None


def test_the_plugin_warns_when_redefinition_is_off(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plugin warns if `allow_redefinition` is off, because a loop then gets no stable type."""
    monkeypatch.chdir(tmp_path)
    options = Options()
    options.config_file = None
    options.allow_redefinition = False
    with caplog.at_level(logging.WARNING):
        AstrolabePlugin(options)
    assert "allow_redefinition is off" in caplog.text
    caplog.clear()
    options.allow_redefinition = True
    with caplog.at_level(logging.WARNING):
        AstrolabePlugin(options)
    assert "allow_redefinition" not in caplog.text


def test_the_cache_key_names_the_modules_the_plugin_forces_into_the_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache key lists the modules that the plugin adds to each build.

    If that list changes, the key changes, so mypy does not reuse a cache
    that was built with another list.
    """
    monkeypatch.chdir(tmp_path)
    options = Options()
    options.config_file = None
    options.allow_redefinition = True
    key = AstrolabePlugin(options).report_config_data(ReportConfigContext("m", "m.py", True))
    assert key == {
        "config": AstrolabeConfig().cache_key(),
        "forced_modules": [
            "astrolabe.frame",
            "astrolabe.remainder",
            "astrolabe.expr",
            "astrolabe.declare",
        ],
        "code": CodeFingerprint().digest(),
    }


def test_the_cache_key_changes_when_a_rule_of_the_plugin_changes(tmp_path: Path) -> None:
    """The fingerprint of the code reads every source file, so a changed rule changes the key.

    mypy hashes only the module that loads the plugin, so without the
    fingerprint it reuses the results of the earlier code of a rule.
    """
    package = Path(astrolabe.__file__).parent
    copy = tmp_path / "astrolabe"
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert CodeFingerprint().of(copy) == CodeFingerprint().digest()
    rule = copy / "_rules" / "casts.py"
    rule.write_text(rule.read_text() + "\n# a changed rule\n")
    assert CodeFingerprint().of(copy) != CodeFingerprint().digest()
