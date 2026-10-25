"""Check the catalog of the website's examples against real runs of mypy with the plugin.

Each test writes small example files, runs the command over them, and
reads the catalog that it writes. The pages of the website show that
catalog, so its messages and its columns must be the ones that mypy gives.
"""

from __future__ import annotations

import io
import json
import sys
from typing import TYPE_CHECKING

import pytest

from tools.site_examples import __main__ as entry
from tools.site_examples.catalog import ExampleCatalog, HiddenMessage
from tools.site_examples.command import SiteExamplesCommand
from tools.site_examples.messages import MypyBatch, MypyFailure

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = """\
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):  # hide
    symbol: pl.String  # hide
    price: pl.Float64  # hide


"""


def _write(directory: Path, name: str, text: str) -> None:
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.py").write_text(text, encoding="utf-8")


def _command(tmp_path: Path, *extra: str) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    argv = [
        "--examples",
        str(tmp_path / "examples"),
        "--output",
        str(tmp_path / "out" / "examples.json"),
        *extra,
    ]
    code = SiteExamplesCommand(out, err).run(argv)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, dict[str, object]]]:
    root = tmp_path_factory.mktemp("site")
    examples = root / "examples"
    _write(
        examples,
        "select_typo",
        "# file: pipeline.py\n# start: 4\n"
        + SCHEMA
        + "trades: TypedDataFrame[Trades]\n"
        + 'trades.select("prcie")\n'
        + 'reveal_type(trades.select("symbol"))  # dropped: price\n',
    )
    _write(
        examples,
        "strict_select",
        '# setting: allow_unproven = "none"\n'
        + SCHEMA
        + "def pick(df: TypedDataFrame[Trades], name: str) -> None:\n    df.select(name)\n",
    )
    code, _, _ = _command(root)
    assert code == 0
    data = json.loads((root / "out" / "examples.json").read_text(encoding="utf-8"))
    return root, data


def test_shown_lines_keep_their_page_numbers(
    catalog: tuple[Path, dict[str, dict[str, object]]],
) -> None:
    _, data = catalog
    entry = data["select_typo"]
    assert entry["file"] == "pipeline.py"
    assert entry["start"] == 4
    assert entry["lines"] == ["trades: TypedDataFrame[Trades]", 'trades.select("prcie")']
    assert entry["messages"] == [
        {
            "line": 5,
            "severity": "error",
            "text": 'Unknown column "prcie"',
            "code": "unknown-column",
        },
    ]


def test_a_revealed_frame_becomes_its_columns(
    catalog: tuple[Path, dict[str, dict[str, object]]],
) -> None:
    _, data = catalog
    assert data["select_typo"]["results"] == [
        {
            "label": 'trades.select("symbol")',
            "frames": [
                {
                    "kind": "TypedDataFrame",
                    "columns": [["symbol", "String", ""], ["price", "", "gone"]],
                    "remainder": None,
                },
            ],
            "revealed": "TypedDataFrame[TypedDict({'symbol': String}, closed=True), None]",
        },
    ]


def test_a_setting_changes_the_messages(catalog: tuple[Path, dict[str, dict[str, object]]]) -> None:
    _, data = catalog
    messages = data["strict_select"]["messages"]
    assert isinstance(messages, list)
    assert [message["severity"] for message in messages] == ["error", "note"]


def test_check_passes_on_a_fresh_catalog(
    catalog: tuple[Path, dict[str, dict[str, object]]],
) -> None:
    root, _ = catalog
    code, out, _ = _command(root, "--check")
    assert code == 0
    assert "up to date" in out


def test_check_fails_on_a_changed_example(tmp_path: Path) -> None:
    _write(tmp_path / "examples", "plain", "x = 1\n")
    assert _command(tmp_path)[0] == 0
    _write(tmp_path / "examples", "plain", "x = 2\n")
    code, _, err = _command(tmp_path, "--check")
    assert code == 1
    assert "out of date" in err


def test_check_fails_without_a_catalog(tmp_path: Path) -> None:
    _write(tmp_path / "examples", "plain", "x = 1\n")
    assert _command(tmp_path, "--check")[0] == 1


def test_a_message_on_a_hidden_line_stops_the_build(tmp_path: Path) -> None:
    _write(tmp_path / "examples", "hidden", "x: int = 'a'  # hide\n")
    catalog = ExampleCatalog(tmp_path / "examples", tmp_path / "out.json")
    with pytest.raises(HiddenMessage, match="hidden line 1"):
        catalog.build()


def test_a_type_that_is_not_a_frame_has_no_columns(tmp_path: Path) -> None:
    _write(tmp_path / "examples", "number", "reveal_type(1 + 1)\n")
    catalog = ExampleCatalog(tmp_path / "examples", tmp_path / "out.json")
    entry = json.loads(catalog.build())["number"]
    assert entry["results"][0]["frames"] == []
    assert entry["results"][0]["revealed"] == "int"


def test_main_reads_the_command_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path / "examples", "plain", "x = 1\n")
    output = tmp_path / "examples.json"
    argv = ["site_examples", "--examples", str(tmp_path / "examples"), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    assert SiteExamplesCommand.main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["plain"]["lines"] == ["x = 1"]


def test_the_module_entry_is_the_command() -> None:
    assert entry.SiteExamplesCommand is SiteExamplesCommand


def test_a_configuration_that_mypy_cannot_read_stops_the_build(tmp_path: Path) -> None:
    _write(tmp_path / "examples", "broken", "# setting: allow_unproven = = 1\nx = 1\n")
    catalog = ExampleCatalog(tmp_path / "examples", tmp_path / "out.json")
    with pytest.raises(MypyFailure, match="mypy did not finish"):
        catalog.build()


def test_a_file_that_mypy_cannot_read_stops_the_build(tmp_path: Path) -> None:
    with pytest.raises(MypyFailure, match="Cannot read file"):
        MypyBatch([]).run([tmp_path / "missing.py"], tmp_path)
