"""Check the command line of the mutation harness: its options, its limit and one full run."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from coverage import CoverageData

from tools.mutation import __main__ as entry
from tools.mutation.command import MutationCommand
from tools.mutation.mutants import Mutant
from tools.mutation.runner import MutantRunner
from tools.mutation.store import StaleStore
from tools.mutation.sweep import Sweep


class _FixedSweep(Sweep):
    """Stand in for the sweep, and give a fixed list of mutants in place of real ones."""

    def __init__(self, count: int) -> None:
        """Make the given number of mutants, one on each line."""
        super().__init__(None)
        self._found = [Mutant(Path("a.py"), "core/AddNot", n, n) for n in range(count)]

    def mutants(self, scope: Path) -> list[Mutant]:
        """Give the fixed list, whatever the scope.

        Returns:
            The mutants.
        """
        return self._found


def _command() -> MutationCommand:
    """Build the command with its output going to memory, so that a test shows nothing.

    Returns:
        The command.
    """
    return MutationCommand(io.StringIO(), io.StringIO())


def test_the_parser_needs_a_data_file() -> None:
    args = _command().parser().parse_args(["--data", "d"])
    assert args.data == "d"
    assert args.scope == MutationCommand.SCOPE
    assert args.workers == MutationCommand.WORKERS
    assert args.limit == 0
    assert args.timeout == MutantRunner.LIMIT
    assert args.retry_undecided is False


def test_the_parser_reads_every_option() -> None:
    args = (
        _command()
        .parser()
        .parse_args(
            [
                "--data",
                "d",
                "--scope",
                "s",
                "--root",
                "r",
                "--home",
                "h",
                "--workers",
                "2",
                "--limit",
                "5",
            ],
        )
    )
    assert (args.scope, args.root, args.home, args.workers, args.limit) == ("s", "r", "h", 2, 5)


def test_the_parser_reads_the_retry_options() -> None:
    args = (
        _command()
        .parser()
        .parse_args(
            ["--data", "d", "--retry-survived", "--retry-undecided"],
        )
    )
    assert (args.retry_survived, args.retry_undecided) == (True, True)


def _chosen(count: int, limit: int) -> list[Mutant]:
    """Give the mutants that the command picks from a fixed list, under a limit.

    Args:
        count: The number of mutants in the list.
        limit: The most mutants to run, or 0 for no limit.

    Returns:
        The mutants that the command runs.
    """
    sweep = _FixedSweep(count)
    return _command().chosen(sweep, sweep.mutants(Path()), limit)


def test_no_limit_keeps_every_mutant() -> None:
    assert len(_chosen(10, 0)) == 10


def test_a_limit_above_the_count_keeps_every_mutant() -> None:
    assert len(_chosen(3, 99)) == 3


def test_a_limit_takes_a_spread_of_the_work_left() -> None:
    found = _chosen(100, 10)
    assert [m.occurrence for m in found] == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def _toy_map(toy: Path) -> Path:
    """Write a coverage data file that says one toy test runs every line of the toy module.

    Args:
        toy: The root of the toy project.

    Returns:
        The path of the file.
    """
    data_file = toy / "toy.coverage"
    data = CoverageData(basename=str(data_file))
    data.set_context("tests/test_toy.py::test_positive|run")
    data.add_arcs({str(toy / "src" / "toy.py"): {(n, n + 1) for n in range(1, 40)}})
    data.write()
    return data_file


def test_it_runs_a_whole_sweep_from_the_command_line(
    toy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_file = _toy_map(toy)
    monkeypatch.chdir(toy)
    out, err = io.StringIO(), io.StringIO()
    code = MutationCommand(out, err).run(
        [
            "--data",
            str(data_file),
            "--scope",
            "src",
            "--root",
            ".",
            "--home",
            str(tmp_path / "h"),
            "--workers",
            "1",
            "--limit",
            "1",
            "--package",
            "toy",
        ],
    )
    assert code == 0
    assert "mutation score" in out.getvalue()
    assert "1 to run, 1 workers" in err.getvalue()


def test_a_relative_root_and_home_run_a_sweep(
    toy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_file = _toy_map(toy)
    monkeypatch.chdir(toy)
    out, err = io.StringIO(), io.StringIO()
    code = MutationCommand(out, err).run(
        [
            "--data",
            str(data_file),
            "--scope",
            "src",
            "--root",
            ".",
            "--home",
            "../h",
            "--workers",
            "1",
            "--limit",
            "1",
            "--package",
            "toy",
            "--all-tests",
        ],
    )
    assert code == 0
    assert "mutation score" in out.getvalue()
    assert (tmp_path / "h").is_dir()


def test_main_reads_the_real_command_line(
    toy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_file = _toy_map(toy)
    monkeypatch.chdir(toy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tools.mutation",
            "--data",
            str(data_file),
            "--scope",
            "src",
            "--root",
            ".",
            "--home",
            str(tmp_path / "h2"),
            "--workers",
            "1",
            "--limit",
            "1",
            "--package",
            "toy",
        ],
    )
    assert MutationCommand.main() == 0
    assert "mutation score" in capsys.readouterr().out


def test_a_second_run_reuses_what_the_first_finished(
    toy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_file = _toy_map(toy)
    monkeypatch.chdir(toy)
    home = tmp_path / "h3"
    argv = [
        "--data",
        str(data_file),
        "--scope",
        "src",
        "--root",
        ".",
        "--home",
        str(home),
        "--workers",
        "1",
        "--package",
        "toy",
    ]
    first_err = io.StringIO()
    MutationCommand(io.StringIO(), first_err).run(argv)
    assert "to run" in first_err.getvalue()

    second_err = io.StringIO()
    MutationCommand(io.StringIO(), second_err).run([*argv, "--resume"])
    assert "0 to run" in second_err.getvalue()


def test_without_resume_a_second_run_does_the_work_again(
    toy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_file = _toy_map(toy)
    monkeypatch.chdir(toy)
    home = tmp_path / "h4"
    argv = [
        "--data",
        str(data_file),
        "--scope",
        "src",
        "--root",
        ".",
        "--home",
        str(home),
        "--workers",
        "1",
        "--limit",
        "1",
        "--package",
        "toy",
    ]
    MutationCommand(io.StringIO(), io.StringIO()).run(argv)
    err = io.StringIO()
    MutationCommand(io.StringIO(), err).run(argv)
    assert "1 to run" in err.getvalue()


def test_the_entry_point_offers_the_command() -> None:
    assert entry.MutationCommand is MutationCommand


def test_a_resume_after_a_source_change_stops_the_run(
    toy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_file = _toy_map(toy)
    monkeypatch.chdir(toy)
    argv = [
        "--data",
        str(data_file),
        "--scope",
        "src",
        "--root",
        ".",
        "--home",
        str(tmp_path / "h5"),
        "--workers",
        "1",
        "--package",
        "toy",
    ]
    MutationCommand(io.StringIO(), io.StringIO()).run(argv)

    source = toy / "src" / "toy.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n# one more line\n", encoding="utf-8")
    with pytest.raises(StaleStore, match="toy"):
        MutationCommand(io.StringIO(), io.StringIO()).run([*argv, "--resume"])
