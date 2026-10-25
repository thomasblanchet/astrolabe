"""Check the private copy of each worker, and the run of one mutant in it.

The check of the copy matters most. If a worker imports the real
source of the project, and not its own changed copy, no test sees the
change. Every mutant then survives, and the whole run proves nothing.
"""

from __future__ import annotations

import shutil
from itertools import chain
from pathlib import Path

import pytest

from tools.mutation.mutants import Mutant
from tools.mutation.pytest_call import CallResult, PytestCall, checked_stream
from tools.mutation.runner import MutantRunner, MutantResult, Mutation, Outcome, Worker


def _worker(toy: Path, home: Path) -> Worker:
    """Make one worker with its own copy of the toy project.

    Args:
        toy: The root of the toy project.
        home: The directory for the copies of the workers.

    Returns:
        The worker, with its copy ready.
    """
    worker = Worker(toy, home / "w0", package="toy")
    worker.prepare()
    return worker


def _mutant(operator: str, occurrence: int) -> Mutant:
    """Name one change to the toy module by its operator and its position.

    Args:
        operator: The name of the change operator, for example `core/AddNot`.
        occurrence: Which place in the file, counted from 0, the operator changes.

    Returns:
        The mutant.
    """
    return Mutant(Path("src/toy.py"), operator, occurrence, 0)


def test_worker_copies_the_project(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    assert (worker.home / "src" / "toy.py").exists()
    assert (worker.home / "tests" / "test_toy.py").exists()
    assert (worker.home / "pyproject.toml").exists()


def test_worker_replaces_an_older_copy(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    (worker.home / "stale.txt").write_text("x", encoding="utf-8")
    worker.prepare()
    assert not (worker.home / "stale.txt").exists()


def test_worker_skips_an_entry_the_project_does_not_have(toy: Path, tmp_path: Path) -> None:
    shutil.rmtree(toy / "spec")
    worker = _worker(toy, tmp_path)
    assert not (worker.home / "spec").exists()


def test_worker_reads_and_writes_its_own_copy(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    worker.write(Path("src/toy.py"), "x = 1\n")
    assert worker.read(Path("src/toy.py")) == "x = 1\n"
    assert (toy / "src" / "toy.py").read_text(encoding="utf-8") != "x = 1\n"


def test_worker_accepts_a_copy_that_supplies_the_package(toy: Path, tmp_path: Path) -> None:
    _worker(toy, tmp_path).verify()


def test_worker_refuses_a_copy_that_does_not_supply_the_package(toy: Path, tmp_path: Path) -> None:
    worker = Worker(toy, tmp_path / "w1", package="astrolabe")
    worker.prepare()
    with pytest.raises(RuntimeError, match="not the copy under"):
        worker.verify()


def test_runner_kills_a_mutant_that_a_test_catches(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    mutant = _mutant("core/ReplaceComparisonOperator_Gt_GtE", 0)
    result = MutantRunner().run(worker, mutant, ["tests/test_toy.py::test_positive"])
    assert result.outcome is Outcome.KILLED
    assert result.killer == "tests/test_toy.py::test_positive"


def test_runner_restores_the_file_after_a_run(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    before = worker.read(Path("src/toy.py"))
    mutant = _mutant("core/ReplaceComparisonOperator_Gt_GtE", 0)
    MutantRunner().run(worker, mutant, ["tests/test_toy.py::test_positive"])
    assert worker.read(Path("src/toy.py")) == before


def test_runner_reports_a_mutant_that_no_test_catches(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    mutant = _mutant("core/ReplaceComparisonOperator_Gt_GtE", 1)
    result = MutantRunner().run(worker, mutant, ["tests/test_toy.py::test_clamp"])
    assert result.outcome is Outcome.SURVIVED
    assert result.killer is None


def test_runner_skips_a_mutant_that_no_test_covers(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    mutant = _mutant("core/ReplaceComparisonOperator_Gt_GtE", 0)
    result = MutantRunner().run(worker, mutant, [])
    assert result.outcome is Outcome.UNSELECTED


def test_runner_skips_a_change_the_operator_does_not_make(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    mutant = _mutant("core/ReplaceBreakWithContinue", 0)
    result = MutantRunner().run(worker, mutant, ["tests/test_toy.py::test_clamp"])
    assert result.outcome is Outcome.UNCHANGED


def test_runner_builds_each_operator_once() -> None:
    runner = MutantRunner()
    assert runner.operator("core/AddNot") is runner.operator("core/AddNot")


def test_runner_holds_one_limit_for_the_whole_run() -> None:
    runner = MutantRunner()
    assert runner.timeout() == pytest.approx(MutantRunner.LIMIT)
    runner.classify(_mutant("core/AddNot", 0), CallResult(0, "", timed_out=False), 400.0)
    assert runner.timeout() == pytest.approx(MutantRunner.LIMIT)


def test_runner_takes_a_limit_of_its_own() -> None:
    assert MutantRunner(limit=45.0).timeout() == pytest.approx(45.0)


def test_runner_reads_a_failing_test_out_of_the_summary() -> None:
    assert MutantRunner().killer("x\nFAILED t.py::a - boom\n") == "t.py::a"


def test_runner_reads_a_collection_error_out_of_the_summary() -> None:
    assert MutantRunner().killer("ERROR t.py::a\n") == "t.py::a"


def test_runner_answers_no_killer_where_the_summary_names_none() -> None:
    assert MutantRunner().killer("nothing here") is None


def test_result_counts_only_a_measured_outcome() -> None:
    assert MutantResult(None, Outcome.KILLED, None, 0.0).scored()
    assert MutantResult(None, Outcome.SURVIVED, None, 0.0).scored()
    assert not MutantResult(None, Outcome.UNSELECTED, None, 0.0).scored()
    assert not MutantResult(None, Outcome.UNCHANGED, None, 0.0).scored()
    assert not MutantResult(None, Outcome.TIMED_OUT, None, 0.0).scored()
    assert not MutantResult(None, Outcome.ERRORED, None, 0.0).scored()


def test_call_result_reads_a_timeout_as_a_failure() -> None:
    assert CallResult(0, "", timed_out=True).failed()


def test_pytest_call_keeps_the_flags_the_tests_need() -> None:
    arguments = PytestCall().arguments(["a.py"], stop_early=True)
    assert "--doctest-modules" in " ".join(arguments)
    assert "-x" in arguments
    assert arguments[-1] == "a.py"


def test_pytest_call_keeps_the_flag_that_reaches_the_patched_plugin() -> None:
    arguments = PytestCall().arguments(["a.py"], stop_early=True)
    assert "--mypy-same-process" in " ".join(arguments)


def test_pytest_call_cuts_a_long_selection_into_batches() -> None:
    targets = [f"t.py::case{n}" for n in range(95)]
    batches = PytestCall().batches(targets)
    assert [len(batch) for batch in batches] == [40, 40, 15]
    assert list(chain.from_iterable(batches)) == targets


def test_pytest_call_keeps_a_short_selection_in_one_batch() -> None:
    assert PytestCall().batches(["a", "b"]) == [["a", "b"]]


def test_pytest_call_runs_nothing_for_an_empty_selection() -> None:
    assert PytestCall().batches([]) == []


def test_a_batch_after_the_deadline_reports_the_limit(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    result = PytestCall().run(["tests/test_toy.py"], worker.home, worker.source, -1.0)
    assert result.timed_out


def test_pytest_call_can_run_without_stopping_early() -> None:
    assert "-x" not in PytestCall().arguments(["a.py"], stop_early=False)


def test_pytest_call_points_the_import_at_one_tree(tmp_path: Path) -> None:
    environment = PytestCall().environment(tmp_path / "src")
    assert environment["PYTHONPATH"] == str(tmp_path / "src")
    assert "COVERAGE_FILE" not in environment


def test_pytest_call_reports_a_run_that_passes_its_limit(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    result = PytestCall().run(["tests/test_toy.py"], worker.home, worker.source, 0.001)
    assert result.timed_out
    assert result.failed()


def test_runner_does_not_read_a_run_out_of_time_as_a_kill() -> None:
    runner = MutantRunner()
    mutant = _mutant("core/AddNot", 0)
    result = runner.classify(mutant, CallResult(-1, "", timed_out=True), 9.0)
    assert result.outcome is Outcome.TIMED_OUT
    assert not result.scored()


def test_runner_names_a_run_stuck_in_one_test_hung() -> None:
    runner = MutantRunner()
    mutant = _mutant("core/AddNot", 0)
    stuck = CallResult(-1, "", timed_out=True, stalled=True)
    result = runner.classify(mutant, stuck, 9.0)
    assert result.outcome is Outcome.HUNG
    assert not result.scored()


def test_a_run_that_writes_until_the_limit_is_not_stalled(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    result = PytestCall().run(["tests/test_toy.py"], worker.home, worker.source, 0.001)
    assert result.timed_out
    assert not result.stalled


def test_a_run_that_writes_nothing_for_the_stall_time_is_stalled(
    toy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PytestCall, "STALL", 0.0)
    worker = _worker(toy, tmp_path)
    result = PytestCall().run(["tests/test_toy.py"], worker.home, worker.source, 0.001)
    assert result.stalled


def test_pytest_call_reads_the_output_of_a_passing_run(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    result = PytestCall().run(["tests/test_toy.py"], worker.home, worker.source, 120.0)
    assert not result.failed()


def test_pytest_call_reads_the_output_of_a_failing_run(toy: Path, tmp_path: Path) -> None:
    worker = _worker(toy, tmp_path)
    (worker.home / "tests" / "test_toy.py").write_text(
        "def test_no() -> None:\n    assert False\n",
        encoding="utf-8",
    )
    result = PytestCall().run(["tests/test_toy.py"], worker.home, worker.source, 120.0)
    assert result.failed()
    assert "test_no" in result.output


def test_pytest_call_makes_the_output_arrive_as_each_test_ends(tmp_path: Path) -> None:
    assert PytestCall().environment(tmp_path)["PYTHONUNBUFFERED"] == "1"


DOTTED_EXCEPT = """import tomllib


def read(path):
    try:
        return path.read_text()
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError from exc
"""


def test_runner_reports_an_operator_that_raises_instead_of_ending_the_run() -> None:
    runner = MutantRunner()
    mutant = _mutant("core/ExceptionReplacer", 1)
    found = runner.mutation(DOTTED_EXCEPT, mutant)
    assert found.text is None
    assert found.refusal is Outcome.ERRORED


def test_runner_still_mutates_a_plain_exception_name() -> None:
    runner = MutantRunner()
    found = runner.mutation(DOTTED_EXCEPT, _mutant("core/ExceptionReplacer", 0))
    assert found.text is not None
    assert found.refusal is Outcome.UNCHANGED


def test_runner_reports_a_change_the_operator_did_not_make() -> None:
    runner = MutantRunner()
    found = runner.mutation("x = 1\n", _mutant("core/ReplaceBreakWithContinue", 0))
    assert found.text is None
    assert found.refusal is Outcome.UNCHANGED


def test_a_mutation_carries_the_text_the_operator_built() -> None:
    assert Mutation("x = 2", Outcome.UNCHANGED).text == "x = 2"


def test_a_process_with_no_pipe_is_refused() -> None:
    with pytest.raises(TypeError, match="without a pipe"):
        checked_stream(None)
