"""Check the sweep, which hands the mutants to the workers and runs them.

Each worker takes its next mutant from one shared queue. The costs of
mutants differ a lot, and a queue keeps every worker busy until the
last mutant. The tests check that each mutant goes to exactly one
worker, even when several threads take at once.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

from tools.mutation.coverage_map import CoverageMap
from tools.mutation.mutants import Mutant
from tools.mutation.runner import MutantResult, Outcome
from tools.mutation.store import ResultStore
from tools.mutation.sweep import MutantQueue, Progress, Sweep


def _numbered(count: int) -> list[Mutant]:
    """Build some mutants that differ only by their line.

    Args:
        count: The number of mutants.

    Returns:
        The mutants, on the lines 0, 1, 2 and so on.
    """
    return [Mutant(Path("a.py"), "core/AddNot", n, n) for n in range(count)]


def test_the_queue_gives_each_mutant_once_and_in_order() -> None:
    queue = MutantQueue(_numbered(3))
    taken = [queue.take() for _ in range(3)]
    assert [m.line for m in taken if m is not None] == [0, 1, 2]
    assert queue.take() is None


def test_the_queue_ends_when_it_is_empty() -> None:
    assert MutantQueue([]).take() is None
    assert list(MutantQueue([]).each()) == []


def test_the_queue_iterates_every_mutant() -> None:
    assert [m.line for m in MutantQueue(_numbered(4)).each()] == [0, 1, 2, 3]


def _drained(queue: MutantQueue, into: list[int]) -> None:
    """Take mutants from the queue until it is empty, and keep the line of each."""
    into.extend(m.line for m in queue.each())


def _joined(groups: list[list[int]]) -> list[int]:
    """Join the lines that each thread took into one list.

    Args:
        groups: The lines of each thread.

    Returns:
        Every line taken, in no fixed order.
    """
    found: list[int] = []
    for group in groups:
        found.extend(group)
    return found


def _awaited(threads: list[threading.Thread]) -> None:
    """Start all the threads, and then wait for each one to end."""
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _taken_by(queue: MutantQueue, takers: int) -> list[int]:
    """Empty one queue from several threads at the same time.

    Args:
        queue: The queue.
        takers: The number of threads.

    Returns:
        Every line that a thread took.
    """
    groups: list[list[int]] = [[] for _ in range(takers)]
    _awaited([threading.Thread(target=_drained, args=(queue, g)) for g in groups])
    return _joined(groups)


def test_the_queue_hands_each_mutant_to_one_taker_only() -> None:
    assert sorted(_taken_by(MutantQueue(_numbered(200)), 4)) == list(range(200))


def test_it_answers_no_tests_without_a_map() -> None:
    assert Sweep(None).tests(Mutant(Path("a.py"), "core/AddNot", 0, 1)) == ()


def test_it_reads_the_tests_out_of_a_map() -> None:
    found = Sweep(CoverageMap()).tests(Mutant(Path("a.py"), "core/AddNot", 0, 1))
    assert found == ()


def test_progress_counts_each_finished_mutant() -> None:
    counter = Progress(None, 4)
    counter.step()
    counter.step()
    assert counter.finished == 2


def test_progress_writes_at_each_step() -> None:
    out = io.StringIO()
    counter = Progress(out, Progress.STEP)
    for _ in range(Progress.STEP):
        counter.step()
    assert out.getvalue().strip() == f"{Progress.STEP}/{Progress.STEP}"


def test_progress_stays_quiet_between_steps() -> None:
    out = io.StringIO()
    Progress(out, 100).step()
    assert not out.getvalue()


def test_it_runs_a_whole_scope_over_real_copies(toy: Path, tmp_path: Path) -> None:
    sweep = Sweep(None, package="toy")
    mutants = sweep.mutants(toy / "src")
    assert mutants != []
    results = sweep.execute(mutants[:2], (toy, tmp_path / "home"), 2)
    assert len(results) == 2
    assert all(r.outcome is Outcome.UNSELECTED for r in results)


def _held_store(tmp_path: Path, mutant: Mutant) -> ResultStore:
    """Build a store that already holds one finished mutant.

    Args:
        tmp_path: The directory for the store file.
        mutant: The mutant to store as caught.

    Returns:
        The store.
    """
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(MutantResult(mutant, Outcome.KILLED, "t.py::a", 0.5))
    return store


def test_it_treats_a_mutant_with_no_store_as_unfinished() -> None:
    assert Sweep(None).finished(Mutant(Path("a.py"), "core/AddNot", 0, 1)) is None


def test_it_skips_a_mutant_an_earlier_run_finished(tmp_path: Path) -> None:
    mutant = Mutant(Path("a.py"), "core/AddNot", 0, 1)
    sweep = Sweep(None, store=_held_store(tmp_path, mutant))
    assert sweep.pending([mutant]) == []
    assert [r.outcome for r in sweep.held([mutant])] == [Outcome.KILLED]


def test_it_runs_nothing_when_every_mutant_is_finished(tmp_path: Path) -> None:
    mutant = Mutant(Path("a.py"), "core/AddNot", 0, 1)
    sweep = Sweep(None, store=_held_store(tmp_path, mutant))
    results = sweep.execute([mutant], (tmp_path, tmp_path / "home"), 2)
    assert [r.outcome for r in results] == [Outcome.KILLED]
    assert not (tmp_path / "home").exists()


def test_it_writes_each_finished_mutant_to_the_store(toy: Path, tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    sweep = Sweep(None, package="toy", store=store)
    mutants = sweep.mutants(toy / "src")[:2]
    sweep.execute(mutants, (toy, tmp_path / "home"), 1)
    assert len(store.results()) == 2
