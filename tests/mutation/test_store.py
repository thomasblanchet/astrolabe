"""Check the store that keeps each finished mutant, so that a stopped run can go on.

A full audit runs for hours. If it stops, a new run reads the store
and runs only the mutants that are left. The store also stops a run
when the source changed since the stored results, because a stored
result then names a different change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.mutation.mutants import Mutant
from tools.mutation.runner import MutantResult, Outcome
from tools.mutation.store import ResultStore, SourceDigests, StaleStore, not_none_mutant


def _mutant(occurrence: int = 0) -> Mutant:
    """Build one mutant of a fixed file and operator.

    Args:
        occurrence: Which place in the file the operator changes.

    Returns:
        The mutant.
    """
    return Mutant(Path("src/a.py"), "core/AddNot", occurrence, 7)


def _result(occurrence: int = 0, killer: str | None = "t.py::a") -> MutantResult:
    """Build the result of one finished mutant.

    Args:
        occurrence: Which place in the file the operator changes.
        killer: The test that caught the mutant, or None if it survived.

    Returns:
        The result.
    """
    outcome = Outcome.KILLED if killer else Outcome.SURVIVED
    return MutantResult(_mutant(occurrence), outcome, killer, 1.25)


def test_it_names_a_mutant_the_same_way_every_run() -> None:
    assert ResultStore(Path("r")).key(_mutant(2)) == "src/a.py|core/AddNot|2"


def test_it_reads_back_what_it_wrote(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(_result())
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True)
    found = later.finished(_mutant())
    assert found is not None
    assert found.outcome is Outcome.KILLED
    assert found.killer == "t.py::a"
    assert found.seconds == pytest.approx(1.25)


def test_it_reads_back_a_survivor_with_no_killer(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(_result(killer=None))
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True)
    found = later.finished(_mutant())
    assert found is not None
    assert found.outcome is Outcome.SURVIVED
    assert found.killer is None


def test_it_starts_the_file_again_without_resume(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(_result())
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    assert later.finished(_mutant()) is None
    assert later.results() == []


def test_it_starts_the_file_when_resume_finds_nothing(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "deep" / "r.jsonl").begin(resume=True)
    assert store.results() == []
    assert (tmp_path / "deep" / "r.jsonl").exists()


def test_it_holds_every_result_it_read(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(_result(0))
    store.record(_result(1))
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True)
    assert len(later.results()) == 2


def test_it_skips_a_blank_line(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    assert ResultStore(path).begin(resume=True).results() == []


def test_it_writes_nothing_for_a_result_with_no_mutant(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(MutantResult(None, Outcome.KILLED, None, 0.0))
    assert store.results() == []


def test_it_refuses_to_read_a_result_with_no_mutant() -> None:
    with pytest.raises(TypeError, match="no result without a mutant"):
        not_none_mutant(MutantResult(None, Outcome.KILLED, None, 0.0))


def test_a_retry_run_reads_an_undecided_mutant_as_unfinished(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(MutantResult(_mutant(), Outcome.TIMED_OUT, None, 180.0))
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True, retry=True)
    assert later.finished(_mutant()) is None


def test_a_plain_resume_keeps_an_undecided_mutant(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(MutantResult(_mutant(), Outcome.TIMED_OUT, None, 180.0))
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True)
    found = later.finished(_mutant())
    assert found is not None
    assert found.outcome is Outcome.TIMED_OUT


def test_a_retry_run_keeps_every_decided_mutant(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(_result(0))
    store.record(MutantResult(_mutant(1), Outcome.TIMED_OUT, None, 180.0))
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True, retry=True)
    assert later.finished(_mutant(0)) is not None
    assert later.finished(_mutant(1)) is None


def test_a_retry_run_keeps_a_mutant_that_ran_out_as_long_a_limit(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(MutantResult(_mutant(), Outcome.TIMED_OUT, None, 180.0))
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True, retry=True, limit=180.0)
    assert later.finished(_mutant()) is not None


def test_a_retry_run_under_a_longer_limit_runs_the_mutant_again(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(MutantResult(_mutant(), Outcome.TIMED_OUT, None, 180.0))
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True, retry=True, limit=600.0)
    assert later.finished(_mutant()) is None


def test_a_retry_run_keeps_a_hung_mutant(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(MutantResult(_mutant(), Outcome.HUNG, None, 180.0))
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True, retry=True, limit=600.0)
    assert later.finished(_mutant()) is not None


def test_a_later_line_replaces_an_earlier_verdict(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(MutantResult(_mutant(), Outcome.TIMED_OUT, None, 180.0))
    store.record(_result())
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True)
    found = later.finished(_mutant())
    assert found is not None
    assert found.outcome is Outcome.KILLED


def test_a_digest_names_a_file_by_its_bytes(tmp_path: Path) -> None:
    one = tmp_path / "a.py"
    one.write_text("x = 1\n", encoding="utf-8")
    first = SourceDigests().digest(one)
    one.write_text("x = 2\n", encoding="utf-8")
    assert SourceDigests().digest(one) != first


def test_digests_mark_every_file_of_a_scope(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    marks = SourceDigests().of([tmp_path / "a.py", tmp_path / "b.py"])
    assert sorted(marks) == [str(tmp_path / "a.py"), str(tmp_path / "b.py")]


def test_an_unchanged_file_is_not_reported() -> None:
    assert SourceDigests().changed({"a.py": "x"}, {"a.py": "x"}) == []


def test_a_changed_file_is_reported() -> None:
    assert SourceDigests().changed({"a.py": "x"}, {"a.py": "y"}) == ["a.py"]


def test_a_file_the_earlier_run_did_not_hold_is_not_reported() -> None:
    assert SourceDigests().changed({}, {"new.py": "x"}) == []


def test_a_resume_over_changed_source_stops(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False, digests={"a.py": "x"})
    store.record(_result())
    with pytest.raises(StaleStore, match=r"a\.py"):
        ResultStore(tmp_path / "r.jsonl").begin(resume=True, digests={"a.py": "y"})


def test_a_resume_over_the_same_source_goes_on(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False, digests={"a.py": "x"})
    store.record(_result())
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True, digests={"a.py": "x"})
    assert later.finished(_mutant()) is not None


def test_a_resume_without_marks_goes_on(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(_result())
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True, digests={"a.py": "x"})
    assert later.finished(_mutant()) is not None


def test_a_resume_that_names_no_marks_skips_the_check(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False, digests={"a.py": "x"})
    store.record(_result())
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True)
    assert later.finished(_mutant()) is not None


def test_the_marks_sit_beside_the_results(tmp_path: Path) -> None:
    ResultStore(tmp_path / "r.jsonl").begin(resume=False, digests={"a.py": "x"})
    assert (tmp_path / "r-digests.json").exists()


def test_a_run_after_new_cases_decides_each_survivor_again(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(_result(0, killer=None))
    store.record(MutantResult(_mutant(1), Outcome.UNREACHED, None, 0.0))
    store.record(MutantResult(_mutant(2), Outcome.UNSELECTED, None, 0.0))
    store.record(_result(3))
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True, again=True)
    assert [later.finished(_mutant(n)) is None for n in range(4)] == [True, True, True, False]


def test_a_plain_resume_keeps_each_survivor(tmp_path: Path) -> None:
    store = ResultStore(tmp_path / "r.jsonl").begin(resume=False)
    store.record(_result(0, killer=None))
    later = ResultStore(tmp_path / "r.jsonl").begin(resume=True)
    assert later.finished(_mutant(0)) is not None
