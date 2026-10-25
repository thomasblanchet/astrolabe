"""Check the report of a mutation run: the score of each module and the list of survivors.

A survivor is a mutant that no test caught.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from tools.mutation.mutants import Mutant
from tools.mutation.report import MutationReport, Score
from tools.mutation.runner import MutantResult, Outcome


def _result(name: str, line: int, outcome: Outcome, killer: str | None = None) -> MutantResult:
    """Build the result of one mutant in a given file.

    Args:
        name: The path of the file.
        line: The changed line.
        outcome: What the run found, for example that a test caught it.
        killer: The test that caught it, if any.

    Returns:
        The result.
    """
    return MutantResult(Mutant(Path(name), "core/AddNot", 0, line), outcome, killer, 1.0)


def _written(results: list[MutantResult]) -> str:
    """Write the report of some results to memory, and give its text.

    Args:
        results: The results.

    Returns:
        The text of the report.
    """
    out = io.StringIO()
    MutationReport(results).write(out)
    return out.getvalue()


def test_a_score_reads_the_share_that_died() -> None:
    assert Score(3, 1).percent() == pytest.approx(75.0)
    assert Score(3, 1).total() == 4


def test_a_score_of_nothing_reads_as_whole() -> None:
    assert Score(0, 0).percent() == pytest.approx(100.0)


def test_it_counts_every_outcome() -> None:
    results = [
        _result("a.py", 1, Outcome.KILLED, "t::a"),
        _result("a.py", 2, Outcome.SURVIVED),
        _result("a.py", 3, Outcome.UNSELECTED),
        _result("a.py", 4, Outcome.UNCHANGED),
        _result("a.py", 5, Outcome.UNREACHED),
        _result("a.py", 6, Outcome.HUNG),
    ]
    assert MutationReport(results).counts() == {
        "killed": 1,
        "survived": 1,
        "timed out": 0,
        "hung": 1,
        "unselected": 1,
        "unreached": 1,
        "unchanged": 1,
        "errored": 0,
    }


def test_it_scores_only_the_mutants_that_measured_something() -> None:
    results = [
        _result("a.py", 1, Outcome.KILLED, "t::a"),
        _result("a.py", 2, Outcome.SURVIVED),
        _result("a.py", 3, Outcome.UNSELECTED),
    ]
    assert MutationReport(results).score() == Score(1, 1)


def test_it_lists_the_survivors_in_file_order() -> None:
    results = [
        _result("b.py", 9, Outcome.SURVIVED),
        _result("a.py", 2, Outcome.SURVIVED),
        _result("a.py", 1, Outcome.KILLED, "t::a"),
    ]
    found = MutationReport(results).survivors()
    labels = [r.mutant.label() for r in found if r.mutant is not None]
    assert labels == ["a.py:2 AddNot", "b.py:9 AddNot"]


def test_it_scores_each_file_worst_first() -> None:
    results = [
        _result("good.py", 1, Outcome.KILLED, "t::a"),
        _result("bad.py", 1, Outcome.SURVIVED),
    ]
    assert [name for name, _score in MutationReport(results).by_module()] == ["bad.py", "good.py"]


def test_it_writes_the_score_and_the_survivors() -> None:
    text = _written([_result("a.py", 2, Outcome.SURVIVED)])
    assert "mutation score 0.0%" in text
    assert "survived  a.py:2 AddNot" in text


def test_it_writes_an_empty_run() -> None:
    assert "mutation score 100.0%" in _written([])


def test_it_reads_a_result_that_carries_no_mutant() -> None:
    results = [MutantResult(None, Outcome.SURVIVED, None, 0.0)]
    assert "survived  -" in _written(results)
    assert [name for name, _score in MutationReport(results).by_module()] == ["-"]
