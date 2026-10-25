"""Run the code of each `.yml` case with real polars, and check the types that it states.

A `.yml` case is only checked with mypy, and never run. This test runs
the code of each case on real sample data, and compares each real
result with the type that the case states. The classes that do the
work are in `tests/runtime_cross_check.py`, see `RuntimeCrossCheck`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.runtime_cross_check import (
    CaseSource,
    DTypeSampleFactory,
    RevealedTypeReference,
    RuntimeCrossCheck,
    SampleFrameBuilder,
    SchemaExpectation,
    YamlCorpus,
)

_MINIMUM_EXPECTED_CHECKS = 500

_CASES = list(YamlCorpus(Path(__file__).parent / "typing").cases())
_CROSS_CHECK = RuntimeCrossCheck(
    SampleFrameBuilder(DTypeSampleFactory()),
    SchemaExpectation(),
    RevealedTypeReference(),
)


def _case_id(case: CaseSource) -> str:
    return f"{case.file.stem}::{case.case}"


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_case_runtime_matches_declared_type(case: CaseSource) -> None:
    results = _CROSS_CHECK.checked(case.main)
    mismatches = [result for result in results if not result.matched]
    assert not mismatches, "\n".join(
        f"{_case_id(case)}: {result.label}: expected {result.expected!r}, got {result.actual!r}"
        for result in mismatches
    )


def test_cross_check_covers_a_meaningful_share_of_the_corpus() -> None:
    total_checks = sum(len(_CROSS_CHECK.checked(case.main)) for case in _CASES)
    assert total_checks >= _MINIMUM_EXPECTED_CHECKS
