"""Check how a failed step in a chain of steps carries the reason why it failed.

Plugin code writes a chain of steps that can each fail as one flat
method. A step such as `not_none(value)` stops the whole method when
its value is missing. The decorator `propagates` then gives a default
value. The decorator `explains` gives a `Refusal` that holds the reason
why the chain gave up, so that the hook can report that reason.
`not_refused` passes the reason of an inner chain through an outer one.

The doctests of `propagates` cover that decorator. These tests cover
`explains`, `not_refused`, and the reason that a failed step carries
through each decorator.
"""

from __future__ import annotations

from astrolabe._hooks.propagate import (
    Refusal,
    explains,
    not_falsy,
    not_none,
    not_refused,
    propagates,
)
from astrolabe._hooks.unproven import Reason


@explains(Reason.CHECKER_GAP)
def _frame_names(frame: str | None, names: list[str] | None) -> list[str]:
    not_none(frame)
    return not_falsy(names, reason=Reason.DYNAMIC_NAME)


@explains(Reason.REPORTED)
def _outer(frame: str | None, names: list[str] | None) -> int:
    return len(not_refused(_frame_names(frame, names)))


@propagates(None)
def _swallowed(names: list[str] | None) -> list[str] | None:
    return not_none(names, reason=Reason.DYNAMIC_NAME)


def test_explains_returns_the_value_when_every_step_passes() -> None:
    assert _frame_names("trades", ["a"]) == ["a"]


def test_explains_gives_the_default_reason_for_an_unnamed_step() -> None:
    assert _frame_names(None, ["a"]) == Refusal(Reason.CHECKER_GAP)


def test_explains_gives_the_reason_the_failing_step_names() -> None:
    assert _frame_names("trades", None) == Refusal(Reason.DYNAMIC_NAME)


def test_not_falsy_carries_its_reason_for_an_empty_value() -> None:
    assert _frame_names("trades", []) == Refusal(Reason.DYNAMIC_NAME)


def test_not_refused_passes_an_inner_reason_through_an_outer_boundary() -> None:
    assert _outer("trades", None) == Refusal(Reason.DYNAMIC_NAME)
    assert _outer(None, ["a"]) == Refusal(Reason.CHECKER_GAP)


def test_not_refused_gives_the_value_of_an_inner_success() -> None:
    assert _outer("trades", ["a", "b"]) == 2


def test_propagates_drops_the_reason_of_a_failed_step() -> None:
    assert _swallowed(None) is None
    assert _swallowed(["a"]) == ["a"]
