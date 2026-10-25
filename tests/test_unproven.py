"""Check the levels of `allow_unproven`, the reasons, and the policy that compares them.

When the checker gives up on a proof, it names a `Reason`. Each reason
has a `Level`, and the `allow_unproven` setting lets each reason at or
below its level pass. The table in this module fixes the level of each
reason. A change to that table changes what the checker reports, so it
is a change of design and not a cleanup.
"""

from __future__ import annotations

import json

import pytest

from astrolabe._hooks.unproven import Level, Reason, UnprovenPolicy

_CENSUS_LEVELS: dict[Reason, Level | None] = {
    Reason.DYNAMIC_NAME: Level.STATED,
    Reason.DYNAMIC_OPTION: Level.STATED,
    Reason.PLAIN_EXPRESSION: Level.STATED,
    Reason.NO_SCHEMA: Level.STATED,
    Reason.PARTIAL_PICK: Level.STATED,
    Reason.OPAQUE_NO_KEYWORD: Level.STATED,
    Reason.OPAQUE_KEYWORD_OMITTED: Level.OMITTED,
    Reason.SERIES_NO_DTYPE: Level.OMITTED,
    Reason.BARE_INT: Level.OMITTED,
    Reason.NON_LITERAL_DTYPE: Level.OMITTED,
    Reason.REMAINDER_MAY_HOLD: Level.POSSIBLE,
    Reason.CALLER_DEPENDENT: Level.POSSIBLE,
    Reason.CALLER_KEY: Level.POSSIBLE,
    Reason.UNPROVEN_REQUIREMENT: Level.ASSERTED,
    Reason.UNTYPED_ARGUMENT: Level.ASSERTED,
    Reason.CHECKER_GAP: None,
    Reason.REPORTED: None,
}

_ON_SCALE = [reason for reason, level in _CENSUS_LEVELS.items() if level is not None]


def test_every_reason_sits_at_its_census_level() -> None:
    assert {reason: reason.level for reason in Reason} == _CENSUS_LEVELS


def test_every_reason_has_its_own_wording() -> None:
    wordings = [reason.wording for reason in Reason]
    assert len(set(wordings)) == len(wordings)


def test_levels_are_ordered_from_never_to_asserted() -> None:
    assert list(Level) == sorted(Level)
    assert [level.spelling for level in Level] == [
        "never",
        "none",
        "stated",
        "omitted",
        "possible",
        "asserted",
    ]


@pytest.mark.parametrize("level", list(Level))
def test_named_reads_every_spelling_back(level: Level) -> None:
    assert Level.named(level.spelling) is level


@pytest.mark.parametrize("name", ["NONE", "Omitted", "", "lenient", "strict"])
def test_named_refuses_anything_but_an_exact_spelling(name: str) -> None:
    assert Level.named(name) is None


@pytest.mark.parametrize("level", list(Level))
@pytest.mark.parametrize("reason", _ON_SCALE)
def test_a_reason_on_the_scale_passes_at_its_level_and_above(reason: Reason, level: Level) -> None:
    policy = UnprovenPolicy(level)
    assert policy.allows(reason, "select") is (reason.level is not None and reason.level <= level)


@pytest.mark.parametrize("reason", _ON_SCALE)
def test_never_and_none_refuse_the_same_reasons_on_the_scale(reason: Reason) -> None:
    assert not UnprovenPolicy(Level.NEVER).allows(reason, "select")
    assert not UnprovenPolicy(Level.NONE).allows(reason, "select")


def test_never_reports_checker_gaps_and_none_does_not() -> None:
    assert not UnprovenPolicy(Level.NEVER).allows(Reason.CHECKER_GAP, "select")
    assert UnprovenPolicy(Level.NONE).allows(Reason.CHECKER_GAP, "select")


def test_report_gaps_reports_checker_gaps_at_a_high_level() -> None:
    policy = UnprovenPolicy(Level.ASSERTED, report_gaps=True)
    assert not policy.allows(Reason.CHECKER_GAP, "select")
    assert policy.allows(Reason.UNTYPED_ARGUMENT, "select")


def test_an_override_to_never_reports_gaps_for_that_operation_only() -> None:
    policy = UnprovenPolicy(Level.OMITTED, {"join": Level.NEVER})
    assert not policy.allows(Reason.CHECKER_GAP, "join")
    assert policy.allows(Reason.CHECKER_GAP, "select")


@pytest.mark.parametrize("level", list(Level))
@pytest.mark.parametrize("report_gaps", [False, True])
def test_a_reason_already_reported_always_passes(level: Level, report_gaps: bool) -> None:
    assert UnprovenPolicy(level, report_gaps=report_gaps).allows(Reason.REPORTED, "select")


def test_an_override_lowers_one_operation_under_the_global_level() -> None:
    policy = UnprovenPolicy(Level.ASSERTED, {"select": Level.NONE})
    assert not policy.allows(Reason.DYNAMIC_NAME, "select")
    assert policy.allows(Reason.DYNAMIC_NAME, "drop")


def test_an_override_raises_one_operation_over_the_global_level() -> None:
    policy = UnprovenPolicy(Level.NONE, {"map_elements": Level.OMITTED})
    assert policy.allows(Reason.OPAQUE_KEYWORD_OMITTED, "map_elements")
    assert not policy.allows(Reason.OPAQUE_KEYWORD_OMITTED, "map_batches")


def test_the_default_policy_is_omitted_with_gaps_off() -> None:
    policy = UnprovenPolicy()
    assert policy.level is Level.OMITTED
    assert policy.overrides == {}
    assert policy.report_gaps is False


def test_the_allowance_names_the_global_setting_and_the_override() -> None:
    allowance = UnprovenPolicy(Level.NONE).allowance(Reason.DYNAMIC_NAME, "select")
    assert allowance == (
        'set allow_unproven = "stated" in [tool.astrolabe] to allow this,'
        ' or set [tool.astrolabe.allow_unproven_overrides] select = "stated"'
    )


def test_the_allowance_names_only_the_override_where_one_exists() -> None:
    policy = UnprovenPolicy(Level.ASSERTED, {"select": Level.NONE})
    expected = 'set [tool.astrolabe.allow_unproven_overrides] select = "possible" to allow this'
    assert policy.allowance(Reason.REMAINDER_MAY_HOLD, "select") == expected


def test_the_gap_allowance_names_the_gap_switch_where_it_is_on() -> None:
    policy = UnprovenPolicy(Level.NEVER, report_gaps=True)
    expected = "set report_checker_gaps = false in [tool.astrolabe] to allow this"
    assert policy.allowance(Reason.CHECKER_GAP, "select") == expected


def test_the_gap_allowance_names_the_none_level_under_never() -> None:
    assert UnprovenPolicy(Level.NEVER).allowance(Reason.CHECKER_GAP, "select") == (
        'set allow_unproven = "none" in [tool.astrolabe] to allow this,'
        ' or set [tool.astrolabe.allow_unproven_overrides] select = "none"'
    )


def test_the_cache_key_is_json_encodable_and_independent_of_override_order() -> None:
    first = UnprovenPolicy(Level.STATED, {"select": Level.NONE, "join": Level.NEVER}, True)
    second = UnprovenPolicy(Level.STATED, {"join": Level.NEVER, "select": Level.NONE}, True)
    assert json.dumps(first.cache_key()) == json.dumps(second.cache_key())
    assert first.cache_key() == {
        "level": "stated",
        "overrides": {"join": "never", "select": "none"},
        "report_gaps": True,
    }


def test_the_roster_holds_the_fit_key_and_no_retired_tag() -> None:
    assert "fit" in UnprovenPolicy.KNOWN_OPERATIONS
    retired = {"boolean_op", "binary_op", "comparison", "to_array", "struct"}
    assert not retired & UnprovenPolicy.KNOWN_OPERATIONS
