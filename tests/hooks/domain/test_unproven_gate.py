"""Check `UnprovenGate`, which reports an unproven result unless the setting lets it pass.

A hook that gives up on a proof names the reason, and the gate
compares the level of that reason with the `allow_unproven` setting.
If the reason does not pass, the gate reports an error and adds a
note that names the setting which lets it pass. The gate reads only
the API, to report, and the node to report at, so these tests give it
a plain object with those parts, and need no mypy build.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from astrolabe._hooks.domain.unproven_gate import UnprovenGate
from astrolabe._hooks.errors import ImplicitlyUncheckedError, ImplicitlyUntypedError
from astrolabe._hooks.unproven import Level, Reason, UnprovenPolicy

if TYPE_CHECKING:
    from mypy.errorcodes import ErrorCode
    from mypy.plugin import MethodContext


class _RecordingApi:
    """Stand in for the checker API, and keep each error and each note, to check later."""

    def __init__(self) -> None:
        self.codes: list[str] = []
        self.messages: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str, ctx: object, *, code: ErrorCode | None = None) -> None:
        self.messages.append(msg)
        if code is not None:
            self.codes.append(code.code)

    def note(self, msg: str, ctx: object, *, code: ErrorCode | None = None) -> None:
        self.notes.append(msg)


def _ctx() -> tuple[MethodContext, _RecordingApi]:
    api = _RecordingApi()
    return cast("MethodContext", SimpleNamespace(api=api, context=object())), api


def test_report_follows_the_policy() -> None:
    ctx, api = _ctx()
    gate = UnprovenGate(UnprovenPolicy(Level.STATED))
    assert gate.report(ctx, ImplicitlyUntypedError(Reason.DYNAMIC_NAME, "select")) is None
    assert api.codes == []
    gate.report(ctx, ImplicitlyUncheckedError(Reason.BARE_INT, "select", "x"))
    assert api.codes == ["implicitly-unchecked"]


def test_report_is_silent_for_an_allowed_reason() -> None:
    ctx, api = _ctx()
    error = ImplicitlyUncheckedError(Reason.BARE_INT, "select", "x")
    UnprovenGate(UnprovenPolicy()).report(ctx, error)
    assert api.codes == []
    assert api.notes == []


def test_report_fails_with_the_reason_and_adds_the_hint_note() -> None:
    ctx, api = _ctx()
    gate = UnprovenGate(UnprovenPolicy(Level.NONE))
    gate.report(ctx, ImplicitlyUntypedError(Reason.DYNAMIC_NAME, "select"))
    assert api.codes == ["implicitly-untyped"]
    assert api.messages == [
        'The result of "select" has no known schema: a column name is not a literal string',
    ]
    assert api.notes == [
        'set allow_unproven = "stated" in [tool.astrolabe] to allow this,'
        ' or set [tool.astrolabe.allow_unproven_overrides] select = "stated"'
    ]


def test_report_names_only_the_override_where_the_operation_has_one() -> None:
    ctx, api = _ctx()
    gate = UnprovenGate(UnprovenPolicy(Level.ASSERTED, {"select": Level.NONE}))
    gate.report(ctx, ImplicitlyUncheckedError(Reason.BARE_INT, "select", "x"))
    assert api.codes == ["implicitly-unchecked"]
    assert api.notes == [
        'set [tool.astrolabe.allow_unproven_overrides] select = "omitted" to allow this',
    ]


def test_report_is_silent_for_a_reason_already_reported() -> None:
    ctx, api = _ctx()
    gate = UnprovenGate(UnprovenPolicy(Level.NEVER, report_gaps=True))
    gate.report(ctx, ImplicitlyUntypedError(Reason.REPORTED, "select"))
    assert api.codes == []
