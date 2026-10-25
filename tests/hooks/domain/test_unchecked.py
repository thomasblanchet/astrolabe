"""Check `UncheckedFallback`, which keeps a column under the data type `Unchecked`.

When a hook knows the name of a result column and cannot prove its
data type, it keeps the column with the data type `Unchecked`. The
`allow_unproven` setting decides whether the checker also reports an
error. `UncheckedFallback` makes that choice and reports through the
context that mypy gives the hook. It reads only the API, to report,
and the node to report at, so these tests give it a plain object with
those parts, and need no mypy build.

The doctests of `UncheckedCompatibility` cover each of its branches,
so that class has no test module of its own.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from mypy.types import AnyType, Instance, TypeOfAny, get_proper_type

from astrolabe._hooks.domain.unchecked import UncheckedFallback
from astrolabe._hooks.unproven import Level, Reason, UnprovenPolicy

if TYPE_CHECKING:
    from mypy.errorcodes import ErrorCode
    from mypy.plugin import MethodContext

    from tests.hooks.conftest import FakePlugin


class _FakeReportApi:
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


def make_ctx() -> tuple[MethodContext, _FakeReportApi]:
    """Build a stand-in for the context of a method hook, with an API that keeps reports.

    Returns:
        The context, and the API inside it, which the test reads.
    """
    api = _FakeReportApi()
    default_return_type = AnyType(TypeOfAny.explicit)
    ctx = SimpleNamespace(api=api, context=object(), default_return_type=default_return_type)
    return cast("MethodContext", ctx), api


def test_substituted_reports_a_reason_over_the_level(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    fallback = UncheckedFallback(UnprovenPolicy(Level.STATED), plugin)
    fallback.substituted(ctx, Reason.BARE_INT, "select", "x")
    assert api.codes == ["implicitly-unchecked"]
    assert api.messages == [
        'Column "x" of "select" is Unchecked: a number variable has no fixed width',
    ]


def test_substituted_is_silent_for_a_reason_at_the_level(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    fallback = UncheckedFallback(UnprovenPolicy(Level.OMITTED), plugin)
    fallback.substituted(ctx, Reason.BARE_INT, "select", "x")
    assert api.codes == []


def test_substituted_honors_an_override_that_silences_an_operation(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    fallback = UncheckedFallback(UnprovenPolicy(Level.STATED, {"select": Level.OMITTED}), plugin)
    fallback.substituted(ctx, Reason.BARE_INT, "select", "x")
    assert api.codes == []


def test_substituted_honors_an_override_that_flags_an_operation(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    fallback = UncheckedFallback(UnprovenPolicy(Level.OMITTED, {"select": Level.STATED}), plugin)
    fallback.substituted(ctx, Reason.BARE_INT, "select", "x")
    assert api.codes == ["implicitly-unchecked"]


def test_substituted_override_does_not_affect_other_operations(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    fallback = UncheckedFallback(UnprovenPolicy(Level.STATED, {"select": Level.OMITTED}), plugin)
    fallback.substituted(ctx, Reason.BARE_INT, "agg", "x")
    assert api.codes == ["implicitly-unchecked"]


def test_substituted_returns_a_minted_unchecked_when_reported(plugin: FakePlugin) -> None:
    ctx, _api = make_ctx()
    fallback = UncheckedFallback(UnprovenPolicy(Level.NEVER), plugin)
    result = fallback.substituted(ctx, Reason.CHECKER_GAP, "select", "x")
    proper = get_proper_type(result)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "astrolabe.frame.Unchecked"


def test_substituted_returns_a_minted_unchecked_when_allowed(plugin: FakePlugin) -> None:
    ctx, _api = make_ctx()
    fallback = UncheckedFallback(UnprovenPolicy(), plugin)
    result = fallback.substituted(ctx, Reason.CHECKER_GAP, "select", "x")
    proper = get_proper_type(result)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "astrolabe.frame.Unchecked"


def test_reported_names_the_omitted_keyword_of_an_opaque_member(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    fallback = UncheckedFallback(UnprovenPolicy(Level.STATED), plugin)
    fallback.reported(ctx, Reason.OPAQUE_KEYWORD_OMITTED, "map_elements", "x")
    assert api.codes == ["implicitly-unchecked"]
    assert "no return_dtype= names it" in api.messages[0]
    assert api.notes == [
        'set allow_unproven = "omitted" in [tool.astrolabe] to allow this,'
        ' or set [tool.astrolabe.allow_unproven_overrides] map_elements = "omitted"'
    ]


def test_reported_lets_an_opaque_member_with_no_keyword_pass_at_stated(
    plugin: FakePlugin,
) -> None:
    ctx, api = make_ctx()
    fallback = UncheckedFallback(UnprovenPolicy(Level.STATED), plugin)
    fallback.reported(ctx, Reason.OPAQUE_NO_KEYWORD, "rolling_map", "x")
    assert api.codes == []
