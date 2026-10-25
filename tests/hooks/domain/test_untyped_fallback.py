"""Check `UntypedFallback`, which gives the plain frame type when a hook cannot type a result.

When a frame hook cannot prove the columns of its result, it falls
back: it gives the plain polars frame of the same kind, which the
checker does not track. The `allow_unproven` setting decides whether
the checker also reports an error. `UntypedFallback` reads only the
API, the node to report at, and the return type that mypy gives when
no plugin acts. These tests give it a plain object with those parts,
and need no mypy build.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from mypy.types import AnyType, Instance, TypeOfAny, get_proper_type

from astrolabe._hooks.domain.untyped_fallback import UntypedFallback
from astrolabe._hooks.unproven import Level, Reason, UnprovenPolicy
from tests.hooks.conftest import make_column_ref, make_frame

if TYPE_CHECKING:
    from mypy.errorcodes import ErrorCode
    from mypy.plugin import MethodContext
    from mypy.test.typefixture import TypeFixture

    from tests.hooks.conftest import FakePlugin


class _FakeReportApi:
    """Stand in for the checker API, and keep the code of each error and each note."""

    def __init__(self) -> None:
        self.codes: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str, ctx: object, *, code: ErrorCode | None = None) -> None:
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


def test_report_if_disallowed_reports_a_reason_over_the_level(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    resolver = UntypedFallback(UnprovenPolicy(Level.NONE), plugin)
    resolver.report_if_disallowed(ctx, Reason.DYNAMIC_NAME, "select")
    assert api.codes == ["implicitly-untyped"]
    assert len(api.notes) == 1


def test_report_if_disallowed_is_silent_for_a_reason_at_the_level(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    resolver = UntypedFallback(UnprovenPolicy(Level.STATED), plugin)
    resolver.report_if_disallowed(ctx, Reason.DYNAMIC_NAME, "select")
    assert api.codes == []
    assert api.notes == []


def test_report_if_disallowed_honors_an_override_that_silences_an_operation(
    plugin: FakePlugin,
) -> None:
    ctx, api = make_ctx()
    resolver = UntypedFallback(UnprovenPolicy(Level.NONE, {"select": Level.STATED}), plugin)
    resolver.report_if_disallowed(ctx, Reason.DYNAMIC_NAME, "select")
    assert api.codes == []


def test_report_if_disallowed_override_does_not_affect_other_operations(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    resolver = UntypedFallback(UnprovenPolicy(Level.NONE, {"select": Level.STATED}), plugin)
    resolver.report_if_disallowed(ctx, Reason.DYNAMIC_NAME, "join")
    assert api.codes == ["implicitly-untyped"]


def test_resolve_reports_a_gap_under_never_and_falls_back_to_default(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    resolver = UntypedFallback(UnprovenPolicy(Level.NEVER), plugin)
    result = resolver.resolve(ctx, Reason.CHECKER_GAP, "select", None)
    assert result is ctx.default_return_type
    assert api.codes == ["implicitly-untyped"]


def test_resolve_falls_back_to_default_silently_by_default(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    resolver = UntypedFallback(UnprovenPolicy(), plugin)
    result = resolver.resolve(ctx, Reason.CHECKER_GAP, "select", None)
    assert result is ctx.default_return_type
    assert api.codes == []


def test_resolve_is_silent_for_a_reason_already_reported(plugin: FakePlugin) -> None:
    ctx, api = make_ctx()
    resolver = UntypedFallback(UnprovenPolicy(Level.NEVER, report_gaps=True), plugin)
    resolver.resolve(ctx, Reason.REPORTED, "select", None)
    assert api.codes == []


def test_resolve_falls_back_to_the_plain_type_and_reports(
    fixture: TypeFixture,
    int_dtype: Instance,
    plugin: FakePlugin,
) -> None:
    ctx, api = make_ctx()
    concrete = make_frame(fixture, [make_column_ref(fixture, "price", int_dtype).as_instance()])
    resolver = UntypedFallback(UnprovenPolicy(Level.NONE), plugin)
    result = resolver.resolve(ctx, Reason.NO_SCHEMA, "select", concrete)
    proper = get_proper_type(result)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "polars.dataframe.frame.DataFrame"
    assert api.codes == ["implicitly-untyped"]


def test_resolve_falls_back_to_the_plain_type_silently_when_allowed(
    fixture: TypeFixture,
    int_dtype: Instance,
    plugin: FakePlugin,
) -> None:
    ctx, api = make_ctx()
    concrete = make_frame(fixture, [make_column_ref(fixture, "price", int_dtype).as_instance()])
    resolver = UntypedFallback(UnprovenPolicy(), plugin)
    result = resolver.resolve(ctx, Reason.NO_SCHEMA, "select", concrete)
    proper = get_proper_type(result)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "polars.dataframe.frame.DataFrame"
    assert api.codes == []


def test_resolve_picks_the_first_resolved_frame_to_fall_back_from(
    fixture: TypeFixture,
    int_dtype: Instance,
    plugin: FakePlugin,
) -> None:
    ctx, api = make_ctx()
    concrete = make_frame(fixture, [make_column_ref(fixture, "price", int_dtype).as_instance()])
    resolver = UntypedFallback(UnprovenPolicy(), plugin)
    result = resolver.resolve(ctx, Reason.CHECKER_GAP, "join", None, concrete)
    proper = get_proper_type(result)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "polars.dataframe.frame.DataFrame"
    assert api.codes == []
