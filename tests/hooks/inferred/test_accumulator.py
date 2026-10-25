"""Check `InferAccumulator`, which keeps the type of each `infer(...)` call by function.

A user writes `return infer(value)` in a function whose return type is
inferred. One hook records the type of each such call, and another
reads the records when the function is called. The records live as
long as the plugin, so they can hold entries for code that no longer
exists. These tests check that a read keeps only the calls in the
current body of the function, and that a call outside a `return` is
reported once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import ExpressionStmt, ReturnStmt

from astrolabe._hooks.inferred.accumulator import InferAccumulator
from tests.support import func_def_with_body, infer_call_node

if TYPE_CHECKING:
    from mypy.test.typefixture import TypeFixture
    from mypy.types import Instance


def test_resolve_variants_returns_recorded_type(fixture: TypeFixture, int_dtype: Instance) -> None:
    call = infer_call_node()
    funcdef = func_def_with_body(ReturnStmt(call))
    accumulator = InferAccumulator()
    accumulator.record(funcdef, id(call), int_dtype)
    assert accumulator.resolve_variants(funcdef) == [int_dtype]


def test_resolve_variants_evicts_stale_entry(fixture: TypeFixture, int_dtype: Instance) -> None:
    live_call = infer_call_node()
    stale_call_id = 999999
    funcdef = func_def_with_body(ReturnStmt(live_call))
    accumulator = InferAccumulator()
    accumulator.record(funcdef, id(live_call), int_dtype)
    accumulator.record(funcdef, stale_call_id, int_dtype)
    assert accumulator.resolve_variants(funcdef) == [int_dtype]
    # A second read gives the same answer, because the first read deleted the old entry.
    assert accumulator.resolve_variants(funcdef) == [int_dtype]


def test_resolve_variants_trusts_raw_when_body_is_stripped(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    call_id = 42
    funcdef = func_def_with_body()  # empty, as mypy gives a function it loads from its cache
    accumulator = InferAccumulator()
    accumulator.record(funcdef, call_id, int_dtype)
    assert accumulator.resolve_variants(funcdef) == [int_dtype]


def test_resolve_variants_empty_for_unrecorded_function() -> None:
    funcdef = func_def_with_body(ReturnStmt(infer_call_node()))
    assert InferAccumulator().resolve_variants(funcdef) == []


def test_misplaced_calls_flags_reachable_non_returned_call(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    call = infer_call_node()
    funcdef = func_def_with_body(ExpressionStmt(call))
    accumulator = InferAccumulator()
    accumulator.record(funcdef, id(call), int_dtype)
    assert accumulator.misplaced_calls_once(funcdef) == {id(call)}


def test_misplaced_calls_empty_for_returned_call(fixture: TypeFixture, int_dtype: Instance) -> None:
    call = infer_call_node()
    funcdef = func_def_with_body(ReturnStmt(call))
    accumulator = InferAccumulator()
    accumulator.record(funcdef, id(call), int_dtype)
    assert accumulator.misplaced_calls_once(funcdef) == set()


def test_misplaced_calls_empty_when_body_is_stripped(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    funcdef = func_def_with_body()  # empty, as mypy gives a function it loads from its cache
    accumulator = InferAccumulator()
    accumulator.record(funcdef, 1, int_dtype)
    assert accumulator.misplaced_calls_once(funcdef) == set()


def test_misplaced_calls_ignores_stale_id_not_in_current_body(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    funcdef = func_def_with_body(ReturnStmt(infer_call_node()))
    accumulator = InferAccumulator()
    accumulator.record(funcdef, 123456, int_dtype)  # no call in the current body has this id
    assert accumulator.misplaced_calls_once(funcdef) == set()


def test_misplaced_calls_once_reports_only_on_first_call(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    call = infer_call_node()
    funcdef = func_def_with_body(ExpressionStmt(call))
    accumulator = InferAccumulator()
    accumulator.record(funcdef, id(call), int_dtype)
    assert accumulator.misplaced_calls_once(funcdef) == {id(call)}
    assert accumulator.misplaced_calls_once(funcdef) == set()
