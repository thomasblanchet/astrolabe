"""Keep the type of each value that `infer(...)` marks, until a call of its function needs it.

A function declared to return `InferredExpr`, `InferredDataFrame` or
`InferredLazyFrame` gets its return type from the values that its body
marks with `infer(...)`. mypy gives the type of such a value while it
checks the body, and asks for the return type later, at each call of
the function. The record here carries the types from the first moment
to the second.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.inferred.walker import LiveInferCallWalker

if TYPE_CHECKING:
    from mypy.nodes import Block, FuncDef
    from mypy.types import Type as MypyType


class InferAccumulator:
    """Record the type of each `infer(...)` call, by the function whose body holds it.

    Every other hook of the plugin works out its answer from the mypy
    context of one call. This record is different: it is state that
    lives as long as the plugin, and that one hook writes and another
    reads. The hook on `infer` writes one entry each time mypy checks an
    `infer(...)` call. The hook on a call of the enclosing function reads
    the entries of that function. Each entry is keyed by the identity of
    the call node, so that two calls in one body stay apart.

    The mypy daemon, `dmypy`, keeps one plugin across the checks of an
    edited file. An entry from an older version of a function can then
    stay behind. To find the live entries, each read walks the current
    body of the function. The read deletes every other entry from the
    record itself, so that the record does not grow over many edits. A
    plain `mypy` run starts a new process with an empty record, so it
    has no old entries.
    """

    def __init__(self) -> None:
        """Start with no entry, and no function whose misplaced calls were reported."""
        self._entries: dict[FuncDef, dict[int, MypyType]] = {}
        self._reported_misplaced: set[FuncDef] = set()

    def record(self, enclosing: FuncDef, call_id: int, arg_type: MypyType) -> None:
        """Record the type of the value that one `infer(...)` call marks.

        Args:
            enclosing: The function whose body holds the call.
            call_id: The `id()` of the call node.
            arg_type: The type of the argument of the call.
        """
        self._entries.setdefault(enclosing, {})[call_id] = arg_type

    def resolve_variants(self, funcdef: FuncDef) -> list[MypyType]:
        """Give the types that make up the return type of a function.

        Each such type is a variant: the type of one value that the
        function returns as `return infer(value)`. If the body is there
        to walk, an entry whose call is not in that position in the
        current body is deleted.

        Args:
            funcdef: The function whose variants a call of it needs.

        Returns:
            The type of each live variant. If mypy holds no body for the
            function, every recorded type.

        Example:
            >>> call = infer_call_node()
            >>> funcdef = func_def_with_body(*return_block(call).body)
            >>> dtype = int_dtype(new_type_fixture())
            >>> accumulator = InferAccumulator()
            >>> accumulator.record(funcdef, id(call), dtype)
            >>> accumulator.resolve_variants(funcdef) == [dtype]
            True
        """
        raw = self._entries.get(funcdef, {})
        body = self._current_body(funcdef)
        if body is None:
            return list(raw.values())
        live_ids = LiveInferCallWalker(only_returned=True).find(body)
        trimmed = {call_id: arg_type for call_id, arg_type in raw.items() if call_id in live_ids}
        self._entries[funcdef] = trimmed
        return list(trimmed.values())

    def live_variants(self, funcdef: FuncDef) -> list[MypyType]:
        """Give the types that make up the return type of a function, and delete nothing.

        The hook on `infer` writes these types beside the function for a
        later run of mypy. It runs while mypy checks the body, before a call
        of the function looks for misplaced calls, so it must keep every
        entry.

        Args:
            funcdef: The function whose body mypy checks.

        Returns:
            The type of each variant whose call is the whole expression of a
            `return` statement in the current body.

        Example:
            >>> call = infer_call_node()
            >>> funcdef = func_def_with_body(*return_block(call).body)
            >>> dtype = int_dtype(new_type_fixture())
            >>> accumulator = InferAccumulator()
            >>> accumulator.record(funcdef, id(call), dtype)
            >>> accumulator.live_variants(funcdef) == [dtype]
            True
        """
        raw = self._entries.get(funcdef, {})
        body = checked_not_none(self._current_body(funcdef))
        live_ids = LiveInferCallWalker(only_returned=True).find(body)
        return [arg_type for call_id, arg_type in raw.items() if call_id in live_ids]

    def misplaced_calls_once(self, funcdef: FuncDef) -> set[int]:
        """Find each `infer(...)` call that the body uses in a wrong place, once for each function.

        `infer(x)` works only as the whole expression of a `return`
        statement. A call elsewhere in the body, for example
        `result = infer(x)`, is a misuse. Code that reads the record runs
        at every call of the function, but a misuse is one fact about the
        body, and the user must see it once. This method therefore gives
        the misplaced calls only the first time that it is asked about a
        function.

        Args:
            funcdef: The function to check.

        Returns:
            The `id()` of each recorded call that is still in the current
            body, but not as the expression of a `return` statement.
            Empty after the first request for the same function, or if
            mypy holds no body for it.
        """
        if funcdef in self._reported_misplaced:
            return set()
        self._reported_misplaced.add(funcdef)
        raw = self._entries.get(funcdef, {})
        body = self._current_body(funcdef)
        if body is None:
            return set()
        all_ids = LiveInferCallWalker(only_returned=False).find(body)
        returned_ids = LiveInferCallWalker(only_returned=True).find(body)
        return (raw.keys() & all_ids) - returned_ids

    def _current_body(self, funcdef: FuncDef) -> Block | None:
        """Give the current body of a function, if mypy holds one to walk.

        Args:
            funcdef: The function to read.

        Returns:
            The body, or None if it is empty. A function that mypy loads
            from its cache for another module has an empty body, because
            the cache does not store one.
        """
        body = funcdef.body
        if body is None:  # pragma: no cover
            # A `def` statement that mypy parses always has a block. None
            # is the default of the `FuncDef` constructor, which only a
            # function that mypy builds itself keeps, and no hook sees one.
            return None
        if not body.body:
            return None
        return body
