"""Carry the binding of one call from the check of its arguments to the type of its result.

A function with an inferred return type can take an open frame, that is,
a frame whose type has a remainder: columns that the type does not name.
Each value that the body returns is then written in the terms of the
function's own remainder, for example `Minus[Rest, Literal["x"]]`. At a
call, `Rest` stands for the caller's own columns. One hook checks the
arguments of the call and binds `Rest`. A later hook gives the call its
type. The binding passes from the first to the second through this store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mypy.nodes import Context
    from mypy.types import Type as MypyType


class FrameEvaluation(Protocol):
    """Write one frame type in the terms of one call's arguments."""

    def evaluated(self, frame_type: MypyType) -> MypyType:
        """Give a frame type with each remainder of the function replaced by the caller's.

        Args:
            frame_type: A frame type written in the terms of the function.

        Returns:
            The frame type in the terms of the call. A type that names no
            remainder of the function stays the same.
        """
        ...


class CallEvaluations:
    """Hold the binding of each call that the fit checked, until the call gets its type.

    mypy runs the check of the arguments of a call before the hook that
    gives the call its type, so the first one records and the second one
    takes. The store keys each evaluation by the node of the call, and
    keeps the node itself: the identity of an AST node is unique only
    while the node lives, so a lookup compares the node too.

    Example:
        >>> CallEvaluations().taken(object()) is None
        True
    """

    def __init__(self) -> None:
        """Start with no evaluation."""
        self._by_call: dict[int, tuple[object, FrameEvaluation]] = {}

    def record(self, call: Context, evaluation: FrameEvaluation) -> None:
        """Keep the evaluation of one call until the hook that types the call takes it.

        Args:
            call: The node of the call.
            evaluation: The evaluation under the binding of the call.
        """
        self._by_call[id(call)] = (call, evaluation)

    def taken(self, call: object) -> FrameEvaluation | None:
        """Take the evaluation of one call out of the store.

        Args:
            call: The node of the call.

        Returns:
            The evaluation, or None if the store holds none for this node.
        """
        entry = self._by_call.pop(id(call), None)
        if entry is None or entry[0] is not call:
            return None
        return entry[1]
