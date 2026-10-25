"""Check that the frame given to `.vstack()`, `.extend()` or `.merge_sorted()` has the same schema.

These three methods append or merge the rows of a second frame. polars
accepts the second frame only if it has the same column names, in the
same order, with the same data types. A different data type raises
`SchemaError`, and a different set or order of columns raises
`ShapeError`. The typed frame declares each method to return the type of
its receiver, which is right for a call that succeeds. This module checks
that the call can succeed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.domain.dtype_promotion import DTypePromotion
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.schema_equality import SchemaEquality
from astrolabe._hooks.errors import SchemaMismatchError
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from astrolabe._hooks.domain.frame import PluginLookup
    from mypy.plugin import MethodContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.column import ColumnRef


@dataclass(frozen=True)
class _VstackExtendPlan:
    """Hold the columns of the two frames of one call, once the hook can read both.

    Attributes:
        left_columns: The declared columns of the receiver.
        right_columns: The declared columns of the frame in the argument.
    """

    left_columns: list[ColumnRef]
    right_columns: list[ColumnRef]

    @classmethod
    @propagates(None)
    def from_context(cls, ctx: MethodContext) -> _VstackExtendPlan | None:
        """Read the columns of the receiver and of the argument.

        The typed frame declares these three methods with the parameters of
        polars, so mypy files the frame under `other`, whether the caller
        passes it by position or by name.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The plan, or None if the receiver or the argument is not a
            typed frame whose columns the hook can read.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        left_columns = not_none(frame.columns)
        other_type = not_none(CallArgs(ctx).node_type("other"))
        other = not_none(Frame.from_proper_type(other_type))
        right_columns = not_none(other.columns)
        return cls(left_columns, right_columns)


class VstackExtendHook:
    """Report a `SchemaMismatchError` if the two frames of a stacking call differ.

    Without this hook, the checker accepts `a.vstack(b)` for any frame `b`,
    and the call fails when the program runs. The hook compares the
    declared columns of both frames, position by position, with
    `SchemaEquality`: the same count, and at each position the same name
    and the same data type. The data type `Unchecked` (a column whose data
    type is not known) matches any data type. The hook never changes the
    result type. If it cannot read either frame, it checks nothing.

    It is a mypy method hook in the plugin layer, with one instance for
    each of the three methods.
    """

    def __init__(self, operation: str, plugin: PluginLookup) -> None:
        """Keep the name of the method, and build the schema comparison.

        Args:
            operation: `".vstack()"`, `".extend()"` or `".merge_sorted()"`,
                for the error message.
            plugin: The mypy plugin, which the type promotion rule of
                `SchemaEquality` needs. This hook compares with no
                promotion, so the rule never runs here.
        """
        self._operation = operation
        unchecked = UncheckedCompatibility()
        self._schema_equality = SchemaEquality(unchecked, DTypePromotion(unchecked, plugin))

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Check the call, and give back the return type that the typed frame declares.

        If the two frames do not match, the hook reports
        `SchemaMismatchError`.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The declared return type, always unchanged.
        """
        plan = _VstackExtendPlan.from_context(ctx)
        if plan is None:
            return ctx.default_return_type
        left = self._schema_equality.sorted_by_name(plan.left_columns)
        right = self._schema_equality.sorted_by_name(plan.right_columns)
        if not self._schema_equality.columns_match(left, right, relaxed=False):
            SchemaMismatchError.stacked(self._operation).fail(ctx)
        return ctx.default_return_type
