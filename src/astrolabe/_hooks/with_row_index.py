"""Type `.with_row_index()`, which adds a column that numbers the rows.

polars declares that the call returns a plain frame, so without this hook
the checker loses every column of a typed frame at this call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TYPE_CHECKING

import polars as pl
from mypy.nodes import TypeInfo
from mypy.types import Instance

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.errors import DuplicateColumnError
from astrolabe._hooks.propagate import Refusal, explains, not_none, propagates
from astrolabe._hooks.row_counts import RowCountCheck

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _WithRowIndexPlan:
    """Hold what the hook reads from one `.with_row_index()` call.

    Attributes:
        frame: The receiver frame.
        known: The declared columns of the receiver, and its remainder.
        name: The literal name of the new index column.
    """

    frame: Frame
    known: KnownColumns
    name: str

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(cls, ctx: MethodContext) -> _WithRowIndexPlan:
        """Read the receiver and the name of the index column from the call.

        If the call passes no `name`, the column is called `"index"`, as
        in polars.

        Args:
            ctx: The context that mypy gives for `.with_row_index()`.

        Returns:
            The plan, or a `Refusal`. The reason is `DYNAMIC_NAME` if
            `name` is not a literal string, and `CHECKER_GAP` if the
            receiver is not a typed frame.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        name = not_none(
            CallArgs(ctx).literal_str("name", default="index"),
            reason=Reason.DYNAMIC_NAME,
        )
        return cls(frame, known, name)


class WithRowIndexHook:
    """Give `.with_row_index()` on a typed frame a result with one new `UInt32` column.

    polars puts the new column before every other column and gives it the
    data type `UInt32`. The `offset=` argument changes only the values,
    never the position or the data type, so the hook does not read it. The
    other columns, and the remainder of an open frame, stay the same. The
    remainder is the set of columns of an open frame that its type does
    not name.

    If a column with the new name already exists, polars raises
    `DuplicateError`. So the hook reports a `DuplicateColumnError` unless
    it can prove that the frame has no such column. On an open frame, the
    remainder must also prove that it cannot hold the name.

    It is a mypy method hook in the plugin layer. If it cannot read the
    call, it falls back through `UntypedFallback`, which gives the plain
    frame that polars declares and reports it unless the `allow_unproven`
    setting lets it pass.
    """

    _UINT32_TARGET: Final[Target] = Target.of(pl.UInt32)

    def __init__(self, plugin: Plugin, untyped_fallback: UntypedFallback) -> None:
        """Keep the plugin and the fallback for a call that the hook cannot read.

        Args:
            plugin: The mypy plugin, which finds `pl.UInt32`, and the
                `Column` class if the receiver has no column to copy it
                from.
            untyped_fallback: Gives the plain frame for a call that the
                hook cannot type, and reports it unless the setting lets
                it pass.
        """
        self._plugin = plugin
        self._untyped = untyped_fallback

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.with_row_index()` call.

        The index column comes first, then the declared columns of the
        receiver. The remainder stays unchanged. If the receiver can
        already have a column with the new name, the hook reports a
        `DuplicateColumnError`. The fallback then gets the reason
        `REPORTED`, so that it adds no second error.

        Args:
            ctx: The context that mypy gives for `.with_row_index()`.

        Returns:
            The typed frame with the new index column. If the hook cannot
            read the call, or the name of the new column can already
            exist, the plain `pl.DataFrame` or `pl.LazyFrame`.
        """
        if RowCountCheck.of_receiver(ctx.type, "with_row_index").reported(ctx):
            frame = Frame.from_proper_type(ctx.type)
            return self._untyped.resolve(ctx, Reason.REPORTED, "with_row_index", frame)
        plan = _WithRowIndexPlan.from_context(ctx)
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                "with_row_index",
                Frame.from_proper_type(ctx.type),
            )
        if not self._is_free_name(plan.known, plan.name):
            DuplicateColumnError("with_row_index", plan.name).fail(ctx)
            return self._untyped.resolve(ctx, Reason.REPORTED, "with_row_index", plan.frame)
        index_column = self._index_column(ctx, plan)
        if index_column is None:  # pragma: no cover
            # mypy has always loaded `pl.UInt32` and `Column` when this hook
            # runs, because the typed frame module imports both.
            return self._untyped.resolve(ctx, Reason.CHECKER_GAP, "with_row_index", plan.frame)
        result = [index_column, *plan.known.columns]
        return plan.frame.rebuilt(result, ctx.api, rest=plan.known.rest).as_instance()

    def _is_free_name(self, known: KnownColumns, name: str) -> bool:
        """Answer whether the receiver surely has no column with the new name.

        polars never replaces a column with the index. It raises
        `DuplicateError` if the name exists. So the name must be free in
        the declared columns and in the remainder.

        A closed frame has no remainder. For an open frame,
        `RemainderKnowledge` reads what the type states about the
        remainder. Only a stated rule proves the name absent, for example
        `other_columns(without=["index"])` in the declaration. Any other
        remainder can hold the name when the program runs.

        Args:
            known: The declared columns of the receiver, and its remainder.
            name: The literal name of the new index column.

        Returns:
            True if no declared column has the name and the remainder
            cannot hold it.

        Example:
            >>> fixture = new_type_fixture()
            >>> foo = column_ref(fixture, "foo", int_dtype(fixture))
            >>> hook = WithRowIndexHook(new_plugin(fixture), untyped_fallback())
            >>> concrete = frame(fixture, [foo.as_instance()]).known_columns
            >>> hook._is_free_name(concrete, "index")
            True
            >>> hook._is_free_name(concrete, "foo")
            False
            >>> opaque = known_columns([foo], declared_rest(fixture))
            >>> hook._is_free_name(opaque, "index")
            False
            >>> excluding = declared_rest(fixture, without=("index",))
            >>> hook._is_free_name(known_columns([foo], excluding), "index")
            True
        """
        if any(column.name == name for column in known.columns):
            return False
        return RemainderKnowledge().proves_absent(known.rest, name)

    @propagates(None)
    def _index_column(self, ctx: MethodContext, plan: _WithRowIndexPlan) -> ColumnRef | None:
        """Build the index column, `Column[name, UInt32]`.

        If the receiver has a declared column, the new column reuses the
        class record of `Column` from that column, and needs no lookup.
        Else the plugin looks up `Column` by its full name.

        Args:
            ctx: The context that mypy gives for `.with_row_index()`.
            plan: What the hook read from the call.

        Returns:
            The index column, or None if mypy cannot find `pl.UInt32` or
            `Column`.
        """
        info = self._plugin.lookup_fully_qualified(self._UINT32_TARGET.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # `pl.UInt32` is part of polars, and mypy has always loaded
            # polars when a frame method runs.
            return None
        dtype = Instance(info.node, [])
        if plan.known.columns:
            return ColumnRef.build(plan.name, dtype, plan.known.columns[0], ctx.api)
        return ColumnRef.build_fresh(plan.name, dtype, ctx.api, self._plugin)
