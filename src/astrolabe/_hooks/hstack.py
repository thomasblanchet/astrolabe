"""The hook for `.hstack()`, which adds new columns to the right of a frame.

The new columns come from another frame or from a list of series.
Without this hook, mypy gives `.hstack()` the type of the receiver, and
the new columns are missing from the type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mypy.nodes import ListExpr

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.series import SeriesRef
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.encoding.concat_remainder import ConcatMember, ConcatRemainderStep
from astrolabe._hooks.propagate import Refusal, explains, not_falsy, not_none, propagates

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _Side:
    """The columns that one side of a `.hstack()` call brings.

    Attributes:
        columns: The declared columns, in order.
        rest: The remainder, that is, the columns that the type does not
            name. None for a closed frame, or for a list of series.
    """

    columns: list[ColumnRef]
    rest: MypyType | None

    def member(self) -> ConcatMember:
        """Describe this side for the proof that no column name repeats.

        Returns:
            The declared names and the remainder of this side.
        """
        names = frozenset(column.name for column in self.columns if column.name is not None)
        return ConcatMember(self.rest, names)


class SeriesListColumns:
    """Read the columns of a list of series that the code writes at the `.hstack()` call.

    A variable of type `list[pl.Series]` loses the type of each element, so
    it holds no column name and no data type. A list written at the call
    is different: mypy knows the type of each item. The plugin types each
    `pl.Series(name, values)` call as a `TypedSeries` that holds the literal
    name and the data type. The data type comes from the `dtype=` argument,
    or it is `Unchecked` without one. This holds whether the code writes
    the series in the list or assigns it to a variable first. This class
    reads the type of each item and makes a column from it. It never reads
    the name from the syntax of the item.
    """

    def __init__(self, plugin: Plugin) -> None:
        """Keep the plugin, which the reader needs to mint each column.

        Args:
            plugin: The mypy plugin, which looks up the class of a column.
        """
        self._plugin = plugin

    @propagates(None)
    def resolved(self, ctx: MethodContext) -> list[ColumnRef] | None:
        """Read the `columns` argument as a list of typed series written at the call.

        The method makes one column from the type of each list item, or gives
        up at the first item that fails.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            One column for each item, in order. None if the argument is not a
            list written at the call, if an item is not a `TypedSeries` with a
            literal name, or if the lookup of the column class fails.
        """
        nodes = CallArgs(ctx).node_list("columns")
        if len(nodes) != 1 or not isinstance(nodes[0], ListExpr):
            return None
        columns: list[ColumnRef] = []
        for item in nodes[0].items:
            item_type = ctx.api.get_expression_type(item)
            series = not_none(SeriesRef.from_proper_type(item_type))
            name = not_none(SeriesRef.literal_name(series))
            dtype = SeriesRef.dtype(series)
            column = not_none(ColumnRef.build_fresh(name, dtype, ctx.api, self._plugin))
            columns.append(column)
        return columns


class HstackHook:
    """Give the type of a frame after `.hstack()`, with the new columns after the old ones.

    The hook reads two forms of the `columns` argument: a typed frame, and a
    list of `pl.Series(...)` items written at the call. If the new columns
    repeat a name, the hook reports the duplicate, because polars fails on
    it.

    Either frame can be open, that is, have columns that its type does not
    name, its remainder. polars fails if an undeclared column of one side has
    the name of a column of the other side. So the hook types the call only
    if it can prove that no such name exists: a remainder must not hold a
    declared name of the other side, and two remainders must share no name.
    The result keeps both remainders. Without the proof, the call goes
    through the `allow_unproven` gate, as a horizontal `pl.concat` does.

    `in_place=True` changes the receiver and gives back the same object.
    The columns of the result are the same as with `in_place=False`, so the
    hook does not read that argument.

    The hook sits in the plugin layer. It reads the frames through `Frame`,
    each series through `SeriesListColumns`, and the remainders through
    `ConcatRemainderStep`.
    """

    def __init__(
        self,
        untyped_fallback: UntypedFallback,
        series_columns: SeriesListColumns,
        plugin: Plugin,
    ) -> None:
        """Keep the fallback, the reader of a list of series, and the remainder step.

        Args:
            untyped_fallback: The fallback that gives the plain frame type when
                the hook cannot read the call, subject to the `allow_unproven`
                setting.
            series_columns: The reader for a `columns` argument that is a list
                of `pl.Series(...)` items and not a typed frame.
            plugin: The mypy plugin, which the remainder step needs.
        """
        self._untyped = untyped_fallback
        self._series_columns = series_columns
        self._duplicate_check = DuplicateNameCheck("hstack")
        self._remainder_step = ConcatRemainderStep(plugin, untyped_fallback, "hstack")

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the result type of one `.hstack()` call.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The columns of the receiver, then the new columns, with the
            remainders of both sides. A series with no `dtype=` gives an
            `Unchecked` column. If the hook cannot read the call, the plain
            `pl.DataFrame`. See `UntypedFallback`.
        """
        result = self._typed(ctx)
        if isinstance(result, Refusal):
            frame = Frame.from_proper_type(ctx.type)
            return self._untyped.resolve(ctx, result.reason, "hstack", frame)
        return result

    @explains(Reason.CHECKER_GAP)
    def _typed(self, ctx: MethodContext) -> MypyType:
        """Build the result frame from the two sides.

        The `columns` argument is a typed frame, or a list of series written
        at the call. A variable of type `list[Series]` does not give a side:
        its type holds no column name and no data type.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The result frame, or a refusal: `REPORTED` after an error, and
            `CHECKER_GAP` if the hook cannot read a side.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        left = self._frame_side(frame)
        other_type = CallArgs(ctx).node_type("columns")
        other = None if other_type is None else Frame.from_proper_type(other_type)
        if other is not None:
            right = self._frame_side(other)
        else:
            right = _Side(not_none(self._series_columns.resolved(ctx)), None)
        merged = [*left.columns, *right.columns]
        names = [column.name for column in merged]
        not_falsy(not self._duplicate_check.fails(ctx, names), reason=Reason.REPORTED)
        members = [left.member(), right.member()]
        outcome = not_none(self._remainder_step.decided(ctx, members), reason=Reason.REPORTED)
        return frame.rebuilt(merged, ctx.api, rest=outcome.slot).as_instance()

    def _frame_side(self, frame: Frame) -> _Side:
        """Read the columns and the remainder of a typed frame.

        A typed frame always holds its dictionary of columns. The checker
        reports a typed frame with no schema class, and gives it the type
        `Any`, so no such frame reaches this hook.

        Args:
            frame: The frame.

        Returns:
            The side.
        """
        known = checked_not_none(frame.known_columns)
        return _Side(known.columns, known.rest)
