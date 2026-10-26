"""The hook for `pl.concat()`, which types a concat by name or reports a sure mismatch.

Without this hook, mypy types every concat as a plain frame. What the
hook can prove depends on the `how` argument.

A vertical concat (`"vertical"`, the default, and `"vertical_relaxed"`)
stacks the frames by column position. The hook never types its result.
A function whose frame parameter has a remainder, the columns that its
type does not name, can return those columns in any order, and nothing
checks that order. The hook still reports a sure mismatch that does not
depend on order: frames whose number of columns, names or data types
differ. `"vertical"` needs equal data types, and `"vertical_relaxed"`
accepts a pair that `DTypePromotion` can widen to one type. The setting
`disallow_vertical_concat` makes each vertical concat an error.

A diagonal concat (`"diagonal"` and `"diagonal_relaxed"`) matches the
columns by name, so order does not matter. The result holds every
column of every frame, in the order the names first appear, and polars
fills a column that a frame lacks with nulls. A name that more than one
frame declares needs equal data types under `"diagonal"`, and the hook
reports a pair that differs. Under `"diagonal_relaxed"` the pair must
widen to one type, and a pair that does not widen becomes `Unchecked`.

The five align strategies (`"align"`, `"align_full"`, `"align_inner"`,
`"align_left"` and `"align_right"`) give the same schema as
`"diagonal"`. They differ only in which rows stay. Polars also needs at
least one column name that every frame holds, and the hook reports a
call without one.

`"horizontal_extend"` puts the columns of each frame side by side, in
frame order. Every column name of every frame must be different. Polars
1.42.0 adds this `how`. An older release raises `ValueError` for it, and
mypy reports the value, because the stub of polars does not list it.

The hook does not type any other `how`, for example `"horizontal"`,
which is ambiguous about the number of rows. The call then falls back:
it gives the plain frame type in place of a typed result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TYPE_CHECKING

from mypy.nodes import ListExpr, TupleExpr
from mypy.types import Instance, ProperType, get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility, UncheckedFallback
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.dtype_promotion import DTypePromotion
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.schema_equality import DecimalParameters, SchemaEquality
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.encoding.concat_remainder import (
    ConcatMember,
    ConcatRemainderStep,
)
from astrolabe._hooks.errors import (
    InvalidCallShapeError,
    SchemaMismatchError,
    VerticalConcatDisallowedError,
)
from astrolabe._hooks.propagate import Refusal, explains, not_falsy, not_none, propagates
from astrolabe._rules.releases import polars_releases

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext, Plugin

    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _ColumnMerge:
    """The two columns with one name that a diagonal concat must join into one column of its result.

    Attributes:
        name: The name of the column.
        existing: The column with that name in the result built so far.
        column: The column with the same name from a later frame.
        columns_by_name: The result built so far. The merge updates it in
            place if the two columns agree.
    """

    name: str
    existing: ColumnRef
    column: ColumnRef
    columns_by_name: dict[str, ColumnRef]


@dataclass(frozen=True)
class _DiagonalMergePolicies:
    """The rules that a diagonal concat uses to join two data types for one column name.

    Attributes:
        unchecked: The rule that accepts equal data types, or a pair where
            one side is `Unchecked`.
        promotion: The rule that widens a pair of data types to one type,
            for a relaxed concat.
        plugin: The mypy plugin, which mints a widened data type.
        unchecked_fallback: The fallback that gives `Unchecked` to a column
            of a relaxed concat whose data types do not widen to one type.
    """

    unchecked: UncheckedCompatibility
    promotion: DTypePromotion
    plugin: Plugin
    unchecked_fallback: UncheckedFallback


class _DiagonalColumnReconciler:
    """Join the two data types of one shared column name in a diagonal concat.

    When two frames of a diagonal concat declare the same name, the result
    holds one column with that name, and its data type must be one that
    both frames accept. With `"diagonal"`, the two data types must be equal,
    or one of them `Unchecked`. With `"diagonal_relaxed"`, polars widens the
    pair to a common type, for example `Int32` and `UInt32` to `Int64`. A
    pair with no common type becomes `Unchecked`.

    The values that stay the same for the whole call are state of the
    instance. Each merge is a method parameter, because it is new for each
    shared name.
    """

    def __init__(self, ctx: FunctionContext, how: str, policies: _DiagonalMergePolicies) -> None:
        """Keep the values that stay the same for the whole call.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.
            how: `"diagonal"` or `"diagonal_relaxed"`.
            policies: The rules that join two data types.
        """
        self._ctx = ctx
        self._how = how
        self._policies = policies
        self._reported = False

    @property
    def failure_reason(self) -> Reason:
        """Give the reason why the result schema has no answer.

        Returns:
            `REPORTED` if a strict merge reported a mismatch. Otherwise
            `CHECKER_GAP`, for a frame or a data type that the checker cannot
            read.

        Example:
            >>> reconciler = _DiagonalColumnReconciler.__new__(_DiagonalColumnReconciler)
            >>> reconciler._reported = True
            >>> reconciler.failure_reason.name
            'REPORTED'
            >>> reconciler._reported = False
            >>> reconciler.failure_reason.name
            'CHECKER_GAP'
        """
        if self._reported:
            return Reason.REPORTED
        return Reason.CHECKER_GAP

    def reconcile(self, merge: _ColumnMerge) -> bool:
        """Join a column with the column of the same name in the result built so far.

        Args:
            merge: The two columns with the shared name, and the result built
                so far.

        Returns:
            True if the two data types are equal, or one of them is
            `Unchecked`. In that case the method puts the side with the agreed
            data type into the result. Also True for `"diagonal"` where the
            two differ only in the parameters of a `Decimal`: the result keeps
            the first. Always
            True for a relaxed concat, see `_merge_relaxed_column`. False only
            for a strict concat whose two data types differ.
        """
        existing_dtype = get_proper_type(merge.existing.dtype)
        column_dtype = get_proper_type(merge.column.dtype)
        relaxed = self._how == "diagonal_relaxed"
        unchecked = self._policies.unchecked
        # A relaxed concat widens to the supertype, which an `Unchecked` side decides.
        merging = unchecked.joined if relaxed else unchecked.resolved
        merged = merging(existing_dtype, column_dtype)
        if merged is not None:
            merge.columns_by_name[merge.name] = self._merged_column(merged, merge)
            return True
        if relaxed:
            self._merge_relaxed_column(merge, existing_dtype, column_dtype)
            return True
        if self._how == "diagonal" and DecimalParameters().same(existing_dtype, column_dtype):
            # Polars reads two `Decimal` types as one, and keeps the first. An align refuses.
            return True
        return self._merge_strict_column(merge, existing_dtype, column_dtype)

    def _merged_column(self, merged: MypyType, merge: _ColumnMerge) -> ColumnRef:
        """Give the column for a shared name with the data type that the two sides merge into.

        Args:
            merged: The merged data type.
            merge: The two columns with the shared name.

        Returns:
            The side whose data type is the merged one. Otherwise a new column with the name
            and the merged data type, as for two structs whose `Unchecked` fields differ.
        """
        if merged == merge.existing.dtype:
            return merge.existing
        if merged == merge.column.dtype:
            return merge.column
        return ColumnRef.build(merge.name, merged, merge.existing, self._ctx.api)

    def _merge_strict_column(
        self,
        merge: _ColumnMerge,
        existing_dtype: ProperType,
        column_dtype: ProperType,
    ) -> bool:
        """Handle a shared column of a strict diagonal concat whose data types differ.

        Args:
            merge: The two columns with the shared name, and the result built
                so far.
            existing_dtype: The data type of the column in the result so far.
            column_dtype: The data type of the new column.

        Returns:
            Always False. `reconcile` already accepted equal types and
            `Unchecked`, so here the two types differ. The method reports
            `SchemaMismatchError` only when both types are known polars data
            types. A type variable inside a generic function is unknown, not
            wrong, so the method gives False without an error.
        """
        if not (isinstance(existing_dtype, Instance) and isinstance(column_dtype, Instance)):
            return False
        error = SchemaMismatchError.concat_column(
            self._how,
            merge.name,
            existing_dtype.type.name,
            column_dtype.type.name,
        )
        error.fail(self._ctx)
        self._reported = True
        return False

    def _merge_relaxed_column(
        self,
        merge: _ColumnMerge,
        existing_dtype: ProperType,
        column_dtype: ProperType,
    ) -> None:
        """Join a shared column of a relaxed diagonal concat, and always find a data type for it.

        If both data types are known polars data types, the method first
        widens the pair to a common type. `DTypePromotion` gives two numbers
        the supertype that polars measures for the pair, and lets `String`
        take in a type that polars can print. Otherwise the method gives the
        column the data type `Unchecked`. That case covers a pair with no
        common type, for example `String` and `Duration`, which polars refuses
        to cast. It also covers a type variable inside a generic function,
        which is unknown and not wrong. `Unchecked` on one column keeps the
        rest of the result typed.

        Args:
            merge: The two columns with the shared name, and the result built
                so far. The method updates the result in each case.
            existing_dtype: The data type of the column in the result so far.
            column_dtype: The data type of the new column.
        """
        if isinstance(existing_dtype, Instance) and isinstance(column_dtype, Instance):
            promoted = self._policies.promotion.promoted(existing_dtype, column_dtype)
            if promoted is not None:
                merge.columns_by_name[merge.name] = self._promoted_column(promoted, merge)
                return
        substituted = self._policies.unchecked_fallback.substituted(
            self._ctx,
            Reason.CHECKER_GAP,
            "concat",
            merge.name,
        )
        dtype = substituted if substituted is not None else existing_dtype
        new_column = ColumnRef.build(merge.name, dtype, merge.existing, self._ctx.api)
        merge.columns_by_name[merge.name] = new_column

    def _promoted_column(self, promoted: Instance, merge: _ColumnMerge) -> ColumnRef:
        """Give the column for a shared name with the common data type.

        Args:
            promoted: The common type of the two data types.
            merge: The two columns with the shared name.

        Returns:
            The side that already has the common type. Otherwise a new column
            with the name and the common type, for example `Int64` for `Int32`
            and `UInt32`.
        """
        if promoted is get_proper_type(merge.existing.dtype):
            return merge.existing
        if promoted is get_proper_type(merge.column.dtype):
            return merge.column
        return ColumnRef.build(merge.name, promoted, merge.existing, self._ctx.api)


class ConcatHook:
    """Give the result type of `pl.concat([frame, ...])`, and report a concat that cannot run.

    The module docstring gives the rule for each value of `how`. The hook
    reads the frames from a list or a tuple written at the call. It works
    on open frames too: `ConcatRemainderStep` decides the remainder of the
    result, and reports a concat whose columns depend on the caller.

    The hook sits in the plugin layer. It reads each frame through `Frame`.
    If it cannot type the call, it falls back: it gives the plain frame
    type in place of a typed result.
    """

    _VERTICAL_HOW: Final[frozenset[str]] = frozenset({"vertical", "vertical_relaxed"})
    _DIAGONAL_HOW: Final[frozenset[str]] = frozenset({"diagonal", "diagonal_relaxed"})
    _RELAXED_HOW: Final[frozenset[str]] = frozenset({"vertical_relaxed", "diagonal_relaxed"})
    _ALIGN_HOW: Final[frozenset[str]] = frozenset({
        "align",
        "align_full",
        "align_inner",
        "align_left",
        "align_right",
    })
    _UNION_BY_NAME_HOW: Final[frozenset[str]] = _DIAGONAL_HOW | _ALIGN_HOW
    _HORIZONTAL_EXTEND_HOW: Final[str] = "horizontal_extend"

    def __init__(
        self,
        untyped_fallback: UntypedFallback,
        disallow_vertical: bool,
        plugin: Plugin,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Build the rules and the fallbacks that the hook needs.

        Args:
            untyped_fallback: The fallback that gives the plain frame type when
                the hook cannot type the call, subject to the `allow_unproven`
                setting.
            disallow_vertical: Whether each vertical concat is an error. See
                `AstrolabeConfig`.
            plugin: The mypy plugin, which mints a widened data type and the
                remainder of the result.
            unchecked_fallback: The fallback that gives `Unchecked` to a column
                of a relaxed diagonal concat whose data types do not widen to
                one type.
        """
        self._untyped = untyped_fallback
        self._disallow_vertical = disallow_vertical
        self._unchecked = UncheckedCompatibility()
        self._promotion = DTypePromotion(self._unchecked, plugin)
        self._schema_equality = SchemaEquality(self._unchecked, self._promotion)
        self._merge_policies = _DiagonalMergePolicies(
            self._unchecked,
            self._promotion,
            plugin,
            unchecked_fallback,
        )
        self._duplicate_name_check = DuplicateNameCheck("concat")
        self._remainder_step = ConcatRemainderStep(plugin, untyped_fallback)

    def __call__(self, ctx: FunctionContext) -> MypyType:
        """Give the result type of one `pl.concat(...)` call.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.

        Returns:
            The typed result for a diagonal, align or `"horizontal_extend"`
            concat that the hook can read. Otherwise the plain `pl.DataFrame` or
            `pl.LazyFrame` of the kind of the first typed frame, or the type
            that mypy chose. See `UntypedFallback`. Before the merge, the hook
            reports an empty list of items for any `how`. It can also report
            `[vertical-concat-disallowed]` or `[schema-mismatch]` for a vertical
            concat.
        """
        how = CallArgs(ctx).literal_str("how", default="vertical")
        empty = self._report_if_empty_items(ctx)
        vertical = how in self._VERTICAL_HOW and self._check_vertical(ctx, how)
        reported = empty or vertical
        result = self._merged_result(ctx, how)
        if not isinstance(result, Refusal):
            return result
        return self._untyped.resolve(
            ctx,
            self._fallback_reason(reported, result),
            "concat",
            *self._best_effort_frames(ctx),
        )

    def _fallback_reason(self, reported: bool, refusal: Refusal) -> Reason:
        """Give the reason for the fallback of a concat that has no typed result.

        Args:
            reported: Whether a check before the merge already reported an
                error.
            refusal: Why the merge gave no result.

        Returns:
            `REPORTED` if a check already reported. Otherwise the reason of the
            merge.

        Example:
            >>> hook = ConcatHook.__new__(ConcatHook)
            >>> hook._fallback_reason(True, Refusal(Reason.CHECKER_GAP)).name
            'REPORTED'
            >>> hook._fallback_reason(False, Refusal(Reason.CHECKER_GAP)).name
            'CHECKER_GAP'
        """
        if reported:
            return Reason.REPORTED
        return refusal.reason

    def _merged_result(self, ctx: FunctionContext, how: str | None) -> MypyType | Refusal:
        """Build the typed result for a concat by name or a `"horizontal_extend"` concat.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.
            how: The value of `how`.

        Returns:
            The typed result, or a refusal. The reason is a checker gap for any
            other `how`, which includes a vertical concat. Otherwise it is the
            reason that the builder of the result gives.
        """
        if how in self._UNION_BY_NAME_HOW:
            return self._diagonal_frame(ctx, how)
        if how == self._HORIZONTAL_EXTEND_HOW:
            return self._extended_frame(ctx)
        return Refusal(Reason.CHECKER_GAP)

    @polars_releases
    def _extended_frame(self, ctx: FunctionContext) -> MypyType | Refusal:
        """Build the typed result of a `"horizontal_extend"` concat.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.

        Returns:
            The typed result, or the refusal of `_horizontal_extend_frame`.
        """
        return self._horizontal_extend_frame(ctx)

    @_extended_frame.before("1.42.0")
    def _no_extend(self, ctx: FunctionContext) -> MypyType | Refusal:
        """Give up on `"horizontal_extend"`, which the older releases do not have.

        Polars raises `ValueError` for that value of `how`. The stub of polars
        does not list it, so mypy already reports the call.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call, which the method does not read.

        Returns:
            A refusal for an error that is already reported.

        Example:
            >>> ConcatHook.__new__(ConcatHook)._no_extend(None).reason.name
            'REPORTED'
        """
        del ctx
        return Refusal(Reason.REPORTED)

    def _report_if_empty_items(self, ctx: FunctionContext) -> bool:
        """Report `pl.concat([])`, a list of items written empty.

        Polars raises `ValueError: cannot concat empty list` for any `how`. The
        hook can read this call. The call can never run, so the checker
        reports it.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.

        Returns:
            True if the method reported the error.
        """
        items_node = self._single_items_node(CallArgs(ctx))
        if items_node is None or items_node.items:
            return False
        InvalidCallShapeError("concat([]) raises ValueError: cannot concat empty list").fail(ctx)
        return True

    def _check_vertical(self, ctx: FunctionContext, how: str) -> bool:
        """Report a vertical concat that the setting refuses, or one whose schemas surely differ.

        The method never types the result. It only decides whether to report
        `[vertical-concat-disallowed]` or `[schema-mismatch]`. The caller falls
        back to the plain type after it in each case.

        The method reports the first frame whose schema differs from the
        schema of the first frame, in any order. It reports nothing if it
        cannot read the frames or the columns of the first frame.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.
            how: `"vertical"` or `"vertical_relaxed"`.

        Returns:
            True if the method reported either error.
        """
        if self._disallow_vertical:
            VerticalConcatDisallowedError(how).fail(ctx)
            return True
        frames = self._same_kind_frames(ctx)
        if frames is None:
            return False
        first_columns = frames[0].columns
        if first_columns is None:
            return False
        relaxed = how in self._RELAXED_HOW
        first_sorted = self._schema_equality.sorted_by_name(first_columns)
        for index, frame in enumerate(frames[1:], start=1):
            if not self._frame_matches(first_sorted, frame, relaxed):
                SchemaMismatchError.concat_frame(how, index).fail(ctx)
                return True
        return False

    @explains(Reason.CHECKER_GAP)
    def _diagonal_frame(self, ctx: FunctionContext, how: str) -> MypyType:
        """Build the typed result of a diagonal or align concat, which matches columns by name.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.
            how: `"diagonal"`, `"diagonal_relaxed"` or an align strategy.

        Returns:
            The result, or a refusal. The reason is a checker gap if the hook
            cannot read the frames. It is `Reason.REPORTED` after an error: an
            align concat with no column that every frame holds, a shared column
            whose data types differ, or a remainder that makes the result
            depend on the caller.
        """
        frames = not_none(self._same_kind_frames(ctx))
        not_falsy(self._align_precondition_holds(ctx, how, frames), reason=Reason.REPORTED)
        reconciler = _DiagonalColumnReconciler(ctx, how, self._merge_policies)
        union = self._diagonal_columns(frames, reconciler)
        columns = not_none(union, reason=reconciler.failure_reason)
        members = not_none(self._members(frames))
        outcome = not_none(self._remainder_step.decided(ctx, members), reason=Reason.REPORTED)
        rebuilt = frames[0].rebuilt(columns, ctx.api, rest=outcome.slot)
        return rebuilt.as_instance()

    @propagates(True)
    def _align_precondition_holds(
        self,
        ctx: FunctionContext,
        how: str,
        frames: list[Frame],
    ) -> bool:
        """Report an align concat that has no column name that every frame holds.

        Each align strategy joins the frames on every column name that all of
        them hold, and polars raises when there is none. A name that only some
        of the frames share is not enough.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.
            how: The value of `how`.
            frames: The frames of the call.

        Returns:
            True if `how` is not an align strategy, if the column names of the
            frames are not all known, or if at least one name is in every frame.
            Otherwise the method reports `InvalidCallShapeError` and gives False.
        """
        if how not in self._ALIGN_HOW:
            return True
        name_sets: list[set[str | None]] = []
        for frame in frames:
            known = not_none(frame.known_columns)
            name_sets.append({column.name for column in known.columns})
        if set.intersection(*name_sets):
            return True
        error = InvalidCallShapeError(
            "concat(how=\"align\") raises: 'align' strategy requires at least one common column",
        )
        error.fail(ctx)
        return False

    @propagates(None)
    def _members(self, frames: list[Frame]) -> list[ConcatMember] | None:
        """Read the declared names and the remainder of each frame, for the remainder decision.

        Args:
            frames: The frames, in call order.

        Returns:
            One member for each frame, or None if the schema of a frame is not
            known.
        """
        members: list[ConcatMember] = []
        for frame in frames:
            known = not_none(frame.known_columns)
            named = (column.name for column in known.columns if column.name is not None)
            members.append(ConcatMember(known.rest, frozenset(named)))
        return members

    @propagates(None)
    def _diagonal_columns(
        self,
        frames: list[Frame],
        reconciler: _DiagonalColumnReconciler,
    ) -> list[ColumnRef] | None:
        """Build the columns of a diagonal concat, in the order the names first appear.

        A name that is new to the result adds its column. A name that an
        earlier frame already holds goes to `reconciler`, which joins the two
        columns into one.

        Args:
            frames: The frames, in call order.
            reconciler: The rule that joins two columns with one name.

        Returns:
            The columns of the result, or None if the schema of a frame is not
            known, a column has no literal name, or a shared name has data types
            that do not agree.
        """
        columns_by_name: dict[str, ColumnRef] = {}
        order: list[str] = []
        for frame in frames:
            known = not_none(frame.known_columns)
            for column in known.columns:
                # The plugin refuses a `Column[...]` item whose name is not a
                # literal when it reads the annotation, so every column of a
                # typed frame has a name.
                name = checked_not_none(column.name)
                existing = columns_by_name.get(name)
                if existing is None:
                    columns_by_name[name] = column
                    order.append(name)
                    continue
                merge = _ColumnMerge(name, existing, column, columns_by_name)
                not_falsy(reconciler.reconcile(merge))
        return [columns_by_name[name] for name in order]

    @explains(Reason.CHECKER_GAP)
    def _horizontal_extend_frame(self, ctx: FunctionContext) -> MypyType:
        """Build the typed result of a `"horizontal_extend"` concat.

        With `"horizontal_extend"`, polars needs every column name of every
        frame to be different. It raises `column with name '...' has more than
        one occurrence` otherwise, and the method reports `DuplicateColumnError`.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.

        Returns:
            The columns of every frame side by side, in frame order, or a
            refusal. The reason is a checker gap if the hook cannot read the
            frames. It is `Reason.REPORTED` after an error: two columns with one
            name, or a remainder that makes the result depend on the caller.
        """
        frames = not_none(self._same_kind_frames(ctx))
        columns: list[ColumnRef] = []
        for frame in frames:
            columns.extend(not_none(frame.known_columns).columns)
        names = [column.name for column in columns]
        has_duplicate = self._duplicate_name_check.fails(ctx, names)
        not_falsy(not has_duplicate, reason=Reason.REPORTED)
        members = not_none(self._members(frames))
        outcome = not_none(self._remainder_step.decided(ctx, members), reason=Reason.REPORTED)
        rebuilt = frames[0].rebuilt(columns, ctx.api, rest=outcome.slot)
        return rebuilt.as_instance()

    def _best_effort_frames(self, ctx: FunctionContext) -> list[Frame | None]:
        """Read each item of the call as a frame, and keep going past an item that is not one.

        `_same_kind_frames` stops at the first item that is not a typed frame.
        This method does not stop, because `UntypedFallback.resolve` falls back
        from the first typed frame that it finds. An item that is not typed must
        not hide a typed frame after it.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.

        Returns:
            One entry for each item: the frame, or None for an item that is not
            a typed frame. An empty list if the items are not a list or a tuple
            written at the call.
        """
        items_node = self._single_items_node(CallArgs(ctx))
        if items_node is None:
            return []
        return [Frame.from_proper_type(ctx.api.get_expression_type(e)) for e in items_node.items]

    def _same_kind_frames(self, ctx: FunctionContext) -> list[Frame] | None:
        """Read the frames of the call, and make sure that they are all of the same kind.

        A call to `pl.concat(...)` cannot mix `TypedDataFrame` and
        `TypedLazyFrame` items.

        Args:
            ctx: The mypy context of the `pl.concat(...)` call.

        Returns:
            The frames, in call order. None if `items` is not a list or a tuple
            written at the call with at least one item, if an item is not a
            typed frame, or if the frames are not all of one kind.
        """
        items_node = self._single_items_node(CallArgs(ctx))
        if items_node is None or not items_node.items:
            return None
        prepared: list[Frame] = []
        for element in items_node.items:
            frame = Frame.from_proper_type(ctx.api.get_expression_type(element))
            if frame is None:
                return None
            prepared.append(frame)
        first_kind = prepared[0].as_instance().type.fullname
        if any(frame.as_instance().type.fullname != first_kind for frame in prepared[1:]):
            # polars cannot concat an eager frame with a lazy frame. mypy
            # already reports such a list, because the polars stub binds one
            # type variable to one kind, and the call keeps the plain frame.
            return None
        return prepared

    def _frame_matches(self, first_sorted: list[ColumnRef], frame: Frame, relaxed: bool) -> bool:
        """Tell whether the schema of one frame matches the schema of the first frame, in any order.

        Args:
            first_sorted: The columns of the first frame, sorted by name. The
                caller sorts them once for all the frames.
            frame: The frame to compare.
            relaxed: Whether a pair of data types that widen to one type counts
                as a match, not only an equal pair.

        Returns:
            True if the columns of the frame are not known, which leaves nothing
            to compare. True if its columns, sorted by name, match
            `first_sorted`.
        """
        columns = frame.columns
        if columns is None:
            return True
        sorted_columns = self._schema_equality.sorted_by_name(columns)
        return self._schema_equality.columns_match(first_sorted, sorted_columns, relaxed=relaxed)

    def _single_items_node(self, args: CallArgs) -> ListExpr | TupleExpr | None:
        """Read the syntax node of the `items` argument.

        Args:
            args: The arguments of the call.

        Returns:
            The list or tuple node, or None if `items` is not one list or tuple
            written at the call.
        """
        items_nodes = args.node_list("items")
        if len(items_nodes) != 1:
            return None
        items_node = items_nodes[0]
        if not isinstance(items_node, ListExpr | TupleExpr):
            return None
        return items_node
