"""Type `.join_asof()` on a typed frame: join each row to the nearest key, not an equal one.

An as-of join matches each left row with the right row whose `on` key is
nearest, for example the last price before each trade. It can also require
exact matches on `by` keys first, for example the same stock. polars types
the call as a plain frame, so without this hook the checker loses every
column at this call.

The result has the same layout as a left join: the left columns in order,
then the right columns that are not keys. A right column whose name is
already in the result takes a suffix, `_right` by default. The two kinds of
key follow different rules in polars:

- `on=`, or `left_on=` with `right_on=`, names exactly one column on each
  side, never a list. The key pair merges into the left column only if
  `coalesce=True`, the default, and the two keys have one name. A right
  key with another name stays in the result, whatever `coalesce=` says.
  The two data types can differ within one family, for example `Int32`
  and `Int64`.
- `by=`, or `by_left=` with `by_right=`, names one column or a list. The
  key pairs always merge into the left columns, whatever `coalesce=` says.
  The two data types must be equal. A `by` pair of `Int32` and `Int64`
  fails.
- A merged key always keeps the data type of the left column. The join
  never widens it to a common type.
- If a call gives `on=` together with `left_on=` and `right_on=`, polars
  uses `on=` and ignores the others. The same holds for `by=`. The hook
  reads them in the same order. A call that gives `by=` together with
  `by_left=` and `by_right=` that name different columns is rare, and the
  hook reads only `by=` there.
- `strategy=`, `tolerance=` and the other arguments that tune the match do
  not change the columns of the result, so the hook does not read them.

These rules differ from `.join()` in too many places to share its merge
code, so this module has its own plan, key reader and merge classes. It
shares the key rules `UncheckedCompatibility` and `JoinKeyCompatibility`
with `.join()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Final, TYPE_CHECKING

from mypy.types import Instance, get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.join_key_compatibility import JoinKeyCompatibility
from astrolabe._hooks.domain.planned import PlannedType
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.join_key_tree import JoinKeyTree
from astrolabe._hooks.encoding.join_remainder import (
    DeclaredNames,
    JoinRemainderStep,
    JoinSides,
    ProbeNames,
)
from astrolabe._hooks.errors import (
    InvalidCallShapeError,
    JoinKeyDtypeMismatchError,
    UnknownColumnError,
)
from astrolabe._hooks.duplicate_name import CollisionSuffix
from astrolabe._hooks.propagate import Refusal, explains, not_falsy, not_none, not_refused

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from mypy.plugin import MethodContext, Plugin

    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _JoinAsofPlan:
    """Hold what the hook reads from one `.join_asof()` call, with the keys checked.

    Attributes:
        frame: The receiver frame.
        left_columns: The declared columns of the receiver.
        right_columns: The declared columns of `other`.
        on_name: The `on` key of the receiver.
        right_on_name: The `on` key of `other`.
        by_left_names: The `by` keys of the receiver, or an empty list if
            the call gives no `by` key.
        by_right_names: The `by` keys of `other`, in the same order as
            `by_left_names`.
        coalesce: Whether the `on` key pair merges into one column. The
            `by` key pairs always merge.
        left_rest: The encoded remainder of the receiver, that is its
            columns that the type does not name, or None for a closed
            frame.
        right_rest: The encoded remainder of `other`, or None.
    """

    frame: Frame
    left_columns: list[ColumnRef]
    right_columns: list[ColumnRef]
    on_name: str
    right_on_name: str
    by_left_names: list[str]
    by_right_names: list[str]
    coalesce: bool
    left_rest: MypyType | None = None
    right_rest: MypyType | None = None

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(
        cls,
        ctx: MethodContext,
        on_compatibility: JoinKeyCompatibility,
        by_compatibility: UncheckedCompatibility,
    ) -> _JoinAsofPlan:
        """Read the two frames, the keys and `coalesce=` of a `.join_asof()` call.

        Args:
            ctx: The context that mypy gives for `.join_asof()`.
            on_compatibility: Decides whether an `on` key pair can join,
                which allows two data types of one family.
            by_compatibility: Decides whether a `by` key pair can join,
                which needs equal data types.

        Returns:
            The plan, or a `Refusal`. The reason is `CHECKER_GAP` if
            either frame is not a typed frame or `coalesce=` is not a
            literal boolean, and `REPORTED` after an alias in a key. Else it
            is the reason that the key reader gives.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        other = not_none(cls.other_frame(ctx))
        left_known = not_none(frame.known_columns)
        right_known = not_none(other.known_columns)
        args = CallArgs(ctx)
        tree = JoinKeyTree("join_asof")
        keys = ("on", "left_on", "right_on")
        not_falsy(tree.holds_no_alias(ctx, keys), reason=Reason.REPORTED)
        not_falsy(tree.keeps_rows(ctx, keys), reason=Reason.REPORTED)
        resolver = _JoinAsofKeyResolver(
            ctx,
            left_known.columns,
            right_known.columns,
            on_compatibility,
            by_compatibility,
        )
        on_name = not_refused(resolver.on_name)
        right_on_name = not_refused(resolver.right_on_name)
        by_left_names = not_refused(resolver.by_left_names)
        by_right_names = not_refused(resolver.by_right_names)
        coalesce = not_none(args.literal_bool("coalesce", default=True))
        return cls(
            frame,
            left_known.columns,
            right_known.columns,
            on_name,
            right_on_name,
            by_left_names,
            by_right_names,
            coalesce,
            left_known.rest,
            right_known.rest,
        )

    @staticmethod
    def other_frame(ctx: MethodContext) -> Frame | None:
        """Read the frame in the argument `other`.

        Args:
            ctx: The context that mypy gives for `.join_asof()`.

        Returns:
            The frame, or None if `other` is not a typed frame.
        """
        other_type = CallArgs(ctx).node_type("other")
        if other_type is None:
            return None
        return Frame.from_proper_type(other_type)


class _JoinAsofKeyResolver:
    """Read and check the `on` and `by` keys of one `.join_asof()` call.

    One instance serves one call. It holds what stays the same for every
    key: the mypy context, the columns of both frames, and the two rules
    for the data types of a key pair. The two groups of keys are
    independent. Each one is read and checked the first time a property
    asks for it, and the answer is kept.

    Each check reports its own error: a key that a frame does not have,
    `by` lists of different lengths, or a key pair whose data types
    cannot join. The chain then stops with the reason `REPORTED`.
    """

    def __init__(
        self,
        ctx: MethodContext,
        left_columns: list[ColumnRef],
        right_columns: list[ColumnRef],
        on_compatibility: JoinKeyCompatibility,
        by_compatibility: UncheckedCompatibility,
    ) -> None:
        """Keep the state of one call.

        Args:
            ctx: The context that mypy gives for `.join_asof()`.
            left_columns: The declared columns of the receiver.
            right_columns: The declared columns of `other`.
            on_compatibility: Decides whether an `on` key pair can join.
            by_compatibility: Decides whether a `by` key pair can join.
        """
        self._ctx = ctx
        self._left_columns = left_columns
        self._right_columns = right_columns
        self._on_compatibility = on_compatibility
        self._by_compatibility = by_compatibility

    @property
    @explains(Reason.CHECKER_GAP)
    def on_name(self) -> str:
        """The checked `on` key of the receiver, or a `Refusal`."""
        return not_refused(self._validated_on_pair)[0]

    @property
    @explains(Reason.CHECKER_GAP)
    def right_on_name(self) -> str:
        """The checked `on` key of `other`, or a `Refusal`."""
        return not_refused(self._validated_on_pair)[1]

    @property
    @explains(Reason.CHECKER_GAP)
    def by_left_names(self) -> list[str]:
        """The checked `by` keys of the receiver in call order, or a `Refusal`."""
        return not_refused(self._validated_by_pair)[0]

    @property
    @explains(Reason.CHECKER_GAP)
    def by_right_names(self) -> list[str]:
        """The checked `by` keys of `other`, in the order of `by_left_names`, or a `Refusal`."""
        return not_refused(self._validated_by_pair)[1]

    @cached_property
    def _args(self) -> CallArgs:
        return CallArgs(self._ctx)

    @cached_property
    @explains(Reason.REPORTED)
    def _validated_on_pair(self) -> tuple[str, str]:
        """Read and check the `on` key pair.

        Each side reads `on=` if the call gives it. Else the receiver
        reads `left_on=` and `other` reads `right_on=`.

        Returns:
            The key of each side, or a `Refusal`. The reason is
            `DYNAMIC_NAME` if a key is not a literal string. It is
            `REPORTED` if a frame does not have its key or the data types
            cannot join, after the error.
        """
        on_given = self._args.is_provided("on")
        left_param = "on" if on_given else "left_on"
        right_param = "on" if on_given else "right_on"
        left = not_none(self._args.literal_str(left_param), reason=Reason.DYNAMIC_NAME)
        right = not_none(self._args.literal_str(right_param), reason=Reason.DYNAMIC_NAME)
        not_falsy(self._left_key_exists(left))
        not_falsy(self._right_key_exists(right))
        not_refused(self._on_key_mismatch(left, right))
        return left, right

    @cached_property
    @explains(Reason.REPORTED)
    def _validated_by_pair(self) -> tuple[list[str], list[str]]:
        """Read and check the `by` keys of both sides.

        Each side reads `by=` if the call gives it. Else the receiver reads
        `by_left=` and `other` reads `by_right=`. A side with neither
        argument has no `by` keys.

        Returns:
            The keys of each side, empty lists if the call gives none, or
            a `Refusal`. The reason is `DYNAMIC_NAME` if a side is not a
            literal string or list of strings. It is `REPORTED` if the
            lists differ in length, a frame does not have a key, or a key
            pair has two different data types, after the error.
        """
        by_given = self._args.is_provided("by")
        left_param = "by" if by_given else "by_left"
        right_param = "by" if by_given else "by_right"
        left_names = self._args.literal_str_list_or_empty(left_param)
        right_names = self._args.literal_str_list_or_empty(right_param)
        left = not_none(left_names, reason=Reason.DYNAMIC_NAME)
        right = not_none(right_names, reason=Reason.DYNAMIC_NAME)
        not_falsy(self._by_keys_equal_length(left, right))
        not_falsy(self._left_by_keys_exist(left))
        not_falsy(self._right_by_keys_exist(right))
        not_refused(self._by_key_mismatch(left, right))
        return left, right

    def _left_column(self, name: str) -> ColumnRef | None:
        """Find a declared column of the receiver by name.

        Args:
            name: The name of the column.

        Returns:
            The column, or None if the receiver has no such column.
        """
        return self._left_index.get(name)

    def _right_column(self, name: str) -> ColumnRef | None:
        """Find a declared column of `other` by name.

        Args:
            name: The name of the column.

        Returns:
            The column, or None if `other` has no such column.
        """
        return self._right_index.get(name)

    @cached_property
    def _left_index(self) -> dict[str, ColumnRef]:
        return self._indexed(self._left_columns)

    @cached_property
    def _right_index(self) -> dict[str, ColumnRef]:
        return self._indexed(self._right_columns)

    def _indexed(self, columns: list[ColumnRef]) -> dict[str, ColumnRef]:
        """Key a list of columns by name.

        Args:
            columns: The columns of one frame.

        Returns:
            A dictionary from each name to its column. A column with no
            literal name is left out.
        """
        return {column.name: column for column in columns if column.name is not None}

    def _left_key_exists(self, name: str) -> bool:
        """Answer whether the receiver has its `on` key, and report it if not.

        Args:
            name: The `on` key of the receiver.

        Returns:
            True if the receiver declares the column. Else False, after
            an `UnknownColumnError` is reported.
        """
        if self._left_column(name) is None:
            UnknownColumnError(name, frame="left").fail(self._ctx)
            return False
        return True

    def _right_key_exists(self, name: str) -> bool:
        """Answer whether `other` has its `on` key, and report it if not.

        Args:
            name: The `on` key of `other`.

        Returns:
            True if `other` declares the column. Else False, after an
            `UnknownColumnError` is reported.
        """
        if self._right_column(name) is None:
            UnknownColumnError(name, frame="right").fail(self._ctx)
            return False
        return True

    def _on_key_mismatch(self, left_name: str, right_name: str) -> Refusal | None:
        """Check that the data types of the `on` key pair can join.

        Args:
            left_name: The `on` key of the receiver.
            right_name: The `on` key of `other`.

        Returns:
            None if the data types can join: equal, one of them
            `Unchecked`, or both in the same integer or float family.
            Else the `Refusal` that `_mismatch` gives.
        """
        planned = PlannedType()
        left = self._left_index[left_name].dtype
        right = self._right_index[right_name].dtype
        for read in (planned.stored, planned.run):
            left_dtype, right_dtype = get_proper_type(read(left)), get_proper_type(read(right))
            if not self._on_compatibility.compatible(left_dtype, right_dtype):
                return self._mismatch(left_name, left_dtype, right_dtype)
        return None

    def _by_keys_equal_length(self, left_names: list[str], right_names: list[str]) -> bool:
        """Answer whether the two lists of `by` keys have the same length, and report it if not.

        `by=` gives the same list to both sides, so only `by_left=` and
        `by_right=` can differ in length. polars then raises
        `InvalidOperationError`.

        Args:
            left_names: The `by` keys of the receiver.
            right_names: The `by` keys of `other`.

        Returns:
            True if the lengths are equal. Else False, after an
            `InvalidCallShapeError` is reported.
        """
        if len(left_names) != len(right_names):
            error = InvalidCallShapeError(
                "join_asof() with by_left=/by_right= of different lengths raises "
                "InvalidOperationError: expected equal number of columns in "
                "'by_left' and 'by_right' in 'asof_join'",
            )
            error.fail(self._ctx)
            return False
        return True

    def _left_by_keys_exist(self, names: list[str]) -> bool:
        """Answer whether the receiver has each of its `by` keys, and report the first it lacks.

        Args:
            names: The `by` keys of the receiver.

        Returns:
            True if the receiver declares each column. Else False, after
            an `UnknownColumnError` for the first missing one.
        """
        for name in names:
            if self._left_column(name) is None:
                UnknownColumnError(name, frame="left").fail(self._ctx)
                return False
        return True

    def _right_by_keys_exist(self, names: list[str]) -> bool:
        """Answer whether `other` has each of its `by` keys, and report the first it lacks.

        Args:
            names: The `by` keys of `other`.

        Returns:
            True if `other` declares each column. Else False, after an
            `UnknownColumnError` for the first missing one.
        """
        for name in names:
            if self._right_column(name) is None:
                UnknownColumnError(name, frame="right").fail(self._ctx)
                return False
        return True

    def _by_key_mismatch(self, left_names: list[str], right_names: list[str]) -> Refusal | None:
        """Check that each `by` key pair has equal data types.

        The rule for `by` is stricter than the rule for `on`. A pair of
        `Int32` and `Int64` joins as an `on` key, and fails as a `by` key.

        Args:
            left_names: The `by` keys of the receiver.
            right_names: The `by` keys of `other`, paired in order with
                `left_names`.

        Returns:
            None if each pair has equal data types, or one side is
            `Unchecked`. Else the `Refusal` that `_mismatch` gives for the
            first other pair.
        """
        planned = PlannedType()
        for left_name, right_name in zip(left_names, right_names, strict=True):
            left = self._left_index[left_name].dtype
            right = self._right_index[right_name].dtype
            for read in (planned.stored, planned.run):
                left_dtype, right_dtype = get_proper_type(read(left)), get_proper_type(read(right))
                if not self._by_compatibility.compatible(left_dtype, right_dtype):
                    return self._mismatch(left_name, left_dtype, right_dtype)
        return None

    def _mismatch(self, name: str, left_dtype: object, right_dtype: object) -> Refusal:
        """Report a `JoinKeyDtypeMismatchError` for a key pair that cannot join.

        The error needs two known data types. If a side is not an
        instance type, for example a type variable in a generic function,
        the checker cannot tell a real mismatch from a type it does not
        see. It then reports nothing and gives up with `CHECKER_GAP`.

        Args:
            name: The name of the key.
            left_dtype: The data type of the key in the receiver.
            right_dtype: The data type of the key in `other`.

        Returns:
            A `Refusal` with `REPORTED` after the error, or with
            `CHECKER_GAP` if a side is not an instance type.
        """
        if not (isinstance(left_dtype, Instance) and isinstance(right_dtype, Instance)):
            return Refusal(Reason.CHECKER_GAP)
        error = JoinKeyDtypeMismatchError(
            name,
            left=left_dtype.type.name,
            right=right_dtype.type.name,
        )
        error.fail(self._ctx)
        return Refusal(Reason.REPORTED)


class _JoinAsofResultMerger:
    """Build the result columns of one `.join_asof()` call from the columns of both frames.

    The layout is that of a left join. The columns of the receiver come
    first, in order. A merged key keeps its left name and position. Then
    come the right columns that are not merged keys, each renamed with the
    suffix if its name is already in the result. The `on` key merges only
    if `coalesce` is true, and the `by` keys always merge. One lookup of
    merged pairs holds both kinds, so the merge of a column does not need
    to know which kind of key it is.

    A merged key never widens to a common data type. It keeps the data
    type of the left column, unless that is `Unchecked` and the right one
    is known. Then it takes the known data type.
    """

    def __init__(
        self,
        plan: _JoinAsofPlan,
        suffix: str,
        api: object,
        unchecked: UncheckedCompatibility,
    ) -> None:
        """Keep the plan, the suffix and the tools for one merge.

        Args:
            plan: What the hook read from the call.
            suffix: The text that a right name takes if it clashes.
            api: The mypy plugin interface, which builds a renamed or
                retyped column.
            unchecked: Picks the known data type over `Unchecked` in a
                merged key pair.
        """
        self._plan = plan
        self._suffix = suffix
        self._api = api
        self._unchecked = unchecked

    @property
    def merged_columns(self) -> list[ColumnRef]:
        """The columns of the result: the left columns, then the right columns that stay.

        Each merged key can take the known data type of its right column.
        Each right column that clashes with a left name takes the suffix.
        """
        left_columns = self._plan.left_columns
        resolved_left = [self._resolved_left_column(column) for column in left_columns]
        left_names = {column.name for column in resolved_left if column.name is not None}
        return [*resolved_left, *self._non_key_right_columns(left_names)]

    def _resolved_left_column(self, column: ColumnRef) -> ColumnRef:
        """Give one left column, with the known data type of the pair if it is a merged key.

        A merged key takes the data type of its right key only if its own
        data type is `Unchecked` and the right one is known. A pair such as
        `Int32` and `Int64` keeps the left data type, because an as-of join
        never widens a key.

        Args:
            column: A column of the receiver.

        Returns:
            `column` unchanged if it is not a merged key, or if it keeps
            its data type. Else a copy with the data type of its right key.
        """
        if column.name is None or column.name not in self._coalesced_pair_by_name:
            return column
        other = self._right_index[self._coalesced_pair_by_name[column.name]]
        resolved = self._unchecked.resolved(column.dtype, other.dtype)
        if resolved is None or get_proper_type(resolved) == get_proper_type(column.dtype):
            return column
        return ColumnRef.build(column.name, resolved, column, self._api)

    def _non_key_right_columns(self, collision_names: AbstractSet[str]) -> list[ColumnRef]:
        """List the right columns that stay in the result, with the suffix on a clash.

        A right key that merges into its left key does not appear in the
        result.

        Args:
            collision_names: The names of the left columns in the result.

        Returns:
            Each right column that is not a merged key, in order. A column
            whose name is in `collision_names` takes the suffix. A column
            with no literal name is left out.
        """
        excluded = set(self._coalesced_pair_by_name.values())
        result_columns: list[ColumnRef] = []
        for column in self._plan.right_columns:
            if column.name is None or column.name in excluded:
                continue
            new_name = CollisionSuffix.renamed(column.name, collision_names, self._suffix)
            if new_name == column.name:
                result_columns.append(column)
                continue
            result_columns.append(column.with_name(new_name, self._api))
        return result_columns

    @cached_property
    def _right_index(self) -> dict[str, ColumnRef]:
        columns = self._plan.right_columns
        return {column.name: column for column in columns if column.name is not None}

    @cached_property
    def _coalesced_pair_by_name(self) -> dict[str, str]:
        """Map each left key that merges to its right key.

        The `by` pairs always merge. The `on` pair merges only if
        `coalesce` is true and both keys have one name.
        """
        pairs = dict(zip(self._plan.by_left_names, self._plan.by_right_names, strict=True))
        if self._plan.coalesce and self._plan.on_name == self._plan.right_on_name:
            pairs[self._plan.on_name] = self._plan.right_on_name
        return pairs


class JoinAsofHook:
    """Give `.join_asof()` on a typed frame the columns of its result, and report calls that fail.

    Unlike `.join()`, an as-of join has no `how=` argument. It always adds
    the columns of `other`. The hook reads the call into a plan, merges the
    declared columns, and checks that no two result columns share a name.

    Either frame can be an open frame, whose type does not name all of its
    columns. A hidden column can then clash with a declared column and
    change which columns take the suffix. `JoinRemainderStep` decides the
    remainder of the result, or reports that the result depends on the
    columns that the caller passes.

    It is a mypy method hook in the plugin layer. If it cannot type the
    call, it falls back through `UntypedFallback` to the plain frame that
    polars declares.
    """

    DEFAULT_SUFFIX: Final[str] = "_right"

    def __init__(self, untyped_fallback: UntypedFallback, plugin: Plugin) -> None:
        """Build the key rules, the duplicate check and the remainder step.

        Args:
            untyped_fallback: Gives the plain frame for a call that the
                hook cannot type, and reports it unless the
                `allow_unproven` setting lets it pass.
            plugin: The mypy plugin, which the `on` key rule and the
                remainder step need to find classes.
        """
        self._untyped = untyped_fallback
        self._unchecked = UncheckedCompatibility()
        self._on_compatibility = JoinKeyCompatibility(self._unchecked, plugin)
        self._duplicate_check = DuplicateNameCheck("join_asof")
        self._remainder_step = JoinRemainderStep(plugin, untyped_fallback, "join_asof")

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.join_asof()` call.

        Args:
            ctx: The context that mypy gives for `.join_asof()`.

        Returns:
            A typed frame with the merged columns. If the hook cannot type
            the call, the plain `pl.DataFrame` or `pl.LazyFrame`.
        """
        plan = _JoinAsofPlan.from_context(ctx, self._on_compatibility, self._unchecked)
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                "join_asof",
                Frame.from_proper_type(ctx.type),
                _JoinAsofPlan.other_frame(ctx),
            )
        result = self._merged_result(ctx, plan)
        if isinstance(result, Refusal):
            return self._untyped.resolve(ctx, result.reason, "join_asof", plan.frame)
        return result

    @explains(Reason.REPORTED)
    def _merged_result(self, ctx: MethodContext, plan: _JoinAsofPlan) -> MypyType:
        """Build the result frame from a plan.

        Args:
            ctx: The context that mypy gives for `.join_asof()`.
            plan: What the hook read from the call.

        Returns:
            The result frame, or a `Refusal`. The reason is `DYNAMIC_NAME`
            if `suffix=` is not a literal and a column takes it. It is
            `REPORTED` if two result columns share a name, or if a
            remainder makes the renames depend on the caller, after the
            error. A renamed right column can still clash, for example if
            the receiver already has a column `x_right`.
        """
        suffix = not_none(self._resolved_suffix(ctx, plan), reason=Reason.DYNAMIC_NAME)
        merger = _JoinAsofResultMerger(plan, suffix, ctx.api, self._unchecked)
        checked_columns = merger.merged_columns
        names = [column.name for column in checked_columns]
        not_falsy(not self._duplicate_check.fails(ctx, names))
        outcome = not_none(self._remainder_step.decided(ctx, self._sides(plan, suffix, ctx.api)))
        rebuilt = plan.frame.rebuilt(checked_columns, ctx.api, rest=outcome.slot)
        return rebuilt.as_instance()

    def _sides(self, plan: _JoinAsofPlan, suffix: str, api: object) -> JoinSides:
        """Sort the names of one call by what the join does to each, for the remainder step.

        The method merges the columns once with a probe suffix that no
        real name can hold. Each output name that ends with the probe
        suffix is then surely a right column that the join renames.

        Args:
            plan: What the hook read from the call.
            suffix: The real suffix that a right name takes on a clash.
            api: The mypy plugin interface, for the probe merge.

        Returns:
            The sorted names, with the real suffix.
        """
        merger = _JoinAsofResultMerger(plan, ProbeNames.SUFFIX, api, self._unchecked)
        names = [column.name for column in merger.merged_columns if column.name is not None]
        declared = DeclaredNames(
            self._declared_names(plan.left_columns),
            self._declared_names(plan.right_columns),
        )
        return JoinSides.from_probe(
            ProbeNames(names),
            plan.left_rest,
            plan.right_rest,
            declared,
            suffix,
        )

    def _declared_names(self, columns: list[ColumnRef]) -> frozenset[str]:
        """Collect the names of the declared columns of one frame.

        Args:
            columns: The declared columns of one frame.

        Returns:
            The names. A column with no literal name is left out.
        """
        return frozenset(column.name for column in columns if column.name is not None)

    def _resolved_suffix(self, ctx: MethodContext, plan: _JoinAsofPlan) -> str | None:
        """Choose the suffix for the merge.

        A `suffix=` that is not a literal matters only if a right column
        clashes and takes it. If no column clashes, the text of the suffix
        does not appear in the result, so the call can still be typed.

        Args:
            ctx: The context that mypy gives for `.join_asof()`.
            plan: What the hook read from the call.

        Returns:
            The literal suffix, or `"_right"` if the call omits it. If the
            suffix is not a literal: `"_right"` if no column clashes,
            else None.
        """
        literal = CallArgs(ctx).literal_str("suffix", default=self.DEFAULT_SUFFIX)
        if literal is not None:
            return literal
        if self._collides(plan, ctx.api):
            return None
        return self.DEFAULT_SUFFIX

    def _collides(self, plan: _JoinAsofPlan, api: object) -> bool:
        """Answer whether the merge renames any column, whatever the suffix is.

        The method merges twice, with two different probe suffixes. If the
        output names differ, at least one right column took the suffix.

        Args:
            plan: What the hook read from the call.
            api: The mypy plugin interface, for the two merges.

        Returns:
            True if a right column takes the suffix.
        """
        probe_a = _JoinAsofResultMerger(
            plan,
            "\0astrolabe_join_asof_probe_a",
            api,
            self._unchecked,
        ).merged_columns
        probe_b = _JoinAsofResultMerger(
            plan,
            "\0astrolabe_join_asof_probe_b",
            api,
            self._unchecked,
        ).merged_columns
        return [column.name for column in probe_a] != [column.name for column in probe_b]
