"""Type `.update(other)` on a typed frame: each column that `other` also has takes a common type.

`.update()` joins the receiver with `other` and fills the values of the
receiver from `other`. It does not check that the two frames agree. The
result always has exactly the column names of the receiver, so a column
that only `other` has is dropped, even with `how="full"`. A column that
both frames have takes the supertype of the two data types, that is the
smallest data type that holds the values of both. For example, `Int64`
updated from `Float64` gives `Float64`, and updated from `String` gives
`String`. A join key is never replaced, so it keeps the data type of the
receiver: `Int64` updated from `Int32` on that key stays `Int64`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.dtype_promotion import DTypePromotion
from astrolabe._hooks.domain.frame import Frame, KnownColumns
from astrolabe._hooks.propagate import Refusal, explains, not_none, propagates

if TYPE_CHECKING:
    from astrolabe._hooks.domain.frame import PluginLookup
    from mypy.plugin import MethodContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _UpdatePlan:
    """Hold what the hook reads from one `.update()` call.

    Attributes:
        frame: The receiver frame.
        left_columns: The columns of the receiver, all declared.
        right_columns: The columns of `other`, keyed by name.
        right_keys: The names of the join keys on the side of `other`. A
            key column never gives values, so it replaces nothing.
    """

    frame: Frame
    left_columns: list[ColumnRef]
    right_columns: dict[str, ColumnRef]
    right_keys: frozenset[str]

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(cls, ctx: MethodContext) -> _UpdatePlan:
        """Read the two frames and the join keys of an `.update()` call.

        Both frames must be closed, that is their types must name all of
        their columns. If either one has a remainder of columns that its
        type does not name, a hidden column of `other` can replace a
        hidden column of the receiver. The hook cannot name either
        column, so it gives up.

        `right_on=` names the keys of `other` alone, and wins over `on=`,
        which names the keys of both frames. If the call names no key,
        polars matches rows by position, and each shared column gives
        values.

        Args:
            ctx: The context that mypy gives for `.update()`.

        Returns:
            The plan, or a `Refusal`. The reason is `DYNAMIC_NAME` if the
            join keys are not literal names. It is `CHECKER_GAP` if either
            frame is not a closed typed frame.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        left_columns = not_none(frame.columns)
        args = CallArgs(ctx)
        other_type = not_none(args.node_type("other"))
        other = not_none(Frame.from_proper_type(other_type))
        right_columns = not_none(other.columns)
        key_argument = "right_on" if args.is_provided("right_on") else "on"
        right_keys: frozenset[str] = frozenset()
        if args.is_provided(key_argument):
            names = not_none(args.literal_str_list(key_argument), reason=Reason.DYNAMIC_NAME)
            right_keys = frozenset(names)
        by_name = KnownColumns(right_columns, None).by_name()
        return cls(frame, left_columns, by_name, right_keys)


class UpdateHook:
    """Give `.update(other)` on a typed frame a result with the column names of the receiver.

    polars declares that `.update()` returns a plain frame, so without this
    hook the checker loses every column at this call. The hook keeps each
    column of the receiver, in order. A column that
    `other` also has, and does not use as a join key, takes the supertype
    of the two data types.

    `DTypePromotion` computes the supertype. It models only part of the
    rules of polars. For a pair that it does not model, that one column
    gets the data type `Unchecked`, which means that its name is known and
    its data type is not. Every other column keeps an exact type.

    It is a mypy method hook in the plugin layer. If it cannot read the
    call, it falls back through `UntypedFallback` to the plain frame that
    polars declares.
    """

    def __init__(
        self,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
        plugin: PluginLookup,
    ) -> None:
        """Keep the two fallbacks, and build the supertype rule.

        Args:
            untyped_fallback: Gives the plain frame for a call that the
                hook cannot type, and reports it unless the
                `allow_unproven` setting lets it pass.
            unchecked_fallback: Gives the data type `Unchecked` to a column
                whose supertype the hook does not know, and reports it
                unless the setting lets it pass.
            plugin: The mypy plugin, which the supertype rule needs to
                find data type classes.
        """
        self._untyped = untyped_fallback
        self._unchecked = unchecked_fallback
        self._promotion = DTypePromotion(UncheckedCompatibility(), plugin)

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.update()` call.

        Args:
            ctx: The context that mypy gives for `.update()`.

        Returns:
            The schema of the receiver, with each replaced column at the
            supertype of both frames. If the hook cannot read the call,
            the plain `pl.DataFrame` or `pl.LazyFrame`.
        """
        plan = _UpdatePlan.from_context(ctx)
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                "update",
                Frame.from_proper_type(ctx.type),
            )
        columns = self._result_columns(ctx, plan)
        if columns is None:  # pragma: no cover
            # `Unchecked` is declared in the same module as the typed
            # frames. The hook runs only on a typed frame, so mypy has
            # already loaded that module.
            return self._untyped.resolve(ctx, Reason.CHECKER_GAP, "update", plan.frame)
        return plan.frame.rebuilt(columns, ctx.api).as_instance()

    @propagates(None)
    def _result_columns(self, ctx: MethodContext, plan: _UpdatePlan) -> list[ColumnRef] | None:
        """Give each column of the receiver as it is after the update.

        A column that `other` replaces takes the supertype of the two data
        types. A pair that `DTypePromotion` does not model is a case that
        the checker does not handle, not a fault in the code of the user.
        So that column gets a new `Unchecked` with the reason `CHECKER_GAP`.

        Args:
            ctx: The context that mypy gives for `.update()`.
            plan: What the hook read from the call.

        Returns:
            The columns of the receiver in order, each one that `other`
            replaces at the supertype. None if mypy cannot find
            `Unchecked`.
        """
        columns: list[ColumnRef] = []
        for column in plan.left_columns:
            source = self._source(plan, column)
            if source is None:
                columns.append(column)
                continue
            name = not_none(column.name)
            dtype = self._promotion.promoted_dtypes(column.dtype, source.dtype)
            if dtype is None:
                unchecked = self._unchecked.substituted(ctx, Reason.CHECKER_GAP, "update", name)
                dtype = not_none(unchecked)
            columns.append(ColumnRef.from_name_and_dtype(name, dtype))
        return columns

    def _source(self, plan: _UpdatePlan, column: ColumnRef) -> ColumnRef | None:
        """Find the column of `other` that gives values to one column of the receiver.

        A join key only matches rows, and never gives a value. So a key
        column is not a source.

        Args:
            plan: What the hook read from the call.
            column: One column of the receiver.

        Returns:
            The column of `other` with the same name. None if `other` has
            no such column, or uses it as a join key.

        Example:
            >>> fixture = new_type_fixture()
            >>> key = column_ref(fixture, "k", int_dtype(fixture))
            >>> value = column_ref(fixture, "v", int_dtype(fixture))
            >>> other = column_ref(fixture, "v", str_dtype(fixture))
            >>> hook = UpdateHook(
            ...     untyped_fallback(), unchecked_fallback(), new_plugin(fixture)
            ... )
            >>> plan = update_plan(fixture, [key, value], [key, other], ["k"])
            >>> hook._source(plan, value).dtype
            polars.datatypes.classes.String
            >>> hook._source(plan, key) is None
            True
        """
        if column.name in plan.right_keys:
            return None
        return plan.right_columns.get(not_none(column.name))
