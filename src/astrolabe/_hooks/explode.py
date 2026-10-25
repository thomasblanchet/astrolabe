"""The hook for `.explode()`, which gives each named `List` or `Array` column its inner data type.

`.explode()` puts each element of a list on its own row. The columns stay
in their places. Each exploded column takes the data type of its
elements, so a `List[Int64]` column becomes an `Int64` column. Without
this hook, mypy keeps the old type, and the checker then believes that
the column still holds lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TYPE_CHECKING

from mypy.types import get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.frame import Frame, KnownColumns
from astrolabe._hooks.domain.requested_columns import RequestedColumns
from astrolabe._hooks.errors import InvalidCallShapeError, WrongColumnShapeError
from astrolabe._hooks.propagate import (
    Refusal,
    explains,
    not_falsy,
    not_none,
    not_refused,
)
from astrolabe.frame import Array as _ArrayMarker
from astrolabe.frame import List as _ListMarker

if TYPE_CHECKING:
    from mypy.plugin import MethodContext

    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _ExplodePlan:
    """The inputs of one `.explode()` call, read from the call once.

    Attributes:
        frame: The receiver.
        requested: The names of the columns to explode, in call order.
        known: The declared columns of the receiver, and its remainder: the
            columns that its type does not name.
    """

    frame: Frame
    requested: list[str]
    known: KnownColumns

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(cls, ctx: MethodContext) -> _ExplodePlan:
        """Read the plan from an `.explode()` call.

        The receiver can be an open frame, whose type does not name all of its
        columns. Only the named columns change, so the undeclared ones keep
        their data types and the result keeps the remainder.

        The method reads the `columns` and `*more_columns` arguments as
        literal names. Polars raises for `.explode()` with no columns,
        whatever the data, so the method reports `InvalidCallShapeError` for
        a call that names no column.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The plan, or a refusal with its reason. The reason is a checker gap
            if the receiver is not a typed frame. It is a dynamic name if an
            argument is not a literal string. It is `Reason.REPORTED` if the
            call names no column at all.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        args = CallArgs(ctx)
        names = not_none(args.literal_str_list("columns"), reason=Reason.DYNAMIC_NAME)
        more_names = not_none(args.vararg_str_list("more_columns"), reason=Reason.DYNAMIC_NAME)
        requested = names + more_names
        if not requested:
            InvalidCallShapeError("explode() with no columns to explode always raises").fail(ctx)
        not_falsy(requested, reason=Reason.REPORTED)
        return cls(frame, requested, known)


class ExplodeHook:
    """Give the type of a frame after `.explode()`.

    For each requested name, the hook finds the column of the receiver and
    puts in its place a column of the same name with the inner data type.
    It reports a name that the frame does not have, and a column that is
    not a `List` or an `Array`. An `Unchecked` column stays `Unchecked`: the
    checker does not know its data type, so it can neither unwrap it nor
    report it.

    The receiver can be an open frame, with columns that its type does not
    name, its remainder. Only the named columns change, so the result keeps
    the remainder. A name that the frame does not declare is an error if the
    remainder cannot hold it, and goes through the `allow_unproven` gate if
    it can.

    The hook sits in the plugin layer. It reads the receiver through
    `Frame`. If it cannot read the call, it falls back: it gives the plain
    frame type in place of a typed result.
    """

    LIST_TARGET: Final[Target] = Target.of(_ListMarker)
    ARRAY_TARGET: Final[Target] = Target.of(_ArrayMarker)

    def __init__(self, untyped_fallback: UntypedFallback) -> None:
        """Keep the fallback for a call that the hook cannot read.

        Args:
            untyped_fallback: The fallback that gives the plain frame type,
                subject to the `allow_unproven` setting.
        """
        self._untyped = untyped_fallback
        self._unchecked = UncheckedCompatibility()
        self._requested_columns = RequestedColumns(untyped_fallback, "explode")

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the result type of one `.explode()` call.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The frame with each requested column changed to its inner data type.
            If the hook cannot read the call, or it reported an error, the plain
            `pl.DataFrame` or `pl.LazyFrame`. See `UntypedFallback`.
        """
        plan = _ExplodePlan.from_context(ctx)
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                "explode",
                Frame.from_proper_type(ctx.type),
            )
        result_columns = self._exploded(ctx, plan)
        if isinstance(result_columns, Refusal):
            return self._untyped.resolve(ctx, result_columns.reason, "explode", plan.frame)
        rest = plan.known.rest
        return plan.frame.rebuilt(result_columns, ctx.api, rest=rest).as_instance()

    @explains(Reason.REPORTED)
    def _exploded(self, ctx: MethodContext, plan: _ExplodePlan) -> list[ColumnRef]:
        """Replace each requested column with its exploded form, in its own place.

        A name that the frame does not declare is an error if the remainder
        cannot hold it, and goes through the `allow_unproven` gate if it can.
        The exploded form of a column has its inner data type in place of the
        `List` or `Array`. A requested column that is neither gets a
        `WrongColumnShapeError`.

        Args:
            ctx: The mypy context of the method call.
            plan: The inputs of the call.

        Returns:
            The declared columns of the receiver with each requested column
            replaced, or a refusal: `REPORTED` after an error, or
            `REMAINDER_MAY_HOLD` for a name that only the remainder can hold.
        """
        not_refused(self._requested_columns.resolved(ctx, plan.requested, plan.known))
        by_name = plan.known.by_name()
        names = [column.name for column in plan.known.columns]
        result_columns = list(plan.known.columns)
        for name in plan.requested:
            matched = checked_not_none(by_name.get(name))
            inner = self._unwrap_list_or_array(matched.dtype)
            if inner is None:
                error = WrongColumnShapeError.on_column(
                    name,
                    expected="List/Array",
                    operation="explode",
                )
                error.fail(ctx)
            exploded = ColumnRef.build(name, not_none(inner), matched, ctx.api)
            result_columns[names.index(name)] = exploded
        return result_columns

    def _unwrap_list_or_array(self, dtype: MypyType) -> MypyType | None:
        """Give the data type that one column has after the explode.

        Args:
            dtype: The data type of the column.

        Returns:
            The same type if it is `Unchecked`, because the real type can be a
            `List` or an `Array` and the checker has nothing to check. The inner
            data type for a `List[...]` or an `Array[..., ...]`. Otherwise None.

        Example:
            >>> fixture = new_type_fixture()
            >>> inner = int_dtype(fixture)
            >>> column_dtype = list_dtype(fixture, inner)
            >>> column_dtype
            astrolabe.frame.List[polars.datatypes.classes.Int64]
            >>> hook = ExplodeHook(untyped_fallback())
            >>> hook._unwrap_list_or_array(column_dtype) is inner
            True
            >>> hook._unwrap_list_or_array(inner) is None
            True
            >>> unresolved = unchecked_instance(fixture)
            >>> hook._unwrap_list_or_array(unresolved) is unresolved
            True
        """
        proper = get_proper_type(dtype)
        if self._unchecked.matches(proper):
            return proper
        if not (self.LIST_TARGET.matches(proper) or self.ARRAY_TARGET.matches(proper)):
            return None
        if not proper.args:  # pragma: no cover
            # mypy gives a generic instance all of its type arguments,
            # with `Any` in place of a missing one. A `List` or an `Array`
            # instance therefore always has at least one argument.
            return None
        return proper.args[0]
