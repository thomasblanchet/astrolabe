"""Find the columns that an operation names, such as the columns of `drop("a", "b")`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.errors import UndeclaredColumnError, UnknownColumnError
from astrolabe._hooks.propagate import Refusal, explains, not_none, not_refused

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.plugin import MethodContext

    from astrolabe._hooks.calls.call_args import CallArgs
    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.unproven_gate import UnprovenGate


class RequestedColumns:
    """Look up the columns that an operation names in its receiver, and report a missing one.

    Many operations take column names, for example `drop`, `unpivot`, `pivot` and `df["a"]`.
    Each needs the declared column for each name, and an error for a name that the frame does
    not have. This class holds that one rule, so every such operation reports in the same way.

    The receiver can be an open frame. An open frame has a remainder: columns that its type
    does not name. A declared name still resolves, because the remainder cannot change whether
    a declared column exists or what its data type is. A name that is not declared gets one of
    two answers. If the remainder cannot hold the name, the name is surely missing, and the
    class reports an error. If the remainder can hold it, the call works for some callers and
    fails for others. The class then reports through an `UnprovenGate`, so the
    `allow_unproven` setting decides.
    """

    def __init__(self, gate: UnprovenGate, operation: str) -> None:
        """Keep the gate for a name that the remainder can hold, and the name of the operation.

        Args:
            gate: The gate that decides whether such a name gets an error.
            operation: The name of the operation. The setting can give each operation its own
                level.
        """
        self._gate = gate
        self._operation = operation

    @explains(Reason.REPORTED)
    def resolved(
        self,
        ctx: MethodContext,
        names: Sequence[str],
        known: KnownColumns,
    ) -> list[ColumnRef]:
        """Give the declared column for each name, in the order of the names.

        The method checks every name before it gives up on one that the setting lets pass. A
        name that is surely missing later in the list therefore still gets its error.

        Args:
            ctx: The mypy context of the operation.
            names: The column names, in call order.
            known: The declared columns of the receiver, and its remainder.

        Returns:
            The columns in the order of `names`. Otherwise a refusal: `REPORTED` if a name got
            an error, or `REMAINDER_MAY_HOLD` if the only missing names are ones that the
            remainder can hold and the setting lets them pass.
        """
        by_name = known.by_name()
        found: list[ColumnRef | None] = []
        for name in names:
            column = by_name.get(name)
            if column is None:
                column = not_refused(self._missing(ctx, name, known))
            found.append(column)
        return [not_none(column, reason=Reason.REMAINDER_MAY_HOLD) for column in found]

    def _missing(self, ctx: MethodContext, name: str, known: KnownColumns) -> Refusal | None:
        """Report a name that the receiver does not declare.

        The operation then fails at run time for at least one caller. If the remainder cannot
        hold the name, the method reports `UnknownColumnError`. If the remainder can hold it,
        the method reports `UndeclaredColumnError` through the gate, unless the setting lets it
        pass.

        Args:
            ctx: The mypy context of the operation.
            name: The column name, which is not declared.
            known: The declared columns of the receiver, and its remainder.

        Returns:
            None if the setting lets the name pass. Otherwise, the refusal `REPORTED`.
        """
        if RemainderKnowledge().proves_absent(known.rest, name):
            UnknownColumnError(name).fail(ctx)
            return Refusal(Reason.REPORTED)
        return self._gate.report(ctx, UndeclaredColumnError(name, self._operation))

    @staticmethod
    def remaining_names(known: KnownColumns, excluded: set[str]) -> list[str]:
        """Give the names of the declared columns, less some names.

        Args:
            known: The declared columns of the receiver, and its remainder.
            excluded: The names to leave out.

        Returns:
            The name of each declared column in order. A column with no literal name, or with
            a name in `excluded`, is left out.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> tax = column_ref(fixture, "tax", int_dtype(fixture))
            >>> RequestedColumns.remaining_names(known_columns([price, tax]), {"tax"})
            ['price']
        """
        return [
            column.name
            for column in known.columns
            if column.name is not None and column.name not in excluded
        ]

    @staticmethod
    def names_or_remaining(
        args: CallArgs,
        param: str,
        known: KnownColumns,
        excluded: set[str],
    ) -> list[str] | None:
        """Read the names for one parameter, or give every other column if the caller passes none.

        For some parameters, no argument means "every column that no other parameter names".
        Examples are `on` of `unpivot` and `values` of `pivot`. This method gives that list for
        a closed frame. For an open frame it cannot, because the remainder holds columns that
        have no known name.

        Args:
            args: The arguments of the call.
            param: The name of the parameter.
            known: The declared columns of the receiver, and its remainder.
            excluded: The names that other parameters take, which the default leaves out.

        Returns:
            The literal names of the argument, if the caller passes one. Otherwise, every
            declared column not in `excluded`, in order. None if the argument is not a list of
            known strings, or if there is no argument and the receiver is an open frame.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> tax = column_ref(fixture, "tax", int_dtype(fixture))
            >>> known = known_columns([price, tax])
            >>> args = call_args(["on"], [[]])
            >>> RequestedColumns.names_or_remaining(args, "on", known, {"tax"})
            ['price']
        """
        if not args.is_omitted_or_none(param):
            return args.literal_str_list(param)
        if known.rest is not None:
            return None
        return RequestedColumns.remaining_names(known, excluded)

    @staticmethod
    def unread_reason(args: CallArgs, param: str) -> Reason:
        """Give the reason why `names_or_remaining` gave no names for one parameter.

        Args:
            args: The arguments of the call.
            param: The name of the parameter.

        Returns:
            `DYNAMIC_NAME` if the caller passes names that are not known strings.
            `CHECKER_GAP` if the caller passes nothing and the receiver is an open frame, so
            the checker cannot list the default columns.

        Example:
            >>> RequestedColumns.unread_reason(call_args(["on"], [[]]), "on").name
            'CHECKER_GAP'
            >>> passed = call_args(["on"], [[str_literal_node("a")]])
            >>> RequestedColumns.unread_reason(passed, "on").name
            'DYNAMIC_NAME'
        """
        if args.is_omitted_or_none(param):
            return Reason.CHECKER_GAP
        return Reason.DYNAMIC_NAME
