"""Check that a frame holds each column that a declaration names, with a data type that agrees.

A declared frame type is a promise about some columns. A function parameter
`TypedDataFrame[Trades, Rest]` promises its body the columns of `Trades`,
and a return type `TypedDataFrame[WithTotal, InferredRemainder]` promises
its callers the columns of `WithTotal`. Code that checks a frame against
such a promise needs the same three answers each time, so they live here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.errors import (
    MissingRequiredColumnError,
    UndeclaredColumnError,
    WrongDtypeError,
)

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext, FunctionSigContext, MethodContext

    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.unproven_gate import UnprovenGate

    # The context of a call, where an error points.
    type CallContext = MethodContext | FunctionContext | FunctionSigContext


class DeclaredColumnsFit:
    """Report each declared column that a frame lacks or holds with a data type that does not agree.

    The frame can be open, that is, it can have a remainder: columns that
    its type does not name. A declared column that the frame does not
    declare is then missing for sure only if the remainder cannot hold
    the name. If the remainder can hold it, the frame fits for some
    callers and not for others, so the report goes through the gate, and
    the `allow_unproven` setting decides. A data type agrees when the two
    are the same, or when one of them is `Unchecked`, the data type of a
    column whose name is known and whose data type is not.
    """

    def __init__(self, gate: UnprovenGate, operation: str) -> None:
        """Keep the gate for a column that only the remainder can hold, and the operation.

        Args:
            gate: The gate that decides whether such a column gets an error.
            operation: The name of the operation, which the setting can
                give its own level.
        """
        self._gate = gate
        self._operation = operation
        self._unchecked = UncheckedCompatibility()

    def check(
        self,
        ctx: CallContext,
        declared: list[ColumnRef],
        actual: KnownColumns,
    ) -> None:
        """Report each declared column that the frame does not hold as declared.

        A column that the frame does not declare goes to `_missing`. A column that the frame
        holds with a data type that does not agree gets `WrongDtypeError`.

        Args:
            ctx: The mypy context where the errors point.
            declared: The columns that the declaration names.
            actual: The declared columns of the frame, and its remainder.
        """
        by_name = actual.by_name()
        for column in declared:
            name = column.name
            if name is None:  # pragma: no cover
                # The plugin refuses a `Column[...]` item whose name is not a
                # literal when it reads the annotation.
                continue
            present = by_name.get(name)
            if present is None:
                self._missing(ctx, name, actual)
            elif self._unchecked.resolved(column.dtype, present.dtype) is None:
                WrongDtypeError(name).fail(ctx)

    def _missing(self, ctx: CallContext, name: str, actual: KnownColumns) -> None:
        """Report a declared column that the frame does not declare.

        Args:
            ctx: The mypy context where the errors point.
            name: The name of the column.
            actual: The declared columns of the frame, and its remainder.
        """
        if actual.rest is None or RemainderKnowledge().proves_absent(actual.rest, name):
            MissingRequiredColumnError(name).fail(ctx)
            return
        self._gate.report(ctx, UndeclaredColumnError(name, self._operation))
