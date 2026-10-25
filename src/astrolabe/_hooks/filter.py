"""The hooks for `.filter()` and `.remove()`, which check that each predicate gives `Boolean`.

A row filter keeps every column, so the result has the type of the
receiver and the hook mints nothing. The hook exists to check the
arguments. Polars refuses a predicate that does not give a `Boolean`
column before it reads any data. The checker reports the same fault at
the call.

A positional argument is a predicate. The hook works out the column that
the predicate gives on the receiver's columns, with the same rules as
`select`. That column must be `Boolean`. An `Unchecked` column passes, because the
checker does not know its data type and cannot prove it wrong. A keyword
argument, for example `.filter(ticker="AAPL")`, compares a column with a
value. Its name must be a column of the receiver. Polars compares the
value itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.encoding.marker_minter import ProperReads
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.errors import (
    UndeclaredColumnError,
    UnknownColumnError,
    WrongColumnShapeError,
)
from astrolabe._hooks.expression_requests import (
    ExpressionRequest,
    RequestPolicies,
    RequestReader,
    RequestResolver,
)
from astrolabe._hooks.propagate import Refusal, not_none, propagates

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.unchecked import UncheckedFallback


class FilterHook:
    """Check the predicates and the keyword constraints of one row filter.

    The hook sits in the plugin layer. It reads the receiver through
    `Frame`, and it turns each predicate into a column through
    `RequestResolver`, which asks the expression engine. It reports each
    wrong predicate and each unknown name, and it always gives back the
    receiver's own type.
    """

    def __init__(
        self,
        plugin: Plugin,
        unchecked_fallback: UncheckedFallback,
        operation: str,
    ) -> None:
        """Build the shared rules that turn a predicate into a column.

        Args:
            plugin: The mypy plugin, which the resolver needs to mint the
                column of a predicate.
            unchecked_fallback: The fallback that reports a predicate or a name
                that the checker cannot prove, subject to the `allow_unproven`
                setting.
            operation: `filter` or `remove`, for the error messages.
        """
        self._operation = operation
        self._policies = RequestPolicies(operation, "row", plugin, unchecked_fallback)

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Check one `.filter()` or `.remove()` call.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The type of the receiver, in every case. A row filter never changes
            the columns. The hook reports each fault that it finds.
        """
        self._checked(ctx)
        return ctx.default_return_type

    @propagates(None)
    def _checked(self, ctx: MethodContext) -> bool | None:
        """Check each argument of the call.

        The typed frames declare the parameters as `*args` and `**kwargs`. A
        positional argument is a predicate, and a keyword argument is a
        constraint on the column that it names.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            True when the method checked every argument. None if the receiver
            is not a typed frame, or the arguments cannot be read. The hook then
            checks nothing.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        resolver = RequestResolver(ctx, known, self._policies)
        for request in not_none(RequestReader("predicates", "constraints").of(ctx)):
            if request.forced_name is not None:
                self._check_constraint(ctx, known, request.forced_name)
            else:
                BooleanPredicate(self._operation).check(ctx, resolver, request)
        return True

    def _check_constraint(self, ctx: MethodContext, known: KnownColumns, name: str) -> None:
        """Report a keyword constraint whose name is not a declared column.

        The error depends on the remainder, the columns of an open frame that
        its type does not name. If the remainder cannot hold the name, the
        column is surely missing, and the method reports a hard error. If the
        remainder can hold it, the method reports an unproven result, which the
        `allow_unproven` setting can let pass.

        Args:
            ctx: The mypy context of the method call.
            known: The declared columns and the remainder of the receiver.
            name: The column name of the keyword.
        """
        if name in known.by_name():
            return
        if RemainderKnowledge().proves_absent(known.rest, name):
            UnknownColumnError(name).fail(ctx)
        else:
            self._policies.unchecked_fallback.report(
                ctx,
                UndeclaredColumnError(name, self._operation),
            )


class BooleanPredicate:
    """Report each column of a predicate that is not `Boolean`, as polars refuses it.

    A row filter, as `filter`, and the predicates of `join_where` keep a row
    where a predicate holds. Polars refuses a predicate that does not give a
    `Boolean` column before it reads any data. A selection can give many
    columns, so each one gets the check. An `Unchecked` column passes,
    because the checker does not know its data type and cannot prove it
    wrong.

    Example:
        >>> BooleanPredicate("filter")._operation
        'filter'
    """

    BOOLEAN: ClassVar[Target] = Target.of(pl.Boolean)

    def __init__(self, operation: str) -> None:
        """Keep the name of the operation, for the message.

        Args:
            operation: The name of the method, as `filter`.
        """
        self._operation = operation

    def check(
        self,
        ctx: MethodContext,
        resolver: RequestResolver,
        request: ExpressionRequest,
    ) -> None:
        """Report each column of one predicate that is not `Boolean`.

        If the resolver cannot turn the predicate into columns, the method
        checks nothing more.

        Args:
            ctx: The mypy context of the method call.
            resolver: The resolver for this call.
            request: The predicate argument.
        """
        resolution = resolver.resolved_all(request)
        if isinstance(resolution, Refusal):
            return
        for predicate in resolution.columns:
            if self.passes(predicate):
                continue
            error = WrongColumnShapeError.on_column(
                predicate.name or "",
                expected="Boolean",
                operation=f".{self._operation}()",
            )
            error.fail(ctx)

    def passes(self, predicate: ColumnRef) -> bool:
        """Tell whether the data type of a predicate column is acceptable.

        Args:
            predicate: The column that the predicate gives.

        Returns:
            True for `Boolean`, and for `Unchecked`, whose data type the checker
            does not know.

        Example:
            >>> fixture = new_type_fixture()
            >>> BooleanPredicate("filter").passes(column_ref(fixture, "b", boolean_dtype(fixture)))
            True
            >>> BooleanPredicate("filter").passes(column_ref(fixture, "b", int_dtype(fixture)))
            False
        """
        if UncheckedCompatibility().matches(predicate.dtype):
            return True
        instance = ProperReads().instance(predicate.dtype)
        return instance is not None and self.BOOLEAN.matches(instance)
