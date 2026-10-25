"""Give the plain polars frame when a hook cannot type the result of a frame operation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.domain.unproven_gate import UnprovenGate
from astrolabe._hooks.errors import ImplicitlyUntypedError

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext, MethodContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import Frame, PluginLookup
    from astrolabe._hooks.unproven import Reason, UnprovenPolicy


class UntypedFallback(UnprovenGate):
    """Fall back to the plain frame for a frame operation that a hook cannot type.

    If a hook cannot find the columns of a result, it must not guess. To keep the columns of
    the receiver is also a guess: the real call can add a column, delete one, or change its
    data type. The hook falls back instead. It gives the plain `pl.DataFrame` or
    `pl.LazyFrame`, which the checker does not track, of the same kind as the typed frame it
    received. That tells the truth: the checker does not know the columns.

    It is an `UnprovenGate`. Before it falls back, it reports `ImplicitlyUntypedError` unless
    the `allow_unproven` setting lets the reason pass. The user then sees where the tracking of
    columns stops.

    Example:
        >>> from astrolabe._hooks.unproven import UnprovenPolicy
        >>> resolver = UntypedFallback(UnprovenPolicy(), new_plugin(new_type_fixture()))
        >>> resolver._first_resolved(None) is None
        True
    """

    def __init__(self, policy: UnprovenPolicy, plugin: PluginLookup) -> None:
        """Keep the setting for the gate, and the plugin that finds the plain frame class.

        Args:
            policy: The `allow_unproven` setting, already read from the configuration.
            plugin: The mypy plugin, which finds `pl.DataFrame` or `pl.LazyFrame`.
        """
        super().__init__(policy)
        self._plugin = plugin

    def report_if_disallowed(
        self,
        ctx: MethodContext | FunctionContext,
        reason: Reason,
        operation: str,
    ) -> None:
        """Report `ImplicitlyUntypedError` unless the setting lets the reason pass.

        A hook whose mypy default result is already the plain frame calls this alone. It needs
        the error and not a new result type.

        Args:
            ctx: The mypy context of the operation.
            reason: Why the operation gave up.
            operation: The name of the operation.
        """
        self.report(ctx, ImplicitlyUntypedError(reason, operation))

    def resolve(
        self,
        ctx: MethodContext | FunctionContext,
        reason: Reason,
        operation: str,
        *frames: Frame | None,
    ) -> MypyType:
        """Report the error if the setting requires it, and give the plain frame as the result.

        Use this in a hook whose result is the same kind of frame as one of its operands. A
        hook with a result of another class, such as a group-by, does not need it. The mypy
        default for such a call is already the plain polars type.

        Args:
            ctx: The mypy context of the operation.
            reason: Why the operation gave up.
            operation: The name of the operation.
            *frames: The typed frames of the operation. Pass None for an operand that is not a
                typed frame.

        Returns:
            The plain polars frame of the same kind as the first typed frame in `frames`. The
            mypy default result if no operand is a typed frame.
        """
        self.report_if_disallowed(ctx, reason, operation)
        frame = self._first_resolved(*frames)
        if frame is None:
            # A hook with a typed receiver always passes that receiver. Only a call with no
            # receiver, such as pl.read_csv() with no typed schema, can pass no frame. Its mypy
            # default result is then the right answer.
            return ctx.default_return_type
        plain = frame.as_plain(self._plugin)
        if plain is None:  # pragma: no cover
            # mypy always loads polars.DataFrame and polars.LazyFrame before a typed frame
            # exists that can fall back to them.
            return ctx.default_return_type
        return plain

    def _first_resolved(self, *frames: Frame | None) -> Frame | None:
        """Give the first typed frame among the operands of an operation.

        Args:
            *frames: The typed frames of the operation, with None for an operand that is not
                one.

        Returns:
            The first frame that is not None, or None if all are None.

        Example:
            >>> from astrolabe._hooks.unproven import UnprovenPolicy
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> concrete = frame(fixture, [price.as_instance()])
            >>> resolver = UntypedFallback(UnprovenPolicy(), new_plugin(fixture))
            >>> resolver._first_resolved(None, concrete) is concrete
            True
            >>> resolver._first_resolved(None, None) is None
            True
        """
        for frame in frames:
            if frame is not None:
                return frame
        return None
