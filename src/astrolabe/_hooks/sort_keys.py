"""Check the keys of `sort`, `top_k` and `bottom_k`, which order the rows of a frame.

A key orders the rows and adds no column, so the result has the type of
the receiver. Polars still evaluates each key over the frame, and raises
where a key reads a column that the frame does not hold, or where polars
refuses the key at the data types of the frame:
`df.sort(pl.col("s").str.len_chars())` over an `Int64` column `s` raises
`SchemaError`. The hook resolves each key at the columns of the receiver,
as `select` resolves an expression, and the resolver reports each fault.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import ListExpr, TupleExpr

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.expression_requests import (
    ExpressionRequest,
    RequestPolicies,
    RequestResolver,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mypy.nodes import Expression
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.row_counts import RowCountCheck


class SortKeysHook:
    """Check the keys of one call that orders the rows of a typed frame.

    `sort` takes its keys as `by` and `more_by`, by position or by name.
    `top_k` and `bottom_k` take them as `by=`, beside a count of rows, which
    the hook checks too. A key is a column name, an expression, or a list of
    them. The parameters are those of polars, which the signature of the
    call holds. It is a mypy method hook in the plugin layer, and it gives
    the type that the plugin declares for the method.

    Example:
        >>> SortKeysHook.KEYS
        ('by', 'more_by')
    """

    KEYS: ClassVar[tuple[str, ...]] = ("by", "more_by")

    def __init__(
        self,
        plugin: Plugin,
        unchecked_fallback: UncheckedFallback,
        operation: str,
        counts: RowCountCheck | None = None,
    ) -> None:
        """Keep the rules that resolve a key, and the check of the count.

        Args:
            plugin: The mypy plugin, which the resolver needs.
            unchecked_fallback: The fallback that reports a key that the
                checker cannot prove, subject to the `allow_unproven` setting.
            operation: `sort`, `top_k` or `bottom_k`.
            counts: The check of the count of rows of `top_k` and
                `bottom_k`, or `None` for `sort`.
        """
        self._policies = RequestPolicies(operation, "row", plugin, unchecked_fallback)
        self._counts = counts

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Check one call, and give its declared type.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The type that the plugin declares for the method.
        """
        if self._counts is not None:
            self._counts.reported(ctx)
        frame = Frame.from_proper_type(ctx.type)
        known = None if frame is None else frame.known_columns
        if known is None:  # pragma: no cover
            # The plugin registers the hook only on the two typed frames.
            return ctx.default_return_type
        resolver = RequestResolver(ctx, known, self._policies)
        for request in self._keys(ctx):
            resolver.resolved_all(request)
        return ctx.default_return_type

    def _keys(self, ctx: MethodContext) -> Iterator[ExpressionRequest]:
        """Give one argument for each key of the call.

        Args:
            ctx: The context that mypy gives for the call.

        Yields:
            The key, with its type.
        """
        args = CallArgs(ctx)
        for name in self.KEYS:
            nodes = args.node_list(name)
            if nodes:
                for node, typ in zip(nodes, args.type_list(name), strict=True):
                    yield from self._listed(ctx, node, typ)

    def _listed(
        self,
        ctx: MethodContext,
        node: Expression,
        typ: MypyType,
    ) -> Iterator[ExpressionRequest]:
        """Give each key of one argument, which can be a list or a tuple of keys.

        Args:
            ctx: The context that mypy gives for the call.
            node: The node of the argument.
            typ: The type of the argument.

        Yields:
            The keys.
        """
        if isinstance(node, (ListExpr, TupleExpr)):
            for item in node.items:
                yield ExpressionRequest(item, ctx.api.get_expression_type(item), None)
            return
        yield ExpressionRequest(node, typ, None)
