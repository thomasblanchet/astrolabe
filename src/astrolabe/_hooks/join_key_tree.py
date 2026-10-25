"""The checks of the tree of a join key, which polars reads as the code writes it.

Most operations of polars read only what an expression computes. A join also
reads the form of each key. Polars refuses a key that holds `.alias()`
anywhere in its tree, and a key that changes the rows. It merges a pair of
keys into one column only where each key is a bare column. The hooks of
`.join()`, `.join_asof()` and `.join_where()` share these checks, so they
live here once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.deferred.argument import DeferredArgument
from astrolabe._hooks.deferred.evaluation import KeyCode
from astrolabe._hooks.errors import InvalidCallShapeError
from astrolabe._hooks.recording.deferred_types import DeferredReader

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.plugin import MethodContext
    from mypy.types import Type as MypyType


class JoinKeyTree:
    """Read the tree of each join key of a call, from the code that the type of the key holds.

    The checker records the code of an expression in its type. A key whose
    type holds a code shows its tree, and a column name or a plain
    expression shows none. A declared expression, such as a parameter typed
    by `column(...)`, fixes what the key computes, and not its tree: the
    caller can pass any expression that computes it. So a key that holds a
    declared expression gives a join that the caller decides.

    Example:
        >>> JoinKeyTree("join")._operation
        'join'
    """

    def __init__(self, operation: str) -> None:
        """Keep the name of the join method for the error message.

        Args:
            operation: The name of the join method, for example `"join"`.
        """
        self._operation = operation

    def holds_no_alias(self, ctx: MethodContext, parameters: Sequence[str]) -> bool:
        """Report a call whose join key holds `.alias()` anywhere in its tree.

        Polars refuses the call with `InvalidOperationError`, eager or lazy.
        The method reports one error for the call.

        Args:
            ctx: The mypy context of the join call.
            parameters: The parameters that hold the keys, for example `"on"`.

        Returns:
            True if no key holds an alias. Otherwise the method reports an
            error and gives False.
        """
        if any(self._holds_alias(typ) for typ in self._types(ctx, parameters)):
            InvalidCallShapeError.alias_in_join_key(self._operation).fail(ctx)
            return False
        return True

    def keeps_rows(self, ctx: MethodContext, parameters: Sequence[str]) -> bool:
        """Report a call whose join key changes the rows, as `shift` or `sort` does.

        Polars refuses such a key with `InvalidOperationError`, eager or
        lazy, unless the key gives one value. The method reports one error for
        the call.

        Args:
            ctx: The mypy context of the join call.
            parameters: The parameters that hold the keys.

        Returns:
            True if no key changes the rows. Otherwise the method reports an
            error and gives False.
        """
        if any(self._changes_rows(typ) for typ in self._types(ctx, parameters)):
            InvalidCallShapeError.key_changes_rows(self._operation).fail(ctx)
            return False
        return True

    def holds_declared(self, ctx: MethodContext, parameters: Sequence[str]) -> bool:
        """Say whether a join key of the call holds a declared expression.

        Args:
            ctx: The mypy context of the join call.
            parameters: The parameters that hold the keys.

        Returns:
            Whether one does, so that the caller decides the tree of the key.
        """
        return any(self._holds_declared(typ) for typ in self._types(ctx, parameters))

    def _holds_alias(self, typ: MypyType) -> bool:
        """Say whether `.alias()` stands anywhere in the code that the type of a key holds.

        Args:
            typ: The type of the key.

        Returns:
            Whether it does in the code of a member of the type.

        Example:
            >>> JoinKeyTree("join")._holds_alias(new_type_fixture().anyt)
            False
        """
        argument = DeferredArgument("row")
        codes = argument.codes(typ) or []
        return any(KeyCode().holds_alias(argument.declared(code)) for code in codes)

    def _changes_rows(self, typ: MypyType) -> bool:
        """Say whether the code that the type of a key holds changes the rows.

        Args:
            typ: The type of the key.

        Returns:
            Whether the code of a member of the type does.

        Example:
            >>> JoinKeyTree("join")._changes_rows(new_type_fixture().anyt)
            False
        """
        argument = DeferredArgument("row")
        codes = argument.codes(typ) or []
        return any(KeyCode().refused(argument.declared(code)) for code in codes)

    def _holds_declared(self, typ: MypyType) -> bool:
        """Say whether the type of a key holds a declared expression.

        The type is a declared expression itself, or a code that reads one as
        a leaf.

        Args:
            typ: The type of the key.

        Returns:
            Whether a member of the type holds one.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = typed_column_instance(fixture, "price", int64_dtype(fixture))
            >>> JoinKeyTree("join")._holds_declared(price)
            True
            >>> JoinKeyTree("join")._holds_declared(fixture.anyt)
            False
        """
        reader = DeferredReader()
        for member in reader.members(typ):
            code = reader.code(member)
            if reader.declared(member) or (code is not None and code.leaves):
                return True
        return False

    def _types(self, ctx: MethodContext, parameters: Sequence[str]) -> list[MypyType]:
        """Give the type of each key of the call, one list or tuple literal opened up.

        Args:
            ctx: The mypy context of the join call.
            parameters: The parameters that hold the keys.

        Returns:
            The types, in call order.
        """
        args = CallArgs(ctx)
        types: list[MypyType] = []
        for parameter in parameters:
            nodes = args.vararg_node_list(parameter)
            types.extend(ctx.api.get_expression_type(node) for node in nodes)
        return types
