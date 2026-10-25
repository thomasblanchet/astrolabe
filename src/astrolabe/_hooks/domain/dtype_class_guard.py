"""Report a class that polars refuses where a call needs one data type.

A call such as `.cast(...)` or `pl.lit(None, dtype=...)` takes one data
type. Some classes look like a data type and are not one there. The class
`pl.Decimal` has no precision, and polars refuses it inside an
expression, in a cast and in the schema of a function that reads a file.
A marker class of this project, such as `List[pl.Int64]`, writes a data
type in a type annotation, and polars refuses it in every such call. A
subclass of a polars data type that user code defines, such as
`class Tagged(pl.Int64)`, is no data type that polars reads either,
except where polars reads it by a name that a polars data type has. A
check of the call reports the class before it reads the data type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.calls.dtype_expr import DtypeExpr
from astrolabe._hooks.errors import InvalidCallShapeError

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import FunctionContext, MethodContext


class DtypeClassGuard:
    """Report a `dtype` argument that polars refuses for every input, before the check reads it.

    Code that reads the data type of a cast, a literal, a `return_dtype=`
    or a schema asks this class first. The answer says whether the call
    already reported, so that the code gives the plain type of the call
    and reads nothing more. A series and a frame built from Python data
    read `pl.Decimal` as `Decimal(38, 0)`, so code for those calls asks
    `refused_marker`, `refused_in_dict_rows` or `refused_in_series` only.
    """

    def refused(self, ctx: MethodContext | FunctionContext, node: Expression) -> bool:
        """Report `pl.Decimal`, a marker class or a class of user code that polars refuses.

        polars reads a class of user code by its name here, so it refuses
        a class whose name is not the name of a polars data type.

        Args:
            ctx: The context of the call.
            node: The AST node of the argument that names a data type.

        Returns:
            True if the hook reported the argument.
        """
        if self.refused_marker(ctx, node):
            return True
        if self._reported_user(ctx, DtypeExpr().unread_user_class(ctx.api, node)):
            return True
        if not DtypeExpr().names_decimal_class(ctx.api, node):
            return False
        error = InvalidCallShapeError(
            "pl.Decimal without a precision is not a data type here, "
            "and polars raises: pass pl.Decimal(precision, scale)"
        )
        error.fail(ctx)
        return True

    def refused_marker(self, ctx: MethodContext | FunctionContext, node: Expression) -> bool:
        """Report the argument if it is a marker class of this project that polars refuses.

        Args:
            ctx: The context of the call.
            node: The AST node of the argument that names a data type.

        Returns:
            True if the hook reported the argument.
        """
        return self._reported(ctx, DtypeExpr().refused_marker(ctx.api, node))

    def refused_in_selector(self, ctx: MethodContext | FunctionContext, node: Expression) -> bool:
        """Report a marker class or a class of user code that a selector refuses.

        A selector such as `cs.by_dtype(...)` refuses every marker class
        that `refused_marker` names. It reads a class of user code by its
        name, and raises for a name that no polars data type has.

        Args:
            ctx: The context of the call.
            node: The AST node of the argument that names a data type.

        Returns:
            True if the hook reported the argument.
        """
        if self.refused_marker(ctx, node):
            return True
        return self._reported_user(ctx, DtypeExpr().unread_user_class(ctx.api, node))

    def refused_in_dict_rows(self, ctx: MethodContext | FunctionContext, node: Expression) -> bool:
        """Report a marker class or a class of user code that `pl.from_dicts` refuses.

        `pl.from_dicts` reads `pl.Decimal` as `Decimal(38, 0)`, so this
        method does not report it. It reads a class of user code by its
        name for every data, and raises for a name that no polars data
        type has.

        Args:
            ctx: The context of the call.
            node: The AST node of the argument that names a data type.

        Returns:
            True if the hook reported the argument.
        """
        if self.refused_marker(ctx, node):
            return True
        return self._reported_user(ctx, DtypeExpr().unread_user_class(ctx.api, node))

    def refused_in_series(self, ctx: MethodContext | FunctionContext, node: Expression) -> bool:
        """Report a marker class of this project that `pl.Series` refuses.

        `pl.Series` refuses every marker class that `refused_marker` names.
        It also refuses a bare marker class, such as `List` with no type
        arguments, for every input, where other calls read the polars class.
        It refuses a class of user code that subclasses a polars data type
        other than `List`, `Array` or `Struct`, whatever its name.

        Args:
            ctx: The context of the call.
            node: The AST node of the argument that names a data type.

        Returns:
            True if the hook reported the argument.
        """
        if self.refused_marker(ctx, node):
            return True
        if self._reported(ctx, DtypeExpr().bare_marker(ctx.api, node)):
            return True
        return self._reported_user(ctx, DtypeExpr().scalar_user_class(ctx.api, node))

    def refused_in_nested(self, ctx: MethodContext | FunctionContext, node: Expression) -> bool:
        """Report a marker with its type arguments inside `pl.List(...)` or a similar call.

        `pl.List`, `pl.Array`, `pl.Struct` and `pl.Field` raise for a
        marker with its type arguments, such as `List[pl.Int64]`. They accept `Unchecked`
        and the bare class `Decimal`, and only a later use of the data type
        raises. They read a class of user code by its name, and raise for a
        name that no polars data type has.

        Args:
            ctx: The context of the call.
            node: The AST node of the argument that names a data type.

        Returns:
            True if the hook reported the argument.
        """
        if self._reported(ctx, DtypeExpr().marker_with_arguments(ctx.api, node)):
            return True
        return self._reported_user(ctx, DtypeExpr().unread_user_class(ctx.api, node))

    def _reported_user(self, ctx: MethodContext | FunctionContext, name: str | None) -> bool:
        """Report a class of user code, if there is one.

        Args:
            ctx: The context of the call.
            name: The short name of the class, or None.

        Returns:
            True if the method reported the class.
        """
        if name is None:
            return False
        InvalidCallShapeError.user_class(name).fail(ctx)
        return True

    def _reported(self, ctx: MethodContext | FunctionContext, marker: str | None) -> bool:
        """Report a marker class, if there is one.

        Args:
            ctx: The context of the call.
            marker: The name of the marker class, or None.

        Returns:
            True if the method reported the marker class.
        """
        if marker is None:
            return False
        InvalidCallShapeError.marker_class(marker).fail(ctx)
        return True
