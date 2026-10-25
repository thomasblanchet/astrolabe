"""Report an argument of a recorded call that polars refuses for every input.

Some values pass the check of mypy and make polars raise `TypeError` while it
builds the expression, so the call can never run. A marker class of this
project, such as `List[pl.Int64]`, writes a data type in a type annotation,
and polars refuses it in the place of a data type. So does the class
`pl.Decimal` with no precision, and a subclass of a polars data type that
user code defines. A `datetime.timezone` object in the place of the name of
a zone raises too. The code of a recorded call keeps such a value as a hole,
so a build of the code cannot find the error. This check reads each argument
by the name of its parameter, and reports the value at the call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import DictExpr, ListExpr, SetExpr, TupleExpr

from astrolabe._hooks.domain.dtype_class_guard import DtypeClassGuard
from astrolabe._hooks.domain.zone_objects import ZoneObjects
from astrolabe._hooks.errors import InvalidCallShapeError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.nodes import Expression
    from mypy.plugin import FunctionContext, MethodContext
    from mypy.types import Type as MypyType


class ArgumentCheck:
    """Report the first argument of a recorded call that polars refuses for every input.

    A parameter that takes one data type, as `dtype=` and `return_dtype=`,
    and each value of a `schema=`, refuse the three kinds of class. A
    parameter that picks columns, as the data types of `cs.by_dtype` and
    the columns of `exclude`, refuses a marker class and a class of user
    code, and reads `pl.Decimal` as every `Decimal`. A `time_zone=`
    refuses a `datetime.timezone` object.

    Example:
        >>> sorted(ArgumentCheck.DATA_TYPES)
        ['dtype', 'return_dtype']
    """

    DATA_TYPES: ClassVar[frozenset[str]] = frozenset({"dtype", "return_dtype"})
    SCHEMAS: ClassVar[frozenset[str]] = frozenset({"schema"})
    SELECTED: ClassVar[frozenset[str]] = frozenset({"dtypes", "columns", "more_columns"})
    ZONES: ClassVar[frozenset[str]] = frozenset({"time_zone"})

    def refused(self, ctx: FunctionContext | MethodContext) -> bool:
        """Report the first argument of a call that polars refuses for every input.

        Args:
            ctx: The context of the call.

        Returns:
            Whether the method reported an error.
        """
        parameters = zip(ctx.callee_arg_names, ctx.args, ctx.arg_types, strict=True)
        for name, nodes, types in parameters:
            for node, typ in zip(nodes, types, strict=True):
                if self._refused(ctx, name, node, typ):
                    return True
        return False

    def _refused(
        self,
        ctx: FunctionContext | MethodContext,
        name: str | None,
        node: Expression,
        typ: MypyType,
    ) -> bool:
        """Report one argument, by the parameter that it fills, where polars refuses it.

        Args:
            ctx: The context of the call.
            name: The name of the parameter.
            node: The node of the argument.
            typ: The type of the argument.

        Returns:
            Whether the method reported an error.
        """
        guard = DtypeClassGuard()
        if name in self.DATA_TYPES:
            return guard.refused(ctx, node)
        if name in self.SCHEMAS and isinstance(node, DictExpr):
            return any(guard.refused(ctx, value) for _, value in node.items)
        if name in self.SELECTED:
            return any(guard.refused_in_selector(ctx, item) for item in self._items(node))
        if name in self.ZONES and ZoneObjects(ctx.api).among(node, typ):
            InvalidCallShapeError.zone_object().fail(ctx)
            return True
        return False

    def _items(self, node: Expression) -> Sequence[Expression]:
        """Give the items of a list, a tuple or a set that the call writes, or the node itself.

        Args:
            node: The node of the argument.

        Returns:
            The nodes.

        Example:
            >>> from mypy.nodes import IntExpr
            >>> items = ArgumentCheck()._items(ListExpr([IntExpr(1), IntExpr(2)]))
            >>> [item.value for item in items]
            [1, 2]
        """
        if isinstance(node, ListExpr | TupleExpr | SetExpr):
            return node.items
        return [node]
