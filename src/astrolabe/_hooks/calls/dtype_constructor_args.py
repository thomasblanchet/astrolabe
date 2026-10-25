"""Read the literal arguments of a data type written as a call, such as `pl.Datetime("ms")`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import IntExpr, ListExpr, NameExpr, StrExpr, UnaryExpr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.nodes import CallExpr, Expression


class DtypeConstructorArgs:
    """Read the literal arguments of a call to a polars data type class, by parameter name.

    Some data types have parameters, for example the time unit in `pl.Datetime("ms")`. The
    parameters are part of the data type, so the checker must read them. Such a call often sits
    inside another call, as in `schema={"t": pl.Datetime("ms")}`, or is read before mypy has
    matched its arguments to parameters. There is then no mypy call context to read it through.
    This class reads the abstract syntax tree (AST) of the call itself.

    The AST does not say which parameter a positional argument fills. The code that builds this
    object therefore gives the parameter names of the class in their declared order. This class
    reads only literals that the user wrote in the call. It does not read the value of a
    `Final` variable, and it asks mypy for no type.

    Example:
        >>> call = dtype_constructor_call([str_literal_node("us")])
        >>> DtypeConstructorArgs(call, ["time_unit", "time_zone"]).literal_str("time_unit")
        'us'
    """

    def __init__(self, call: CallExpr, param_names: Sequence[str]) -> None:
        """Match each argument of the call to its parameter.

        A keyword argument names its parameter. A positional argument takes the next name from
        `param_names`. A positional argument with no parameter left is ignored.

        Args:
            call: The AST node of the call, for example `pl.Datetime("ms")`.
            param_names: The parameter names of the data type class, in declared order.
        """
        self._nodes: dict[str, Expression] = {}
        positional = iter(param_names)
        for name, node in zip(call.arg_names, call.args, strict=True):
            resolved_name = name if name is not None else next(positional, None)
            if resolved_name is not None:
                self._nodes[resolved_name] = node

    def is_omitted_or_none(self, name: str) -> bool:
        """Tell whether a parameter whose default is `None` gets its default.

        The caller can leave the argument out or write `None`, as in
        `pl.Datetime("ms", time_zone=None)`. Both give the default.

        Args:
            name: The name of the parameter.

        Returns:
            True if the call has no argument for `name`, or if the argument is `None`.

        Example:
            >>> call = dtype_constructor_call([str_literal_node("us")])
            >>> args = DtypeConstructorArgs(call, ["time_unit", "time_zone"])
            >>> args.is_omitted_or_none("time_zone")
            True
        """
        node = self._nodes.get(name)
        if node is None:
            return True
        return isinstance(node, NameExpr) and node.fullname == "builtins.None"

    def literal_str(self, name: str, default: str | None = None) -> str | None:
        """Read a string literal that the call passes for one parameter, such as a time unit.

        Args:
            name: The name of the parameter.
            default: The value to give if the call has no argument for `name`.

        Returns:
            The string, or `default` if the argument is missing. None if the argument is not a
            string literal.

        Example:
            >>> call = dtype_constructor_call([str_literal_node("us")])
            >>> DtypeConstructorArgs(call, ["time_unit"]).literal_str("time_unit")
            'us'
        """
        node = self._nodes.get(name)
        if node is None:
            return default
        return node.value if isinstance(node, StrExpr) else None

    def literal_int(self, name: str, default: int | None = None) -> int | None:
        """Read a whole number that the call passes for one parameter, such as a scale.

        A leading `-` is part of the number, so `-1` reads as -1.

        Args:
            name: The name of the parameter.
            default: The value to give if the call has no argument for `name`.

        Returns:
            The number, or `default` if the argument is missing. None if the argument is not an
            integer literal.

        Example:
            >>> call = dtype_constructor_call([int_literal_node(2)], ["scale"])
            >>> DtypeConstructorArgs(call, ["precision", "scale"]).literal_int("scale")
            2
        """
        node = self._nodes.get(name)
        if node is None:
            return default
        if isinstance(node, IntExpr):
            return node.value
        if isinstance(node, UnaryExpr) and node.op == "-" and isinstance(node.expr, IntExpr):
            return -node.expr.value
        return None

    def literal_str_list(self, name: str) -> list[str] | None:
        """Read a list of string literals that the call passes, such as the categories of an enum.

        Args:
            name: The name of the parameter.

        Returns:
            The strings in order. None if the argument is missing, is not a list literal, or
            holds an item that is not a string literal.

        Example:
            >>> call = dtype_constructor_call([str_list_literal_node("a", "b")])
            >>> DtypeConstructorArgs(call, ["categories"]).literal_str_list("categories")
            ['a', 'b']
        """
        node = self._nodes.get(name)
        if not isinstance(node, ListExpr):
            return None
        values: list[str] = []
        for item in node.items:
            if not isinstance(item, StrExpr):
                return None
            values.append(item.value)
        return values
