"""Read the value of a plain Python literal from a call, for code that needs the value itself.

Polars often decides a data type from a value, not from its type. `pl.lit(200)` and
`pl.lit(2**40)` give different integer types, and a column name is a string value. This module
reads such a value from the abstract syntax tree (AST) of the call, or from the type that mypy
gives it.

Python computes a constant expression, such as `2**10` or `-(2**63)`, before polars sees it.
The reader computes it too, with `ConstantFold`.

mypy knows the value of a variable in two cases, and the reader treats them the same:

- A `Final` variable, for example `NAME: Final = "a"`. mypy keeps its value on the
  `last_known_value` of its type. That also covers a negated name: `-NAME` for
  `NAME: Final = 5` has the value -5.
- A variable or an expression whose type is a `Literal[...]`, for example
  `x: Literal["a"] = "a"`. Its type is the value itself.

mypy keeps no value for any other variable, so the reader cannot find one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import NameExpr, StrExpr
from mypy.types import Instance, LiteralType, get_proper_type

from astrolabe._hooks.calls.constant_fold import ConstantFold

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.types import Type as MypyType


class BareLiteralReader:
    """Read the value of a literal that a user wrote, or that mypy knows, for a checked call.

    Many checks need a value, not a type. The width of an integer literal, the name that a
    string gives a column, and the result of `col + 1` all depend on the value. A value can
    reach a call in several forms: a literal, arithmetic of literals such as `2**10`, or a
    `Final` or `Literal[...]` variable. This class reads all of them
    and answers a plain Python value, or None if the value is not known.

    It sits in the plugin layer. It reads mypy AST nodes and mypy types. The engine receives
    only the plain values that it answers.

    Example:
        >>> fixture = new_type_fixture()
        >>> reader = BareLiteralReader()
        >>> reader.numeric(IntExpr(200), int_dtype(fixture))
        200
        >>> reader.string(StrExpr("x"), str_dtype(fixture))
        'x'
    """

    def numeric(self, node: Expression, instance_type: MypyType) -> int | float | None:
        """Read an `int` or a `float` value from a node, or from the type that mypy gives it.

        Args:
            node: The AST node. It can be arithmetic of literals, so `-1` reads as the
                number -1 and `2**10` as 1024.
            instance_type: The type that mypy gives `node`. It carries the value of a `Final`
                or `Literal[...]` variable. The method reads it only if `node` is not a literal.

        Returns:
            The value, or None if neither `node` nor `instance_type` gives a number. A `bool`
            value gives None. Python makes `bool` a subclass of `int`, but polars gives a
            `bool` the `Boolean` data type, not a number type.

        Example:
            >>> fixture = new_type_fixture()
            >>> reader = BareLiteralReader()
            >>> reader.numeric(IntExpr(200), int_dtype(fixture))
            200
            >>> from mypy.nodes import OpExpr, UnaryExpr
            >>> reader.numeric(UnaryExpr("-", IntExpr(1)), int_dtype(fixture))
            -1
            >>> reader.numeric(OpExpr("**", IntExpr(2), IntExpr(10)), int_dtype(fixture))
            1024
            >>> reader.numeric(StrExpr("x"), int_dtype(fixture)) is None
            True
            >>> node = NameExpr("NAME")
            >>> reader.numeric(node, literal_receiver_instance(fixture, 5))
            5
            >>> reader.numeric(node, literal_int_type(fixture, 3))
            3
            >>> reader.numeric(node, literal_bool_type(fixture, True)) is None
            True
        """
        ast_value = self._ast_numeric(node)
        if ast_value is not None:
            return ast_value
        value = self._static_value(instance_type)
        if type(value) is int or type(value) is float:
            return value
        return None

    def _ast_numeric(self, node: Expression) -> int | float | None:
        """Read an `int` or a `float` that the user wrote in the call, as a literal or arithmetic.

        Args:
            node: The AST node.

        Returns:
            The value, or None if `node` does not compute an int or a float.
        """
        value = ConstantFold().value(node)
        if isinstance(value, bool):
            return None
        return value

    def string(self, node: Expression, instance_type: MypyType) -> str | None:
        """Read a string value from a node, or from the type that mypy gives it.

        Args:
            node: The AST node.
            instance_type: The type that mypy gives `node`. It carries the value of a `Final`
                or `Literal[...]` variable. The method reads it only if `node` is not a literal.

        Returns:
            The value, or None if neither `node` nor `instance_type` gives a string.

        Example:
            >>> fixture = new_type_fixture()
            >>> reader = BareLiteralReader()
            >>> reader.string(StrExpr("x"), str_dtype(fixture))
            'x'
            >>> reader.string(IntExpr(1), str_dtype(fixture)) is None
            True
            >>> node = NameExpr("NAME")
            >>> reader.string(node, literal_receiver_instance(fixture, "a"))
            'a'
            >>> reader.string(node, literal_str_type(fixture, "b"))
            'b'
        """
        ast_value = self._ast_string(node)
        if ast_value is not None:
            return ast_value
        value = self._static_value(instance_type)
        if type(value) is str:
            return value
        return None

    def _ast_string(self, node: Expression) -> str | None:
        """Read a string literal that the user wrote in the call.

        Args:
            node: The AST node.

        Returns:
            The value, or None if `node` is not a string literal.
        """
        inner = node
        if isinstance(inner, StrExpr):
            return inner.value
        return None

    def string_via_api(self, node: Expression, api: object) -> str | None:
        """Read a string value from a node, and ask mypy for its type only if that can help.

        Code that walks the AST of an argument has a node but not its type. This method reads a
        string literal directly. If the node is a variable name, it asks mypy for the type of
        the node and reads a `Final` or `Literal[...]` value from it. It never asks for the type
        of any other node, such as a call or an attribute. That keeps the cost low, and a test
        stand-in for the API then needs to know only the names.

        Args:
            node: The AST node.
            api: The mypy plugin API, which gives the type of a variable name.

        Returns:
            The value, or None if the node is neither a string literal nor a name with a known
            string value.

        Example:
            >>> BareLiteralReader().string_via_api(str_literal_node("price"), api=None)
            'price'
        """
        inner = node
        if isinstance(inner, StrExpr):
            return inner.value
        if not isinstance(inner, NameExpr):
            return None
        return self.string(node, api.get_expression_type(node))  # type: ignore[attr-defined]

    def scalar_via_api(self, node: Expression, api: object) -> int | float | str | bool | None:
        """Read a number, string or boolean value from a node, and ask mypy only for a name.

        This is `scalar` for code that holds a node but not its type. It asks mypy for the type
        only if the node is a variable name, for the same reasons as `string_via_api`.

        Args:
            node: The AST node.
            api: The mypy plugin API, which gives the type of a variable name.

        Returns:
            The value, or None if the node is neither such a literal nor a name with a known
            value of one of those kinds.

        Example:
            >>> BareLiteralReader().scalar_via_api(IntExpr(5), api=None)
            5
        """
        direct = self._ast_scalar(node)
        if direct is not None:
            return direct
        if not isinstance(node, NameExpr):
            return None
        return self._static_value(api.get_expression_type(node))  # type: ignore[attr-defined]

    def _ast_scalar(self, node: Expression) -> int | float | str | bool | None:
        """Read a number, string or boolean literal that the user wrote in the call.

        Args:
            node: The AST node. A number can have a leading `-`.

        Returns:
            The value, or None if `node` is none of those literals.
        """
        numeric = self._ast_numeric(node)
        if numeric is not None:
            return numeric
        string = self._ast_string(node)
        if string is not None:
            return string
        return self._ast_boolean(node)

    def boolean(self, node: Expression, instance_type: MypyType) -> bool | None:
        """Read a boolean value from a node, or from the type that mypy gives it.

        Args:
            node: The AST node.
            instance_type: The type that mypy gives `node`. It carries the value of a `Final`
                or `Literal[...]` variable. The method reads it only if `node` is not a literal.

        Returns:
            The value, or None if neither `node` nor `instance_type` gives a boolean.

        Example:
            >>> fixture = new_type_fixture()
            >>> reader = BareLiteralReader()
            >>> true_expr = NameExpr("True")
            >>> true_expr.fullname = "builtins.True"
            >>> reader.boolean(true_expr, boolean_dtype(fixture))
            True
            >>> reader.boolean(IntExpr(1), boolean_dtype(fixture)) is None
            True
            >>> node = NameExpr("NAME")
            >>> reader.boolean(node, literal_receiver_instance(fixture, True))
            True
            >>> reader.boolean(node, literal_bool_type(fixture, False))
            False
        """
        ast_value = self._ast_boolean(node)
        if ast_value is not None:
            return ast_value
        value = self._static_value(instance_type)
        if type(value) is bool:
            return value
        return None

    def _ast_boolean(self, node: Expression) -> bool | None:
        """Read a boolean that the user wrote in the call, such as `True` or `True & False`.

        Args:
            node: The AST node.

        Returns:
            The value, or None if `node` does not compute a boolean.
        """
        value = ConstantFold().value(node)
        return value if isinstance(value, bool) else None

    def _static_value(self, instance_type: MypyType) -> int | float | str | bool | None:
        """Read the value that a type carries, of any kind.

        mypy keeps a known value in one of two places. A `Literal[...]` type is the value
        itself. The type of a `Final` variable keeps the value on its `last_known_value`.

        Args:
            instance_type: The type that mypy gives the variable.

        Returns:
            The value, or None for the type of an ordinary variable, which carries no value.

        Example:
            >>> fixture = new_type_fixture()
            >>> reader = BareLiteralReader()
            >>> reader._static_value(literal_receiver_instance(fixture, 5))
            5
            >>> reader._static_value(literal_int_type(fixture, 3))
            3
            >>> reader._static_value(int_dtype(fixture)) is None
            True
        """
        proper = get_proper_type(instance_type)
        if isinstance(proper, LiteralType):
            return proper.value
        if not isinstance(proper, Instance) or proper.last_known_value is None:
            return None
        return proper.last_known_value.value
