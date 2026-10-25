"""Find a column name in an argument that the checker cannot type as an expression.

Polars accepts a string in place of `pl.col(...)`: `df.select("a")` is `df.select(pl.col("a"))`.
And if a chain of calls ends in `.alias("b")`, the output column is named `b`, whatever the rest
of the chain does. This module reads both from the abstract syntax tree (AST) of an argument. A
hook can then keep a column under its name even when it cannot type the whole argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from mypy.nodes import CallExpr, Expression, MemberExpr

from astrolabe._hooks.calls.bare_literal import BareLiteralReader
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.propagate import not_none, propagates


@dataclass(frozen=True)
class ColumnExpr:
    """A column name that the caller wrote as a string in place of `pl.col(...)`.

    In `df.select("price")`, the string names a column, and the argument has the type `str`,
    not an expression type. A hook that reads the arguments of a frame operation needs the name.
    `from_node` reads it from a string literal, or from a `Final` or `Literal[...]` string
    variable. Every argument that is a real expression, such as `pl.col("price").alias("cost")`,
    has a typed expression already, so this class reads only the string form.

    It sits in the plugin layer and reads mypy AST nodes.

    Attributes:
        source_name: The name of the column that the string names.

    Example:
        >>> ColumnExpr.from_node(StrExpr("price"), api=None)
        ColumnExpr(source_name='price')
    """

    # This is not Target.of(pl.col). pl.col is an instance of the class Col, and Col overrides
    # __getattr__ so that pl.col.name gives an expression. pl.col.__qualname__ therefore gives
    # an expression and not a name. The full name here is the name "col" in its module, which
    # is how a call such as pl.col("a") refers to it in the AST.
    COL_TARGET: ClassVar[Target] = Target("polars.functions.col.col")

    source_name: str

    @classmethod
    def from_node(cls, node: Expression, api: object) -> ColumnExpr | None:
        """Read an argument as a column name written as a string.

        Args:
            node: The AST node of the argument.
            api: The mypy plugin API, which gives the value of a `Final` or `Literal[...]`
                string variable.

        Returns:
            The column name, or None if the argument is not a known string.
        """
        name = cls._bare_string_name(node, api)
        if name is None:
            return None
        return cls(source_name=name)

    @classmethod
    def outermost_alias(cls, node: Expression, api: object) -> str | None:
        """Read the last `.alias(...)` of a call chain that the checker cannot type.

        Some chains have no typed expression, for example one that calls a method the checker
        does not model. If the chain ends in `.alias("b")`, the output column is still named `b`.
        A hook can then keep the column as `b` with the data type `Unchecked`, which means that
        the name is known and the data type is not. Without the name, the hook falls back: it
        gives the plain frame, which the checker does not track.

        Args:
            node: The AST node of the argument.
            api: The mypy plugin API, which gives the value of a `Final` or `Literal[...]`
                string variable in `.alias(...)`.

        Returns:
            The name from the last `.alias(...)` of the chain, or None if the chain has none.

        Example:
            >>> ColumnExpr.outermost_alias(StrExpr("price"), api=None) is None
            True
        """
        walker = AliasCastUnwrapper(node)
        walker.unwrapped(api)
        return walker.alias

    @classmethod
    def _bare_string_name(cls, node: Expression, api: object) -> str | None:
        """Read the argument as a string, from the literal or from a known variable.

        Args:
            node: The AST node of the argument.
            api: The mypy plugin API. It gives the type of a variable name, and nothing else.

        Returns:
            The string, or None if the argument is not a known string.

        Example:
            >>> ColumnExpr._bare_string_name(StrExpr("price"), api=None)
            'price'
        """
        return BareLiteralReader().string_via_api(node, api)


class AliasCastUnwrapper:
    """Walk a call chain inward past its last `.alias(...)` and `.cast(...)` steps.

    In `expr.cast(pl.Int64).alias("b")`, the outermost call is `.alias("b")`. Polars applies it
    last, so it names the output column. The walk starts at the outermost call and keeps the
    first alias that it meets. It steps past a `.cast(...)`, because a cast does not change the
    name, and an `.alias(...)` can sit under it. It stops at the first other call.

    The walk does not type the node where it stops. The code that uses this class already knows
    that the chain has no typed expression. It wants only the name.
    """

    def __init__(self, node: Expression) -> None:
        """Start a walk at the outermost call of a chain.

        Args:
            node: The AST node of the outermost call.
        """
        self._alias: str | None = None
        self._current: Expression = node
        self._bare_literal = BareLiteralReader()

    @property
    def alias(self) -> str | None:
        """The name from the outermost `.alias(...)` so far, or None if the walk met none."""
        return self._alias

    def unwrapped(self, api: object) -> Expression:
        """Walk inward past every `.alias(...)` and `.cast(...)` step that it can read.

        Args:
            api: The mypy plugin API, which gives the value of a `Final` or `Literal[...]`
                string variable in `.alias(...)`.

        Returns:
            The node where the walk stops: the first node that is not a readable
            `.alias(name)` or `.cast(dtype)` call. That can be an `.alias(...)` or `.cast(...)`
            call with an argument that the walk cannot read.

        Example:
            >>> receiver = NameExpr("e")
            >>> aliased = CallExpr(MemberExpr(receiver, "alias"), [StrExpr("b")], [], [])
            >>> outer = CallExpr(MemberExpr(aliased, "cast"), [NameExpr("t")], [], [])
            >>> walker = AliasCastUnwrapper(outer)
            >>> walker.unwrapped(api=None) is receiver
            True
            >>> walker.alias
            'b'
        """
        while self._steps_inward(api):
            pass
        return self._current

    @propagates(False)
    def _steps_inward(self, api: object) -> bool:
        """Take one step inward, if the current node is a method call that the walk can pass.

        The walk passes an `.alias(name)` or a `.cast(dtype)` call with one argument. After a
        step, the current node is the receiver of the method call. At an alias, the walk keeps
        the name if it is the first one. The first alias that the walk meets is the outermost
        one, which names the column.

        At a cast, the walk does not read the data type. The walk looks only for a name, and a
        cast does not change the name. A chain that the checker can type, such as
        `pl.col("a").cast(pl.Int64)`, already has a typed expression, so no code needs the data
        type from here.

        Args:
            api: The mypy plugin API. See `unwrapped`.

        Returns:
            True if the walk took a step. False if the current node is not a method call, is
            another method, does not have one argument, or is an alias whose argument is not a
            known string.

        Example:
            >>> node = CallExpr(StrExpr("not a call"), [], [], [])
            >>> walker = AliasCastUnwrapper(node)
            >>> walker._steps_inward(api=object())
            False
        """
        current = self._current
        if not isinstance(current, CallExpr) or not isinstance(current.callee, MemberExpr):
            return False
        callee = current.callee
        if callee.name not in {"alias", "cast"} or len(current.args) != 1:
            return False
        if callee.name == "alias":
            name = not_none(self._bare_literal.string_via_api(current.args[0], api))
            if self._alias is None:
                self._alias = name
        self._current = callee.expr
        return True
