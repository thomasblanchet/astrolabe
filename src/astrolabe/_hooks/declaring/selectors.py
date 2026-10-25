"""Read the selector of a selection declaration from the syntax that the user wrote.

The first argument of `typed_selection(...)` is a selector, written with
the ordinary polars calls: a `cs.*` function, `pl.all()` or `pl.col(...)`,
combined with `~`, `&`, `|`, `-` and `^`. A declaration is read during
semantic analysis, the mypy phase that resolves names before any expression
has a type. The selector therefore has no type to read yet, and the code
here reads its abstract syntax tree (AST) instead.

The code writes the AST as the code of polars that it spells. The reader of
deferred codes builds the selector of polars from that code, and gives the
rule of the columns that it picks. A selector in a declaration and the same
selector in a function body therefore give the same rule.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from mypy.nodes import (
    CallExpr,
    FloatExpr,
    IntExpr,
    ListExpr,
    NameExpr,
    OpExpr,
    RefExpr,
    StrExpr,
    TupleExpr,
    UnaryExpr,
)

from astrolabe._hooks.deferred.argument import DeferredArgument
from astrolabe._hooks.deferred.code import Code
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.propagate import not_falsy, not_none, propagates

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.types import Type as MypyType

    from astrolabe._hooks.remainder.selector import Selector


class SelectorSyntax:
    """Read the AST of one selector into the engine value `Selector`.

    A selector picks columns by name, by data type or both. The class writes
    the AST as code: a call of a function of `cs`, `pl.all()` or `pl.col`,
    with literal arguments and data types of polars, a negation with `~`, or
    two selectors joined by an operator. It gives the rule of the selector in
    canonical form, the one spelling that two equal selectors share.

    Example:
        >>> SelectorSyntax().parsed(str_literal_node("a")) is None
        True
    """

    OPERATORS: ClassVar[dict[str, type[ast.operator]]] = {
        "&": ast.BitAnd,
        "|": ast.BitOr,
        "-": ast.Sub,
        "^": ast.BitXor,
    }
    SELECTORS: ClassVar[str] = "polars.selectors."
    FUNCTIONS: ClassVar[dict[str, str]] = {
        f"{type(pl.col).__module__}.col": "pl.col",
        Target.of(pl.all).fullname: "pl.all",
    }
    DATA_TYPES: ClassVar[str] = f"{pl.DataType.__module__}."
    NAMED: ClassVar[dict[str, bool | None]] = {"True": True, "False": False, "None": None}

    def parsed(self, node: Expression) -> Selector | None:
        """Read one selector, whatever its shape.

        Args:
            node: The node of the selector.

        Returns:
            The canonical selector, or None if the node is not a selector
            that polars builds and the checker reads.

        Example:
            >>> call = named_call_node("polars.selectors.numeric")
            >>> SelectorSyntax().parsed(UnaryExpr("~", call))
            SelNot(inner=OfType(family=NumericType))
        """
        written = self._selector(node)
        if written is None:
            return None
        code: Code[MypyType] = Code(ast.unparse(written))
        rule = DeferredArgument("row").selector_rule(code)
        return None if rule is None else rule.canonical()

    @propagates(None)
    def _selector(self, node: Expression) -> ast.expr | None:
        """Write the code of a selector: a negation, an operator over two selectors, or a call.

        Args:
            node: The node.

        Returns:
            The code, or None for another node.

        Example:
            >>> call = named_call_node("polars.selectors.numeric")
            >>> ast.unparse(SelectorSyntax()._selector(OpExpr("-", call, call)))
            'cs.numeric() - cs.numeric()'
        """
        if isinstance(node, UnaryExpr):
            not_falsy(node.op == "~")
            return ast.UnaryOp(ast.Invert(), not_none(self._selector(node.expr)))
        if isinstance(node, OpExpr):
            operator = not_none(self.OPERATORS.get(node.op))
            left = not_none(self._selector(node.left))
            right = not_none(self._selector(node.right))
            return ast.BinOp(left, operator(), right)
        call = not_none(node if isinstance(node, CallExpr) else None)
        function = not_none(self._function(call.callee))
        return self._call(ast.parse(function, mode="eval").body, call)

    def _function(self, callee: Expression) -> str | None:
        """Give the code of a function that builds a selector.

        Args:
            callee: The node of the called function.

        Returns:
            `cs.<name>` for a function of `polars.selectors`, `pl.col` or
            `pl.all`, or None for another function.

        Example:
            >>> SelectorSyntax()._function(named_reference_node("polars.selectors.numeric"))
            'cs.numeric'
            >>> SelectorSyntax()._function(named_reference_node("main.f")) is None
            True
        """
        if not isinstance(callee, RefExpr):
            return None
        fullname = callee.fullname
        if fullname.startswith(self.SELECTORS):
            return f"cs.{fullname.removeprefix(self.SELECTORS)}"
        return self.FUNCTIONS.get(fullname)

    @propagates(None)
    def _call(self, function: ast.expr, call: CallExpr) -> ast.expr | None:
        """Write a call with its arguments, each one a literal or a data type of polars.

        Args:
            function: The code of the function.
            call: The call node.

        Returns:
            The code of the call, or None where an argument is not a literal.

        Example:
            >>> from mypy.nodes import ARG_POS
            >>> call = CallExpr(NameExpr("col"), [str_literal_node("a")], [ARG_POS], [None])
            >>> function = ast.parse("pl.col", mode="eval").body
            >>> ast.unparse(SelectorSyntax()._call(function, call))
            "pl.col('a')"
        """
        positional: list[ast.expr] = []
        keywords: list[ast.keyword] = []
        for argument, name in zip(call.args, call.arg_names, strict=True):
            value = not_none(self._argument(argument))
            if name is None:
                positional.append(value)
            else:
                keywords.append(ast.keyword(name, value))
        return ast.Call(function, positional, keywords)

    @propagates(None)
    def _argument(self, node: Expression) -> ast.expr | None:
        """Write one argument: a literal, a list or a tuple of them, or a data type of polars.

        Args:
            node: The argument.

        Returns:
            The code, or None for another node.

        Example:
            >>> ast.unparse(SelectorSyntax()._argument(str_list_literal_node("a", "b")))
            "['a', 'b']"
            >>> SelectorSyntax()._argument(named_reference_node("main.x")) is None
            True
        """
        if isinstance(node, StrExpr | IntExpr | FloatExpr):
            return ast.Constant(node.value)
        if isinstance(node, NameExpr) and node.name in self.NAMED:
            return ast.Constant(self.NAMED[node.name])
        if isinstance(node, ListExpr | TupleExpr):
            items = [not_none(self._argument(item)) for item in node.items]
            return ast.List(items, ast.Load())
        if isinstance(node, CallExpr):
            return self._call(not_none(self._data_type(node.callee)), node)
        return self._data_type(node)

    def _data_type(self, node: Expression) -> ast.expr | None:
        """Write a data type class of polars, such as `pl.Int64`.

        Args:
            node: The node.

        Returns:
            The code, or None for another node.

        Example:
            >>> int64 = named_reference_node("polars.datatypes.classes.Int64")
            >>> ast.unparse(SelectorSyntax()._data_type(int64))
            'pl.Int64'
        """
        if not isinstance(node, RefExpr) or not node.fullname.startswith(self.DATA_TYPES):
            return None
        name = node.fullname.removeprefix(self.DATA_TYPES)
        return ast.Attribute(ast.Name("pl", ast.Load()), name, ast.Load())
