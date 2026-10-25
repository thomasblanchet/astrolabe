"""Read the syntax of a declaration call, and give its errors a place to point at.

A declaration such as `typed_expr(Quantity, gives=pl.Boolean)` is read
during semantic analysis, the mypy phase that resolves names. No
expression has a type yet, so the code that reads a declaration works
on the abstract syntax tree (AST) of the call: its argument nodes, its
callee and its string literals. The classes here hold the small reads
that every declaration parser needs, so that each parser reads the call
in the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import ARG_NAMED, ARG_POS, CallExpr, DictExpr, RefExpr, SetExpr, StrExpr

from astrolabe._hooks.dispatch.generic import Target

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import DynamicClassDefContext, SemanticAnalyzerPluginInterface


class DeclarationCall:
    """Find the argument nodes of one declaration call, by position or by keyword.

    A mypy call node keeps its arguments in three parallel lists: the
    nodes, their kinds and their keywords. A parser wants a question
    such as "which node did the user pass as `gives=`?", and not three
    lists to walk in step. This class answers such questions.

    Example:
        >>> call = dtype_constructor_call([str_literal_node("q")], ["named"])
        >>> DeclarationCall(call).keyword("named").value
        'q'
        >>> DeclarationCall(call).positional()
        []
    """

    def __init__(self, call: CallExpr) -> None:
        """Keep the call node whose arguments this object reads.

        Args:
            call: The call node of the declaration, or of a call inside
                it such as `row(...)`.
        """
        self._call = call

    def positional(self) -> list[Expression]:
        """Give every argument that the user passed by position, in order.

        Returns:
            The argument nodes. A `*args` argument is not among them.

        Example:
            >>> call = dtype_constructor_call([str_literal_node("a")])
            >>> len(DeclarationCall(call).positional())
            1
        """
        return [
            node
            for kind, node in zip(self._call.arg_kinds, self._call.args, strict=True)
            if kind == ARG_POS
        ]

    def keyword(self, name: str) -> Expression | None:
        """Give the node that the user passed under one keyword.

        Args:
            name: The keyword, without the `=`.

        Returns:
            The node, or None if the call does not pass that keyword.

        Example:
            >>> call = dtype_constructor_call([str_literal_node("a")])
            >>> DeclarationCall(call).keyword("gives") is None
            True
            >>> named = dtype_constructor_call([str_literal_node("a")], ["gives"])
            >>> DeclarationCall(named).keyword("gives").value
            'a'
        """
        for keyword, kind, node in self._items():
            if keyword == name and kind == ARG_NAMED:
                return node
        return None

    def keywords(self) -> list[str]:
        """Give every keyword that the call passes, so that a parser can refuse an unknown one.

        Returns:
            The keywords, in the order of the call. A `**kwargs` argument
            is not among them.

        Example:
            >>> call = dtype_constructor_call([str_literal_node("a")], ["gives"])
            >>> DeclarationCall(call).keywords()
            ['gives']
        """
        return [keyword for keyword, kind, _ in self._items() if kind == ARG_NAMED and keyword]

    def argument(self, position: int, keyword: str) -> Expression | None:
        """Give the node for a parameter that the user can pass by position or by keyword.

        `column("price", pl.Float64)` and `column(name="price", dtype=pl.Float64)`
        give the same nodes. The keyword wins if the call passes both.

        Args:
            position: The index of the parameter among the arguments
                passed by position.
            keyword: The keyword of the parameter.

        Returns:
            The node, or None if the call passes the parameter in
            neither way.

        Example:
            >>> node = str_literal_node("bar")
            >>> DeclarationCall(dtype_constructor_call([node])).argument(0, "name") is node
            True
            >>> call = dtype_constructor_call([node], ["name"])
            >>> DeclarationCall(call).argument(0, "name") is node
            True
        """
        named = self.keyword(keyword)
        if named is not None:
            return named
        positional = self.positional()
        if position >= len(positional):
            return None
        return positional[position]

    def _items(self) -> list[tuple[str | None, object, Expression]]:
        """Put the three parallel argument lists of the call side by side.

        Returns:
            One `(keyword, kind, node)` triple for each argument, in the
            order of the call.

        Example:
            >>> call = dtype_constructor_call([str_literal_node("a")])
            >>> DeclarationCall(call)._items()[0][0] is None
            True
        """
        return list(zip(self._call.arg_names, self._call.arg_kinds, self._call.args, strict=True))


class CallShape:
    """Recognize a call to one declaration function, or a reference to one value.

    A declaration is built from calls such as `row(...)`, `wrap(...)` and
    `column(...)`. A parser must know which function a node calls. A
    user can import the function under another name, so the parser
    compares the fullname that mypy resolved, and not the text of the
    call. This class takes the function as a real Python object and
    derives its fullname, so that a rename of the function cannot break
    the comparison.

    Example:
        >>> import polars as pl
        >>> CallShape().call(dtype_constructor_call([]), pl.Datetime) is not None
        True
    """

    def call(self, node: Expression, function: object) -> CallExpr | None:
        """Give a node as a call to one function, so that a parser can read its arguments.

        Args:
            node: The node to look at.
            function: The function, as a real Python object.

        Returns:
            The call node, or None for any other node.

        Example:
            >>> CallShape().call(str_literal_node("a"), str) is None
            True
        """
        return self._call_named(node, self.fullname_of(function))

    def call_of_value(self, node: Expression, value: object, name: str) -> CallExpr | None:
        """Give a node as a call of a callable object, for example `same("price")`.

        `same` is an instance, and not a function or a class. It has no
        name of its own to derive a fullname from, so the caller gives
        the name of the variable that holds it. The module comes from the
        class of the value.

        Args:
            node: The node to look at.
            value: The value. Only its module is read.
            name: The name of the variable that holds the value in that
                module.

        Returns:
            The call node, or None for any other node.

        Example:
            >>> CallShape().call_of_value(str_literal_node("a"), object(), "same") is None
            True
        """
        return self._call_named(node, f"{type(value).__module__}.{name}")

    def _call_named(self, node: Expression, fullname: str) -> CallExpr | None:
        """Give a node as a call whose callee resolves to one fullname.

        Args:
            node: The node to look at.
            fullname: The fullname that the callee must resolve to.

        Returns:
            The call node, or None if the node is not a call, or its
            callee is not a name, or the name resolves elsewhere.

        Example:
            >>> call = dtype_constructor_call([])
            >>> CallShape()._call_named(call, "polars.datatypes.classes.Datetime") is call
            True
        """
        if not isinstance(node, CallExpr) or not isinstance(node.callee, RefExpr):
            return None
        if node.callee.fullname != fullname:
            return None
        return node

    def refers_to(self, node: Expression, value: object, name: str) -> bool:
        """Answer whether a node is a bare reference to one value, for example `same`.

        Args:
            node: The node to look at.
            value: The value. Only its module is read.
            name: The name of the variable that holds the value in that
                module.

        Returns:
            True if the node is a name that resolves to that variable.

        Example:
            >>> CallShape().refers_to(str_literal_node("a"), object(), "same")
            False
        """
        if not isinstance(node, RefExpr):
            return False
        return node.fullname == f"{type(value).__module__}.{name}"

    def fullname_of(self, function: object) -> str:
        """Give the fullname that mypy uses for a function.

        Args:
            function: The function, as a real Python object.

        Returns:
            The fullname, derived from the object and not typed by hand.

        Example:
            >>> from astrolabe.declare import typed_expr
            >>> CallShape().fullname_of(typed_expr)
            'astrolabe.declare.typed_expr'
        """
        return Target.of(function).fullname  # type: ignore[arg-type]

    def literal(self, node: Expression | None) -> str | None:
        """Read the text of a string literal, for example a column name.

        Args:
            node: The node, or None if the argument is missing.

        Returns:
            The string, or None if the node is not a string literal.

        Example:
            >>> CallShape().literal(str_literal_node("a"))
            'a'
            >>> CallShape().literal(None) is None
            True
        """
        if isinstance(node, StrExpr):
            return node.value
        return None


class DeclarationSite:
    """Let an error class report at a declaration call.

    An error class reports through an object with two attributes:
    `context`, the node that the error points at, and `api`, the mypy
    interface that prints it. The context of a hook call has both. The
    context that mypy gives a dynamic-class hook holds the node under
    `call` instead. This class gives that node under the expected name.
    """

    def __init__(self, ctx: DynamicClassDefContext) -> None:
        """Keep the context of the declaration.

        Args:
            ctx: The mypy context of the declaration.
        """
        self._ctx = ctx

    @property
    def context(self) -> CallExpr:
        """The declaration call, where an error points."""
        return self._ctx.call

    @property
    def api(self) -> SemanticAnalyzerPluginInterface:
        """The mypy interface that reports an error."""
        return self._ctx.api


class UnresolvedReferences:
    """Find a name in a declaration call that mypy has not resolved yet.

    A declaration can name something that the module defines further
    down, for example a `Columns` class. mypy analyzes a module in
    several passes, and on an early pass that name has no node yet. A
    hook that reads the call on that pass reports the name as wrong, and
    mypy keeps that error after the name resolves. A hook therefore asks
    this class first, and does nothing on such a pass.

    The hook does not ask mypy to defer the statement. Another hook
    registers its class again on each pass, and a deferral beside it
    makes mypy repeat its passes up to its limit.

    Example:
        >>> UnresolvedReferences().found(dtype_constructor_call([str_literal_node("us")]))
        False
    """

    def found(self, node: Expression) -> bool:
        """Answer whether a node, or any node inside it, is a name that is not resolved.

        Args:
            node: The declaration call, or a part of it.

        Returns:
            True if a name at any depth has neither a node nor a
            fullname yet.

        Example:
            >>> from mypy.nodes import NameExpr
            >>> UnresolvedReferences().found(NameExpr("Later"))
            True
        """
        if isinstance(node, RefExpr):
            return node.node is None and not node.fullname
        return any(self.found(part) for part in self._parts(node))

    def _parts(self, node: Expression) -> list[Expression]:
        """Give the nodes one level inside a node, where a name can hide.

        Args:
            node: The node.

        Returns:
            The callee and the arguments of a call, the items of a set,
            or the keys and the values of a dict. A `**` item of a dict
            has no key, so it gives only its value. Any other node gives
            nothing.

        Example:
            >>> len(UnresolvedReferences()._parts(str_literal_node("a")))
            0
            >>> from mypy.nodes import DictExpr
            >>> item = (str_literal_node("a"), str_literal_node("b"))
            >>> len(UnresolvedReferences()._parts(DictExpr([item])))
            2
            >>> unpacked = (None, str_literal_node("b"))
            >>> len(UnresolvedReferences()._parts(DictExpr([unpacked])))
            1
        """
        if isinstance(node, CallExpr):
            return [node.callee, *node.args]
        if isinstance(node, SetExpr):
            return list(node.items)
        if not isinstance(node, DictExpr):
            return []
        parts: list[Expression] = []
        for key, value in node.items:
            if key is not None:
                parts.append(key)
            parts.append(value)
        return parts
