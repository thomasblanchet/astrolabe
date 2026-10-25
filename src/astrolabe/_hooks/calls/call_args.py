"""Read the arguments of a checked call by parameter name, as values and not as types."""

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

from mypy.nodes import (
    DictExpr,
    Expression,
    ListExpr,
    NameExpr,
    StrExpr,
    TupleExpr,
)

from astrolabe._hooks.calls.bare_literal import BareLiteralReader
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext, MethodContext
    from mypy.types import Type as MypyType


class CallArgs:
    """Read the argument values a caller typed, for code that checks the call.

    A check often needs the value itself, and not its type. In the
    call `head(3)`, the abstract syntax tree (AST) holds the number 3.
    The type of that argument says only that it is a whole number. A
    check that must know the number therefore reads the AST.

    Reading the AST is not simple. An argument can arrive by name or
    by position. A literal value can sit inside a wrapper call. This
    class does that work once, and answers a plain Python value.

    It sits in the plugin layer, between the type checker and the
    engine that decides the answer. The engine never reads the AST.
    """

    _BOOL_LITERAL_FULLNAMES: ClassVar[dict[str, bool]] = {
        "builtins.True": True,
        "builtins.False": False,
    }

    def __init__(self, ctx: MethodContext | FunctionContext) -> None:
        """Keep the mypy context of the call whose arguments this object reads.

        Args:
            ctx: The mypy context of a method call or a function call.
        """
        self._ctx = ctx
        self._bare_literal = BareLiteralReader()

    def node_list(self, name: str) -> list[Expression]:
        """Give the AST nodes that the caller passed for one parameter.

        mypy groups the arguments by parameter, so a node is found whether the caller wrote it
        by name or by position. Every other reader in this class starts here.

        Args:
            name: The name of the parameter.

        Returns:
            The nodes in call order. The list is empty if the caller left the parameter out. It
            is also empty if the called function has no parameter of that name. One hook can
            serve several functions, and only some of them can have the parameter.
        """
        if name not in self._ctx.callee_arg_names:
            return []
        return self._ctx.args[self._ctx.callee_arg_names.index(name)]

    def is_provided(self, name: str) -> bool:
        """Tell whether the caller passed a value for one parameter.

        Args:
            name: The name of the parameter.

        Returns:
            True if the call has an argument for `name`.
        """
        return bool(self.node_list(name))

    def is_omitted_or_none(self, name: str) -> bool:
        """Tell whether a parameter whose default is `None` gets its default.

        The caller can leave the argument out or write `None`. Both give the default, and a
        check treats them the same.

        Args:
            name: The name of the parameter.

        Returns:
            True if the call has no argument for `name`, or if the argument is `None`.

        Example:
            >>> call_args(["coalesce"], [[]]).is_omitted_or_none("coalesce")
            True
            >>> none_node = NameExpr("None")
            >>> none_node.fullname = "builtins.None"
            >>> call_args(["coalesce"], [[none_node]]).is_omitted_or_none("coalesce")
            True
        """
        nodes = self.node_list(name)
        if not nodes:
            return True
        node = nodes[0]
        return isinstance(node, NameExpr) and node.fullname == "builtins.None"

    def literal_bool(self, name: str, default: bool | None = None) -> bool | None:
        """Read a `True` or `False` that the caller wrote for one parameter.

        A flag such as `coalesce=True` can change the result of an operation. A check needs to
        know which way the caller set it.

        Args:
            name: The name of the parameter.
            default: The value to give if the call has no argument for `name`.

        Returns:
            The boolean, or `default` if the argument is missing. None if the argument is
            something other than the literal `True` or `False`, for example a variable.

        Example:
            >>> true_node = NameExpr("True")
            >>> true_node.fullname = "builtins.True"
            >>> call_args(["coalesce"], [[true_node]]).literal_bool("coalesce")
            True
            >>> call_args(["coalesce"], [[]]).literal_bool("coalesce", default=False)
            False
            >>> false_node = NameExpr("False")
            >>> false_node.fullname = "builtins.False"
            >>> call_args(["coalesce"], [[false_node]]).literal_bool("coalesce")
            False
            >>> call_args(["coalesce"], [[str_literal_node("nope")]]).literal_bool("coalesce")
        """
        nodes = self.node_list(name)
        if not nodes:
            return default
        node = nodes[0]
        if not isinstance(node, NameExpr):
            return None
        return self._BOOL_LITERAL_FULLNAMES.get(node.fullname)

    @propagates(None)
    def keyword_items(self, name: str) -> list[tuple[str, Expression]] | None:
        """Read the keywords and the values that the caller passed to a `**kwargs` parameter.

        In `with_columns(total=expr)`, the keyword `total` names the new column. mypy keeps the
        values of a `**kwargs` parameter in one list, and the keywords in a second list of the
        same order. This method pairs them.

        Args:
            name: The name of the `**kwargs` parameter.

        Returns:
            Each keyword with the AST node of its value, in call order. The list is empty if the
            caller passed no keyword argument. None if a value has no keyword, which a
            `**kwargs` parameter does not allow.

        Example:
            >>> bar_node = str_literal_node("bar-value")
            >>> args = call_args(["kwargs"], [[bar_node]], [["bar"]])
            >>> args.keyword_items("kwargs") == [("bar", bar_node)]
            True
        """
        index = self._ctx.callee_arg_names.index(name)
        names = self._ctx.arg_names[index]
        pairs: list[tuple[str, Expression]] = []
        for keyword_name, node in zip(names, self._ctx.args[index], strict=True):
            pairs.append((not_none(keyword_name), node))
        return pairs

    def node_type(self, name: str) -> MypyType | None:
        """Give the type that mypy inferred for the argument of one parameter.

        Some arguments matter for their type and not for how the caller wrote them. The `other`
        frame of a `join` is one example: its columns are in its type.

        Args:
            name: The name of the parameter.

        Returns:
            The type of the first argument, or None if the call has no argument for `name`.
        """
        types = self._ctx.arg_types[self._ctx.callee_arg_names.index(name)]
        if not types:
            return None
        return types[0]

    def type_list(self, name: str) -> list[MypyType]:
        """Give the types that mypy inferred for every argument of one parameter.

        This is `node_type` for a `*args` parameter. The list has the same order as the list
        from `node_list`, so item `i` of each describes the same argument.

        Args:
            name: The name of the parameter.

        Returns:
            The types in call order. The list is empty if the caller passed nothing for `name`.
        """
        return self._ctx.arg_types[self._ctx.callee_arg_names.index(name)]

    def literal_str(self, name: str, default: str | None = None) -> str | None:
        """Read a string that the caller passed for one parameter, such as a column name.

        The string can be a literal, or a variable whose value mypy knows because it is `Final`
        or has a `Literal[...]` type.

        Args:
            name: The name of the parameter.
            default: The value to give if the call has no argument for `name`.

        Returns:
            The string, or `default` if the argument is missing. None if the value of the
            argument is not a known string.
        """
        nodes = self.node_list(name)
        if not nodes:
            return default
        if len(nodes) != 1:  # pragma: no cover
            # Code calls this method only for a parameter that is not `*args`. Such a
            # parameter gets at most one node, so a list that is not empty has one item.
            return None
        node = nodes[0]
        if isinstance(node, StrExpr):
            return node.value
        instance_type = checked_not_none(self.node_type(name))
        return self._bare_literal.string(node, instance_type)

    def literal_str_list(self, name: str) -> list[str] | None:
        """Read one string, or a list of strings, that the caller passed for one parameter.

        Polars accepts both `columns="a"` and `columns=["a", "b"]`. This method reads both
        forms. A known `Final` or `Literal[...]` string variable can take the place of any of
        the strings.

        Args:
            name: The name of the parameter.

        Returns:
            The strings in order. None if the argument is missing or is neither form.
        """
        nodes = self.node_list(name)
        if not nodes:
            return None
        node = nodes[0]
        single = self._bare_literal.string_via_api(node, self._ctx.api)
        if single is not None:
            return [single]
        if isinstance(node, ListExpr | TupleExpr):
            return self._resolve_flat_str_list(node.items)
        return None

    def literal_str_list_or_empty(self, name: str) -> list[str] | None:
        """Read one string or a list of strings, where a missing argument means no strings.

        For some parameters, a missing argument or `None` means an empty list. For example,
        `index=None` in `unpivot` picks no index column. This method gives an empty list in that
        case, and keeps None for an argument that it cannot read.

        Args:
            name: The name of the parameter.

        Returns:
            An empty list if the argument is missing or is `None`. Otherwise the strings, or
            None if the argument is neither one string nor a list of them.

        Example:
            >>> call_args(["index"], [[]]).literal_str_list_or_empty("index")
            []
            >>> call_args(["index"], [[str_literal_node("x")]]).literal_str_list_or_empty("index")
            ['x']
        """
        if self.is_omitted_or_none(name):
            return []
        return self.literal_str_list(name)

    def vararg_str_list(self, name: str) -> list[str] | None:
        """Read the strings that the caller passed to a `*args` parameter, such as column names.

        A `*args` parameter gets one node for each argument. One list or tuple literal counts as
        its items, as `vararg_node_list` explains.

        Args:
            name: The name of the `*args` parameter.

        Returns:
            The strings in call order. The list is empty if the caller passed nothing. None if
            one argument is not a known string.
        """
        return self._resolve_flat_str_list(self.vararg_node_list(name))

    def vararg_node_list(self, name: str) -> list[Expression]:
        """Give the argument nodes of a `*args` parameter, with one list literal opened up.

        Polars reads one list or tuple passed to a `*args` parameter as its items.
        `drop(["a", "b"])` is the same call as `drop("a", "b")`. This method therefore gives
        the items of such a literal in place of the literal.

        Args:
            name: The name of the `*args` parameter.

        Returns:
            The AST node of each argument, in call order. The list is empty if the caller passed
            nothing.

        Example:
            >>> literal = str_list_literal_node("a", "b")
            >>> args = call_args(["columns"], [[literal]])
            >>> args.vararg_node_list("columns") == literal.items
            True
        """
        nodes = self.node_list(name)
        if len(nodes) != 1:
            return nodes
        node = nodes[0]
        if isinstance(node, ListExpr | TupleExpr):
            return node.items
        return nodes

    @propagates(None)
    def literal_mapping(self, name: str) -> dict[str, str] | None:
        """Read a dict literal of strings to strings, such as the mapping of `rename`.

        This reads `rename({"a": "b"})`. A known `Final` or `Literal[...]` string variable can
        be a key or a value. A function argument, such as `rename(str.upper)`, is not a dict
        literal, so this method cannot read it.

        Args:
            name: The name of the parameter.

        Returns:
            The mapping. None if the argument is missing, is not a dict literal, or has a key or
            a value that is not a known string. A `**other` entry has no key, so it also gives
            None.
        """
        nodes = self.node_list(name)
        if not nodes:
            return None
        node = nodes[0]
        if not isinstance(node, DictExpr):
            return None
        api = self._ctx.api
        mapping: dict[str, str] = {}
        for key, value in node.items:
            checked_key = not_none(key)
            str_key = not_none(self._bare_literal.string_via_api(checked_key, api))
            str_value = not_none(self._bare_literal.string_via_api(value, api))
            mapping[str_key] = str_value
        return mapping

    @propagates(None)
    def _resolve_flat_str_list(self, nodes: list[Expression]) -> list[str] | None:
        """Read each of several AST nodes as a string.

        Args:
            nodes: The AST nodes to read.

        Returns:
            The strings in order, or None if one node is not a known string.
        """
        api = self._ctx.api
        names: list[str] = []
        for node in nodes:
            names.append(not_none(self._bare_literal.string_via_api(node, api)))
        return names
