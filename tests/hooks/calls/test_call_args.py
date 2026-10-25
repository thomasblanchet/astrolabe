"""Check `CallArgs`, the plugin class that reads the values a caller wrote in a call.

A hook often needs an argument value and not only its type, for
example the column names in `drop("a", "b")`. `CallArgs` reads those
values from the context that mypy gives a hook. It reads only the
argument nodes, their types, the parameter names and the API from
that context. These tests therefore give it a plain object with those
four parts, and need no mypy build.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from mypy.nodes import DictExpr, Expression, ListExpr, NameExpr, StrExpr

from astrolabe._hooks.calls.call_args import CallArgs

if TYPE_CHECKING:
    from mypy.plugin import MethodContext
    from mypy.types import Instance
    from mypy.types import Type as MypyType

    from tests.hooks.conftest import FakeApi


def make_call_args(api: FakeApi, **named_arg_nodes: list[Expression]) -> CallArgs:
    """Build a `CallArgs` over a call that the test describes by parameter.

    Args:
        api: The stand-in checker API. A method that reads a variable
            asks it for the type of the variable. Each node with a type in
            `api.expression_types` also gets that type in the argument
            types of the call, in the same position, as in a real mypy
            context.
        named_arg_nodes: One keyword for each parameter. Its value is the
            list of AST nodes that the caller wrote for it, or `[]` if the
            caller left it out.

    Returns:
        The reader over the call.
    """
    names = list(named_arg_nodes)
    nodes = [named_arg_nodes[name] for name in names]
    arg_types = [_registered_types(api, group) for group in nodes]
    ctx = SimpleNamespace(args=nodes, arg_types=arg_types, callee_arg_names=names, api=api)
    return CallArgs(cast("MethodContext", ctx))


def _registered_types(api: FakeApi, nodes: list[Expression]) -> list[MypyType]:
    """Give the types that the test stored for some nodes.

    Args:
        api: The stand-in checker API that holds the stored types.
        nodes: The AST nodes.

    Returns:
        The stored type of each node that has one, in order. A node with
        no stored type is left out.
    """
    types: list[MypyType] = []
    for node in nodes:
        if id(node) in api.expression_types:
            types.append(api.expression_types[id(node)])
    return types


def test_node_list_returns_the_nodes_for_a_provided_argument(api: FakeApi) -> None:
    node = StrExpr("foo")
    call_args = make_call_args(api, columns=[node])
    assert call_args.node_list("columns") == [node]


def test_node_list_is_empty_for_an_unprovided_argument(api: FakeApi) -> None:
    call_args = make_call_args(api, columns=[])
    assert call_args.node_list("columns") == []


def test_is_provided_true_when_nodes_are_present(api: FakeApi) -> None:
    call_args = make_call_args(api, columns=[StrExpr("foo")])
    assert call_args.is_provided("columns") is True


def test_is_provided_false_when_absent(api: FakeApi) -> None:
    call_args = make_call_args(api, columns=[])
    assert call_args.is_provided("columns") is False


def test_literal_str_reads_a_single_string_literal(api: FakeApi) -> None:
    call_args = make_call_args(api, name=[StrExpr("foo")])
    assert call_args.literal_str("name") == "foo"


def test_literal_str_returns_the_default_when_not_passed(api: FakeApi) -> None:
    call_args = make_call_args(api, name=[])
    assert call_args.literal_str("name", default="fallback") == "fallback"


def test_literal_str_is_none_for_a_non_literal_argument(api: FakeApi, str_dtype: Instance) -> None:
    # str_dtype stands in for the type of an ordinary variable. Like that
    # type, it holds no known value, so the reader cannot find the string.
    node = NameExpr("some_variable")
    api.expression_types[id(node)] = str_dtype
    call_args = make_call_args(api, name=[node])
    assert call_args.literal_str("name") is None


def test_literal_str_list_accepts_a_single_string(api: FakeApi) -> None:
    call_args = make_call_args(api, columns=[StrExpr("foo")])
    assert call_args.literal_str_list("columns") == ["foo"]


def test_literal_str_list_accepts_a_list_of_strings(api: FakeApi) -> None:
    node = ListExpr([StrExpr("foo"), StrExpr("bar")])
    call_args = make_call_args(api, columns=[node])
    assert call_args.literal_str_list("columns") == ["foo", "bar"]


def test_literal_str_list_is_none_when_not_passed(api: FakeApi) -> None:
    call_args = make_call_args(api, columns=[])
    assert call_args.literal_str_list("columns") is None


def test_literal_str_list_is_none_for_a_list_with_a_non_literal_item(
    api: FakeApi,
    str_dtype: Instance,
) -> None:
    variable = NameExpr("some_variable")
    api.expression_types[id(variable)] = str_dtype
    node = ListExpr([StrExpr("foo"), variable])
    call_args = make_call_args(api, columns=[node])
    assert call_args.literal_str_list("columns") is None


def test_vararg_str_list_collects_every_positional_node(api: FakeApi) -> None:
    call_args = make_call_args(api, columns=[StrExpr("foo"), StrExpr("bar")])
    assert call_args.vararg_str_list("columns") == ["foo", "bar"]


def test_vararg_str_list_is_empty_not_none_when_absent(api: FakeApi) -> None:
    call_args = make_call_args(api, columns=[])
    assert call_args.vararg_str_list("columns") == []


def test_literal_mapping_reads_string_key_value_pairs(api: FakeApi) -> None:
    node = DictExpr([(StrExpr("a"), StrExpr("b"))])
    call_args = make_call_args(api, mapping=[node])
    assert call_args.literal_mapping("mapping") == {"a": "b"}


def test_literal_mapping_is_none_when_not_a_dict_literal(api: FakeApi) -> None:
    call_args = make_call_args(api, mapping=[NameExpr("str.upper")])
    assert call_args.literal_mapping("mapping") is None


def test_literal_mapping_is_none_for_a_non_literal_value(api: FakeApi, str_dtype: Instance) -> None:
    variable = NameExpr("some_variable")
    api.expression_types[id(variable)] = str_dtype
    node = DictExpr([(StrExpr("a"), variable)])
    call_args = make_call_args(api, mapping=[node])
    assert call_args.literal_mapping("mapping") is None
