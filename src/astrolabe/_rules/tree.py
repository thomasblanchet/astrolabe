"""Read the tree that polars builds for an expression: each node with its identity and its inputs.

Polars serializes an expression as a tree in JSON, with `meta.serialize()`.
The evaluator of an expression follows that tree, so it needs, for each node,
the identity that finds its definition, the options that the definition
reads, and the inputs in the order of polars. Each shape of node keeps its
inputs under fixed keys: a function under `input`, a binary operator under
`left` and `right`, a `when` under `predicate`, `truthy` and `falsy`, an
aggregation under its own name. Polars decides which arguments of a method are
inputs, so this module keeps only where each shape holds them.

A leaf has no inputs: a column, a literal, a selector, `pl.len()`, the element
of `eval` and the field of a struct `eval`. An `eval` and a struct `eval` also
hold an evaluation, which runs over the elements of the input and not over the
frame. The tree keeps it apart from the inputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import KnownNan
from astrolabe._rules.definitions.node_options import NodeIdentity, NodeOptions
from astrolabe._rules.failures import ErrorClass, PolarsRefusal
from astrolabe._rules.releases import polars_releases

if TYPE_CHECKING:
    import polars as pl


class HorizontalLogic:
    """Rewrite `all_horizontal` and `any_horizontal` as the conversion of polars rewrites them.

    Polars turns these functions into nodes that its optimizer reads before
    it plans the expression. With no input, the function is a literal: `true`
    for `all_horizontal` and `false` for `any_horizontal`. One input becomes a
    strict cast of the input to `Boolean`. Two inputs or more, up to 127,
    become a chain of the binary `&` or `|`, from the left. The rewrite keeps
    the function of two inputs, whose definition is that binary operator,
    and nests it. With 128 inputs or more, the function stays.

    Example:
        >>> function = {"Boolean": "AllHorizontal"}
        >>> tree = {"Function": {"input": [{"Column": "a"}], "function": function}}
        >>> HorizontalLogic().rewritten(tree)["Cast"]["dtype"]
        {'Literal': 'Boolean'}
    """

    EMPTY: ClassVar[dict[str, bool]] = {"AllHorizontal": True, "AnyHorizontal": False}
    LIMIT: ClassVar[int] = 128

    def rewritten(self, serialized: object) -> object:
        """Give a node as the conversion leaves it.

        Args:
            serialized: The node, as `json.loads` gives it.

        Returns:
            The node, rewritten where it is one of the two functions with
            other than two inputs.

        Example:
            >>> parts = [{"Column": name} for name in "abc"]
            >>> function = {"Boolean": "AnyHorizontal"}
            >>> tree = {"Function": {"input": parts, "function": function}}
            >>> HorizontalLogic().rewritten(tree)["Function"]["input"][0]["Function"]["input"]
            [{'Column': 'a'}, {'Column': 'b'}]
            >>> HorizontalLogic().rewritten({"Function": {"input": [], "function": function}})
            {'Literal': {'Scalar': {'Boolean': False}}}
            >>> HorizontalLogic().rewritten({"Column": "a"})
            {'Column': 'a'}
            >>> between = {"Function": {"input": [], "function": {"Boolean": {"IsBetween": {}}}}}
            >>> HorizontalLogic().rewritten(between) == between
            True
        """
        body = serialized.get("Function") if isinstance(serialized, dict) else None
        function = body.get("function") if isinstance(body, dict) else None
        name = function.get("Boolean") if isinstance(function, dict) else None
        if not isinstance(body, dict) or not isinstance(name, str) or name not in self.EMPTY:
            return serialized
        inputs = list(body.get("input", []))
        if len(inputs) == 2 or len(inputs) >= self.LIMIT:
            return serialized
        if not inputs:  # pragma: no cover -- polars refuses a fold of no input
            return {"Literal": {"Scalar": {"Boolean": self.EMPTY[str(name)]}}}
        if len(inputs) == 1:
            boolean = {"Literal": "Boolean"}
            return {"Cast": {"expr": inputs[0], "dtype": boolean, "options": "Strict"}}
        folded = inputs[0]
        for part in inputs[1:]:
            folded = {"Function": {**body, "input": [folded, part]}}
        return folded


class AnyCast:
    """Take away a cast to the planner type `Unknown(Any)`, as the conversion of polars does.

    The engine of polars runs a cast to `Unknown(Any)` as no cast at all. From
    polars 1.44.1, the conversion builds no node for such a cast, so the node
    above reads the input of the cast directly, with its own type. Older
    releases keep the cast node, whose planner type is `Unknown(Any)`.

    Example:
        >>> cast = {"Cast": {"expr": {"Column": "a"}, "dtype": AnyCast.ANY, "options": "Strict"}}
        >>> from astrolabe._rules.releases import PolarsRelease
        >>> with PolarsRelease.pretend("1.44.1"):
        ...     AnyCast().rewritten(cast)
        {'Column': 'a'}
    """

    ANY: ClassVar[dict[str, object]] = {"Literal": {"Unknown": "Any"}}

    @polars_releases
    def rewritten(self, serialized: object) -> object:
        """Give a node as the conversion leaves it: the input of a cast to `Unknown(Any)`.

        Args:
            serialized: The node, as `json.loads` gives it.

        Returns:
            The input of the cast, where the node is a cast to `Unknown(Any)`,
            and the node otherwise.

        Example:
            >>> AnyCast().rewritten({"Column": "a"})
            {'Column': 'a'}
        """
        body = serialized.get("Cast") if isinstance(serialized, dict) else None
        if isinstance(body, dict) and body.get("dtype") == self.ANY:
            return self.rewritten(body.get("expr"))
        return serialized

    @rewritten.before("1.44.1")
    def _kept(self, serialized: object) -> object:
        """Give a node as the conversion of the older releases leaves it: unchanged.

        Args:
            serialized: The node, as `json.loads` gives it.

        Returns:
            The node.

        Example:
            >>> cast = {"Cast": {"expr": {"Column": "a"}, "dtype": AnyCast.ANY}}
            >>> AnyCast()._kept(cast) == cast
            True
        """
        return serialized


class KeptName:
    """Check that `.name.keep()` reads a column, as the conversion of polars does.

    `.name.keep()` names its output after the first column that its input
    reads. Polars refuses it while it converts the expression where the
    input reads no column, as over a literal or `pl.len()`.

    Example:
        >>> KeptName().check(ExpressionTree().read({"KeepName": {"Column": "a"}})) is None
        True
    """

    NAMED: ClassVar[frozenset[str]] = frozenset({"Column", "Selector", "Field"})

    def check(self, tree: TreeNode) -> None:
        """Refuse a `.name.keep()` whose input reads no column.

        Args:
            tree: The node of `.name.keep()`.

        Raises:
            PolarsRefusal: The input reads no column.

        Example:
            >>> literal = {"Literal": {"Dyn": {"Int": 1}}}
            >>> KeptName().check(ExpressionTree().read({"KeepName": literal}))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `name.keep_name` ...
        """
        pending = list(tree.inputs)
        while pending:
            node = pending.pop()
            if node.identity.shape in self.NAMED:
                return
            pending.extend(node.inputs)
        message = "`name.keep_name` expected at least one column name"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


@dataclass(frozen=True)
class TreeNode:
    """Hold one node of the tree of polars, with its identity, its inputs and its evaluation.

    Example:
        >>> node = ExpressionTree().read({"Agg": {"Sum": {"Column": "a"}}})
        >>> node.identity, node.inputs[0].identity
        (NodeIdentity(shape='Agg', function='Sum'), NodeIdentity(shape='Column', function=''))
    """

    serialized: object
    identity: NodeIdentity
    inputs: tuple[TreeNode, ...] = ()
    evaluation: tuple[TreeNode, ...] = ()

    def options(self) -> NodeOptions:
        """Give the options of the node, which its definition reads.

        Returns:
            The serialized node, as the definitions read it.

        Example:
            >>> body = {"expr": {"Column": "a"}, "dtype": {"Literal": "Int8"}, "options": "Strict"}
            >>> ExpressionTree().read({"Cast": body}).options().value(("dtype",))
            {'Literal': 'Int8'}
            >>> ExpressionTree().read("Len").options().value(())
            {}
        """
        if isinstance(self.serialized, dict):
            return NodeOptions(self.serialized)
        return NodeOptions({str(self.serialized): {}})  # pragma: no cover -- no definition reads it

    def is_leaf(self) -> bool:
        """Say whether the node is a leaf, which reads the frame or holds a value.

        Returns:
            Whether it is.

        Example:
            >>> ExpressionTree().read({"Column": "a"}).is_leaf()
            True
        """
        return self.identity.shape in ExpressionTree.LEAVES


class NonFinite:
    """Carry an infinity or a NaN through the tree of polars, which writes such a float as null.

    The JSON form of the tree of polars writes `inf`, `-inf` and NaN as
    `null`. Code that builds an expression from a value that it knows puts a
    stand-in in place of such a float: a finite value next to the largest
    float, which no user writes. The tree reads each stand-in back as the
    float that it stands for, a NaN as a `KnownNan`.

    Example:
        >>> NonFinite().of("-inf")
        -1.7976931348623155e+308
        >>> NonFinite().parsed("-1.7976931348623155e308")
        -inf
    """

    STAND_INS: ClassVar[dict[float, float]] = {
        1.7976931348623155e308: math.inf,
        -1.7976931348623155e308: -math.inf,
        1.7976931348623153e308: KnownNan("nan"),
    }

    def of(self, text: str) -> float:
        """Give the float of a text, with a stand-in for an infinity or a NaN.

        Args:
            text: The text, as `float` reads it.

        Returns:
            The float, or its stand-in.

        Example:
            >>> NonFinite().of("1.5"), NonFinite().of("nan")
            (1.5, 1.7976931348623153e+308)
        """
        value = float(text)
        if math.isnan(value):
            return 1.7976931348623153e308
        if math.isinf(value):
            return math.copysign(1.7976931348623155e308, value)
        return value

    def parsed(self, text: str) -> float:
        """Read a float of the JSON form of a tree, with the float that a stand-in stands for.

        Args:
            text: The text of the float.

        Returns:
            The float.

        Example:
            >>> NonFinite().parsed("0.5")
            0.5
        """
        value = float(text)
        return self.STAND_INS.get(value, value)


class ExpressionTree:
    """Read the tree of polars into nodes, each with its inputs in the order of polars.

    Example:
        >>> import polars as pl
        >>> tree = ExpressionTree().of(pl.col("a") + 1)
        >>> [node.identity.shape for node in tree.inputs]
        ['Column', 'Literal']
    """

    INPUTS: ClassVar[dict[str, tuple[str, ...]]] = {
        "Function": ("input",),
        "BinaryExpr": ("left", "right"),
        "Ternary": ("predicate", "truthy", "falsy"),
        "Filter": ("input", "by"),
        "Gather": ("expr", "idx"),
        "Slice": ("input", "offset", "length"),
        "Over": ("function", "partition_by", "order_by"),
        "Rolling": ("function", "index_column"),
        "Sort": ("expr",),
        "SortBy": ("expr", "by"),
        "Cast": ("expr",),
        "Explode": ("input",),
        "Eval": ("expr",),
        "StructEval": ("expr",),
        "RenameAlias": ("expr",),
    }
    LEAVES: ClassVar[frozenset[str]] = frozenset({
        "Column",
        "Literal",
        "Selector",
        "Len",
        "Element",
        "Field",
    })

    def of(self, expression: pl.Expr) -> TreeNode:
        """Give the tree of an expression.

        Args:
            expression: The expression, which polars serializes.

        Returns:
            The root of the tree.

        Example:
            >>> import polars as pl
            >>> ExpressionTree().of(pl.len()).identity
            NodeIdentity(shape='Len', function='')
        """
        serialized = expression.meta.serialize(format="json")
        return self.read(json.loads(serialized, parse_float=NonFinite().parsed))

    def read(self, serialized: object) -> TreeNode:
        """Give the node of a serialized tree, with its inputs and its evaluation.

        Args:
            serialized: The tree, as `json.loads` gives it.

        Returns:
            The node.

        Example:
            >>> tree = ExpressionTree().read({"Alias": [{"Column": "a"}, "b"]})
            >>> tree.identity.shape, tree.inputs[0].identity.shape
            ('Alias', 'Column')
        """
        serialized = AnyCast().rewritten(HorizontalLogic().rewritten(serialized))
        if not isinstance(serialized, dict):
            return TreeNode(serialized, NodeIdentity(str(serialized), ""))
        shape = next(iter(serialized))
        body = serialized[shape]
        inputs = tuple(self.read(part) for part in self._inputs(shape, body))
        evaluation = tuple(self.read(part) for part in self._evaluation(shape, body))
        return TreeNode(serialized, NodeIdentity.of(serialized), inputs, evaluation)

    def _inputs(self, shape: str, body: object) -> list[object]:
        """Give the serialized inputs of a node, in the order of polars.

        Args:
            shape: The shape of the node.
            body: Its body.

        Returns:
            The inputs.

        Example:
            >>> body = {"expr": {"Column": "a"}, "by": [{"Column": "k"}]}
            >>> ExpressionTree()._inputs("SortBy", body)
            [{'Column': 'a'}, {'Column': 'k'}]
            >>> ExpressionTree()._inputs("KeepName", {"Column": "a"})
            [{'Column': 'a'}]
            >>> ExpressionTree()._inputs("Explode", "a")
            []
        """
        if shape in self.LEAVES:
            return []
        if shape == "Agg" and isinstance(body, dict):
            return self._aggregated(body)
        if shape == "Alias" and isinstance(body, list):
            return body[:1]
        if shape == "KeepName":
            return [body]
        if not isinstance(body, dict):  # pragma: no cover -- each other node holds an object
            return []
        parts: list[object] = []
        for key in self.INPUTS.get(shape, ()):
            parts.extend(self._under(shape, key, body.get(key)))
        return parts

    def _under(self, shape: str, key: str, value: object) -> list[object]:
        """Give the inputs under one key of a body: one input, or each input of a list.

        The sort key of `over` stands as the first item of a pair with its
        sort options.

        Args:
            shape: The shape of the node.
            key: The key.
            value: The value under the key.

        Returns:
            The inputs.

        Example:
            >>> pair = [{"Column": "k"}, {"descending": False}]
            >>> ExpressionTree()._under("Over", "order_by", pair)
            [{'Column': 'k'}]
        """
        if value is None:
            return []
        if shape == "Over" and key == "order_by" and isinstance(value, list):
            return value[:1]
        if isinstance(value, list):
            return list(value)
        return [value]

    def _aggregated(self, body: dict[str, object]) -> list[object]:
        """Give the input of an aggregation, which sits under the name of the aggregation.

        The input is the value itself, the first item of a list, or the value
        of the key `input`.

        Args:
            body: The body of the `Agg` node.

        Returns:
            The input.

        Example:
            >>> body = {"Max": {"input": {"Column": "a"}, "propagate_nans": False}}
            >>> ExpressionTree()._aggregated(body)
            [{'Column': 'a'}]
        """
        value = next(iter(body.values()))
        if isinstance(value, list):
            return value[:1]
        if isinstance(value, dict) and "input" in value:
            return [value["input"]]
        return [value]

    def _evaluation(self, shape: str, body: object) -> list[object]:
        """Give the evaluation of an `eval` or a struct `eval`, which runs over the elements.

        Args:
            shape: The shape of the node.
            body: Its body.

        Returns:
            The trees of the evaluation, or nothing for another shape.

        Example:
            >>> ExpressionTree()._evaluation("Eval", {"expr": {}, "evaluation": "Element"})
            ['Element']
        """
        if shape not in {"Eval", "StructEval"} or not isinstance(body, dict):
            return []
        evaluation = body.get("evaluation")
        return list(evaluation) if isinstance(evaluation, list) else [evaluation]
