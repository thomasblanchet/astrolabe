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
from astrolabe._rules.definitions.node_options import NodeIdentity, NodeOptions, OptionReader
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


class EarlierForms:
    """Rewrite a node that polars builds before 1.41.0 into the form that polars 1.41.0 builds.

    Before polars 1.41.0, `quantile` is an aggregation that holds its input and
    its quantile, and the members `reverse`, `unique` and `n_unique` of `list`
    and `arr` are functions of their own. From that release, `quantile` is a
    function of two inputs, and each of these members is an `eval` of the
    element. The rules hold the later forms, and the two forms give the same
    types. Before polars 1.40.0, `tail(n)` also builds its offset as `-n`
    cast to `Int64` where the later releases write the literal `-n`
    (`_tail`).

    Example:
        >>> member = {"ListExpr": "Reverse"}
        >>> reversed_list = {"Function": {"input": [{"Column": "l"}], "function": member}}
        >>> EarlierForms()._later(reversed_list)["Eval"]["variant"]
        'List'
    """

    VARIANTS: ClassVar[dict[tuple[str, str], object]] = {
        ("ListExpr", "Reverse"): "List",
        ("ListExpr", "Unique"): "List",
        ("ListExpr", "NUnique"): "ListAgg",
        ("ArrayExpr", "Reverse"): {"Array": {"as_list": False}},
        ("ArrayExpr", "Unique"): {"Array": {"as_list": True}},
        ("ArrayExpr", "NUnique"): "ArrayAgg",
    }

    @polars_releases
    def rewritten(self, serialized: object) -> object:
        """Give a node in the form that the installed release builds: unchanged.

        Args:
            serialized: The node, as `json.loads` gives it.

        Returns:
            The node.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.41.0"):
            ...     EarlierForms().rewritten({"Column": "a"})
            {'Column': 'a'}
        """
        return serialized

    @rewritten.before("1.40.0")
    def _earliest(self, serialized: object) -> object:
        """Give a node of a release before 1.40.0 in the form of polars 1.41.0.

        Args:
            serialized: The node, as `json.loads` gives it.

        Returns:
            The node, with the offset of `tail` as a literal (`_tail`) and the
            later forms of the releases before 1.41.0 (`_later`).

        Example:
            >>> EarlierForms()._earliest({"Column": "a"})
            {'Column': 'a'}
        """
        return self._later(self._tail(serialized))

    def _tail(self, serialized: object) -> object:
        """Give the slice of `tail(n)` with its offset as the literal `-n`.

        Args:
            serialized: The node, as `json.loads` gives it.

        Returns:
            The slice with the literal offset where the node is a slice whose
            offset is `-n` cast to `Int64`, and the node otherwise.

        Example:
            >>> three = {"Literal": {"Dyn": {"Int": 3}}}
            >>> to_int64 = {"expr": three, "dtype": {"Literal": "Int64"}, "options": "Overflowing"}
            >>> cast = {"Cast": to_int64}
            >>> offset = {"Function": {"input": [cast], "function": "Negate"}}
            >>> body = {"input": {"Column": "x"}, "offset": offset, "length": three}
            >>> EarlierForms()._tail({"Slice": body})["Slice"]["offset"]
            {'Literal': {'Dyn': {'Int': -3}}}
            >>> EarlierForms()._tail({"Slice": {**body, "offset": three}})["Slice"]["offset"]
            {'Literal': {'Dyn': {'Int': 3}}}
        """
        body = serialized.get("Slice") if isinstance(serialized, dict) else None
        if not isinstance(body, dict):
            return serialized
        count = self._negated_count(body.get("offset"))
        if count is None:
            return serialized
        return {"Slice": {**body, "offset": {"Literal": {"Dyn": {"Int": -count}}}}}

    def _negated_count(self, offset: object) -> int | None:
        """Give `n` where an offset is `-n` cast to `Int64`, as an older `tail(n)` builds it.

        Args:
            offset: The offset, as `json.loads` gives it.

        Returns:
            The count, or `None` for any other offset.

        Example:
            >>> EarlierForms()._negated_count({"Literal": {"Dyn": {"Int": 3}}}) is None
            True
        """
        function = offset.get("Function") if isinstance(offset, dict) else None
        if not isinstance(function, dict) or function.get("function") != "Negate":
            return None
        inputs = function.get("input")
        first = inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else None
        return self._cast_count(first.get("Cast") if isinstance(first, dict) else None)

    def _cast_count(self, cast: object) -> int | None:
        """Give `n` where a cast is the integer literal `n` cast to `Int64`.

        Args:
            cast: The body of the cast, as `json.loads` gives it.

        Returns:
            The count, or `None` for any other cast.

        Example:
            >>> EarlierForms()._cast_count({"expr": {"Column": "n"}, "dtype": {"Literal": "Int64"}})
            >>> EarlierForms()._cast_count({"expr": {"Column": "n"}, "dtype": {"Literal": "Int8"}})
        """
        if not isinstance(cast, dict) or cast.get("dtype") != {"Literal": "Int64"}:
            return None
        literal = cast.get("expr")
        value = literal.get("Literal") if isinstance(literal, dict) else None
        dynamic = value.get("Dyn") if isinstance(value, dict) else None
        count = dynamic.get("Int") if isinstance(dynamic, dict) else None
        return count if isinstance(count, int) else None

    @rewritten.before("1.41.0")
    def _later(self, serialized: object) -> object:
        """Give a node of an older release in the form of polars 1.41.0.

        Args:
            serialized: The node, as `json.loads` gives it.

        Returns:
            The function of `quantile`, or the `eval` of a member, where the node
            is one of them, and the node otherwise.

        Example:
            >>> body = {"expr": {"Column": "x"}, "quantile": {"Column": "q"}, "method": "Linear"}
            >>> EarlierForms()._later({"Agg": {"Quantile": body}})["Function"]["function"]
            {'Quantile': {'method': 'Linear'}}
            >>> EarlierForms()._later({"Agg": {"Sum": {"Column": "x"}}})
            {'Agg': {'Sum': {'Column': 'x'}}}
            >>> EarlierForms()._later({"Function": {"input": [{"Column": "x"}], "function": "Abs"}})
            {'Function': {'input': [{'Column': 'x'}], 'function': 'Abs'}}
            >>> EarlierForms()._later("Element")
            'Element'
        """
        if not isinstance(serialized, dict):
            return serialized
        aggregation = serialized.get("Agg")
        if isinstance(aggregation, dict) and isinstance(aggregation.get("Quantile"), dict):
            return self._quantile(aggregation["Quantile"])
        body = serialized.get("Function")
        member = self._member(body) if isinstance(body, dict) else None
        return serialized if member is None else member

    def _quantile(self, body: dict[str, object]) -> object:
        """Give the function of `quantile`, whose inputs are the value and the quantile.

        Args:
            body: The body of the aggregation.

        Returns:
            The function.

        Example:
            >>> body = {"expr": {"Column": "x"}, "quantile": {"Column": "q"}, "method": "Nearest"}
            >>> EarlierForms()._quantile(body)["Function"]["input"]
            [{'Column': 'x'}, {'Column': 'q'}]
        """
        inputs = [body.get("expr"), body.get("quantile")]
        function = {"Quantile": {"method": body.get("method")}}
        return {"Function": {"input": inputs, "function": function}}

    def _member(self, body: dict[str, object]) -> object | None:
        """Give the `eval` of the element that a member of `list` or `arr` is in polars 1.41.0.

        Args:
            body: The body of the function.

        Returns:
            The `eval`, or `None` where the function is not one of the members.

        Example:
            >>> member = {"ArrayExpr": "NUnique"}
            >>> body = {"input": [{"Column": "a"}], "function": member}
            >>> EarlierForms()._member(body)["Eval"]["evaluation"]
            {'Agg': {'NUnique': 'Element'}}
            >>> unique = {"input": [{"Column": "l"}], "function": {"ListExpr": {"Unique": True}}}
            >>> EarlierForms()._member(unique)["Eval"]["evaluation"]["Function"]["function"]
            {'Unique': True}
            >>> EarlierForms()._member({"input": [], "function": {"ListExpr": "Sum"}}) is None
            True
        """
        function = body.get("function")
        if not isinstance(function, dict):
            return None
        namespace, member = next(iter(function.items()))
        name = next(iter(member)) if isinstance(member, dict) else str(member)
        variant = self.VARIANTS.get((namespace, name))
        if variant is None:
            return None
        if name == "NUnique":
            evaluation: object = {"Agg": {"NUnique": "Element"}}
        else:
            evaluation = {"Function": {"input": ["Element"], "function": member}}
        inputs = body.get("input")
        value = inputs[0] if isinstance(inputs, list) else None
        return {"Eval": {"expr": value, "evaluation": evaluation, "variant": variant}}


class WindowKeys:
    """Check the names of the keys of `over`, as the conversion of polars does.

    The polars 1.40 releases refuse an `over` with two keys of one output
    name, as two literals, with `DuplicateError`, while they convert the
    expression. The releases before and after them do not compare the names.

    Example:
        >>> body = {"function": {"Column": "x"}, "partition_by": []}
        >>> WindowKeys()._distinct(ExpressionTree().read({"Over": body})) is None
        True
    """

    @polars_releases
    def check(self, tree: TreeNode) -> None:
        """Check the keys of an `over`: no check.

        Args:
            tree: The node of `over`.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> body = {"function": {"Column": "x"}, "partition_by": [{"Column": "g"}] * 2}
            >>> tree = ExpressionTree().read({"Over": body})
            >>> with PolarsRelease.pretend("1.41.0"):
            ...     WindowKeys().check(tree)
        """
        del tree

    @check.before("1.41.0")
    def _distinct(self, tree: TreeNode) -> None:
        """Refuse two keys of one output name, as the polars 1.40 releases do.

        Args:
            tree: The node of `over`.

        Raises:
            PolarsRefusal: Two keys have one output name.

        Example:
            >>> keys = [{"Literal": {"Dyn": {"Int": 0}}}, {"Literal": {"Dyn": {"Int": 1}}}]
            >>> body = {"function": {"Column": "x"}, "partition_by": keys}
            >>> WindowKeys()._distinct(ExpressionTree().read({"Over": body}))
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: DuplicateError: column with name 'literal' ...
        """
        body = tree.serialized.get("Over") if isinstance(tree.serialized, dict) else None
        keys = body.get("partition_by") if isinstance(body, dict) else None
        seen: set[str] = set()
        for name in OptionReader().names(keys if isinstance(keys, list) else []):
            if name in seen:
                message = f"column with name '{name}' has more than one occurrence"
                raise PolarsRefusal(ErrorClass.DUPLICATE, message)
            seen.add(name)

    @check.before("1.40.0")
    def _unchecked(self, tree: TreeNode) -> None:
        """Check the keys of an `over` in the releases before 1.40.0: no check.

        Args:
            tree: The node of `over`.

        Example:
            >>> keys = [{"Literal": {"Dyn": {"Int": 0}}}, {"Literal": {"Dyn": {"Int": 1}}}]
            >>> body = {"function": {"Column": "x"}, "partition_by": keys}
            >>> WindowKeys()._unchecked(ExpressionTree().read({"Over": body}))
        """
        del tree


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
        serialized = EarlierForms().rewritten(serialized)
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
