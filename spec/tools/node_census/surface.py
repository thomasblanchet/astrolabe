"""The nodes of the census, read from polars' own tree of each census call.

The census calls of `CensusCalls` fix their arguments, for example
`pl.col('x').clip(0)`. A test varies a node over the outcomes of all its
inputs, not only of its receiver. So this module
serializes each call, finds the inputs of the root node in polars' tree,
and builds the node again with any input at each position. Polars decides
which arguments are inputs, so this module keeps no list of them.
"""

from __future__ import annotations

import copy
import io
import json
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import ClassVar

import polars as pl

from spec.tools.node_census.calls import CensusCalls


@dataclass(frozen=True)
class Slot:
    """Name one input position inside the body of a root node.

    `path` leads from the body of the root to the input, through dictionary
    keys and list indexes.

    Example:
        >>> Slot(("input", 1)).read({"input": [{"Column": "a"}, {"Column": "b"}]})
        {'Column': 'b'}
    """

    path: tuple[str | int, ...]

    def read(self, body: object) -> object:
        """Give the part of a body at this slot.

        Args:
            body: The body of a root node, as parsed JSON.

        Returns:
            The input at the slot.

        Example:
            >>> Slot(("left",)).read({"left": {"Column": "a"}, "op": "Plus"})
            {'Column': 'a'}
        """
        part = body
        for step in self.path:
            part = part[step]  # type: ignore[index]
        return part

    def write(self, body: object, value: object) -> None:
        """Put a value at this slot of a body.

        Args:
            body: The body of a root node, as parsed JSON. It changes in place.
            value: The input to put there.

        Example:
            >>> body = {"left": {"Column": "a"}}
            >>> Slot(("left",)).write(body, {"Column": "z"})
            >>> body
            {'left': {'Column': 'z'}}
        """
        part = body
        for step in self.path[:-1]:
            part = part[step]  # type: ignore[index]
        part[self.path[-1]] = value  # type: ignore[index]


class SlotFinder:
    """Find the input slots of a root node from its shape.

    Each shape of node keeps its inputs under fixed keys. An `Eval` and a
    `StructEval` also hold an evaluation, which is an option and not an
    input. An `Agg` holds its input under the name of the aggregation.

    Example:
        >>> SlotFinder().slots("BinaryExpr", {"left": {}, "op": "Plus", "right": {}})
        [Slot(path=('left',)), Slot(path=('right',))]
    """

    KEYS: ClassVar[dict[str, tuple[str, ...]]] = {
        "Function": ("input",),
        "BinaryExpr": ("left", "right"),
        "Ternary": ("predicate", "truthy", "falsy"),
        "Filter": ("input", "by"),
        "Gather": ("expr", "idx"),
        "Slice": ("input", "offset", "length"),
        "Over": ("function", "partition_by"),
        "Rolling": ("function", "index_column"),
        "Sort": ("expr",),
        "SortBy": ("expr", "by"),
        "Cast": ("expr",),
        "Explode": ("input",),
        "Eval": ("expr",),
        "StructEval": ("expr",),
    }

    def slots(self, shape: str, body: object) -> list[Slot]:
        """Give the input slots of a root node, in polars' order.

        Args:
            shape: The shape of the root, for example `Function`.
            body: The body of the root, as parsed JSON.

        Returns:
            The slots, or an empty list for a shape that has no known inputs.

        Example:
            >>> SlotFinder().slots("Function", {"input": [{}, {}], "function": "Abs"})
            [Slot(path=('input', 0)), Slot(path=('input', 1))]
        """
        if shape == "Agg" and isinstance(body, dict):
            return self._aggregation(body)
        if shape == "Alias" and isinstance(body, list):
            return [Slot((0,))]
        if not isinstance(body, dict):
            return []
        slots: list[Slot] = []
        for key in self.KEYS.get(shape, ()):
            slots.extend(self._under(body, key))
        return slots

    def _under(self, body: dict[str, object], key: str) -> list[Slot]:
        """Give the slots under one key: one for an input, one for each input of a list.

        Args:
            body: The body of the root.
            key: The key.

        Returns:
            The slots.

        Example:
            >>> SlotFinder()._under({"by": [{"Column": "a"}, {"Column": "b"}]}, "by")
            [Slot(path=('by', 0)), Slot(path=('by', 1))]
        """
        value = body.get(key)
        if isinstance(value, list):
            return [Slot((key, index)) for index, _ in enumerate(value)]
        if isinstance(value, dict):
            return [Slot((key,))]
        return []

    def _aggregation(self, body: dict[str, object]) -> list[Slot]:
        """Give the slot of the input of an aggregation.

        The input sits under the name of the aggregation, as the value itself,
        as the first item of a list, or under the key `input`.

        Args:
            body: The body of the `Agg` root.

        Returns:
            The slot.

        Example:
            >>> SlotFinder()._aggregation({"Max": {"input": {}, "propagate_nans": False}})
            [Slot(path=('Max', 'input'))]
        """
        name, value = next(iter(body.items()))
        if isinstance(value, list):
            return [Slot((name, 0))]
        if isinstance(value, dict) and "input" in value:
            return [Slot((name, "input"))]
        return [Slot((name,))]


class NodeTemplate:
    """Build one node of the census with any input at each position.

    It keeps the serialized tree of a census call and the slots of its root.
    `build` puts the serialized inputs in the slots and reads the tree back.

    Example:
        >>> node = NodeTemplate("root.clip", "pl.col('x').clip(0)")
        >>> node.inputs(), node.shape()
        (2, 'Function')
        >>> str(node.build([pl.col("a"), pl.col("b")]))
        'col("a").clip_min([col("b")])'
    """

    def __init__(self, key: str, code: str) -> None:
        """Serialize the call and find the slots of its root.

        Args:
            key: The name of the call in the census.
            code: The code of the call, which reads the column `x`.

        A function of `pl` that names its node with an alias, such as
        `pl.repeat`, is read at the node under the alias.

        Example:
            >>> NodeTemplate("root.abs", "pl.col('x').abs()").key
            'root.abs'
            >>> NodeTemplate("pl.repeat", "pl.repeat(1, 2)").shape()
            'Function'
        """
        self.key = key
        self.code = code
        names = {"pl": pl, "time": time, "date": date, "datetime": datetime}
        # The code comes from the census, not from a user.
        expression = eval(code, names)  # ruff: ignore[suspicious-eval-usage]
        self._tree: dict[str, object] = json.loads(expression.meta.serialize(format="json"))
        alias = self._tree.get("Alias")
        if key.startswith("pl.") and isinstance(alias, list) and isinstance(alias[0], dict):
            self._tree = alias[0]
        self._original = copy.deepcopy(self._tree)
        self._shape: str = next(iter(self._tree))
        self._slots = SlotFinder().slots(self._shape, self._tree[self._shape])

    def shape(self) -> str:
        """Give the shape of the root node.

        Returns:
            The shape, for example `BinaryExpr`.

        Example:
            >>> NodeTemplate("root.add", "pl.col('x').add(pl.col('x'))").shape()
            'BinaryExpr'
        """
        return self._shape

    def inputs(self) -> int:
        """Give the number of inputs of the root node.

        Returns:
            The number of slots.

        Example:
            >>> NodeTemplate("root.sum", "pl.col('x').sum()").inputs()
            1
        """
        return len(self._slots)

    def original(self, position: int) -> pl.Expr:
        """Give the input that the census call itself has at a position.

        A node with four inputs or more keeps the inputs of its census call
        at the positions that the census does not vary.

        Args:
            position: The index of the input.

        Returns:
            The input of the call at that position.

        Example:
            >>> str(NodeTemplate("root.clip", "pl.col('x').clip(0)").original(1))
            'dyn int: 0'
        """
        slot = self._slots[position]
        part = slot.read(self._original[self._shape])
        text = io.StringIO(json.dumps(part))
        return pl.Expr.deserialize(text, format="json")

    def build(self, inputs: list[pl.Expr]) -> pl.Expr:
        """Build the node with the given inputs.

        Args:
            inputs: One expression for each slot.

        Returns:
            The node.

        Example:
            >>> node = NodeTemplate("root.add", "pl.col('x').add(pl.col('x'))")
            >>> str(node.build([pl.col("a"), pl.lit(1)]))
            '[(col("a")) + (dyn int: 1)]'
        """
        tree = copy.deepcopy(self._tree)
        body = tree[self._shape]
        for slot, expression in zip(self._slots, inputs, strict=True):
            slot.write(body, json.loads(expression.meta.serialize(format="json")))
        text = io.StringIO(json.dumps(tree))
        return pl.Expr.deserialize(text, format="json")


class CensusSurface:
    """List the nodes of the census: one template for each census call that polars can rebuild.

    A call that polars cannot serialize, or whose root has no input that
    this tool can free, is left out with its reason. `EXTRA_CALLS` adds the
    calls that polars builds as a node of their own and that the census
    calls of the checker leave out: `replace_strict` with no `default` is a
    node of three inputs, beside the node of four. It also gives `and_` and
    `or_` a second input, because with none polars gives the column itself.

    Example:
        >>> surface = CensusSurface()
        >>> len(surface.templates()) + len(surface.skipped()) == len(surface._calls())
        True
    """

    EXTRA_CALLS: ClassVar[dict[str, str]] = {
        "root.replace_strict.no_default": "pl.col('x').replace_strict({1: 2})",
        "root.and_": "pl.col('x').and_(pl.col('x'))",
        "root.or_": "pl.col('x').or_(pl.col('x'))",
    }

    def __init__(self) -> None:
        """Build the templates of all census calls.

        Example:
            >>> CensusSurface().skipped()["pl.col"]
            'no input'
        """
        self._templates: list[NodeTemplate] = []
        self._skipped: dict[str, str] = {}
        for key, code in self._calls().items():
            self._add(key, code)

    def _calls(self) -> dict[str, str]:
        """Give the code of every census call, by name.

        The calls are the methods of `pl.Expr`, the members of its
        namespaces, and the functions of `pl` that build an expression,
        under the key `pl.<name>`.

        Returns:
            The calls.

        Example:
            >>> CensusSurface()._calls()["root.abs"]
            "pl.col('x').abs()"
            >>> CensusSurface()._calls()["pl.coalesce"]
            "pl.coalesce('x', 'y')"
        """
        census = CensusCalls()
        calls = {f"root.{name}": code for name, code in census.root_calls().items()}
        calls |= census.members()
        for name, code in census.constructors().items():
            calls[f"pl.{name}"] = code
        return calls | self.EXTRA_CALLS

    def _add(self, key: str, code: str) -> None:
        """Add the template of one call, or record why it is left out.

        Args:
            key: The name of the call.
            code: Its code.

        Example:
            >>> surface = CensusSurface()
            >>> "root.inspect" in surface.skipped()
            True
        """
        try:
            template = NodeTemplate(key, code)
            placeholders: list[pl.Expr] = [
                pl.col(f"__{index}") for index in range(template.inputs())
            ]
            template.build(placeholders)
        except Exception as error:  # any failure of polars leaves the call out
            self._skipped[key] = type(error).__name__
            return
        if template.inputs() == 0:
            self._skipped[key] = "no input"
            return
        self._templates.append(template)

    def templates(self) -> list[NodeTemplate]:
        """Give the templates.

        Returns:
            One template for each call of the census that polars can rebuild.

        Example:
            >>> CensusSurface().templates()[0].key
            'root.abs'
        """
        return list(self._templates)

    def skipped(self) -> dict[str, str]:
        """Give the calls that the census leaves out, with the reason.

        Returns:
            The reason for each left-out call.

        Example:
            >>> CensusSurface().skipped()["root.inspect"]
            'ComputeError'
        """
        return dict(self._skipped)
