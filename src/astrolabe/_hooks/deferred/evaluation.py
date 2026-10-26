"""Evaluate the code of a deferred expression at the columns of one frame.

A deferred type stores the code of an expression, not its data types. When the
expression meets a frame, for example as an argument of `select`, the checker
needs the column that the expression gives there. This module computes it. It
builds the expression again from its code (`CodeReader`), reads the tree of
nodes that polars builds (`ExpressionTree`), and evaluates the tree with the
rules of polars at the data types of the frame (`TreeEvaluation`).

The module works on plain values: the code, and the rendering of each data
type, that is, the text that polars prints for it. So it needs no access to
mypy, and the code that calls it turns mypy types into renderings and back.
"""

from __future__ import annotations

import ast
import datetime
import decimal
import io
import itertools
import json
import math
import tokenize
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import polars as pl
import polars.selectors as cs

from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.deferred.code import CodeRefs
from astrolabe._hooks.deferred.code_reader import (
    BuildRefusal,
    CodeReader,
    CodeReadError,
    HoleValues,
    OpaqueCall,
    RunRefusal,
    UnwrittenTree,
)
from astrolabe._hooks.deferred.open_types import (
    Generalization,
    LeafChoices,
    OpenParameters,
    OpenType,
)
from astrolabe._hooks.deferred.polars_selector import NameRequirement
from astrolabe._rules.definitions.node_options import OptionReader
from astrolabe._rules.dtypes import DType, Is, NoPartError, Rendering, RenderingError
from astrolabe._rules.execution import GroupBy, QueryFlags
from astrolabe._rules.failures import PolarsRefusal, UndecidedError
from astrolabe._rules.node_evaluation import ColumnLeaf, Refused, Unchecked
from astrolabe._rules.options import Height, RowForm
from astrolabe._rules.tree import ExpressionTree
from astrolabe._rules.tree_evaluation import JoinKeyRows, TreeEvaluation

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from astrolabe._hooks.deferred.code import Code
    from astrolabe._hooks.deferred.code_reader import ReadCode
    from astrolabe._hooks.remainder.selector import Selector
    from astrolabe._rules.node_evaluation import Answer, CellAnswers
    from astrolabe._rules.tree import TreeNode


@dataclass(frozen=True)
class DeclaredLeaf:
    """Hold a declaration that a code reads by reference: its name, reads, rows and type.

    A parameter such as `price: Price` gives the code a leaf, `$0`. Where the
    declaration is a column, the leaf reads the column of its name, and has
    its data type, as the rendering of polars. Any other declaration, as
    `typed_agg(...)`, is a declared map: `reads` names the columns that it
    reads, `form` says how it counts rows, and `outcome` gives the type that
    it promises at the data types of the frame. The name is the output name,
    and the empty name of a blank that no call binds. A declared selection
    stands for each column that it picks, so its leaf reads no column until
    the evaluation puts one picked column in its place (`selection`). A
    column whose name is a union of literals, as `Literal["a", "b"]`, keeps
    those names in `choices`: it reads one of them, and no code knows which.

    Example:
        >>> DeclaredLeaf("price", "Float64").rendering
        'Float64'
    """

    name: str
    rendering: str
    reads: tuple[str, ...] | None = None
    form: RowForm = RowForm.PER_ROW
    outcome: LeafOutcome | None = None
    selection: LeafSelection | None = None
    choices: tuple[str, ...] = ()

    def read_names(self) -> tuple[str, ...]:
        """Give the names of the columns that the leaf reads.

        Returns:
            The names: the declared reads of a map, or the name of a column.

        Example:
            >>> DeclaredLeaf("price", "Float64").read_names()
            ('price',)
        """
        return (self.name,) if self.reads is None else self.reads


class LeafOutcome(Protocol):
    """Give the type that a declared map promises at the data types of a frame."""

    def at(self, columns: Mapping[str, str]) -> str | None:
        """Give the rendering of the type, or `None` where the frame is outside the declaration."""


class LeafSelection(Protocol):
    """Give the leaf of a declared selection at one picked column, and the rule of its picks."""

    @property
    def rule(self) -> Selector:
        """Give the rule of the columns that the selection picks."""

    def picked(self, column: str) -> DeclaredLeaf:
        """Give the declared map that the selection is at the column of this name."""


@dataclass(frozen=True)
class DeferredAnswer:
    """Hold the column that a deferred expression gives on one frame, or why it gives none.

    `rendering` is the data type of the column, or `Unchecked` where the
    rules do not know it. `refused` says that polars refuses the expression
    at these data types, and `reason` says why the answer is not a type.
    `cause` names the source of an `Unchecked` answer: `code` for a code
    that the reader cannot build, `number` for a number that the checker
    cannot know, `hole` for another value that the checker cannot know,
    `opaque` for a function of the user, `open` for a column whose data type
    the frame does not fully declare, `name` for a column whose name the
    checker cannot know, and `rules` for a question that the rules cannot
    answer. `reads` names the columns of the frame that the
    expression reads. `cast` names the source and the target of a cast at
    the root that polars refuses. `build` names the class and the message of
    the error that polars raises while it builds the expression, whatever
    the frame. `source` names the method of a function of the user that
    leaves the answer `Unchecked`, as `map_elements`. `refusals` holds each
    distinct refusal where every choice of the data types refuses, and
    `reason` holds the one that most choices share. `stored` is the type
    that a lazy frame stores for the column, where it differs from the type
    of the data. `partial` names the column that the selector of the code
    picks only at some data types of its sample, where the frame does not
    fully declare the data type of the column: the data then decides
    whether the selector picks it.

    Example:
        >>> DeferredAnswer("a", "Int64").refused
        False
    """

    name: str | None
    rendering: str
    refused: bool = False
    reason: str = ""
    reads: tuple[str, ...] = ()
    cause: str = ""
    cast: tuple[str, str] | None = None
    build: tuple[str, str] | None = None
    source: str = ""
    refusals: tuple[str, ...] = ()
    stored: str = ""
    partial: str = ""


@dataclass(frozen=True)
class Sample:
    """Hold the answer of a code at one choice of the values of its holes.

    `opened` says that the parameters of the answer that are not data types
    are open, because they can change with the parameters of the columns.

    Example:
        >>> Sample("Int8").refused
        False
    """

    rendering: str
    refused: bool = False
    reason: str = ""
    cause: str = ""
    cast: tuple[str, str] | None = None
    opened: bool = False


class Substitution:
    """Put a column in the place of the selector of an expression, as the expansion of polars does.

    Polars serializes a selector as a leaf of the tree. The substitution
    replaces that leaf with a column, and builds the expression again. With
    no selector, it gives the expression as it is.

    Example:
        >>> import polars.selectors as cs
        >>> expression = cs.numeric().abs()
        >>> node = Substitution().selectors(expression)[0]
        >>> Substitution(node, "a")(expression).meta.root_names()
        ['a']
    """

    def __init__(
        self, selector: object = None, column: str = "", replacement: object = None
    ) -> None:
        """Keep the serialized node and the node that takes its place.

        Args:
            selector: The serialized node, or `None` for no substitution.
            column: The name of the column that takes the place of the node.
            replacement: Another serialized node that takes its place, where
                no column does.

        Example:
            >>> Substitution()._selector is None
            True
        """
        self._selector = selector
        self._column = column
        self._replacement = {"Column": column} if replacement is None else replacement

    def __call__(self, expression: pl.Expr) -> pl.Expr:
        """Give the expression with the column in the place of the selector.

        Args:
            expression: The expression.

        Returns:
            The new expression, or the same one where there is no selector.

        Example:
            >>> Substitution()(pl.col("a")).meta.output_name()
            'a'
        """
        if self._selector is None:
            return expression
        tree = json.loads(expression.meta.serialize(format="json"))
        replaced = self._replaced(tree)
        return pl.Expr.deserialize(io.StringIO(json.dumps(replaced)), format="json")

    def calls(self, calls: Sequence[OpaqueCall]) -> list[OpaqueCall]:
        """Give the calls of a function of the user, with the column in the place of the selector.

        The code reader keeps the receiver of each such call apart from the
        tree, so the receivers take the column too.

        Args:
            calls: The calls.

        Returns:
            The new calls.

        Example:
            >>> import polars.selectors as cs
            >>> call = OpaqueCall(cs.numeric(), "map_batches")
            >>> selector = {"Selector": {"ByDType": "Numeric"}}
            >>> Substitution(selector, "a").calls([call])[0].receiver.meta.root_names()
            ['a']
        """
        return [replace(call, receiver=self(call.receiver)) for call in calls]

    def selectors(
        self,
        expression: pl.Expr,
        calls: Sequence[OpaqueCall] = (),
    ) -> list[object]:
        """Give each distinct selector of an expression, serialized.

        Args:
            expression: The expression.
            calls: The calls of a function of the user, whose receivers can hold
                a selector too.

        Returns:
            The selectors.

        Example:
            >>> import polars.selectors as cs
            >>> Substitution().selectors(cs.numeric() + 1)
            [{'Selector': {'ByDType': 'Numeric'}}]
        """
        found: list[object] = []
        roots = [expression, *(call.receiver for call in calls)]
        pending: list[object] = [json.loads(root.meta.serialize(format="json")) for root in roots]
        while pending:
            node = pending.pop()
            if self._is_selector(node):
                found.extend([] if node in found else [node])
            elif isinstance(node, dict):
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)
        return found

    def _is_selector(self, node: object) -> bool:
        """Say whether a serialized node is a selector.

        Args:
            node: The node.

        Returns:
            Whether the node is a map of `Selector` to its body.

        Example:
            >>> Substitution()._is_selector({"Selector": "Wildcard"})
            True
        """
        return isinstance(node, dict) and "Selector" in node and len(node) == 1

    def selector(self, node: object) -> cs.Selector:
        """Give the selector of polars from its serialized form.

        Args:
            node: The serialized selector.

        Returns:
            The selector.

        Example:
            >>> Substitution().selector({"Selector": {"ByDType": "Numeric"}})
            cs.numeric()
        """
        expression = pl.Expr.deserialize(io.StringIO(json.dumps(node)), format="json")
        return expression.meta.as_selector()

    def _replaced(self, node: object) -> object:
        """Give a serialized tree with the column in the place of the selector.

        Args:
            node: The serialized tree.

        Returns:
            The new tree.

        Example:
            >>> selector = {"Selector": {"ByDType": "Numeric"}}
            >>> Substitution(selector, "a")._replaced({"Alias": [selector, "b"]})
            {'Alias': [{'Column': 'a'}, 'b']}
        """
        if node == self._selector:
            return self._replacement
        if isinstance(node, dict):
            return {key: self._replaced(value) for key, value in node.items()}
        if isinstance(node, list):
            return [self._replaced(value) for value in node]
        return node


class FieldNodes:
    """Find and rebuild the node of a read of several fields of a struct, in a serialized tree.

    Polars writes `.struct.unnest()` and `.struct.field("a", "b")` as one
    function, `SelectFields`, of the struct and of a selector of its fields.

    Example:
        >>> node = FieldNodes().of(pl.col("s").struct.unnest())
        >>> FieldNodes().struct_of(node).meta.root_names()
        ['s']
    """

    def of(self, expression: pl.Expr) -> dict[str, Any] | None:
        """Give the one node of a read of several fields of an expression.

        Args:
            expression: The expression.

        Returns:
            The node, or `None` where the expression holds no such node, or more than one.

        Example:
            >>> FieldNodes().of(pl.col("s").struct.field("a")) is None
            True
        """
        found: list[dict[str, Any]] = []
        pending: list[object] = [json.loads(expression.meta.serialize(format="json"))]
        while pending:
            node = pending.pop()
            if isinstance(node, list):
                pending.extend(node)
            if not isinstance(node, dict):
                continue
            if self._selects(node):
                found.append(node)
            pending.extend(node.values())
        return found[0] if len(found) == 1 else None

    def _selects(self, node: Mapping[str, Any]) -> bool:
        """Say whether a serialized node is a read of several fields.

        Args:
            node: The node.

        Returns:
            Whether it is.

        Example:
            >>> FieldNodes()._selects({"Column": "a"})
            False
        """
        body = node.get("Function")
        function = body.get("function") if isinstance(body, dict) else None
        struct = function.get("StructExpr") if isinstance(function, dict) else None
        return isinstance(struct, dict) and "SelectFields" in struct

    def struct_of(self, node: Mapping[str, Any]) -> pl.Expr:
        """Give the struct that a read of several fields reads.

        Args:
            node: The node of the read.

        Returns:
            The expression of the struct.

        Example:
            >>> node = FieldNodes().of(pl.col("s").struct.unnest())
            >>> str(FieldNodes().struct_of(node))
            'col("s")'
        """
        struct = node["Function"]["input"][0]
        return pl.Expr.deserialize(io.StringIO(json.dumps(struct)), format="json")

    def selector_of(self, node: Mapping[str, Any]) -> cs.Selector:
        """Give the selector of the fields that a read of several fields picks.

        Args:
            node: The node of the read.

        Returns:
            The selector.

        Example:
            >>> node = FieldNodes().of(pl.col("s").struct.field("a", "b"))
            >>> FieldNodes().selector_of(node)
            [cs.empty() | cs.by_name('a', 'b', require_all=true)]
        """
        body = node["Function"]["function"]["StructExpr"]["SelectFields"]
        return Substitution().selector({"Selector": body})

    def required_of(self, node: Mapping[str, Any]) -> list[str]:
        """Give the names of the fields that a read of several fields requires the struct to hold.

        Args:
            node: The node of the read.

        Returns:
            The names.

        Example:
            >>> FieldNodes().required_of(FieldNodes().of(pl.col("s").struct.field("a", "b")))
            ['a', 'b']
        """
        return NameRequirement().required(
            node["Function"]["function"]["StructExpr"]["SelectFields"]
        )

    def read_of(self, node: Mapping[str, Any], name: str) -> dict[str, Any]:
        """Give the node of a read of one field, in the place of a read of several.

        Args:
            node: The node of the read of several fields.
            name: The name of the field.

        Returns:
            The node of the read of that field.

        Example:
            >>> node = FieldNodes().of(pl.col("s").struct.unnest())
            >>> FieldNodes().read_of(node, "a")["Function"]["function"]
            {'StructExpr': {'FieldByName': 'a'}}
        """
        inputs = node["Function"]["input"]
        return {"Function": {"input": inputs, "function": {"StructExpr": {"FieldByName": name}}}}


class KeyCode:
    """Read the tree of the code of a join key, which polars checks as the code writes it.

    Most operations of polars read only what an expression computes. A join
    also reads the tree of each key: polars refuses an alias anywhere in it,
    and a key that changes the rows. This class reads the code of a key into
    the expression of polars, and answers both from its tree.

    Example:
        >>> from astrolabe._hooks.deferred.code import Code
        >>> KeyCode().holds_alias(Code("pl.col('a').alias('b')"))
        True
    """

    ALIASED: ClassVar[frozenset[str]] = frozenset({"cum_fold", "cum_reduce"})

    def holds_alias(self, code: Code[DeclaredLeaf | None]) -> bool:
        """Say whether `.alias()` stands anywhere in the tree of a code.

        Polars refuses an alias anywhere in a join key. A rename through
        `.name`, such as `.name.suffix()`, is no alias. The method reads the
        expressions that a call of a function of the user reads too, as the
        code writes them. `pl.cum_fold` and `pl.cum_reduce` name their result
        with an alias of their own, so a key with one of them holds an alias.

        Args:
            code: The code.

        Returns:
            Whether it does. A code that the reader cannot read holds none.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> evaluation = KeyCode()
            >>> evaluation.holds_alias(Code("pl.col('a').alias('x') + 1"))
            True
            >>> evaluation.holds_alias(Code("pl.col('a').name.suffix('x')"))
            False
            >>> evaluation.holds_alias(Code("pl.col("))
            False
            >>> evaluation.holds_alias(Code("1"))
            False
            >>> evaluation.holds_alias(Code("pl.cum_reduce(hole('function'), [pl.col('a')])"))
            True
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:
            return False
        if not isinstance(read.expression, pl.Expr):
            return False
        if any(call.method in self.ALIASED for call in read.opaque):
            return True
        roots = [read.expression]
        for call in read.opaque:
            roots.extend(call.written())
        return self._aliased_tree(roots)

    def refused(self, code: Code[DeclaredLeaf | None]) -> bool:
        """Say whether polars refuses a code as a key of a join, because the key changes the rows.

        Polars needs a key that works value by value at each node, or one
        value (`JoinKeyRows`). A call of a function of the user stands as an
        expression of the same facts (`KeyStandIn`). A call whose facts the
        code does not show, and a node that no definition states, leave the
        key unknown, so they refuse nothing.

        Args:
            code: The code.

        Returns:
            Whether polars refuses it.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> evaluation = KeyCode()
            >>> evaluation.refused(Code("pl.col('a').shift(1)"))
            True
            >>> evaluation.refused(Code("pl.col('a') + pl.col('b').sum()"))
            True
            >>> evaluation.refused(Code("pl.col('a').cast(pl.Int8) + 1"))
            False
            >>> evaluation.refused(Code("pl.col("))
            False
            >>> evaluation.refused(Code("pl.col('a').map_batches(hole('function'))"))
            True
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:
            return False
        expression = read.expression
        stand_ins = KeyStandIn().all_of(read)
        if not isinstance(expression, pl.Expr) or stand_ins is None:
            return False
        leaf = ColumnLeaf(DType("Int64"), RowForm.PER_ROW)
        roots = [expression, *stand_ins.values()]
        columns = dict.fromkeys(itertools.chain(*(root.meta.root_names() for root in roots)), leaf)
        rows = JoinKeyRows(TreeEvaluation(columns, GroupBy((3,))))
        return rows.refuses(KeyStandIn().substituted(expression, stand_ins)) is True

    def _aliased_tree(self, roots: Sequence[pl.Expr]) -> bool:
        """Say whether an alias node stands anywhere in the trees of some expressions.

        Args:
            roots: The expressions.

        Returns:
            Whether one does.

        Example:
            >>> evaluation = KeyCode()
            >>> evaluation._aliased_tree([pl.col("a"), pl.struct(pl.col("b").alias("x"))])
            True
            >>> evaluation._aliased_tree([pl.col("a").name.keep()])
            False
        """
        pending: list[object] = [json.loads(root.meta.serialize(format="json")) for root in roots]
        while pending:
            node = pending.pop()
            if isinstance(node, list):
                pending.extend(node)
            if not isinstance(node, dict):
                continue
            if "Alias" in node:
                return True
            pending.extend(node.values())
        return False


class KeyStandIn:
    """Give an expression that stands for a call of a function of the user in a join key.

    Polars checks a join key node by node: a key of one value moves into a
    column first, and every other key must work value by value at each node.
    A call that takes a function of the user has no tree that the rules
    read, but polars reads both facts from the call itself. `map_batches`
    and `map_groups` take them from `is_elementwise=` and
    `returns_scalar=`, which are `False` by default. `pl.fold` and
    `pl.reduce` never work value by value, and take `returns_scalar=`.
    `map_elements` works value by value, and `rolling_map` does not.

    The stand-in is a stated expression of the same facts over the receiver
    of the call: the receiver itself for a call that works value by value,
    its `first()` for a call that gives one value, and its `cum_sum()` for a
    call that keeps the rows and does not work value by value. The check of
    the key then reads the stand-in as polars reads the call.

    Example:
        >>> call = OpaqueCall(pl.col("a"), "map_batches", {"returns_scalar": True})
        >>> str(KeyStandIn().of(call))
        'col("a").first()'
    """

    FIXED: ClassVar[dict[str, tuple[bool, bool]]] = {
        "map_elements": (True, False),
        "rolling_map": (False, False),
    }
    FLAGGED: ClassVar[frozenset[str]] = frozenset({"map_batches", "map_groups"})
    FOLDS: ClassVar[frozenset[str]] = frozenset({"fold", "reduce"})

    def all_of(self, read: ReadCode) -> dict[str, pl.Expr] | None:
        """Give the stand-in of each call of a code, by the reserved column that holds its place.

        A hole of a `bool` or of a known set takes several values, and a
        keyword of a call can hold it. The reader gives one of the values,
        so where the code holds such a hole beside a call, the code does not
        show the facts of the call.

        Args:
            read: The code, as the reader builds it.

        Returns:
            The stand-ins, or `None` where the code does not show the facts of
            a call.

        Example:
            >>> from astrolabe._hooks.deferred.code_reader import ReadCode
            >>> elements = OpaqueCall(pl.col("a"), "map_elements")
            >>> KeyStandIn().all_of(ReadCode(None, ("function",), (elements,)))
            {'__opaque_0__': <Expr ['col("a")'] at ...>}
            >>> KeyStandIn().all_of(ReadCode(None, ("function", "bool"), (elements,))) is None
            True
            >>> pipe = OpaqueCall(pl.col("a"), "pipe")
            >>> KeyStandIn().all_of(ReadCode(None, ("function",), (pipe,))) is None
            True
        """
        chosen = any(kind == "bool" or HoleValues().listed(kind) for kind in read.holes)
        if read.opaque and chosen:
            return None
        stand_ins = {}
        for place, call in enumerate(read.opaque):
            stand_in = self.of(call)
            if stand_in is None:
                return None
            stand_ins[DeferredEvaluation.OPAQUE.format(place)] = stand_in
        return stand_ins

    def of(self, call: object) -> pl.Expr | None:
        """Give the stand-in of one call.

        Args:
            call: The call of a function of the user.

        Returns:
            The stand-in, or `None` where the code does not show its facts.

        Example:
            >>> str(KeyStandIn().of(OpaqueCall(pl.col("a"), "rolling_map")))
            'col("a").cum_sum()'
        """
        facts = self._facts(call) if isinstance(call, OpaqueCall) else None
        if not isinstance(call, OpaqueCall) or facts is None:
            return None
        elementwise, scalar = facts
        if scalar:
            return call.receiver.first()
        return call.receiver if elementwise else call.receiver.cum_sum()

    def substituted(self, expression: pl.Expr, stand_ins: Mapping[str, pl.Expr]) -> TreeNode:
        """Give the tree of an expression with each reserved column replaced by its stand-in.

        A stand-in can read the reserved column of an earlier call, as a call
        over the result of another call does, so the replacement goes down
        into each stand-in too.

        Args:
            expression: The expression, which reads the reserved columns.
            stand_ins: The stand-ins, by reserved column.

        Returns:
            The tree.

        Example:
            >>> stand_ins = {"__opaque_0__": pl.col("a").cum_sum()}
            >>> tree = KeyStandIn().substituted(pl.col("__opaque_0__") + 1, stand_ins)
            >>> tree.inputs[0].identity.function
            'CumSum'
        """
        serialized = {
            name: json.loads(stand_in.meta.serialize(format="json"))
            for name, stand_in in stand_ins.items()
        }
        return ExpressionTree().read(
            self._replaced(json.loads(expression.meta.serialize(format="json")), serialized),
        )

    def _replaced(self, node: object, serialized: Mapping[str, object]) -> object:
        """Replace each reserved column in a serialized tree by its stand-in.

        Args:
            node: The serialized tree.
            serialized: The serialized stand-ins, by reserved column.

        Returns:
            The tree.

        Example:
            >>> KeyStandIn()._replaced([{"Column": "x"}], {"x": {"Column": "a"}})
            [{'Column': 'a'}]
        """
        if isinstance(node, list):
            return [self._replaced(child, serialized) for child in node]
        if not isinstance(node, dict):
            return node
        column = node.get("Column")
        if isinstance(column, str) and column in serialized:
            return self._replaced(serialized[column], serialized)
        return {key: self._replaced(value, serialized) for key, value in node.items()}

    def _facts(self, call: OpaqueCall) -> tuple[bool, bool] | None:
        """Give whether a call works value by value, and whether it gives one value.

        Args:
            call: The call.

        Returns:
            The two facts, or `None` where the code does not show them, as for
            a keyword whose value is not a literal.

        Example:
            >>> elementwise = OpaqueCall(pl.col("a"), "map_batches", {"is_elementwise": True})
            >>> KeyStandIn()._facts(elementwise)
            (True, False)
            >>> scalar = OpaqueCall(pl.col("a"), "map_elements", {"returns_scalar": True})
            >>> KeyStandIn()._facts(scalar) is None
            True
            >>> KeyStandIn()._facts(OpaqueCall(pl.struct("a"), "fold"))
            (False, False)
        """
        if call.method in self.FIXED:
            return None if "returns_scalar" in call.keywords else self.FIXED[call.method]
        flagged = call.method in self.FLAGGED
        elementwise = call.keywords.get("is_elementwise", False) if flagged else False
        scalar = call.keywords.get("returns_scalar", False)
        known = flagged or call.method in self.FOLDS
        if not known or not isinstance(elementwise, bool) or not isinstance(scalar, bool):
            return None
        return elementwise, scalar


class DeferredEvaluation:
    """Give the column that the code of a deferred expression gives at the data types of a frame.

    The rules answer for a `select` and for `agg`. In `agg`, they take a
    frame of two groups of other sizes, the normal case of a `group_by`. The
    answer is the run type, the type of the data that polars computes. A fault
    of the rules, such as a value of a type that they do not expect, gives
    `Unchecked` and does not end the run of mypy.

    A hole is a value that the checker cannot know, such as the count of a
    `shift` in a variable. The evaluation tries each hole of a `bool`, an
    `int`, a `float` or a `str` at a few values of its type, and keeps the
    answer where every choice gives the same one. Otherwise the value decides
    the answer, and the answer is `Unchecked`. A function of the user gives
    the type that `return_dtype=` names, or `Unchecked`.

    Example:
        >>> from astrolabe._hooks.deferred.code import Code
        >>> evaluation = DeferredEvaluation("row")
        >>> answer = evaluation.answer(Code("pl.col('a') + 1"), {"a": "Int8"})
        >>> answer.name, answer.rendering, answer.reads
        ('a', 'Int8', ('a',))
    """

    UNCHECKED: ClassVar[str] = "Unchecked"
    GROUPS: ClassVar[GroupBy] = GroupBy((3, 2))
    WINDOWS: ClassVar[dict[str, GroupBy]] = {
        "rolling": GroupBy((1, 2, 2, 2, 2), height=5, overlapping=True),
        "dynamic": GroupBy((3, 3, 1), height=5, overlapping=True),
        "disjoint": GroupBy((2, 2, 1), height=5),
    }
    LAZY: ClassVar[str] = "lazy "
    ALONE: ClassVar[str] = "alone"
    STORED: ClassVar[str] = " stored"
    LEAF: ClassVar[str] = "__leaf_{}__"
    OPAQUE: ClassVar[str] = "__opaque_{}__"
    OPAQUE_PREFIX: ClassVar[str] = "__opaque_"
    SAMPLES: ClassVar[dict[str, tuple[object, ...]]] = {
        "int": (1, -1, 2**15, 2**40, 2**63),
        "float": (0.5, 2.0**100),
        "number": (0.5, 2.0**100, 1, -1, 2**15, 2**40, 2**63),
        "str": ("a", "%Y"),
        "bool": (False, True),
        "datetime": (
            datetime.datetime(1970, 1, 1),
            datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC),
        ),
        "decimal": (decimal.Decimal("1.5"), decimal.Decimal("1.25")),
        "date": (datetime.date(1970, 1, 1), datetime.datetime(1970, 1, 1)),
    }
    LIMIT: ClassVar[int] = 64
    BUILD_SAMPLES: ClassVar[dict[str, tuple[object, ...]]] = {
        "str": ("{}", "{}{}", "{}{}{}", "{}{}{}{}"),
    }
    ANY_VALUE: ClassVar[str] = "value"
    UNTYPED_VALUE: ClassVar[str] = "any"
    PLAIN_EXPRESSION: ClassVar[str] = "expr"
    NO_INPUT: ClassVar[str] = "literal"
    UNKNOWN: ClassVar[dict[str, str]] = {
        "untyped": "a value of the type `Any`",
        "plain": "an expression that is not typed",
        "hole": "a value that the checker cannot know",
        "name": "a column name that the checker cannot know",
    }
    EVALUATIONS: ClassVar[int] = 512
    NO_EXPRESSION: ClassVar[str] = "the code gives no expression"
    FORMAT_HOLE: ClassVar[str] = (
        "a format that the checker cannot know decides whether the text parses"
    )
    SELECTION: ClassVar[str] = "the code reads a declared selection beside other parts"

    def __init__(self, context: str, planned: Mapping[str, str] | None = None) -> None:
        """Keep the context of an operation, the passes of its optimizer, and the stored types.

        An eager frame simplifies the expressions of a `select`, and those of
        `agg` as `EagerAgg` says. A lazy frame runs every pass when it collects, the
        slice pushdown included. A lazy frame also stores a type for each
        column, and polars plans each step at that type. Where it differs
        from the type of the data, the leaf of the column holds both. The
        windows of a `rolling` or a `group_by_dynamic` take the place of the
        two groups in `agg`: windows that overlap, or windows that do not
        (`WINDOWS`).

        Args:
            context: `row` for a context such as `select` on an eager frame,
                `agg` for the aggregation context of an eager `agg`, `lazy row`
                and `lazy agg` for the same contexts on a lazy frame, or
                `planner` for the type that polars plans in a `select`, which
                a fit compares with the type that it runs. `lazy row stored`
                and `lazy agg stored` give the type that a lazy frame stores
                for the output. `agg rolling`, `agg dynamic` and `agg disjoint`
                take the windows of their name in `agg`. The word `alone`
                after `lazy row` says that the code is the only expression
                of its `select`.
            planned: The rendering of the stored type of each column of a
                lazy frame where it differs from the type of its data, by its name.

        Example:
            >>> DeferredEvaluation("lazy agg")._groups.flags.slice_pushdown
            True
            >>> DeferredEvaluation("agg rolling")._groups.overlapping
            True
            >>> DeferredEvaluation("lazy row alone")._groups.flags.lone
            True
        """
        self._context = context
        self._planned = dict(planned or {})
        lazy = context.startswith(self.LAZY)
        words = context.split()
        flags = QueryFlags.lazy(lone=self.ALONE in words) if lazy else QueryFlags.eager()
        windows = [groups for name, groups in self.WINDOWS.items() if name in words]
        self._groups = replace(windows[0] if windows else self.GROUPS, flags=flags)

    def answer(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
    ) -> DeferredAnswer:
        """Give the column of a code at the columns of a frame.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            columns: The rendering of the data type of each column of the
                frame, by its name.

        Returns:
            The answer.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row").answer(Code("pl.col('t') + 1"), {"t": "String"}).refused
            True
            >>> shifted = Code("pl.col('a').shift(hole('int'))")
            >>> DeferredEvaluation("row").answer(shifted, {"a": "Int8"}).rendering
            'Int8'
            >>> added = Code("pl.col('a') + hole('int')")
            >>> DeferredEvaluation("row").answer(added, {"a": "Int8"}).cause
            'number'
            >>> DeferredEvaluation("row").answer(Code("pl.col("), {}).cause
            'code'
        """
        if self._reads_selection(code):
            return DeferredAnswer(None, self.UNCHECKED, reason=self.SELECTION, cause="code")
        return self._answered(code, columns, Substitution())

    def leaf_selected(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        node: object,
        rule: Selector,
        selectable: Collection[str],
    ) -> list[DeferredAnswer]:
        """Give one column for each column of the frame that the declared selection of a code picks.

        The declared selection picks a column where its rule picks the data
        type of the column, or one data type of its sample where the frame
        does not fully declare it. At each picked column, the leaf of the
        selection is the declared map there.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            columns: The rendering of the data type of each column of the frame.
            node: The leaf of the selection, as a column of the code.
            rule: The rule of the columns that the selection picks.
            selectable: The names of the columns that the selection can pick.

        Returns:
            One answer for each picked column, in the order of the frame.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = OfType(Family.numeric())
            >>> node = {"Column": "__leaf_0__"}
            >>> text = {"t": "String"}
            >>> DeferredEvaluation("row").leaf_selected(Code("$0"), text, node, numeric, ["t"])
            []
        """
        picked: list[str] = []
        for name in selectable:
            renderings = OpenType().renderings(columns[name])
            if any(rule.picks(name, rendering) for rendering in renderings):
                picked.append(name)
        evaluation = self._apart(len(picked))
        return [evaluation.at_column(code, columns, node, name) for name in picked]

    def _apart(self, outputs: int) -> DeferredEvaluation:
        """Give the evaluation of a code that polars expands into some number of outputs.

        A `len` that stands alone in a lazy `select` counts the rows, but
        polars counts the expressions after it expands a selector. So a code
        that gives another number of outputs than one does not stand alone.

        Args:
            outputs: The number of outputs of the code.

        Returns:
            This evaluation, or one that does not read the code as alone.

        Example:
            >>> DeferredEvaluation("lazy row alone")._apart(2)._groups.flags.lone
            False
            >>> DeferredEvaluation("lazy row alone")._apart(1)._groups.flags.lone
            True
        """
        if outputs == 1 or not self._groups.flags.lone:
            return self
        words = [word for word in self._context.split() if word != self.ALONE]
        return DeferredEvaluation(" ".join(words), self._planned)

    def _reads_selection(self, code: Code[DeclaredLeaf | None]) -> bool:
        """Say whether a code reads a declared selection as one of its leaves.

        Such a code gives one column for each column that the selection
        picks. The evaluation reads it only at one picked column.

        Args:
            code: The code.

        Returns:
            Whether a leaf is a declared selection.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row")._reads_selection(Code("pl.col('a')"))
            False
        """
        return any(leaf is not None and leaf.selection is not None for leaf in code.leaves)

    def _picked_code(
        self,
        code: Code[DeclaredLeaf | None],
        selector: object,
        column: str,
    ) -> Code[DeclaredLeaf | None] | None:
        """Give a code with one picked column in the place of the declared selection that it reads.

        The selector of such a code is the leaf of the selection. At the
        picked column, the leaf is the declared map that the selection
        promises there: its type, its name after the rename, and a read of
        that column.

        Args:
            code: The code.
            selector: The serialized selector of the code.
            column: The name of the picked column.

        Returns:
            The code, or `None` where the selector is not the leaf of a
            declared selection.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> numeric = {"Selector": {"ByDType": "Numeric"}}
            >>> DeferredEvaluation("row")._picked_code(Code("cs.numeric()"), numeric, "a") is None
            True
        """
        name = selector.get("Column") if isinstance(selector, dict) else None
        places = {self.LEAF.format(place): place for place in range(len(code.leaves))}
        place = places.get(name) if isinstance(name, str) else None
        leaf = None if place is None else code.leaves[place]
        if place is None or leaf is None or leaf.selection is None:
            return None
        leaves = list(code.leaves)
        leaves[place] = leaf.selection.picked(column)
        return replace(code, leaves=tuple(leaves))

    def bare_column(self, code: Code[DeclaredLeaf | None]) -> str | None:
        """Give the name of the column of a code that reads one column and gives it back as it is.

        Code that reads a key of a join needs this form: a bare `pl.col(name)`
        acts there as its name.

        Args:
            code: The code.

        Returns:
            The name, or `None` where the code is not `pl.col(name)` alone.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row").bare_column(Code("pl.col('id')"))
            'id'
            >>> DeferredEvaluation("row").bare_column(Code("pl.col('id').alias('k')")) is None
            True
            >>> DeferredEvaluation("row").bare_column(Code("pl.col(")) is None
            True
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:
            return None
        expression = read.expression
        if not isinstance(expression, pl.Expr) or not expression.meta.is_column():
            return None
        return self._name(expression, code.leaves, read.opaque)

    def struct_column(self, code: Code[DeclaredLeaf | None]) -> str | None:
        """Give the column of the struct that a read of several fields reads, where it is a column.

        Args:
            code: The code.

        Returns:
            The name of the column, or `None` where the code reads no several
            fields, or reads them from another expression.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row").struct_column(Code("pl.col('s').struct.unnest()"))
            's'
            >>> DeferredEvaluation("row").struct_column(Code("pl.col(")) is None
            True
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:  # pragma: no cover -- the caller read this code first
            return None
        node = (
            None if not isinstance(read.expression, pl.Expr) else FieldNodes().of(read.expression)
        )
        struct = None if node is None else FieldNodes().struct_of(node)
        if struct is None or not struct.meta.is_column():
            return None
        return self._name(struct, code.leaves, read.opaque)

    def aliased(self, code: Code[DeclaredLeaf | None]) -> bool:
        """Say whether the outermost call of a code is `.alias()`.

        Args:
            code: The code.

        Returns:
            Whether it is.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row").aliased(Code("cs.numeric().alias('z')"))
            True
            >>> DeferredEvaluation("row").aliased(Code("pl.col("))
            False
            >>> DeferredEvaluation("row").aliased(Code("1"))
            False
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:  # pragma: no cover -- the caller read this code first
            return False
        expression = read.expression
        if not isinstance(expression, pl.Expr):  # pragma: no cover -- the caller read an expression
            return False
        return "Alias" in json.loads(expression.meta.serialize(format="json"))

    def is_literal(self, code: Code[DeclaredLeaf | None]) -> bool:
        """Say whether a code is a literal, a value that reads no column, with or without an alias.

        Args:
            code: The code.

        Returns:
            Whether polars reads the expression as a literal.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row").is_literal(Code("pl.lit(1).alias('a')"))
            True
            >>> DeferredEvaluation("row").is_literal(Code("pl.col("))
            False
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:  # pragma: no cover -- the caller read this code first
            return False
        expression = read.expression
        return isinstance(expression, pl.Expr) and expression.meta.is_literal(allow_aliasing=True)

    def is_dynamic(self, code: Code[DeclaredLeaf | None]) -> bool:
        """Say whether a code is a literal whose value decides its data type, as `pl.lit(1)`.

        Beside a column, polars joins such a literal by the range of its
        value, and not by a data type of its own.

        Args:
            code: The code.

        Returns:
            Whether the planned type of the code is `Unknown`.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row").is_dynamic(Code("pl.lit(1)"))
            True
            >>> DeferredEvaluation("row").is_dynamic(Code("pl.lit(1, dtype=pl.Int32)"))
            False
            >>> DeferredEvaluation("row").is_dynamic(Code("pl.col("))
            False
        """
        try:
            read = CodeReader().read(code)
            expression = checked_cast(read.expression, pl.Expr)
            operand = TreeEvaluation({}, self._groups).planned(ExpressionTree().of(expression))
        except (CodeReadError, TypeError, PolarsRefusal, UndecidedError):
            return False
        return Is.unknown(operand.dtype)

    def selector(self, code: Code[DeclaredLeaf | None]) -> object | None:
        """Give the one selector of a code, serialized, as the tree of polars writes it.

        Args:
            code: The code.

        Returns:
            The selector. `None` where the code holds no selector, or more
            than one.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row").selector(Code("cs.numeric().abs()"))
            {'Selector': {'ByDType': 'Numeric'}}
            >>> DeferredEvaluation("row").selector(Code("pl.col('a')")) is None
            True
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:
            return None
        if not isinstance(read.expression, pl.Expr):
            return None
        selectors = Substitution().selectors(read.expression, read.opaque)
        return selectors[0] if len(selectors) == 1 else None

    def multiple(self, code: Code[DeclaredLeaf | None]) -> bool:
        """Say whether polars expands a code into one output for each column that it picks.

        A code with a selector at its root, such as `pl.all().alias('o')`,
        gives a number of outputs that only the columns of the frame decide.

        Args:
            code: The code.

        Returns:
            Whether polars marks the expression of the code as one of several
            outputs. False for a code that the reader cannot build.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row").multiple(Code("pl.all().alias('o')"))
            True
            >>> DeferredEvaluation("row").multiple(Code("pl.col('a') + 1"))
            False
            >>> DeferredEvaluation("row").multiple(Code("pl.col("))
            False
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:
            return False
        expression = read.expression
        return isinstance(expression, pl.Expr) and expression.meta.has_multiple_outputs()

    def name_at(
        self,
        code: Code[DeclaredLeaf | None],
        selector: object,
        column: str,
    ) -> str | None:
        """Give the output name of a code with one column in the place of its selector.

        Args:
            code: The code.
            selector: The serialized selector of the code.
            column: The name of the column.

        Returns:
            The name, or `None` where polars gives none.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> code = Code("cs.numeric().map_batches(hole('function')).name.suffix('_x')")
            >>> numeric = {"Selector": {"ByDType": "Numeric"}}
            >>> DeferredEvaluation("row").name_at(code, numeric, "a")
            'a_x'
            >>> DeferredEvaluation("row").name_at(Code("1"), numeric, "a") is None
            True
            >>> DeferredEvaluation("row").name_at(Code("pl.col("), numeric, "a") is None
            True
        """
        picked = self._picked_code(code, selector, column)
        try:
            read = CodeReader().read(code)
        except CodeReadError:  # pragma: no cover -- the caller read this code first
            return None
        if not isinstance(read.expression, pl.Expr):  # pragma: no cover -- it reads one
            return None
        substitution = Substitution(selector, column) if picked is None else Substitution()
        leaves = code.leaves if picked is None else picked.leaves
        calls = substitution.calls(read.opaque)
        return self._name(substitution(read.expression), leaves, calls)

    def selected(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        selector: object,
        selectable: Collection[str] | None = None,
    ) -> list[DeferredAnswer] | None:
        """Give one column for each column of the frame that the selector of a code picks.

        Polars expands a selector over the columns of the frame, and runs the
        expression once for each column that it picks. The evaluation does the
        same: it expands the selector over the data types of the frame, and
        evaluates the code with each picked column in place of the selector.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            columns: The rendering of the data type of each column of the
                frame, by its name.
            selector: The serialized selector of the code.
            selectable: The names of the columns that the selector can pick,
                or `None` for every column. `agg` keeps its group keys out of
                every selector, and the expression can still read them.

        Returns:
            One answer for each picked column. `None` where a column of the
            frame has a data type that the rules cannot read, so the columns
            that the selector picks are not known, or where polars refuses to
            expand the selector, as for a regular expression that does not
            parse.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> code = Code("cs.numeric().abs()")
            >>> evaluation = DeferredEvaluation("row")
            >>> numeric = evaluation.selector(code)
            >>> answers = evaluation.selected(code, {"a": "Int8", "t": "String"}, numeric)
            >>> [(answer.name, answer.rendering) for answer in answers]
            [('a', 'Int8')]
            >>> pattern = {"Selector": {"Matches": "^a["}}
            >>> evaluation.selected(Code("cs.matches('^a[')"), {"a": "Int8"}, pattern) is None
            True

            A code that also reads a declared selection gives no answer:

            >>> from astrolabe._hooks.deferred.argument import DeclaredSelection
            >>> from astrolabe._hooks.expression.selection import SelectionType
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> declared = DeclaredSelection(SelectionType.identity(OfType(Family.numeric())))
            >>> leaf = DeclaredLeaf("", "Unchecked", (), selection=declared)
            >>> both = Code("cs.numeric() + $0", (leaf,))
            >>> evaluation.selected(both, {"a": "Int8"}, numeric) is None
            True
        """
        if self._reads_selection(code):
            return None
        names = columns.keys() if selectable is None else selectable
        candidates = {name: rendering for name, rendering in columns.items() if name in names}
        picked = self.picked_columns(candidates, selector)
        if picked is None:
            return None
        apart = self._apart(len(picked))
        if apart is not self:
            return apart.selected(code, columns, selector, selectable)
        return [
            self._picked_answer(code, columns, (selector, name), samples)
            for name, samples in picked.items()
        ]

    def _picked_answer(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        pick: tuple[object, str],
        samples: tuple[DType, ...],
    ) -> DeferredAnswer:
        """Give the answer of a code at one column that its selector picks.

        The selector picks a column whose data type has holes at some data
        types of its sample. Where it does not pick it at each of them, the
        data decide whether it picks the column, and the answer names it in
        `partial`.

        Args:
            code: The code.
            columns: The rendering of the data type of each column of the frame.
            pick: The serialized selector, and the name of the picked column.
            samples: The data types at which the selector picks the column,
                none for a column of a full data type.

        Returns:
            The answer.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> pick = ({"Selector": {"ByDType": "Numeric"}}, "u")
            >>> evaluation = DeferredEvaluation("row")
            >>> code = Code("cs.numeric()")
            >>> answer = evaluation._picked_answer(code, {"u": "Unchecked"}, pick, (DType("Int8"),))
            >>> answer.partial
            'u'
        """
        selector, name = pick
        substitution = Substitution(selector, name)
        answer = self._answered(code, columns, substitution, {name: samples} if samples else {})
        partial = bool(samples) and len(samples) < len(OpenType().samples(columns[name]))
        return replace(answer, partial=name) if partial else answer

    def picked_columns(
        self,
        columns: Mapping[str, str],
        selector: object,
    ) -> dict[str, tuple[DType, ...]] | None:
        """Give each column of the frame that a selector picks, in the order of the frame.

        A column whose data type has holes, as `Unchecked`, can be picked or
        not, by its data type. The selector picks it where it picks a data
        type of its sample, and the evaluation then takes only those data
        types for it. A selector of positions needs every data type, so it
        gets no answer over such a column. A name that the selector requires
        and the frame does not hold picks nothing: over an open frame, it can
        be a column of the remainder.

        Args:
            columns: The rendering of the data type of each column, by its name.
            selector: The serialized selector.

        Returns:
            The data types that each picked column takes, by its name: none
            for a column of a full data type. `None` where polars refuses to
            expand the selector, where the selector reads positions over a
            column with holes, or where a name or a pattern of the selector
            comes from a variable, so that the data decides it.

        Example:
            >>> numeric = {"Selector": {"ByDType": "Numeric"}}
            >>> columns = {"a": "Unchecked", "t": "String"}
            >>> picked = DeferredEvaluation("row").picked_columns(columns, numeric)
            >>> list(picked), len(picked["a"])
            (['a'], 15)
            >>> first = {"Selector": {"ByIndex": {"indices": [0], "strict": True}}}
            >>> DeferredEvaluation("row").picked_columns({"a": "Unchecked"}, first) is None
            True
            >>> named = {"Selector": {"Matches": HoleValues.STR}}
            >>> DeferredEvaluation("row").picked_columns({"a": "Int8"}, named) is None
            True
        """
        if HoleValues().held(selector):
            return None
        chosen = Substitution().selector(NameRequirement().relaxed(selector))
        closed = [LeafChoices().closed(rendering) for rendering in columns.values()]
        if None in closed and "ByIndex" in json.dumps(selector):
            return None
        try:
            names = set(cs.expand_selector(self._placed_schema(columns), chosen))
        except pl.exceptions.PolarsError:
            return None
        picked: dict[str, tuple[DType, ...]] = {}
        for name, rendering in columns.items():
            samples = self._column_pick(chosen, name, rendering, picked=name in names)
            if samples is not None:
                picked[name] = samples
        return picked

    def _column_pick(
        self,
        chosen: cs.Selector,
        name: str,
        rendering: str,
        *,
        picked: bool,
    ) -> tuple[DType, ...] | None:
        """Give the data types at which a selector picks one column.

        A column of a full data type is picked or not by the expansion over
        the whole frame. A column whose data type has holes is picked at each
        data type of its sample that the selector picks.

        Args:
            chosen: The selector.
            name: The name of the column.
            rendering: The rendering of the data type of the column.
            picked: Whether the expansion over the whole frame picks the column.

        Returns:
            No data type for a picked column of a full data type, the picked
            data types for a column with holes, or `None` where the selector
            does not pick the column.

        Example:
            >>> evaluation = DeferredEvaluation("row")
            >>> evaluation._column_pick(cs.numeric(), "a", "Int8", picked=True)
            ()
            >>> evaluation._column_pick(cs.temporal(), "d", "Duration", picked=False)
            (Duration(time_unit='ns'), Duration(time_unit='us'), Duration(time_unit='ms'))
            >>> evaluation._column_pick(cs.string(), "d", "Duration", picked=False) is None
            True
        """
        if LeafChoices().closed(rendering) is not None:
            return () if picked else None
        samples = OpenType().samples(rendering)
        chosen_samples = tuple(sample for sample in samples if self._picks(chosen, name, sample))
        return chosen_samples or None

    def _placed_schema(self, columns: Mapping[str, str]) -> dict[str, pl.DataType]:
        """Give the schema of a frame for polars, with `Null` for each data type with holes.

        A test of names needs every column. The test of the data type of a
        column with holes is a step of its own.

        Args:
            columns: The rendering of the data type of each column, by its name.

        Returns:
            The schema.

        Example:
            >>> DeferredEvaluation("row")._placed_schema({"a": "Int8", "u": "Unchecked"})
            {'a': Int8, 'u': Null}
        """
        schema: dict[str, pl.DataType] = {}
        for name, rendering in columns.items():
            closed = LeafChoices().closed(self._planned.get(name, rendering))
            schema[name] = pl.Null() if closed is None else closed.polars()
        return schema

    def _picks(self, chosen: cs.Selector, name: str, dtype: DType) -> bool:
        """Say whether a selector picks a column of one name and one data type.

        Args:
            chosen: The selector.
            name: The name of the column.
            dtype: The data type.

        Returns:
            Whether it does.

        Example:
            >>> DeferredEvaluation("row")._picks(cs.numeric(), "a", DType("Int8"))
            True
            >>> DeferredEvaluation("row")._picks(cs.by_name("b"), "a", DType("Int8"))
            False
        """
        try:
            return bool(cs.expand_selector({name: dtype.polars()}, chosen))
        except pl.exceptions.PolarsError:  # pragma: no cover -- the relaxed selector picks by names
            return False

    def fields(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
    ) -> list[DeferredAnswer] | None:
        """Give one column for each field that a read of several fields of a struct picks.

        `.struct.unnest()` and `.struct.field("a", "b")` hold a selector of the
        fields. Polars expands it into one read of each field that it picks,
        in the order of the struct, as it expands a selector of columns. The
        evaluation reads the type of the struct, picks its fields, and
        evaluates the code with a read of each field in the place of the node.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            columns: The rendering of the data type of each column of the frame.

        Returns:
            One answer for each picked field. `None` where the code reads no
            several fields. No answer where the checker does not know the
            fields of the struct, or where polars refuses the selector.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> struct = "Struct({'a': Int64, 'b': String})"
            >>> unnest = Code("pl.col('s').struct.unnest()")
            >>> answers = DeferredEvaluation("row").fields(unnest, {"s": struct})
            >>> [(answer.name, answer.rendering) for answer in answers]
            [('a', 'Int64'), ('b', 'String')]
            >>> DeferredEvaluation("row").fields(Code("pl.col('s')"), {"s": struct}) is None
            True
        """
        node = self._field_node(code)
        if node is None:
            return None
        names = self._field_names(node, code, columns) or []
        answers: list[DeferredAnswer] = []
        for name in names:
            replacement = FieldNodes().read_of(node, name)
            answers.append(
                self._answered(code, columns, Substitution(node, replacement=replacement))
            )
        return answers

    def picks_no_field(self, code: Code[DeclaredLeaf | None], columns: Mapping[str, str]) -> bool:
        """Say whether a read of several fields of a struct of known fields picks none of them.

        A pattern such as `struct.field("^z.*$")` that matches no field gives
        no column, and polars runs it.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            columns: The rendering of the data type of each column of the frame.

        Returns:
            Whether the code picks no field. `False` where the code reads no
            several fields, where the fields of the struct are not known, or
            where polars refuses the selector.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> struct = {"s": "Struct({'a': Int64})"}
            >>> none = Code("pl.col('s').struct.field('^z.*$')")
            >>> DeferredEvaluation("row").picks_no_field(none, struct)
            True
            >>> DeferredEvaluation("row").picks_no_field(none, {"s": "Unchecked"})
            False
            >>> DeferredEvaluation("row").picks_no_field(Code("pl.col('s')"), struct)
            False
        """
        node = self._field_node(code)
        return node is not None and self._field_names(node, code, columns) == []

    def _field_node(self, code: Code[DeclaredLeaf | None]) -> Mapping[str, Any] | None:
        """Give the node of a read of several fields of a struct, where the code holds one.

        Args:
            code: The code.

        Returns:
            The serialized node, or `None`.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row")._field_node(Code("pl.col(")) is None
            True
            >>> DeferredEvaluation("row")._field_node(Code("1")) is None
            True
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:
            return None
        if not isinstance(read.expression, pl.Expr):
            return None
        return FieldNodes().of(read.expression)

    def _field_names(
        self,
        node: Mapping[str, Any],
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
    ) -> list[str] | None:
        """Give the fields of a struct that a read of several fields picks.

        Args:
            node: The serialized node of the read.
            code: The code.
            columns: The rendering of the data type of each column of the frame.

        Returns:
            The names, in the order of the struct. None where the type of the
            struct is not known, or where polars refuses the selector.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> node = FieldNodes().of(pl.col("s").struct.field("b", "a"))
            >>> struct = "Struct({'a': Int64, 'b': String})"
            >>> DeferredEvaluation("row")._field_names(node, Code("pl.col('s')"), {"s": struct})
            ['a', 'b']
        """
        schema = self._struct_schema(node, code, columns)
        if schema is None:
            return None
        try:
            return list(cs.expand_selector(schema, FieldNodes().selector_of(node)))
        except pl.exceptions.PolarsError:
            return None

    def missing_fields(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
    ) -> list[str]:
        """Give each field that a read of several fields requires and the struct does not hold.

        Polars raises `StructFieldNotFoundError` for such a field, as
        `.struct.field("a", "q")` over a struct that holds no field `q`.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            columns: The rendering of the data type of each column of the frame.

        Returns:
            The names, in the order of the code. No name where the code reads
            no several fields, or where the type of the struct is not known.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> read = Code("pl.col('s').struct.field('a', 'q')")
            >>> DeferredEvaluation("row").missing_fields(read, {"s": "Struct({'a': Int64})"})
            ['q']
            >>> DeferredEvaluation("row").missing_fields(Code("pl.col("), {})
            []
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:  # pragma: no cover -- the caller read this code first
            return []
        node = (
            None if not isinstance(read.expression, pl.Expr) else FieldNodes().of(read.expression)
        )
        schema = None if node is None else self._struct_schema(node, code, columns)
        if node is None or schema is None:  # pragma: no cover -- the caller checked the struct
            return []
        return [name for name in FieldNodes().required_of(node) if name not in schema]

    def _struct_schema(
        self,
        node: Mapping[str, Any],
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
    ) -> dict[str, pl.DataType] | None:
        """Give the fields of the struct that a read of several fields reads, as a schema of polars.

        Args:
            node: The serialized node of the read.
            code: The code.
            columns: The rendering of the data type of each column of the frame.

        Returns:
            The schema, or `None` where the type of the struct is not known.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> node = FieldNodes().of(pl.col("s").struct.field("a", "b"))
            >>> columns = {"s": "Struct({'a': Int64})"}
            >>> DeferredEvaluation("row")._struct_schema(node, Code("pl.col('s')"), columns)
            {'a': Int64}
            >>> DeferredEvaluation("row")._struct_schema(node, Code("pl.col('s')"), {}) is None
            True
        """
        struct = FieldNodes().struct_of(node)
        planned = LeafChoices(self._leaf_planned(code))
        (leaves,) = planned.of({**columns, **self._declared(code.leaves, columns)}, ())
        try:
            tree = ExpressionTree().of(struct)
            dtype = TreeEvaluation(leaves, self._groups).answers(tree).run_select
        except Exception:  # pragma: no cover
            # A fault of the rules must not end the run of mypy.
            return None
        if not isinstance(dtype, DType) or dtype.name != "Struct":
            return None
        schema: dict[str, pl.DataType] = {}
        for pair in checked_cast(dtype.args[0], tuple):
            name, inner = pair
            schema[str(name)] = checked_cast(inner, DType).polars()
        return schema

    def at_column(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        selector: object,
        column: str,
    ) -> DeferredAnswer:
        """Give the column of a code with one column in the place of its selector.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            columns: The rendering of the data type of each column of the frame.
            selector: The serialized selector of the code.
            column: The name of the column that takes the place of the selector.

        Returns:
            The answer.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> numeric = {"Selector": {"ByDType": "Numeric"}}
            >>> code = Code("cs.numeric() + 300")
            >>> DeferredEvaluation("row").at_column(code, {"a": "Int8"}, numeric, "a").rendering
            'Int16'
        """
        own = columns.get(column)
        picked = self._picked_code(code, selector, column)
        if picked is not None:
            return self._answered(picked, columns, Substitution(), own=own)
        return self._answered(code, columns, Substitution(selector, column), own=own)

    def reads(self, code: Code[DeclaredLeaf | None]) -> tuple[str, ...]:
        """Give the columns of the frame that a code reads, without an evaluation.

        Args:
            code: The code.

        Returns:
            The names, in the order of polars. No name where the code gives no expression.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row").reads(Code("pl.col('b') + pl.col('a')"))
            ('b', 'a')
            >>> DeferredEvaluation("row").reads(Code("pl.col("))
            ()
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:
            return ()
        if not isinstance(read.expression, pl.Expr):
            return ()
        return self._reads(read.expression, read.opaque, code.leaves)

    def _answered(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        substitution: Substitution,
        restricted: Mapping[str, tuple[DType, ...]] | None = None,
        *,
        own: str | None = None,
    ) -> DeferredAnswer:
        """Give the column of a code at the columns of a frame, with a column for its selector.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            columns: The rendering of the data type of each column of the frame.
            substitution: The column that stands for the selector of the code, if any.
            restricted: The data types that a column with holes takes in
                place of its whole sample, by its name.
            own: The rendering of the column that stands for the selector,
                which the answer can give back as it is.

        Returns:
            The answer.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row")._answered(Code("pl.col("), {}, Substitution()).cause
            'code'
            >>> wide = Code("2**127 + pl.col('a')")
            >>> DeferredEvaluation("row")._answered(wide, {"a": "Int8"}, Substitution()).build
            ('InvalidOperationError', 'integer too large for Polars: OverflowError: ...')
            >>> objects = Code("pl.col('a').replace([1], [pl.lit(2)])")
            >>> DeferredEvaluation("row")._answered(objects, {"a": "Int8"}, Substitution()).refused
            True
            >>> numbers = Code("pl.col('a') + hole('int') + hole('int') + hole('int')")
            >>> DeferredEvaluation("row")._answered(numbers, {"a": "Int8"}, Substitution()).cause
            'number'
            >>> many = Code("pl.col('a') + hole('int') + hole('int')")
            >>> DeferredEvaluation("row")._answered(many, {"a": "Unchecked"}, Substitution()).cause
            'open'
        """
        try:
            read = CodeReader().read(code)
        except BuildRefusal as refusal:
            rebuilt = self._rebuilt(code, refusal)
            if rebuilt is None:
                return self._unread(code, refusal)
            read = rebuilt
        except CodeReadError as error:
            return self._unread(code, error)
        if not isinstance(read.expression, pl.Expr):
            return DeferredAnswer(None, self.UNCHECKED, reason=self.NO_EXPRESSION, cause="code")
        expression = substitution(read.expression)
        calls = substitution.calls(read.opaque)
        name = self._name(expression, code.leaves, calls)
        reads = self._reads(expression, calls, code.leaves)
        base = DeferredAnswer(name, self.UNCHECKED, reads=reads)
        cause = self._unknown(read.holes, self._roots(expression, calls))
        if cause is not None:
            return replace(base, reason=self.UNKNOWN[cause], cause=cause)
        leaves = self._leaf_choices(code, expression, calls, columns, restricted or {})
        values = self._choices(read.holes, read.choice)
        types = self._typed(read.holes, OpenType(self._fields(expression), self._texts(code)))
        crowded = self._crowded(base, read.holes, len(values), len(types) * len(leaves))
        if crowded is not None:
            return crowded
        kinds = self._typed_kinds(code, leaves, values, types, substitution)
        signatures = [self._signature(typed, leaf) for typed in types for leaf in leaves]
        answer = self._formatted(
            self._combined(base, read.holes, kinds, signatures), base, expression
        )
        return self._decided_by_user(self._unsampled(answer, own), calls)

    def _formatted(
        self,
        answer: DeferredAnswer,
        base: DeferredAnswer,
        expression: pl.Expr,
    ) -> DeferredAnswer:
        """Give `Unchecked` in place of a refusal where a parse takes its format from a hole.

        A hole of a text takes two sample texts, and neither needs to be a
        format that fits a literal text, as `"2020-01-01"`. A format that the
        sample does not hold can fit it. So a refusal at every choice does not
        show that polars refuses every format.

        Args:
            answer: The answer at the choices of the holes.
            base: The answer with the name and the columns that the code reads.
            expression: The expression, with the placeholder of each hole.

        Returns:
            The answer, or `Unchecked` with the cause `hole` in place of a refusal.

        Example:
            >>> evaluation = DeferredEvaluation("row")
            >>> base = DeferredAnswer("literal", "Unchecked")
            >>> refused = replace(base, refused=True)
            >>> parse = pl.lit("2020-01-01").str.to_date(HoleValues.STR)
            >>> evaluation._formatted(refused, base, parse).cause
            'hole'
            >>> evaluation._formatted(refused, base, pl.lit("2020-01-01").str.to_date("%Y")).refused
            True
        """
        if not answer.refused or not self._format_hole(expression):
            return answer
        return replace(base, reason=self.FORMAT_HOLE, cause="hole")

    def _format_hole(self, expression: pl.Expr) -> bool:
        """Say whether a node of an expression takes its format from a hole of a text.

        Args:
            expression: The expression, with the placeholder of each hole.

        Returns:
            Whether a `format` option holds the placeholder of a text.

        Example:
            >>> evaluation = DeferredEvaluation("row")
            >>> evaluation._format_hole(pl.col("s").str.to_time(HoleValues.STR))
            True
            >>> evaluation._format_hole(pl.col("s").str.contains(HoleValues.STR))
            False
        """
        pending: list[object] = [json.loads(expression.meta.serialize(format="json"))]
        while pending:
            node = pending.pop()
            if isinstance(node, list):
                pending.extend(node)
            if not isinstance(node, dict):
                continue
            if node.get("format") == HoleValues.STR:
                return True
            pending.extend(node.values())
        return False

    def _unsampled(self, answer: DeferredAnswer, own: str | None) -> DeferredAnswer:
        """Give `Unchecked` in place of each struct of an answer that holds the field of a sample.

        A sample of an open `Struct` holds one field of a reserved name, which
        stands for fields that the code does not name. A column that the caller
        gave as such a sample, and that comes back as it is, keeps that field,
        because the caller reads the answer against its own sample.

        Args:
            answer: The answer.
            own: The rendering of the column that stands for the selector, or `None`.

        Returns:
            The answer, with the cause `open` where it becomes `Unchecked`.

        Example:
            >>> sampled = "Struct({'__field__': Int8})"
            >>> evaluation = DeferredEvaluation("row")
            >>> evaluation._unsampled(DeferredAnswer("a", sampled), None).rendering
            'Unchecked'
            >>> evaluation._unsampled(DeferredAnswer("a", sampled), sampled).rendering
            "Struct({'__field__': Int8})"
            >>> evaluation._unsampled(DeferredAnswer("a", f"List({sampled})"), None).cause
            ''

            A struct that holds a part of no known type is `Unchecked` too:

            >>> opened = "Struct({'__field__': Unchecked})"
            >>> evaluation._unsampled(DeferredAnswer("a", opened), None).rendering
            'Unchecked'
        """
        if OpenType.FIELD not in answer.rendering or answer.rendering == own:
            return answer
        try:
            named = Generalization().named(Rendering().parse(answer.rendering)).render()
        except RenderingError:
            named = self.UNCHECKED
        cause = "open" if named == self.UNCHECKED else answer.cause
        return replace(answer, rendering=named, cause=cause)

    def _crowded(
        self,
        base: DeferredAnswer,
        holes: Sequence[str],
        values: int,
        others: int,
    ) -> DeferredAnswer | None:
        """Give the answer of a code whose choices are more than the evaluation tries.

        Each choice of the values of the holes meets each choice of the data
        types of the holes and of the columns, and the evaluation runs polars
        once for each pair. Where no choice of the values is left, the values
        decide the answer.

        Args:
            base: The answer with the name and the columns that the code reads.
            holes: The kind of each hole of the code.
            values: The number of choices of the values of the holes, where
                none means too many.
            others: The number of choices of the data types.

        Returns:
            The answer `Unchecked`, or `None` where the evaluation tries every pair.

        Example:
            >>> base = DeferredAnswer("a", "Unchecked")
            >>> DeferredEvaluation("row")._crowded(base, ["int"] * 3, 0, 1).cause
            'number'
            >>> DeferredEvaluation("row")._crowded(base, [], 1, 1000).cause
            'open'
            >>> DeferredEvaluation("row")._crowded(base, [], 1, 1) is None
            True
        """
        if not values:
            reason = "the values of the holes give too many choices"
            return replace(base, reason=reason, cause=self._hole_cause(holes))
        if values * others > self.EVALUATIONS:
            reason = "the holes and the data types of the columns give too many choices"
            return replace(base, reason=reason, cause="open")
        return None

    def _hole_cause(self, holes: Sequence[str]) -> str:
        """Give the cause of an answer that the values of the holes leave `Unchecked`.

        Args:
            holes: The kind of each hole of the code.

        Returns:
            `number` where a hole holds a number, and `hole` otherwise.

        Example:
            >>> DeferredEvaluation("row")._hole_cause(["str", "int"])
            'number'
            >>> DeferredEvaluation("row")._hole_cause(["str"])
            'hole'
        """
        return "number" if {"int", "float", "number"} & set(holes) else "hole"

    def _unread(self, code: Code[DeclaredLeaf | None], error: CodeReadError) -> DeferredAnswer:
        """Give the answer of a code that the reader cannot build into an expression.

        Args:
            code: The code.
            error: Why the reader cannot build it.

        Returns:
            A refusal where polars refuses the code while it builds it, or at
            every frame. Otherwise `Unchecked`, with the cause `code`.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> error = CodeReadError("a form that the recorder does not write")
            >>> DeferredEvaluation("row")._unread(Code("pl.col('a').alias('b')"), error).name
            'b'
        """
        if isinstance(error, BuildRefusal):
            return self._refused_build(code, error)
        if isinstance(error, RunRefusal):
            reason = f"{error}, and polars raises {error.error}"
            return DeferredAnswer(
                error.name,
                self.UNCHECKED,
                refused=True,
                reason=reason,
                refusals=(reason,),
            )
        if isinstance(error, UnwrittenTree):
            return self._unwritten(error)
        name = self._alias_of(code)
        return DeferredAnswer(name, self.UNCHECKED, reason=str(error), cause="code")

    def _unwritten(self, error: UnwrittenTree) -> DeferredAnswer:
        """Give the answer of an expression whose tree polars cannot write: a named `Unchecked`.

        Polars names the output of such an expression and the columns that it
        reads without its tree, so the frame keeps the name and checks the
        reads.

        Args:
            error: The expression.

        Returns:
            The answer, with the cause `code`.

        Example:
            >>> unwritten = UnwrittenTree("no tree", pl.cum_sum_horizontal("b", "a"))
            >>> answer = DeferredEvaluation("row")._unwritten(unwritten)
            >>> answer.name, answer.reads, answer.cause
            ('cum_sum', ('b', 'a'), 'code')
        """
        expression = error.expression
        name = expression.meta.output_name(raise_if_undetermined=False)
        reads = tuple(dict.fromkeys(expression.meta.root_names()))
        return DeferredAnswer(name, self.UNCHECKED, reason=str(error), reads=reads, cause="code")

    def _decided_by_user(
        self, answer: DeferredAnswer, calls: Sequence[OpaqueCall]
    ) -> DeferredAnswer:
        """Give the cause `opaque` to an answer that a function of the user leaves `Unchecked`.

        Args:
            answer: The answer.
            calls: The calls of a function of the user.

        Returns:
            The answer, with the cause `opaque` and the method of the first
            such call where it is `Unchecked` and a call names no type.

        Example:
            >>> call = OpaqueCall(pl.col("a"), "map_batches")
            >>> unknown = DeferredAnswer("a", "Unchecked")
            >>> DeferredEvaluation("row")._decided_by_user(unknown, [call]).cause
            'opaque'
        """
        unnamed = [call.method for call in calls if self._return_dtype(call) is None]
        if answer.rendering != self.UNCHECKED or answer.refused or not unnamed:
            return answer
        reason = "a function of the user decides the type"
        return replace(answer, reason=reason, cause="opaque", source=unnamed[0])

    def _typed_kinds(
        self,
        code: Code[DeclaredLeaf | None],
        leaves: Sequence[Mapping[str, ColumnLeaf]],
        values: Sequence[Mapping[int, object]],
        types: Sequence[Mapping[int, object]],
        substitution: Substitution,
    ) -> list[set[Sample]]:
        """Give the answers at each choice of the data types of the holes and of the leaves.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            leaves: Each choice of the leaves of the columns.
            values: Each choice of the values of the holes.
            types: Each choice of the data types of the holes of a data type.
            substitution: The column that stands for the selector of the code, if any.

        Returns:
            One set of answers for each choice of the data types of the holes
            and of the leaves, in that order, with one answer for each
            choice of the values of the holes.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> leaves = [{"a": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}]
            >>> code = Code("pl.col('a').abs()")
            >>> evaluation = DeferredEvaluation("row")
            >>> kinds = evaluation._typed_kinds(code, leaves, [{}], [{}], Substitution())
            >>> [sample.rendering for samples in kinds for sample in samples]
            ['Int8']
        """
        kinds: list[set[Sample]] = []
        for typed in types:
            chosen = [{**value, **typed} for value in values]
            kinds.extend(self._kinds(code, leaves, chosen, substitution))
        return kinds

    def _unknown(self, holes: Sequence[str], roots: Sequence[str]) -> str | None:
        """Give why no choice of the holes can give the answer of a code, where none can.

        A hole of any value can hold a value that no sample stands for. A
        value of the type `Any` is such a hole, which the user can give a
        type, and so is an expression that is not typed. A column whose name
        is a hole of a text is a column that the checker cannot know.

        Args:
            holes: The kind of each hole of the code.
            roots: The columns that the code reads, by the names that the reader gives them.

        Returns:
            The cause: `untyped`, `plain`, `hole` or `name`, or `None` where
            the choices can give the answer.

        Example:
            >>> DeferredEvaluation("row")._unknown(["str"], [HoleValues.STR])
            'name'
            >>> DeferredEvaluation("row")._unknown(["int"], ["a"]) is None
            True
            >>> DeferredEvaluation("row")._unknown(["any", "value"], ["a"])
            'untyped'
            >>> DeferredEvaluation("row")._unknown(["expr"], ["a"])
            'plain'
        """
        if self.UNTYPED_VALUE in holes:
            return "untyped"
        if self.PLAIN_EXPRESSION in holes:
            return "plain"
        if self.ANY_VALUE in holes:
            return "hole"
        if any(HoleValues.STR in root for root in roots):
            return "name"
        return None

    def _leaf_choices(
        self,
        code: Code[DeclaredLeaf | None],
        expression: pl.Expr,
        calls: Sequence[OpaqueCall],
        columns: Mapping[str, str],
        restricted: Mapping[str, tuple[DType, ...]],
    ) -> list[dict[str, ColumnLeaf]]:
        """Give each choice of the leaves of the columns and the declared columns that a code reads.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            expression: The expression of the code.
            calls: The calls of a function of the user, whose receivers read columns too.
            columns: The rendering of the data type of each column of the frame.
            restricted: The data types that a column with holes takes in
                place of its whole sample, by its name.

        Returns:
            The choices.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> evaluation = DeferredEvaluation("row")
            >>> code, columns = Code("pl.col('a')"), {"a": "Int8"}
            >>> choices = evaluation._leaf_choices(code, pl.col("a"), [], columns, {})
            >>> choices[0]["a"].dtype
            Int8
        """
        unnamed = {
            self.OPAQUE.format(place): self.UNCHECKED
            for place, call in enumerate(calls)
            if self._return_dtype(call) is None
        }
        renderings = {**columns, **self._declared(code.leaves, columns), **unnamed}
        roots = self._roots(expression, calls)
        fields = self._fields(expression)
        texts = self._texts(code)
        cast = {} if fields else self._cast_samples(expression, renderings, texts)
        planned = LeafChoices(self._leaf_planned(code))
        choices = planned.of(renderings, roots, fields, {**cast, **restricted}, texts)
        forms = {
            self.LEAF.format(place): leaf.form for place, leaf in enumerate(code.leaves) if leaf
        }
        return [self._formed(choice, forms) for choice in choices]

    def _cast_samples(
        self,
        expression: pl.Expr,
        renderings: Mapping[str, str],
        texts: Sequence[str],
    ) -> dict[str, tuple[DType, ...]]:
        """Give the sample of each open column that the code casts to a type that is not nested.

        Polars keeps a struct under a cast to a number, on a frame with rows,
        so a sample with a struct would leave the cast of an open column
        `Unchecked`. The checker takes that cast as an oddity of polars, so
        the sample of such a column leaves the structs out, and the cast gives
        its target. A column with no other data type in its sample keeps it.

        Args:
            expression: The expression.
            renderings: The rendering of the data type of each column, by its name.
            texts: The texts of the code, which the sample of an `Enum` holds.

        Returns:
            The sample of each such column, by its name.

        Example:
            >>> evaluation = DeferredEvaluation("row")
            >>> sample = evaluation._cast_samples(pl.col("u").cast(pl.Int8), {"u": "Unchecked"}, [])
            >>> any(dtype.name == "Struct" for dtype in sample["u"])
            False
            >>> evaluation._cast_samples(pl.col("u").cast(pl.Int8), {"u": "Int8"}, [])
            {}
        """
        restricted: dict[str, tuple[DType, ...]] = {}
        for name in self._cast_columns(expression):
            samples = OpenType((), texts).samples(renderings.get(name, self.UNCHECKED))
            kept = tuple(dtype for dtype in samples if dtype.name != "Struct")
            if kept and len(kept) < len(samples):
                restricted[name] = kept
        return restricted

    def _cast_columns(self, expression: pl.Expr) -> set[str]:
        """Give the columns that an expression casts, as they are, to a type that is not nested.

        Args:
            expression: The expression.

        Returns:
            The names.

        Example:
            >>> evaluation = DeferredEvaluation("row")
            >>> lists = pl.col("b").cast(pl.List(pl.Int8))
            >>> evaluation._cast_columns(pl.col("a").cast(pl.Int8) + lists)
            {'a'}
            >>> evaluation._cast_columns(pl.col("l").list.eval(pl.element().cast(pl.Int8)))
            set()
        """
        names: set[str] = set()
        pending: list[object] = [json.loads(expression.meta.serialize(format="json"))]
        while pending:
            node = pending.pop()
            if isinstance(node, list):
                pending.extend(node)
            if not isinstance(node, dict):
                continue
            pending.extend(node.values())
            column = self._cast_column(node)
            if column is not None:
                names.add(column)
        return names

    def _cast_column(self, node: Mapping[str, object]) -> str | None:
        """Give the column that one node of a tree casts, as it is, to a type that is not nested.

        Args:
            node: The node, as polars serializes it.

        Returns:
            The name, or `None` where the node is no such cast.

        Example:
            >>> cast = {"Cast": {"expr": "Element", "dtype": {"Literal": "Int8"}}}
            >>> DeferredEvaluation("row")._cast_column(cast) is None
            True
        """
        body = node.get("Cast")
        if not isinstance(body, dict) or not isinstance(body.get("expr"), dict):
            return None
        column = body["expr"].get("Column")
        return column if isinstance(column, str) and self._flat_target(body) else None

    def _flat_target(self, cast: Mapping[str, Any]) -> bool:
        """Say whether the target of a cast node is a type that is not nested.

        Args:
            cast: The body of the cast node, as polars serializes it.

        Returns:
            Whether it is. `False` where the target does not read.

        Example:
            >>> body = json.loads(pl.col("a").cast(pl.Int8).meta.serialize(format="json"))["Cast"]
            >>> DeferredEvaluation("row")._flat_target(body)
            True
            >>> DeferredEvaluation("row")._flat_target({})
            False
        """
        try:
            target = OptionReader().dtype(cast["dtype"])
        except (KeyError, UndecidedError):  # pragma: no cover -- every data type of polars reads
            return False
        return target.name not in {"Struct", "List", "Array"}

    def _formed(
        self,
        choice: Mapping[str, ColumnLeaf],
        forms: Mapping[str, RowForm],
    ) -> dict[str, ColumnLeaf]:
        """Give one choice of the leaves, each declared leaf in the row form that it declares.

        Args:
            choice: The leaf of each column, by its name.
            forms: The row form of each declared leaf, by its reserved name.

        Returns:
            The choice.

        Example:
            >>> leaf = ColumnLeaf(DType("Float64"), RowForm.PER_ROW)
            >>> forms = {"__leaf_0__": RowForm.ONE_VALUE}
            >>> formed = DeferredEvaluation("row")._formed({"__leaf_0__": leaf}, forms)
            >>> formed["__leaf_0__"].form
            <RowForm.ONE_VALUE: 'one value'>
            >>> DeferredEvaluation("row")._formed({}, forms)
            {}
        """
        formed = dict(choice)
        for name, form in forms.items():
            if name in formed:
                formed[name] = replace(formed[name], form=form)
        return formed

    def _texts(self, code: Code[DeclaredLeaf | None]) -> list[str]:
        """Give the texts that a code writes, apart from the kinds of its holes.

        Args:
            code: The code.

        Returns:
            The texts.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row")._texts(Code("pl.col('e').replace('a', hole('str'))"))
            ['a', 'e']
        """
        nodes = ast.walk(ast.parse(CodeRefs().parseable(code.text), mode="eval"))
        texts = [node.value for node in nodes if isinstance(node, ast.Constant)]
        kinds = HoleValues.KINDS
        return [text for text in texts if isinstance(text, str) and text not in kinds]

    def _signature(
        self,
        typed: Mapping[int, object],
        leaf: Mapping[str, ColumnLeaf],
    ) -> tuple[str, ...]:
        """Give the kinds of one choice of the data types of the holes and of the columns.

        Args:
            typed: The data type of each hole of a data type, by its place.
            leaf: The leaf of each column, by its name.

        Returns:
            The kind of each data type, in order.

        Example:
            >>> leaf = {"d": ColumnLeaf(DType("Decimal", (38, 0)), RowForm.PER_ROW)}
            >>> DeferredEvaluation("row")._signature({}, leaf)
            ('Decimal',)
        """
        dtypes = [DType.of(value) for value in typed.values() if isinstance(value, pl.DataType)]
        dtypes.extend(column.dtype for column in leaf.values())
        return tuple(OpenParameters().kind(dtype) for dtype in dtypes)

    def _kinds(
        self,
        code: Code[DeclaredLeaf | None],
        leaves: Sequence[Mapping[str, ColumnLeaf]],
        choices: Sequence[Mapping[int, object]],
        substitution: Substitution,
    ) -> list[set[Sample]]:
        """Give the answers at each choice of the leaves, one answer for each choice of the holes.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            leaves: Each choice of the leaves of the columns.
            choices: Each choice of the values of the holes.
            substitution: The column that stands for the selector of the code, if any.

        Returns:
            One set of answers for each choice of the leaves.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> leaves = [{"a": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}]
            >>> code = Code("pl.col('a') + hole('int')")
            >>> evaluation = DeferredEvaluation("row")
            >>> kinds = evaluation._kinds(code, leaves, [{0: 1}, {0: 2**40}], Substitution())
            >>> [len(answers) for answers in kinds]
            [2]
        """
        rows = [self._sampled(code, leaves, choice, substitution) for choice in choices]
        return [{row[place] for row in rows} for place in range(len(leaves))]

    def _typed(self, holes: Sequence[str], sample: OpenType) -> list[dict[int, object]]:
        """Give each choice of the data types of the holes of a data type.

        A data type whose parameters the checker does not know stands for
        several data types, as a column with holes does. Its choices join the
        choices of the leaves, so the answers at its data types give what they
        share, and need not agree.

        Args:
            holes: The kind of each hole, in the order of the code.
            sample: The samples of an open type, with the fields that the code
                reads, so a struct of the sample holds each of them.

        Returns:
            The choices, a data type of polars for each hole of a data type by
            its place. No choice where the choices are too many.

        Example:
            >>> choices = DeferredEvaluation("row")._typed(["int", "dtype:Duration"], OpenType())
            >>> len(choices), choices[0]
            (3, {1: Duration(time_unit='ns')})
        """
        places = [place for place, kind in enumerate(holes) if kind.startswith(HoleValues.DTYPE)]
        options = []
        for place in places:
            samples = sample.samples(holes[place].removeprefix(HoleValues.DTYPE))
            options.append([sample.polars() for sample in samples])
        if math.prod(len(dtypes) for dtypes in options) > self.LIMIT:
            return []
        return [dict(zip(places, dtypes, strict=True)) for dtypes in itertools.product(*options)]

    def _combined(
        self,
        base: DeferredAnswer,
        holes: Sequence[str],
        kinds: Sequence[set[Sample]],
        signatures: Sequence[tuple[str, ...]],
    ) -> DeferredAnswer:
        """Give the answer of a code from its answers at each choice of its holes and its leaves.

        The value of a hole must not change the answer at any choice of the
        leaves. A value where polars refuses drops out, because the code can
        hold another value, as a choice of the data types of the columns
        where polars refuses drops out, because the frame can hold another
        data type. Where every
        choice refuses, the answer is the refusal. The other answers give what
        they share (`Generalization`), with their parameters left open where a
        choice of the same kinds refuses.

        Args:
            base: The answer with the name and the columns that the code reads.
            holes: The kind of each hole of the code.
            kinds: The answers at each choice of the leaves, one for each choice
                of the values of the holes.
            signatures: The kinds of each choice of the leaves, in the same order.

        Returns:
            The answer.

        Example:
            >>> base = DeferredAnswer("a", "Unchecked")
            >>> evaluation = DeferredEvaluation("row")
            >>> kinds = [{Sample("Int8")}, {Sample("Unchecked", refused=True)}]
            >>> evaluation._combined(base, [], kinds, [("Int8",), ("String",)]).rendering
            'Int8'
            >>> evaluation._combined(base, ["int"], [{Sample("Int8"), Sample("Int16")}], [()]).cause
            'number'
            >>> evaluation._combined(base, [], [], []).cause
            'open'
        """
        if not kinds:
            reason = "the data types of the columns give too many choices"
            return replace(base, reason=reason, cause="open")
        kept = [self._valued(samples) for samples in kinds]
        if any(len(samples) != 1 for samples in kept):
            reason = "the value of a hole decides the answer"
            return replace(base, reason=reason, cause=self._hole_cause(holes))
        samples = [next(iter(samples)) for samples in kept]
        answered = self._answered_samples(samples, signatures)
        if not answered:
            shared = self._shared_refusal(samples)
            refusals = tuple(sorted({sample.reason for sample in samples}))
            return replace(
                base,
                refused=True,
                reason=shared.reason,
                cast=self._shared_cast(samples, shared),
                refusals=refusals,
            )
        return self._generalized(base, answered)

    def _shared_refusal(self, samples: Sequence[Sample]) -> Sample:
        """Give the refusal that most choices of the leaves share, where every choice refuses.

        A refusal that does not depend on the data types, as a column that
        an `eval` must not name, repeats at each choice that reaches it. A
        refusal of one data type names that type, so it differs at each one.

        Args:
            samples: The refusals, one for each choice.

        Returns:
            The first refusal among those that most choices share.

        Example:
            >>> samples = [Sample("Unchecked", True, "a"), Sample("Unchecked", True, "b")]
            >>> DeferredEvaluation("row")._shared_refusal([*samples, samples[1]]).reason
            'b'
        """
        counts = Counter(sample.reason for sample in samples)
        return max(samples, key=lambda sample: counts[sample.reason])

    def _shared_cast(self, samples: Sequence[Sample], shared: Sample) -> tuple[str, str] | None:
        """Give the cast that every choice refuses, with the target that the choices share.

        Where a hole of a data type stands in the target of a cast, each choice
        refuses the cast to its own sample, so no one sample names the target
        that the code writes. The target that the samples share does, with
        `Unchecked` at the parts that the hole decides.

        Args:
            samples: The refusals, one for each choice.
            shared: The refusal that most choices share.

        Returns:
            The source and the shared target, or the cast of the shared refusal
            where the choices do not all refuse a cast from one source.

        Example:
            >>> listed = [
            ...     Sample("Unchecked", True, "a", cast=("Int64", "Array(Null, shape=(2,))")),
            ...     Sample("Unchecked", True, "b", cast=("Int64", "Array(Int8, shape=(2,))")),
            ... ]
            >>> DeferredEvaluation("row")._shared_cast(listed, listed[0])
            ('Int64', 'Array(Unchecked, shape=(2,))')
        """
        casts = [sample.cast for sample in samples if sample.cast is not None]
        sources = {source for source, _ in casts}
        if shared.cast is None or len(casts) != len(samples) or len(sources) != 1:
            return shared.cast
        targets = [Rendering().parse(target) for _, target in casts]
        return shared.cast[0], Generalization().of(targets).render()

    def _valued(self, samples: set[Sample]) -> set[Sample]:
        """Give the answers of the values of the holes that polars does not refuse, at one choice.

        Args:
            samples: The answer at each choice of the values of the holes.

        Returns:
            The answers that are not refusals, or one refusal where every value refuses.

        Example:
            >>> refused = Sample("Unchecked", refused=True)
            >>> DeferredEvaluation("row")._valued({Sample("Int8"), refused})
            {Sample(rendering='Int8', refused=False, reason='', cause='', cast=None, opened=False)}
        """
        answered = {sample for sample in samples if not sample.refused}
        if answered:
            return answered
        return {min(samples, key=lambda sample: sample.reason)}

    def _answered_samples(
        self,
        samples: Sequence[Sample],
        signatures: Sequence[tuple[str, ...]],
    ) -> list[Sample]:
        """Give the answers that are not refusals, with their parameters open where they can differ.

        A choice of the same kinds as a choice that refuses differs from it in
        a parameter only, as the precision of a `Decimal`. The sample holds a
        few values of each parameter, so the answer of such a kind holds its
        kinds and leaves its other parameters open.

        Args:
            samples: The answer at each choice of the leaves.
            signatures: The kinds of each choice, in the same order.

        Returns:
            The answers.

        Example:
            >>> wide = Sample("Decimal(precision=38, scale=0)")
            >>> samples = [wide, Sample("Unchecked", refused=True)]
            >>> answered = DeferredEvaluation("row")._answered_samples(samples, [("Decimal",)] * 2)
            >>> [sample.opened for sample in answered]
            [True]
        """
        pairs = list(zip(signatures, samples, strict=True))
        refused = {signature for signature, sample in pairs if sample.refused}
        answered: list[Sample] = []
        for signature, sample in pairs:
            if not sample.refused:
                answered.append(replace(sample, opened=signature in refused))
        return answered

    def _generalized(self, base: DeferredAnswer, samples: Sequence[Sample]) -> DeferredAnswer:
        """Give what the answers at several choices of the leaves share.

        Args:
            base: The answer with the name and the columns that the code reads.
            samples: The answers that are not refusals.

        Returns:
            The answer. An answer that the rules do not know at one choice
            gives `Unchecked`, with its cause.

        Example:
            >>> base = DeferredAnswer("a", "Unchecked")
            >>> evaluation = DeferredEvaluation("row")
            >>> lists = [Sample("List(Int8)"), Sample("List(String)")]
            >>> evaluation._generalized(base, lists).rendering
            'List(Unchecked)'
            >>> unknown = [Sample("Int8"), Sample("Unchecked", cause="rules")]
            >>> evaluation._generalized(base, unknown).cause
            'rules'
        """
        unknown = [sample for sample in samples if sample.rendering == self.UNCHECKED]
        if unknown:
            return replace(base, reason=unknown[0].reason, cause=unknown[0].cause)
        shared = Generalization().shared([self._dtype_of(sample) for sample in samples])
        rendering = shared.render()
        cause = "open" if rendering == self.UNCHECKED else ""
        return replace(base, rendering=rendering, cause=cause)

    def _dtype_of(self, sample: Sample) -> DType:
        """Give the data type of an answer, with its parameters open where they can change.

        Args:
            sample: The answer.

        Returns:
            The data type.

        Example:
            >>> wide = Sample("Decimal(precision=38, scale=0)", opened=True)
            >>> DeferredEvaluation("row")._dtype_of(wide)
            Decimal
        """
        dtype = Rendering().parse(sample.rendering)
        return OpenParameters().opened(dtype) if sample.opened else dtype

    def _choices(
        self,
        holes: Sequence[str],
        built: Mapping[int, object] | None = None,
    ) -> list[dict[int, object]]:
        """Give each choice of the values of the holes that the evaluation tries.

        A hole of another kind, such as an expression or a function, takes its
        placeholder in every choice: the evaluation reads it in another way.
        A choice that polars builds where the placeholders do not build comes
        first, so that the evaluation tries a value that the code can hold.

        Args:
            holes: The kind of each hole, in the order of the code.
            built: The values that built the expression where the placeholders
                do not build, by the places of the holes.

        Returns:
            The choices, one value for each hole by its place, where `None`
            stands for the placeholder. No choice where the choices are too many.

        Example:
            >>> DeferredEvaluation("row")._choices(["bool", "function"])
            [{0: False, 1: None}, {0: True, 1: None}]
            >>> DeferredEvaluation("row")._choices(["int"] * 3)
            []
            >>> DeferredEvaluation("row")._choices([HoleValues().listing(("a", "b"))])
            [{0: 'a'}, {0: 'b'}]
            >>> DeferredEvaluation("row")._choices(["str"], {0: "{}"})
            [{0: '{}'}, {0: 'a'}, {0: '%Y'}]
        """
        options = [HoleValues().listed(kind) or self.SAMPLES.get(kind, (None,)) for kind in holes]
        if math.prod(len(values) for values in options) > self.LIMIT:
            return []
        choices = [dict(enumerate(values)) for values in itertools.product(*options)]
        typed = {place for place, kind in enumerate(holes) if kind.startswith(HoleValues.DTYPE)}
        first = {place: value for place, value in (built or {}).items() if place not in typed}
        if not first:
            return choices
        first = {**dict.fromkeys(range(len(holes))), **first}
        return [first, *(choice for choice in choices if choice != first)]

    def _rebuilt(self, code: Code[DeclaredLeaf | None], refusal: BuildRefusal) -> ReadCode | None:
        """Read a code again with other values of its holes, where polars refuses the placeholders.

        The placeholder of a hole is one value of its kind, and polars can
        refuse to build the code with it, as `pl.format` with a text that has
        no `{}` for its arguments. The code can hold another value, so the
        reader tries the samples of each hole, with a few texts that hold
        `{}` (`BUILD_SAMPLES`), until polars builds the code.

        Args:
            code: The code.
            refusal: The refusal of the placeholders, with the holes before the refused call.

        Returns:
            The read with the first values that polars builds, or `None` where
            none builds.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> code = Code("pl.format(hole('str'), 1, 2)")
            >>> refusal = BuildRefusal("ValueError", "no", ("str",))
            >>> DeferredEvaluation("row")._rebuilt(code, refusal).choice
            {0: '{}{}'}
            >>> DeferredEvaluation("row")._rebuilt(code, BuildRefusal("ValueError", "no")) is None
            True
        """
        options = [self._build_values(kind) for kind in refusal.holes]
        if not options:
            return None
        for values in itertools.islice(itertools.product(*options), self.EVALUATIONS):
            try:
                return CodeReader().read(code, dict(enumerate(values)))
            except CodeReadError:
                continue
        return None

    def _build_values(self, kind: str) -> tuple[object, ...]:
        """Give the values of a hole that a new build of its code tries.

        Args:
            kind: The kind of the hole.

        Returns:
            The values: the listed values, the sample of a data type, or the
            samples of the kind with the texts that hold `{}`.

        Example:
            >>> DeferredEvaluation("row")._build_values("bool")
            (False, True)
            >>> len(DeferredEvaluation("row")._build_values("dtype:Duration"))
            3
        """
        listed = HoleValues().listed(kind)
        if listed:
            return listed
        if kind.startswith(HoleValues.DTYPE):
            samples = OpenType().samples(kind.removeprefix(HoleValues.DTYPE))
            return tuple(sample.polars() for sample in samples)
        return (*self.SAMPLES.get(kind, (None,)), *self.BUILD_SAMPLES.get(kind, ()))

    def _sampled(
        self,
        code: Code[DeclaredLeaf | None],
        leaves: Sequence[Mapping[str, ColumnLeaf]],
        choice: Mapping[int, object],
        substitution: Substitution,
    ) -> list[Sample]:
        """Give the answers of a code at one choice of the values of its holes.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            leaves: Each choice of the leaves of the columns, by their names.
            choice: The value of each hole, by its place.
            substitution: The column that stands for the selector of the code, if any.

        Returns:
            The answer at each choice of the leaves.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> added = Code("pl.col('a') + hole('int')")
            >>> leaves = [{"a": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}]
            >>> DeferredEvaluation("row")._sampled(added, leaves, {0: 2**40}, Substitution())
            [Sample(rendering='Int64', refused=False, reason='', cause='', cast=None, opened=False)]
        """
        try:
            read = CodeReader().read(code, choice)
        except BuildRefusal as refusal:
            return [Sample(self.UNCHECKED, refused=True, reason=str(refusal))] * len(leaves)
        expression = substitution(checked_cast(read.expression, pl.Expr))
        samples: list[Sample] = []
        calls = substitution.calls(read.opaque)
        for columns in leaves:
            opaque = self._opaque(calls, columns)
            known = {**columns, **opaque}
            refusal = self._refused_receiver(calls, known) or self._undeclared(code.leaves, known)
            samples.append(refusal or self._sample(expression, known))
        return samples

    def _undeclared(
        self,
        declared: Sequence[DeclaredLeaf | None],
        leaves: Mapping[str, ColumnLeaf],
    ) -> Sample | None:
        """Give the refusal of a choice where a declared leaf has a data type it does not allow.

        A declaration promises its outcome only at the data types that it
        allows. An argument that fits it can give any other type, or none,
        at another data type, so the code has no answer there.

        Args:
            declared: A declared column, or `None`, for each leaf of the code.
            leaves: The leaf of each column of the choice, by its name.

        Returns:
            The refusal, or `None`.

        Example:
            >>> from astrolabe._hooks.deferred.argument import DeclaredOutcome
            >>> from astrolabe._hooks.expression.fit import Read, Row, SliceDecl
            >>> from astrolabe._hooks.expression.outcome import Axis
            >>> decl = SliceDecl(
            ...     (Read("p", frozenset({"Float32"})),),
            ...     (Row((frozenset({"Float32"}),), Axis("p")),),
            ...     None, "elementwise",
            ... )
            >>> leaf = DeclaredLeaf("p", "Float32", outcome=DeclaredOutcome(decl))
            >>> wide = {"__leaf_0__": ColumnLeaf(DType("Int64"), RowForm.PER_ROW)}
            >>> DeferredEvaluation("row")._undeclared([leaf], wide).reason
            'column "p" is Int64, which the declared type of an argument does not allow'
        """
        for place, leaf in enumerate(declared):
            if leaf is None or leaf.outcome is None:
                continue
            renderings = self._leaf_frame(leaf, place, leaves)
            if renderings is None or leaf.outcome.at(renderings) is not None:
                continue
            described = ", ".join(f'column "{name}" is {text}' for name, text in renderings.items())
            reason = f"{described}, which the declared type of an argument does not allow"
            return Sample(self.UNCHECKED, refused=True, reason=reason)
        return None

    def _leaf_frame(
        self,
        leaf: DeclaredLeaf,
        place: int,
        leaves: Mapping[str, ColumnLeaf],
    ) -> dict[str, str] | None:
        """Give the data type of each column that a declared leaf reads, at one choice.

        A declared column reads its own leaf. A declared map reads the
        columns of its declared reads.

        Args:
            leaf: The declared leaf.
            place: The place of the leaf in the code.
            leaves: The leaf of each column of the choice, by its name.

        Returns:
            The renderings, by the declared name of each read. `None` where
            the choice holds no data type for a read.

        Example:
            >>> int8 = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> evaluation = DeferredEvaluation("row")
            >>> evaluation._leaf_frame(DeclaredLeaf("p", "Int8"), 0, {"__leaf_0__": int8})
            {'p': 'Int8'}
            >>> evaluation._leaf_frame(DeclaredLeaf("s", "Unchecked", ("a",)), 0, {}) is None
            True
        """
        if leaf.reads is None:
            reads = {leaf.name: self.LEAF.format(place)}
        else:
            reads = {name: name for name in leaf.reads}
        if not all(key in leaves for key in reads.values()):
            return None
        return {name: leaves[key].dtype.render() for name, key in reads.items()}

    def _refused_receiver(
        self,
        calls: Sequence[OpaqueCall],
        leaves: Mapping[str, ColumnLeaf],
    ) -> Sample | None:
        """Give the refusal of the receiver of a function of the user, if polars refuses it.

        Polars runs the receiver before the function, so a receiver that it
        refuses refuses the whole expression.

        Args:
            calls: The calls of a function of the user.
            leaves: The leaf of each column, by its name.

        Returns:
            The refusal, or `None`.

        Example:
            >>> text = {"s": ColumnLeaf(DType("String"), RowForm.PER_ROW)}
            >>> call = OpaqueCall(pl.col("s").abs(), "map_batches")
            >>> DeferredEvaluation("row")._refused_receiver([call], text).refused
            True
        """
        for call in calls:
            sample = self._sample(call.receiver, leaves)
            if sample.refused:
                return sample
        return None

    def _sample(self, expression: pl.Expr, leaves: Mapping[str, ColumnLeaf]) -> Sample:
        """Give the answer of an expression at one choice of its leaves.

        Args:
            expression: The expression.
            leaves: The leaf of each column, by its name.

        Returns:
            The answer.

        Example:
            >>> leaf = ColumnLeaf(DType("String"), RowForm.PER_ROW)
            >>> DeferredEvaluation("row")._sample(pl.col("t") + 1, {"t": leaf}).refused
            True
        """
        try:
            answer = self._evaluated(expression, leaves)
        except Exception as error:  # a fault of the rules must not end the run of mypy
            return Sample(self.UNCHECKED, reason=f"a fault of the rules: {error!r}", cause="rules")
        if isinstance(answer, DType):
            return Sample(answer.render())
        if isinstance(answer, Refused):
            cast = self._cast(expression, leaves)
            return Sample(self.UNCHECKED, refused=True, reason=answer.text(), cast=cast)
        cause = "opaque" if self.OPAQUE_PREFIX in repr(answer) else "rules"
        return Sample(self.UNCHECKED, reason=repr(answer), cause=cause)

    def _evaluated(self, expression: pl.Expr, leaves: Mapping[str, ColumnLeaf]) -> Answer:
        """Give the run answer of an expression in the context, at the leaves of its columns.

        Args:
            expression: The expression.
            leaves: The leaf of each column, by its name.

        Returns:
            The answer.

        Example:
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> DeferredEvaluation("row")._evaluated(pl.col("a").abs(), {"a": leaf})
            Int8
        """
        tree = ExpressionTree().of(expression)
        return self._picked(TreeEvaluation(leaves, self._groups).answers(tree))

    def _cast(
        self,
        expression: pl.Expr,
        leaves: Mapping[str, ColumnLeaf],
    ) -> tuple[str, str] | None:
        """Give the data types of a refused cast at the root of a refused expression.

        A cast that polars refuses gets an error that names both data types.

        Args:
            expression: The refused expression.
            leaves: The leaf of each column, by its name.

        Returns:
            The rendering of the source and of the target, or `None` where the
            root is not a cast, or where its input refuses too.

        Example:
            >>> leaf = ColumnLeaf(DType("String"), RowForm.PER_ROW)
            >>> cast = pl.col("s").cast(pl.Struct({"a": pl.Int8}))
            >>> DeferredEvaluation("row")._cast(cast, {"s": leaf})
            ('String', "Struct({'a': Int8})")
            >>> DeferredEvaluation("row")._cast(pl.col("s").abs(), {"s": leaf}) is None
            True
        """
        tree = json.loads(expression.meta.serialize(format="json"))
        body = tree.get("Cast") if isinstance(tree, dict) else None
        if not isinstance(body, dict):
            return None
        source = pl.Expr.deserialize(io.StringIO(json.dumps(body["expr"])), format="json")
        answer = self._evaluated(source, leaves)
        if not isinstance(answer, DType):
            return None
        target = OptionReader().dtype(body["dtype"])
        return answer.render(), target.render()

    def _refused_build(
        self,
        code: Code[DeclaredLeaf | None],
        refusal: BuildRefusal,
    ) -> DeferredAnswer:
        """Give the answer of a code that polars refuses to build.

        A placeholder of a hole, of a value or of a function, can make polars
        refuse where the value of the user does not, so such a code gives
        `Unchecked`. A code with no hole is refused at every frame.

        Args:
            code: The code.
            refusal: The refusal of polars.

        Returns:
            The answer.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> refusal = BuildRefusal("ComputeError", "no")
            >>> DeferredEvaluation("row")._refused_build(Code("pl.lit(1)"), refusal).refused
            True
            >>> DeferredEvaluation("row")._refused_build(Code("pl.lit(hole('int'))"), refusal).cause
            'hole'
            >>> evaluation = DeferredEvaluation("row")
            >>> evaluation._refused_build(Code("cs.by_dtype([hole('function')])"), refusal).cause
            'hole'
        """
        reason = str(refusal)
        if HoleValues().any_in_text(code.text):
            name = self._alias_of(code)
            return DeferredAnswer(name, self.UNCHECKED, reason=reason, cause="hole")
        build = (refusal.error, reason)
        return DeferredAnswer(None, self.UNCHECKED, refused=True, reason=reason, build=build)

    def _alias_of(self, code: Code[DeclaredLeaf | None]) -> str | None:
        """Give the name of the outermost `.alias()` of a code that the reader cannot build.

        The name of such a code is still known where its last call names it.

        Args:
            code: The code.

        Returns:
            The name, or `None`.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row")._alias_of(Code("pl.col('a').cast().alias('t')"))
            't'
            >>> DeferredEvaluation("row")._alias_of(Code("pl.col(")) is None
            True
            >>> DeferredEvaluation("row")._alias_of(Code("1")) is None
            True
        """
        try:
            node = ast.parse(CodeRefs().parseable(code.text), mode="eval").body
        except (SyntaxError, tokenize.TokenError):
            return None
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return None
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        named = node.func.attr == "alias" and len(arguments) == 1
        value = arguments[0] if named else None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        return None

    def _picked(self, answers: CellAnswers) -> Answer:
        """Give the answer of the context: the run type, the planner type or the stored type.

        Args:
            answers: The four answers of the tree.

        Returns:
            The answer in `agg` or in a `select`.

        Example:
            >>> from astrolabe._rules.node_evaluation import CellAnswers
            >>> int8, listed = DType("Int8"), DType("List", (DType("Int8"),))
            >>> DeferredEvaluation("agg")._picked(CellAnswers(int8, listed, int8, listed))
            List(Int8)
            >>> literal = DType.dynamic_int(1)
            >>> stored = DeferredEvaluation("lazy row stored")
            >>> stored._picked(CellAnswers(literal, literal, int8, int8))
            Int32
        """
        agg = "agg" in self._context
        if self._context == "planner":
            return answers.planner_select
        if self._context.endswith(self.STORED):
            return self._stored(answers.planner_agg if agg else answers.planner_select)
        return answers.run_agg if agg else answers.run_select

    def _stored(self, planned: Answer) -> Answer:
        """Give the type that a lazy frame stores for a planner type: a literal takes its own type.

        Args:
            planned: The planner answer.

        Returns:
            The stored answer.

        Example:
            >>> DeferredEvaluation("lazy row stored")._stored(DType.unknown("float"))
            Float64
            >>> DeferredEvaluation("lazy row stored")._stored(DType("Unknown", ("int", "a")))
            unchecked: the stored type of a literal of no value
        """
        if not isinstance(planned, DType) or planned.name != "Unknown":
            return planned
        try:
            return planned.materialized()
        except NoPartError:  # pragma: no cover -- a dynamic integer holds its value
            return Unchecked("the stored type of a literal of no value")

    def _leaf_planned(self, code: Code[DeclaredLeaf | None]) -> dict[str, str]:
        """Give the stored type of each column and each declared column of a code that has one.

        A declared column that reads a column of the frame takes the stored
        type of that column, as it takes its data type. A declared map, as the
        leaf of a declared selection at one picked column, takes the type that
        it promises at the stored types of the columns that it reads.

        Args:
            code: The code, with a declared column or `None` for each leaf.

        Returns:
            The renderings of the stored types, by the name of the column or
            the reserved name of the declared column.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> code = Code("$0", (DeclaredLeaf("d", "Unchecked"),))
            >>> DeferredEvaluation("lazy row", {"d": "Float32"})._leaf_planned(code)
            {'d': 'Float32', '__leaf_0__': 'Float32'}
        """
        planned = dict(self._planned)
        for place, leaf in enumerate(code.leaves):
            stored = None if leaf is None else self._stored_leaf(leaf)
            if stored is not None:
                planned[self.LEAF.format(place)] = stored
        return planned

    def _stored_leaf(self, leaf: DeclaredLeaf) -> str | None:
        """Give the stored type of a declared column, or of a declared map over stored columns.

        Args:
            leaf: The declared column or map.

        Returns:
            The rendering, or `None` where a column that it reads has no stored
            type, or where the map promises no type there.

        Example:
            >>> evaluation = DeferredEvaluation("lazy row", {"d": "Float32"})
            >>> evaluation._stored_leaf(DeclaredLeaf("d", "Unchecked"))
            'Float32'
            >>> evaluation._stored_leaf(DeclaredLeaf("x", "Unchecked", ("e",))) is None
            True
        """
        if leaf.outcome is None or leaf.reads is None:
            return self._planned.get(leaf.name)
        if not leaf.reads or any(read not in self._planned for read in leaf.reads):
            return None
        return leaf.outcome.at({read: self._planned[read] for read in leaf.reads})

    def _declared(
        self,
        declared: Sequence[DeclaredLeaf | None],
        columns: Mapping[str, str],
    ) -> dict[str, str]:
        """Give the rendering of each declared column of the code, under its reserved name.

        A declared column reads the column of its name, so where the frame
        has that column, the leaf takes the data type of the frame. A choice
        of that data type that the declaration does not allow refuses, at
        each choice of the leaves.

        Args:
            declared: A declared column, or `None`, for each leaf of the code.
            columns: The rendering of the data type of each column of the frame.

        Returns:
            The renderings, by the reserved name of the code reader.

        Example:
            >>> leaves = [DeclaredLeaf("p", "Unchecked"), DeclaredLeaf("q", "Int8"), None]
            >>> DeferredEvaluation("row")._declared(leaves, {"p": "Float32"})
            {'__leaf_0__': 'Float32', '__leaf_1__': 'Int8'}
        """
        named: dict[str, str] = {}
        for place, leaf in enumerate(declared):
            if leaf is not None and leaf.outcome is not None and leaf.reads is not None:
                named[self.LEAF.format(place)] = leaf.outcome.at(columns) or self.UNCHECKED
            elif leaf is not None:
                named[self.LEAF.format(place)] = columns.get(leaf.name, leaf.rendering)
        return named

    def _opaque(
        self,
        calls: Sequence[OpaqueCall],
        columns: Mapping[str, ColumnLeaf],
    ) -> dict[str, ColumnLeaf]:
        """Give the leaf of each call of a function of the user, in the rows that the call gives.

        A call whose `return_dtype=` names a type has that type. Any other
        call is a column with holes, whose sampled type the leaf of the
        reserved name holds.

        Args:
            calls: The calls, by their place.
            columns: The leaf of each column, where a receiver reads its height,
                and the sampled leaf of each call with no `return_dtype=`.

        Returns:
            The leaves, by the reserved name of the code reader.

        Example:
            >>> call = OpaqueCall(pl.col("a"), "map_batches", {"return_dtype": pl.Int16})
            >>> DeferredEvaluation("row")._opaque([call], {})["__opaque_0__"].dtype
            Int16
            >>> DeferredEvaluation("row")._opaque([OpaqueCall(pl.col("a"), "map_batches")], {})
            {}
        """
        leaves: dict[str, ColumnLeaf] = {}
        for place, call in enumerate(calls):
            name = self.OPAQUE.format(place)
            named = self._return_dtype(call)
            sampled = columns[name].dtype if name in columns else None
            dtype = named if named is not None else sampled
            if dtype is not None:
                leaves[name] = ColumnLeaf(dtype, self._form(call, columns))
        return leaves

    def _return_dtype(self, call: OpaqueCall) -> DType | None:
        """Give the type that the `return_dtype=` of a call of a function of the user names.

        Args:
            call: The call.

        Returns:
            The type, or `None` where the call names none.

        Example:
            >>> call = OpaqueCall(pl.col("a"), "map_batches", {"return_dtype": pl.Int16})
            >>> DeferredEvaluation("row")._return_dtype(call)
            Int16
        """
        named = call.keywords.get("return_dtype")
        if isinstance(named, pl.DataType) or (
            isinstance(named, type) and issubclass(named, pl.DataType)
        ):
            return DType.of(named)
        return None

    def _form(self, call: OpaqueCall, columns: Mapping[str, ColumnLeaf]) -> RowForm:
        """Give the rows that a call of a function of the user gives, as polars runs it.

        A function that runs element by element, as `map_elements` or a call
        with `is_elementwise=True`, keeps the height of its receiver: over an
        aggregation in `agg`, it gives one value for each group. Another call
        gives one value with `returns_scalar=True`, and a value for each row
        otherwise, so `agg` gathers its values in a list.

        Args:
            call: The call.
            columns: The leaf of each column, where the receiver reads its height.

        Returns:
            The row form of the leaf.

        Example:
            >>> leaves = {"a": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
            >>> summed = OpaqueCall(pl.col("a").sum(), "map_elements")
            >>> DeferredEvaluation("agg")._form(summed, leaves)
            <RowForm.ONE_VALUE: 'one value'>
            >>> batched = OpaqueCall(pl.col("a").sum(), "map_batches")
            >>> DeferredEvaluation("agg")._form(batched, leaves)
            <RowForm.PER_ROW: 'per row'>
            >>> unknown = OpaqueCall(pl.col("b"), "map_elements")
            >>> DeferredEvaluation("agg")._form(unknown, leaves)
            <RowForm.PER_ROW: 'per row'>
        """
        elementwise = call.method == "map_elements" or call.keywords.get("is_elementwise") is True
        if not elementwise:
            scalar = call.keywords.get("returns_scalar") is True
            return RowForm.ONE_VALUE if scalar else RowForm.PER_ROW
        try:
            receiver = ExpressionTree().of(call.receiver)
            height = TreeEvaluation(columns, self._groups).height(receiver)
        except (PolarsRefusal, UndecidedError):
            return RowForm.PER_ROW
        return RowForm.ONE_VALUE if height is Height.SCALAR else RowForm.PER_ROW

    def _name(
        self,
        expression: pl.Expr,
        declared: Sequence[DeclaredLeaf | None],
        calls: Sequence[OpaqueCall],
    ) -> str | None:
        """Give the output name of an expression, with the name of a reserved column put back.

        A declared column gives its own name, and a call of a function of the
        user gives the name of its receiver. A name that a hole of text gives
        is not known, and neither is a name that an expression that is not
        typed gives, nor the name of a declared column whose name the checker
        does not read or whose name is a plain `str`.

        Args:
            expression: The expression.
            declared: A declared column, or `None`, for each leaf of the code.
            calls: The calls of a function of the user, by their place.

        Returns:
            The name, or `None` where polars gives none.

        Example:
            >>> leaf = [DeclaredLeaf("p", "Int8")]
            >>> DeferredEvaluation("row")._name(pl.col("__leaf_0__").abs(), leaf, [])
            'p'
            >>> call = OpaqueCall(pl.col("a"), "map_batches")
            >>> DeferredEvaluation("row")._name(pl.col("__opaque_0__"), [], [call])
            'a'
            >>> DeferredEvaluation("row")._name(pl.col(HoleValues.STR), [], []) is None
            True
            >>> DeferredEvaluation("row")._name(pl.struct(pl.col("__hole_0__")), [], []) is None
            True
            >>> DeferredEvaluation("row")._name(pl.col("__leaf_0__") + 1, [None], []) is None
            True
        """
        name = self.output_name(expression)
        if name is None or HoleValues.EXPR_NAME.search(name):
            return None
        for place, leaf in enumerate(declared):
            reserved = self.LEAF.format(place)
            if leaf is None and reserved in name:
                return None
            if leaf is not None:
                name = name.replace(reserved, leaf.name)
        return None if HoleValues.STR in name else self._receivers(name, calls)

    def _receivers(self, name: str, calls: Sequence[OpaqueCall]) -> str:
        """Put the name of the receiver of each call of a function of the user in its place.

        A rename after the call keeps the reserved name inside the output
        name, as `__opaque_0___x` for `.name.suffix("_x")`.

        Args:
            name: The output name.
            calls: The calls, by their place.

        Returns:
            The name.

        Example:
            >>> call = OpaqueCall(pl.col("a"), "map_batches")
            >>> DeferredEvaluation("row")._receivers("__opaque_0___x", [call])
            'a_x'
        """
        for place, call in enumerate(calls):
            reserved = self.OPAQUE.format(place)
            receiver = self.output_name(call.receiver) if reserved in name else None
            if receiver is not None:
                name = name.replace(reserved, receiver)
        return name

    def output_name(self, expression: pl.Expr) -> str | None:
        """Give the output name that polars gives an expression.

        A frame names a node of no input, as `pl.format` of a text with no
        placeholder, `literal`.

        Args:
            expression: The expression.

        Returns:
            The name, or `None` where polars gives none.

        Example:
            >>> DeferredEvaluation("row").output_name(pl.lit(1).over("g").name.keep())
            'g'
            >>> DeferredEvaluation("row").output_name(pl.all()) is None
            True
            >>> DeferredEvaluation("row").output_name(pl.format("x"))
            'literal'
        """
        tree = json.loads(expression.meta.serialize(format="json"))
        try:
            return OptionReader().first_name([tree])
        except pl.exceptions.ComputeError:
            return self._no_input_name(tree)

    def _no_input_name(self, tree: object) -> str | None:
        """Give the name of a function of no input, which a frame names `literal`.

        Args:
            tree: The serialized expression.

        Returns:
            `literal` for a function of no input, or `None`.

        Example:
            >>> tree = {"Function": {"input": [], "function": "X"}}
            >>> DeferredEvaluation("row")._no_input_name(tree)
            'literal'
        """
        body = tree.get("Function") if isinstance(tree, dict) else None
        inputs = body.get("input") if isinstance(body, dict) else None
        return self.NO_INPUT if inputs == [] else None

    def _reads(
        self,
        expression: pl.Expr,
        calls: Sequence[OpaqueCall],
        declared: Sequence[DeclaredLeaf | None],
    ) -> tuple[str, ...]:
        """Give the columns of the frame that an expression reads, by the names of the frame.

        A declared column reads the column of its name. Any other reserved
        name of the code reader reads no column.

        Args:
            expression: The expression.
            calls: The calls of a function of the user, whose receivers read columns too.
            declared: A declared column, or `None`, for each leaf of the code.

        Returns:
            The names, in the order of polars.

        Example:
            >>> call = OpaqueCall(pl.col("b"), "map_batches")
            >>> leaf = [DeclaredLeaf("p", "Int8")]
            >>> expression = pl.col("a") + pl.col("__opaque_0__") + pl.col("__leaf_0__")
            >>> DeferredEvaluation("row")._reads(expression, [call], leaf)
            ('a', 'p', 'b')
        """
        names: dict[str, tuple[str, ...]] = {}
        for place, leaf in enumerate(declared):
            if leaf is not None:
                names[self.LEAF.format(place)] = leaf.read_names()
        roots = [
            name for root in self._roots(expression, calls) for name in names.get(root, (root,))
        ]
        known = [root for root in roots if HoleValues.STR not in root]
        return tuple(root for root in known if not root.startswith("__"))

    def fields_of(self, code: Code[DeclaredLeaf | None]) -> tuple[str, ...]:
        """Give the names of the fields of a struct that a code reads by name.

        Args:
            code: The code.

        Returns:
            The names, each once, and no name where the reader cannot read the code.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredEvaluation("row").fields_of(Code("cs.struct().struct.field('x')"))
            ('x',)
            >>> DeferredEvaluation("row").fields_of(Code("pl.col("))
            ()
        """
        try:
            read = CodeReader().read(code)
        except CodeReadError:
            return ()
        expression = read.expression
        return tuple(self._fields(expression)) if isinstance(expression, pl.Expr) else ()

    def _fields(self, expression: pl.Expr) -> list[str]:
        """Give the names of the fields of a struct that an expression reads by name.

        Args:
            expression: The expression.

        Returns:
            The names, each once.

        Example:
            >>> evaluation = DeferredEvaluation("row")
            >>> fields = pl.col("s").struct.field("f") + pl.col("s").struct.field("g", "f")
            >>> sorted(evaluation._fields(fields))
            ['f', 'g']
            >>> evaluation._fields(pl.col("s").struct.with_fields(pl.field("h")))
            ['h']
        """
        names: list[str] = []
        pending: list[object] = [json.loads(expression.meta.serialize(format="json"))]
        while pending:
            node = pending.pop()
            if isinstance(node, list):
                pending.extend(node)
            if not isinstance(node, dict):
                continue
            pending.extend(node.values())
            found = [node.get("FieldByName"), *node.get("Field", []), *node.get("names", [])]
            names.extend(name for name in found if isinstance(name, str) and name not in names)
        return names

    def _roots(self, expression: pl.Expr, calls: Sequence[OpaqueCall]) -> tuple[str, ...]:
        """Give the columns that an expression reads, with the reserved names of its leaves.

        Args:
            expression: The expression.
            calls: The calls of a function of the user, whose receivers read columns too.

        Returns:
            The names, in the order of polars.

        Example:
            >>> DeferredEvaluation("row")._roots(pl.col("__leaf_0__") + pl.col("a"), [])
            ('__leaf_0__', 'a')
        """
        roots = [expression, *(call.receiver for call in calls)]
        return tuple(name for root in roots for name in root.meta.root_names())
