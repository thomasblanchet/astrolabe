"""Evaluate a whole tree of polars from the definition of each node, from the leaves up.

A tree holds columns and literals at its leaves, and a node at each other
place (`ExpressionTree`). The evaluation gives the four answers of polars for
the whole expression: the planner type and the run type, in a `select` and
in `group_by().agg()`. It follows the three passes of polars:

1. the planner type of each node is its type function at the planner types
   of its inputs;
2. the cast pass visits the root once, after its inputs, and visits every
   other node twice: before its inputs, and after them. Each node then has a
   type after its visits, which the node above reads;
3. the run gives each node its rows and its data, in a `select` and over the
   groups of `agg`, from the outputs of its inputs.

Each node that is not the root stands as an input of the node above it, in
the form of a leaf (`NodeState`). The conversion of polars folds the
literals of an input of the root that holds only literals
(`FoldedLeaf`). A refusal anywhere below the root reaches the root, and a
node whose identity has no definition makes the answers `Unchecked`.
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard

import polars as pl

from astrolabe._hooks.checked import checked_not_none
from astrolabe._rules.definitions.node_options import NodeIdentity, OptionReader, SerializedDType
from astrolabe._rules.definitions.registry import Definitions
from astrolabe._rules.dtypes import DType
from astrolabe._rules.execution import (
    AggState,
    BinaryFold,
    GroupBy,
    InGroups,
    InSelect,
    QueryFlags,
    RowCount,
    SliceInput,
)
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop, Undecided, UndecidedError
from astrolabe._rules.kernels.dispatch import Data
from astrolabe._rules.node_evaluation import (
    Answers,
    CellAnswers,
    ColumnLeaf,
    LiteralLeaf,
    NodeEvaluation,
    NodeOutput,
    Refused,
)
from astrolabe._rules.operators import Operator
from astrolabe._rules.options import Height, KernelLength, NodeKind, RowForm, Shape
from astrolabe._rules.tree import ExpressionTree, KeptName
from astrolabe._rules.type_functions import ExprFields, Operand

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from astrolabe._rules.node_evaluation import Leaf
    from astrolabe._rules.tree import TreeNode


@dataclass(frozen=True)
class TypedLiteralLeaf:
    """Hold a literal whose type the code gives, such as `pl.lit(1, dtype=pl.Int8)`.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> TypedLiteralLeaf(1, DType("Int8")).operand()
        Operand(dtype=Int8, literal=True, value=1)
    """

    value: object
    dtype: DType

    def operand(self) -> Operand:
        """Give the literal as the type function reads it.

        Returns:
            The literal at its type.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> TypedLiteralLeaf(None, DType("String")).operand().dtype
            String
        """
        return Operand.typed_literal(self.value, self.dtype)

    def folded(self) -> Operand:
        """Give the literal as the cast pass reads it: the same literal.

        Returns:
            The literal.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> TypedLiteralLeaf(1, DType("Int8")).folded().literal
            True
        """
        return self.operand()

    def optimized(self) -> Operand:
        """Give the input as the optimizer reads it, after the conversion: the same as `folded`.

        Returns:
            The operand.

        Example:
            >>> TypedLiteralLeaf(1, DType("Int8")).optimized().literal
            True
        """
        return self.folded()

    def computed(self) -> Operand:
        """Give the literal as a run computes it: the same literal.

        Returns:
            The literal.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> TypedLiteralLeaf(1, DType("Int8")).computed().value
            1
        """
        return self.operand()

    def data(self) -> Data:
        """Give the data of the literal: one value of its type.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> TypedLiteralLeaf(None, DType("String")).data().nulls
            True
        """
        return Data(self.dtype, nulls=self.value is None)

    def grouped_data(self, frame: GroupBy) -> Data:
        """Give the data of the literal in `agg`: its data in a `select`.

        Args:
            frame: The groups of the `group_by`, which the data does not read.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> TypedLiteralLeaf(1, DType("Int8")).grouped_data(GroupBy((3, 2))).dtype
            Int8
        """
        del frame
        return self.data()

    def height(self) -> Height:
        """Give the projection height of the literal.

        Returns:
            One value.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> TypedLiteralLeaf(1, DType("Int8")).height()
            <Height.SCALAR: 'scalar'>
        """
        return Height.SCALAR

    def elementwise(self) -> bool:
        """Say that a literal of one value works value by value.

        Returns:
            True.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> TypedLiteralLeaf(1, DType("Int8")).elementwise()
            True
        """
        return True

    def selected(self, rows: int) -> InSelect:
        """Give the value of the literal in a `select`.

        Args:
            rows: The height of the frame, which a literal does not read.

        Returns:
            One value.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> TypedLiteralLeaf(1, DType("Int8")).selected(5).length
            1
        """
        del rows
        return InSelect.literal()

    def grouped(self, frame: GroupBy) -> InGroups:
        """Give the value of the literal over the groups of `agg`.

        Args:
            frame: The groups of the `group_by`.

        Returns:
            One value for all the groups.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> TypedLiteralLeaf(1, DType("Int8")).grouped(GroupBy((3, 2))).state
            <AggState.LITERAL_SCALAR: 'literal scalar'>
        """
        return frame.literal(self.dtype)


@dataclass
class NodeState:
    """Stand for a node that is not the root, as an input of the node above it.

    The node above reads the planner type of this node at its first visit of
    the cast pass, and its type after its own visits at its second visit. A
    run reads the output of this node. Each answer comes from the evaluation
    of this node over its own inputs, once, and a refusal raises again at
    each read. A refusal of the cast pass of this node refuses the planner
    type too, because `collect_schema()` runs the cast pass.

    Example:
        >>> from astrolabe._rules.definitions.elementwise import Elementwise
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.node_evaluation import ColumnLeaf
        >>> from astrolabe._rules.options import RowForm
        >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
        >>> node = NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3,)), root=False)
            >>> state = NodeState(node)
        >>> state.operand(), state.data().dtype
        (Operand(dtype=Int8, literal=False, value=None), Int8)
    """

    evaluation: NodeEvaluation
    _cache: dict[str, object] = field(default_factory=dict)

    def operand(self) -> Operand:
        """Give the node as the type function of the node above reads it: its planner type.

        Returns:
            A column of the planner type.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> node = NodeEvaluation(Aggregations.SUM, [leaf], GroupBy((3,)), root=False)
            >>> state = NodeState(node)
            >>> state.operand().dtype
            Int64
        """
        self._read("steps", self.evaluation.inner_steps)
        planner = Operand.column(self._read("planner", self.evaluation.planner))
        return replace(planner, scalar=self.height() is Height.SCALAR)

    def folded(self) -> Operand:
        """Give the node as the second visit of the node above reads it: its type after its visits.

        Returns:
            A column of that type.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> leaves = [ColumnLeaf(DType("Int8"), RowForm.PER_ROW), LiteralLeaf(300)]
            >>> fill = NodeEvaluation(Elementwise.FILL_NULL, leaves, GroupBy((3,)), root=False)
            >>> NodeState(fill).folded().dtype
            Int16
        """
        return Operand.column(self._read("after", self.evaluation.after))

    def optimized(self) -> Operand:
        """Give the node as the optimizer of the node above reads it.

        The optimizer works from the leaves up. Where it puts a literal in
        place of this node, as `True` for `x | True` where `x` is one value,
        the node above reads that literal. Otherwise it reads the type of the
        node after its visits.

        Returns:
            The operand.

        Example:
            >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> first = ColumnLeaf(DType("Boolean"), RowForm.ONE_VALUE)
            >>> leaves = [first, LiteralLeaf(True)]
            >>> node = NodeEvaluation(BinaryOperators.OR, leaves, GroupBy((3,)), root=False)
            >>> NodeState(node).optimized().value
            True
        """
        try:
            literal = self._output().literal
        except (PolarsRefusal, UndecidedError):
            literal = None
        return self.folded() if literal is None else literal

    def computed(self) -> Operand:
        """Give the node as the run computes it: a literal of a fixed value, or a column.

        The value of a binary node of two literals is their fold, and the
        value of a node over a literal comes from its rule (`ValueRule`). A
        column has the type of the node after its visits.

        Returns:
            The literal or the column.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> node = NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3,)), root=False)
            >>> state = NodeState(node)
            >>> state.computed().literal
            False
        """
        fold = self.evaluation.literal_fold()
        if fold is not None:
            return fold
        column = self.folded()
        value = self.evaluation.literal_value(column.dtype)
        return column if value is None else value

    def data(self) -> Data:
        """Give the data of the node in a `select`.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> leaf = ColumnLeaf(DType("Null"), RowForm.PER_ROW)
            >>> node = NodeEvaluation(Aggregations.FIRST, [leaf], GroupBy((3,)), root=False)
            >>> state = NodeState(node)
            >>> state.data().nulls
            True
        """
        return self._output().data

    def grouped_data(self, frame: GroupBy) -> Data:
        """Give the data of the node in `agg`: the type of its value over the groups.

        A kernel can answer in `agg` where it refuses in a `select`, as the
        mean of text gives a null of text for each group.

        Args:
            frame: The groups of the `group_by`, which the evaluation of the node holds.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> leaf = ColumnLeaf(DType("String"), RowForm.PER_ROW)
            >>> node = NodeEvaluation(Aggregations.MEAN, [leaf], GroupBy((3, 2)), root=False)
            >>> NodeState(node).grouped_data(GroupBy((3, 2))).dtype
            String
        """
        dtype = self.grouped(frame).dtype
        if dtype is None:  # pragma: no cover - the evaluation of a node gives the type of its value
            return self.data()
        nulls = self.evaluation.grouped_nulls(dtype)
        return Data(dtype, nulls=nulls, some_null=self.evaluation.grouped_some_null(dtype))

    def height(self) -> Height:
        """Give the projection height of the node.

        Returns:
            The height.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> node = NodeEvaluation(Aggregations.SUM, [leaf], GroupBy((3,)), root=False)
            >>> state = NodeState(node)
            >>> state.height()
            <Height.SCALAR: 'scalar'>
        """
        return self.evaluation.height()

    def elementwise(self) -> bool:
        """Say whether the node and each node below it work value by value.

        Returns:
            Whether they do.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> node = NodeEvaluation(Aggregations.SUM, [leaf], GroupBy((3,)), root=False)
            >>> NodeState(node).elementwise()
            False
        """
        return self.evaluation.elementwise()

    def selected(self, rows: int) -> InSelect:
        """Give the rows of the node in a `select`.

        Args:
            rows: The height of the frame, which the evaluation of the node holds.

        Returns:
            The value.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> node = NodeEvaluation(Aggregations.SUM, [leaf], GroupBy((3,)), root=False)
            >>> state = NodeState(node)
            >>> state.selected(3).length
            1
        """
        del rows
        return self._output().value

    def grouped(self, frame: GroupBy) -> InGroups:
        """Give the value of the node over the groups of `agg`.

        Args:
            frame: The groups of the `group_by`, which the evaluation of the node holds.

        Returns:
            The value.

        Raises:
            TypeError: The read gives another object, which is a bug in this class.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> node = NodeEvaluation(Aggregations.SUM, [leaf], GroupBy((3, 2)), root=False)
            >>> state = NodeState(node)
            >>> state.grouped(GroupBy((3, 2))).dtype
            Int64
        """
        del frame
        value = self._read("grouped", self.evaluation.grouped)
        if not isinstance(value, InGroups):  # pragma: no cover - the read gives an InGroups
            raise TypeError(value)
        return value

    def _output(self) -> NodeOutput:
        """Give the output of the node in a `select`, once.

        Returns:
            The output.

        Raises:
            TypeError: The read gives another object, which is a bug in this class.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> node = NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3,)), root=False)
            >>> state = NodeState(node)
            >>> state._output().value.length
            3
        """
        output = self._read("selected", self.evaluation.selected)
        if not isinstance(output, NodeOutput):  # pragma: no cover - the read gives a NodeOutput
            raise TypeError(output)
        return output

    def _read[T](self, name: str, compute: Callable[[], T]) -> T:
        """Give an answer of the node once, and raise its refusal again at each read.

        Args:
            name: The name of the answer.
            compute: The computation of the answer.

        It raises again the `PolarsRefusal` or the `UndecidedError` of the computation.

        Returns:
            The answer.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> state = NodeState(NodeEvaluation(Elementwise.ABS, [], GroupBy((3,)), root=False))
            >>> state._read("x", lambda: 1), state._read("x", lambda: 2)
            (1, 1)
        """
        if name not in self._cache:
            try:
                self._cache[name] = compute()
            except (PolarsRefusal, UndecidedError) as error:
                self._cache[name] = error
        value = self._cache[name]
        if isinstance(value, PolarsRefusal | UndecidedError):
            raise value
        return value  # type: ignore[return-value]


@dataclass(frozen=True)
class FoldedLeaf:
    """Stand for an input of the root that holds only literals, which the conversion folds.

    The planner reads the input before the fold, as the node that it is. The
    cast pass of the root and the run read the one literal of the fold.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> state = TypedLiteralLeaf(300, DType("Int64"))
        >>> FoldedLeaf(state, LiteralLeaf(300)).folded()
        Operand(dtype=Unknown(int, 300), literal=True, value=300)
    """

    planned: Leaf
    literal: LiteralLeaf

    def operand(self) -> Operand:
        """Give the input before the fold, as the type function reads it.

        Returns:
            The operand of the node.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> FoldedLeaf(TypedLiteralLeaf(1, DType("Int8")), LiteralLeaf(1)).operand().dtype
            Int8
        """
        return self.planned.operand()

    def folded(self) -> Operand:
        """Give the literal of the fold.

        Returns:
            The literal.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> FoldedLeaf(TypedLiteralLeaf(1, DType("Int8")), LiteralLeaf(2)).folded().value
            2
        """
        return self.literal.folded()

    def optimized(self) -> Operand:
        """Give the input as the optimizer reads it, after the conversion: the same as `folded`.

        Returns:
            The operand.

        Example:
            >>> FoldedLeaf(TypedLiteralLeaf(1, DType("Int8")), LiteralLeaf(2)).optimized().literal
            True
        """
        return self.folded()

    def computed(self) -> Operand:
        """Give the literal of the fold, as a run computes it.

        Returns:
            The literal.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> FoldedLeaf(TypedLiteralLeaf(1, DType("Int8")), LiteralLeaf(2)).computed().value
            2
        """
        return self.literal.computed()

    def data(self) -> Data:
        """Give the data of the literal of the fold.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> FoldedLeaf(TypedLiteralLeaf(1, DType("Int8")), LiteralLeaf(2)).data().dtype
            Int32
        """
        return self.literal.data()

    def grouped_data(self, frame: GroupBy) -> Data:
        """Give the data of the literal of the fold in `agg`: its data in a `select`.

        Args:
            frame: The groups of the `group_by`, which the data does not read.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> leaf = FoldedLeaf(TypedLiteralLeaf(1, DType("Int8")), LiteralLeaf(2))
            >>> leaf.grouped_data(GroupBy((3, 2))).dtype
            Int32
        """
        del frame
        return self.data()

    def height(self) -> Height:
        """Give the height of the input: one value.

        Returns:
            The height.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> FoldedLeaf(TypedLiteralLeaf(1, DType("Int8")), LiteralLeaf(2)).height()
            <Height.SCALAR: 'scalar'>
        """
        return self.literal.height()

    def elementwise(self) -> bool:
        """Say whether the literal works value by value: a literal of one value does.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> FoldedLeaf(TypedLiteralLeaf(1, DType("Int8")), LiteralLeaf(2)).elementwise()
            True
        """
        return self.literal.elementwise()

    def selected(self, rows: int) -> InSelect:
        """Give the value of the literal of the fold in a `select`.

        Args:
            rows: The height of the frame.

        Returns:
            One value.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> FoldedLeaf(TypedLiteralLeaf(1, DType("Int8")), LiteralLeaf(2)).selected(3).length
            1
        """
        return self.literal.selected(rows)

    def grouped(self, frame: GroupBy) -> InGroups:
        """Give the value of the literal of the fold over the groups of `agg`.

        Args:
            frame: The groups of the `group_by`.

        Returns:
            One value for all the groups.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> leaf = FoldedLeaf(TypedLiteralLeaf(1, DType("Int8")), LiteralLeaf(2))
            >>> leaf.grouped(GroupBy((3, 2))).state
            <AggState.LITERAL_SCALAR: 'literal scalar'>
        """
        return self.literal.grouped(frame)


class EvalVariant:
    """Give the types of an `eval` node from its variant: the element that it reads, and its output.

    A list variant reads the inner type of a list, and an array variant the
    inner type of an array. `cumulative_eval` reads the values themselves.
    `List` gives a list of the result, an `Array` that is not `as_list` gives
    an array of the same width, and an aggregation variant gives the result
    itself where it is one value for each list. `cumulative_eval` gives the
    result itself, and polars refuses a body that is not one value.

    Example:
        >>> lists = DType("List", (DType("Int8"),))
        >>> EvalVariant("ListAgg").output(lists, DType("Int64"), scalar=True)
        Int64
    """

    def __init__(self, variant: object) -> None:
        """Keep the variant, as the tree writes it.

        Args:
            variant: The variant.

        Example:
            >>> EvalVariant({"Array": {"as_list": True}})._name
            'Array'
        """
        self._variant = variant
        self._name = variant if isinstance(variant, str) else next(iter(variant), "")  # type: ignore[call-overload]

    def elementwise(self) -> bool:
        """Say whether the `eval` works value by value: every variant but `cumulative_eval` does.

        Returns:
            Whether it does.

        Example:
            >>> EvalVariant({"Cumulative": {"min_samples": 0}}).elementwise()
            False
        """
        return self._name != "Cumulative"

    def element(self, dtype: DType) -> DType:
        """Give the type of the element that the evaluation reads.

        Args:
            dtype: The type of the input of the `eval`.

        Returns:
            The type of the element.

        Raises:
            PolarsRefusal: The input is not of the kind that the variant reads.

        Example:
            >>> EvalVariant("List").element(DType("List", (DType("Date"),)))
            Date
        """
        if self._name == "Cumulative":
            return dtype
        kind = "List" if self._name in {"List", "ListAgg"} else "Array"
        if dtype.name != kind:
            message = f"expected {kind} datatype for this operation, got: {dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return dtype.inner()

    def shape_first(self) -> bool:
        """Say whether polars checks the shape of the evaluation before its types.

        Polars refuses a `cumulative_eval` whose body gives more than one value
        from the shape of the body, before it types the body.

        Returns:
            Whether it does: for `cumulative_eval`.

        Example:
            >>> EvalVariant("Cumulative").shape_first(), EvalVariant("List").shape_first()
            (True, False)
        """
        return self._name == "Cumulative"

    def check_shape(self, *, scalar: bool) -> None:
        """Refuse an evaluation whose shape the variant does not take.

        Args:
            scalar: Whether the evaluation gives one value for each list or window.

        Raises:
            PolarsRefusal: The variant is `cumulative_eval`, and the evaluation
                gives more than one value.

        Example:
            >>> EvalVariant("List").check_shape(scalar=False)
            >>> EvalVariant("Cumulative").check_shape(scalar=False)
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `cumulative_eval` ...
        """
        if self.shape_first() and not scalar:
            message = "`cumulative_eval` is not allowed with non-scalar output"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def output(self, dtype: DType, element: DType, *, scalar: bool) -> DType:
        """Give the type of the output of the `eval`, after the check of its shape (`check_shape`).

        Args:
            dtype: The type of the input of the `eval`.
            element: The type of the result of the evaluation.
            scalar: Whether the evaluation gives one value for each list.

        Returns:
            The type.

        Example:
            >>> EvalVariant({"Array": {"as_list": False}}).output(
            ...     DType("Array", (DType("Int8"), 2)), DType("Int16"), scalar=False
            ... )
            Array(Int16, shape=(2,))
            >>> EvalVariant("Cumulative").output(DType("Int8"), DType("Int8"), scalar=False)
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `cumulative_eval` ...
        """
        self.check_shape(scalar=scalar)
        if self._name == "Cumulative":
            return element
        if self._name in {"ListAgg", "ArrayAgg"} and scalar:
            return element
        as_list = isinstance(self._variant, dict) and self._variant.get("Array", {}).get("as_list")
        if self._name == "Array" and not as_list:
            return DType("Array", (element, dtype.args[1]))
        return DType("List", (element,))


@dataclass
class EvalState:
    """Stand for an `eval` node: its input, and its evaluation over the elements of each row.

    The evaluation runs over the values of each list, as a group. So its
    result for each list is one value where the evaluation in `agg` is not
    wrapped in a list, and a list otherwise. The `eval` gives one row for
    each row of its input.

    The number of values in each list is a fact of the data. Where the run
    refuses lists of the lengths of `frame` and runs over lists of one value,
    as `item()` does, the run answer is the one over lists of one value
    (`ONE_VALUE`): the data can hold such lists.

    Example:
        >>> from astrolabe._rules.node_evaluation import ColumnLeaf
        >>> from astrolabe._rules.options import RowForm
        >>> from astrolabe._rules.tree import ExpressionTree
        >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
        >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
        >>> state = EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,)))
        >>> state.operand().dtype
        Int64
    """

    ONE_VALUE: ClassVar[GroupBy] = GroupBy((1, 1))

    input: Leaf
    variant: EvalVariant
    evaluation: TreeNode
    frame: GroupBy

    def _answers(self, dtype: DType) -> CellAnswers:
        """Give the answers of the evaluation at one type of the input.

        Args:
            dtype: The type of the input.

        Returns:
            The answers, with the element as a column. The run in `agg` is
            the run over lists of one value where only that run answers.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> state = EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,)))
            >>> state._answers(lists.dtype).planner_agg
            Int64
            >>> body = {"input": "Element", "allow_empty": False}
            >>> item = ExpressionTree().read({"Agg": {"Item": body}})
            >>> state = EvalState(lists, EvalVariant("ListAgg"), item, GroupBy((3, 2)))
            >>> state._answers(lists.dtype).run_agg
            Int8
        """
        element = ColumnLeaf(self.variant.element(dtype), RowForm.PER_ROW)
        answers = TreeEvaluation({}, self.frame, element=element).answers(self.evaluation)
        if isinstance(answers.run_agg, DType) or not isinstance(answers.planner_agg, DType):
            return answers
        single = TreeEvaluation({}, self.ONE_VALUE, element=element).answers(self.evaluation)
        if not isinstance(single.run_agg, DType):
            return answers
        return replace(answers, run_agg=single.run_agg)

    def _shaped_answers(self, dtype: DType) -> CellAnswers:
        """Give the answers of the evaluation, after a check of its shape that polars runs first.

        Where the body refuses its element, the rules read the shape of the
        body at a `Null` element, which most nodes accept, and the variant
        refuses a shape that it does not take (`check_shape`).

        Args:
            dtype: The type of the input.

        Returns:
            The answers.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> text = ColumnLeaf(DType("String"), RowForm.PER_ROW)
            >>> two = {"Literal": {"Dyn": {"Int": 2}}}
            >>> body = {"left": "Element", "op": "Multiply", "right": two}
            >>> inner = ExpressionTree().read({"BinaryExpr": body})
            >>> state = EvalState(text, EvalVariant("Cumulative"), inner, GroupBy((3,)))
            >>> state._shaped_answers(DType("String"))
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `cumulative_eval` ...
        """
        answers = self._answers(dtype)
        if not self.variant.shape_first() or self._typed(answers):
            return answers
        probe = self._answers(DType("Null"))
        if self._typed(probe):
            self.variant.check_shape(scalar=probe.planner_select == probe.planner_agg)
        return answers

    def _typed(self, answers: CellAnswers) -> bool:
        """Say whether the evaluation gives a type in a `select` and in `agg`, at its planner.

        Args:
            answers: The answers of the evaluation.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> state = EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,)))
            >>> state._typed(state._answers(lists.dtype))
            True
        """
        planner = answers.planner_select
        return isinstance(planner, DType) and isinstance(answers.planner_agg, DType)

    def operand(self) -> Operand:
        """Give the `eval` at its planner type.

        It raises the refusal of the evaluation at its planner type.

        Returns:
            A column of the planner type.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> body = {"left": "Element", "op": "Plus", "right": "Element"}
            >>> inner = ExpressionTree().read({"BinaryExpr": body})
            >>> EvalState(lists, EvalVariant("List"), inner, GroupBy((3,))).operand().dtype
            List(Int8)
        """
        dtype = self.input.operand().dtype
        answers = self._shaped_answers(dtype)
        planner, wrapped = self.known(answers.planner_select), self.known(answers.planner_agg)
        scalar = planner == wrapped
        return Operand.column(self.variant.output(dtype, planner, scalar=scalar))

    def folded(self) -> Operand:
        """Give the `eval` as the second visit of the node above reads it: its planner type.

        Returns:
            The operand.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,))).folded().literal
            False
        """
        return self.operand()

    def optimized(self) -> Operand:
        """Give the input as the optimizer reads it, after the conversion: the same as `folded`.

        Returns:
            The operand.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,))).optimized().literal
            False
        """
        return self.folded()

    def computed(self) -> Operand:
        """Give the `eval` as a run computes it: a column.

        Returns:
            The operand.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,))).computed().literal
            False
        """
        return self.operand()

    def data(self) -> Data:
        """Give the data of the `eval`: the result of the evaluation at the data of its input.

        It raises the refusal of the evaluation at its run.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,))).data().dtype
            Int64
        """
        dtype = self.input.data().dtype
        answers = self._shaped_answers(dtype)
        planner, wrapped = self.known(answers.planner_select), self.known(answers.planner_agg)
        scalar = planner == wrapped
        run = self.known(answers.run_agg)
        element = run if scalar or run.name != "List" else run.inner()
        return Data(self.variant.output(dtype, element, scalar=scalar))

    def grouped_data(self, frame: GroupBy) -> Data:
        """Give the data of the `eval` in `agg`: its data in a `select`.

        Args:
            frame: The groups of the `group_by`, which the data does not read.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> state = EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,)))
            >>> state.grouped_data(GroupBy((3,))).dtype
            Int64
        """
        del frame
        return self.data()

    def height(self) -> Height:
        """Give the height of the `eval`: the height of its input.

        Returns:
            The height.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,))).height()
            <Height.COLUMN: 'column'>
        """
        return self.input.height()

    def elementwise(self) -> bool:
        """Say whether the `eval` works value by value: over each list, not over growing windows.

        Returns:
            Whether the variant is elementwise and the input is.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,))).elementwise()
            True
        """
        return self.variant.elementwise() and self.input.elementwise()

    def selected(self, rows: int) -> InSelect:
        """Give the rows of the `eval` in a `select`: the rows of its input.

        Args:
            rows: The height of the frame.

        Returns:
            The value.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,))).selected(3).length
            3
        """
        self.data()
        return self.input.selected(rows)

    def grouped(self, frame: GroupBy) -> InGroups:
        """Give the value of the `eval` over the groups of `agg`: the value of its input.

        `cumulative_eval` runs over the growing windows of the values of each
        group, so it gives one list for each group, also over one value for
        each group.

        Args:
            frame: The groups of the `group_by`.

        Returns:
            The value, with the type of the `eval`.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> state = EvalState(lists, EvalVariant("ListAgg"), inner, GroupBy((3,)))
            >>> state.grouped(GroupBy((3,))).dtype
            Int64
        """
        value = replace(self.input.grouped(frame), dtype=self.data().dtype)
        if self.variant.elementwise():
            return value
        sizes = value.values()
        listed = replace(value, state=AggState.AGGREGATED_LIST, groups=sizes, rows=sum(sizes))
        return replace(listed, rebuilt=True)

    @staticmethod
    def known(answer: object) -> DType:
        """Give a type of the evaluation, and raise the refusal of an answer that is not a type.

        Args:
            answer: The answer.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The answer is a refusal.
            UndecidedError: The answer is `Unchecked`.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> lists = ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)
            >>> inner = ExpressionTree().read({"Agg": {"Sum": "Element"}})
            >>> EvalState.known(DType("Int8"))
            Int8
        """
        if isinstance(answer, DType):
            return answer
        if isinstance(answer, Refused):
            raise PolarsRefusal(answer.error or answer.stop, answer.message)
        raise UndecidedError(Undecided.NOT_TRANSLATED, str(answer))


@dataclass
class StructEvalState:
    """Stand for `struct.with_fields`: a struct whose fields its evaluations replace or add.

    Each evaluation reads the fields of the struct as columns
    (`pl.field("a")`), and the columns of the frame by their names. It gives
    one field, named by its output name. A field of that name replaces the
    field of the struct, and any other name adds a field at the end. The node
    gives one row for each row of its input.

    Example:
        >>> import polars as pl
        >>> from astrolabe._rules.options import RowForm
        >>> from astrolabe._rules.tree import ExpressionTree
        >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
        >>> tree = ExpressionTree().of(pl.col("s").struct.with_fields(pl.field("a").cast(pl.Int64)))
        >>> StructEvalState(struct, tree.evaluation, GroupBy((3,))).operand().dtype
        Struct({'a': Int64})
    """

    input: Leaf
    evaluation: tuple[TreeNode, ...]
    frame: GroupBy
    columns: Mapping[str, Leaf] = field(default_factory=dict)

    def _fields(self, dtype: DType, *, run: bool) -> DType:
        """Give the struct after the evaluations, at one type of the input.

        Args:
            dtype: The type of the input.
            run: Whether to read the run type of each evaluation, or its planner type.

        Returns:
            The struct type.

        Raises:
            PolarsRefusal: The input is not a struct.

        Example:
            >>> from astrolabe._rules.options import RowForm
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> StructEvalState(struct, (), GroupBy((3,)))._fields(struct.dtype, run=False)
            Struct({'a': Int8})
        """
        if dtype.name != "Struct":
            message = f"`with_fields` operation not supported for dtype `{dtype!r}`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        fields = dict(dtype.fields())
        columns = {
            name: ColumnLeaf(field_type, RowForm.PER_ROW) for name, field_type in fields.items()
        }
        evaluation = TreeEvaluation(self.columns, self.frame, fields=columns)
        for tree in self.evaluation:
            answers = evaluation.answers(tree)
            answer = answers.run_select if run else answers.planner_select
            fields[OutputName().of(tree)] = EvalState.known(answer)
        return DType("Struct", (tuple(fields.items()),))

    def operand(self) -> Operand:
        """Give the node at its planner type.

        Returns:
            A column of the planner type.

        Example:
            >>> from astrolabe._rules.options import RowForm
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> StructEvalState(struct, (), GroupBy((3,))).operand().literal
            False
        """
        return Operand.column(self._fields(self.input.operand().dtype, run=False))

    def folded(self) -> Operand:
        """Give the node as the second visit of the node above reads it.

        Returns:
            The operand.

        Example:
            >>> from astrolabe._rules.options import RowForm
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> StructEvalState(struct, (), GroupBy((3,))).folded().dtype
            Struct({'a': Int8})
        """
        return self.operand()

    def optimized(self) -> Operand:
        """Give the input as the optimizer reads it, after the conversion: the same as `folded`.

        Returns:
            The operand.

        Example:
            >>> from astrolabe._rules.options import RowForm
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> StructEvalState(struct, (), GroupBy((3,))).optimized().literal
            False
        """
        return self.folded()

    def computed(self) -> Operand:
        """Give the node as a run computes it.

        Returns:
            The operand.

        Example:
            >>> from astrolabe._rules.options import RowForm
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> StructEvalState(struct, (), GroupBy((3,))).computed().literal
            False
        """
        return self.operand()

    def data(self) -> Data:
        """Give the data of the node: the struct after the runs of the evaluations.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.options import RowForm
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> StructEvalState(struct, (), GroupBy((3,))).data().dtype
            Struct({'a': Int8})
        """
        return Data(self._fields(self.input.data().dtype, run=True))

    def grouped_data(self, frame: GroupBy) -> Data:
        """Give the data of the node in `agg`: its data in a `select`.

        Args:
            frame: The groups of the `group_by`, which the data does not read.

        Returns:
            The data.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> tree = ExpressionTree().of(pl.col("s").struct.with_fields(pl.field("a")))
            >>> state = StructEvalState(struct, tree.evaluation, GroupBy((3,)))
            >>> state.grouped_data(GroupBy((3,))).dtype
            Struct({'a': Int8})
        """
        del frame
        return self.data()

    def height(self) -> Height:
        """Give the height of the node: the height of its input.

        Returns:
            The height.

        Example:
            >>> from astrolabe._rules.options import RowForm
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> StructEvalState(struct, (), GroupBy((3,))).height()
            <Height.COLUMN: 'column'>
        """
        return self.input.height()

    def elementwise(self) -> bool:
        """Say whether the input and each evaluation of the fields work value by value.

        Returns:
            Whether they do. A field reads the value of its row.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> summed = pl.col("s").struct.with_fields(pl.field("a").sum())
            >>> tree = ExpressionTree().of(summed)
            >>> StructEvalState(struct, tree.evaluation, GroupBy((3,))).elementwise()
            False
            >>> number = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> StructEvalState(number, tree.evaluation, GroupBy((3,))).elementwise()
            False
        """
        dtype = self.input.operand().dtype
        if not self.input.elementwise() or dtype.name != "Struct":
            return False
        fields = {name: ColumnLeaf(kind, RowForm.PER_ROW) for name, kind in dtype.fields()}
        evaluation = TreeEvaluation(self.columns, self.frame, fields=fields)
        return all(evaluation.elementwise(tree) for tree in self.evaluation)

    def selected(self, rows: int) -> InSelect:
        """Give the rows of the node in a `select`: the rows of its input.

        Args:
            rows: The height of the frame.

        Returns:
            The value.

        Example:
            >>> from astrolabe._rules.options import RowForm
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> StructEvalState(struct, (), GroupBy((3,))).selected(3).length
            3
        """
        self.data()
        return self.input.selected(rows)

    def grouped(self, frame: GroupBy) -> InGroups:
        """Give the value of the node over the groups of `agg`: the value of its input.

        Args:
            frame: The groups of the `group_by`.

        Returns:
            The value, with the type of the node.

        Example:
            >>> from astrolabe._rules.options import RowForm
            >>> struct = ColumnLeaf(DType("Struct", ((("a", DType("Int8")),),)), RowForm.PER_ROW)
            >>> StructEvalState(struct, (), GroupBy((3,))).grouped(GroupBy((3,))).dtype
            Struct({'a': Int8})
        """
        return replace(self.input.grouped(frame), dtype=self.data().dtype)


class OutputName:
    """Give the output name of a tree, as polars names the column that it gives.

    Polars gives the name of a tree (`meta.output_name()`), but it does not
    name a tree that reads a field of a struct. Inside `struct.with_fields`,
    a field gives its name, an alias its alias, and any other node the name of
    its first input.

    Example:
        >>> from astrolabe._rules.tree import ExpressionTree
        >>> OutputName().of(ExpressionTree().read({"Column": "a"}))
        'a'
    """

    def of(self, tree: TreeNode) -> str:
        """Give the output name.

        Args:
            tree: The tree.

        Returns:
            The name.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> OutputName().of(ExpressionTree().of(pl.field("b") * 2))
            'b'
        """
        if not self._reads_field(tree):
            return OptionReader().first_name([tree.serialized])
        body = (
            tree.serialized.get(tree.identity.shape) if isinstance(tree.serialized, dict) else None
        )
        if tree.identity.shape == "Field" and isinstance(body, list):
            return str(body[0])
        if tree.identity.shape == "Alias" and isinstance(body, list):
            return str(body[1])
        return self.of(tree.inputs[0])

    def _reads_field(self, tree: TreeNode) -> bool:
        """Say whether a tree reads a field of a struct.

        Args:
            tree: The tree.

        Returns:
            Whether a leaf of the tree is a field.

        Example:
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> OutputName()._reads_field(ExpressionTree().read({"Column": "a"}))
            False
        """
        if tree.identity.shape == "Field":
            return True
        return any(self._reads_field(part) for part in tree.inputs)


@dataclass(frozen=True)
class SeriesLeaf:
    """Stand for a literal of a series: a column of its own length whose values the code holds.

    The series is a column for every pass. A kernel that reads the values of
    an input, as `replace` reads its old and new values after the `implode`
    of the cast pass, reads them from the computed input. The series holds
    its own number of rows in a `select`, whatever the height of the frame.
    In `agg`, polars gives each group a list of all the values, on groups of
    its own.

    Example:
        >>> column = ColumnLeaf(DType("Int64"), RowForm.OTHER_LENGTH)
        >>> SeriesLeaf(column, (1, 2)).computed().value
        (1, 2)
    """

    column: ColumnLeaf
    values: tuple[object, ...]

    def operand(self) -> Operand:
        """Give the series as the type function reads it: a column.

        Returns:
            The operand.

        Example:
            >>> column = ColumnLeaf(DType("Int64"), RowForm.OTHER_LENGTH)
            >>> SeriesLeaf(column, (1, 2)).operand().literal
            False
        """
        return self.column.operand()

    def folded(self) -> Operand:
        """Give the series as the cast pass reads it: a column.

        Returns:
            The operand.

        Example:
            >>> column = ColumnLeaf(DType("Int64"), RowForm.OTHER_LENGTH)
            >>> SeriesLeaf(column, (1, 2)).folded().dtype
            Int64
        """
        return self.column.folded()

    def optimized(self) -> Operand:
        """Give the series as the optimizer reads it: a column.

        Returns:
            The operand.

        Example:
            >>> column = ColumnLeaf(DType("Int64"), RowForm.OTHER_LENGTH)
            >>> SeriesLeaf(column, (1, 2)).optimized().literal
            False
        """
        return self.column.optimized()

    def computed(self) -> Operand:
        """Give the series as a run computes it: a column that holds its values.

        Returns:
            The operand.

        Example:
            >>> column = ColumnLeaf(DType("Int64"), RowForm.ONE_ROW)
            >>> SeriesLeaf(column, (1,)).computed()
            Operand(dtype=Int64, literal=False, value=(1,))
        """
        return Operand(self.column.dtype, value=self.values)

    def data(self) -> Data:
        """Give the data of the series in a `select`.

        Returns:
            The data.

        Example:
            >>> column = ColumnLeaf(DType("Int64"), RowForm.ONE_ROW, nulls=True)
            >>> SeriesLeaf(column, (None,)).data().nulls
            True
        """
        return self.column.data()

    def grouped_data(self, frame: GroupBy) -> Data:
        """Give the data of the series in `agg`.

        Args:
            frame: The groups.

        Returns:
            The data.

        Example:
            >>> column = ColumnLeaf(DType("Int64"), RowForm.ONE_ROW)
            >>> SeriesLeaf(column, (1,)).grouped_data(GroupBy((3,))).dtype
            Int64
        """
        return self.column.grouped_data(frame)

    def height(self) -> Height:
        """Give the projection height of the series.

        Returns:
            The height.

        Example:
            >>> column = ColumnLeaf(DType("Int64"), RowForm.OTHER_LENGTH)
            >>> SeriesLeaf(column, (1, 2)).height()
            <Height.UNKNOWN: 'unknown'>
        """
        return self.column.height()

    def elementwise(self) -> bool:
        """Say that a literal of several values does not work value by value.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> column = ColumnLeaf(DType("Int64"), RowForm.OTHER_LENGTH)
            >>> SeriesLeaf(column, (1, 2)).elementwise()
            False
        """
        return False

    def selected(self, rows: int) -> InSelect:
        """Give the value of the series in a `select` of a frame: its own number of rows.

        Args:
            rows: The rows of the frame, which the series does not read.

        Returns:
            The value.

        Example:
            >>> column = ColumnLeaf(DType("Int64"), RowForm.OTHER_LENGTH)
            >>> SeriesLeaf(column, (1, 2, 3)).selected(5).length
            3
        """
        return replace(self.column.selected(rows), length=len(self.values))

    def grouped(self, frame: GroupBy) -> InGroups:
        """Give the value of the series over the groups of `agg`: every value in each group.

        Args:
            frame: The groups.

        Returns:
            The value.

        Example:
            >>> column = ColumnLeaf(DType("Int64"), RowForm.OTHER_LENGTH)
            >>> SeriesLeaf(column, (1, 2, 3)).grouped(GroupBy((3, 2))).groups
            (3, 3)
        """
        sizes = (len(self.values),) * len(frame.sizes)
        value = self.column.grouped(frame)
        listed = AggState.AGGREGATED_LIST
        return replace(value, state=listed, groups=sizes, rows=sum(sizes), original=False)


class LiteralParse:
    """Read a literal leaf of the tree of polars into a leaf of the rules.

    A literal of Python with no type in the code is dynamic: an `int` or a
    `float`. Polars writes a text, a `bool` and a null as a scalar, with no
    type of its own. A scalar of another type has the type that the code
    gives. A literal of a nested value, of bytes or of a category has the
    type of its value, which polars writes with the value. A literal of a
    series is a column of its own length whose values the code holds
    (`SeriesLeaf`), as the holidays of `pl.business_day_count` or the old
    values of `replace`.

    Example:
        >>> LiteralParse().of({"Dyn": {"Int": 1}})
        LiteralLeaf(value=1, operator=None, neutral=None)
        >>> LiteralParse().of({"Scalar": {"Int8": 1}})
        TypedLiteralLeaf(value=1, dtype=Int8)
    """

    UNTYPED: ClassVar[frozenset[str]] = frozenset({"String", "Boolean"})
    NESTED: ClassVar[frozenset[str]] = frozenset({"Binary", "List", "Array", "Enum", "Struct"})

    def of(self, body: object) -> Leaf:
        """Give the leaf of a serialized literal.

        A literal that holds a value that this reader does not read raises
        `UndecidedError`.

        Args:
            body: The body of the `Literal` node.

        Returns:
            The leaf.

        Example:
            >>> LiteralParse().of({"Scalar": {"Null": "String"}})
            TypedLiteralLeaf(value=None, dtype=String)
        """
        kind, value = self._entry(body)
        if kind == "Dyn":
            _, number = self._entry(value)
            return LiteralLeaf(number)  # type: ignore[arg-type]
        if kind == "Series":
            return self._series(value)
        return self._scalar(value)

    def _scalar(self, value: object) -> Leaf:
        """Give the leaf of a literal of one value, which polars writes as a scalar.

        Args:
            value: The scalar, as a map of the name of its type to its value.

        Returns:
            The leaf.

        Raises:
            UndecidedError: The scalar holds a value that this reader does not read.

        Example:
            >>> LiteralParse()._scalar({"UInt16": 3})
            TypedLiteralLeaf(value=3, dtype=UInt16)
        """
        name, scalar = self._entry(value)
        if name in self.UNTYPED:
            return LiteralLeaf(scalar)  # type: ignore[arg-type]
        if name == "Null":
            return self._null(scalar)
        if name in self.NESTED:
            return self._nested(name, scalar)
        if isinstance(scalar, list):
            return self._parametric(name, scalar)
        if isinstance(scalar, int | float | str | bool):
            return TypedLiteralLeaf(scalar, self._dtype(name))
        raise UndecidedError(Undecided.NOT_TRANSLATED, f"a literal of {name}")

    def _nested(self, name: str, scalar: object) -> Leaf:
        """Give a literal of a nested value, of text bytes or of a category.

        Polars writes the values of a `List` and an `Array` as an Arrow stream
        of one series, the categories of an `Enum` too, and each field of a
        `Struct` as a literal of its own.

        Args:
            name: The name of the type.
            scalar: The value, as polars serializes it.

        Returns:
            The leaf, with the type that polars gives the value.

        Raises:
            UndecidedError: The value is not of the form that polars writes.

        Example:
            >>> LiteralParse()._nested("Binary", [120, 121])
            TypedLiteralLeaf(value=b'xy', dtype=Binary)
            >>> LiteralParse()._nested("Struct", [["a", {"Int8": 1}]]).dtype
            Struct({'a': Int8})
            >>> try:
            ...     LiteralParse()._nested("Map", 1)
            ... except UndecidedError as error:
            ...     print(error)
            not translated: a literal of Map
        """
        if name == "Binary" and isinstance(scalar, list):
            return TypedLiteralLeaf(bytes(scalar), DType("Binary"))
        if name in {"List", "Array"} and isinstance(scalar, list):
            return self._listed(name, scalar)
        if name == "Enum" and isinstance(scalar, dict):
            return self._category(scalar)
        if name == "Struct" and isinstance(scalar, list):
            return self._struct(scalar)
        # Polars writes each literal in one of the forms above.
        raise UndecidedError(Undecided.NOT_TRANSLATED, f"a literal of {name}")  # pragma: no cover

    def _listed(self, name: str, scalar: list[object]) -> Leaf:
        """Give a literal of a `List` or an `Array`, whose values polars writes as an Arrow stream.

        Args:
            name: `List` or `Array`.
            scalar: The stream of a `List`, or the stream and the size of an `Array`.

        Returns:
            The leaf.

        Example:
            >>> import json
            >>> import polars as pl
            >>> tree = json.loads(pl.lit([1, 2]).meta.serialize(format="json"))
            >>> LiteralParse()._listed("List", tree["Literal"]["Scalar"]["List"]).dtype
            List(Int64)
        """
        if name == "List":
            inner = SerializedDType().stream(scalar)
            return TypedLiteralLeaf(inner.to_list(), DType("List", (DType.of(inner.dtype),)))
        inner = SerializedDType().stream(scalar[0])
        dtype = DType("Array", (DType.of(inner.dtype), scalar[1]))
        return TypedLiteralLeaf(inner.to_list(), dtype)

    def _series(self, stream: object) -> Leaf:
        """Give a literal of a series, whose values polars writes as an Arrow stream.

        Polars reads a series of one value as a column of one row, which a
        node can spread over the rows of the frame, and a longer series as a
        column of its own length.

        Args:
            stream: The stream.

        Returns:
            The leaf.

        Example:
            >>> import json
            >>> import polars as pl
            >>> series = pl.lit(pl.Series([None], dtype=pl.Int8))
            >>> tree = json.loads(series.meta.serialize(format="json"))
            >>> LiteralParse()._series(tree["Literal"]["Series"]).column
            ColumnLeaf(dtype=Int8, form=<RowForm.ONE_ROW: 'one row'>, nulls=True, planned=None)
        """
        series = SerializedDType().stream(stream)
        form = RowForm.ONE_ROW if series.len() == 1 else RowForm.OTHER_LENGTH
        nulls = series.len() > 0 and series.null_count() == series.len()
        column = ColumnLeaf(DType.of(series.dtype), form, nulls=nulls)
        return SeriesLeaf(column, tuple(series.to_list()))

    def _category(self, scalar: Mapping[str, Any]) -> Leaf:
        """Give a literal of an `Enum`: the place of its value among the categories.

        Args:
            scalar: The place of the value, and the categories as an Arrow stream.

        Returns:
            The leaf.

        Example:
            >>> import json
            >>> import polars as pl
            >>> literal = pl.lit("b", dtype=pl.Enum(["a", "b"]))
            >>> tree = json.loads(literal.meta.serialize(format="json"))
            >>> LiteralParse()._category(tree["Literal"]["Scalar"]["Enum"])
            TypedLiteralLeaf(value='b', dtype=Enum(categories=['a', 'b']))
        """
        values = SerializedDType().stream(scalar["categories"]).to_list()
        categories = tuple(str(value) for value in values)
        return TypedLiteralLeaf(categories[scalar["value"]], DType("Enum", (categories,)))

    def _struct(self, fields: list[object]) -> Leaf:
        """Give a literal of a struct, whose fields polars writes as literals of their own.

        Args:
            fields: A pair of a name and a serialized scalar for each field.

        Returns:
            The leaf.

        Raises:
            UndecidedError: A field is not of the form that polars writes.

        Example:
            >>> LiteralParse()._struct([["a", {"Int8": 1}], ["b", {"String": "x"}]]).dtype
            Struct({'a': Int8, 'b': String})
            >>> try:
            ...     LiteralParse()._struct([["a"]])
            ... except UndecidedError as error:
            ...     print(error)
            not translated: a literal of Struct
        """
        parts: list[tuple[str, DType]] = []
        values: dict[str, object] = {}
        for pair in fields:
            if (  # pragma: no cover -- polars writes this form
                not isinstance(pair, list) or len(pair) != len(("name", "value"))
            ):
                raise UndecidedError(Undecided.NOT_TRANSLATED, "a literal of Struct")
            leaf = self.of({"Scalar": pair[1]})
            parts.append((str(pair[0]), leaf.data().dtype))
            values[str(pair[0])] = getattr(leaf, "value", None)
        return TypedLiteralLeaf(values, DType("Struct", (tuple(parts),)))

    def _parametric(self, name: str, parts: list[object]) -> Leaf:
        """Give a literal of a type with parameters, which polars writes with its parameters.

        A moment or a duration keeps its physical value, a count of its time
        unit since 1970 or in all. A `Decimal` holds its digits, which the
        scale places.

        Args:
            name: The name of the type.
            parts: The value, then the parameters.

        Returns:
            The leaf.

        Raises:
            UndecidedError: The type is not one that this reader reads.

        Example:
            >>> LiteralParse()._parametric("Datetime", [1, "Milliseconds", {"inner": "UTC"}])
            TypedLiteralLeaf(value=1, dtype=Datetime(time_unit='ms', time_zone='UTC'))
            >>> LiteralParse()._parametric("Decimal", [15, 38, 1]).value
            Decimal('1.5')
            >>> try:
            ...     LiteralParse()._parametric("Time", [1])
            ... except UndecidedError as error:
            ...     print(error)
            not translated: a literal of Time
        """
        reader = SerializedDType()
        if name == "Datetime" and len(parts) == len(("value", "unit", "zone")):
            value, unit, zone = parts
            text = zone.get("inner") if isinstance(zone, dict) else zone
            return TypedLiteralLeaf(value, DType("Datetime", (reader.unit(unit), text)))
        if name == "Duration" and len(parts) == len(("value", "unit")):
            return TypedLiteralLeaf(parts[0], DType("Duration", (reader.unit(parts[1]),)))
        if name == "Decimal":
            return self._decimal(parts)
        # Polars writes each literal in one of the forms above.
        raise UndecidedError(Undecided.NOT_TRANSLATED, f"a literal of {name}")  # pragma: no cover

    def _decimal(self, parts: list[object]) -> Leaf:
        """Give a literal of a `Decimal`, from its digits, its precision and its scale.

        Args:
            parts: The digits, the precision and the scale.

        Returns:
            The leaf.

        Raises:
            UndecidedError: The parts are not three, or the digits or the scale
                are not integers.

        Example:
            >>> LiteralParse()._decimal([15, 38, 1])
            TypedLiteralLeaf(value=Decimal('1.5'), dtype=Decimal(precision=38, scale=1))
            >>> LiteralParse()._decimal([15])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not translated: a literal of Decimal
            >>> try:
            ...     LiteralParse()._decimal(["15", 38, 1])
            ... except UndecidedError as error:
            ...     print(error)
            not translated: a literal of Decimal
        """
        if len(parts) == len(("digits", "precision", "scale")):
            digits, precision, scale = parts
            if isinstance(digits, int) and isinstance(scale, int):
                value = decimal.Decimal(digits).scaleb(-scale)
                return TypedLiteralLeaf(value, DType("Decimal", (precision, scale)))
        # Polars writes each literal in one of the forms above.
        raise UndecidedError(Undecided.NOT_TRANSLATED, "a literal of Decimal")  # pragma: no cover

    def _null(self, name: object) -> Leaf:
        """Give a null literal: of no type, or of the type that the code gives.

        Polars writes the type of a null as it writes any data type, so a
        type with parameters, as `List(Int8)`, is a map of its name to them.

        Args:
            name: The type of the null, as polars serializes it.

        Returns:
            The leaf.

        Example:
            >>> LiteralParse()._null("Null")
            LiteralLeaf(value=None, operator=None, neutral=None)
            >>> LiteralParse()._null({"List": "Int8"})
            TypedLiteralLeaf(value=None, dtype=List(Int8))
        """
        if name == "Null":
            return LiteralLeaf(None)
        if isinstance(name, dict):
            return TypedLiteralLeaf(None, SerializedDType().of(name))
        return TypedLiteralLeaf(None, self._dtype(name))

    def _dtype(self, name: object) -> DType:
        """Give the data type of a scalar from its name, where it has no parameter.

        Args:
            name: The name.

        Returns:
            The type.

        Raises:
            UndecidedError: The name holds parameters, which this reader does not read.

        Example:
            >>> LiteralParse()._dtype("UInt16")
            UInt16
            >>> LiteralParse()._dtype({"Duration": "Milliseconds"})  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not translated: a literal of ...
        """
        if not isinstance(name, str):  # pragma: no cover -- polars writes this form
            raise UndecidedError(Undecided.NOT_TRANSLATED, f"a literal of {name!r}")
        return DType(name)

    def _entry(self, part: object) -> tuple[str, object]:
        """Give the one key of a map of the tree, and its value.

        Args:
            part: The map.

        Returns:
            The key and the value.

        Raises:
            UndecidedError: The part is not a map of one key.

        Example:
            >>> LiteralParse()._entry({"Int": 1})
            ('Int', 1)
        """
        if not isinstance(part, dict) or len(part) != 1:
            raise UndecidedError(Undecided.NOT_TRANSLATED, f"a literal written as {part!r}")
        return next(iter(part.items()))


class TreeEvaluation:
    """Give the four answers of polars for a whole tree, from the definition of each node.

    `columns` gives the leaf of each column that the tree reads, by its
    name. Each node takes the leaves and the states of its inputs, in the
    order of polars. The root takes its inputs as they are, except an input of
    only literals, which the conversion folds (`FoldedLeaf`).

    Example:
        >>> import polars as pl
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.node_evaluation import ColumnLeaf
        >>> from astrolabe._rules.options import RowForm
        >>> from astrolabe._rules.tree import ExpressionTree
        >>> tree = ExpressionTree().of((pl.col("u") + 300).abs())
        >>> columns = {"u": ColumnLeaf(DType("UInt8"), RowForm.PER_ROW)}
        >>> TreeEvaluation(columns, GroupBy((3, 2))).answers(tree).run_select
        UInt16
    """

    DROPPED: ClassVar[frozenset[str]] = frozenset({"Alias", "RenameAlias"})
    RENAMES: ClassVar[frozenset[str]] = DROPPED | {"KeepName"}

    def __init__(
        self,
        columns: Mapping[str, Leaf],
        frame: GroupBy,
        *,
        element: Leaf | None = None,
        fields: Mapping[str, Leaf] | None = None,
    ) -> None:
        """Keep the leaves of the columns, the groups of `agg`, and the leaves of an evaluation.

        Args:
            columns: The leaf of each column, by its name.
            frame: The groups of the `group_by`. The frame of a `select` has their rows.
            element: The leaf of `pl.element()`, inside the evaluation of an `eval`.
            fields: The leaf of each field of a struct, by its name, inside the
                evaluation of `struct.with_fields`.

        Example:
            >>> TreeEvaluation({}, GroupBy((3,)))._frame.sizes
            (3,)
        """
        self._columns = columns
        self._frame = frame
        self._element = element
        self._struct = fields is not None
        self._fields = fields or {}

    def answers(self, tree: TreeNode) -> CellAnswers:
        """Give the four answers of the tree.

        Args:
            tree: The root of the tree.

        Returns:
            The answers.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> tree = ExpressionTree().of(pl.col("x"))
            >>> TreeEvaluation({}, GroupBy((3,))).answers(tree).planner_select
            unchecked: the leaf Column x
        """
        try:
            evaluation = self._evaluation(tree, root=True)
        except (PolarsRefusal, UndecidedError) as error:
            failure = Answers().failure(error)
            return CellAnswers(failure, failure, failure, failure)
        if isinstance(evaluation, CellAnswers):
            return evaluation
        answers = evaluation.answers()
        if not self._frame.flags.lone:
            return answers
        lifted = self._lifted(tree, answers)
        return LoneCount(self, self._frame.flags).counted(tree, lifted)

    def _lifted(self, tree: TreeNode, answers: CellAnswers) -> CellAnswers:
        """Give the answers, with the run of a `select` whose slices polars moves into the plan.

        In a lazy `select` of one expression, the slice pushdown can move the
        slices of the expression into the plan (`SlicePushdown`). The
        expression then runs as the pushdown leaves it, over the rows that the
        slice keeps. So in `pl.col("x").head(2).over(pl.lit(0))`, the window
        has the two rows of the frame, and its function fits them. Where the
        pushdown builds a cycle that the expression reads, the run never
        returns.

        Args:
            tree: The root of the tree.
            answers: The answers of the tree.

        Returns:
            The answers, with the run in the `select` over the rows of the plan.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.execution import QueryFlags
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> columns = {"x": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
            >>> tree = ExpressionTree().of(pl.col("x").head(2).over(pl.lit(0)))
            >>> TreeEvaluation(columns, GroupBy((3, 2))).answers(tree).run_select
            raises ShapeError
            >>> lone = TreeEvaluation(columns, GroupBy((3, 2), QueryFlags.lazy(lone=True)))
            >>> lone.answers(tree).run_select
            Int8
            >>> after_first = pl.col("x").head(2).over(pl.lit(0)).first()
            >>> lone.answers(ExpressionTree().of(after_first)).run_select
            Int8
        """
        try:
            pushed = SlicePushdown(self).pushed(tree)
        except PolarsRefusal as refusal:
            return replace(answers, run_select=Answers().failure(refusal))
        if pushed.bounds is None:
            return answers
        moved = self._moved(*pushed.bounds).answers(pushed.tree)
        return replace(answers, run_select=moved.run_select)

    def _moved(self, offset: int, length: int | None) -> TreeEvaluation:
        """Give the evaluation over the rows that a slice in the plan keeps.

        The slice keeps no more than `length` rows of any frame. Where the
        frame holds that many, the code bounds the rows of each column, as it
        bounds the slice that the plan took.

        Args:
            offset: The offset of the slice.
            length: The length of the slice, or `None` for a slice to the end.

        Returns:
            The evaluation.

        Example:
            >>> columns = {"x": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
            >>> moved = TreeEvaluation(columns, GroupBy((3, 2)))._moved(1, 2)
            >>> moved._frame.sizes, moved._columns["x"].selected(2).bounded
            ((2,), True)
        """
        height = sum(self._frame.sizes)
        rows = KernelLength.slice(offset, height if length is None else length).length([height])
        columns = dict(self._columns)
        if rows == length:
            columns = {name: self._bounded(leaf) for name, leaf in self._columns.items()}
        # The plan already holds the slice, so the run over its rows moves none again.
        return TreeEvaluation(columns, GroupBy((rows,), replace(self._frame.flags, lone=False)))

    def _bounded(self, leaf: Leaf) -> Leaf:
        """Mark a column of the frame as bounded by the code, and keep any other leaf.

        Args:
            leaf: The leaf of a column.

        Returns:
            The leaf.

        Example:
            >>> column = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> TreeEvaluation({}, GroupBy((3,)))._bounded(column).bounded
            True
            >>> literal = LiteralLeaf(1)
            >>> TreeEvaluation({}, GroupBy((3,)))._bounded(literal) is literal
            True
        """
        return replace(leaf, bounded=True) if isinstance(leaf, ColumnLeaf) else leaf

    def _evaluation(self, tree: TreeNode, *, root: bool) -> NodeEvaluation | CellAnswers:
        """Give the evaluation of a node over the leaves and the states of its inputs.

        A root that is itself a leaf gives the answers of that leaf. A rename
        at the root, as `.alias()`, leaves the node below it the root of the
        cast pass, so that node is evaluated as the root.

        Args:
            tree: The node.
            root: Whether the node is the root.

        Returns:
            The evaluation, or the answers of a leaf at the root.

        Raises:
            UndecidedError: No definition holds the identity of the node.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> columns = {"a": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
            >>> evaluation = TreeEvaluation(columns, GroupBy((3,)))
            >>> type(evaluation._evaluation(ExpressionTree().of(pl.col("a").abs()), root=True))
            <class 'astrolabe._rules.node_evaluation.NodeEvaluation'>
        """
        if tree.is_leaf():
            return self._leaf_answers(self.leaf(tree))
        if tree.identity.shape == "Eval":
            return self._leaf_answers(self._eval(tree))
        if tree.identity.shape == "StructEval":
            return self._leaf_answers(self._struct_eval(tree))
        if tree.identity.shape == "KeepName":
            KeptName().check(tree)
        if root and tree.identity.shape in self.RENAMES:
            return self._evaluation(tree.inputs[0], root=True)
        definition = Definitions.of(tree.identity)
        if definition is None:
            raise UndecidedError(Undecided.NOT_TRANSLATED, f"the node {tree.identity}")
        inputs = [self._input(node, root=root) for node in tree.inputs]
        rules = definition.at(tree.options())
        pushed = self._slice_inputs(tree) if rules.shape().kind is NodeKind.SLICE else None
        return NodeEvaluation(rules, inputs, self._frame, root=root, slice_inputs=pushed)

    def _slice_inputs(self, tree: TreeNode) -> list[SliceInput] | None:
        """Give the inputs of a slice as the slice pushdown of a lazy frame reads them.

        Args:
            tree: The slice node.

        Returns:
            The value, the offset and the length, or `None` where no slice
            pushdown runs over the frame.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.execution import QueryFlags
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> columns = {"a": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
            >>> head = ExpressionTree().of(pl.col("a").head(2))
            >>> TreeEvaluation(columns, GroupBy((3,)))._slice_inputs(head) is None
            True
            >>> lazy = TreeEvaluation(columns, GroupBy((3,), QueryFlags.lazy()))
            >>> [entry.literal for entry in lazy._slice_inputs(head)]
            [False, True, True]
        """
        if not self._frame.flags.slice_pushdown:
            return None
        return [self._slice_input(node) for node in tree.inputs]

    def _slice_input(self, node: TreeNode) -> SliceInput:
        """Give one input of a slice as the slice pushdown reads it.

        A leaf is its own place. A node passes the places of its inputs where
        it is elementwise over inputs that keep their rows (`SliceInput.of`).

        Args:
            node: The input.

        Returns:
            The input.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> columns = {"a": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
            >>> evaluation = TreeEvaluation(columns, GroupBy((3,)))
            >>> evaluation._slice_input(ExpressionTree().of(pl.col("a").abs())).passes
            True
            >>> lists = {"l": ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)}
            >>> each = ExpressionTree().of(pl.col("l").list.eval(pl.element() + 1))
            >>> TreeEvaluation(lists, GroupBy((3,)))._slice_input(each).passes
            False
        """
        if node.is_leaf():
            leaf = self.leaf(node)
            return SliceInput(leaf.height(), literal=leaf.operand().literal)
        definition = Definitions.of(node.identity)
        if definition is None:
            return SliceInput(self.height(node))
        heights = [self.height(child) for child in node.inputs]
        return SliceInput.of(definition.at(node.options()).shape(), heights)

    def elementwise(self, tree: TreeNode) -> bool:
        """Say whether every node of a tree works value by value.

        The state of each node answers, as it does for the input of a node:
        an `eval` reads its variant, and `is_in` reads a literal set of values
        as one value. It raises the `UndecidedError` of a node that no
        definition states.

        Args:
            tree: The tree.

        Returns:
            Whether it does.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> columns = {"a": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
            >>> evaluation = TreeEvaluation(columns, GroupBy((3,)))
            >>> evaluation.elementwise(ExpressionTree().of(pl.col("a").abs() + 1))
            True
            >>> evaluation.elementwise(ExpressionTree().of(pl.col("a").sort()))
            False
            >>> evaluation.elementwise(ExpressionTree().of(pl.col("a").is_in([1, 2])))
            True
            >>> unknown = {"Function": {"input": [{"Column": "a"}], "function": "Nothing"}}
            >>> evaluation.elementwise(ExpressionTree().read(unknown))
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: ...
        """
        return self._input(tree, root=False).elementwise()

    def planned(self, tree: TreeNode) -> Operand:
        """Give the planner operand of a tree: its planned type, and whether it is a literal.

        A literal whose value decides its data type, as `pl.lit(1)`, has an
        `Unknown` planned type. It raises the `PolarsRefusal` or the
        `UndecidedError` of a node of the tree.

        Args:
            tree: The tree.

        Returns:
            The operand.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> TreeEvaluation({}, GroupBy((3,))).planned(ExpressionTree().of(pl.lit(300))).dtype
            Unknown(int, 300)
        """
        return self._input(tree, root=False).operand()

    def computed(self, tree: TreeNode) -> Operand:
        """Give a tree as the node above it computes it: a literal of a fixed value, or a column.

        It raises the `PolarsRefusal` or the `UndecidedError` of a node of the tree.

        Args:
            tree: The tree.

        Returns:
            The operand.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> TreeEvaluation({}, GroupBy((3,))).computed(ExpressionTree().of(pl.lit("a").first()))
            Operand(dtype=String, literal=True, value='a')
        """
        return self._input(tree, root=False).computed()

    def height(self, tree: TreeNode) -> Height:
        """Give the projection height of a tree: one value, or a value for each row.

        It raises the `PolarsRefusal` or the `UndecidedError` of a node of the tree.

        Args:
            tree: The tree.

        Returns:
            The height.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.node_evaluation import ColumnLeaf
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> columns = {"x": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
            >>> summed = ExpressionTree().of(pl.col("x").sum())
            >>> TreeEvaluation(columns, GroupBy((3, 2))).height(summed)
            <Height.SCALAR: 'scalar'>
        """
        return self._input(tree, root=False).height()

    def _input(self, node: TreeNode, *, root: bool) -> Leaf:
        """Give an input of a node: a leaf, the state of a node, or a fold of literals at the root.

        Polars drops an alias and a rename by a function, as `.name.suffix()`,
        below another node while it converts the expression. So such an input
        is the input of the alias, and an alias of a literal is that literal.
        `.name.keep()` stays, because polars refuses it over a literal.

        Args:
            node: The input.
            root: Whether the node that reads the input is the root.

        Returns:
            The input as a leaf of the rules.

        Raises:
            TypeError: The node gives answers, which only a leaf gives.

        Example:
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> literal = ExpressionTree().read({"Literal": {"Dyn": {"Int": 1}}})
            >>> TreeEvaluation({}, GroupBy((3,)))._input(literal, root=True)
            LiteralLeaf(value=1, operator=None, neutral=None)
            >>> aliased = ExpressionTree().read({"Alias": [{"Literal": {"Dyn": {"Int": 1}}}, "z"]})
            >>> TreeEvaluation({}, GroupBy((3,)))._input(aliased, root=False)
            LiteralLeaf(value=1, operator=None, neutral=None)
        """
        if node.is_leaf():
            return self.leaf(node)
        if node.identity.shape in self.DROPPED:
            return self._input(node.inputs[0], root=root)
        if node.identity.shape == "Eval":
            return self._eval(node)
        if node.identity.shape == "StructEval":
            return self._struct_eval(node)
        evaluation = self._evaluation(node, root=False)
        if not isinstance(
            evaluation, NodeEvaluation
        ):  # pragma: no cover - only a leaf gives answers
            raise TypeError(evaluation)
        state = NodeState(evaluation)
        folded = self._folded(node) if root else None
        return state if folded is None else FoldedLeaf(state, folded)

    def leaf(self, node: TreeNode) -> Leaf:
        """Give the leaf of the rules for a leaf of the tree: a column, a literal or `pl.len()`.

        It raises `UndecidedError` where the leaf is a column or a field that
        this evaluation does not hold, or a kind of leaf that this module does
        not evaluate (`_named`).

        Args:
            node: The leaf of the tree.

        Returns:
            The leaf.

        Raises:
            PolarsRefusal: The leaf is `pl.field()` outside `struct.with_fields`,
                which polars refuses while it converts the expression.

        Example:
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> TreeEvaluation({}, GroupBy((3,))).leaf(ExpressionTree().read("Len")).height()
            <Height.SCALAR: 'scalar'>
        """
        shape = node.identity.shape
        body = node.serialized[shape] if isinstance(node.serialized, dict) else None
        if shape == "Literal":
            return LiteralParse().of(body)
        if shape == "Len":
            return LengthLeaf()
        if shape == "Element":
            return self._element_leaf()
        if shape == "Field" and not self._struct:
            message = "`pl.field()` called outside of struct context"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return self._named(shape, body)

    def _element_leaf(self) -> Leaf:
        """Give the leaf of `pl.element()`, which only the evaluation of an `eval` holds.

        Polars refuses `pl.element()` outside an `eval` while it converts the
        expression.

        Returns:
            The leaf.

        Raises:
            PolarsRefusal: The evaluation is not the evaluation of an `eval`.

        Example:
            >>> TreeEvaluation({}, GroupBy((3,)))._element_leaf()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `element` is not ...
        """
        if self._element is None:
            message = "`element` is not allowed in this context"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return self._element

    def _named(self, shape: str, body: object) -> Leaf:
        """Give the leaf of a column, or of a field of the struct that an evaluation reads.

        Args:
            shape: The shape of the leaf.
            body: The body of the leaf: the name of a column, or a list that
                holds the name of a field.

        Returns:
            The leaf.

        Raises:
            PolarsRefusal: The leaf is a column inside an `eval`, where polars
                reads only `pl.element()`.
            UndecidedError: The leaf is not a column or a field that this
                evaluation holds.

        Example:
            >>> TreeEvaluation({}, GroupBy((3,)))._named("Column", "x")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not translated: the leaf Column x
        """
        if shape == "Column" and self._element is not None:
            message = "named columns are not allowed in `eval` functions; consider using `element`"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        named: dict[str, Mapping[str, Leaf]] = {"Column": self._columns, "Field": self._fields}
        name = body[0] if isinstance(body, list) and body else body
        leaves = named.get(shape, {})
        if isinstance(name, str) and name in leaves:
            return leaves[name]
        raise UndecidedError(Undecided.NOT_TRANSLATED, f"the leaf {shape} {name}")

    def _struct_eval(self, node: TreeNode) -> StructEvalState:
        """Give the state of a `struct.with_fields` node, from its input and its evaluations.

        Args:
            node: The node.

        Returns:
            The state.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> columns = {"s": ColumnLeaf(struct, RowForm.PER_ROW)}
            >>> tree = ExpressionTree().of(pl.col("s").struct.with_fields(b=pl.field("a") * 2))
            >>> TreeEvaluation(columns, GroupBy((3,)))._struct_eval(tree).operand().dtype
            Struct({'a': Int8, 'b': Int8})
        """
        source = self._input(node.inputs[0], root=False)
        return StructEvalState(source, node.evaluation, self._frame, self._columns)

    def _eval(self, node: TreeNode) -> EvalState:
        """Give the state of an `eval` node, from its input and its evaluation.

        Args:
            node: The `eval` node.

        Returns:
            The state.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.options import RowForm
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> columns = {"l": ColumnLeaf(DType("List", (DType("Int8"),)), RowForm.PER_ROW)}
            >>> tree = ExpressionTree().of(pl.col("l").list.eval(pl.element() * 2))
            >>> TreeEvaluation(columns, GroupBy((3,)))._eval(tree).operand().dtype
            List(Int8)
        """
        body = node.serialized["Eval"] if isinstance(node.serialized, dict) else {}
        variant = EvalVariant(body.get("variant") if isinstance(body, dict) else None)
        source = self._input(node.inputs[0], root=False)
        return EvalState(source, variant, node.evaluation[0], self._frame)

    def _folded(self, node: TreeNode) -> LiteralLeaf | None:
        """Give the literal that the conversion folds a node of only literals into, or `None`.

        Args:
            node: The node.

        Returns:
            The literal, or `None` where the node holds a column or does not fold.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> tree = ExpressionTree().of(pl.lit(200) + pl.lit(100))
            >>> TreeEvaluation({}, GroupBy((3,)))._folded(tree)
            LiteralLeaf(value=300, operator=None, neutral=None)
        """
        operand = self._fold(node)
        if operand is None or not operand.literal:
            return None
        return LiteralLeaf.of(operand)

    def _fold(self, node: TreeNode) -> Operand | None:
        """Give the operand that a subtree of literals folds into, or `None`.

        Args:
            node: The root of the subtree.

        Returns:
            The literal, or `None`.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> TreeEvaluation({}, GroupBy((3,)))._fold(ExpressionTree().of(pl.col("a"))) is None
            True
        """
        if node.identity.shape == "Literal":
            leaf = self.leaf(node)
            return leaf.folded() if isinstance(leaf, LiteralLeaf) else None
        if node.identity.shape != "BinaryExpr":
            return None
        sides = [self._fold(part) for part in node.inputs]
        if any(side is None for side in sides):
            return None
        operator = Operator(node.identity.function)
        return BinaryFold().binary(operator, sides[0], sides[1])  # type: ignore[arg-type]

    def _leaf_answers(self, leaf: Leaf) -> CellAnswers:
        """Give the four answers of a tree that is one leaf, such as an `eval` at the root.

        A planner that refuses refuses the run too. Otherwise the run reads the
        data, which can refuse where the planner gives a type, as a kernel can.

        Args:
            leaf: The leaf.

        Returns:
            The answers.

        Example:
            >>> TreeEvaluation({}, GroupBy((3,)))._leaf_answers(LiteralLeaf(1)).run_select
            Int32
        """
        answers = Answers()
        planner = answers.of(partial(self._planner_type, leaf))
        if not isinstance(planner, DType):
            return CellAnswers(planner, planner, planner, planner)
        wrapped = leaf.height() is not Height.SCALAR
        agg = DType("List", (planner,)) if wrapped else planner
        run = answers.of(lambda: leaf.data().dtype)
        run_agg = answers.of(lambda: self._grouped_type(leaf, wrapped=wrapped))
        return CellAnswers(planner, agg, run, run_agg)

    def _planner_type(self, leaf: Leaf) -> DType:
        """Give the planner type of a leaf, with a literal at the type that it takes alone.

        Args:
            leaf: The leaf.

        Returns:
            The type.

        Example:
            >>> TreeEvaluation({}, GroupBy((3,)))._planner_type(LiteralLeaf(1))
            Int32
        """
        return leaf.operand().dtype.materialized()

    def _grouped_type(self, leaf: Leaf, *, wrapped: bool) -> DType:
        """Give the run type of a leaf in `agg`: its data, in a list where `agg` lists each group.

        Args:
            leaf: The leaf.
            wrapped: Whether the planner wraps the leaf, which does not give one value.

        Returns:
            The type.

        Example:
            >>> leaf = LiteralLeaf(1)
            >>> TreeEvaluation({}, GroupBy((3, 2)))._grouped_type(leaf, wrapped=False)
            Int32
        """
        data = leaf.data().dtype
        listed = self._frame.output(leaf.grouped(self._frame), wrapped=wrapped).listed()
        return DType("List", (data,)) if listed else data


@dataclass(frozen=True)
class CommonSlice:
    """Hold the rows that every slice of the columns of an expression keeps, as polars joins them.

    Polars can move the slices that cut the columns of an expression into the
    plan as one slice. Slices from the start join into the least offset and the
    greatest end, and slices from the end join the same way, counted from the
    end. A slice from the start and one from the end do not join.

    Example:
        >>> joined = CommonSlice.of(0, 2).joined(CommonSlice.of(1, 3))
        >>> joined.plan()
        (0, 4, 0)
    """

    from_end: bool
    offset: int
    end: int

    MAX_LENGTH: ClassVar[int] = 2**32 - 1

    @classmethod
    def of(cls, offset: int, length: int) -> CommonSlice:
        """Give the rows that one slice keeps.

        Args:
            offset: The offset of the slice, which counts from the end where it is negative.
            length: The length of the slice.

        Returns:
            The rows.

        Example:
            >>> CommonSlice.of(-2, 5)
            CommonSlice(from_end=True, offset=-2, end=0)
        """
        if offset >= 0:
            end = min(min(offset, cls.MAX_LENGTH) + length, cls.MAX_LENGTH)
            return cls(from_end=False, offset=offset, end=end)
        return cls(from_end=True, offset=offset, end=min(0, offset + length))

    def joined(self, other: CommonSlice) -> CommonSlice | None:
        """Give the rows that two sets of rows keep together.

        Args:
            other: The other rows.

        Returns:
            The rows, or `None` where one counts from the start and the other from the end.

        Example:
            >>> CommonSlice.of(0, 2).joined(CommonSlice.of(-1, 1)) is None
            True
        """
        if self.from_end != other.from_end:
            return None
        return CommonSlice(self.from_end, min(self.offset, other.offset), max(self.end, other.end))

    def plan(self) -> tuple[int, int, int]:
        """Give the slice that polars puts in the plan, and how far it moves each slice of a column.

        Returns:
            The offset and the length of the slice of the plan, and the
            correction to take from the offset of each slice of a column that
            stays.

        Example:
            >>> CommonSlice.of(-3, 2).plan()
            (-3, 2, -1)
        """
        if not self.from_end:
            return self.offset, self.end - min(self.offset, self.MAX_LENGTH), self.offset
        return self.offset, min(abs(self.offset - self.end), self.MAX_LENGTH), self.end


@dataclass(frozen=True)
class PushedSlices:
    """Hold an expression after the slice pushdown of polars, and the slice that it adds.

    `bounds` holds the offset and the length of the slice of the plan, the
    length `None` for a slice to the end, or `None` where polars adds no
    slice. `height` is the projection height of the expression after the
    pushdown.

    Example:
        >>> import polars as pl
        >>> from astrolabe._rules.tree import ExpressionTree
        >>> PushedSlices(ExpressionTree().of(pl.col("x")), None, Height.COLUMN).bounds is None
        True
    """

    tree: TreeNode
    bounds: tuple[int, int | None] | None
    height: Height


@dataclass(frozen=True)
class _Cut:
    """Hold a slice that the pushdown moves: the place of its input, its bounds and their places.

    `bounds` is `None` for bounds that are not literals, and `edges` names the
    places of the nodes of the bounds of a slice, `None` for `first` and `last`.

    Example:
        >>> _Cut(0, (0, 1), None).bounds
        (0, 1)
    """

    source: int
    bounds: tuple[int, int] | None
    edges: tuple[int, int] | None


@dataclass(frozen=True)
class _Placement:
    """Hold what one place did with a slice: whether it took it, the bounds there, and the count.

    Example:
        >>> _Placement(True, (0, 2), 1).hits
        1
    """

    sliced: bool
    bounds: tuple[int, int] | None
    hits: int


@dataclass
class _Cell:
    """Hold one node of an expression that the slice pushdown changes in place.

    The pushdown of polars works on an arena of nodes, and replaces a node in
    place, so a node that holds it sees the change. A cell is that place: its
    node gives the kind and the options, and `inputs` names the cells below.

    Example:
        >>> import polars as pl
        >>> from astrolabe._rules.tree import ExpressionTree
        >>> _Cell(ExpressionTree().of(pl.col("x")), []).inputs
        []
    """

    node: TreeNode
    inputs: list[int]


@dataclass
class _PushState:
    """Hold what the pushdown knows of one input: where a slice above it can go, and its height.

    Example:
        >>> _PushState([], Height.UNKNOWN).places
        []
    """

    places: list[int]
    height: Height


class SlicePushdown:
    """Translate the slice pushdown of polars over a lazy `select` of one expression.

    Polars reads a slice, `first` and `last` in an expression, and moves each
    one down the nodes that work value by value, to the deepest places that it
    can reach. A slice over a slice joins into one slice. Where every column
    that the expression reads, and each `pl.len()`, then lies directly below a
    slice of literal bounds, and the slices join into one set of rows
    (`CommonSlice`), polars puts that slice in the plan. Each slice of a column
    as long as it goes away, and each other one moves its offset.

    The expression then runs over the rows of the plan. The class follows the
    pushdown of polars step by step, on a copy of the tree, as `pushed` shows.

    Example:
        >>> import polars as pl
        >>> from astrolabe._rules.tree import ExpressionTree
        >>> columns = {"x": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
        >>> pushdown = SlicePushdown(TreeEvaluation(columns, GroupBy((3, 2))))
        >>> pushdown.pushed(ExpressionTree().of(pl.col("x").head(2).over(pl.lit(0)))).bounds
        (0, 2)
    """

    ENDS: ClassVar[dict[NodeIdentity, tuple[int, int]]] = {
        NodeIdentity("Agg", "First"): (0, 1),
        NodeIdentity("Agg", "Last"): (-1, 1),
    }
    SLICE: ClassVar[NodeIdentity] = NodeIdentity("Slice", "")
    NOT_LITERAL: ClassVar[object] = object()

    def __init__(self, evaluation: TreeEvaluation) -> None:
        """Keep the evaluation, which gives the leaves of the tree.

        Args:
            evaluation: The evaluation of the tree.
        """
        self._evaluation = evaluation
        self._traits = NodeTraits(evaluation)
        self._cells: list[_Cell] = []
        self._limit: CommonSlice | None = None
        self._hits: int | None = 0
        self._direct: dict[int, None] = {}

    def pushed(self, tree: TreeNode) -> PushedSlices:
        """Give the expression after the pushdown, and the slice of the plan where polars adds one.

        Where the expression reads a slice that moved into its own offset or
        length, the query never returns, and the rebuild of the tree raises
        `PolarsRefusal`.

        Args:
            tree: The expression, which stands alone in a lazy `select`.

        Returns:
            The result.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> columns = {"x": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
            >>> pushdown = SlicePushdown(TreeEvaluation(columns, GroupBy((3, 2))))
            >>> pushdown.pushed(ExpressionTree().of(pl.col("x").sort().head(2))).bounds is None
            True

            `first` joins the slice below it, which keeps its own rows of the plan:

            >>> pushed = pushdown.pushed(ExpressionTree().of(pl.col("x").head(3).first()))
            >>> pushed.bounds, pushed.height
            ((0, 3), <Height.SCALAR: 'scalar'>)
        """
        self._cells, self._limit, self._hits, self._direct = [], None, 0, {}
        root = self._added(tree)
        self._visit(root)
        plan = self._limit.plan() if self._limit is not None and self._hits == 0 else None
        bounds = None
        if plan is not None:
            offset, length, correction = plan
            for place in self._direct:
                self._corrected(place, length, correction)
            bounds = (offset, None if length == CommonSlice.MAX_LENGTH else length)
        rebuilt = self._rebuilt(root)
        return PushedSlices(rebuilt, bounds, self._traits.height(rebuilt))

    def _added(self, node: TreeNode) -> int:
        """Put a tree into the cells, its inputs first.

        Args:
            node: The tree.

        Returns:
            The place of its root.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> pushdown = SlicePushdown(TreeEvaluation({}, GroupBy((3,))))
            >>> pushdown._added(ExpressionTree().of(pl.col("x") + 1))
            2
        """
        inputs = [self._added(child) for child in node.inputs]
        self._cells.append(_Cell(node, inputs))
        return len(self._cells) - 1

    def _rebuilt(self, place: int, path: frozenset[int] = frozenset()) -> TreeNode:
        """Give the tree that the cells hold below a place.

        A slice that moves into its own offset or length reads itself, so the
        cells hold a cycle. Where the root reaches that cycle, polars walks it
        with no end, and the query never returns.

        Args:
            place: The place of the root.
            path: The places above it, from the root of the expression.

        Returns:
            The tree.

        Raises:
            PolarsRefusal: The place reads itself, so the query never returns.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> pushdown = SlicePushdown(TreeEvaluation({}, GroupBy((3,))))
            >>> root = pushdown._added(ExpressionTree().of(pl.col("x") + 1))
            >>> pushdown._rebuilt(root).identity.shape
            'BinaryExpr'
            >>> pushdown._rebuilt(root, frozenset({root}))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: hangs: ...
        """
        if place in path:
            message = "a place below the value reads the cycle of the offset or the length"
            raise PolarsRefusal(Stop.HANGS, message)
        cell = self._cells[place]
        inputs = tuple(self._rebuilt(child, path | {place}) for child in cell.inputs)
        return replace(cell.node, inputs=inputs)

    def _visit(self, place: int) -> _PushState:
        """Run the pushdown over the cells below a place, its inputs first, as polars does.

        Args:
            place: The place of the node.

        Returns:
            Where a slice above the node can go, and its height.
        """
        cell = self._cells[place]
        states = [self._visit(child) for child in cell.inputs]
        node = cell.node
        if node.identity.shape in TreeEvaluation.RENAMES:
            return states[0]
        state = self._joined(states)
        state.height = self._traits.step(node, [entry.height for entry in states])
        if node.is_leaf() and node.identity.shape in {"Column", "Len"}:
            self._hits = None if self._hits is None else self._hits + 1
        self._moved_down(place, state)
        return self._closed(place, state)

    def _moved_down(self, place: int, state: _PushState) -> None:
        """Move the slice of a node to each place below it where a slice can go.

        Args:
            place: The place of the node.
            state: The state that the inputs of the node give.
        """
        cut = self._slice_of(place)
        if cut is None:
            return
        for candidate in list(state.places):
            self._pushed(place, cut, candidate)

    def _closed(self, place: int, state: _PushState) -> _PushState:
        """Give the places where a slice above a node can go, after the node.

        A node that does not work value by value stops a slice, so the node
        itself is the one place for it, as it is where no place is left.

        Args:
            place: The place of the node.
            state: The state that the inputs of the node give.

        Returns:
            The state.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> pushdown = SlicePushdown(TreeEvaluation({}, GroupBy((3,))))
            >>> root = pushdown._added(ExpressionTree().of(pl.col("x").sort()))
            >>> pushdown._closed(root, _PushState([0], Height.COLUMN)).places
            [1]
        """
        if not self._traits.elementwise(self._cells[place].node):
            state.places = []
        if not state.places:
            state.places = [place]
        return state

    def _joined(self, states: Sequence[_PushState]) -> _PushState:
        """Give the places where a slice above a node can go, from the states of its inputs.

        The first input that is not one value gives its places. Each later
        input of the height of the frame adds its places, and an input of any
        other height that is not one value leaves no place.

        Args:
            states: The states of the inputs.

        Returns:
            The state.

        Example:
            >>> pushdown = SlicePushdown(TreeEvaluation({}, GroupBy((3,))))
            >>> column = _PushState([1], Height.COLUMN)
            >>> pushdown._joined([_PushState([0], Height.SCALAR), column]).places
            [1]
        """
        firsts = [index for index, state in enumerate(states) if state.height is not Height.SCALAR]
        if not firsts:
            return _PushState([], Height.UNKNOWN)
        first = states[firsts[0]]
        joined = _PushState(list(first.places), first.height)
        for other in states[firsts[0] + 1 :]:
            if other.height is Height.SCALAR:
                continue
            if joined.height is Height.COLUMN and other.height is Height.COLUMN:
                joined.places.extend(other.places)
                continue
            joined.places = []
            break
        return joined

    def _slice_of(self, place: int) -> _Cut | None:
        """Give the slice that polars reads in a node: a slice, `first` or `last`.

        Args:
            place: The place of the node.

        Returns:
            The slice, or `None` where the node is no slice.
        """
        cell = self._cells[place]
        ends = self.ENDS.get(cell.node.identity)
        if ends is not None:
            return _Cut(cell.inputs[0], ends, None)
        if cell.node.identity != self.SLICE:
            return None
        edges = (cell.inputs[1], cell.inputs[2])
        return _Cut(cell.inputs[0], self._bounds(*edges), edges)

    def _bounds(self, offset: int, length: int) -> tuple[int, int] | None:
        """Give the offset and the length of a slice where both are literals of one value.

        A null length keeps every row to the end.

        Args:
            offset: The place of the offset.
            length: The place of the length.

        Returns:
            The bounds, or `None` where one is not a literal integer.
        """
        start, size = self._literal(offset), self._literal(length)
        if not self._whole(start):
            return None
        if size is None:
            return start, -start if start < 0 else CommonSlice.MAX_LENGTH
        return (start, size) if self._whole(size) else None

    def _literal(self, place: int) -> object:
        """Give the value of a literal of one value at a place.

        Args:
            place: The place.

        Returns:
            The value, or `NOT_LITERAL` where the node is not such a literal.
        """
        node = self._cells[place].node
        if not node.is_leaf() or node.identity.shape != "Literal":
            return self.NOT_LITERAL
        leaf = self._evaluation.leaf(node)
        if leaf.height() is not Height.SCALAR:
            return self.NOT_LITERAL
        return leaf.operand().value

    def _whole(self, value: object) -> TypeGuard[int]:
        """Say whether a value is a whole number, and not a boolean.

        Args:
            value: The value.

        Returns:
            Whether it is.

        Example:
            >>> pushdown = SlicePushdown(TreeEvaluation({}, GroupBy((3,))))
            >>> pushdown._whole(2), pushdown._whole(True), pushdown._whole(pushdown.NOT_LITERAL)
            (True, False, False)
        """
        return isinstance(value, int) and not isinstance(value, bool)

    def _pushed(self, place: int, cut: _Cut, candidate: int) -> None:
        """Move the slice of a node to one place below it, as polars does.

        Where the slice goes below its own node, the node gives way to its
        input. Where the slice then cuts a column directly, it counts.

        Args:
            place: The place of the node that polars reads as a slice.
            cut: The slice of the node.
            candidate: The place where the slice goes.
        """
        placed = self._placed(cut, candidate)
        own = self._cells[place].node.identity == self.SLICE
        if placed.sliced and own:
            self._cells[place] = replace(self._cells[cut.source])
        below = self._cells[candidate].inputs[0] if placed.sliced else cut.source
        if self._cells[below].node.identity.shape != "Column":
            return
        if placed.sliced:
            self._direct.setdefault(candidate)
        elif own:
            self._direct.setdefault(place)
        self._counted(placed.hits, placed.bounds)

    def _placed(self, cut: _Cut, candidate: int) -> _Placement:
        """Put a slice at one place, where it joins a slice that is there already.

        Args:
            cut: The slice.
            candidate: The place.

        Returns:
            Whether the place took a slice, the bounds of the slice there, and
            the columns that it adds to the count.
        """
        inner = self._cells[candidate]
        if inner.node.identity != self.SLICE:
            return _Placement(self._plain(cut, candidate), cut.bounds, 1)
        inner_bounds = self._bounds(inner.inputs[1], inner.inputs[2])
        joined = None
        if cut.bounds is not None and inner_bounds is not None:
            joined = self._inner(cut.bounds, inner_bounds)
        if joined is None:
            return _Placement(self._plain(cut, candidate), cut.bounds, 0)
        if joined != inner_bounds:
            self._cells[candidate] = replace(self._cells[inner.inputs[0]])
            self._wrapped(candidate, _Cut(cut.source, joined, None))
        return _Placement(True, joined, 0)

    def _plain(self, cut: _Cut, candidate: int) -> bool:
        """Put a slice over a place below the input of the slice, as polars does.

        Args:
            cut: The slice.
            candidate: The place.

        Returns:
            Whether the place took the slice: not where it is the input itself.
        """
        if candidate == cut.source:
            return False
        self._wrapped(candidate, cut)
        return True

    def _counted(self, hits: int, extracted: tuple[int, int] | None) -> None:
        """Count a slice of a column, and join its rows into the rows of every such slice.

        Args:
            hits: The columns that the slice adds to the count of sliced columns.
            extracted: The bounds of the slice, or `None` for bounds that are not literals.
        """
        self._hits = None if self._hits is None or self._hits < hits else self._hits - hits
        if extracted is None:
            self._hits = None
            return
        rows = CommonSlice.of(*extracted)
        joined = rows if self._limit is None else self._limit.joined(rows)
        if joined is None:
            self._hits = None
            return
        self._limit = joined

    def _inner(self, outer: tuple[int, int], inner: tuple[int, int]) -> tuple[int, int] | None:
        """Give the one slice that a slice of a slice keeps, as polars joins them.

        Args:
            outer: The bounds of the slice above.
            inner: The bounds of the slice below.

        Returns:
            The bounds, or `None` where one counts from the start and the other from the end.

        Example:
            >>> pushdown = SlicePushdown(TreeEvaluation({}, GroupBy((3,))))
            >>> pushdown._inner((1, 5), (0, 3)), pushdown._inner((-1, 1), (-3, 2))
            ((1, 2), (-2, 1))
        """
        (outer_offset, outer_length), (inner_offset, inner_length) = outer, inner
        if outer_offset >= 0 and inner_offset >= 0:
            left = inner_length - outer_offset if inner_length > outer_offset else 0
            return inner_offset + outer_offset, min(left, outer_length)
        if outer_offset < 0 and inner_offset < 0:
            inner_stop = inner_offset + inner_length
            start = max(inner_stop + outer_offset, inner_offset)
            stop = max(inner_stop + outer_offset + outer_length, start)
            return start, min(stop - start, CommonSlice.MAX_LENGTH)
        return None

    def _wrapped(self, place: int, cut: _Cut) -> None:
        """Put a slice in place of a node, over a copy of the node.

        A slice of literal bounds takes new literals, and any other slice
        keeps the nodes of its own bounds, as polars does.

        Args:
            place: The place of the node.
            cut: The slice.
        """
        self._cells.append(replace(self._cells[place]))
        copied = len(self._cells) - 1
        if cut.bounds is None:
            self._cells[place] = _Cell(self._template(0, 0), [copied, *checked_not_none(cut.edges)])
            return
        template = self._template(*cut.bounds)
        self._cells.append(_Cell(template.inputs[1], []))
        self._cells.append(_Cell(template.inputs[2], []))
        count = len(self._cells)
        self._cells[place] = _Cell(template, [copied, count - 2, count - 1])

    def _template(self, offset: int, length: int) -> TreeNode:
        """Give the tree of a slice of literal bounds, as polars writes a slice that it moves.

        Args:
            offset: The offset.
            length: The length.

        Returns:
            The tree, whose first input the caller replaces.

        Example:
            >>> pushdown = SlicePushdown(TreeEvaluation({}, GroupBy((3,))))
            >>> pushdown._template(0, 2).identity
            NodeIdentity(shape='Slice', function='')
        """
        bounds = (pl.lit(offset, dtype=pl.Int64), pl.lit(length, dtype=pl.UInt32))
        return ExpressionTree().of(pl.col("_").slice(*bounds))

    def _corrected(self, place: int, length: int, correction: int) -> None:
        """Take a slice of a column out where the plan keeps its rows, or move its offset.

        Args:
            place: The place of the slice.
            length: The length of the slice of the plan.
            correction: How far the plan moves the rows.
        """
        cell = self._cells[place]
        bounds = checked_not_none(self._bounds(cell.inputs[1], cell.inputs[2]))
        if bounds[1] == length:
            self._cells[place] = replace(self._cells[cell.inputs[0]])
        elif correction != 0:
            source = cell.inputs[0]
            self._cells[place] = replace(self._cells[source])
            self._wrapped(place, _Cut(source, (bounds[0] - correction, bounds[1]), None))


class NodeTraits:
    """Give the facts of a node that the optimizer of polars reads: its height and its kind of work.

    The slice pushdown and the check of the keys of a join read each node of
    an expression by these two facts, as polars reads them from the kind of
    the node and its options.

    Example:
        >>> import polars as pl
        >>> from astrolabe._rules.tree import ExpressionTree
        >>> columns = {"x": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
        >>> traits = NodeTraits(TreeEvaluation(columns, GroupBy((3,))))
        >>> traits.height(ExpressionTree().of(pl.col("x").sum()))
        <Height.SCALAR: 'scalar'>
    """

    EVALS: ClassVar[frozenset[str]] = frozenset({"Eval", "StructEval"})

    def __init__(self, evaluation: TreeEvaluation) -> None:
        """Keep the evaluation, which gives the leaves of the tree.

        Args:
            evaluation: The evaluation of the tree.
        """
        self._evaluation = evaluation

    def known(self, node: TreeNode) -> bool:
        """Say whether the rules know the facts of a node: a leaf, a rename, an `eval`, or stated.

        Args:
            node: The node.

        Returns:
            Whether they do.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> traits = NodeTraits(TreeEvaluation({}, GroupBy((3,))))
            >>> traits.known(ExpressionTree().of(pl.col("x").abs().alias("y")))
            True
        """
        named = node.identity.shape in TreeEvaluation.RENAMES | self.EVALS
        return node.is_leaf() or named or self._shape(node) is not None

    def elementwise(self, node: TreeNode) -> bool:
        """Say whether polars reads a node as one that works value by value.

        A column and a literal of one value do, and so does an `eval` over
        each list, array or struct. `pl.len()`, a slice, an aggregation and a
        cumulative `eval` do not.

        Args:
            node: The node.

        Returns:
            Whether it does.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> traits = NodeTraits(TreeEvaluation({}, GroupBy((3,))))
            >>> traits.elementwise(ExpressionTree().of(pl.col("x").shift(1)))
            False
        """
        if node.is_leaf():
            shape = node.identity.shape
            literal = shape == "Literal" and self._evaluation.leaf(node).height() is Height.SCALAR
            return shape == "Column" or literal
        if node.identity.shape in TreeEvaluation.RENAMES:
            return True
        if node.identity.shape == "StructEval":
            return True
        if node.identity.shape == "Eval":
            variant = node.options().value(("variant",))
            return not (isinstance(variant, dict) and "Cumulative" in variant)
        shape = self._shape(node)
        return shape is not None and shape.is_elementwise()

    def height(self, node: TreeNode) -> Height:
        """Give the projection height of a tree, as the planner reads it.

        Args:
            node: The tree.

        Returns:
            The height.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> columns = {"x": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
            >>> traits = NodeTraits(TreeEvaluation(columns, GroupBy((3,))))
            >>> traits.height(ExpressionTree().of(pl.col("x").abs().alias("y")))
            <Height.COLUMN: 'column'>
        """
        if node.identity.shape in TreeEvaluation.RENAMES:
            return self.height(node.inputs[0])
        return self.step(node, [self.height(child) for child in node.inputs])

    def step(self, node: TreeNode, inputs: Sequence[Height]) -> Height:
        """Give the projection height of a node from the heights of its inputs.

        A node that no definition states takes the height that the
        evaluation gives it.

        Args:
            node: The node.
            inputs: The heights of its inputs.

        Returns:
            The height, or `UNKNOWN` where the evaluation refuses the node.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> traits = NodeTraits(TreeEvaluation({}, GroupBy((3,))))
            >>> traits.step(ExpressionTree().of(pl.col("x").head(2)), [Height.COLUMN] * 3)
            <Height.UNKNOWN: 'unknown'>
        """
        if node.is_leaf():
            return self._evaluation.leaf(node).height()
        if node.identity.shape in self.EVALS:
            return inputs[0]
        shape = self._shape(node)
        if shape is not None:
            return shape.height(list(inputs))
        try:
            return self._evaluation.height(node)
        except (PolarsRefusal, UndecidedError):
            return Height.UNKNOWN

    def _shape(self, node: TreeNode) -> Shape | None:
        """Give the shape of a node from its definition.

        Args:
            node: The node.

        Returns:
            The shape, or `None` for a node that no definition states.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> head = ExpressionTree().of(pl.col("s").head(2))
            >>> NodeTraits(TreeEvaluation({}, GroupBy((3,))))._shape(head).kind
            <NodeKind.SLICE: 'Slice'>
        """
        definition = Definitions.of(node.identity)
        return None if definition is None else definition.at(node.options()).shape()


class JoinKeyRows:
    """Say whether polars refuses an expression as a key of a join, because it changes the rows.

    Polars moves a key of one value, as an aggregation or a literal, into a
    column of the frame first. It then needs every other key to work value by
    value at each of its nodes, so that the keys of both frames have the
    lengths of their frames. Only the literal of an `is_in` stays outside the
    check. Polars raises `InvalidOperationError` for any other key.

    Example:
        >>> import polars as pl
        >>> from astrolabe._rules.tree import ExpressionTree
        >>> columns = {"a": ColumnLeaf(DType("Int64"), RowForm.PER_ROW)}
        >>> rows = JoinKeyRows(TreeEvaluation(columns, GroupBy((3,))))
        >>> rows.refuses(ExpressionTree().of(pl.col("a").shift(1)))
        True
        >>> total, listed = pl.col("a").sum(), pl.col("a").is_in([1])
        >>> rows.refuses(ExpressionTree().of(total)), rows.refuses(ExpressionTree().of(listed))
        (False, False)
    """

    IS_IN: ClassVar[NodeIdentity] = NodeIdentity("Function", "Boolean.IsIn")

    def __init__(self, evaluation: TreeEvaluation) -> None:
        """Keep the facts of the nodes of the key.

        Args:
            evaluation: The evaluation of the tree, which gives its leaves.
        """
        self._traits = NodeTraits(evaluation)

    def refuses(self, tree: TreeNode) -> bool | None:
        """Say whether polars refuses a key.

        Args:
            tree: The key.

        Returns:
            Whether it does, or `None` where a node that no definition states
            leaves it unknown.
        """
        if self._traits.height(tree) is Height.SCALAR:
            return False
        kept = self._elementwise(tree)
        return None if kept is None else not kept

    def _elementwise(self, node: TreeNode) -> bool | None:
        """Say whether every node of a tree works value by value, as polars checks a key.

        Args:
            node: The tree.

        Returns:
            Whether every node does, or `None` where the rules do not know a node.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> columns = {"a": ColumnLeaf(DType("Int64"), RowForm.PER_ROW)}
            >>> rows = JoinKeyRows(TreeEvaluation(columns, GroupBy((3,))))
            >>> rows._elementwise(ExpressionTree().of(pl.col("a").is_in(pl.col("a").implode())))
            False
        """
        if not self._traits.known(node):
            return None
        if not self._traits.elementwise(node):
            return False
        inputs = node.inputs
        if node.identity == self.IS_IN and inputs[1].identity.shape == "Literal":
            inputs = inputs[:1]
        answers = [self._elementwise(child) for child in inputs]
        if False in answers:
            return False
        return None if None in answers else True


class LoneCount:
    """Give the run of a `len` that stands alone in a lazy `select`, where polars counts the rows.

    The projection pushdown of polars puts a count of the rows in place of
    such a `len` where its input has the height of the frame (`RowCount`). The
    input then never runs. The slice pushdown runs first (`SlicePushdown`),
    and where it moves a slice into the plan, the input loses the slice and
    can get the height of the frame.

    Example:
        >>> import polars as pl
        >>> from astrolabe._rules.execution import QueryFlags
        >>> from astrolabe._rules.tree import ExpressionTree
        >>> columns = {"s": ColumnLeaf(DType("String"), RowForm.PER_ROW)}
        >>> lone = TreeEvaluation(columns, GroupBy((3,), QueryFlags.lazy(lone=True)))
        >>> lone.answers(ExpressionTree().of(pl.col("s").abs().len())).run_select
        UInt32
        >>> lone.answers(ExpressionTree().of(pl.col("s").abs().count())).run_select
        raises InvalidOperationError
    """

    COUNT: ClassVar[NodeIdentity] = NodeIdentity("Agg", "Count")

    def __init__(self, evaluation: TreeEvaluation, flags: QueryFlags) -> None:
        """Keep the evaluation, which gives the leaves of the tree, and the passes of the frame.

        Args:
            evaluation: The evaluation of the tree.
            flags: The passes of the optimizer over the frame.
        """
        self._evaluation = evaluation
        self._flags = flags

    def counted(self, tree: TreeNode, answers: CellAnswers) -> CellAnswers:
        """Give the answers, with the count of the rows as the run where polars counts them.

        Args:
            tree: The root of the tree, which can be a rename.
            answers: The answers of the tree.

        Returns:
            The answers, with the planner type as the run type in a `select`
            where polars counts the rows, and a run that never returns where
            the input reads a cycle that the slice pushdown builds.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.execution import QueryFlags
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> columns = {"s": ColumnLeaf(DType("String"), RowForm.PER_ROW)}
            >>> lone = TreeEvaluation(columns, GroupBy((3,), QueryFlags.lazy(lone=True)))
            >>> lone.answers(ExpressionTree().of(pl.col("s").abs().head(2).len())).run_select
            UInt32
            >>> sorted_head = pl.col("s").abs().sort().head(2).len()
            >>> lone.answers(ExpressionTree().of(sorted_head)).run_select
            raises InvalidOperationError
            >>> beside_first = (pl.col("s").abs().head(2) + pl.col("s").abs().first()).len()
            >>> lone.answers(ExpressionTree().of(beside_first)).run_select
            UInt32
            >>> lone.answers(ExpressionTree().of((pl.col("s") * 2).len())).run_select
            raises InvalidOperationError
            >>> refused = Answers().failure(PolarsRefusal(ErrorClass.COMPUTE, "no plan"))
            >>> plain = CellAnswers(refused, refused, refused, refused)
            >>> count = LoneCount(lone, QueryFlags.lazy(lone=True))
            >>> count.counted(ExpressionTree().of(pl.col("s").len()), plain) == plain
            True
            >>> count.counted(ExpressionTree().of(pl.col("s").abs()), plain) == plain
            True
        """
        root = self._below_renames(tree)
        if root.identity != self.COUNT or isinstance(answers.run_select, DType):
            return answers
        if not isinstance(answers.planner_select, DType):  # pragma: no cover -- it plans UInt32
            return answers
        nulls = root.options().value(("Count", "include_nulls")) is True
        try:
            pushed = SlicePushdown(self._evaluation).pushed(root.inputs[0])
        except PolarsRefusal as refusal:
            return replace(answers, run_select=Answers().failure(refusal))
        if not RowCount().counts(self._flags, nulls=nulls, height=pushed.height):
            return answers
        return replace(answers, run_select=answers.planner_select)

    def _below_renames(self, tree: TreeNode) -> TreeNode:
        """Give the node below the renames at the root, which polars counts.

        Args:
            tree: The root.

        Returns:
            The first node that is not a rename.

        Example:
            >>> import polars as pl
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> named = ExpressionTree().of(pl.col("s").len().alias("n"))
            >>> count = LoneCount(TreeEvaluation({}, GroupBy((3,))), QueryFlags.lazy())
            >>> count._below_renames(named).identity
            NodeIdentity(shape='Agg', function='Count')
        """
        while tree.identity.shape in TreeEvaluation.RENAMES:
            tree = tree.inputs[0]
        return tree


@dataclass(frozen=True)
class LengthLeaf:
    """Hold `pl.len()`: the number of rows of the frame, or of each group.

    Example:
        >>> LengthLeaf().data()
        Data(dtype=UInt32, nulls=False, one_row=False)
    """

    def operand(self) -> Operand:
        """Give the length as the type function reads it: a column of the index type.

        Returns:
            The operand.

        Example:
            >>> LengthLeaf().operand().dtype
            UInt32
        """
        return Operand.column(ExprFields().length())

    def folded(self) -> Operand:
        """Give the length as the cast pass reads it.

        Returns:
            The operand.

        Example:
            >>> LengthLeaf().folded().literal
            False
        """
        return self.operand()

    def optimized(self) -> Operand:
        """Give the input as the optimizer reads it, after the conversion: the same as `folded`.

        Returns:
            The operand.

        Example:
            >>> LengthLeaf().optimized().literal
            False
        """
        return self.folded()

    def computed(self) -> Operand:
        """Give the length as a run computes it.

        Returns:
            The operand.

        Example:
            >>> LengthLeaf().computed().literal
            False
        """
        return self.operand()

    def data(self) -> Data:
        """Give the data of the length: one value of the index type.

        Returns:
            The data.

        Example:
            >>> LengthLeaf().data().dtype
            UInt32
        """
        return Data(self.operand().dtype)

    def grouped_data(self, frame: GroupBy) -> Data:
        """Give the data of the length in `agg`: its data in a `select`.

        Args:
            frame: The groups of the `group_by`, which the data does not read.

        Returns:
            The data.

        Example:
            >>> LengthLeaf().grouped_data(GroupBy((3, 2))).dtype
            UInt32
        """
        del frame
        return self.data()

    def height(self) -> Height:
        """Give the height of the length: one value.

        Returns:
            The height.

        Example:
            >>> LengthLeaf().height()
            <Height.SCALAR: 'scalar'>
        """
        return Height.SCALAR

    def elementwise(self) -> bool:
        """Say that `pl.len()` does not work value by value.

        Returns:
            False.

        Example:
            >>> LengthLeaf().elementwise()
            False
        """
        return False

    def selected(self, rows: int) -> InSelect:
        """Give the length in a `select`: one value.

        Args:
            rows: The height of the frame.

        Returns:
            The value.

        Example:
            >>> LengthLeaf().selected(3).length
            1
        """
        del rows
        return InSelect(1, Height.SCALAR, bounded=True)

    def grouped(self, frame: GroupBy) -> InGroups:
        """Give the length over the groups of `agg`: one value for each group.

        Args:
            frame: The groups of the `group_by`.

        Returns:
            The value.

        Example:
            >>> LengthLeaf().grouped(GroupBy((3, 2))).rows
            2
        """
        groups = len(frame.sizes)
        dtype = self.data().dtype
        return InGroups(AggState.AGGREGATED_SCALAR, frame.sizes, Height.SCALAR, groups, dtype=dtype)
