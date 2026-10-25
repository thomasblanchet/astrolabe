"""Evaluate one node of polars whose inputs are leaves, from its definition alone.

A cell is one node of polars with one leaf at each input. A leaf is a
column of a kind in a row form, or a literal. Polars gives such a cell four
answers: the planner type and the run type, in a `select` and in
`group_by().agg()`. Each answer is a data type, or a refusal.

This module computes the four answers from the definition of the node and the
modules of the vocabulary, in the order of the passes of polars:

1. the type function, at the planner types of the inputs;
2. the cast pass: one visit where the node is the root, two visits where it is
   not, as in `agg` under the list that the planner adds;
3. the cast of each input at run time, and the kernel, at the data types;
4. the run in `agg`: the state of each input in the groups, the run of the
   kind of node over those states, and the wrap of the output in a list.

The module holds no rule of any one node. Every rule comes from the definition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from itertools import starmap
from typing import TYPE_CHECKING, ClassVar, Protocol

from astrolabe._rules.cast_pass import (
    CastFold,
    CastOption,
    RootKind,
    SettledCasts,
    StepKind,
    StepOperand,
)
from astrolabe._rules.casts import CastKernel, SeriesCast
from astrolabe._rules.dtypes import DType
from astrolabe._rules.definitions.kernel_adapters import KernelContext, OwnLength
from astrolabe._rules.definitions.node_options import NodeIdentity
from astrolabe._rules.execution import (
    AggState,
    BinaryFold,
    GroupBy,
    InGroups,
    InSelect,
    Kernel,
    LiteralInPlace,
    OverSelect,
    PhysicalExpr,
    Pushdown,
    RollingRun,
    Run,
    SliceInput,
    SlicePushdown,
)
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Undecided, UndecidedError
from astrolabe._rules.kernels.dispatch import Data, RunCast
from astrolabe._rules.kernels.nodes import Mask
from astrolabe._rules.options import Height, KernelLength, NodeKind, RowForm, Shape
from astrolabe._rules.type_functions import ExprFields, Operand, ProjectionSchema

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from astrolabe._rules.cast_pass import NewRoot, Step, Visit
    from astrolabe._rules.definitions.definition import Definition
    from astrolabe._rules.execution import Replacement
    from astrolabe._rules.failures import Stop
    from astrolabe._rules.operators import Operator


@dataclass(frozen=True)
class Refused:
    """Hold how polars refuses a query: the way that it stops, and the class of its error.

    `message` says what polars checks. Two refusals with the same way and the
    same class are equal, whatever their messages, because polars words one
    check in several ways.

    Example:
        >>> from astrolabe._rules.failures import ErrorClass
        >>> Refused.of(PolarsRefusal(ErrorClass.SHAPE, "lengths"))
        raises ShapeError
    """

    stop: Stop
    error: ErrorClass | None = None
    message: str = field(default="", compare=False)

    LITERAL: ClassVar[re.Pattern[str]] = re.compile(r"Unknown\((int|float|str), [^()]*\)")

    def __repr__(self) -> str:
        """Give the way, and the class of the error where polars raises.

        Returns:
            The text.

        Example:
            >>> from astrolabe._rules.failures import Stop
            >>> Refused(Stop.PANICS)
            panics
        """
        if self.error is None:
            return self.stop.value
        return f"{self.stop.value} {self.error.value}"

    def text(self) -> str:
        """Give the reason of the refusal, then how polars stops and the class of its error.

        The message is what the rules check, in their own words. Polars words
        its message in another way, and adds facts of the run, as the name of
        a column, so the text does not quote polars. A literal has a data type
        that the rules write with its value, as `Unknown(int, 2)`. The text
        writes it as polars writes it in its errors, as `dyn int`, because a
        reader knows no other name for it.

        Returns:
            The text.

        Example:
            >>> from astrolabe._rules.failures import ErrorClass, Stop
            >>> Refused.of(PolarsRefusal(ErrorClass.SHAPE, "the lengths differ")).text()
            'the lengths differ, and polars raises ShapeError'
            >>> Refused(Stop.PANICS).text()
            'polars panics'
            >>> message = "arithmetic on dtypes String and Unknown(int, 2) is not allowed"
            >>> refusal = PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
            >>> Refused.of(refusal).text()[:56]
            'arithmetic on dtypes String and dyn int is not allowed, '
        """
        if not self.message:  # pragma: no cover -- each refusal of the rules has a message
            return f"polars {self!r}"
        message = self.LITERAL.sub(r"dyn \1", self.message)
        return f"{message}, and polars {self!r}"

    @classmethod
    def of(cls, refusal: PolarsRefusal) -> Refused:
        """Read the refusal of a rule.

        Args:
            refusal: The refusal.

        Returns:
            The answer.

        Example:
            >>> from astrolabe._rules.failures import Stop
            >>> Refused.of(PolarsRefusal(Stop.CRASHES, "stack")).stop
            <Stop.CRASHES: 'crashes'>
        """
        return cls(refusal.stop, refusal.error, refusal.message)


@dataclass(frozen=True)
class Unchecked:
    """Hold a question that the rules cannot answer, so the answer is not known.

    `cause` says why the rules do not know: a part of polars that the rules
    do not translate, or a fact that no type holds, such as the values of a
    column.

    Example:
        >>> Unchecked("the fill of an Int8 value by a Date fill")
        unchecked: the fill of an Int8 value by a Date fill
    """

    question: str
    cause: Undecided = Undecided.NOT_TRANSLATED

    def __repr__(self) -> str:
        """Give the question.

        Returns:
            The text.

        Example:
            >>> Unchecked("a cast")
            unchecked: a cast
        """
        return f"unchecked: {self.question}"


type Answer = DType | Refused | Unchecked
type Steps = tuple[tuple[Step, ...], ...]


@dataclass(frozen=True)
class CellAnswers:
    """Hold the four answers of a cell: the planner type and the run type, in each context.

    Example:
        >>> int8 = DType("Int8")
        >>> CellAnswers(int8, DType("List", (int8,)), int8, DType("List", (int8,))).planner_agg
        List(Int8)
    """

    planner_select: Answer
    planner_agg: Answer
    run_select: Answer
    run_agg: Answer


@dataclass(frozen=True)
class ContextAnswers:
    """Hold the two answers of one context: the planner type and the run type.

    Example:
        >>> ContextAnswers(DType("Int8"), DType("Int8")).run
        Int8
    """

    planner: Answer
    run: Answer


class Answers:
    """Run one rule and give its answer: a data type, a refusal, or a question it cannot answer.

    Example:
        >>> from astrolabe._rules.failures import ErrorClass
        >>> def refuse() -> DType:
        ...     raise PolarsRefusal(ErrorClass.SCHEMA, "no join")
        >>> Answers().of(refuse)
        raises SchemaError
    """

    def of(self, rule: Callable[[], DType]) -> Answer:
        """Give the answer of the rule.

        Args:
            rule: The rule.

        Returns:
            The type, the refusal, or the question.

        Example:
            >>> Answers().of(lambda: DType("Int8"))
            Int8
        """
        try:
            return rule()
        except (PolarsRefusal, UndecidedError) as error:
            return self.failure(error)

    def failure(self, error: PolarsRefusal | UndecidedError) -> Refused | Unchecked:
        """Give the answer of a rule that did not give a type.

        Args:
            error: The refusal of polars, or the question that the rules cannot answer.

        Returns:
            The refusal, or the question.

        Example:
            >>> Answers().failure(UndecidedError(Undecided.PROCESS, "a category"))
            unchecked: a category
        """
        if isinstance(error, PolarsRefusal):
            return Refused.of(error)
        return Unchecked(error.question, error.cause)


class Leaf(Protocol):
    """Give the facts of one input of a node that each pass of polars reads."""

    def operand(self) -> Operand:
        """Give the input as the type function reads it, before polars folds any literal."""

    def folded(self) -> Operand:
        """Give the input as the cast pass of the root reads it, after polars folds literals."""

    def computed(self) -> Operand:
        """Give the input as a run computes it: a literal of its value, where the code fixes it."""

    def optimized(self) -> Operand:
        """Give the input as the optimizer reads it: a literal where it replaced a node by one."""

    def data(self) -> Data:
        """Give the data that the input gives a kernel in a `select`, before any cast."""

    def grouped_data(self, frame: GroupBy) -> Data:
        """Give the data that the input gives a kernel in `agg`, before any cast.

        Args:
            frame: The groups of the `group_by`.
        """

    def height(self) -> Height:
        """Give the projection height of the input, which decides the wrap in `agg`."""

    def elementwise(self) -> bool:
        """Say whether the input and every node below it work value by value."""

    def selected(self, rows: int) -> InSelect:
        """Give the value of the input in a `select` of a frame.

        Args:
            rows: The height of the frame.
        """

    def grouped(self, frame: GroupBy) -> InGroups:
        """Give the value of the input over the groups of `agg`.

        Args:
            frame: The groups of the `group_by`.
        """


class ColumnForms:
    """Give the node that puts a column in a row form, and the value of that node.

    A column gives one value per row. `.first()` gives one value, `.head(2)`
    another length, and `.head(1)` one row. Each of these is a node of polars,
    so its value in a `select` and in `agg` comes from the run of its kind of
    node (`Run`). A slice also reads its two literal arguments.

    Example:
        >>> ColumnForms().selected(RowForm.OTHER_LENGTH, 3)
        InSelect(length=2, height=<Height.UNKNOWN: 'unknown'>, decided=False, bounded=True)
    """

    NODES: ClassVar[dict[RowForm, tuple[Shape, int]]] = {
        RowForm.ONE_VALUE: (Shape.of(NodeKind.AGG), 0),
        RowForm.OTHER_LENGTH: (Shape(NodeKind.SLICE, length=KernelLength.slice(0, 2)), 2),
        RowForm.ONE_ROW: (Shape(NodeKind.SLICE, length=KernelLength.slice(0, 1)), 2),
    }

    def selected(self, form: RowForm, rows: int) -> InSelect:
        """Give the value of a column in a row form in a `select`.

        Args:
            form: The row form.
            rows: The height of the frame.

        Returns:
            The value.

        Example:
            >>> ColumnForms().selected(RowForm.ONE_VALUE, 3).height
            <Height.SCALAR: 'scalar'>
        """
        column = InSelect.column(rows)
        if form is RowForm.PER_ROW:
            return column
        shape, arguments = self.NODES[form]
        inputs = [column, *[InSelect.literal()] * arguments]
        return Run(shape).select(inputs)

    def grouped(self, form: RowForm, frame: GroupBy, dtype: DType) -> InGroups:
        """Give the value of a column in a row form over the groups of `agg`.

        Args:
            form: The row form.
            frame: The groups of the `group_by`.
            dtype: The data type of the column.

        Returns:
            The value.

        Example:
            >>> ColumnForms().grouped(RowForm.ONE_ROW, GroupBy((3, 2)), DType("Int8")).groups
            (1, 1)
        """
        column = frame.column(dtype)
        if form is RowForm.PER_ROW:
            return column
        shape, arguments = self.NODES[form]
        inputs = [column, *[frame.literal()] * arguments]
        return Run(shape, Kernel(data=dtype)).agg(inputs, frame)


@dataclass(frozen=True)
class ColumnLeaf:
    """Hold an input that reads a column of a frame, in one row form.

    `nulls` says that every value of the column is null. `planned` is the type that a
    lazy frame stores for the column, where it differs from the type of its
    data (`dtype`). Polars plans the node at the stored type, as it plans a
    node at the planner type of an input node, and runs it on the data.
    `bounded` says that the code bounds the rows of the frame: a slice that
    polars moves into the plan keeps no more rows than the frame holds.

    Example:
        >>> leaf = ColumnLeaf(DType("Int8"), RowForm.ONE_VALUE)
        >>> leaf.operand(), leaf.height()
        (Operand(dtype=Int8, literal=False, value=None), <Height.SCALAR: 'scalar'>)
    """

    dtype: DType
    form: RowForm
    nulls: bool = False
    planned: DType | None = None
    bounded: bool = field(default=False, repr=False)

    def operand(self) -> Operand:
        """Give the input as the type function reads it.

        Returns:
            An input of the stored type of the column, which is not a literal.

        Example:
            >>> ColumnLeaf(DType("Date"), RowForm.PER_ROW).operand().dtype
            Date
            >>> stored = ColumnLeaf(DType("Float64"), RowForm.PER_ROW, planned=DType("Float32"))
            >>> stored.operand().dtype
            Float32
        """
        stored = self.dtype if self.planned is None else self.planned
        return replace(Operand.column(stored), scalar=self.form is RowForm.ONE_VALUE)

    def folded(self) -> Operand:
        """Give the input as the cast pass of the root reads it.

        Returns:
            The same input: a column holds no literal to fold.

        Example:
            >>> ColumnLeaf(DType("Date"), RowForm.PER_ROW).folded().literal
            False
        """
        return self.operand()

    def optimized(self) -> Operand:
        """Give the input as the optimizer reads it, after the conversion: the same as `folded`.

        Returns:
            The operand.

        Example:
            >>> ColumnLeaf(DType("Int8"), RowForm.PER_ROW).optimized().literal
            False
        """
        return self.folded()

    def computed(self) -> Operand:
        """Give the input as a run computes it.

        Returns:
            The same input: the values of a column are not in the code.

        Example:
            >>> ColumnLeaf(DType("Date"), RowForm.PER_ROW).computed().value is None
            True
        """
        return self.operand()

    def data(self) -> Data:
        """Give the data of the column.

        Returns:
            Data of the type of the column. A column of the type `Null` holds only nulls.

        Example:
            >>> ColumnLeaf(DType("Null"), RowForm.PER_ROW).data().nulls
            True
        """
        return Data(self.dtype, nulls=self.nulls)

    def grouped_data(self, frame: GroupBy) -> Data:
        """Give the data of the column in `agg`: its data in a `select`.

        Args:
            frame: The groups of the `group_by`, which a column does not read.

        Returns:
            The data.

        Example:
            >>> ColumnLeaf(DType("Int8"), RowForm.PER_ROW).grouped_data(GroupBy((3, 2))).dtype
            Int8
        """
        del frame
        return self.data()

    def height(self) -> Height:
        """Give the projection height of the column in its row form.

        Returns:
            The height.

        Example:
            >>> ColumnLeaf(DType("Int8"), RowForm.ONE_ROW).height()
            <Height.UNKNOWN: 'unknown'>
        """
        return self.form.height()

    def elementwise(self) -> bool:
        """Say whether the input works value by value: a column of the frame does.

        A column in another row form stands for a node over it, such as
        `.first()` or `.head(2)`, which does not.

        Returns:
            Whether the row form is one value per row.

        Example:
            >>> ColumnLeaf(DType("Int8"), RowForm.ONE_VALUE).elementwise()
            False
        """
        return self.form is RowForm.PER_ROW

    def selected(self, rows: int) -> InSelect:
        """Give the value of the column in a `select`.

        Args:
            rows: The height of the frame.

        Returns:
            The value.

        Example:
            >>> ColumnLeaf(DType("Int8"), RowForm.PER_ROW).selected(5).length
            5
            >>> ColumnLeaf(DType("Int8"), RowForm.PER_ROW, bounded=True).selected(2).bounded
            True
        """
        value = ColumnForms().selected(self.form, rows)
        return replace(value, bounded=True) if self.bounded else value

    def grouped(self, frame: GroupBy) -> InGroups:
        """Give the value of the column over the groups of `agg`.

        Args:
            frame: The groups of the `group_by`.

        Returns:
            The value.

        Example:
            >>> ColumnLeaf(DType("Int8"), RowForm.ONE_VALUE).grouped(GroupBy((3, 2))).state
            <AggState.AGGREGATED_SCALAR: 'aggregated scalar'>
        """
        return ColumnForms().grouped(self.form, frame, self.dtype)


@dataclass(frozen=True)
class LiteralLeaf:
    """Hold an input that is a literal: `pl.lit(value)`, or a literal derived from it.

    A derived literal combines the value with a neutral value, as in
    `pl.lit(300) + pl.lit(0)`. The type function of polars reads it as a binary
    node, which is not a literal. Where the node is the root, the conversion of
    polars folds the derived literal into one literal before the cast pass of
    the root reads it (`BinaryFold`), where the two values fold.

    Example:
        >>> from astrolabe._rules.operators import Operator
        >>> LiteralLeaf(300).operand()
        Operand(dtype=Unknown(int, 300), literal=True, value=300)
        >>> LiteralLeaf(300, Operator.PLUS, 0).operand().literal
        False
    """

    value: bool | int | float | str | None
    operator: Operator | None = None
    neutral: bool | int | float | str | None = None

    @classmethod
    def of(cls, operand: Operand) -> LiteralLeaf:
        """Give the literal of the value of a literal operand.

        Args:
            operand: The operand.

        Returns:
            The literal.

        Raises:
            TypeError: The value is not one that a literal of this module holds,
                which is a bug in the caller.

        Example:
            >>> LiteralLeaf.of(Operand.of_value(1.5))
            LiteralLeaf(value=1.5, operator=None, neutral=None)
            >>> LiteralLeaf.of(Operand(DType("Int8"), literal=True, value=(1,)))
            Traceback (most recent call last):
            ...
            TypeError: (1,)
        """
        value = operand.value
        if (  # pragma: no cover -- a fold gives back one of these values
            value is not None and not isinstance(value, bool | int | float | str)
        ):
            raise TypeError(value)
        return cls(value)

    def operand(self) -> Operand:
        """Give the literal as the type function reads it.

        Returns:
            The literal, or the binary node of a derived literal.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> LiteralLeaf(True, Operator.OR, False).operand().dtype
            Boolean
        """
        if self.operator is None:
            return Operand.of_value(self.value)
        left, right = Operand.of_value(self.value), Operand.of_value(self.neutral)
        derived = Operand.column(ExprFields().binary(self.operator, left, right))
        return replace(derived, scalar=True)

    def folded(self) -> Operand:
        """Give the literal as the cast pass of the root reads it.

        Returns:
            The literal, or the fold of a derived literal where its values fold.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> LiteralLeaf(300, Operator.PLUS, 0).folded().literal
            True
            >>> LiteralLeaf("a", Operator.PLUS, "").folded().literal
            False
        """
        if self.operator is None:
            return self.operand()
        left, right = Operand.of_value(self.value), Operand.of_value(self.neutral)
        folded = BinaryFold().binary(self.operator, left, right)
        return self.operand() if folded is None else folded

    def optimized(self) -> Operand:
        """Give the input as the optimizer reads it, after the conversion: the same as `folded`.

        Returns:
            The operand.

        Example:
            >>> LiteralLeaf(True).optimized().value
            True
        """
        return self.folded()

    def computed(self) -> Operand:
        """Give the literal as a run computes it.

        A derived literal combines its value with a neutral value, so the run
        computes the value itself. Where polars folds the literal, the run
        reads the fold. Otherwise it computes a column of the value, at the
        type of the data, and a cast of that column meets the value.

        Returns:
            The fold, or a literal of the value at the type of its data.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> LiteralLeaf("a", Operator.PLUS, "").computed()
            Operand(dtype=String, literal=True, value='a')
        """
        folded = self.folded()
        if folded.literal:
            return folded
        return Operand.typed_literal(self.value, self.data().dtype)

    def data(self) -> Data:
        """Give the data of the literal: one value at the type that it takes alone.

        Returns:
            The data.

        Example:
            >>> LiteralLeaf(2**40).data()
            Data(dtype=Int64, nulls=False, one_row=False)
        """
        return Data(self.operand().dtype.materialized(), nulls=self.value is None)

    def grouped_data(self, frame: GroupBy) -> Data:
        """Give the data of the literal in `agg`: its data in a `select`.

        Args:
            frame: The groups of the `group_by`, which a literal does not read.

        Returns:
            The data.

        Example:
            >>> LiteralLeaf(None).grouped_data(GroupBy((3, 2))).nulls
            True
        """
        del frame
        return self.data()

    def height(self) -> Height:
        """Give the projection height of the literal.

        Returns:
            One value.

        Example:
            >>> LiteralLeaf(None).height()
            <Height.SCALAR: 'scalar'>
        """
        return Height.SCALAR

    def elementwise(self) -> bool:
        """Say that a literal of one value works value by value.

        Returns:
            True.

        Example:
            >>> LiteralLeaf(1).elementwise()
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
            >>> LiteralLeaf(1).selected(5).length
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
            >>> LiteralLeaf(1).grouped(GroupBy((3, 2))).state
            <AggState.LITERAL_SCALAR: 'literal scalar'>
        """
        return frame.literal(self.data().dtype)


class CastVisits:
    """Give the changes that the cast pass of polars makes to each input of a node.

    The pass reaches the inputs of the root before the root, so the root reads
    its inputs once, after polars folds their literals. It reaches any other
    node before its inputs, and once more after them. That second visit reads
    the inputs after the changes of the first visit. Each visit inserts cast
    nodes, and the pass then visits each of them (`SettledCasts`).

    Example:
        >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
        >>> inputs = [Operand.column(DType("UInt8")), Operand.of_value(1)]
        >>> CastVisits(BinaryOperators.ADD).root(inputs)
        ((), (literal(UInt8),))
    """

    def __init__(self, definition: Definition) -> None:
        """Keep the definition of the node.

        Args:
            definition: The definition.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> CastVisits(Aggregations.SUM)._definition.identity.function
            'Sum'
        """
        self._definition = definition

    def root(self, folded: Sequence[Operand]) -> tuple[tuple[Step, ...], ...]:
        """Give the changes of the one visit of the root.

        It raises `PolarsRefusal` where the visit refuses the plan.

        Args:
            folded: The inputs after polars folds their literals.

        Returns:
            The steps of each input.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> CastVisits(Elementwise.ABS).root([Operand.column(DType("Int8"))])
            ((),)
        """
        return self._settled(folded, again=False).steps

    def inner(
        self,
        planned: Sequence[Operand],
        folded: Sequence[Operand],
    ) -> tuple[tuple[Step, ...], ...]:
        """Give the changes of the two visits of a node that is not the root.

        The first visit reads the inputs before polars folds their literals.
        The second visit reads them after the first visit and after the fold.
        Where the first visit puts a new node in place of the node, the pass
        visits that new node next, so the node has no second visit.

        It raises `PolarsRefusal` where a visit refuses the plan.

        Args:
            planned: The inputs before polars folds their literals.
            folded: The inputs after polars folds their literals.

        Returns:
            The steps of each input, those of the first visit first.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> inputs = [Operand.column(DType("Int8")), Operand.column(DType("Int16"))]
            >>> CastVisits(Elementwise.FILL_NULL).inner(inputs, inputs)
            ((cast(Int16, overflowing),), ())
        """
        first = self._settled(planned, again=False)
        if first.root is not None and first.root.kind is not RootKind.SWAPPED:
            return first.steps
        pairs = zip(folded, first.steps, strict=True)
        after = list(starmap(StepOperand().after, pairs))
        second = self._settled(after, again=True)
        joined = zip(first.steps, second.steps, strict=True)
        return tuple(earlier + later for earlier, later in joined)

    def _settled(self, inputs: Sequence[Operand], *, again: bool) -> Visit:
        """Give one visit, with the visits of the cast nodes that it inserts.

        Args:
            inputs: The inputs as the node reads them at the visit.
            again: Whether the pass visits the node a second time.

        Returns:
            The visit.

        A first visit that replaces the node with a new root gives no steps
        to its inputs, and `new_root` gives that root. A comparison that moves
        its literal to the right keeps its steps.

        Raises:
            UndecidedError: The second visit of the node replaces it with a new
                root, which this module does not follow.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> CastVisits(Aggregations.SUM)._settled([Operand.of_value(1)], again=False).steps
            ((),)
            >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
            >>> inputs = [Operand.column(DType("UInt8")), Operand.of_value(300)]
            >>> CastVisits(BinaryOperators.GT)._settled(inputs, again=True)
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not translated: a new root of the cast pass
        """
        definition = self._definition
        options = definition.shape().options
        visit = definition.casts.visit(inputs, options, again=again)
        if again and visit.root is not None:  # pragma: no cover -- the first visit gives one type
            raise UndecidedError(Undecided.NOT_TRANSLATED, "a new root of the cast pass")
        return SettledCasts().of(visit, inputs)

    def new_root(self, inputs: Sequence[Operand]) -> NewRoot | None:
        """Give the new node that the first visit puts in place of the node, or `None`.

        A comparison that moves its literal to the right stays the same
        comparison, so it gives `None`.

        Args:
            inputs: The inputs as the first visit reads them.

        Returns:
            The new node, or `None`.

        Example:
            >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
            >>> inputs = [Operand.column(DType("Int8")), Operand.of_value(None)]
            >>> CastVisits(BinaryOperators.EQ_MISSING).new_root(inputs)
            is_null
        """
        definition = self._definition
        options = definition.shape().options
        root = definition.casts.visit(inputs, options, again=False).root
        if root is None or root.kind is RootKind.SWAPPED:
            return None
        return root


class StepRun:
    """Run the changes of the cast pass on one input at run time.

    A cast changes the type of the data, with the cast of a series at run
    time (`RunCast`), and it is a node of its own: in `agg`, a strict cast of
    values whose groups a node rebuilt first gathers them into lists (`Run`).
    A new type for a literal changes only the type. An `implode` puts the
    values in a list.

    Example:
        >>> from astrolabe._rules.cast_pass import CastOption, Step
        >>> cast = (Step.cast(DType("Int16"), CastOption.NON_STRICT),)
        >>> StepRun().data(Data(DType("Int8")), Operand.column(DType("Int8")), cast)
        Data(dtype=Int16, nulls=False, one_row=False)
    """

    CAST: ClassVar[Shape] = Shape.of(NodeKind.CAST)
    IMPLODE: ClassVar[Shape] = Shape.of(NodeKind.AGG)

    def data(self, data: Data, operand: Operand, steps: Sequence[Step]) -> Data:
        """Give the data after the steps.

        A literal that the pass casts, or folds into a new type, holds the
        value of that cast. The value is null where it does not fit the
        type (`CastFold.keeps`). A number cast to a struct is a struct value
        whose field is null (`_lost`). An overflowing cast wraps the value, so
        it keeps one. A strict cast of a literal raises where the value does
        not fit.

        It raises `PolarsRefusal` where a cast has no kernel, or where a strict
        cast loses the value of a literal.

        Args:
            data: The data before the steps.
            operand: The input as a run computes it, with the value of a literal.
            steps: The steps, in the order that they apply.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.cast_pass import Step
            >>> literal = (Step.literal(DType("Int8")),)
            >>> StepRun().data(Data(DType("Int64")), Operand.of_value(2**40), literal)
            Data(dtype=Int8, nulls=True, one_row=False)
        """
        for step in steps:
            data = self._step(data, operand, step)
            operand = StepOperand().after(operand, (step,))
        return data

    def _step(self, data: Data, operand: Operand, step: Step) -> Data:
        """Give the data after one step.

        It raises `PolarsRefusal` where a cast has no kernel, or where a strict
        cast loses the value of a literal (`_strict`).

        Args:
            data: The data before the step.
            operand: The input as a run computes it, before the step.
            step: The step.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.cast_pass import CastOption, Step
            >>> overflowing = Step.cast(DType("UInt8"), CastOption.OVERFLOWING)
            >>> StepRun()._step(Data(DType("Int32")), Operand.of_value(300), overflowing)
            Data(dtype=UInt8, nulls=False, one_row=False)
        """
        if step.kind is StepKind.IMPLODE:
            return replace(data.with_dtype(DType("List", (data.dtype,))), some_null=False)
        cast = step.kind is StepKind.CAST
        after = RunCast().of(data, step.dtype) if cast else data.with_dtype(step.dtype)
        self._strict(data, operand, step)
        if cast and step.option is CastOption.OVERFLOWING:
            return after
        return self._lost(after, operand, step.dtype)

    def _lost(self, data: Data, operand: Operand, dtype: DType) -> Data:
        """Give the data after a cast, where the cast of a literal does not keep its value.

        The value becomes null. A cast to a struct of one field puts the value
        in the field, so the field holds the null, and the struct value is not
        null. A column keeps its data: its values are not in the code.

        Args:
            data: The data after the cast.
            operand: The input as a run computes it, with the value of a literal.
            dtype: The type of the cast.

        Returns:
            The data.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> StepRun()._lost(Data(struct), Operand.of_value(300), struct)
            Data(dtype=Struct({'a': Int8}), nulls=False, one_row=False, field_nulls=True)
        """
        if not operand.literal or CastFold().keeps(operand, dtype):
            return data
        if dtype.name == "Struct":
            return replace(data, field_nulls=True)
        return replace(data, nulls=True)

    def _strict(self, data: Data, operand: Operand, step: Step) -> None:
        """Refuse a strict cast that loses a value at every run.

        A strict cast of a column raises on a value that does not fit, and the
        values of a column are a fact of the data. Two cases do not depend on
        the data. The value of a literal is in the code, so the cast meets it
        at every run (`SeriesCast`). And a cast that drops every value at one
        place of the type, as a cast to `Null` does, raises on every series
        that holds a value (`CastKernel.drops_every_value`).

        Args:
            data: The data before the step.
            operand: The input as a run computes it.
            step: The step.

        Raises:
            PolarsRefusal: The strict cast loses a value at every run.

        Example:
            >>> from astrolabe._rules.cast_pass import CastOption, Step
            >>> strict = Step.cast(DType("Int32"), CastOption.STRICT)
            >>> text = Operand.typed_literal("a", DType("String"))
            >>> StepRun()._strict(Data(DType("String")), text, strict)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> nulled = Step.cast(DType("Null"), CastOption.STRICT)
            >>> StepRun()._strict(Data(DType("Int8")), Operand.column(DType("Int8")), nulled)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if step.kind is not StepKind.CAST or step.option is not CastOption.STRICT:
            return
        if operand.literal:
            SeriesCast(data.dtype, step.dtype, strict=True).output([operand.value])
        elif not data.nulls and CastKernel().drops_every_value(data.dtype, step.dtype):
            message = f"a strict cast of {data.dtype!r} to {step.dtype!r} drops every value"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def selected(self, value: InSelect, steps: Sequence[Step]) -> InSelect:
        """Give the value of an input in a `select` after the steps.

        Args:
            value: The value before the steps.
            steps: The steps.

        Returns:
            The value.

        Example:
            >>> from astrolabe._rules.cast_pass import Step
            >>> StepRun().selected(InSelect.column(3), (Step.implode(DType("Int8")),)).height
            <Height.SCALAR: 'scalar'>
        """
        for step in steps:
            if step.kind is not StepKind.LITERAL:
                value = self._run(step).select([value])
        return value

    def grouped(self, value: InGroups, steps: Sequence[Step], frame: GroupBy) -> InGroups:
        """Give the value of an input over the groups of `agg` after the steps.

        Args:
            value: The value before the steps.
            steps: The steps.
            frame: The groups of the `group_by`.

        Returns:
            The value.

        Example:
            >>> from astrolabe._rules.cast_pass import CastOption, Step
            >>> frame = GroupBy((3, 2))
            >>> sliced = ColumnForms().grouped(RowForm.OTHER_LENGTH, frame, DType("Int8"))
            >>> strict = (Step.cast(DType("Int16"), CastOption.STRICT),)
            >>> StepRun().grouped(sliced, strict, frame).state
            <AggState.AGGREGATED_LIST: 'aggregated list'>
        """
        for step in steps:
            if step.kind is not StepKind.LITERAL:
                value = self._run(step).agg([value], frame)
        return value

    def _run(self, step: Step) -> Run:
        """Give the run of the node that a step inserts.

        Args:
            step: A cast or an `implode`.

        Returns:
            The run.

        Example:
            >>> from astrolabe._rules.cast_pass import Step
            >>> StepRun()._run(Step.implode(DType("Int8"))).select([InSelect.column(3)]).length
            1
        """
        if step.kind is StepKind.IMPLODE:
            return Run(self.IMPLODE, Kernel(data=step.dtype, physical=PhysicalExpr.IMPLODE))
        strict = step.option is not None and step.option.name == "STRICT"
        return Run(self.CAST, Kernel(data=step.dtype, strict=strict))


class SingleRows:
    """Mark the data of each input that has exactly one row.

    A kernel spreads such an input beside an input of more rows, and some
    kernels take another path for it (`Data.one_row`). In a `select`, the
    lengths of the inputs decide. In `agg`, the rows of each call of the
    kernel decide (`Run.calls`).

    Example:
        >>> SingleRows().of([1, 3])
        (True, False)
    """

    def of(self, lengths: Sequence[int]) -> tuple[bool, ...]:
        """Say, for each input, whether it has exactly one row.

        Args:
            lengths: The number of rows of each input.

        Returns:
            One flag for each input.

        Example:
            >>> SingleRows().of([1, 1])
            (True, True)
        """
        return tuple(length == 1 for length in lengths)

    def distinct(self, calls: Sequence[Sequence[int]], inputs: int) -> list[tuple[int, ...]]:
        """Give the rows of each input at each call of a kernel, each distinct call once.

        Args:
            calls: The rows of each input at each call.
            inputs: The number of inputs.

        Returns:
            The calls, in their order. With no call, one call of one row for
            each input.

        Example:
            >>> SingleRows().distinct([(1, 3), (1, 3), (2, 2)], 2)
            [(1, 3), (2, 2)]
            >>> SingleRows().distinct([], 2)
            [(1, 1)]
        """
        distinct: list[tuple[int, ...]] = []
        for call in calls:
            if tuple(call) not in distinct:
                distinct.append(tuple(call))
        return distinct or [(1,) * inputs]


@dataclass(frozen=True)
class NodeOutput:
    """Hold the output of a node in a `select`: its rows and its data.

    `literal` is the literal that the optimizer puts in place of the node, as
    `True` for `x | True` where `x` is one value. The optimizer of the node
    above then reads that literal.

    Example:
        >>> NodeOutput(InSelect.literal(), Data(DType("Int8"))).value.length
        1
    """

    value: InSelect
    data: Data
    literal: Operand | None = None


class NodeEvaluation:
    """Give the four answers of a node whose inputs are leaves, from its definition.

    The planner type is the type function at the planner types of the inputs.
    Then each context runs the cast pass and the kernel. The planner of polars,
    `collect_schema()`, reads the tree before the conversion folds literals,
    and a run reads the tree after it. So the cast pass runs twice in each
    context: at the inputs as they are, for the planner, and at the folded
    inputs, for the run. A refusal of the type function refuses everything. A
    refusal of the cast pass for the planner refuses the planner and the run
    of that context, and a refusal for the run refuses the run only. A
    refusal of the kernel refuses the run only.

    Example:
        >>> from astrolabe._rules.definitions.length_preserving import LengthPreserving
        >>> leaf = ColumnLeaf(DType("Decimal", (10, 2)), RowForm.PER_ROW)
        >>> answers = NodeEvaluation(LengthPreserving.CUM_SUM, [leaf], GroupBy((3, 2))).answers()
        >>> answers.run_select, answers.run_agg
        (Decimal(precision=38, scale=2), List(Decimal(precision=10, scale=2)))
    """

    ROOT_TYPE: ClassVar[DType] = DType("Boolean")
    IS_IN: ClassVar[NodeIdentity] = NodeIdentity("Function", "Boolean.IsIn")

    def __init__(
        self,
        definition: Definition,
        leaves: Sequence[Leaf],
        frame: GroupBy,
        *,
        root: bool = True,
        slice_inputs: Sequence[SliceInput] | None = None,
    ) -> None:
        """Keep the node, its inputs and the groups of `agg`.

        Args:
            definition: The definition of the node.
            leaves: One leaf for each input, in the order of polars.
            frame: The groups of the `group_by`. The frame of a `select` has their rows.
            root: Whether the node is the root of its expression. The conversion
                rewrites the inputs of the root only, and the cast pass visits
                any other node twice.
            slice_inputs: The inputs of a slice as the slice pushdown of a lazy
                frame reads them, or `None` where no pushdown runs.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))._rows
            3
        """
        self._definition = definition
        self._shape = definition.shape()
        self._frame = frame
        self._slice_inputs = slice_inputs
        self._rows = sum(frame.sizes)
        self._visits = CastVisits(definition)
        self._root = root
        self._planned = [leaf.operand() for leaf in leaves]
        self._heights = [leaf.height() for leaf in leaves]
        self._leaves = self._converted(leaves) if root else list(leaves)
        self._folded = [leaf.folded() for leaf in self._leaves]
        self._computed = [leaf.computed() for leaf in self._leaves]

    def _converted(self, leaves: Sequence[Leaf]) -> list[Leaf]:
        """Give the inputs as the conversion rewrites them before the cast pass of the node.

        `collect_schema()` does not rewrite them, so the planner type reads the
        inputs as they are. A literal that the conversion rewrites becomes a
        literal of its new value.

        Args:
            leaves: The inputs.

        Returns:
            The inputs after the rewrite.

        Example:
            >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
            >>> leaves = [LiteralLeaf(1), LiteralLeaf(1.5)]
            >>> evaluation = NodeEvaluation(BinaryOperators.XOR, leaves, GroupBy((3,)))
            >>> evaluation._converted(leaves)[0]
            LiteralLeaf(value=1.0, operator=None, neutral=None)
        """
        folded = [leaf.folded() for leaf in leaves]
        rewritten = self._definition.simplify.rewritten(folded)
        converted: list[Leaf] = []
        for leaf, before, after in zip(leaves, folded, rewritten, strict=True):
            converted.append(leaf if after == before else LiteralLeaf.of(after))
        return converted

    def answers(self) -> CellAnswers:
        """Give the four answers.

        Returns:
            The answers.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaf = ColumnLeaf(DType("String"), RowForm.PER_ROW)
            >>> NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3, 2))).answers().run_select
            raises InvalidOperationError
        """
        planner = Answers().of(self._planner)
        if not isinstance(planner, DType):
            return CellAnswers(planner, planner, planner, planner)
        select = self._select(planner)
        grouped = self._agg(planner)
        return CellAnswers(select.planner, grouped.planner, select.run, grouped.run)

    def planner(self) -> DType:
        """Give the planner type of the node, for the node above it.

        It raises `PolarsRefusal` where the conversion or the type function refuses the node.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3,)), root=False).planner()
            Int8
        """
        return self._planner()

    def inner_steps(self) -> Steps:
        """Give the steps of the two visits of the cast pass to a node that is not the root.

        It raises `PolarsRefusal` where a visit refuses the plan.

        Returns:
            The steps of each input.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaves = [ColumnLeaf(DType("Int8"), RowForm.PER_ROW), LiteralLeaf(300)]
            >>> fill = NodeEvaluation(Elementwise.FILL_NULL, leaves, GroupBy((3,)), root=False)
            >>> fill.inner_steps()
            ((cast(Int16, overflowing),), (literal(Int16),))
        """
        return self._visits.inner(self._planned, self._folded)

    def literal_fold(self) -> Operand | None:
        """Give the literal that the run computes for a node of literals, or `None` for any other.

        The run of a binary node of two literals computes their fold wherever
        the node stands, so a rule of the node above that reads the value of a
        literal input meets that value. The conversion folds such a node only
        where it is the root, so the planner and the cast pass read the node.

        Returns:
            The fold, or `None`.

        Example:
            >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
            >>> leaves = [LiteralLeaf(1), LiteralLeaf(2)]
            >>> plus = NodeEvaluation(BinaryOperators.ADD, leaves, GroupBy((3,)), root=False)
            >>> plus.literal_fold()
            Operand(dtype=Unknown(int, 3), literal=True, value=3)
        """
        return self._definition.simplify.folded(self._computed)

    def literal_value(self, dtype: DType) -> Operand | None:
        """Give the literal that the run computes for a node over a literal, or `None`.

        A node such as `first` or `str.join` runs over the one value of a
        literal, and a later rule that reads the value meets the value that
        the node computes (`ValueRule`).

        Args:
            dtype: The type of the node after its visits.

        Returns:
            The literal, or `None` where the rules do not know the value.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> first = NodeEvaluation(Aggregations.FIRST, [LiteralLeaf("a")], GroupBy((3,)))
            >>> first.literal_value(DType("String")).value
            'a'
        """
        rule = self._definition.computes
        return None if rule is None else rule.of(self._computed, dtype)

    def after(self) -> DType:
        """Give the type of the node after its visits of the cast pass, which the node above reads.

        It raises `PolarsRefusal` where a visit or the type function refuses.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaves = [ColumnLeaf(DType("Int8"), RowForm.PER_ROW), LiteralLeaf(300)]
            >>> NodeEvaluation(Elementwise.FILL_NULL, leaves, GroupBy((3,)), root=False).after()
            Int16
        """
        return self._after(self.inner_steps())

    def selected(self) -> NodeOutput:
        """Give the output of a node that is not the root, in a `select`.

        It raises `PolarsRefusal` where a step of the run fails.

        Returns:
            The output.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3,)), root=False).selected().data
            Data(dtype=Int8, nulls=False, one_row=False)
        """
        return self._select_output(self.inner_steps(), self._planned)

    def grouped(self) -> InGroups:
        """Give the value of a node that is not the root, over the groups of `agg`.

        It raises `PolarsRefusal` where a step of the run fails.

        Returns:
            The value, with the type of its data.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3,)), root=False).grouped().dtype
            Int8
        """
        return self._agg_output(self.inner_steps(), self._planned)

    def height(self) -> Height:
        """Give the projection height of the node, from the heights of its inputs.

        Returns:
            The height.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> NodeEvaluation(Aggregations.SUM, [leaf], GroupBy((3,))).height()
            <Height.SCALAR: 'scalar'>
        """
        return self._shape.height(self._heights)

    def _planner(self) -> DType:
        """Give the planner type: the checks of the conversion, then the type function.

        The conversion refuses an input that must give one value and does not
        (`Shape.check_inputs`), in both contexts, before polars types the node.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))._planner()
            Unknown(int, 1)
        """
        self._shape.check_inputs(self._heights)
        return self._definition.type_function.planner(self._planned)

    def _select(self, planner: DType) -> ContextAnswers:
        """Give the two answers of a `select`, where the node is the root.

        Args:
            planner: The planner type of the node.

        Returns:
            The answers.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._select(DType.dynamic_int(1))
            ContextAnswers(planner=Int32, run=Int32)
        """
        planned = ProjectionSchema().select(planner)
        return self._context(planned, self._visits.root, self._run_select, self._visited())

    def _visited(self) -> list[Operand]:
        """Give the inputs as the one visit of the root reads them while polars plans a `select`.

        The cast pass visits each node below the root first, so the root
        reads such a node at its type after its own visits. It reads an input
        that folds into a literal as the code writes it, because
        `collect_schema()` folds no literal: `pl.lit(1) + 0` stays a node of
        two literals. A node under the root reads the planner type of each
        input at its first visit, which can refuse where the root accepts.

        Returns:
            The inputs.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> from astrolabe._rules.operators import Operator
            >>> derived = LiteralLeaf(300, Operator.PLUS, 0)
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [derived], GroupBy((3,)))
            >>> evaluation._visited()[0].literal
            False
        """
        pairs = zip(self._planned, self._folded, strict=True)
        return [planned if folded.literal else folded for planned, folded in pairs]

    def _agg(self, planner: DType) -> ContextAnswers:
        """Give the two answers of `agg`.

        The planner wraps the node in a list where its height is not one value.
        The node is then not the root, and the cast pass visits it twice. While
        polars plans, the second visit reads each node below at its type after
        its own visits (`_visited`).

        Args:
            planner: The planner type of the node.

        Returns:
            The answers.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3,)))._agg(DType("Int8"))
            ContextAnswers(planner=List(Int8), run=List(Int8))
        """
        heights = [leaf.height() for leaf in self._leaves]
        wrapped = self._shape.wraps_in_agg(heights)
        planned = ProjectionSchema().agg(planner, scalar=not wrapped)
        return self._context(
            planned,
            lambda inputs: self._agg_steps(inputs, wrapped=wrapped),
            lambda steps: self._run_agg(steps, wrapped=wrapped),
            self._visited(),
        )

    def _context(
        self,
        planned: DType,
        visits: Callable[[Sequence[Operand]], Steps],
        run: Callable[[Steps], DType],
        inputs: Sequence[Operand],
    ) -> ContextAnswers:
        """Give the two answers of one context from the cast pass and the run of that context.

        Args:
            planned: The planner type of the context.
            visits: The cast pass, which gives the steps of each input from the
                inputs as the pass reads them: folded or not.
            run: The run at the steps of the folded inputs.
            inputs: The inputs as the cast pass reads them while polars plans.

        Returns:
            The answers.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [], GroupBy((3,)))
            >>> int8 = DType("Int8")
            >>> evaluation._context(int8, lambda inputs: (), lambda steps: int8, [])
            ContextAnswers(planner=Int8, run=Int8)
        """
        refusal = self._refusal(lambda: visits(inputs))
        if refusal is not None:
            return ContextAnswers(refusal, refusal)
        try:
            steps = visits(self._folded)
        except (PolarsRefusal, UndecidedError) as error:
            return ContextAnswers(planned, Answers().failure(error))
        return ContextAnswers(planned, Answers().of(lambda: run(steps)))

    def _refusal(self, visits: Callable[[], Steps]) -> Refused | Unchecked | None:
        """Give the refusal of the cast pass of the planner, or `None` where it accepts the node.

        Args:
            visits: The cast pass at the inputs before the fold.

        Returns:
            The refusal, the question that the rules cannot answer, or `None`.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [], GroupBy((3,)))
            >>> evaluation._refusal(lambda: ()) is None
            True
        """
        try:
            visits()
        except (PolarsRefusal, UndecidedError) as error:
            return Answers().failure(error)
        return None

    def _agg_steps(self, inputs: Sequence[Operand], *, wrapped: bool) -> Steps:
        """Give the steps of the cast pass in `agg`.

        Args:
            inputs: The inputs as the pass reads them after its first visit:
                folded or not.
            wrapped: Whether the planner wraps the node, so that it is not the root.

        Returns:
            The steps of each input.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._agg_steps([Operand.of_value(1)], wrapped=True)
            ((),)
        """
        if not wrapped:
            return self._visits.root(inputs)
        return self._visits.inner(self._planned, inputs)

    def _run_select(self, steps: Steps) -> DType:
        """Give the run type in a `select`.

        Where the conversion folds the node into a literal, the run gives that
        literal. Where the cast pass puts a new node in place of the node, the
        run gives the `Boolean` of that node. Otherwise polars first computes
        the type of the node after the cast pass, when it converts the
        expression, before any data runs (`_after`). Where the optimizer then
        puts an input or a literal in place of the node, the run gives it
        (`_replacement`). Otherwise polars runs each input with its casts,
        checks the lengths of the inputs, and runs the kernel. A kernel that
        checks the lengths itself, with an error of its own, runs with no
        check before it (`KernelRule.checks_lengths`). It raises
        `PolarsRefusal` at the first of these that fails.

        Args:
            steps: The steps of the cast pass on each input.

        Returns:
            The type of the data.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaf = ColumnLeaf(DType("Boolean"), RowForm.PER_ROW)
            >>> NodeEvaluation(Aggregations.SUM, [leaf], GroupBy((3,)))._run_select(((),))
            UInt32
            >>> sliced = ColumnLeaf(DType("Float64"), RowForm.OTHER_LENGTH)
            >>> evaluation = NodeEvaluation(Elementwise.LOG, [leaf, sliced], GroupBy((3,)))
            >>> evaluation._run_select(((), ()))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot do a binary ...
        """
        return self._select_output(steps, self._folded).data.dtype

    def _select_output(self, steps: Steps, visited: Sequence[Operand]) -> NodeOutput:
        """Give the output of the node in a `select`: its value and its data.

        It follows the order of `_run_select`, and raises `PolarsRefusal` at
        the first step that fails. Only the root folds.

        Args:
            steps: The steps of the cast pass on each input.
            visited: The inputs as the first visit of the cast pass reads them.

        Returns:
            The output.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._select_output(((),), [Operand.of_value(1)]).value.length
            1
        """
        rewritten = self._rewritten_select(steps, visited)
        if rewritten is not None:
            return rewritten
        if self._shape.kind in {NodeKind.OVER, NodeKind.ROLLING}:
            return self._window_output(steps)
        states = self._selected_states(steps)
        lengths = tuple(state.length for state in states)
        bounds = [state.length if state.bounded else None for state in states]
        if self._shape.kind is NodeKind.TERNARY:
            self._mask_first(steps)
        datas = self._datas(steps, SingleRows().of(lengths), grouped=False, bounds=bounds)
        if self._pushed_away():
            return NodeOutput(states[0], datas[0])
        return self._kernel_output(states, datas, steps)

    def _kernel_output(
        self,
        states: Sequence[InSelect],
        datas: Sequence[Data],
        steps: Steps,
    ) -> NodeOutput:
        """Give the output of the kernel of the node in a `select`: its value and its data.

        A kernel that checks the lengths of its inputs itself runs before the
        check of the rows of the node. The data holds only nulls where the
        rules of the definition say so.

        Args:
            states: The value of each input.
            datas: The data of each input.
            steps: The steps of the cast pass on each input.

        Returns:
            The output.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> literal = InSelect.literal()
            >>> evaluation._kernel_output([literal], [Data(DType("Int8"))], ((),)).data.dtype
            Int8
        """
        kernel = self._definition.kernel
        node = Run(self._shape, Kernel(physical=self._definition.physical_expr()))
        value = None if kernel.checks_lengths() else node.select(states)
        dtype = kernel.output(datas, self._select_context(states, steps))
        if value is None:
            value = self._settled_value(node, states)
        value = self._owned(value, datas, states)
        rows = states[0].length if states and states[0].bounded else None
        nulls = self._definition.nulls.output(datas, dtype) or self._short_nulls(rows)
        return NodeOutput(value, Data(dtype, nulls=nulls, some_null=self._some_null(datas, dtype)))

    def _some_null(self, datas: Sequence[Data], dtype: DType) -> bool:
        """Say whether the output holds a null wherever it holds a row, as the code decides it.

        A node gives such a null where its rule of few rows says so, as
        `shift(1)` does in its first row (`ShortNulls`). A node that works
        value by value keeps such a null of an input that its rule of nulls
        reads (`NullRule`).

        Args:
            datas: The data of each input.
            dtype: The type of the output.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._some_null([Data(DType("Int8"), some_null=True)], DType("Int8"))
            True
        """
        short = self._definition.short
        if short is not None and short.some_null(self._computed):
            return True
        return self._shape.is_elementwise() and self._definition.nulls.held(datas, dtype)

    def _short_nulls(self, rows: int | None) -> bool:
        """Say whether the node gives only nulls because its first input holds few rows.

        Args:
            rows: The most rows of the first input, or `None` where the code bounds none.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))._short_nulls(1)
            False
        """
        short = self._definition.short
        return short is not None and rows is not None and short.all_null(rows, self._computed)

    def _selected_states(self, steps: Steps) -> list[InSelect]:
        """Give the value of each input in a `select`, after the steps of the cast pass.

        Args:
            steps: The steps of each input.

        Returns:
            The values.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._selected_states(((),))[0].length
            1
        """
        run = StepRun()
        pairs = zip(self._leaves, steps, strict=True)
        return [run.selected(leaf.selected(self._rows), changes) for leaf, changes in pairs]

    def _select_context(self, states: Sequence[InSelect], steps: Steps) -> KernelContext:
        """Give how the data reaches the kernel in a `select`.

        Args:
            states: The value of each input.
            steps: The steps of each input.

        Returns:
            The context.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._select_context([InSelect.literal()], ((),)).scalars
            (True,)
        """
        return KernelContext(
            grouped=False,
            scalars=tuple(state.height is Height.SCALAR for state in states),
            lengths=tuple(state.length for state in states),
            planned=self._node_type(),
            computed=self._computed_after(steps),
            decided=tuple(state.decided for state in states),
        )

    def _owned(
        self,
        value: InSelect,
        datas: Sequence[Data],
        states: Sequence[InSelect],
    ) -> InSelect:
        """Give the rows of a kernel that gives the rows of its first input at some types of data.

        Such a kernel (`OwnLength`) keeps whether the values decide those rows.

        Args:
            value: The output by the rule of the node.
            datas: The data of each input.
            states: The value of each input.

        Returns:
            The output.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> literal = InSelect.literal()
            >>> evaluation._owned(literal, [Data(DType("Int8"))], [literal]) == literal
            True
        """
        kernel = self._definition.kernel
        if not isinstance(kernel, OwnLength):
            return value
        lengths = [state.length for state in states]
        own = kernel.output_length(datas, lengths)
        if own is None:
            return value
        return replace(value, length=own, decided=states[0].decided)

    def _mask_first(self, steps: Steps) -> None:
        """Refuse the mask of a `when` in a `select` before the casts of its branches run.

        Polars computes the mask and checks that it is `Boolean` before it
        computes `then` and `otherwise`. In `agg` it computes the three
        inputs first, so a cast of a branch fails before the check. The check
        raises `PolarsRefusal` where the mask is not `Boolean` (`Mask`).

        Args:
            steps: The steps of the cast pass on each input.

        Example:
            >>> from astrolabe._rules.definitions.node_kinds import NodeKinds
            >>> mask = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> leaves = [mask, LiteralLeaf(1), LiteralLeaf(2)]
            >>> evaluation = NodeEvaluation(NodeKinds.WHEN, leaves, GroupBy((3,)))
            >>> evaluation._mask_first(((), (), ()))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: invalid series dtype: ...
        """
        mask = StepRun().data(self._leaves[0].data(), self._computed[0], steps[0])
        Mask().read(mask)

    def _pushed_away(self) -> bool:
        """Say whether the slice pushdown of a lazy frame deletes this slice, so it gives its value.

        A slice whose offset or length is a column, over a value whose root
        does not pass a slice down, leaves the tree, and the node gives its
        value unchanged. Where a place below the value reads the offset or
        the length, the query never returns, and the method raises the
        `PolarsRefusal` of that hang.

        Returns:
            Whether the pushdown deletes the node.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))._pushed_away()
            False
        """
        if self._slice_inputs is None:
            return False
        return SlicePushdown().outcome(*self._slice_inputs) is Pushdown.DELETED

    def _rewritten_select(self, steps: Steps, visited: Sequence[Operand]) -> NodeOutput | None:
        """Give the output in a `select` where polars puts something in place of the node.

        The conversion folds the root of two literals into a literal. The cast
        pass can put a new node in place of a comparison, which gives the
        `Boolean` of that node over the rows of the input that is not the
        literal. Polars then computes the type of the node after the cast
        pass, which can refuse. The optimizer can put an input or a literal in
        place of the node.

        Args:
            steps: The steps of the cast pass on each input.
            visited: The inputs as the first visit of the cast pass reads them.

        Returns:
            The output, or `None` where the node stays.

        Example:
            >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
            >>> leaves = [LiteralLeaf(1), LiteralLeaf(2)]
            >>> evaluation = NodeEvaluation(BinaryOperators.ADD, leaves, GroupBy((3,)))
            >>> literals = [Operand.of_value(1), Operand.of_value(2)]
            >>> evaluation._rewritten_select(((), ()), literals).data.dtype
            Int32
        """
        folded = self._definition.simplify.folded(visited) if self._root else None
        if folded is not None:
            return NodeOutput(InSelect.literal(), Data(folded.dtype.materialized()))
        if self._visits.new_root(visited) is not None:
            other = self._leaves[1 if visited[0].literal and not visited[1].literal else 0]
            if other.height() is Height.SCALAR:
                return NodeOutput(InSelect.literal(), Data(self.ROOT_TYPE))
            return NodeOutput(other.selected(self._rows), Data(self.ROOT_TYPE))
        self._after(steps)
        replacement = self._replacement(steps) if self._frame.flags.simplify_select else None
        if replacement is None:
            return None
        data = self._replaced_data(replacement, steps)
        value = self._replaced_value(replacement, steps)
        return NodeOutput(value, data, self._replaced_literal(replacement, steps))

    def _replaced_literal(self, replacement: Replacement, steps: Steps) -> Operand | None:
        """Give the literal that the optimizer puts in place of the node, where it puts one.

        The optimizer puts a literal in place of the node, or an input that
        can be a literal itself.

        Args:
            replacement: The input or the literal.
            steps: The steps of the cast pass on each input.

        Returns:
            The literal, or `None`.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> from astrolabe._rules.execution import InputInPlace
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(True)], GroupBy((3,)))
            >>> evaluation._replaced_literal(InputInPlace(0), ((),)).value
            True
        """
        if isinstance(replacement, LiteralInPlace):
            return replacement.literal
        index = replacement.index
        operand = StepOperand().after(self._leaves[index].optimized(), steps[index])
        return operand if operand.literal else None

    def _window_output(self, steps: Steps) -> NodeOutput:
        """Give the output of an `over` or a `rolling` in a `select`: its function runs as in `agg`.

        Polars runs the function of a window over groups, as it runs an
        expression in `agg`, even inside a `select`. So the mean of text gives
        text there, and the function reads the data of its inputs in groups.
        The keys and the index of the window run as in a `select`.

        It raises `PolarsRefusal` where a key, a cast or the kernel fails.

        Args:
            steps: The steps of the cast pass on each input.

        Returns:
            The output.

        Example:
            >>> from astrolabe._rules.definitions.node_kinds import NodeKinds
            >>> from astrolabe._rules.definitions.node_options import NodeOptions
            >>> body = {"function": {"Column": "x"}, "partition_by": [], "mapping": "GroupsToRows"}
            >>> column = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> definition = NodeKinds.OVER.at(NodeOptions({"Over": body}))
            >>> over = NodeEvaluation(definition, [column, column], GroupBy((3, 2)))
            >>> over._window_output(((), ())).data.dtype
            Int8
        """
        run = StepRun()
        keys: list[InSelect] = []
        for leaf, changes in zip(self._leaves[1:], steps[1:], strict=True):
            keys.append(run.selected(leaf.selected(self._rows), changes))
        datas = self._datas(steps, [False] * len(self._leaves), grouped=True)
        return self._over_output(steps, keys, datas)

    def _over_output(
        self,
        steps: Steps,
        keys: Sequence[InSelect],
        datas: Sequence[Data],
    ) -> NodeOutput:
        """Give the output of an `over` or a `rolling` in a `select`, from its function.

        The function runs over the groups of the keys, which the rule takes as
        the groups of `agg` (`OverSelect`). A `rolling` runs it over a window
        for each row, and gives one row for each row of the frame
        (`RollingRun`). A function that gives one value for each group
        gives that value, and the kernel reads that fact.

        Args:
            steps: The steps of the cast pass on each input.
            keys: The value of each input after the function in the `select`:
                the keys of an `over`, or the index of a `rolling`.
            datas: The data of each input.

        Returns:
            The output.

        Example:
            >>> from astrolabe._rules.definitions.node_kinds import NodeKinds
            >>> from astrolabe._rules.definitions.node_options import NodeOptions
            >>> body = {"function": {"Column": "x"}, "partition_by": [], "mapping": "GroupsToRows"}
            >>> tree = {"Over": body}
            >>> column = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> leaves = [column, column]
            >>> over = NodeEvaluation(NodeKinds.OVER.at(NodeOptions(tree)), leaves, GroupBy((3, 2)))
            >>> keys = [InSelect.column(5)]
            >>> over._over_output(((), ()), keys, [leaf.data() for leaf in leaves]).data
            Data(dtype=Int8, nulls=False, one_row=False)
        """
        function = StepRun().grouped(self._leaves[0].grouped(self._frame), steps[0], self._frame)
        if self._shape.kind is NodeKind.OVER:
            value = self._over_select(function, keys)
        else:
            value = RollingRun().select(keys[0], self._frame)
        spread = function.state in {AggState.AGGREGATED_SCALAR, AggState.LITERAL_SCALAR}
        scalars = (spread,) + (False,) * len(keys)
        context = KernelContext(grouped=False, scalars=scalars)
        return NodeOutput(value, Data(self._definition.kernel.output(datas, context)))

    def _over_select(self, function: InGroups, keys: Sequence[InSelect]) -> InSelect:
        """Give the value of an `over` in a `select`, over the groups that its keys make.

        A key of the rows of the frame makes groups that its values decide,
        and the rule takes them as the groups of `agg`. Where every key has one
        row, the keys make one group of the whole frame. The function then
        runs over all the rows, so it has the rows that it has in the `select`.

        Args:
            function: The value of the function over the groups of `agg`.
            keys: The value of each key in the `select`.

        Returns:
            The value.

        Example:
            >>> from astrolabe._rules.definitions.node_kinds import NodeKinds
            >>> from astrolabe._rules.definitions.node_options import NodeOptions
            >>> body = {"function": {"Column": "x"}, "partition_by": [], "mapping": "Explode"}
            >>> rules = NodeKinds.OVER.at(NodeOptions({"Over": body}))
            >>> leaves = [ColumnLeaf(DType("Int8"), RowForm.PER_ROW), LiteralLeaf(1)]
            >>> frame = GroupBy((3, 2))
            >>> over = NodeEvaluation(rules, leaves, frame)
            >>> over._over_select(frame.column(), [InSelect.literal()]).length
            5
        """
        over = OverSelect(explode=self._shape.explode, joined=self._shape.joined)
        if any(key.length != 1 for key in keys):
            return over.select(function, keys, self._frame)
        whole = self._leaves[0].selected(self._rows).length
        alone = replace(function, groups=(whole,), rows=whole)
        return over.select(alone, keys, GroupBy((self._rows,), self._frame.flags))

    def _settled_value(self, node: Run, states: Sequence[InSelect]) -> InSelect:
        """Give the value of a node whose kernel checked the lengths of its inputs itself.

        The kernel accepted the lengths, so the run gives the longest input.

        Args:
            node: The run of the node.
            states: The value of each input.

        Returns:
            The value of the output.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> run = Run(Shape.of(NodeKind.BINARY))
            >>> evaluation._settled_value(run, [InSelect.column(3), InSelect.literal()]).length
            3
        """
        try:
            return node.select(states)
        except PolarsRefusal:
            longest = max(states, key=lambda state: state.length)
            return InSelect(longest.length, longest.height)

    def _replaced_value(self, replacement: Replacement, steps: Steps) -> InSelect:
        """Give the value in a `select` of what the optimizer puts in place of the node.

        Args:
            replacement: The input or the literal.
            steps: The steps of the cast pass on each input.

        Returns:
            The value.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> from astrolabe._rules.execution import InputInPlace
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._replaced_value(InputInPlace(0), ((),)).length
            1
        """
        if isinstance(replacement, LiteralInPlace):
            return InSelect.literal()
        index = replacement.index
        leaf = self._leaves[index]
        return StepRun().selected(leaf.selected(self._rows), steps[index])

    def _run_agg(self, steps: Steps, *, wrapped: bool) -> DType:
        """Give the run type in `agg`: the type of each group, in a list where a group holds a list.

        Where the conversion folds the node into a literal, the run gives that
        literal for each group. The conversion reads the inputs folded only
        where the node is the root. Where the cast pass puts a new node in
        place of the node, the run gives the `Boolean` of that node, with the
        rows of the input that is not the literal. Otherwise polars first computes the type
        of the node after the cast pass, when it converts the expression
        (`_after`). Where the optimizer then puts an input or a literal in
        place of the node, the run gives it (`_replacement`). Otherwise polars
        runs each input with its casts, and the run of the node checks the
        groups of its inputs and calls the kernel. Last, a node that runs once
        for each group builds the list of each group at that type. It raises
        `PolarsRefusal` at the first of these that fails.

        Args:
            steps: The steps of the cast pass on each input.
            wrapped: Whether the planner wraps the node in a list.

        Returns:
            The type of the column of `agg`.

        Example:
            >>> from astrolabe._rules.definitions.aggregations import Aggregations
            >>> leaf = ColumnLeaf(DType("Date"), RowForm.ONE_VALUE)
            >>> evaluation = NodeEvaluation(Aggregations.SUM, [leaf], GroupBy((3, 2)))
            >>> evaluation._run_agg(((),), wrapped=False)
            Date
        """
        visited = self._planned if wrapped else self._folded
        value = self._agg_output(steps, visited)
        return self._column(value, value.dtype or self.ROOT_TYPE, wrapped=wrapped)

    def _agg_output(self, steps: Steps, visited: Sequence[Operand]) -> InGroups:
        """Give the value of the node over the groups of `agg`, with the type of its data.

        It follows the order of `_run_agg`, and raises `PolarsRefusal` at the
        first step that fails. Only the root folds.

        Args:
            steps: The steps of the cast pass on each input.
            visited: The inputs as the first visit of the cast pass reads them.

        Returns:
            The value, whose type is the type of the data that the node gives.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3,)))
            >>> evaluation._agg_output(((),), [leaf.operand()]).dtype
            Int8
        """
        rewritten = self._rewritten_agg(steps, visited)
        if rewritten is not None:
            return rewritten
        after = self._after(steps)
        run = StepRun()
        states: list[InGroups] = []
        for leaf, changes in zip(self._leaves, steps, strict=True):
            states.append(run.grouped(leaf.grouped(self._frame), changes, self._frame))
        if self._shape.kind is NodeKind.ROLLING:
            return self._rolling_agg(steps, states)
        if self._shape.kind is NodeKind.OVER:
            return self._over_agg(steps, states)
        reads = self._cast_reads(states, steps)
        plain = self._datas(steps, [False] * len(states), grouped=True, reads=reads)
        if self._pushed_away():
            return replace(states[0], dtype=plain[0].dtype)
        fallible = self._definition.kernel.fallible(plain)
        node = self._shape
        # The run checks the groups of the inputs before the kernel runs. The
        # run below then checks the output of the kernel, and builds the lists
        # at the type that the kernel gives.
        own_lengths = self._definition.kernel.checks_group_lengths()
        before = Kernel(
            fallible=fallible,
            physical=self._definition.physical_expr(),
            before=True,
            own_lengths=own_lengths,
        )
        Run(node, before).agg(states, self._frame)
        data = self._grouped_data(steps, states, fallible=fallible)
        facts = Kernel(after, data, fallible=fallible, physical=before.physical)
        value = Run(node, facts).agg(states, self._frame)
        if self._keeps_rows(plain, states):
            value = replace(value, decided=states[0].decided)
        return value if value.dtype is not None else replace(value, dtype=data)

    def _keeps_rows(self, datas: Sequence[Data], states: Sequence[InGroups]) -> bool:
        """Say whether the kernel gives the rows of its first input at the types of the inputs.

        Such a kernel (`OwnLength`) keeps whether the values decide those rows.

        Args:
            datas: The data of each input.
            states: The value of each input over the groups.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> frame = GroupBy((3, 2))
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], frame)
            >>> evaluation._keeps_rows([Data(DType("Int8"))], [frame.column()])
            False
        """
        kernel = self._definition.kernel
        if not isinstance(kernel, OwnLength):
            return False
        rows = tuple(state.rows for state in states)
        return kernel.output_length(datas, rows) is not None

    def grouped_nulls(self, dtype: DType) -> bool:
        """Say whether the output of the node in `agg` holds only nulls, by its rule (`NullRule`).

        Args:
            dtype: The type of the output.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaf = ColumnLeaf(DType("Null"), RowForm.PER_ROW)
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3, 2)))
            >>> evaluation.grouped_nulls(DType("Null"))
            True
        """
        datas = [leaf.grouped_data(self._frame) for leaf in self._leaves]
        if self._definition.nulls.output(datas, dtype):
            return True
        if not self._leaves or self._definition.short is None:
            return False
        first = self._leaves[0].grouped(self._frame)
        if not first.bounded and first.state not in InGroups.SINGLE:
            return False
        return self._short_nulls(max(first.values(), default=0))

    def grouped_some_null(self, dtype: DType) -> bool:
        """Say whether the output of the node in `agg` holds a null wherever it holds a row.

        Args:
            dtype: The type of the output.

        Returns:
            Whether it does (`_some_null`), in each group.

        Example:
            >>> from astrolabe._rules.definitions.length_preserving import LengthPreserving
            >>> leaves = [ColumnLeaf(DType("Int8"), RowForm.PER_ROW), LiteralLeaf(1)]
            >>> evaluation = NodeEvaluation(LengthPreserving.SHIFT, leaves, GroupBy((3, 2)))
            >>> evaluation.grouped_some_null(DType("Int8"))
            True
        """
        datas = [leaf.grouped_data(self._frame) for leaf in self._leaves]
        return self._some_null(datas, dtype)

    def elementwise(self) -> bool:
        """Say whether the node and every node below it work value by value.

        `is_in` works value by value only where its set of values is a
        literal, and it then does not read the set as an input that works
        value by value: the set is the whole literal.

        Returns:
            Whether they do.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> leaf = ColumnLeaf(DType("Int8"), RowForm.PER_ROW)
            >>> NodeEvaluation(Elementwise.ABS, [leaf], GroupBy((3,))).elementwise()
            True
        """
        leaves = self._leaves
        if self._definition.identity == self.IS_IN:
            if not leaves[1].computed().literal:
                return False
            leaves = leaves[:1]
        return self._shape.is_elementwise() and all(leaf.elementwise() for leaf in leaves)

    def _over_agg(self, steps: Steps, states: Sequence[InGroups]) -> InGroups:
        """Give the value of an `over` over the groups of `agg`, with the type of its data.

        Polars runs an `over` in `agg` only where each key is elementwise: a
        column, a literal, or an elementwise function of them, at every node
        (`Leaf.elementwise`). The function runs over the subgroups that the
        keys make in each group. A
        function that does not give one value for each subgroup must keep the
        rows of each subgroup: the rule checks this where every key is a
        literal, so that the subgroups are the groups. The output has the rows
        of each group.

        Args:
            steps: The steps of the cast pass on each input.
            states: The value of each input over the groups.

        Returns:
            The value.

        Raises:
            PolarsRefusal: A key is not elementwise, or the function does not
                keep the rows of a group.
            UndecidedError: The mapping is `explode`, which this module does not translate in `agg`.

        Example:
            >>> from astrolabe._rules.definitions.node_kinds import NodeKinds
            >>> from astrolabe._rules.definitions.node_options import NodeOptions
            >>> body = {"function": {"Column": "x"}, "partition_by": [], "mapping": "GroupsToRows"}
            >>> rules = NodeKinds.OVER.at(NodeOptions({"Over": body}))
            >>> leaves = [ColumnLeaf(DType("Int8"), RowForm.PER_ROW), LiteralLeaf(1)]
            >>> frame = GroupBy((3, 2))
            >>> evaluation = NodeEvaluation(rules, leaves, frame)
            >>> evaluation._over_agg(((), ()), [frame.column(), frame.literal()]).dtype
            Int8
            >>> from dataclasses import replace
            >>> fewer = replace(frame.column(), groups=(1, 1))
            >>> evaluation._over_agg(((), ()), [fewer, frame.literal()])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: the length of the window ...
            >>> joined = NodeKinds.OVER.at(NodeOptions({"Over": {**body, "mapping": "Join"}}))
            >>> joining = NodeEvaluation(joined, leaves, frame)
            >>> joining._over_agg(((), ()), [fewer, frame.literal()]).dtype
            List(Int8)
            >>> body = {**body, "mapping": "Explode"}
            >>> rules = NodeKinds.OVER.at(NodeOptions({"Over": body}))
            >>> exploded = NodeEvaluation(rules, leaves, frame)
            >>> exploded._over_agg(((), ()), [frame.column(), frame.literal()])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not translated: an `over` with explode ...
        """
        if self._shape.explode:
            raise UndecidedError(Undecided.NOT_TRANSLATED, "an `over` with explode in `agg`")
        for leaf in self._leaves[1:]:
            if not leaf.elementwise():
                message = "window expression with non-elementwise `partition_by` not allowed"
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        function = states[0]
        spread = function.state in {AggState.AGGREGATED_SCALAR, AggState.LITERAL_SCALAR}
        literal_keys = all(leaf.computed().literal for leaf in self._leaves[1:])
        checked = not spread and not self._shape.joined
        if checked and literal_keys and function.groups != self._frame.sizes:
            message = "the length of the window expression did not match that of the group"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        scalars = (spread,) + (False,) * (len(states) - 1)
        context = KernelContext(grouped=True, scalars=scalars)
        datas = self._datas(steps, [False] * len(states), grouped=True)
        value = replace(self._frame.column(), original=False)
        return replace(value, dtype=self._definition.kernel.output(datas, context))

    def _rolling_agg(self, steps: Steps, states: Sequence[InGroups]) -> InGroups:
        """Give the value of a `rolling` over the groups of `agg`, with the type of its data.

        The kernel checks the index first. The function then runs over a
        window for each row of each group (`RollingRun`). A function that
        gives one value for each window gives that value, and the kernel reads
        that fact.

        Args:
            steps: The steps of the cast pass on each input.
            states: The value of each input over the groups.

        Returns:
            The value.

        Example:
            >>> from astrolabe._rules.definitions.node_kinds import NodeKinds
            >>> from astrolabe._rules.definitions.node_options import NodeOptions
            >>> from astrolabe._rules.tree import ExpressionTree
            >>> import polars as pl
            >>> tree = ExpressionTree().of(pl.col("x").rolling(index_column="t", period="2i"))
            >>> rules = NodeKinds.ROLLING.at(NodeOptions(tree.serialized))
            >>> leaves = [ColumnLeaf(DType("Int8"), RowForm.PER_ROW)] * 2
            >>> frame = GroupBy((3, 2))
            >>> evaluation = NodeEvaluation(rules, leaves, frame)
            >>> evaluation._rolling_agg(((), ()), [frame.column(), frame.column()]).dtype
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        spread = states[0].state in {AggState.AGGREGATED_SCALAR, AggState.LITERAL_SCALAR}
        scalars = (spread,) + (False,) * (len(states) - 1)
        context = KernelContext(grouped=True, scalars=scalars)
        datas = self._datas(steps, [False] * len(states), grouped=True)
        dtype = self._definition.kernel.output(datas, context)
        return replace(RollingRun().agg(states[1], self._frame), dtype=dtype)

    def _rewritten_agg(self, steps: Steps, visited: Sequence[Operand]) -> InGroups | None:
        """Give the value in `agg` where polars puts something in place of the node.

        The same rewrites as in a `select` apply (`_rewritten_select`).

        Args:
            steps: The steps of the cast pass on each input.
            visited: The inputs as the first visit of the cast pass reads them.

        Returns:
            The value, with the type of its data, or `None` where the node stays.

        Example:
            >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
            >>> leaves = [LiteralLeaf(1), LiteralLeaf(2)]
            >>> evaluation = NodeEvaluation(BinaryOperators.ADD, leaves, GroupBy((3,)))
            >>> literals = [Operand.of_value(1), Operand.of_value(2)]
            >>> evaluation._rewritten_agg(((), ()), literals).state
            <AggState.LITERAL_SCALAR: 'literal scalar'>
        """
        folded = self._definition.simplify.folded(visited) if self._root else None
        if folded is not None:
            return self._frame.literal(folded.dtype.materialized())
        if self._visits.new_root(visited) is not None:
            other = self._leaves[1 if visited[0].literal and not visited[1].literal else 0]
            if other.height() is Height.SCALAR:
                return self._frame.literal(self.ROOT_TYPE)
            return replace(other.grouped(self._frame), dtype=self.ROOT_TYPE)
        self._after(steps)
        replacement = self._replacement(steps) if self._frame.flags.simplify_agg else None
        if replacement is None:
            return None
        data = self._replaced_data(replacement, steps).dtype
        value = self._replaced_groups(replacement, steps)
        return value if value.dtype is not None else replace(value, dtype=data)

    def _column(self, value: InGroups, data: DType, *, wrapped: bool) -> DType:
        """Give the type of the column of `agg` for the value that the run of the node gives.

        It raises `PolarsRefusal` where the column does not have one row for each group.

        Args:
            value: The value of the node over the groups.
            data: The type of the data, where the value does not hold a type.
            wrapped: Whether the planner wraps the node in a list.

        Returns:
            The type, in a list where each group holds a list.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> frame = GroupBy((3, 2))
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], frame)
            >>> evaluation._column(frame.column(DType("Int8")), DType("Int8"), wrapped=True)
            List(Int8)
        """
        output = self._frame.output(value, wrapped=wrapped)
        dtype = data if value.dtype is None else value.dtype
        return DType("List", (dtype,)) if output.listed() else dtype

    def _replacement(self, steps: Steps) -> Replacement | None:
        """Give what the optimizer puts in place of the node, or `None` where the node stays.

        The optimizer runs after the conversion, so it reads each input after
        the steps of the cast pass, with its literals folded. An `implode`
        makes an input one value.

        Args:
            steps: The steps of the cast pass on each input.

        Returns:
            The replacement, or `None`.

        Example:
            >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
            >>> leaves = [ColumnLeaf(DType("String"), RowForm.PER_ROW), LiteralLeaf(True)]
            >>> evaluation = NodeEvaluation(BinaryOperators.AND, leaves, GroupBy((3,)))
            >>> evaluation._replacement(((), ()))
            InputInPlace(index=0)
        """
        optimized = [leaf.optimized() for leaf in self._leaves]
        inputs = list(starmap(StepOperand().after, zip(optimized, steps, strict=True)))
        heights: list[Height] = []
        for leaf, changes in zip(self._leaves, steps, strict=True):
            imploded = any(step.kind is StepKind.IMPLODE for step in changes)
            heights.append(Height.SCALAR if imploded else leaf.height())
        return self._definition.simplify.replaced(inputs, heights)

    def _replaced_data(self, replacement: Replacement, steps: Steps) -> Data:
        """Give the data of what the optimizer puts in place of the node.

        It raises `PolarsRefusal` where a cast of the input fails.

        Args:
            replacement: The input or the literal.
            steps: The steps of the cast pass on each input.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
            >>> from astrolabe._rules.execution import InputInPlace
            >>> leaves = [ColumnLeaf(DType("String"), RowForm.PER_ROW), LiteralLeaf(True)]
            >>> evaluation = NodeEvaluation(BinaryOperators.AND, leaves, GroupBy((3,)))
            >>> evaluation._replaced_data(InputInPlace(0), ((), ())).dtype
            String
        """
        if isinstance(replacement, LiteralInPlace):
            return Data(replacement.literal.dtype.materialized())
        index = replacement.index
        leaf = self._leaves[index]
        return StepRun().data(leaf.data(), self._computed[index], steps[index])

    def _replaced_groups(self, replacement: Replacement, steps: Steps) -> InGroups:
        """Give the value over the groups of what the optimizer puts in place of the node.

        Args:
            replacement: The input or the literal.
            steps: The steps of the cast pass on each input.

        Returns:
            The value.

        Example:
            >>> from astrolabe._rules.definitions.binary_operators import BinaryOperators
            >>> from astrolabe._rules.execution import InputInPlace
            >>> leaves = [ColumnLeaf(DType("String"), RowForm.ONE_VALUE), LiteralLeaf(True)]
            >>> evaluation = NodeEvaluation(BinaryOperators.AND, leaves, GroupBy((3, 2)))
            >>> evaluation._replaced_groups(InputInPlace(0), ((), ())).state
            <AggState.AGGREGATED_SCALAR: 'aggregated scalar'>
        """
        if isinstance(replacement, LiteralInPlace):
            return self._frame.literal(replacement.literal.dtype.materialized())
        index = replacement.index
        leaf = self._leaves[index]
        return StepRun().grouped(leaf.grouped(self._frame), steps[index], self._frame)

    def _grouped_data(self, steps: Steps, states: Sequence[InGroups], *, fallible: bool) -> DType:
        """Give the type that the kernel gives in `agg`, at the rows of each of its calls.

        The run of the node calls the kernel once for each group, or once on
        the flat values of all the groups (`Run.calls`). The rows of a call
        decide which input has one row beside more rows, and a kernel that
        checks the lengths of its inputs reads them. Where two calls see
        other rows, each one runs, and a call that fails makes the run fail.

        It raises `PolarsRefusal` where a cast or the kernel fails.

        Args:
            steps: The steps of the cast pass on each input.
            states: The value of each input over the groups, after the steps.
            fallible: Whether the kernel can fail on one value.

        Returns:
            The type of the data of the first call.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3, 2)))
            >>> literal = GroupBy((3, 2)).literal(DType("Int32"))
            >>> evaluation._grouped_data(((),), [literal], fallible=False)
            Int32
        """
        facts = Kernel(fallible=fallible, physical=self._definition.physical_expr())
        calls = Run(self._shape, facts).calls(states)
        outputs: list[DType] = []
        lists = self._reads_lists(states, calls)
        reads = self._cast_reads(states, steps)
        for call in SingleRows().distinct(calls, len(states)):
            bounds = self._call_bounds(states, call)
            singles = SingleRows().of(call)
            datas = self._datas(steps, singles, grouped=True, bounds=bounds, reads=reads)
            context = replace(self._grouped_context(states, steps), lengths=call, lists=lists)
            outputs.append(self._definition.kernel.output(datas, context))
            if len(calls) == 1:
                self._own_length(datas, call)
        return outputs[0]

    def _reads_lists(self, states: Sequence[InGroups], calls: Sequence[Sequence[int]]) -> bool:
        """Say whether the run in `agg` reads the inputs group by group or as lists of the groups.

        The run reads flat values on one call over values that are not
        aggregated, or that share their groups. A call for each group, an
        input of lists, or two inputs whose groups can differ read lists.

        Args:
            states: The value of each input over the groups.
            calls: The rows of each input at each call of the kernel.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> frame = GroupBy((3, 2))
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], frame)
            >>> evaluation._reads_lists([frame.column(), frame.column()], [(5, 5)])
            False
        """
        lists = any(state.state is AggState.AGGREGATED_LIST for state in states)
        groups = {state.groups for state in states if state.state is AggState.NOT_AGGREGATED}
        return len(calls) > 1 or lists or len(groups) > 1

    def _own_length(self, datas: Sequence[Data], call: Sequence[int]) -> None:
        """Refuse the one call in `agg` whose kernel gives other rows than its inputs hold.

        A kernel whose input types decide the rows of its output (`OwnLength`)
        must give the rows of the flat values where it runs once over them.
        Where it runs once for each group, each group takes the rows that
        the kernel gives.

        Args:
            datas: The data of each input at the call.
            call: The rows of each input at the call.

        Raises:
            PolarsRefusal: The kernel gives other rows than the group holds.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._own_length([Data(DType("Int8"))], (3,))
        """
        kernel = self._definition.kernel
        own = kernel.output_length(datas, call) if isinstance(kernel, OwnLength) else None
        if own is not None and own != max(call):
            message = f"output length of `map` ({own}) must be equal to the input length"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def _computed_after(self, steps: Steps) -> tuple[Operand, ...]:
        """Give each input as the run computes it after the cast pass, for a kernel that reads it.

        A literal that the pass gives a new type stays a literal, and a cast
        makes the input a column (`StepOperand`).

        Args:
            steps: The steps of each input.

        Returns:
            The inputs.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf("a")], GroupBy((3,)))
            >>> evaluation._computed_after(((),))
            (Operand(dtype=String, literal=True, value='a'),)
        """
        pairs = zip(self._computed, steps, strict=True)
        return tuple(starmap(StepOperand().after, pairs))

    def _node_type(self) -> DType:
        """Give the type of the node at the planner types of its inputs, as polars converts it.

        Some functions keep this type as an option that their kernel reads
        (`KernelContext.planned`). The code that calls this method has the
        planner type, so the type function does not refuse here.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))._node_type()
            Unknown(int, 1)
        """
        return self._definition.type_function.planner(self._planned)

    def _datas(
        self,
        steps: Steps,
        singles: Sequence[bool],
        *,
        grouped: bool,
        bounds: Sequence[int | None] = (),
        reads: Sequence[int | None] = (),
    ) -> list[Data]:
        """Give the data that reaches the kernel from each input, after the cast pass.

        An input can give other data in `agg` than in a `select`: the mean of
        text refuses in a `select`, and gives a null of text for each group.
        Where the code bounds an input to zero rows, the input holds no value,
        so its data holds only nulls: polars casts it to any type with no
        kernel, and a kernel reads it as it reads nulls. The casts of the
        cast pass can read other rows than the kernel (`_cast_reads`).

        It raises `PolarsRefusal` where a cast fails.

        Args:
            steps: The steps of each input.
            singles: Whether each input has one row beside an input of more rows.
            grouped: Whether the kernel runs in `agg`.
            bounds: The most rows that the code allows each input at the
                kernel, or `None` where the data decides its rows (`Data.bound`).
            reads: The most rows that the casts of each input read, where
                they differ from `bounds`.

        Returns:
            The data of each input.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._datas(((),), [True], grouped=False)
            [Data(dtype=Int32, nulls=False, one_row=True)]
        """
        run = StepRun()
        datas: list[Data] = []
        limits = list(bounds) or [None] * len(singles)
        casts = list(reads) or limits
        inputs = zip(self._leaves, self._computed, steps, singles, limits, casts, strict=True)
        for leaf, operand, changes, single, bound, cast in inputs:
            source = leaf.grouped_data(self._frame) if grouped else leaf.data()
            if cast == 0:
                source = replace(source, nulls=True)
            data = run.data(source, operand, changes)
            if bound == 0:
                data = replace(data, nulls=True)
            datas.append(replace(data, one_row=single, bound=bound))
        return datas

    def _call_bounds(self, states: Sequence[InGroups], call: Sequence[int]) -> list[int | None]:
        """Give the most rows that the code allows each input at one call of the kernel in `agg`.

        Args:
            states: The value of each input over the groups.
            call: The rows of each input at the call.

        Returns:
            The rows, or `None` where the data decides them.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> frame = GroupBy((3, 2))
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], frame)
            >>> evaluation._call_bounds([frame.column(), frame.literal()], (3, 1))
            [None, 1]
        """
        bounded = [state.bounded or state.state in InGroups.SINGLE for state in states]
        return [rows if limit else None for rows, limit in zip(call, bounded, strict=True)]

    def _grouped_context(self, states: Sequence[InGroups], steps: Steps) -> KernelContext:
        """Give how the data reaches the kernel in `agg`, before the rows of a call.

        Args:
            states: The value of each input over the groups.
            steps: The steps of each input.

        Returns:
            The context.

        Example:
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> frame = GroupBy((3, 2))
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], frame)
            >>> evaluation._grouped_context([frame.literal()], ((),)).scalars
            (True,)
        """
        return KernelContext(
            grouped=True,
            scalars=tuple(state.state is AggState.LITERAL_SCALAR for state in states),
            planned=self._node_type(),
            computed=self._computed_after(steps),
            decided=tuple(state.decided for state in states),
        )

    def _cast_reads(self, states: Sequence[InGroups], steps: Steps) -> list[int | None]:
        """Give the most rows that the casts of the cast pass read from each input in `agg`.

        Polars casts the flat values of an input that is not aggregated, the
        rows that a slice drops included, where a cast is not strict or the
        groups are those of the `group_by`. A strict cast over other groups
        casts the values of each group, and a cast of lists casts their values.

        Args:
            states: The value of each input over the groups.
            steps: The steps of each input.

        Returns:
            The rows, or `None` where the data decides them.

        Example:
            >>> from astrolabe._rules.cast_pass import Step
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> frame = GroupBy((3, 2))
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], frame)
            >>> empty = replace(frame.column(), groups=(0, 0), original=False, bounded=True)
            >>> loose = (Step.cast(DType("Float64"), CastOption.NON_STRICT),)
            >>> evaluation._cast_reads([empty], (loose,))
            [5]
        """
        reads: list[int | None] = []
        for state, changes in zip(states, steps, strict=True):
            loose = state.original or self._loose(changes)
            flat = state.state is AggState.NOT_AGGREGATED and loose
            rows = state.rows if flat else sum(state.values())
            reads.append(rows if state.bounded else None)
        return reads

    def _loose(self, changes: Sequence[Step]) -> bool:
        """Say whether the cast pass puts a cast that is not strict above an input.

        Args:
            changes: The steps of the input.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.cast_pass import Step
            >>> from astrolabe._rules.definitions.elementwise import Elementwise
            >>> evaluation = NodeEvaluation(Elementwise.ABS, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._loose((Step.cast(DType("Int8"), CastOption.STRICT),))
            False
        """
        casts = [step for step in changes if step.kind is StepKind.CAST]
        return any(step.option is not CastOption.STRICT for step in casts)

    def _after(self, steps: Steps) -> DType:
        """Give the type of the node in the plan that runs: the type function after the cast pass.

        Polars computes this type when it converts the expression, so a
        refusal of the type function here refuses the run before any data
        runs. For example, the cast pass casts a number beside a struct to the
        struct, and the division of two structs refuses a field that is not a
        number. A node that runs once for each group builds the list of each
        group at this type.

        Args:
            steps: The steps of each input.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.cast_pass import CastOption, Step
            >>> from astrolabe._rules.definitions.length_preserving import LengthPreserving
            >>> cum_sum = LengthPreserving.CUM_SUM
            >>> evaluation = NodeEvaluation(cum_sum, [LiteralLeaf(1)], GroupBy((3,)))
            >>> evaluation._after(((Step.cast(DType("Int8"), CastOption.STRICT),),))
            Int64
        """
        pairs = zip(self._folded, steps, strict=True)
        after = list(starmap(StepOperand().after, pairs))
        return self._definition.type_function.planner(after)
