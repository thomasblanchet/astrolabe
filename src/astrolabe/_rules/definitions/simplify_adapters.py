"""Give the rewrites of the simplifier of polars, which can take the place of a node before it runs.

Polars simplifies an expression twice. When it converts the expression, it
visits each node before the cast pass of that node. There it turns an integer
literal into a float beside a float literal, and folds a node whose inputs
are literals into one literal, for example `pl.lit(6) // pl.lit(4)` into
`1`. At `collect()`, its optimizer reads the plan after the cast pass,
and puts an input or a literal in place of some nodes, for example `x` in
place of `x & True`. Where a rewrite applies, the cast pass of the node or its
kernel never runs, so the type of the run can differ from what the kernel
gives. Most nodes have no rewrite. Each class here adapts the rewrites of one
kind of node to the calls that code that evaluates a node makes
(`SimplifyRule`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from astrolabe._rules.dtypes import IntegerRange, Is
from astrolabe._rules.execution import BinaryFold, BooleanSimplify, LiteralInPlace
from astrolabe._rules.type_functions import Operand

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.definitions.node_options import NodeOptions
    from astrolabe._rules.execution import Replacement
    from astrolabe._rules.operators import Operator
    from astrolabe._rules.options import Height


class SimplifyRule(Protocol):
    """Give the rewrites of the simplifier of polars for one node."""

    def rewritten(self, inputs: Sequence[Operand]) -> list[Operand]:
        """Give the inputs as the conversion rewrites them before the cast pass of the node.

        Args:
            inputs: The inputs as the conversion reads them when it visits the node.
        """

    def folded(self, inputs: Sequence[Operand]) -> Operand | None:
        """Give the literal that the conversion folds the node into, or `None` where it does not.

        Args:
            inputs: The inputs as the conversion reads them when it visits the node.
        """

    def replaced(self, inputs: Sequence[Operand], heights: Sequence[Height]) -> Replacement | None:
        """Give what the optimizer puts in place of the node, or `None` where the node stays.

        Args:
            inputs: The inputs, with the types that the cast pass gives them.
            heights: The projection height of each input.
        """

    def at(self, node: NodeOptions) -> SimplifyRule:
        """Give the rule at the options of one node.

        Args:
            node: The serialized root of the node.
        """


@dataclass(frozen=True)
class NoSimplify:
    """Give no rewrite: the node runs as written.

    Example:
        >>> NoSimplify().folded([]) is None
        True
    """

    def rewritten(self, inputs: Sequence[Operand]) -> list[Operand]:
        """Give the inputs unchanged.

        Args:
            inputs: The inputs.

        Returns:
            The same inputs.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> NoSimplify().rewritten([Operand.of_value(1)])
            [Operand(dtype=Unknown(int, 1), literal=True, value=1)]
        """
        return list(inputs)

    def folded(self, inputs: Sequence[Operand]) -> Operand | None:
        """Give no fold.

        Args:
            inputs: The inputs, which the rule does not read.

        Returns:
            `None`.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> NoSimplify().folded([Operand.of_value(1)]) is None
            True
        """
        del inputs
        return None

    def replaced(self, inputs: Sequence[Operand], heights: Sequence[Height]) -> Replacement | None:
        """Give no replacement.

        Args:
            inputs: The inputs, which the rule does not read.
            heights: The heights of the inputs, which the rule does not read.

        Returns:
            `None`.

        Example:
            >>> NoSimplify().replaced([], []) is None
            True
        """
        del inputs, heights
        return None

    def at(self, node: NodeOptions) -> NoSimplify:
        """Give the same rule: it reads no option.

        Args:
            node: The serialized root of the node.

        Returns:
            The same rule.

        Example:
            >>> from astrolabe._rules.definitions.node_options import NodeOptions
            >>> NoSimplify().at(NodeOptions({"Agg": {}}))
            NoSimplify()
        """
        del node
        return self


@dataclass(frozen=True)
class OperatorSimplify:
    """Give the rewrites of a binary operator: the fold of two literals, and those of `&` and `|`.

    The conversion folds a binary node whose two inputs are literals of one
    kind (`BinaryFold`). The optimizer replaces `&` and `|` beside a `Boolean`
    literal (`BooleanSimplify`).

    Example:
        >>> from astrolabe._rules.operators import Operator
        >>> from astrolabe._rules.type_functions import Operand
        >>> pair = [Operand.of_value(2**40), Operand.of_value(1)]
        >>> OperatorSimplify(Operator.MODULUS).folded(pair)
        Operand(dtype=Unknown(int, 0), literal=True, value=0)
    """

    operator: Operator

    def rewritten(self, inputs: Sequence[Operand]) -> list[Operand]:
        """Give the inputs with an integer literal turned into a float beside a float literal.

        Args:
            inputs: The two inputs as the conversion reads them.

        Returns:
            The two inputs.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> from astrolabe._rules.type_functions import Operand
            >>> pair = [Operand.of_value(1), Operand.of_value(1.5)]
            >>> OperatorSimplify(Operator.XOR).rewritten(pair)[0]
            Operand(dtype=Unknown, literal=True, value=1.0)
        """
        left, right = inputs
        if not (left.literal and right.literal):
            return [left, right]
        return list(BinaryFold().promoted(left, right))

    def folded(self, inputs: Sequence[Operand]) -> Operand | None:
        """Give the literal of the fold, where the two inputs are literals that fold.

        Args:
            inputs: The two inputs as the conversion reads them.

        Returns:
            The literal, or `None`.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.operators import Operator
            >>> from astrolabe._rules.type_functions import Operand
            >>> pair = [Operand.column(DType("Int8")), Operand.of_value(1)]
            >>> OperatorSimplify(Operator.PLUS).folded(pair) is None
            True
        """
        left, right = inputs
        if not (left.literal and right.literal):
            return None
        return BinaryFold().binary(self.operator, left, right)

    def replaced(self, inputs: Sequence[Operand], heights: Sequence[Height]) -> Replacement | None:
        """Give the replacement of `&` or `|` beside a `Boolean` literal.

        Args:
            inputs: The two inputs, with the types that the cast pass gives them.
            heights: The projection height of each input.

        Returns:
            The replacement, or `None`.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.operators import Operator
            >>> from astrolabe._rules.options import Height
            >>> from astrolabe._rules.type_functions import Operand
            >>> false = Operand.typed_literal(False, DType("Boolean"))
            >>> pair = [Operand.column(DType("String")), false]
            >>> heights = [Height.COLUMN, Height.SCALAR]
            >>> OperatorSimplify(Operator.OR).replaced(pair, heights)
            InputInPlace(index=0)
        """
        return BooleanSimplify().replaced(self.operator, inputs, heights)

    def at(self, node: NodeOptions) -> OperatorSimplify:
        """Give the same rule: the operator is not an option.

        Args:
            node: The serialized root of the node.

        Returns:
            The same rule.

        Example:
            >>> from astrolabe._rules.definitions.node_options import NodeOptions
            >>> from astrolabe._rules.operators import Operator
            >>> OperatorSimplify(Operator.OR).at(NodeOptions({"BinaryExpr": {}})).operator
            <Operator.OR: 'Or'>
        """
        del node
        return self


@dataclass(frozen=True)
class NegateSimplify:
    """Give the rewrite of a negation: the optimizer puts the negated literal in its place.

    At `collect()`, the optimizer of polars puts a literal of the negated
    value in place of the negation of a literal. A dynamic number stays
    dynamic, so `-pl.lit(2**31)` becomes the literal `-2**31`, which takes
    `Int32` where the negation of the `Int64` literal would keep `Int64`. A
    signed integer literal negates where its type holds the result, and a
    float literal always does. Any other literal, as an unsigned integer,
    stays below the node. `collect_schema()` does not run the optimizer, so
    the planner type does not change.

    Example:
        >>> NegateSimplify().replaced([Operand.of_value(2**31)], []).literal.value
        -2147483648
    """

    SIGNED: ClassVar[frozenset[str]] = frozenset({"Int8", "Int16", "Int32", "Int64", "Int128"})
    FLOATS: ClassVar[frozenset[str]] = frozenset({"Float16", "Float32", "Float64"})

    def rewritten(self, inputs: Sequence[Operand]) -> list[Operand]:
        """Give the input unchanged: the conversion does not rewrite a negation.

        Args:
            inputs: The input.

        Returns:
            The same input.

        Example:
            >>> NegateSimplify().rewritten([Operand.of_value(1)])
            [Operand(dtype=Unknown(int, 1), literal=True, value=1)]
        """
        return list(inputs)

    def folded(self, inputs: Sequence[Operand]) -> Operand | None:
        """Give no fold: the conversion does not fold a negation.

        Args:
            inputs: The input, which the rule does not read here.

        Returns:
            `None`.

        Example:
            >>> NegateSimplify().folded([Operand.of_value(1)]) is None
            True
        """
        del inputs
        return None

    def replaced(self, inputs: Sequence[Operand], heights: Sequence[Height]) -> Replacement | None:
        """Give the literal of the negated value, where the input is a literal that negates.

        Args:
            inputs: The input, with the type that the cast pass gives it.
            heights: The height of the input, which the rule does not read.

        Returns:
            The replacement, or `None`.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> NegateSimplify().replaced([Operand.column(DType("Int8"))], []) is None
            True
            >>> NegateSimplify().replaced([Operand.typed_literal(-128, DType("Int8"))], []) is None
            True
        """
        del heights
        negated = self._negated(inputs[0])
        return None if negated is None else LiteralInPlace(negated)

    def _negated(self, operand: Operand) -> Operand | None:
        """Give the literal of the negated value, or `None` where the literal does not negate.

        Args:
            operand: The input.

        Returns:
            The literal, or `None`.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> NegateSimplify()._negated(Operand.typed_literal(1.5, DType("Float32"))).value
            -1.5
            >>> NegateSimplify()._negated(Operand.typed_literal(1, DType("UInt8"))) is None
            True
            >>> NegateSimplify()._negated(Operand.of_value(-(2**127))) is None
            True
        """
        value = operand.value
        if not operand.literal or isinstance(value, bool) or not isinstance(value, int | float):
            return None
        if Is.unknown(operand.dtype):
            fits = isinstance(value, float) or IntegerRange().holds("Int128", -value)
            return Operand.of_value(-value) if fits else None
        name = operand.dtype.name
        if name in self.FLOATS:
            return Operand.typed_literal(-value, operand.dtype)
        if name in self.SIGNED and isinstance(value, int) and IntegerRange().holds(name, -value):
            return Operand.typed_literal(-value, operand.dtype)
        return None

    def at(self, node: NodeOptions) -> NegateSimplify:
        """Give the same rule: it reads no option.

        Args:
            node: The serialized root of the node.

        Returns:
            The same rule.

        Example:
            >>> from astrolabe._rules.definitions.node_options import NodeOptions
            >>> NegateSimplify().at(NodeOptions({"Function": {}}))
            NegateSimplify()
        """
        del node
        return self
