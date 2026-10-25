"""The definitions of the binary operators of an expression, such as `+` or `==`.

Polars builds a binary operator as a node of its own kind, not as a function.
Its type function, its cast pass, its kernel and its simplifier each follow
the rule of the operator, so a definition here names the operator at each
place. The Rust API of polars also has `RustDivide`, `LogicalAnd` and
`LogicalOr`. The Python API does not build them, so they have no definition
here.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.definitions.cast_adapters import OperatorCasts
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import OperatorRun
from astrolabe._rules.definitions.node_options import NodeIdentity
from astrolabe._rules.definitions.simplify_adapters import OperatorSimplify
from astrolabe._rules.definitions.type_function_adapters import BinaryField
from astrolabe._rules.operators import Operator
from astrolabe._rules.options import NodeKind, Shape


class OperatorNode:
    """Build the definition of a binary operator, which reads the operator at each place.

    The node of an operator has the same parts for every operator: the type
    function, the cast pass, the kernel and the simplifier of polars each read
    the operator. So one call builds the definition of each operator. A rule
    of one operator that the parts do not state replaces one field of its
    definition, with `dataclasses.replace`. The arithmetic and the comparisons
    give a null where a side is null, and the logical operators and the
    comparisons that count nulls do not (`NullRule`).

    Example:
        >>> OperatorNode().of(Operator.MINUS).identity
        NodeIdentity(shape='BinaryExpr', function='Minus')
    """

    NULLS: ClassVar[frozenset[Operator]] = frozenset({
        Operator.PLUS,
        Operator.MINUS,
        Operator.MULTIPLY,
        Operator.TRUE_DIVIDE,
        Operator.FLOOR_DIVIDE,
        Operator.MODULUS,
        Operator.EQ,
        Operator.NOT_EQ,
        Operator.LT,
        Operator.LT_EQ,
        Operator.GT,
        Operator.GT_EQ,
    })

    def of(self, operator: Operator) -> Definition:
        """Give the definition of one operator.

        Args:
            operator: The operator, as polars names it in the tree.

        Returns:
            The definition.

        Example:
            >>> OperatorNode().of(Operator.EQ).kernel
            OperatorRun(operator=<Operator.EQ: 'Eq'>)
        """
        return Definition(
            identity=NodeIdentity("BinaryExpr", operator.value),
            type_function=BinaryField(operator),
            options=Shape.of(NodeKind.BINARY),
            casts=OperatorCasts(operator),
            kernel=OperatorRun(operator),
            simplify=OperatorSimplify(operator),
            nulls=NullRule.ANY if operator in self.NULLS else NullRule.NONE,
        )


class BinaryOperators:
    """Hold the definitions of the binary operators.

    Example:
        >>> BinaryOperators.ADD.identity
        NodeIdentity(shape='BinaryExpr', function='Plus')
        >>> BinaryOperators.EQ_MISSING.identity
        NodeIdentity(shape='BinaryExpr', function='EqValidity')
    """

    ADD: ClassVar[Definition] = OperatorNode().of(Operator.PLUS)
    SUBTRACT: ClassVar[Definition] = OperatorNode().of(Operator.MINUS)
    MULTIPLY: ClassVar[Definition] = OperatorNode().of(Operator.MULTIPLY)
    TRUE_DIVIDE: ClassVar[Definition] = OperatorNode().of(Operator.TRUE_DIVIDE)
    FLOOR_DIVIDE: ClassVar[Definition] = OperatorNode().of(Operator.FLOOR_DIVIDE)
    MODULUS: ClassVar[Definition] = OperatorNode().of(Operator.MODULUS)
    EQ: ClassVar[Definition] = OperatorNode().of(Operator.EQ)
    EQ_MISSING: ClassVar[Definition] = OperatorNode().of(Operator.EQ_VALIDITY)
    NOT_EQ: ClassVar[Definition] = OperatorNode().of(Operator.NOT_EQ)
    NOT_EQ_MISSING: ClassVar[Definition] = OperatorNode().of(Operator.NOT_EQ_VALIDITY)
    LT: ClassVar[Definition] = OperatorNode().of(Operator.LT)
    LT_EQ: ClassVar[Definition] = OperatorNode().of(Operator.LT_EQ)
    GT: ClassVar[Definition] = OperatorNode().of(Operator.GT)
    GT_EQ: ClassVar[Definition] = OperatorNode().of(Operator.GT_EQ)
    AND: ClassVar[Definition] = OperatorNode().of(Operator.AND)
    OR: ClassVar[Definition] = OperatorNode().of(Operator.OR)
    XOR: ClassVar[Definition] = OperatorNode().of(Operator.XOR)
