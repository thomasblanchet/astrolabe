"""Give the casts that the cast pass of polars puts above the inputs of a node.

Before a query runs, the type coercion pass of polars can put a cast above an
input of a node, so that the inputs of an operator or a function have a
common type. It can also refuse the plan. The rule depends on the kind of the
node. A binary operator, a function, a cast node and a node with no cast pass
each have one class here. Each class adapts its rule to the one call that
code that evaluates a node makes (`CastRule`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from astrolabe._rules.cast_pass import (
    BinaryCasts,
    CastNodeVisit,
    CastOption,
    FunctionCasts,
    Namespace,
    Step,
    StepOperand,
    Visit,
    VisitTurn,
)
from astrolabe._rules.options import CastingRules
from astrolabe._rules.supertype import Supertype
from astrolabe._rules.definitions.node_options import (
    NodeOptions,
    OptionValue,
    OptionValues,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.dtypes import DType
    from astrolabe._rules.operators import Operator
    from astrolabe._rules.options import FunctionOptions
    from astrolabe._rules.type_functions import Operand


class CastRule(Protocol):
    """Give the changes of one visit of the cast pass of polars to a node."""

    def visit(
        self,
        inputs: Sequence[Operand],
        options: FunctionOptions,
        *,
        turn: VisitTurn,
    ) -> Visit:
        """Give the steps of each input, or raise `PolarsRefusal` where the pass refuses.

        Args:
            inputs: The inputs as the node reads them at that visit.
            options: The options of the node.
            turn: The visit of the cast pass.
        """

    def at(self, node: NodeOptions) -> CastRule:
        """Give the rule at the options of one node.

        Args:
            node: The serialized root of the node.
        """


@dataclass(frozen=True)
class NoCasts:
    """State that the cast pass of polars changes nothing on the node, as on an aggregation.

    Example:
        >>> from astrolabe._rules.options import FunctionOptions
        >>> from astrolabe._rules.type_functions import Operand
        >>> NoCasts().visit([Operand.of_value(1)], FunctionOptions(), turn=VisitTurn.FIRST)
        Visit(steps=((),), root=None)
    """

    def visit(
        self,
        inputs: Sequence[Operand],
        options: FunctionOptions,
        *,
        turn: VisitTurn,
    ) -> Visit:
        """Give a visit that changes nothing.

        Args:
            inputs: The inputs.
            options: The options of the node.
            turn: The visit of the cast pass.

        Returns:
            The visit.

        Example:
            >>> from astrolabe._rules.options import FunctionOptions
            >>> NoCasts().visit([], FunctionOptions(), turn=VisitTurn.SECOND).steps
            ()
        """
        del options, turn
        return Visit.unchanged(len(inputs))

    def at(self, node: NodeOptions) -> NoCasts:
        """Give the rule, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same rule.

        Example:
            >>> NoCasts().at(NodeOptions({"Agg": {}}))
            NoCasts()
        """
        del node
        return self


@dataclass(frozen=True)
class OperatorCasts:
    """Give the casts of the cast pass of polars on a binary operator, at each visit.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.operators import Operator
        >>> from astrolabe._rules.options import FunctionOptions
        >>> from astrolabe._rules.type_functions import Operand
        >>> inputs = [Operand.column(DType("Int8")), Operand.column(DType("Int16"))]
        >>> OperatorCasts(Operator.PLUS).visit(inputs, FunctionOptions(), turn=VisitTurn.FIRST)
        Visit(steps=((cast(Int16, non_strict),), ()), root=None)
    """

    operator: Operator

    def visit(
        self,
        inputs: Sequence[Operand],
        options: FunctionOptions,
        *,
        turn: VisitTurn,
    ) -> Visit:
        """Give the casts of the visit (`BinaryCasts`).

        Args:
            inputs: The left input and the right input.
            options: The options of the node, which a binary operator does not have.
            turn: The visit of the cast pass.

        Returns:
            The visit.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.operators import Operator
            >>> from astrolabe._rules.options import FunctionOptions
            >>> from astrolabe._rules.type_functions import Operand
            >>> same = [Operand.column(DType("Int8"))] * 2
            >>> turn = VisitTurn.SECOND
            >>> OperatorCasts(Operator.PLUS).visit(same, FunctionOptions(), turn=turn).steps
            ((), ())
        """
        del options, turn
        return BinaryCasts().visit(self.operator, inputs[0], inputs[1])

    def at(self, node: NodeOptions) -> OperatorCasts:
        """Give the rule, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same rule.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> OperatorCasts(Operator.PLUS).at(NodeOptions({"BinaryExpr": {}})).operator
            <Operator.PLUS: 'Plus'>
        """
        del node
        return self


@dataclass(frozen=True)
class FunctionCasting:
    """Give the casts of the general rule of the cast pass of polars on a function.

    A function that has no rule of its own in the cast pass gets the general
    rule. The rule applies the casting rule of the options of the function,
    where the options have one (`FunctionCasts`): for example, `fill_null`
    casts its inputs to their supertype. Polars then takes the casting rule
    away, so a second visit changes nothing. A function whose options have no
    casting rule, such as `abs`, gets no cast.

    Example:
        >>> from astrolabe._rules.options import FunctionOptions
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.supertype import SupertypeFlag
        >>> from astrolabe._rules.type_functions import Operand
        >>> fill = FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default())
        >>> inputs = [Operand.column(DType("Int8")), Operand.of_value(300)]
        >>> FunctionCasting().visit(inputs, fill, turn=VisitTurn.FIRST)
        Visit(steps=((cast(Int16, strict),), (literal(Int16),)), root=None)
        >>> FunctionCasting().visit(inputs, fill, turn=VisitTurn.SECOND).steps
        ((), ())
    """

    namespace: Namespace = Namespace.NONE

    def visit(
        self,
        inputs: Sequence[Operand],
        options: FunctionOptions,
        *,
        turn: VisitTurn,
    ) -> Visit:
        """Give the casts of the visit.

        Args:
            inputs: The inputs of the function.
            options: The options of the function.
            turn: The visit of the cast pass.

        Returns:
            The visit.

        Example:
            >>> from astrolabe._rules.options import FunctionOptions
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.type_functions import Operand
            >>> inputs = [Operand.column(DType("Int8"))]
            >>> turn = VisitTurn.FIRST
            >>> FunctionCasting().visit(inputs, FunctionOptions.elementwise(), turn=turn).steps
            ((),)
        """
        rules = options.cast_options
        if rules is None or turn is VisitTurn.SECOND:
            return Visit.unchanged(len(inputs))
        return FunctionCasts(rules, self.namespace).visit(inputs)

    def at(self, node: NodeOptions) -> FunctionCasting:
        """Give the rule, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same rule.

        Example:
            >>> FunctionCasting().at(NodeOptions({"Function": {}}))
            FunctionCasting(namespace=<Namespace.NONE: 'none'>)
        """
        del node
        return self


@dataclass(frozen=True)
class SupertypeConversion:
    """Give the casts of a function whose inputs the conversion casts to their supertype.

    `append` is such a function. When polars converts the expression, it puts
    a cast to the supertype of the two inputs above each input of another
    type, and the node keeps the casting rule of a supertype. The cast pass
    then visits those casts as it visits any cast node (`CastNodeVisit`).
    Where the supertype is a known type, both inputs reach the node with that
    type, so its casting rule changes nothing. Where the supertype is the
    planner type `Unknown(Any)`, the cast pass of some releases takes those
    casts away. The root reads its inputs after the visits of the casts
    (`VisitTurn.ROOT`). Where the casts went, it reads the types of its inputs
    again, and its casting rule refuses them. Any other node reads the casts
    at its first visit, and the pass takes its casting rule away there.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.options import FunctionOptions
        >>> from astrolabe._rules.type_functions import Operand
        >>> inputs = [Operand.column(DType("Int8")), Operand.of_value(300)]
        >>> SupertypeConversion().visit(inputs, FunctionOptions(), turn=VisitTurn.ROOT).steps
        ((), ())
    """

    RULE: ClassVar[CastingRules] = CastingRules.cast_to_supertypes()

    def visit(
        self,
        inputs: Sequence[Operand],
        options: FunctionOptions,
        *,
        turn: VisitTurn,
    ) -> Visit:
        """Give the casts of the visit: none, or the refusal of the root.

        Args:
            inputs: The inputs of the function, before the casts of the conversion.
            options: The options of the function.
            turn: The visit of the cast pass.

        Returns:
            A visit that changes nothing.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.options import FunctionOptions
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> from astrolabe._rules.type_functions import Operand
            >>> inputs = [Operand.column(DType("Binary")), Operand.of_value(1)]
            >>> turn = VisitTurn.FIRST
            >>> SupertypeConversion().visit(inputs, FunctionOptions(), turn=turn).steps
            ((), ())
            >>> with PolarsRelease.pretend("1.44.1"):
            ...     SupertypeConversion().visit(inputs, FunctionOptions(), turn=VisitTurn.ROOT)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        del options
        if turn is VisitTurn.ROOT:
            supertype = Supertype().require(inputs[0].dtype, inputs[1].dtype)
            read = [self._read(operand, supertype) for operand in inputs]
            FunctionCasts(self.RULE).visit(read)
        return Visit.unchanged(len(inputs))

    def _read(self, operand: Operand, supertype: DType) -> Operand:
        """Give an input as the root reads it, after the visit of its cast to the supertype.

        Args:
            operand: The input.
            supertype: The supertype of the inputs.

        Returns:
            The input after its cast, or the input itself where it has no cast
            or the visit took the cast away.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.type_functions import Operand
            >>> SupertypeConversion()._read(Operand.column(DType("Int8")), DType("Int16"))
            Operand(dtype=Int16, literal=False, value=None)
        """
        if operand.dtype.same(supertype):
            return operand
        step = CastNodeVisit().visit(operand, supertype, CastOption.NON_STRICT)
        return operand if step is None else StepOperand().after(operand, (step,))

    def at(self, node: NodeOptions) -> SupertypeConversion:
        """Give the rule, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same rule.

        Example:
            >>> SupertypeConversion().at(NodeOptions({"Function": {}}))
            SupertypeConversion()
        """
        del node
        return self


@dataclass(frozen=True)
class CastNode:
    """Give the visit of the cast pass to a cast node: the cast itself, which the pass settles.

    A cast that the user writes and a cast that the cast pass puts in are one
    kind of node in polars, and the pass visits both in one way
    (`CastNodeVisit`). It takes away a cast that changes nothing, relaxes a
    strict cast that cannot fail, and folds the cast of a literal into the
    literal. So the rule gives the input of the node one step: the cast, with
    its target and its option. The visit of the inserted cast settles that
    step. A second visit finds the cast settled, and changes nothing.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.options import FunctionOptions
        >>> from astrolabe._rules.type_functions import Operand
        >>> cast = CastNode(DType("Int16"), CastOption.STRICT)
        >>> cast.visit([Operand.column(DType("Int8"))], FunctionOptions(), turn=VisitTurn.FIRST)
        Visit(steps=((cast(Int16, strict),),), root=None)
    """

    target: DType | OptionValue[DType]
    option: CastOption | OptionValue[CastOption]

    def visit(
        self,
        inputs: Sequence[Operand],
        options: FunctionOptions,
        *,
        turn: VisitTurn,
    ) -> Visit:
        """Give the step of the cast.

        Args:
            inputs: The input of the cast.
            options: The options of the node, which a cast does not have.
            turn: The visit of the cast pass.

        Returns:
            The visit.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.options import FunctionOptions
            >>> from astrolabe._rules.type_functions import Operand
            >>> cast = CastNode(DType("Int16"), CastOption.STRICT)
            >>> turn = VisitTurn.SECOND
            >>> cast.visit([Operand.column(DType("Int8"))], FunctionOptions(), turn=turn).steps
            ((),)
        """
        del options
        if turn is VisitTurn.SECOND:
            return Visit.unchanged(len(inputs))
        values = OptionValues()
        target = values.known(self.target, "the target of the cast")
        option = values.known(self.option, "the option of the cast")
        return Visit(((Step.cast(target, option),),))

    def at(self, node: NodeOptions) -> CastNode:
        """Give the rule with the target and the option of one cast node.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> from astrolabe._rules.definitions.node_options import NodeOption
            >>> cast = CastNode(NodeOption.dtype("dtype"), NodeOption.cast_option("options"))
            >>> body = {"Cast": {"dtype": {"Literal": "Date"}, "options": "NonStrict"}}
            >>> cast.at(NodeOptions(body))
            CastNode(target=Date, option=<CastOption.NON_STRICT: 1>)
        """
        values = OptionValues()
        return CastNode(values.at(self.target, node), values.at(self.option, node))


class FunctionRule(Protocol):
    """Give the changes of a rule of the cast pass of polars that belongs to one function."""

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the steps of each input, or raise `PolarsRefusal` where the rule refuses.

        Args:
            operands: The inputs as the node reads them at the visit.
        """


@dataclass(frozen=True)
class OwnRuleCasts:
    """Give the casts of a function that has a rule of its own in the cast pass of polars.

    The cast pass of polars matches a few functions by name, before its
    general rule, for example `str.pad_start`, whose length it casts to
    `UInt64`. Such a rule runs at each visit of the node, because polars keeps
    no flag that stops it. It changes nothing once the inputs have the types
    that it wants, so a second visit changes nothing. The rule can read
    options of the node, as `int_range` casts its bounds to the type of the
    call.

    Example:
        >>> from astrolabe._rules.cast_pass import PadLengthCasts
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.options import FunctionOptions
        >>> from astrolabe._rules.type_functions import Operand
        >>> inputs = [Operand.column(DType("String")), Operand.column(DType("Int8"))]
        >>> turn = VisitTurn.FIRST
        >>> OwnRuleCasts(PadLengthCasts()).visit(inputs, FunctionOptions(), turn=turn).steps
        ((), (cast(UInt64, strict),))
    """

    rule: FunctionRule | OptionValue[FunctionRule]

    def visit(
        self,
        inputs: Sequence[Operand],
        options: FunctionOptions,
        *,
        turn: VisitTurn,
    ) -> Visit:
        """Give the casts of the visit.

        Args:
            inputs: The inputs of the function.
            options: The options of the function, which the rule does not read.
            turn: The visit of the cast pass, which the rule does not read.

        Returns:
            The visit.

        Example:
            >>> from astrolabe._rules.cast_pass import PadLengthCasts
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.options import FunctionOptions
            >>> from astrolabe._rules.type_functions import Operand
            >>> inputs = [Operand.column(DType("String")), Operand.column(DType("UInt64"))]
            >>> turn = VisitTurn.SECOND
            >>> OwnRuleCasts(PadLengthCasts()).visit(inputs, FunctionOptions(), turn=turn).steps
            ((), ())
        """
        del options, turn
        return OptionValues().known(self.rule, "the rule of the cast pass").visit(inputs)

    def at(self, node: NodeOptions) -> OwnRuleCasts:
        """Give the rule, with the options that it reads at one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> from astrolabe._rules.cast_pass import PadLengthCasts
            >>> rule = OwnRuleCasts(PadLengthCasts()).at(NodeOptions({"Function": {}})).rule
            >>> type(rule).__name__
            'PadLengthCasts'
        """
        return OwnRuleCasts(OptionValues().at(self.rule, node))
