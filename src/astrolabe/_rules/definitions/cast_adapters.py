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
from typing import TYPE_CHECKING, Protocol

from astrolabe._rules.cast_pass import (
    BinaryCasts,
    CastOption,
    FunctionCasts,
    Namespace,
    Step,
    Visit,
)
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

    def visit(self, inputs: Sequence[Operand], options: FunctionOptions, *, again: bool) -> Visit:
        """Give the steps of each input, or raise `PolarsRefusal` where the pass refuses.

        Args:
            inputs: The inputs as the node reads them at that visit.
            options: The options of the node.
            again: Whether the pass visits the node a second time.
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
        >>> NoCasts().visit([Operand.of_value(1)], FunctionOptions(), again=False)
        Visit(steps=((),), root=None)
    """

    def visit(self, inputs: Sequence[Operand], options: FunctionOptions, *, again: bool) -> Visit:
        """Give a visit that changes nothing.

        Args:
            inputs: The inputs.
            options: The options of the node.
            again: Whether the pass visits the node a second time.

        Returns:
            The visit.

        Example:
            >>> from astrolabe._rules.options import FunctionOptions
            >>> NoCasts().visit([], FunctionOptions(), again=True).steps
            ()
        """
        del options, again
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
        >>> OperatorCasts(Operator.PLUS).visit(inputs, FunctionOptions(), again=False)
        Visit(steps=((cast(Int16, non_strict),), ()), root=None)
    """

    operator: Operator

    def visit(self, inputs: Sequence[Operand], options: FunctionOptions, *, again: bool) -> Visit:
        """Give the casts of the visit (`BinaryCasts`).

        Args:
            inputs: The left input and the right input.
            options: The options of the node, which a binary operator does not have.
            again: Whether the pass visits the node a second time.

        Returns:
            The visit.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.operators import Operator
            >>> from astrolabe._rules.options import FunctionOptions
            >>> from astrolabe._rules.type_functions import Operand
            >>> same = [Operand.column(DType("Int8"))] * 2
            >>> OperatorCasts(Operator.PLUS).visit(same, FunctionOptions(), again=True).steps
            ((), ())
        """
        del options, again
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
        >>> FunctionCasting().visit(inputs, fill, again=False)
        Visit(steps=((cast(Int16, strict),), (literal(Int16),)), root=None)
        >>> FunctionCasting().visit(inputs, fill, again=True).steps
        ((), ())
    """

    namespace: Namespace = Namespace.NONE

    def visit(self, inputs: Sequence[Operand], options: FunctionOptions, *, again: bool) -> Visit:
        """Give the casts of the visit.

        Args:
            inputs: The inputs of the function.
            options: The options of the function.
            again: Whether the pass visits the function a second time.

        Returns:
            The visit.

        Example:
            >>> from astrolabe._rules.options import FunctionOptions
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.type_functions import Operand
            >>> inputs = [Operand.column(DType("Int8"))]
            >>> FunctionCasting().visit(inputs, FunctionOptions.elementwise(), again=False).steps
            ((),)
        """
        rules = options.cast_options
        if rules is None or again:
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
        >>> cast.visit([Operand.column(DType("Int8"))], FunctionOptions(), again=False)
        Visit(steps=((cast(Int16, strict),),), root=None)
    """

    target: DType | OptionValue[DType]
    option: CastOption | OptionValue[CastOption]

    def visit(self, inputs: Sequence[Operand], options: FunctionOptions, *, again: bool) -> Visit:
        """Give the step of the cast.

        Args:
            inputs: The input of the cast.
            options: The options of the node, which a cast does not have.
            again: Whether the pass visits the node a second time.

        Returns:
            The visit.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.options import FunctionOptions
            >>> from astrolabe._rules.type_functions import Operand
            >>> cast = CastNode(DType("Int16"), CastOption.STRICT)
            >>> cast.visit([Operand.column(DType("Int8"))], FunctionOptions(), again=True).steps
            ((),)
        """
        del options
        if again:
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
        >>> OwnRuleCasts(PadLengthCasts()).visit(inputs, FunctionOptions(), again=False).steps
        ((), (cast(UInt64, strict),))
    """

    rule: FunctionRule | OptionValue[FunctionRule]

    def visit(self, inputs: Sequence[Operand], options: FunctionOptions, *, again: bool) -> Visit:
        """Give the casts of the visit.

        Args:
            inputs: The inputs of the function.
            options: The options of the function, which the rule does not read.
            again: Whether the pass visits the function a second time, which the rule does not read.

        Returns:
            The visit.

        Example:
            >>> from astrolabe._rules.cast_pass import PadLengthCasts
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.options import FunctionOptions
            >>> from astrolabe._rules.type_functions import Operand
            >>> inputs = [Operand.column(DType("String")), Operand.column(DType("UInt64"))]
            >>> OwnRuleCasts(PadLengthCasts()).visit(inputs, FunctionOptions(), again=True).steps
            ((), ())
        """
        del options, again
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
