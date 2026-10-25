"""Give the planner type of a node from the type function that polars names for it.

The planner of polars computes the type of each node before the query runs,
with the type function of the node. A type function has one of four shapes
in polars. A function uses a method of `FieldsMapper`. A binary operator, an
aggregation and a node of another kind, such as a cast, each have a rule of
their own. Each class here adapts one shape to the one call that code that
evaluates a node makes (`TypeFunction`). An adapter computes nothing of its
own. It gives the fields of the inputs, and the options that the node reads,
to the rule of the vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from astrolabe._rules.definitions.node_options import (
    Arguments,
    NodeOptions,
    OptionValue,
    OptionValues,
)
from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.type_functions import (
    AggFields,
    ExprFields,
    FieldsMapper,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from astrolabe._rules.operators import Operator
    from astrolabe._rules.type_functions import Operand


class TypeFunction(Protocol):
    """Give the planner type of a node from its inputs, as the type function of polars does."""

    def planner(self, inputs: Sequence[Operand]) -> DType:
        """Give the planner type, or raise `PolarsRefusal` where polars refuses.

        Args:
            inputs: The inputs of the node, each with its planner type and, for
                a literal, its value.
        """

    def at(self, node: NodeOptions) -> TypeFunction:
        """Give the rule at the options of one node.

        Args:
            node: The serialized root of the node.
        """


class MapperCheck:
    """Check the inputs of a function with a method of `FieldsMapper` that gives the mapper back.

    Polars chains a check before the type of some functions, for example
    `mapper.ensure_satisfies(check, "sign")?.with_same_dtype()` for `sign`. The
    check refuses the inputs with `PolarsRefusal`. Otherwise the call goes on
    with the same mapper. This adapter holds the method and its arguments.

    Example:
        >>> from astrolabe._rules.type_functions import FunctionClosures
        >>> MapperCheck(FieldsMapper.ensure_satisfies, check=FunctionClosures().numeric, op="sign")
        MapperCheck(ensure_satisfies, check=numeric, op='sign')
    """

    def __init__(self, method: Callable[..., FieldsMapper], **arguments: object) -> None:
        """Keep the method of `FieldsMapper` and its arguments.

        Args:
            method: The method, taken from the class, for example `FieldsMapper.ensure_is_list`.
            **arguments: The arguments that polars passes to the method.

        Example:
            >>> MapperCheck(FieldsMapper.ensure_is_list)._arguments.known()
            {}
        """
        self._method = method
        self._arguments = Arguments(arguments)

    def __repr__(self) -> str:
        """Give the name of the method and its arguments.

        Returns:
            The text.

        Example:
            >>> MapperCheck(FieldsMapper.ensure_is_array)
            MapperCheck(ensure_is_array)
        """
        parts = [self._method.__name__, *self._arguments.texts()]
        return f"MapperCheck({', '.join(parts)})"

    def checked(self, mapper: FieldsMapper) -> FieldsMapper:
        """Check the inputs, and give the mapper for the next call.

        It raises `PolarsRefusal` where the check refuses an input.

        Args:
            mapper: The mapper of the planner types of the inputs.

        Returns:
            The mapper.

        Example:
            >>> MapperCheck(FieldsMapper.ensure_is_list).checked(FieldsMapper(DType("Int8")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: expected List ...
        """
        return self._method(mapper, **self._arguments.known())

    def at(self, node: NodeOptions) -> MapperCheck:
        """Give the check with each argument that is an option read at one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The check.

        Example:
            >>> MapperCheck(FieldsMapper.ensure_is_list).at(NodeOptions({"Function": {}}))
            MapperCheck(ensure_is_list)
        """
        return MapperCheck(self._method, **self._arguments.at(node).known())


class Mapper:
    """Type a function with one call of the `FieldsMapper` of polars, after its checks.

    Polars types most functions with one method of its mapper, for example
    `mapper.with_same_dtype()` for `abs`, or `mapper.map_dtype(cum_sum)` with a
    closure for `cum_sum`. This adapter holds the method and its arguments,
    and calls the method on a mapper of the planner types of the inputs. An
    argument can be an option of the node, for example the field name that the
    closure of `struct.field` reads. Where polars checks the inputs first, as
    `sign` does, the checks run before the method (`MapperCheck`).

    Example:
        >>> from astrolabe._rules.type_functions import FunctionClosures
        >>> from astrolabe._rules.type_functions import Operand
        >>> Mapper(FieldsMapper.with_same_dtype).planner([Operand.column(DType("Date"))])
        Date
        >>> Mapper(FieldsMapper.map_dtype, func=FunctionClosures().cum_sum)
        Mapper(map_dtype, func=cum_sum)
    """

    def __init__(
        self,
        method: Callable[..., DType],
        *,
        checks: Sequence[MapperCheck] = (),
        **arguments: object,
    ) -> None:
        """Keep the method of `FieldsMapper`, its arguments and the checks before it.

        Args:
            method: The method, taken from the class, for example `FieldsMapper.with_same_dtype`.
            checks: The checks that polars chains before the method, in order.
            **arguments: The arguments that polars passes to the method.

        Example:
            >>> sqrt = Mapper(FieldsMapper.map_numeric_to_float_dtype, coerce_decimal=True)
            >>> sqrt._arguments.known()
            {'coerce_decimal': True}
        """
        self._method = method
        self._checks = tuple(checks)
        self._arguments = Arguments(arguments)

    def __repr__(self) -> str:
        """Give the name of the method, its arguments and its checks.

        Returns:
            The text.

        Example:
            >>> Mapper(FieldsMapper.map_to_supertype)
            Mapper(map_to_supertype)
            >>> checks = [MapperCheck(FieldsMapper.ensure_is_list)]
            >>> Mapper(FieldsMapper.with_same_dtype, checks=checks)
            Mapper(with_same_dtype, checks=(MapperCheck(ensure_is_list),))
        """
        parts = [self._method.__name__, *self._arguments.texts()]
        if self._checks:
            parts.append(f"checks={self._checks!r}")
        return f"Mapper({', '.join(parts)})"

    def planner(self, inputs: Sequence[Operand]) -> DType:
        """Give the planner type.

        It raises `PolarsRefusal` where a check or the method refuses the inputs.

        Args:
            inputs: The inputs of the function.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> sqrt = Mapper(FieldsMapper.map_numeric_to_float_dtype, coerce_decimal=True)
            >>> sqrt.planner([Operand.column(DType("Int8"))])
            Float64
        """
        mapper = FieldsMapper(*(operand.dtype for operand in inputs))
        for check in self._checks:
            mapper = check.checked(mapper)
        return self._method(mapper, **self._arguments.known())

    def at(self, node: NodeOptions) -> Mapper:
        """Give the call with each argument that is an option read at one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The call.

        Example:
            >>> from astrolabe._rules.definitions.node_options import FromOptions, NodeOption
            >>> from astrolabe._rules.type_functions import StructClosures
            >>> from astrolabe._rules.type_functions import Operand
            >>> name = FromOptions(StructClosures().field_by_name, NodeOption.text("function", "x"))
            >>> field = Mapper(FieldsMapper.try_map_field, func=name)
            >>> struct = Operand.column(DType("Struct", ((("a", DType("Date")),),)))
            >>> field.at(NodeOptions({"Function": {"function": {"x": "a"}}})).planner([struct])
            Date
        """
        checks = [check.at(node) for check in self._checks]
        return Mapper(self._method, checks=checks, **self._arguments.at(node).known())


class ExprField:
    """Type a node that is not a function with the rule of polars for its kind of node.

    A cast, a `when` or an `explode` has no `FieldsMapper`. Polars types it
    from the types of its inputs and its options (`ExprFields`). This adapter
    holds the rule and the options that it reads, and passes the planner types
    of the inputs in order.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> cast = ExprField(ExprFields.cast, target=DType("Int8"))
        >>> cast.planner([Operand.column(DType("Date"))])
        Int8
    """

    def __init__(self, method: Callable[..., DType], **arguments: object) -> None:
        """Keep the rule and its arguments.

        Args:
            method: The rule, taken from the class, for example `ExprFields.cast`.
            **arguments: The options that the rule reads.

        Example:
            >>> ExprField(ExprFields.explode)
            ExprField(explode)
        """
        self._method = method
        self._arguments = Arguments(arguments)

    def __repr__(self) -> str:
        """Give the name of the rule and its arguments.

        Returns:
            The text.

        Example:
            >>> from astrolabe._rules.definitions.node_options import NodeOption
            >>> ExprField(ExprFields.cast, target=NodeOption.dtype("dtype"))
            ExprField(cast, target=NodeOption.dtype('dtype'))
        """
        parts = [self._method.__name__, *self._arguments.texts()]
        return f"ExprField({', '.join(parts)})"

    def planner(self, inputs: Sequence[Operand]) -> DType:
        """Give the planner type.

        Args:
            inputs: The inputs of the node.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> listed = Operand.column(DType("List", (DType("Int8"),)))
            >>> ExprField(ExprFields.explode).planner([listed])
            Int8
        """
        dtypes = [operand.dtype for operand in inputs]
        return self._method(ExprFields(), *dtypes, **self._arguments.known())

    def at(self, node: NodeOptions) -> ExprField:
        """Give the rule with each argument that is an option read at one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> from astrolabe._rules.definitions.node_options import NodeOption
            >>> cast = ExprField(ExprFields.cast, target=NodeOption.dtype("dtype"))
            >>> cast.at(NodeOptions({"Cast": {"dtype": {"Literal": "Time"}}}))._arguments.known()
            {'target': Time}
        """
        return ExprField(self._method, **self._arguments.at(node).known())


@dataclass(frozen=True)
class BinaryField:
    """Type a binary operator with the rule of polars for its kind of node.

    A binary operator has no `FieldsMapper`. Polars types it from the operator
    and its two inputs (`ExprFields.binary`). The rule reads whether an input
    is a literal, because a literal with no data type joins by its value.

    Example:
        >>> from astrolabe._rules.operators import Operator
        >>> from astrolabe._rules.type_functions import Operand
        >>> inputs = [Operand.column(DType("UInt8")), Operand.of_value(300)]
        >>> BinaryField(Operator.PLUS).planner(inputs)
        UInt16
    """

    operator: Operator

    def planner(self, inputs: Sequence[Operand]) -> DType:
        """Give the planner type.

        Args:
            inputs: The left input and the right input.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> from astrolabe._rules.type_functions import Operand
            >>> BinaryField(Operator.PLUS).planner([Operand.column(DType("Boolean"))] * 2)
            UInt32
        """
        return ExprFields().binary(self.operator, inputs[0], inputs[1])

    def at(self, node: NodeOptions) -> BinaryField:
        """Give the rule, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same rule.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> BinaryField(Operator.PLUS).at(NodeOptions({"BinaryExpr": {}}))
            BinaryField(operator=<Operator.PLUS: 'Plus'>)
        """
        del node
        return self


class AggField:
    """Type an aggregation with the rule of polars for it.

    An aggregation such as `sum` is a node of its own kind, with no
    `FieldsMapper`. Polars types it with one rule of `AggFields`. Before it
    types the node, the conversion of polars gives a literal input with no
    data type the type that it takes alone. So `pl.lit(1).sum()` reads an
    `Int32` input.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> AggField(AggFields.sum).planner([Operand.of_value(2**40)])
        Int64
    """

    def __init__(self, method: Callable[[AggFields, DType], DType]) -> None:
        """Keep the rule of `AggFields`.

        Args:
            method: The rule, taken from the class, for example `AggFields.sum`.

        Example:
            >>> AggField(AggFields.sum)
            AggField(sum)
        """
        self._method = method

    def __repr__(self) -> str:
        """Give the name of the rule.

        Returns:
            The text.

        Example:
            >>> AggField(AggFields.mean)
            AggField(mean)
        """
        return f"AggField({self._method.__name__})"

    def planner(self, inputs: Sequence[Operand]) -> DType:
        """Give the planner type.

        Args:
            inputs: The input of the aggregation.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> AggField(AggFields.sum).planner([Operand.column(DType("Int8"))])
            Int64
        """
        operand = inputs[0]
        dynamic = operand.literal and Is.unknown(operand.dtype)
        dtype = operand.dtype.materialized() if dynamic else operand.dtype
        return self._method(AggFields(), dtype)

    def at(self, node: NodeOptions) -> AggField:
        """Give the rule, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same rule.

        Example:
            >>> AggField(AggFields.sum).at(NodeOptions({"Agg": {}}))
            AggField(sum)
        """
        del node
        return self


class InputCheck(Protocol):
    """Refuse inputs of a node where the planner of polars checks them before the type function."""

    def check(self, inputs: Sequence[Operand]) -> None:
        """Raise `PolarsRefusal` where the planner refuses the inputs.

        Args:
            inputs: The inputs of the node, each with its planner type and, for
                a literal, its value.
        """


class CheckedInputs:
    """Check the inputs of a node as the planner of polars does, and then give its type.

    The planner of polars checks a few nodes before it calls their type
    function, and such a check can read the value of a literal input: for
    example, the index of `arr.get` must lie inside the width of the arrays.
    This adapter runs the check (`InputCheck`), which can be built from the
    options of the node, and then the type function of the node.

    Example:
        >>> from astrolabe._rules.type_functions import ArrayIndexBounds, Operand
        >>> inner = Mapper(FieldsMapper.map_to_list_and_array_inner_dtype)
        >>> get = CheckedInputs(ArrayIndexBounds(null_on_oob=False), inner)
        >>> array = Operand.column(DType("Array", (DType("Date"), 2)))
        >>> get.planner([array, Operand.of_value(1)])
        Date
    """

    def __init__(self, check: InputCheck | OptionValue[InputCheck], then: TypeFunction) -> None:
        """Keep the check and the type function.

        Args:
            check: The check, or an option of the node that builds it.
            then: The type function of the node.

        Example:
            >>> from astrolabe._rules.type_functions import ArrayIndexBounds
            >>> same = Mapper(FieldsMapper.with_same_dtype)
            >>> CheckedInputs(ArrayIndexBounds(null_on_oob=True), same)._then
            Mapper(with_same_dtype)
        """
        self._check = check
        self._then = then

    def __repr__(self) -> str:
        """Give the check and the type function.

        Returns:
            The text.

        Example:
            >>> from astrolabe._rules.type_functions import ArrayIndexBounds
            >>> same = Mapper(FieldsMapper.with_same_dtype)
            >>> CheckedInputs(ArrayIndexBounds(null_on_oob=True), same)
            CheckedInputs(ArrayIndexBounds(null_on_oob=True), Mapper(with_same_dtype))
        """
        return f"CheckedInputs({self._check!r}, {self._then!r})"

    def planner(self, inputs: Sequence[Operand]) -> DType:
        """Give the planner type, after the check.

        It raises `PolarsRefusal` where the check or the type function refuses the inputs.

        Args:
            inputs: The inputs of the node.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.type_functions import ArrayIndexBounds, Operand
            >>> inner = Mapper(FieldsMapper.map_to_list_and_array_inner_dtype)
            >>> get = CheckedInputs(ArrayIndexBounds(null_on_oob=False), inner)
            >>> array = Operand.column(DType("Array", (DType("Date"), 2)))
            >>> get.planner([array, Operand.of_value(3)])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        OptionValues().known(self._check, "the check of the inputs").check(inputs)
        return self._then.planner(inputs)

    def at(self, node: NodeOptions) -> CheckedInputs:
        """Give the check and the type function at the options of one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> from astrolabe._rules.definitions.node_options import FromOptions, NodeOption
            >>> from astrolabe._rules.type_functions import ArrayIndexBounds
            >>> flag = NodeOption.flag("function", "ArrayExpr", "Get")
            >>> bounds = FromOptions(ArrayIndexBounds, null_on_oob=flag)
            >>> rule = CheckedInputs(bounds, Mapper(FieldsMapper.with_same_dtype))
            >>> rule.at(NodeOptions({"Function": {"function": {"ArrayExpr": {"Get": True}}}}))
            CheckedInputs(ArrayIndexBounds(null_on_oob=True), Mapper(with_same_dtype))
        """
        check = OptionValues().at(self._check, node)
        return CheckedInputs(check, self._then.at(node))


@dataclass(frozen=True)
class ImplodedField:
    """Type an `over` or a `rolling`: the type of its function, or a list of it.

    `join` says that the node puts the values of a group in a list, as an
    `over` with the mapping `join` and every `rolling` do. A function that
    gives one value for each group gives that value (`Operand.scalar`).

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> ImplodedField(join=True).planner([Operand.column(DType("Int8"))])
        List(Int8)
    """

    join: bool | OptionValue[bool]

    def planner(self, inputs: Sequence[Operand]) -> DType:
        """Give the planner type.

        Args:
            inputs: The function, then the other inputs of the node.

        Returns:
            The type.

        Example:
            >>> from dataclasses import replace
            >>> from astrolabe._rules.type_functions import Operand
            >>> one = replace(Operand.column(DType("Int8")), scalar=True)
            >>> ImplodedField(join=True).planner([one])
            Int8
        """
        join = OptionValues().known(self.join, "whether the node puts a group in a list")
        function = inputs[0]
        return DType("List", (function.dtype,)) if join and not function.scalar else function.dtype

    def at(self, node: NodeOptions) -> ImplodedField:
        """Give the rule, with the option that it reads at one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> ImplodedField(join=False).at(NodeOptions({"Over": {}})).join
            False
        """
        return ImplodedField(OptionValues().at(self.join, node))
