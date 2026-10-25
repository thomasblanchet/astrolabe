"""The cast pass of polars: the changes it makes to the inputs of a node, and the plans it refuses.

After the schema pass, polars changes the tree so that the inputs of each node
agree, before anything runs. At each visit of a node it reads the types of the
inputs at that moment, and it gives each input a list of steps: a cast with an
option, a new type for a literal, or an `implode`. A comparison with a literal
can also become a new node. This module translates the rules of one visit of
one node: a binary operator, a `when`, a function that casts its inputs to
their supertype, and `is_in`. It also translates the visit of a cast node,
which folds a cast into a literal and relaxes a strict cast that cannot fail.
The order in which the pass visits the nodes of a tree belongs to the code that
follows a tree.

Each rule takes the inputs as `Operand` values and gives a `Visit`. A rule
that refuses the plan raises `PolarsRefusal`, with the class of the error of
polars and the lines of the refusal.
"""

from __future__ import annotations

import enum
import itertools
from itertools import starmap
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.casts import (
    CastCheck,
    CastKernel,
    LosslessCast,
    Outcome,
    PhysicalValue,
    SeriesCast,
)
from astrolabe._rules.dtypes import DType, Is, Parameter, TimeUnit
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.operators import NumericOp, Operator
from astrolabe._rules.releases import polars_releases
from astrolabe._rules.supertype import LosslessUpcast, Supertype, SupertypeFlag
from astrolabe._rules.type_functions import Operand

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.kernels.windows import WindowDuration
    from astrolabe._rules.options import CastingRules


class CastOption(enum.Enum):
    """Name what a cast does with a value that does not fit its target.

    The values are those of the cast options of polars. A strict cast raises, a
    cast that is not strict gives a null, and an overflowing cast wraps the
    number.

    Example:
        >>> CastOption(2)
        <CastOption.OVERFLOWING: 2>
    """

    STRICT = 0
    NON_STRICT = 1
    OVERFLOWING = 2


class StepKind(enum.Enum):
    """Name the kinds of change that the cast pass makes to one input.

    A `CAST` puts a cast node above the input. A `LITERAL` gives a literal
    input a new type, with no cast node. An `IMPLODE` puts the input in a list.

    Example:
        >>> StepKind.LITERAL.value
        'literal'
    """

    CAST = "cast"
    LITERAL = "literal"
    IMPLODE = "implode"


@dataclass(frozen=True)
class Step:
    """Hold one change that the cast pass makes to one input, and the type it gives the input.

    Its text names the kind, the type and the option of a cast.

    Example:
        >>> Step.cast(DType("Int16"), CastOption.NON_STRICT)
        cast(Int16, non_strict)
    """

    kind: StepKind
    dtype: DType
    option: CastOption | None = None

    def __repr__(self) -> str:
        """Give the kind, the type and the option of a cast.

        Returns:
            The text.

        Example:
            >>> Step.literal(DType("UInt8"))
            literal(UInt8)
        """
        if self.option is None:
            return f"{self.kind.value}({self.dtype!r})"
        return f"{self.kind.value}({self.dtype!r}, {self.option.name.lower()})"

    @classmethod
    def cast(cls, dtype: DType, option: CastOption) -> Step:
        """Give a cast of the input.

        Args:
            dtype: The target of the cast.
            option: The option of the cast.

        Returns:
            The step.

        Example:
            >>> Step.cast(DType("Float64"), CastOption.STRICT).option
            <CastOption.STRICT: 0>
        """
        return cls(StepKind.CAST, dtype, option)

    @classmethod
    def literal(cls, dtype: DType) -> Step:
        """Give a new type for a literal input, which the pass folds at plan time.

        Args:
            dtype: The new type of the literal.

        Returns:
            The step.

        Example:
            >>> Step.literal(DType("UInt8")).kind
            <StepKind.LITERAL: 'literal'>
        """
        return cls(StepKind.LITERAL, dtype)

    @classmethod
    def implode(cls, dtype: DType) -> Step:
        """Give an `implode` of the input.

        Args:
            dtype: The type of the input inside the list.

        Returns:
            The step, which gives the list type.

        Example:
            >>> Step.implode(DType("Int8"))
            implode(List(Int8))
        """
        return cls(StepKind.IMPLODE, DType("List", (dtype,)))


class RootKind(enum.Enum):
    """Name the new node that the rewrite of a comparison with a literal gives.

    `SWAPPED` keeps the comparison, with the literal moved to the right and the
    operator swapped. `IS_NULL` and `IS_NOT_NULL` test the other input.
    `NULL_BOOLEAN` gives a null `Boolean` for each row, and `CONSTANT` gives
    one value for each row. `WHEN_NOT_NULL` gives a value where the other input
    is not null, and a null otherwise. `IS_BETWEEN` tests a range. Each new
    node gives `Boolean`, as the comparison does.

    Example:
        >>> RootKind.WHEN_NOT_NULL.value
        'when_not_null'
    """

    SWAPPED = "swapped"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    NULL_BOOLEAN = "null_boolean"
    CONSTANT = "constant"
    WHEN_NOT_NULL = "when_not_null"
    IS_BETWEEN = "is_between"


@dataclass(frozen=True)
class NewRoot:
    """Hold the new node of a rewrite, and the `Boolean` value that it gives where it has one.

    Example:
        >>> NewRoot(RootKind.WHEN_NOT_NULL, value=True)
        when_not_null(True)
    """

    kind: RootKind
    value: bool | None = None

    def __repr__(self) -> str:
        """Give the kind of the node, and its value where it has one.

        Returns:
            The text.

        Example:
            >>> NewRoot(RootKind.IS_NULL)
            is_null
        """
        return self.kind.value if self.value is None else f"{self.kind.value}({self.value})"


class VisitTurn(enum.Enum):
    """Name one visit of the cast pass to a node, by the time that it reads the inputs of the node.

    The pass reaches the inputs of the root before the root, so the root has
    one visit, which reads its inputs after their own visits (`ROOT`). The
    pass reaches every other node before its inputs (`FIRST`), and once more
    after them (`SECOND`).

    Example:
        >>> VisitTurn.ROOT.value
        'root'
    """

    ROOT = "root"
    FIRST = "first"
    SECOND = "second"


@dataclass(frozen=True)
class Visit:
    """Hold the changes of one visit of the cast pass to one node: the steps of each input.

    `steps` has one tuple for each input, in the order of the node, with the
    steps in the order that they apply. `root` names a new node where the
    pass replaces the node itself.

    Example:
        >>> Visit.unchanged(2)
        Visit(steps=((), ()), root=None)
    """

    steps: tuple[tuple[Step, ...], ...]
    root: NewRoot | None = None

    @classmethod
    def unchanged(cls, count: int) -> Visit:
        """Give a visit that changes nothing.

        Args:
            count: The number of inputs of the node.

        Returns:
            The visit.

        Example:
            >>> Visit.unchanged(1).steps
            ((),)
        """
        return cls(((),) * count)


class StepOperand:
    """Give the input that a node reads after some steps of the cast pass.

    A cast and an `implode` put a node above the input, so the input is no
    longer a literal. A new type for a literal keeps a literal, of that type.
    Polars folds the `implode` of a literal into a literal list, also where
    the literal comes from a fold of the optimizer. So the `implode` of a
    literal keeps its value, as the one value of the list, for a kernel that
    reads it. The `implode` of a literal series keeps its values.

    Example:
        >>> StepOperand().after(Operand.of_value(1), (Step.literal(DType("UInt8")),))
        Operand(dtype=UInt8, literal=True, value=1)
    """

    def after(self, operand: Operand, steps: Sequence[Step]) -> Operand:
        """Give the input after the steps.

        Args:
            operand: The input before the steps.
            steps: The steps, in the order that they apply.

        Returns:
            The input.

        Example:
            >>> StepOperand().after(Operand.of_value(1), (Step.implode(DType.dynamic_int(1)),))
            Operand(dtype=List(Unknown(int, 1)), literal=False, value=(1,))
        """
        current = operand
        for step in steps:
            if step.kind is StepKind.LITERAL:
                current = Operand.typed_literal(current.value, step.dtype)
            elif step.kind is StepKind.IMPLODE and current.literal:
                current = Operand(step.dtype, value=(current.value,))
            elif step.kind is StepKind.IMPLODE and isinstance(current.value, tuple):
                current = Operand(step.dtype, value=current.value)
            else:
                current = Operand.column(step.dtype)
        return current


class CastFold:
    """Fold a cast into a literal at plan time, as the cast pass of polars does.

    A literal with no data type casts from the type that holds its value:
    `Int128` for an integer, `Float64` for a float and `String` for text. A
    typed literal casts from its own type. The literal then takes the type that
    the cast gives (`SeriesCast`), and the planner raises where the cast
    raises, with the same error. A null literal folds only into the planner
    type `Unknown`: it becomes a null with no type. A strict cast of an integer
    to `Categorical` keeps a value only where the process holds a category with
    that id, so that fold raises `UndecidedError`.

    Example:
        >>> CastFold().of(Operand.of_value(1), DType("UInt8"), CastOption.STRICT)
        UInt8
        >>> CastFold().of(Operand.of_value(None), DType("Int8"), CastOption.STRICT) is None
        True
    """

    STORAGE: ClassVar[dict[str, DType]] = {
        "int": DType("Int128"),
        "float": DType("Float64"),
        "str": DType("String"),
    }
    UNKNOWN_TARGETS: ClassVar[frozenset[str]] = frozenset({"int", "float", "str"})

    def of(self, literal: Operand, target: DType, option: CastOption) -> DType | None:
        """Give the type of the folded literal.

        It raises `PolarsRefusal` where the cast of the value raises.

        Args:
            literal: The literal input.
            target: The target of the cast.
            option: The option of the fold.

        Returns:
            The new type of the literal, or `None` where the literal stays below a cast node.

        Example:
            >>> CastFold().of(Operand.of_value(300), DType("Int8"), CastOption.STRICT)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if literal.value is None and not Is.unknown(literal.dtype):
            return self._null(target)
        source = self._source(literal)
        if source.same(target):
            return source
        strict = option is CastOption.STRICT
        return SeriesCast(source, target, strict=strict).output([literal.value])

    def keeps(self, literal: Operand, target: DType) -> bool:
        """Say whether the value of a literal survives its fold into a target.

        A fold that is not strict turns a value that does not fit into null,
        as the cast of that value does (`CastKernel.outcome`). A kernel that
        later reads the literal reads that null.

        Args:
            literal: The literal input.
            target: The type of the fold.

        Returns:
            Whether the folded literal holds a value that is not null.

        Example:
            >>> CastFold().keeps(Operand.of_value(2**40), DType("Int8"))
            False
            >>> CastFold().keeps(Operand.of_value(1), DType("Struct", ((("a", DType("Int8")),),)))
            True
        """
        return self.outcome(literal, target) is Outcome.KEPT

    def nulled(self, literal: Operand, target: DType) -> bool:
        """Say whether the cast of a literal into a target turns its value into null.

        A kernel that casts a literal with a cast that is not strict then reads
        a null. A cast that keeps the value, or that raises, gives no null.

        Args:
            literal: The literal input.
            target: The type of the cast.

        Returns:
            Whether the value becomes null.

        Example:
            >>> CastFold().nulled(Operand.of_value("a"), DType("Int64"))
            True
            >>> listed = Operand(DType("List", (DType("Int64"),)), literal=True, value=[1])
            >>> CastFold().nulled(listed, DType("Int64"))
            False
        """
        return self.outcome(literal, target) not in {Outcome.KEPT, Outcome.RAISES}

    def outcome(self, literal: Operand, target: DType) -> Outcome:
        """Give what the cast of a literal into a target does to its value.

        A null value stays null, so its outcome is `LOST`. A kernel that casts
        a literal reads a value that is `LOST` as a null. Where the outcome is
        `RAISES`, the cast of the kernel raises.

        Args:
            literal: The literal input.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> CastFold().outcome(Operand.of_value(None), DType("Int8"))
            <Outcome.LOST: 'lost'>
            >>> listed = Operand(DType("List", (DType("Int64"),)), literal=True, value=[1])
            >>> CastFold().outcome(listed, DType("Float64"))
            <Outcome.RAISES: 'raises'>
        """
        if literal.value is None:
            return Outcome.LOST
        return CastKernel().outcome(literal.value, self._source(literal), target)

    def _source(self, literal: Operand) -> DType:
        """Give the type that a literal casts from.

        Args:
            literal: The literal input.

        Returns:
            Its type, or the type that holds its value where it has no data type.

        Example:
            >>> CastFold()._source(Operand.of_value(1.5))
            Float64
        """
        source = literal.dtype
        return self.STORAGE[str(source.args[0])] if Is.unknown(source) else source

    def _null(self, target: DType) -> DType | None:
        """Give the type of a null literal folded into a target.

        Args:
            target: The target of the cast.

        Returns:
            `Null` for a planner type `Unknown` of a literal, and `None` otherwise.

        Example:
            >>> CastFold()._null(DType.dynamic_int(1))
            Null
        """
        unknown = Is.unknown(target) and target.args[0] in self.UNKNOWN_TARGETS
        return DType("Null") if unknown else None


class CastColumnsPolicy:
    """Say whether a strict cast needs to run, by the policy that a lazy query gives it.

    The cast pass reads every strict cast with this policy. The policy allows
    an integer upcast that keeps every value, any cast between floats, any
    change of the unit or the zone of a `Datetime`, and a cast of `Null` to any
    type. It does not allow a cast of an integer to a float. A struct target
    allows missing fields and ignores extra fields. A list and an array check
    their inner types. Where the policy does not allow a cast, polars gives a
    schema error, so the policy raises `PolarsRefusal`. The cast pass ignores
    that error.

    Example:
        >>> CastColumnsPolicy().should_cast(DType("Int16"), DType("Int8"))
        True
        >>> CastColumnsPolicy().should_cast(DType("Int8"), DType("Int8"))
        False
    """

    def should_cast(self, target: DType, incoming: DType) -> bool:
        """Give the answer of the policy.

        It raises `PolarsRefusal` where the policy does not allow the cast.

        Args:
            target: The target of the cast.
            incoming: The type of the input.

        Returns:
            `True` where the cast must run, and `False` where no cast is needed.

        Example:
            >>> CastColumnsPolicy().should_cast(DType("Int8"), DType("Null"))
            True
            >>> CastColumnsPolicy().should_cast(DType("Float32"), DType("Int8"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        if incoming.name == "Null" and target.name != "Null":
            return True
        if Is.nested(target):
            return self._nested(target, incoming)
        if not Is.known(incoming):
            return True
        materialized = target.materialized()
        if materialized.same(incoming):
            return False
        return self._flat(materialized, incoming)

    def _nested(self, target: DType, incoming: DType) -> bool:
        """Read a nested target, of which the input must be of the same kind.

        It raises `PolarsRefusal` where the policy does not allow the cast of
        an inner type.

        Args:
            target: The target, a struct, a list or an array.
            incoming: The type of the input.

        Returns:
            The answer of the policy.

        Raises:
            PolarsRefusal: The input is not of the kind of the target, or an array
                has another width.

        Example:
            >>> listed = DType("List", (DType("Int16"),))
            >>> CastColumnsPolicy()._nested(listed, DType("List", (DType("Int8"),)))
            True
        """
        same_width = target.name != "Array" or target.args[1:] == incoming.args[1:]
        if target.name != incoming.name or not same_width:
            message = f"data type mismatch: incoming: {incoming!r} != target: {target!r}"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        if target.name == "Struct":
            return self._struct(target.fields(), incoming.fields())
        return self.should_cast(target.inner(), incoming.inner())

    def _struct(
        self,
        target: tuple[tuple[str, DType], ...],
        incoming: tuple[tuple[str, DType], ...],
    ) -> bool:
        """Read the fields of a struct target by name.

        A missing field or an extra field needs the cast. A field at another
        place needs it too. A field whose type the policy refuses refuses
        the whole cast, with `PolarsRefusal`.

        Args:
            target: The fields of the target.
            incoming: The fields of the input.

        Returns:
            The answer of the policy.

        Example:
            >>> field = (("a", DType("Int8")),)
            >>> CastColumnsPolicy()._struct(field, field)
            False
        """
        places = {name: (index, dtype) for index, (name, dtype) in enumerate(incoming)}
        needed = len(incoming) != len(target)
        for index, (name, dtype) in enumerate(target):
            if name not in places:
                needed = True
                continue
            place, field = places[name]
            answer = self.should_cast(dtype, field)
            needed = needed or answer or place != index
        names = {name for name, _ in target}
        return needed or any(name not in names for name, _ in incoming)

    def _flat(self, target: DType, incoming: DType) -> bool:
        """Read two flat types that differ.

        Args:
            target: The target, with its literal types materialized.
            incoming: The type of the input.

        Returns:
            `True`, where the policy allows the cast.

        Raises:
            PolarsRefusal: The policy does not allow the cast.

        Example:
            >>> moment = DType("Datetime", ("ms", "UTC"))
            >>> CastColumnsPolicy()._flat(moment, DType("Datetime", ("ns", None)))
            True
        """
        if Is.integer(target) and Is.integer(incoming):
            upcast = LosslessUpcast().of(incoming, target)
            allowed = upcast is not None and upcast.same(target)
        else:
            floats = Is.float(target) and Is.float(incoming)
            allowed = floats or target.name == incoming.name == "Datetime"
        if allowed:
            return True
        message = f"data type mismatch: incoming: {incoming!r} != target: {target!r}"
        raise PolarsRefusal(ErrorClass.SCHEMA, message)


class CastNodeVisit:
    """Give the result of the visit of the cast pass to one cast node.

    A strict cast first meets the policy of `CastColumnsPolicy`. Where no cast
    is needed, the node goes away. Where the policy allows the cast, it cannot
    fail, so it becomes an overflowing cast between two numbers and a cast that
    is not strict otherwise. Where the policy does not allow it, the cast stays
    strict. Then a cast of a literal to a known type folds into the literal,
    with the option that the cast had before the policy. A failed fold raises
    `PolarsRefusal`. From polars 1.44.1, a cast to the planner type
    `Unknown(Any)` goes away, because the engine runs it as no cast at all.

    Example:
        >>> int8 = Operand.column(DType("Int8"))
        >>> CastNodeVisit().visit(int8, DType("Int16"), CastOption.STRICT)
        cast(Int16, overflowing)
        >>> CastNodeVisit().visit(Operand.of_value(1), DType("Int8"), CastOption.STRICT)
        literal(Int8)
    """

    @polars_releases
    def visit(self, operand: Operand, target: DType, option: CastOption) -> Step | None:
        """Give the step that the cast node becomes.

        A cast to the planner type `Unknown(Any)` goes away. Every other cast
        follows the rule of the older releases (`_keeps_cast_to_any`).

        Args:
            operand: The input of the cast.
            target: The target of the cast.
            option: The option of the cast.

        Returns:
            The cast with its new option, the literal of its new type, or
            `None` where the cast goes away.

        Example:
            >>> int8 = Operand.column(DType("Int8"))
            >>> CastNodeVisit().visit(int8, DType("Int8"), CastOption.STRICT) is None
            True
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.44.1"):
            ...     CastNodeVisit().visit(int8, DType.unknown("any"), CastOption.STRICT) is None
            True
        """
        if Is.unknown_any(target):
            return None
        return self._keeps_cast_to_any(operand, target, option)

    @visit.before("1.44.1")
    def _keeps_cast_to_any(
        self,
        operand: Operand,
        target: DType,
        option: CastOption,
    ) -> Step | None:
        """Give the step that the cast node becomes, where a cast to `Unknown(Any)` stays a cast.

        Args:
            operand: The input of the cast.
            target: The target of the cast.
            option: The option of the cast.

        Returns:
            The cast with its new option, the literal of its new type, or
            `None` where the cast goes away.

        Example:
            >>> int8 = Operand.column(DType("Int8"))
            >>> any_type = DType.unknown("any")
            >>> CastNodeVisit()._keeps_cast_to_any(int8, any_type, CastOption.STRICT)
            cast(Unknown, strict)
        """
        settled = self._option(operand.dtype, target, option)
        if settled is None:
            return None
        if operand.literal and Is.known(target):
            folded = CastFold().of(operand, target, option)
            if folded is not None:
                return Step.literal(folded)
        return Step.cast(target, settled)

    def _option(self, source: DType, target: DType, option: CastOption) -> CastOption | None:
        """Give the option of the cast after the policy of a strict cast.

        Args:
            source: The type of the input.
            target: The target of the cast.
            option: The option of the cast.

        Returns:
            The option, or `None` where the policy finds that no cast is needed.

        Example:
            >>> CastNodeVisit()._option(DType("Int8"), DType("Float32"), CastOption.STRICT)
            <CastOption.STRICT: 0>
        """
        if option is not CastOption.STRICT:
            return option
        try:
            answer = CastColumnsPolicy().should_cast(target, source)
        except PolarsRefusal:
            return option
        if not answer:
            return None
        numbers = Is.primitive_numeric(source) and Is.primitive_numeric(target)
        return CastOption.OVERFLOWING if numbers else CastOption.NON_STRICT


class CastExprIr:
    """Cast one input of a function to a type, as the cast pass of polars does.

    An input of the target type stays. Otherwise `CastCheck` checks the cast,
    and refuses it where it is sure that the cast cannot run. A literal folds
    the cast with the option that the caller gives. Every other input gets a
    strict cast node, whatever the option: the visit of that node relaxes it
    later, by the policy of `CastNodeVisit`.

    Example:
        >>> int8 = Operand.column(DType("Int8"))
        >>> CastExprIr().steps(int8, DType("Int16"), CastOption.NON_STRICT)
        (cast(Int16, strict),)
    """

    def steps(self, operand: Operand, target: DType, option: CastOption) -> tuple[Step, ...]:
        """Give the steps of the input.

        Args:
            operand: The input.
            target: The type to cast to.
            option: The option of a fold into a literal.

        Returns:
            No step, a new type for a literal, or a strict cast.

        Raises:
            PolarsRefusal: `CastCheck` refuses the cast.

        Example:
            >>> date = Operand.column(DType("Date"))
            >>> CastExprIr().steps(date, DType("Boolean"), CastOption.STRICT)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if operand.dtype.same(target):
            return ()
        if CastCheck().of(operand.dtype, target) is False:
            message = f"casting from {operand.dtype!r} to {target!r} not supported"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if operand.literal:
            folded = CastFold().of(operand, target, option)
            if folded is not None:
                return (Step.literal(folded),)
        return (Step.cast(target, CastOption.STRICT),)


class ModifySupertype:
    """Change the supertype of two inputs where one is a literal.

    A string literal beside a `Categorical` or an `Enum` takes the type of that
    input, so the column is not cast. A list literal beside a list of another
    inner type takes the type of the other list.

    Example:
        >>> enum = Operand.column(DType("Enum", (("a",),)))
        >>> ModifySupertype().of(DType("String"), enum, Operand.of_value("a"))
        Enum(categories=['a'])
    """

    def of(self, supertype: DType, left: Operand, right: Operand) -> DType:
        """Give the supertype after the change.

        Args:
            supertype: The supertype of the two inputs.
            left: The left input.
            right: The right input.

        Returns:
            The supertype that the casts use.

        Example:
            >>> int8 = Operand.column(DType("Int8"))
            >>> ModifySupertype().of(DType("Int16"), int8, Operand.of_value(300))
            Int16
            >>> listed, narrow = DType("List", (DType("Int64"),)), DType("List", (DType("Int8"),))
            >>> literal = Operand.typed_literal([1], narrow)
            >>> column, modify = Operand.column(listed), ModifySupertype()
            >>> modify.of(listed, column, literal), modify.of(listed, literal, column)
            (List(Int64), List(Int64))
        """
        ltype, rtype = left.dtype, right.dtype
        if ltype.name in Is.CATEGORIES and Is.string(rtype) and right.literal:
            return ltype
        if Is.string(ltype) and rtype.name in Is.CATEGORIES and left.literal:
            return rtype
        lists = ltype.name == rtype.name == "List"
        if lists and not ltype.inner().same(rtype.inner()):
            if right.literal:
                return ltype
            if left.literal:
                return rtype
        return supertype


class ComparisonLiteral:
    """Rewrite a comparison of an input with a literal, as the cast pass of polars does.

    It casts the literal to the type of the other input where the value fits,
    so the input needs no cast. Where the value does not fit, every row that is
    not null compares the same way, so the comparison becomes a `when` that
    gives that value. A null literal makes `eq_missing` an `is_null` and every
    other comparison a null. The rewrite applies only to two integers, or to a
    `Datetime` or a `Duration` beside a value of its kind, and only where the
    supertype keeps every value of the input.

    The literal reads as polars materializes it: an integer with no data type
    takes the type that it takes alone.

    Example:
        >>> uint8 = DType("UInt8")
        >>> ComparisonLiteral().rewrite(uint8, Operator.LT, Operand.of_value(300))
        when_not_null(True)
        >>> ComparisonLiteral().rewrite(uint8, Operator.EQ, Operand.of_value(1))
        UInt8
    """

    NULL_TESTS: ClassVar[dict[Operator, RootKind]] = {
        Operator.EQ_VALIDITY: RootKind.IS_NULL,
        Operator.NOT_EQ_VALIDITY: RootKind.IS_NOT_NULL,
    }
    CONSTANTS: ClassVar[dict[Operator, NewRoot]] = {
        Operator.EQ_VALIDITY: NewRoot(RootKind.CONSTANT, value=False),
        Operator.NOT_EQ_VALIDITY: NewRoot(RootKind.CONSTANT, value=True),
        Operator.EQ: NewRoot(RootKind.WHEN_NOT_NULL, value=False),
        Operator.NOT_EQ: NewRoot(RootKind.WHEN_NOT_NULL, value=True),
    }

    def rewrite(self, lhs: DType, op: Operator, literal: Operand) -> DType | NewRoot | None:
        """Give the rewrite of `lhs <op> literal`.

        Args:
            lhs: The type of the input that is not the literal.
            op: The operator, with the literal on the right.
            literal: The literal.

        Returns:
            The new type of the literal, a new node, or `None` where the comparison stays.

        Example:
            >>> ComparisonLiteral().rewrite(DType("Int8"), Operator.EQ, Operand.of_value(None))
            null_boolean
        """
        scalar = literal.dtype.materialized()
        supertype = Supertype().of(lhs, scalar)
        if supertype is None:
            return None
        if literal.value is None:
            return self._null(lhs, op, supertype)
        if lhs.same(scalar) or not self._applies(lhs, scalar, supertype):
            return None
        if self._fits(literal.value, scalar, lhs):
            return self._replaced(lhs, op, scalar)
        return self._out_of_range(lhs, op, literal.value, scalar, supertype)

    def _applies(self, lhs: DType, scalar: DType, supertype: DType) -> bool:
        """Say whether the types allow the rewrite.

        Args:
            lhs: The type of the other input.
            scalar: The type of the literal.
            supertype: Their supertype.

        Returns:
            Whether the kinds match and the supertype keeps every value of the input.

        Example:
            >>> ComparisonLiteral()._applies(DType("Int64"), DType("UInt64"), DType("Float64"))
            False
        """
        matching = self._matching(lhs, scalar) and self._matching(lhs, supertype)
        return matching and not self._introduces_nulls(lhs, supertype)

    def _matching(self, lhs: DType, other: DType) -> bool:
        """Say whether two types are of the kinds that the rewrite takes.

        Args:
            lhs: The type of the other input.
            other: The type of the literal or of the supertype.

        Returns:
            Whether both are integers, or a `Datetime` beside a moment, or two durations.

        Example:
            >>> ComparisonLiteral()._matching(DType("Datetime", ("ms", None)), DType("Date"))
            True
        """
        integers = Is.integer(lhs) and Is.integer(other)
        moments = lhs.name == "Datetime" and other.name in {"Datetime", "Date"}
        return integers or moments or lhs.name == other.name == "Duration"

    def _introduces_nulls(self, lhs: DType, supertype: DType) -> bool:
        """Say whether a cast of the input to the supertype can lose a value.

        Args:
            lhs: The type of the other input.
            supertype: The supertype.

        Returns:
            Whether the input is an integer that does not cast to the supertype without loss.

        Example:
            >>> ComparisonLiteral()._introduces_nulls(DType("UInt128"), DType("Int128"))
            True
        """
        if not Is.integer(lhs) or lhs.same(supertype):
            return False
        upcast = LosslessUpcast().of(lhs, supertype)
        return upcast is None or not upcast.same(supertype)

    def _null(self, lhs: DType, op: Operator, supertype: DType) -> NewRoot | None:
        """Give the rewrite of a comparison with a null literal.

        Args:
            lhs: The type of the other input.
            op: The operator.
            supertype: The supertype of the input and the literal.

        Returns:
            `is_null` or `is_not_null` for the validity forms, a null for the
            rest, or `None` where the supertype can lose a value.

        Example:
            >>> ComparisonLiteral()._null(DType("Int8"), Operator.NOT_EQ_VALIDITY, DType("Int8"))
            is_not_null
            >>> uint64, int64 = DType("UInt64"), DType("Int64")
            >>> ComparisonLiteral()._null(uint64, Operator.EQ_VALIDITY, int64) is None
            True
        """
        if op not in self.NULL_TESTS:
            return NewRoot(RootKind.NULL_BOOLEAN)
        if self._introduces_nulls(lhs, supertype):
            return None
        return NewRoot(self.NULL_TESTS[op])

    def _fits(self, value: object, scalar: DType, lhs: DType) -> bool:
        """Say whether a cast of the literal to the type of the input keeps the value.

        It is not strict, so a value that does not fit becomes a null.

        Args:
            value: The value of the literal.
            scalar: The type of the literal.
            lhs: The type of the other input.

        Returns:
            Whether the cast runs and keeps the value.

        Example:
            >>> ComparisonLiteral()._fits(300, DType("Int32"), DType("UInt8"))
            False
        """
        try:
            CastKernel().output(scalar, lhs)
        except PolarsRefusal:  # pragma: no cover -- the rewrite reads integers and moments only,
            return False  # whose casts to each other exist.
        return CastKernel().outcome(value, scalar, lhs) in {Outcome.KEPT, Outcome.HIDDEN}

    def _replaced(self, lhs: DType, op: Operator, scalar: DType) -> DType | NewRoot | None:
        """Give the rewrite of a literal that fits the type of the input.

        A temporal input compares `<=` with the last value of the coarser
        unit of the literal, and `==` becomes `is_between`. Both need a
        literal of a coarser unit than the input. The other operators take
        the literal cast to the type of the input.

        Args:
            lhs: The type of the other input.
            op: The operator.
            scalar: The type of the literal.

        Returns:
            The new type of the literal, `is_between`, or `None`.

        Example:
            >>> moment = DType("Datetime", ("ns", None))
            >>> ComparisonLiteral()._replaced(moment, Operator.EQ, DType("Datetime", ("ms", None)))
            is_between
        """
        if not Is.temporal(lhs) or op not in {Operator.LT_EQ, Operator.EQ}:
            return lhs
        if not self._has_upper_bound(scalar, lhs):
            return None
        return lhs if op is Operator.LT_EQ else NewRoot(RootKind.IS_BETWEEN)

    def _has_upper_bound(self, scalar: DType, lhs: DType) -> bool:
        """Say whether the literal has a coarser unit than the input.

        Args:
            scalar: The type of the literal.
            lhs: The type of the input.

        Returns:
            Whether both are a `Datetime` or a `Duration`, and the literal unit is coarser.

        Example:
            >>> micro, milli = DType("Duration", ("us",)), DType("Duration", ("ms",))
            >>> ComparisonLiteral()._has_upper_bound(micro, milli)
            False
        """
        if scalar.name != lhs.name or scalar.name not in {"Datetime", "Duration"}:
            return False
        return TimeUnit.finer(Parameter().unit(lhs), Parameter().unit(scalar))

    def _out_of_range(
        self,
        lhs: DType,
        op: Operator,
        value: object,
        scalar: DType,
        supertype: DType,
    ) -> NewRoot | None:
        """Give the rewrite of a literal that does not fit the type of the input.

        Every value of the input is below or above the literal. So `==` and
        `!=` give a fixed answer, and the order of the literal and zero decides
        `<`, `<=`, `>` and `>=`.

        Args:
            lhs: The type of the other input.
            op: The operator.
            value: The value of the literal.
            scalar: The type of the literal.
            supertype: The supertype of the input and the literal.

        Returns:
            The new node, or `None` where the comparison stays.

        Example:
            >>> uint8, int32, int16 = DType("UInt8"), DType("Int32"), DType("Int16")
            >>> ComparisonLiteral()._out_of_range(uint8, Operator.GT, -1, int32, int16)
            when_not_null(True)
            >>> from datetime import datetime
            >>> nano, micro = DType("Datetime", ("ns", None)), DType("Datetime", ("us", None))
            >>> far = datetime(3000, 1, 1)
            >>> ComparisonLiteral()._out_of_range(nano, Operator.GT, far, micro, nano)
            null_boolean
        """
        kernel = CastKernel()
        try:
            kernel.output(scalar, supertype)
        except PolarsRefusal:  # pragma: no cover -- the rewrite reads integers and moments only,
            return None  # whose casts to their supertype exist.
        outcome = kernel.outcome(value, scalar, supertype)
        if outcome is Outcome.RAISES:  # pragma: no cover -- a cast that is not strict between
            return None  # integers or moments loses a value, and does not raise.
        if outcome is Outcome.LOST:
            return self._null(lhs, op, supertype)
        if op in self.CONSTANTS:
            return self.CONSTANTS[op]
        below = self._below_zero(value, scalar)
        if below is None:  # pragma: no cover -- the physical value of an integer or a moment
            return None  # is an integer.
        above = op in {Operator.LT, Operator.LT_EQ}
        return NewRoot(RootKind.WHEN_NOT_NULL, not below if above else below)

    def _below_zero(self, value: object, scalar: DType) -> bool | None:
        """Say whether the physical value of the literal is below zero.

        Args:
            value: The value of the literal.
            scalar: The type of the literal.

        Returns:
            `False` for an unsigned type, whether the value is negative for
            an integer, and `None` where the value is not an integer.

        Example:
            >>> ComparisonLiteral()._below_zero(-5, DType("Int32"))
            True
        """
        if Is.unsigned(scalar.physical()):
            return False
        physical = PhysicalValue().of(value, scalar)
        return physical < 0 if isinstance(physical, int) else None


class CoercedBinopDtype:
    """Give the type that the cast pass casts both inputs of a binary operator to.

    A comparison of text with a number gets no cast, so the run refuses it. A
    comparison of a moment with text refuses the plan. Arithmetic on temporal
    types, lists and arrays gets no cast here, and arithmetic of `String` with
    a number refuses the plan. The other pairs take their supertype, changed by
    `ModifySupertype`. A text type beside `String` takes `String`.

    Example:
        >>> int8, float64 = Operand.column(DType("Int8")), Operand.column(DType("Float64"))
        >>> CoercedBinopDtype().of(Operator.PLUS, int8, float64)
        Float64
    """

    DATES: ClassVar[frozenset[str]] = frozenset({"Date", "Datetime"})
    UNCAST: ClassVar[frozenset[str]] = frozenset({
        "Datetime",
        "Date",
        "Duration",
        "Time",
        "List",
        "Array",
    })

    def of(self, op: Operator, left: Operand, right: Operand) -> DType | None:
        """Give the type, or `None` where the pass casts nothing.

        Args:
            op: The operator.
            left: The left input.
            right: The right input.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The pass refuses the pair.

        Example:
            >>> date, text = Operand.column(DType("Date")), Operand.column(DType("String"))
            >>> CoercedBinopDtype().of(Operator.EQ, date, text)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        ltype, rtype = left.dtype, right.dtype
        compares = op.is_comparison_or_bitwise()
        if compares and self._text_beside_number(ltype, rtype):
            return None
        if compares and self._moment_beside_text(ltype, rtype):
            message = "cannot compare 'date/datetime/time' to a string value"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if ltype.name == rtype.name == "Struct":
            return None
        if op.is_arithmetic() and self._arithmetic_uncast(ltype, rtype):
            return None
        if compares and self._text_beside_categorical(ltype, rtype):
            return None
        return self._supertype(left, right)

    def _text(self, dtype: DType) -> bool:
        """Say whether a type is text in the sense of the comparison arms.

        Args:
            dtype: The type.

        Returns:
            Whether it is `String`, a string literal, `Categorical` or `Enum`.

        Example:
            >>> CoercedBinopDtype()._text(DType("Enum", (("a",),)))
            True
        """
        return Is.string(dtype) or dtype.name in Is.CATEGORIES

    def _text_beside_number(self, left: DType, right: DType) -> bool:
        """Say whether a text type meets a number.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            Whether one input is text and the other a number.

        Example:
            >>> CoercedBinopDtype()._text_beside_number(DType("Categorical"), DType.dynamic_int(1))
            True
        """
        forward = self._text(left) and Is.primitive_numeric(right)
        return forward or (self._text(right) and Is.primitive_numeric(left))

    def _moment_beside_text(self, left: DType, right: DType) -> bool:
        """Say whether a moment meets text, which the pass refuses.

        A `Date` or a `Datetime` refuses `String` or a string literal on
        either side. A `Time`, or a string literal, refuses `String` only on
        the right.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            Whether the pass refuses the pair.

        Example:
            >>> CoercedBinopDtype()._moment_beside_text(DType("String"), DType("Time"))
            False
        """
        forward = left.name in self.DATES and Is.string(right)
        either = forward or (right.name in self.DATES and Is.string(left))
        literal_text = Is.unknown(left) and left.args[0] == "str"
        time = (left.name == "Time" or literal_text) and right.name == "String"
        return either or time

    def _arithmetic_uncast(self, left: DType, right: DType) -> bool:
        """Say whether arithmetic leaves the pair uncast here.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            Whether a temporal type, a list or an array is on a side.

        Raises:
            PolarsRefusal: `String` meets a number.

        Example:
            >>> CoercedBinopDtype()._arithmetic_uncast(DType("Duration", ("us",)), DType("Int8"))
            True
        """
        forward = left.name == "String" and Is.primitive_numeric(right)
        if forward or (right.name == "String" and Is.primitive_numeric(left)):
            message = f"arithmetic on dtypes {left!r} and {right!r} is not allowed"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return bool({left.name, right.name} & self.UNCAST)

    def _text_beside_categorical(self, left: DType, right: DType) -> bool:
        """Say whether a comparison meets text beside categories.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            Whether one input is text and the other a `Categorical` or an `Enum`.

        Example:
            >>> CoercedBinopDtype()._text_beside_categorical(DType("String"), DType("Categorical"))
            True
        """
        forward = Is.string(left) and right.name in Is.CATEGORIES
        return forward or (Is.string(right) and left.name in Is.CATEGORIES)

    def _supertype(self, left: Operand, right: Operand) -> DType | None:
        """Give the supertype that both inputs take.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The type, or `None` where there is none or it is a planner type `Unknown`.

        Example:
            >>> text, enum = DType("String"), DType("Enum", (("a",),))
            >>> CoercedBinopDtype()._supertype(Operand.column(text), Operand.column(enum))
            String
        """
        ltype, rtype = left.dtype, right.dtype
        joined = Supertype().of(ltype, rtype)
        if joined is None:
            return None
        joined = ModifySupertype().of(joined, left, right)
        names = {ltype.name, rtype.name}
        if "String" in names and names & Is.CATEGORIES:
            joined = DType("String")
        return None if Is.unknown(joined) else joined


class BinaryCasts:
    """Give the changes of one visit of the cast pass to a binary operator.

    A literal type beside the planner type `Unknown` of an input takes the type
    that it takes alone. An integer literal type beside a float literal type
    casts to the float literal type. A comparison with a literal first tries
    the rewrite of `ComparisonLiteral`. Two inputs of one type stay. In arithmetic, a number
    beside a struct casts to a struct of the first field, and a list or an
    array casts only where its leaf is a planner type `Unknown`. Every other
    pair casts each input that differs to the type of `CoercedBinopDtype`.
    Every cast of this visit is not strict.

    Example:
        >>> int8, int32 = Operand.column(DType("Int8")), Operand.column(DType("Int32"))
        >>> BinaryCasts().visit(Operator.PLUS, int8, int32)
        Visit(steps=((cast(Int32, non_strict),), ()), root=None)
    """

    LIST_OPS: ClassVar[dict[Operator, NumericOp]] = {
        Operator.PLUS: NumericOp.ADD,
        Operator.MINUS: NumericOp.SUB,
        Operator.MULTIPLY: NumericOp.MUL,
        Operator.TRUE_DIVIDE: NumericOp.DIV,
        Operator.RUST_DIVIDE: NumericOp.DIV,
        Operator.FLOOR_DIVIDE: NumericOp.FLOOR_DIV,
        Operator.MODULUS: NumericOp.REM,
    }

    def visit(self, op: Operator, left: Operand, right: Operand) -> Visit:
        """Give the changes of the visit.

        Args:
            op: The operator.
            left: The left input.
            right: The right input.

        Returns:
            The steps of the two inputs, and the new node of a rewrite.

        Example:
            >>> uint8 = Operand.column(DType("UInt8"))
            >>> BinaryCasts().visit(Operator.LT, Operand.of_value(-1), uint8).root
            when_not_null(True)
        """
        literal_types = self._beside_any(left, right) or self._beside_float(left.dtype, right.dtype)
        if literal_types is not None:
            return literal_types
        if op.is_comparison():
            rewritten = self._literal_comparison(op, left, right)
            if rewritten is not None:
                return rewritten
        ltype, rtype = left.dtype, right.dtype
        if ltype.same(rtype):
            return Visit.unchanged(2)
        arithmetic = self._arithmetic(op, ltype, rtype) if op.is_arithmetic() else None
        if arithmetic is not None:
            return arithmetic
        joined = CoercedBinopDtype().of(op, left, right)
        if joined is None:
            return Visit.unchanged(2)
        return Visit((self._cast(ltype, joined), self._cast(rtype, joined)))

    def _arithmetic(self, op: Operator, left: DType, right: DType) -> Visit | None:
        """Give the changes of arithmetic on a struct beside a number, or on a list or an array.

        Args:
            op: The operator, an arithmetic one.
            left: The left type.
            right: The right type.

        Returns:
            The changes, or `None` where neither arm applies.

        Example:
            >>> BinaryCasts()._arithmetic(Operator.PLUS, DType("Int8"), DType("Int16")) is None
            True
        """
        if self._struct_beside_number(left, right):
            return self._struct(left, right)
        if {left.name, right.name} & Is.LISTS:
            return self._nested(op, left, right)
        return None

    def _cast(self, dtype: DType, joined: DType) -> tuple[Step, ...]:
        """Give the cast of one input to the joined type, where its type differs.

        Args:
            dtype: The type of the input.
            joined: The type that both inputs take.

        Returns:
            No step, or a cast that is not strict.

        Example:
            >>> BinaryCasts()._cast(DType("Int8"), DType("Int8"))
            ()
        """
        return () if dtype.same(joined) else (Step.cast(joined, CastOption.NON_STRICT),)

    def _beside_any(self, left: Operand, right: Operand) -> Visit | None:
        """Give the changes beside an input of the planner type `Unknown(Any)`.

        Two inputs of that type stay. A literal type beside it takes the type
        that it takes alone, where the input is a literal node. Any other pair
        goes on to the other rules.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The changes, or `None` where the pair goes on to the other rules.

        Example:
            >>> any_input = Operand.column(DType.unknown("any"))
            >>> BinaryCasts()._beside_any(any_input, Operand.of_value(1))
            Visit(steps=((), (literal(Int32),)), root=None)
            >>> BinaryCasts()._beside_any(any_input, Operand.of_value("a")) is None
            True
            >>> BinaryCasts()._beside_any(any_input, any_input).steps
            ((), ())
        """
        left_any, right_any = Is.unknown_any(left.dtype), Is.unknown_any(right.dtype)
        if left_any and right_any:
            return Visit.unchanged(2)
        if right_any and self._literal_type(left.dtype):
            return self._materialized(left, 0)
        if left_any and self._literal_type(right.dtype):
            return self._materialized(right, 1)
        return None

    def _literal_type(self, dtype: DType) -> bool:
        """Say whether a type is a planner type `Unknown` of a literal: an integer, a float or text.

        Args:
            dtype: The type.

        Returns:
            Whether it is.

        Example:
            >>> BinaryCasts()._literal_type(DType.unknown("float"))
            True
        """
        return Is.unknown(dtype) and not Is.unknown_any(dtype)

    def _beside_float(self, left: DType, right: DType) -> Visit | None:
        """Give the cast of an integer literal type beside a float literal type.

        The integer side casts to the float literal type with a cast that is
        not strict, so the kernel reads two floats. The cast does not fold into
        a literal, because its target is not a known type. Any other pair goes
        on to the other rules.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The changes, or `None` where the pair goes on to the other rules.

        Example:
            >>> dynamic, floats = DType.dynamic_int(7), DType.unknown("float")
            >>> visit = BinaryCasts()._beside_float(floats, dynamic)
            >>> visit.steps  # the right side casts to the float literal type
            ((), (cast(Unknown, non_strict),))
            >>> visit.steps[1][0].dtype.same(floats)
            True
            >>> BinaryCasts()._beside_float(dynamic, dynamic) is None
            True
        """
        cast = (Step.cast(DType.unknown("float"), CastOption.NON_STRICT),)
        if Is.dynamic_int(left) and self._dynamic_float(right):
            return Visit((cast, ()))
        if self._dynamic_float(left) and Is.dynamic_int(right):
            return Visit(((), cast))
        return None

    def _dynamic_float(self, dtype: DType) -> bool:
        """Say whether a type is the planner type of a float literal with no data type.

        Args:
            dtype: The type.

        Returns:
            Whether it is.

        Example:
            >>> BinaryCasts()._dynamic_float(DType("Float64"))
            False
        """
        return Is.unknown(dtype) and Is.float(dtype)

    def _materialized(self, operand: Operand, index: int) -> Visit:
        """Give a literal input the type that it takes alone, where it is a literal node.

        Args:
            operand: The input of a literal type.
            index: Its place in the node.

        Returns:
            The changes: none where the input is not a literal node.

        Example:
            >>> BinaryCasts()._materialized(Operand.of_value(1.5), 0)
            Visit(steps=((literal(Float64),), ()), root=None)
            >>> BinaryCasts()._materialized(Operand.column(DType("Int8")), 1).steps
            ((), ())
        """
        if not operand.literal:
            return Visit.unchanged(2)
        step = (Step.literal(operand.dtype.materialized()),)
        return Visit(((), step) if index == 1 else (step, ()))

    def _literal_comparison(self, op: Operator, left: Operand, right: Operand) -> Visit | None:
        """Give the rewrite of a comparison with a literal.

        A literal on the right is read first. A literal on the left is read
        only where the right input is not a literal, with the operator
        swapped, and its new type moves it to the right.

        Args:
            op: The operator.
            left: The left input.
            right: The right input.

        Returns:
            The changes, or `None` where the comparison stays.

        Example:
            >>> uint8 = Operand.column(DType("UInt8"))
            >>> BinaryCasts()._literal_comparison(Operator.GT, Operand.of_value(1), uint8)
            Visit(steps=((literal(UInt8),), ()), root=swapped)
        """
        if right.literal:
            rewrite = ComparisonLiteral().rewrite(left.dtype, op, right)
            if isinstance(rewrite, DType):
                return Visit(((), (Step.literal(rewrite),)))
            return None if rewrite is None else Visit(((), ()), rewrite)
        swapped = op.swapped()
        if not left.literal or swapped is None:
            return None
        rewrite = ComparisonLiteral().rewrite(right.dtype, swapped, left)
        if isinstance(rewrite, DType):
            return Visit(((Step.literal(rewrite),), ()), NewRoot(RootKind.SWAPPED))
        return None if rewrite is None else Visit(((), ()), rewrite)

    def _struct_beside_number(self, left: DType, right: DType) -> bool:
        """Say whether a struct meets a number in arithmetic.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            Whether one input is a struct and the other a number.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> BinaryCasts()._struct_beside_number(DType("Int8"), struct)
            True
        """
        forward = left.name == "Struct" and Is.primitive_numeric(right)
        return forward or (right.name == "Struct" and Is.primitive_numeric(left))

    def _struct(self, left: DType, right: DType) -> Visit:
        """Cast the number to a struct of the first field of the struct.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The changes: none for a struct with no fields.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")), ("b", DType("Date"))),))
            >>> BinaryCasts()._struct(struct, DType("Float32"))
            Visit(steps=((), (cast(Struct({'a': Int8}), non_strict),)), root=None)
            >>> BinaryCasts()._struct(DType("Struct", ((),)), DType("Float32")).steps
            ((), ())
        """
        struct, index = (left, 1) if left.name == "Struct" else (right, 0)
        fields = struct.fields()
        if not fields:
            return Visit.unchanged(2)
        cast = (Step.cast(DType("Struct", (fields[:1],)), CastOption.NON_STRICT),)
        return Visit(((), cast) if index == 1 else (cast, ()))

    def _nested(self, op: Operator, left: DType, right: DType) -> Visit:
        """Cast a list or an array whose leaf is a planner type `Unknown`.

        The pass maps the operator to its list operation, and casts nothing
        where the leaves have no join.

        Args:
            op: The operator.
            left: The left type.
            right: The right type.

        Returns:
            The changes: each input with a leaf `Unknown` casts its leaf to the joined leaf.

        Example:
            >>> listed = DType("List", (DType.dynamic_int(1),))
            >>> BinaryCasts()._nested(Operator.PLUS, listed, DType("Int8"))
            Visit(steps=((cast(List(Int8), non_strict),), ()), root=None)
        """
        list_op = self.LIST_OPS.get(op)
        if list_op is None:
            return Visit.unchanged(2)
        try:
            leaf = list_op.leaf_supertype(left.leaf(), right.leaf())
        except PolarsRefusal:
            return Visit.unchanged(2)
        if Is.unknown(leaf):
            return Visit.unchanged(2)
        steps = []
        for dtype in (left, right):
            cast = Step.cast(dtype.with_leaf(leaf), CastOption.NON_STRICT)
            steps.append((cast,) if Is.unknown(dtype.leaf()) else ())
        return Visit((steps[0], steps[1]))


class TernaryCasts:
    """Give the changes of one visit of the cast pass to `when(...).then(...).otherwise(...)`.

    The two inputs are `then` and `otherwise`: the condition never casts. Two
    inputs of one type stay, and so do two inputs with no supertype. The others
    cast each input that differs to their supertype, changed by
    `ModifySupertype`, with a strict cast.

    Example:
        >>> then, otherwise = Operand.column(DType("Int8")), Operand.of_value(300)
        >>> TernaryCasts().visit(then, otherwise)
        Visit(steps=((cast(Int16, strict),), (cast(Int16, strict),)), root=None)
    """

    def visit(self, truthy: Operand, falsy: Operand) -> Visit:
        """Give the changes of the visit.

        Args:
            truthy: The `then` input.
            falsy: The `otherwise` input.

        Returns:
            The steps of the two inputs.

        Example:
            >>> date, boolean = Operand.column(DType("Date")), Operand.column(DType("Boolean"))
            >>> TernaryCasts().visit(date, boolean)
            Visit(steps=((), ()), root=None)
        """
        ttype, ftype = truthy.dtype, falsy.dtype
        if ttype.same(ftype):
            return Visit.unchanged(2)
        joined = Supertype().of(ttype, ftype)
        if joined is None:
            return Visit.unchanged(2)
        joined = ModifySupertype().of(joined, truthy, falsy)
        cast = (Step.cast(joined, CastOption.STRICT),)
        steps = [() if dtype.same(joined) else cast for dtype in (ttype, ftype)]
        return Visit((steps[0], steps[1]))


class WhenCasts:
    """Give the changes of one visit of the cast pass to a `when` node, from its three inputs.

    The node holds the mask, `then` and `otherwise`. The mask never casts,
    and the two branches cast as `TernaryCasts` says.

    Example:
        >>> mask, column = Operand.column(DType("Boolean")), Operand.column(DType("Int8"))
        >>> inputs = [mask, column, Operand.of_value(300)]
        >>> WhenCasts().visit(inputs).steps
        ((), (cast(Int16, strict),), (cast(Int16, strict),))
    """

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        Args:
            operands: The mask, `then` and `otherwise`.

        Returns:
            The steps of the three inputs.

        Example:
            >>> inputs = [Operand.column(DType("Boolean"))] * 3
            >>> WhenCasts().visit(inputs).steps
            ((), (), ())
        """
        branches = TernaryCasts().visit(operands[1], operands[2])
        return Visit(((), *branches.steps))


class EwmCasts:
    """Give the changes of one visit of the cast pass to an exponentially weighted function.

    `ewm_mean`, `ewm_sum`, `ewm_std` and `ewm_var` have a rule of their own in
    the cast pass, and no casting rule in their options. The rule casts an
    input that is not a float to `Float64`, with a cast that is not strict. A
    float, a float literal included, stays. The rule applies at each visit,
    and a second visit finds a float. From polars 1.44.0, the rule refuses a
    `Struct`, whose cast to a number gives a `Struct` again. The rule also
    refuses a weight `alpha` outside of 0 to 1, which the Python API of
    polars never builds.

    Example:
        >>> EwmCasts().visit([Operand.column(DType("Int8"))])
        Visit(steps=((cast(Float64, non_strict),),), root=None)
        >>> EwmCasts().visit([Operand.of_value(1.5)])
        Visit(steps=((),), root=None)
    """

    TARGET: ClassVar[DType] = DType("Float64")

    @polars_releases
    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the steps of the one input.

        Args:
            operands: The one input.

        Returns:
            The visit.

        Raises:
            PolarsRefusal: The input is a `Struct`.

        Example:
            >>> EwmCasts().visit([Operand.column(DType("Float16"))]).steps
            ((),)
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> nested = Operand.column(DType("Struct", ((("a", DType("Int8")),),)))
            >>> with PolarsRelease.pretend("1.44.0"):
            ...     EwmCasts().visit([nested])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `ewm` operation ...
        """
        dtype = operands[0].dtype
        if dtype.name == "Struct":
            message = f"`ewm` operation not supported for dtype `{dtype!r}`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return self._casts_any_input(operands)

    @visit.before("1.44.0")
    def _casts_any_input(self, operands: Sequence[Operand]) -> Visit:
        """Give the steps of the one input, where the rule casts a `Struct` as it casts any type.

        Args:
            operands: The one input.

        Returns:
            The visit.

        Example:
            >>> EwmCasts()._casts_any_input([Operand.column(DType("Int8"))]).steps
            ((cast(Float64, non_strict),),)
        """
        if Is.float(operands[0].dtype):
            return Visit.unchanged(1)
        return Visit(((Step.cast(self.TARGET, CastOption.NON_STRICT),),))


class DurationCasts:
    """Cast each part of `pl.duration` to `Int64` or to `Float64`, as the cast pass does.

    The cast pass of polars has a rule of its own for `pl.duration`. Where
    every part is an `Int64` or a `Float64`, it changes nothing. Otherwise it
    casts an integer to `Int64` and a float to `Float64`, with a strict cast
    (`CastExprIr`), and it refuses a part of any other type, `Null` included.

    Example:
        >>> parts = [Operand.column(DType("Int8")), Operand.of_value(0)]
        >>> DurationCasts().visit(parts)
        Visit(steps=((cast(Int64, strict),), (literal(Int64),)), root=None)
    """

    KEPT: ClassVar[frozenset[str]] = frozenset({"Int64", "Float64"})

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the steps of each part.

        It raises `PolarsRefusal` where a part is not a number (`_steps`).

        Args:
            operands: The parts.

        Returns:
            The visit.

        Example:
            >>> DurationCasts().visit([Operand.column(DType("String"))])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: expected integer or ...
        """
        if all(operand.dtype.name in self.KEPT for operand in operands):
            return Visit.unchanged(len(operands))
        steps: list[tuple[Step, ...]] = []
        for place, operand in enumerate(operands):
            steps.append(self._steps(place, operand))
        return Visit(tuple(steps))

    def _steps(self, place: int, operand: Operand) -> tuple[Step, ...]:
        """Give the steps of one part.

        Args:
            place: The place of the part, for the message.
            operand: The part.

        Returns:
            The steps.

        Raises:
            PolarsRefusal: The part is not an integer or a float.

        Example:
            >>> DurationCasts()._steps(0, Operand.column(DType("Float32")))
            (cast(Float64, strict),)
        """
        if operand.dtype.name in self.KEPT:
            return ()
        if Is.integer(operand.dtype):
            return CastExprIr().steps(operand, DType("Int64"), CastOption.STRICT)
        if Is.float(operand.dtype):
            return CastExprIr().steps(operand, DType("Float64"), CastOption.STRICT)
        message = (
            f"expected integer or float dtype, (got {operand.dtype!r}) in input {place} of duration"
        )
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


class Namespace(enum.Enum):
    """Name the namespace of a function, which the cast pass checks on its first input.

    The check runs only for a function that casts its inputs to their
    supertype.

    Example:
        >>> Namespace.STRING.value
        'str'
    """

    NONE = "none"
    STRING = "str"
    BINARY = "bin"
    LIST = "list"
    CATEGORICAL = "cat"

    def accepts(self, dtype: DType) -> bool:
        """Say whether the first input fits the namespace.

        Args:
            dtype: The type of the first input.

        Returns:
            Whether it fits: `str` needs `String` itself, and each other
            namespace its own kind.

        Example:
            >>> Namespace.NONE.accepts(DType("Time"))
            True
            >>> Namespace.STRING.accepts(DType.unknown("str"))
            False
        """
        if self is Namespace.NONE:
            return True
        kinds = {
            Namespace.STRING: "String",
            Namespace.BINARY: "Binary",
            Namespace.LIST: "List",
            Namespace.CATEGORICAL: "Categorical",
        }
        return dtype.name == kinds[self]


class FunctionCasts:
    """Give the changes of one visit of the cast pass to a function with a casting rule.

    It applies to a function whose options carry a casting rule
    (`CastingRules`). That rule is the supertype rule with the flags of its
    join, or the rule of the first input. Inputs of one type, or with an input
    of the planner type `Unknown(Any)`, stay. With the supertype rule, the pass
    folds the join of the inputs from the first one. `ModifySupertype` changes
    the join where the function has two inputs. With the rule of the first
    input, every other input must cast to the type of the first with no loss.
    Each input then casts to the result, as `CastExprIr` gives it. A plan that
    the rule refuses raises `PolarsRefusal`.

    Example:
        >>> from astrolabe._rules.options import CastingRules
        >>> casts = FunctionCasts(CastingRules.cast_to_supertypes())
        >>> casts.visit([Operand.column(DType("Int8")), Operand.of_value(300)])
        Visit(steps=((cast(Int16, strict),), (literal(Int16),)), root=None)
    """

    def __init__(self, rules: CastingRules, namespace: Namespace = Namespace.NONE) -> None:
        """Keep the casting rule and the namespace of the function.

        Args:
            rules: The casting rule that the options of the function declare.
            namespace: The namespace of the function.

        Example:
            >>> from astrolabe._rules.options import CastingRules
            >>> lossless = FunctionCasts(CastingRules.first_arg_lossless())
            >>> lossless.visit([Operand.column(DType("Int8")), Operand.of_value(1)]).steps[1]
            (literal(Int8),)
        """
        self._flags = rules.supertype_flags
        self._namespace = namespace

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        It raises `PolarsRefusal` where a cast of an input to the target is
        refused (`CastExprIr`).

        Args:
            operands: The inputs of the function.

        Returns:
            The steps of each input.

        Raises:
            PolarsRefusal: The first input does not fit the namespace, or no type takes the inputs.

        Example:
            >>> from astrolabe._rules.options import CastingRules
            >>> casts = FunctionCasts(CastingRules.cast_to_supertypes(), Namespace.STRING)
            >>> casts.visit([Operand.column(DType("Int8")), Operand.column(DType("String"))])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        first = operands[0].dtype
        if not self._namespace.accepts(first):
            message = f"expected {self._namespace.value} type, got: {first!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if self._unchanged([operand.dtype for operand in operands]):
            return Visit.unchanged(len(operands))
        target = self._target(operands)
        if Is.unknown_any(target):
            message = RaiseSupertype().message([operand.dtype for operand in operands])
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if Is.unknown(target) and target.args[0] in {"int", "float"}:
            target = target.materialized()
        steps = [CastExprIr().steps(operand, target, CastOption.NON_STRICT) for operand in operands]
        return Visit(tuple(steps))

    def _unchanged(self, dtypes: Sequence[DType]) -> bool:
        """Say whether the inputs stay as they are.

        Args:
            dtypes: The types of the inputs.

        Returns:
            Whether an input is `Unknown(Any)`, or all the types are equal.

        Example:
            >>> from astrolabe._rules.options import CastingRules
            >>> lossless = FunctionCasts(CastingRules.first_arg_lossless())
            >>> lossless._unchanged([DType("Int8"), DType.unknown("any")])
            True
        """
        any_input = any(Is.unknown_any(dtype) for dtype in dtypes)
        return any_input or all(dtype.same(dtypes[0]) for dtype in dtypes)

    def _target(self, operands: Sequence[Operand]) -> DType:
        """Give the type that every input casts to.

        It raises `PolarsRefusal` where a step of the join fails (`_joined`),
        or where a cast of an input to the first type can lose values
        (`LosslessCast.check`).

        Args:
            operands: The inputs.

        Returns:
            The type, before the change of a literal type.

        Example:
            >>> from astrolabe._rules.options import CastingRules
            >>> casts = FunctionCasts(CastingRules.first_arg_lossless())
            >>> inputs = [Operand.column(DType("Int8")), Operand.column(DType("Int16"))]
            >>> casts._target(inputs)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        first = operands[0]
        if self._flags is not None:
            return self._joined(operands, self._flags)
        for other in operands[1:]:
            LosslessCast().check(other.dtype, first.dtype)
        return first.dtype

    def _joined(self, operands: Sequence[Operand], flags: SupertypeFlag) -> DType:
        """Fold the join of the inputs from the first one, with the flags of the function.

        Args:
            operands: The inputs.
            flags: The flags of the join.

        Returns:
            The join.

        Raises:
            PolarsRefusal: A step of the fold has no join.

        Example:
            >>> inputs = [Operand.column(DType("Int8")), Operand.of_value("a")]
            >>> from astrolabe._rules.options import CastingRules
            >>> casts = FunctionCasts(CastingRules.cast_to_supertypes())
            >>> casts._joined(inputs, SupertypeFlag.default())
            String
        """
        first = operands[0]
        joined = first.dtype
        for other in operands[1:]:
            step = Supertype(flags).of(joined, other.dtype)
            if step is None:
                message = RaiseSupertype().message([operand.dtype for operand in operands])
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
            joined = ModifySupertype().of(step, first, other) if len(operands) == 2 else step
        return joined


class RaiseSupertype:
    """Give the message of a function whose inputs have no type to cast to.

    Polars then raises an `InvalidOperationError`. The message says whether the
    default join of the inputs exists: then the flags of the function refused
    it.

    Example:
        >>> RaiseSupertype().message([DType("Date"), DType("Boolean")])
        'could not determine supertype of: Date, Boolean'
    """

    def message(self, dtypes: Sequence[DType]) -> str:
        """Give the message.

        Args:
            dtypes: The types of the inputs.

        Returns:
            The message.

        Example:
            >>> RaiseSupertype().message([DType("Int8"), DType("String")])
            'got invalid or ambiguous dtypes: Int8, String'
        """
        joined: DType | None = dtypes[0]
        for dtype in dtypes[1:]:
            joined = None if joined is None else Supertype().of(joined, dtype)
        listed = ", ".join(repr(dtype) for dtype in dtypes)
        if joined is not None:
            return f"got invalid or ambiguous dtypes: {listed}"
        return f"could not determine supertype of: {listed}"


class IsInChange(enum.Enum):
    """Name the change that one round of the cast pass of `is_in` asks for.

    Example:
        >>> IsInChange.IMPLODE.value
        'implode'
    """

    SUPERTYPE = "supertype"
    SELF_CAST = "self_cast"
    OTHER_CAST = "other_cast"
    IMPLODE = "implode"


@dataclass(frozen=True)
class IsInRound:
    """Hold the change of one round of `is_in`: its kind, its target types and its option.

    For `SUPERTYPE`, `dtype` is the target of the values and `nested` the
    target of the collection. For a cast of one side, `dtype` is its target.

    Example:
        >>> IsInRound(IsInChange.SELF_CAST, DType("Enum", (("a",),)), strict=True)
        self_cast(Enum(categories=['a']), strict)
    """

    change: IsInChange
    dtype: DType | None = None
    nested: DType | None = None
    strict: bool = False

    def __repr__(self) -> str:
        """Give the change, its targets and its option.

        Returns:
            The text.

        Example:
            >>> IsInRound(IsInChange.IMPLODE)
            implode()
        """
        parts = [repr(dtype) for dtype in (self.dtype, self.nested) if dtype is not None]
        if self.strict:
            parts.append("strict")
        return f"{self.change.value}({', '.join(parts)})"


class IsInResolution:
    """Give the change that the cast pass of `is_in` asks for, from the types of the two inputs.

    The value side has one level of nesting less than the collection side, a
    level being one `List` or `Array`. Two sides of the same level make
    `is_in`, but not `contains`, implode the collection. Two equal types need
    nothing. Text beside categories casts one side. A `Decimal` value takes a
    collection of numbers. Units of time must not get coarser. Numbers take
    their lossless join, or their supertype. The other pairs refuse the plan
    with `PolarsRefusal`.

    Example:
        >>> resolution = IsInResolution("is_in", contains=False)
        >>> resolution.resolve(DType("Int8"), DType("List", (DType("UInt8"),)))
        supertype(Int16, List(Int16))
    """

    TEXT_CASTS: ClassVar[dict[tuple[str, str], tuple[IsInChange, bool]]] = {
        ("Enum", "String"): (IsInChange.OTHER_CAST, True),
        ("String", "Enum"): (IsInChange.SELF_CAST, True),
        ("String", "Categorical"): (IsInChange.SELF_CAST, False),
        ("Categorical", "String"): (IsInChange.OTHER_CAST, False),
    }

    def __init__(self, op: str, *, contains: bool) -> None:
        """Keep the name of the function and whether it is a `contains`.

        Args:
            op: The name of the function, for the messages.
            contains: Whether the function is `list.contains` or `arr.contains`.

        Example:
            >>> contains = IsInResolution("list.contains", contains=True)
            >>> contains.resolve(DType("Int8"), DType("Int8"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        self._op = op
        self._contains = contains

    def resolve(self, value: DType, collection: DType) -> IsInRound | None:
        """Give the change of one round.

        Args:
            value: The type of the values to look for.
            collection: The type of the collection.

        Returns:
            The change, or `None` where the two sides agree.

        Raises:
            PolarsRefusal: The levels of nesting do not fit.

        Example:
            >>> IsInResolution("is_in", contains=False).resolve(DType("Int8"), DType("Int8"))
            implode()
        """
        value_level, collection_level = self._level(value), self._level(collection)
        if not self._contains and value_level == collection_level:
            return IsInRound(IsInChange.IMPLODE)
        if value_level + 1 != collection_level:
            message = self._refusal(value, collection)
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        inner = collection.inner()
        if value.same(inner):
            return None
        text = self._text(value, collection, inner)
        return text if text is not None else self._rest(value, collection, inner)

    def _level(self, dtype: DType) -> int:
        """Give the number of `List` and `Array` levels of a type.

        Args:
            dtype: The type.

        Returns:
            The level.

        Example:
            >>> nested = DType("List", (DType("Array", (DType("Int8"), 2)),))
            >>> IsInResolution("is_in", contains=False)._level(nested)
            2
        """
        level = 0
        while dtype.name in Is.LISTS:
            level += 1
            dtype = dtype.inner()
        return level

    def _refusal(self, value: DType, collection: DType) -> str:
        """Give the message of the refusal of a pair.

        Args:
            value: The type of the values.
            collection: The type of the collection.

        Returns:
            The message.

        Example:
            >>> IsInResolution("is_in", contains=False)._refusal(DType("Int8"), DType("Date"))
            "'is_in' cannot check for Date values in Int8 data"
        """
        return f"'{self._op}' cannot check for {collection!r} values in {value!r} data"

    def _wrapped(self, collection: DType, inner: DType) -> DType:
        """Give the collection type of a new inner type: a list, or an array of the same width.

        Args:
            collection: The type of the collection.
            inner: The new inner type.

        Returns:
            The type.

        Example:
            >>> array = DType("Array", (DType("Int8"), 3))
            >>> IsInResolution("is_in", contains=False)._wrapped(array, DType("Int16"))
            Array(Int16, shape=(3,))
        """
        if collection.name == "List":
            return DType("List", (inner,))
        return DType("Array", (inner, collection.args[1]))

    def _text(self, value: DType, collection: DType, inner: DType) -> IsInRound | None:
        """Give the change of a null collection or of text beside categories.

        Args:
            value: The type of the values.
            collection: The type of the collection.
            inner: The inner type of the collection.

        Returns:
            The change, or `None` where no arm of this group applies.

        Example:
            >>> enum = DType("Enum", (("a",),))
            >>> resolution = IsInResolution("is_in", contains=False)
            >>> resolution._text(DType("String"), DType("List", (enum,)), enum)
            self_cast(Enum(categories=['a']), strict)
        """
        other = self._wrapped(collection, value)
        if inner.name == "Null":
            return IsInRound(IsInChange.OTHER_CAST, other)
        cast = self.TEXT_CASTS.get((value.name, inner.name))
        if cast is None:
            return None
        change, strict = cast
        target = inner if change is IsInChange.SELF_CAST else other
        return IsInRound(change, target, strict=strict)

    def _rest(self, value: DType, collection: DType, inner: DType) -> IsInRound | None:
        """Give the change of a decimal, a temporal type or a number, and refuse the rest.

        Args:
            value: The type of the values.
            collection: The type of the collection.
            inner: The inner type of the collection.

        Returns:
            The change, or `None` where the units of time agree.

        It raises `PolarsRefusal` where two numbers have no supertype, as
        `Supertype.require` does.

        Raises:
            PolarsRefusal: The pair has no change that makes it agree.

        Example:
            >>> moment = DType("Datetime", ("ns", None))
            >>> coarse = DType("List", (DType("Datetime", ("ms", None)),))
            >>> resolution = IsInResolution("is_in", contains=False)
            >>> resolution._rest(moment, coarse, coarse.inner()) is None
            True
        """
        if value.name == "Decimal" and Is.primitive_numeric(inner):
            return IsInRound(IsInChange.OTHER_CAST, self._wrapped(collection, value))
        names = (value.name, inner.name)
        units = names in {("Datetime", "Datetime"), ("Duration", "Duration")}
        if units and TimeUnit.as_fine(Parameter().unit(value), Parameter().unit(inner)):
            return None
        numbers = Is.primitive_numeric(value) and Is.primitive_numeric(inner)
        refused = "Decimal" in names or inner.name in Is.LISTS or "Struct" in names
        if units or refused or not (numbers or value.name == "Null"):
            message = self._refusal(value, collection)
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        joined = LosslessUpcast().of(value, inner) or Supertype().require(value, inner)
        return IsInRound(IsInChange.SUPERTYPE, joined, self._wrapped(collection, joined))


class IsInCasts:
    """Give the changes of one visit of the cast pass to `is_in`, `list.contains` or `arr.contains`.

    The pass applies the change of `IsInResolution`, then reads the node again,
    until the two sides agree: an `implode` of the collection can need a cast
    in the next round. A cast of a side goes through `CastExprIr`, with the
    option of the change. `is_in` reads the values from its first input and the
    collection from its second. `list.contains` and `arr.contains` read them
    the other way round.

    A round that leaves the types of both sides as they were repeats with
    no end, so the planner never returns. That happens to a null literal
    whose join with the collection is a planner type `Unknown`: the fold
    turns it into a null with no type, of type `Null` again. The probe
    `pl.LazyFrame({"a": [1]}).select(pl.lit(None).is_in(1)).collect_schema()`
    shows it. The rule raises `PolarsRefusal` with the way `HANGS` there.

    Example:
        >>> int8, int16 = Operand.column(DType("Int8")), Operand.column(DType("Int16"))
        >>> IsInCasts.is_in().visit([int8, int16])
        Visit(steps=((cast(Int16, strict),), (implode(List(Int16)),)), root=None)
    """

    def __init__(self, op: str, *, contains: bool) -> None:
        """Keep the name of the function and the order of its inputs.

        Args:
            op: The name of the function.
            contains: Whether the collection is the first input.

        Example:
            >>> array = Operand.column(DType("Array", (DType("Int8"), 2)))
            >>> contains = IsInCasts("arr.contains", contains=True)
            >>> contains.visit([array, Operand.of_value(1)]).steps[1]
            (literal(Int8),)
        """
        self._resolution = IsInResolution(op, contains=contains)
        self._value_index, self._collection_index = (1, 0) if contains else (0, 1)

    @classmethod
    def is_in(cls) -> IsInCasts:
        """Give the rule of `is_in`.

        Returns:
            The rule.

        Example:
            >>> listed = Operand.column(DType("List", (DType("Int8"),)))
            >>> IsInCasts.is_in().visit([Operand.column(DType("Int8")), listed])
            Visit(steps=((), ()), root=None)
        """
        return cls("is_in", contains=False)

    @classmethod
    def list_contains(cls) -> IsInCasts:
        """Give the rule of `list.contains`.

        Returns:
            The rule.

        Example:
            >>> listed = Operand.column(DType("List", (DType("Int8"),)))
            >>> IsInCasts.list_contains().visit([listed, Operand.of_value(300)]).steps[1]
            (literal(Int16),)
        """
        return cls("list.contains", contains=True)

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit, round after round.

        Args:
            operands: The two inputs.

        Returns:
            The steps of the two inputs.

        It raises `PolarsRefusal` where a round refuses the plan.

        Raises:
            PolarsRefusal: A round changes nothing, so the planner hangs.

        Example:
            >>> IsInCasts.is_in().visit([Operand.of_value(None), Operand.of_value(1)])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: hangs: ...
        """
        rounds = _IsInRounds(operands)
        while True:
            value = rounds.current[self._value_index]
            collection = rounds.current[self._collection_index]
            change = self._resolution.resolve(value.dtype, collection.dtype)
            if change is None:
                return rounds.visit()
            if not rounds.apply(self._round(change, rounds.current)):
                message = "the cast pass repeats with no end"
                raise PolarsRefusal(Stop.HANGS, message)

    def _round(self, change: IsInRound, current: Sequence[Operand]) -> dict[int, tuple[Step, ...]]:
        """Give the steps of one round, by input.

        Args:
            change: The change of the round.
            current: The inputs at the start of the round.

        Returns:
            The new steps of each input that changes.

        Example:
            >>> casts = IsInCasts.is_in()
            >>> casts._round(IsInRound(IsInChange.IMPLODE), [Operand.column(DType("Int8"))] * 2)
            {1: (implode(List(Int8)),)}
        """
        value, collection = self._value_index, self._collection_index
        option = CastOption.STRICT if change.strict else CastOption.NON_STRICT
        if change.change is IsInChange.IMPLODE:
            return {1: (Step.implode(current[1].dtype),)}
        if change.change is IsInChange.SELF_CAST:
            return {value: self._cast(current[value], change.dtype, option)}
        if change.change is IsInChange.OTHER_CAST:
            return {collection: self._cast(current[collection], change.dtype, option)}
        value_steps = self._cast(current[value], change.dtype, CastOption.NON_STRICT)
        collection_steps = self._cast(current[collection], change.nested, CastOption.NON_STRICT)
        return {value: value_steps, collection: collection_steps}

    def _cast(self, operand: Operand, target: DType | None, option: CastOption) -> tuple[Step, ...]:
        """Cast one side, as `CastExprIr` does.

        Args:
            operand: The side.
            target: The target of the change.
            option: The option of the change.

        Returns:
            The steps.

        Example:
            >>> IsInCasts.is_in()._cast(Operand.of_value(1), DType("Int8"), CastOption.STRICT)
            (literal(Int8),)
        """
        if target is None:  # pragma: no cover - a change that casts always has a target
            return ()
        return CastExprIr().steps(operand, target, option)


class _IsInRounds:
    """Hold the inputs and the steps of one visit of `is_in` while its rounds run.

    Example:
        >>> rounds = _IsInRounds([Operand.column(DType("Int8"))])
        >>> rounds.apply({0: (Step.implode(DType("Int8")),)}), rounds.visit()
        (True, Visit(steps=((implode(List(Int8)),),), root=None))
    """

    def __init__(self, operands: Sequence[Operand]) -> None:
        """Start with the inputs of the node and no step.

        Args:
            operands: The inputs.

        Example:
            >>> _IsInRounds([Operand.of_value(1)]).current[0].literal
            True
        """
        self.current = list(operands)
        self._steps: list[list[Step]] = [[] for _ in operands]

    def apply(self, changes: dict[int, tuple[Step, ...]]) -> bool:
        """Add the steps of one round to the inputs.

        Args:
            changes: The new steps of each input that changes.

        Returns:
            Whether the type of an input changed.

        Example:
            >>> _IsInRounds([Operand.of_value(None)]).apply({0: (Step.literal(DType("Null")),)})
            False
        """
        before = [operand.dtype for operand in self.current]
        for index, steps in changes.items():
            self._steps[index].extend(steps)
            self.current[index] = StepOperand().after(self.current[index], steps)
        pairs = zip(self.current, before, strict=True)
        return not all(operand.dtype.same(dtype) for operand, dtype in pairs)

    def visit(self) -> Visit:
        """Give the steps so far as a visit.

        Returns:
            The visit.

        Example:
            >>> _IsInRounds([Operand.column(DType("Int8"))]).visit()
            Visit(steps=((),), root=None)
        """
        return Visit(tuple(tuple(steps) for steps in self._steps))


class InlineImplode:
    """Give the type of a literal that the cast pass puts in a list.

    The visit of an `implode` node of a literal folds it into a list literal.
    The literal first takes the type that it takes alone.

    Example:
        >>> InlineImplode().of(Operand.of_value(1))
        List(Int32)
        >>> InlineImplode().of(Operand.column(DType("Int8"))) is None
        True
    """

    def of(self, operand: Operand) -> DType | None:
        """Give the type of the list literal.

        Args:
            operand: The input of the `implode`.

        Returns:
            The list type, or `None` where the input is not a literal.

        Example:
            >>> InlineImplode().of(Operand.of_value(None))
            List(Null)
        """
        if not operand.literal:
            return None
        return DType("List", (operand.dtype.materialized(),))


class SettledCasts:
    """Give the steps of a visit after the pass visits the nodes that the visit put in.

    The pass visits a node before its inputs, so it reaches each cast node that
    a visit inserts right after the visit, from the outermost one in. Each cast
    then takes the result of `CastNodeVisit`, at the type that its input has at
    that moment. An `implode` of a literal becomes a list literal, as
    `InlineImplode` gives it. A cast that sat above that `implode` stays a
    cast, because the pass visited it first.

    Example:
        >>> inputs = [Operand.column(DType("UInt8")), Operand.of_value(1)]
        >>> visit = BinaryCasts().visit(Operator.PLUS, *inputs)
        >>> SettledCasts().of(visit, inputs)
        Visit(steps=((), (literal(UInt8),)), root=None)
    """

    def of(self, visit: Visit, operands: Sequence[Operand]) -> Visit:
        """Give the settled visit.

        Args:
            visit: The visit of the node.
            operands: The inputs before the visit.

        Returns:
            The visit with each inserted node visited.

        Example:
            >>> inputs = [Operand.column(DType("Int8")), Operand.column(DType("Int16"))]
            >>> SettledCasts().of(TernaryCasts().visit(*inputs), inputs)
            Visit(steps=((cast(Int16, overflowing),), ()), root=None)
        """
        pairs = zip(operands, visit.steps, strict=True)
        settled = tuple(itertools.starmap(self._input, pairs))
        return Visit(settled, visit.root)

    def _input(self, operand: Operand, steps: tuple[Step, ...]) -> tuple[Step, ...]:
        """Settle the steps of one input, from the outermost one in.

        Args:
            operand: The input before the visit.
            steps: Its steps, in the order that they apply.

        Returns:
            The settled steps, in the order that they apply.

        Example:
            >>> listed = DType("List", (DType("Int8"),))
            >>> steps = (Step.implode(DType.dynamic_int(1)), Step.cast(listed, CastOption.STRICT))
            >>> SettledCasts()._input(Operand.of_value(1), steps)
            (literal(List(Int32)), cast(List(Int8), non_strict))
        """
        settled: list[Step] = []
        for index in reversed(range(len(steps))):
            below = StepOperand().after(operand, steps[:index])
            result = self._step(steps[index], below)
            if result is not None:
                settled.append(result)
        return tuple(reversed(settled))

    def _step(self, step: Step, below: Operand) -> Step | None:
        """Settle one step, at the input below it.

        Args:
            step: The step.
            below: The input of the step.

        Returns:
            The settled step, or `None` where the cast goes away.

        Example:
            >>> cast = Step.cast(DType("Int8"), CastOption.STRICT)
            >>> SettledCasts()._step(cast, Operand.column(DType("Int8"))) is None
            True
        """
        if step.kind is StepKind.CAST and step.option is not None:
            return CastNodeVisit().visit(below, step.dtype, step.option)
        if step.kind is StepKind.IMPLODE:
            folded = InlineImplode().of(below)
            return step if folded is None else Step.literal(folded)
        return step


class PadLengthCasts:
    """Cast the length of `str.pad_start`, `str.pad_end` and `str.zfill` to `UInt64`.

    The cast pass of polars has a rule of its own for these functions. Where
    the second input, the length, is not a `UInt64`, the pass casts it there
    with a strict cast (`CastExprIr`). A literal folds that cast, so a
    negative length refuses the plan. The rule runs at each visit, and a
    length that is already a `UInt64` stays.

    Example:
        >>> text = Operand.column(DType("String"))
        >>> PadLengthCasts().visit([text, Operand.column(DType("Int8"))])
        Visit(steps=((), (cast(UInt64, strict),)), root=None)
        >>> PadLengthCasts().visit([text, Operand.of_value(3)]).steps[1]
        (literal(UInt64),)
    """

    TARGET: ClassVar[DType] = DType("UInt64")

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        It raises `PolarsRefusal` where the cast of the length is refused
        (`CastExprIr`).

        Args:
            operands: The text and the length.

        Returns:
            The steps of each input.

        Example:
            >>> text, length = Operand.column(DType("String")), Operand.column(DType("UInt64"))
            >>> PadLengthCasts().visit([text, length])
            Visit(steps=((), ()), root=None)
        """
        length = operands[1]
        if length.dtype.name == self.TARGET.name:
            return Visit.unchanged(len(operands))
        steps = CastExprIr().steps(length, self.TARGET, CastOption.STRICT)
        return Visit(((), steps))


class TextInputCasts:
    """Cast each input of `concat_str` that is not text to `String`.

    The cast pass of polars has a rule of its own for `concat_str`. An input
    of `String`, or of the text literal type, stays. Every other input casts
    to `String` (`CastExprIr`): the pass refuses a cast that cannot run,
    folds a literal with a cast that is not strict, and puts a strict cast
    above any other input. The cast of an input then runs before the kernel,
    so a value that does not cast refuses before the kernel checks the
    lengths. The rule runs at each visit, and an input of `String` stays.

    Example:
        >>> text = Operand.column(DType("String"))
        >>> TextInputCasts().visit([text, Operand.column(DType("Int8"))])
        Visit(steps=((), (cast(String, strict),)), root=None)
        >>> TextInputCasts().visit([Operand.of_value(3), text]).steps[0]
        (literal(String),)
    """

    TARGET: ClassVar[DType] = DType("String")

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        It raises `PolarsRefusal` where the cast of an input is refused
        (`CastExprIr`). The first input in the order of the call refuses
        first.

        Args:
            operands: The inputs of the function.

        Returns:
            The steps of each input.

        Example:
            >>> date = Operand.column(DType("Date"))
            >>> TextInputCasts().visit([Operand.of_value("a"), date]).steps
            ((), (cast(String, strict),))
        """
        steps: list[tuple[Step, ...]] = []
        for operand in operands:
            text = Is.string(operand.dtype)
            cast = () if text else CastExprIr().steps(operand, self.TARGET, CastOption.NON_STRICT)
            steps.append(cast)
        return Visit(tuple(steps))


class PatternListCasts:
    """Put the patterns of `str.contains_any` and of its relatives in a list, as the cast pass does.

    `str.contains_any`, `str.find_many`, `str.extract_many` and
    `str.replace_many` read their patterns, and the replacements of
    `str.replace_many`, as a list. The cast pass of polars has a rule of its
    own for them. Where such an input is not a `List`, the pass puts it in a
    list with an `implode`, and visits the node again. Where every such input
    is a `List`, the pass checks that the text is a `String`, and that each
    list holds `String` values. Another type refuses the plan with an
    `InvalidOperationError`.

    Example:
        >>> text = Operand.column(DType("String"))
        >>> PatternListCasts((1,)).visit([text, Operand.of_value("a")])
        Visit(steps=((), (implode(List(String)),)), root=None)
        >>> PatternListCasts((1,)).visit([text, Operand.of_value(1)])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    def __init__(self, places: tuple[int, ...]) -> None:
        """Keep the places of the inputs that the function reads as lists.

        Args:
            places: The places, for example `(1,)` for the patterns.

        Example:
            >>> PatternListCasts((1, 2))._places
            (1, 2)
        """
        self._places = places

    def __repr__(self) -> str:
        """Give the places of the inputs that the rule reads as lists.

        Returns:
            The text.

        Example:
            >>> PatternListCasts((1,))
            PatternListCasts((1,))
        """
        return f"PatternListCasts({self._places!r})"

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit: an `implode` of each input that is not a list.

        Args:
            operands: The inputs of the function.

        Returns:
            The steps of each input.

        Raises:
            PolarsRefusal: The text is not a `String`, or a list does not hold `String` values.

        Example:
            >>> listed = Operand.column(DType("List", (DType("String"),)))
            >>> PatternListCasts((1,)).visit([Operand.column(DType("String")), listed]).steps
            ((), ())
        """
        steps: list[tuple[Step, ...]] = [() for _ in operands]
        dtypes = [operand.dtype for operand in operands]
        for place in self._places:
            if dtypes[place].name != "List":
                steps[place] = (Step.implode(dtypes[place]),)
                dtypes[place] = DType("List", (dtypes[place],))
        inners = [dtypes[place].inner() for place in self._places]
        if not Is.string(dtypes[0]) or not all(Is.string(inner) for inner in inners):
            listed = ", ".join(repr(dtype) for dtype in dtypes)
            message = f"operation not supported for dtypes {listed}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return Visit(tuple(steps))


@dataclass(frozen=True)
class FirstInputCast:
    """Cast the first input of a function to one type, as the cast pass does for a few aggregations.

    The cast pass of polars has a rule of its own for `any` and `all`, which
    cast their input to `Boolean` with a cast that is not strict, and for
    `skew` and `kurtosis`, which cast it to `Float64` with a strict cast. An
    input of that type stays. Otherwise the pass refuses a cast that cannot
    run, and inserts the cast (`CastExprIr`). A literal folds the cast with
    the option of the rule, so a strict fold refuses a value that does not
    fit. The rule runs at each visit, and an input that has the type stays.

    Example:
        >>> any_ = FirstInputCast(DType("Boolean"), CastOption.NON_STRICT)
        >>> any_.visit([Operand.column(DType("Int8"))])
        Visit(steps=((cast(Boolean, strict),),), root=None)
        >>> any_.visit([Operand.column(DType("Boolean"))]).steps
        ((),)
    """

    target: DType
    option: CastOption

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        It raises `PolarsRefusal` where the cast of the input is refused
        (`CastExprIr`).

        Args:
            operands: The inputs of the function.

        Returns:
            The steps of each input.

        Example:
            >>> skew = FirstInputCast(DType("Float64"), CastOption.STRICT)
            >>> skew.visit([Operand.of_value("a")])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        steps = CastExprIr().steps(operands[0], self.target, self.option)
        others: list[tuple[Step, ...]] = [() for _ in operands[1:]]
        return Visit((steps, *others))


@dataclass(frozen=True)
class IntRangeCasts:
    """Cast the bounds of `int_range` to the integer type of the call, as the cast pass does.

    The type of the call must be an integer, and each bound must be a
    number or `Null`. Where both bounds have the type of the call, they
    stay. Otherwise each bound casts to it with a strict cast (`CastExprIr`),
    so a literal bound that does not fit refuses the plan. `int_ranges` has
    the same rule for its three inputs, with `Int64` as the target.

    Example:
        >>> IntRangeCasts(DType("Int16")).visit([Operand.of_value(0), Operand.of_value(3)])
        Visit(steps=((literal(Int16),), (literal(Int16),)), root=None)
        >>> IntRangeCasts(DType("Float64")).visit([Operand.of_value(0), Operand.of_value(3)])
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: non-integer `dtype` ...
    """

    dtype: DType
    names: tuple[str, ...] = ("start", "end")

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        It raises `PolarsRefusal` where the cast of a bound is refused (`CastExprIr`).

        Args:
            operands: The inputs of the function.

        Returns:
            The steps of each input.

        Raises:
            PolarsRefusal: The type of the call is not an integer, or a bound is
                not a number.

        Example:
            >>> IntRangeCasts(DType("Int64")).visit([Operand.of_value("a"), Operand.of_value(3)])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `start` must be ...
        """
        if not Is.integer(self.dtype):
            message = f"non-integer `dtype` passed to `int_range`: {self.dtype!r}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        bounds = operands[: len(self.names)]
        for name, operand in zip(self.names, bounds, strict=True):
            if not Is.numeric(operand.dtype) and operand.dtype.name != "Null":
                message = f"`{name}` must be numeric for `int_range`, got {operand.dtype!r}"
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if all(operand.dtype.same(self.dtype) for operand in bounds):
            return Visit.unchanged(len(operands))
        steps = [CastExprIr().steps(operand, self.dtype, CastOption.STRICT) for operand in bounds]
        return Visit(tuple(steps))


@dataclass(frozen=True)
class TemporalRangeCasts:
    """Cast the bounds of a range of dates or of datetimes to its type, as the cast pass does.

    Each bound must be a `Date` or a `Datetime`. `date_range` casts each
    bound to `Date`. `datetime_range` casts each bound to a `Datetime` that
    the supertype of the bounds and the options of the call give: a `Date`
    takes the unit of the call, or nanoseconds where the interval holds a
    part of a microsecond, or microseconds. A zone of the call replaces the
    zone. A bound of another type gets a strict cast node.

    Example:
        >>> date = Operand.column(DType("Date"))
        >>> datetime = Operand.column(DType("Datetime", ("ms", None)))
        >>> TemporalRangeCasts.dates().visit([date, datetime])
        Visit(steps=((), (cast(Date, strict),)), root=None)
    """

    MOMENTS: ClassVar[frozenset[str]] = frozenset({"Date", "Datetime"})

    target: DType | None
    time_unit: str | None = None
    time_zone: str | None = None
    nanoseconds: int = 0

    @classmethod
    def dates(cls) -> TemporalRangeCasts:
        """Give the rule of `date_range` and `date_ranges`.

        Returns:
            The rule.

        Example:
            >>> TemporalRangeCasts.dates().target
            Date
        """
        return cls(DType("Date"))

    @classmethod
    def datetimes(
        cls,
        time_unit: str | None,
        time_zone: str | None,
        interval: WindowDuration,
    ) -> TemporalRangeCasts:
        """Give the rule of `datetime_range` and `datetime_ranges`, from the options of the call.

        Only the nanoseconds of the interval can hold a part of a microsecond.

        Args:
            time_unit: The unit that the call names, or `None`.
            time_zone: The zone that the call names, or `None`.
            interval: The interval between two values.

        Returns:
            The rule.

        Example:
            >>> from astrolabe._rules.kernels.windows import WindowDuration
            >>> interval = WindowDuration(0, 0, 0, 90, negative=False, parsed_int=False)
            >>> TemporalRangeCasts.datetimes(None, None, interval).nanoseconds
            90
        """
        return cls(None, time_unit, time_zone, interval.nsecs)

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        Args:
            operands: The inputs of the function.

        Returns:
            The steps of each input.

        Raises:
            PolarsRefusal: A bound is not a `Date` or a `Datetime`, or two
                bounds have no supertype, where polars panics.

        Example:
            >>> dates = [Operand.column(DType("Date"))] * 2
            >>> TemporalRangeCasts(None, nanoseconds=1).visit(dates).steps[0]
            (cast(Datetime(time_unit='ns', time_zone=None), strict),)
        """
        bounds = operands[:2]
        for name, operand in zip(("start", "end"), bounds, strict=True):
            if operand.dtype.name not in self.MOMENTS:
                message = f"'{name}' must be Date or Datetime, got {operand.dtype!r}"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)
        target = self.target or self._datetime(bounds[0].dtype, bounds[1].dtype)
        steps = [
            () if bound.dtype.same(target) else (Step.cast(target, CastOption.STRICT),)
            for bound in bounds
        ]
        return Visit(tuple(steps))

    def _datetime(self, start: DType, end: DType) -> DType:
        """Give the `Datetime` of a range of datetimes.

        Args:
            start: The type of the start.
            end: The type of the end.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The bounds have no supertype, where polars panics.

        Example:
            >>> datetime = DType("Datetime", ("ms", "UTC"))
            >>> TemporalRangeCasts(None, time_unit="ns")._datetime(datetime, datetime)
            Datetime(time_unit='ns', time_zone='UTC')
            >>> paris = DType("Datetime", ("ms", "Europe/Paris"))
            >>> TemporalRangeCasts(None)._datetime(datetime, paris)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: no supertype ...
        """
        joined = Supertype().of(start, end)
        if joined is None:  # pragma: no cover -- the type function of the range refuses bounds
            message = f"no supertype of {start!r} and {end!r}"  # of no supertype first.
            raise PolarsRefusal(Stop.PANICS, message)
        if joined.name == "Date":
            fine = "ns" if self.nanoseconds % 1_000 != 0 else "us"
            unit, zone = self.time_unit or fine, None
        else:
            unit, zone = self.time_unit or Parameter().unit(joined), Parameter().text(joined, 1)
        return DType("Datetime", (unit, self.time_zone or zone))


@dataclass(frozen=True)
class IsBetweenCasts:
    """Give the changes of the cast pass to `is_between`, which has a rule of its own.

    Each bound is compared with the input: the lower one with `>=`, or `>`
    where `closed` leaves it out, and the upper one with `<=` or `<`. A
    literal bound of another type meets the rewrite of a comparison
    (`ComparisonLiteral`). A value that fits takes the type of the input. A
    bound that holds at every row that is not null drops out, and the node
    compares the input with the other bound only (`BinaryCasts`). Where both
    hold, the node compares the input with the lower bound. A bound that
    fails at every row puts its rewrite in place of the node. Otherwise
    each bound meets the input as a comparison does (`CoercedBinopDtype`),
    and the three inputs cast to the supertype of the two types that this
    gives. Where a comparison casts nothing, or the two types have no
    supertype, the node stays as it is, and its run compares the types.

    Example:
        >>> uint8 = Operand.column(DType("UInt8"))
        >>> IsBetweenCasts("Both").visit([uint8, Operand.of_value(1), Operand.of_value(2)])
        Visit(steps=((), (literal(UInt8),), (literal(UInt8),)), root=None)
    """

    OPERATORS: ClassVar[dict[str, tuple[Operator, Operator]]] = {
        "Both": (Operator.GT_EQ, Operator.LT_EQ),
        "Left": (Operator.GT_EQ, Operator.LT),
        "Right": (Operator.GT, Operator.LT_EQ),
        "None": (Operator.GT, Operator.LT),
    }

    closed: str

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        Args:
            operands: The input, the lower bound and the upper bound.

        Returns:
            The steps of each input, or the new node.

        Example:
            >>> text = Operand.column(DType("String"))
            >>> IsBetweenCasts("Both").visit([text, Operand.of_value(1), Operand.of_value(2)])
            Visit(steps=((), (), ()), root=None)
        """
        needle = operands[0]
        pairs = zip(self.OPERATORS[self.closed], operands[1:3], strict=True)
        rewrites = [self._rewrite(needle, op, bound) for op, bound in pairs]
        for rewrite in rewrites:
            if isinstance(rewrite, NewRoot) and rewrite.value is not True:
                return Visit(((), (), ()), root=rewrite)
        return self._settled(needle, operands[1:3], rewrites)

    def _settled(
        self,
        needle: Operand,
        bounds: Sequence[Operand],
        rewrites: Sequence[DType | NewRoot | None],
    ) -> Visit:
        """Give the visit where no bound puts a new node in place of `is_between`.

        A bound that holds at every row drops out, and the node compares the
        input with the other bound only. Otherwise both bounds stay.

        Args:
            needle: The input.
            bounds: The two bounds.
            rewrites: The rewrite of each bound.

        Returns:
            The steps of each input.

        Example:
            >>> int8 = Operand.column(DType("Int8"))
            >>> bounds = [Operand.of_value(1), Operand.of_value(300)]
            >>> rewrites = [DType("Int8"), NewRoot(RootKind.WHEN_NOT_NULL, value=True)]
            >>> IsBetweenCasts("Both")._settled(int8, bounds, rewrites).steps
            ((), (literal(Int8),), ())
        """
        retyped = list(starmap(self._retyped, zip(bounds, rewrites, strict=True)))
        literals = ((Step.literal(new),) if isinstance(new, DType) else () for new in rewrites)
        steps: list[tuple[Step, ...]] = [(), *literals]
        kept = [place for place, rewrite in enumerate(rewrites) if not isinstance(rewrite, NewRoot)]
        if len(kept) < 2:
            place = kept[0] if kept else 0
            return self._one_side(needle, retyped[place], place, steps)
        return self._joined(needle, retyped, steps)

    def _rewrite(self, needle: Operand, op: Operator, bound: Operand) -> DType | NewRoot | None:
        """Give the rewrite of one bound, where it is a literal of another type.

        Args:
            needle: The input.
            op: The comparison of the input with the bound.
            bound: The bound.

        Returns:
            The new type of the literal, a new node, or `None` where the bound stays.

        Example:
            >>> uint8 = Operand.column(DType("UInt8"))
            >>> IsBetweenCasts("Both")._rewrite(uint8, Operator.LT_EQ, Operand.of_value(300))
            when_not_null(True)
        """
        if not bound.literal or bound.dtype.same(needle.dtype):
            return None
        return ComparisonLiteral().rewrite(needle.dtype, op, bound)

    def _retyped(self, bound: Operand, rewrite: DType | NewRoot | None) -> Operand:
        """Give a bound after its rewrite: a literal of the new type, or the bound as it is.

        Args:
            bound: The bound.
            rewrite: Its rewrite.

        Returns:
            The bound.

        Example:
            >>> IsBetweenCasts("Both")._retyped(Operand.of_value(1), DType("UInt8")).dtype
            UInt8
        """
        if isinstance(rewrite, DType):
            return Operand.typed_literal(bound.value, rewrite)
        return bound

    def _one_side(
        self,
        needle: Operand,
        bound: Operand,
        place: int,
        steps: list[tuple[Step, ...]],
    ) -> Visit:
        """Give the visit where one bound holds at every row: a comparison with the other bound.

        Args:
            needle: The input.
            bound: The bound that stays, after its rewrite.
            place: The place of that bound: 0 for the lower one, 1 for the upper one.
            steps: The steps of the rewrites of the literals.

        Returns:
            The steps of each input.

        Example:
            >>> int8 = Operand.column(DType("Int8"))
            >>> lower = Operand.column(DType("Int16"))
            >>> IsBetweenCasts("Both")._one_side(int8, lower, 0, [(), (), ()]).steps
            ((cast(Int16, non_strict),), (), ())
        """
        op = self.OPERATORS[self.closed][place]
        compared = BinaryCasts().visit(op, needle, bound)
        steps[0] = compared.steps[0]
        steps[place + 1] += compared.steps[1]
        return Visit(tuple(steps))

    def _joined(
        self,
        needle: Operand,
        bounds: Sequence[Operand],
        steps: list[tuple[Step, ...]],
    ) -> Visit:
        """Give the visit where both bounds stay: the casts to the supertype of the two comparisons.

        Args:
            needle: The input.
            bounds: The two bounds, after the rewrites of the literals.
            steps: The steps of the rewrites of the literals.

        Returns:
            The steps of each input, or no change where a comparison or the
            supertype gives nothing.

        Example:
            >>> int8, uint8 = Operand.column(DType("Int8")), Operand.column(DType("UInt8"))
            >>> IsBetweenCasts("Both")._joined(int8, [uint8, int8], [(), (), ()]).steps
            ((cast(Int16, strict),), (cast(Int16, strict),), (cast(Int16, strict),))
        """
        lower_op, upper_op = self.OPERATORS[self.closed]
        lower = CoercedBinopDtype().of(lower_op, needle, bounds[0])
        if lower is None:
            return Visit.unchanged(3)
        upper = CoercedBinopDtype().of(upper_op, needle, bounds[1])
        if upper is None:
            return Visit.unchanged(3)
        flags = SupertypeFlag.default() & ~SupertypeFlag.ALLOW_PRIMITIVE_TO_STRING
        joined = Supertype(flags).of(lower, upper)
        if joined is None:
            return Visit.unchanged(3)
        inputs = [needle, *bounds]
        for place, operand in enumerate(inputs):
            cast = CastExprIr().steps(operand, joined, CastOption.NON_STRICT)
            steps[place] += cast
        return Visit(tuple(steps))


class ShiftFillCasts:
    """Cast the values and the fill of `shift` with a fill to their supertype, as polars does.

    The cast pass of polars has a rule of its own for `shift` with a fill:
    it casts the values and the fill to their supertype, and it leaves the
    count alone. Unlike the general rule of a function (`FunctionCasts`), it
    keeps a dynamic supertype as it is. So a `Null` column beside the fill
    `300` casts to the dynamic type of `300`, and the node above reads the
    shift as a value of that dynamic type. Where the two have no supertype,
    the rule leaves them as they are, and refuses nothing. So does a supertype
    of any type, as of a `List` and a number.

    Example:
        >>> operands = [Operand.column(DType("Int8")), Operand.of_value(1), Operand.of_value(300)]
        >>> ShiftFillCasts().visit(operands).steps
        ((cast(Int16, strict),), (), (literal(Int16),))
        >>> nulls = Operand.column(DType("Null"))
        >>> ShiftFillCasts().visit([nulls, Operand.of_value(1), Operand.of_value(300)]).steps
        ((cast(Unknown(int, 300), strict),), (), ())
    """

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        Args:
            operands: The values, the count and the fill.

        Returns:
            The steps of each input.

        Example:
            >>> int8 = Operand.column(DType("Int8"))
            >>> ShiftFillCasts().visit([int8, Operand.of_value(1), int8]).steps
            ((), (), ())
            >>> listed = Operand.column(DType("List", (DType("Int8"),)))
            >>> ShiftFillCasts().visit([listed, Operand.of_value(1), Operand.of_value(1)]).steps
            ((), (), ())

            A `Boolean` does not cast to the type of a float fill, so the rule
            leaves both as they are:

            >>> flags = Operand.column(DType("Boolean"))
            >>> ShiftFillCasts().visit([flags, Operand.of_value(1), Operand.of_value(1.5)]).steps
            ((), (), ())
        """
        values, fill = operands[0], operands[2]
        dtypes = [values.dtype, fill.dtype]
        if values.dtype == fill.dtype or any(Is.unknown_any(dtype) for dtype in dtypes):
            return Visit.unchanged(len(operands))
        joined = Supertype().of(values.dtype, fill.dtype)
        if joined is None or Is.unknown_any(joined):
            return Visit.unchanged(len(operands))
        target = ModifySupertype().of(joined, values, fill)
        try:
            left = self._steps(values, target)
            right = self._steps(fill, target)
        except PolarsRefusal:  # pragma: no cover
            # The cast check accepts a cast of each input to their supertype.
            return Visit.unchanged(len(operands))
        return Visit((left, (), right))

    def _steps(self, operand: Operand, target: DType) -> tuple[Step, ...]:
        """Give the steps of one input cast to the supertype.

        A null literal does not fold into a dynamic type: it stays below the
        cast, and the run gives it the fixed type of the dynamic type, as it
        gives a `Null` column.

        Args:
            operand: The values or the fill.
            target: The supertype.

        Returns:
            The steps.

        Example:
            >>> ShiftFillCasts()._steps(Operand.of_value(None), DType.dynamic_int(300))
            (cast(Unknown(int, 300), non_strict),)
            >>> ShiftFillCasts()._steps(Operand.of_value(300), DType("Int16"))
            (literal(Int16),)
        """
        if operand.literal and operand.value is None and Is.unknown(target):
            return (Step.cast(target, CastOption.NON_STRICT),)
        return CastExprIr().steps(operand, target, CastOption.NON_STRICT)


class ReplaceCasts:
    """Put the old and the new values of `replace` in lists, as the cast pass does.

    The kernel reads one list of old values and one list of new values. So
    the cast pass gives each of the two inputs that is not a `List` an
    `implode`, and an input that is a `List` stays.

    Example:
        >>> int8 = Operand.column(DType("Int8"))
        >>> ReplaceCasts().visit([int8, Operand.of_value(1), int8]).steps
        ((), (implode(List(Unknown(int, 1))),), (implode(List(Int8)),))
    """

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        Args:
            operands: The input, the old values, the new values, and the default where it has one.

        Returns:
            The steps of each input.

        Example:
            >>> listed = Operand.column(DType("List", (DType("Int8"),)))
            >>> ReplaceCasts().visit([listed, listed, listed]).steps
            ((), (), ())
        """
        steps: list[tuple[Step, ...]] = [() for _ in operands]
        for place in (1, 2):
            dtype = operands[place].dtype
            if dtype.name != "List":
                steps[place] = (Step.implode(dtype),)
        return Visit(tuple(steps))


class SliceCasts:
    """Check the offset and the length of a slice, as the cast pass of polars does.

    The pass casts nothing for a slice. It refuses an offset that is not an
    integer, and a length that is neither an integer nor `Null`.

    Example:
        >>> operands = [Operand.column(DType("Int8")), Operand.of_value(1.5), Operand.of_value(1)]
        >>> SliceCasts().visit(operands)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: offset must be ...
    """

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit: none, after the checks.

        Args:
            operands: The input, the offset and the length.

        Returns:
            No change.

        Raises:
            PolarsRefusal: The offset or the length has another type.

        Example:
            >>> int8 = Operand.column(DType("Int8"))
            >>> SliceCasts().visit([int8, Operand.of_value(0), Operand.of_value(None)]).steps
            ((), (), ())
        """
        offset, length = operands[1].dtype, operands[2].dtype
        if not Is.integer(offset):
            message = f"offset must be integral for slice expression, not {offset!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if not Is.integer(length) and length.name != "Null":
            message = f"length must be integral for slice expression, not {length!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return Visit.unchanged(len(operands))


class GatherIndexCasts:
    """Put the indices of `list.gather` in a list, as the cast pass of polars does.

    `list.gather` reads its second input as a list of indices for each row.
    The cast pass of polars has a rule of its own for it. Where the indices are
    not a `List`, the pass puts them in a list with an `implode`, and visits
    the node again. Where they are a `List`, its values must be integers, or
    the pass refuses the plan with an `InvalidOperationError`. So indices that
    are not integers refuse the plan, in a list or not. The rule runs at each
    visit, and indices that are a list of integers stay.

    Example:
        >>> lists = Operand.column(DType("List", (DType("Int8"),)))
        >>> GatherIndexCasts().visit([lists, Operand.column(DType("Int8"))])
        Visit(steps=((), (implode(List(Int8)),)), root=None)
        >>> texts = Operand.column(DType("List", (DType("String"),)))
        >>> GatherIndexCasts().visit([lists, texts])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        Args:
            operands: The lists and the indices.

        Returns:
            The steps of each input.

        Raises:
            PolarsRefusal: The indices are not integers.

        Example:
            >>> lists = Operand.column(DType("List", (DType("Int8"),)))
            >>> indices = Operand.column(DType("List", (DType("UInt32"),)))
            >>> GatherIndexCasts().visit([lists, indices]).steps
            ((), ())
            >>> GatherIndexCasts().visit([lists, Operand.of_value(1)]).steps
            ((), (implode(List(Unknown(int, 1))),))
        """
        indices = operands[1].dtype
        steps: tuple[Step, ...] = ()
        if indices.name != "List":
            steps = (Step.implode(indices),)
            indices = DType("List", (indices.materialized(),))
        if not Is.integer(indices.inner()):
            listed = f"{operands[0].dtype!r}, {indices!r}"
            message = f"`list.gather` operation not supported for dtypes {listed}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return Visit(((), steps))


@dataclass(frozen=True)
class HolidaysCasts:
    """Check the holidays of a function of business days, as the cast pass of polars does.

    `dt.add_business_days`, `dt.is_business_day` and `pl.business_day_count`
    have a rule of their own in the cast pass. A literal series of dates
    becomes a list of those dates. Any other holidays must already be a
    `List(Date)`, or the pass refuses the plan with a `ComputeError`. The rule
    runs at each visit, and changes nothing where the holidays are a list of
    dates. `index` is the place of the holidays among the inputs. The inputs of
    this module do not hold a literal series, so the rule always checks the
    type.

    Example:
        >>> holidays = Operand.column(DType("List", (DType("Date"),)))
        >>> HolidaysCasts(1).visit([Operand.column(DType("Date")), holidays])
        Visit(steps=((), ()), root=None)
    """

    index: int

    LIST: ClassVar[DType] = DType("List", (DType("Date"),))

    def visit(self, operands: Sequence[Operand]) -> Visit:
        """Give the changes of the visit.

        Args:
            operands: The inputs of the function.

        Returns:
            The steps of each input: none.

        Raises:
            PolarsRefusal: The holidays are not a `List(Date)`.

        Example:
            >>> HolidaysCasts(1).visit([Operand.column(DType("Date")), Operand.of_value(None)])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: dtype of holidays ...
        """
        dtype = operands[self.index].dtype
        if dtype != self.LIST:
            message = f"dtype of holidays list must be List(Date), got {dtype!r} instead"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return Visit.unchanged(len(operands))
