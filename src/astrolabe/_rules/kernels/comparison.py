"""The comparison kernels of polars: `==`, `!=`, `<`, `<=`, `>`, `>=` and the variants with nulls.

A comparison of two series checks a few pairs of kinds first: text beside a
number, a category beside a type that is not text or a category, and a date
beside a duration raise. A category beside text or the same category compares
at once. Every other pair is coerced to one type, as the arithmetic of two
series does (`Coercion`), and the comparison then runs on the physical type. So
two nested values compare after their casts, and only their outer types matter:
`==` gives `Boolean`, and an ordering raises on a list, an array or a struct. A
coercion that fails raises a `SchemaError`, whatever the cause.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.arithmetic import Coercion, Operands
from astrolabe._rules.kernels.dispatch import Data  # ruff: ignore[typing-only-first-party-import]
from astrolabe._rules.releases import polars_releases

if TYPE_CHECKING:
    from collections.abc import Sequence


class CompareOp(enum.Enum):
    """Name the comparisons of two series.

    The value is the name of the method of polars.

    Example:
        >>> CompareOp.LT_EQ.orders(), CompareOp.EQUAL_MISSING.orders()
        (True, False)
    """

    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    EQUAL_MISSING = "equal_missing"
    NOT_EQUAL_MISSING = "not_equal_missing"
    GT = "gt"
    GT_EQ = "gt_eq"
    LT = "lt"
    LT_EQ = "lt_eq"

    def orders(self) -> bool:
        """Say whether the comparison orders its inputs, as `<`, `<=`, `>` and `>=` do.

        An ordering refuses a list, an array and a struct, where an equality
        accepts them.

        Returns:
            Whether it is `<`, `<=`, `>` or `>=`.

        Example:
            >>> CompareOp.GT.orders()
            True
        """
        return self in {CompareOp.GT, CompareOp.GT_EQ, CompareOp.LT, CompareOp.LT_EQ}


class Comparison:
    """Give the type of a comparison of two series, as polars runs it.

    It is the kernel that an eager `Series` comparison reaches, with no cast
    pass in front of it. A comparison that runs gives `Boolean`.

    Example:
        >>> lists = Data(DType("List", (DType("Int8"),)))
        >>> Comparison(CompareOp.EQUAL).output(lists, Data(DType("List", (DType("String"),))))
        Boolean
        >>> Comparison(CompareOp.LT).output(lists, lists)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    PHYSICAL: ClassVar[frozenset[str]] = frozenset({
        "Null",
        "Boolean",
        "String",
        "Binary",
        "BinaryOffset",
        *Is.SIGNED,
        *Is.UNSIGNED,
        *Is.FLOATS,
        "List",
        "Array",
        "Struct",
    })

    def __init__(self, op: CompareOp) -> None:
        """Keep the comparison.

        Args:
            op: The comparison.

        Example:
            >>> Comparison(CompareOp.GT)._op
            <CompareOp.GT: 'gt'>
        """
        self._op = op

    def output(self, left: Data, right: Data, lengths: Sequence[int] = ()) -> DType:
        """Give the type of the output.

        The lengths of the two series, where the caller knows them, must agree
        or hold one row. Polars checks them after the pairs of kinds that no
        comparison accepts, and before the coercion.

        Args:
            left: The left input.
            right: The right input.
            lengths: The rows of the left and the right input, or none.

        Returns:
            `Boolean`.

        It raises `PolarsRefusal` where the pair of kinds is refused
        (`_validate`), or where two structs do not compare (`_fields`).

        Raises:
            PolarsRefusal: The coercion fails, or the physical type has no comparison.

        Example:
            >>> text, small = Data(DType("String")), Data(DType("Int8"))
            >>> Comparison(CompareOp.EQUAL).output(text, small)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
            >>> objects = Data(DType("Object"))
            >>> Comparison(CompareOp.EQUAL).output(objects, objects)
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot compare Object
        """
        self._validate(left.dtype, right.dtype)
        longer = {length for length in lengths if length != 1}
        if len(longer) > 1:
            message = "could not compare between two series of different length"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        if self._categories(left.dtype, right.dtype):
            return DType("Boolean")
        try:
            coerced = Coercion().of(left, right)
        except PolarsRefusal as refusal:
            message = f"could not evaluate comparison of {left.dtype!r} and {right.dtype!r}"
            raise PolarsRefusal(ErrorClass.SCHEMA, message) from refusal
        physical = coerced.left.dtype.physical()
        if physical.name not in self.PHYSICAL:
            message = f"cannot compare {physical!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if self._op.orders() and Is.nested(physical):
            message = f"cannot order {physical!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if physical.name == "Struct":
            self._fields(coerced)
            self._validity(coerced)
        return DType("Boolean")

    def _validate(self, left: DType, right: DType) -> None:
        """Raise on the pairs of kinds that no comparison accepts.

        Args:
            left: The left type.
            right: The right type.

        Raises:
            PolarsRefusal: The pair is text beside a number, a category beside
                a type that is not text or a category, or a date beside a duration.

        Example:
            >>> span = DType("Duration", ("us",))
            >>> Comparison(CompareOp.EQUAL)._validate(span, DType("Date"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        for one, other in ((left, right), (right, left)):
            if one.name == "String" and Is.primitive_numeric(other):
                message = f"cannot compare String with {other!r}"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)
            if one.name in Is.CATEGORIES and other.name not in Is.CATEGORIES | {"String"}:
                message = f"cannot compare a category with {other!r}"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)
            if one.name == "Date" and other.name == "Duration":
                message = "cannot compare Date with a duration"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def _categories(self, left: DType, right: DType) -> bool:
        """Say whether the pair compares as categories at once.

        Two `Categorical` series share the categories of the process. Two
        `Enum` series need the same categories. A category beside text
        compares each text with the categories.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            Whether the pair compares at once.

        Raises:
            PolarsRefusal: Two `Enum` types have different categories.

        Example:
            >>> Comparison(CompareOp.EQUAL)._categories(DType("Enum", (("a",),)), DType("String"))
            True
        """
        names = (left.name, right.name)
        if names == ("Enum", "Enum"):
            if left.args != right.args:
                message = "the categories of the enums differ"
                raise PolarsRefusal(ErrorClass.SCHEMA, message)
            return True
        if names == ("Categorical", "Categorical"):
            return True
        return "String" in names and bool(set(names) & Is.CATEGORIES)

    @polars_releases
    def _validity(self, coerced: Operands) -> None:
        """Panic where a struct of one null row meets a struct of more rows.

        A comparison that does not count nulls gives the output the joined
        validity of both structs. The validity of a struct of one row with a
        null has one bit, so the join or the output of more rows refuses it. A
        comparison that counts nulls reads the validities in another way, and
        runs. Before polars 1.41.0, the comparison first repeats the struct of
        one row to the length of the other (`_spread_validity`).

        Args:
            coerced: The two structs, of one type after the coercion.

        Raises:
            PolarsRefusal: A struct of one null row meets a struct of more rows.

        Example:
            >>> pair = DType("Struct", ((("a", DType("Int8")),),))
            >>> null = Data(pair, nulls=True, one_row=True)
            >>> operands = Operands(Data(pair), null)
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.41.0"):
            ...     Comparison(CompareOp.EQUAL)._validity(operands)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        if self._op in {CompareOp.EQUAL_MISSING, CompareOp.NOT_EQUAL_MISSING}:
            return
        left, right = coerced.left, coerced.right
        left_short = left.one_row and left.nulls and not right.one_row
        right_short = right.one_row and right.nulls and not left.one_row
        if left_short or right_short:
            message = "the validity of one null row meets more rows"
            raise PolarsRefusal(Stop.PANICS, message)

    @_validity.before("1.41.0")
    def _spread_validity(self, coerced: Operands) -> None:
        """Refuse no pair of structs, as the older releases do.

        The older releases repeat a struct of one row to the length of the
        other before they join the validities, so the lengths always fit.

        Args:
            coerced: The two structs, of one type after the coercion.

        Example:
            >>> pair = DType("Struct", ((("a", DType("Int8")),),))
            >>> null = Data(pair, nulls=True, one_row=True)
            >>> Comparison(CompareOp.EQUAL)._spread_validity(Operands(Data(pair), null)) is None
            True
        """
        del coerced

    def _fields(self, coerced: Operands) -> None:
        """Compare each pair of fields of two structs, as the comparison of two structs does.

        The kernel compares each pair as `CompareOp.EQUAL_MISSING` does, and it
        panics where a pair cannot compare.

        Args:
            coerced: The two structs, of one type after the coercion.

        Raises:
            PolarsRefusal: A pair of fields cannot compare.

        Example:
            >>> pair = DType("Struct", ((("a", DType("Int8")),),))
            >>> Comparison(CompareOp.EQUAL)._fields(Operands(Data(pair), Data(pair))) is None
            True
            >>> objects = Data(DType("Struct", ((("a", DType("Object")),),)))
            >>> Comparison(CompareOp.EQUAL)._fields(Operands(objects, objects))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: a pair of fields fails: cannot ...
        """
        fields = Comparison(CompareOp.EQUAL_MISSING)
        mine, theirs = coerced.left.dtype.fields(), coerced.right.dtype.fields()
        for (_, one), (_, other) in zip(mine, theirs, strict=False):
            try:
                fields.output(coerced.left.field(one), coerced.right.field(other))
            except PolarsRefusal as refusal:
                message = f"a pair of fields fails: {refusal.message}"
                raise PolarsRefusal(Stop.PANICS, message) from refusal
