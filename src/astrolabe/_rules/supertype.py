"""The supertype of polars: the one data type that two data types both cast to.

Polars joins two data types in many places. The planner joins the inputs of
`fill_null`, of `when` and of a binary operator, and the cast pass casts each
input to the join. This module translates that join, with its flags, the
version that raises, the fold over many types, and the lossless join of two
numbers. A rule that needs a join reads it here and does not compute one of its
own. Polars can find no join of two types. `Supertype.of` then gives `None`,
and `Supertype.require` raises `PolarsRefusal`.

The join of polars is one long list of arms over the two types, and the first
arm that applies decides. Each group of its arms is one method or one small
class here, in the order in which polars tries them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import (
    Decimals,
    DType,
    DynamicInt,
    IntegerRange,
    Is,
    Parameter,
    TimeUnit,
)
from astrolabe._rules.failures import ErrorClass, PolarsRefusal
from astrolabe._rules.releases import polars_releases

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class SupertypeFlag(enum.Flag):
    """Name the two flags that change the join of polars.

    Polars keeps the flags in one bit set, and the values here are its bits.
    `ALLOW_PRIMITIVE_TO_STRING` lets a number or a `Boolean` join a `String`.
    `ALLOW_IMPLODE_LIST` lets a type join a `List` of a type that it joins. The
    default has only the first flag. Polars takes the first flag away for
    `is_between`, `is_close`, `search_sorted`, `max_horizontal` and
    `min_horizontal`, and it adds the second for the set operations of `list`.
    The flags apply only at the top level: a join inside a `List` or a `Struct`
    always uses the default.

    Example:
        >>> SupertypeFlag.default() & ~SupertypeFlag.ALLOW_PRIMITIVE_TO_STRING
        <SupertypeFlag: 0>
    """

    ALLOW_IMPLODE_LIST = 1 << 0
    ALLOW_PRIMITIVE_TO_STRING = 1 << 1

    @classmethod
    def default(cls) -> SupertypeFlag:
        """Give the default flags of polars.

        Returns:
            The flags: a number joins a `String`, and a value does not join a list.

        Example:
            >>> SupertypeFlag.default()
            <SupertypeFlag.ALLOW_PRIMITIVE_TO_STRING: 2>
        """
        return cls.ALLOW_PRIMITIVE_TO_STRING


@dataclass(frozen=True)
class _Matched:
    """Hold the answer of an arm of the join that applies to two types.

    An arm that applies can give no supertype. Its `None` then ends the join of
    the two types in that order, and no later arm runs.

    Example:
        >>> _Matched(None).dtype is None
        True
    """

    dtype: DType | None


class Supertype:
    """Join two data types as polars does, with the options of one call site.

    The join is `inner(left, right)`, and where that gives nothing,
    `inner(right, left)`. So the join is nearly symmetric. It keeps the left
    type where two types are equal for polars, for example two integer literals
    of different values, and the order of the left fields where two structs
    have the same number of fields.

    Example:
        >>> Supertype().of(DType("Int8"), DType("UInt8"))
        Int16
        >>> Supertype().of(DType("Date"), DType("Boolean")) is None
        True
        >>> no_text = SupertypeFlag(0)
        >>> Supertype(no_text).of(DType("Int8"), DType("String")) is None
        True
    """

    def __init__(self, flags: SupertypeFlag | None = None) -> None:
        """Keep the flags.

        Args:
            flags: The flags of the call site, or `None` for the default flags.

        Example:
            >>> implode = Supertype(SupertypeFlag.default() | SupertypeFlag.ALLOW_IMPLODE_LIST)
            >>> implode.of(DType("Int8"), DType("List", (DType("Int16"),)))
            List(Int16)
        """
        self._arms = _Arms(SupertypeFlag.default() if flags is None else flags)

    def of(self, left: DType, right: DType) -> DType | None:
        """Give the supertype with the flags of the call site.

        Where polars finds no supertype of the two types, this method gives
        `None`.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The supertype, or `None` where the two types have none.

        Example:
            >>> Supertype().of(DType("Int8"), DType.dynamic_int(300))
            Int16
        """
        first = self._arms.inner(left, right)
        return first if first is not None else self._arms.inner(right, left)

    def require(self, left: DType, right: DType) -> DType:
        """Give the supertype, or refuse as polars does where it needs one.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The supertype.

        Raises:
            PolarsRefusal: The two types have no supertype. Polars raises a
                `SchemaError`.

        Example:
            >>> Supertype().require(DType("Float32"), DType("Int64"))
            Float64
            >>> Supertype().require(DType("Date"), DType("Boolean"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        joined = self.of(left, right)
        if joined is None:
            message = f"failed to determine supertype of {left!r} and {right!r}"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        return joined

    def fold(self, dtypes: Iterable[DType]) -> DType:
        """Join many types from the left, as polars joins a list of types.

        The fold starts at `Null`, so it gives `Null` for no types. The join so
        far is the left operand of each step. The cast pass of a function with
        supertyping folds the same way with its own options.

        A step with no supertype raises `PolarsRefusal`, as `require` does.

        Args:
            dtypes: The types, in order.

        Returns:
            The supertype of all of them.

        Example:
            >>> Supertype().fold([DType("Int8"), DType("UInt8"), DType("Float32")])
            Float32
        """
        joined = DType("Null")
        for dtype in dtypes:
            joined = self.require(joined, dtype)
        return joined


class _Arms:
    """Hold the arms of the join of two types, in the order in which polars tries them.

    Each group gives `None` where none of its arms applies, and the answer of
    the arm that applies otherwise. The first group that answers decides. A
    join inside a nested type uses the default flags, as in polars.

    Example:
        >>> _Arms(SupertypeFlag.default()).inner(DType("Int64"), DType("UInt32"))
        Int64
    """

    def __init__(self, flags: SupertypeFlag) -> None:
        """Keep the flags and list the groups of arms in the order of polars.

        Args:
            flags: The flags of the call site.

        Example:
            >>> len(_Arms(SupertypeFlag.default())._groups)
            11
        """
        self._flags = flags
        self._groups: tuple[Callable[[DType, DType], _Matched | None], ...] = (
            _FixedPairs().match,
            self._wide_integer,
            self._to_string,
            self._binary_or_null,
            self._duration,
            self._datetime,
            self._list,
            self._array,
            _StructArms().match,
            _DecimalArms().match,
            self._dynamic,
        )

    def inner(self, left: DType, right: DType) -> DType | None:
        """Give the supertype of two types in one order.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The supertype, or `None` where no arm gives one in this order.

        Example:
            >>> _Arms(SupertypeFlag.default()).inner(DType("UInt8"), DType("UInt16")) is None
            True
        """
        if left.same(right):
            return left
        for group in self._groups:
            matched = group(left, right)
            if matched is not None:
                return matched.dtype
        return None

    def _wide_integer(self, left: DType, right: DType) -> _Matched | None:
        """Join a 128-bit integer with a number or a `Boolean`.

        An integer literal is an integer here, so `Int128` beside it stays
        `Int128`.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> _Arms(SupertypeFlag.default())._wide_integer(DType("UInt128"), DType("Int8"))
            _Matched(dtype=Int128)
        """
        if left.name == "Int128" and (Is.integer(right) or right.name == "Boolean"):
            return _Matched(DType("Int128"))
        if left.name == "Int128" and Is.float(right):
            return _Matched(DType("Float64"))
        if left.name != "UInt128":
            return None
        if Is.unsigned(right) or right.name == "Boolean":
            return _Matched(DType("UInt128"))
        if Is.signed(right):
            return _Matched(DType("Int128"))
        return _Matched(DType("Float64")) if Is.float(right) else None

    def _to_string(self, left: DType, right: DType) -> _Matched | None:
        """Join a type with `String` into `String`.

        A type that is not `Unknown(Any)` and not `Binary` joins `String` where
        the flags allow it. A type whose physical type is not a number,
        `Boolean`, `String` or `Binary` joins `String` always: a `List`, an
        `Array`, a `Struct` or `Null`. A `Date` holds an `Int32`, so it needs
        the options.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> no_text = _Arms(SupertypeFlag(0))
            >>> no_text._to_string(DType("List", (DType("Int8"),)), DType("String"))
            _Matched(dtype=String)
            >>> no_text._to_string(DType("Date"), DType("String")) is None
            True
        """
        if right.name != "String":
            return None
        text_flag = SupertypeFlag.ALLOW_PRIMITIVE_TO_STRING in self._flags
        allowed = not Is.unknown_any(left) and left.name != "Binary" and text_flag
        if allowed or not Is.primitive(left.physical()):
            return _Matched(DType("String"))
        return None

    def _binary_or_null(self, left: DType, right: DType) -> _Matched | None:
        """Join `String` with `Binary`, and any type with `Null`.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> _Arms(SupertypeFlag.default())._binary_or_null(DType("String"), DType("Binary"))
            _Matched(dtype=Binary)
        """
        if left.name == "String" and right.name == "Binary":
            return _Matched(DType("Binary"))
        if right.name == "Null":
            return _Matched(left)
        return None

    def _duration(self, left: DType, right: DType) -> _Matched | None:
        """Join a `Duration` with a `Datetime`, a `Date` or a `Duration`.

        A `Duration` beside a `Datetime` gives the `Datetime` with the joined
        unit. An empty time zone becomes no time zone.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> moment = DType("Datetime", ("us", "UTC"))
            >>> _Arms(SupertypeFlag.default())._duration(DType("Duration", ("ns",)), moment)
            _Matched(dtype=Datetime(time_unit='us', time_zone='UTC'))
        """
        names = {left.name, right.name}
        if "Duration" not in names or not names <= {"Duration", "Datetime", "Date"}:
            return None
        if "Date" in names:
            return _Matched(DType("Date"))
        unit = TimeUnit.joined(Parameter().unit(left), Parameter().unit(right))
        if names == {"Duration"}:
            return _Matched(DType("Duration", (unit,)))
        moment = left if left.name == "Datetime" else right
        zone = Parameter().text(moment, 1)
        return _Matched(DType("Datetime", (unit, zone or None)))

    def _datetime(self, left: DType, right: DType) -> _Matched | None:
        """Join two `Datetime` types of the same time zone.

        Two zones that differ, or a zone beside no zone, have no supertype.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> nano, milli = DType("Datetime", ("ns", None)), DType("Datetime", ("ms", None))
            >>> _Arms(SupertypeFlag.default())._datetime(nano, milli)
            _Matched(dtype=Datetime(time_unit='ms', time_zone=None))
        """
        if (left.name, right.name) != ("Datetime", "Datetime"):
            return None
        left_zone = Parameter().text(left, 1)
        right_zone = Parameter().text(right, 1)
        if left_zone != right_zone:
            return None
        unit = TimeUnit.joined(Parameter().unit(left), Parameter().unit(right))
        return _Matched(DType("Datetime", (unit, right_zone)))

    def _list(self, left: DType, right: DType) -> _Matched | None:
        """Join a `List` with a `List`, an `Array`, or a type that it wraps.

        The inner types join with the default flags. With the flag
        `ALLOW_IMPLODE_LIST`, a type beside a `List` joins its inner type.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> listed = DType("List", (DType("Int8"),))
            >>> _Arms(SupertypeFlag.default())._list(listed, DType("Array", (DType("UInt8"), 2)))
            _Matched(dtype=List(Int16))
        """
        names = {left.name, right.name}
        if names in ({"List"}, {"List", "Array"}):
            return self._wrapped(Supertype().of(left.inner(), right.inner()))
        implode = SupertypeFlag.ALLOW_IMPLODE_LIST in self._flags
        if not (implode and "List" in names):
            return None
        if left.name == "List":
            return self._wrapped(Supertype().of(left.inner(), right))
        return self._wrapped(Supertype().of(right.inner(), left))

    def _array(self, left: DType, right: DType) -> _Matched | None:
        """Join two `Array` types.

        Two arrays of the same width give an `Array` of that width. Two
        widths that differ give a `List`.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> pair = DType("Array", (DType("Int8"), 2))
            >>> triple = DType("Array", (DType("Int8"), 3))
            >>> _Arms(SupertypeFlag.default())._array(pair, triple)
            _Matched(dtype=List(Int8))
        """
        if (left.name, right.name) != ("Array", "Array"):
            return None
        joined = Supertype().of(left.inner(), right.inner())
        width = Parameter().integer(left, 1)
        if width != Parameter().integer(right, 1):
            return self._wrapped(joined)
        return _Matched(None if joined is None else DType("Array", (joined, width)))

    def _wrapped(self, inner: DType | None) -> _Matched:
        """Give a `List` of a joined inner type, or no supertype where the inner join failed.

        Args:
            inner: The joined inner type, or `None`.

        Returns:
            The answer of the arm.

        Example:
            >>> _Arms(SupertypeFlag.default())._wrapped(DType("Int8"))
            _Matched(dtype=List(Int8))
        """
        return _Matched(None if inner is None else DType("List", (inner,)))

    def _dynamic(self, left: DType, right: DType) -> _Matched | None:
        """Join a type with the planner type of a literal that has no data type.

        `_DynamicJoin` holds the arms of this join.

        Args:
            left: The left type.
            right: The right type, which must be `Unknown` for the arm to apply.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> _Arms(SupertypeFlag.default())._dynamic(DType("UInt8"), DType.dynamic_int(-1))
            _Matched(dtype=Int16)
        """
        if not Is.unknown(right):
            return None
        return _Matched(_DynamicJoin(self._flags).join(left, right))


class _FixedPairs:
    """Hold the arms that map two named kinds to a fixed supertype.

    A kind is a data type with its parameters taken away. The arms join the
    integers, the floats and `Boolean`, and a temporal kind with a number. A
    row does not read the parameters of its types, so `Datetime` stands for
    every unit and time zone. The standard build of polars has 32-bit row
    indices, and in that build `Int32` and `Int64` beside `UInt64` give
    `Float64`.

    Example:
        >>> _FixedPairs().match(DType("Int16"), DType("Float16"))
        _Matched(dtype=Float32)
    """

    ROWS: ClassVar[dict[str, dict[str, str]]] = {
        "Int8": {
            "Boolean": "Int8",
            "Int16": "Int16",
            "Int32": "Int32",
            "Int64": "Int64",
            "UInt8": "Int16",
            "UInt16": "Int32",
            "UInt32": "Int64",
            "UInt64": "Float64",
            "Float16": "Float16",
            "Float32": "Float32",
            "Float64": "Float64",
        },
        "Int16": {
            "Boolean": "Int16",
            "Int8": "Int16",
            "Int32": "Int32",
            "Int64": "Int64",
            "UInt8": "Int16",
            "UInt16": "Int32",
            "UInt32": "Int64",
            "UInt64": "Float64",
            "Float16": "Float32",
            "Float32": "Float32",
            "Float64": "Float64",
        },
        "Int32": {
            "Boolean": "Int32",
            "Int8": "Int32",
            "Int16": "Int32",
            "Int64": "Int64",
            "UInt8": "Int32",
            "UInt16": "Int32",
            "UInt32": "Int64",
            "UInt64": "Float64",
            "Float16": "Float64",
            "Float32": "Float64",
            "Float64": "Float64",
        },
        "Int64": {
            "Boolean": "Int64",
            "Int8": "Int64",
            "Int16": "Int64",
            "Int32": "Int64",
            "UInt8": "Int64",
            "UInt16": "Int64",
            "UInt32": "Int64",
            "UInt64": "Float64",
            "Float16": "Float64",
            "Float32": "Float64",
            "Float64": "Float64",
        },
        "UInt8": {"UInt32": "UInt32", "UInt64": "UInt64"},
        "UInt16": {"UInt8": "UInt16", "UInt32": "UInt32", "UInt64": "UInt64"},
        "UInt32": {"UInt64": "UInt64"},
        "Boolean": {
            "UInt8": "UInt8",
            "UInt16": "UInt16",
            "UInt32": "UInt32",
            "UInt64": "UInt64",
            "Float16": "Float16",
            "Float32": "Float32",
            "Float64": "Float64",
        },
        "Float16": {
            "UInt8": "Float16",
            "UInt16": "Float32",
            "UInt32": "Float64",
            "UInt64": "Float64",
            "Float32": "Float32",
            "Float64": "Float64",
        },
        "Float32": {
            "UInt8": "Float32",
            "UInt16": "Float32",
            "UInt32": "Float64",
            "UInt64": "Float64",
            "Float16": "Float32",
            "Float64": "Float64",
        },
        "Float64": {
            "UInt8": "Float64",
            "UInt16": "Float64",
            "UInt32": "Float64",
            "UInt64": "Float64",
            "Float16": "Float64",
            "Float32": "Float64",
        },
        "Date": {
            "UInt32": "Int64",
            "UInt64": "Int64",
            "Int32": "Int32",
            "Int64": "Int64",
            "Float16": "Float32",
            "Float32": "Float32",
            "Float64": "Float64",
        },
        "Datetime": {
            "UInt32": "Int64",
            "UInt64": "Int64",
            "Int32": "Int64",
            "Int64": "Int64",
            "Float16": "Float64",
            "Float32": "Float64",
            "Float64": "Float64",
        },
        "Duration": {
            "UInt32": "Int64",
            "UInt64": "Int64",
            "Int32": "Int64",
            "Int64": "Int64",
            "Float16": "Float64",
            "Float32": "Float64",
            "Float64": "Float64",
        },
        "Time": {
            "Int32": "Int64",
            "Int64": "Int64",
            "Float16": "Float64",
            "Float32": "Float64",
            "Float64": "Float64",
        },
    }

    def match(self, left: DType, right: DType) -> _Matched | None:
        """Give the fixed supertype of two kinds, or the `Datetime` beside a `Date`.

        The arms of a `Date` beside a `Datetime` keep the `Datetime` with its
        parameters.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> _FixedPairs().match(DType("Date"), DType("Datetime", ("ns", "UTC")))
            _Matched(dtype=Datetime(time_unit='ns', time_zone='UTC'))
        """
        if (left.name, right.name) == ("Date", "Datetime"):
            return _Matched(right)
        if (left.name, right.name) == ("Datetime", "Date"):
            return _Matched(left)
        joined = self.ROWS.get(left.name, {}).get(right.name)
        return None if joined is None else _Matched(DType(joined))


class _StructArms:
    """Join a `Struct` with a literal, a `Struct` or a number.

    A struct beside a literal with no data type joins its first field with
    the literal, so the join is not a struct. A struct beside a number joins
    each field with the number. Two structs with the same names in the same
    order join field by field. Other pairs of structs join into the union of
    their fields.

    Example:
        >>> _StructArms().match(DType("Struct", ((("a", DType("Int8")),),)), DType.dynamic_int(300))
        _Matched(dtype=Int16)
    """

    def match(self, left: DType, right: DType) -> _Matched | None:
        """Give the answer of the struct arm that applies.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> _StructArms().match(DType("Struct", ((("a", DType("Int8")),),)), DType("Float32"))
            _Matched(dtype=Struct({'a': Float32}))
        """
        if left.name != "Struct":
            return None
        if Is.dynamic_int(right) or (Is.unknown(right) and right.args[0] == "float"):
            fields = left.fields()
            return _Matched(Supertype().of(fields[0][1], right) if fields else None)
        if right.name == "Struct":
            return _Matched(self._structs(left.fields(), right.fields()))
        if Is.primitive_numeric(right):
            return _Matched(self._each_field(left.fields(), right))
        return None

    def _each_field(self, fields: tuple[tuple[str, DType], ...], number: DType) -> DType | None:
        """Join each field of a struct with a number.

        Args:
            fields: The fields of the struct.
            number: The number type.

        Returns:
            The struct of the joined fields, or `None` where a field has no join.

        Example:
            >>> _StructArms()._each_field((("a", DType("Date")),), DType("Int8")) is None
            True
        """
        joined = []
        for name, dtype in fields:
            field = Supertype().of(dtype, number)
            if field is None:
                return None
            joined.append((name, field))
        return DType("Struct", (tuple(joined),))

    def _structs(
        self,
        left: tuple[tuple[str, DType], ...],
        right: tuple[tuple[str, DType], ...],
    ) -> DType | None:
        """Join two structs field by field, or by the union of their fields.

        The loop stops at the first pair of fields with no join, before it
        reads the names of the next pair.

        Args:
            left: The fields of the left struct.
            right: The fields of the right struct.

        Returns:
            The joined struct, or `None`.

        Example:
            >>> _StructArms()._structs((("a", DType("Int8")),), (("a", DType("UInt8")),))
            Struct({'a': Int16})
        """
        if len(left) != len(right):
            return self._union(left, right)
        joined = []
        for (name, dtype), (other_name, other) in zip(left, right, strict=True):
            if name != other_name:
                return self._union(left, right)
            field = Supertype().of(dtype, other)
            if field is None:
                return None
            joined.append((name, field))
        return DType("Struct", (tuple(joined),))

    def _union(
        self,
        left: tuple[tuple[str, DType], ...],
        right: tuple[tuple[str, DType], ...],
    ) -> DType | None:
        """Join two structs into the union of their fields by name.

        The fields of the longer struct come first, in its order, and the left
        struct counts as the longer one where both have the same length. A
        field that only the shorter struct has comes after them. A field in
        both joins its two types, with the type of the shorter struct on the
        left.

        Args:
            left: The fields of the left struct.
            right: The fields of the right struct.

        Returns:
            The joined struct, or `None` where a shared field has no join.

        Example:
            >>> short = (("a", DType("Int8")),)
            >>> long = (("b", DType("String")), ("a", DType("Int16")))
            >>> _StructArms()._union(short, long)
            Struct({'b': String, 'a': Int16})
        """
        longest, shortest = (left, right) if len(left) >= len(right) else (right, left)
        merged = dict(longest)
        for name, dtype in shortest:
            current = merged.setdefault(name, dtype)
            if dtype.same(current):
                continue
            field = Supertype().of(dtype, current)
            if field is None:
                return None
            merged[name] = field
        return DType("Struct", (tuple(merged.items()),))


class _DecimalArms:
    """Join a `Decimal` with a `Decimal`, a float or an integer.

    Two decimals take the larger precision and the larger scale. A float
    gives `Float64`. An integer keeps the decimal where every value of the
    integer type fits its precision and scale (`Decimals.holds`), and gives
    the largest precision with the same scale otherwise.

    Example:
        >>> _DecimalArms().match(DType("Decimal", (10, 2)), DType("Int16"))
        _Matched(dtype=Decimal(precision=10, scale=2))
        >>> _DecimalArms().match(DType("Decimal", (10, 2)), DType("Int64"))
        _Matched(dtype=Decimal(precision=38, scale=2))
    """

    WIDE: ClassVar[frozenset[str]] = frozenset({"Int128", "UInt128"})

    def match(self, left: DType, right: DType) -> _Matched | None:
        """Give the answer of the decimal arm that applies.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> _DecimalArms().match(DType("Decimal", (10, 2)), DType.unknown("float"))
            _Matched(dtype=Float64)
        """
        if left.name != "Decimal":
            return None
        precision = Parameter().integer(left, 0)
        scale = Parameter().integer(left, 1)
        if right.name == "Decimal":
            joined_precision = max(precision, Parameter().integer(right, 0))
            joined_scale = max(scale, Parameter().integer(right, 1))
            return _Matched(DType("Decimal", (joined_precision, joined_scale)))
        if Is.float(right):
            return _Matched(DType("Float64"))
        if not (Is.signed(right) or Is.unsigned(right)):
            return None
        if self._fits(right.name, precision, scale):
            return _Matched(left)
        return _Matched(Decimals().widest(scale))

    def _fits(self, integer: str, precision: int, scale: int) -> bool:
        """Say whether the largest value of an integer type fits a decimal.

        No 128-bit integer type fits.

        Args:
            integer: The name of the integer type.
            precision: The precision of the decimal.
            scale: The scale of the decimal.

        Returns:
            Whether every value of the integer type fits.

        Example:
            >>> _DecimalArms()._fits("Int8", 5, 2), _DecimalArms()._fits("Int8", 4, 2)
            (True, False)
        """
        if integer in self.WIDE:
            return False
        largest = IntegerRange().BOUNDS[integer][1]
        return Decimals().holds(largest, precision, scale)


class _DynamicJoin:
    """Join a type with the planner type `Unknown` of a literal.

    A literal with no data type, such as `pl.lit(300)`, has the type
    `Unknown(Int(300))` in the planner. Polars joins it with a type by its
    value: beside an `Int8`, the literal 300 needs the smallest integer type
    that holds it, `Int16`, and the join gives `Int16`. A join that no arm
    covers gives `Unknown(Any)`, a type that the cast pass refuses.

    Example:
        >>> _DynamicJoin(SupertypeFlag.default()).join(DType("Int8"), DType.dynamic_int(300))
        Int16
        >>> _DynamicJoin(SupertypeFlag.default()).join(DType("Date"), DType.dynamic_int(1))
        Unknown
    """

    def __init__(self, flags: SupertypeFlag) -> None:
        """Keep the flags.

        Args:
            flags: The flags of the call site.

        Example:
            >>> no_text = _DynamicJoin(SupertypeFlag(0))
            >>> no_text.join(DType("String"), DType.dynamic_int(1)) is None
            True
        """
        self._flags = flags

    def join(self, dtype: DType, literal: DType) -> DType | None:
        """Give the join of a type with the planner type of a literal, in the order of polars.

        Args:
            dtype: The type on the left.
            literal: The planner type `Unknown` on the right.

        Returns:
            The join, `Unknown(Any)` where no arm applies, or `None`.

        Example:
            >>> join = _DynamicJoin(SupertypeFlag.default())
            >>> join.join(DType("Null"), DType.unknown("float")).args
            ('float', None)
        """
        steps = (self._number, self._null, self._integer)
        for step in steps:
            matched = step(dtype, literal)
            if matched is not None:
                return matched.dtype
        return DType.unknown("any")

    def _number(self, dtype: DType, literal: DType) -> _Matched | None:
        """Join a number literal with text or a float, and a float literal with an integer.

        A number literal beside text keeps the text type where the flags allow
        it, and has no join otherwise.

        Args:
            dtype: The type on the left.
            literal: The planner type of the literal.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> join = _DynamicJoin(SupertypeFlag.default())
            >>> with PolarsRelease.pretend("1.43.0"):
            ...     join._number(DType("Int8"), DType.unknown("float"))
            _Matched(dtype=Float64)
        """
        what = literal.args[0]
        if what not in {"int", "float"}:
            return None
        if Is.string(dtype):
            text_flag = SupertypeFlag.ALLOW_PRIMITIVE_TO_STRING in self._flags
            return _Matched(dtype if text_flag else None)
        if Is.float(dtype):
            return _Matched(dtype)
        if what == "float" and Is.integer(dtype):
            return _Matched(self._float_beside_integer(dtype))
        return None

    @polars_releases
    def _float_beside_integer(self, dtype: DType) -> DType:
        """Join a float literal with an integer: `Float64` beside a known integer.

        Args:
            dtype: The integer type.

        Returns:
            `Float64` beside a known integer, and the planner type of a float
            literal beside an integer literal.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> join = _DynamicJoin(SupertypeFlag.default())
            >>> with PolarsRelease.pretend("1.43.0"):
            ...     known = join._float_beside_integer(DType("Int8"))
            ...     literal = join._float_beside_integer(DType.dynamic_int(1))
            >>> known, literal.args
            (Float64, ('float', None))
        """
        return DType("Float64") if Is.known(dtype) else DType.unknown("float")

    @_float_beside_integer.before("1.43.0")
    def _float_literal_beside_integer(self, dtype: DType) -> DType:
        """Join a float literal with any integer into the planner type of a float literal.

        Args:
            dtype: The integer type.

        Returns:
            The planner type of a float literal.

        Example:
            >>> join = _DynamicJoin(SupertypeFlag.default())
            >>> join._float_literal_beside_integer(DType("Int8")).args
            ('float', None)
        """
        del dtype
        return DType.unknown("float")

    def _null(self, dtype: DType, literal: DType) -> _Matched | None:
        """Keep a literal beside `Null`.

        Args:
            dtype: The type on the left.
            literal: The planner type of the literal.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> join = _DynamicJoin(SupertypeFlag.default())
            >>> join._null(DType("Null"), DType.dynamic_int(5))
            _Matched(dtype=Unknown(int, 5))
        """
        if dtype.name == "Null":
            return _Matched(literal)
        return None

    def _integer(self, dtype: DType, literal: DType) -> _Matched | None:
        """Join an integer literal with a number, a `Boolean` or a `Decimal`.

        Beside a `Boolean`, the literal takes the type that it takes alone.
        Beside a `Decimal`, the join has the largest precision and the scale
        of the decimal. Two integer literals never reach this arm, because the
        join keeps the left one of two types that are the same for polars.

        Args:
            dtype: The type on the left.
            literal: The planner type of the literal.

        Returns:
            The answer of the arm that applies, or `None`.

        Example:
            >>> join = _DynamicJoin(SupertypeFlag.default())
            >>> join._integer(DType("UInt64"), DType.dynamic_int(-1))
            _Matched(dtype=Int64)
            >>> join._integer(DType("Boolean"), DType.dynamic_int(-1))
            _Matched(dtype=Int32)
        """
        if not Is.dynamic_int(literal):
            return None
        value = Parameter().integer(literal, 1)
        if Is.primitive_numeric(dtype):
            return _Matched(self._smallest(dtype, value))
        if dtype.name == "Boolean":
            return _Matched(Supertype().of(dtype, DynamicInt.standalone(value)))
        if dtype.name == "Decimal":
            return _Matched(Decimals().widest(Parameter().integer(dtype, 1)))
        return None

    def _smallest(self, dtype: DType, value: int) -> DType | None:
        """Join an integer type with the smallest type that holds a literal.

        The literal takes an unsigned type beside an unsigned type where its
        value is not negative. A signed type beside `UInt64` gives `Int64`,
        and not the `Float64` of the two types alone.

        Args:
            dtype: The integer type.
            value: The value of the literal.

        Returns:
            The join.

        Example:
            >>> _DynamicJoin(SupertypeFlag.default())._smallest(DType("UInt8"), 300)
            UInt16
        """
        if Is.unsigned(dtype) and value >= 0:
            fitting = DynamicInt.positive(value)
        else:
            fitting = DynamicInt.smallest(value)
        if dtype.name == "UInt64" and Is.signed(fitting):
            return DType("Int64")
        return Supertype().of(dtype, fitting)


class LosslessUpcast:
    """Give the smallest number type that two number types both cast to with no loss.

    Polars uses this join where a cast must keep every value, for example to
    cast the keys of a join. A float joins only a float, and an integer joins
    only an integer. A signed type beside an unsigned type gives the wider of
    the signed type and the signed type of twice the unsigned width, and no
    signed type holds a `UInt128`. Two equal types, a literal type and a nested
    type give nothing.

    Example:
        >>> LosslessUpcast().of(DType("Int8"), DType("UInt32"))
        Int64
        >>> LosslessUpcast().of(DType("Int8"), DType("Float32")) is None
        True
    """

    FLOATS: ClassVar[tuple[str, ...]] = ("Float64", "Float32")
    SIGNED: ClassVar[tuple[str, ...]] = ("Int128", "Int64", "Int32", "Int16", "Int8")
    UNSIGNED: ClassVar[tuple[str, ...]] = ("UInt128", "UInt64", "UInt32", "UInt16", "UInt8")
    MIXED: ClassVar[tuple[tuple[frozenset[str], str | None], ...]] = (
        (frozenset({"UInt128"}), None),
        (frozenset({"UInt64", "Int128"}), "Int128"),
        (frozenset({"UInt32", "Int64"}), "Int64"),
        (frozenset({"UInt16", "Int32"}), "Int32"),
        (frozenset({"UInt8", "Int16"}), "Int16"),
    )

    def of(self, left: DType, right: DType) -> DType | None:
        """Give the lossless join of two number types.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The join, or `None`.

        Example:
            >>> LosslessUpcast().of(DType("UInt64"), DType("Int8"))
            Int128
        """
        either = (left, right)
        if any(Is.unknown(dtype) or Is.nested(dtype) for dtype in either) or left.same(right):
            return None
        names = {left.name, right.name}
        if Is.float(left) and Is.float(right):
            return self._first(self.FLOATS, names)
        if Is.signed(left) and Is.signed(right):
            return self._first(self.SIGNED, names)
        if Is.unsigned(left) and Is.unsigned(right):
            return self._first(self.UNSIGNED, names)
        if Is.integer(left) and Is.integer(right):
            return self._mixed(names)
        return None

    def _first(self, order: tuple[str, ...], names: set[str]) -> DType | None:
        """Give the first type of an order that is one of the two names.

        Two `Float16` types are equal, so no pair of floats lacks an arm.

        Args:
            order: The names, widest first.
            names: The names of the two types.

        Returns:
            The type, or `None`.

        Example:
            >>> LosslessUpcast()._first(LosslessUpcast.SIGNED, {"Int8", "Int32"})
            Int32
            >>> LosslessUpcast()._first(LosslessUpcast.SIGNED, {"Float32"}) is None
            True
        """
        for name in order:
            if name in names:
                return DType(name)
        return None  # pragma: no cover -- the order holds both names

    def _mixed(self, names: set[str]) -> DType | None:
        """Join a signed and an unsigned integer type.

        Args:
            names: The names of the two types.

        Returns:
            The signed type that holds both, or `None` beside `UInt128`.

        Example:
            >>> LosslessUpcast()._mixed({"Int8", "UInt8"})
            Int16
            >>> LosslessUpcast()._mixed({"Int8", "UInt128"}) is None
            True
            >>> LosslessUpcast()._mixed({"Int8"}) is None
            True
        """
        for keys, joined in self.MIXED:
            if keys & names:
                return None if joined is None else DType(joined)
        return None  # pragma: no cover -- the table holds each unsigned type
