"""The casts of polars: which casts the planner allows, what they give, and which values they keep.

Polars casts in two places. The planner checks a cast that its cast pass
inserts, and refuses it where the two types cannot meet. The run then casts a
series with a kernel for each source kind, and a strict cast raises where a
value does not fit the target. This module translates both, in the split of
polars. It holds the plan checks of the data types and of the cast pass. It
also holds the kernels of the series and of the arrow arrays, and the fate of
each value. A rule that needs a cast reads it here.

A cast of a series runs on three levels. The cast of the series takes the
shortcuts that do not read the data. The kernel of the source kind then casts
the logical type. Most kernels end in the arrow kernel, which casts the
physical arrays, that is, the integers that hold a date, a decimal or a
category.

A kernel that has no cast for a pair raises `PolarsRefusal`, and so does a run
that a value makes fail. The cast check of polars can give no answer, so
`CastCheck` gives `None` where polars is not sure. A strict cast whose answer
the categories of the process decide raises `UndecidedError`.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Protocol

from astrolabe._hooks.checked import checked_cast
from astrolabe._rules.dtypes import Decimals, DType, IntegerRange, Is, Parameter, TimeUnit
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop, Undecided, UndecidedError
from astrolabe._rules.supertype import LosslessUpcast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


class CastCheck:
    """Say whether the planner lets one type cast to another.

    The cast pass of polars inserts casts, for example to bring the inputs of
    `fill_null` to their supertype. Before it inserts one, it checks the cast,
    and it refuses the query where the answer is `False`. The check of polars
    can also give no answer. This check then gives `None`, and the cast pass
    lets the cast through.

    Example:
        >>> CastCheck().of(DType("Boolean"), DType("Date"))
        False
        >>> CastCheck().of(DType("String"), DType("Date")) is None
        True
    """

    def of(self, source: DType, target: DType) -> bool | None:
        """Give the answer of the cast check of polars.

        Args:
            source: The type that the cast reads.
            target: The type that the cast gives.

        Returns:
            `True`, `False`, or `None` where polars is not sure.

        Example:
            >>> CastCheck().of(DType("Categorical"), DType("Binary"))
            False
        """
        numbers = Is.primitive_numeric(source) and Is.primitive_numeric(target)
        if source.same(target) or numbers or source.name == "Null":
            return True
        names = (source.name, target.name)
        categorical = set(names) & Is.CATEGORIES
        if categorical and {"Binary", "String"} & set(names):
            return "String" in names
        if "Boolean" in names:
            return self._beside_boolean(target if source.name == "Boolean" else source)
        return self._nested(source, target)

    def _beside_boolean(self, other: DType) -> bool:
        """Say whether a `Boolean` casts to or from a type.

        Args:
            other: The type on the other side of the `Boolean`.

        Returns:
            Whether the cast makes sense: only a number, a `Decimal`, `String` or `Binary`.

        Example:
            >>> CastCheck()._beside_boolean(DType("Decimal", (10, 2)))
            True
        """
        return Is.primitive_numeric(other) or other.name in {"Decimal", "String", "Binary"}

    def _nested(self, source: DType, target: DType) -> bool | None:
        """Check a cast between two nested types of the same kind.

        A struct with no fields casts to any struct. Two structs need the same
        number of fields, and each field casts by position. An inner answer of
        `None` makes the whole answer `None`, as in polars.

        Args:
            source: The type that the cast reads.
            target: The type that the cast gives.

        Returns:
            The answer, or `None` for every other pair.

        Example:
            >>> one = DType("Struct", ((("a", DType("Int8")),),))
            >>> two = DType("Struct", ((("a", DType("Int8")), ("b", DType("Int8"))),))
            >>> CastCheck()._nested(one, two)
            False
        """
        names = (source.name, target.name)
        if names == ("Struct", "Struct"):
            return self._structs(source.fields(), target.fields())
        if names == ("List", "List"):
            return self.of(source.inner(), target.inner())
        if names != ("Array", "Array"):
            return None
        same_width = Parameter().integer(source, 1) == Parameter().integer(target, 1)
        return same_width and self.of(source.inner(), target.inner())

    def _structs(
        self,
        source: tuple[tuple[str, DType], ...],
        target: tuple[tuple[str, DType], ...],
    ) -> bool | None:
        """Check a cast between two structs by the position of their fields.

        Args:
            source: The fields that the cast reads.
            target: The fields that the cast gives.

        Returns:
            `True` for a struct with no fields, `False` where the counts differ,
            and otherwise the first answer of a field that is not `True`.

        Example:
            >>> CastCheck()._structs((), (("a", DType("Int8")),))
            True
        """
        if not source:
            return True
        if len(source) != len(target):
            return False
        for (_, field), (_, other) in zip(source, target, strict=True):
            answer = self.of(field, other)
            if answer is not True:
                return answer
        return True


class LosslessCast:
    """Say whether a cast keeps every value, as the planner checks it for `index_of`.

    A function with the casting rule of the first input
    (`CastingRules.first_arg_lossless`) casts its other inputs to the type of
    its first input. The cast pass refuses the query where such a cast can lose
    a value. An integer literal is checked by its value, so `pl.lit(300)`
    cannot meet an `Int8`. `of` gives the test, and `check` gives the refusal
    of the cast pass.

    Example:
        >>> LosslessCast().of(DType("Int8"), DType("Float32"))
        True
        >>> LosslessCast().of(DType.dynamic_int(300), DType("Int8"))
        False
    """

    FLOAT_HOLDS: ClassVar[dict[str, frozenset[str]]] = {
        "Float16": frozenset({"UInt8", "Int8"}),
        "Float32": frozenset({"UInt8", "UInt16", "Int8", "Int16"}),
        "Float64": frozenset({"UInt8", "UInt16", "UInt32", "Int8", "Int16", "Int32"}),
    }
    FLOAT_DIGITS: ClassVar[dict[str, int]] = {"Float16": 11, "Float32": 24, "Float64": 53}

    def check(self, source: DType, target: DType) -> None:
        """Refuse a cast that can lose a value, as the cast pass of polars does.

        It reads the largest value of an integer type as a signed 128-bit
        integer, so it panics for `UInt128` beside a `Decimal`.

        Args:
            source: The type that the cast reads.
            target: The type that the cast gives.

        Raises:
            PolarsRefusal: The cast can lose a value.

        Example:
            >>> LosslessCast().check(DType("Int8"), DType("Int16")) is None
            True
            >>> LosslessCast().check(DType("UInt128"), DType("Decimal", (10, 2)))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        if (source.name, target.name) == ("UInt128", "Decimal"):
            message = "the largest value of UInt128 does not fit an i128"
            raise PolarsRefusal(Stop.PANICS, message)
        if not self.of(source, target):
            message = f"cannot cast losslessly from {source!r} to {target!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def of(self, source: DType, target: DType) -> bool:
        """Say whether the lossless check of polars lets the cast through.

        Where two arms apply to one pair, the checks here keep the order of
        polars.

        Args:
            source: The type that the cast reads.
            target: The type that the cast gives.

        Returns:
            Whether the cast keeps every value. Polars raises where it does not.

        Example:
            >>> LosslessCast().of(DType("Datetime", ("ms", None)), DType("Datetime", ("ns", None)))
            True
        """
        if source.same(target) or source.name == "Null":
            return True
        if Is.dynamic_int(source):
            return self._literal(Parameter().integer(source, 1), target)
        if target.name in {"String", "Enum", "Categorical"}:
            return Is.string(source)
        if Is.primitive_numeric(target) and Is.primitive_numeric(source):
            return self._number(source, target)
        if target.name == "Decimal":
            return self._decimal(source, target)
        return self._parametric(source, target)

    def _number(self, source: DType, target: DType) -> bool:
        """Check a cast between two number types.

        A float holds the small integers whose digits fit its mantissa.
        `Float64` takes a float literal. Other pairs need the lossless join of
        the two types to be the target.

        Args:
            source: The type that the cast reads.
            target: The type that the cast gives.

        Returns:
            Whether the cast keeps every value.

        Example:
            >>> LosslessCast()._number(DType("Int32"), DType("Float32"))
            False
        """
        if source.name in self.FLOAT_HOLDS.get(target.name, frozenset()):
            return True
        if target.name == "Float64" and Is.unknown(source):
            return True
        return LosslessUpcast().of(target, source) == target

    def _literal(self, value: int, target: DType) -> bool:
        """Check an integer literal by its value.

        Args:
            value: The value of the literal.
            target: The type that the cast gives.

        Returns:
            Whether the target holds the value with no loss.

        Example:
            >>> LosslessCast()._literal(2**24, DType("Float32"))
            False
        """
        if target.name == "Decimal":
            precision = Parameter().integer(target, 0)
            return Decimals().holds(value, precision, Parameter().integer(target, 1))
        if Is.integer(target):
            return IntegerRange().holds(target.name, value)
        bits = self.FLOAT_DIGITS.get(target.name)
        return bits is not None and abs(value) < 2**bits

    def _decimal(self, source: DType, target: DType) -> bool:
        """Check a cast to a `Decimal` from a `Decimal` or an integer.

        Polars reads the largest value of the integer type as a signed 128-bit
        integer, so it panics for `UInt128`, and `check` gives that panic. The
        test says that the cast does not keep every value.

        Args:
            source: The type that the cast reads.
            target: The decimal type.

        Returns:
            Whether every value of the source fits the decimal.

        Example:
            >>> LosslessCast()._decimal(DType("Int8"), DType("Decimal", (5, 2)))
            True
        """
        precision = Parameter().integer(target, 0)
        scale = Parameter().integer(target, 1)
        if source.name == "Decimal":
            source_digits = Parameter().integer(source, 0) - Parameter().integer(source, 1)
            wider_scale = scale >= Parameter().integer(source, 1)
            return precision - scale >= source_digits and wider_scale
        bounds = IntegerRange().BOUNDS.get(source.name)
        if bounds is None:
            return False
        return all(Decimals().holds(bound, precision, scale) for bound in bounds)

    def _parametric(self, source: DType, target: DType) -> bool:
        """Check a cast between two temporal or nested types.

        A `Datetime` or a `Duration` keeps its values where the target unit is
        as fine as the source unit. Lists and arrays check their inner types,
        and two structs check their fields by position.

        Args:
            source: The type that the cast reads.
            target: The type that the cast gives.

        Returns:
            Whether the cast keeps every value.

        Example:
            >>> LosslessCast()._parametric(DType("Duration", ("ns",)), DType("Duration", ("ms",)))
            False
        """
        names = (target.name, source.name)
        if names in {("Datetime", "Datetime"), ("Duration", "Duration")}:
            return TimeUnit.as_fine(Parameter().unit(target), Parameter().unit(source))
        nested = {("List", "List"), ("List", "Array"), ("Array", "List")}
        if names in nested:
            return self.of(source.inner(), target.inner())
        if names == ("Array", "Array"):
            same_width = Parameter().integer(source, 1) == Parameter().integer(target, 1)
            return same_width and self.of(source.inner(), target.inner())
        if names == ("Struct", "Struct"):
            return self._structs(source.fields(), target.fields())
        return False

    def _structs(
        self,
        source: tuple[tuple[str, DType], ...],
        target: tuple[tuple[str, DType], ...],
    ) -> bool:
        """Check two structs field by field, by position and by name.

        Two fields at the same position must have the same name.

        Args:
            source: The fields that the cast reads.
            target: The fields that the cast gives.

        Returns:
            Whether every field keeps its values.

        Example:
            >>> LosslessCast()._structs((("a", DType("Int8")),), (("b", DType("Int8")),))
            False
        """
        if len(source) != len(target):
            return False
        pairs = zip(source, target, strict=True)
        return all(a[0] == b[0] and self.of(a[1], b[1]) for a, b in pairs)


class ArrowCast:
    """Say whether the arrow kernel casts one physical type to another.

    Polars stores a date, a time, a decimal and a category as plain integers,
    and most of its kernels end in this arrow kernel on those physical
    types. The kernel casts a number to every number, to `Boolean`, to text
    and to a `Decimal`, and it wraps a flat value in a list of one. It
    refuses a cast between a struct and a type that is not a struct, and a
    cast of text to `Boolean`. A `List` here is the arrow large list and an
    `Array` the fixed-size list.

    Example:
        >>> ArrowCast().exists(DType("String"), DType("Int64"))
        True
        >>> ArrowCast().exists(DType("Int64"), DType("Array", (DType("Int64"), 2)))
        False
    """

    def exists(self, source: DType, target: DType) -> bool:
        """Say whether the kernel casts the physical source type to the physical target type.

        The arms are in the order of polars. A struct comes first, then a list
        or an array beside a list or an array, and then a list of `UInt8` to
        `Binary`. `Binary` and a flat value wrapped in a list come last, in
        that order. The other arms cast flat types.

        Args:
            source: The physical type that the cast reads.
            target: The physical type that the cast gives.

        Returns:
            Whether the kernel has an arm for the pair.

        Example:
            >>> ArrowCast().exists(DType("Boolean"), DType("List", (DType("String"),)))
            True
        """
        if source == target or "Null" in {source.name, target.name}:
            return True
        if "Struct" in {source.name, target.name}:
            return self._structs(source, target)
        listed = Is.LISTS
        if source.name in listed and target.name in listed:
            return self._lists(source, target)
        if source.name in {"List", "Binary"} or target.name == "List":
            return self._bytes_or_wrap(source, target)
        return _FlatArrow().exists(source, target)

    def _bytes_or_wrap(self, source: DType, target: DType) -> bool:
        """Say whether a list casts to `Binary`, `Binary` casts, or a flat value wraps in a list.

        A list casts to `Binary` only where it holds `UInt8`. `Binary` casts to
        `String` and to a list of `UInt8`. A flat value wraps in a list where
        it casts to the inner type.

        Args:
            source: The physical type that the cast reads.
            target: The physical type that the cast gives.

        Returns:
            Whether the kernel has an arm for the pair.

        Example:
            >>> ArrowCast()._bytes_or_wrap(DType("Binary"), DType("List", (DType("UInt8"),)))
            True
        """
        if source.name == "List":
            return target.name == "Binary" and source.inner().name == "UInt8"
        if source.name == "Binary":
            return target.name == "String" or target == DType("List", (DType("UInt8"),))
        return self.exists(source, target.inner())

    def _structs(self, source: DType, target: DType) -> bool:
        """Say whether two structs cast by the position of their fields.

        Args:
            source: The physical type that the cast reads.
            target: The physical type that the cast gives.

        Returns:
            Whether both are structs with the same number of fields, each of which casts.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> ArrowCast()._structs(struct, DType("Int8"))
            False
            >>> wide = DType("Struct", ((("a", DType("Int8")), ("b", DType("Int8"))),))
            >>> ArrowCast()._structs(struct, wide), ArrowCast()._structs(struct, struct)
            (False, True)
        """
        if (source.name, target.name) != ("Struct", "Struct"):
            return False
        if len(source.fields()) != len(target.fields()):
            return False
        pairs = zip(source.fields(), target.fields(), strict=True)
        return all(self.exists(a[1], b[1]) for a, b in pairs)

    def _lists(self, source: DType, target: DType) -> bool:
        """Say whether a list or an array casts to a list or an array.

        A list casts to an array and back, and a list to a list, by the cast
        of the inner type. An array does not cast to an array of another type.

        Args:
            source: The physical type that the cast reads.
            target: The physical type that the cast gives.

        Returns:
            Whether the kernel has an arm for the pair.

        Example:
            >>> pair = DType("Array", (DType("Int8"), 2))
            >>> ArrowCast()._lists(pair, DType("List", (DType("Int64"),)))
            True
            >>> ArrowCast()._lists(pair, pair)
            False
        """
        if (source.name, target.name) == ("Array", "Array"):
            return False
        return self.exists(source.inner(), target.inner())


class _FlatArrow:
    """Say whether the arrow kernel casts one flat physical type to another.

    A flat type here is a number, `Boolean`, `String`, `Binary` or a
    `Decimal`. Text parses to every number and to a `Decimal`. A number
    and a `Decimal` cast to each other, to `Boolean` and to text.

    Example:
        >>> _FlatArrow().exists(DType("String"), DType("Boolean"))
        False
    """

    def exists(self, source: DType, target: DType) -> bool:
        """Say whether the kernel has an arm for two flat physical types.

        Args:
            source: The physical type that the cast reads.
            target: The physical type that the cast gives.

        Returns:
            Whether the kernel has an arm for the pair.

        Example:
            >>> _FlatArrow().exists(DType("Decimal", (10, 2)), DType("Boolean"))
            True
        """
        number = Is.primitive_numeric(source) or source.name == "Decimal"
        if source.name == "String":
            return target.name in {"Binary", "Decimal"} or Is.primitive_numeric(target)
        if target.name == "Boolean":
            return number
        if source.name == "Boolean":
            return Is.primitive_numeric(target) or target.name in {"String", "Binary"}
        if target.name == "Binary":
            return Is.primitive_numeric(source)
        if target.name == "String":
            return number
        return number and (Is.primitive_numeric(target) or target.name == "Decimal")


class Outcome(Enum):
    """Name what a cast does to one value that is not null.

    `KEPT` keeps a value. `LOST` turns it into null where a strict cast sees
    the null and raises. `HIDDEN` turns a value inside a list into null where
    the strict check does not look, so a strict cast keeps the null. `RAISES`
    makes even a cast that is not strict raise. `PROCESS` depends on the
    state of the process, so no rule decides it.

    Example:
        >>> Outcome.worst([Outcome.KEPT, Outcome.LOST, Outcome.HIDDEN])
        <Outcome.LOST: 'lost'>
    """

    KEPT = "kept"
    HIDDEN = "hidden"
    PROCESS = "process"
    LOST = "lost"
    RAISES = "raises"

    @classmethod
    def worst(cls, outcomes: Iterable[Outcome]) -> Outcome:
        """Give the outcome that decides a cast of many values, `KEPT` for none.

        Args:
            outcomes: The outcomes of the values.

        Returns:
            The outcome that decides the cast.

        Example:
            >>> Outcome.worst([])
            <Outcome.KEPT: 'kept'>
        """
        order = list(cls)
        return max(outcomes, key=order.index, default=cls.KEPT)


class PhysicalValue:
    """Give the value that the physical array of a type holds for a Python value.

    A date is the number of days since 1970-01-01, a `Datetime` and a
    `Duration` count their unit, a `Time` counts nanoseconds since midnight,
    a `Decimal` holds its digits with no point, and an `Enum` holds the index
    of its category. A `Categorical` holds an id that the process assigns,
    so its value stays the text. A list and a struct convert each part.

    Example:
        >>> PhysicalValue().of(date(1970, 1, 3), DType("Date"))
        2
        >>> PhysicalValue().of(Decimal("1.50"), DType("Decimal", (10, 2)))
        150
    """

    EPOCH: ClassVar[datetime] = datetime(1970, 1, 1)

    def of(self, value: object, dtype: DType) -> object:
        """Give the physical value.

        A literal of the tree of polars holds a moment or a duration as its
        physical value already, so an integer for such a type is the physical
        value.

        Args:
            value: The Python value, or `None`.
            dtype: Its data type.

        Returns:
            The physical value, or `None`.

        Example:
            >>> PhysicalValue().of(time(0, 0, 1), DType("Time"))
            1000000000
            >>> PhysicalValue().of(None, DType("Time")) is None
            True
            >>> PhysicalValue().of(18262, DType("Date"))
            18262
        """
        if value is None:
            return None
        physical = isinstance(value, int) and not isinstance(value, bool)
        if physical and dtype.name in Is.TEMPORAL:
            return value
        readers: dict[str, Callable[[object], object]] = {
            "Date": lambda v: (self._date(v) - self.EPOCH.date()).days,
            "Datetime": lambda v: self._count(self._moment(v) - self.EPOCH, dtype),
            "Duration": lambda v: self._count(self._delta(v), dtype),
            "Time": self._nanoseconds,
            "Decimal": lambda v: self._digits(v, Parameter().integer(dtype, 1)),
            "Enum": lambda v: Parameter().categories(dtype).index(str(v)),
            "List": lambda v: self._each(v, dtype.inner()),
            "Array": lambda v: self._each(v, dtype.inner()),
            "Struct": lambda v: self._fields(v, dtype),
        }
        reader = readers.get(dtype.name)
        return value if reader is None else reader(value)

    def _date(self, value: object) -> date:
        """Read a Python date.

        Args:
            value: The value.

        Returns:
            The date.

        Example:
            >>> PhysicalValue()._date(date(2020, 1, 1))
            datetime.date(2020, 1, 1)
        """
        return checked_cast(value, date)

    def _moment(self, value: object) -> datetime:
        """Read a Python datetime as a naive moment in UTC.

        A series reads a datetime with no zone as UTC, also for a type with a
        zone. A literal reads it in the zone of its type, and the tree of
        polars holds its physical value, so this reader does not meet it.

        Args:
            value: The value.

        Returns:
            The moment with no time zone.

        Example:
            >>> PhysicalValue()._moment(datetime(2020, 1, 1, tzinfo=UTC))
            datetime.datetime(2020, 1, 1, 0, 0)
        """
        value = checked_cast(value, datetime)
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def _delta(self, value: object) -> timedelta:
        """Read a Python timedelta.

        Args:
            value: The value.

        Returns:
            The timedelta.

        Example:
            >>> PhysicalValue()._delta(timedelta(days=1))
            datetime.timedelta(days=1)
        """
        return checked_cast(value, timedelta)

    def _count(self, delta: timedelta, dtype: DType) -> int:
        """Count a timedelta in the unit of a `Datetime` or a `Duration`.

        Args:
            delta: The timedelta.
            dtype: The type, which holds the unit.

        Returns:
            The count.

        Example:
            >>> PhysicalValue()._count(timedelta(seconds=1), DType("Duration", ("ms",)))
            1000
        """
        per_second = TimeUnit.PER_SECOND[Parameter().unit(dtype)]
        seconds = delta.days * TimeUnit.SECONDS_PER_DAY + delta.seconds
        return seconds * per_second + delta.microseconds * per_second // 10**6

    def _nanoseconds(self, value: object) -> int:
        """Count the nanoseconds of a Python time since midnight.

        Args:
            value: The value.

        Returns:
            The count.

        Example:
            >>> PhysicalValue()._nanoseconds(time(1))
            3600000000000
        """
        value = checked_cast(value, time)
        seconds = (value.hour * 60 + value.minute) * 60 + value.second
        nanoseconds = TimeUnit.PER_SECOND["ns"]
        return seconds * nanoseconds + value.microsecond * (nanoseconds // 10**6)

    def _digits(self, value: object, scale: int) -> int:
        """Give the digits of a decimal value at a scale, with no point.

        Args:
            value: The value.
            scale: The scale of its type.

        Returns:
            The digits as an integer.

        Example:
            >>> PhysicalValue()._digits(Decimal("2.5"), 2)
            250
        """
        with localcontext() as context:
            context.prec = 200
            return int(Decimal(str(value)).scaleb(scale))

    def _each(self, value: object, inner: DType) -> list[object]:
        """Convert each element of a list.

        Args:
            value: The list.
            inner: The type of its elements.

        Returns:
            The physical elements.

        Example:
            >>> PhysicalValue()._each([date(1970, 1, 2)], DType("Date"))
            [1]
        """
        return [self.of(element, inner) for element in self._sequence(value)]

    def _sequence(self, value: object) -> list[object]:
        """Read a Python list.

        Args:
            value: The value.

        Returns:
            The list.

        Example:
            >>> PhysicalValue()._sequence([1])
            [1]
        """
        return checked_cast(value, list)

    def _fields(self, value: object, dtype: DType) -> dict[str, object]:
        """Convert each field of a struct value.

        Args:
            value: The struct value, a dictionary.
            dtype: The struct type.

        Returns:
            The physical fields by name.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Date")),),))
            >>> PhysicalValue()._fields({"a": date(1970, 1, 1)}, struct)
            {'a': 0}
        """
        fields = checked_cast(value, dict)
        return {name: self.of(fields.get(name), inner) for name, inner in dtype.fields()}


class TextParse:
    """Parse text as the arrow kernel of polars parses it.

    A cast of text to a number, a date, a `Datetime` or a `Decimal` parses each
    value, and a value that does not parse becomes null. An integer takes an
    optional sign and digits only, and an unsigned type does not take a minus
    sign, not even in `-0`. A float takes a sign, digits, a point and an
    exponent, and also `inf` and `nan`. A date takes `YYYY-MM-DD`, where the
    month and the day can have one digit. A `Datetime` takes the RFC 3339
    format `%Y-%m-%dT%H:%M:%S%.f%:z`. With no time zone, the parse keeps the
    fields that it read before an error, so the text needs the date, the hour
    and the minute. With a time zone, the whole format must parse, the offset
    included. A decimal takes a sign, digits, a point and an exponent, and
    rounds half to even.

    Example:
        >>> TextParse().integer("+1", signed=True), TextParse().integer(" 1", signed=True)
        (1, None)
    """

    SIGNED: ClassVar[re.Pattern[str]] = re.compile(r"[+-]?[0-9]+")
    UNSIGNED: ClassVar[re.Pattern[str]] = re.compile(r"\+?[0-9]+")
    FLOAT: ClassVar[re.Pattern[str]] = re.compile(
        r"[+-]?(?:inf|infinity|nan|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?)",
        re.IGNORECASE,
    )
    DATE: ClassVar[re.Pattern[str]] = re.compile(r"([+-]?[0-9]{4,})-([0-9]{1,2})-([0-9]{1,2})")
    NAIVE: ClassVar[re.Pattern[str]] = re.compile(
        r"([+-]?[0-9]{4,})-([0-9]{1,2})-([0-9]{1,2})T([0-9]{1,2}):([0-9]{1,2})",
    )
    ZONED: ClassVar[re.Pattern[str]] = re.compile(
        r"([+-]?[0-9]{4,})-([0-9]{1,2})-([0-9]{1,2})T([0-9]{1,2}):([0-9]{1,2}):[0-9]{1,2}"
        r"(?:\.[0-9]+)?[+-][0-9]{2}:[0-9]{2}",
    )
    DECIMAL: ClassVar[re.Pattern[str]] = re.compile(
        r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
    )

    def integer(self, text: str, *, signed: bool) -> int | None:
        """Parse an integer for a signed or an unsigned type.

        Args:
            text: The text.
            signed: Whether the target type is signed.

        Returns:
            The integer, or `None`.

        Example:
            >>> TextParse().integer("-0", signed=True), TextParse().integer("-0", signed=False)
            (0, None)
        """
        grammar = self.SIGNED if signed else self.UNSIGNED
        return int(text) if grammar.fullmatch(text) else None

    def number(self, text: str) -> bool:
        """Say whether text parses as a float.

        Args:
            text: The text.

        Returns:
            Whether it parses.

        Example:
            >>> TextParse().number(".5"), TextParse().number("2020-01-01")
            (True, False)
        """
        return self.FLOAT.fullmatch(text) is not None

    def date(self, text: str) -> bool:
        """Say whether text parses as a date.

        Args:
            text: The text.

        Returns:
            Whether it parses to a valid date.

        Example:
            >>> TextParse().date("2020-1-1"), TextParse().date("2020-02-30")
            (True, False)
        """
        match = self.DATE.fullmatch(text)
        if match is None:
            return False
        year, month, day = (int(part) for part in match.groups())
        try:
            date(year, month, day)
        except ValueError:
            return False
        return True

    def moment(self, text: str, *, zoned: bool) -> bool:
        """Say whether text parses as a `Datetime` in the RFC 3339 format.

        Args:
            text: The text.
            zoned: Whether the target type has a time zone.

        Returns:
            Whether it parses to a valid moment.

        Example:
            >>> TextParse().moment("2020-01-01T00:00", zoned=False)
            True
            >>> TextParse().moment("2020-01-01", zoned=False)
            False
            >>> TextParse().moment("2020-01-01T00:00:00+00:00", zoned=True)
            True
        """
        match = self.ZONED.fullmatch(text) if zoned else self.NAIVE.match(text)
        if match is None:
            return False
        year, month, day, hour, minute = (int(part) for part in match.groups())
        try:
            datetime(year, month, day, hour, minute)
        except ValueError:
            return False
        return True

    def decimal(self, text: str, precision: int, scale: int) -> bool:
        """Say whether text parses as a decimal that fits a precision and a scale.

        Args:
            text: The text.
            precision: The precision of the target.
            scale: The scale of the target.

        Returns:
            Whether it parses and fits after the rounding.

        Example:
            >>> TextParse().decimal("1e2", 10, 2), TextParse().decimal("999.5", 3, 0)
            (True, False)
            >>> TextParse().decimal("1e99999999999999999999", 10, 2)
            False
        """
        if self.DECIMAL.fullmatch(text) is None:
            return False
        with localcontext() as context:
            context.prec = 200
            try:
                value = Decimal(text)
            except InvalidOperation:
                return False
            digits = value.scaleb(scale).to_integral_value(rounding=ROUND_HALF_EVEN)
            return Decimals().fits(int(digits), precision)


@dataclass(frozen=True)
class ProcessCategories:
    """Name the one fact of the casts that the state of the process decides.

    An integer cast to `Categorical` keeps a value only where the process holds
    a category with that id, and the global categories live only while some
    series holds them. The id of a `Categorical` value cast to a narrow integer
    also depends on how many categories the process made before. The probe is a
    cast of the values 0, 1 and 2 to `Categorical`: it keeps them while a
    series of three categories is alive, and gives three nulls after that
    series is gone.

    Example:
        >>> ProcessCategories().holds_every_id("UInt32"), ProcessCategories().holds_every_id("Int8")
        (True, False)
    """

    def holds_every_id(self, integer: str) -> bool:
        """Say whether an integer type holds every id of a `Categorical`, a 32-bit unsigned integer.

        Args:
            integer: The name of the integer type.

        Returns:
            Whether every id fits.

        Example:
            >>> ProcessCategories().holds_every_id("Int64")
            True
        """
        ids = DType("Categorical").physical().name
        largest = IntegerRange().BOUNDS[ids][1]
        return integer in IntegerRange().BOUNDS and IntegerRange().holds(integer, largest)


class ArrowValues:
    """Give the outcome of the arrow kernel for one physical value.

    A number that does not fit an integer type becomes null, and so does a
    float that is NaN or infinite. A cast to a float never loses a value: a
    value out of range becomes infinite. A cast of text parses the text. A
    cast that wraps a flat value in a list keeps a null inside the list, where
    the strict check does not look. A list cast to an array raises where a
    list does not have the width of the array.

    Example:
        >>> ArrowValues().outcome(300, DType("Int64"), DType("Int8"))
        <Outcome.LOST: 'lost'>
        >>> ArrowValues().outcome(1e300, DType("Float64"), DType("Float32"))
        <Outcome.KEPT: 'kept'>
    """

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of the cast of one physical value.

        Args:
            value: The physical value, or `None`.
            source: The physical type of the value.
            target: The physical type that the cast gives.

        Returns:
            The outcome.

        Example:
            >>> ArrowValues().outcome("a", DType("String"), DType("List", (DType("Int64"),)))
            <Outcome.HIDDEN: 'hidden'>
        """
        if value is None or source == target:
            return Outcome.KEPT
        if target.name == "Null":
            return Outcome.LOST
        if source.name in {"List", "Array", "Struct"}:
            return self._nested(value, source, target)
        if target.name == "List":
            inner = self.outcome(value, source, target.inner())
            return Outcome.HIDDEN if inner is Outcome.LOST else inner
        return self._flat(value, source, target)

    def _nested(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome for a list, an array or a struct.

        A struct casts its fields by position. A list cast to an array raises
        where its length differs from the width. The inner values cast one by
        one, and the strict check sees their nulls. A list or an array casts to
        bytes, and a cast to any other type that is not nested raises, as a
        struct beside a type that is not a struct does.

        Args:
            value: The physical value.
            source: The physical type of the value.
            target: The physical type that the cast gives.

        Returns:
            The outcome.

        Example:
            >>> listed, pair = DType("List", (DType("Int64"),)), DType("Array", (DType("Int64"), 2))
            >>> ArrowValues()._nested([1], listed, pair)
            <Outcome.RAISES: 'raises'>
            >>> struct = DType("Struct", ((("a", DType("Int64")),),))
            >>> narrow = DType("Struct", ((("a", DType("Int8")),),))
            >>> ArrowValues()._nested({"a": 300}, struct, narrow)
            <Outcome.LOST: 'lost'>
            >>> ArrowValues()._nested([1], listed, DType("Binary"))
            <Outcome.KEPT: 'kept'>
            >>> ArrowValues()._nested([1], listed, DType("Float64"))
            <Outcome.RAISES: 'raises'>
        """
        if "Struct" in {source.name, target.name}:
            return self._fields(value, source, target)
        elements = value if isinstance(value, list) else []
        if target.name == "Binary":
            return Outcome.KEPT
        if target.name not in {"List", "Array"}:  # pragma: no cover -- `CastKernel.outcome`
            return Outcome.RAISES  # first refuses a list into another flat type.
        if target.name == "Array" and len(elements) != Parameter().integer(target, 1):
            return Outcome.RAISES
        inner = (source.inner(), target.inner())
        return Outcome.worst(self.outcome(element, *inner) for element in elements)

    def _fields(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome for a struct, which casts its fields by position to another struct.

        Args:
            value: The physical value: a dictionary of the fields.
            source: The physical type of the value.
            target: The physical type that the cast gives.

        Returns:
            The outcome, `RAISES` where one side is not a struct.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int64")),),))
            >>> ArrowValues()._fields({"a": 1}, struct, DType("Int64"))
            <Outcome.RAISES: 'raises'>
        """
        if (source.name, target.name) != ("Struct", "Struct"):  # pragma: no cover -- the cast
            return Outcome.RAISES  # of the types refuses a struct beside another type in a list.
        values = value.values() if isinstance(value, dict) else []
        pairs = zip(values, source.fields(), target.fields(), strict=False)
        return Outcome.worst(self.outcome(v, a[1], b[1]) for v, a, b in pairs)

    def _flat(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome for a flat value.

        Args:
            value: The physical value.
            source: The physical type of the value.
            target: The physical type that the cast gives.

        Returns:
            The outcome.

        Example:
            >>> ArrowValues()._flat("1.5", DType("String"), DType("Float64"))
            <Outcome.KEPT: 'kept'>
            >>> ArrowValues()._flat(1000, DType("Int64"), DType("Decimal", (5, 2)))
            <Outcome.LOST: 'lost'>
        """
        if Is.integer(target):
            return self._in_range(self.integer(value, target), target)
        if target.name == "Decimal":  # pragma: no cover -- a cast of an element reads physical
            return DecimalFit().outcome(value, target)  # types, where a decimal is `Int128`.
        if Is.float(target) and isinstance(value, str):
            return Outcome.KEPT if TextParse().number(value) else Outcome.LOST
        if source.name == "Binary" and target.name == "String":
            return Outcome.KEPT if self._utf8(value) else Outcome.RAISES
        return Outcome.KEPT

    def _in_range(self, number: int | None, target: DType) -> Outcome:
        """Keep an integer that the integer target holds, and lose the others.

        Args:
            number: The integer that the cast reads, or `None` where there is none.
            target: The integer type.

        Returns:
            The outcome.

        Example:
            >>> ArrowValues()._in_range(256, DType("UInt8"))
            <Outcome.LOST: 'lost'>
        """
        in_range = number is not None and IntegerRange().holds(target.name, number)
        return Outcome.KEPT if in_range else Outcome.LOST

    def integer(self, value: object, target: DType) -> int | None:
        """Give the integer that a cast to an integer type reads, before it checks the range.

        A float truncates toward zero, and NaN or an infinite value gives
        nothing. Text parses with the grammar of the target type.

        Args:
            value: The physical value.
            target: The integer type of the cast.

        Returns:
            The integer, or `None`.

        Example:
            >>> ArrowValues().integer(-1.9, DType("Int8"))
            -1
            >>> ArrowValues().integer(b"1", DType("Int8")) is None
            True
        """
        if isinstance(value, str):
            return TextParse().integer(value, signed=not Is.unsigned(target))
        if isinstance(value, float):
            return int(value) if math.isfinite(value) else None
        if isinstance(value, (bool, int)):
            return int(value)
        return None  # pragma: no cover -- the cast of the types refuses any other value first.

    def _utf8(self, value: object) -> bool:
        r"""Say whether bytes are valid UTF-8, as a cast of `Binary` to `String` needs.

        The cast raises for bytes that are not text.

        Args:
            value: The bytes.

        Returns:
            Whether they are bytes that decode.

        Example:
            >>> ArrowValues()._utf8(b"\xff")
            False
            >>> ArrowValues()._utf8("a")
            False
        """
        if not isinstance(value, bytes):  # pragma: no cover -- the physical value of `Binary`
            return False  # is bytes.
        try:
            value.decode()
        except UnicodeDecodeError:
            return False
        return True


class DecimalFit:
    """Give the outcome of a cast of one value to a `Decimal`.

    An integer keeps its value where the decimal holds it (`Decimals.holds`). A
    float is checked against ten to the precision before the scale applies,
    then it rounds half to even. So a value that passes the check can still
    overflow the precision, and the check of the whole column then raises.

    Example:
        >>> DecimalFit().outcome(1.0, DType("Decimal", (3, 3)))
        <Outcome.RAISES: 'raises'>
        >>> DecimalFit().outcome(1e30, DType("Decimal", (10, 2)))
        <Outcome.LOST: 'lost'>
    """

    def outcome(self, value: object, target: DType) -> Outcome:
        """Give the outcome.

        Args:
            value: The physical value: a number or text.
            target: The decimal type.

        Returns:
            The outcome.

        Example:
            >>> DecimalFit().outcome(1000, DType("Decimal", (5, 2)))
            <Outcome.LOST: 'lost'>
            >>> DecimalFit().outcome("1.5", DType("Decimal", (5, 2)))
            <Outcome.KEPT: 'kept'>
        """
        precision = Parameter().integer(target, 0)
        scale = Parameter().integer(target, 1)
        if isinstance(value, str):
            kept = TextParse().decimal(value, precision, scale)
            return Outcome.KEPT if kept else Outcome.LOST
        if isinstance(value, float):
            return self._float(value, precision, scale)
        number = ArrowValues().integer(value, DType("Int128"))
        fits = number is not None and Decimals().holds(number, precision, scale)
        return Outcome.KEPT if fits else Outcome.LOST

    def _float(self, value: float, precision: int, scale: int) -> Outcome:
        """Give the outcome of a float.

        Args:
            value: The float.
            precision: The precision of the target.
            scale: The scale of the target.

        Returns:
            The outcome.

        Example:
            >>> DecimalFit()._float(float("nan"), 10, 2)
            <Outcome.LOST: 'lost'>
        """
        if not abs(value) < 10.0**precision:
            return Outcome.LOST
        digits = round(value * 10.0**scale)
        return Outcome.KEPT if Decimals().fits(digits, precision) else Outcome.RAISES


class CastKernel:
    """Give the type that a cast gives a series that holds values, by the kind of its source.

    The cast of a series first adjusts the target and takes the shortcuts that
    read no data. It then calls the kernel of the source kind, one strategy
    class here for each kernel of polars. Each strategy also gives the outcome
    of one value, so a strict cast knows which values it loses. A kernel that
    has no cast for the pair raises `PolarsRefusal` with the class of the error
    of polars.

    Example:
        >>> CastKernel().output(DType("Int8"), DType("Date"))
        Date
        >>> CastKernel().output(DType("Date"), DType("Boolean"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    def __init__(self) -> None:
        """Build the strategy of each source kind.

        Example:
            >>> sorted(CastKernel()._kinds)[:3]
            ['Array', 'Binary', 'Boolean']
        """
        flat = _FlatCast(self)
        temporal = _TemporalCast()
        categorical = _CategoricalCast()
        self._nested: dict[str, _NestedCast] = {
            "List": _ListCast(self),
            "Array": _ArrayCast(self),
            "Struct": _StructCast(self),
        }
        self._kinds: dict[str, _KindCast] = {
            "String": _TextCast(flat),
            "Binary": flat,
            "Boolean": flat,
            "Null": _NullCast(),
            "Object": _ObjectCast(),
            **self._nested,
            "Decimal": _DecimalCast(),
            "Categorical": categorical,
            "Enum": categorical,
            **dict.fromkeys(Is.TEMPORAL, temporal),
        }
        self._number = flat

    def clones(self, source: DType, target: DType) -> bool:
        """Say whether the cast returns the series as it is.

        A cast to a flat type that equals the source, or to a planner type
        `Unknown` that the source already fits, reads no data.

        Args:
            source: The type of the series.
            target: The type of the cast.

        Returns:
            Whether the series stays as it is.

        Example:
            >>> CastKernel().clones(DType("Int8"), DType.dynamic_int(1))
            True
        """
        if Is.unknown(target):
            what = target.args[0]
            fits = {
                "any": True,
                "int": Is.integer(source),
                "float": Is.float(source),
                "str": Is.string(source) or source.name == "Categorical",
            }
            return bool(fits.get(str(what)))
        return Is.primitive(target) and target.same(source)

    def target(self, source: DType, target: DType) -> DType:
        """Give the target that the cast uses.

        A planner type `Unknown` in the target takes the type of its literal.
        A cast of a list or an array to `List(Null)` keeps the inner type of
        the source.

        Args:
            source: The type of the series.
            target: The type of the cast.

        Returns:
            The target.

        Example:
            >>> listed = DType("List", (DType("Int8"),))
            >>> CastKernel().target(listed, DType("List", (DType("Null"),)))
            List(Int8)
        """
        adjusted = target if Is.known(target) else target.materialized()
        wants_inner = adjusted.name == "List" and adjusted.inner().name == "Null"
        if wants_inner and source.name in Is.LISTS:
            return DType("List", (source.inner(),))
        return adjusted

    def output(self, source: DType, target: DType) -> DType:
        """Give the type that the cast gives a series that holds values.

        It raises `PolarsRefusal` where the kernel has no cast for the pair.

        Args:
            source: The type of the series.
            target: The type of the cast.

        Returns:
            The type.

        Example:
            >>> CastKernel().output(DType("Int8"), DType("List", (DType("Date"),)))
            List(Int32)
        """
        if self.clones(source, target):
            return source
        return self._kind(source).output(source, self.target(source, target))

    def inner(self, source: DType, target: DType) -> list[tuple[str, DType | None, DType]] | None:
        """Give the inner casts of a nested cast that casts each inner series as a series.

        A cast of a list to a list, of an array to an array, and of a struct
        to a type other than `String` casts the inner values, or each field,
        with a cast of a series. That cast takes its own shortcuts, so inner
        values that are all null cast to any type. The checks of the outer
        types run first, and raise `PolarsRefusal` where they fail.

        Args:
            source: The type of the series.
            target: The type of the cast.

        Returns:
            For each inner series: the name of its field (empty for a list or
            an array), its type or `None` for a field that the source lacks,
            and the type of its cast. `None` where the cast does not cast an
            inner series.

        Example:
            >>> listed = DType("List", (DType("Binary"),))
            >>> CastKernel().inner(listed, DType("List", (DType("Date"),)))
            [('', Binary, Date)]
        """
        strategy = self._nested.get(source.name)
        if strategy is None or self.clones(source, target):
            return None
        return strategy.inner(source, self.target(source, target))

    def drops_every_value(self, source: DType, target: DType) -> bool:
        """Say whether the cast turns every value at one place of the type into a null.

        A cast to `Null` gives a null for each value, and a strict cast sees
        those nulls. A struct casts each field with the options of the cast, so
        a field that becomes `Null` drops its values too. A strict cast of such
        a type raises on every series that
        holds a value there, so it is a fault of the types, not of the values.
        The strict check does not look inside a list, so a list is never such a
        place.

        It raises `PolarsRefusal` where the kernel has no cast for the pair.

        Args:
            source: The type of the series.
            target: The type of the cast.

        Returns:
            Whether the cast drops every value at one place.

        Example:
            >>> CastKernel().drops_every_value(DType("Int8"), DType("Null"))
            True
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> CastKernel().drops_every_value(struct, DType("Null"))
            True
            >>> CastKernel().drops_every_value(DType("Int8"), DType("Int16"))
            False
        """
        return self._drops(source, self.output(source, target))

    def _drops(self, source: DType, output: DType) -> bool:
        """Say whether an output type holds `Null` at a place where the source holds another type.

        Args:
            source: The type of the series.
            output: The type that the cast gives.

        Returns:
            Whether it does, at the top or in a field of a struct, by name.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")), ("b", DType("Null"))),))
            >>> CastKernel()._drops(struct, struct)
            False
        """
        if output.name == "Null":
            return source.name != "Null"
        if output.name != "Struct" or source.name != "Struct":
            return False
        fields = dict(source.fields())
        pairs = [(fields[name], dtype) for name, dtype in output.fields() if name in fields]
        return any(itertools.starmap(self._drops, pairs))

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of the cast of one value of the series.

        A pair of types that the kernel has no cast for raises at each value.

        Args:
            value: The Python value, or `None`.
            source: The type of the series.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> CastKernel().outcome("a", DType("String"), DType("Enum", (("a",),)))
            <Outcome.KEPT: 'kept'>
            >>> CastKernel().outcome(b"1", DType("Binary"), DType("Int64"))
            <Outcome.RAISES: 'raises'>
        """
        if value is None or self.clones(source, target):
            return Outcome.KEPT
        try:
            self.output(source, target)
        except PolarsRefusal:
            return Outcome.RAISES
        return self._kind(source).outcome(value, source, self.target(source, target))

    def _kind(self, source: DType) -> _KindCast:
        """Give the strategy of the source kind.

        Args:
            source: The type of the series.

        Returns:
            The strategy.

        Example:
            >>> type(CastKernel()._kind(DType("Float32"))).__name__
            '_FlatCast'
        """
        if Is.primitive_numeric(source):
            return self._number
        return self._kinds[source.name]


class _KindCast(Protocol):
    """Give the output type of a cast and the outcome of each value, for one source kind."""

    def output(self, source: DType, target: DType) -> DType:
        """Give the output type, or raise `PolarsRefusal` where the kernel has no cast.

        Args:
            source: The type of the series.
            target: The type of the cast.
        """

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of one value.

        Args:
            value: The value.
            source: The type of the series.
            target: The type of the cast.
        """


class _NestedCast(_KindCast, Protocol):
    """Give the inner casts of a nested cast, for the kind of a list, an array or a struct."""

    def inner(self, source: DType, target: DType) -> list[tuple[str, DType | None, DType]] | None:
        """Give the inner casts, or `None` where the cast does not cast an inner series.

        Args:
            source: The type of the series.
            target: The type of the cast, adjusted.
        """


class _NoCast:
    """Give the text of a cast kernel that has no cast for a pair of types.

    Most kernels of polars raise an `InvalidOperationError` with this text, for
    example the kernel of `Date`.

    Example:
        >>> _NoCast().message(DType("Date"), DType("Binary"))
        'casting from Date to Binary not supported'
    """

    def message(self, source: DType, target: DType) -> str:
        """Give the text.

        Args:
            source: The type of the series.
            target: The type of the cast.

        Returns:
            The text.

        Example:
            >>> _NoCast().message(DType("Int8"), DType("Time"))
            'casting from Int8 to Time not supported'
        """
        return f"casting from {source!r} to {target!r} not supported"


class _FlatCast:
    """Cast a number, `Boolean` or `Binary`.

    A number casts to a `Categorical` or an `Enum` through the ids of its
    categories, and to a struct of one field. `Boolean` does not cast to a
    `Categorical` or an `Enum`. Every other cast goes to the arrow kernel on
    the physical types, and the result keeps a flat logical target, a date,
    a time, a `Datetime`, a `Duration` or a `Decimal`. A nested target keeps
    its physical type, so a cast of `Int8` to `List(Date)` gives `List(Int32)`.

    Example:
        >>> _FlatCast(CastKernel()).output(DType("Int8"), DType("Int16"))
        Int16
    """

    LOGICAL: ClassVar[frozenset[str]] = Is.TEMPORAL | {"Decimal"}

    def __init__(self, kernel: CastKernel) -> None:
        """Keep the kernel, which reads the categories of an `Enum`.

        Args:
            kernel: The cast kernel.

        Example:
            >>> _FlatCast(CastKernel())._kernel is not None
            True
        """
        self._kernel = kernel

    def output(self, source: DType, target: DType) -> DType:
        """Give the output type.

        Args:
            source: The type of the series.
            target: The type of the cast.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The source is not a number and the target holds categories.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Date")),),))
            >>> _FlatCast(CastKernel()).output(DType("Int8"), struct)
            Struct({'a': Date})
        """
        if target.name == "Struct":
            return self._single_field(source, target)
        if target.name in Is.CATEGORIES and not Is.primitive_numeric(source):
            message = f"cannot cast {source!r} to {target.name}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if target.name in Is.CATEGORIES:
            return target
        return self.inner_output(source, target)

    def inner_output(self, source: DType, target: DType) -> DType:
        """Give the type of the arrow cast of the physical types.

        Args:
            source: The type of the series.
            target: The type of the cast.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The arrow kernel has no arm for the physical types.

        Example:
            >>> _FlatCast(CastKernel()).inner_output(DType("String"), DType("Time"))
            Time
        """
        physical_target = target if target.name == "Decimal" else target.physical()
        if not ArrowCast().exists(source.physical(), physical_target):
            message = _NoCast().message(source, target)
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return target if target.name in self.LOGICAL else target.physical()

    def _single_field(self, source: DType, target: DType) -> DType:
        """Give the type of a cast of a flat value to a struct of one field.

        It raises `PolarsRefusal` where the field does not cast.

        Args:
            source: The type of the series.
            target: The struct type.

        Returns:
            The struct.

        Raises:
            PolarsRefusal: The struct does not have exactly one field.

        Example:
            >>> pair = DType("Struct", ((("a", DType("Int8")), ("b", DType("Int8"))),))
            >>> _FlatCast(CastKernel())._single_field(DType("Int8"), pair)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if len(target.fields()) != 1:
            message = "must specify one field in the struct"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        name, field = target.fields()[0]
        return DType("Struct", (((name, self.inner_output(source, field)),),))

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of one value.

        Args:
            value: The value.
            source: The type of the series.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> letters = DType("Enum", (("a", "b", "c"),))
            >>> _FlatCast(CastKernel()).outcome(3, DType("Int64"), letters)
            <Outcome.LOST: 'lost'>
        """
        if target.name == "Struct":
            return self.inner_outcome(value, source, target.fields()[0][1])
        if target.name == "Categorical":
            to_id = ArrowValues().outcome(value, source, target.physical())
            return Outcome.PROCESS if to_id is Outcome.KEPT else to_id
        if target.name == "Enum":
            return self._enum(value, target)
        return self.inner_outcome(value, source, target)

    def inner_outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of the arrow cast of the physical types for one value.

        A cast to `Time` first casts to the physical `Int64`, then nulls a
        value that is not a time of day.

        Args:
            value: The value.
            source: The type of the series.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> _FlatCast(CastKernel()).inner_outcome(-1, DType("Int64"), DType("Time"))
            <Outcome.LOST: 'lost'>
        """
        physical = PhysicalValue().of(value, source)
        if target.name == "Time":
            number = ArrowValues().integer(physical, DType("Int64"))
            valid = number is not None and 0 <= number < TimeUnit.per_day("ns")
            return Outcome.KEPT if valid else Outcome.LOST
        if target.name == "Decimal":
            return DecimalFit().outcome(physical, target)
        return ArrowValues().outcome(physical, source.physical(), target.physical())

    def _enum(self, value: object, target: DType) -> Outcome:
        """Give the outcome of a number cast to an `Enum`.

        The number casts to the physical type of the ids, and an id with no
        category becomes null.

        Args:
            value: The number.
            target: The `Enum` type.

        Returns:
            The outcome.

        Example:
            >>> _FlatCast(CastKernel())._enum(1.9, DType("Enum", (("a", "b"),)))
            <Outcome.KEPT: 'kept'>
        """
        number = ArrowValues().integer(value, target.physical())
        count = len(Parameter().categories(target))
        known = number is not None and 0 <= number < count
        return Outcome.KEPT if known else Outcome.LOST


class _TextCast:
    """Cast `String`.

    Text casts to a `Categorical`, an `Enum`, a `Decimal`, a date and a
    `Datetime` by parsing or by looking up each value. Every other cast goes
    the way of a flat type, so text parses to a number, and to a `Time` or a
    `Duration` through the physical `Int64`.

    Example:
        >>> _TextCast(_FlatCast(CastKernel())).output(DType("String"), DType("Int8"))
        Int8
    """

    DIRECT: ClassVar[frozenset[str]] = Is.CATEGORIES | {"Decimal", "Date", "Datetime"}

    def __init__(self, flat: _FlatCast) -> None:
        """Keep the strategy of flat types, for the casts that go the same way.

        Args:
            flat: The strategy of flat types.

        Example:
            >>> _TextCast(_FlatCast(CastKernel()))._flat is not None
            True
        """
        self._flat = flat

    def output(self, source: DType, target: DType) -> DType:
        """Give the output type.

        It raises `PolarsRefusal` where the cast goes the way of a flat type
        and that kernel has no cast, for example to `Boolean`.

        Args:
            source: `String`.
            target: The type of the cast.

        Returns:
            The type.

        Example:
            >>> _TextCast(_FlatCast(CastKernel())).output(DType("String"), DType("Enum", (("a",),)))
            Enum(categories=['a'])
        """
        if target.name in self.DIRECT:
            return target
        return self._flat.output(source, target)

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of one value.

        Args:
            value: The text.
            source: `String`.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> text = _TextCast(_FlatCast(CastKernel()))
            >>> text.outcome("2020-01-01", DType("String"), DType("Date"))
            <Outcome.KEPT: 'kept'>
        """
        text = str(value)
        checks: dict[str, Callable[[], bool]] = {
            "Categorical": lambda: True,
            "Enum": lambda: text in Parameter().categories(target),
            "Decimal": lambda: TextParse().decimal(
                text,
                Parameter().integer(target, 0),
                Parameter().integer(target, 1),
            ),
            "Date": lambda: TextParse().date(text),
            "Datetime": lambda: TextParse().moment(text, zoned=target.args[1] is not None),
        }
        check = checks.get(target.name)
        if check is None:
            return self._flat.outcome(value, source, target)
        return Outcome.KEPT if check() else Outcome.LOST


class _NullCast:
    """Cast a series of `Null`, which holds only nulls.

    The kernel builds a series of nulls of the target type.

    Example:
        >>> _NullCast().output(DType("Null"), DType("Date"))
        Date
    """

    def output(self, source: DType, target: DType) -> DType:
        """Give the target: a series of nulls casts to any type.

        Args:
            source: `Null`.
            target: The type of the cast.

        Returns:
            The target.

        Example:
            >>> _NullCast().output(DType("Null"), DType("Binary"))
            Binary
        """
        return target

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Keep every value: a series of `Null` holds no value that a cast can lose.

        Args:
            value: The value, always `None`.
            source: `Null`.
            target: The type of the cast.

        Returns:
            `KEPT`.

        Example:
            >>> _NullCast().outcome(None, DType("Null"), DType("Int8"))
            <Outcome.KEPT: 'kept'>
        """
        return Outcome.KEPT  # pragma: no cover -- `CastKernel.outcome` keeps a null first.


class _ObjectCast:
    """Cast a series of `Object`, which holds Python objects that polars does not read.

    The kernel casts such a series only to `Object`, and raises a
    `ComputeError` for any other target, strict or not.

    Example:
        >>> _ObjectCast().output(DType("Object"), DType("Object"))
        Object
    """

    def output(self, source: DType, target: DType) -> DType:
        """Give the target `Object`, or refuse every other target.

        Args:
            source: `Object`.
            target: The type of the cast.

        Returns:
            The target.

        Raises:
            PolarsRefusal: The target is not `Object`.

        Example:
            >>> _ObjectCast().output(DType("Object"), DType("Int8"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: cannot cast 'Object' type
        """
        if target.name != "Object":
            raise PolarsRefusal(ErrorClass.COMPUTE, "cannot cast 'Object' type")
        return target

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Keep a value: the only cast that runs gives the series back.

        Args:
            value: The value.
            source: `Object`.
            target: `Object`.

        Returns:
            `KEPT`.

        Example:
            >>> _ObjectCast().outcome(object(), DType("Object"), DType("Object"))
            <Outcome.KEPT: 'kept'>
        """
        return Outcome.KEPT


class _ListCast:
    """Cast a `List`.

    A list casts to a list by the cast of its inner type, which runs as a
    cast of a series. A list casts to an array through the arrow kernel, and
    raises where a list does not have the width of the array. A list of
    `UInt8` casts to `Binary`. A list of a type that is not text or
    categories does not cast to a list of categories.

    Example:
        >>> listed = DType("List", (DType("Int64"),))
        >>> _ListCast(CastKernel()).output(listed, DType("List", (DType("Date"),)))
        List(Date)
    """

    TEXT: ClassVar[frozenset[str]] = Is.CATEGORIES | {"String", "Null"}

    def __init__(self, kernel: CastKernel) -> None:
        """Keep the kernel, which casts the inner type.

        Args:
            kernel: The cast kernel.

        Example:
            >>> _ListCast(CastKernel())._kernel is not None
            True
        """
        self._kernel = kernel

    def output(self, source: DType, target: DType) -> DType:
        """Give the output type.

        It raises `PolarsRefusal` where the inner type does not cast to a
        list (`_listed`).

        Args:
            source: The list type.
            target: The type of the cast.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The kernel has no cast for the pair.

        Example:
            >>> listed = DType("List", (DType("UInt8"),))
            >>> _ListCast(CastKernel()).output(listed, DType("Binary"))
            Binary
        """
        if target.name == "Array":
            if not ArrowCast().exists(source.physical(), target.physical()):
                message = _NoCast().message(source, target)
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
            return target
        binary = target.name == "Binary" and source.inner().name == "UInt8"
        if binary:
            return target
        if target.name != "List":
            message = f"cannot cast List type (inner: {source.inner()!r}, to: {target!r})"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return self._listed(source, target)

    def _listed(self, source: DType, target: DType) -> DType:
        """Give the type of a cast of a list to a list, by the cast of its inner type.

        It raises `PolarsRefusal` where the inner type does not cast, or
        where the list holds numbers and the target holds categories.

        Args:
            source: The list type.
            target: The list type of the cast.

        Returns:
            The list of the cast inner type.

        Example:
            >>> numbers = DType("List", (DType("Int8"),))
            >>> letters = DType("List", (DType("Enum", (("a",),)),))
            >>> _ListCast(CastKernel())._listed(numbers, letters)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if self.inner(source, target) is None:
            return source
        return DType("List", (self._kernel.output(source.inner(), target.inner()),))

    def inner(self, source: DType, target: DType) -> list[tuple[str, DType | None, DType]] | None:
        """Give the cast of the inner values of a list cast to a list, after a check of categories.

        Args:
            source: The list type.
            target: The type of the cast.

        Returns:
            The inner type and its target, or `None` where the cast does not
            cast the inner values.

        Raises:
            PolarsRefusal: The list holds numbers and the target holds
                categories.

        Example:
            >>> listed = DType("List", (DType("Int8"),))
            >>> _ListCast(CastKernel()).inner(listed, DType("List", (DType("Date"),)))
            [('', Int8, Date)]
        """
        if target.name != "List" or source.inner().same(target.inner()):
            return None
        categorical = target.inner().name in Is.CATEGORIES
        if categorical and source.inner().name not in self.TEXT:
            message = f"cannot cast List inner type: {source.inner()!r} to Categorical"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return [("", source.inner(), target.inner())]

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of one list.

        Args:
            value: The list.
            source: The list type.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> listed = DType("List", (DType("Int64"),))
            >>> _ListCast(CastKernel()).outcome([300], listed, DType("List", (DType("Int8"),)))
            <Outcome.LOST: 'lost'>
        """
        elements = value if isinstance(value, list) else []
        if target.name == "List":
            pair = (source.inner(), target.inner())
            return Outcome.worst(self._kernel.outcome(element, *pair) for element in elements)
        physical = PhysicalValue().of(value, source)
        return ArrowValues().outcome(physical, source.physical(), target.physical())


class _ArrayCast:
    """Cast an `Array`.

    An array casts to an array of the same width by the cast of its inner
    type, which runs as a cast of a series. It casts to a list through the
    arrow kernel. An array of a type that is not text does not cast to an
    array of categories.

    Example:
        >>> pair = DType("Array", (DType("Int64"), 2))
        >>> _ArrayCast(CastKernel()).output(pair, DType("List", (DType("String"),)))
        List(String)
    """

    def __init__(self, kernel: CastKernel) -> None:
        """Keep the kernel, which casts the inner type.

        Args:
            kernel: The cast kernel.

        Example:
            >>> _ArrayCast(CastKernel())._kernel is not None
            True
        """
        self._kernel = kernel

    def output(self, source: DType, target: DType) -> DType:
        """Give the output type.

        It raises `PolarsRefusal` where the inner type does not cast.

        Args:
            source: The array type.
            target: The type of the cast.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The kernel has no cast for the pair.

        Example:
            >>> pair = DType("Array", (DType("Int64"), 2))
            >>> _ArrayCast(CastKernel()).output(pair, DType("Array", (DType("Int8"), 2)))
            Array(Int8, shape=(2,))
            >>> _ArrayCast(CastKernel()).output(pair, DType("Array", (DType("Categorical"), 2)))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot cast Array ...
        """
        if target.name == "List":
            if not ArrowCast().exists(source.physical(), target.physical()):
                message = _NoCast().message(source, target)
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
            return target
        if target.name != "Array":
            message = f"cannot cast Array type (inner: {source.inner()!r}, to: {target!r})"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if self.inner(source, target) is None:
            return source
        width = Parameter().integer(source, 1)
        return DType("Array", (self._kernel.output(source.inner(), target.inner()), width))

    def inner(self, source: DType, target: DType) -> list[tuple[str, DType | None, DType]] | None:
        """Give the cast of the inner values of an array cast to an array, after the outer checks.

        Args:
            source: The array type.
            target: The type of the cast.

        Returns:
            The inner type and its target, or `None` where the cast does not
            cast the inner values.

        Raises:
            PolarsRefusal: The widths differ, or the array holds other than
                text and the target holds categories.

        Example:
            >>> pair = DType("Array", (DType("Int64"), 2))
            >>> _ArrayCast(CastKernel()).inner(pair, DType("Array", (DType("Int8"), 2)))
            [('', Int64, Int8)]
        """
        if target.name != "Array":
            return None
        if Parameter().integer(source, 1) != Parameter().integer(target, 1):
            message = "cannot cast Array to a different width"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if source.inner().same(target.inner()):
            return None
        categorical = target.inner().name in Is.CATEGORIES
        if categorical and source.inner().name != "String":
            message = f"cannot cast Array inner type: {source.inner()!r} to {target.inner()!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return [("", source.inner(), target.inner())]

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of one array.

        Args:
            value: The array, a Python list.
            source: The array type.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> pair = DType("Array", (DType("Int64"), 2))
            >>> _ArrayCast(CastKernel()).outcome([1, 300], pair, DType("List", (DType("Int8"),)))
            <Outcome.LOST: 'lost'>
        """
        elements = value if isinstance(value, list) else []
        if target.name == "Array":
            pair = (source.inner(), target.inner())
            return Outcome.worst(self._kernel.outcome(element, *pair) for element in elements)
        physical = PhysicalValue().of(value, source)
        return ArrowValues().outcome(physical, source.physical(), target.physical())


class _StructCast:
    """Cast a `Struct`.

    A struct casts to a struct field by field by name. A field of the target
    that the source lacks becomes null, and a field of the source that the
    target lacks is dropped. A struct casts to `String` as text such as
    `{1,a}`. A cast to any other type casts each field to that type, so the
    result is still a struct.

    Example:
        >>> struct = DType("Struct", ((("a", DType("Int8")),),))
        >>> _StructCast(CastKernel()).output(struct, DType("Int64"))
        Struct({'a': Int64})
    """

    def __init__(self, kernel: CastKernel) -> None:
        """Keep the kernel, which casts each field.

        Args:
            kernel: The cast kernel.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> _StructCast(CastKernel()).output(struct, DType("String"))
            String
        """
        self._kernel = kernel

    def output(self, source: DType, target: DType) -> DType:
        """Give the output type.

        It raises `PolarsRefusal` where a field does not cast.

        Args:
            source: The struct type.
            target: The type of the cast.

        Returns:
            The type.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> other = DType("Struct", ((("b", DType("Date")),),))
            >>> _StructCast(CastKernel()).output(struct, other)
            Struct({'b': Date})
        """
        if target.name == "String":
            return target
        fields = []
        for name, field, other in self._pairs(source, target):
            cast = other if field is None else self._kernel.output(field, other)
            fields.append((name, cast))
        return DType("Struct", (tuple(fields),))

    def inner(self, source: DType, target: DType) -> list[tuple[str, DType | None, DType]] | None:
        """Give the cast of each field, which a cast to a type other than `String` runs as a series.

        Args:
            source: The struct type.
            target: The type of the cast.

        Returns:
            The fields of the result with their casts, or `None` for a cast to `String`.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> _StructCast(CastKernel()).inner(struct, DType("String")) is None
            True
        """
        if target.name == "String":
            return None
        return self._pairs(source, target)

    def _pairs(self, source: DType, target: DType) -> list[tuple[str, DType | None, DType]]:
        """Pair each field of the result with the source field that it reads and its target type.

        Args:
            source: The struct type.
            target: The type of the cast.

        Returns:
            For each field of the result: its name, the source field or `None`, and the target type.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> _StructCast(CastKernel())._pairs(struct, DType("Date"))
            [('a', Int8, Date)]
        """
        if target.name != "Struct":
            return [(name, field, target) for name, field in source.fields()]
        fields = dict(source.fields())
        return [(name, fields.get(name), other) for name, other in target.fields()]

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of one struct value.

        Args:
            value: The struct value, a dictionary.
            source: The struct type.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int64")),),))
            >>> _StructCast(CastKernel()).outcome({"a": 300}, struct, DType("Int8"))
            <Outcome.LOST: 'lost'>
            >>> wider = DType("Struct", ((("a", DType("Int8")), ("b", DType("Int8"))),))
            >>> _StructCast(CastKernel()).outcome({"a": 1}, struct, wider)
            <Outcome.KEPT: 'kept'>
        """
        fields = value if isinstance(value, dict) else {}
        if target.name == "String":
            has_null = any(fields.get(name) is None for name, _ in source.fields())
            return Outcome.LOST if has_null else Outcome.KEPT
        outcomes = []
        for name, field, other in self._pairs(source, target):
            if field is not None:
                outcomes.append(self._kernel.outcome(fields.get(name), field, other))
        return Outcome.worst(outcomes)


class _TemporalCast:
    """Cast a `Date`, a `Datetime`, a `Duration` or a `Time`.

    Each temporal kind casts to the numbers through its physical integer,
    and to a short list of other kinds. A `Duration` does not cast to text.
    A cast to a finer unit multiplies, and a value that overflows becomes
    null.

    Example:
        >>> _TemporalCast().output(DType("Datetime", ("ms", None)), DType("Time"))
        Time
    """

    TARGETS: ClassVar[dict[str, frozenset[str]]] = {
        "Date": frozenset({"String", "Datetime", "Date"}),
        "Datetime": frozenset({"String", "Datetime", "Date", "Time"}),
        "Duration": frozenset({"Duration"}),
        "Time": frozenset({"String", "Time", "Duration"}),
    }

    def output(self, source: DType, target: DType) -> DType:
        """Give the output type.

        Args:
            source: The temporal type.
            target: The type of the cast.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The kind has no cast to the target.

        Example:
            >>> span = DType("Duration", ("us",))
            >>> _TemporalCast().output(span, DType("String"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if Is.primitive_numeric(target) or target.name in self.TARGETS[source.name]:
            return target
        message = _NoCast().message(source, target)
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of one value.

        A cast to a finer unit of time multiplies the count, and a count that
        leaves the range of `Int64` becomes null.

        Args:
            value: The value.
            source: The temporal type.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> nano = DType("Datetime", ("ns", None))
            >>> _TemporalCast().outcome(date(9999, 12, 31), DType("Date"), nano)
            <Outcome.LOST: 'lost'>
        """
        physical = PhysicalValue().of(value, source)
        if Is.primitive_numeric(target):
            return ArrowValues().outcome(physical, source.physical(), target)
        factor = self._factor(source, target)
        number = physical if isinstance(physical, int) else 0
        in_range = IntegerRange().holds("Int64", number * factor)
        return Outcome.KEPT if in_range else Outcome.LOST

    def _factor(self, source: DType, target: DType) -> int:
        """Give the factor that a cast to a finer unit multiplies by, or 1.

        Args:
            source: The temporal type.
            target: The type of the cast.

        Returns:
            The factor.

        Example:
            >>> _TemporalCast()._factor(DType("Date"), DType("Datetime", ("us", None)))
            86400000000
        """
        names = (source.name, target.name)
        if names == ("Date", "Datetime"):
            return TimeUnit.per_day(Parameter().unit(target))
        if names in {("Datetime", "Datetime"), ("Duration", "Duration")}:
            to_unit = TimeUnit.PER_SECOND[Parameter().unit(target)]
            from_unit = TimeUnit.PER_SECOND[Parameter().unit(source)]
            return max(to_unit // from_unit, 1)
        return 1


class _DecimalCast:
    """Cast a `Decimal`.

    A decimal casts to a decimal by a change of scale that rounds half to even,
    and a value that then exceeds the precision becomes null. It casts to a
    number, to text and to `Boolean` through the arrow kernel. A cast to an
    integer rounds half to even.

    Example:
        >>> _DecimalCast().output(DType("Decimal", (10, 2)), DType("Boolean"))
        Boolean
    """

    def output(self, source: DType, target: DType) -> DType:
        """Give the output type.

        Args:
            source: The decimal type.
            target: The type of the cast.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The target is not a number, text or `Boolean`.

        Example:
            >>> decimal = DType("Decimal", (10, 2))
            >>> _DecimalCast().output(decimal, DType("Date"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        flat = Is.primitive_numeric(target) or target.name in {"String", "Boolean", "Decimal"}
        if not flat:
            message = _NoCast().message(source, target)
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return target

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of one value.

        Args:
            value: The decimal value.
            source: The decimal type.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> wide, narrow = DType("Decimal", (10, 2)), DType("Decimal", (3, 0))
            >>> _DecimalCast().outcome(Decimal("999.50"), wide, narrow)
            <Outcome.LOST: 'lost'>
        """
        if target.name != "Decimal" and not Is.integer(target):
            return Outcome.KEPT
        digits = PhysicalValue().of(value, source)
        number = Decimal(digits if isinstance(digits, int) else 0)
        shift = -Parameter().integer(source, 1)
        if target.name == "Decimal":
            shift += Parameter().integer(target, 1)
        with localcontext() as context:
            context.prec = 200
            rounded = int(number.scaleb(shift).to_integral_value(rounding=ROUND_HALF_EVEN))
        if target.name == "Decimal":
            fits = Decimals().fits(rounded, Parameter().integer(target, 0))
        else:
            fits = IntegerRange().holds(target.name, rounded)
        return Outcome.KEPT if fits else Outcome.LOST


class _CategoricalCast:
    """Cast a `Categorical` or an `Enum`.

    Categories cast to text, to an `Enum`, to a `Categorical`, and to an
    integer through their ids. A category that an `Enum` target lacks
    becomes null. They do not cast to a float.

    Example:
        >>> _CategoricalCast().output(DType("Enum", (("a",),)), DType("UInt8"))
        UInt8
    """

    TARGETS: ClassVar[frozenset[str]] = Is.CATEGORIES | {"String"}

    def output(self, source: DType, target: DType) -> DType:
        """Give the output type.

        Args:
            source: The `Categorical` or `Enum` type.
            target: The type of the cast.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The target is not text, categories or an integer.

        Example:
            >>> _CategoricalCast().output(DType("Categorical"), DType("Float64"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        if source.same(target):
            return source
        if target.name in self.TARGETS or Is.integer(target):
            return target
        message = f"cannot cast categorical types to {target!r}"
        raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def outcome(self, value: object, source: DType, target: DType) -> Outcome:
        """Give the outcome of one value.

        The id of an `Enum` value is its index. The id of a `Categorical`
        value belongs to the process (see `ProcessCategories`).

        Args:
            value: The category, as text.
            source: The `Categorical` or `Enum` type.
            target: The type of the cast.

        Returns:
            The outcome.

        Example:
            >>> _CategoricalCast().outcome("c", DType("Categorical"), DType("Enum", (("a", "b"),)))
            <Outcome.LOST: 'lost'>
        """
        if target.name == "Enum" and not source.same(target):
            return Outcome.KEPT if value in Parameter().categories(target) else Outcome.LOST
        if not Is.integer(target):
            return Outcome.KEPT
        if source.name == "Enum":
            index = PhysicalValue().of(value, source)
            return ArrowValues().outcome(index, source.physical(), target)
        wide = ProcessCategories().holds_every_id(target.name)
        return Outcome.KEPT if wide else Outcome.PROCESS


class SeriesCast:
    """Give the type of a run of `Series.cast` on given values, or the refusal of polars.

    The run takes the shortcuts of the cast of a series first. A cast that
    changes nothing returns the series, and a series of only nulls casts to any
    type with no kernel. Otherwise the kernel must exist for the pair
    (`CastKernel`), and no value can make it raise. A list cast to an array
    checks the width of each list before it casts the inner values, so a list
    of another width raises first. A strict cast also raises where it loses a
    value that its check sees.

    Example:
        >>> SeriesCast(DType("Date"), DType("Boolean"), strict=False).output([None])
        Boolean
        >>> SeriesCast(DType("Int64"), DType("Int8"), strict=False).output([1, 300])
        Int8
    """

    def __init__(self, source: DType, target: DType, *, strict: bool) -> None:
        """Keep the cast.

        Args:
            source: The type of the series.
            target: The type of the cast.
            strict: Whether the cast is strict.

        Example:
            >>> SeriesCast(DType("Int64"), DType("Int8"), strict=True)._strict
            True
        """
        self._source = source
        self._target = target
        self._strict = strict
        self._kernel = CastKernel()

    def output(self, values: Sequence[object]) -> DType:
        """Give the type of the run.

        It raises `PolarsRefusal` where the kernel has no cast for the pair.

        Args:
            values: The values of the series, `None` for a null.

        Returns:
            The type.

        Raises:
            PolarsRefusal: A value makes the kernel raise, or a strict cast loses a value.
            UndecidedError: The answer depends on the categories of the process.

        Example:
            >>> SeriesCast(DType("String"), DType("Int8"), strict=True).output(["a"])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        shortcut = self._shortcut(values)
        if shortcut is not None:
            return shortcut
        descended = self._descended(values)
        if descended is not None:
            return descended
        self._check_widths(values)
        output = self._kernel.output(self._source, self._target)
        worst = self._worst(values)
        if worst is Outcome.RAISES:
            message = f"a value makes the cast of {self._source!r} to {self._target!r} fail"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if self._strict and worst is Outcome.LOST:
            message = f"conversion from {self._source!r} to {self._target!r} failed"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if self._strict and worst is Outcome.PROCESS:
            question = f"the strict cast of {self._source!r} to {self._target!r}"
            raise UndecidedError(Undecided.PROCESS, question)
        return output

    def _check_widths(self, values: Sequence[object]) -> None:
        """Refuse a list cast to an array where a list has another width.

        The arrow kernel checks the widths before it casts the inner values,
        so this refusal comes before the refusal of an inner cast. A null
        list has no width.

        Args:
            values: The values of the series.

        Raises:
            PolarsRefusal: A list does not have the width of the array.

        Example:
            >>> pair = DType("Array", (DType("Int64"), 2))
            >>> cast = SeriesCast(DType("List", (DType("Int64"),)), pair, strict=False)
            >>> cast._check_widths([[1, 2], None]) is None
            True
        """
        target = self._kernel.target(self._source, self._target)
        if (self._source.name, target.name) != ("List", "Array"):
            return
        width = Parameter().integer(target, 1)
        lists = [value for value in values if isinstance(value, list)]
        if any(len(value) != width for value in lists):
            message = f"not all elements have the specified width {width}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def _shortcut(self, values: Sequence[object]) -> DType | None:
        """Give the type of a run that reads no data, or `None` where the kernel must run.

        Args:
            values: The values of the series.

        Returns:
            The source for a cast that changes nothing, the target for a series of only nulls.

        Example:
            >>> SeriesCast(DType("Binary"), DType("Date"), strict=True)._shortcut([None, None])
            Date
        """
        if self._kernel.clones(self._source, self._target):
            return self._source
        if all(value is None for value in values):
            return self._kernel.target(self._source, self._target)
        return None

    def _descended(self, values: Sequence[object]) -> DType | None:
        """Give the type of a nested cast that casts each inner series as a series, or `None`.

        The cast of each inner series takes the shortcuts again, so inner
        values that are all null cast to any type. A null list holds no inner
        value, and a null struct holds a null in each field.

        Args:
            values: The values of the series.

        Returns:
            The type, or `None` where the cast does not cast an inner series (`CastKernel.inner`).

        Example:
            >>> listed = DType("List", (DType("Binary"),))
            >>> cast = SeriesCast(listed, DType("List", (DType("Date"),)), strict=True)
            >>> cast._descended([[None], None, []])
            List(Date)
        """
        pairs = self._kernel.inner(self._source, self._target)
        if pairs is None:
            return None
        if self._source.name == "Struct":
            fields = [
                (name, self._cast(field, other, values, name)) for name, field, other in pairs
            ]
            return DType("Struct", (tuple(fields),))
        _, inner, other = pairs[0]
        cast = self._cast(inner, other, values, None)
        if self._source.name == "List":
            return DType("List", (cast,))
        return DType("Array", (cast, Parameter().integer(self._source, 1)))

    def _cast(
        self,
        source: DType | None,
        target: DType,
        values: Sequence[object],
        field: str | None,
    ) -> DType:
        """Give the type of the cast of one inner series: the values of the lists, or one field.

        Args:
            source: The inner type, or `None` for a field that the source lacks.
            target: The type of the cast of the inner series.
            values: The values of the outer series.
            field: The name of the field, or `None` for the values of the lists.

        Returns:
            The type.

        Example:
            >>> cast = SeriesCast(DType("Null"), DType("Null"), strict=True)
            >>> cast._cast(DType("Int64"), DType("Int8"), [{"a": None}], "a")
            Int8
        """
        if source is None:
            return target
        inner = self._inner(values, field)
        return SeriesCast(source, target, strict=self._strict).output(inner)

    def _inner(self, values: Sequence[object], field: str | None) -> list[object]:
        """Give the values of one inner series: the values of the lists, or one field of each.

        A null list holds no inner value, and a null struct holds a null in each field.

        Args:
            values: The values of the outer series.
            field: The name of the field, or `None` for the values of the lists.

        Returns:
            The values.

        Example:
            >>> cast = SeriesCast(DType("Null"), DType("Null"), strict=True)
            >>> cast._inner([[1], None, [2, 3]], None)
            [1, 2, 3]
        """
        if field is not None:
            return [value.get(field) if isinstance(value, dict) else None for value in values]
        lists = [value for value in values if isinstance(value, list)]
        return [element for value in lists for element in value]

    def _worst(self, values: Sequence[object]) -> Outcome:
        """Give the outcome that decides the cast of the values.

        Args:
            values: The values of the series.

        Returns:
            The outcome.

        Example:
            >>> SeriesCast(DType("Int64"), DType("Int8"), strict=True)._worst([1, 300])
            <Outcome.LOST: 'lost'>
        """
        kernel, source, target = self._kernel, self._source, self._target
        return Outcome.worst(kernel.outcome(value, source, target) for value in values)


class CastAbility:
    """Say whether polars marks a cast as possible before it tries the cast.

    Some kernels check a pair of types against this coarse table before a
    strict cast, because a strict cast takes pairs that the kernel cannot
    use, such as text to a struct. The table answers `True`, `False`, or
    `None` where it has no entry, and such a kernel needs `True`. A type
    casts to itself, a number to a number, and `Null` to every type.

    Example:
        >>> CastAbility().of(DType("Int8"), DType("Float32"))
        True
        >>> CastAbility().of(DType("String"), DType("Struct", ((("a", DType("Int8")),),))) is None
        True
    """

    CATEGORIES: ClassVar[frozenset[str]] = frozenset({"Categorical", "Enum"})
    BOOLEAN_PEERS: ClassVar[frozenset[str]] = frozenset({"Decimal", "String", "Binary"})

    def of(self, source: DType, target: DType) -> bool | None:
        """Give the answer of the table for a pair of types.

        Args:
            source: The type of the values.
            target: The type of the cast.

        Returns:
            Whether the cast is possible, or `None` where the table has no entry.

        Example:
            >>> CastAbility().of(DType("Null"), DType("Date"))
            True
            >>> CastAbility().of(DType("Date"), DType("Time")) is None
            True
            >>> CastAbility().of(DType("Categorical"), DType("String"))
            True
            >>> CastAbility().of(DType("Boolean"), DType("Int8"))
            True
            >>> CastAbility().of(DType("List", (DType("Int8"),)), DType("List", (DType("Int16"),)))
            True
        """
        numbers = Is.primitive_numeric(source) and Is.primitive_numeric(target)
        if numbers or source.same(target) or source.name == "Null":
            return True
        if source.name in self.CATEGORIES or target.name in self.CATEGORIES:
            return self._categories(source, target)
        if "Boolean" in {source.name, target.name}:
            return self._boolean(source, target)
        if source.name == target.name and source.name in {"List", "Array", "Struct"}:
            return self._nested(source, target)
        return None

    def _categories(self, source: DType, target: DType) -> bool | None:
        """Give the answer for a pair where one side is a category.

        Args:
            source: The type of the values.
            target: The type of the cast.

        Returns:
            `False` beside `Binary`, `True` beside `String`, and `None` otherwise.

        Example:
            >>> CastAbility()._categories(DType("Categorical"), DType("Binary"))
            False
            >>> CastAbility()._categories(DType("String"), DType("Categorical"))
            True
            >>> CastAbility()._categories(DType("Categorical"), DType("Date")) is None
            True
        """
        other = target if source.name in self.CATEGORIES else source
        if other.name == "Binary":
            return False
        if other.name == "String":
            return True
        return None

    def _boolean(self, source: DType, target: DType) -> bool:
        """Give the answer for a pair where one side is `Boolean`.

        Args:
            source: The type of the values.
            target: The type of the cast.

        Returns:
            Whether the other side is a number, a `Decimal`, text or binary data.

        Example:
            >>> CastAbility()._boolean(DType("Boolean"), DType("Date"))
            False
        """
        other = target if source.name == "Boolean" else source
        return Is.primitive_numeric(other) or other.name in self.BOOLEAN_PEERS

    def _nested(self, source: DType, target: DType) -> bool | None:
        """Give the answer for two lists, two arrays or two structs, from their inner types.

        Args:
            source: The type of the values.
            target: The type of the cast, of the same kind.

        Returns:
            The answer of the inner types, or `False` where the widths or the
            counts of the fields differ.

        Example:
            >>> array = DType("Array", (DType("Int8"), 2))
            >>> CastAbility()._nested(array, DType("Array", (DType("Int8"), 3)))
            False
            >>> CastAbility()._nested(array, DType("Array", (DType("Int16"), 2)))
            True
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> CastAbility()._nested(struct, DType("Struct", ((("b", DType("Date")),),))) is None
            True
        """
        if source.name == "List":
            return self.of(source.inner(), target.inner())
        if source.name == "Array":
            if source.args[1] != target.args[1]:
                return False
            return self.of(source.inner(), target.inner())
        return self._fields(source, target)

    def _fields(self, source: DType, target: DType) -> bool | None:
        """Give the answer for two structs, field by field in order.

        Args:
            source: The struct of the values.
            target: The struct of the cast.

        Returns:
            `True` for a source with no field, `False` where the counts of the
            fields differ, and otherwise the first answer of a pair of fields
            that is not `True`.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> pair = DType("Struct", ((("a", DType("Int8")), ("b", DType("Int8"))),))
            >>> CastAbility()._fields(struct, pair)
            False
            >>> CastAbility()._fields(DType("Struct", ((),)), struct)
            True
            >>> CastAbility()._fields(struct, DType("Struct", ((("a", DType("Int16")),),)))
            True
            >>> CastAbility()._fields(DType("Struct", ((("a", DType("Date")),),)), struct) is None
            True
        """
        mine, theirs = source.fields(), target.fields()
        if not mine:
            return True
        if len(mine) != len(theirs):
            return False
        for (_, one), (_, other) in zip(mine, theirs, strict=True):
            answer = self.of(one, other)
            if answer is not True:
                return answer
        return True
