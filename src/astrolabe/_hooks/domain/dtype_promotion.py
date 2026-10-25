"""Find the data type that two columns of different data types become when polars merges them."""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

import polars as pl
from mypy.types import Instance, LiteralType, get_proper_type

from astrolabe._hooks.checked import checked_cast, checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe._rules.dtypes import DType
from astrolabe._rules.supertype import Supertype
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe.frame import Decimal
from astrolabe.frame import Struct as StructMarker

if TYPE_CHECKING:
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup
    from astrolabe._hooks.domain.unchecked import UncheckedCompatibility


class DTypePromotion:
    """Give the common data type of two columns that polars puts into one column.

    Some operations merge two columns of different data types into one. Examples are a
    `pl.concat` with `how="diagonal_relaxed"` and the `value` column of `unpivot`. Polars then
    casts both to a common data type, its supertype. Code that types such an operation asks
    this class for it. One class holds the rules, so every operation that merges columns agrees.

    These are the rules that it knows:

    - `Unchecked` beside any data type gives `Unchecked`, at any depth, because the unknown side
      decides the supertype (see `UncheckedCompatibility.joined`).
    - Two numbers give the supertype of polars for the pair (`Supertype`): `Int8` beside
      `UInt16` gives `Int32`, and `Int64` beside `Float32` gives `Float64`.
    - `Boolean` beside a number gives the number.
    - Two `Decimal` types give a `Decimal` of the larger precision and the larger scale.
    - `String` beside a number, `Boolean`, `Decimal`, `Datetime` or `Enum` gives `String`.
    - Two structs give the union of their fields, each field of both at its common data type.

    For any other pair the answer is None. The code that asks then treats the pair as one it
    cannot type.

    It sits in the plugin layer, and it reads and builds mypy instance types.
    """

    _NUMERIC: Final[frozenset[Target]] = frozenset(
        Target.of(dtype)
        for dtype in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.Int128,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.UInt128,
            pl.Float16,
            pl.Float32,
            pl.Float64,
        )
    )
    _BOOLEAN_TARGET: Final[Target] = Target.of(pl.Boolean)
    _STRUCT: Final[Target] = Target.of(StructMarker)
    _DECIMAL: Final[Target] = Target.of(Decimal)
    _LARGEST: Final[int] = 38
    _STRING_TARGET: Final[Target] = Target.of(pl.String)
    _STRING_ABSORBABLE_TARGETS: Final[tuple[Target, ...]] = (
        Target.of(pl.Decimal),
        Target.of(pl.Datetime),
        Target.of(pl.Enum),
    )

    def __init__(self, unchecked: UncheckedCompatibility, plugin: PluginLookup) -> None:
        """Keep the `Unchecked` rule and the class lookup that the promotion needs.

        Args:
            unchecked: The rule for a pair where one side is `Unchecked`. It runs before every
                other rule.
            plugin: The mypy plugin, which finds the class of a result that is neither input,
                for example `Int64` for `Int32` beside `UInt32`.
        """
        self._unchecked = unchecked
        self._types = FreshTypeInfo(plugin)

    def promoted(self, a: Instance, b: Instance) -> Instance | None:
        """Give the common data type of two data types.

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            The common data type by the rules of this class, or None for a pair that the rules
            do not cover.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> promotion.promoted(int16_dtype(fixture), int64_dtype(fixture))
            polars.datatypes.classes.Int64
            >>> promotion.promoted(int64_dtype(fixture), str_dtype(fixture))
            polars.datatypes.classes.String
            >>> dtype = int64_dtype(fixture)
            >>> promotion.promoted(dtype, dtype)
            polars.datatypes.classes.Int64
            >>> promotion.promoted(unchecked_instance(fixture), int64_dtype(fixture))
            astrolabe.frame.Unchecked
        """
        unchecked_result = self._unchecked.joined(a, b)
        if unchecked_result is not None:
            return checked_cast(unchecked_result, Instance)
        widened = self._widened(a, b)
        if widened is not None:
            return widened
        decimal = self._decimal(a, b)
        if decimal is not None:
            return decimal
        structs = self._structs(a, b)
        if structs is not None:
            return structs
        return self._absorbed(a, b)

    @propagates(None)
    def _structs(self, a: Instance, b: Instance) -> Instance | None:
        """Give the common data type of two structs, as polars joins their fields.

        The result holds the fields of the longer struct, or of `a` where both
        have as many, in their order, then each field of the other that it
        lacks. A field of both takes the common data type of its two types.

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            The struct, or None where a side is not a struct with a list of
            fields, or where two types of one field have no common data type.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> x = column_ref(fixture, "x", int16_dtype(fixture))
            >>> wide = column_ref(fixture, "x", int64_dtype(fixture))
            >>> y = column_ref(fixture, "y", str_dtype(fixture))
            >>> narrow, both = struct_instance(fixture, x), struct_instance(fixture, wide, y)
            >>> joined = promotion._structs(narrow, both)
            >>> [(field.name, field.dtype) for field in ColumnRef.from_argument(joined, 0)]
            [('x', polars.datatypes.classes.Int64), ('y', polars.datatypes.classes.String)]
        """
        not_falsy(self._STRUCT.matches(a) and self._STRUCT.matches(b))
        fields_a = not_none(ColumnRef.from_argument(a, 0))
        fields_b = not_none(ColumnRef.from_argument(b, 0))
        longer = a if len(fields_a) >= len(fields_b) else b
        longer_fields, other_fields = (fields_a, fields_b) if longer is a else (fields_b, fields_a)
        merged = {field.name: field.as_instance() for field in longer_fields}
        for field in other_fields:
            kept = merged.get(field.name)
            if kept is None:
                merged[field.name] = field.as_instance()
                continue
            dtype = not_none(self.promoted_dtypes(kept.args[1], field.dtype))
            merged[field.name] = kept.copy_modified(args=[kept.args[0], dtype])
        items = not_none(ColumnRef.tuple_argument(longer, 0))
        return longer.copy_modified(args=[items.copy_modified(items=list(merged.values()))])

    def _decimal(self, a: Instance, b: Instance) -> Instance | None:
        """Give the common data type of two `Decimal` types: the larger precision and scale.

        A precision of `None` is the largest precision of polars, 38.

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            The `Decimal`, with the parameter of the side that holds the larger
            one. None if one side is not a `Decimal` with its parameters.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> promotion._decimal(decimal_dtype(fixture, 10, 2), decimal_dtype(fixture, 5, 3))
            astrolabe.frame.Decimal[Literal[10], Literal[3]]
        """
        if not (self._DECIMAL.matches(a) and self._DECIMAL.matches(b)):
            return None
        precision = max((a, b), key=lambda side: self._digits(side.args[0])).args[0]
        scale = max((a, b), key=lambda side: self._digits(side.args[1])).args[1]
        return a.copy_modified(args=[precision, scale])

    def _digits(self, parameter: MypyType) -> int:
        """Give the number that a parameter of a `Decimal` holds, where `None` is 38.

        Args:
            parameter: The type argument.

        Returns:
            The number.

        Example:
            >>> from mypy.types import NoneType
            >>> fixture = new_type_fixture()
            >>> DTypePromotion(unchecked_compatibility(), new_plugin(fixture))._digits(NoneType())
            38
        """
        proper = get_proper_type(parameter)
        if isinstance(proper, LiteralType) and isinstance(proper.value, int):
            return proper.value
        return self._LARGEST  # pragma: no cover -- type analysis writes each parameter

    def promoted_dtypes(self, a: MypyType, b: MypyType) -> Instance | None:
        """Give the common data type of two types that are not yet known to be instances.

        A data type read from a schema is a general mypy type. It can be a type variable, for
        example. Use this method for such a type, and `promoted` for two instances.

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            The common data type, or None if one side is not an instance or the rules do not
            cover the pair.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> promotion.promoted_dtypes(int16_dtype(fixture), int64_dtype(fixture))
            polars.datatypes.classes.Int64
            >>> promotion.promoted_dtypes(int64_dtype(fixture), plain_typevar(fixture)) is None
            True
        """
        if not (isinstance(a, Instance) and isinstance(b, Instance)):
            return None
        return self.promoted(a, b)

    def compatible(self, a: MypyType, b: MypyType) -> bool:
        """Tell whether two data types have a common data type that this class knows.

        An operation that merges columns can use this to decide whether to report an error
        before it builds its result.

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            True if `promoted_dtypes` gives a result for the pair.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> promotion.compatible(int16_dtype(fixture), int64_dtype(fixture))
            True
            >>> promotion.compatible(int64_dtype(fixture), duration_dtype(fixture))
            False
            >>> promotion.compatible(int64_dtype(fixture), plain_typevar(fixture))
            False
        """
        return self.promoted_dtypes(a, b) is not None

    def _widened(self, a: Instance, b: Instance) -> Instance | None:
        """Give the common data type of two numbers.

        polars gives two numbers the supertype that holds every value of both:
        the wider of two integers of one sign, a signed integer wide enough for
        an unsigned one, or a float that holds an integer (`Supertype`).

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            A new instance of the common data type, with no type arguments. None if one side
            is not a number.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> promotion._widened(uint32_dtype(fixture), uint64_dtype(fixture))
            polars.datatypes.classes.UInt64
            >>> promotion._widened(int64_dtype(fixture), float32_dtype(fixture))
            polars.datatypes.classes.Float64
            >>> promotion._widened(int64_dtype(fixture), str_dtype(fixture)) is None
            True
        """
        if not (self._is_numeric(a) and self._is_numeric(b)):
            return None
        joined = Supertype().of(DType(self._kind(a)), DType(self._kind(b)))
        target = Target.of(getattr(pl, checked_not_none(joined).name))
        # mypy loads every polars data type class with polars. The plugin adds polars to the
        # dependencies of every build, so the lookup always finds the class.
        info = checked_not_none(self._types.resolved(target.fullname))
        return Instance(info, [])

    def _kind(self, dtype: Instance) -> str:
        """Give the name of the polars class of a data type, for example `Int64`.

        Args:
            dtype: The data type.

        Returns:
            The last part of the full name of its class.
        """
        return dtype.type.fullname.rsplit(".", maxsplit=1)[-1]

    def _absorbed(self, a: Instance, b: Instance) -> Instance | None:
        """Give the side that takes in the other, for a pair from two different families.

        The order of the pair does not matter.

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            The number for `Boolean` beside a number. `String` for `String` beside a data type
            that `_string_absorbable` accepts. None for any other pair.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> promotion._absorbed(boolean_dtype(fixture), int64_dtype(fixture))
            polars.datatypes.classes.Int64
            >>> promotion._absorbed(int64_dtype(fixture), boolean_dtype(fixture))
            polars.datatypes.classes.Int64
            >>> promotion._absorbed(str_dtype(fixture), int_dtype(fixture))
            polars.datatypes.classes.String
            >>> promotion._absorbed(int_dtype(fixture), str_dtype(fixture))
            polars.datatypes.classes.String
            >>> promotion._absorbed(int_dtype(fixture), int64_dtype(fixture)) is None
            True
        """
        if self._bool_numeric_pair(a, b):
            return b
        if self._bool_numeric_pair(b, a):
            return a
        if self._string_absorbs_pair(a, b):
            return a
        if self._string_absorbs_pair(b, a):
            return b
        return None

    def _bool_numeric_pair(self, boolean_side: Instance, numeric_side: Instance) -> bool:
        """Tell whether the first data type is `Boolean` and the second is a number.

        Args:
            boolean_side: The data type that must be `Boolean`.
            numeric_side: The data type that must be a number.

        Returns:
            True if both hold.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> promotion._bool_numeric_pair(boolean_dtype(fixture), int64_dtype(fixture))
            True
            >>> promotion._bool_numeric_pair(int64_dtype(fixture), int64_dtype(fixture))
            False
        """
        return self._BOOLEAN_TARGET.matches(boolean_side) and self._is_numeric(numeric_side)

    def _string_absorbs_pair(self, string_side: Instance, other_side: Instance) -> bool:
        """Tell whether the first data type is `String` and polars turns the second into a string.

        Args:
            string_side: The data type that must be `String`.
            other_side: The data type that `_string_absorbable` tests.

        Returns:
            True if both hold.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> promotion._string_absorbs_pair(str_dtype(fixture), int_dtype(fixture))
            True
            >>> promotion._string_absorbs_pair(int_dtype(fixture), int64_dtype(fixture))
            False
        """
        if not self._STRING_TARGET.matches(string_side):
            return False
        return self._string_absorbable(other_side)

    def _string_absorbable(self, dtype: Instance) -> bool:
        """Tell whether polars turns a column of this data type into `String` when it merges.

        A number, `Boolean`, `Decimal`, `Datetime` or `Enum` column becomes `String` beside a
        `String` column. `Duration` does not: polars raises `InvalidOperationError` when it
        casts a `Duration` column to `String`.

        Args:
            dtype: The data type to test.

        Returns:
            True for a number, `Boolean`, `Decimal`, `Datetime` or `Enum`. The markers of this
            project for the last three are subclasses of the polars classes, so they count too.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> promotion._string_absorbable(decimal_dtype(fixture, 10, 2))
            True
            >>> promotion._string_absorbable(boolean_dtype(fixture))
            True
            >>> promotion._string_absorbable(duration_dtype(fixture))
            False
            >>> promotion._string_absorbable(str_dtype(fixture))
            False
        """
        if self._is_numeric(dtype) or self._BOOLEAN_TARGET.matches(dtype):
            return True
        return any(target.is_base_of(dtype) for target in self._STRING_ABSORBABLE_TARGETS)

    def _is_numeric(self, dtype: Instance) -> bool:
        """Tell whether a data type is a signed integer, an unsigned integer or a float.

        Args:
            dtype: The data type to test.

        Returns:
            True if the data type is in one of those three families.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> promotion._is_numeric(int_dtype(fixture))
            True
            >>> promotion._is_numeric(str_dtype(fixture))
            False
        """
        return Target(dtype.type.fullname) in self._NUMERIC
