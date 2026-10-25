"""Give the data type that a reducer such as `sum` or `mean` produces from a column.

A reducer turns a column into one value, and the data type of that value depends on the reducer
and on the input. `sum` of an `Int8` column gives `Int64`, and `mean` of an integer column gives
`Float64`. The whole-frame reducers, such as `df.sum()`, and the aggregate of `pivot` share the
rules in this module.
"""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

import polars as pl
from mypy.nodes import TypeInfo
from mypy.types import Instance, LiteralType, NoneType, get_proper_type

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe.frame import Datetime as _DatetimeMarker
from astrolabe.frame import Unchecked as _UncheckedMarker

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.dtype_families import DtypeFamilies
    from astrolabe._hooks.domain.frame import PluginLookup


class ReducerPromotion:
    """Answer the questions about an input data type that the rules for reducers ask.

    Each rule of a reducer turns on one fact about the input: is it `Boolean`, a narrow
    integer, `Float32` or `Unchecked`? This class answers each question from the full name of
    the class of the data type. It builds no type, so code that has its own way to build the
    result can still share these tests.

    Example:
        >>> fixture = new_type_fixture()
        >>> promotion = ReducerPromotion()
        >>> promotion.is_boolean(boolean_dtype(fixture).type.fullname)
        True
        >>> promotion.is_boolean(int_dtype(fixture).type.fullname)
        False
    """

    _BOOLEAN_TARGET: Final[Target] = Target.of(pl.Boolean)
    _FLOAT32_TARGET: Final[Target] = Target.of(pl.Float32)
    _UNCHECKED_TARGET: Final[Target] = Target.of(_UncheckedMarker)
    _SUM_UPCAST_TO_INT64: Final[frozenset[str]] = frozenset({
        Target.of(pl.Int8).fullname,
        Target.of(pl.Int16).fullname,
        Target.of(pl.UInt8).fullname,
        Target.of(pl.UInt16).fullname,
    })

    def is_boolean(self, source_fullname: str) -> bool:
        """Tell whether the input data type is `Boolean`.

        `sum()` of a `Boolean` column counts the true values and gives `UInt32`.

        Args:
            source_fullname: The mypy full name of the input data type.

        Returns:
            True for `Boolean`.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = ReducerPromotion()
            >>> promotion.is_boolean(boolean_dtype(fixture).type.fullname)
            True
            >>> promotion.is_boolean(int_dtype(fixture).type.fullname)
            False
        """
        return source_fullname == self._BOOLEAN_TARGET.fullname

    def is_narrow_int(self, source_fullname: str) -> bool:
        """Tell whether `sum()` widens the input data type to `Int64`.

        Args:
            source_fullname: The mypy full name of the input data type.

        Returns:
            True for `Int8`, `Int16`, `UInt8` and `UInt16`.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = ReducerPromotion()
            >>> promotion.is_narrow_int(int8_dtype(fixture).type.fullname)
            True
            >>> promotion.is_narrow_int(int_dtype(fixture).type.fullname)
            False
        """
        return source_fullname in self._SUM_UPCAST_TO_INT64

    def is_float32(self, source_fullname: str) -> bool:
        """Tell whether the input data type is `Float32`.

        `mean()`, `median()`, `std()` and `var()` give `Float64` for every number and for
        `Boolean`, except `Float32`, which stays `Float32`.

        Args:
            source_fullname: The mypy full name of the input data type.

        Returns:
            True for `Float32`.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = ReducerPromotion()
            >>> promotion.is_float32(float32_dtype(fixture).type.fullname)
            True
            >>> promotion.is_float32(float64_dtype(fixture).type.fullname)
            False
        """
        return source_fullname == self._FLOAT32_TARGET.fullname

    def is_unchecked(self, source_fullname: str) -> bool:
        """Tell whether the input data type is `Unchecked`.

        `Unchecked` means that the data type is not known. `mean()` and the similar reducers
        must then not give `Float64`, because the column can hold something other than numbers.

        Args:
            source_fullname: The mypy full name of the input data type.

        Returns:
            True for `Unchecked`.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = ReducerPromotion()
            >>> promotion.is_unchecked(unchecked_instance(fixture).type.fullname)
            True
            >>> promotion.is_unchecked(int_dtype(fixture).type.fullname)
            False
        """
        return source_fullname == self._UNCHECKED_TARGET.fullname


class ReducerDtype:
    """Give the data type that one reducer produces from one input column.

    Code that types `df.sum()`, `df.mean()` and the other whole-frame reducers asks this class
    for each column. So does code that types the `aggregate_function` of `pivot`. One class
    holds the rules, so the two agree. The rules are:

    - `min`, `max`, `first` and `last` keep the input data type.
    - `sum` gives `UInt32` for `Boolean`, `Int64` for an integer narrower than 32 bits, and a
      `Decimal` with precision 38 for a `Decimal`. It keeps any other accepted data type.
    - `mean`, `median`, `std`, `var` and `quantile` give `Float64` for a number, `Boolean`,
      `Decimal` or `Int128`. They keep `Float32`, `Unchecked` and every other data type. Two
      cases differ: `quantile` keeps `Boolean`, and `mean`, `median` and `quantile` turn a
      `Date` into a `Datetime` in microseconds.
    - `len` gives `UInt32`, whatever the input.

    `sum` is the one reducer that rejects an input. `source_family_matches` tells whether it
    accepts one, so that the code can report an error.

    Example:
        >>> fixture = new_type_fixture()
        >>> resolver = reducer_dtype(new_plugin(fixture))
        >>> resolver.resolve("sum", int8_dtype(fixture))
        polars.datatypes.classes.Int64
        >>> resolver.resolve("min", int8_dtype(fixture))
        polars.datatypes.classes.Int8
        >>> resolver.resolve("len", None)
        polars.datatypes.classes.UInt32
    """

    IDENTITY_REDUCERS: Final[frozenset[str]] = frozenset({"min", "max", "first", "last"})
    FLOAT_COERCE_REDUCERS: Final[frozenset[str]] = frozenset({
        "mean",
        "median",
        "std",
        "var",
        "quantile",
    })
    SUM_REDUCER: Final[str] = "sum"
    ALL_REDUCERS: Final[frozenset[str]] = IDENTITY_REDUCERS | FLOAT_COERCE_REDUCERS | {SUM_REDUCER}
    _UINT32_TARGET: Final[Target] = Target.of(pl.UInt32)
    _INT64_TARGET: Final[Target] = Target.of(pl.Int64)
    _FLOAT64_TARGET: Final[Target] = Target.of(pl.Float64)
    _BOOLEAN_TARGET: Final[Target] = Target.of(pl.Boolean)
    _DATE_TARGET: Final[Target] = Target.of(pl.Date)
    _DATETIME_MARKER: Final[Target] = Target.of(_DatetimeMarker)
    # mean(), median() and quantile() turn a Date into a Datetime in microseconds. std() and
    # var() keep the Date.
    _DATE_TO_DATETIME: Final[frozenset[str]] = frozenset({"mean", "median", "quantile"})
    # sum() also accepts these data types, which are not "numeric or Boolean". It keeps each
    # one, except Decimal, whose precision it widens. The test uses is_base_of and not matches,
    # because the markers of this project, such as Datetime[Unit, Zone], are subclasses of
    # these polars classes.
    _SUM_ACCEPTED_TARGETS: Final[frozenset[Target]] = frozenset({
        Target.of(pl.Int128),
        Target.of(pl.Decimal),
        Target.of(pl.Duration),
        Target.of(pl.Datetime),
        Target.of(pl.Enum),
        Target.of(pl.Categorical),
    })
    # mean(), median(), std() and var() give Float64 for these two data types as well. They
    # keep every other data type that is not a number or Boolean, such as String, List,
    # Duration, Datetime, Enum and Categorical.
    _FLOAT_COERCED_TARGETS: Final[frozenset[Target]] = frozenset({
        Target.of(pl.Decimal),
        Target.of(pl.Int128),
    })

    def __init__(self, plugin: PluginLookup, families: DtypeFamilies) -> None:
        """Keep the class lookup and the family tests that the rules need.

        Args:
            plugin: The mypy plugin, which finds the class of each result data type.
            families: The tests by family, and the widening of a `Decimal` for `sum`.
        """
        self._plugin = plugin
        self._promotion = ReducerPromotion()
        self._families = families
        self._unchecked = UncheckedCompatibility()
        self._family_checks: dict[str, Callable[[MypyType], bool]] = {
            self.SUM_REDUCER: self._is_unknown_or_valid_sum_source,
        }

    def source_family_matches(self, reducer: str, source_dtype: MypyType | None) -> bool:
        """Tell whether a reducer accepts a column of the given data type.

        Only `sum()` rejects some inputs. Polars raises `InvalidOperationError` for the `sum()`
        of a `String` or a `List` column, for example. Every other reducer accepts any column:
        `mean()`, `median()` and `std()` give `Float64` or keep the input data type, and
        `min()`, `max()`, `first()` and `last()` keep it.

        Args:
            reducer: The name of the reducer, or `"len"`.
            source_dtype: The data type of the input column, or None for `"len"`.

        Returns:
            True if the reducer has no rule about its input, or if `source_dtype` is None, or
            if the rule accepts the data type. A data type that is not an instance, such as a
            type variable, is not known to be wrong, so it passes. `Unchecked` passes too.

        Example:
            >>> fixture = new_type_fixture()
            >>> resolver = reducer_dtype(new_plugin(fixture))
            >>> resolver.source_family_matches("sum", str_dtype(fixture))
            False
            >>> resolver.source_family_matches("mean", str_dtype(fixture))
            True
        """
        check = self._family_checks.get(reducer)
        if source_dtype is None or check is None:
            return True
        return check(source_dtype)

    def _is_unknown_or_valid_sum_source(self, source_dtype: MypyType) -> bool:
        """Tell whether `sum()` accepts the input, or whether the input is not known.

        Args:
            source_dtype: The data type of the input column.

        Returns:
            True if the data type is not an instance, is `Unchecked`, is a number or `Boolean`,
            or is one of `_SUM_ACCEPTED_TARGETS` or a marker subclass of one.
        """
        proper = get_proper_type(source_dtype)
        if not isinstance(proper, Instance) or self._unchecked.matches(proper):
            return True
        if self._families.is_numeric_or_boolean(proper):
            return True
        return any(target.is_base_of(proper) for target in self._SUM_ACCEPTED_TARGETS)

    def resolve(self, reducer: str, source_dtype: MypyType | None) -> Instance | None:
        """Give the data type that a reducer produces from an input column.

        Call this only after `source_family_matches` accepts the input.

        Args:
            reducer: The name of the reducer, one of `ALL_REDUCERS`, or `"len"`.
            source_dtype: The data type of the input column, or None for `"len"`.

        Returns:
            The data type of the result. None if the input is not an instance, or if the
            plugin cannot find a class.

        Example:
            >>> fixture = new_type_fixture()
            >>> resolver = reducer_dtype(new_plugin(fixture))
            >>> resolver.resolve("mean", int_dtype(fixture))
            polars.datatypes.classes.Float64
        """
        if reducer == "len":
            return self._fresh(self._UINT32_TARGET.fullname)
        proper_source = get_proper_type(source_dtype)
        if not isinstance(proper_source, Instance):
            return None
        if reducer == self.SUM_REDUCER:
            return self._sum_dtype(proper_source)
        return self._coerced_dtype(reducer, proper_source)

    def _sum_dtype(self, proper_source: Instance) -> Instance | None:
        """Give the data type of the `sum()` of a column.

        Args:
            proper_source: The data type of the input column.

        Returns:
            `UInt32` for `Boolean`. `Int64` for an integer narrower than 32 bits. A `Decimal`
            with precision 38 for a `Decimal`. The input data type for any other input.
        """
        fullname = proper_source.type.fullname
        if self._promotion.is_boolean(fullname):
            return self._fresh(self._UINT32_TARGET.fullname)
        if self._promotion.is_narrow_int(fullname):
            return self._fresh(self._INT64_TARGET.fullname)
        widened = self._families.widened_decimal(proper_source)
        return widened if widened is not None else proper_source

    def _coerced_dtype(self, reducer: str, proper_source: Instance) -> Instance | None:
        """Give the result for every reducer other than `sum`.

        Args:
            reducer: The name of the reducer.
            proper_source: The data type of the input column.

        Returns:
            The input data type for `min`, `max`, `first` and `last`. The result of the float
            rule, or of one of its exceptions, for `mean`, `median`, `std`, `var` and
            `quantile`. None for a name that has no rule.
        """
        if reducer in self.IDENTITY_REDUCERS:
            return proper_source
        if reducer in self.FLOAT_COERCE_REDUCERS:
            return self._departed(reducer, proper_source) or self._float_coerced_dtype(
                proper_source,
            )
        # Code that calls this class passes only a name from ALL_REDUCERS or "len". The
        # whole-frame hooks register that set, and pivot checks the name against it first.
        return None  # pragma: no cover

    def _float_coerced_dtype(self, proper_source: Instance) -> Instance | None:
        """Give the result of the float rule, which `mean()`, `median()` and their kin share.

        `_departed` handles the two cases that differ from this rule.

        Args:
            proper_source: The data type of the input column.

        Returns:
            The input data type for `Float32` and for `Unchecked`. `Float64` for any other
            number, for `Boolean`, for `Decimal` and for `Int128`. The input data type for
            every other input, for example `String`, `List`, `Duration` or `Datetime`.
        """
        fullname = proper_source.type.fullname
        if self._promotion.is_float32(fullname) or self._promotion.is_unchecked(fullname):
            return proper_source
        # The test uses is_base_of, so a Decimal with or without its parameters matches.
        extended = any(target.is_base_of(proper_source) for target in self._FLOAT_COERCED_TARGETS)
        if extended or self._families.is_numeric_or_boolean(proper_source):
            return self._fresh(self._FLOAT64_TARGET.fullname)
        return proper_source

    def _departed(self, reducer: str, proper_source: Instance) -> Instance | None:
        """Give the result for the two cases that differ from the float rule.

        `quantile()` keeps a `Boolean` input. `mean()`, `median()` and `quantile()` turn a
        `Date` input into a `Datetime` in microseconds.

        Args:
            reducer: The name of the reducer.
            proper_source: The data type of the input column.

        Returns:
            The data type of the result, or None if the float rule applies.
        """
        if reducer == "quantile" and self._BOOLEAN_TARGET.matches(proper_source):
            return proper_source
        if reducer in self._DATE_TO_DATETIME and self._DATE_TARGET.matches(proper_source):
            return self._microsecond_datetime()
        return None

    def _microsecond_datetime(self) -> Instance | None:
        """Build the type of a `Datetime` in microseconds with no time zone.

        Returns:
            `Datetime[Literal["us"], None]`, or None if the plugin cannot find a class.
        """
        marker = self._type_info(self._DATETIME_MARKER.fullname)
        text = self._type_info("builtins.str")
        if marker is None or text is None:  # pragma: no cover
            # mypy always loads both classes before a reducer runs.
            return None
        unit = LiteralType("us", Instance(text, []))
        return Instance(marker, [unit, NoneType()])

    def _fresh(self, fullname: str) -> Instance | None:
        """Build an instance of a data type class, with no type arguments.

        Args:
            fullname: The mypy full name of the class.

        Returns:
            The instance, or None if the plugin cannot find the class.
        """
        info = self._type_info(fullname)
        if info is None:  # pragma: no cover
            # Every full name here is a polars data type class. mypy loads polars before a
            # reducer runs, so the lookup always finds it.
            return None
        return Instance(info, [])

    def _type_info(self, fullname: str) -> TypeInfo | None:
        """Find a class by its full name.

        Args:
            fullname: The mypy full name of the class.

        Returns:
            The `TypeInfo` of the class, or None if the plugin cannot find it.
        """
        symbol = self._plugin.lookup_fully_qualified(fullname)
        if symbol is None or not isinstance(symbol.node, TypeInfo):  # pragma: no cover
            return None
        return symbol.node
