"""Sort the data type of a column into a family, such as numeric or `Decimal`.

Some operations reduce each column to one value: the whole-frame reducers such as `df.sum()`,
and the aggregate of `.pivot`. Their result depends on the family of the input data type, not
on its exact width, so the checker decides them by family, which covers every integer width that
polars has. This module holds those family tests, and the one change of precision that `sum` and
`product` make to a `Decimal`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import polars as pl
from mypy.types import Instance, LiteralType, get_proper_type

from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe.frame import Decimal as _DecimalMarker

if TYPE_CHECKING:
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup


class DtypeFamilies:
    """Tell which family a data type belongs to, for code that types a reduction by family.

    A reduction such as `sum` gives an integer, a float or a `Decimal` depending on the family of
    its input. Code that types such an operation asks this class, and never compares class names
    itself. Each test reads a mypy type and answers True or False. A type that is not an
    instance, for example a type variable, belongs to no family.

    Example:
        >>> fixture = new_type_fixture()
        >>> families = DtypeFamilies(new_plugin(fixture))
        >>> families.is_numeric_or_boolean(int_dtype(fixture))
        True
        >>> families.is_numeric_or_boolean(str_dtype(fixture))
        False
    """

    _SIGNED: Final[frozenset[Target]] = frozenset({
        Target.of(pl.Int8),
        Target.of(pl.Int16),
        Target.of(pl.Int32),
        Target.of(pl.Int64),
        Target.of(pl.Int128),
    })
    _UNSIGNED: Final[frozenset[Target]] = frozenset({
        Target.of(pl.UInt8),
        Target.of(pl.UInt16),
        Target.of(pl.UInt32),
        Target.of(pl.UInt64),
    })
    _FLOAT: Final[frozenset[Target]] = frozenset({Target.of(pl.Float32), Target.of(pl.Float64)})
    _BOOLEAN: Final[Target] = Target.of(pl.Boolean)
    _DECIMAL: Final[Target] = Target.of(pl.Decimal)
    _DECIMAL_MARKER: Final[Target] = Target.of(_DecimalMarker)
    _INT: Final[Target] = Target.of(int)
    # sum and product of a Decimal column give a Decimal with this precision, the maximum, and
    # the same scale. The precision of the input does not matter.
    _DECIMAL_RESULT_PRECISION: Final[int] = 38

    def __init__(self, plugin: PluginLookup) -> None:
        """Keep the class lookup that `widened_decimal` needs to build a new `Decimal`.

        Args:
            plugin: The mypy plugin, which finds a class by its full name.
        """
        self._type_info = FreshTypeInfo(plugin)

    def is_numeric(self, dtype: MypyType) -> bool:
        """Tell whether a data type is an integer or a float.

        Args:
            dtype: The data type to test.

        Returns:
            True for a signed integer, an unsigned integer or a float.

        Example:
            >>> fixture = new_type_fixture()
            >>> DtypeFamilies(new_plugin(fixture)).is_numeric(boolean_dtype(fixture))
            False
            >>> from mypy.types import UninhabitedType
            >>> DtypeFamilies(new_plugin(fixture)).is_numeric(UninhabitedType())
            False
        """
        instance = self._instance(dtype)
        if instance is None:
            return False
        target = Target(instance.type.fullname)
        return target in self._SIGNED or target in self._UNSIGNED or target in self._FLOAT

    def is_numeric_or_boolean(self, dtype: MypyType) -> bool:
        """Tell whether a data type is numeric or `Boolean`.

        Polars can sum a `Boolean` column as a count of the true values, so some reductions
        accept this wider family.

        Args:
            dtype: The data type to test.

        Returns:
            True for an integer, a float or `Boolean`.
        """
        return self.is_numeric(dtype) or self.is_boolean(dtype)

    def is_numeric_or_decimal(self, dtype: MypyType) -> bool:
        """Tell whether a data type is numeric or `Decimal`.

        Args:
            dtype: The data type to test.

        Returns:
            True for an integer, a float or `Decimal`.

        Example:
            >>> fixture = new_type_fixture()
            >>> families = DtypeFamilies(new_plugin(fixture))
            >>> families.is_numeric_or_decimal(decimal_dtype(fixture, 10, 2))
            True
        """
        return self.is_numeric(dtype) or self.is_decimal(dtype)

    def is_boolean(self, dtype: MypyType) -> bool:
        """Tell whether a data type is `Boolean`.

        Args:
            dtype: The data type to test.

        Returns:
            True for `Boolean`.

        Example:
            >>> fixture = new_type_fixture()
            >>> DtypeFamilies(new_plugin(fixture)).is_boolean(boolean_dtype(fixture))
            True
        """
        return self._is_exactly(dtype, self._BOOLEAN)

    def is_decimal(self, dtype: MypyType) -> bool:
        """Tell whether a data type is `Decimal`, with or without its precision and scale.

        Args:
            dtype: The data type to test.

        Returns:
            True for the polars `Decimal` class, and for the marker `Decimal[Precision, Scale]`
            of this project, which is a subclass of it.

        Example:
            >>> fixture = new_type_fixture()
            >>> DtypeFamilies(new_plugin(fixture)).is_decimal(int_dtype(fixture))
            False
        """
        instance = self._instance(dtype)
        return instance is not None and self._DECIMAL.is_base_of(instance)

    def widened_decimal(self, dtype: MypyType) -> Instance | None:
        """Give the data type of the `sum` or the `product` of a `Decimal` column.

        Polars sets the precision of the result to 38, the maximum, and keeps the scale.

        Args:
            dtype: The data type of the input column.

        Returns:
            A new `Decimal[Literal[38], Literal[scale]]`. The input itself if it is a `Decimal`
            with no known scale. None if the input is not a `Decimal`.

        Example:
            >>> fixture = new_type_fixture()
            >>> families = DtypeFamilies(new_plugin(fixture))
            >>> families.widened_decimal(decimal_dtype(fixture, 10, 2))
            astrolabe.frame.Decimal[Literal[38], Literal[2]]
            >>> families.widened_decimal(int_dtype(fixture)) is None
            True
            >>> from mypy.types import Instance
            >>> bare = Instance(decimal_dtype(fixture, 10, 2).type, [])
            >>> families.widened_decimal(bare)
            astrolabe.frame.Decimal
        """
        instance = self._instance(dtype)
        if instance is None or not self._DECIMAL.is_base_of(instance):
            return None
        scale = self._marker_scale(instance)
        if scale is None:
            return instance
        return self._fresh_decimal(scale)

    def _is_exactly(self, dtype: MypyType, target: Target) -> bool:
        """Tell whether a data type is an instance of one class, and not of a subclass.

        Args:
            dtype: The data type to test.
            target: The class.

        Returns:
            True if the data type is an instance of exactly that class.
        """
        instance = self._instance(dtype)
        return instance is not None and target.matches(instance)

    def _instance(self, dtype: MypyType) -> Instance | None:
        """Give the data type as a mypy instance, if it is one.

        Args:
            dtype: The data type to read.

        Returns:
            The instance, or None for any other kind of type.
        """
        proper = get_proper_type(dtype)
        if isinstance(proper, Instance):
            return proper
        return None

    def _marker_scale(self, instance: Instance) -> int | None:
        """Read the scale of a `Decimal[Precision, Scale]` marker.

        Args:
            instance: The `Decimal` instance.

        Returns:
            The scale, or None for the polars `Decimal` class with no type arguments.
        """
        if len(instance.args) != 2:
            return None
        proper = get_proper_type(instance.args[1])
        if not isinstance(proper, LiteralType):  # pragma: no cover
            # The checker builds every Decimal[...] with an integer literal as its scale, and it
            # reports an error for a Decimal[...] that a user writes with any other scale.
            return None
        return checked_cast(proper.value, int)

    @propagates(None)
    def _fresh_decimal(self, scale: int) -> Instance | None:
        """Build the type `Decimal[Literal[38], Literal[scale]]`.

        Args:
            scale: The scale of the result.

        Returns:
            The new instance, or None if the plugin cannot find the `Decimal` marker class or
            `int`.
        """
        info = not_none(self._type_info.resolved(self._DECIMAL_MARKER.fullname))
        int_info = not_none(self._type_info.resolved(self._INT.fullname))
        fallback = Instance(int_info, [])
        return Instance(
            info,
            [
                LiteralType(self._DECIMAL_RESULT_PRECISION, fallback),
                LiteralType(scale, fallback),
            ],
        )
