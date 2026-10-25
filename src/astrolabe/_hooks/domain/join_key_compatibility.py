"""Decide whether two join keys can meet, and give the data type of a merged key column."""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

import polars as pl
from mypy.types import Instance, get_proper_type

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
    from astrolabe._hooks.domain.frame import PluginLookup


class JoinKeyCompatibility:
    """Check the data types of a pair of join keys, and give the type of their merged column.

    A join compares a key column of the left frame with a key column of the right frame. Polars
    accepts the pair if the data types are equal, if both are integers of any width and any
    signedness, or if both are floats. Any other pair raises an error at run time. The checker
    reports that error before the code runs, and code that checks a join asks this class.
    `Unchecked`, the data type of a column whose data type is not known, matches any key.

    The data type of the key in the result depends on the kind of join. Most joins keep the
    key column of one side as it is, even if it is the narrower one. Only `how="full"` with
    `coalesce=True` merges the two key columns into one, and polars then widens:

    - Two integers of the same signedness, or two floats, give the wider one.
    - A signed and an unsigned integer give the signed side if it is wider. Otherwise they give
      the signed integer with twice the unsigned width. That always exists, up to `Int128`.

    It sits in the plugin layer, and it reads and builds mypy instance types.

    Example:
        >>> fixture = new_type_fixture()
        >>> compat = join_key_compatibility(new_plugin(fixture))
        >>> compat.compatible(int32_dtype(fixture), int64_dtype(fixture))
        True
        >>> compat.full_coalesce_supertype(int8_dtype(fixture), uint8_dtype(fixture))
        polars.datatypes.classes.Int16
    """

    _SIGNED_WIDTH: Final[dict[Target, int]] = {
        Target.of(pl.Int8): 8,
        Target.of(pl.Int16): 16,
        Target.of(pl.Int32): 32,
        Target.of(pl.Int64): 64,
        Target.of(pl.Int128): 128,
    }
    _UNSIGNED_WIDTH: Final[dict[Target, int]] = {
        Target.of(pl.UInt8): 8,
        Target.of(pl.UInt16): 16,
        Target.of(pl.UInt32): 32,
        Target.of(pl.UInt64): 64,
    }
    _FLOAT_WIDTH: Final[dict[Target, int]] = {
        Target.of(pl.Float32): 32,
        Target.of(pl.Float64): 64,
    }
    _INTEGER_TARGETS: Final[frozenset[Target]] = frozenset(_SIGNED_WIDTH) | frozenset(
        _UNSIGNED_WIDTH,
    )
    _FLOAT_TARGETS: Final[frozenset[Target]] = frozenset(_FLOAT_WIDTH)
    _SIGNED_PAIRS: Final = [(width, target) for target, width in _SIGNED_WIDTH.items()]
    _SIGNED_TARGET_BY_WIDTH: Final[dict[int, Target]] = dict(_SIGNED_PAIRS)

    def __init__(self, unchecked: UncheckedCompatibility, plugin: PluginLookup) -> None:
        """Keep the `Unchecked` rule and the class lookup that this check needs.

        Args:
            unchecked: The equality of data types that lets `Unchecked` match anything. It
                runs before the family rule.
            plugin: The mypy plugin, which finds the class of a widened key, for example
                `Int64` for `Int32` beside `UInt32`.
        """
        self._unchecked = unchecked
        self._type_info = FreshTypeInfo(plugin)

    def compatible(self, a: MypyType, b: MypyType) -> bool:
        """Tell whether polars can join a left key of one data type with a right key of another.

        Args:
            a: The data type of the left key.
            b: The data type of the right key.

        Returns:
            True if the data types are equal, if one is `Unchecked`, if both are integers, or if
            both are floats.

        Example:
            >>> fixture = new_type_fixture()
            >>> compat = join_key_compatibility(new_plugin(fixture))
            >>> compat.compatible(int32_dtype(fixture), str_dtype(fixture))
            False
        """
        if self._unchecked.compatible(a, b):
            return True
        a_proper = get_proper_type(a)
        b_proper = get_proper_type(b)
        if not (isinstance(a_proper, Instance) and isinstance(b_proper, Instance)):
            return False
        a_target = Target(a_proper.type.fullname)
        b_target = Target(b_proper.type.fullname)
        if self._both_in(a_target, b_target, self._INTEGER_TARGETS):
            return True
        return self._both_in(a_target, b_target, self._FLOAT_TARGETS)

    def _both_in(self, a: Target, b: Target, targets: frozenset[Target]) -> bool:
        """Tell whether two classes are both in one family.

        Args:
            a: The class of the left key.
            b: The class of the right key.
            targets: The classes of the family.

        Returns:
            True if both classes are in `targets`.
        """
        return a in targets and b in targets

    def full_coalesce_supertype(self, a: MypyType, b: MypyType) -> Instance | None:
        """Give the data type of the merged key column of a full join with `coalesce=True`.

        Call it only for a pair that `compatible` accepts.

        Args:
            a: The data type of the left key.
            b: The data type of the right key.

        Returns:
            The wider float for two floats, or the result of `_integer_supertype`. None if one
            side is not an instance, for example a type variable, or if the pair is neither two
            floats nor two integers.

        Example:
            >>> fixture = new_type_fixture()
            >>> compat = join_key_compatibility(new_plugin(fixture))
            >>> compat.full_coalesce_supertype(plain_typevar(fixture), int64_dtype(fixture))
        """
        a_proper, b_proper = get_proper_type(a), get_proper_type(b)
        if not (isinstance(a_proper, Instance) and isinstance(b_proper, Instance)):
            return None
        a_target = Target(a_proper.type.fullname)
        b_target = Target(b_proper.type.fullname)
        if self._both_in(a_target, b_target, self._FLOAT_TARGETS):
            return self._wider(a_proper, b_proper, self._FLOAT_WIDTH)
        return self._integer_supertype(a_proper, b_proper)

    @propagates(None)
    def _integer_supertype(self, a: Instance, b: Instance) -> Instance | None:
        """Give the widened data type of two integer keys.

        Args:
            a: The data type of the left key.
            b: The data type of the right key.

        Returns:
            The wider side if both have the same signedness, and the result of
            `_mixed_signedness_supertype` if not. None if one side is not an integer.
        """
        a_signed = not_none(self._is_signed(a))
        b_signed = not_none(self._is_signed(b))
        if a_signed == b_signed:
            widths = self._SIGNED_WIDTH if a_signed else self._UNSIGNED_WIDTH
            return self._wider(a, b, widths)
        if a_signed:
            return self._mixed_signedness_supertype(a, b)
        return self._mixed_signedness_supertype(b, a)

    def _is_signed(self, dtype: Instance) -> bool | None:
        """Tell whether an integer data type is signed.

        Args:
            dtype: The data type to test.

        Returns:
            True for a signed integer, False for an unsigned integer, and None for any other
            data type.
        """
        target = Target(dtype.type.fullname)
        if target in self._SIGNED_WIDTH:
            return True
        if target in self._UNSIGNED_WIDTH:
            return False
        return None

    def _wider(self, a: Instance, b: Instance, widths: dict[Target, int]) -> Instance:
        """Give the wider of two data types of the same family.

        Args:
            a: The data type of the left key.
            b: The data type of the right key.
            widths: The bit width of each class in the family.

        Returns:
            `a` if it is at least as wide as `b`, else `b`.
        """
        a_width = widths[Target(a.type.fullname)]
        b_width = widths[Target(b.type.fullname)]
        return a if a_width >= b_width else b

    def _mixed_signedness_supertype(self, signed: Instance, unsigned: Instance) -> Instance | None:
        """Give the widened data type of a signed and an unsigned integer key.

        Args:
            signed: The data type of the signed key.
            unsigned: The data type of the unsigned key.

        Returns:
            `signed` if it is wider than `unsigned`. Otherwise, a new instance of the signed
            integer with twice the width of `unsigned`, which is at most `Int128`. None if the
            plugin cannot find that class.
        """
        signed_width = self._SIGNED_WIDTH[Target(signed.type.fullname)]
        unsigned_width = self._UNSIGNED_WIDTH[Target(unsigned.type.fullname)]
        if signed_width > unsigned_width:
            return signed
        target = self._SIGNED_TARGET_BY_WIDTH[unsigned_width * 2]
        info = self._type_info.resolved(target.fullname)
        if info is None:  # pragma: no cover
            # The class is always a polars data type class. mypy loads polars before it checks
            # any .join() call, so the lookup always finds it.
            return None
        return Instance(info, [])
