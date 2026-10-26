"""The kernel of `fill_null`: the value where it holds a value, and the fill where it holds a null.

`fill_null` has two inputs, the value and the fill. Its cast pass gives both
inputs one type, and then the kernel reads the data. It does not check the
types again. It reads the nulls of the value and the lengths of the two inputs:

- A value of one row beside a longer fill gives the fill where the value is
  null. Otherwise it spreads the value to the length of the fill.
- A value with no null comes back as it is.
- A value of the type `Null` gives the fill. Before polars 1.40.0, it takes
  the last path and stays `Null` (`FillNull._of_nulls`).
- Any other value takes the fill in the place of each null, and keeps its own
  type. A `Categorical` value first casts a `String` fill to its own type.

The last case reads the fill as data of the type of the value. Where the two
types differ, each type of polars reads the fill in a way of its own
(`FillNull._zipped`): a date or a time reads a fill of its physical type, a
`Decimal` reads any `Decimal`, a `Decimal`, a struct or a categorical value
raises beside a fill of another kind, and any other value panics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType, Is, Parameter
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.releases import polars_releases

if TYPE_CHECKING:
    from astrolabe._rules.kernels.dispatch import Data


class FillNull:
    """Give the type of `fill_null` at the data of its value and of its fill.

    A value whose `Data.nulls` is false holds at least one value that is not
    null. Where some of its values are null, the kernel fills them, and the
    type stays the type of the value. So the rule gives the type of the value,
    and a fill of another type is a fact of the values: it fails only where a
    value is null.

    Example:
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> FillNull().output(Data(DType("Int8")), Data(DType("Int16")))
        Int8
        >>> FillNull().output(Data(DType("Null")), Data(DType("Int16")))
        Int16
    """

    def output(self, value: Data, fill: Data) -> DType:
        """Give the type of the output.

        Args:
            value: The data of the value.
            fill: The data of the fill.

        Returns:
            The type of the output.

        Example:
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> one_null = Data(DType("Int8"), nulls=True, one_row=True)
            >>> FillNull().output(one_null, Data(DType("Int16")))
            Int16
        """
        if value.one_row and not fill.one_row:
            return fill.dtype if value.nulls else value.dtype
        if not value.nulls:
            return value.dtype
        if value.dtype.name == "Null":
            return self._of_nulls(fill)
        return self._zipped(value.dtype, fill.dtype)

    @polars_releases
    def _of_nulls(self, fill: Data) -> DType:
        """Give the type of the output for a value of the type `Null`: the type of the fill.

        Args:
            fill: The data of the fill.

        Returns:
            The type of the fill.

        Example:
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     FillNull()._of_nulls(Data(DType("Int16")))
            Int16
        """
        return fill.dtype

    @_of_nulls.before("1.40.0")
    def _kept_null(self, fill: Data) -> DType:
        """Give the type of the output for a value of the type `Null` in the older releases: `Null`.

        These releases take the fill in the place of each null as data of the
        type of the value, so the output stays `Null`.

        Args:
            fill: The data of the fill, which the output does not read.

        Returns:
            `Null`.

        Example:
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> FillNull()._kept_null(Data(DType("Int16")))
            Null
        """
        del fill
        return DType("Null")

    TEMPORAL: ClassVar[frozenset[str]] = frozenset({"Date", "Datetime", "Duration", "Time"})
    CHECKED: ClassVar[frozenset[str]] = frozenset({"Decimal", "Struct", "Categorical", "Enum"})

    def _zipped(self, value: DType, fill: DType) -> DType:
        """Give the type of a value of nulls that takes the fill in the place of each null.

        Each type of polars reads the fill in a way of its own. A date or a
        time reads the physical data of the fill, so a fill of the same
        physical type runs. A `Decimal` reads any `Decimal`, and keeps its own
        parameters. A struct zips its fields with the fields of a struct
        fill, in order, by these same rules, and keeps as many fields as the
        shorter of the two (`_struct`). A list and an array read the physical
        data of their inner values, so a fill of the same kind runs where the
        physical types inside agree, and where an array fill is at least as
        wide, and panics otherwise. A `Decimal`, a struct, a `Categorical` and
        an `Enum` check the kind of the fill and raise a `SchemaError`. Any
        other value reads the fill as data of its own type, and panics.

        Args:
            value: The type of the value.
            fill: The type of the fill.

        Returns:
            The type of the value, with the fields of a struct that the fill also has.

        Raises:
            PolarsRefusal: The type of the value does not read the fill.

        Example:
            >>> FillNull()._zipped(DType("Categorical"), DType("String"))
            Categorical
            >>> FillNull()._zipped(DType("Datetime", ("us", None)), DType("Int64"))
            Datetime(time_unit='us', time_zone=None)
            >>> FillNull()._zipped(DType("Int8"), DType("Date"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        if value.name == "Struct" and fill.name == "Struct":
            return self._struct(value, fill)
        if self._reads(value, fill):
            return value
        if value.name in self.CHECKED:
            message = f"invalid series dtype: expected `{value!r}`, got `{fill!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        raise PolarsRefusal(Stop.PANICS, f"cannot get ref {value!r} from {fill!r}")

    def _reads(self, value: DType, fill: DType) -> bool:
        """Say whether a value reads a fill of another type as its own data.

        Args:
            value: The type of the value.
            fill: The type of the fill.

        Returns:
            Whether it does: a `Categorical` beside text, a date or a time
            beside a fill of its physical type, a `Decimal` beside a `Decimal`,
            and a list or an array beside one of the same physical layout.

        Example:
            >>> FillNull()._reads(DType("Date"), DType("Int32"))
            True
        """
        if value.same(fill) or (value.name == "Categorical" and Is.string(fill)):
            return True
        if value.name in self.TEMPORAL:
            return value.physical().same(fill.physical())
        if value.name == "Decimal":
            return fill.name == "Decimal"
        if value.name not in {"List", "Array"} or fill.name != value.name:
            return False
        inner = self._layout(value.inner()) == self._layout(fill.inner())
        return inner and (value.name == "List" or self._width(fill) >= self._width(value))

    def _struct(self, value: DType, fill: DType) -> DType:
        """Give the type of a struct value of nulls that takes a struct fill, field by field.

        Args:
            value: The type of the value.
            fill: The type of the fill.

        Returns:
            The fields of the value, in order, as many as the shorter struct holds.

        Example:
            >>> wide = DType("Struct", ((("a", DType("Int8")), ("b", DType("String"))),))
            >>> narrow = DType("Struct", ((("c", DType("Int8")),),))
            >>> FillNull()._struct(wide, narrow)
            Struct({'a': Int8})
        """
        pairs = zip(value.fields(), fill.fields(), strict=False)
        fields = tuple((name, self._zipped(kind, other)) for (name, kind), (_, other) in pairs)
        return DType("Struct", (fields,))

    def _width(self, dtype: DType) -> int:
        """Give the number of values of each row of an array.

        Args:
            dtype: The array type.

        Returns:
            The width of its first dimension.

        Example:
            >>> FillNull()._width(DType("Array", (DType("Int8"), 3)))
            3
        """
        return Parameter().integer(dtype, 1)

    def _layout(self, dtype: DType) -> object:
        """Give the physical layout of a type: its physical type, with no width of an array.

        Args:
            dtype: The type.

        Returns:
            The layout, which two types share where nested data of one reads the other.

        Example:
            >>> FillNull()._layout(DType("List", (DType("Datetime", ("us", None)),)))
            ('List', 'Int64')
        """
        physical = dtype.physical()
        if physical.name in {"List", "Array"}:
            return (physical.name, self._layout(physical.inner()))
        if physical.name == "Struct":
            return ("Struct", tuple(self._layout(kind) for _, kind in physical.fields()))
        return physical.render()
