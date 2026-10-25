"""The data types of the census, and the values that polars runs them on.

A rule of polars can depend on the values as well as on the types. A
division of a `Decimal` by zero raises, and a cast of a column whose values
are all null takes a shortcut. So a test runs each node on several samples
of values, and on columns whose values are all null. This module
holds the 27 data types of the census and builds the frames of values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal, localcontext
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Protocol

import polars as pl

if TYPE_CHECKING:
    from polars.datatypes import DataTypeClass


class Sample(Enum):
    """Name one sample of values.

    `TYPICAL` holds small distinct values. `SMALL` holds zeros and empty
    text, `LARGE` the largest values, and `TEXT` text that parses as a
    number or a date. `SHAPES` shows the lengths that the values decide: a
    float NaN, lists of 0 and 2 elements, and text of several characters,
    in a frame of 400 rows. Its height differs from the others, so a node
    whose length only adds up to 3 rows by chance does not look per row, and
    a literal count such as 300 stays under it.

    Example:
        >>> [sample.value for sample in Sample]
        ['typical', 'small', 'large', 'text', 'shapes']
    """

    TYPICAL = "typical"
    SMALL = "small"
    LARGE = "large"
    TEXT = "text"
    SHAPES = "shapes"

    def height(self) -> int:
        """Give the number of rows of the frame of this sample.

        Returns:
            The height.

        Example:
            >>> Sample.TYPICAL.height(), Sample.SHAPES.height()
            (3, 400)
        """
        return 400 if self is Sample.SHAPES else 3


@dataclass(frozen=True)
class KindValues:
    """Give three values of one data type for each sample.

    A value must be valid for its data type, so each kind has its own
    values. A sample that has no special values for a kind takes the
    typical values.

    Example:
        >>> KindValues({Sample.TYPICAL: (1, 2, 3)}).of(Sample.LARGE)
        (1, 2, 3)
    """

    by_sample: dict[Sample, tuple[object, ...]]

    def of(self, sample: Sample) -> tuple[object, ...]:
        """Give the three values of a sample.

        Args:
            sample: The sample.

        Returns:
            The values of that sample, or the typical values.

        Example:
            >>> KindValues({Sample.TYPICAL: (1, 2, 3), Sample.SMALL: (0, 0, 0)}).of(Sample.SMALL)
            (0, 0, 0)
        """
        return self.by_sample.get(sample, self.by_sample[Sample.TYPICAL])


class Kinds(Protocol):
    """Give the data types of a frame and the values of each."""

    def dtypes(self) -> list[pl.DataType]:
        """Give the data types, in the order of their column names."""

    def values(self, dtype: pl.DataType) -> KindValues:
        """Give the values of one data type.

        Args:
            dtype: One of the data types.
        """


class CensusKinds:
    """Hold the 27 data types of the census and their values.

    Each kind has one point of its parameters, for example `Decimal(10, 2)`
    and `Datetime("ms")`. A frame names the column of kind `i` `k{i}`, and
    its twin whose values are all null `n{i}`. `UInt128` and `Float16` come
    last, so the other kinds keep their numbers.

    Example:
        >>> kinds = CensusKinds()
        >>> len(kinds.dtypes())
        27
        >>> kinds.dtypes()[11]
        Decimal(precision=10, scale=2)
    """

    INT_MAX: ClassVar[dict[str, int]] = {
        "Int8": 127,
        "UInt8": 255,
        "Int16": 32767,
        "UInt16": 65535,
        "Int32": 2**31 - 1,
        "UInt32": 2**32 - 1,
        "Int64": 2**63 - 1,
        "UInt64": 2**64 - 1,
        "Int128": 2**100,
        "UInt128": 2**128 - 1,
    }

    def dtypes(self) -> list[pl.DataType]:
        """Give the 27 data types, in the order of their column names.

        Returns:
            The data types.

        Example:
            >>> CensusKinds().dtypes()[0]
            Int8
        """
        integers = [pl.Int8(), pl.UInt8(), pl.Int16(), pl.UInt16(), pl.Int32()]
        integers += [pl.UInt32(), pl.Int64(), pl.UInt64(), pl.Int128()]
        scalars = [pl.Float32(), pl.Float64(), pl.Decimal(10, 2), pl.Boolean()]
        scalars += [pl.String(), pl.Binary(), pl.Null(), pl.Date()]
        scalars += [pl.Datetime("ms"), pl.Duration("us"), pl.Time()]
        scalars += [pl.Enum(["a", "b", "c"]), pl.Categorical()]
        nested = [pl.List(pl.Int64()), pl.Array(pl.Int64(), 2), pl.Struct({"a": pl.Int8()})]
        wide = [pl.UInt128(), pl.Float16()]
        return integers + scalars + nested + wide

    def values(self, dtype: pl.DataType) -> KindValues:
        """Give the values of one data type.

        Args:
            dtype: One of the data types of the census.

        Returns:
            Its values for each sample.

        Example:
            >>> CensusKinds().values(pl.UInt8()).of(Sample.LARGE)
            (255, 255, 255)
        """
        name = type(dtype).__name__
        if name in self.INT_MAX:
            return self._integer(self.INT_MAX[name])
        return self._other(dtype)

    def _integer(self, largest: int) -> KindValues:
        """Give the values of an integer kind with the given largest value.

        Args:
            largest: The largest value of the kind.

        Returns:
            The values for each sample.

        Example:
            >>> CensusKinds()._integer(127).of(Sample.SMALL)
            (0, 0, 0)
        """
        return KindValues({
            Sample.TYPICAL: (1, 2, 3),
            Sample.SMALL: (0, 0, 0),
            Sample.LARGE: (largest, largest, largest),
        })

    def _other(self, dtype: pl.DataType) -> KindValues:
        """Give the values of a kind that is not an integer.

        Args:
            dtype: The data type.

        Returns:
            The values for each sample.

        Example:
            >>> CensusKinds()._other(pl.String()).of(Sample.TEXT)
            ('1', '2020-01-01', '1.5')
        """
        table = self._scalar_table() | self._temporal_table() | self._nested_table()
        return table[type(dtype).__name__]

    def _scalar_table(self) -> dict[str, KindValues]:
        """Give the values of the scalar kinds that are not integers or temporal.

        Returns:
            The values for each kind name.

        Example:
            >>> CensusKinds()._scalar_table()["Boolean"].of(Sample.TYPICAL)
            (True, False, True)
        """
        return {
            "Float16": KindValues({
                Sample.TYPICAL: (1.0, 2.0, 3.0),
                Sample.SMALL: (0.0, 0.0, 0.0),
                Sample.LARGE: (65504.0, 65504.0, 65504.0),
                Sample.SHAPES: (1.0, float("nan"), 3.0),
            }),
            "Float32": KindValues({
                Sample.TYPICAL: (1.0, 2.0, 3.0),
                Sample.SMALL: (0.0, 0.0, 0.0),
                Sample.LARGE: (1e30, 1e30, 1e30),
                Sample.SHAPES: (1.0, float("nan"), 3.0),
            }),
            "Float64": KindValues({
                Sample.TYPICAL: (1.0, 2.0, 3.0),
                Sample.SMALL: (0.0, 0.0, 0.0),
                Sample.LARGE: (1e300, 1e300, 1e300),
                Sample.SHAPES: (1.0, float("nan"), 3.0),
            }),
            "Decimal": KindValues({
                Sample.TYPICAL: (Decimal("1.00"), Decimal("2.00"), Decimal("3.00")),
                Sample.SMALL: (Decimal("0.00"), Decimal("0.00"), Decimal("0.00")),
                Sample.LARGE: (Decimal("99999999.99"),) * 3,
            }),
            "Boolean": KindValues({
                Sample.TYPICAL: (True, False, True),
                Sample.SMALL: (False, False, False),
            }),
            "String": KindValues({
                Sample.TYPICAL: ("a", "b", "c"),
                Sample.SMALL: ("", "", ""),
                Sample.TEXT: ("1", "2020-01-01", "1.5"),
                Sample.SHAPES: ("ab", "cde", "f"),
            }),
            "Binary": KindValues({
                Sample.TYPICAL: (b"a", b"b", b"c"),
                Sample.SMALL: (b"", b"", b""),
                Sample.SHAPES: (b"ab", b"cde", b"f"),
            }),
            "Null": KindValues({Sample.TYPICAL: (None, None, None)}),
            "Enum": KindValues({Sample.TYPICAL: ("a", "b", "a"), Sample.LARGE: ("c", "c", "c")}),
            "Categorical": KindValues({Sample.TYPICAL: ("a", "b", "a")}),
        }

    def _temporal_table(self) -> dict[str, KindValues]:
        """Give the values of the temporal kinds.

        Returns:
            The values for each kind name.

        Example:
            >>> CensusKinds()._temporal_table()["Time"].of(Sample.SMALL)
            (datetime.time(0, 0), datetime.time(0, 0), datetime.time(0, 0))
        """
        days = (date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3))
        moments = tuple(datetime(day.year, day.month, day.day) for day in days)
        return {
            "Date": KindValues({
                Sample.TYPICAL: days,
                Sample.SMALL: (date(1970, 1, 1),) * 3,
                Sample.LARGE: (date(9999, 12, 31),) * 3,
            }),
            "Datetime": KindValues({
                Sample.TYPICAL: moments,
                Sample.SMALL: (datetime(1970, 1, 1),) * 3,
                Sample.LARGE: (datetime(2262, 1, 1),) * 3,
            }),
            "Duration": KindValues({
                Sample.TYPICAL: (timedelta(days=1), timedelta(days=2), timedelta(days=3)),
                Sample.SMALL: (timedelta(0),) * 3,
                Sample.LARGE: (timedelta(days=100000),) * 3,
            }),
            "Time": KindValues({
                Sample.TYPICAL: (time(1), time(2), time(3)),
                Sample.SMALL: (time(0),) * 3,
                Sample.LARGE: (time(23, 59, 59),) * 3,
            }),
        }

    def _nested_table(self) -> dict[str, KindValues]:
        """Give the values of the nested kinds.

        Returns:
            The values for each kind name.

        Example:
            >>> CensusKinds()._nested_table()["Struct"].of(Sample.LARGE)
            ({'a': 127}, {'a': 127}, {'a': 127})
        """
        largest = self.INT_MAX["Int64"]
        return {
            "List": KindValues({
                Sample.TYPICAL: ([1], [2], [3]),
                Sample.SMALL: ([0], [0], [0]),
                Sample.LARGE: ([largest],) * 3,
                Sample.SHAPES: ([1, 2], [], [3]),
            }),
            "Array": KindValues({
                Sample.TYPICAL: ([1, 1], [2, 2], [3, 3]),
                Sample.SMALL: ([0, 0],) * 3,
                Sample.LARGE: ([largest, largest],) * 3,
            }),
            "Struct": KindValues({
                Sample.TYPICAL: ({"a": 1}, {"a": 2}, {"a": 3}),
                Sample.SMALL: ({"a": 0},) * 3,
                Sample.LARGE: ({"a": 127},) * 3,
            }),
        }


class SampleFrames:
    """Build the frame of one sample, with a column for each kind and its all-null twin.

    The frame has the height of its sample: three rows, or 400 for `SHAPES`,
    which repeat the three values of each kind. Column `k{i}` holds the
    values of kind `i`, column `n{i}` holds nulls of the same kind, `b` holds
    a mask and `g` a group key with one group.

    Example:
        >>> frame = SampleFrames(CensusKinds()).frame(Sample.TYPICAL)
        >>> frame.height, frame.width
        (3, 56)
        >>> frame["n11"].null_count()
        3
    """

    def __init__(self, kinds: Kinds) -> None:
        """Keep the kinds.

        Args:
            kinds: The kinds of the frame.

        Example:
            >>> SampleFrames(CensusKinds())._kinds is not None
            True
        """
        self._kinds = kinds
        self._frames: dict[Sample, pl.DataFrame] = {}

    def frame(self, sample: Sample) -> pl.DataFrame:
        """Give the frame of a sample, built once.

        Args:
            sample: The sample.

        Returns:
            The frame.

        Example:
            >>> SampleFrames(CensusKinds()).frame(Sample.SMALL)["k0"].to_list()
            [0, 0, 0]
        """
        if sample not in self._frames:
            self._frames[sample] = self._build(sample)
        return self._frames[sample]

    def _build(self, sample: Sample) -> pl.DataFrame:
        """Build the frame of a sample.

        Args:
            sample: The sample.

        Returns:
            The frame.

        Example:
            >>> SampleFrames(CensusKinds())._build(Sample.LARGE)["k1"].to_list()
            [255, 255, 255]
        """
        height = sample.height()
        columns: list[pl.Series] = []
        for index, dtype in enumerate(self._kinds.dtypes()):
            values = self._kinds.values(dtype).of(sample)
            repeated = [values[row % len(values)] for row in range(height)]
            column = pl.Series(f"k{index}", repeated, dtype=dtype)
            twin = pl.Series(f"n{index}", [None] * height, dtype=dtype)
            columns.extend((column, twin))
        mask = pl.Series("b", [row % 3 != 2 for row in range(height)])
        key = pl.Series("g", [1] * height)
        columns.extend((mask, key))
        return pl.DataFrame(columns)


class AnyValues:
    """Give three values of any polars data type for a sample, its parameters included.

    A test runs a node at data types that the census kinds do not hold, for
    example `Decimal(38, 2)` or `Struct({"a": Int16})`. Those need values too,
    on every sample, so that a value that does not fit shows on one sample and
    not on another. A flat type takes the values of the census
    kind of its name, and a nested type takes values built from its parts.

    Example:
        >>> AnyValues().of(pl.Decimal(38, 2))
        (Decimal('1.00'), Decimal('2.00'), Decimal('3.00'))
        >>> AnyValues().of(pl.Struct({"p": pl.Int16(), "q": pl.Boolean()}))
        ({'p': 1, 'q': True}, {'p': 2, 'q': False}, {'p': 3, 'q': True})
        >>> AnyValues().of(pl.List(pl.Int8()), Sample.SHAPES)
        ([1, 2], [], [3])
    """

    def of(self, dtype: pl.DataType, sample: Sample = Sample.TYPICAL) -> tuple[object, ...]:
        """Give three values of a data type for a sample.

        Args:
            dtype: The data type.
            sample: The sample.

        Returns:
            The values.

        Example:
            >>> AnyValues().of(pl.List(pl.Date()))
            ([datetime.date(2020, 1, 1)], [datetime.date(2020, 1, 2)], [datetime.date(2020, 1, 3)])
        """
        if isinstance(dtype, pl.Decimal):
            return self._decimal(dtype, sample)
        if isinstance(dtype, pl.Enum):
            return self._enum(dtype, sample)
        if isinstance(dtype, (pl.List, pl.Array, pl.Struct)):
            return self._nested(dtype, sample)
        return CensusKinds().values(self._census_point(dtype)).of(sample)

    def _decimal(self, dtype: pl.Decimal, sample: Sample) -> tuple[object, ...]:
        """Give three values that fit a decimal type, as the census kind gives them.

        The sample of small values holds zeros, and the sample of large
        values holds the largest value that the precision holds, so that a
        count or a divisor of the type meets the values that the census kind
        meets.

        Args:
            dtype: The decimal type.
            sample: The sample.

        Returns:
            The values.

        Example:
            >>> AnyValues()._decimal(pl.Decimal(4, 4), Sample.TYPICAL)
            (Decimal('0.0001'), Decimal('0.0002'), Decimal('0.0003'))
            >>> AnyValues()._decimal(pl.Decimal(38, 2), Sample.LARGE)[0]
            Decimal('999999999999999999999999999999999999.99')
        """
        scale = dtype.scale or 0
        step = Decimal(1).scaleb(-scale)
        digits = (dtype.precision or 38) - scale
        if sample is Sample.SMALL:
            return (Decimal(0).quantize(step),) * 3
        with localcontext() as context:
            context.prec = 80
            largest = Decimal(10) ** max(digits, 0) - step
            if sample is Sample.LARGE:
                return (largest.quantize(step),) * 3
        if digits <= 0:
            return tuple((step * index).quantize(step) for index in (1, 2, 3))
        return tuple(Decimal(index).quantize(step) for index in (1, 2, 3))

    def _enum(self, dtype: pl.Enum, sample: Sample) -> tuple[object, ...]:
        """Give three categories of an `Enum`, as the census kind gives them.

        The sample of large values holds the last category three times.

        Args:
            dtype: The `Enum` type.
            sample: The sample.

        Returns:
            The values.

        Example:
            >>> AnyValues()._enum(pl.Enum(["x", "y"]), Sample.LARGE)
            ('y', 'y', 'y')
        """
        categories = list(dtype.categories)
        if sample is Sample.LARGE:
            return (categories[-1],) * 3
        return tuple(categories[index % len(categories)] for index in range(3))

    def _nested(self, dtype: pl.List | pl.Array | pl.Struct, sample: Sample) -> tuple[object, ...]:
        """Give three values of a nested type from the values of its parts.

        On the sample `SHAPES`, a list type holds lists of 2, 0 and 1 elements.

        Args:
            dtype: The nested type.
            sample: The sample.

        Returns:
            The values.

        Example:
            >>> AnyValues()._nested(pl.Array(pl.Int8(), 2), Sample.TYPICAL)
            ([1, 1], [2, 2], [3, 3])
        """
        if isinstance(dtype, pl.Struct):
            return self._struct(dtype, sample)
        if isinstance(dtype, pl.Array):
            inner = self.of(self._instance(dtype.inner), sample)
            return tuple([value] * dtype.size for value in inner)
        if sample is Sample.SHAPES:
            first, second, third = self.of(self._instance(dtype.inner), Sample.TYPICAL)
            return ([first, second], [], [third])
        inner = self.of(self._instance(dtype.inner), sample)
        return tuple([value] for value in inner)

    def _struct(self, dtype: pl.Struct, sample: Sample) -> tuple[object, ...]:
        """Give three values of a struct type, one row of field values each.

        Args:
            dtype: The struct type.
            sample: The sample.

        Returns:
            The values.

        Example:
            >>> AnyValues()._struct(pl.Struct({"a": pl.Boolean()}), Sample.TYPICAL)
            ({'a': True}, {'a': False}, {'a': True})
        """
        fields = {
            field.name: self.of(self._instance(field.dtype), sample) for field in dtype.fields
        }
        rows: list[dict[str, object]] = []
        for index in range(3):
            rows.append({name: values[index] for name, values in fields.items()})
        return tuple(rows)

    def _instance(self, dtype: pl.DataType | DataTypeClass) -> pl.DataType:
        """Give an instance of a data type that polars can also give as a class.

        Args:
            dtype: The data type, as an instance or as a class.

        Returns:
            The instance.

        Example:
            >>> AnyValues()._instance(pl.Int8)
            Int8
        """
        if isinstance(dtype, pl.DataType):
            return dtype
        instance: pl.DataType = dtype()
        return instance

    def _census_point(self, dtype: pl.DataType) -> pl.DataType:
        """Give the census kind that has the same values as a flat data type.

        A `Datetime` of any unit and zone takes the values of the census
        `Datetime`, and the same for `Duration`.

        Args:
            dtype: A flat data type.

        Returns:
            The census kind of the same name.

        Raises:
            ValueError: If no census kind has the name, for example `Unknown`.

        Example:
            >>> AnyValues()._census_point(pl.Datetime("ns", "UTC"))
            Datetime(time_unit='ms', time_zone=None)
        """
        name = type(dtype).__name__
        for kind in CensusKinds().dtypes():
            if type(kind).__name__ == name:
                return kind
        message = f"no census kind has the values of {dtype}"
        raise ValueError(message)
