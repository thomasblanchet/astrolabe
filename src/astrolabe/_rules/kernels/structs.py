"""The kernels of the `struct` namespace, which read the fields of a struct.

A function of the `struct` namespace reads its input as a struct column.
Data of another type stops it with a `SchemaError`. The kernel then picks a
field, or gives the struct new names for its fields. The planner of
`struct.rename_fields` types any input, and its kernel refuses an input that
is not a struct. `struct.json_encode` is the exception: its kernel writes
data of any type as JSON text, and panics on binary data, which the writer
of JSON does not take.

Some kernels read an option of their node, such as the name of
`struct.field`, so they are built from the value of the option.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.dispatch import Arm, Data, Dispatch, Fixed, Kinds, Refuse
from astrolabe._rules.type_functions import StructClosures

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.type_functions import FieldSelection


@dataclass(frozen=True)
class FieldOf:
    """Give the type of one field of a `Struct`, or refuse a name that the struct does not hold.

    Example:
        >>> struct = DType("Struct", ((("a", DType("Int8")),),))
        >>> FieldOf("a").give(Data(struct), StructKernels().field_by_name("a"))
        Int8
    """

    name: str

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the field.

        Args:
            data: The data, of a `Struct` type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The struct has no field of that name.

        Example:
            >>> struct = DType("Struct", ((("b", DType("Int8")),),))
            >>> FieldOf("a").give(Data(struct), StructKernels().field_by_name("a"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: StructFieldNotFoundError: a
        """
        del kernel
        fields = dict(data.dtype.fields())
        if self.name not in fields:  # pragma: no cover -- the type function refuses it first
            raise PolarsRefusal(ErrorClass.STRUCT_FIELD_NOT_FOUND, self.name)
        return fields[self.name]


@dataclass(frozen=True)
class Renamed:
    """Give a struct with new names for its fields, in order, as `struct.rename_fields` does.

    The kernel pairs each field with a name, in order. Before polars 2.0.0,
    a struct with more fields than names keeps only the fields that have a
    name. From that release, the type function refuses another count of names.

    Example:
        >>> pair = DType("Struct", ((("a", DType("Int8")), ("b", DType("Date"))),))
        >>> from astrolabe._rules.releases import PolarsRelease
        >>> with PolarsRelease.pretend("1.44.2"):
        ...     Renamed(("x",)).give(Data(pair), StructKernels().rename_fields(("x",)))
        Struct({'x': Int8})
    """

    names: Sequence[str]

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the struct type.

        Args:
            data: The data, of a `Struct` type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Example:
            >>> one = DType("Struct", ((("a", DType("Int8")),),))
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.44.2"):
            ...     Renamed(("x", "y")).give(Data(one), StructKernels().rename_fields(("x", "y")))
            Struct({'x': Int8})
        """
        del kernel
        return StructClosures().rename_fields(self.names)(data.dtype)


@dataclass(frozen=True)
class Selected:
    """Give the type of the one field that a selector picks, as an expanded `struct.unnest` does.

    Polars expands `struct.unnest` into one `struct.field` for each field
    that its selector picks, so the kernel is the kernel of that field.

    Example:
        >>> from astrolabe._rules.type_functions import FieldSelection
        >>> struct = DType("Struct", ((("a", DType("Date")),),))
        >>> every = FieldSelection("Wildcard")
        >>> Selected(every).give(Data(struct), StructKernels().select_fields(every))
        Date
    """

    selection: FieldSelection

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the field.

        It raises `PolarsRefusal` where the selector picks two fields or more.

        Args:
            data: The data, of a `Struct` type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.type_functions import FieldSelection
            >>> pair = DType("Struct", ((("a", DType("Int8")), ("b", DType("Date"))),))
            >>> second = FieldSelection({"ByIndex": {"indices": [1], "strict": True}})
            >>> Selected(second).give(Data(pair), StructKernels().select_fields(second))
            Date
        """
        del kernel
        return StructClosures().select_fields(self.selection)(data.dtype)


class HoldsBinary:
    """Test a data type for binary data at any level, which the writer of JSON does not take.

    Example:
        >>> HoldsBinary()(DType("List", (DType("Binary"),)))
        True
        >>> HoldsBinary()(DType("Struct", ((("a", DType("Int8")),),)))
        False
    """

    def __call__(self, dtype: DType) -> bool:
        """Say whether the type holds `Binary` data.

        Args:
            dtype: The data type.

        Returns:
            Whether it does.

        Example:
            >>> HoldsBinary()(DType("Struct", ((("a", DType("Binary")),),)))
            True
        """
        if dtype.name in Is.LISTS:
            return self(dtype.inner())
        if dtype.name == "Struct":
            return any(self(field) for _, field in dtype.fields())
        return dtype.name == "Binary"


class StructKernels:
    """Build the kernels of the `struct` namespace.

    Each kernel but the JSON writer takes a `Struct`, and any other type
    raises a `SchemaError`. A kernel that reads an option of its node is built
    from the value of the option.

    Example:
        >>> struct = DType("Struct", ((("a", DType("Date")),),))
        >>> StructKernels().field_by_name("a").output(Data(struct))
        Date
        >>> StructKernels.JSON_ENCODE.output(Data(DType("Date")))
        String
    """

    STRUCTS: ClassVar[Kinds] = Kinds("Struct")
    JSON_ENCODE: ClassVar[Dispatch] = Dispatch(
        "struct.json_encode",
        (Arm(HoldsBinary(), Refuse(Stop.PANICS)),),
        Fixed(DType("String")),
    )

    def field_by_name(self, name: str) -> Dispatch:
        """Give the kernel of `struct.field`.

        The kernel reads its input as a `Struct`. Another type raises a
        `SchemaError`, and a struct with no field of the name raises a
        `StructFieldNotFoundError`.

        Args:
            name: The name of the field.

        Returns:
            The dispatch.

        Example:
            >>> StructKernels().field_by_name("a").output(Data(DType("Int8")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: struct.field ...
        """
        arms = (Arm(self.STRUCTS, FieldOf(name)),)
        return Dispatch("struct.field", arms, Refuse(ErrorClass.SCHEMA))

    def rename_fields(self, names: Sequence[str]) -> Dispatch:
        """Give the kernel of `struct.rename_fields`.

        Args:
            names: The new names of the fields.

        Returns:
            The dispatch.

        Example:
            >>> StructKernels().rename_fields(("x",)).output(Data(DType("Int8")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: struct.rename_fields ...
        """
        arms = (Arm(self.STRUCTS, Renamed(names)),)
        return Dispatch("struct.rename_fields", arms, Refuse(ErrorClass.SCHEMA))

    def select_fields(self, selection: FieldSelection) -> Dispatch:
        """Give the kernel of `struct.unnest`, and of `struct.field` with several names.

        Args:
            selection: The selector of the fields.

        Returns:
            The dispatch.

        Example:
            >>> from astrolabe._rules.type_functions import FieldSelection
            >>> StructKernels().select_fields(FieldSelection("Wildcard")).name
            'struct.unnest'
        """
        arms = (Arm(self.STRUCTS, Selected(selection)),)
        return Dispatch("struct.unnest", arms, Refuse(ErrorClass.SCHEMA))
