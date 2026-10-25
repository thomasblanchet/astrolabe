"""Read and build the mypy type of a typed series, which carries its name and its data type."""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

from mypy.nodes import TypeInfo
from mypy.types import Instance, LiteralType, get_proper_type

from astrolabe._hooks.dispatch.generic import Target
from astrolabe.frame import TypedSeries

if TYPE_CHECKING:
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup


class SeriesRef:
    """Build a `TypedSeries[Name, DType]` type, and read the name and the data type of one.

    The checker tracks a series the way it tracks a column: `TypedSeries[Literal["price"],
    pl.Int64]` names the series and gives its data type. Most code here builds such a type as
    a result, for example for `df["price"]`, `pl.Series(...)` or `series.cast(...)`. Other code
    reads a series that it receives: a method checks the data type of its receiver, and passes
    the name on to its result. This class does both, so no hook touches the type arguments
    directly.

    It sits in the plugin layer. Each method takes or gives a mypy `Instance`.

    Example:
        >>> fixture = new_type_fixture()
        >>> name = literal_str_type(fixture, "price")
        >>> SeriesRef.build(name, int_dtype(fixture), new_plugin(fixture))
        astrolabe.frame.TypedSeries[Literal['price'], polars.datatypes.classes.Int64]
    """

    TARGET: Final[Target] = Target.of(TypedSeries)

    @classmethod
    def build(cls, name: MypyType, dtype: MypyType, plugin: PluginLookup) -> Instance | None:
        """Build the type of a series with a given name and data type.

        Args:
            name: The name of the series, as a type. It is usually a string literal type.
            dtype: The data type of the series.
            plugin: The mypy plugin, which finds the `TypedSeries` class.

        Returns:
            A `TypedSeries[name, dtype]` instance, or None if the plugin cannot find the class.
        """
        info = plugin.lookup_fully_qualified(cls.TARGET.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # TypedSeries lives in astrolabe.frame. The plugin adds that module to the
            # dependencies of every build, so the lookup always finds it.
            return None
        return Instance(info.node, [name, dtype])

    @classmethod
    def from_proper_type(cls, typ: MypyType) -> Instance | None:
        """Give a type as an instance if it is a typed series.

        Args:
            typ: The type to test.

        Returns:
            The `TypedSeries[...]` instance, or None for any other type.

        Example:
            >>> fixture = new_type_fixture()
            >>> name = literal_str_type(fixture, "price")
            >>> series = SeriesRef.build(name, int_dtype(fixture), new_plugin(fixture))
            >>> SeriesRef.from_proper_type(series)
            astrolabe.frame.TypedSeries[Literal['price'], polars.datatypes.classes.Int64]
            >>> SeriesRef.from_proper_type(int_dtype(fixture)) is None
            True
        """
        proper = get_proper_type(typ)
        if not cls.TARGET.matches(proper):
            return None
        return proper

    @classmethod
    def name(cls, instance: Instance) -> MypyType:
        """Give the name of a series as a type, to pass on to a result.

        The name can be a string literal type or any other type. This method gives it unchanged,
        so a result keeps whatever name its receiver had.

        Args:
            instance: A `TypedSeries[...]` instance, as `from_proper_type` gives it.

        Returns:
            The `Name` type argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> name = literal_str_type(fixture, "price")
            >>> series = SeriesRef.build(name, int_dtype(fixture), new_plugin(fixture))
            >>> SeriesRef.name(series)
            Literal['price']
        """
        return instance.args[0]

    @classmethod
    def literal_name(cls, instance: Instance) -> str | None:
        """Give the name of a series as a string, if it is a string literal.

        Args:
            instance: A `TypedSeries[...]` instance, as `from_proper_type` gives it.

        Returns:
            The name, or None if the `Name` type argument is not a string literal type.

        Example:
            >>> fixture = new_type_fixture()
            >>> name = literal_str_type(fixture, "price")
            >>> series = SeriesRef.build(name, int_dtype(fixture), new_plugin(fixture))
            >>> SeriesRef.literal_name(series)
            'price'
            >>> plugin = new_plugin(fixture)
            >>> non_literal = SeriesRef.build(int_dtype(fixture), int_dtype(fixture), plugin)
            >>> SeriesRef.literal_name(non_literal) is None
            True
        """
        proper = get_proper_type(cls.name(instance))
        if isinstance(proper, LiteralType) and isinstance(proper.value, str):
            return proper.value
        return None

    @classmethod
    def dtype(cls, instance: Instance) -> MypyType:
        """Give the data type of a series.

        Args:
            instance: A `TypedSeries[...]` instance, as `from_proper_type` gives it.

        Returns:
            The `DType` type argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> name = literal_str_type(fixture, "price")
            >>> series = SeriesRef.build(name, int_dtype(fixture), new_plugin(fixture))
            >>> SeriesRef.dtype(series)
            polars.datatypes.classes.Int64
        """
        return instance.args[1]
