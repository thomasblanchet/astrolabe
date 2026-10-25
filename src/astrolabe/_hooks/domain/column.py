"""Read and build one named, typed column, so that hooks never touch its raw mypy form."""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

from mypy.nodes import TypeInfo
from mypy.types import Instance, LiteralType, TupleType, get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.dtype_constructor_readers import CheckerLiteralTypeBuilder
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe.frame import Column as _ColumnMarker

if TYPE_CHECKING:
    from mypy.plugin import Plugin


class ColumnRef:
    """One column as a name and a data type, for hook code that reads or builds a schema.

    A column reaches a hook in two forms. The schema of a typed frame is a `TypedDict`, whose
    key is the name and whose value is the data type. Other types hold an ordered tuple of
    `Column[Literal[name], dtype]` items instead, for example the fields of a `Struct` or the
    key columns of a group-by. This class gives both forms the same `name` and `dtype`, so hook
    code does not care which form it has. It can also build the `Column[...]` form for code that
    must put a column into such a tuple.

    It sits in the plugin layer. It reads and builds mypy types, and the engine never sees it.

    Example:
        >>> fixture = new_type_fixture()
        >>> column = column_ref(fixture, "price", int_dtype(fixture))
        >>> column.name
        'price'
    """

    TARGET: Final[Target] = Target.of(_ColumnMarker)
    _LITERAL_TYPE: Final[CheckerLiteralTypeBuilder] = CheckerLiteralTypeBuilder()

    def __init__(
        self,
        instance: Instance | None = None,
        *,
        name: str | None = None,
        dtype: MypyType | None = None,
    ) -> None:
        """Wrap a `Column[...]` instance, or a name and a data type with no instance.

        Pass `instance` to wrap a `Column[...]` item. For a name and a data type, use
        `from_name_and_dtype`, which says what it builds.

        Args:
            instance: A `Column[...]` instance. Leave it out for a name and a data type.
            name: The name of the column, used only when there is no instance.
            dtype: The data type of the column, used only when there is no instance.
        """
        self._instance = instance
        self._name = name
        self._dtype = dtype

    @classmethod
    def from_name_and_dtype(cls, name: str, dtype: MypyType) -> ColumnRef:
        """Wrap a name and a data type, with no `Column[...]` instance.

        This is the form of a column in the schema of a typed frame. The schema is a
        `TypedDict`, so its key holds the name and its value holds the data type, and no
        `Column[...]` item is needed. A column built this way has no instance for `as_instance`
        to give. Use `materialized` if code needs one.

        Args:
            name: The name of the column.
            dtype: The data type of the column.

        Returns:
            The new column.

        Example:
            >>> fixture = new_type_fixture()
            >>> column = ColumnRef.from_name_and_dtype("price", int_dtype(fixture))
            >>> column.name
            'price'
        """
        return cls(name=name, dtype=dtype)

    @classmethod
    def from_proper_type(cls, typ: MypyType) -> ColumnRef | None:
        """Wrap a type as a column if it is a `Column[...]` instance.

        Args:
            typ: The type to test.

        Returns:
            The wrapped column, or None if the type is not a `Column[...]` instance.

        Example:
            >>> fixture = new_type_fixture()
            >>> column = column_ref(fixture, "price", int_dtype(fixture))
            >>> ColumnRef.from_proper_type(column.as_instance()).name
            'price'
            >>> ColumnRef.from_proper_type(int_dtype(fixture)) is None
            True
        """
        proper = get_proper_type(typ)
        if not cls.TARGET.matches(proper):
            return None
        return cls(proper)

    @property
    def name(self) -> str | None:
        """The name of the column, or None if the `Column[...]` item has no string literal name.

        Example:
            >>> fixture = new_type_fixture()
            >>> column_ref(fixture, "price", int_dtype(fixture)).name
            'price'
            >>> info = fixture.make_type_info(ColumnRef.TARGET.fullname)
            >>> bad = Instance(info, [int_dtype(fixture), int_dtype(fixture)])
            >>> ColumnRef(bad).name
        """
        if self._instance is None:
            return self._name
        name_type = get_proper_type(self._instance.args[0])
        if isinstance(name_type, LiteralType) and isinstance(name_type.value, str):
            return name_type.value
        return None

    @property
    def dtype(self) -> MypyType:
        """The data type of the column, as a mypy type."""
        if self._instance is None:
            return checked_not_none(self._dtype)
        return self._instance.args[1]

    def with_name(self, name: str, api: object) -> ColumnRef:
        """Give the same column under a new name, as for a rename.

        Args:
            name: The new name.
            api: The mypy plugin API, which builds the literal type of the new name if this
                column has a `Column[...]` instance.

        Returns:
            A new column with the new name and the same data type. It has a `Column[...]`
            instance if this column has one, and no instance if this column has none.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> column = column_ref(fixture, "price", int_dtype(fixture))
            >>> column.with_name("cost", api).name
            'cost'
        """
        if self._instance is None:
            return ColumnRef.from_name_and_dtype(name, self.dtype)
        name_type = self._LITERAL_TYPE.string(name, api)
        return ColumnRef(Instance(self._instance.type, [name_type, self._instance.args[1]]))

    @classmethod
    def build(cls, name: str, dtype: MypyType, model: ColumnRef, api: object) -> ColumnRef:
        """Build a new column in the same form as a column that the code already holds.

        An operation such as `with_columns` adds columns that the user did not declare. The new
        column takes the form of `model`. If `model` has a `Column[...]` instance, the new one
        gets an instance of the same class. That reuses the class of `model`, so there is no
        lookup by name. If `model` has no instance, the new column has none either.

        Args:
            name: The name of the new column.
            dtype: The data type of the new column.
            model: A column from the same frame or tuple.
            api: The mypy plugin API, which builds the literal type of the name if `model` has
                an instance.

        Returns:
            The new column.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> tax = ColumnRef.build("tax", str_dtype(fixture), price, api)
            >>> tax.name
            'tax'
        """
        if model._instance is None:
            return cls.from_name_and_dtype(name, dtype)
        name_type = cls._LITERAL_TYPE.string(name, api)
        return cls(Instance(model._instance.type, [name_type, dtype]))

    @classmethod
    def build_fresh(
        cls,
        name: str,
        dtype: MypyType,
        api: object,
        plugin: Plugin,
    ) -> ColumnRef | None:
        """Build a `Column[...]` instance when there is no column to copy its class from.

        This looks up the `Column` class through the plugin. Use it only when no column with an
        instance is at hand. If there is one, use `build`, which reuses its class.

        Args:
            name: The name of the new column.
            dtype: The data type of the new column.
            api: The mypy plugin API, which builds the literal type of the name.
            plugin: The mypy plugin, which finds the `Column` class.

        Returns:
            The new column, or None if the plugin cannot find the `Column` class.
        """
        info = plugin.lookup_fully_qualified(cls.TARGET.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # Column lives in astrolabe.frame, with TypedDataFrame and TypedLazyFrame. The plugin
            # adds that module to the dependencies of every build, so the lookup always finds it.
            return None
        name_type = cls._LITERAL_TYPE.string(name, api)
        return cls(Instance(info.node, [name_type, dtype]))

    def as_instance(self) -> Instance:
        """Return the `Column[...]` instance that this column wraps.

        Call it only on a column that has an instance. A column read from the schema of a
        typed frame has none, and a call then raises `TypeError`. Use `materialized` for such a
        column.
        """
        return checked_not_none(self._instance)

    def materialized(self, api: object, plugin: Plugin) -> ColumnRef | None:
        """Return this column with a `Column[...]` instance, and build one if it has none.

        Code that puts a column into a tuple of `Column[...]` items needs the instance. Examples
        are the fields of a `Struct` and the key columns of a group-by. A column read from the
        schema of a typed frame has no instance, so this method builds one.

        Args:
            api: The mypy plugin API, which builds the literal type of the name.
            plugin: The mypy plugin, which finds the `Column` class.

        Returns:
            This column if it has an instance. Otherwise, a new column with an instance. None
            if the column has no name, or if the plugin cannot find the `Column` class.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> plugin = new_plugin(fixture)
            >>> column = ColumnRef.from_name_and_dtype("price", int_dtype(fixture))
            >>> column.materialized(api, plugin).name
            'price'
            >>> already_built = column_ref(fixture, "price", int_dtype(fixture))
            >>> already_built.materialized(api, plugin) is already_built
            True
            >>> nameless = ColumnRef(dtype=int_dtype(fixture))
            >>> nameless.materialized(api, plugin) is None
            True
        """
        if self._instance is not None:  # pragma: no cover -- a column of a schema holds no instance
            return self
        name = self.name
        if name is None:  # pragma: no cover -- a key column has a name
            return None
        return ColumnRef.build_fresh(name, self.dtype, api, plugin)

    @classmethod
    @propagates(None)
    def from_tuple(cls, tuple_type: TupleType) -> list[ColumnRef] | None:
        """Wrap each item of a tuple of `Column[...]` items.

        Some types hold their columns as a tuple type argument, for example the fields of a
        `Struct` or the key columns of a group-by. This reads them all.

        Args:
            tuple_type: A tuple type whose items are `Column[...]` instances.

        Returns:
            The columns in order, or None if one item is not a `Column[...]` instance.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> items = columns_tuple_type(fixture, [price.as_instance()])
            >>> [column.name for column in ColumnRef.from_tuple(items)]
            ['price']
        """
        result: list[ColumnRef] = []
        for item in tuple_type.items:
            result.append(not_none(cls.from_proper_type(item)))
        return result

    @classmethod
    def tuple_argument(cls, instance: Instance, index: int) -> TupleType | None:
        """Give one type argument of an instance if it is a tuple type.

        Some markers hold a tuple type as an argument, for example the `Fields` argument of
        `Struct[Fields]`. Code that reads the tuple its own way starts here. `from_argument`
        then wraps each item as a column.

        Args:
            instance: The instance to read.
            index: The position of the type argument.

        Returns:
            The type argument as a `TupleType`, or None if it is not a tuple type.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> struct = struct_instance(fixture, price)
            >>> ColumnRef.tuple_argument(struct, 0).items
            [astrolabe.frame.Column[Literal['price'], polars.datatypes.classes.Int64]]
            >>> ColumnRef.tuple_argument(price.as_instance(), 1) is None
            True
        """
        tuple_type = get_proper_type(instance.args[index])
        if not isinstance(tuple_type, TupleType):
            return None
        return tuple_type

    @classmethod
    @propagates(None)
    def from_argument(cls, instance: Instance, index: int) -> list[ColumnRef] | None:
        """Read one type argument of an instance as a tuple of columns.

        For example, this reads the fields of `Struct[Fields]` as a list of columns.

        Args:
            instance: The instance to read.
            index: The position of the type argument.

        Returns:
            The columns in order, or None if the type argument is not a tuple of `Column[...]`
            instances.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> struct = struct_instance(fixture, price)
            >>> [column.name for column in ColumnRef.from_argument(struct, 0)]
            ['price']
        """
        tuple_type = not_none(cls.tuple_argument(instance, index))
        return cls.from_tuple(tuple_type)
