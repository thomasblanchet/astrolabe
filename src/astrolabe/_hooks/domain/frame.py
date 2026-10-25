"""Read and build the mypy type of a typed frame, so that hooks never touch its raw form.

A typed frame is a `TypedDataFrame` or a `TypedLazyFrame`. Its mypy type holds two arguments: a
`TypedDict` of its declared columns, and a slot for its remainder. The remainder is the set of
columns that the type does not name. Every frame hook reads that type and builds a new one.
`Frame` does both jobs in one place, so a change to the encoding touches one class.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import Final, Protocol, TYPE_CHECKING

import polars as pl
from mypy.nodes import TypeInfo
from mypy.maptype import map_instance_to_supertype
from mypy.subtypes import is_same_type
from mypy.types import Instance, NoneType, TupleType, TypedDictType, get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.planned import PlannedType
from astrolabe.frame import TypedDataFrame, TypedLazyFrame

if TYPE_CHECKING:
    from collections.abc import Sequence
    from mypy.nodes import SymbolTableNode


class PluginLookup(Protocol):
    """The one service of a mypy `Plugin` that code which builds a type needs: find a class.

    To build a type, code often needs a class that it does not hold yet, for example
    `TypedLazyFrame` when it turns a `TypedDataFrame` lazy. It asks the plugin with
    `lookup_fully_qualified`. The real mypy `Plugin` matches this protocol with no change. A
    doctest can pass a small stand-in instead, and so it needs no mypy build.
    """

    def lookup_fully_qualified(self, fullname: str) -> SymbolTableNode | None: ...


@dataclass(frozen=True)
class KnownColumns:
    """The columns that a frame declares, next to the remainder that it does not.

    An open frame has a remainder: columns that its type does not name. A hook can still check
    an operation that touches only named columns, because the remainder cannot change whether a
    named column exists or what its data type is. This class gives such a hook the declared
    columns and keeps the remainder aside, raw, for the code that reads it.

    A schema has no order. It is a set of names, each with a data type. No operation in this
    project finds a column by its position, only by its name.

    Attributes:
        columns: The columns that the frame's type names.
        rest: The remainder slot of the frame, raw and not decoded, or None if the frame is
            closed. The encoding codecs decode it.
    """

    columns: list[ColumnRef]
    rest: MypyType | None

    def by_name(self) -> dict[str, ColumnRef]:
        """Look up the declared columns by name.

        Returns:
            The declared columns, keyed by name. A column with no literal name is left out.
        """
        return {column.name: column for column in self.columns if column.name is not None}


class Frame:
    """Read the columns of a typed frame, and build the typed frame that an operation gives.

    Hook code works with a `Frame` and never with the raw mypy `Instance`. A frame's type holds
    its columns in a `TypedDict` and its remainder in a second slot. Only this class knows that
    layout, so a change to it touches no hook. This class reads the columns, builds a new frame
    of the same kind, and turns a frame lazy, eager or plain.

    It sits in the plugin layer. It reads a mypy `Instance` and answers `ColumnRef` values and
    new mypy types. The engine never sees it.

    Example:
        >>> fixture = new_type_fixture()
        >>> price = column_ref(fixture, "price", int_dtype(fixture))
        >>> built = frame(fixture, [price.as_instance()])
        >>> [column.name for column in built.columns]
        ['price']
    """

    TARGETS: Final[frozenset[Target]] = frozenset({
        Target.of(TypedDataFrame),
        Target.of(TypedLazyFrame),
    })

    _OTHER_KIND_TARGET: Final[dict[Target, Target]] = {
        Target.of(TypedDataFrame): Target.of(TypedLazyFrame),
        Target.of(TypedLazyFrame): Target.of(TypedDataFrame),
    }

    def __init__(self, instance: Instance) -> None:
        """Wrap the mypy type of a typed frame.

        Args:
            instance: A `TypedDataFrame` or `TypedLazyFrame` instance.
        """
        self._instance = instance

    @classmethod
    def from_proper_type(cls, typ: MypyType) -> Frame | None:
        """Wrap a type as a `Frame` if it is a typed frame.

        A hook reads the type of its receiver or its argument from mypy. This step tells a
        typed frame from every other type, for example a plain frame.

        A class that a user derives from a typed frame is read through its typed base: the
        frame that it holds is the frame of the base.

        Args:
            typ: The type to test.

        Returns:
            The wrapped frame, or None if the type is not a `TypedDataFrame` or a
            `TypedLazyFrame` instance, or an instance of a class derived from one.
        """
        proper = get_proper_type(typ)
        if not isinstance(proper, Instance):
            return None
        base = cls._typed_base(proper.type)
        if base is None:
            return None
        return cls(map_instance_to_supertype(proper, base))

    @classmethod
    def _typed_base(cls, info: TypeInfo) -> TypeInfo | None:
        """Give the first class in the order of bases that is a typed frame.

        Args:
            info: The class of the type.

        Returns:
            The class itself if it is a typed frame, its typed base, or None.
        """
        return next((klass for klass in info.mro if Target(klass.fullname) in cls.TARGETS), None)

    @property
    def columns(self) -> list[ColumnRef] | None:
        """The columns of a closed frame, or None for any other frame.

        A closed frame has no remainder, so this list is all of its columns. The value is None
        if the frame has a remainder, or if its type has no `TypedDict` of columns. Use
        `known_columns` to read the declared columns of an open frame.

        Example:
            >>> fixture = new_type_fixture()
            >>> Frame(bare_frame_instance(fixture)).columns is None
            True
        """
        columns_dict = self._columns_dict()
        if columns_dict is None:
            return None
        items = self.schema_items()
        if items:
            return None
        return self._columns_from(columns_dict)

    def _columns_from(self, columns_dict: dict[str, MypyType]) -> list[ColumnRef]:
        """Wrap each entry of the columns `TypedDict` as a column.

        Args:
            columns_dict: The declared columns of the frame, keyed by name.

        Returns:
            One column for each entry, in the order of `columns_dict`.
        """
        return list(starmap(ColumnRef.from_name_and_dtype, columns_dict.items()))

    @property
    def known_columns(self) -> KnownColumns | None:
        """The declared columns of the frame, open or closed, with its raw remainder.

        Use this for a frame that can have a remainder, for example a parameter of a function
        that is generic over its remainder. An operation that touches only named columns stays
        safe to check. Whatever the remainder holds, it cannot change whether a named column
        exists or what its data type is.

        The value is None only if the type of the frame has no `TypedDict` of columns. A
        remainder alone never makes it None.
        """
        columns_dict = self._columns_dict()
        if columns_dict is None:
            return None
        items = self.schema_items()
        if items is None:  # pragma: no cover
            # schema_items() gives None if the columns or the remainder slot are missing. The
            # check above excludes the first, and mypy gives every typed frame both arguments.
            return None
        rest = items[0] if items else None
        return KnownColumns(self._columns_from(columns_dict), rest)

    NO_REST: Final[NoneType] = NoneType()
    """The value in the remainder slot of a closed frame, which has no remainder.

    The type of a frame always has a remainder slot. A closed frame holds
    the mypy `None` type there. A marker class of its own needs a class
    lookup to build, and the `None` type does not. No remainder codec reads a
    `NoneType` as a remainder, so this value never looks like a real one.
    """

    def rebuilt(
        self,
        columns: Sequence[ColumnRef],
        api: object,
        rest: MypyType | None = None,
    ) -> Frame:
        """Build the frame that an operation gives, of the same kind as this frame.

        A frame operation keeps the kind of its receiver: a `TypedDataFrame` gives a
        `TypedDataFrame`. The hook gives the new columns, and the new remainder if one
        survives. This method puts them in a new frame type.

        Args:
            columns: The columns of the result, in order.
            api: The mypy plugin API, which builds the `TypedDict` of the columns.
            rest: The remainder of the result, already encoded. Leave it out for a closed
                result.

        Returns:
            A new frame of the same kind, with `columns` and `rest` as its schema.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> built = frame(fixture, [price.as_instance()])
            >>> tax = ColumnRef.build("tax", str_dtype(fixture), price, api)
            >>> rebuilt = built.rebuilt([price, tax], api)
            >>> [column.name for column in rebuilt.columns]
            ['price', 'tax']
        """
        columns_dict = self._kept_schema(self.object_typeddict(columns, api))
        rest_type = self._rest_type(rest)
        return Frame(Instance(self._instance.type, [columns_dict, rest_type]))

    def _kept_schema(self, built: MypyType) -> MypyType:
        """Keep the columns argument of the receiver if the result has the same columns.

        The columns argument of the receiver can be a class that the user named. An error that
        names that class reads better than one that lists its columns again. If the new columns
        equal the old ones, the result keeps the old argument. It is the same type, and it
        keeps the name.

        Args:
            built: The new columns argument of the result.

        Returns:
            The argument of the receiver if the two are the same type, and `built` if not.
        """
        own = self._instance.args[0]
        if is_same_type(get_proper_type(own), get_proper_type(built)):
            return own
        return built

    @classmethod
    def build_fresh(
        cls,
        target: Target,
        columns: Sequence[ColumnRef],
        plugin: PluginLookup,
        api: object,
        rest: MypyType | None = None,
    ) -> Frame | None:
        """Build a new typed frame when there is no typed frame to start from.

        Some operations give a typed frame with no typed receiver, for example a function that
        reads a file with a declared schema. Such code uses this method. If a typed receiver
        exists, use `rebuilt`, which keeps the name of the receiver's columns class.

        Args:
            target: The kind of frame to build, `TypedDataFrame` or `TypedLazyFrame`.
            columns: The columns of the new frame, in order.
            plugin: The mypy plugin, which finds the class of `target`.
            api: The mypy plugin API, which builds the `TypedDict` of the columns.
            rest: The remainder of the new frame, already encoded. Leave it out for a closed
                frame.

        Returns:
            A new frame of the kind `target`, with `columns` and `rest` as its schema. None if
            the plugin cannot find the class of `target`.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> plugin = new_plugin(fixture)
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> built = Frame.build_fresh(Target.of(TypedDataFrame), [price], plugin, api)
            >>> [column.name for column in built.columns]
            ['price']
        """
        info = plugin.lookup_fully_qualified(target.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # TypedDataFrame and TypedLazyFrame live in astrolabe.frame. The plugin adds that
            # module to the dependencies of every build, so the lookup always finds them.
            return None
        columns_dict = cls.object_typeddict(columns, api)
        rest_type = cls._rest_type(rest)
        return cls(Instance(info.node, [columns_dict, rest_type]))

    @staticmethod
    def _rest_type(rest: MypyType | None) -> MypyType:
        """Give the value for the remainder slot, with `NO_REST` for a closed frame.

        It is a static method because `build_fresh`, a class method, calls it too, and a class
        method has no instance.

        Args:
            rest: The encoded remainder, or None for a closed frame.

        Returns:
            `rest` itself, or `Frame.NO_REST`.
        """
        if rest is None:
            return Frame.NO_REST
        return rest

    def other_kind(self, plugin: PluginLookup) -> Frame | None:
        """Build the same frame as the other kind, for `.lazy()` and `.collect()`.

        A `TypedDataFrame` becomes a `TypedLazyFrame`, and a `TypedLazyFrame` becomes a
        `TypedDataFrame`. The columns and the remainder stay the same, even if the remainder is
        still a type variable.

        Args:
            plugin: The mypy plugin, which finds the class of the other kind.

        Returns:
            A frame of the other kind with the same type arguments. None if the kind of this
            frame is not known, or if the plugin cannot find the other class.
        """
        other_target = self._OTHER_KIND_TARGET.get(Target(self._instance.type.fullname))
        if other_target is None:  # pragma: no cover
            # _OTHER_KIND_TARGET has the same two full names as TARGETS, so the full name of a
            # Frame is always one of its keys.
            return None
        info = plugin.lookup_fully_qualified(other_target.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # TypedDataFrame and TypedLazyFrame live in the same module. mypy already loaded it
            # when this Frame came into existence.
            return None
        return Frame(Instance(info.node, list(self._instance.args)))

    def collected(self) -> Frame:
        """Give the frame with the type of the data of each column, as `.collect()` gives it.

        A column of a lazy frame holds `Planned[run, planner]` where polars
        stores another type than the type of its data. An eager frame holds
        its data, so each such column takes its run type.

        Returns:
            The frame. The frame itself where it has no `TypedDict` of columns.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> collected = frame(fixture, [price.as_instance()]).collected()
            >>> [column.name for column in collected.columns]
            ['price']
            >>> bare = Frame(bare_frame_instance(fixture))
            >>> bare.collected() is bare
            True
        """
        columns_arg = get_proper_type(self._instance.args[0]) if self._instance.args else None
        if not isinstance(columns_arg, TypedDictType):
            return self
        planned = PlannedType()
        types = [planned.run(typ) for typ in columns_arg.items.values()]
        schema = columns_arg.copy_modified(item_types=types)
        return Frame(self._instance.copy_modified(args=[schema, *self._instance.args[1:]]))

    _PLAIN_KIND_TARGET: Final[dict[Target, Target]] = {
        Target.of(TypedDataFrame): Target.of(pl.DataFrame),
        Target.of(TypedLazyFrame): Target.of(pl.LazyFrame),
    }

    def as_plain(self, plugin: PluginLookup) -> MypyType | None:
        """Give the plain polars frame of the same kind, for a hook that falls back.

        If a hook cannot find the result of an operation, the real result can add, delete or
        change the data type of any column. To keep the old columns is then wrong. The
        hook falls back instead: it gives the plain `pl.DataFrame` or `pl.LazyFrame`, which the
        checker does not track. That is also the type of the same call with no plugin.

        Args:
            plugin: The mypy plugin, which finds the class of the plain frame.

        Returns:
            A `pl.DataFrame` or a `pl.LazyFrame` instance, of the same kind as this frame. None
            if the kind of this frame is not known, or if the plugin cannot find the class.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> built = frame(fixture, [price.as_instance()])
            >>> built.as_plain(new_plugin(fixture))
            polars.dataframe.frame.DataFrame
        """
        plain_target = self._PLAIN_KIND_TARGET.get(Target(self._instance.type.fullname))
        if plain_target is None:  # pragma: no cover
            # _PLAIN_KIND_TARGET has the same two full names as TARGETS, so the full name of a
            # Frame is always one of its keys.
            return None
        info = plugin.lookup_fully_qualified(plain_target.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # mypy always loads polars.DataFrame and polars.LazyFrame before a Frame exists
            # that can fall back to them.
            return None
        return Instance(info.node, [])

    @staticmethod
    def object_typeddict(columns: Sequence[ColumnRef], api: object) -> TypedDictType:
        """Build the `TypedDict` that holds the columns of a typed frame.

        The first type argument of every typed frame is this `TypedDict`. Each key is a column
        name, and each value is the data type of that column. A `TypedDict` has no order, so
        two frames with the same columns in a different order are the same type. That matches
        a schema, which has no order either.

        Args:
            columns: The columns to put in the `TypedDict`.
            api: The mypy plugin API, which builds the fallback type of the `TypedDict`.

        Returns:
            A closed `TypedDict` type with one required key for each column.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> sorted(Frame.object_typeddict([price], api).items)
            ['price']
        """
        fallback = api.named_generic_type("typing._TypedDict", [])  # type: ignore[attr-defined]
        named = [(checked_not_none(column.name), column.dtype) for column in columns]
        items: dict[str, MypyType] = dict(named)
        return TypedDictType(
            items,
            required_keys=set(items),
            readonly_keys=set(),
            fallback=fallback,
            is_closed=True,
        )

    @staticmethod
    def object_tuple(items: Sequence[MypyType], api: object) -> TupleType:
        """Build a tuple type over some items, with `tuple[object, ...]` as its fallback.

        Some types hold an ordered list of column items as one type argument, for example the
        key columns of a group-by, or the fields of a `Struct`. This method builds that
        argument.

        Args:
            items: The items of the tuple, in order.
            api: The mypy plugin API, which builds the fallback type of the tuple.

        Returns:
            A `TupleType` over `items`.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> len(Frame.object_tuple([price.as_instance()], api).items)
            1
        """
        tuple_fallback = api.named_generic_type(  # type: ignore[attr-defined]
            "builtins.tuple",
            [api.named_generic_type("builtins.object", [])],  # type: ignore[attr-defined]
        )
        return TupleType(list(items), fallback=tuple_fallback)

    def as_instance(self) -> Instance:
        """Return the wrapped mypy type, for a hook that hands a type back to mypy."""
        return self._instance

    def schema_items(self) -> list[MypyType] | None:
        """Read the remainder slot of the frame, raw and not decoded.

        Only the remainder can hold a type variable that a call binds later. A declared column
        is a key of the `TypedDict`, and it never can. Most code wants `columns` or
        `known_columns`. Use this method to reach the raw remainder itself.

        The second type argument of a typed frame is the remainder slot. It holds the encoded
        remainder, or `NO_REST` for a closed frame.

        Returns:
            A list with the one encoded remainder, or an empty list for a closed frame. None if
            the type of the frame has no `TypedDict` of columns or no remainder slot.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> frame(fixture, [price.as_instance()]).schema_items()
            []
            >>> open_frame = frame(fixture, [price.as_instance(), declared_rest(fixture)])
            >>> len(open_frame.schema_items())
            1
            >>> columns_dict = columns_typeddict_type(fixture, [price.as_instance()])
            >>> frame_info = fixture.make_type_info(Target.of(TypedDataFrame).fullname)
            >>> one_arg = Frame(Instance(frame_info, [columns_dict]))
            >>> one_arg.schema_items() is None
            True
        """
        if self._columns_dict() is None:
            return None
        if len(self._instance.args) < 2:
            return None
        candidate = get_proper_type(self._instance.args[1])
        if isinstance(candidate, NoneType):
            return []
        return [candidate]

    def _columns_dict(self) -> dict[str, MypyType] | None:
        """Read the entries of the `TypedDict` in the first type argument.

        Returns:
            The declared columns, keyed by name. None if the first type argument is missing or
            is not a `TypedDict`.

        Example:
            >>> fixture = new_type_fixture()
            >>> frame_info = fixture.make_type_info(Target.of(TypedDataFrame).fullname)
            >>> malformed = Frame(Instance(frame_info, [declared_rest(fixture)]))
            >>> malformed._columns_dict() is None
            True
        """
        if not self._instance.args:
            return None
        columns_arg = get_proper_type(self._instance.args[0])
        if not isinstance(columns_arg, TypedDictType):
            return None
        return columns_arg.items
