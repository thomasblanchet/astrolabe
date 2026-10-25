"""Build the marker types that every codec writes, and read their parts back.

A codec writes an engine value as a mypy type made of marker classes. A marker
class is a small generic class in a public module of astrolabe, for example
`Both`, and its type arguments hold the parts of the value. The leaves
are literal types: a column name is `Literal['price']`, a width is `Literal[2]`.

This module holds the few steps that every codec repeats. `MarkerMinter` finds a
marker class in mypy and builds a type of it. `ProperReads` reads such a type
back apart. `NameSetCodec` writes a set of names in the one form that ignores
order and repeats, so that the same set never gives two different types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.types import (
    Instance,
    LiteralType,
    TupleType,
    UninhabitedType,
    UnionType,
    get_proper_type,
)
from mypy.types import Type as MypyType

from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe.expr import Blank

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._hooks.calls.dtype_constructor_readers import LiteralTypeBuilder
    from astrolabe._hooks.domain.frame import PluginLookup


class MarkerMinter:
    """Build a mypy type of one marker class, or of one literal value, for a codec.

    Every type that a codec writes comes from here, so this class is the one
    place where the encoding turns a Python class into a mypy `Instance`. It
    takes the marker as the real Python class and derives its full name from it.
    A hand-typed name cannot then go out of step with a rename.

    mypy runs the plugin in two phases: semantic analysis, which reads
    declarations, and type checking, which checks calls. A literal type needs an
    object from the plugin API of the current phase, so one minter serves one
    phase. The caller gives it the literal builder and the API object of that
    phase.

    Example:
        >>> from astrolabe.remainder import Matching
        >>> fixture = new_type_fixture()
        >>> minter = new_marker_minter(fixture)
        >>> minter.marker(Matching, [minter.string("0[.>0]")])
        astrolabe.remainder.Matching[Literal['0[.>0]']]
    """

    def __init__(
        self,
        plugin: PluginLookup,
        literals: LiteralTypeBuilder,
        api: object,
    ) -> None:
        """Keep what the minter needs to find classes and build literals in one phase.

        Args:
            plugin: The mypy plugin, which finds a class by its full name.
            literals: The literal builder that matches the phase.
            api: The plugin API of the phase. A literal type needs it to build
                its `str` or `int` fallback.
        """
        self._types = FreshTypeInfo(plugin)
        self._literals = literals
        self._api = api

    def marker(self, marker: type, args: Sequence[MypyType]) -> Instance | None:
        """Build a type of one marker class, with the given type arguments.

        Args:
            marker: The marker class, as the real Python object, so that its
                full name comes from the class itself.
            args: The type arguments, in order.

        Returns:
            The type, or None if mypy did not load the module of the marker. The
            plugin adds every marker module to each build, so a real run always
            gets the type.

        Example:
            >>> from astrolabe.remainder import Both
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> one = minter.resolved("kept.R")
            >>> minter.marker(Both, [one, one])
            astrolabe.remainder.Both[kept.R, kept.R]
        """
        return self.instance(Target.of(marker).fullname, args)

    def instance(self, fullname: str, args: Sequence[MypyType]) -> Instance | None:
        """Build a type of the class with the given full name.

        A codec uses this for a class that it knows only by name, for example a
        polars data type class that it found from a name in the engine.

        Args:
            fullname: The full name of the class in mypy, for example
                `polars.datatypes.classes.Int64`.
            args: The type arguments, in order.

        Returns:
            The type, or None if the name does not lead to a class.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> minter.instance("kept.R", [])
            kept.R
        """
        info = self._types.resolved(fullname)
        if info is None:  # pragma: no cover
            # A codec asks only for a marker class or a declaration, and
            # the plugin adds the module of each one to every build. The
            # lookup therefore always finds the class.
            return None
        return Instance(info, list(args))

    def tuple_of(self, items: Sequence[MypyType]) -> TupleType | None:
        """Build a tuple type over the given items.

        A `Struct[...]` or an `Enum[...]` marker holds its fields or its
        categories in one tuple argument, because that is the form mypy gives
        such a declaration. A minted struct or enum uses the same form, so that
        it is equal to the declared one.

        Args:
            items: The items of the tuple, in order.

        Returns:
            The tuple type, or None if mypy did not load `builtins`. That does
            not occur in a real run.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> minter.tuple_of([minter.string("a")])
            tuple[Literal['a']]
        """
        item = self.instance("builtins.object", [])
        if item is None:  # pragma: no cover
            # mypy loads `builtins` in every build.
            return None
        fallback = self.instance("builtins.tuple", [item])
        if fallback is None:  # pragma: no cover
            # mypy loads `builtins` in every build.
            return None
        return TupleType(list(items), fallback=fallback)

    def resolved(self, fullname: str) -> Instance | None:
        """Build a type of the class with the given full name, with no type arguments.

        Args:
            fullname: The full name of the class in mypy.

        Returns:
            The type, or None if the name does not lead to a class.

        Example:
            >>> fixture = new_type_fixture()
            >>> new_marker_minter(fixture).resolved("kept.R")
            kept.R
        """
        return self.instance(fullname, [])

    def string(self, value: str) -> LiteralType:
        """Build a string literal type, for example `Literal['price']`.

        Args:
            value: The string that the literal holds.

        Returns:
            The literal type.

        Example:
            >>> fixture = new_type_fixture()
            >>> new_marker_minter(fixture).string("_prev")
            Literal['_prev']
        """
        return self._literals.string(value, self._api)

    def name(self, value: str) -> MypyType:
        """Build the type that holds one column name.

        The empty name stands for the blank: the column name that a
        `some_column` declaration leaves for each call to fill in. It is written
        as the `Blank` marker, so that no message shows an empty literal.

        Args:
            value: The column name, or the empty string for the blank.

        Returns:
            The string literal type, or the `Blank` marker for the empty name.

        Example:
            >>> fixture = new_type_fixture()
            >>> new_marker_minter(fixture).name("price")
            Literal['price']
            >>> new_marker_minter(fixture).name("")
            astrolabe.expr.Blank
        """
        if value:
            return self.string(value)
        return checked_not_none(self.marker(Blank, []))

    def integer(self, value: int) -> LiteralType:
        """Build an integer literal type, for example `Literal[2]`.

        Args:
            value: The integer that the literal holds.

        Returns:
            The literal type.

        Example:
            >>> fixture = new_type_fixture()
            >>> new_marker_minter(fixture).integer(2)
            Literal[2]
        """
        return self._literals.integer(value, self._api)


class ProperReads:
    """Read the parts of a mypy type that the codecs need to take apart.

    mypy can wrap a type in an alias that must be expanded before its shape is
    visible. Each read here expands the type first, and then answers a plain
    Python value, or None if the type has a different shape. The decoders use
    this class so that none of them repeats that step. It builds no type, so it
    needs no access to the plugin.

    Example:
        >>> ProperReads().instance(UninhabitedType()) is None
        True
    """

    def instance(self, typ: MypyType) -> Instance | None:
        """Read a type as an instance of a class.

        Args:
            typ: The type to read.

        Returns:
            The instance, or None if the type is not an instance.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> ProperReads().instance(minter.resolved("kept.R"))
            kept.R
        """
        proper = get_proper_type(typ)
        if isinstance(proper, Instance):
            return proper
        return None

    def union_items(self, typ: MypyType) -> list[MypyType]:
        """Split a union type into its items.

        Args:
            typ: The type to split.

        Returns:
            The items of the union. A type that is not a union gives a list with
            the type alone.

        Example:
            >>> from mypy.types import UninhabitedType
            >>> len(ProperReads().union_items(UninhabitedType()))
            1
        """
        proper = get_proper_type(typ)
        if isinstance(proper, UnionType):
            return list(proper.items)
        return [typ]

    def string_literal(self, typ: MypyType) -> str | None:
        """Read a type as a string literal.

        Args:
            typ: The type to read.

        Returns:
            The string that the literal holds, or None if the type is not a
            string literal.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> ProperReads().string_literal(minter.string("_p"))
            '_p'
            >>> ProperReads().string_literal(minter.resolved("kept.R")) is None
            True
        """
        proper = get_proper_type(typ)
        if isinstance(proper, LiteralType) and isinstance(proper.value, str):
            return proper.value
        return None

    def name(self, typ: MypyType) -> str | None:
        """Read a type as a column name, and read the `Blank` marker as the empty name.

        This is the reverse of `MarkerMinter.name`.

        Args:
            typ: The type to read.

        Returns:
            The column name, the empty string for the `Blank` marker, or None
            for a type of another shape.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> ProperReads().name(minter.name(""))
            ''
            >>> ProperReads().name(minter.name("a"))
            'a'
        """
        instance = self.instance(typ)
        if instance is not None and self.matches(instance, Blank):
            return ""
        return self.string_literal(typ)

    def int_literal(self, typ: MypyType) -> int | None:
        """Read a type as an integer literal.

        A literal `True` or `False` is not read as an integer, although a Python
        `bool` is an `int`. A declaration that gives a `bool` as a width, a
        precision or a scale is an error, so a data type that holds one is
        not well formed, and this reader agrees with that check.

        Args:
            typ: The type to read.

        Returns:
            The integer that the literal holds, or None if the type is not an
            integer literal.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> ProperReads().int_literal(minter.integer(3))
            3
            >>> ProperReads().int_literal(minter.string("3")) is None
            True
        """
        proper = get_proper_type(typ)
        if isinstance(proper, LiteralType) and type(proper.value) is int:
            return proper.value
        return None

    def matches(self, instance: Instance, marker: type) -> bool:
        """Tell whether an instance is of the given marker class.

        The test compares full names, which the marker class itself supplies.

        Args:
            instance: The instance to test.
            marker: The marker class, as the real Python object.

        Returns:
            True if the instance is of that class.

        Example:
            >>> from astrolabe.remainder import Both
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> one = minter.marker(Both, [])
            >>> ProperReads().matches(one, Both)
            True
        """
        return instance.type.fullname == Target.of(marker).fullname


class NameSetCodec:
    """Write a set of column names as one type, and read it back.

    Many markers hold a set of names, for example the names that a remainder
    lacks. The remainder is the part of an open frame that its type does not
    name. A set has no order and no repeats, and its type must have none either.
    If one set has two spellings, mypy sees two different types.

    A union of string literals gives that property. The codec sorts the names,
    and mypy collapses the union into one of three forms: `Never` for the empty
    set, one literal for one name, and the union for several names. Both
    directions handle all three forms.

    Example:
        >>> fixture = new_type_fixture()
        >>> codec = NameSetCodec(new_marker_minter(fixture))
        >>> codec.built(frozenset({"b", "a"}))
        Literal['a'] | Literal['b']
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the minter that builds the literals.

        Args:
            minter: The marker minter for the current phase.
        """
        self._minter = minter

    def built(self, names: frozenset[str]) -> MypyType:
        """Write one set of names as its type.

        Args:
            names: The names to write.

        Returns:
            `Never` for the empty set, one literal for one name, or the union of
            the sorted literals.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = NameSetCodec(new_marker_minter(fixture))
            >>> codec.built(frozenset())
            Never
            >>> codec.built(frozenset({"a"}))
            Literal['a']
        """
        literals = [self._minter.string(name) for name in sorted(names)]
        if not literals:
            return UninhabitedType()
        if len(literals) == 1:
            return literals[0]
        return UnionType(literals)

    def read(self, typ: MypyType) -> frozenset[str] | None:
        """Read a type back into its set of names.

        Args:
            typ: The type to read.

        Returns:
            The names, the empty set for `Never`, or None if the type does not
            hold a set of names.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = NameSetCodec(new_marker_minter(fixture))
            >>> sorted(codec.read(codec.built(frozenset({"b", "a"}))))
            ['a', 'b']
            >>> codec.read(UninhabitedType())
            frozenset()
        """
        return NameSetReads().read(typ)


class NameSetReads:
    """Read a set of column names from its type, with no access to the plugin.

    This is the reading half of `NameSetCodec`. Code that only reads types, and
    so has no access to the plugin, uses it alone.

    Example:
        >>> NameSetReads().read(UninhabitedType())
        frozenset()
    """

    def read(self, typ: MypyType) -> frozenset[str] | None:
        """Read a type back into its set of names.

        A union reads each item as one name.

        Args:
            typ: The type to read.

        Returns:
            The names, the empty set for `Never`, or None if the type, or one
            item of the union, is not a string literal.

        Example:
            >>> from mypy.types import UnionType
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> codec = NameSetCodec(minter)
            >>> NameSetReads().read(codec.built(frozenset({"a"})))
            frozenset({'a'})
            >>> sorted(NameSetReads().read(codec.built(frozenset({"a", "b"}))))
            ['a', 'b']
            >>> mixed = UnionType([minter.string("a"), minter.resolved("kept.R")])
            >>> NameSetReads().read(mixed) is None
            True
        """
        proper = get_proper_type(typ)
        if isinstance(proper, UninhabitedType):
            return frozenset()
        if not isinstance(proper, UnionType):
            return self._one_name(proper)
        names: set[str] = set()
        for item in proper.items:
            one = self._one_name(get_proper_type(item))
            if one is None:
                return None
            names |= one
        return frozenset(names)

    def _one_name(self, proper: object) -> frozenset[str] | None:
        """Read one string literal as a set with one name.

        Args:
            proper: The type to read, already expanded.

        Returns:
            The set with that name, or None if the type is not a string literal.

        Example:
            >>> fixture = new_type_fixture()
            >>> built = NameSetCodec(new_marker_minter(fixture)).built(
            ...     frozenset({"a"})
            ... )
            >>> NameSetReads()._one_name(built)
            frozenset({'a'})
            >>> minter = new_marker_minter(fixture)
            >>> NameSetReads()._one_name(minter.resolved("kept.R")) is None
            True
        """
        if isinstance(proper, LiteralType) and isinstance(proper.value, str):
            return frozenset({proper.value})
        return None
