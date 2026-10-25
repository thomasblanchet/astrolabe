"""Write a family of data types as a mypy type, and read it back.

A family is a named group of data types, for example every numeric data type. A
selector such as `cs.numeric()` picks the columns whose data type is in one
family, and a remainder can record that its columns are in one family. A
remainder is the part of an open frame that its type does not name. Both facts
must travel inside a type, so the family must have a type too.

A family that polars gives a class, for example `NumericType` or `String`, is
written as that class. A family with no class of its own, for example
`Datetime("us")` or a group of several data types, is written as the union of
its members.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.types import UnionType, get_proper_type

from astrolabe._hooks.encoding.marker_minter import ProperReads
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeCodec, ConstDtypeReads
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe._hooks.remainder.type_set import Family, FamilyRoster

if TYPE_CHECKING:
    from mypy.types import Type as MypyType

    from astrolabe._hooks.encoding.marker_minter import MarkerMinter


class FamilyCodec:
    """Build the mypy type for one family of data types.

    A codec that writes a selector or a remainder record uses this class for the
    family part. The union form gives each member its own type through
    `ConstDtypeCodec`, so that a member with parameters looks the same as in a
    frame declaration.

    Example:
        >>> fixture = new_type_fixture()
        >>> FamilyCodec(new_marker_minter(fixture)).built(Family.numeric())
        polars.datatypes.classes.NumericType
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the minter that builds the types.

        Args:
            minter: The marker minter for the current phase.
        """
        self._minter = minter

    @propagates(None)
    def built(self, family: Family) -> MypyType | None:
        """Build the type for one family.

        A family with no polars class of its own is the union of its members.
        The members are sorted, so that one family always gives the same union.

        Args:
            family: The family to write.

        Returns:
            The polars class of the family, if it has one. Otherwise the union
            of its members, or None if a member has no type.

        Example:
            >>> fixture = new_type_fixture()
            >>> us = Family("us", frozenset({"Datetime(time_unit='us', time_zone=None)"}))
            >>> FamilyCodec(new_marker_minter(fixture)).built(us)
            astrolabe.frame.Datetime[Literal['us'], None]
        """
        fullname = FamilyRoster().class_fullname(family)
        if fullname is not None:
            return self._minter.resolved(fullname)
        spelling = ConstDtypeCodec(self._minter)
        items = [not_none(spelling.built(member)) for member in sorted(family.members)]
        return items[0] if len(items) == 1 else UnionType(list(items))


class FamilyReads:
    """Read a family of data types back from its mypy type.

    This is the reading half of `FamilyCodec`. It also reads a family that the
    user writes, for example the data types given to a selector. It builds
    no type, so it needs no access to the plugin.

    Example:
        >>> from mypy.types import UninhabitedType
        >>> FamilyReads().read(UninhabitedType()) is None
        True
    """

    def __init__(self, *, required: bool = False) -> None:
        """Set up the reader of instances.

        Args:
            required: Whether the type states a requirement, where
                `pl.DataType` stands for every kind (see
                `FamilyRoster.required`).
        """
        self._reads = ProperReads()
        self._required = required

    @propagates(None)
    def read(self, argument: MypyType) -> Family | None:
        """Read one type as a family.

        A union stands for the union of the families of its items. Its name
        joins the names of the items, in sorted order. A data type with
        parameters, for example `Datetime("us")`, stands for a family with that
        one data type.

        Args:
            argument: The type to read.

        Returns:
            The family, or None for a class that is not a data type or a family.

        Example:
            >>> fixture = new_type_fixture()
            >>> string = new_marker_minter(fixture).resolved("polars.datatypes.classes.String")
            >>> FamilyReads().read(string)
            String
            >>> us = Family("us", frozenset({"Datetime(time_unit='us', time_zone=None)"}))
            >>> FamilyReads().read(FamilyCodec(new_marker_minter(fixture)).built(us))
            Datetime(time_unit='us', time_zone=None)
        """
        proper = get_proper_type(argument)
        if isinstance(proper, UnionType):
            parts = [not_none(self.read(item)) for item in proper.items]
            names = "|".join(sorted(part.name for part in parts))
            members = frozenset().union(*(part.members for part in parts))
            return Family(names, members)
        instance = not_none(self._reads.instance(proper))
        rendering = ConstDtypeReads().read(instance)
        if rendering is not None and "(" in rendering:
            return Family(rendering, frozenset({rendering}))
        bare = instance.type.fullname.rsplit(".", maxsplit=1)[-1]
        roster = FamilyRoster()
        return roster.required(bare) if self._required else roster.named(bare)
