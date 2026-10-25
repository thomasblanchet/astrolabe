"""Type sets and families: which data types the columns of a remainder can have.

A remainder is the columns of a frame that its type does not name.
The checker cannot list them, but it can limit their data types. A
declaration such as `other_columns(all_of=NumericType)` promises
that every such column is numeric, and `select(cs.numeric())` makes
it so. A type set holds this fact. It is one half of a record, the
value that holds all that the checker knows about a remainder. The
other half, the absence rule, is about names.

This module speaks of kinds. A kind is one data type class of
polars with no subclass, for example `Int64`, `List` or `Datetime`.
A kind with parameters holds many data types: `Datetime` holds one
for each time unit and each time zone. The list of kinds comes from
the installed polars (see `KindRoster`). A type set is exact for
each kind. It holds a finite list of that kind's data types, or the
whole kind except a finite list.

A family is a named group of kinds or single data types, for
example `NumericType`. It is what a user writes in `all_of=` or
`none_of=`, and what a data type selector such as `cs.numeric()`
tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from astrolabe._hooks.expression.kind import KindRoster

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class Family:
    """A named group of data types, such as every numeric kind, that a user or a selector names.

    A member is a kind, for example `Int64`, which stands for every
    data type of that class. A member can also be one data type, for
    example `Datetime(time_unit='us', time_zone=None)`, or every zone
    of one time unit, `Datetime(time_unit='us')`. The name is the
    polars class that spells the family in a type, or a name that the
    engine builds from the members.

    Example:
        >>> Family("NumericType", frozenset({"Int64"})).holds("Int64")
        True
    """

    name: str
    members: frozenset[str]

    @classmethod
    def of_members(cls, members: frozenset[str]) -> Family:
        """Build a family whose name joins its sorted members, so that equal sets get one name.

        A family that a selector or a remainder reads back from its type
        gets this name, so a family built with it reads back equal.

        Args:
            members: The kinds or the data types of the family.

        Returns:
            The family.

        Example:
            >>> Family.of_members(frozenset({"Null", "Int8"})).name
            'Int8|Null'
        """
        return cls("|".join(sorted(members)), members)

    @classmethod
    def numeric(cls) -> Family:
        """Build the family that `cs.numeric()` picks.

        Returns:
            The family of every numeric kind of the installed polars.

        Example:
            >>> Family.numeric().holds("Int64")
            True
        """
        return cls("NumericType", frozenset(KindRoster().numeric()))

    @classmethod
    def temporal(cls) -> Family:
        """Build the family that `cs.temporal()` picks.

        No hook calls it, because a hook reads the family from the
        selector. The examples of the remainder selectors build a second
        family with it, beside the numeric one.

        Returns:
            The family of every temporal kind of the installed polars.

        Example:
            >>> Family.temporal().holds("Datetime")
            True
        """
        return cls("TemporalType", frozenset(KindRoster().temporal()))

    def __repr__(self) -> str:
        """Give the name of the family, so that the text is the same in every process.

        The `repr` of a `frozenset` lists its members in hash order,
        and that order changes between processes. A test ID built
        from such text then changes too, and a test runner that splits
        the tests between workers fails.

        Returns:
            The name of the family.

        Example:
            >>> Family("NumericType", frozenset({"Int64"}))
            NumericType
        """
        return self.name

    def holds(self, dtype: str) -> bool:
        """Answer whether the family holds one data type.

        Args:
            dtype: A kind, or the full text of one data type.

        Returns:
            True if the family holds it.

        Example:
            >>> Family.numeric().holds("String")
            False
        """
        return self.type_set().holds(dtype)

    def type_set(self) -> TypeSet:
        """Give the data types of the family as a type set, so that set operations apply.

        Returns:
            The type set of the members.

        Example:
            >>> sorted(Family("Time", frozenset({"Time"})).type_set().members)
            ['Time']
        """
        return TypeSet(self.members)


class FamilyRoster:
    """Map between a polars data type class and the family it stands for.

    A type spells a family as a polars class. It is a group base
    such as `NumericType`, or one kind such as `String`, which is a
    family of one kind. Code that reads a declaration needs the
    family of a class name. Code that writes a type needs the class
    of a family. This roster answers both, and it reads the members
    from the installed polars, so no list is written by hand.

    Example:
        >>> FamilyRoster().named("NumericType").holds("Int64")
        True
    """

    _ANY: ClassVar[str] = pl.DataType.__name__

    def required(self, name: str) -> Family | None:
        """Give the family that a polars class name stands for in a requirement.

        A requirement, for example `other_columns(all_of=pl.DataType)`, and
        the column of a frame that a fit checks can also name `DataType`,
        the base of every polars class, which stands for every kind. A
        selector cannot name it: polars refuses `cs.by_dtype(pl.DataType)`.

        Args:
            name: The name of the class, with no module.

        Returns:
            The family, or None for a name that polars does not define.

        Example:
            >>> "Struct" in FamilyRoster().required("DataType").members
            True
            >>> FamilyRoster().required("FloatType").holds("Float32")
            True
        """
        if name == self._ANY:
            return Family(name, frozenset(KindRoster().names()))
        return self.named(name)

    def named(self, name: str) -> Family | None:
        """Give the family that a polars class name stands for.

        Args:
            name: The name of the class, with no module.

        Returns:
            The family, or None for a name that polars does not
            define.

        Example:
            >>> FamilyRoster().named("String").members
            frozenset({'String'})
            >>> FamilyRoster().named("NoSuchType") is None
            True
            >>> FamilyRoster().named("DataType") is None
            True
        """
        for base in KindRoster().group_bases():
            if base.__name__ == name:
                return Family(name, frozenset(KindRoster().family(base)))
        if name in KindRoster().names():
            return Family(name, frozenset({name}))
        return None

    def class_fullname(self, family: Family) -> str | None:
        """Give the full name of the polars class that spells one family in a type.

        A class spells a family if it has the name of the family, and
        if the family of the class has exactly the members of the
        family.

        Args:
            family: The family to spell.

        Returns:
            The full name of the class, or None if no polars class
            has the name and the members of the family.

        Example:
            >>> roster = FamilyRoster()
            >>> roster.class_fullname(Family.numeric())
            'polars.datatypes.classes.NumericType'
            >>> roster.class_fullname(Family("Weird", frozenset())) is None
            True
            >>> roster.class_fullname(Family("String", frozenset({"Int64"}))) is None
            True
        """
        for cls in (*KindRoster().group_bases(), *KindRoster().classes()):
            if cls.__name__ != family.name:
                continue
            derived = self.named(cls.__name__)
            if derived is not None and derived.members == family.members:
                return f"{cls.__module__}.{cls.__qualname__}"
        return None


@dataclass(frozen=True)
class KindSectors:
    """Split a kind into sectors, the pieces that a type set can hold whole.

    Most kinds have one sector with no name, the empty text. Their
    data types are few, or a list of parameters describes them.
    `Datetime` is different, because its time zone is any text. The
    selector `cs.datetime("us")` picks every zone of one time unit.
    No finite list states that set, and neither does a finite list of
    exceptions. So `Datetime` has one sector for each time unit, and
    a type set can hold one sector whole.

    Example:
        >>> sorted(KindSectors().of("Datetime"))
        ['ms', 'ns', 'us']
        >>> sorted(KindSectors().of("Int64"))
        ['']
    """

    KIND: ClassVar[str] = "Datetime"
    UNITS: ClassVar[tuple[str, ...]] = ("ms", "ns", "us")
    ANY: ClassVar[str] = ""

    def of(self, kind: str) -> frozenset[str]:
        """Give the sectors of one kind.

        Args:
            kind: The name of the kind.

        Returns:
            One sector for each time unit for `Datetime`, and the one
            sector with no name for every other kind.

        Example:
            >>> KindSectors().of("Datetime") == {"ms", "ns", "us"}
            True
        """
        if kind == self.KIND:
            return frozenset(self.UNITS)
        return frozenset({self.ANY})

    def sector_of(self, rendering: str) -> str:
        """Give the sector that one data type is in.

        Args:
            rendering: The full text of a data type, or a kind.

        Returns:
            The time unit of a `Datetime` that names one, and the
            sector with no name for anything else.

        Example:
            >>> KindSectors().sector_of("Datetime(time_unit='us', time_zone=None)")
            'us'
            >>> KindSectors().sector_of("Duration(time_unit='us')")
            ''
            >>> KindSectors().sector_of("Datetime")
            ''
            >>> KindSectors().sector_of("Int64")
            ''
        """
        if rendering.split("(", maxsplit=1)[0] != self.KIND:
            return self.ANY
        _, marker, rest = rendering.partition("time_unit='")
        if not marker:
            return self.ANY
        return rest.split("'", maxsplit=1)[0]

    def atom(self, kind: str, sector: str) -> str:
        """Give the text that stands for one whole sector in a type set.

        Args:
            kind: The name of the kind.
            sector: The name of the sector.

        Returns:
            The kind for the sector with no name. For a time unit,
            the short text that stands for every zone of that unit.

        Example:
            >>> KindSectors().atom("Datetime", "us")
            "Datetime(time_unit='us')"
            >>> KindSectors().atom("Int64", "")
            'Int64'
        """
        if sector == self.ANY:
            return kind
        return f"{kind}(time_unit='{sector}')"

    def whole_sector(self, atom: str) -> str | None:
        """Give the sector that one member of a type set stands for whole.

        Args:
            atom: A kind, the short text for one time unit, or the
                full text of one data type.

        Returns:
            The name of the sector, or None for the full text of one
            data type, which stands for no whole sector. Only the short
            text of a `Datetime` time unit names a sector with parameters.
            The text of any other kind with parameters, for example
            `Duration(time_unit='us')` or `List(Null)`, is one data type.

        Example:
            >>> KindSectors().whole_sector("Datetime(time_unit='us')")
            'us'
            >>> KindSectors().whole_sector("Datetime(time_unit='us', time_zone=None)") is None
            True
            >>> KindSectors().whole_sector("Duration(time_unit='us')") is None
            True
        """
        if "(" not in atom:
            return self.ANY
        if "," in atom or not atom.startswith(f"{self.KIND}("):
            return None
        return self.sector_of(atom)


@dataclass(frozen=True)
class KindPart:
    """The data types of one kind that a type set holds.

    A kind splits into sectors (see `KindSectors`). A part holds some
    sectors whole, except a finite set `removed`. It also holds a
    finite set `singles` from the sectors it does not hold whole.
    The sectors are few and both sets are finite. So the union, the
    intersection and the complement of two parts are parts again,
    and every question about a type set has an exact answer.

    Example:
        >>> us = "Datetime(time_unit='us', time_zone=None)"
        >>> KindPart.whole("Datetime").without(frozenset({us})).holds(us)
        False
    """

    sectors: frozenset[str]
    held: frozenset[str] = frozenset()
    singles: frozenset[str] = frozenset()
    removed: frozenset[str] = frozenset()

    @classmethod
    def empty(cls, kind: str) -> KindPart:
        """Build the part of one kind that holds no data type.

        Args:
            kind: The name of the kind.

        Returns:
            The empty part.

        Example:
            >>> KindPart.empty("Int64").is_empty()
            True
        """
        return cls(KindSectors().of(kind))

    @classmethod
    def whole(cls, kind: str) -> KindPart:
        """Build the part that holds every data type of one kind.

        Args:
            kind: The name of the kind.

        Returns:
            The part that holds every sector whole.

        Example:
            >>> KindPart.whole("Int64").holds("Int64")
            True
        """
        sectors = KindSectors().of(kind)
        return cls(sectors, sectors)

    @classmethod
    def one_sector(cls, kind: str, sector: str) -> KindPart:
        """Build the part that holds one whole sector of a kind.

        Args:
            kind: The name of the kind.
            sector: The name of the sector.

        Returns:
            The part that holds that sector alone.

        Example:
            >>> zoned = "Datetime(time_unit='us', time_zone='UTC')"
            >>> KindPart.one_sector("Datetime", "us").holds(zoned)
            True
        """
        return cls(KindSectors().of(kind), frozenset({sector}))

    @classmethod
    def listed(cls, kind: str, renderings: frozenset[str]) -> KindPart:
        """Build the part that holds a finite list of data types.

        Args:
            kind: The name of the kind.
            renderings: The full text of each data type.

        Returns:
            The part that holds exactly those data types.

        Example:
            >>> KindPart.listed("Int64", frozenset({"Int64"})).holds("Int64")
            True
        """
        return cls(KindSectors().of(kind), frozenset(), renderings)

    def without(self, renderings: frozenset[str]) -> KindPart:
        """Take a set of data types out of this part.

        A data type in a sector that the part holds whole goes into
        `removed`. A data type in the list `singles` leaves the list.

        Args:
            renderings: The full text of each data type to take out.

        Returns:
            The part without those data types.

        Example:
            >>> us = "Datetime(time_unit='us', time_zone=None)"
            >>> KindPart.whole("Datetime").without(frozenset({us})).excluded()
            frozenset({"Datetime(time_unit='us', time_zone=None)"})
        """
        sectors = KindSectors()
        inside = frozenset(one for one in renderings if sectors.sector_of(one) in self.held)
        return KindPart(self.sectors, self.held, self.singles - renderings, inside)

    def holds(self, rendering: str) -> bool:
        """Answer whether the part holds one data type.

        Args:
            rendering: The full text of a data type of this kind.

        Returns:
            True if the data type is in the part.

        Example:
            >>> KindPart.listed("Int64", frozenset({"Int64"})).holds("Int64")
            True
        """
        if rendering in self.singles:
            return True
        sector = KindSectors().sector_of(rendering)
        return sector in self.held and rendering not in self.removed

    def union(self, other: KindPart) -> KindPart:
        """Give the part that holds what either part holds.

        The union is the complement of the intersection of the two
        complements, so it needs no rule of its own.

        Args:
            other: The other part, of the same kind.

        Returns:
            The union.

        Example:
            >>> one = KindPart.listed("Int64", frozenset({"Int64"}))
            >>> one.union(KindPart.whole("Int64")).holds("Int64")
            True
        """
        return self.complement().intersection(other.complement()).complement()

    def intersection(self, other: KindPart) -> KindPart:
        """Give the part that holds what both parts hold.

        Args:
            other: The other part, of the same kind.

        Returns:
            The intersection.

        Example:
            >>> whole = KindPart.whole("Int64")
            >>> whole.intersection(KindPart.empty("Int64")).is_empty()
            True
        """
        held = self.held & other.held
        return KindPart(
            self.sectors,
            held,
            self._shared_singles(other, held),
            self._shared_removed(other, held),
        )

    def _shared_singles(self, other: KindPart, held: frozenset[str]) -> frozenset[str]:
        """Give the single data types of the intersection.

        These are the listed data types of either part that both
        parts hold, and that no shared whole sector already holds.

        Args:
            other: The other part.
            held: The sectors that both parts hold whole.

        Returns:
            The list `singles` of the intersection.

        Example:
            >>> one = KindPart.listed("Int64", frozenset({"Int64"}))
            >>> one._shared_singles(one, frozenset())
            frozenset({'Int64'})
        """
        sectors = KindSectors()
        candidates = self.singles | other.singles
        return frozenset(
            one
            for one in candidates
            if sectors.sector_of(one) not in held and self.holds(one) and other.holds(one)
        )

    def _shared_removed(self, other: KindPart, held: frozenset[str]) -> frozenset[str]:
        """Give the data types that the intersection takes out of its whole sectors.

        A data type that either part takes out stays out, if it is
        in a sector that both parts hold whole.

        Args:
            other: The other part.
            held: The sectors that both parts hold whole.

        Returns:
            The list `removed` of the intersection.

        Example:
            >>> whole = KindPart.whole("Int64")
            >>> whole._shared_removed(whole, frozenset({""}))
            frozenset()
        """
        sectors = KindSectors()
        taken = self.removed | other.removed
        return frozenset(one for one in taken if sectors.sector_of(one) in held)

    def complement(self) -> KindPart:
        """Give the part that holds every other data type of the kind.

        The held sectors swap with the others, and the two lists swap
        too. A data type that a whole sector took out is now a single
        data type that the part holds. A single data type is now taken
        out of a sector that the part holds whole.

        Returns:
            The complement within the kind.

        Example:
            >>> KindPart.empty("Int64").complement().holds("Int64")
            True
        """
        return KindPart(self.sectors, self.sectors - self.held, self.removed, self.singles)

    def within(self, other: KindPart) -> bool:
        """Answer whether the other part holds every data type of this part.

        Args:
            other: The other part, of the same kind.

        Returns:
            True if this part is inside the other.

        Example:
            >>> one = KindPart.listed("Int64", frozenset({"Int64"}))
            >>> one.within(KindPart.whole("Int64"))
            True
        """
        return self.intersection(other.complement()).is_empty()

    def atoms(self, kind: str) -> frozenset[str]:
        """Give the members that spell this part in a type set.

        Args:
            kind: The kind of the part.

        Returns:
            The kind if every sector is held whole, or else the short
            text of each held sector. Then the listed data types.

        Example:
            >>> KindPart.one_sector("Datetime", "us").atoms("Datetime")
            frozenset({"Datetime(time_unit='us')"})
        """
        sectors = KindSectors()
        if self.held == self.sectors and self.held:
            return frozenset({kind}) | self.singles
        whole = {sectors.atom(kind, sector) for sector in self.held}
        return frozenset(whole) | self.singles

    def excluded(self) -> frozenset[str]:
        """Give the data types that the whole sectors of this part leave out.

        Returns:
            The list `removed`.

        Example:
            >>> KindPart.empty("Int64").excluded()
            frozenset()
        """
        return self.removed

    def is_empty(self) -> bool:
        """Answer whether the part holds no data type.

        Returns:
            True if no sector is held whole and no data type is
            listed.

        Example:
            >>> KindPart.empty("Datetime").is_empty()
            True
        """
        return not self.held and not self.singles


@dataclass(frozen=True)
class TypeSet:
    """The data types that the columns of a remainder can have.

    A record holds all that the checker knows about a remainder, and
    this is its half about data types. A declaration such as
    `other_columns(all_of=NumericType)` is checked against it. A data
    type selector such as `cs.numeric()` reads it to decide whether it
    picks every column of the remainder, or none.

    Each member stands for some data types. A kind, for example
    `Int64` or `List`, stands for every data type of that kind. The
    full text of one data type, for example
    `Datetime(time_unit='us', time_zone=None)`, stands for that one.
    The short text `Datetime(time_unit='us')` stands for every zone of
    one time unit. `excluded` lists the data types taken out of a
    whole kind or unit. Internally, each operation works kind by kind
    on `KindPart` values, where every answer is exact.

    Example:
        >>> TypeSet(frozenset({"Int64"})).contained_in(Family.numeric())
        True
    """

    members: frozenset[str]
    excluded: frozenset[str] = frozenset()

    @classmethod
    def anything(cls) -> TypeSet:
        """Build the type set that allows every data type.

        The kinds come from the installed polars, so the set cannot
        miss a data type that a new polars release adds.

        Returns:
            The type set of a remainder with no promise about its
            data types.

        Example:
            >>> TypeSet.anything().contained_in(Family.numeric())
            False
        """
        return cls(frozenset(KindRoster().names()))

    @classmethod
    def of_parts(cls, parts: Mapping[str, KindPart]) -> TypeSet:
        """Build a type set back from its parts, one for each kind.

        Args:
            parts: The part of each kind, by kind name.

        Returns:
            The type set. An empty part adds no member.

        Example:
            >>> TypeSet.of_parts({"Int64": KindPart.whole("Int64")}).members
            frozenset({'Int64'})
        """
        members = frozenset().union(*(part.atoms(kind) for kind, part in parts.items()))
        excluded = frozenset().union(*(part.excluded() for part in parts.values()))
        return cls(members, excluded)

    def parts(self) -> dict[str, KindPart]:
        """Split the set into one part for each kind, where set operations are exact.

        Returns:
            One part for each kind that a member names.

        Example:
            >>> us = "Datetime(time_unit='us', time_zone=None)"
            >>> TypeSet(frozenset({"Datetime"}), frozenset({us})).parts()["Datetime"].holds(us)
            False
        """
        parts: dict[str, KindPart] = {}
        for atom in self.members:
            kind = self.kind_of(atom)
            standing = parts.get(kind, KindPart.empty(kind))
            parts[kind] = standing.union(self._atom(atom))
        return parts

    def _atom(self, atom: str) -> KindPart:
        """Give the part that one member stands for.

        A member that names a whole sector stands for that sector, or
        for the whole kind if the sector has no name. The data types of
        the kind that the set takes out are then left out of it.

        Args:
            atom: A kind, the short text for one time unit, or the
                full text of one data type.

        Returns:
            The whole kind or unit without the excluded data types,
            or the one data type.

        Example:
            >>> TypeSet(frozenset())._atom("Int8").holds("Int8")
            True
            >>> TypeSet(frozenset())._atom("Datetime(time_unit='us')").atoms("Datetime")
            frozenset({"Datetime(time_unit='us')"})
        """
        kind = self.kind_of(atom)
        sector = KindSectors().whole_sector(atom)
        if sector is None:
            return KindPart.listed(kind, frozenset({atom}))
        if sector == KindSectors.ANY:
            whole = KindPart.whole(kind)
        else:
            whole = KindPart.one_sector(kind, sector)
        left_out = frozenset(one for one in self.excluded if self.kind_of(one) == kind)
        return whole.without(left_out)

    def kind_of(self, atom: str) -> str:
        """Give the kind of one member, which is its text before any parameters.

        Args:
            atom: A kind, or the text of a data type with its
                parameters.

        Returns:
            The kind.

        Example:
            >>> TypeSet(frozenset()).kind_of("List(Int64)")
            'List'
        """
        return atom.split("(", maxsplit=1)[0]

    def kinds(self) -> frozenset[str]:
        """Give the kinds of which the set holds at least one data type.

        Returns:
            The kinds.

        Example:
            >>> TypeSet(frozenset({"List(Int64)", "Int8"})).kinds() == {"List", "Int8"}
            True
        """
        return frozenset(kind for kind, part in self.parts().items() if not part.is_empty())

    def holds(self, rendering: str) -> bool:
        """Answer whether the set certainly holds one data type.

        Args:
            rendering: The full text of a data type, or a kind. A kind
                with parameters, for example `Datetime`, stands for a
                data type whose parameters are not known. Only a set
                that holds the whole kind certainly holds it.

        Returns:
            True if the data type is in the set.

        Example:
            >>> TypeSet(frozenset({"Datetime"})).holds("Datetime(time_unit='us', time_zone=None)")
            True
        """
        return TypeSet(frozenset({rendering})).within(self)

    def union(self, other: TypeSet) -> TypeSet:
        """Give the set that holds what either set holds.

        Args:
            other: The other set.

        Returns:
            The union.

        Example:
            >>> sorted(TypeSet(frozenset({"Int8"})).union(TypeSet(frozenset({"Date"}))).members)
            ['Date', 'Int8']
        """
        return self._combined(other, KindPart.union)

    def intersection(self, other: TypeSet) -> TypeSet:
        """Give the set that holds what both sets hold.

        Args:
            other: The other set.

        Returns:
            The intersection.

        Example:
            >>> dates = TypeSet(frozenset({"Date"}))
            >>> both = TypeSet(frozenset({"Int8", "Date"})).intersection(dates)
            >>> sorted(both.members)
            ['Date']
        """
        return self._combined(other, KindPart.intersection)

    def minus(self, other: TypeSet) -> TypeSet:
        """Give the set that holds what this set holds and the other does not.

        A declaration with `none_of=` uses this to take data types out
        of the set.

        Args:
            other: The other set.

        Returns:
            The difference.

        Example:
            >>> us = "Datetime(time_unit='us', time_zone=None)"
            >>> rest = TypeSet(frozenset({"Datetime"})).minus(TypeSet(frozenset({us})))
            >>> rest.holds(us), rest.holds("Datetime(time_unit='ms', time_zone=None)")
            (False, True)
        """
        return self._combined(other, lambda mine, theirs: mine.intersection(theirs.complement()))

    def within(self, other: TypeSet) -> bool:
        """Answer whether the other set holds every data type of this set.

        A check that a remainder keeps a data type promise asks this,
        with the remainder's set on the left.

        Args:
            other: The other set.

        Returns:
            True if this set is inside the other.

        Example:
            >>> TypeSet(frozenset({"Int8"})).within(TypeSet.anything())
            True
        """
        return self.minus(other).is_empty()

    def is_empty(self) -> bool:
        """Answer whether the set holds no data type.

        A remainder whose type set is empty holds no column.

        Returns:
            True for the empty set.

        Example:
            >>> TypeSet(frozenset()).is_empty()
            True
        """
        return not self.kinds()

    def _combined(
        self,
        other: TypeSet,
        combine: Callable[[KindPart, KindPart], KindPart],
    ) -> TypeSet:
        """Combine two sets kind by kind, with one operation on parts.

        A kind that only one set names meets an empty part of that
        kind in the other set.

        Args:
            other: The other set.
            combine: The operation on two parts of one kind.

        Returns:
            The combined set.

        Example:
            >>> pair = (TypeSet(frozenset()), KindPart.union)
            >>> TypeSet(frozenset({"Int8"}))._combined(*pair).members
            frozenset({'Int8'})
        """
        mine, theirs = self.parts(), other.parts()
        return TypeSet.of_parts({
            kind: combine(
                mine.get(kind, KindPart.empty(kind)),
                theirs.get(kind, KindPart.empty(kind)),
            )
            for kind in mine.keys() | theirs.keys()
        })

    def contained_in(self, family: Family) -> bool:
        """Answer whether `family` holds every data type that this set allows.

        A data type selector asks this to know whether it picks every
        column of a remainder.

        Args:
            family: The family to compare with.

        Returns:
            True if no allowed data type is outside the family.

        Example:
            >>> TypeSet(frozenset({"Int64", "String"})).contained_in(
            ...     Family.numeric()
            ... )
            False
        """
        return self.within(family.type_set())

    def disjoint_from(self, family: Family) -> bool:
        """Answer whether `family` holds none of the data types that this set allows.

        A data type selector asks this to know whether it picks no
        column of a remainder.

        Args:
            family: The family to compare with.

        Returns:
            True if no allowed data type is inside the family.

        Example:
            >>> TypeSet(frozenset({"String"})).disjoint_from(Family.numeric())
            True
        """
        return self.intersection(family.type_set()).is_empty()

    def narrowed_to(self, family: Family) -> TypeSet:
        """Keep only the allowed data types that `family` holds.

        A data type selector such as `cs.numeric()` does this to a
        remainder, and so does a declaration with `all_of=`.

        Args:
            family: The family to intersect with.

        Returns:
            The allowed data types that are in the family.

        Example:
            >>> narrowed = TypeSet(frozenset({"Int64", "String"}))
            >>> sorted(narrowed.narrowed_to(Family.numeric()).members)
            ['Int64']
        """
        return self.intersection(family.type_set())

    def combined_with(self, other: TypeSet) -> TypeSet:
        """Allow every data type that either set allows.

        Two remainders that join into one give this type set.

        Args:
            other: The type set of the other remainder.

        Returns:
            The union of the two type sets.

        Example:
            >>> a = TypeSet(frozenset({"Int64"}))
            >>> sorted(a.combined_with(TypeSet(frozenset({"String"}))).members)
            ['Int64', 'String']
        """
        return self.union(other)


class TypeSamples:
    """Give a finite list of data types that stands for every data type, for some type tests.

    Some questions ask about every possible column, for example
    whether one selector picks no more columns than another. There
    are infinitely many data types, but a set of tests on data types
    can tell only a few groups apart. A test names whole kinds and
    some single data types. Within one kind, every data type that no
    test names gets the same answer from each test. So one new sample
    stands for all of them, beside each data type that a test names.
    The question then has an exact answer from these samples alone.

    Example:
        >>> us = "Datetime(time_unit='us', time_zone=None)"
        >>> found = TypeSamples().of(frozenset({us}))
        >>> [d for d in found if d.startswith("Datetime")][-1]
        "Datetime(time_unit='us', time_zone=None)"
    """

    def of(self, mentioned: frozenset[str]) -> tuple[str, ...]:
        """Give the samples for a set of tests, from the single data types they name.

        Args:
            mentioned: The single data types that the tests name.

        Returns:
            The kind itself for each kind with no named data type.
            For each other kind, new samples and the named data types.

        Example:
            >>> "Int64" in TypeSamples().of(frozenset())
            True
        """
        samples: list[str] = []
        for kind in KindRoster().names():
            named = sorted(m for m in mentioned if m.split("(", maxsplit=1)[0] == kind)
            if named:
                samples.extend(self._fresh(kind))
                samples.extend(named)
            else:
                samples.append(kind)
        return tuple(samples)

    def _fresh(self, kind: str) -> list[str]:
        """Give one new data type for each sector of a kind, which no test names.

        A test can tell one `Datetime` time unit from another. So each
        unit needs its own sample, which stands for the zones that no
        test names. Every other kind needs one sample. A sample uses
        `*` as its parameter, which no real data type has.

        Args:
            kind: The kind.

        Returns:
            The new samples.

        Example:
            >>> TypeSamples()._fresh("Int8")
            ['Int8(*)']
            >>> TypeSamples()._fresh("Datetime")[0]
            "Datetime(time_unit='ms', time_zone='*')"
        """
        if kind != KindSectors.KIND:
            return [f"{kind}(*)"]
        return [f"Datetime(time_unit='{unit}', time_zone='*')" for unit in KindSectors.UNITS]
