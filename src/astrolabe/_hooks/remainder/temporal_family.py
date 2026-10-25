"""The data types that `cs.datetime(...)` and `cs.duration(...)` pick, as families.

A family is a named group of data types that a selector tests. A
`Datetime` data type has a time unit and a time zone, and a
`Duration` has a time unit. The checker must know which of them a
selector such as `cs.datetime("us", ["UTC", None])` picks.

If the selector lists its units and its zones, it picks a finite set
of data types. If its zones include both `"*"` and `None`, it picks
every zone of each unit, and one short text for each unit states
that. `"*"` without `None` picks every zone that has a name, and it
leaves out the naive columns, which have no zone. No family holds
exactly that set. The checker writes it as a combination of
selectors instead: every zone of the unit, but not the naive data
type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe._hooks.remainder.selector import OfType, SelAnd, Selector, SelNot
from astrolabe._hooks.remainder.type_set import Family


@dataclass(frozen=True)
class TemporalFamilies:
    """Build the family or the selector for one `cs.datetime(...)` or `cs.duration(...)` call.

    Code that types these selector calls reads their units and zones,
    and then asks this class for the data types they pick. An unknown
    unit gives None, and the selector then keeps its plain type.

    Example:
        >>> TemporalFamilies().datetime(["us"], [None]).members
        frozenset({"Datetime(time_unit='us', time_zone=None)"})
    """

    UNITS: ClassVar[tuple[str, ...]] = ("ns", "us", "ms")
    ANY_ZONE: ClassVar[str] = "*"

    @propagates(None)
    def datetime(self, units: list[str] | None, zones: list[str | None]) -> Family | None:
        """Give the family of the `Datetime` data types of some units and zones.

        Args:
            units: The time units, or None for every unit.
            zones: The time zones. None in the list stands for the
                naive data type, and `"*"` for every named zone.

        Returns:
            The family. None if a unit is unknown, or if the zones
            are every named zone without the naive one, which no
            family states.

        Example:
            >>> sorted(TemporalFamilies().datetime(["us"], ["*", None]).members)
            ["Datetime(time_unit='us')"]
            >>> TemporalFamilies().datetime(["us"], ["*"]) is None
            True
            >>> len(TemporalFamilies().datetime(None, [None, "UTC"]).members)
            6
        """
        chosen = not_none(self._units(units))
        if self.ANY_ZONE in zones:
            not_falsy(None in zones)
            return Family.of_members(frozenset(map(self._every_zone, chosen)))
        members = frozenset(
            f"Datetime(time_unit='{unit}', time_zone={self._zone(zone)})"
            for unit in chosen
            for zone in zones
        )
        return Family.of_members(members)

    def _every_zone(self, unit: str) -> str:
        """Give the short text that stands for every zone of one time unit.

        Args:
            unit: The time unit.

        Returns:
            The text, with the unit and no zone.

        Example:
            >>> TemporalFamilies()._every_zone("us")
            "Datetime(time_unit='us')"
        """
        return f"Datetime(time_unit='{unit}')"

    @propagates(None)
    def datetime_selector(
        self,
        units: list[str] | None,
        zones: list[str | None],
    ) -> Selector | None:
        """Give the selector that one `cs.datetime(...)` call states.

        This covers the case that `datetime` cannot: every named zone
        without the naive one becomes every zone of the unit, and not
        the naive data type of the unit.

        Args:
            units: The time units, or None for every unit.
            zones: The time zones. None in the list stands for the
                naive data type, and `"*"` for every named zone.

        Returns:
            The selector, or None if a unit is unknown.

        Example:
            >>> TemporalFamilies().datetime_selector(["us"], ["*", None])
            OfType(family=Datetime(time_unit='us'))
            >>> starred = TemporalFamilies().datetime_selector(["us"], ["*"])
            >>> starred.left
            OfType(family=Datetime(time_unit='us'))
            >>> starred.right.inner
            OfType(family=Datetime(time_unit='us', time_zone=None))
        """
        listed = self.datetime(units, zones)
        if listed is not None:
            return OfType(listed)
        chosen = not_none(self._units(units))
        sector = Family.of_members(frozenset(map(self._every_zone, chosen)))
        naive = Family.of_members(frozenset(self._naive(unit) for unit in chosen))
        return SelAnd(OfType(sector), SelNot(OfType(naive)))

    def _naive(self, unit: str) -> str:
        """Give the text of the naive data type of one time unit, which has no zone.

        Args:
            unit: The time unit.

        Returns:
            The full text of the data type.

        Example:
            >>> TemporalFamilies()._naive("us")
            "Datetime(time_unit='us', time_zone=None)"
        """
        return f"Datetime(time_unit='{unit}', time_zone=None)"

    def duration(self, units: list[str] | None) -> Family | None:
        """Give the family of the `Duration` data types of some units.

        Args:
            units: The time units, or None for every unit.

        Returns:
            The family, or None if a unit is unknown.

        Example:
            >>> TemporalFamilies().duration(["ms"]).members
            frozenset({"Duration(time_unit='ms')"})
            >>> TemporalFamilies().duration(["s"]) is None
            True
        """
        chosen = self._units(units)
        if chosen is None:
            return None
        return Family.of_members(frozenset(f"Duration(time_unit='{unit}')" for unit in chosen))

    def _units(self, units: list[str] | None) -> tuple[str, ...] | None:
        """Give the chosen time units, or every unit for None.

        Args:
            units: The listed units, or None.

        Returns:
            The units, or None if polars has no such unit.

        Example:
            >>> TemporalFamilies()._units(["s"]) is None
            True
        """
        if units is None:
            return self.UNITS
        return tuple(units) if set(units) <= set(self.UNITS) else None

    def _zone(self, zone: str | None) -> str:
        """Write a time zone the way that polars shows it in a data type.

        Args:
            zone: The zone, or None.

        Returns:
            The zone in quotes, or the text `None`.

        Example:
            >>> TemporalFamilies()._zone("UTC")
            "'UTC'"
        """
        return "None" if zone is None else f"'{zone}'"
