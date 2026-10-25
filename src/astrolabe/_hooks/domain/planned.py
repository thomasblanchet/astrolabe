"""Read the two data types of a column of a lazy frame that polars plans at another type."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy.subtypes import is_same_type
from mypy.types import Instance, get_proper_type

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe.frame import Planned

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup


class PlannedType:
    """Read the two data types of a column of a lazy frame: the stored one and the run one.

    A lazy frame plans each step at the data type that it stores for a
    column, and the data of the column can have another type. Such a column
    holds `Planned[run, planner]`. A frame operation that checks a data type,
    such as the key of a join, plans at the stored type, as polars does.
    `collect()` gives the run type.

    It sits in the plugin layer, and it reads mypy instance types. It builds
    no type, so it needs no access to the plugin.

    Example:
        >>> fixture = new_type_fixture()
        >>> PlannedType().stored(int_dtype(fixture))
        polars.datatypes.classes.Int64
    """

    TARGET: ClassVar[Target] = Target.of(Planned)

    def stored(self, typ: MypyType) -> MypyType:
        """Give the data type that a lazy frame stores for a column, at which polars plans.

        Args:
            typ: The data type of the column.

        Returns:
            The planner type of `Planned[run, planner]`, or the type itself.

        Example:
            >>> fixture = new_type_fixture()
            >>> PlannedType().stored(fixture.anyt)
            Any
        """
        proper = get_proper_type(typ)
        if self.TARGET.matches(proper) and len(proper.args) == 2:
            return proper.args[1]
        return typ

    def run(self, typ: MypyType) -> MypyType:
        """Give the data type of the data of a column, which `collect()` gives.

        Args:
            typ: The data type of the column.

        Returns:
            The run type of `Planned[run, planner]`, or the type itself.

        Example:
            >>> fixture = new_type_fixture()
            >>> PlannedType().run(int_dtype(fixture))
            polars.datatypes.classes.Int64
        """
        proper = get_proper_type(typ)
        if self.TARGET.matches(proper) and len(proper.args) == 2:
            return proper.args[0]
        return typ


class PlannedMap:
    """Map the two data types of a column of a lazy frame, and hold both where they still differ.

    A frame method that maps the data type of each column, such as `max` or
    `sum`, plans the map of the stored type and runs the map of the type of
    the data. A column that holds `Planned[run, planner]` then gives the two
    maps, and the result holds both where they differ.

    It sits in the plugin layer, and it reads and builds mypy instance types.

    Example:
        >>> fixture = new_type_fixture()
        >>> PlannedMap(new_plugin(fixture)).mapped(int_dtype(fixture), lambda typ: typ)
        polars.datatypes.classes.Int64
    """

    def __init__(self, plugin: PluginLookup) -> None:
        """Keep the lookup of the class of the marker, which a mapped type needs.

        Args:
            plugin: The mypy plugin, which finds the class `Planned`.
        """
        self._types = FreshTypeInfo(plugin)
        self._planned = PlannedType()

    def mapped(
        self,
        typ: MypyType,
        rule: Callable[[MypyType], MypyType | None],
    ) -> MypyType | None:
        """Map each of the two data types of a column, and hold both where they still differ.

        Args:
            typ: The data type of the column.
            rule: The map of one data type, which gives `None` where it has
                no answer.

        Returns:
            The mapped type: `Planned[run, planner]` where the two maps
            differ, the one type where they agree, or `None` where either map
            has no answer.

        Example:
            >>> fixture = new_type_fixture()
            >>> PlannedMap(new_plugin(fixture)).mapped(int_dtype(fixture), lambda typ: None)
        """
        run = rule(self._planned.run(typ))
        stored = rule(self._planned.stored(typ))
        if run is None or stored is None:
            return None
        if is_same_type(run, stored):
            return run
        info = self._types.resolved(PlannedType.TARGET.fullname)
        return None if info is None else Instance(info, [run, stored])
