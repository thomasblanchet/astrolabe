"""Give the columns that a group-by puts before its aggregations, and the windows of its groups.

A `group_by` puts its keys first. A group-by over windows of the rows, as
`rolling(...)` or `group_by_dynamic(...)` gives, puts its keys, then the
bounds of each window where the call asks for them, then its index column.
Its aggregations run over the windows, which the evaluation of each argument
takes in place of the groups of a `group_by`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.calls.dtype_constructor_readers import CheckerLiteralTypeBuilder
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.encoding.marker_minter import MarkerMinter
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeReads
from astrolabe._hooks.encoding.schema_renderings import MaterializedColumn
from astrolabe._rules.releases import polars_releases
from astrolabe.frame import TypedLazyDynamicGroupBy, TypedLazyRollingGroupBy

if TYPE_CHECKING:
    from mypy.plugin import CheckerPluginInterface, Plugin

    from astrolabe._hooks.group_by.receiver import GroupByReceiver


class WindowColumns:
    """Give the leading columns of a group-by, and the windows that its aggregations run over.

    The bounds take the data type of the index, with one exception: over a
    `Date` index, the run gives `Datetime("us")` bounds. Before polars 2.0.0,
    a lazy frame plans them as `Date`, and then holds both types for each
    bound. From that release, it plans them as `Datetime("us")` too.

    Example:
        >>> WindowColumns.BOUNDS
        ('_lower_boundary', '_upper_boundary')
    """

    BOUNDS: ClassVar[tuple[str, str]] = ("_lower_boundary", "_upper_boundary")
    DATETIME: ClassVar[str] = "Datetime(time_unit='us', time_zone=None)"
    LAZY: ClassVar[frozenset[str]] = frozenset({
        Target.of(TypedLazyRollingGroupBy).fullname,
        Target.of(TypedLazyDynamicGroupBy).fullname,
    })

    def __init__(self, plugin: Plugin) -> None:
        """Keep the plugin, which finds the classes of a column that holds two types.

        Args:
            plugin: The mypy plugin.
        """
        self._plugin = plugin

    def kind(self, receiver: GroupByReceiver) -> str:
        """Give the windows that the aggregations run over, as the evaluation names them.

        Args:
            receiver: The group-by.

        Returns:
            `rolling`, `dynamic` or `disjoint`, or the empty text for the
            groups of a `group_by`.
        """
        return receiver.windows.split(" ")[0]

    def leading(self, receiver: GroupByReceiver, api: CheckerPluginInterface) -> list[ColumnRef]:
        """Give the columns that the group-by puts before its aggregations.

        Args:
            receiver: The group-by.
            api: The mypy plugin API, which builds the types of the bounds.

        Returns:
            The keys, then the bounds where the call asks for them, then the index column.
        """
        if not receiver.windows.endswith("bounds"):
            return [*receiver.key_columns, *receiver.index_columns]
        bounds = self._bounds(receiver, api)
        return [*receiver.key_columns, *bounds, *receiver.index_columns]

    def _bounds(self, receiver: GroupByReceiver, api: CheckerPluginInterface) -> list[ColumnRef]:
        """Give the two bounds of each window.

        Args:
            receiver: The group-by over windows, with its index column.
            api: The mypy plugin API.

        Returns:
            The lower bound, then the upper bound.
        """
        index = receiver.index_columns[0]
        if ConstDtypeReads().read_type(index.dtype) != "Date":
            return [ColumnRef.from_name_and_dtype(name, index.dtype) for name in self.BOUNDS]
        minter = MarkerMinter(self._plugin, CheckerLiteralTypeBuilder(), api)
        materialized = MaterializedColumn(minter)
        lazy = receiver.receiver_fullname in self.LAZY
        stored = self._stored()
        bounds: list[ColumnRef] = []
        for name in self.BOUNDS:
            if lazy and stored is not None:
                built = materialized.planned(name, self.DATETIME, stored)
            else:
                built = materialized.built(name, self.DATETIME)
            bounds.append(checked_not_none(built))
        return bounds

    @polars_releases
    def _stored(self) -> str | None:
        """Give the type that a lazy frame plans for a bound over a `Date` index, where it differs.

        Returns:
            `None`: the planner gives the type of the run.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> columns = WindowColumns(new_plugin(new_type_fixture()))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     columns._stored() is None
            True
        """
        return None

    @_stored.before("2.0.0")
    def _planned_as_date(self) -> str | None:
        """Give the type that a lazy frame of the older releases plans for such a bound.

        Returns:
            `Date`.

        Example:
            >>> WindowColumns(new_plugin(new_type_fixture()))._planned_as_date()
            'Date'
        """
        return "Date"
