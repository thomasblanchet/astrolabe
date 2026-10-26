"""Register the hooks that mypy calls for a method call on a frame, a group or a series.

mypy calls a method hook for a call `receiver.method(...)`, by the full
name of the method. This module registers the hooks of the frame
operations, such as `select`, `join`, `group_by` and `agg`, on
`TypedDataFrame` and `TypedLazyFrame`, and the hooks of a few methods of
`TypedSeries`. The recorder takes every call on an expression
(`RecordingHooks`), so no expression method has a hook here.

mypy gives the full name through the class of the receiver, not through
the class that defines the method. `**` on `int` and `float` gets a hook
too: mypy types a power of literals as `Any` otherwise.
"""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

import polars as pl
from mypy.plugin import MethodContext, Plugin
from mypy.types import Type as MypyType

from astrolabe._hooks.collect import CollectHook
from astrolabe._hooks.constant_power import ConstantPowerHook
from astrolabe._hooks.dispatch.generic import HookDispatcher, Target
from astrolabe._hooks.domain.dtype_families import DtypeFamilies
from astrolabe._hooks.domain.reducer_promotion import ReducerDtype
from astrolabe._hooks.domain.unchecked import UncheckedFallback
from astrolabe._hooks.domain.untyped_fallback import UntypedFallback
from astrolabe._hooks.drop import DropHook
from astrolabe._hooks.explode import ExplodeHook
from astrolabe._hooks.filter import FilterHook
from astrolabe._hooks.fill_nan import FrameFillNanHook
from astrolabe._hooks.fill_null import FrameFillNullHook
from astrolabe._hooks.frame_reducer import (
    CorrHook,
    WholeFrameReducerFallback,
    WholeFrameReducerHook,
)
from astrolabe._hooks.group_by.agg import AggHook
from astrolabe._hooks.group_by.group_by import GroupByHook
from astrolabe._hooks.group_by.shorthand import GroupByShorthandHook
from astrolabe._hooks.group_by.windows import WindowHook, WindowMethod
from astrolabe._hooks.hstack import HstackHook, SeriesListColumns
from astrolabe._hooks.join import JoinHook, JoinWhereHook
from astrolabe._hooks.join_asof import JoinAsofHook
from astrolabe._hooks.lazy import LazyHook
from astrolabe._hooks.map_columns import MapColumnsHook
from astrolabe._hooks.pivot import PivotHook
from astrolabe._hooks.rename import RenameHook
from astrolabe._hooks.row_counts import RowCountCheck, RowCountHook
from astrolabe._hooks.sort_keys import SortKeysHook
from astrolabe._hooks.select import SelectHook
from astrolabe._hooks.series_access import GetColumnHook, GetItemHook
from astrolabe._hooks.series_methods import CastHook, RoundHook, SeriesRenameHook
from astrolabe._hooks.unnest import UnnestHook
from astrolabe._hooks.unpivot import UnpivotConfig, UnpivotHook
from astrolabe._hooks.untyped_select import UntypedSelectHook
from astrolabe._hooks.update import UpdateHook
from astrolabe._hooks.vstack_extend import VstackExtendHook
from astrolabe._hooks.with_columns import WithColumnsHook
from astrolabe._hooks.with_row_index import WithRowIndexHook
from astrolabe._hooks.write_parquet import WriteParquetPartitionByHook
from astrolabe.frame import (
    TypedDataFrame,
    TypedDynamicGroupBy,
    TypedGroupBy,
    TypedLazyDynamicGroupBy,
    TypedLazyFrame,
    TypedLazyGroupBy,
    TypedLazyRollingGroupBy,
    TypedRollingGroupBy,
    TypedSeries,
)

if TYPE_CHECKING:
    from astrolabe._hooks.unproven import UnprovenPolicy


class MethodHooks:
    """Build the method hooks, and register each one by the full name of its method.

    The plugin asks this class for the hook of a method when mypy
    checks a call of it. The frame hooks share the objects that apply
    the `allow_unproven` setting, so that every operation reports in
    the same way.
    """

    _POWER_TYPES: Final[tuple[type, ...]] = (int, float)
    GROUP_BYS: Final[tuple[type, ...]] = (
        TypedGroupBy,
        TypedLazyGroupBy,
        TypedRollingGroupBy,
        TypedDynamicGroupBy,
        TypedLazyRollingGroupBy,
        TypedLazyDynamicGroupBy,
    )

    def __init__(
        self,
        plugin: Plugin,
        allow_unproven: UnprovenPolicy,
        elevate_untyped_select: bool,
    ) -> None:
        """Build every method hook, and register it in a new dispatcher.

        The hooks of the operations on the typed frames and on
        `TypedSeries` come in groups of related operations.

        Args:
            plugin: The mypy plugin, which the hooks use to look up a
                class by its full name.
            allow_unproven: The resolved `allow_unproven` setting. The
                hooks share one `UntypedFallback` and one
                `UncheckedFallback` built from it.
            elevate_untyped_select: Whether `.select()` on a plain frame
                gets `UntypedSelectHook`. See `AstrolabeConfig`.
        """
        self._dispatcher = HookDispatcher[MethodContext, MypyType]()
        untyped_fallback = UntypedFallback(allow_unproven, plugin)
        unchecked_fallback = UncheckedFallback(allow_unproven, plugin)
        self._register_reshape_hooks(plugin, untyped_fallback, unchecked_fallback)
        self._register_relational_hooks(plugin, untyped_fallback, unchecked_fallback)
        self._register_wide_reshape_hooks(plugin, untyped_fallback, unchecked_fallback)
        self._register_misc_frame_hooks(plugin, untyped_fallback, unchecked_fallback)
        self._register_frame_utility_hooks(plugin, untyped_fallback, unchecked_fallback)
        map_columns_hook = MapColumnsHook(untyped_fallback, plugin, unchecked_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "map_columns"), map_columns_hook)
        corr_hook = CorrHook(plugin, DtypeFamilies(plugin), untyped_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "corr"), corr_hook)
        self._register_whole_frame_reducer_hooks(plugin)
        self._register_power_hooks()
        if elevate_untyped_select:
            self._register_untyped_select_hook(plugin, unchecked_fallback)

    def _register_untyped_select_hook(
        self,
        plugin: Plugin,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Register the hook of `.select()` on a plain frame.

        Args:
            plugin: The mypy plugin, which looks up the classes of the
                result by full name.
            unchecked_fallback: The object that gives `Unchecked` to a
                requested column whose data type the checker cannot
                read.
        """
        untyped_select_hook = UntypedSelectHook(plugin, unchecked_fallback)
        self._dispatcher.register(Target.of(pl.DataFrame, "select"), untyped_select_hook)
        self._dispatcher.register(Target.of(pl.LazyFrame, "select"), untyped_select_hook)

    def _register_whole_frame_reducer_hooks(self, plugin: Plugin) -> None:
        """Register the frame methods that reduce every column to one row, such as `.sum()`.

        `product` exists only on `pl.DataFrame`, so it is registered
        only on `TypedDataFrame`. Every other reducer exists on both
        frame classes.

        Args:
            plugin: The mypy plugin, which looks up the data type class
                of each result by full name.
        """
        families = DtypeFamilies(plugin)
        reducer_dtype = ReducerDtype(plugin, families)
        fallback = WholeFrameReducerFallback(plugin, reducer_dtype, families)
        for reducer in WholeFrameReducerFallback.ALL_REDUCERS:
            reducer_hook = WholeFrameReducerHook(fallback, reducer, plugin)
            self._dispatcher.register(Target.of(TypedDataFrame, reducer), reducer_hook)
            if reducer != WholeFrameReducerFallback.PRODUCT:
                self._dispatcher.register(Target.of(TypedLazyFrame, reducer), reducer_hook)

    def _register_reshape_hooks(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Register the hooks of `select`, `drop`, `rename`, `with_columns`, `filter` and `remove`.

        Args:
            plugin: The mypy plugin, which the hooks use to look up a
                class by its full name.
            untyped_fallback: The object that falls back to the plain
                frame, which every hook here except the filter hooks
                receives.
            unchecked_fallback: The object that gives `Unchecked` to a
                column whose data type the checker cannot read, and
                handles a predicate whose data type the checker cannot
                read.
        """
        select_hook = SelectHook(untyped_fallback, plugin, unchecked_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "select"), select_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "select"), select_hook)
        self._dispatcher.register(Target.of(TypedDataFrame, "select_seq"), select_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "select_seq"), select_hook)

        drop_hook = DropHook(untyped_fallback, plugin)
        self._dispatcher.register(Target.of(TypedDataFrame, "drop"), drop_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "drop"), drop_hook)

        rename_hook = RenameHook(untyped_fallback, plugin)
        self._dispatcher.register(Target.of(TypedDataFrame, "rename"), rename_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "rename"), rename_hook)

        with_columns_hook = WithColumnsHook(untyped_fallback, plugin, unchecked_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "with_columns"), with_columns_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "with_columns"), with_columns_hook)
        self._dispatcher.register(Target.of(TypedDataFrame, "with_columns_seq"), with_columns_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "with_columns_seq"), with_columns_hook)

        for operation in ("filter", "remove"):
            filter_hook = FilterHook(plugin, unchecked_fallback, operation)
            self._dispatcher.register(Target.of(TypedDataFrame, operation), filter_hook)
            self._dispatcher.register(Target.of(TypedLazyFrame, operation), filter_hook)

    def _register_relational_hooks(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Register the hooks of the joins, `group_by`, `explode` and `unnest`.

        Args:
            plugin: The mypy plugin, which the join and group-by hooks
                use to look up a class by its full name.
            untyped_fallback: The object that falls back to the plain
                frame, which every hook here receives.
            unchecked_fallback: The object that reports a column that the
                checker cannot prove, which the predicates of `join_where` need.
        """
        join_hook = JoinHook(untyped_fallback, plugin)
        self._dispatcher.register(Target.of(TypedDataFrame, "join"), join_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "join"), join_hook)

        join_where_hook = JoinWhereHook(untyped_fallback, plugin, unchecked_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "join_where"), join_where_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "join_where"), join_where_hook)

        join_asof_hook = JoinAsofHook(untyped_fallback, plugin)
        self._dispatcher.register(Target.of(TypedDataFrame, "join_asof"), join_asof_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "join_asof"), join_asof_hook)

        group_by_hook = GroupByHook(plugin, untyped_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "group_by"), group_by_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "group_by"), group_by_hook)

        explode_hook = ExplodeHook(untyped_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "explode"), explode_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "explode"), explode_hook)

        unnest_hook = UnnestHook(untyped_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "unnest"), unnest_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "unnest"), unnest_hook)

    def _register_wide_reshape_hooks(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Register the hooks of `unpivot`, `melt` and `pivot`.

        `melt` does the same as `unpivot` with other parameter names,
        so it shares the hook class with a configuration of its own.

        Args:
            plugin: The mypy plugin, which the hooks use to look up a
                class by its full name.
            untyped_fallback: The object that falls back to the plain
                frame, which every hook here receives.
            unchecked_fallback: The object that gives `Unchecked` to a
                column whose data type the checker cannot read.
        """
        unpivot_config = UnpivotConfig()
        unpivot_hook = UnpivotHook(
            untyped_fallback,
            plugin,
            unchecked_fallback,
            unpivot_config,
        )
        self._dispatcher.register(Target.of(TypedDataFrame, "unpivot"), unpivot_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "unpivot"), unpivot_hook)

        melt_config = UnpivotConfig(on_param="value_vars", index_param="id_vars", operation="melt")
        melt_hook = UnpivotHook(
            untyped_fallback,
            plugin,
            unchecked_fallback,
            melt_config,
        )
        self._dispatcher.register(Target.of(TypedDataFrame, "melt"), melt_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "melt"), melt_hook)

        pivot_hook = PivotHook(untyped_fallback, plugin, unchecked_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "pivot"), pivot_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "pivot"), pivot_hook)

    def _register_misc_frame_hooks(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Register the hooks of `agg` and of the other operations that fit no group above.

        These are the group-by shorthands, `lazy`, `collect`,
        `with_row_index`, `fill_null` and `update`.

        Args:
            plugin: The mypy plugin, which the hooks use to look up a
                class by its full name.
            untyped_fallback: The object that falls back to the plain
                frame.
            unchecked_fallback: The object that gives `Unchecked` to a
                column whose data type the checker cannot read.
        """
        agg_hook = AggHook(plugin, untyped_fallback, unchecked_fallback)
        for group_by in self.GROUP_BYS:
            self._dispatcher.register(Target.of(group_by, "agg"), agg_hook)
        for frame in (TypedDataFrame, TypedLazyFrame):
            for method in (WindowMethod.ROLLING, WindowMethod.DYNAMIC):
                window_hook = WindowHook(plugin, untyped_fallback, method)
                self._dispatcher.register(Target.of(frame, method.name), window_hook)
        self._register_shorthands(plugin, untyped_fallback, unchecked_fallback)

        self._dispatcher.register(Target.of(TypedDataFrame, "lazy"), LazyHook(plugin))
        self._dispatcher.register(Target.of(TypedLazyFrame, "collect"), CollectHook(plugin))

        for method in ("head", "tail", "limit", "slice", "gather_every"):
            self._dispatcher.register(
                Target.of(TypedDataFrame, method),
                RowCountHook("DataFrame", method),
            )
            self._dispatcher.register(
                Target.of(TypedLazyFrame, method),
                RowCountHook("LazyFrame", method),
            )
        for frame, kind in ((TypedDataFrame, "DataFrame"), (TypedLazyFrame, "LazyFrame")):
            self._dispatcher.register(
                Target.of(frame, "sort"),
                SortKeysHook(plugin, unchecked_fallback, "sort"),
            )
            for method in ("top_k", "bottom_k"):
                counts = RowCountCheck(kind, method)
                hook = SortKeysHook(plugin, unchecked_fallback, method, counts)
                self._dispatcher.register(Target.of(frame, method), hook)

        with_row_index_hook = WithRowIndexHook(plugin, untyped_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "with_row_index"), with_row_index_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "with_row_index"), with_row_index_hook)

        fill_null_hook = FrameFillNullHook(plugin, untyped_fallback, unchecked_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "fill_null"), fill_null_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "fill_null"), fill_null_hook)

        fill_nan_hook = FrameFillNanHook(plugin, untyped_fallback, unchecked_fallback)
        self._dispatcher.register(Target.of(TypedDataFrame, "fill_nan"), fill_nan_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "fill_nan"), fill_nan_hook)

        update_hook = UpdateHook(untyped_fallback, unchecked_fallback, plugin)
        self._dispatcher.register(Target.of(TypedDataFrame, "update"), update_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "update"), update_hook)

    def _register_shorthands(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Register each group-by shorthand, such as `.sum()`, on both group-by classes.

        Args:
            plugin: The mypy plugin.
            untyped_fallback: The object that falls back to the plain
                frame for a call that the checker cannot read.
            unchecked_fallback: The object that gives `Unchecked` to a
                column whose data type the checker cannot read.
        """
        members = (*GroupByShorthandHook.RUNS, *GroupByShorthandHook.KEEPING)
        lazy = (TypedLazyGroupBy, TypedLazyRollingGroupBy, TypedLazyDynamicGroupBy)
        for member in members:
            hook = GroupByShorthandHook(plugin, member, untyped_fallback, unchecked_fallback)
            for group_by in (TypedGroupBy, *lazy):
                self._dispatcher.register(Target.of(group_by, member), hook)

    def _register_frame_utility_hooks(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Register the other frame and series operations, such as `hstack` and `get_column`.

        Args:
            plugin: The mypy plugin, which the hooks use to look up a
                class by its full name.
            untyped_fallback: The object that falls back to the plain
                frame. `HstackHook` and the two hooks that read one
                column as a series receive it. The two series hooks use
                it for a name that the frame does not declare.
            unchecked_fallback: The object that gives a column the data type
                `Unchecked`. The cast of a series uses it for a target that
                the hook cannot read.
        """
        series_columns = SeriesListColumns(plugin)
        hstack_hook = HstackHook(untyped_fallback, series_columns, plugin)
        self._dispatcher.register(Target.of(TypedDataFrame, "hstack"), hstack_hook)

        self._dispatcher.register(
            Target.of(TypedDataFrame, "vstack"),
            VstackExtendHook(".vstack()", plugin),
        )
        self._dispatcher.register(
            Target.of(TypedDataFrame, "extend"),
            VstackExtendHook(".extend()", plugin),
        )

        merge_sorted_hook = VstackExtendHook(".merge_sorted()", plugin, merges=True)
        self._dispatcher.register(Target.of(TypedDataFrame, "merge_sorted"), merge_sorted_hook)
        self._dispatcher.register(Target.of(TypedLazyFrame, "merge_sorted"), merge_sorted_hook)

        self._dispatcher.register(
            Target.of(TypedDataFrame, "write_parquet"),
            WriteParquetPartitionByHook(),
        )

        self._dispatcher.register(
            Target.of(TypedDataFrame, "__getitem__"),
            GetItemHook(plugin, untyped_fallback),
        )
        self._dispatcher.register(
            Target.of(TypedDataFrame, "get_column"),
            GetColumnHook(plugin, untyped_fallback),
        )

        cast_hook = CastHook(plugin, unchecked_fallback)
        self._dispatcher.register(Target.of(TypedSeries, "cast"), cast_hook)
        self._dispatcher.register(Target.of(TypedSeries, "round"), RoundHook(plugin))
        self._dispatcher.register(Target.of(TypedSeries, "alias"), SeriesRenameHook(plugin))
        self._dispatcher.register(Target.of(TypedSeries, "rename"), SeriesRenameHook(plugin))

    def _register_power_hooks(self) -> None:
        """Register `**` on `int` and `float`, so that a power of literals keeps a number type.

        mypy types `2**(N + 1)` as `Any`, and then calls no hook for an
        operation with it. `ConstantPowerHook` gives such a power the
        type of its value.
        """
        hook = ConstantPowerHook()
        for owner in self._POWER_TYPES:
            self._dispatcher.register(Target.of(owner, "__pow__"), hook)

    @property
    def dispatcher(self) -> HookDispatcher[MethodContext, MypyType]:
        """The dispatcher that holds every registered method hook."""
        return self._dispatcher
