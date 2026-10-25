"""Register the hooks that mypy calls for an assignment such as `Name = column(...)`.

mypy calls a dynamic-class hook for an assignment `Name = f(...)` if
the plugin registers a hook for `f`. The hook runs during semantic
analysis, the phase that resolves names, and it can declare `Name` as
a new class or type. The declarations of this project work this way:
`column`, `typed_expr`, `typed_agg`, `typed_lit`, `typed_selection`,
`typed_agg_selection`, `other_columns` and `columns`.

Each polars data type class also gets a hook. With it,
`Stamp = pl.Datetime("us")` keeps its time unit, and a declaration
that reads `Stamp` later gets the parameters of the data type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.plugin import DynamicClassDefContext

from astrolabe._hooks.declaring.dtype_alias import DtypeAliasHook
from astrolabe._hooks.declaring.hooks import (
    ColumnDeclarationHook,
    SelectionDeclarationHook,
    SliceDeclarationHook,
)
from astrolabe._hooks.declaring.schema_hook import SchemaDeclarationHook
from astrolabe._hooks.dispatch.generic import HookDispatcher, Target
from astrolabe._hooks.dynamic_class import RemainderDeclarationHook
from astrolabe._hooks.expression.kind import KindRoster
from astrolabe.remainder import other_columns
from astrolabe.declare import (
    typed_agg,
    typed_agg_selection,
    typed_expr,
    typed_lit,
    typed_selection,
)
from astrolabe.frame import column, columns

if TYPE_CHECKING:
    from astrolabe._hooks.domain.frame import PluginLookup


class DynamicClassHooks:
    """Build the dynamic-class hooks, and register each one by the full name of its function.

    The plugin asks this class for the hook of a function when mypy
    finds an assignment from a call of that function. Each declaration
    of an expression gets its character here: `typed_expr` works row by
    row, `typed_agg` combines rows into one value, and `typed_lit` is a
    fixed value.
    """

    def __init__(self, plugin: PluginLookup) -> None:
        """Build every dynamic-class hook, and register it in a new dispatcher.

        Args:
            plugin: The mypy plugin, which the declarations use to look
                up a class by its full name.
        """
        self._dispatcher = HookDispatcher[DynamicClassDefContext, None]()
        self._dispatcher.register(Target.of(column), ColumnDeclarationHook(plugin))
        self._dispatcher.register(Target.of(other_columns), RemainderDeclarationHook())
        self._dispatcher.register(Target.of(columns), SchemaDeclarationHook())
        characters = {typed_expr: "elementwise", typed_agg: "aggregating", typed_lit: "literal"}
        for function, character in characters.items():
            self._dispatcher.register(Target.of(function), SliceDeclarationHook(plugin, character))
        selections = {typed_selection: "elementwise", typed_agg_selection: "aggregating"}
        for function, character in selections.items():
            self._dispatcher.register(
                Target.of(function),
                SelectionDeclarationHook(plugin, character),
            )
        # The hook keeps the data type that an assignment such as
        # `Stamp = pl.Datetime("us")` holds, with its parameters, for a
        # later declaration to read.
        hook = DtypeAliasHook()
        for kind in KindRoster().classes():
            self._dispatcher.register(Target.of(kind), hook)

    @property
    def dispatcher(self) -> HookDispatcher[DynamicClassDefContext, None]:
        """The dispatcher that holds every registered dynamic-class hook."""
        return self._dispatcher
