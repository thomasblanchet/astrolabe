"""Give `df["a"]` and `df.get_column("a")` the name and the data type of the column.

polars types both reads as a plain `Series`, so the checker forgets which
column the series came from. These hooks give a `TypedSeries[name, dtype]`
instead, if the name is a literal string and the frame declares the
column. Else they keep the plain `Series`.

A schema has no order in this project, so no hook reads a column by its
position. `.to_series(index)` therefore keeps the plain `Series` that
polars declares.
"""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

import polars as pl
from mypy.types import get_proper_type

from astrolabe._hooks.calls.bare_literal import BareLiteralReader
from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.calls.dtype_constructor_readers import CheckerLiteralTypeBuilder
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.requested_columns import RequestedColumns
from astrolabe._hooks.domain.series import SeriesRef
from astrolabe._hooks.propagate import Refusal

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unproven_gate import UnprovenGate


class _SeriesColumnLookup:
    """Find one declared column of the receiver by name, and mint its `TypedSeries`.

    The two ways to read one column as a series share this work. It reads
    the receiver as a typed frame, finds the name among its declared
    columns and builds the series type through `SeriesRef`. To mint means
    to build a type in a hook.

    The receiver can be an open frame, that is a frame whose type does not
    name all of its columns. The columns that it does not name are its
    remainder. A declared column has the same data type whatever the
    remainder holds, so the lookup works on an open frame too.
    `RequestedColumns` reports a name that the frame does not declare.
    """

    def __init__(self, plugin: Plugin, requested_columns: RequestedColumns) -> None:
        """Keep the plugin and the rule that finds a requested name.

        Args:
            plugin: The mypy plugin, which finds the `TypeInfo` of
                `TypedSeries`.
            requested_columns: Finds a name among the declared columns,
                and reports a name that the frame does not declare.
        """
        self._plugin = plugin
        self._requested_columns = requested_columns
        self._literal_type = CheckerLiteralTypeBuilder()

    def by_name(self, ctx: MethodContext, name: str) -> MypyType:
        """Give the series type for one literal column name.

        mypy runs this hook only for a receiver whose own class is a typed
        frame, and a typed frame always holds its dictionary of columns. A
        declared column is present, with its data type, whatever the
        remainder holds. So the answer is the same for an open frame and for
        a closed one.

        Args:
            ctx: The context that mypy gives for the call.
            name: The literal name of the requested column.

        Returns:
            `TypedSeries[name, dtype]` if the receiver declares the
            column. Else the plain `Series` that polars declares. If the
            frame cannot have the column, an `UnknownColumnError` is
            reported first. If only the remainder can hold it, an
            `UndeclaredColumnError` is reported unless the
            `allow_unproven` setting lets it pass.
        """
        frame = checked_not_none(Frame.from_proper_type(ctx.type))
        known = checked_not_none(frame.known_columns)
        resolved = self._requested_columns.resolved(ctx, [name], known)
        if isinstance(resolved, Refusal):
            return ctx.default_return_type
        matched = resolved[0]
        literal = self._literal_type.string(checked_not_none(matched.name), ctx.api)
        built = SeriesRef.build(literal, matched.dtype, self._plugin)
        if built is None:  # pragma: no cover
            # `TypedSeries` is declared in the same module as the typed
            # frames. The hook runs only on a typed frame, so mypy has
            # already loaded that module.
            return ctx.default_return_type
        return built


class GetItemHook:
    """Give the type of `df[key]` on a typed frame, for a column name or a row selection.

    `__getitem__` accepts many shapes of key, and the declared types of
    polars are right for most of them. This hook changes two cases. A
    literal string key reads one column, so it gets a `TypedSeries` with
    the name and the data type of that column. An integer or a slice
    selects rows and keeps every column, so it gets the type of the
    receiver. Any other key keeps the type that polars declares.

    A string key reads a column in the same way as `.get_column()`. So
    it uses the same operation name for the `allow_unproven` override.
    It is a mypy method hook in the plugin layer.
    """

    _DATAFRAME_TARGET: Final[Target] = Target.of(pl.DataFrame)

    def __init__(self, plugin: Plugin, gate: UnprovenGate) -> None:
        """Build the column lookup, and the reader for a literal key.

        Args:
            plugin: The mypy plugin, which finds the `TypeInfo` of
                `TypedSeries`.
            gate: Decides whether an undeclared key is reported, from the
                `allow_unproven` setting.
        """
        self._lookup = _SeriesColumnLookup(plugin, RequestedColumns(gate, "get_column"))
        self._bare_literal = BareLiteralReader()

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `df[key]` expression.

        polars declares an overload of `__getitem__` that returns
        `DataFrame` for each key that selects rows. mypy has already chosen
        the overload, so its return type tells the key shape without a
        second reading of the key.

        Args:
            ctx: The context that mypy gives for `__getitem__`.

        Returns:
            `TypedSeries[name, dtype]` for a literal string key that names
            a declared column. The type of the receiver for a key that
            selects rows, for example an integer or a slice. Else the
            type that polars declares. The hook reports an error only for
            a string key.
        """
        name = self._literal_name(ctx)
        if name is not None:
            return self._lookup.by_name(ctx, name)
        chosen = get_proper_type(ctx.default_return_type)
        if self._DATAFRAME_TARGET.matches(chosen):
            return ctx.type
        return ctx.default_return_type

    def _literal_name(self, ctx: MethodContext) -> str | None:
        """Read the key as a string, if it is a literal.

        The one parameter of `__getitem__` is positional-only, and mypy
        gives it no name. So this method reads the first argument by
        position, and not by name through `CallArgs`.

        Args:
            ctx: The context that mypy gives for `__getitem__`.

        Returns:
            The string, or None if the key is not a literal string or a
            string variable declared `Final`.
        """
        nodes = ctx.args[0]
        if len(nodes) != 1:  # pragma: no cover
            # A subscript gives exactly one syntax node to the parameter.
            # Even `df[a, b]` gives one tuple node, not two nodes.
            return None
        return self._bare_literal.string(nodes[0], ctx.arg_types[0][0])


class GetColumnHook:
    """Give `.get_column(name)` on a typed frame the name and the data type of the column.

    The overloads of polars already use the `default=` argument. With no
    default, or a default that is a `Series`, the call returns `Series`.
    With any other default, it returns `Any`, because the call can return
    that default. This hook changes only the first case, and it reads the
    overload that mypy chose to tell them apart. It is a mypy method hook
    in the plugin layer.
    """

    _SERIES_TARGET: Final[Target] = Target.of(pl.Series)

    def __init__(self, plugin: Plugin, gate: UnprovenGate) -> None:
        """Build the column lookup.

        Args:
            plugin: The mypy plugin, which finds the `TypeInfo` of
                `TypedSeries`.
            gate: Decides whether an undeclared name is reported, from the
                `allow_unproven` setting.
        """
        self._lookup = _SeriesColumnLookup(plugin, RequestedColumns(gate, "get_column"))

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.get_column(name)` call.

        Args:
            ctx: The context that mypy gives for `.get_column()`.

        Returns:
            `TypedSeries[name, dtype]` for a literal name that the frame
            declares, if the overload returns `Series`. Else the type
            that polars declares.
        """
        chosen = get_proper_type(ctx.default_return_type)
        if not self._SERIES_TARGET.matches(chosen):
            return ctx.default_return_type
        name = CallArgs(ctx).literal_str("name")
        if name is None:
            return ctx.default_return_type
        return self._lookup.by_name(ctx, name)
