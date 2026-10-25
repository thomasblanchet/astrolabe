"""The hooks for a reducer on a whole frame, for example `df.sum()`, and for `df.corr()`.

A reducer called on a frame turns each column into one value, so the
result has one row and the same column names. The data type of each
column can change: `mean()` gives a float, and `count()` gives `UInt32`.
Without these hooks, mypy gives the result the type of the receiver,
and the checker then believes the old data types.

`sum`, `max`, `min`, `first`, `last`, `mean`, `std`, `var` and `median`
share their data type rules with `.agg()`, through `ReducerDtype`. On a
whole frame, polars does not refuse a column outside the family that a
reducer accepts. The shared rules then keep that column's data type.
`median` on a `Decimal` column, `count`, `null_count` and `product` have
rules of their own on a whole frame. `WholeFrameReducerFallback` holds
them.
"""

from __future__ import annotations

import functools
from typing import ClassVar, Final, TYPE_CHECKING

import polars as pl
from mypy.types import get_proper_type, Instance

from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.frame import Frame, KnownColumns
from astrolabe._hooks.domain.planned import PlannedMap
from astrolabe._hooks.domain.reducer_promotion import ReducerDtype, ReducerPromotion
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.remainder.type_set import Family
from astrolabe._hooks.errors import DuplicateColumnError, WrongColumnShapeError
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe._hooks.expression.outcome import UNCHECKED
from astrolabe._hooks.unproven import Reason
from astrolabe.frame import Unchecked

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.plugin import MethodContext, Plugin

    from astrolabe._hooks.domain.dtype_families import DtypeFamilies
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


class WholeFrameReducerFallback:
    """Give the data type of one column after a reducer on a whole frame.

    A reducer on a whole frame mostly follows the same data type rules as
    the same reducer in `.agg()`, and this class passes those cases to
    `ReducerDtype`. Four reducers differ, and this class holds their rules:
    `count` and `null_count` give `UInt32`, `product` widens small integers
    to `Int64`, and `median` keeps a `Decimal` column. Code that types the
    result of a reducer on a frame asks this class once for each column.

    Example:
        >>> fixture = new_type_fixture()
        >>> plugin = new_plugin(fixture)
        >>> families = dtype_families(plugin)
        >>> fallback = WholeFrameReducerFallback(plugin, reducer_dtype(plugin), families)
        >>> fallback.resolve("null_count", int_dtype(fixture))
        polars.datatypes.classes.UInt32
    """

    NULL_COUNT: Final[str] = "null_count"
    COUNT: Final[str] = "count"
    PRODUCT: Final[str] = "product"
    MEDIAN: Final[str] = "median"
    ALL_REDUCERS: Final[frozenset[str]] = (
        ReducerDtype.IDENTITY_REDUCERS
        | ReducerDtype.FLOAT_COERCE_REDUCERS
        | {ReducerDtype.SUM_REDUCER, NULL_COUNT, COUNT, PRODUCT}
    )

    _UINT32_TARGET: Final[Target] = Target.of(pl.UInt32)
    _INT64_TARGET: Final[Target] = Target.of(pl.Int64)
    _NULL_TARGET: Final[Target] = Target.of(pl.Null)
    _DECIMAL_TARGET: Final[Target] = Target.of(pl.Decimal)
    # product() turns each integer of fewer than 64 bits into Int64.
    # This set is wider than the one for sum(), which keeps Int32 and
    # UInt32. Int64, UInt64 and Int128 stay as they are.
    _PRODUCT_WIDENS_TO_INT64: Final[frozenset[str]] = frozenset({
        Target.of(pl.Int8).fullname,
        Target.of(pl.Int16).fullname,
        Target.of(pl.Int32).fullname,
        Target.of(pl.UInt8).fullname,
        Target.of(pl.UInt16).fullname,
        Target.of(pl.UInt32).fullname,
    })

    def __init__(
        self,
        plugin: PluginLookup,
        reducer_dtype: ReducerDtype,
        families: DtypeFamilies,
    ) -> None:
        """Keep the shared reducer rules, and the tools for the four rules of this class.

        Args:
            plugin: The mypy plugin, which looks up the polars classes
                `UInt32`, `Int64` and `Null`.
            reducer_dtype: The rules that the reducers share with `.agg()`.
            families: The tests for a family of data types, and the widening of
                a `Decimal`, which the `product` rule needs.
        """
        self._type_info = FreshTypeInfo(plugin)
        self._reducer_dtype = reducer_dtype
        self._families = families
        self._promotion = ReducerPromotion()
        self._unchecked = UncheckedCompatibility()

    def resolve(self, reducer: str, source_dtype: MypyType) -> MypyType | None:
        """Give the data type of one column after one reducer.

        Args:
            reducer: The name of the reducer, one of `ALL_REDUCERS`.
            source_dtype: The data type of the column before the reducer.

        Returns:
            The data type after the reducer, or None if the lookup of a polars
            class fails.

        Example: see class docstring.
        """
        own_rules: dict[str, Callable[[MypyType], MypyType | None]] = {
            self.NULL_COUNT: self._counted_dtype,
            self.COUNT: self._counted_dtype,
            self.PRODUCT: self._product_dtype,
            self.MEDIAN: self._median_dtype,
        }
        own_rule = own_rules.get(reducer)
        if own_rule is not None:
            return own_rule(source_dtype)
        return self._reducer_dtype.resolve(reducer, source_dtype)

    def _counted_dtype(self, source_dtype: MypyType) -> MypyType | None:
        """Give the data type after `count()` or `null_count()`, which is `UInt32` for every column.

        Args:
            source_dtype: The data type of the column. The rule does not
                read it.

        Returns:
            `UInt32`.
        """
        del source_dtype
        return self._fresh(self._UINT32_TARGET.fullname)

    def _median_dtype(self, source_dtype: MypyType) -> MypyType | None:
        """Give the data type after `median()` on a whole frame.

        On a whole frame, `median()` keeps a `Decimal` column as it is. Every
        other reducer that gives a float, and `median()` in a pivot, turn a
        `Decimal` into `Float64`.

        Args:
            source_dtype: The data type of the column.

        Returns:
            The same `Decimal` type for a `Decimal` column. Otherwise the shared
            rule.
        """
        if self._DECIMAL_TARGET.is_base_of(get_proper_type(source_dtype)):
            return source_dtype
        return self._reducer_dtype.resolve(self.MEDIAN, source_dtype)

    def _product_dtype(self, source_dtype: MypyType) -> MypyType | None:
        """Give the data type after `product()`.

        Args:
            source_dtype: The data type of the column.

        Returns:
            The same type if it is `Unchecked` or not a class instance, because
            the rule has nothing to read. `Int64` for `Boolean` and for an
            integer of fewer than 64 bits. The same type for `Int64`, `UInt64`,
            `Int128`, `Float32` and `Float64`. A `Decimal` with a precision of
            38 for a `Decimal`. `Null` for any other data type, for example
            `String`, `List`, `Date`, `Datetime` or `Duration`.
        """
        proper = get_proper_type(source_dtype)
        if not isinstance(proper, Instance) or self._unchecked.matches(proper):
            return source_dtype
        fullname = proper.type.fullname
        if self._promotion.is_boolean(fullname) or fullname in self._PRODUCT_WIDENS_TO_INT64:
            return self._fresh(self._INT64_TARGET.fullname)
        if self._families.is_numeric_or_boolean(proper):
            return proper
        widened_decimal = self._families.widened_decimal(proper)
        if widened_decimal is not None:
            return widened_decimal
        return self._fresh(self._NULL_TARGET.fullname)

    def _fresh(self, fullname: str) -> MypyType | None:
        """Build a polars data type with no parameters, from its full name.

        Args:
            fullname: The full name of the polars data type class.

        Returns:
            The new type, or None if the lookup of the class fails.
        """
        info = self._type_info.resolved(fullname)
        if info is None:  # pragma: no cover
            # Each name here is a built-in polars data type, and mypy
            # always loads polars before a reducer needs one.
            return None
        return Instance(info, [])


class WholeFrameReducerHook:
    """Give the result type of one reducer called on a whole frame.

    The plugin registers one instance for each reducer name, for example
    `sum` or `max`, on both `TypedDataFrame` and `TypedLazyFrame`. The hook
    keeps the columns in order and replaces the data type of each declared
    column with the answer of `WholeFrameReducerFallback`.

    An open frame has a remainder: the columns that its type does not name.
    The reducer keeps the names of those columns, and the hook changes the
    data type that the remainder carries. `min`, `max`, `first` and `last`
    keep each data type. `count` and `null_count` give `UInt32`. Any other
    reducer gives an answer that depends on the data type of each column,
    and the remainder does not name those, so the hook marks them
    `Unchecked`.
    """

    # The data type that a reducer gives to every column of the
    # remainder. A reducer that keeps each value keeps each data type.
    # Any other reducer that is not here gives `Unchecked`, because its
    # answer depends on the data type of each column.
    REST_DTYPES: ClassVar[dict[str, str]] = {
        WholeFrameReducerFallback.COUNT: "UInt32",
        WholeFrameReducerFallback.NULL_COUNT: "UInt32",
    }

    def __init__(self, fallback: WholeFrameReducerFallback, reducer: str, plugin: Plugin) -> None:
        """Keep the rules for each column, the name of the reducer and the plugin.

        Args:
            fallback: The rules that give the data type of each column.
            reducer: The name of the reducer that this instance serves.
            plugin: The mypy plugin, which mints the new remainder.
        """
        self._fallback = fallback
        self._reducer = reducer
        self._plugin = plugin

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the result type of one call of the reducer.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The frame with the data type of each column reduced. If the receiver
            is not a typed frame, or a lookup fails, the type that mypy chose.
        """
        result = self._result(ctx)
        if result is None:  # pragma: no cover
            # The plugin calls this hook only for a typed frame, and a
            # column that no rule answers keeps its name as `Unchecked`.
            return ctx.default_return_type
        return result

    @propagates(None)
    def _result(self, ctx: MethodContext) -> MypyType | None:
        """Build the reduced frame, or give up at the first step that fails.

        Each declared column keeps its name and gets the data type that the
        rules give. A column of a lazy frame that holds a stored type of its
        own gets the reducer of each of its two types. A data type that no
        rule answers, for example a type variable of a generic schema class,
        depends on the caller. The column then gets the data type
        `Unchecked`, as the remainder does.

        Every reducer keeps the name of each column of the remainder. A
        reducer that gives each value back unchanged keeps the data type too.
        A count gives `UInt32` to every column. Any other reducer gives an
        answer that depends on the data type of the column, which the
        remainder does not name, so the result is `Unchecked`.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The result frame, or None if a step fails.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        reduced: list[ColumnRef] = []
        rule = functools.partial(self._fallback.resolve, self._reducer)
        planned = PlannedMap(self._plugin)
        for column in known.columns:
            dtype = planned.mapped(column.dtype, rule)
            if dtype is None:
                unchecked = FreshTypeInfo(self._plugin).resolved(Target.of(Unchecked).fullname)
                dtype = Instance(checked_not_none(unchecked), [])
            name = not_none(column.name)
            reduced.append(ColumnRef.build(name, dtype, column, ctx.api))
        rest = known.rest
        if rest is not None and self._reducer not in ReducerDtype.IDENTITY_REDUCERS:
            edit = RemainderEdit.for_checker(self._plugin, ctx.api)
            rest = edit.retyped(rest, self.REST_DTYPES.get(self._reducer, UNCHECKED))
        return frame.rebuilt(reduced, ctx.api, rest=rest).as_instance()


class CorrHook:
    """Give the result type of `DataFrame.corr()`, and check that polars can correlate each column.

    Polars correlates each pair of columns. The result has one `Float64`
    column for each column of the receiver, with the same name, and one row
    for each column. With `label=`, the result starts with a `String` column
    of that name, which holds the column names. Polars raises if a column is
    not numeric or `Boolean`, so the hook reports such a column.

    On an open frame, the hook reads what the remainder can hold. If a
    column of the remainder can have a data type that polars cannot
    correlate, the result depends on the caller, and the hook falls back:
    it gives the plain frame type in place of a typed result, subject to
    the `allow_unproven` setting.
    """

    _FLOAT64_TARGET: Final[Target] = Target.of(pl.Float64)
    _STRING_TARGET: Final[Target] = Target.of(pl.String)

    def __init__(
        self,
        plugin: PluginLookup,
        families: DtypeFamilies,
        untyped_fallback: UntypedFallback,
    ) -> None:
        """Keep the lookups, the family tests and the fallback that the hook needs.

        Args:
            plugin: The mypy plugin, which looks up `Float64` and `String` and
                mints the new remainder.
            families: The test for a numeric or `Boolean` data type.
            untyped_fallback: The fallback that gives the plain frame type,
                subject to the `allow_unproven` setting.
        """
        self._type_info = FreshTypeInfo(plugin)
        self._plugin = plugin
        self._families = families
        self._untyped = untyped_fallback
        self._unchecked = UncheckedCompatibility()

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the result type of one `.corr()` call.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The frame of correlations. The plain frame type after an error, or
            if the label is not a literal string.
        """
        frame = Frame.from_proper_type(ctx.type)
        known = frame.known_columns if frame is not None else None
        if frame is None or known is None:  # pragma: no cover
            # The plugin calls this hook only for a typed frame, whose
            # schema is always a `TypedDict`.
            return ctx.default_return_type
        columns = known.columns
        if not self._in_domain(ctx, columns) or self._label_collides(ctx, columns):
            return self._untyped.resolve(ctx, Reason.REPORTED, "corr", frame)
        if not self._rest_correlates(known.rest):
            return self._untyped.resolve(ctx, Reason.CALLER_DEPENDENT, "corr", frame)
        built = self._built(ctx, frame, known)
        if built is None:
            return self._untyped.resolve(ctx, Reason.DYNAMIC_NAME, "corr", frame)
        return built

    def _rest_correlates(self, rest: MypyType | None) -> bool:
        """Tell whether polars can correlate every column that the remainder can hold.

        Polars can correlate the numeric data types and `Boolean`.

        Args:
            rest: The remainder of the receiver, or None for a closed frame.

        Returns:
            True for a closed frame. True if the remainder can hold only numeric
            and `Boolean` data types.
        """
        if rest is None:
            return True
        record = RemainderKnowledge().record(rest)
        if record is None:  # pragma: no cover
            # The remainder of a typed frame is always a declaration, a
            # split, or an effect on one of them. The engine can read
            # each of those forms.
            return False
        boolean = Family("Boolean", frozenset({"Boolean"})).type_set()
        correlated = Family.numeric().type_set().union(boolean)
        return record.types.within(correlated)

    def _in_domain(self, ctx: MethodContext, columns: list[ColumnRef]) -> bool:
        """Report the first declared column that polars cannot correlate.

        Polars can correlate a numeric or `Boolean` column. The checker does
        not know the data type of an `Unchecked` column, so that column passes.

        Args:
            ctx: The mypy context of the method call.
            columns: The declared columns of the receiver.

        Returns:
            True if every column is numeric, `Boolean` or `Unchecked`.
            Otherwise the method reported an error and gives False.
        """
        for column in columns:
            dtype = column.dtype
            if self._families.is_numeric_or_boolean(dtype) or self._unchecked.matches(dtype):
                continue
            error = WrongColumnShapeError.on_column(
                str(column.name),
                expected="numeric or Boolean",
                operation="corr",
            )
            error.fail(ctx)
            return False
        return True

    def _label_collides(self, ctx: MethodContext, columns: list[ColumnRef]) -> bool:
        """Report a label that is already the name of a column.

        Args:
            ctx: The mypy context of the method call.
            columns: The declared columns of the receiver.

        Returns:
            True if the label repeats a column name. The method then reported
            the duplicate.
        """
        label = CallArgs(ctx).literal_str("label")
        if label is None or label not in {column.name for column in columns}:
            return False
        DuplicateColumnError("corr", label).fail(ctx)
        return True

    @propagates(None)
    def _built(self, ctx: MethodContext, frame: Frame, known: KnownColumns) -> MypyType | None:
        """Build the frame of correlations.

        Each declared column becomes a `Float64` column of the same name. If
        the call gives a label, a `String` column of that name comes first.
        Every column of the remainder becomes `Float64` too.

        Args:
            ctx: The mypy context of the method call.
            frame: The receiver.
            known: The declared columns and the remainder of the receiver.

        Returns:
            The result frame, or None if the label is not a literal string.
        """
        float64 = not_none(self._instance(self._FLOAT64_TARGET.fullname))
        correlated = [
            ColumnRef.build(not_none(column.name), float64, column, ctx.api)
            for column in known.columns
        ]
        args = CallArgs(ctx)
        if not args.is_omitted_or_none("label"):
            label = not_none(args.literal_str("label"))
            text = not_none(self._instance(self._STRING_TARGET.fullname))
            correlated = [ColumnRef.from_name_and_dtype(label, text), *correlated]
        rest = known.rest
        if rest is not None:
            rest = RemainderEdit.for_checker(self._plugin, ctx.api).retyped(rest, "Float64")
        return frame.rebuilt(correlated, ctx.api, rest=rest).as_instance()

    @propagates(None)
    def _instance(self, fullname: str) -> Instance | None:
        """Build a polars data type with no parameters, from its full name.

        Args:
            fullname: The full name of the polars data type class.

        Returns:
            The new type, or None if the lookup of the class fails.
        """
        return Instance(not_none(self._type_info.resolved(fullname)), [])
