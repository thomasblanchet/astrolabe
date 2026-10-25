"""Type `.unpivot()` and `.melt()` on a typed frame: fold columns into name and value pairs.

`.unpivot(on=[...], index=[...])` turns a wide frame into a long one. It
keeps each `index` column. It replaces the `on` columns with two new
columns: `variable`, a `String` that holds the name of the folded column,
and `value`, which holds its value. The `value` column has the supertype
of the data types of all `on` columns, that is the smallest data type that
holds the values of each. The keyword arguments `variable_name=` and
`value_name=` rename the two new columns.

The hook needs literal names in `on` and `index`. A selector, or a name
that the hook cannot read, makes it fall back to the plain frame.

`.melt()` is the older name of `.unpivot()`. It takes `id_vars` for `index`
and `value_vars` for `on`, and behaves the same. So one hook class serves
both methods, and an `UnpivotConfig` holds the parameter names of each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TYPE_CHECKING

import polars as pl
from mypy.nodes import TypeInfo
from mypy.types import Instance, get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.dtype_promotion import DTypePromotion
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.requested_columns import RequestedColumns
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.propagate import Refusal, explains, not_none, not_refused, propagates

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin

    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


class _ValueDtypeResolver:
    """Compute the data type of the `value` column from the data types of the `on` columns.

    The data type is the supertype of all `on` columns. `DTypePromotion`
    computes it one pair at a time. For a pair that it does not model, the
    column gets the data type `Unchecked`, which means that its name is
    known and its data type is not. `UncheckedFallback` reports that unless
    the `allow_unproven` setting lets it pass. An empty `on` gives
    `pl.Null`, as in polars.

    This class keeps the promotion rule, the fallback and the plugin
    together, so that the code that reads the call needs only one object
    for the `value` column.
    """

    _NULL_TARGET: Final[Target] = Target.of(pl.Null)

    def __init__(
        self,
        promotion: DTypePromotion,
        unchecked_fallback: UncheckedFallback,
        plugin: Plugin,
    ) -> None:
        """Keep the promotion rule, the fallback and the plugin.

        Args:
            promotion: The rule that gives the supertype of two data types.
            unchecked_fallback: Gives `Unchecked` for a pair that the rule
                does not model, and reports it unless the setting lets it
                pass.
            plugin: The mypy plugin, which finds `pl.Null` for an empty
                `on`.
        """
        self._promotion = promotion
        self._unchecked_fallback = unchecked_fallback
        self._plugin = plugin

    def resolved(
        self,
        ctx: MethodContext,
        on_columns: list[ColumnRef],
        operation: str,
        value_name: str,
    ) -> MypyType:
        """Give the data type of the `value` column.

        An empty `on` is a valid call. polars gives the `value` column the
        data type `Null`, whatever the other columns are.

        Args:
            ctx: The context that mypy gives for the call.
            on_columns: The `on` columns, in call order.
            operation: The name of the method, for the error message.
            value_name: The name of the `value` column, for the error
                message.

        Returns:
            `pl.Null` for an empty `on`. Else the supertype of the `on`
            columns, or `Unchecked` if the promotion rule does not model a
            pair.
        """
        if not on_columns:
            return checked_not_none(self._null_dtype())
        dtype = self._folded_value_dtype(on_columns)
        if dtype is not None:
            return dtype
        substituted = self._unchecked_fallback.substituted(
            ctx,
            Reason.CHECKER_GAP,
            operation,
            value_name,
        )
        return checked_not_none(substituted)

    def _null_dtype(self) -> MypyType | None:
        """Build the type `pl.Null`.

        Returns:
            An instance type of `pl.Null`, or None if mypy cannot find the
            class.
        """
        info = self._plugin.lookup_fully_qualified(self._NULL_TARGET.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # `pl.Null` is part of polars, and mypy has always loaded
            # polars when a frame method runs.
            return None
        return Instance(info.node, [])

    @propagates(None)
    def _folded_value_dtype(self, columns: list[ColumnRef]) -> MypyType | None:
        """Combine the data types of the `on` columns into one supertype, a pair at a time.

        The promotion rule widens within one family of numbers. It puts a
        signed and an unsigned integer in a signed integer wide enough for
        both. It lets `String` or a float take in a narrower data type.

        Args:
            columns: The `on` columns, in call order. The list is not
                empty.

        Returns:
            The data type of the one column if there is one. Else the
            supertype of all of them, or None if a data type is not an
            instance type or the rule does not model a pair.

        Example:
            >>> fixture = new_type_fixture()
            >>> promotion = DTypePromotion(unchecked_compatibility(), new_plugin(fixture))
            >>> plugin = new_plugin(fixture)
            >>> resolver = _ValueDtypeResolver(promotion, unchecked_fallback(), plugin)
            >>> a = column_ref(fixture, "a", int_dtype(fixture))
            >>> b = column_ref(fixture, "b", str_dtype(fixture))
            >>> resolver._folded_value_dtype([a, b])
            polars.datatypes.classes.String
            >>> c = column_ref(fixture, "c", int32_dtype(fixture))
            >>> d = column_ref(fixture, "d", int64_dtype(fixture))
            >>> resolver._folded_value_dtype([c, d])
            polars.datatypes.classes.Int64
        """
        dtype: MypyType = columns[0].dtype
        for column in columns[1:]:
            proper_acc = get_proper_type(dtype)
            proper_next = get_proper_type(column.dtype)
            if not (isinstance(proper_acc, Instance) and isinstance(proper_next, Instance)):
                return None
            dtype = not_none(self._promotion.promoted(proper_acc, proper_next))
        return dtype


@dataclass(frozen=True)
class _UnpivotPlan:
    """Hold what the hook reads from one `.unpivot()` or `.melt()` call.

    Attributes:
        frame: The receiver frame.
        index_columns: The `index` columns, which the result keeps as
            they are, in call order.
        model: A column of the receiver whose class record of `Column`
            the two new columns reuse, or None if the receiver declares no
            column.
        value_dtype: The data type of the `value` column.
        variable_name: The literal name of the `variable` column.
        value_name: The literal name of the `value` column.
    """

    frame: Frame
    index_columns: list[ColumnRef]
    model: ColumnRef | None
    value_dtype: MypyType
    variable_name: str
    value_name: str

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(
        cls,
        ctx: MethodContext,
        requested_columns: RequestedColumns,
        value_resolver: _ValueDtypeResolver,
        config: UnpivotConfig,
    ) -> _UnpivotPlan:
        """Read the receiver, the `index` and `on` columns and the two new names.

        If the call gives no `on`, polars folds every column that is not
        in `index`. The hook can list those columns only for a closed
        frame, whose type names all of its columns. An open frame also
        has a remainder of columns with no known names, so the hook gives
        up. An empty `on`, given or left over after `index`, is valid.

        Args:
            ctx: The context that mypy gives for the call.
            requested_columns: Finds each named column among the declared
                columns, and reports a name that the frame does not
                declare.
            value_resolver: Gives the data type of the `value` column.
            config: The parameter names and the operation name of the
                method.

        Returns:
            The plan, or a `Refusal`. The reason is `DYNAMIC_NAME` if a
            name argument is not literal. It is `REPORTED` if a named
            column does not exist, after the error. It is `CHECKER_GAP`
            if the receiver is not a typed frame, or if `on` is missing
            and the frame is open.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        args = CallArgs(ctx)
        index_names = not_none(
            args.literal_str_list_or_empty(config.index_param),
            reason=Reason.DYNAMIC_NAME,
        )
        index_columns = not_refused(requested_columns.resolved(ctx, index_names, known))
        excluded = set(index_names)
        on_names = not_none(
            RequestedColumns.names_or_remaining(args, config.on_param, known, excluded),
            reason=RequestedColumns.unread_reason(args, config.on_param),
        )
        on_columns = not_refused(requested_columns.resolved(ctx, on_names, known))
        variable_name = not_none(
            cls._resolved_str_param(args, "variable_name", "variable"),
            reason=Reason.DYNAMIC_NAME,
        )
        value_name = not_none(
            cls._resolved_str_param(args, "value_name", "value"),
            reason=Reason.DYNAMIC_NAME,
        )
        value_dtype = value_resolver.resolved(ctx, on_columns, config.operation, value_name)
        model = cls._model_column(on_columns, known.columns)
        return cls(frame, index_columns, model, value_dtype, variable_name, value_name)

    @staticmethod
    def _model_column(
        on_columns: list[ColumnRef],
        known_columns: list[ColumnRef],
    ) -> ColumnRef | None:
        """Pick a column of the receiver for the two new columns to copy their class from.

        A new column built from an existing one reuses its class record
        of `Column`, so the plugin needs no lookup. The first `on` column
        serves. If `on` is empty, any declared column serves.

        Args:
            on_columns: The `on` columns, in call order.
            known_columns: The declared columns of the receiver.

        Returns:
            A column, or None if the receiver declares no column, for
            example `TypedDataFrame[()]`.
        """
        if on_columns:
            return on_columns[0]
        if known_columns:
            return known_columns[0]
        return None

    @staticmethod
    def _resolved_str_param(args: CallArgs, param: str, default: str) -> str | None:
        """Read a string argument, with the default of polars if it is missing or None.

        Args:
            args: The arguments of the call.
            param: The name of the parameter.
            default: The name that polars uses if the argument is missing
                or None.

        Returns:
            `default` if the argument is missing or None. Else its literal
            string, or None if it is not a literal string.

        Example:
            >>> args = call_args(["variable_name"], [[]])
            >>> _UnpivotPlan._resolved_str_param(args, "variable_name", "variable")
            'variable'
            >>> args = call_args(["variable_name"], [[str_literal_node("kind")]])
            >>> _UnpivotPlan._resolved_str_param(args, "variable_name", "variable")
            'kind'
        """
        if args.is_omitted_or_none(param):
            return default
        return args.literal_str(param)


@dataclass(frozen=True)
class UnpivotConfig:
    """Name the parameters of `.unpivot()` or of `.melt()`, for one instance of the hook.

    The two methods do the same thing with different parameter names. This
    is the only part that differs, so the hook takes it as data, and one
    hook class serves both.

    Attributes:
        on_param: The parameter that names the folded columns: `"on"`
            for `.unpivot()`, `"value_vars"` for `.melt()`.
        index_param: The parameter that names the kept columns:
            `"index"` for `.unpivot()`, `"id_vars"` for `.melt()`.
        operation: The name of the method, `"unpivot"` or `"melt"`, for
            error messages and the `allow_unproven` override.
    """

    on_param: str = "on"
    index_param: str = "index"
    operation: str = "unpivot"


class UnpivotHook:
    """Give `.unpivot()` or `.melt()` on a typed frame its result: the `index` columns and two more.

    The result holds the `index` columns, then `variable` and `value`.
    polars fails if two of these share a name, and the hook reports that.

    The receiver can be an open frame, whose type does not name all of its
    columns. The hook still types the call if it names `on` and every named
    column is declared. The result holds only the named columns and the
    two new ones, so the unnamed columns of the receiver cannot reach it.

    It is a mypy method hook in the plugin layer. If it cannot type the
    call, it falls back through `UntypedFallback` to the plain frame that
    polars declares.
    """

    _STRING_TARGET: Final[Target] = Target.of(pl.String)

    def __init__(
        self,
        untyped_fallback: UntypedFallback,
        plugin: Plugin,
        unchecked_fallback: UncheckedFallback,
        config: UnpivotConfig,
    ) -> None:
        """Build the helpers that the hook needs for each call.

        Args:
            untyped_fallback: Gives the plain frame for a call that the
                hook cannot type, and reports each unproven result unless
                the `allow_unproven` setting lets it pass.
            plugin: The mypy plugin, which finds `pl.String` for the
                `variable` column.
            unchecked_fallback: Gives `Unchecked` to a `value` column
                whose supertype the hook does not know.
            config: The parameter names and the operation name of the
                method.
        """
        self._requested_columns = RequestedColumns(untyped_fallback, config.operation)
        self._untyped = untyped_fallback
        self._plugin = plugin
        self._value_resolver = _ValueDtypeResolver(
            DTypePromotion(UncheckedCompatibility(), plugin),
            unchecked_fallback,
            plugin,
        )
        self._config = config
        self._operation = config.operation
        self._duplicate_check = DuplicateNameCheck(config.operation)

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.unpivot()` or `.melt()` call.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            A typed frame with the `index` columns, then `variable` and
            `value`. If the hook cannot type the call, the plain
            `pl.DataFrame` or `pl.LazyFrame`. The plain frame also follows
            the error for two result columns that share a name.
        """
        plan = _UnpivotPlan.from_context(
            ctx,
            self._requested_columns,
            self._value_resolver,
            self._config,
        )
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                self._operation,
                Frame.from_proper_type(ctx.type),
            )
        index_names = [column.name for column in plan.index_columns]
        if self._duplicate_check.fails(ctx, [*index_names, plan.variable_name, plan.value_name]):
            return self._untyped.resolve(ctx, Reason.REPORTED, self._operation, plan.frame)
        columns = self._built_columns(plan, ctx.api)
        if columns is None:  # pragma: no cover
            # mypy has always loaded `pl.String` and `Column` when this
            # hook runs, so neither lookup fails.
            return self._untyped.resolve(ctx, Reason.CHECKER_GAP, self._operation, plan.frame)
        return plan.frame.rebuilt(columns, ctx.api).as_instance()

    @propagates(None)
    def _built_columns(self, plan: _UnpivotPlan, api: object) -> list[ColumnRef] | None:
        """List the columns of the result: the `index` columns, then the two new ones.

        The `variable` column has the data type `String`. The `value` column
        has the data type that the plan gives.

        Args:
            plan: What the hook read from the call.
            api: The mypy plugin interface, which builds the name types.

        Returns:
            The columns, or None if mypy cannot find `pl.String`, or cannot
            find `Column` when there is no column to copy it from.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> built = frame(fixture, [price.as_instance()])
            >>> plan = _UnpivotPlan(built, [], price, int_dtype(fixture), "variable", "value")
            >>> plugin = new_plugin(fixture)
            >>> fallback = unchecked_fallback()
            >>> hook = UnpivotHook(untyped_fallback(), plugin, fallback, UnpivotConfig())
            >>> for column in hook._built_columns(plan, api):
            ...     print(column.name, column.dtype)
            variable polars.datatypes.classes.String
            value polars.datatypes.classes.Int64
        """
        string_dtype = not_none(self._string_dtype())
        variable = self._built_column(plan.variable_name, string_dtype, plan.model, api)
        variable_column = not_none(variable)
        value = self._built_column(plan.value_name, plan.value_dtype, plan.model, api)
        value_column = not_none(value)
        return [*plan.index_columns, variable_column, value_column]

    def _built_column(
        self,
        name: str,
        dtype: MypyType,
        model: ColumnRef | None,
        api: object,
    ) -> ColumnRef | None:
        """Build one new column, from a model column if there is one, else by a lookup.

        Args:
            name: The literal name of the new column.
            dtype: The data type of the new column.
            model: A column of the receiver whose class record the new
                column reuses, or None if the receiver declares no column.
            api: The mypy plugin interface, which builds the name type.

        Returns:
            The column, or None if there is no model and mypy cannot find
            `Column`.
        """
        if model is not None:
            return ColumnRef.build(name, dtype, model, api)
        return ColumnRef.build_fresh(name, dtype, api, self._plugin)

    def _string_dtype(self) -> MypyType | None:
        """Build the type `pl.String`.

        Returns:
            An instance type of `pl.String`, or None if mypy cannot find
            the class.

        Example:
            >>> fixture = new_type_fixture()
            >>> plugin = new_plugin(fixture)
            >>> fallback = unchecked_fallback()
            >>> hook = UnpivotHook(untyped_fallback(), plugin, fallback, UnpivotConfig())
            >>> hook._string_dtype()
            polars.datatypes.classes.String
        """
        info = self._plugin.lookup_fully_qualified(self._STRING_TARGET.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # `pl.String` is part of polars, and mypy has always loaded
            # polars when a frame method runs.
            return None
        return Instance(info.node, [])
