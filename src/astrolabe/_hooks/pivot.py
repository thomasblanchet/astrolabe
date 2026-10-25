"""Type `.pivot()` on a typed frame: spread the values of one column into new columns.

`.pivot(on="kind", index="day", values="price", on_columns=["a", "b"])`
gives one row for each `day`, and one new column for each value of `kind`.
The names of the new columns come from the data, so the checker can know
them only if the call lists them. The hook therefore types a call only if
`on_columns` is a literal list. With `on_columns`, polars makes exactly
those columns, whatever the data holds. Without it, polars reads the names
from the data, and the hook falls back to the plain frame.

polars turns each value of `on_columns` into a column name. A string stays
as it is. An integer or a float becomes its Python `str()`. A boolean
becomes the lowercase `"true"` or `"false"`, which is not the Python
`str()`.

`on` must name exactly one column, as a string or a list of one string.
An `on` with more columns stays untyped. Such a call needs `on_columns` as
a frame of value tuples, and polars rejects a list literal there, so the
hook reports that shape as an error.

The data type of each new column follows the same rules as a reducer in
`.agg()`, which `ReducerDtype` holds. With no `aggregate_function`, or with
`"item"`, a new column keeps the data type of its `values` column. If the
hook cannot compute the data type of one `values` column, only the new
columns from it get `Unchecked`, the data type of a column whose name is
known and whose data type is not. The other new columns keep their types.
If no `values` column is left, polars always raises, and the hook reports
an `InvalidCallShapeError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, TYPE_CHECKING

import polars as pl
from mypy.nodes import ListExpr, TupleExpr
from mypy.types import Instance, get_proper_type

from astrolabe._hooks.calls.bare_literal import BareLiteralReader
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.dtype_families import DtypeFamilies
from astrolabe._hooks.domain.reducer_promotion import ReducerDtype
from astrolabe._hooks.domain.requested_columns import RequestedColumns
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.errors import InvalidCallShapeError, WrongColumnShapeError
from astrolabe._hooks.propagate import (
    Refusal,
    explains,
    not_falsy,
    not_none,
    not_refused,
)

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import MethodContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.frame import KnownColumns, PluginLookup
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


class _OnColumnResolver:
    """Read the one column that `on` names in a `.pivot()` call.

    `on` must be a literal string, or a literal list of one string. A
    selector from `polars.selectors` is not read, so the call falls back.

    Example:
        >>> resolver = _OnColumnResolver()
        >>> resolver.resolved(call_args(["on"], [[str_literal_node("cat")]]))
        'cat'
        >>> two = ListExpr([str_literal_node("cat1"), str_literal_node("cat2")])
        >>> resolver.resolved(call_args(["on"], [[two]])) is None
        True
    """

    def resolved(self, args: CallArgs) -> str | None:
        """Read the one column name in `on`.

        Args:
            args: The arguments of the call.

        Returns:
            The name, or None if `on` is not a literal string or a literal
            list of exactly one string.
        """
        literal_names = args.literal_str_list("on")
        if literal_names is None or len(literal_names) != 1:
            return None
        return literal_names[0]

    def unread_reason(self, args: CallArgs) -> Reason:
        """Give the reason why `resolved` found no single name in `on`.

        Args:
            args: The arguments of the call.

        Returns:
            `DYNAMIC_NAME` if `on` is not literal names. `CHECKER_GAP` if
            it names more than one column, which the hook does not model.

        Example:
            >>> passed = call_args(["on"], [[str_literal_node("a")]])
            >>> _OnColumnResolver().unread_reason(passed).name
            'CHECKER_GAP'
        """
        if args.literal_str_list("on") is None:
            return Reason.DYNAMIC_NAME
        return Reason.CHECKER_GAP

    def name_count(self, args: CallArgs) -> int | None:
        """Count the columns that `on` names, with no limit of one.

        Args:
            args: The arguments of the call.

        Returns:
            The count, or None if `on` is not a literal string or a
            literal list of strings.
        """
        literal_names = args.literal_str_list("on")
        if literal_names is None:
            return None
        return len(literal_names)


class _OnColumnsReader:
    """Read the `on_columns` argument of `.pivot()` as the names of the new columns.

    Each value becomes a column name in the way polars makes it: a string
    stays as it is, a boolean becomes `"true"` or `"false"` in lowercase,
    and an integer or a float becomes its Python `str()`. A value can be a
    literal or a variable declared `Final`.

    Example:
        >>> nodes = ListExpr([int_literal_node(1), int_literal_node(2)])
        >>> _OnColumnsReader().resolved(call_args(["on_columns"], [[nodes]]), api=None)
        ['1', '2']
    """

    EXPRESSION: ClassVar[str] = Target.of(pl.Expr).fullname

    def __init__(self) -> None:
        """Build the reader for literal values."""
        self._bare_literal = BareLiteralReader()

    def holds_no_expression(self, ctx: MethodContext, args: CallArgs) -> bool:
        """Report an expression in a list or tuple literal of `on_columns`.

        Polars builds a series of the values of `on_columns`. An expression
        in a list makes a series of the type `Object`, which polars cannot
        cast, and an expression beside a value makes no series. So the call
        raises at every frame. The signature of `pivot` already refuses an
        expression in place of the list.

        Args:
            ctx: The context that mypy gives for `.pivot()`.
            args: The arguments of the call.

        Returns:
            True if `on_columns` holds no expression. Otherwise the method
            reports an error and gives False.
        """
        items: list[Expression] = []
        for node in args.node_list("on_columns"):
            items.extend(node.items if isinstance(node, ListExpr | TupleExpr) else [])
        if not any(self._is_expression(ctx.api.get_expression_type(item)) for item in items):
            return True
        error = InvalidCallShapeError(
            "pivot() with an expression in on_columns= always raises: polars cannot turn "
            'an expression into a column name. Pass the values, for example on_columns=["x"]',
        )
        error.fail(ctx)
        return False

    def _is_expression(self, typ: MypyType) -> bool:
        """Say whether a type is an expression of polars.

        Args:
            typ: The type of a value.

        Returns:
            Whether it is `pl.Expr` or a class that inherits it.

        Example:
            >>> _OnColumnsReader()._is_expression(new_type_fixture().anyt)
            False
        """
        proper = get_proper_type(typ)
        return isinstance(proper, Instance) and proper.type.has_base(self.EXPRESSION)

    def resolved(self, args: CallArgs, api: object) -> list[str] | None:
        """Read the names of the new columns from `on_columns`.

        Each item of the list or tuple must be a literal integer, float,
        string or boolean, or a variable declared `Final` with such a
        value. A single scalar is not read. polars raises for an
        `on_columns` that is not a sequence, so the call can never run.

        Args:
            args: The arguments of the call.
            api: The mypy plugin interface, which reads the value of a
                variable declared `Final`.

        Returns:
            The names in order, or None if the call does not pass
            `on_columns`, or its value is not a list or tuple literal of
            scalar values.
        """
        nodes = args.node_list("on_columns")
        if len(nodes) != 1:
            return None
        node = nodes[0]
        if not isinstance(node, ListExpr | TupleExpr):
            return None
        names: list[str] = []
        for item in node.items:
            value = self._bare_literal.scalar_via_api(item, api)
            if value is None:
                return None
            names.append(self._stringified(value))
        return names

    def unread_reason(self, args: CallArgs) -> Reason:
        """Give the reason why `resolved` found no names in `on_columns`.

        Args:
            args: The arguments of the call.

        Returns:
            `DYNAMIC_NAME` if the call passes `on_columns` but it is not a
            literal. `CHECKER_GAP` if the call omits it, because polars
            then takes the names from the data.

        Example:
            >>> _OnColumnsReader().unread_reason(call_args(["on_columns"], [[]])).name
            'CHECKER_GAP'
        """
        if args.is_omitted_or_none("on_columns"):
            return Reason.CHECKER_GAP
        return Reason.DYNAMIC_NAME

    def _stringified(self, value: int | float | str | bool) -> str:
        """Turn one `on_columns` value into a column name, as polars does.

        Args:
            value: The value.

        Returns:
            `"true"` or `"false"` for a boolean. The value itself for a
            string. Else the Python `str()` of the value.

        Example:
            >>> _OnColumnsReader()._stringified(5)
            '5'
            >>> _OnColumnsReader()._stringified(True)
            'true'
            >>> _OnColumnsReader()._stringified(False)
            'false'
            >>> _OnColumnsReader()._stringified("a")
            'a'
        """
        if isinstance(value, bool):
            # polars writes a boolean in lowercase, which is not the Python `str()`.
            if value:
                return "true"
            return "false"
        if isinstance(value, str):
            return value
        return str(value)


@dataclass(frozen=True)
class _PivotedColumn:
    """Hold the name and the data type of one new column that `.pivot()` makes.

    Attributes:
        name: The literal name of the column.
        dtype: The data type of the column.
    """

    name: str
    dtype: MypyType


class _PivotedColumnBuilder:
    """Build the new columns of one `.pivot()` call, with their names and data types.

    polars makes the new columns in this order: for each `values` column,
    one new column for each `on_columns` value. The builder holds the three
    arguments that shape each new column (`aggregate_function`,
    `separator` and `column_naming`) and the tools that compute a data
    type. So the code that reads the call does not pass them through a
    function with many arguments.
    """

    _IDENTITY_REDUCERS: Final[frozenset[str | None]] = frozenset({None, "item"})

    def __init__(
        self,
        reducer: str | None,
        separator: str,
        column_naming: str,
        reducer_dtype: ReducerDtype,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Keep the arguments of the call and the tools for the data types.

        Args:
            reducer: The name of the `aggregate_function`, or None. None
                and `"item"` both keep the data type of the `values`
                column.
            separator: The text between a `values` name and an
                `on_columns` value, if the new names combine the two.
            column_naming: `"auto"` or `"combine"`.
            reducer_dtype: Gives the data type that a reducer makes from
                a data type.
            unchecked_fallback: Gives `Unchecked` for a `values` column
                whose new data type the builder cannot compute, and
                reports it unless the setting lets it pass.
        """
        self._reducer = reducer
        self._separator = separator
        self._column_naming = column_naming
        self._reducer_dtype = reducer_dtype
        self._unchecked_fallback = unchecked_fallback

    def built(
        self,
        ctx: MethodContext,
        values_columns: list[ColumnRef],
        on_columns: list[str],
    ) -> list[_PivotedColumn]:
        """Build each new column.

        With `column_naming="auto"` and one `values` column, a new column
        takes the bare `on_columns` value as its name. With more than one
        `values` column, or with `"combine"`, the name also holds the
        `values` name and the separator, as in polars.

        Args:
            ctx: The context that mypy gives for `.pivot()`.
            values_columns: The `values` columns, in call order.
            on_columns: The names from `on_columns`, in call order.

        Returns:
            The new columns: for each `values` column, one new column for
            each `on_columns` value. The new columns of one `values` column
            share one data type. If the data type of one `values` column
            does not suit the reducer, or cannot be computed, only its new
            columns get `Unchecked`.
        """
        combine = self._column_naming == "combine" or len(values_columns) != 1
        result: list[_PivotedColumn] = []
        for values_column in values_columns:
            dtype = self._resolved_dtype(ctx, values_column)
            for on_value in on_columns:
                name = self._pivoted_name(values_column.name, on_value, combine)
                result.append(_PivotedColumn(name, dtype))
        return result

    def _resolved_dtype(self, ctx: MethodContext, values_column: ColumnRef) -> MypyType:
        """Compute the data type of the new columns from one `values` column.

        Some reducers accept only some data types. For example, `"sum"`
        needs a number or a boolean. For a data type that the reducer
        does not accept, the method reports a `WrongColumnShapeError` and
        gives `Unchecked`. For a pair that `ReducerDtype` does not model,
        it gives `Unchecked` with the reason `CHECKER_GAP`.

        Args:
            ctx: The context that mypy gives for `.pivot()`.
            values_column: The `values` column.

        Returns:
            The data type that the reducer gives, or `Unchecked`.
        """
        name = checked_not_none(values_column.name)
        if not self._family_matches(values_column.dtype):
            error = WrongColumnShapeError.on_column(
                name,
                expected="numeric or Boolean",
                operation=f"pivot(aggregate_function={self._reducer!r})",
            )
            error.fail(ctx)
            return self._unchecked_or_source(ctx, Reason.REPORTED, name, values_column.dtype)
        dtype = self._pivoted_dtype(values_column.dtype)
        if dtype is not None:
            return dtype
        return self._unchecked_or_source(ctx, Reason.CHECKER_GAP, name, values_column.dtype)

    def _unchecked_or_source(
        self,
        ctx: MethodContext,
        reason: Reason,
        name: str,
        source_dtype: MypyType,
    ) -> MypyType:
        """Give `Unchecked` for the new columns of one `values` column.

        The fallback reports the `Unchecked` column unless the reason is
        `REPORTED` or the `allow_unproven` setting lets it pass.

        Args:
            ctx: The context that mypy gives for `.pivot()`.
            reason: Why the data type is not known: an error that is
                already reported, or a case that the rules do not model.
            name: The name of the `values` column.
            source_dtype: The data type of the `values` column.

        Returns:
            `Unchecked`, or `source_dtype` if mypy cannot find
            `Unchecked`.
        """
        substituted = self._unchecked_fallback.substituted(ctx, reason, "pivot", name)
        return substituted if substituted is not None else source_dtype

    def _pivoted_dtype(self, source_dtype: MypyType) -> MypyType | None:
        """Give the data type that the reducer makes from one `values` column.

        Args:
            source_dtype: The data type of the `values` column.

        Returns:
            `source_dtype` unchanged if the reducer is None or `"item"`.
            Else the data type that the reducer gives, or None if
            `ReducerDtype` does not model it.

        Example:
            >>> fixture = new_type_fixture()
            >>> reducer_dtype_ = reducer_dtype(new_plugin(fixture))
            >>> fallback = unchecked_fallback()
            >>> identity = _PivotedColumnBuilder(None, "_", "auto", reducer_dtype_, fallback)
            >>> identity._pivoted_dtype(int_dtype(fixture))
            polars.datatypes.classes.Int64
            >>> item = _PivotedColumnBuilder("item", "_", "auto", reducer_dtype_, fallback)
            >>> item._pivoted_dtype(int_dtype(fixture))
            polars.datatypes.classes.Int64
            >>> builder = _PivotedColumnBuilder("len", "_", "auto", reducer_dtype_, fallback)
            >>> builder._pivoted_dtype(int_dtype(fixture))
            polars.datatypes.classes.UInt32
        """
        if self._reducer in self._IDENTITY_REDUCERS:
            return source_dtype
        return self._reducer_dtype.resolve(checked_not_none(self._reducer), source_dtype)

    def _family_matches(self, source_dtype: MypyType) -> bool:
        """Answer whether the reducer accepts the data type of one `values` column.

        Args:
            source_dtype: The data type of the `values` column.

        Returns:
            True if there is no reducer. Else the answer of
            `ReducerDtype.source_family_matches`.
        """
        if self._reducer is None:
            return True
        return self._reducer_dtype.source_family_matches(self._reducer, source_dtype)

    def _pivoted_name(self, values_name: str | None, on_value: str, combine: bool) -> str:
        """Build the name of one new column.

        Args:
            values_name: The name of the `values` column.
            on_value: The `on_columns` value of this column.
            combine: Whether the name starts with the `values` name and
                the separator.

        Returns:
            `on_value` if not `combine`. Else
            `f"{values_name}{separator}{on_value}"`.

        Example:
            >>> fixture = new_type_fixture()
            >>> reducer_dtype_ = reducer_dtype(new_plugin(fixture))
            >>> fallback = unchecked_fallback()
            >>> builder = _PivotedColumnBuilder("sum", "_", "auto", reducer_dtype_, fallback)
            >>> builder._pivoted_name("v1", "a", combine=False)
            'a'
            >>> builder._pivoted_name("v1", "a", combine=True)
            'v1_a'
        """
        if not combine:
            return on_value
        return f"{values_name}{self._separator}{on_value}"


@dataclass(frozen=True)
class _PivotPlan:
    """Hold what the hook reads from one `.pivot()` call, and the checks on the call.

    The static methods are the checks and readers that the plan builder
    runs on the arguments before it builds a plan.

    Attributes:
        frame: The receiver frame.
        index_columns: The `index` columns, which the result keeps as
            they are, in call order.
        model: A column of the receiver whose class record of `Column`
            each new column reuses.
        pivoted: The new columns, in the order that polars makes them:
            for each `values` column, one for each `on_columns` value.
        rest: The remainder that the result keeps. A call with no
            `index` uses every other column as the index, the undeclared
            ones included, so the remainder of the receiver survives.
            None for a call that names its `index`.
    """

    frame: Frame
    index_columns: list[ColumnRef]
    model: ColumnRef
    pivoted: list[_PivotedColumn]
    rest: MypyType | None = None

    @staticmethod
    def multi_on_needs_unreadable_on_columns(
        ctx: MethodContext,
        args: CallArgs,
        on_resolver: _OnColumnResolver,
    ) -> bool:
        """Report a call whose `on` names more columns than one and whose `on_columns` is a list.

        If `on` names more than one column, polars needs `on_columns` as
        a frame whose rows match the `on` columns. A list literal is never
        that shape, so polars always raises, whether the items are strings
        or tuples.

        Args:
            ctx: The context that mypy gives for `.pivot()`.
            args: The arguments of the call.
            on_resolver: Counts the columns that `on` names.

        Returns:
            True if `on` names at most one column, or `on_columns` is not
            a list literal. Else False, after the error is reported.
        """
        on_count = on_resolver.name_count(args)
        if on_count is None or on_count <= 1:
            return True
        if not _PivotPlan.on_columns_is_list_literal(args):
            return True
        error = InvalidCallShapeError(
            "pivot() with on= naming more than one column and a list literal "
            "on_columns= always raises",
        )
        error.fail(ctx)
        return False

    @staticmethod
    def on_columns_is_list_literal(args: CallArgs) -> bool:
        """Answer whether `on_columns` is a list literal.

        Args:
            args: The arguments of the call.

        Returns:
            True if the call passes `on_columns` once, as a `[...]`
            literal, whatever its items are.
        """
        nodes = args.node_list("on_columns")
        if len(nodes) != 1:
            return False
        return isinstance(nodes[0], ListExpr)

    @staticmethod
    def names_index_or_values(ctx: MethodContext, args: CallArgs) -> bool:
        """Report `InvalidCallShapeError` if the call names neither `index` nor `values`.

        polars needs one of the two, and raises if both are missing.

        Args:
            ctx: The context that mypy gives for `.pivot()`.
            args: The arguments of the call.

        Returns:
            True if the call names at least one of them. Else False,
            after the error is reported.
        """
        if not args.is_omitted_or_none("index") or not args.is_omitted_or_none("values"):
            return True
        InvalidCallShapeError("pivot() with neither index nor values always raises").fail(ctx)
        return False

    @staticmethod
    def has_values_columns(ctx: MethodContext, values_names: list[str]) -> bool:
        """Report `InvalidCallShapeError` if no `values` column is left to spread.

        If the call omits `values`, polars uses every column that `on`
        and `index` do not name. If there is none, polars always raises.

        Args:
            ctx: The context that mypy gives for `.pivot()`.
            values_names: The names of the `values` columns.

        Returns:
            True if there is at least one name. Else False, after the
            error is reported.
        """
        if values_names:
            return True
        InvalidCallShapeError("pivot() with no values columns always raises").fail(ctx)
        return False

    @staticmethod
    def aggregate_function_is_recognized(args: CallArgs) -> bool:
        """Answer whether the hook can read the `aggregate_function` argument.

        None is a valid value, so the value itself cannot go through
        `not_none`. This test gives a boolean for the chain to check
        first, and `aggregate_function_value` reads the value after it
        passes.

        Args:
            args: The arguments of the call.

        Returns:
            True if `aggregate_function` is missing or None, or is a
            literal string that names a reducer in
            `ReducerDtype.ALL_REDUCERS`. Also True for `"len"`, which
            `ReducerDtype.resolve` reads with no source data type, and
            for `"item"`, which keeps the data type.

        Example:
            >>> args = call_args(["aggregate_function"], [[]])
            >>> _PivotPlan.aggregate_function_is_recognized(args)
            True
            >>> args = call_args(["aggregate_function"], [[str_literal_node("sum")]])
            >>> _PivotPlan.aggregate_function_is_recognized(args)
            True
            >>> args = call_args(["aggregate_function"], [[str_literal_node("len")]])
            >>> _PivotPlan.aggregate_function_is_recognized(args)
            True
            >>> args = call_args(["aggregate_function"], [[str_literal_node("item")]])
            >>> _PivotPlan.aggregate_function_is_recognized(args)
            True
            >>> args = call_args(["aggregate_function"], [[str_literal_node("bogus")]])
            >>> _PivotPlan.aggregate_function_is_recognized(args)
            False
        """
        if args.is_omitted_or_none("aggregate_function"):
            return True
        reducer = args.literal_str("aggregate_function")
        if reducer == "item":
            return True
        return reducer == "len" or reducer in ReducerDtype.ALL_REDUCERS

    @staticmethod
    def aggregate_function_value(args: CallArgs) -> str | None:
        """Read `aggregate_function`, after `aggregate_function_is_recognized` passes.

        Args:
            args: The arguments of the call.

        Returns:
            The literal name of the reducer, or None if the argument is
            missing or None, which is the default of polars.

        Example:
            >>> args = call_args(["aggregate_function"], [[]])
            >>> _PivotPlan.aggregate_function_value(args) is None
            True
            >>> args = call_args(["aggregate_function"], [[str_literal_node("sum")]])
            >>> _PivotPlan.aggregate_function_value(args)
            'sum'
        """
        if args.is_omitted_or_none("aggregate_function"):
            return None
        return args.literal_str("aggregate_function")


class _PivotPlanBuilder:
    """Read a `.pivot()` call into a `_PivotPlan`, or refuse with a reason.

    It holds the readers and the tools that every call needs, so that the
    one method that reads a call does not take them as arguments. That
    method is a flat chain of steps. Each step can fail and stop the
    chain with its own reason.
    """

    def __init__(
        self,
        requested_columns: RequestedColumns,
        reducer_dtype: ReducerDtype,
        unchecked_fallback: UncheckedFallback,
        on_resolver: _OnColumnResolver,
        on_columns_reader: _OnColumnsReader,
    ) -> None:
        """Keep the readers and the tools for each call.

        Args:
            requested_columns: Finds the columns that `on`, `index` and
                `values` name, and reports a name that the frame does not
                declare.
            reducer_dtype: Gives the data type that a reducer makes.
            unchecked_fallback: Gives `Unchecked` for a `values` column
                whose new data type the hook cannot compute.
            on_resolver: Reads the one column that `on` names.
            on_columns_reader: Reads the names of the new columns from
                `on_columns`.
        """
        self._requested_columns = requested_columns
        self._reducer_dtype = reducer_dtype
        self._unchecked_fallback = unchecked_fallback
        self._on_resolver = on_resolver
        self._on_columns_reader = on_columns_reader

    @explains(Reason.CHECKER_GAP)
    def built(self, ctx: MethodContext) -> _PivotPlan:
        """Read a `.pivot()` call into a plan.

        Args:
            ctx: The context that mypy gives for `.pivot()`.

        Returns:
            The plan, or a `Refusal`. The reason is `DYNAMIC_NAME` if
            `on`, `on_columns`, `index`, `values` or `separator=` is not
            a literal. It is `REPORTED` if a named column does not exist
            or polars always rejects the call, after the error. Else it
            is `CHECKER_GAP`: the receiver is not a typed frame, `on`
            names more than one column, `on_columns` is missing, or
            `aggregate_function` is not a reducer that the hook knows.
            An expression in `on_columns` is `REPORTED`.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        args = CallArgs(ctx)
        not_falsy(
            _PivotPlan.multi_on_needs_unreadable_on_columns(ctx, args, self._on_resolver),
            reason=Reason.REPORTED,
        )
        not_falsy(self._on_columns_reader.holds_no_expression(ctx, args), reason=Reason.REPORTED)
        on_name = not_none(
            self._on_resolver.resolved(args),
            reason=self._on_resolver.unread_reason(args),
        )
        on_column = not_refused(self._requested_columns.resolved(ctx, [on_name], known))[0]
        on_columns = not_falsy(
            not_none(
                self._on_columns_reader.resolved(args, ctx.api),
                reason=self._on_columns_reader.unread_reason(args),
            )
        )
        not_falsy(_PivotPlan.names_index_or_values(ctx, args), reason=Reason.REPORTED)
        given_index = not_none(args.literal_str_list_or_empty("index"), reason=Reason.DYNAMIC_NAME)
        excluded = {on_name, *given_index}
        values_names = not_none(
            RequestedColumns.names_or_remaining(args, "values", known, excluded),
            reason=RequestedColumns.unread_reason(args, "values"),
        )
        not_falsy(_PivotPlan.has_values_columns(ctx, values_names), reason=Reason.REPORTED)
        index_names = self._index_names(args, known, given_index, {on_name, *values_names})
        index_columns = not_refused(self._requested_columns.resolved(ctx, index_names, known))
        values_columns = not_refused(self._requested_columns.resolved(ctx, values_names, known))
        not_falsy(_PivotPlan.aggregate_function_is_recognized(args))
        builder = _PivotedColumnBuilder(
            reducer=_PivotPlan.aggregate_function_value(args),
            separator=not_none(
                args.literal_str("separator", default="_"),
                reason=Reason.DYNAMIC_NAME,
            ),
            column_naming=not_none(args.literal_str("column_naming", default="auto")),
            reducer_dtype=self._reducer_dtype,
            unchecked_fallback=self._unchecked_fallback,
        )
        pivoted = builder.built(ctx, values_columns, on_columns)
        rest = self._kept_rest(args, known)
        return _PivotPlan(frame, index_columns, on_column, pivoted, rest)

    def _index_names(
        self,
        args: CallArgs,
        known: KnownColumns,
        given: list[str],
        taken: set[str],
    ) -> list[str]:
        """Give the names of the `index` columns.

        Args:
            args: The arguments of the call.
            known: The declared columns of the receiver, and its remainder.
            given: The names that `index` lists, or none.
            taken: The names that `on` and `values` take.

        Returns:
            The names that `index` lists. With no `index`, every declared
            column that `on` and `values` do not take, as polars does.
        """
        if args.is_omitted_or_none("index"):
            return RequestedColumns.remaining_names(known, taken)
        return given

    def _kept_rest(self, args: CallArgs, known: KnownColumns) -> MypyType | None:
        """Give the remainder that the result keeps.

        Args:
            args: The arguments of the call.
            known: The declared columns of the receiver, and its remainder.

        Returns:
            The remainder of the receiver for a call with no `index`, and
            None for a call that names its `index`.
        """
        if args.is_omitted_or_none("index"):
            return known.rest
        return None


class PivotHook:
    """Give `.pivot()` on a typed frame its result: the `index` columns and the new columns.

    The hook types a call only if `on_columns` lists the new columns as a
    literal. The result holds the `index` columns, then the new columns.
    polars fails if two of them share a name, and the hook reports that.

    The receiver can be an open frame, whose type does not name all of its
    columns. If the call gives `index`, the result holds only the `index`
    columns and the new ones, so the unnamed columns of the receiver cannot
    reach it. If the call gives no `index`, polars uses every other column
    as the index, the unnamed ones included. The result then keeps the
    remainder of the receiver, and the hook types it only if that
    remainder cannot hold the name of a new column.

    It is a mypy method hook in the plugin layer. `_PivotPlanBuilder`
    reads the call. If the hook cannot type the call, it falls back
    through `UntypedFallback` to the plain frame that polars declares.
    """

    def __init__(
        self,
        untyped_fallback: UntypedFallback,
        plugin: PluginLookup,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Build the plan builder and the duplicate check.

        Args:
            untyped_fallback: Gives the plain frame for a call that the
                hook cannot type, and reports each unproven result unless
                the `allow_unproven` setting lets it pass.
            plugin: The mypy plugin, which finds the data type classes
                that the reducer rules give.
            unchecked_fallback: Gives `Unchecked` for a `values` column
                whose new data type the hook cannot compute.
        """
        self._untyped = untyped_fallback
        self._plan_builder = _PivotPlanBuilder(
            requested_columns=RequestedColumns(untyped_fallback, "pivot"),
            reducer_dtype=ReducerDtype(plugin, DtypeFamilies(plugin)),
            unchecked_fallback=unchecked_fallback,
            on_resolver=_OnColumnResolver(),
            on_columns_reader=_OnColumnsReader(),
        )
        self._duplicate_check = DuplicateNameCheck("pivot")

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.pivot()` call.

        Args:
            ctx: The context that mypy gives for `.pivot()`.

        Returns:
            A typed frame with the `index` columns, then the new columns.
            If the hook cannot type the call, the plain `pl.DataFrame` or
            `pl.LazyFrame`. The plain frame also follows the error for two
            result columns that share a name, `index` columns and new
            columns together.
        """
        plan = self._plan_builder.built(ctx)
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                "pivot",
                Frame.from_proper_type(ctx.type),
            )
        index_names = [column.name for column in plan.index_columns]
        pivoted_names = [pivoted.name for pivoted in plan.pivoted]
        if self._duplicate_check.fails(ctx, [*index_names, *pivoted_names]):
            return self._untyped.resolve(ctx, Reason.REPORTED, "pivot", plan.frame)
        if not self._rest_proven(plan):
            return self._untyped.resolve(ctx, Reason.REMAINDER_MAY_HOLD, "pivot", plan.frame)
        columns = list(plan.index_columns)
        for pivoted in plan.pivoted:
            columns.append(ColumnRef.build(pivoted.name, pivoted.dtype, plan.model, ctx.api))
        return plan.frame.rebuilt(columns, ctx.api, rest=plan.rest).as_instance()

    def _rest_proven(self, plan: _PivotPlan) -> bool:
        """Tell whether the kept remainder is proven to hold none of the new names.

        polars fails if an undeclared column that the result keeps has
        the name of a new column. The result is typed only if the
        remainder cannot hold any of those names.

        Args:
            plan: What the hook read from the call.

        Returns:
            True if the result keeps no remainder, or if the remainder
            is proven to hold none of the new names.
        """
        knowledge = RemainderKnowledge()
        return all(knowledge.proves_absent(plan.rest, pivoted.name) for pivoted in plan.pivoted)
