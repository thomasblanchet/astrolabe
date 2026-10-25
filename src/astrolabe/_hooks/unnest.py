"""Type `.unnest()` on a typed frame: each named `Struct` column becomes its fields.

A `Struct` column holds a record in each row, with named fields. `.unnest()`
replaces the column with one column for each field, at the position of the
struct and in the order of its fields. polars declares that `.unnest()`
returns a plain frame, so without this hook the checker loses every
column at this call.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Final, TYPE_CHECKING

from mypy.types import get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.errors import UnknownColumnError, WrongColumnShapeError
from astrolabe._hooks.propagate import Refusal, explains, not_falsy, not_none, propagates
from astrolabe.frame import Struct as _StructMarker

if TYPE_CHECKING:
    from mypy.plugin import MethodContext

    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _UnnestPlan:
    """Hold what the hook reads from one `.unnest()` call.

    Attributes:
        frame: The receiver frame.
        requested: The names of the columns to unnest, in call order. If
            the call names no column, this is each `Struct` column of the
            receiver, in column order, as polars does.
        separator: The text between the struct name and the field name
            in each new column name, or None if a new column takes the
            bare field name.
        known_columns: The columns of the receiver, all declared.
    """

    frame: Frame
    requested: list[str]
    separator: str | None
    known_columns: list[ColumnRef]

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(cls, ctx: MethodContext) -> _UnnestPlan:
        """Read the receiver, the separator and the column names of an `.unnest()` call.

        Args:
            ctx: The context that mypy gives for `.unnest()`.

        Returns:
            The plan, or a `Refusal`. The reason is `DYNAMIC_NAME` if
            `separator=` is not a literal string or None, or if the column
            names are not literal. It is `CHECKER_GAP` if the receiver is
            not a closed typed frame, that is one whose type names all of
            its columns.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known_columns = not_none(frame.columns)
        args = CallArgs(ctx)
        not_falsy(cls._separator_is_recognized(args), reason=Reason.DYNAMIC_NAME)
        separator = cls._separator_value(args)
        requested = not_none(cls._requested_names(args, known_columns), reason=Reason.DYNAMIC_NAME)
        return cls(frame, requested, separator, known_columns)

    @staticmethod
    def _separator_is_recognized(args: CallArgs) -> bool:
        """Answer whether the hook can read the `separator=` argument.

        None is a valid value of the separator, so the value itself
        cannot go through `not_none`. This test gives a boolean for the
        chain to check first, and `_separator_value` reads the value
        after it passes.

        Args:
            args: The arguments of the call.

        Returns:
            True if `separator=` is missing, None, or a literal string.
        """
        if args.is_omitted_or_none("separator"):
            return True
        return args.literal_str("separator") is not None

    @staticmethod
    def _separator_value(args: CallArgs) -> str | None:
        """Read the separator, after `_separator_is_recognized` passes.

        Args:
            args: The arguments of the call.

        Returns:
            The literal separator, or None if the argument is missing or
            None. None is also the default of polars.
        """
        if args.is_omitted_or_none("separator"):
            return None
        return args.literal_str("separator")

    @classmethod
    @propagates(None)
    def _requested_names(cls, args: CallArgs, known_columns: list[ColumnRef]) -> list[str] | None:
        """Read the names of the columns to unnest, or find every `Struct` column.

        If the call names no column, polars unnests each `Struct` column.

        Args:
            args: The arguments of the call.
            known_columns: The columns of the receiver.

        Returns:
            The names in call order, or None if `columns` or
            `*more_columns` is not a literal name or a literal list of
            names.
        """
        more_names = not_none(args.vararg_str_list("more_columns"))
        if args.is_omitted_or_none("columns") and not more_names:
            return cls._struct_column_names(known_columns)
        names = not_none(cls._columns_list(args))
        return names + more_names

    @staticmethod
    def _columns_list(args: CallArgs) -> list[str] | None:
        """Read `columns` as a literal list of names, with an empty list if it is missing.

        An explicit `columns=[]` names no column through this parameter.
        A missing or None `columns` is different: if `*more_columns` is
        also empty, polars unnests every `Struct` column. The caller
        decides that case before this method runs.

        Args:
            args: The arguments of the call.

        Returns:
            The literal names of `columns`, an empty list if `columns` is
            missing or None, or None if it is not literal.
        """
        if args.is_omitted_or_none("columns"):
            return []
        return args.literal_str_list("columns")

    @staticmethod
    def _struct_column_names(known_columns: list[ColumnRef]) -> list[str]:
        """List the name of each `Struct` column, in the column order of the receiver.

        Args:
            known_columns: The columns of the receiver.

        Returns:
            The names in order. A column with no literal name is left
            out, because no name in a call can match it either.
        """
        return [
            column.name
            for column in known_columns
            if column.name is not None and UnnestHook.is_struct_dtype(column.dtype)
        ]


class _UnnestSplicer:
    """Find the new columns that replace each requested `Struct` column of one call.

    One instance serves one call. It holds what stays the same for every
    requested name: the mypy context, the columns of the receiver and the
    separator. Each step then reads that state from `self`, and takes only
    the name as an argument.
    """

    def __init__(
        self,
        ctx: MethodContext,
        known_columns: list[ColumnRef],
        separator: str | None,
    ) -> None:
        """Keep the state of one call.

        Args:
            ctx: The context that mypy gives for `.unnest()`.
            known_columns: The columns of the receiver.
            separator: The text between the struct name and the field
                name, or None if a new column takes the bare field name.
        """
        self._ctx = ctx
        self._known_columns = known_columns
        self._separator = separator

    def index_of(self, name: str) -> int:
        """Give the position of one requested column among the columns of the receiver.

        Args:
            name: The name of a requested column that the receiver has.

        Returns:
            The position of the column.
        """
        return self._index_by_name[name]

    @propagates(None)
    def fields_for(self, name: str) -> list[ColumnRef] | None:
        """Give the new columns that replace one requested column.

        Args:
            name: The name of the requested column.

        Returns:
            The fields of the column, renamed with the separator if the
            call gives one. None if the column does not exist or is not a
            `Struct`, after the error is reported.
        """
        matched = not_none(self._matched_column(name))
        fields = not_none(self._struct_fields(name, matched))
        return self._renamed_fields(name, fields)

    def _matched_column(self, name: str) -> ColumnRef | None:
        """Find the requested column among the columns of the receiver.

        If the receiver has no such column, the method reports an
        `UnknownColumnError`.

        Args:
            name: The name of the requested column.

        Returns:
            The column, or None if the receiver has no such column.
        """
        matched = self._by_name.get(name)
        if matched is None:
            UnknownColumnError(name).fail(self._ctx)
        return matched

    def _struct_fields(self, name: str, matched: ColumnRef) -> list[ColumnRef] | None:
        """Read the fields of the requested column, or report that it has none.

        If the column is not a `Struct` with declared fields, the method
        reports a `WrongColumnShapeError`. This includes a column of data
        type `Unchecked`, whose data type is not known. Its field names
        are not known either, and they are the names of the result
        columns.

        Args:
            name: The name of the requested column, for the error message.
            matched: The requested column.

        Returns:
            The fields, or None if the column is not a `Struct` with
            declared fields.
        """
        fields: list[ColumnRef] | None = None
        proper = get_proper_type(matched.dtype)
        # The plugin builds each `Struct[...]` type with exactly one
        # argument, the tuple of fields. So a `Struct` type always has an
        # argument here, and the test of `proper.args` only guards the index.
        if UnnestHook.STRUCT_TARGET.matches(proper) and proper.args:
            fields = ColumnRef.from_argument(proper, 0)
        if fields is None:
            error = WrongColumnShapeError.on_column(
                name,
                expected="a Struct with declared fields",
                operation="unnest",
            )
            error.fail(self._ctx)
        return fields

    @propagates(None)
    def _renamed_fields(self, struct_name: str, fields: list[ColumnRef]) -> list[ColumnRef] | None:
        """Put the struct name and the separator before each field name, if there is a separator.

        Args:
            struct_name: The name of the struct column.
            fields: The fields of the struct.

        Returns:
            `fields` unchanged if the separator is None. Else each field
            named `f"{struct_name}{separator}{field_name}"`. None if a
            field has no literal name.
        """
        if self._separator is None:
            return fields
        result: list[ColumnRef] = []
        for field in fields:
            name = not_none(field.name)
            result.append(
                field.with_name(f"{struct_name}{self._separator}{name}", self._ctx.api),
            )
        return result

    @cached_property
    def _index_by_name(self) -> dict[str, int]:
        return {
            column.name: i
            for i, column in enumerate(self._known_columns)
            if column.name is not None
        }

    @cached_property
    def _by_name(self) -> dict[str, ColumnRef]:
        return {column.name: column for column in self._known_columns if column.name is not None}


class UnnestHook:
    """Give `.unnest()` on a typed frame a result where each struct becomes its fields.

    The receiver must be a closed frame, whose type names all of its
    columns. The hook replaces each requested `Struct` column with its
    fields, at the same position. With a separator, a field `b` of a
    struct `s` becomes the column `s_b` for the separator `"_"`.

    polars fails if a requested column does not exist, is not a `Struct`,
    or gives a field name that another column of the result also has. The
    hook reports each of these. It checks the names of all result columns
    together, because a field can clash with a column that was not a
    struct.

    It is a mypy method hook in the plugin layer. `_UnnestSplicer` does
    the work for one call. If the hook cannot read the call, it falls back
    through `UntypedFallback` to the plain frame that polars declares.
    """

    STRUCT_TARGET: Final[Target] = Target.of(_StructMarker)

    @staticmethod
    def is_struct_dtype(dtype: MypyType) -> bool:
        """Answer whether a data type is a `Struct`.

        A call that names no column unnests each `Struct` column, and
        this test finds them.

        Args:
            dtype: The data type of a column.

        Returns:
            True if `dtype` is a `Struct[...]` instance type.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> UnnestHook.is_struct_dtype(struct_instance(fixture, price))
            True
            >>> UnnestHook.is_struct_dtype(int_dtype(fixture))
            False
        """
        return UnnestHook.STRUCT_TARGET.matches(get_proper_type(dtype))

    def __init__(self, untyped_fallback: UntypedFallback) -> None:
        """Keep the fallback, and build the duplicate check.

        Args:
            untyped_fallback: Gives the plain frame for a call that the
                hook cannot type, and reports it unless the
                `allow_unproven` setting lets it pass.
        """
        self._untyped = untyped_fallback
        self._duplicate_check = DuplicateNameCheck("unnest")

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.unnest()` call.

        A field of one struct can clash with another column of the frame,
        and not only with a field of another struct. So the duplicate check
        reads all result columns together, and not each struct alone.

        Args:
            ctx: The context that mypy gives for `.unnest()`.

        Returns:
            A typed frame with each requested column replaced by its
            fields. If the hook cannot type the call, the plain
            `pl.DataFrame` or `pl.LazyFrame`. The plain frame also follows
            an error: a requested column that is wrong, or two result
            columns that share a name.
        """
        plan = _UnnestPlan.from_context(ctx)
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                "unnest",
                Frame.from_proper_type(ctx.type),
            )
        result_columns = self._unnested(ctx, plan)
        if result_columns is None:
            return self._untyped.resolve(ctx, Reason.REPORTED, "unnest", plan.frame)
        names = [column.name for column in result_columns]
        if self._duplicate_check.fails(ctx, names):
            return self._untyped.resolve(ctx, Reason.REPORTED, "unnest", plan.frame)
        return plan.frame.rebuilt(result_columns, ctx.api).as_instance()

    @propagates(None)
    def _unnested(self, ctx: MethodContext, plan: _UnnestPlan) -> list[ColumnRef] | None:
        """Replace each requested column with its fields in the list of columns.

        Args:
            ctx: The context that mypy gives for `.unnest()`.
            plan: What the hook read from the call.

        Returns:
            The columns of the receiver, with each requested column
            replaced by its fields at the position of that column. None if
            an error is reported.
        """
        splicer = _UnnestSplicer(ctx, plan.known_columns, plan.separator)
        replacement_by_position: dict[int, list[ColumnRef]] = {}
        for name in plan.requested:
            fields = not_none(splicer.fields_for(name))
            replacement_by_position[splicer.index_of(name)] = fields
        result_columns: list[ColumnRef] = []
        for i, column in enumerate(plan.known_columns):
            replacement = replacement_by_position.get(i)
            if replacement is None:
                result_columns.append(column)
                continue
            result_columns.extend(replacement)
        return result_columns
