"""Check that the frame given to `.vstack()`, `.extend()` or `.merge_sorted()` has the same schema.

These three methods append or merge the rows of a second frame. polars
accepts the second frame only if it has the same column names, in the
same order, with the same data types. A different data type raises
`SchemaError`, and a different set or order of columns raises
`ShapeError`. The typed frame declares each method to return the type of
its receiver, which is right for a call that succeeds. This module checks
that the call can succeed. Before polars 1.41.0, `merge_sorted` also panics
over a column that holds a `List` or an `Array` (`MergedKinds`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.domain.dtype_promotion import DTypePromotion
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.schema_equality import SchemaEquality
from astrolabe._hooks.encoding.schema_renderings import SchemaRenderings
from astrolabe._hooks.errors import SchemaMismatchError, UnsupportedDtypeOperationError
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe._rules.dtypes import DType, Rendering, RenderingError
from astrolabe._rules.releases import polars_releases

if TYPE_CHECKING:
    from astrolabe._hooks.domain.frame import PluginLookup
    from mypy.plugin import MethodContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.column import ColumnRef


@dataclass(frozen=True)
class _VstackExtendPlan:
    """Hold the columns of the two frames of one call, once the hook can read both.

    Attributes:
        left_columns: The declared columns of the receiver.
        right_columns: The declared columns of the frame in the argument.
    """

    left_columns: list[ColumnRef]
    right_columns: list[ColumnRef]

    @classmethod
    @propagates(None)
    def from_context(cls, ctx: MethodContext) -> _VstackExtendPlan | None:
        """Read the columns of the receiver and of the argument.

        The typed frame declares these three methods with the parameters of
        polars, so mypy files the frame under `other`, whether the caller
        passes it by position or by name.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The plan, or None if the receiver or the argument is not a
            typed frame whose columns the hook can read.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        left_columns = not_none(frame.columns)
        other_type = not_none(CallArgs(ctx).node_type("other"))
        other = not_none(Frame.from_proper_type(other_type))
        right_columns = not_none(other.columns)
        return cls(left_columns, right_columns)


class MergedKinds:
    """Report a column that `merge_sorted` cannot merge in the installed release of polars.

    Before polars 1.41.0, `merge_sorted` panics over a column that holds a
    `List` or an `Array`, at the top or in a field of a struct, or ends the
    process. It decodes the merged rows of such a column with the field of the
    encoded rows, which has another type. From that release, it merges every
    column.

    Example:
        >>> MergedKinds()._holds_a_list(Rendering().parse("Struct({'a': List(Int8)})"))
        True
    """

    REASON: ClassVar[str] = (
        "merge_sorted does not merge a List or an Array before polars 1.41.0,"
        " and polars panics or ends the process"
    )
    LISTS: ClassVar[frozenset[str]] = frozenset({"List", "Array"})

    @polars_releases
    def check(self, ctx: MethodContext, renderings: dict[str, str]) -> None:
        """Report no column: the installed release merges every column.

        Args:
            ctx: The context that mypy gives for the call.
            renderings: The rendering of the data type of each column of the receiver.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> ctx = new_fail_context()
            >>> with PolarsRelease.pretend("1.41.0"):
            ...     MergedKinds().check(ctx, {"l": "List(Int8)"})
            >>> ctx.api.messages
            []
        """
        del ctx, renderings

    @check.before("1.41.0")
    def _refuse_lists(self, ctx: MethodContext, renderings: dict[str, str]) -> None:
        """Report the first column that holds a `List` or an `Array`, over which polars panics.

        Args:
            ctx: The context that mypy gives for the call.
            renderings: The rendering of the data type of each column of the receiver.

        Example:
            >>> ctx = new_fail_context()
            >>> MergedKinds()._refuse_lists(ctx, {"k": "Int64", "l": "List(Int8)"})
            >>> ctx.api.messages[0][:24]
            '"l" has no result type a'
            >>> unknown = new_fail_context()
            >>> MergedKinds()._refuse_lists(unknown, {"u": "Unchecked"})
            >>> unknown.api.messages
            []
        """
        for name, rendering in renderings.items():
            if self._holds_a_list(self._dtype(rendering)):
                UnsupportedDtypeOperationError.at_frame(name, self.REASON).fail(ctx)
                return

    def _dtype(self, rendering: str) -> DType | None:
        """Read the data type of a column, or `None` where the rules do not know it.

        Args:
            rendering: The rendering of the data type.

        Returns:
            The data type, or `None` for a type such as `Unchecked`.

        Example:
            >>> MergedKinds()._dtype("Unchecked") is None
            True
        """
        try:
            return Rendering().parse(rendering)
        except RenderingError:
            return None

    def _holds_a_list(self, dtype: DType | None) -> bool:
        """Say whether a data type is a `List`, an `Array`, or a struct with a field that holds one.

        Args:
            dtype: The data type, or `None` where it is not known.

        Returns:
            Whether it holds one.

        Example:
            >>> MergedKinds()._holds_a_list(Rendering().parse("Struct({'a': Int8})"))
            False
            >>> MergedKinds()._holds_a_list(None)
            False
        """
        if dtype is None:
            return False
        if dtype.name in self.LISTS:
            return True
        if dtype.name != "Struct":
            return False
        return any(self._holds_a_list(field) for _, field in dtype.fields())


class VstackExtendHook:
    """Report a `SchemaMismatchError` if the two frames of a stacking call differ.

    Without this hook, the checker accepts `a.vstack(b)` for any frame `b`,
    and the call fails when the program runs. The hook compares the
    declared columns of both frames, position by position, with
    `SchemaEquality`: the same count, and at each position the same name
    and the same data type. The data type `Unchecked` (a column whose data
    type is not known) matches any data type. The hook never changes the
    result type. If it cannot read either frame, it checks nothing.

    It is a mypy method hook in the plugin layer, with one instance for
    each of the three methods.
    """

    def __init__(self, operation: str, plugin: PluginLookup, *, merges: bool = False) -> None:
        """Keep the name of the method, and build the schema comparison.

        Args:
            operation: `".vstack()"`, `".extend()"` or `".merge_sorted()"`,
                for the error message.
            plugin: The mypy plugin, which the type promotion rule of
                `SchemaEquality` needs. This hook compares with no
                promotion, so the rule never runs here.
            merges: Whether the method is `merge_sorted`, whose columns
                `MergedKinds` checks.
        """
        self._operation = operation
        self._merges = merges
        unchecked = UncheckedCompatibility()
        self._schema_equality = SchemaEquality(unchecked, DTypePromotion(unchecked, plugin))

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Check the call, and give back the return type that the typed frame declares.

        If the two frames do not match, the hook reports
        `SchemaMismatchError`.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The declared return type, always unchanged.
        """
        plan = _VstackExtendPlan.from_context(ctx)
        if plan is None:
            return ctx.default_return_type
        left = self._schema_equality.sorted_by_name(plan.left_columns)
        right = self._schema_equality.sorted_by_name(plan.right_columns)
        if not self._schema_equality.columns_match(left, right, relaxed=False):
            SchemaMismatchError.stacked(self._operation).fail(ctx)
        elif self._merges:
            MergedKinds().check(ctx, SchemaRenderings().of(plan.left_columns))
        return ctx.default_return_type
