"""Type `df.group_by(...)` as a group-by that remembers its keys and the columns of the frame.

A `group_by` call does not give a frame. The call after it does, for example `.agg(...)` or
`.sum()`. That later call needs the key columns and the columns of the frame, but its own
receiver is the group-by. So this hook gives the group-by a type, `TypedGroupBy` or
`TypedLazyGroupBy`, whose type arguments hold the keys, the columns and the remainder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import Expression, TypeInfo
from mypy.types import Instance, NoneType
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.bare_literal import BareLiteralReader
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.frame import Frame, KnownColumns
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.errors import UnknownColumnError
from astrolabe._hooks.propagate import Refusal, explains, not_none, not_refused
from astrolabe.frame import TypedDataFrame, TypedGroupBy, TypedLazyFrame, TypedLazyGroupBy

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin

    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _GroupByPlan:
    """What one `group_by` call needs to build its typed group-by.

    Attributes:
        key_columns: The key columns, in call order. A key given by keyword has the keyword
            as its name.
        known_columns: The declared columns of the frame.
        rest: The remainder of the frame, or None for a closed frame.
        result_info: The class of the result, `TypedGroupBy` or `TypedLazyGroupBy`.
    """

    key_columns: list[ColumnRef]
    known_columns: list[ColumnRef]
    rest: MypyType | None
    result_info: TypeInfo

    RESULT_TARGET_BY_RECEIVER: ClassVar[dict[Target, Target]] = {
        Target.of(TypedDataFrame): Target.of(TypedGroupBy),
        Target.of(TypedLazyFrame): Target.of(TypedLazyGroupBy),
    }

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(cls, ctx: MethodContext, plugin: Plugin) -> _GroupByPlan:
        """Read the keys and the columns of one `group_by` call.

        The key columns are the positional keys in `by`, in call order, then the keys that the
        caller passes by keyword. A keyword key takes its keyword as its name: in
        `group_by(year="date_year")`, the key column in the result is named `year`. A
        `group_by()` call with no key is valid. The next call then reduces the whole frame to
        one row, so an empty list of keys is a real answer.

        Args:
            ctx: The mypy context of the `group_by` call.
            plugin: The mypy plugin, which finds the class of the result.

        Returns:
            The plan, or a refusal: the reason of the first key that the hook cannot match (see
            `_KeyReader`), or `CHECKER_GAP` if the receiver is not a typed frame or a keyword
            value has no keyword.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        keys = _KeyReader(ctx, known)
        args = CallArgs(ctx)
        key_columns = [not_refused(keys.matched(node)) for node in args.vararg_node_list("by")]
        for keyword, node in not_none(args.keyword_items("named_by")):
            matched = not_refused(keys.matched(node))
            key_columns.append(matched.with_name(keyword, ctx.api))
        result_info = not_none(cls.result_type_info(frame, plugin))
        return cls(key_columns, known.columns, known.rest, result_info)

    @classmethod
    def result_type_info(cls, frame: Frame, plugin: Plugin) -> TypeInfo | None:
        """Find the class of the group-by that matches the kind of frame.

        Args:
            frame: The frame that receives the `group_by` call.
            plugin: The mypy plugin, which finds the class.

        Returns:
            `TypedGroupBy` for a `TypedDataFrame`, and `TypedLazyGroupBy` for a
            `TypedLazyFrame`. None if the frame is neither, or if the plugin cannot find the
            class.
        """
        receiver_target = Target(frame.as_instance().type.fullname)
        result_target = cls.RESULT_TARGET_BY_RECEIVER.get(receiver_target)
        if result_target is None:  # pragma: no cover
            # RESULT_TARGET_BY_RECEIVER has the same two full names as Frame.TARGETS, so the
            # full name of a Frame is always one of its keys.
            return None
        info = plugin.lookup_fully_qualified(result_target.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # TypedGroupBy and TypedLazyGroupBy live in astrolabe.frame with the frame classes.
            # mypy already loaded that module when this Frame came into existence.
            return None
        return info.node


class _KeyReader:
    """Match each key of one `group_by` call to a declared column, and report a missing one.

    A key is a column name, written as a string or as a known `Final` or `Literal[...]`
    variable. If the frame does not declare the name and its remainder cannot hold it, the key
    is surely missing, and the reader reports `UnknownColumnError`. If the remainder can hold
    it, the data type of the key is not known, so the reader gives up, and the hook reports
    through the `allow_unproven` gate.
    """

    def __init__(self, ctx: MethodContext, known: KnownColumns) -> None:
        """Keep the context of the call, and the declared columns and remainder of the frame.

        Args:
            ctx: The mypy context of the `group_by` call.
            known: The declared columns of the frame, and its remainder.
        """
        self._ctx = ctx
        self._rest = known.rest
        self._by_name = known.by_name()
        self._literals = BareLiteralReader()

    def matched(self, node: Expression) -> ColumnRef | Refusal:
        """Give the declared column that one key names.

        Args:
            node: The AST node of the key.

        Returns:
            The declared column, or a refusal: `DYNAMIC_NAME` if the key is not a known string,
            `REPORTED` if the remainder cannot hold the name, after the method reports it, and
            `REMAINDER_MAY_HOLD` if it can.
        """
        name = self._literals.string(node, self._ctx.api.get_expression_type(node))
        if name is None:
            return Refusal(Reason.DYNAMIC_NAME)
        matched = self._by_name.get(name)
        if matched is not None:
            return matched
        if RemainderKnowledge().proves_absent(self._rest, name):
            UnknownColumnError(name).fail(self._ctx)
            return Refusal(Reason.REPORTED)
        return Refusal(Reason.REMAINDER_MAY_HOLD)


class GroupByHook:
    """Type `df.group_by(...)` on a typed frame as a typed group-by.

    The result holds the key columns, the declared columns of the frame and its remainder, so
    that the call after it can build its result frame. If the hook cannot read the keys, it
    reports through the `allow_unproven` setting and gives the plain polars group-by. The call
    after it then gives a plain frame, which the checker does not track.
    """

    def __init__(self, plugin: Plugin, untyped_fallback: UntypedFallback) -> None:
        """Keep the plugin and the fallback for a call that the hook cannot type.

        Args:
            plugin: The mypy plugin, which finds the classes of the result.
            untyped_fallback: The gate that reports a call that the hook cannot type.
        """
        self._plugin = plugin
        self._untyped = untyped_fallback

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `group_by` call.

        Args:
            ctx: The mypy context of the `group_by` call.

        Returns:
            A `TypedGroupBy` or `TypedLazyGroupBy` with the keys, the columns and the
            remainder. If the hook cannot read the call, the mypy default, which is the plain
            polars `GroupBy` or `LazyGroupBy`. It first reports `ImplicitlyUntypedError` unless
            the setting lets the reason pass.
        """
        plan = _GroupByPlan.from_context(ctx, self._plugin)
        if isinstance(plan, Refusal):
            self._untyped.report_if_disallowed(ctx, plan.reason, "group_by")
            return ctx.default_return_type
        keys_tuple = Frame.object_tuple(self._materialized(plan.key_columns, ctx), ctx.api)
        original_tuple = Frame.object_tuple(self._materialized(plan.known_columns, ctx), ctx.api)
        rest = NoneType() if plan.rest is None else plan.rest
        return Instance(plan.result_info, [keys_tuple, original_tuple, rest])

    def _materialized(self, columns: list[ColumnRef], ctx: MethodContext) -> list[Instance]:
        """Give a `Column[...]` instance for each column, to put in a tuple type argument.

        A typed group-by holds its columns as tuples of `Column[...]` items. A column read from
        the `TypedDict` of a typed frame has no such item, so this method builds one.

        Args:
            columns: The columns.
            ctx: The mypy context of the `group_by` call.

        Returns:
            One `Column[...]` instance for each column, in order.
        """
        return [
            checked_not_none(column.materialized(ctx.api, self._plugin)).as_instance()
            for column in columns
        ]
