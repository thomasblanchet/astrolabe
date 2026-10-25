"""Type `rolling(...)` and `group_by_dynamic(...)` as group-bys over windows of the rows.

Both calls group the rows of a frame into windows along an index column, and
the `.agg(...)` after them runs over each window as over a group. That call
needs the keys of `group_by=`, the index column, the columns of the frame and
the kind of windows, so this hook gives the group-by a type that holds them,
such as `TypedRollingGroupBy`. The windows of `rolling` always overlap. The
windows of `group_by_dynamic` overlap where `period` is longer than `every`,
or as long with `closed="both"`. Polars refuses an index column of another
data type than the call reads, and a duration whose unit does not fit the
index: a count of rows, as `2i`, for an integer index, and a span of time, as
`2d`, for a temporal one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import TypeInfo
from mypy.types import Instance, LiteralType, NoneType

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.frame import Frame, KnownColumns
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeReads
from astrolabe._hooks.errors import InvalidCallShapeError, UnknownColumnError, WrongColumnShapeError
from astrolabe._hooks.propagate import Refusal, explains, not_none, not_refused
from astrolabe._hooks.unproven import Reason
from astrolabe.frame import (
    TypedDataFrame,
    TypedDynamicGroupBy,
    TypedLazyDynamicGroupBy,
    TypedLazyFrame,
    TypedLazyRollingGroupBy,
    TypedRollingGroupBy,
)

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class WindowDuration:
    """Hold a duration of polars, as `2i` or `1d12h`: its length, and whether it counts rows.

    Polars reads a duration as numbers, each with a unit. An index unit, `i`,
    counts the values of an integer index. The other units are spans of time,
    and polars orders two durations by their nanoseconds, with a month of 28
    days.

    Example:
        >>> WindowDuration.parse("2i")
        WindowDuration(length=2, index=True)
        >>> WindowDuration.parse("1d12h").length == 36 * 3600 * 10**9
        True
    """

    NANOSECONDS: ClassVar[dict[str, int]] = {
        "ns": 1,
        "us": 10**3,
        "ms": 10**6,
        "s": 10**9,
        "m": 60 * 10**9,
        "h": 3600 * 10**9,
        "d": 86400 * 10**9,
        "w": 7 * 86400 * 10**9,
        "mo": 28 * 86400 * 10**9,
        "q": 3 * 28 * 86400 * 10**9,
        "y": 12 * 28 * 86400 * 10**9,
        "i": 1,
    }
    PART: ClassVar[re.Pattern[str]] = re.compile(r"(\d+)([a-z]+)")

    length: int
    index: bool

    @classmethod
    def parse(cls, text: str) -> WindowDuration | None:
        """Read a duration from its text.

        Args:
            text: The text, with an optional sign first.

        Returns:
            The duration, or `None` where polars cannot read the text.

        Example:
            >>> WindowDuration.parse("2x") is None
            True
            >>> WindowDuration.parse("") is None
            True
        """
        body = text[1:] if text[:1] in {"+", "-"} else text
        parts = cls.PART.findall(body)
        written = "".join(number + unit for number, unit in parts)
        if not parts or written != body or any(unit not in cls.NANOSECONDS for _, unit in parts):
            return None
        length = sum(int(number) * cls.NANOSECONDS[unit] for number, unit in parts)
        return cls(length, any(unit == "i" for _, unit in parts))


@dataclass(frozen=True)
class WindowMethod:
    """Hold the rules of one method that groups rows into windows: its index types and durations.

    `rolling` reads an index of `Int32`, `Int64`, `UInt32`, `UInt64`, `Date`
    or `Datetime`, and `group_by_dynamic` the same but the unsigned integers.
    Each duration of the call must fit the index.

    Example:
        >>> WindowMethod.ROLLING.reads("UInt32"), WindowMethod.DYNAMIC.reads("UInt32")
        (True, False)
    """

    INTEGERS: ClassVar[frozenset[str]] = frozenset({"Int32", "Int64", "UInt32", "UInt64"})

    name: str
    integers: frozenset[str]
    durations: tuple[str, ...]

    ROLLING: ClassVar[WindowMethod]
    DYNAMIC: ClassVar[WindowMethod]

    def reads(self, rendering: str) -> bool:
        """Say whether the method reads an index of a data type.

        Args:
            rendering: The rendering of the data type of the index.

        Returns:
            Whether it does.

        Example:
            >>> WindowMethod.ROLLING.reads("Datetime(time_unit='us', time_zone='UTC')")
            True
        """
        return rendering in self.integers or rendering == "Date" or rendering.startswith("Datetime")

    def kinds(self) -> str:
        """Give the data types that the method reads, in words.

        Returns:
            The text.

        Example:
            >>> WindowMethod.DYNAMIC.kinds()
            'Int32, Int64, Date or Datetime'
        """
        return ", ".join(sorted(self.integers)) + ", Date or Datetime"


WindowMethod.ROLLING = WindowMethod(
    "rolling",
    WindowMethod.INTEGERS,
    ("period", "offset"),
)
WindowMethod.DYNAMIC = WindowMethod(
    "group_by_dynamic",
    frozenset({"Int32", "Int64"}),
    ("every", "period", "offset"),
)


@dataclass(frozen=True)
class _WindowPlan:
    """What one call needs to build its typed group-by.

    Attributes:
        key_columns: The keys of `group_by=`, in call order.
        index_column: The index column.
        known_columns: The declared columns of the frame.
        rest: The remainder of the frame, or None for a closed frame.
        windows: The kind of windows, with ` bounds` where the result holds them.
    """

    key_columns: list[ColumnRef]
    index_column: ColumnRef
    known_columns: list[ColumnRef]
    rest: MypyType | None
    windows: str


class _WindowReader:
    """Read one call of `rolling` or `group_by_dynamic`, and report what polars refuses.

    It reads the index column and the keys by name, and the durations and the
    flags as literals. A name that the frame does not declare is an error
    where the remainder cannot hold it. An index of a data type that the
    method does not read, and a duration whose unit does not fit the index,
    are errors.
    """

    def __init__(self, ctx: MethodContext, method: WindowMethod, known: KnownColumns) -> None:
        """Keep the context of the call, the rules of the method and the columns of the frame.

        Args:
            ctx: The mypy context of the call.
            method: The rules of the method.
            known: The declared columns of the frame, and its remainder.
        """
        self._ctx = ctx
        self._method = method
        self._known = known
        self._by_name = known.by_name()
        self._args = CallArgs(ctx)

    @explains(Reason.CHECKER_GAP)
    def plan(self) -> _WindowPlan:
        """Read the call.

        Returns:
            The plan, or a refusal: `DYNAMIC_NAME` for a name that is not a
            literal, `DYNAMIC_OPTION` for a duration or a flag that is not a
            literal, `REMAINDER_MAY_HOLD` for a name that the remainder can
            hold, and `REPORTED` after an error.
        """
        name = not_none(self._args.literal_str("index_column"), reason=Reason.DYNAMIC_NAME)
        index = not_refused(self._column(name))
        keys = [not_refused(self._column(key)) for key in not_refused(self._keys())]
        rendering = self._rendering(index)
        if rendering is not None:
            not_refused(self._index(name, rendering))
        durations = not_refused(self._durations(rendering))
        windows = not_refused(self._windows(durations))
        return _WindowPlan(keys, index, self._known.columns, self._known.rest, windows)

    def _rendering(self, index: ColumnRef) -> str | None:
        """Give the rendering of the data type of the index, where the checker knows it.

        An `Unchecked` index can hold any data type, so the checks of the
        index and of the durations do not read it.

        Args:
            index: The index column.

        Returns:
            The rendering, or None for a data type that the checker does not know.
        """
        rendering = ConstDtypeReads().read_type(index.dtype)
        return None if rendering == "Unchecked" else rendering

    def _keys(self) -> list[str] | Refusal:
        """Give the names of the keys of `group_by=`.

        Returns:
            The names, none where the call groups by nothing, or a refusal.
        """
        if not self._args.is_provided("group_by"):
            return []
        names = self._args.literal_str_list("group_by")
        return Refusal(Reason.DYNAMIC_NAME) if names is None else names

    def _column(self, name: str) -> ColumnRef | Refusal:
        """Give the declared column of a name, or report a name that the frame cannot hold.

        Args:
            name: The name.

        Returns:
            The column, or a refusal.
        """
        column = self._by_name.get(name)
        if column is not None:
            return column
        if RemainderKnowledge().proves_absent(self._known.rest, name):
            UnknownColumnError(name).fail(self._ctx)
            return Refusal(Reason.REPORTED)
        return Refusal(Reason.REMAINDER_MAY_HOLD)

    def _index(self, name: str, rendering: str) -> Refusal | None:
        """Report an index of a data type that the method does not read.

        Args:
            name: The name of the index column.
            rendering: The rendering of its data type.

        Returns:
            None where the method reads it, or a refusal after the error.
        """
        if self._method.reads(rendering):
            return None
        operation = f".{self._method.name}()"
        error = WrongColumnShapeError.on_column(
            name, expected=self._method.kinds(), operation=operation
        )
        error.fail(self._ctx)
        return Refusal(Reason.REPORTED)

    def _durations(self, rendering: str | None) -> dict[str, WindowDuration] | Refusal:
        """Read each duration of the call, and report one that the index does not read.

        Args:
            rendering: The rendering of the data type of the index, or None where it is not known.

        Returns:
            Each duration that the call writes, by its parameter, or a refusal.
        """
        durations: dict[str, WindowDuration] = {}
        for parameter in self._method.durations:
            if not self._args.is_provided(parameter):
                continue
            text = not_none(self._args.literal_str(parameter), reason=Reason.DYNAMIC_OPTION)
            duration = WindowDuration.parse(text)
            refusal = self._refused(parameter, text, duration, rendering)
            if refusal is not None:
                return refusal
            durations[parameter] = not_none(duration)
        return durations

    def _refused(
        self,
        parameter: str,
        text: str,
        duration: WindowDuration | None,
        rendering: str | None,
    ) -> Refusal | None:
        """Report a duration that polars cannot read, or whose unit does not fit the index.

        Args:
            parameter: The parameter of the duration.
            text: The text of the duration.
            duration: The duration, or None where polars cannot read the text.
            rendering: The rendering of the data type of the index, or None where it is not known.

        Returns:
            A refusal after the error, or None.
        """
        if duration is None:
            reason = "polars reads a number and a unit, as '2i' or '1d'"
        elif rendering is None or duration.index == (rendering in WindowMethod.INTEGERS):
            return None
        elif duration.index:
            reason = "polars reads a span of time, as '2d', over a temporal index"
        else:
            reason = "polars reads a count of rows, as '2i', over an integer index"
        call = f"{self._method.name}() with {parameter}={text!r}"
        message = f"{call} raises InvalidOperationError: {reason}"
        InvalidCallShapeError(message).fail(self._ctx)
        return Refusal(Reason.REPORTED)

    def _windows(self, durations: dict[str, WindowDuration]) -> str | Refusal:
        """Give the kind of windows of the call, with ` bounds` where the result holds them.

        Args:
            durations: The duration of each parameter that the call writes.

        Returns:
            `rolling`, `dynamic` for windows of `group_by_dynamic` that
            overlap, or `disjoint` for windows that do not, or a refusal.
        """
        if self._method is WindowMethod.ROLLING:
            return "rolling"
        bounds = not_none(
            self._args.literal_bool("include_boundaries", default=False),
            reason=Reason.DYNAMIC_OPTION,
        )
        closed = not_none(
            self._args.literal_str("closed", default="left"), reason=Reason.DYNAMIC_OPTION
        )
        every = not_none(durations.get("every"), reason=Reason.DYNAMIC_OPTION)
        period = durations.get("period", every)
        longer = period.length >= every.length if closed == "both" else period.length > every.length
        kind = "dynamic" if longer else "disjoint"
        return f"{kind} bounds" if bounds else kind


class WindowHook:
    """Type `rolling(...)` or `group_by_dynamic(...)` on a typed frame as a typed group-by.

    The result holds the keys, the columns of the frame, its remainder, the
    index column and the kind of windows, so that the `.agg(...)` after it can
    build its frame. If the hook cannot read the call, it reports through the
    `allow_unproven` setting and gives the plain polars group-by.
    """

    RESULTS: ClassVar[dict[tuple[str, Target], Target]] = {
        ("rolling", Target.of(TypedDataFrame)): Target.of(TypedRollingGroupBy),
        ("rolling", Target.of(TypedLazyFrame)): Target.of(TypedLazyRollingGroupBy),
        ("group_by_dynamic", Target.of(TypedDataFrame)): Target.of(TypedDynamicGroupBy),
        ("group_by_dynamic", Target.of(TypedLazyFrame)): Target.of(TypedLazyDynamicGroupBy),
    }

    def __init__(
        self, plugin: Plugin, untyped_fallback: UntypedFallback, method: WindowMethod
    ) -> None:
        """Keep the plugin, the fallback and the rules of the method.

        Args:
            plugin: The mypy plugin, which finds the class of the result.
            untyped_fallback: The gate that reports a call that the hook cannot type.
            method: The rules of the method.
        """
        self._plugin = plugin
        self._untyped = untyped_fallback
        self._method = method

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one call.

        Args:
            ctx: The mypy context of the call.

        Returns:
            The typed group-by, or the mypy default, after a report unless the
            setting lets the reason pass.
        """
        typed = self._typed(ctx)
        if isinstance(typed, Refusal):
            self._untyped.report_if_disallowed(ctx, typed.reason, self._method.name)
            return ctx.default_return_type
        return typed

    @explains(Reason.CHECKER_GAP)
    def _typed(self, ctx: MethodContext) -> Instance:
        """Read the call and build the typed group-by.

        Args:
            ctx: The mypy context of the call.

        Returns:
            The typed group-by, or a refusal.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        plan = not_refused(_WindowReader(ctx, self._method, known).plan())
        target = self.RESULTS[self._method.name, Target(frame.as_instance().type.fullname)]
        info = not_none(self._plugin.lookup_fully_qualified(target.fullname)).node
        info = not_none(info if isinstance(info, TypeInfo) else None)
        keys = Frame.object_tuple(self._materialized(plan.key_columns, ctx), ctx.api)
        original = Frame.object_tuple(self._materialized(plan.known_columns, ctx), ctx.api)
        index = Frame.object_tuple(self._materialized([plan.index_column], ctx), ctx.api)
        text = ctx.api.named_generic_type("builtins.str", [])
        windows = LiteralType(plan.windows, text)
        rest = NoneType() if plan.rest is None else plan.rest
        return Instance(info, [keys, original, rest, index, windows])

    def _materialized(self, columns: list[ColumnRef], ctx: MethodContext) -> list[Instance]:
        """Give a `Column[...]` instance for each column, to put in a tuple type argument.

        Args:
            columns: The columns.
            ctx: The mypy context of the call.

        Returns:
            One `Column[...]` instance for each column, in order.
        """
        return [
            checked_not_none(column.materialized(ctx.api, self._plugin)).as_instance()
            for column in columns
        ]
