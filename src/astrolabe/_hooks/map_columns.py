"""Type `.map_columns(names, f)`, which runs a Python function over whole columns.

The function `f` can return any data type, and the checker does not read
Python functions. So each mapped column keeps its name and gets the data
type `Unchecked`, which means that its name is known and its data type is
not. The `allow_unproven` setting decides whether the checker reports that.

The call names its columns in one of two ways: with literal strings, or
with a selector. A selector also matches the columns in the remainder of
an open frame, that is the columns that the type of the frame does not
name. Those columns get `Unchecked` too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.deferred.argument import DeferredArgument
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.errors import UndeclaredColumnError, UnknownColumnError
from astrolabe._hooks.expression.lifting import UNCHECKED
from astrolabe._hooks.expression.selection import Affix, RemainderEffect
from astrolabe._hooks.propagate import (
    Refusal,
    explains,
    not_falsy,
    not_none,
    not_refused,
    propagates,
)

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.deferred.code import Code
    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _MapColumnsPlan:
    """Hold what the hook reads from one `.map_columns()` call.

    Attributes:
        frame: The receiver frame.
        known: The declared columns of the receiver, and its remainder.
        target: The columns that the call maps, as literal names or as a
            selector.
    """

    frame: Frame
    known: KnownColumns
    target: _MappedTarget

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(cls, ctx: MethodContext) -> _MapColumnsPlan:
        """Read the receiver and the mapped columns of a `.map_columns()` call.

        Args:
            ctx: The context that mypy gives for `.map_columns()`.

        Returns:
            The plan, or a `Refusal`. The reason is `DYNAMIC_NAME` if the
            columns are not a literal string, a literal list of strings or
            a typed selector. It is `CHECKER_GAP` if the receiver is not a
            typed frame, or if the checker cannot know what the selector
            picks.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        args = CallArgs(ctx)
        names = args.literal_str_list("column_names")
        if names:
            return cls(frame, known, _NamedTarget(names))
        typ = not_none(args.node_type("column_names"))
        codes = not_none(DeferredArgument("row").codes(typ), reason=Reason.DYNAMIC_NAME)
        not_falsy(len(codes) == 1, reason=Reason.DYNAMIC_NAME)
        return cls(frame, known, not_refused(_SelectedTarget.read(codes[0], known)))


class _ColumnMapper:
    """Give the mapped columns of one `.map_columns()` call the data type `Unchecked`.

    One instance serves one call. It holds what stays the same for the
    whole call: the mypy context, the columns of the receiver, and the
    fallback that reports each new `Unchecked` column. Each step then reads
    that state from `self`, and does not pass it along.
    """

    OPERATION: ClassVar[str] = "map_columns"
    MATCHED: ClassVar[str] = "the matched columns"

    def __init__(
        self,
        ctx: MethodContext,
        known: KnownColumns,
        plugin: Plugin,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Keep the state of one call.

        Args:
            ctx: The context that mypy gives for `.map_columns()`.
            known: The declared columns of the receiver, and its remainder.
            plugin: The mypy plugin, which finds the marker classes of the
                new remainder.
            unchecked_fallback: Gives the data type `Unchecked`, and reports
                it unless the `allow_unproven` setting lets it pass.
        """
        self._ctx = ctx
        self._known = known
        self._plugin = plugin
        self._fallback = unchecked_fallback
        self._by_name = known.by_name()

    @property
    def rest(self) -> MypyType | None:
        """Give the encoded remainder of the receiver.

        Returns:
            The encoded remainder, or None for a closed frame.
        """
        return self._known.rest

    @propagates(None)
    def named_columns(self, names: list[str]) -> list[ColumnRef] | None:
        """Give each named column the data type `Unchecked`, and keep the other columns.

        Args:
            names: The mapped names, in call order.

        Returns:
            The declared columns of the receiver, the named ones as
            `Unchecked`. None if a name is not declared. The name is then
            reported, unless the setting lets it pass.
        """
        for name in names:
            not_falsy(self._declared(name))
        return [self._column(column, frozenset(names)) for column in self._known.columns]

    def mapped_remainder(self, effect: RemainderEffect | None) -> MypyType | None:
        """Give the columns of the remainder that a selector picks the data type `Unchecked`.

        If the selector can pick a column of the remainder, the method reports
        the new `Unchecked` columns. They have no names to show, so the error
        calls them "the matched columns".

        Args:
            effect: What the selector does to the remainder, or None for a
                closed frame.

        Returns:
            The new encoded remainder. None for a closed frame, and the
            remainder unchanged where the selector picks none of its kinds.
        """
        record = RemainderKnowledge().record(self._known.rest)
        if effect is None or record is None:
            return self._known.rest
        if not effect.selector.surviving_types(record.types).kinds():
            return self._known.rest
        self._fallback.reported(self._ctx, Reason.OPAQUE_NO_KEYWORD, self.OPERATION, self.MATCHED)
        mapped = RemainderEffect(effect.selector, UNCHECKED, Affix())
        edit = RemainderEdit.for_checker(self._plugin, self._ctx.api)
        return edit.with_effect(self._known.rest, self._known.rest, mapped)

    def _declared(self, name: str) -> bool:
        """Answer whether one mapped name is declared, and report it if it is not.

        If the remainder cannot hold the name, the call always fails, and
        the method reports `UnknownColumnError`. Else it reports
        `UndeclaredColumnError` unless the `allow_unproven` setting lets
        it pass.

        Args:
            name: The mapped name.

        Returns:
            True for a declared name, else False.
        """
        if name in self._by_name:
            return True
        if RemainderKnowledge().proves_absent(self._known.rest, name):
            UnknownColumnError(name).fail(self._ctx)
        else:
            self._fallback.report(self._ctx, UndeclaredColumnError(name, self.OPERATION))
        return False

    def _column(self, column: ColumnRef, mapped: frozenset[str]) -> ColumnRef:
        """Give one column of the receiver, as `Unchecked` if the call maps it.

        Args:
            column: One declared column of the receiver.
            mapped: The mapped names.

        Returns:
            The column with the data type `Unchecked` if it is mapped,
            else the column unchanged.
        """
        name = column.name
        if name is None or name not in mapped:
            return column
        self._fallback.reported(self._ctx, Reason.OPAQUE_NO_KEYWORD, self.OPERATION, name)
        return ColumnRef.from_name_and_dtype(name, checked_not_none(self._fallback.minted()))


class _MappedTarget:
    """Say which columns one `.map_columns()` call maps, as literal names or as a selector.

    The two shapes need different work, so each one is a small subclass
    with the same two methods: `columns` for the declared columns, and
    `remainder` for the remainder. The hook calls both methods and does not
    need to know which shape it holds.
    """

    def columns(self, mapper: _ColumnMapper) -> list[ColumnRef] | None:
        """Give the declared columns of the receiver, the mapped ones as `Unchecked`.

        Args:
            mapper: The mapper of this call.

        Returns:
            The columns, or None after an error is reported.
        """
        raise NotImplementedError  # pragma: no cover

    def remainder(self, mapper: _ColumnMapper) -> MypyType | None:
        """Give the remainder of the receiver, the mapped part as `Unchecked`.

        Args:
            mapper: The mapper of this call.

        Returns:
            The encoded remainder, or None for a closed frame.
        """
        raise NotImplementedError  # pragma: no cover


@dataclass(frozen=True)
class _NamedTarget(_MappedTarget):
    """Map the columns that literal strings name. The remainder stays the same.

    Each name must be a declared column, so no remainder column changes.

    Attributes:
        names: The mapped names, in call order.
    """

    names: list[str]

    def columns(self, mapper: _ColumnMapper) -> list[ColumnRef] | None:
        """Give the declared columns, the named ones as `Unchecked`.

        Args:
            mapper: The mapper of this call.

        Returns:
            The columns, or None after a name that is not declared is
            reported.
        """
        return mapper.named_columns(self.names)

    def remainder(self, mapper: _ColumnMapper) -> MypyType | None:
        """Give the remainder of the receiver unchanged.

        Args:
            mapper: The mapper of this call.

        Returns:
            The encoded remainder of the receiver.
        """
        return mapper.rest


@dataclass(frozen=True)
class _SelectedTarget(_MappedTarget):
    """Map the columns that a selector picks, both declared and in the remainder.

    Attributes:
        names: The declared columns that the selector picks, in the order of the frame.
        effect: What the selector does to the remainder, or None for a closed frame.
    """

    names: list[str]
    effect: RemainderEffect | None

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def read(cls, code: Code[MypyType], known: KnownColumns) -> _SelectedTarget:
        """Read what the selector of a code picks among the declared columns and in the remainder.

        Args:
            code: The code of the argument.
            known: The declared columns of the receiver, and its remainder.

        Returns:
            The target, or a refusal where the code holds no selector, or where
            the checker cannot know what the selector picks.
        """
        argument = DeferredArgument("row")
        names = not_none(argument.picked(code, known.columns)).names
        if known.rest is None:
            return cls(names, None)
        record = not_none(RemainderKnowledge().record(known.rest))
        effect = argument.effect(code, known.columns, record)
        return cls(names, not_none(effect if isinstance(effect, RemainderEffect) else None))

    def columns(self, mapper: _ColumnMapper) -> list[ColumnRef] | None:
        """Give the declared columns, the picked ones as `Unchecked`.

        Args:
            mapper: The mapper of this call.

        Returns:
            The columns.
        """
        return mapper.named_columns(self.names)

    def remainder(self, mapper: _ColumnMapper) -> MypyType | None:
        """Give the remainder, the picked part as `Unchecked`.

        Args:
            mapper: The mapper of this call.

        Returns:
            The new encoded remainder.
        """
        return mapper.mapped_remainder(self.effect)


class MapColumnsHook:
    """Give `.map_columns()` on a typed frame a result whose mapped columns are `Unchecked`.

    polars declares that `.map_columns()` returns a plain frame, so without
    this hook the checker loses every column at this call. This hook keeps
    each column and marks each mapped column as `Unchecked`.

    It is a mypy method hook in the plugin layer. `_ColumnMapper` does the
    work for one call. If the hook cannot read the mapped columns, or a
    name is not declared, it falls back through `UntypedFallback` to the
    plain frame that polars declares.
    """

    def __init__(
        self,
        untyped_fallback: UntypedFallback,
        plugin: Plugin,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Keep the two fallbacks and the plugin.

        Args:
            untyped_fallback: Gives the plain frame for a call that the
                hook cannot type, and reports it unless the
                `allow_unproven` setting lets it pass.
            plugin: The mypy plugin, which finds the marker classes of the
                new remainder.
            unchecked_fallback: Gives the data type `Unchecked`, and reports
                it unless the setting lets it pass.
        """
        self._untyped = untyped_fallback
        self._plugin = plugin
        self._unchecked = unchecked_fallback

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.map_columns()` call.

        Args:
            ctx: The context that mypy gives for `.map_columns()`.

        Returns:
            A typed frame with each mapped column as `Unchecked`. The
            plain frame if the hook cannot read the mapped columns, or a
            name is not declared.
        """
        plan = _MapColumnsPlan.from_context(ctx)
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                "map_columns",
                Frame.from_proper_type(ctx.type),
            )
        mapper = _ColumnMapper(ctx, plan.known, self._plugin, self._unchecked)
        columns = plan.target.columns(mapper)
        if columns is None:
            return self._untyped.resolve(ctx, Reason.REPORTED, "map_columns", plan.frame)
        rest = plan.target.remainder(mapper)
        return plan.frame.rebuilt(columns, ctx.api, rest=rest).as_instance()
