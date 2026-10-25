"""Type `.rename({...})` on a typed frame: the result has the new column names.

The typed frame declares `.rename()` to return its own type, so without this
hook the checker keeps the old names after the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.errors import (
    DuplicateColumnError,
    UndeclaredColumnError,
    UnknownColumnError,
)
from astrolabe._hooks.propagate import Refusal, explains, not_none, propagates

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _RenamePlan:
    """Hold what the hook reads from one `.rename()` call.

    Attributes:
        frame: The receiver frame.
        known: The declared columns of the receiver, and its remainder.
        mapping: The literal dictionary of the call, from old name to new
            name.
        original_names: The names of the declared columns of the
            receiver, in order.
    """

    frame: Frame
    known: KnownColumns
    mapping: dict[str, str]
    original_names: list[str]

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(cls, ctx: MethodContext) -> _RenamePlan:
        """Read the receiver and the literal dictionary of a `.rename()` call.

        Args:
            ctx: The context that mypy gives for `.rename()`.

        Returns:
            The plan, or a `Refusal`. The reason is `DYNAMIC_NAME` if the
            argument is not a literal dictionary of strings. It is
            `CHECKER_GAP` if the receiver is not a typed frame, or if a
            declared column has no literal name.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        mapping = not_none(CallArgs(ctx).literal_mapping("mapping"), reason=Reason.DYNAMIC_NAME)
        known = not_none(frame.known_columns)
        names: list[str] = []
        for column in known.columns:
            names.append(not_none(column.name))
        return cls(frame, known, mapping, names)


class RenameHook:
    """Give `.rename({...})` on a typed frame the renamed columns, and report the calls that fail.

    The hook renames each declared column that the dictionary names, and
    keeps the order and the data types. polars fails if a result has two
    columns with one name. The hook checks the final names together, so a
    swap such as `{"a": "b", "b": "a"}` is correct.

    The receiver can be an open frame, whose type does not name all of its
    columns. The columns that it does not name are its remainder. Each old
    name must be a declared column. A new name must be a declared name, or
    a name that the remainder cannot hold. The remainder never holds a
    declared name, because a frame holds each name once. Any other new
    name can clash with a hidden column of the caller, so the hook reports
    a `DuplicateColumnError`.

    An old name that no column takes back is free after the call. The hook
    takes it out of the remainder, written `Minus[...]`, so that a later
    operation can prove that no hidden column has that name.

    It is a mypy method hook in the plugin layer. If it cannot type the
    call, it falls back through `UntypedFallback`, which gives the plain
    frame that polars declares.
    """

    def __init__(self, untyped_fallback: UntypedFallback, plugin: Plugin) -> None:
        """Keep the fallback and the plugin, and build the duplicate check.

        Args:
            untyped_fallback: Gives the plain frame for a call that the
                hook cannot type, and reports it unless the
                `allow_unproven` setting lets it pass.
            plugin: The mypy plugin, which finds the marker classes of
                the new remainder.
        """
        self._untyped = untyped_fallback
        self._plugin = plugin
        self._duplicate_check = DuplicateNameCheck("Rename")

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.rename()` call.

        A call with no argument always fails, and mypy reports the missing
        `mapping` from the signature of polars. The fallback then gets
        `REPORTED`, so that it adds no second error.

        The duplicate check reads the final names of all columns
        together, and not each rename alone. So a swap such as
        `{"a": "b", "b": "a"}` passes: each new name matches an old one,
        but no two columns share a final name.

        Args:
            ctx: The context that mypy gives for `.rename()`.

        Returns:
            A typed frame of the same kind as the receiver, with the new
            names. If the hook cannot type the call, the plain
            `pl.DataFrame` or `pl.LazyFrame`. The plain frame also if an
            old name is not a declared column, two final names are the
            same, or a new name can clash with the remainder.
        """
        plan = _RenamePlan.from_context(ctx)
        if isinstance(plan, Refusal):
            missing = not CallArgs(ctx).is_provided("mapping")
            reason = Reason.REPORTED if missing else plan.reason
            return self._untyped.resolve(ctx, reason, "rename", Frame.from_proper_type(ctx.type))
        final_names = [plan.mapping.get(name, name) for name in plan.original_names]
        if not self._can_narrow(ctx, plan, final_names):
            return self._untyped.resolve(ctx, Reason.REPORTED, "rename", plan.frame)
        narrowed = self._narrowed(ctx, plan, final_names)
        if narrowed is None:  # pragma: no cover
            # The plugin makes mypy load the modules of the remainder
            # markers in every run, so the rebuild never fails.
            return self._untyped.resolve(ctx, Reason.CHECKER_GAP, "rename", plan.frame)
        return narrowed

    @propagates(None)
    def _narrowed(
        self,
        ctx: MethodContext,
        plan: _RenamePlan,
        final_names: list[str],
    ) -> MypyType | None:
        """Build the result frame, and take each freed name out of the remainder.

        The declared columns keep their order and data types, with their new
        names. A freed name is an old name that the rename gives up and no
        column takes back. A swap such as `{"a": "b", "b": "a"}` frees no
        name, because both names stay declared, on different columns.

        Args:
            ctx: The context that mypy gives for `.rename()`.
            plan: What the hook read from the call.
            final_names: The name of each declared column after the
                rename.

        Returns:
            The result frame, or None if the new remainder cannot be
            built.
        """
        kept = set(final_names)
        freed = [name for name in plan.original_names if name not in kept]
        editor = RemainderEdit.for_checker(self._plugin, ctx.api)
        rest = editor.minus(plan.known.rest, freed)
        pairs = zip(plan.known.columns, final_names, strict=True)
        renamed_columns = [column.with_name(final_name, ctx.api) for column, final_name in pairs]
        return plan.frame.rebuilt(renamed_columns, ctx.api, rest=rest).as_instance()

    def _can_narrow(self, ctx: MethodContext, plan: _RenamePlan, final_names: list[str]) -> bool:
        """Run the three checks on the names, and report the first that fails.

        Each old name must be a declared column. polars raises
        `ColumnNotFoundError` for an old name whenever the remainder does not
        hold it. If the remainder cannot hold the name, the call always
        fails, and the hook reports `UnknownColumnError`. Else the call fails
        for some callers, and the hook reports `UndeclaredColumnError` unless
        the `allow_unproven` setting lets it pass.

        No two final names can be the same.

        No new name can clash with a hidden column of the remainder. polars
        raises `DuplicateError` if the remainder holds a new name. So a new
        name that the remainder can hold makes the call fail for some
        callers. The hook reports it as a `DuplicateColumnError`, as for a
        clash with a declared column.

        Args:
            ctx: The context that mypy gives for `.rename()`.
            plan: What the hook read from the call.
            final_names: The name of each declared column after the
                rename.

        Returns:
            True if each old name is a declared column, no two final names
            are the same, and no new name can clash with the remainder.
        """
        names_present = set(plan.original_names)
        for old_name in plan.mapping:
            if old_name in names_present:
                continue
            if RemainderKnowledge().proves_absent(plan.known.rest, old_name):
                UnknownColumnError(old_name).fail(ctx)
            else:
                self._untyped.report(ctx, UndeclaredColumnError(old_name, "rename"))
            return False
        if self._duplicate_check.fails(ctx, final_names):
            return False
        ambiguous = self._first_ambiguous_destination(plan.known, plan.original_names, final_names)
        if ambiguous is None:
            return True
        DuplicateColumnError("Rename", ambiguous).fail(ctx)
        return False

    def _first_ambiguous_destination(
        self,
        known: KnownColumns,
        original_names: list[str],
        final_names: list[str],
    ) -> str | None:
        """Find the first new name that the remainder can hold.

        A declared name is always safe, because the remainder never holds
        it. Any other name is safe only if the type of the remainder
        proves it absent, for example through
        `other_columns(without=[...])` in the declaration.

        Args:
            known: The declared columns of the receiver, and its remainder.
            original_names: The names of the declared columns before the
                rename.
            final_names: The names of the declared columns after the
                rename.

        Returns:
            The first such name in column order. None if the frame is
            closed, or if every new name is safe.

        Example:
            >>> fixture = new_type_fixture()
            >>> foo = column_ref(fixture, "foo", int_dtype(fixture))
            >>> hook = RenameHook(untyped_fallback(), new_plugin(fixture))
            >>> open_rest = declared_rest(fixture)
            >>> known = known_columns([foo], open_rest)
            >>> hook._first_ambiguous_destination(known, ["foo"], ["bar"])
            'bar'
            >>> hook._first_ambiguous_destination(known, ["foo"], ["foo"]) is None
            True
            >>> excluding = known_columns([foo], declared_rest(fixture, without=("bar",)))
            >>> hook._first_ambiguous_destination(excluding, ["foo"], ["bar"]) is None
            True
            >>> hook._first_ambiguous_destination(excluding, ["foo"], ["baz"])
            'baz'
        """
        if known.rest is None:
            return None
        declared = set(original_names)
        for name in final_names:
            if name in declared:
                continue
            if not RemainderKnowledge().proves_absent(known.rest, name):
                return name
        return None
