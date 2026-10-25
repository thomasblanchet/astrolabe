"""Turn each expression argument of a frame operation into the columns that it gives.

`select`, `with_columns`, `select` on a plain frame, `agg`, `filter` and
more take the same forms of argument. An argument can be a typed
expression, a bare string that names a column, a bare number or
Boolean, a selection such as `cs.numeric()`, or a struct unnest. Each
operation needs to know which columns such an argument gives when it
runs on the frame.

`RequestResolver` answers that for every operation in the same way. It
reads each form, applies the type of an expression to the schema of the
frame through the expression engine, and reports each fault with the
same errors. The hooks that use it differ only in how they put the
columns into their result.

When a fault or an unknown makes the data type unknowable, the resolver
keeps the output name with the data type `Unchecked`, and the rest of
the result keeps its schema.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import polars.selectors as cs
from mypy.types import Instance, get_proper_type
from polars.lazyframe.group_by import LazyGroupBy
from mypy.types import UnionType

from astrolabe._hooks.calls.bare_literal import BareLiteralReader
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.calls.column_expr import ColumnExpr
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.calls.dtype_constructor_readers import (
    CheckerLiteralTypeBuilder,
)
from astrolabe._hooks.deferred.argument import DeferredArgument, DeferredFields
from astrolabe._hooks.deferred.evaluation import DeferredEvaluation
from astrolabe._hooks.deferred.remainder import DeferredRemainder
from astrolabe._hooks.deferred.code import Code
from astrolabe._hooks.deferred.code_reader import HoleValues
from astrolabe._hooks.deferred.open_types import Generalization
from astrolabe._rules.dtypes import Rendering, RenderingError
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.encoding.expr_evaluation import (
    ExpressionArgument,
    ExpressionEvaluation,
)
from astrolabe._hooks.encoding.marker_minter import MarkerMinter, ProperReads
from astrolabe._hooks.encoding.schema_renderings import MaterializedColumn
from astrolabe._hooks.encoding.selection_codec import SelectionReader
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.errors import (
    UncastableDtypeError,
    DuplicateColumnError,
    InvalidCallShapeError,
    SelectionRenameError,
    WrongColumnShapeError,
    UndeclaredColumnError,
    BlankColumnError,
    UnknownColumnError,
    UnnamedReadError,
    UnsupportedDtypeOperationError,
)
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.expression.outcome import UNCHECKED
from astrolabe._hooks.expression.roots import LiteralRoot
from astrolabe._hooks.recording.deferred_types import DeferredReader
from astrolabe._hooks.propagate import (
    explains,
    not_falsy,
    not_none,
    not_refused,
    propagates,
)
from astrolabe.expr import TypedColumn
from astrolabe.frame import Struct as _StructMarker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._hooks.deferred.evaluation import DeferredAnswer
    from mypy.nodes import Expression
    from mypy.plugin import FunctionContext, MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.deferred.selection_fit import CodeSelector

    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.expression.quadruple import ExpressionType
    from astrolabe._hooks.expression.selection import RemainderEffect, SelectionType
    from astrolabe._hooks.remainder.record import Record


@dataclass(frozen=True)
class ExpressionRequest:
    """One expression argument of a frame operation, as the resolver reads it.

    Attributes:
        node: The syntax node of the argument.
        arg_type: The type that mypy gives the argument.
        forced_name: The output name from a keyword argument, for example
            `total` in `with_columns(total=...)`. None for a positional
            argument.

    Example:
        >>> from mypy.nodes import StrExpr
        >>> ExpressionRequest(StrExpr("price"), None, None).forced_name is None
        True
    """

    node: Expression
    arg_type: MypyType
    forced_name: str | None


@dataclass(frozen=True)
class Resolution:
    """The columns that one argument gives, and what it does to the remainder.

    Most arguments give one column. A selection gives one column for each
    declared column that it picks, and it can also change the columns of
    the remainder (the columns of an open frame that its type does not
    name), for example by a new data type or a new suffix.

    Attributes:
        columns: The output columns, in order.
        effect: What a selection does to the remainder of the receiver, or
            None for an argument that does not touch it.

    Example:
        >>> Resolution([], None).effect is None
        True
    """

    columns: list[ColumnRef]
    effect: RemainderEffect | None

    @classmethod
    def columns_of(cls, resolutions: Sequence[Resolution]) -> list[ColumnRef]:
        """Join the columns of several resolutions, in order.

        Args:
            resolutions: The resolutions.

        Returns:
            Every column, in call order.

        Example:
            >>> Resolution.columns_of([Resolution([], None), Resolution([], None)])
            []
        """
        columns: list[ColumnRef] = []
        for resolution in resolutions:
            columns.extend(resolution.columns)
        return columns

    @classmethod
    def effects_of(cls, resolutions: Sequence[Resolution]) -> list[RemainderEffect]:
        """Collect what each selection does to the remainder, in order.

        Args:
            resolutions: The resolutions.

        Returns:
            Each effect that a selection gives.

        Example:
            >>> Resolution.effects_of([Resolution([], None)])
            []
        """
        effects: list[RemainderEffect] = []
        for resolution in resolutions:
            if resolution.effect is not None:
                effects.append(resolution.effect)
        return effects


class SelectionArgument:
    """Tell whether an argument is a selection, typed or plain.

    A selection picks columns by a rule, not by name, so a hook that reads
    an argument must know when it has one.

    Example:
        >>> from mypy.types import UninhabitedType
        >>> SelectionArgument().matches(UninhabitedType())
        False
    """

    _SELECTOR: ClassVar[Target] = Target.of(cs.Selector)

    def matches(self, typ: MypyType) -> bool:
        """Tell whether an argument is a selection.

        Args:
            typ: The type of the argument.

        Returns:
            True for an instance of a polars `Selector` class, for the code of
            an expression with a selector at its root, and for a declared
            selection.
        """
        argument = DeferredArgument("row")
        evaluation = DeferredEvaluation("row")
        codes = argument.codes(typ) or []
        if any(evaluation.multiple(argument.declared(code)) for code in codes):
            return True
        instance = ProperReads().instance(typ)
        if instance is not None and self._SELECTOR.is_base_of(instance):
            return True
        return SelectionReader().decoded(typ) is not None


class RequestReader:
    """Read the positional and keyword expression arguments of a call.

    Frame operations name their variadic parameters differently, for
    example `*exprs` and `**named_exprs`. The reader takes those two names
    and gives one list of requests in call order.

    Example:
        >>> RequestReader("args", "kwargs").positional
        'args'
    """

    def __init__(self, positional: str, keyword: str) -> None:
        """Keep the names of the two parameters that the arguments come from.

        Args:
            positional: The name of the `*args` parameter.
            keyword: The name of the `**kwargs` parameter.
        """
        self.positional = positional
        self.keyword = keyword

    @propagates(None)
    def of(self, ctx: MethodContext | FunctionContext) -> list[ExpressionRequest] | None:
        """Read every positional argument, then every keyword argument, in call order.

        Args:
            ctx: The mypy context of the call.

        Returns:
            One request for each argument. A positional argument has no forced
            name. None if an argument unpacked with `**` has no known keyword.
        """
        args = CallArgs(ctx)
        positional = zip(
            args.node_list(self.positional),
            args.type_list(self.positional),
            strict=True,
        )
        requests = [ExpressionRequest(node, typ, None) for node, typ in positional]
        keyword_items = not_none(args.keyword_items(self.keyword))
        keyword_types = args.type_list(self.keyword)
        for (name, node), typ in zip(keyword_items, keyword_types, strict=True):
            requests.append(ExpressionRequest(node, typ, name))
        return requests


class MemberSplit:
    """Split the arguments of a frame operation by the two members of each branch argument.

    An expression that an `if`/`else` builds has a union type of two
    members, one for each branch. Each member gives its own answer at the
    frame, so a frame operation gives one frame for each choice of a member
    of each argument, and mypy checks each later step once for each frame.
    Choices that give equal frames give one frame. A loop that rebuilds an
    expression leaves a union of more members: mypy stops its analysis after
    a few passes, and a later pass can give an answer that none of them
    gives. So an argument of more than two members keeps one frame, and its
    answers merge into one column, as they do where the choices are more
    than `LIMIT`.

    Example:
        >>> MemberSplit().choices([]) is None
        True
    """

    LIMIT: ClassVar[int] = 8
    BRANCHES: ClassVar[tuple[str, str]] = ("if", "else")

    def choices(
        self,
        requests: Sequence[ExpressionRequest] | None,
    ) -> list[list[ExpressionRequest]] | None:
        """Give each choice of one member of each argument.

        Args:
            requests: The arguments, or None where an argument gives no
                keyword to read.

        Returns:
            The choices, or `None` where no argument is a union of codes, or
            where the choices are more than `LIMIT`.

        Example:
            >>> from mypy.nodes import StrExpr
            >>> fixture = new_type_fixture()
            >>> MemberSplit().choices([ExpressionRequest(StrExpr("a"), fixture.anyt, None)]) is None
            True
        """
        options = [self._members(request) for request in requests or []]
        count = math.prod(len(members) for members in options)
        if count == 1 or count > self.LIMIT:
            return None
        return [list(choice) for choice in itertools.product(*options)]

    def joined(self, frames: Sequence[MypyType]) -> MypyType:
        """Give the union of the frames of the choices, each distinct frame once.

        Args:
            frames: The frame of each choice.

        Returns:
            The union, or the one frame where every choice gives it.

        Example:
            >>> fixture = new_type_fixture()
            >>> MemberSplit().joined([fixture.a, fixture.a]) == fixture.a
            True
        """
        distinct = [frame for place, frame in enumerate(frames) if frame not in frames[:place]]
        return UnionType.make_union(distinct)

    def _members(self, request: ExpressionRequest) -> list[ExpressionRequest]:
        """Give one argument for each member of an argument of two members that hold a code.

        Args:
            request: The argument.

        Returns:
            The arguments, or the argument alone where it is no union of two codes.

        Example:
            >>> from mypy.nodes import StrExpr
            >>> fixture = new_type_fixture()
            >>> request = ExpressionRequest(StrExpr("a"), fixture.anyt, None)
            >>> MemberSplit()._members(request) == [request]
            True
        """
        codes = DeferredArgument("row").codes(request.arg_type)
        if codes is None or len(codes) != len(self.BRANCHES):
            return [request]
        members = DeferredReader().members(request.arg_type)
        return [replace(request, arg_type=member) for member in members]


class Unpicked(Enum):
    """Say what the result of an operation holds for a column that a selector does not pick.

    A frame can declare a column `Unchecked`, or with a data type of open
    parameters. A selector that tests the data type then picks the column
    for some data and not for other data. What that means for the result
    depends on the operation:

    - `ABSENT`: the arguments give the columns of the result, as in `select`
      and `agg`, so the data decides whether the result holds the column;
    - `KEPT`: the column stays with its own data type where the selector
      does not pick it, as in `with_columns`;
    - `UNREAD`: the selector picks keys or predicates, and the result keeps
      every column, as in `sort` and `filter`.

    Example:
        >>> Unpicked.KEPT.value
        'kept'
    """

    ABSENT = "absent"
    KEPT = "kept"
    UNREAD = "unread"


@dataclass(frozen=True)
class RequestPolicies:
    """The values that one hook passes to the resolver unchanged at each call.

    Attributes:
        operation: The name of the operation, for the error messages.
        context: `"row"` or `"agg"`. In an aggregation context, each group
            gives one value, so a column read row by row becomes a list.
        plugin: The mypy plugin.
        unchecked_fallback: The fallback that gives an `Unchecked` column,
            and reports it under the `allow_unproven` setting.
        remainder_refusal: The reason to give up where polars refuses a
            selection at a kind that the remainder can hold, or None to
            report that as an error. A hook that builds the selection
            itself gives up, because the user did not write it.
        windows: The windows of a `rolling` or a `group_by_dynamic`, whose
            groups the evaluation of a code takes in the aggregation
            context, or the empty text for the groups of a `group_by`.
        alone: Whether the call holds one expression. Over a lazy frame,
            polars then counts the rows in place of a `len` of an input
            that it does not run.
        unpicked: What the result holds for a column that a selector does
            not pick (`Unpicked`).

    Example:
        >>> RequestPolicies("select", "row", None, unchecked_fallback()).context
        'row'
    """

    operation: str
    context: str
    plugin: Plugin
    unchecked_fallback: UncheckedFallback
    remainder_refusal: Reason | None = None
    windows: str = ""
    alone: bool = False
    unpicked: Unpicked = Unpicked.UNREAD


class RequestResolver:
    """Turn each argument of one call into the columns it gives on the receiver.

    For a typed expression, the resolver materializes the expression type:
    it reads the data type of each column that the expression needs from
    the receiver, looks up the answer in the table of the expression, and
    gives one column under the output name. For a selection, it does the
    same for each declared column that the selector picks. A bare string
    is the short form of `pl.col(name)`. A bare number or Boolean is a new
    literal value.

    A column that the frame does not declare is a hard error when the
    remainder cannot hold it, and an unproven result when it can. An
    expression with no answer at the data types of the frame is an error.
    After each fault, the output keeps its name with the data type
    `Unchecked`.

    Two cases give `Unchecked` without the user writing it, and the
    `allow_unproven` setting decides whether the checker reports them. The
    first is a needed column whose declared data type the engine cannot
    read, for example a type variable. The second is a bare `int` with no
    literal value, because polars picks the width of the integer from the
    value. A bare `float` or `bool` has one data type whatever its value. A
    column that the user declares `Unchecked` is the user's own choice, and
    the checker never reports it.

    The values that stay the same for the call are state of the instance.
    Each argument is a method parameter.
    """

    _STRUCT: ClassVar[Target] = Target.of(_StructMarker)
    _COLUMN: ClassVar[Target] = Target.of(TypedColumn)
    _MATCHED: ClassVar[str] = "the matched columns"
    _LAZY: ClassVar[tuple[str, ...]] = (
        Target.of(pl.LazyFrame).fullname,
        Target.of(LazyGroupBy).fullname,
    )
    _DEFERRED_REASONS: ClassVar[dict[str, Reason]] = {
        "untyped": Reason.PLAIN_EXPRESSION,
        "plain": Reason.PLAIN_EXPRESSION,
        "number": Reason.BARE_INT,
        "name": Reason.DYNAMIC_NAME,
        "opaque": Reason.OPAQUE_KEYWORD_OMITTED,
        "open": Reason.REPORTED,
    }
    _SCALAR_RENDERINGS: ClassVar[dict[str, str]] = {
        Target.of(int).fullname: UNCHECKED,
        Target.of(float).fullname: UNCHECKED,
        Target.of(bool).fullname: "Boolean",
    }

    def __init__(
        self,
        ctx: MethodContext | FunctionContext,
        known: KnownColumns,
        policies: RequestPolicies,
        selected_from: Sequence[ColumnRef] | None = None,
    ) -> None:
        """Keep the values that stay the same for the call.

        Args:
            ctx: The mypy context of the call.
            known: The declared columns and the remainder of the receiver.
            policies: The values that the hook passes unchanged.
            selected_from: The columns that a selection picks from, if they
                differ from the columns of the receiver. For example, `agg`
                keeps its group keys out of every selection. By default, the
                columns of the receiver.
        """
        self._ctx = ctx
        self._known = known
        self._policies = policies
        words = [
            policies.context,
            policies.windows,
            DeferredEvaluation.ALONE if policies.alone else "",
        ]
        self._context = self._engine_context(ctx, " ".join(word for word in words if word))
        self._selected_from = known.columns if selected_from is None else list(selected_from)
        self._by_name = known.by_name()
        self._literals = BareLiteralReader()
        self._argument = ExpressionArgument()

    def _engine_context(self, ctx: MethodContext | FunctionContext, context: str) -> str:
        """Give the context of the evaluation of a code, with the kind of frame that runs it.

        A lazy frame runs every pass of the optimizer when it collects, and an
        eager frame runs fewer, so the two can give different answers.

        Args:
            ctx: The mypy context of the call.
            context: `row` or `agg`.

        Returns:
            The context, after the word `lazy` where the receiver is a lazy
            frame or a lazy group-by.
        """
        receiver = get_proper_type(getattr(ctx, "type", None))
        if not isinstance(receiver, Instance):
            return context
        lazy = any(receiver.type.has_base(fullname) for fullname in self._LAZY)
        return f"{DeferredEvaluation.LAZY}{context}" if lazy else context

    @explains(Reason.CHECKER_GAP)
    def resolved(self, request: ExpressionRequest) -> ColumnRef:
        """Give the one output column of an argument that is not a selection.

        The method reads the argument by its form. An expression gives one
        expression type for each member of its union, and a bare number or
        Boolean gives the type of a new value. The method materializes each
        member alone and merges the outputs. A bare string names a declared
        column. Any other argument keeps its name, where the method can
        recover one, as an `Unchecked` column.

        Args:
            request: The argument.

        Returns:
            The output column. After a fault that the method reports, the
            column keeps its output name with the data type `Unchecked`. If
            the argument cannot give a column at all, a refusal with its
            reason: a dynamic name, a plain expression with no name to keep,
            or a checker gap.
        """
        codes = DeferredArgument(self._context).codes(request.arg_type)
        if codes is not None:
            deferred = [self._deferred_column(code, request.forced_name) for code in codes]
            return not_none(self._merged([not_refused(column) for column in deferred]))
        decoded = self._decoded_items(request.arg_type)
        variants = self._scalar_variants(request) if decoded is None else decoded
        if variants:
            columns = [self._materialized(variant, request.forced_name) for variant in variants]
            return not_none(self._merged(columns))
        name = self._literals.string(request.node, request.arg_type)
        if name is not None:
            return not_none(self._named_column(name, request.forced_name))
        return not_none(self._unchecked_from_salvaged_name(request))

    @explains(Reason.CHECKER_GAP)
    def resolved_all(self, request: ExpressionRequest) -> Resolution:
        """Give every output column of one argument.

        A selection gives one column for each declared column that its
        selector picks, and an effect on the remainder. Any other argument
        gives one column.

        Args:
            request: The argument.

        Returns:
            The resolution. If the argument cannot give columns at all, a
            refusal. The reason is `Reason.REPORTED` for a selection that
            failed with an error. Otherwise it is the reason that `resolved`
            gives.
        """
        deferred = self._deferred_selection(request)
        if deferred is not None:
            return not_refused(deferred)
        selection = SelectionReader().decoded(request.arg_type)
        if selection is not None:
            not_falsy(not self._picked_by_data(selection), reason=Reason.PARTIAL_PICK)
            resolution = self._selection_resolution(request.forced_name, selection)
            return not_none(resolution, reason=Reason.REPORTED)
        return Resolution([not_refused(self.resolved(request))], None)

    @propagates(None)
    def _selection_resolution(
        self,
        forced_name: str | None,
        selection: SelectionType,
    ) -> Resolution | None:
        """Materialize one selection: the columns that it picks, and its effect on the remainder.

        A selection cannot take a name from a keyword argument, because it has
        many outputs. The method reports that as an error.

        Args:
            forced_name: The output name from a keyword argument, or None.
            selection: The type of the selection.

        Returns:
            The resolution, or None after an error that the method reported.
        """
        if forced_name is not None:
            SelectionRenameError("alias").fail(self._ctx)
            return Resolution([self._unchecked_column(forced_name)], None)
        return self._materialized_selection(selection)

    @propagates(None)
    def _materialized_selection(self, selection: SelectionType) -> Resolution | None:
        """Read one selection at each declared column that it picks.

        Args:
            selection: The type of the selection.

        Returns:
            The resolution, or None after the method reported a picked column
            at which the selection has no result.
        """
        outputs = self._evaluation().selection_outputs(
            selection,
            self._selected_from,
            self._policies.context,
        )
        if isinstance(outputs, str):
            UnsupportedDtypeOperationError.at_frame(outputs).fail(self._ctx)
            return None
        partial = set(self._evaluation().partially_picked(selection, self._selected_from))
        if self._policies.unpicked is Unpicked.KEPT:
            outputs = [
                self._unchecked_column(column.name) if column.name in partial else column
                for column in outputs
            ]
        return Resolution(outputs, self._selection_effect(selection))

    def _picked_by_data(self, selection: SelectionType) -> bool:
        """Say whether the data decides which columns of the result a declared selection gives.

        Args:
            selection: The type of the selection.

        Returns:
            Whether a column that the selection picks for some data only is a
            column of the result, or a new name of one.
        """
        if self._policies.unpicked is Unpicked.UNREAD:
            return False
        partial = self._evaluation().partially_picked(selection, self._selected_from)
        renamed = any(selection.affix.applied(name) != name for name in partial)
        return bool(partial) and (self._policies.unpicked is Unpicked.ABSENT or renamed)

    def _struct_fields(self, name: str, column: ColumnRef) -> list[ColumnRef] | None:
        """Read the fields of a declared struct column, and report a column that has none.

        A type that is not a struct with a list of fields gives no list.
        `Unchecked` gives none either.

        Args:
            name: The name of the column, for the message.
            column: The declared column.

        Returns:
            The fields, or None after the method reported a column whose
            declared type gives no list of fields.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> struct = ColumnRef.from_name_and_dtype("st", struct_instance(fixture, price))
            >>> resolver = RequestResolver.__new__(RequestResolver)
            >>> [field.name for field in resolver._struct_fields("st", struct)]
            ['price']
        """
        instance = ProperReads().instance(column.dtype)
        fields = None
        if instance is not None and self._STRUCT.matches(instance):
            fields = ColumnRef.from_argument(instance, 0)
        if fields is None:
            error = WrongColumnShapeError.on_column(
                name,
                expected="a Struct with declared fields",
                operation=".struct.unnest()",
            )
            error.fail(self._ctx)
        return fields

    def _selection_effect(self, selection: SelectionType) -> RemainderEffect | None:
        """Give what a selection does to the remainder of the receiver.

        Args:
            selection: The type of the selection.

        Returns:
            The effect. None for a closed frame, or for a remainder that the
            engine cannot read. None as well if the remainder can hold a data
            type at which the selection has no result. The method then reports
            that error.
        """
        record = RemainderKnowledge().record(self._known.rest)
        if self._known.rest is None or record is None:
            return None
        effect = self._evaluation().selection_effect(selection, record, self._policies.context)
        if isinstance(effect, str):
            UnsupportedDtypeOperationError.over_remainder().fail(self._ctx)
            return None
        return effect

    @propagates(None)
    def _decoded_items(self, typ: MypyType) -> list[ExpressionType] | None:
        """Read each member of a union as an expression type.

        Args:
            typ: The type of the argument.

        Returns:
            One expression type for each member, or None at the first member
            that is not an expression.
        """
        items = ProperReads().union_items(typ)
        return [not_none(self._argument.decoded(item)) for item in items]

    def _scalar_variants(self, request: ExpressionRequest) -> list[ExpressionType]:
        """Read an argument that is not an expression as a bare scalar value.

        The value of a bare number or Boolean literal gives its data type. A
        number whose value the checker cannot read gives the data type of
        its Python type.

        Args:
            request: The argument.

        Returns:
            The expression type of the new value, alone in a list, for a bare
            number or Boolean. Otherwise an empty list. A bare string is never a
            value here. It is the short form of `pl.col(name)`.
        """
        value: int | float | bool | None = self._literals.numeric(request.node, request.arg_type)
        if value is None:
            value = self._literals.boolean(request.node, request.arg_type)
        if value is None:
            rendering = self._unchecked_scalar(request)
        else:
            code: Code[MypyType] = Code(f"pl.lit({value!r})")
            rendering = DeferredArgument(self._context).answer(code, []).rendering
        if rendering is None:
            return []
        return [LiteralRoot().of(rendering)]

    def _unchecked_scalar(self, request: ExpressionRequest) -> str | None:
        """Give the data type of a scalar argument whose value the checker cannot read.

        Args:
            request: The argument.

        Returns:
            `Boolean` for a `bool`. `Unchecked` for an `int` or a `float`, after
            the fallback reports it, because the checker cannot know the width
            of the integer without its value. mypy accepts an int where the
            code declares a `float`, so a `float` can be an int too. None for
            any other type.
        """
        instance = ProperReads().instance(request.arg_type)
        if instance is None:
            return None
        rendering = self._SCALAR_RENDERINGS.get(instance.type.fullname)
        if rendering == UNCHECKED:
            name = "literal" if request.forced_name is None else request.forced_name
            self._policies.unchecked_fallback.reported(
                self._ctx,
                Reason.BARE_INT,
                self._policies.operation,
                name,
            )
        return rendering

    def _materialized(self, expression: ExpressionType, forced_name: str | None) -> ColumnRef:
        """Materialize one expression type on the columns of the receiver.

        Args:
            expression: The expression type of the argument.
            forced_name: The output name from a keyword argument. It replaces
                the output name of the expression.

        Returns:
            The output column. After the method reports a missing column or a
            combination of data types with no answer, an `Unchecked` column
            under the output name.
        """
        target = expression if forced_name is None else expression.renamed(forced_name)
        if not self._needs_declared(target):
            return self._unchecked_column(target.name)
        answer = self._evaluation().materialized(
            target,
            self._known.columns,
            self._policies.context,
        )
        if isinstance(answer, str):
            UnsupportedDtypeOperationError.at_frame(target.name, answer).fail(self._ctx)
            return self._unchecked_column(target.name)
        return answer

    @explains(Reason.CHECKER_GAP)
    def _deferred_selection(self, request: ExpressionRequest) -> Resolution | None:
        """Give the columns of a deferred argument whose code holds a selector.

        Polars expands the selector over the columns of the receiver, so the
        argument gives one column for each declared column that it picks, and
        an effect on the remainder of an open frame. A selection cannot take a
        name from a keyword argument.

        Args:
            request: The argument.

        Returns:
            The resolution, `None` where the argument is not a deferred
            selection, or a refusal where the checker cannot answer.
        """
        codes = DeferredArgument(self._context).codes(request.arg_type)
        if codes is None or len(codes) != 1:
            return None
        refused = self._policies.remainder_refusal
        return not_refused(self._code_selection(codes[0], request.forced_name, refused))

    @explains(Reason.CHECKER_GAP)
    def _code_selection(
        self,
        code: Code[MypyType],
        forced_name: str | None,
        refused: Reason | None,
    ) -> Resolution | None:
        """Give the columns of a deferred code that holds a selector, or a read of several fields.

        Args:
            code: The code.
            forced_name: The output name from a keyword argument, or `None`.
            refused: The reason to give up where polars refuses the code at a
                kind that the remainder can hold, or `None` to report that
                as an error.

        Returns:
            The resolution, `None` where the code expands to no columns, or a
            refusal where the checker cannot answer.
        """
        argument = DeferredArgument(self._context)
        selector = argument.selector(code)
        leaf = None if selector is not None else argument.leaf_selector(code)
        if leaf is not None:
            return not_refused(self._leaf_columns(code, leaf, refused, forced_name))
        fields = (
            None
            if selector is not None
            else DeferredFields(self._context).fields(code, self._known.columns)
        )
        if selector is None and fields is None:
            return None
        if fields == []:
            not_falsy(not self._refused_fields(code), reason=Reason.REPORTED)
            picks_none = DeferredFields(self._context).picks_no_field(code, self._known.columns)
            not_falsy(picks_none, reason=Reason.CHECKER_GAP)
        if fields is not None:
            aliased = argument.aliased(code)
            return not_refused(self._named_columns(fields, None, aliased, forced_name))
        return not_refused(self._selector_columns(code, selector, refused, forced_name))

    @explains(Reason.CHECKER_GAP)
    def _selector_columns(
        self,
        code: Code[MypyType],
        selector: object,
        refused: Reason | None,
        forced_name: str | None,
    ) -> Resolution:
        """Give the columns of a deferred code whose selector picks columns of the receiver.

        Polars expands the selector over the columns that it can pick, which
        leave out the group keys of `agg`, and runs the code once for each
        picked column. The code itself can read every column of the frame.
        A name or a pattern of the selector from a variable of the type `str`
        is a value that no code knows, so the data decides the columns.

        Args:
            code: The code.
            selector: The serialized selector of the code.
            refused: The reason to give up where polars refuses the code at a
                kind that the remainder can hold, or `None` to report that
                as an error.
            forced_name: The output name from a keyword argument, or `None`.

        Returns:
            The resolution, or a refusal where the checker cannot answer or
            where the method reported an error.
        """
        argument = DeferredArgument(self._context)
        not_falsy(not HoleValues().held(selector), reason=Reason.DYNAMIC_NAME)
        not_falsy(not self._reported_missing(selector), reason=Reason.REPORTED)
        not_falsy(not self._reported_read(code), reason=Reason.REPORTED)
        selectable = [column.name for column in self._selected_from if column.name is not None]
        answers = not_none(argument.selected(code, self._known.columns, selector, selectable))
        answers = not_none(self._settled(answers), reason=Reason.PARTIAL_PICK)
        reported = any(answer.refused for answer in answers)
        effect = not_refused(self._deferred_effect(code, refused, reported=reported))
        aliased = argument.aliased(code)
        return not_refused(self._named_columns(answers, effect, aliased, forced_name))

    def _settled(self, answers: list[DeferredAnswer]) -> list[DeferredAnswer] | None:
        """Give the answers of a selector where the data decides whether it picks a column.

        Where the arguments give the columns of the result, such a column is
        there for some data and not for other data, so no type holds the
        result. Where the column stays either way, under its own name, its
        type is what its answer and its own data type share. A new name, as
        `.name.suffix()` gives, exists only where the selector picks the
        column.

        Args:
            answers: The answer for each column that the selector picks.

        Returns:
            The answers, or `None` where no type holds the result.
        """
        if self._policies.unpicked is Unpicked.UNREAD:
            return answers
        own = DeferredArgument(self._context).renderings(self._known.columns)
        settled = []
        for answer in answers:
            kept = self._policies.unpicked is Unpicked.KEPT and answer.name == answer.partial
            if answer.partial and not kept:
                return None
            settled.append(self._kept(answer, own[answer.partial]) if answer.partial else answer)
        return settled

    def _kept(self, answer: DeferredAnswer, own: str) -> DeferredAnswer:
        """Give the answer of a column that keeps its own data type where it is not picked.

        Args:
            answer: The answer where the selector picks the column.
            own: The rendering of the declared data type of the column.

        Returns:
            The answer, with the data type that both data types share.

        Example:
            >>> from astrolabe._hooks.deferred.evaluation import DeferredAnswer
            >>> resolver = RequestResolver.__new__(RequestResolver)
            >>> resolver._kept(DeferredAnswer("u", "Int8", partial="u"), "Unchecked").rendering
            'Unchecked'
        """
        shared = [self._shared(answer.rendering, own), self._shared(answer.stored, own)]
        stored = shared[1] if answer.stored else ""
        return replace(answer, rendering=shared[0], stored=stored, partial="")

    def _shared(self, rendering: str, own: str) -> str:
        """Give the rendering of the data type that two data types share.

        Args:
            rendering: One rendering.
            own: The other.

        Returns:
            The rendering, `Unchecked` where one of them names no data type.

        Example:
            >>> resolver = RequestResolver.__new__(RequestResolver)
            >>> resolver._shared("Int8", "Int8"), resolver._shared("Int8", "Unchecked")
            ('Int8', 'Unchecked')
        """
        try:
            dtypes = [Rendering().parse(text) for text in (rendering, own)]
        except RenderingError:
            return UNCHECKED
        return Generalization().of(dtypes).render()

    @explains(Reason.CHECKER_GAP)
    def _leaf_columns(
        self,
        code: Code[MypyType],
        leaf: CodeSelector,
        refused: Reason | None,
        forced_name: str | None,
    ) -> Resolution:
        """Give the columns of a deferred code that reads a declared selection, one for each pick.

        The declared selection picks columns by its rule, and the code runs
        once for each picked column, with the declared type and the rename
        there. It also gives an effect on the remainder, kind by kind.

        Args:
            code: The code.
            leaf: The leaf of the declared selection, with its rule.
            refused: The reason to give up where polars refuses the code at a
                kind that the remainder can hold, or `None` to report that
                as an error.
            forced_name: The output name from a keyword argument, or `None`.

        Returns:
            The resolution, or a refusal where the checker cannot answer or
            where the method reported an error.
        """
        argument = DeferredArgument(self._context)
        not_falsy(not self._reported_read(code), reason=Reason.REPORTED)
        selectable = [column.name for column in self._selected_from if column.name is not None]
        answers = argument.leaf_selected(code, self._known.columns, leaf, selectable)
        reported = any(answer.refused for answer in answers)
        effect = not_refused(self._deferred_effect(code, refused, reported=reported))
        aliased = argument.aliased(code)
        return not_refused(self._named_columns(answers, effect, aliased, forced_name))

    @explains(Reason.CHECKER_GAP)
    def _named_columns(
        self,
        answers: Sequence[DeferredAnswer],
        effect: RemainderEffect | None,
        aliased: bool,
        forced_name: str | None,
    ) -> Resolution:
        """Give the columns of an expansion, under the name of a keyword argument if there is one.

        A keyword argument names every output of the expansion, so polars
        refuses the name where the expansion gives several columns. It
        gives one column where it picks one column, and none where it picks
        none.

        Args:
            answers: The answer of each column that the expansion gives.
            effect: The effect on the remainder of the receiver, or `None`.
            aliased: Whether the outermost call of the code is `.alias()`.
            forced_name: The output name from a keyword argument, or `None`.

        Returns:
            The resolution, or a refusal where an answer gives no column or
            where the outputs depend on the remainder of the caller.
        """
        kept = not_refused(self._fixed_effect(effect, forced_name))
        if forced_name is None:
            return not_refused(self._deferred_columns(answers, kept, aliased=aliased))
        if len(answers) > 1:
            SelectionRenameError("alias").fail(self._ctx)
            return Resolution([self._unchecked_column(forced_name)], None)
        named = [not_none(self._answered_column(answer, forced_name)) for answer in answers]
        return Resolution(named, None)

    @explains(Reason.CALLER_DEPENDENT)
    def _fixed_effect(
        self,
        effect: RemainderEffect | None,
        forced_name: str | None,
    ) -> RemainderEffect | None:
        """Keep the effect of an expansion on the remainder, unless one name labels every output.

        A keyword argument, an outermost `.alias()` or a column on the left of
        a selection gives every output one name. Where the remainder of an open
        frame can hold a column that the selector picks, the number of outputs
        depends on the caller, and polars refuses two outputs of one name. So
        the result depends on the caller. Where the remainder holds no such
        column, the declared columns are the whole answer.

        Args:
            effect: The effect on the remainder of the receiver, or `None`.
            forced_name: The output name from a keyword argument, or `None`.

        Returns:
            The effect where each output keeps a name of its own, and `None`
            where one name labels every output and the remainder holds no
            column that the selector picks. A refusal where the remainder can
            hold such a column.
        """
        if effect is None or (forced_name is None and effect.affix.fixed is None):
            return effect
        # A frame with an effect on its remainder has a remainder that the checker reads.
        record = checked_not_none(RemainderKnowledge().record(self._known.rest))
        not_falsy(effect.selector.surviving_types(record.types).is_empty())
        return None

    def _reported_read(self, code: Code[MypyType]) -> bool:
        """Report a column that a selection reads beside its selector, which the frame lacks.

        The selection can pick no declared column and still pick a column of
        the remainder, where polars needs the column. One call reports one
        error, so the columns that the selection picks do not report it again.

        Args:
            code: The code of the selection.

        Returns:
            Whether the method reported an error.
        """
        reads = DeferredArgument(self._context).reads_of(code)
        missing = [name for name in reads if name not in self._by_name]
        if missing:
            self._report_missing(missing[0])
        return bool(missing)

    def _reported_missing(self, selector: object) -> bool:
        """Report a name that a deferred selection requires and a closed frame does not hold.

        Polars raises `ColumnNotFoundError` for it. The remainder of an open
        frame can hold the name, so an open frame gets no error.

        Args:
            selector: The serialized selector of the selection.

        Returns:
            Whether the method reported an error.
        """
        argument = DeferredArgument(self._context)
        missing = argument.missing(selector, self._known.columns)
        if self._known.rest is not None or not missing:
            return False
        self._report_missing(missing[0])
        return True

    def _refused_fields(self, code: Code[MypyType]) -> bool:
        """Report why a deferred read of several fields of a struct picks no field.

        A column that the receiver does not declare is an error, and so is a
        column whose declared type is not a struct with a list of fields, a
        field that the read requires and the struct does not hold, and a place
        past the last field, which names no field.

        Args:
            code: The code of the argument.

        Returns:
            Whether the method reported an error.
        """
        name = DeferredFields(self._context).struct_column(code)
        if name is None:
            return False
        column = self._by_name.get(name)
        if column is None:
            self._report_missing(name)
            return True
        if self._struct_fields(name, column) is None:
            return True
        missing = DeferredFields(self._context).missing_fields(code, self._known.columns)
        if missing:
            UnsupportedDtypeOperationError.at_frame(missing[0]).fail(self._ctx)
            return True
        answer = DeferredArgument(self._context).answer(code, self._known.columns)
        if answer.refused:
            UnsupportedDtypeOperationError.unnamed_at_frame(answer.reason).fail(self._ctx)
        return answer.refused

    @explains(Reason.CHECKER_GAP)
    def _deferred_columns(
        self,
        answers: Sequence[DeferredAnswer],
        effect: RemainderEffect | None,
        *,
        aliased: bool,
    ) -> Resolution:
        """Turn the answers of a deferred expansion into its columns, after a check of the names.

        One call reports one error: after the first column that polars
        refuses, a refused column is `Unchecked` with no error of its own.

        Args:
            answers: The answer of each column that the expansion gives.
            effect: The effect on the remainder of the receiver, or `None`.
            aliased: Whether the outermost call of the code is `.alias()`,
                which gives each column the same name.

        Returns:
            The resolution. A refusal where an answer gives no column.
        """
        repeated = self._repeated_name(answers)
        if repeated is not None and aliased:
            SelectionRenameError("alias").fail(self._ctx)
            return Resolution([self._unchecked_column(repeated)], None)
        if repeated is not None:
            DuplicateColumnError(self._policies.operation, repeated).fail(self._ctx)
            return Resolution([self._unchecked_column(repeated)], None)
        columns: list[ColumnRef] = []
        reported = False
        for answer in answers:
            name = not_none(answer.name)
            if answer.refused and reported:
                columns.append(self._unchecked_column(name))
                continue
            reported = reported or answer.refused
            columns.append(not_none(self._answered_column(answer, name)))
        return Resolution(columns, effect)

    @explains(Reason.CHECKER_GAP)
    def _deferred_effect(
        self,
        code: Code[MypyType],
        refused: Reason | None,
        *,
        reported: bool,
    ) -> RemainderEffect | None:
        """Give what a deferred selection does to the remainder of the receiver.

        One call reports one error: after the error of a declared column, a
        kind of the remainder that polars refuses gives no error of its own.
        The columns of the remainder that the selection leaves `Unchecked`
        are reported together, as each declared column is.

        Args:
            code: The code of the selection.
            refused: The reason to give up where polars refuses the code at a
                kind that the remainder can hold, or `None` to report an error.
            reported: Whether the call already reported the error of a
                declared column.

        Returns:
            The effect. None for a closed frame, or after the method reported
            a kind that the remainder can hold, where polars refuses the code.
            A refusal where the checker cannot read the remainder, the
            selector or the rename, and where polars refuses some data types
            of a kind that the remainder can hold, so that the result depends
            on the caller.
        """
        if self._known.rest is None:
            return None
        record = not_none(RemainderKnowledge().record(self._known.rest))
        argument = DeferredArgument(self._context)
        effect = argument.effect(code, self._known.columns, record)
        if effect == DeferredRemainder.RENAMED:
            renamer = DeferredRemainder(self._context).renamer(code.text)
            SelectionRenameError(renamer).fail(self._ctx)
            return not_none(None, reason=Reason.REPORTED)
        not_falsy(effect != DeferredRemainder.SPLIT, reason=Reason.CALLER_DEPENDENT)
        if isinstance(effect, str) and refused is not None:
            return not_none(None, reason=refused)
        if isinstance(effect, str) and not reported:
            UnsupportedDtypeOperationError.over_remainder().fail(self._ctx)
        if isinstance(effect, str):
            return None
        kept = not_none(effect)
        if any(UNCHECKED in answer for answer in kept.answers()):
            self._report_unchecked_rest(code, record)
        return kept

    def _report_unchecked_rest(
        self,
        code: Code[MypyType],
        record: Record,
    ) -> None:
        """Report the remainder columns that a selection leaves `Unchecked` for a known reason.

        The columns have no names to show, so the error calls them "the
        matched columns". A cause that has no reason, as answers that differ
        across the data types of a kind, gives no error.

        Args:
            code: The code of the selection.
            record: What the checker knows about the remainder.
        """
        argument = DeferredArgument(self._context)
        cause = argument.remainder_cause(code, self._known.columns, record)
        reason = self._DEFERRED_REASONS.get(cause)
        if reason is None:
            return
        fallback = self._policies.unchecked_fallback
        fallback.reported(self._ctx, reason, self._policies.operation, self._MATCHED)

    def _repeated_name(self, answers: Sequence[DeferredAnswer]) -> str | None:
        """Give an output name that two columns of a deferred selection share.

        Args:
            answers: The answer of each column that the selection picks.

        Returns:
            The first name that repeats, or `None`.
        """
        names = [answer.name for answer in answers if answer.name is not None]
        return next((name for name in names if names.count(name) > 1), None)

    @explains(Reason.CHECKER_GAP)
    def _deferred_column(self, code: Code[MypyType], forced_name: str | None) -> ColumnRef:
        """Evaluate the code of a deferred argument on the columns of the receiver.

        A column that the code reads and the receiver does not declare is an
        error, as for a typed expression. An expression that polars refuses
        at the data types of the receiver is an error too, and so is one that
        polars refuses to build. An answer that the rules do not know is an
        `Unchecked` column, which the `allow_unproven` setting reports or
        lets pass.

        Args:
            code: The code of the argument.
            forced_name: The output name from a keyword argument. It replaces
                the output name of the expression.

        Returns:
            The output column. After a fault, an `Unchecked` column under the
            output name. A refusal where the expression has no output name,
            with the reason of the answer, as a name that is not a literal,
            or after the error of a build that polars refuses.
        """
        self._report_unnamed(code)
        answer = DeferredArgument(self._context).answer(code, self._known.columns)
        if answer.build is not None:
            InvalidCallShapeError.refused_by_polars(*answer.build).fail(self._ctx)
            return not_none(None, reason=Reason.REPORTED)
        name = answer.name if forced_name is None else forced_name
        opened = self._opened(code, answer, name)
        if opened is not None:
            return opened
        reason = self._DEFERRED_REASONS.get(answer.cause, Reason.CHECKER_GAP)
        return not_none(self._answered_column(answer, name), reason=reason)

    def _opened(
        self,
        code: Code[MypyType],
        answer: DeferredAnswer,
        name: str | None,
    ) -> ColumnRef | None:
        """Give the column of a code that reads columns that only the remainder can hold.

        The checker cannot prove that the frame holds such a column, so it
        reports the first one. Each caller that holds it can give it any data
        type, so the code takes each one as an `Unchecked` column, and an
        answer that no data type changes, as a call with `return_dtype=`,
        keeps its type.

        Args:
            code: The code of the argument.
            answer: The answer of the code on the receiver.
            name: The output name, or `None`.

        Returns:
            The column, or `None` where the code reads no such column, reads a
            column that the frame cannot hold, or has no output name.
        """
        missing = sorted(set(answer.reads) - set(self._by_name))
        rest = self._known.rest
        absent = [read for read in missing if RemainderKnowledge().proves_absent(rest, read)]
        if name is None or not missing or "" in missing or absent:
            return None
        self._report_missing(missing[0])
        columns = [*self._known.columns, *(self._unchecked_column(read) for read in missing)]
        opened = DeferredArgument(self._context).answer(code, columns)
        minted = None if opened.refused else self._minted(opened, name)
        return minted or self._unchecked_column(name)

    def _report_unnamed(self, code: Code[MypyType]) -> None:
        """Report a declared column whose name is not one literal, where a code reads it.

        Its data type is known, and the column that it reads is not, so the
        checker cannot prove that the frame holds it. A name that is a union
        of literals reads one of its names, so the read is proven where the
        frame declares each of them.

        Args:
            code: The code of the argument.
        """
        leaves = DeferredArgument(self._context).declared(code).leaves
        declared = self._known.by_name().keys()
        unnamed = [leaf for leaf in leaves if leaf is not None and leaf.name == HoleValues.STR]
        if any(not leaf.choices or not declared >= set(leaf.choices) for leaf in unnamed):
            error = UnnamedReadError(self._policies.operation)
            self._policies.unchecked_fallback.report(self._ctx, error)

    def _answered_column(self, answer: DeferredAnswer, name: str | None) -> ColumnRef | None:
        """Turn the answer of a deferred code into its output column, after its errors.

        Args:
            answer: The answer of the code on the receiver.
            name: The output name.

        Returns:
            The output column. After a fault, an `Unchecked` column under the
            output name. `None` where there is no output name.
        """
        if name is None:
            return None
        faulted = self._faulted(answer, name)
        if faulted is not None:
            return faulted
        return self._minted(answer, name)

    def _faulted(self, answer: DeferredAnswer, name: str) -> ColumnRef | None:
        """Report the fault of an answer and give its `Unchecked` column, or `None` for no fault.

        The code reads a column that the frame does not have, or polars
        refuses it: a cast that polars does not have, or another refusal.

        Args:
            answer: The answer of the code on the receiver.
            name: The output name.

        Returns:
            An `Unchecked` column under the output name after a fault, and `None` otherwise.
        """
        missing = sorted(set(answer.reads) - set(self._by_name))
        if missing:
            self._report_missing(missing[0])
            return self._unchecked_column(name)
        if not answer.refused:
            return None
        if answer.cast is not None:
            source, target = answer.cast
            UncastableDtypeError.on_expression(source=source, target=target).fail(self._ctx)
        else:
            UnsupportedDtypeOperationError.at_frame(name, answer.reason).fail(self._ctx)
        return self._unchecked_column(name)

    def _minted(self, answer: DeferredAnswer, name: str) -> ColumnRef | None:
        """Mint the output column of an answer that has no fault.

        Args:
            answer: The answer of the code on the receiver.
            name: The output name.

        Returns:
            The column, an `Unchecked` column where the answer has no type, or
            `None` where the gate gives the plain frame.
        """
        minter = MarkerMinter(self._policies.plugin, CheckerLiteralTypeBuilder(), self._ctx.api)
        materialized = MaterializedColumn(minter)
        built = materialized.built(name, answer.rendering)
        if answer.stored and built is not None:
            built = materialized.planned(name, answer.rendering, answer.stored)
        if answer.rendering == UNCHECKED or built is None:
            dtype = self._policies.unchecked_fallback.substituted(
                self._ctx,
                self._DEFERRED_REASONS.get(answer.cause, Reason.CHECKER_GAP),
                answer.source or self._policies.operation,
                name,
            )
            return None if dtype is None else ColumnRef.from_name_and_dtype(name, dtype)
        return built

    def _merged(self, columns: list[ColumnRef]) -> ColumnRef | None:
        """Merge the outputs of the members of a union argument.

        Args:
            columns: One output for each member.

        Returns:
            The output that every member agrees on. An `Unchecked` column under
            the shared name if the data types differ, which the fallback
            reports. None if the names differ.
        """
        first = columns[0]
        if any(column.name != first.name for column in columns):
            return None
        compatibility = UncheckedCompatibility()
        for column in columns:
            if compatibility.resolved(first.dtype, column.dtype) is None:
                return self._implicit_column(checked_not_none(first.name))
        return first

    def _implicit_column(self, name: str) -> ColumnRef:
        """Keep an output whose data type is lost under its name, and report the loss.

        Args:
            name: The output name.

        Returns:
            The `Unchecked` column.
        """
        self._policies.unchecked_fallback.reported(
            self._ctx,
            Reason.CHECKER_GAP,
            self._policies.operation,
            name,
        )
        return self._unchecked_column(name)

    def _unchecked_column(self, name: str) -> ColumnRef:
        """Keep the output name of an argument after a fault, with the data type `Unchecked`.

        Args:
            name: The output name.

        Returns:
            The `Unchecked` column. The rest of the schema stays as it is.
        """
        # `Unchecked` lives in astrolabe.frame, which the plugin adds to
        # every mypy build, so the lookup always finds it.
        dtype = checked_not_none(self._policies.unchecked_fallback.minted())
        return ColumnRef.from_name_and_dtype(name, dtype)

    def _needs_declared(self, expression: ExpressionType) -> bool:
        """Check that the receiver declares every column that an expression needs.

        The remainder never makes a needed column acceptable. If the remainder
        can hold the name, the column is undeclared. If it cannot, the column is
        unknown.

        Args:
            expression: The expression type of the argument.

        Returns:
            True if every needed column is declared. Otherwise the method reports
            the first missing column and gives False.
        """
        for name in sorted(expression.needs - set(self._by_name)):
            self._report_missing(name)
            return False
        return True

    def _report_missing(self, name: str) -> None:
        """Report one needed column that the receiver does not declare.

        A blank read has no name yet, and gets its own error. A name that the
        remainder cannot hold is unknown, a hard error. A name that the
        remainder can hold is undeclared, which the checker reports unless the
        `allow_unproven` setting lets it pass. The caller keeps the output
        under `Unchecked` in each case.

        Args:
            name: The name of the needed column.
        """
        if not name:
            BlankColumnError().fail(self._ctx)
        elif RemainderKnowledge().proves_absent(self._known.rest, name):
            UnknownColumnError(name).fail(self._ctx)
        else:
            self._policies.unchecked_fallback.report(
                self._ctx,
                UndeclaredColumnError(name, self._policies.operation),
            )

    def _named_column(self, name: str, forced_name: str | None) -> ColumnRef | None:
        """Give the declared column for the short form of `pl.col(name)`.

        Args:
            name: The name of the column.
            forced_name: The output name from a keyword argument, or None.

        Returns:
            The column, renamed if a keyword gives a name. After the method
            reports a missing name, an `Unchecked` column.
        """
        output = name if forced_name is None else forced_name
        matched = self._by_name.get(name)
        if matched is None:
            self._report_missing(name)
            return self._unchecked_column(output)
        return ColumnRef.from_name_and_dtype(output, matched.dtype)

    def _unchecked_from_salvaged_name(self, request: ExpressionRequest) -> ColumnRef | None:
        """Keep an argument that the checker cannot read as an `Unchecked` column under its name.

        A plain expression that no hook typed can still name its output
        through a keyword or an outermost `.alias(...)`. The keyword comes
        first. Keeping that name is better than falling back to the plain
        frame for the whole call.

        Args:
            request: The argument.

        Returns:
            The `Unchecked` column. If no name can be recovered, the chain stops
            with the reason that `_unnamed_reason` gives.
        """
        reason = self._unnamed_reason(request.arg_type)
        salvaged = request.forced_name
        if salvaged is None:
            salvaged = ColumnExpr.outermost_alias(request.node, self._ctx.api)
        name = not_none(salvaged, reason=reason)
        dtype = self._policies.unchecked_fallback.substituted(
            self._ctx,
            reason,
            self._policies.operation,
            name,
        )
        if dtype is None:  # pragma: no cover
            # `Unchecked` lives in astrolabe.frame, which the plugin adds
            # to every mypy build, so the lookup always finds it.
            return None
        return ColumnRef.from_name_and_dtype(name, dtype)

    def _unnamed_reason(self, typ: MypyType) -> Reason:
        """Give the reason why an argument that the checker cannot read gives up.

        Args:
            typ: The type of the argument.

        Returns:
            `DYNAMIC_NAME` for a string, which is a column name held in a
            variable. `PLAIN_EXPRESSION` for any other argument, which is an
            expression that no hook typed.

        Example:
            >>> from mypy.types import Instance
            >>> fixture = new_type_fixture()
            >>> resolver = RequestResolver.__new__(RequestResolver)
            >>> text = Instance(fixture.make_type_info("builtins.str"), [])
            >>> resolver._unnamed_reason(text).name
            'DYNAMIC_NAME'
            >>> resolver._unnamed_reason(Instance(fixture.oi, [])).name
            'PLAIN_EXPRESSION'
        """
        instance = ProperReads().instance(typ)
        if instance is not None and instance.type.fullname == "builtins.str":
            return Reason.DYNAMIC_NAME
        return Reason.PLAIN_EXPRESSION

    def _evaluation(self) -> ExpressionEvaluation:
        """Build the service that asks the expression engine, for this call.

        Returns:
            The service, with a minter for the type-checking phase.
        """
        return ExpressionEvaluation(
            MarkerMinter(self._policies.plugin, CheckerLiteralTypeBuilder(), self._ctx.api),
        )
