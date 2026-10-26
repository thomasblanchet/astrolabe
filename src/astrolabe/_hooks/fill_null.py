"""Type `df.fill_null(...)` on a typed frame, which fills some columns and can change their types.

polars declares that the call returns a plain frame, so without this hook
the checker loses every column of a typed frame at this call.

polars runs the call as another operation over one expression. It picks the
filled columns from the Python class of the value. A number fills the
numeric columns, a `str` fills the `String` and `Categorical` columns, a
`date` fills the `Date` columns, and from polars 1.40.0 each of these also
fills the `Null` columns: polars runs
`with_columns(pl.col([...]).fill_null(value))` over those data types. An
expression, a value of another class and a strategy fill every column:
polars runs `select(pl.all().fill_null(...))`. The hook types the call as
that operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from mypy.nodes import CallExpr
from mypy.types import AnyType, Instance, LiteralType, get_proper_type

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.calls.constant_fold import ConstantFold
from astrolabe._hooks.calls.temporal_value import TemporalValueReader
from astrolabe._hooks.checked import checked_cast, checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.encoding.schema_renderings import SchemaRenderings
from astrolabe._hooks.encoding.marker_minter import ProperReads
from astrolabe._hooks.errors import InvalidCallShapeError, UnsupportedDtypeOperationError
from astrolabe._hooks.expression_requests import ExpressionRequest
from astrolabe._hooks.propagate import Refusal, explains, not_falsy, not_none, not_refused
from astrolabe._hooks.recording.recorder import RecordedShortForm, ShortForm
from astrolabe._hooks.select import SelectHook
from astrolabe._hooks.unproven import Reason
from astrolabe._rules.releases import polars_releases
from astrolabe._hooks.with_columns import WithColumnsHook

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


class FrameFillNullHook:
    """Give `df.fill_null(...)` on a typed frame its result type.

    The hook first reports a call that polars refuses whatever the data:
    no value and no strategy, a value with a strategy, and a `limit` with
    a strategy other than `forward` or `backward`. It then reads the data
    types that the value fills, records the expression that polars runs
    with the arguments of the call, and types the call as `with_columns` or
    `select` over that expression. Where the hook cannot type the call, it
    falls back to the plain frame, subject to the `allow_unproven` setting.

    It is a mypy method hook in the plugin layer.
    """

    OPERATION: ClassVar[str] = "fill_null"
    PARAMETERS: ClassVar[tuple[str, ...]] = ("value", "strategy", "limit")
    EVERY_COLUMN: ClassVar[ShortForm] = ShortForm("pl.all().fill_null()", PARAMETERS)

    def __init__(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Keep the plugin, the fallback and the two hooks that type the call.

        Args:
            plugin: The mypy plugin, which finds the classes of the new
                data types.
            untyped_fallback: The fallback that gives the plain frame,
                subject to the `allow_unproven` setting.
            unchecked_fallback: The fallback that gives an `Unchecked`
                column.
        """
        self._plugin = plugin
        self._untyped = untyped_fallback
        self._with_columns = WithColumnsHook(
            untyped_fallback,
            plugin,
            unchecked_fallback,
            built_for=self.OPERATION,
        )
        self._select = SelectHook(
            untyped_fallback,
            plugin,
            unchecked_fallback,
            built_for=self.OPERATION,
        )

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.fill_null(...)` call on a frame.

        Args:
            ctx: The context that mypy gives for `.fill_null(...)`.

        Returns:
            The typed frame after the fill, or the plain frame where the
            hook reported an error or cannot type the call.
        """
        frame = Frame.from_proper_type(ctx.type)
        known = None if frame is None else frame.known_columns
        if frame is None or known is None:  # pragma: no cover
            # The plugin calls this hook only for a typed frame, whose
            # schema is always a `TypedDict`.
            return ctx.default_return_type
        refusal = FillCallShape(CallArgs(ctx)).refusal()
        if refusal is not None:
            InvalidCallShapeError(refusal).fail(ctx)
            return self._untyped.resolve(ctx, Reason.REPORTED, self.OPERATION, frame)
        kinds = checked_not_none(FrameKinds.of(known))
        refused = self._refused_strategy(ctx, known, kinds)
        if refused is not None:
            return self._untyped.resolve(ctx, refused, self.OPERATION, frame)
        filled = FilledDataTypes(CallArgs(ctx), kinds).code()
        if isinstance(filled, Refusal):
            return self._untyped.resolve(ctx, filled.reason, self.OPERATION, frame)
        form = self.EVERY_COLUMN
        if filled is not None:
            form = ShortForm(f"{filled}.fill_null()", self.PARAMETERS)
        typ = RecordedShortForm(self._plugin, form).recorded(ctx)
        if typ is None:
            # Polars refuses to build the expression, and the recorder reported it.
            return ctx.default_return_type
        request = ExpressionRequest(checked_cast(ctx.context, CallExpr), typ, None)
        hook = self._select if filled is None else self._with_columns
        return hook.typed(ctx, [request])

    def _refused_strategy(
        self,
        ctx: MethodContext,
        known: KnownColumns,
        kinds: FrameKinds,
    ) -> Reason | None:
        """Report a declared column of a refused kind, or find a refused kind in the remainder.

        Polars refuses such a column only where it holds a null, and a fill
        of nulls exists to meet a null, so the call nearly always raises.
        A remainder that can hold such a column makes the result depend on
        the caller.

        Args:
            ctx: The context that mypy gives for the call.
            known: The declared columns of the frame, with its remainder.
            kinds: The kinds that the columns of the frame can have.

        Returns:
            `REPORTED` after the error of a declared column,
            `CALLER_DEPENDENT` for the remainder, or `None` where the
            strategy accepts every column.
        """
        refused = FillStrategy(CallArgs(ctx)).refused()
        for name, rendering in SchemaRenderings().of(known.columns).items():
            kind = rendering.partition("(")[0]
            if kind in refused:
                reason = FillStrategy.REASON.format(kind=kind)
                UnsupportedDtypeOperationError.at_frame(name, reason).fail(ctx)
                return Reason.REPORTED
        return Reason.CALLER_DEPENDENT if kinds.held & refused else None


class RefusedKinds:
    """Give the kinds of the columns that one strategy of `fill_null` refuses.

    A kind is a data type with its parameters taken away, for example
    `Datetime` for every time unit. `min` and `max` need an order of the
    values, `mean` a number, and `zero` and `one` a constant of the physical
    type of the column. Polars computes that value only where the column
    holds a null, and raises `InvalidOperationError` there for a column of
    another kind. `forward` and `backward` accept every column. From polars
    1.44.0, `zero`, `one` and `mean` also refuse a `Categorical` and an
    `Enum`, and `min` and `max` take the smallest and the largest category.

    Example:
        >>> sorted(RefusedKinds().of("min"))
        ['Array', 'List', 'Struct']
    """

    NESTED: ClassVar[frozenset[str]] = frozenset({"Array", "List", "Struct"})
    STRATEGIES: ClassVar[dict[str, frozenset[str]]] = {
        "min": NESTED,
        "max": NESTED,
        "zero": NESTED | {"Null"},
        "one": NESTED | {"Binary", "Null", "String"},
        "mean": NESTED | {"Binary", "Boolean", "String"},
    }
    CATEGORIES: ClassVar[frozenset[str]] = frozenset({"Categorical", "Enum"})
    VALUED: ClassVar[frozenset[str]] = frozenset({"zero", "one", "mean"})

    @polars_releases
    def of(self, strategy: str) -> frozenset[str]:
        """Give the kinds that the strategy refuses, with the categories for a valued strategy.

        Args:
            strategy: The name of the strategy, for example `zero`.

        Returns:
            The names of the kinds. None for a strategy that refuses no kind.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.44.0"):
            ...     sorted(RefusedKinds().of("zero"))
            ['Array', 'Categorical', 'Enum', 'List', 'Null', 'Struct']
        """
        kinds = self._without_categories(strategy)
        return kinds | self.CATEGORIES if strategy in self.VALUED else kinds

    @of.before("1.44.0")
    def _without_categories(self, strategy: str) -> frozenset[str]:
        """Give the kinds that the strategy refuses in every release.

        Args:
            strategy: The name of the strategy, for example `zero`.

        Returns:
            The names of the kinds.

        Example:
            >>> sorted(RefusedKinds()._without_categories("zero"))
            ['Array', 'List', 'Null', 'Struct']
        """
        return self.STRATEGIES.get(strategy, frozenset())


@dataclass(frozen=True)
class FillStrategy:
    """Give the kinds of the columns that the strategy of a `fill_null` call refuses.

    A strategy whose value the call does not state refuses no kind here. The
    kinds of each strategy come from `RefusedKinds`.

    Attributes:
        arguments: The argument values of the call.

    Example:
        >>> mean = call_args(["value", "strategy"], [[], [str_literal_node("mean")]])
        >>> sorted(FillStrategy(mean).refused()) == sorted(RefusedKinds().of("mean"))
        True
    """

    arguments: CallArgs

    REASON: ClassVar[str] = (
        "the fill strategy does not accept {kind} where the column holds a null, "
        "and polars raises InvalidOperationError"
    )

    def refused(self) -> frozenset[str]:
        """Give the kinds that the strategy refuses.

        Returns:
            The kinds that any strategy that the argument can hold refuses.

        Example:
            >>> carried = call_args(["value", "strategy"], [[], [str_literal_node("forward")]])
            >>> FillStrategy(carried).refused()
            frozenset()
        """
        refused: set[str] = set()
        for name in self._names():
            refused |= RefusedKinds().of(name)
        return frozenset(refused)

    def _names(self) -> list[str]:
        """Give each strategy that the argument can hold, where its type states it.

        Returns:
            The literal value of the argument, or each literal value of a
            union of literal types. No name for any other type.

        Example:
            >>> empty = call_args(["value", "strategy"], [[], []])
            >>> FillStrategy(empty)._names()
            []
        """
        if self.arguments.is_omitted_or_none("strategy"):
            return []
        literal = self.arguments.literal_str("strategy")
        if literal is not None:
            return [literal]
        typ = checked_not_none(self.arguments.node_type("strategy"))
        reads = ProperReads()
        names = [reads.string_literal(item) for item in reads.union_items(typ)]
        return [name for name in names if name is not None]


@dataclass(frozen=True)
class FillCallShape:
    """Find the form of a `fill_null` call that polars refuses before it reads any data.

    polars raises `ValueError` at the call for three forms: no value and
    no strategy, a value together with a strategy, and a `limit` with no
    strategy `forward` or `backward`. A strategy that is not a literal
    string hides whether the `limit` is allowed, so it passes.

    Attributes:
        arguments: The argument values of the call.

    Example:
        >>> empty = call_args(["value", "strategy", "limit"], [[], [], []])
        >>> FillCallShape(empty).refusal() == FillCallShape.MISSING
        True
    """

    arguments: CallArgs

    CARRIED: ClassVar[frozenset[str]] = frozenset({"forward", "backward"})

    MISSING: ClassVar[str] = (
        "fill_null() with no value and no strategy raises ValueError: "
        "must specify either a fill `value` or `strategy`"
    )
    BOTH: ClassVar[str] = (
        "fill_null() with a value and a strategy raises ValueError: "
        "cannot specify both `value` and `strategy`"
    )
    LIMIT: ClassVar[str] = (
        "fill_null() with limit= raises ValueError: can only specify `limit` "
        "when strategy is set to 'backward' or 'forward'"
    )

    def refusal(self) -> str | None:
        """Give the message for a call that polars refuses.

        Returns:
            The message, or None for a call that polars accepts.

        Example:
            >>> one = [int_literal_node(1)]
            >>> forward = [str_literal_node("forward")]
            >>> shape = FillCallShape(call_args(["value", "strategy", "limit"], [one, forward, []]))
            >>> shape.refusal() == FillCallShape.BOTH
            True
            >>> carried = call_args(["value", "strategy", "limit"], [[], forward, one])
            >>> FillCallShape(carried).refusal() is None
            True
            >>> limited = FillCallShape(call_args(["value", "strategy", "limit"], [one, [], one]))
            >>> limited.refusal() == FillCallShape.LIMIT
            True
        """
        value = not self.arguments.is_omitted_or_none("value")
        strategy = not self.arguments.is_omitted_or_none("strategy")
        if not value and not strategy:
            return self.MISSING
        if value and strategy:
            return self.BOTH
        named = self.arguments.literal_str("strategy")
        if self.arguments.is_omitted_or_none("limit") or named in self.CARRIED:
            return None
        if strategy and named is None:
            return None
        return self.LIMIT


@dataclass(frozen=True)
class FrameKinds:
    """The kinds that the columns of one frame can have, declared or not.

    Some values make polars touch columns of kinds that the checker cannot
    read from the call. A `datetime` whose time zone the call does not
    state is one. The checker can still type such a call where the frame
    has no column of those kinds. A column of the data type `Unchecked`
    counts as none of them, as everywhere in the checker.

    Attributes:
        declared: The kinds of the declared columns.
        held: The kinds that the undeclared columns can have.

    Example:
        >>> FrameKinds(frozenset({"Int64"}), frozenset()).touches(frozenset({"Null"}))
        False
    """

    declared: frozenset[str]
    held: frozenset[str]

    @classmethod
    def of(cls, known: KnownColumns) -> FrameKinds | None:
        """Read the kinds of a frame from its declared columns and its remainder.

        Args:
            known: The declared columns of the frame, with its remainder.

        Returns:
            The kinds, or None where the remainder does not read.

        Example:
            >>> from astrolabe._hooks.domain.frame import KnownColumns
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> FrameKinds.of(KnownColumns([price], None))
            FrameKinds(declared=frozenset({'Int64'}), held=frozenset())
        """
        renderings = SchemaRenderings().of(known.columns).values()
        declared = frozenset(rendering.partition("(")[0] for rendering in renderings)
        record = RemainderKnowledge().record(known.rest)
        if record is None:  # pragma: no cover
            # The remainder of a typed frame is always a declaration, a
            # split, or an effect on one of them, and each of them reads.
            return None
        return cls(declared, record.types.kinds())

    def touches(self, kinds: frozenset[str]) -> bool:
        """Tell whether a column of the frame can have one of some kinds.

        Args:
            kinds: The kinds.

        Returns:
            True where a declared column, or the remainder, can have one.

        Example:
            >>> FrameKinds(frozenset(), frozenset({"Null"})).touches(frozenset({"Null"}))
            True
        """
        return bool((self.declared | self.held) & kinds)


class FilledDataTypes:
    """Give the data types of the columns that a `fill_null` value fills, in the code of polars.

    polars tests the Python class of the value in a fixed order, a subclass
    before its base, and fills the columns of the data types of that class,
    and from polars 1.40.0 the `Null` columns (`_with_nulls`). An `int` or a
    `float` fills every numeric column. With `matches_supertype=False`, an
    `int` fills only the `Int64` columns, and a `float` only the `Float64`
    columns. An expression, a value of another class and a strategy fill
    every column.

    The type of a value does not always tell its class. mypy accepts an
    `int` where the type is `float`, and the type `date` also covers a
    `datetime`, which polars reads as a `Datetime`. The time zone of a
    `datetime` decides whether polars puts it in a column. A literal or a
    call that builds the value in place tells these facts. Elsewhere, the
    classes that the value can have give the same result only where the
    frame has no column of the kinds where they differ.

    Example:
        >>> empty = FrameKinds(frozenset(), frozenset())
        >>> forward = [str_literal_node("forward")]
        >>> FilledDataTypes(call_args(["value", "strategy"], [[], forward]), empty).code() is None
        True
    """

    DATETIMES: ClassVar[str] = "cs.datetime()"
    DATES: ClassVar[str] = "pl.col([pl.Date])"
    CLASSES: ClassVar[tuple[tuple[str, str], ...]] = (
        (Target.of(bool).fullname, "pl.col([pl.Boolean])"),
        (Target.of(int).fullname, "int"),
        (Target.of(float).fullname, "float"),
        (Target.of(str).fullname, "pl.col([pl.String, pl.Categorical])"),
        (Target.of(datetime).fullname, DATETIMES),
        (Target.of(date).fullname, DATES),
        (Target.of(time).fullname, "pl.col([pl.Time])"),
        (Target.of(timedelta).fullname, "cs.duration()"),
    )
    NUMBERS: ClassVar[str] = (
        "pl.col([pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Int128, pl.UInt8, pl.UInt16, "
        "pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Decimal])"
    )
    EXACT: ClassVar[dict[str, str]] = {
        "int": "pl.col([pl.Int64])",
        "float": "pl.col([pl.Float64])",
    }
    # An int or a float, for a variable of the type `float`.
    EITHER: ClassVar[str] = "pl.col([pl.Int64, pl.Float64])"
    EXPRESSION: ClassVar[str] = Target.of(pl.Expr).fullname

    # The kinds that a `datetime` fills, where the type of the value is `date`.
    DATETIME_KINDS: ClassVar[frozenset[str]] = frozenset({"Datetime"})

    def __init__(self, arguments: CallArgs, kinds: FrameKinds) -> None:
        """Keep the arguments of the call and the kinds of the frame.

        Args:
            arguments: The argument values of the call.
            kinds: The kinds that the columns of the frame can have.
        """
        self._arguments = arguments
        self._kinds = kinds

    @explains(Reason.CHECKER_GAP)
    def code(self) -> str | None:
        """Give the data types that the value fills.

        Returns:
            The data types, as the code of polars spells them. `None` where
            polars fills every column: for a strategy, an expression or a
            value of another class. A refusal where the type of the value
            does not tell what polars fills.
        """
        if not self._arguments.is_omitted_or_none("strategy"):
            return None
        typ = checked_not_none(self._arguments.node_type("value"))
        found = not_refused(self._class(typ))
        node = self._arguments.node_list("value")[0]
        if found in self.EXACT:
            return self._with_nulls(not_refused(self._number(found, node)))
        not_refused(self._temporal(found, node))
        return None if found is None else self._with_nulls(found)

    def _with_nulls(self, code: str) -> str:
        """Add the `Null` columns to the columns that the value fills, where polars fills them.

        Args:
            code: The data types of the class of the value.

        Returns:
            The data types, with `Null` where the installed release fills it
            (`_nulls`).

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> empty = FrameKinds(frozenset(), frozenset())
            >>> fill = FilledDataTypes(None, empty)
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     fill._with_nulls("pl.col([pl.Date])"), fill._with_nulls("cs.datetime()")
            ('pl.col([pl.Date, pl.Null])', '(cs.datetime() | cs.by_dtype(pl.Null))')
            >>> with PolarsRelease.pretend("1.39.3"):
            ...     fill._with_nulls("pl.col([pl.Date])")
            'pl.col([pl.Date])'
        """
        if not self._nulls():
            return code
        if code.startswith("pl.col(["):
            return code.removesuffix("])") + ", pl.Null])"
        return f"({code} | cs.by_dtype(pl.Null))"

    @polars_releases
    def _nulls(self) -> frozenset[str]:
        """Give the kinds that every value fills beside the kinds of its class: `Null`.

        Returns:
            The kinds.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> empty = FrameKinds(frozenset(), frozenset())
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     sorted(FilledDataTypes(None, empty)._nulls())
            ['Null']
        """
        return frozenset({"Null"})

    @_nulls.before("1.40.0")
    def _no_nulls(self) -> frozenset[str]:
        """Give the kinds of the older releases: none, as they leave the `Null` columns as they are.

        Returns:
            No kind.

        Example:
            >>> FilledDataTypes(None, FrameKinds(frozenset(), frozenset()))._no_nulls()
            frozenset()
        """
        return frozenset()

    @explains(Reason.UNTYPED_ARGUMENT)
    def _class(self, typ: MypyType) -> str | None:
        """Give the data types of the Python class of the value, or the class of a number.

        A value of the type `Any` can have any class, so the columns that
        polars fills are not known.

        Args:
            typ: The type that mypy gives the value.

        Returns:
            The data types, `int` or `float`, or `None` for an expression or a
            value of another class.

        Example:
            >>> fixture = new_type_fixture()
            >>> empty = FrameKinds(frozenset(), frozenset())
            >>> FilledDataTypes(None, empty)._class(fixture.lit_str1)
            'pl.col([pl.String, pl.Categorical])'
            >>> FilledDataTypes(None, empty)._class(fixture.a) is None
            True
            >>> FilledDataTypes(None, empty)._class(fixture.anyt)
            Refusal(reason=<Reason.UNTYPED_ARGUMENT: ...>)
        """
        proper = get_proper_type(typ)
        not_falsy(not isinstance(proper, AnyType))
        if isinstance(proper, LiteralType):
            proper = proper.fallback
        if not isinstance(proper, Instance) or proper.type.has_base(self.EXPRESSION):
            return None
        found = (types for fullname, types in self.CLASSES if proper.type.has_base(fullname))
        return next(found, None)

    @explains(Reason.CHECKER_GAP)
    def _number(self, found: str, node: Expression) -> str:
        """Give the data types that a number fills.

        With `matches_supertype=False`, a variable of the type `float` can
        hold an `int`, which fills the `Int64` columns, or a `float`, which
        fills the `Float64` columns. The code then fills both kinds with
        every number. That gives the answers of the two classes: a float
        turns an `Int64` column into `Float64`, so no answer there is
        `Int64`, and an int leaves a `Float64` column as it is.

        Args:
            found: `int` or `float`, from the type of the value.
            node: The value, as the user wrote it.

        Returns:
            The data types. A refusal where `matches_supertype` is not a
            literal.

        Example:
            >>> empty = FrameKinds(frozenset(), frozenset())
            >>> one = [int_literal_node(1)]
            >>> arguments = call_args(["value", "matches_supertype"], [one, []])
            >>> FilledDataTypes(arguments, empty)._number("int", one[0]) == FilledDataTypes.NUMBERS
            True
            >>> exact = call_args(["value", "matches_supertype"], [one, [bool_literal_node(False)]])
            >>> FilledDataTypes(exact, empty)._number("int", one[0])
            'pl.col([pl.Int64])'
        """
        matches = self._arguments.literal_bool("matches_supertype", default=True)
        if not_none(matches):
            return self.NUMBERS
        if found == "float" and not isinstance(ConstantFold().value(node), float):
            return self.EITHER
        return self.EXACT[found]

    def _temporal(self, found: str | None, node: Expression) -> Refusal | None:
        """Check that the class and the time zone of a temporal value are known where they matter.

        A `datetime` whose call does not state its time zone, and a `date`
        that can be a `datetime`, change the `Datetime` and the `Null`
        columns in a way that the checker cannot see.

        Args:
            found: The data types that the class of the value fills.
            node: The value, as the user wrote it.

        Returns:
            None, or a refusal where the frame can hold a column of those kinds.

        Example:
            >>> empty = FrameKinds(frozenset(), frozenset())
            >>> day = NameExpr("day")
            >>> FilledDataTypes(None, empty)._temporal(FilledDataTypes.DATES, day) is None
            True
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> dates = FrameKinds(frozenset({"Datetime"}), frozenset())
            >>> FilledDataTypes(None, dates)._temporal(FilledDataTypes.DATES, day)
            Refusal(reason=<Reason.NON_LITERAL_DTYPE: ...>)
        """
        reader = TemporalValueReader()
        open_zone = found == self.DATETIMES and reader.datetime_rendering(node) is None
        open_class = found == self.DATES and not reader.exact_date(node)
        if (open_zone or open_class) and self._kinds.touches(self.DATETIME_KINDS | self._nulls()):
            return Refusal(Reason.NON_LITERAL_DTYPE)
        return None
