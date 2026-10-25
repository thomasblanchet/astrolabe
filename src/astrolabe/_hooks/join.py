"""The hooks for `.join()` and `.join_where()`, which give the columns of a join of two frames.

Without these hooks, mypy gives a join the type of the receiver, and the
columns of the other frame are missing from the type.

`.join()` adds the columns of `other` to the columns of the receiver. The
`how` and `coalesce` arguments decide what happens to the join keys:
polars can merge each pair of keys into one column, keep both, or leave out
the right one. Each other right column whose name is also a left name
gets the `suffix`, which is `"_right"` by default.

`.join_where()` is an inner join that matches rows by a predicate, not
by keys. Its result always holds every column of both sides, with the
suffix on each right name that collides. The predicates choose only
rows, never columns. That is the same schema as a cross join, so both
hooks share one builder of the result.

On an open frame, a frame whose type does not name all of its columns,
`JoinRemainderStep` decides the remainder of the result, and reports a
join whose renames depend on the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Final, TYPE_CHECKING

from mypy.nodes import NameExpr
from mypy.types import Instance, ProperType, get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.calls.column_expr import ColumnExpr
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.join_key_compatibility import JoinKeyCompatibility
from astrolabe._hooks.domain.planned import PlannedType
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.join_key_tree import JoinKeyTree
from astrolabe._hooks.encoding.join_remainder import (
    DeclaredNames,
    JoinRemainderStep,
    JoinSides,
    ProbeNames,
)
from astrolabe._hooks.errors import (
    InvalidCallShapeError,
    JoinKeyDtypeMismatchError,
    UnknownColumnError,
)
from astrolabe._hooks.duplicate_name import CollisionSuffix
from astrolabe._hooks.propagate import (
    Refusal,
    explains,
    not_falsy,
    not_none,
    not_refused,
    propagates,
)
from astrolabe._hooks.deferred.argument import DeferredArgument
from astrolabe._hooks.expression_requests import ExpressionRequest, RequestPolicies, RequestResolver
from astrolabe._hooks.filter import BooleanPredicate

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from mypy.nodes import Expression
    from mypy.plugin import MethodContext, Plugin

    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _JoinPlan:
    """The inputs of one `.join()` call, read from the call once.

    The plan reads the two frames, `how`, `coalesce` and the join keys in
    one chain that stops at the first gap and names the reason. It also
    reports each form of call that polars refuses whatever the data.

    Attributes:
        frame: The receiver.
        how: The kind of join: `"inner"`, `"left"`, `"right"`, `"full"`,
            `"semi"`, `"anti"` or `"cross"`.
        left_columns: All the declared columns of the receiver.
        right_columns: All the declared columns of `other`.
        left_key_names: The names of the join keys on the receiver's side.
            An empty list for `how="cross"`, which takes no keys, and for a
            key that transforms its column (see
            `_JoinKeyResolver.has_transforming_key`).
        right_key_names: The names of the join keys on the side of `other`,
            in the same order as `left_key_names`.
        coalesce: Whether to merge each pair of keys into one column. False
            for a key that transforms its column. None to use the default of
            `how`.
        left_rest: The remainder of the receiver, in its encoded form, or
            None for a closed receiver.
        right_rest: The remainder of `other`, or None.
    """

    frame: Frame
    how: str
    left_columns: list[ColumnRef]
    right_columns: list[ColumnRef]
    left_key_names: list[str]
    right_key_names: list[str]
    coalesce: bool | None
    left_rest: MypyType | None = None
    right_rest: MypyType | None = None

    KEYS: ClassVar[tuple[str, ...]] = ("on", "left_on", "right_on")
    SUPPORTED_HOW: ClassVar[frozenset[str]] = frozenset({
        "inner",
        "left",
        "right",
        "full",
        "semi",
        "anti",
        "cross",
    })

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(cls, ctx: MethodContext, compatibility: JoinKeyCompatibility) -> _JoinPlan:
        """Read the plan from a `.join()` call.

        Args:
            ctx: The mypy context of the method call.
            compatibility: The rule that decides whether the data types of a
                pair of keys can join.

        Returns:
            The plan, or a refusal with its reason. The reason is a checker gap
            if the receiver or `other` is not a typed frame, if `how` is not
            supported, or if `coalesce` is not a literal bool or `None`. It is
            `Reason.REPORTED` for a form of call that polars refuses, and
            `Reason.CALLER_KEY` for a key that holds a declared expression.
            Otherwise it is the reason that the join keys give.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        other = not_none(cls.other_frame(ctx))
        left_known = not_none(frame.known_columns)
        right_known = not_none(other.known_columns)
        how = not_none(cls._how(ctx))
        args = CallArgs(ctx)
        not_falsy(cls._on_not_mixed_with_left_right(ctx, args), reason=Reason.REPORTED)
        not_falsy(cls._cross_has_no_keys(ctx, how, args), reason=Reason.REPORTED)
        tree = JoinKeyTree("join")
        not_falsy(tree.holds_no_alias(ctx, cls.KEYS), reason=Reason.REPORTED)
        not_falsy(tree.keeps_rows(ctx, cls.KEYS), reason=Reason.REPORTED)
        not_falsy(not tree.holds_declared(ctx, cls.KEYS), reason=Reason.CALLER_KEY)
        not_falsy(cls._coalesce_is_recognized(args))
        schemas = _JoinSchemas(left_known.columns, right_known.columns)
        resolver = _JoinKeyResolver(ctx, how, schemas, compatibility)
        coalesce = cls._effective_coalesce(args, resolver)
        left_key_names = not_refused(resolver.left_key_names)
        right_key_names = not_refused(resolver.right_key_names)
        return cls(
            frame,
            how,
            left_known.columns,
            right_known.columns,
            left_key_names,
            right_key_names,
            coalesce,
            left_known.rest,
            right_known.rest,
        )

    @classmethod
    def _effective_coalesce(cls, args: CallArgs, resolver: _JoinKeyResolver) -> bool | None:
        """Read `coalesce`, which a key that transforms its column turns off.

        Args:
            args: The arguments of the call.
            resolver: The key resolver of the call, which tells whether a key
                transforms its column.

        Returns:
            False if a key transforms its column, because polars then turns
            coalescing off, even with `coalesce=True`, and gives a
            `UserWarning`. Otherwise the literal value of `coalesce`, or None to
            use the default of `how`. None stands for a missing argument and for
            `coalesce=None`, which is the default of polars.
        """
        if resolver.has_transforming_key:
            return False
        if args.is_omitted_or_none("coalesce"):
            return None
        return args.literal_bool("coalesce")

    @staticmethod
    def other_frame(ctx: MethodContext) -> Frame | None:
        """Read the type of the `other` argument as a frame.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            `other` as a frame, or None if it is not a typed frame.
        """
        other_type = CallArgs(ctx).node_type("other")
        if other_type is None:
            return None
        return Frame.from_proper_type(other_type)

    @classmethod
    def _how(cls, ctx: MethodContext) -> str | None:
        """Read the `how` argument as a kind of join that the hook supports.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The literal value of `how`, with `"outer"` changed to `"full"`. None
            if it is not a literal, or not one of `SUPPORTED_HOW`.
        """
        literal = CallArgs(ctx).literal_str("how", default="inner")
        if literal is None:
            return None
        # polars treats `"outer"` as an old name for `"full"`.
        how = "full" if literal == "outer" else literal
        if how not in cls.SUPPORTED_HOW:
            return None
        return how

    @staticmethod
    def _on_not_mixed_with_left_right(ctx: MethodContext, args: CallArgs) -> bool:
        """Report a call that gives `on=` together with `left_on=` or `right_on=`.

        Polars refuses that call with a `ValueError`.

        Args:
            ctx: The mypy context of the method call.
            args: The arguments of the call.

        Returns:
            True if the call does not mix them. Otherwise the method reports an
            error and gives False.
        """
        if args.is_provided("on") and (args.is_provided("left_on") or args.is_provided("right_on")):
            error = InvalidCallShapeError(
                "join() with on= and left_on=/right_on= raises ValueError: cannot use "
                '"on" in conjunction with "left_on" or "right_on"',
            )
            error.fail(ctx)
            return False
        return True

    @staticmethod
    def _cross_has_no_keys(ctx: MethodContext, how: str, args: CallArgs) -> bool:
        """Report a `how="cross"` call that gives a join key.

        A cross join pairs every row with every row, so it takes no keys, and
        polars refuses a key with a `ValueError`.

        Args:
            ctx: The mypy context of the method call.
            how: The kind of join.
            args: The arguments of the call.

        Returns:
            True if `how` is not `"cross"`, or the call gives no key. Otherwise
            the method reports an error and gives False.
        """
        has_key_argument = any(args.is_provided(key) for key in _JoinPlan.KEYS)
        if how == "cross" and has_key_argument:
            error = InvalidCallShapeError(
                'join() with how="cross" and a join key argument raises ValueError: cross '
                "join should not pass join keys",
            )
            error.fail(ctx)
            return False
        return True

    @staticmethod
    def _coalesce_is_recognized(args: CallArgs) -> bool:
        """Tell whether the `coalesce` argument has a form that the hook can read.

        `None` is a real value of `coalesce`, which means the default of `how`.
        So this is a plain test, and not a value for `not_none` to narrow.
        `_effective_coalesce` reads the value after this test passes.

        Args:
            args: The arguments of the call.

        Returns:
            True if `coalesce` is missing, the literal `None`, or a literal
            `True` or `False`.
        """
        if args.is_omitted_or_none("coalesce"):
            return True
        return args.literal_bool("coalesce") is not None


@dataclass(frozen=True)
class _JoinSchemas:
    """The declared columns of both sides of one `.join()` call, as one value.

    One value keeps the constructor of `_JoinKeyResolver` short.

    Attributes:
        left: All the declared columns of the receiver.
        right: All the declared columns of `other`.
    """

    left: list[ColumnRef]
    right: list[ColumnRef]


class _JoinKeyResolver:
    """Read the join keys of a `.join()` call, and check them against both frames.

    A key can be a string, `pl.col(name)`, a variable that holds such an
    expression, or an expression that transforms its column, for example
    `pl.col("id").cast(pl.String)`. The resolver finds the column name of
    each key, and reports a key that a side lacks, lists of keys of
    different lengths, and a pair of keys whose data types cannot join.

    The values that stay the same for the call are state of the instance.
    Each fact is worked out the first time that code asks for it, and kept.
    """

    NO_KEY_HOW: ClassVar[frozenset[str]] = frozenset({"cross"})

    def __init__(
        self,
        ctx: MethodContext,
        how: str,
        schemas: _JoinSchemas,
        compatibility: JoinKeyCompatibility,
    ) -> None:
        """Keep the values that stay the same for the call.

        Args:
            ctx: The mypy context of the method call.
            how: The kind of join.
            schemas: The declared columns of both sides.
            compatibility: The rule that decides whether the data types of a
                pair of keys can join.
        """
        self._ctx = ctx
        self._how = how
        self._schemas = schemas
        self._compatibility = compatibility

    @property
    @explains(Reason.CHECKER_GAP)
    def left_key_names(self) -> list[str]:
        """The checked names of the join keys on the receiver's side, in call order."""
        return not_refused(self._validated_pair)[0]

    @property
    @explains(Reason.CHECKER_GAP)
    def right_key_names(self) -> list[str]:
        """The checked names of the right join keys, in the order of `left_key_names`."""
        return not_refused(self._validated_pair)[1]

    @cached_property
    def has_transforming_key(self) -> bool:
        """Whether a join key of the call transforms its column.

        A key such as `pl.col(name).str.to_lowercase()` or `.cast(dtype)`, or
        any other chain after the column, transforms its column. Such a key
        turns coalescing off for the whole call, even with `coalesce=True`, and
        polars gives a `UserWarning`. In the result, such a key leaves only the
        real columns that it reads, as they are. The output of a `.cast(...)`
        key keeps the data type of its source, not the target of the cast.

        A keyed join without coalescing keeps every column of both sides. So
        the hook needs neither the name nor the data type of a transforming
        key. It only needs to know that one exists.
        """
        if self._how in self.NO_KEY_HOW:
            return False
        arg_names = ("on",) if self._args.is_provided("on") else ("left_on", "right_on")
        for arg_name in arg_names:
            nodes = self._args.vararg_node_list(arg_name)
            if any(self._is_transforming_key_node(node) for node in nodes):
                return True
        return False

    def _is_transforming_key_node(self, node: Expression) -> bool:
        """Tell whether one key, as written in the call, transforms its column.

        The expression type cannot tell a bare `pl.col(name)` from a chain that
        keeps the data type of the column in every case, for example
        `.str.to_lowercase()`. So the method reads the syntax of the call. A
        literal string, a string variable declared `Final` and a `pl.col(name)`
        call written in place, with exactly one argument, are bare. A plain
        variable is unknown, not transforming. Anything else transforms.

        Args:
            node: The syntax node of the key.

        Returns:
            True if the key transforms its column.
        """
        if ColumnExpr.from_node(node, self._ctx.api) is not None:
            return False
        if isinstance(node, NameExpr):
            return False
        is_col_call = ColumnExpr.COL_TARGET.matches_call(node) and len(node.args) == 1
        return not is_col_call

    @cached_property
    @explains(Reason.REPORTED)
    def _validated_pair(self) -> tuple[list[str], list[str]]:
        """Read and check the join key names of both sides together.

        Each side reads `on=` if the call gives it. Else the receiver reads
        `left_on=` and `other` reads `right_on=`.

        Returns:
            The key names of both sides. Two empty lists for `how="cross"`, and
            for a key that transforms its column (see `has_transforming_key`).
            Otherwise a refusal with its reason. The reason is a dynamic name if
            the keys cannot be read. It is `Reason.REPORTED` after an error: a
            key that a side lacks, lists of different lengths, or a pair of keys
            whose data types cannot join.
        """
        if self._how in self.NO_KEY_HOW or self.has_transforming_key:
            return [], []
        on_given = self._args.is_provided("on")
        left_param = "on" if on_given else "left_on"
        right_param = "on" if on_given else "right_on"
        left = not_none(self._key_names(left_param), reason=Reason.DYNAMIC_NAME)
        right = not_none(self._key_names(right_param), reason=Reason.DYNAMIC_NAME)
        not_falsy(self._keys_equal_length(left, right))
        not_falsy(self._left_keys_exist(left))
        not_falsy(self._right_keys_exist(right))
        not_refused(self._key_mismatch(left, right))
        return left, right

    @cached_property
    def _args(self) -> CallArgs:
        return CallArgs(self._ctx)

    @propagates(None)
    def _key_names(self, arg_name: str) -> list[str] | None:
        """Read one join key argument as a list of column names.

        Polars accepts a key as a literal string, a bare `pl.col(name)`, or a
        list that mixes both. A bare `pl.col(name)` acts exactly like its name:
        the same coalescing and the same data type rule. Call this method only
        after `has_transforming_key` is False. A key that transforms its column
        is not a form that this method reads, because the hook needs no name
        for it.

        Args:
            arg_name: The name of the parameter: `"on"`, `"left_on"` or
                `"right_on"`.

        Returns:
            The column name of each key, in call order. None if the call does
            not give the argument, or if the argument has a form that the
            method does not read.
        """
        literal = self._args.literal_str_list(arg_name)
        if literal is not None:
            return literal
        nodes = self._args.vararg_node_list(arg_name)
        if not nodes:
            return None
        return [not_none(self._bare_column_name(node)) for node in nodes]

    def _bare_column_name(self, node: Expression) -> str | None:
        """Read one key as a bare `pl.col(name)` expression.

        The method first reads the expression type that mypy already gave the
        key. That lets a key held in a variable (`x = pl.col(name)`, then
        `on=x`) give the same name as `pl.col(name)` written in the call, which
        the syntax alone cannot show. Then it reads the syntax, which also finds
        a bare string and a string variable declared `Final`. Neither of those
        is an expression type.

        Args:
            node: The syntax node of the key.

        Returns:
            The column name, or None if the key does not have that form.
        """
        instance_type = self._ctx.api.get_expression_type(node)
        deferred = DeferredArgument("row").bare_column(instance_type)
        if deferred is not None:
            return deferred
        parsed = ColumnExpr.from_node(node, self._ctx.api)
        if parsed is None:
            return None
        return parsed.source_name

    def _keys_equal_length(self, left_names: list[str], right_names: list[str]) -> bool:
        """Report lists of join keys that have different lengths.

        `on=` always gives lists of the same length on both sides, so this
        error applies only to `left_on=` and `right_on=`.

        Args:
            left_names: The names of the left join keys.
            right_names: The names of the right join keys.

        Returns:
            True if the two lists have the same length. Otherwise the method
            reports an error and gives False.
        """
        if len(left_names) != len(right_names):
            error = InvalidCallShapeError(
                "join() with left_on=/right_on= of different lengths raises "
                "InvalidOperationError: the number of columns given as join key "
                "should be equal",
            )
            error.fail(self._ctx)
            return False
        return True

    def _left_column(self, name: str) -> ColumnRef | None:
        """Find a column of the receiver by name.

        Args:
            name: The name of the column.

        Returns:
            The column, or None if the receiver has no column with that name.
        """
        return self._left_index.get(name)

    def _right_column(self, name: str) -> ColumnRef | None:
        """Find a column of `other` by name.

        Args:
            name: The name of the column.

        Returns:
            The column, or None if `other` has no column with that name.
        """
        return self._right_index.get(name)

    @cached_property
    def _left_index(self) -> dict[str, ColumnRef]:
        return self._indexed(self._schemas.left)

    @cached_property
    def _right_index(self) -> dict[str, ColumnRef]:
        return self._indexed(self._schemas.right)

    def _indexed(self, columns: list[ColumnRef]) -> dict[str, ColumnRef]:
        """Index a list of columns by name.

        Args:
            columns: The columns.

        Returns:
            The columns by name. A column with no literal name is left out.
        """
        return {column.name: column for column in columns if column.name is not None}

    def _left_keys_exist(self, names: list[str]) -> bool:
        """Report a left join key that is not a column of the receiver.

        Args:
            names: The names of the left join keys.

        Returns:
            True if each name is a column of the receiver. Otherwise the method
            reports the first missing name and gives False.
        """
        for name in names:
            if self._left_column(name) is None:
                UnknownColumnError(name, frame="left").fail(self._ctx)
                return False
        return True

    def _right_keys_exist(self, names: list[str]) -> bool:
        """Report a right join key that is not a column of `other`.

        Args:
            names: The names of the right join keys.

        Returns:
            True if each name is a column of `other`. Otherwise the method
            reports the first missing name and gives False.
        """
        for name in names:
            if self._right_column(name) is None:
                UnknownColumnError(name, frame="right").fail(self._ctx)
                return False
        return True

    def _key_mismatch(self, left_names: list[str], right_names: list[str]) -> Refusal | None:
        """Find the first pair of join keys whose data types cannot join.

        A lazy frame plans the join at the type that it stores for each key,
        and the run checks the keys again at the types of their data.

        Args:
            left_names: The names of the left join keys.
            right_names: The names of the right join keys, in the same order.

        Returns:
            None if every pair can join. Otherwise the refusal that `_mismatch`
            gives for the first pair that cannot.
        """
        planned = PlannedType()
        for left_name, right_name in zip(left_names, right_names, strict=True):
            left = checked_not_none(self._left_column(left_name)).dtype
            right = checked_not_none(self._right_column(right_name)).dtype
            for read in (planned.stored, planned.run):
                left_dtype, right_dtype = get_proper_type(read(left)), get_proper_type(read(right))
                if not self._compatibility.compatible(left_dtype, right_dtype):
                    return self._mismatch(left_name, left_dtype, right_dtype)
        return None

    def _mismatch(self, name: str, left_dtype: ProperType, right_dtype: ProperType) -> Refusal:
        """Report `[join-key-dtype]` for keys that cannot join, if both data types are known.

        If one side is not a known polars data type, for example a type
        variable in a generic function, the checker cannot tell a mismatch from
        a type that it does not see. The method then reports nothing.

        Args:
            name: The name of the join key.
            left_dtype: The data type on the left side.
            right_dtype: The data type on the right side.

        Returns:
            A refusal: `Reason.REPORTED` for a known pair, after the error. A
            checker gap for an unknown one.
        """
        if not (isinstance(left_dtype, Instance) and isinstance(right_dtype, Instance)):
            return Refusal(Reason.CHECKER_GAP)
        error = JoinKeyDtypeMismatchError(
            name,
            left=left_dtype.type.name,
            right=right_dtype.type.name,
        )
        error.fail(self._ctx)
        return Refusal(Reason.REPORTED)


class _JoinResultMerger:
    """Build the columns of the result of one join from its left and right columns.

    The layout depends on `how` and on coalescing. Without coalescing, and
    for a cross join, the result holds every left column, then every right
    column, with the suffix on each right name that collides. With
    coalescing, each pair of keys becomes one column. For `"inner"`,
    `"left"` and `"full"`, that column has the name and the place of the
    left key. For `"right"`, it has the name of the right key and comes
    after the left columns. A semi or anti join keeps only the left columns,
    and the hook handles it without this class.

    The class never reports an error and never reads an argument. It takes
    the mypy API, not the whole context, only to build or rename a column.
    """

    _DEFAULT_COALESCE_BY_HOW: Final[dict[str, bool]] = {
        "inner": True,
        "left": True,
        "right": True,
        "full": False,
    }

    def __init__(
        self,
        plan: _JoinPlan,
        suffix: str,
        api: object,
        compatibility: JoinKeyCompatibility,
        unchecked: UncheckedCompatibility,
    ) -> None:
        """Keep the values that stay the same for the call.

        Args:
            plan: The inputs of the call.
            suffix: The suffix for a right name that collides.
            api: The mypy API, which builds a renamed or retyped column.
            compatibility: The rule that widens a pair of keys to a common type
                for `how="full"` with coalescing.
            unchecked: The rule that prefers a known data type to `Unchecked`.
        """
        self._plan = plan
        self._suffix = suffix
        self._api = api
        self._compatibility = compatibility
        self._unchecked = unchecked

    @property
    def merged_columns(self) -> list[ColumnRef]:
        """Give the columns of the result for the `how` of the call.

        A cross join, and a join with keys that does not coalesce them, keep
        every column of both sides. A join with keys coalesces them if the
        plan says so, or else by the default of `how`. It then uses the
        layout of `"right"` or the layout of the other kinds of join.

        Returns:
            The columns of the result.
        """
        plan = self._plan
        if plan.how == "cross":
            return self._merge_all(plan.left_columns, plan.right_columns)
        coalesce = plan.coalesce
        if coalesce is None:
            coalesce = self._DEFAULT_COALESCE_BY_HOW[plan.how]
        if not coalesce:
            return self._merge_all(plan.left_columns, plan.right_columns)
        if plan.how == "right":
            return self._right_style_merge()
        return self._left_style_merge()

    def _left_style_merge(self) -> list[ColumnRef]:
        """Merge each pair of keys into one column at the place of the left key.

        `how="inner"`, `"left"` and `"full"` share this layout: the merged key
        keeps the name and the place of the left key. `how="full"` also widens a
        pair of keys of one family to their common type. `"inner"` and `"left"`
        only prefer a known data type to `Unchecked`, because polars never
        widens for them, even when the kept side is the narrower type.

        Returns:
            The left columns, then each right column that is not a key, with
            the suffix on a name that collides.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> plugin = new_plugin(fixture)
            >>> shared_dtype = int_dtype(fixture)
            >>> left_id = column_ref(fixture, "id", shared_dtype)
            >>> name = column_ref(fixture, "name", str_dtype(fixture))
            >>> right_id = column_ref(fixture, "id", shared_dtype)
            >>> receiver = frame(fixture, [left_id.as_instance(), name.as_instance()])
            >>> plan = _JoinPlan(
            ...     receiver, "inner", [left_id, name], [right_id, name], ["id"], ["id"], None,
            ... )
            >>> merger = _JoinResultMerger(
            ...     plan, "_right", api, join_key_compatibility(plugin), unchecked_compatibility(),
            ... )
            >>> merged = merger._left_style_merge()
            >>> [column.name for column in merged]
            ['id', 'name', 'name_right']
        """
        use_supertype = self._plan.how == "full"
        left_columns = self._plan.left_columns
        resolved_left = [
            self._resolved_left_column(column, use_supertype) for column in left_columns
        ]
        left_names = self._column_names(resolved_left)
        non_key_right = self._non_key_right_columns(
            self._plan.right_columns,
            self._plan.right_key_names,
            left_names,
        )
        return [*resolved_left, *non_key_right]

    def _resolved_left_column(self, column: ColumnRef, use_supertype: bool) -> ColumnRef:
        """Give one left column, merged with its right key if it is a join key.

        Args:
            column: A left column.
            use_supertype: The flag for `_resolved_key_column`.

        Returns:
            The column as it is if it is not a join key. Otherwise the answer
            of `_resolved_key_column` against its right key.
        """
        if column.name is None or column.name not in self._paired_right_name:
            return column
        other = self._right_index[self._paired_right_name[column.name]]
        return self._resolved_key_column(column, other, use_supertype=use_supertype)

    def _resolved_key_column(
        self,
        kept: ColumnRef,
        other: ColumnRef,
        *,
        use_supertype: bool,
    ) -> ColumnRef:
        """Decide the data type of one merged pair of join keys.

        Args:
            kept: The key whose name and place the layout keeps.
            other: The key on the other side, which the layout leaves out.
            use_supertype: Whether to widen a pair of one family to their
                common type (`how="full"`), or only prefer a known data type to
                `Unchecked`.

        Returns:
            The kept key with the common type, if `use_supertype` is True and
            the two types have one. Otherwise the kept key as it is, except
            that an `Unchecked` kept key takes the known data type of the other
            key, and keeps its own name. That includes a pair of one family and
            different sizes: every layout except `how="full"` with coalescing
            keeps its own side, with no widening.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> plugin = new_plugin(fixture)
            >>> receiver = frame(fixture, [])
            >>> plan = _JoinPlan(receiver, "cross", [], [], [], [], None)
            >>> merger = _JoinResultMerger(
            ...     plan, "_right", api, join_key_compatibility(plugin), unchecked_compatibility(),
            ... )
            >>> unknown_id = column_ref(fixture, "id", unchecked_instance(fixture))
            >>> concrete_id = column_ref(fixture, "id", int_dtype(fixture))
            >>> merger._resolved_key_column(unknown_id, concrete_id, use_supertype=False).dtype
            polars.datatypes.classes.Int64
            >>> resolved = merger._resolved_key_column(concrete_id, unknown_id, use_supertype=False)
            >>> resolved is concrete_id
            True
        """
        if use_supertype:
            planned = PlannedType()
            kept_dtype, other_dtype = planned.stored(kept.dtype), planned.stored(other.dtype)
            supertype = self._compatibility.full_coalesce_supertype(kept_dtype, other_dtype)
            if supertype is not None:
                return self._retyped(kept, supertype)
        resolved = self._unchecked.resolved(kept.dtype, other.dtype)
        if resolved is None:
            return kept
        return self._retyped(kept, resolved)

    def _retyped(self, kept: ColumnRef, dtype: MypyType) -> ColumnRef:
        """Give a merged key a new data type, and keep its name.

        Args:
            kept: The merged key.
            dtype: The new data type.

        Returns:
            The key as it is if it already has that data type. Otherwise the
            key with the new data type.
        """
        if get_proper_type(dtype) == get_proper_type(kept.dtype):
            return kept
        name = checked_not_none(kept.name)
        return ColumnRef.build(name, dtype, kept, self._api)

    def _right_style_merge(self) -> list[ColumnRef]:
        """Merge each pair of keys of a `how="right"` join into one column with the right name.

        In this layout the merged key has the name of the right key. It comes
        after every left column that is not a key, and before every right
        column that is not a key. If a merged key collides with a left name, it
        gets the suffix, as any right name does.

        Each merged key prefers a known data type to `Unchecked` against its
        left key, with no widening.

        Returns:
            The left columns that are not keys, then the merged keys, then the
            right columns that are not keys, each with the suffix on a name that
            collides.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> plugin = new_plugin(fixture)
            >>> left_id = column_ref(fixture, "id", int_dtype(fixture))
            >>> name = column_ref(fixture, "name", str_dtype(fixture))
            >>> right_id = column_ref(fixture, "id", int_dtype(fixture))
            >>> receiver = frame(fixture, [left_id.as_instance(), name.as_instance()])
            >>> plan = _JoinPlan(
            ...     receiver, "right", [left_id, name], [right_id, name], ["id"], ["id"], None,
            ... )
            >>> merger = _JoinResultMerger(
            ...     plan, "_right", api, join_key_compatibility(plugin), unchecked_compatibility(),
            ... )
            >>> [column.name for column in merger._right_style_merge()]
            ['name', 'id', 'name_right']
        """
        left_key_names = set(self._plan.left_key_names)
        left_columns = self._plan.left_columns
        left_non_key = [column for column in left_columns if column.name not in left_key_names]
        left_non_key_names = self._column_names(left_non_key)
        coalesced_keys: list[ColumnRef] = []
        for name in self._plan.right_key_names:
            kept = self._right_index[name]
            other = self._left_index[self._paired_left_name[name]]
            resolved = self._resolved_key_column(kept, other, use_supertype=False)
            new_name = CollisionSuffix.renamed(name, left_non_key_names, self._suffix)
            if new_name == name:
                coalesced_keys.append(resolved)
                continue
            coalesced_keys.append(resolved.with_name(new_name, self._api))
        collision_names = left_non_key_names | self._column_names(coalesced_keys)
        non_key_right = self._non_key_right_columns(
            self._plan.right_columns,
            self._plan.right_key_names,
            collision_names,
        )
        return [*left_non_key, *coalesced_keys, *non_key_right]

    def _merge_all(
        self,
        left_columns: list[ColumnRef],
        right_columns: list[ColumnRef],
    ) -> list[ColumnRef]:
        """Keep every column of both sides, keys included.

        A cross join always merges this way, because it has no keys. A join
        with keys and no coalescing merges this way too. Its layout is the same
        for every `how` with keys. Only the rows differ.

        Args:
            left_columns: All the columns of the receiver.
            right_columns: All the columns of `other`.

        Returns:
            The left columns, then every right column, with the suffix on a name
            that collides.
        """
        left_names = self._column_names(left_columns)
        merged_right = self._non_key_right_columns(right_columns, [], left_names)
        return [*left_columns, *merged_right]

    def _non_key_right_columns(
        self,
        right_columns: list[ColumnRef],
        right_key_names: list[str],
        collision_names: AbstractSet[str],
    ) -> list[ColumnRef]:
        """List the right columns that are not keys, with the suffix on a name that collides.

        Args:
            right_columns: All the columns of `other`.
            right_key_names: The right join keys to leave out, or an empty list
                to keep every column.
            collision_names: The names that a right column must not collide
                with.

        Returns:
            Each right column that is not a key, with the suffix if its name is
            in `collision_names`. A column with no literal name is left out.
        """
        right_key_name_set = set(right_key_names)
        result_columns: list[ColumnRef] = []
        for column in right_columns:
            if column.name is None or column.name in right_key_name_set:
                continue
            new_name = CollisionSuffix.renamed(column.name, collision_names, self._suffix)
            if new_name == column.name:
                result_columns.append(column)
                continue
            result_columns.append(column.with_name(new_name, self._api))
        return result_columns

    def _column_names(self, columns: list[ColumnRef]) -> AbstractSet[str]:
        """Give the names of a list of columns, as a set.

        Args:
            columns: The columns.

        Returns:
            Each name. A column with no literal name is left out.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> plugin = new_plugin(fixture)
            >>> receiver = frame(fixture, [])
            >>> plan = _JoinPlan(receiver, "cross", [], [], [], [], None)
            >>> merger = _JoinResultMerger(
            ...     plan, "_right", api, join_key_compatibility(plugin), unchecked_compatibility(),
            ... )
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> merger._column_names([price]) == {"price"}
            True
        """
        return {column.name for column in columns if column.name is not None}

    @cached_property
    def _left_index(self) -> dict[str, ColumnRef]:
        return self._indexed(self._plan.left_columns)

    @cached_property
    def _right_index(self) -> dict[str, ColumnRef]:
        return self._indexed(self._plan.right_columns)

    def _indexed(self, columns: list[ColumnRef]) -> dict[str, ColumnRef]:
        """Index a list of columns by name.

        Args:
            columns: The columns.

        Returns:
            The columns by name. A column with no literal name is left out.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> plugin = new_plugin(fixture)
            >>> receiver = frame(fixture, [])
            >>> plan = _JoinPlan(receiver, "cross", [], [], [], [], None)
            >>> merger = _JoinResultMerger(
            ...     plan, "_right", api, join_key_compatibility(plugin), unchecked_compatibility(),
            ... )
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> merger._indexed([price]) == {"price": price}
            True
        """
        return {column.name: column for column in columns if column.name is not None}

    @cached_property
    def _paired_right_name(self) -> dict[str, str]:
        return dict(zip(self._plan.left_key_names, self._plan.right_key_names, strict=True))

    @cached_property
    def _paired_left_name(self) -> dict[str, str]:
        return dict(zip(self._plan.right_key_names, self._plan.left_key_names, strict=True))


class _MergeInventory:
    """List the names of one join for the decision about its remainder.

    To decide the remainder of the result, `JoinRemainderStep` needs to
    know which output names come from each side and which right names the
    join renamed. The class builds the columns of the result once with a
    marker suffix that no real name has. Each name with that marker is a
    renamed right column. It works the same for `.join()` and
    `.join_where()`.
    """

    def __init__(
        self,
        plan: _JoinPlan,
        api: object,
        compatibility: JoinKeyCompatibility,
        unchecked: UncheckedCompatibility,
    ) -> None:
        """Keep the values that stay the same for the call.

        Args:
            plan: The inputs of the call.
            api: The mypy API, for the trial merge.
            compatibility: The rule for the trial merge, as for the real one.
            unchecked: The rule for the trial merge, as for the real one.
        """
        self._plan = plan
        self._api = api
        self._compatibility = compatibility
        self._unchecked = unchecked

    def sides(self, suffix: str) -> JoinSides:
        """Build the list of names for the decision, with the real suffix.

        A trial merge with the marker suffix separates the renamed right
        columns from the others. `JoinSides.from_probe` then sorts the other
        output names by side.

        Args:
            suffix: The real suffix of a right name that collides.

        Returns:
            The list of names.
        """
        merger = _JoinResultMerger(
            self._plan,
            ProbeNames.SUFFIX,
            self._api,
            self._compatibility,
            self._unchecked,
        )
        names = [column.name for column in merger.merged_columns if column.name is not None]
        declared = DeclaredNames(
            self._declared_names(self._plan.left_columns),
            self._declared_names(self._plan.right_columns),
        )
        return JoinSides.from_probe(
            ProbeNames(names),
            self._plan.left_rest,
            self._plan.right_rest,
            declared,
            suffix,
        )

    def _declared_names(self, columns: list[ColumnRef]) -> frozenset[str]:
        """List the declared column names of one side.

        Args:
            columns: The declared columns of the side.

        Returns:
            Each name. A column with no literal name is left out.
        """
        return frozenset(column.name for column in columns if column.name is not None)


class JoinHook:
    """Give the result type of `.join()`, with the columns of both frames.

    `how="semi"` and `how="anti"` give the left schema as it is, because the
    columns of `other` never appear in the result. `how="cross"` takes no
    keys, so its result holds every column of both sides, with the suffix
    on each name that collides. Every other `how` adds the columns of
    `other`, merges or keeps each pair of keys as `coalesce` decides, and
    adds the suffix to each other name that collides.

    The hook reports a join key that a frame lacks, a pair of keys whose
    data types cannot join, a form of call that polars refuses, and a
    result with a duplicate name. If it cannot type the call, it falls
    back: it gives the plain frame type in place of a typed result.
    """

    DEFAULT_SUFFIX: Final[str] = "_right"
    _PASSTHROUGH_HOW: Final[frozenset[str]] = frozenset({"semi", "anti"})

    def __init__(self, untyped_fallback: UntypedFallback, plugin: Plugin) -> None:
        """Build the rules and the fallback that the hook needs.

        Args:
            untyped_fallback: The fallback that gives the plain frame type when
                the hook cannot type the call, subject to the `allow_unproven`
                setting.
            plugin: The mypy plugin. `JoinKeyCompatibility` uses it to mint the
                common type of a pair of keys for `how="full"` with coalescing,
                and `RemainderEdit` uses it to mint the remainder of the result.
        """
        self._untyped = untyped_fallback
        self._unchecked = UncheckedCompatibility()
        self._compatibility = JoinKeyCompatibility(self._unchecked, plugin)
        self._duplicate_check = DuplicateNameCheck("join")
        self._remainder_step = JoinRemainderStep(plugin, untyped_fallback, "join")

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the result type of one `.join()` call.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The left schema as it is for `how="semi"` and `"anti"`. Otherwise
            the joined frame. If the hook cannot type the call, the plain
            `pl.DataFrame` or `pl.LazyFrame`. See `UntypedFallback`.
        """
        plan = _JoinPlan.from_context(ctx, self._compatibility)
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                "join",
                Frame.from_proper_type(ctx.type),
                _JoinPlan.other_frame(ctx),
            )
        if plan.how in self._PASSTHROUGH_HOW:
            rebuilt = plan.frame.rebuilt(plan.left_columns, ctx.api, rest=plan.left_rest)
            return rebuilt.as_instance()
        result = self._merged_result(ctx, plan)
        if isinstance(result, Refusal):
            return self._untyped.resolve(ctx, result.reason, "join", plan.frame)
        return result

    @explains(Reason.REPORTED)
    def _merged_result(self, ctx: MethodContext, plan: _JoinPlan) -> MypyType:
        """Build the result frame for a `how` that adds the columns of `other`.

        Args:
            ctx: The mypy context of the method call.
            plan: The inputs of the call.

        Returns:
            The joined frame, or a refusal with its reason. The reason is a
            dynamic name if `suffix` is not a literal and some columns collide.
            It is `Reason.REPORTED` after an error: a duplicate name in the
            result, or a remainder that makes the renames depend on the caller.
            A renamed right column can still collide, for example if the
            receiver already has a column `x_right`.
        """
        suffix = not_none(self._resolved_suffix(ctx, plan), reason=Reason.DYNAMIC_NAME)
        merger = _JoinResultMerger(plan, suffix, ctx.api, self._compatibility, self._unchecked)
        checked_columns = merger.merged_columns
        names = [column.name for column in checked_columns]
        not_falsy(not self._duplicate_check.fails(ctx, names), reason=Reason.REPORTED)
        inventory = _MergeInventory(plan, ctx.api, self._compatibility, self._unchecked)
        outcome = not_none(
            self._remainder_step.decided(ctx, inventory.sides(suffix)),
            reason=Reason.REPORTED,
        )
        rebuilt = plan.frame.rebuilt(checked_columns, ctx.api, rest=outcome.slot)
        return rebuilt.as_instance()

    def _resolved_suffix(self, ctx: MethodContext, plan: _JoinPlan) -> str | None:
        """Give the suffix to build the result with.

        A `suffix` that is not a literal matters only if some columns of the
        join collide. Otherwise polars never uses its text, so the call can
        still get a typed result.

        Args:
            ctx: The mypy context of the method call.
            plan: The inputs of the call.

        Returns:
            The literal value of `suffix`, or the default if the call does not
            give one. If `suffix` is not a literal: `DEFAULT_SUFFIX` if no
            columns collide, otherwise None.
        """
        literal = CallArgs(ctx).literal_str("suffix", default=self.DEFAULT_SUFFIX)
        if literal is not None:
            return literal
        if self._collides(plan, ctx.api):
            return None
        return self.DEFAULT_SUFFIX

    def _collides(self, plan: _JoinPlan, api: object) -> bool:
        """Tell whether the join renames any column, whatever the text of the suffix.

        The method builds the result twice, with two different trial suffixes.
        If the names differ, at least one right column collided and got the
        suffix.

        Args:
            plan: The inputs of the call.
            api: The mypy API, for `_JoinResultMerger`.

        Returns:
            True if the two trial results have different names.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> hook = JoinHook(untyped_fallback(), new_plugin(fixture))
            >>> foo = column_ref(fixture, "foo", int_dtype(fixture))
            >>> bar = column_ref(fixture, "bar", str_dtype(fixture))
            >>> qux = column_ref(fixture, "qux", str_dtype(fixture))
            >>> receiver = frame(fixture, [foo.as_instance(), bar.as_instance()])
            >>> no_collision = _JoinPlan(
            ...     receiver, "cross", [foo, bar], [qux], [], [], None,
            ... )
            >>> hook._collides(no_collision, api)
            False
            >>> with_collision = _JoinPlan(
            ...     receiver, "cross", [foo, bar], [bar], [], [], None,
            ... )
            >>> hook._collides(with_collision, api)
            True
        """
        probe_a = _JoinResultMerger(
            plan,
            "\0astrolabe_join_probe_a",
            api,
            self._compatibility,
            self._unchecked,
        ).merged_columns
        probe_b = _JoinResultMerger(
            plan,
            "\0astrolabe_join_probe_b",
            api,
            self._compatibility,
            self._unchecked,
        ).merged_columns
        return [column.name for column in probe_a] != [column.name for column in probe_b]


class JoinWhereHook:
    """Give the result type of `.join_where()`, an inner join that matches rows by a predicate.

    The predicates choose only rows, never columns. So the result has the
    same schema as a cross join, and the hook builds a `_JoinPlan` with no
    keys and uses the cross layout of `_JoinResultMerger`.
    """

    DEFAULT_SUFFIX: Final[str] = "_right"
    PREDICATES: ClassVar[tuple[str, ...]] = ("predicates",)

    def __init__(
        self,
        untyped_fallback: UntypedFallback,
        plugin: Plugin,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Build the rules and the fallback that the hook needs.

        Args:
            untyped_fallback: The fallback that gives the plain frame type when
                the hook cannot type the call, subject to the `allow_unproven`
                setting.
            plugin: The mypy plugin. `_JoinResultMerger` needs a key rule built
                from it, but a merge with no keys never uses it. `RemainderEdit`
                uses it to mint the remainder of the result, and the resolver
                of the predicates uses it to mint their columns.
            unchecked_fallback: The fallback that reports a predicate that the
                checker cannot prove, subject to the `allow_unproven` setting.
        """
        self._untyped = untyped_fallback
        self._unchecked = UncheckedCompatibility()
        self._compatibility = JoinKeyCompatibility(self._unchecked, plugin)
        self._duplicate_check = DuplicateNameCheck("join_where")
        self._remainder_step = JoinRemainderStep(plugin, untyped_fallback, "join_where")
        self._policies = RequestPolicies("join_where", "row", plugin, unchecked_fallback)

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the result type of one `.join_where()` call.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            Every column of both sides, with the suffix on each name that
            collides. The plain `pl.DataFrame` or `pl.LazyFrame` if a frame is
            not typed, `suffix` is not a literal string, or the result has a
            duplicate name.
        """
        result = self._merged_result(ctx)
        if isinstance(result, Refusal):
            frame = Frame.from_proper_type(ctx.type)
            return self._untyped.resolve(
                ctx,
                result.reason,
                "join_where",
                frame,
                _JoinPlan.other_frame(ctx),
            )
        return result

    @explains(Reason.CHECKER_GAP)
    def _merged_result(self, ctx: MethodContext) -> MypyType:
        """Build the result frame, or give up at the first step that fails.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The joined frame, or a refusal with its reason. The reason is a
            checker gap if a frame is not typed. It is a dynamic name if
            `suffix` is not a literal string, and `Reason.CALLER_KEY` if a
            predicate holds a declared expression. It is `Reason.REPORTED` after
            an error: an alias in a predicate, a duplicate name in the result,
            or a remainder that makes the renames depend on the caller.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        other = not_none(_JoinPlan.other_frame(ctx))
        left_known = not_none(frame.known_columns)
        right_known = not_none(other.known_columns)
        tree = JoinKeyTree("join_where")
        not_falsy(tree.holds_no_alias(ctx, self.PREDICATES), reason=Reason.REPORTED)
        not_falsy(not tree.holds_declared(ctx, self.PREDICATES), reason=Reason.CALLER_KEY)
        suffix = not_none(
            CallArgs(ctx).literal_str("suffix", default=self.DEFAULT_SUFFIX),
            reason=Reason.DYNAMIC_NAME,
        )
        plan = _JoinPlan(
            frame,
            "cross",
            left_known.columns,
            right_known.columns,
            [],
            [],
            None,
            left_known.rest,
            right_known.rest,
        )
        merger = _JoinResultMerger(plan, suffix, ctx.api, self._compatibility, self._unchecked)
        checked_columns = merger.merged_columns
        names = [column.name for column in checked_columns]
        not_falsy(not self._duplicate_check.fails(ctx, names), reason=Reason.REPORTED)
        inventory = _MergeInventory(plan, ctx.api, self._compatibility, self._unchecked)
        outcome = not_none(
            self._remainder_step.decided(ctx, inventory.sides(suffix)),
            reason=Reason.REPORTED,
        )
        rebuilt = frame.rebuilt(checked_columns, ctx.api, rest=outcome.slot)
        self._check_predicates(ctx, not_none(rebuilt.known_columns))
        return rebuilt.as_instance()

    def _check_predicates(self, ctx: MethodContext, joined: KnownColumns) -> None:
        """Report each predicate that does not give `Boolean` over the columns of the join.

        A predicate reads the columns of both frames, a column of the other
        frame under the name that the join gives it, with the suffix where
        the names collide.

        Args:
            ctx: The mypy context of the method call.
            joined: The columns of the result, and its remainder.
        """
        resolver = RequestResolver(ctx, joined, self._policies)
        args = CallArgs(ctx)
        nodes = args.node_list("predicates")
        types = args.type_list("predicates")
        for node, typ in zip(nodes, types, strict=True):
            request = ExpressionRequest(node, typ, None)
            BooleanPredicate("join_where").check(ctx, resolver, request)
