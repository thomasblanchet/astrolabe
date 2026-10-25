"""Compare data types that can be `Unchecked`, and give a column the data type `Unchecked`.

`Unchecked` is the data type of a column whose name the checker knows and whose data type it
does not. It lets the checker keep such a column in the schema, instead of falling back to the
plain frame, which loses every column. It must then match any data type: code that compares two
data types, as a join or a concat does, cannot reject a column whose data type is not known.

`UncheckedCompatibility` compares and merges two data types with that rule. `UncheckedFallback`
gives a column the data type `Unchecked`, and first reports an error unless the
`allow_unproven` setting lets the reason pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TYPE_CHECKING

from mypy.types import get_proper_type, Instance, TupleType
from mypy.types import Type as MypyType

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.unproven_gate import UnprovenGate
from astrolabe._hooks.errors import ImplicitlyUncheckedError
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe.frame import Unchecked as _UncheckedMarker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.plugin import FunctionContext, MethodContext
    from mypy.types import ProperType

    from astrolabe._hooks.unproven import Reason, UnprovenPolicy
    from astrolabe._hooks.domain.frame import PluginLookup


class UncheckedCompatibility:
    """Compare two data types where `Unchecked` matches any data type, and merge them.

    Code that checks that two columns agree, for example the keys of a join or the columns of a
    concat, asks this class instead of testing for equality. `Unchecked` beside a known data type
    matches, and the merged data type is the known one. The rule also applies inside a nested
    data type, so `List[Unchecked]` matches `List[Int64]`. Code that picks a rule by the family of
    a data type also asks this class first, so that an `Unchecked` operand never reaches a rule
    that needs a real data type.

    It holds no state and reads only mypy types.

    Example:
        >>> fixture = new_type_fixture()
        >>> unchecked = UncheckedCompatibility()
        >>> unchecked.matches(unchecked_instance(fixture))
        True
        >>> unchecked.matches(int_dtype(fixture))
        False
    """

    TARGET: Final[Target] = Target.of(_UncheckedMarker)

    def matches(self, dtype: MypyType) -> bool:
        """Tell whether a data type is `Unchecked`.

        Args:
            dtype: The data type to test.

        Returns:
            True if `dtype` is an `Unchecked` instance.

        Example: see class docstring.
        """
        return self.TARGET.matches(get_proper_type(dtype))

    def resolved(self, a: MypyType, b: MypyType) -> MypyType | None:
        """Merge two data types that match, and keep the known side over `Unchecked`.

        The rule applies at every level of a nested data type. `List[Unchecked]` beside
        `List[Int64]` gives `List[Int64]`. The same holds for the fields of a `Struct` and for the
        type arguments of any other marker, such as `Datetime[Unit, Zone]`. Two literal type
        arguments that differ, for example two field names, do not match.

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            `a` if the two are equal. The other side if one side is `Unchecked`. If both are
            instances of the same class, or tuples of the same length, the merge of each pair
            of type arguments or items. None if the two do not match.

        Example:
            >>> fixture = new_type_fixture()
            >>> unchecked = UncheckedCompatibility()
            >>> unchecked.resolved(unchecked_instance(fixture), int_dtype(fixture))
            polars.datatypes.classes.Int64
            >>> unchecked.resolved(int_dtype(fixture), str_dtype(fixture))
            >>> any_typed, int_typed = list_dtype_pair(
            ...     fixture, unchecked_instance(fixture), int_dtype(fixture)
            ... )
            >>> unchecked.resolved(any_typed, int_typed)
            astrolabe.frame.List[polars.datatypes.classes.Int64]
            >>> a_column = column_ref(fixture, "a", int_dtype(fixture))
            >>> b_column = column_ref(fixture, "b", int_dtype(fixture))
            >>> one_field = struct_instance(fixture, a_column)
            >>> two_fields = struct_instance(fixture, a_column, b_column)
            >>> one_field.args[0]  # the fields of a struct, as a tuple type
            tuple[astrolabe.frame.Column[Literal['a'], polars.datatypes.classes.Int64]]
            >>> unchecked.resolved(one_field.args[0], two_fields.args[0]) is None
            True
        """
        return _UncheckedMerge(self, keep_known=True).merged(a, b)

    def joined(self, a: MypyType, b: MypyType) -> MypyType | None:
        """Merge two data types into their supertype, where an `Unchecked` side gives `Unchecked`.

        An operation that widens two columns to the supertype of their data types, as `update`
        or a relaxed concat does, asks this class first. The supertype depends on both sides:
        `String` beside `Int64` gives `String`, and `Int64` beside `Int64` gives `Int64`. So a
        side whose data type is not known leaves the supertype unknown. The rule applies at
        every level of a nested data type, so `List[Unchecked]` beside `List[Int64]` gives
        `List[Unchecked]`.

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            `a` if the two are equal. The `Unchecked` side if one side is `Unchecked`. If both
            are instances of the same class, or tuples of the same length, the merge of each
            pair of type arguments or items. None if the two do not match, which leaves the
            supertype to rules that read known data types.

        Example:
            >>> fixture = new_type_fixture()
            >>> unchecked = UncheckedCompatibility()
            >>> unchecked.joined(int_dtype(fixture), unchecked_instance(fixture))
            astrolabe.frame.Unchecked
            >>> any_typed, int_typed = list_dtype_pair(
            ...     fixture, unchecked_instance(fixture), int_dtype(fixture)
            ... )
            >>> unchecked.joined(int_typed, any_typed)
            astrolabe.frame.List[astrolabe.frame.Unchecked]
            >>> unchecked.joined(int_dtype(fixture), str_dtype(fixture)) is None
            True
        """
        return _UncheckedMerge(self, keep_known=False).merged(a, b)

    def compatible(self, a: MypyType, b: MypyType) -> bool:
        """Tell whether two data types match, where `Unchecked` matches any data type.

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            True if `resolved` gives a merge of the two.

        Example:
            >>> fixture = new_type_fixture()
            >>> unchecked = UncheckedCompatibility()
            >>> unchecked.compatible(unchecked_instance(fixture), int_dtype(fixture))
            True
            >>> unchecked.compatible(int_dtype(fixture), str_dtype(fixture))
            False
        """
        return self.resolved(a, b) is not None


@dataclass(frozen=True)
class _UncheckedMerge:
    """Merge two data types where a side can be `Unchecked`, by one rule for that side.

    A check that two columns agree keeps the known side: a column of another data type makes
    polars raise, so where the code runs, the data type is the known one. A supertype keeps
    `Unchecked`, because the unknown side decides it. Both merges walk a nested data type in
    the same way, the arguments of a marker and the fields of a `Struct`, and differ only in
    that rule.

    Example:
        >>> fixture = new_type_fixture()
        >>> merge = _UncheckedMerge(UncheckedCompatibility(), keep_known=True)
        >>> merge.merged(unchecked_instance(fixture), int_dtype(fixture))
        polars.datatypes.classes.Int64
    """

    compatibility: UncheckedCompatibility
    keep_known: bool

    def merged(self, a: MypyType, b: MypyType) -> MypyType | None:
        """Merge two data types.

        Args:
            a: The first data type.
            b: The second data type.

        Returns:
            `a` if the two are equal, the side that the rule keeps if one side is `Unchecked`,
            the merge of each pair of type arguments or items, or None if the two do not match.

        Example:
            >>> fixture = new_type_fixture()
            >>> merge = _UncheckedMerge(UncheckedCompatibility(), keep_known=False)
            >>> merge.merged(unchecked_instance(fixture), int_dtype(fixture))
            astrolabe.frame.Unchecked
        """
        if get_proper_type(a) == get_proper_type(b):
            return a
        if self.compatibility.matches(a):
            return b if self.keep_known else a
        if self.compatibility.matches(b):
            return a if self.keep_known else b
        proper_a = get_proper_type(a)
        proper_b = get_proper_type(b)
        instance_result = self._instance_pair(proper_a, proper_b)
        if instance_result is not None:
            return instance_result
        return self._tuple_pair(proper_a, proper_b)

    def _instance_pair(self, a: ProperType, b: ProperType) -> MypyType | None:
        """Merge two instances of the same class, one type argument at a time.

        Args:
            a: The first data type. It is not equal to `b`, and it is not `Unchecked`.
            b: The second data type.

        Returns:
            `a` with the merged type arguments. None if the two are not instances of the same
            class with the same number of type arguments, or if one pair does not match.

        Example:
            >>> fixture = new_type_fixture()
            >>> merge = _UncheckedMerge(UncheckedCompatibility(), keep_known=True)
            >>> merge._instance_pair(int_dtype(fixture), str_dtype(fixture)) is None
            True
        """
        if not (isinstance(a, Instance) and isinstance(b, Instance)):
            return None
        if a.type != b.type or len(a.args) != len(b.args):
            return None
        merged_args = self._pairs(a.args, b.args)
        if merged_args is None:
            return None
        return a.copy_modified(args=merged_args)

    def _tuple_pair(self, a: ProperType, b: ProperType) -> MypyType | None:
        """Merge two tuple types of the same length, one item at a time.

        A tuple type holds the fields of a `Struct`, so this is how two structs merge.

        Args:
            a: The first data type. It is not equal to `b`, and it is not `Unchecked`.
            b: The second data type.

        Returns:
            `a` with the merged items. None if the two are not tuples of the same length, or if
            one pair of items does not match.

        Example:
            >>> fixture = new_type_fixture()
            >>> merge = _UncheckedMerge(UncheckedCompatibility(), keep_known=True)
            >>> merge._tuple_pair(int_dtype(fixture), str_dtype(fixture)) is None
            True
        """
        if not (isinstance(a, TupleType) and isinstance(b, TupleType)):
            return None
        if len(a.items) != len(b.items):
            return None
        merged_items = self._pairs(a.items, b.items)
        if merged_items is None:
            return None
        return a.copy_modified(items=merged_items)

    @propagates(None)
    def _pairs(
        self,
        a_items: Sequence[MypyType],
        b_items: Sequence[MypyType],
    ) -> list[MypyType] | None:
        """Merge each pair of two lists of types of the same length.

        Args:
            a_items: The type arguments or the tuple items of the first data type.
            b_items: Those of the second data type, in the same order.

        Returns:
            The merged types in order, or None if one pair does not match.

        Example:
            >>> fixture = new_type_fixture()
            >>> merge = _UncheckedMerge(UncheckedCompatibility(), keep_known=True)
            >>> merge._pairs([int_dtype(fixture)], [unchecked_instance(fixture)])
            [polars.datatypes.classes.Int64]
        """
        result: list[MypyType] = []
        for a_item, b_item in zip(a_items, b_items, strict=True):
            result.append(not_none(self.merged(a_item, b_item)))
        return result


class UncheckedFallback(UnprovenGate):
    """Give one column the data type `Unchecked`, and report it unless the setting lets it pass.

    Sometimes a hook knows the name of an output column but cannot prove its data type, for
    example after a method that runs a Python function. To fall back to the plain frame then
    loses every column of the frame. This class keeps the column under its name with the data
    type `Unchecked`, and the rest of the schema stays typed. It is an `UnprovenGate`: before
    it gives the column, it reports `ImplicitlyUncheckedError` unless the `allow_unproven`
    setting lets the reason pass. The report happens once, at the operation where the unknown
    data type first appears.

    Example:
        >>> from astrolabe._hooks.errors import UnnamedReadError
        >>> from astrolabe._hooks.unproven import UnprovenPolicy
        >>> fallback = UncheckedFallback(UnprovenPolicy(), new_plugin(new_type_fixture()))
        >>> fallback.report(None, UnnamedReadError("select")) is None
        True
    """

    _UNCHECKED_TARGET: Final[Target] = Target.of(_UncheckedMarker)

    def __init__(self, policy: UnprovenPolicy, plugin: PluginLookup) -> None:
        """Keep the setting for the gate, and the plugin that finds the `Unchecked` class.

        Args:
            policy: The `allow_unproven` setting, already read from the configuration.
            plugin: The mypy plugin, which finds the `Unchecked` class.
        """
        super().__init__(policy)
        self._type_info = FreshTypeInfo(plugin)

    def substituted(
        self,
        ctx: MethodContext | FunctionContext,
        reason: Reason,
        operation: str,
        name: str,
    ) -> MypyType | None:
        """Report the column unless the setting lets it pass, then give the type `Unchecked`.

        Args:
            ctx: The mypy context of the operation.
            reason: Why the data type of the column is not proven.
            operation: The name of the operation.
            name: The name of the column, which the error shows.

        Returns:
            A new `Unchecked` instance, or None if the plugin cannot find the class.
        """
        self.reported(ctx, reason, operation, name)
        return self.minted()

    def minted(self) -> MypyType | None:
        """Give a new `Unchecked` instance, and report nothing.

        Code that has already reported a hard error for a column uses this. It keeps the name of
        the column with no second error.

        Returns:
            A new `Unchecked` instance, or None if the plugin cannot find the class.
        """
        info = self._type_info.resolved(self._UNCHECKED_TARGET.fullname)
        if info is None:  # pragma: no cover
            # Unchecked lives in astrolabe.frame. The plugin adds that module to the
            # dependencies of every build, so the lookup always finds it.
            return None
        return Instance(info, [])

    def reported(
        self,
        ctx: MethodContext | FunctionContext,
        reason: Reason,
        operation: str,
        name: str,
    ) -> None:
        """Report one `Unchecked` column unless the setting lets its reason pass.

        Use this when the code already has the `Unchecked` data type and needs only the error.
        For example, the expression engine can already give `Unchecked` as its answer. An
        expression method that runs a Python function also reports here, at the call that
        builds the expression.

        Args:
            ctx: The mypy context of the operation.
            reason: Why the data type of the column is not proven.
            operation: The name of the operation.
            name: The name of the column, which the error shows.
        """
        self.report(ctx, ImplicitlyUncheckedError(reason, operation, name))
