"""Read the key columns and the columns of the frame from the type of a typed group-by.

`df.group_by(...)` gives a `TypedGroupBy` or a `TypedLazyGroupBy`. Its type holds three things
that the next call needs: the key columns, the columns of the frame at the time of the
`group_by`, and the remainder of that frame. The remainder is the set of columns that the type of
an open frame does not name. Every hook on a group-by reads the same three, so one module reads
them. A group-by over windows of the rows, as `rolling(...)` gives, holds two more: its index
column, and the kind of its windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar

from mypy.types import Instance, LiteralType, NoneType, TupleType, get_proper_type

from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from mypy.types import Type as MypyType


@dataclass(frozen=True)
class GroupByTuples:
    """The raw type arguments of a typed group-by, before they become columns.

    A typed group-by holds its key columns and the columns of its frame as two tuple types of
    `Column[...]` items. A tuple keeps the order of the keys, which is the order of the first
    columns of the result. This class reads the three type arguments. `GroupByReceiver` then
    turns the tuples into columns.

    Attributes:
        keys: The tuple type of the key columns.
        original: The tuple type of the columns of the frame at the time of the `group_by`.
        rest: The remainder of that frame, or None for a closed frame.
        receiver_fullname: The mypy full name of the group-by class, which tells an eager
            group-by from a lazy one.
        index: The tuple type of the index column of a group-by over windows, or None.
        windows: The kind of windows of a group-by over windows, or the empty text.
    """

    WINDOW_ARGS: ClassVar[int] = 5

    keys: TupleType
    original: TupleType
    rest: MypyType | None
    receiver_fullname: str
    index: TupleType | None = None
    windows: str = ""

    @classmethod
    @propagates(None)
    def from_receiver_type(cls, receiver_type: MypyType) -> GroupByTuples | None:
        """Read the type arguments of a typed group-by.

        Args:
            receiver_type: The type of the group-by that receives the call.

        Returns:
            The type arguments, or None if the type is not an instance with three type
            arguments whose first two are tuple types.
        """
        receiver = not_none(cls.receiver_instance(receiver_type))
        keys_arg = not_none(cls._tuple_arg(receiver, 0))
        original_arg = not_none(cls._tuple_arg(receiver, 1))
        rest = cls.rest_arg(receiver.args[2])
        tuples = cls(keys_arg, original_arg, rest, receiver.type.fullname)
        if len(receiver.args) == cls.WINDOW_ARGS:
            windows = get_proper_type(receiver.args[4])
            text = not_none(windows if isinstance(windows, LiteralType) else None).value
            index = not_none(cls._tuple_arg(receiver, 3))
            return replace(tuples, index=index, windows=str(text))
        return tuples

    @staticmethod
    def rest_arg(argument: MypyType) -> MypyType | None:
        """Read the third type argument, which holds the remainder.

        Args:
            argument: The third type argument.

        Returns:
            The remainder, or None if the argument is the `None` type of a closed frame.

        Example:
            >>> GroupByTuples.rest_arg(Frame.NO_REST) is None
            True
        """
        return None if isinstance(get_proper_type(argument), NoneType) else argument

    @staticmethod
    def receiver_instance(receiver_type: MypyType) -> Instance | None:
        """Give the type of the group-by as an instance with three type arguments.

        Args:
            receiver_type: The type of the group-by that receives the call.

        Returns:
            The instance, or None if the type is not an instance with three type arguments.
        """
        receiver = get_proper_type(receiver_type)
        if not isinstance(receiver, Instance):  # pragma: no cover
            # The plugin registers the group-by hooks only on TypedGroupBy and TypedLazyGroupBy,
            # so the receiver is always an instance of one of them.
            return None
        if len(receiver.args) not in {3, GroupByTuples.WINDOW_ARGS}:  # pragma: no cover
            # TypedGroupBy and TypedLazyGroupBy each have three type parameters, and the
            # group-by classes over windows five, so an instance always has three or five.
            return None
        return receiver

    @staticmethod
    def _tuple_arg(receiver: Instance, index: int) -> TupleType | None:
        """Give one type argument of the group-by if it is a tuple type.

        Args:
            receiver: The group-by instance.
            index: The position of the type argument.

        Returns:
            The tuple type, or None if the type argument is not a tuple type.
        """
        arg = get_proper_type(receiver.args[index])
        if not isinstance(arg, TupleType):  # pragma: no cover
            # The group_by hook builds every typed group-by with two tuple type arguments. When
            # it cannot, it gives the plain polars GroupBy or LazyGroupBy, which has no hook
            # here. An annotation that a user writes passes `GroupByTypeHook`, which refuses
            # one without two tuples. So the first two type arguments are always tuples.
            return None
        return arg


@dataclass(frozen=True)
class GroupByReceiver:
    """The key columns and the frame columns of a typed group-by, as columns.

    A hook on a group-by, such as `.agg(...)` or `.sum()`, reads its receiver with this class. It
    gets the keys, which come first in the result, and the columns that its expressions read.

    Attributes:
        key_columns: The key columns, in the order of the `group_by` call.
        original_columns: The declared columns of the frame at the time of the `group_by`.
        rest: The remainder of that frame, or None for a closed frame.
        receiver_fullname: The mypy full name of the group-by class, which tells an eager
            group-by from a lazy one.
        index_columns: The index column of a group-by over windows, or none.
        windows: The kind of windows of a group-by over windows, or the empty text.
    """

    key_columns: list[ColumnRef]
    original_columns: list[ColumnRef]
    rest: MypyType | None
    receiver_fullname: str
    index_columns: list[ColumnRef] = field(default_factory=list)
    windows: str = ""

    @classmethod
    @propagates(None)
    def from_receiver_type(cls, receiver_type: MypyType) -> GroupByReceiver | None:
        """Read the key columns and the frame columns from the type of a typed group-by.

        Args:
            receiver_type: The type of the group-by that receives the call.

        Returns:
            The columns, or None if the type is not a `TypedGroupBy` or `TypedLazyGroupBy`
            whose tuples hold `Column[...]` items.
        """
        tuples = not_none(GroupByTuples.from_receiver_type(receiver_type))
        key_columns = not_none(ColumnRef.from_tuple(tuples.keys))
        original_columns = not_none(ColumnRef.from_tuple(tuples.original))
        index = [] if tuples.index is None else not_none(ColumnRef.from_tuple(tuples.index))
        return cls(
            key_columns,
            original_columns,
            tuples.rest,
            tuples.receiver_fullname,
            index,
            tuples.windows,
        )
