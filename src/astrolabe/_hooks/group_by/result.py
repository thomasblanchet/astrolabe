"""Build the typed frame that a call on a group-by gives, such as `.agg(...)` or `.sum()`.

Each such call gives a frame whose columns are the keys and then the outputs of the call. An
eager group-by gives a `TypedDataFrame`, and a lazy one gives a `TypedLazyFrame`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import TypeInfo
from mypy.types import Instance

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.frame import Frame
from astrolabe.frame import (
    TypedDataFrame,
    TypedDynamicGroupBy,
    TypedGroupBy,
    TypedLazyDynamicGroupBy,
    TypedLazyFrame,
    TypedLazyGroupBy,
    TypedLazyRollingGroupBy,
    TypedRollingGroupBy,
)

if TYPE_CHECKING:
    from mypy.plugin import Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.column import ColumnRef


@dataclass(frozen=True)
class GroupByResult:
    """Build the typed frame that one call on a group-by gives.

    Every hook on a group-by ends by building its result frame. The hook gives the columns and
    the remainder. This class picks the frame class from the kind of group-by and builds the
    type. There is no typed frame to copy from here, so the class looks up the frame class
    through the plugin.

    Attributes:
        plugin: The mypy plugin, which finds the class of the result frame.
    """

    FRAMES: ClassVar[dict[Target, Target]] = {
        Target.of(TypedGroupBy): Target.of(TypedDataFrame),
        Target.of(TypedLazyGroupBy): Target.of(TypedLazyFrame),
        Target.of(TypedRollingGroupBy): Target.of(TypedDataFrame),
        Target.of(TypedDynamicGroupBy): Target.of(TypedDataFrame),
        Target.of(TypedLazyRollingGroupBy): Target.of(TypedLazyFrame),
        Target.of(TypedLazyDynamicGroupBy): Target.of(TypedLazyFrame),
    }

    plugin: Plugin

    def frame(
        self,
        receiver_fullname: str,
        columns: list[ColumnRef],
        rest: MypyType | None,
        api: object,
    ) -> MypyType | None:
        """Build the result frame of one call.

        The frame class matches the kind of group-by: `TypedDataFrame` for an eager group-by,
        and `TypedLazyFrame` for a lazy one.

        Args:
            receiver_fullname: The mypy full name of the group-by class.
            columns: The keys, then the outputs of the call.
            rest: The remainder of the result, or None for a closed result.
            api: The mypy plugin API of the type checking phase, which builds the `TypedDict`
                of the columns.

        Returns:
            The frame instance, or None if the name is not a typed group-by or the plugin
            cannot find the frame class.
        """
        target = self.FRAMES.get(Target(receiver_fullname))
        if target is None:  # pragma: no cover
            # The plugin registers the group-by hooks only on the two classes in FRAMES.
            return None
        found = self.plugin.lookup_fully_qualified(target.fullname)
        if found is None or not isinstance(found.node, TypeInfo):  # pragma: no cover
            # TypedDataFrame and TypedLazyFrame live in the module of the group-by classes, so
            # the lookup always finds them.
            return None
        result = Frame.object_typeddict(columns, api)
        return Instance(found.node, [result, Frame.NO_REST if rest is None else rest])
