"""Keep the column types of a typed frame through `.lazy()`.

polars types `DataFrame.lazy()` as a plain `LazyFrame`, so without this hook
the checker loses every column at the first `.lazy()` call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType


class LazyHook:
    """Give the `TypedLazyFrame` with the same columns as the receiver of `.lazy()`.

    A `TypedDataFrame` is an eager frame whose type names its columns and
    their data types. `.lazy()` turns it into a lazy query over the same
    data. The call adds, deletes and retypes no column, and it takes no
    argument that names a column. So the hook has nothing to read from the
    call. It copies the schema of the receiver, including its remainder
    (the columns of an open frame that its type does not name), into the
    lazy kind of frame. If the receiver is a typed frame, the answer is
    always typed.

    It is a mypy method hook in the plugin layer. It reads the receiver
    through `Frame`, which also builds the frame of the other kind.
    """

    def __init__(self, plugin: Plugin) -> None:
        """Keep the plugin, to look up the class of the lazy frame.

        Args:
            plugin: The mypy plugin, which finds the `TypeInfo` of
                `TypedLazyFrame`.
        """
        self._plugin = plugin

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.lazy()` call.

        Args:
            ctx: The context that mypy gives for the `.lazy()` call.

        Returns:
            A `TypedLazyFrame` with the schema of the receiver. If that
            type cannot be built, the plain `pl.LazyFrame` that polars
            declares.
        """
        result = self._result(ctx)
        if result is None:  # pragma: no cover
            # The plugin registers this hook only on `TypedDataFrame.lazy`,
            # so the receiver is always a typed frame. `TypedLazyFrame` is
            # declared in the same module as `TypedDataFrame`, so mypy has
            # always loaded it. Neither step of `_result` can fail.
            return ctx.default_return_type
        return result

    @propagates(None)
    def _result(self, ctx: MethodContext) -> MypyType | None:
        """Build the `TypedLazyFrame` from the receiver, or give None if a step fails.

        Args:
            ctx: The context that mypy gives for the `.lazy()` call.

        Returns:
            A `TypedLazyFrame` with the schema of the receiver, or None
            if the receiver is not a typed frame or mypy cannot find
            `TypedLazyFrame`.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        lazy_frame = not_none(frame.other_kind(self._plugin))
        return lazy_frame.as_instance()
