"""The hook for `.collect()`, which turns a typed lazy frame into a typed eager frame.

Without this hook, mypy types the result of `.collect()` on a
`TypedLazyFrame` as a plain `pl.DataFrame`, and every column that the lazy
frame declared is lost.
"""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

import polars as pl
from mypy.types import get_proper_type

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.propagate import not_falsy, not_none, propagates

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType


class CollectHook:
    """Keep the schema of a typed lazy frame when the code collects it.

    A `TypedLazyFrame` and a `TypedDataFrame` hold the same columns. Only the
    kind of frame differs. The hook reads the schema of the receiver and gives
    a `TypedDataFrame` with the same declared columns and the same remainder
    (the columns of an open frame that its type does not name).

    `pl.LazyFrame.collect` has two overloads. With `background=True` it gives
    an `InProcessQuery`, which is not a frame at all. With the default
    `background=False` it gives a `DataFrame`. mypy picks the overload before
    the hook runs, so the hook acts only when that pick is a plain
    `DataFrame`. In every other case, it gives back the type that mypy chose.
    This includes the `[call-overload]` error that mypy reports when
    `background` is not a literal.

    The hook sits in the plugin layer. It reads the receiver through `Frame`
    and needs no rule from the engine.
    """

    _DATAFRAME_TARGET: Final[Target] = Target.of(pl.DataFrame)

    def __init__(self, plugin: Plugin) -> None:
        """Keep the plugin, which the hook needs to look up `TypedDataFrame`.

        Args:
            plugin: The mypy plugin. The hook looks up the class
                `TypedDataFrame` through it.
        """
        self._plugin = plugin

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the result type of one `.collect()` call.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            A `TypedDataFrame` with the schema of the receiver. If mypy did not
            pick the `DataFrame` overload, or the receiver is not a typed frame,
            the type that mypy chose, unchanged.
        """
        result = self._result(ctx)
        if result is None:
            return ctx.default_return_type
        return result

    @propagates(None)
    def _result(self, ctx: MethodContext) -> MypyType | None:
        """Build the `TypedDataFrame` result, or give up at the first step that fails.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            A `TypedDataFrame` with the schema of the receiver. None if mypy did
            not pick the `DataFrame` overload, if the receiver is not a typed
            frame, or if the lookup of `TypedDataFrame` fails.
        """
        chosen = get_proper_type(ctx.default_return_type)
        not_falsy(self._DATAFRAME_TARGET.matches(chosen))
        frame = not_none(Frame.from_proper_type(ctx.type))
        data_frame = not_none(frame.other_kind(self._plugin))
        return data_frame.collected().as_instance()
