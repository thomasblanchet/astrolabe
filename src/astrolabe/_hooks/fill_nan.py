"""Type `df.fill_nan(value)` on a typed frame, which fills the float columns and can retype them.

polars declares that the call returns a plain frame, so without this hook
the checker loses every column of a typed frame at this call.

polars runs `df.fill_nan(value)` as
`df.with_columns(cs.float().fill_nan(value))`. So each float column takes
the answer of the expression form of `fill_nan`, and every other column
keeps its type. A text value turns a float column into `String`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import CallExpr
from mypy.types import AnyType, get_proper_type

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.expression_requests import ExpressionRequest
from astrolabe._hooks.recording.recorder import RecordedShortForm, ShortForm
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.with_columns import WithColumnsHook

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


class FrameFillNanHook:
    """Give `df.fill_nan(value)` on a typed frame its result type.

    The hook records the expression that polars runs, with the value of the
    call, and types the call as that `with_columns`. The rules of the
    expression decide each float column, and a selection over an open frame
    changes the float columns of the remainder too. A value of the type
    `Any` is an argument that is not typed. Where the hook cannot type the
    call, it falls back to the plain frame, subject to the `allow_unproven`
    setting.

    It is a mypy method hook in the plugin layer.
    """

    OPERATION: ClassVar[str] = "fill_nan"
    FORM: ClassVar[ShortForm] = ShortForm("cs.float().fill_nan()", ("value",))

    def __init__(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Keep the plugin and the hook of `with_columns` that types the call.

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

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.fill_nan(...)` call on a frame.

        Args:
            ctx: The context that mypy gives for `.fill_nan(...)`.

        Returns:
            The typed frame after the fill, or the plain frame where the
            hook reported an error or cannot type the call.
        """
        value = get_proper_type(CallArgs(ctx).node_type("value"))
        if isinstance(value, AnyType):
            # The value decides each filled column, and its type shows nothing.
            frame = Frame.from_proper_type(ctx.type)
            return self._untyped.resolve(ctx, Reason.UNTYPED_ARGUMENT, self.OPERATION, frame)
        typ = RecordedShortForm(self._plugin, self.FORM).recorded(ctx)
        if typ is None:
            # Polars refuses to build the expression, and the recorder reported it.
            return ctx.default_return_type
        request = ExpressionRequest(checked_cast(ctx.context, CallExpr), typ, None)
        return self._with_columns.typed(ctx, [request])
