"""Check the column names that `.write_parquet(partition_by=...)` names.

`write_parquet` writes a frame to a file and returns None, so the checker
has no result type to compute. The one thing it can check is the
`partition_by=` argument, which names columns of the frame. A name that
the frame does not have fails only when the program runs, after the work
that built the frame. This hook finds it before the code runs.

The other `write_*` and `sink_*` methods take no argument that names a
column, and polars declares their return types correctly. So they need
no hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.errors import UnknownColumnError
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from mypy.plugin import MethodContext
    from mypy.types import Type as MypyType


class WriteParquetPartitionByHook:
    """Report each `partition_by=` name that the receiver frame does not declare.

    It is a mypy method hook in the plugin layer. It reads the declared
    columns of the receiver through `Frame`, and the argument through
    `CallArgs`. It reports an `UnknownColumnError` for each name that is
    not one of those columns.

    The hook never changes the result type. So if it cannot check a call,
    it does nothing, and the check is lost but no wrong type appears. This
    is the case if the receiver is not a typed frame, or if
    `partition_by=` is not a literal string or a literal list of strings.
    """

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Check the call, and give back the return type that polars declares.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The return type that polars declares, always unchanged.
        """
        self._checked(ctx)
        return ctx.default_return_type

    @propagates(None)
    def _checked(self, ctx: MethodContext) -> None:
        """Report an `UnknownColumnError` for each `partition_by=` name that is not declared.

        If a step cannot read the receiver or the argument, the method
        stops and reports nothing. A declared column can have no literal
        name, and such a column cannot match a name in the call.

        Args:
            ctx: The context that mypy gives for the call.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        columns = not_none(frame.columns)
        names = not_none(CallArgs(ctx).literal_str_list("partition_by"))
        known = {column.name for column in columns if column.name is not None}
        for name in names:
            if name not in known:
                UnknownColumnError(name).fail(ctx)
