"""Give `.schema` on a typed frame the type `pl.Schema`, as polars does.

A typed frame answers `.schema` in two ways when the program runs. On an
instance it gives the `pl.Schema` object of the data, as any polars frame
does. On the class, for example `TypedDataFrame[Trade].schema`, it gives
the declared columns. One static return type cannot describe both, so the
typed frame class declares `.schema` as `Any`. Without this hook, a user
who reads `frame.schema` gets `Any` and loses all checks on it.
"""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

import polars as pl
from mypy.nodes import TypeInfo
from mypy.types import Instance, TypedDictType, get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.frame import Frame

if TYPE_CHECKING:
    from mypy.plugin import AttributeContext, Plugin


class SchemaAttributeHook:
    """Give `pl.Schema` for `.schema` on a typed frame instance, and `Any` on the class.

    It is a mypy attribute hook in the plugin layer. It reads only the
    type of the receiver, to tell an instance from the class, and it
    answers the type that polars itself gives for an instance.
    """

    POLARS_SCHEMA_TARGET: Final[Target] = Target.of(pl.Schema)

    def __init__(self, plugin: Plugin) -> None:
        """Keep the plugin, to look up the class `pl.Schema`.

        Args:
            plugin: The mypy plugin, which finds the `TypeInfo` of
                `polars.schema.Schema` by its full name.
        """
        self._plugin = plugin

    def __call__(self, ctx: AttributeContext) -> MypyType:
        """Give the type of one read of `.schema`.

        Args:
            ctx: The context that mypy gives for the attribute read.

        Returns:
            `polars.schema.Schema` on an instance. On the class, or if
            mypy cannot find `pl.Schema`, the declared type `Any`.
        """
        if not self._is_instance_access(ctx.type):
            return ctx.default_attr_type
        info = self._plugin.lookup_fully_qualified(self.POLARS_SCHEMA_TARGET.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # The hook runs only on a typed frame, and the module that
            # declares the typed frames imports polars. So mypy has always
            # loaded `polars.schema.Schema` by the time the hook runs.
            return ctx.default_attr_type
        return Instance(info.node, [])

    def _is_instance_access(self, receiver_type: MypyType) -> bool:
        """Tell a read of `.schema` on an instance from a read on the class.

        The first type argument of a typed frame holds its declared
        columns, and a class derived from a typed frame holds them on its
        typed base. On an instance, mypy has filled it with a `TypedDict` of
        those columns. On the class, it is still a type variable, or it is
        missing.

        Args:
            receiver_type: The type of the object before `.schema`.

        Returns:
            True if the receiver is a frame instance, False if it is the
            frame class or anything else.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> hook = SchemaAttributeHook(None)
            >>> hook._is_instance_access(frame(fixture, [price.as_instance()]).as_instance())
            True
            >>> hook._is_instance_access(bare_frame_instance(fixture))
            False
            >>> hook._is_instance_access(literal_str_type(fixture, "not an instance"))
            False
        """
        frame = Frame.from_proper_type(receiver_type)
        if frame is None or not frame.as_instance().args:
            return False
        columns_arg = get_proper_type(frame.as_instance().args[0])
        return isinstance(columns_arg, TypedDictType)
