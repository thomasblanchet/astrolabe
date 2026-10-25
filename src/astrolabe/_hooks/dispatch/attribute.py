"""Register the hooks that mypy calls when code reads an attribute.

`.schema` on a typed frame gives the schema that the type of the frame
declares. The recorder takes the namespace attributes of an expression,
such as `.str` (`RecordingHooks`).

mypy finds an attribute hook by the class that defines the attribute,
not by the class of the receiver.
"""

from __future__ import annotations


from mypy.plugin import AttributeContext, Plugin
from mypy.types import Type as MypyType

from astrolabe._hooks.dispatch.generic import HookDispatcher, Target
from astrolabe._hooks.schema_attribute import SchemaAttributeHook
from astrolabe.frame import _TypedFrameBase


class AttributeHooks:
    """Build the attribute hooks, and register each one by the full name of its attribute.

    The plugin asks this class for the hook of an attribute when mypy
    reads that attribute.
    """

    def __init__(self, plugin: Plugin) -> None:
        """Build every attribute hook, and register it in a new dispatcher.

        Args:
            plugin: The mypy plugin. The `.schema` hook uses it to look
                up `polars.schema.Schema` by full name.
        """
        self._dispatcher = HookDispatcher[AttributeContext, MypyType]()
        self._dispatcher.register(
            Target.of(_TypedFrameBase, "schema"),
            SchemaAttributeHook(plugin),
        )

    @property
    def dispatcher(self) -> HookDispatcher[AttributeContext, MypyType]:
        """The dispatcher that holds every registered attribute hook."""
        return self._dispatcher
