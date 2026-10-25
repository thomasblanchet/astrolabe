"""Look up the mypy class record of one of the plugin's own classes by its full name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import TypeInfo as MypyTypeInfo

if TYPE_CHECKING:
    from astrolabe._hooks.domain.frame import PluginLookup


class FreshTypeInfo:
    """Look up the mypy class record (a `TypeInfo`) of a class by its full name.

    A hook that mints a type builds a new mypy `Instance` of one of the
    plugin's own marker classes, for example `TypedExpr`. To mint means to
    build a type inside a hook. An `Instance` needs the `TypeInfo` of its
    class, and the hook often has no type in hand to take it from. This
    class does that lookup once, for every hook that has the need.

    It needs only the `lookup_fully_qualified` method of the plugin, so it
    takes the small `PluginLookup` protocol and not the whole mypy `Plugin`.
    """

    def __init__(self, plugin: PluginLookup) -> None:
        """Keep the plugin that answers each lookup.

        Args:
            plugin: The mypy plugin, or any object that can look up a full name.
        """
        self._plugin = plugin

    def resolved(self, fullname: str) -> MypyTypeInfo | None:
        """Find the `TypeInfo` of one class.

        Args:
            fullname: The full name of the class, for example
                `astrolabe.expr.TypedExpr`.

        Returns:
            The `TypeInfo` of the class. None if the name is not found or does
            not name a class.
        """
        info = self._plugin.lookup_fully_qualified(fullname)
        if info is None or not isinstance(info.node, MypyTypeInfo):  # pragma: no cover
            # Code asks only for the plugin's own marker classes. The
            # plugin adds their modules to every mypy build, so the lookup
            # always finds them.
            return None
        return info.node
