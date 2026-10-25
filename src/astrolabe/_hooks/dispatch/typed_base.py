"""Send a method or attribute of a class that a user derives from a typed frame to its typed base.

mypy names a method hook by the class that the call goes through. A user
can derive a class from a typed frame, for example
`class MyFrame(TypedDataFrame[A])`, and a call such as `m.select(...)` on
it then asks for the hook of `mod.MyFrame.select`. The table of hooks
knows only `TypedDataFrame` and `TypedLazyFrame`, so without this step no
hook fires, and the call keeps the return type of the polars stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import TypeInfo

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe.frame import TypedDataFrame, TypedLazyFrame

if TYPE_CHECKING:
    from astrolabe._hooks.domain.frame import PluginLookup


class TypedBaseRoute:
    """Give the full name on the typed base for a method or attribute of a derived class.

    A derived class can define its own method of the same name. Its own
    code then runs, so the hook of the base does not apply, and the route
    gives no name. The route reads the class from mypy once for each full
    name, and keeps the answer, because mypy asks for every call in the
    program.
    """

    _BASES: ClassVar[frozenset[str]] = frozenset({
        Target.of(TypedDataFrame).fullname,
        Target.of(TypedLazyFrame).fullname,
    })

    def __init__(self, plugin: PluginLookup) -> None:
        """Keep the plugin, which finds a class by its full name.

        Args:
            plugin: The mypy plugin.
        """
        self._plugin = plugin
        self._routes: dict[str, str | None] = {}

    def of(self, fullname: str) -> str | None:
        """Give the full name of the same member on the typed base of its class.

        Args:
            fullname: The full name of the member, for example
                `mod.MyFrame.select`.

        Returns:
            The full name on the base, for example
            `astrolabe.frame.TypedDataFrame.select`. None if the class does
            not derive from a typed frame, is a typed frame, or defines the
            member itself.
        """
        if fullname not in self._routes:
            self._routes[fullname] = self._routed(fullname)
        return self._routes[fullname]

    @propagates(None)
    def _routed(self, fullname: str) -> str | None:
        """Find the typed base of the member's class, if the class does not define the member.

        The walk goes up the classes of the derived class, and the first
        class that is a typed base or defines the member decides.

        Args:
            fullname: The full name of the member.

        Returns:
            The full name of the member on the first typed base, or None
            where a class before that base defines the member, or where
            no class is a typed base.
        """
        owner, _, member = fullname.rpartition(".")
        node = not_none(self._plugin.lookup_fully_qualified(owner))
        info = not_none(node.node if isinstance(node.node, TypeInfo) else None)
        not_falsy(member not in info.names)
        deciding = (klass for klass in info.mro[1:] if self._decides(klass, member))
        first = not_none(next(deciding, None))
        not_falsy(first.fullname in self._BASES)
        return f"{first.fullname}.{member}"

    def _decides(self, klass: TypeInfo, member: str) -> bool:
        """Answer whether one class of the walk up from the derived class decides the route.

        Args:
            klass: The class.
            member: The name of the member.

        Returns:
            True for a typed base, and for a class that defines the member.
        """
        return klass.fullname in self._BASES or member in klass.names
