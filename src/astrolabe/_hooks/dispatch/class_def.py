"""Register the hooks that mypy calls while it reads the definition of a class.

mypy calls a class hook once for each class that the plugin names, during
semantic analysis, the phase that resolves names. The two frame classes
have one: it gives their remainder parameter the default `None`.
"""

from __future__ import annotations

from mypy.plugin import ClassDefContext

from astrolabe._hooks.dispatch.generic import HookDispatcher, Target
from astrolabe._hooks.remainder_default import RemainderDefault
from astrolabe.frame import TypedDataFrame, TypedLazyFrame


class ClassDefHooks:
    """Build the class hooks, and register each one by the full name of its class.

    The plugin asks this class for the hook of a class when mypy reads the
    definition of that class.
    """

    def __init__(self) -> None:
        """Build every class hook, and register it in a new dispatcher."""
        self._dispatcher = HookDispatcher[ClassDefContext, None]()
        default = RemainderDefault()
        for frame in (TypedDataFrame, TypedLazyFrame):
            self._dispatcher.register(Target.of(frame), default)

    @property
    def dispatcher(self) -> HookDispatcher[ClassDefContext, None]:
        """The dispatcher that holds every registered class hook."""
        return self._dispatcher
