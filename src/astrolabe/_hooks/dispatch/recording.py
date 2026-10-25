"""Register the hooks of the recorder, which writes the code of each expression in its type.

The type of an expression stores the code of the expression, and the checker
computes the data types from the code when the expression meets a frame.

The recorder takes every call that gives an expression: each method of a
typed expression or of one of its namespaces, `pl.col`, each function of
`pl` and each selector of `cs`, the namespace attributes such as `.str`, and
a comparison whose literal stands on the left. A function of `pl` that gives
no expression, such as `pl.concat`, keeps its own hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl
import polars.selectors as cs

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.recording.deferred_types import DeferredHeads
from astrolabe._hooks.recording.recorder import (
    RecordedAccessor,
    RecordedCall,
    RecordedFunction,
    RecordedLiteralComparison,
)
from astrolabe.expr import TypedChainedWhen, TypedWhen

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from mypy.plugin import AttributeContext, FunctionContext, MethodContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup


class RecordingHooks:
    """Give the hook of the recorder for each full name that it serves.

    Example:
        >>> hooks = RecordingHooks(None)
        >>> hooks.method_hook("astrolabe.expr.TypedExpr.abs") is not None
        True
        >>> hooks.method_hook("polars.dataframe.frame.DataFrame.select") is None
        True
    """

    LITERALS: ClassVar[frozenset[str]] = frozenset({
        "builtins.int",
        "builtins.float",
        "builtins.bool",
        "builtins.str",
    })
    COMPARISONS: ClassVar[frozenset[str]] = frozenset({"__eq__", "__ne__"})
    DEFINERS: ClassVar[frozenset[str]] = frozenset({"Expr", "Selector"})

    def __init__(self, plugin: PluginLookup) -> None:
        """Keep the plugin, and the full names of the classes whose methods the recorder takes.

        Args:
            plugin: The plugin, which finds a class by its full name.

        Example:
            >>> "astrolabe.expr.TypedWhen" in RecordingHooks(None)._owners
            True
        """
        self._plugin = plugin
        classes = (
            *DeferredHeads.DECLARED,
            TypedWhen,
            TypedChainedWhen,
            *DeferredHeads.NAMESPACES.values(),
        )
        self._owners = frozenset(Target.of(owner).fullname for owner in classes)
        self._public = {**self._exports(pl, "pl"), **self._exports(cs, "cs")}

    def method_hook(self, fullname: str) -> Callable[[MethodContext], MypyType] | None:
        """Give the hook of a method that the recorder takes.

        Args:
            fullname: The full name of the method, by the class of the receiver.

        Returns:
            The hook, or `None`.

        `None` has no class of its own in mypy, which names its `__eq__` by
        the class that defines it, as `__eq__ of object`.

        Example:
            >>> RecordingHooks(None).method_hook("polars.functions.col.Col.__call__") is not None
            True
            >>> RecordingHooks(None).method_hook("__ne__ of object") is not None
            True
        """
        owner, _, name = fullname.rpartition(".")
        if owner in self._owners:
            return RecordedCall(self._plugin, name)
        operator = self._operator_by_definer(fullname)
        if operator is not None:
            return RecordedCall(self._plugin, operator)
        if fullname == Target.of(type(pl.col), "__call__").fullname:
            return RecordedFunction(self._plugin, "pl.col", None)
        if owner in self.LITERALS and name in self.COMPARISONS:
            return RecordedLiteralComparison(self._plugin, name)
        comparison, separator, definer = fullname.partition(" of ")
        if separator and definer == "object" and comparison in self.COMPARISONS:
            return RecordedLiteralComparison(self._plugin, comparison)
        return None

    def _operator_by_definer(self, fullname: str) -> str | None:
        """Give the operator of a name of the form `__add__ of Expr`.

        mypy names an operator this way where it checks one member of a
        union on the other side, by the class that defines the operator.

        Args:
            fullname: The name that mypy gives the method.

        Returns:
            The name of the operator, or `None` for any other name.

        Example:
            >>> RecordingHooks(None)._operator_by_definer("__radd__ of Expr")
            '__radd__'
            >>> RecordingHooks(None)._operator_by_definer("__eq__ of object") is None
            True
        """
        name, separator, definer = fullname.partition(" of ")
        if not separator or definer not in self.DEFINERS:
            return None
        return name if name.startswith("__") and name.endswith("__") else None

    def function_hook(
        self,
        fullname: str,
        current: Callable[[FunctionContext], MypyType] | None,
    ) -> Callable[[FunctionContext], MypyType] | None:
        """Give the hook of a function that the recorder takes.

        Args:
            fullname: The full name of the function.
            current: The hook of the function in the current model, which a
                call that gives no expression keeps.

        Returns:
            The hook, or `None`.

        Example:
            >>> RecordingHooks(None).function_hook("polars.functions.lit.lit", None) is not None
            True
            >>> RecordingHooks(None).function_hook("builtins.len", None) is None
            True
        """
        if fullname == "builtins.abs":
            return RecordedFunction(self._plugin, "abs", current)
        public = self._public.get(fullname)
        if public is None:
            return None
        return RecordedFunction(self._plugin, public, current)

    def attribute_hook(self, fullname: str) -> Callable[[AttributeContext], MypyType] | None:
        """Give the hook of a namespace attribute of an expression.

        Args:
            fullname: The full name of the attribute, by the class that defines it.

        Returns:
            The hook, or `None`.

        Example:
            >>> RecordingHooks(None).attribute_hook("polars.expr.expr.Expr.str") is not None
            True
        """
        owner, _, name = fullname.rpartition(".")
        if owner == Target.of(pl.Expr).fullname and name in DeferredHeads.NAMESPACES:
            return RecordedAccessor(self._plugin, name)
        return None

    def _exports(self, module: ModuleType, prefix: str) -> dict[str, str]:
        """Give the public name of each function that a module of polars exports, by its full name.

        Polars defines some functions under another name, such as `date_`
        for `pl.date`, so mypy names the call by the full name of the
        definition. The recorder writes the public name. A function of `pl`
        must come from `polars.functions`, which holds the functions that
        give an expression.

        Args:
            module: `polars` or `polars.selectors`.
            prefix: The name that the code writes before the function, `pl` or `cs`.

        Returns:
            The public name with its prefix, by the full name of the definition.

        Example:
            >>> exports = RecordingHooks(None)._exports(pl, "pl")
            >>> exports["polars.functions.as_datatype.date_"], exports["polars.functions.lit.lit"]
            ('pl.date', 'pl.lit')
            >>> "polars.io.csv.functions.read_csv" in exports
            False
        """
        exports: dict[str, str] = {}
        for name in dir(module):
            value = getattr(module, name)
            defined = getattr(value, "__module__", None)
            if name.startswith("_") or not callable(value) or not isinstance(defined, str):
                continue
            if module is pl and not defined.startswith("polars.functions"):
                continue
            exports.setdefault(f"{defined}.{value.__name__}", f"{prefix}.{name}")
        return exports
