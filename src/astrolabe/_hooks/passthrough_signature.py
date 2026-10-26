"""Check the arguments of a method that a typed class passes on to polars against polars."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from mypy.nodes import Decorator, FuncDef
from mypy.typeops import bind_self
from mypy.types import CallableType, get_proper_type

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.errors import InvalidCallShapeError
from astrolabe._rules.releases import PolarsRelease
from astrolabe.frame import TypedDataFrame, TypedLazyFrame, TypedSeries

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.plugin import MethodSigContext

    from astrolabe._hooks.domain.frame import PluginLookup


class PassthroughMethods:
    """Name each method of a typed class that passes its arguments on to the polars method.

    A typed frame or series keeps its type through a method that changes no
    column, such as `head`. The class declares such a method as
    `(self, *args, **kwargs) -> Self`, so that it follows the parameters of
    polars at any version. With that signature, mypy checks no argument of a
    call: `lf.head(1, 2)` passes, and polars raises `TypeError`. This class
    finds those methods from the classes themselves, and pairs each with the
    polars method that it calls, so that a signature hook can give the call
    the parameters of polars. `SINCE` gives the first release of polars that
    has a method that a recent release adds.

    Example:
        >>> methods = PassthroughMethods().targets()
        >>> methods[Target.of(TypedDataFrame, "head")].fullname
        'polars.dataframe.frame.DataFrame.head'
        >>> len(methods)
        63
    """

    BASES: ClassVar[dict[type, type]] = {
        TypedDataFrame: pl.DataFrame,
        TypedLazyFrame: pl.LazyFrame,
        TypedSeries: pl.Series,
    }
    SINCE: ClassVar[dict[str, str]] = {"gather": "1.41.0"}
    PASSES: ClassVar[tuple[inspect._ParameterKind, ...]] = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )

    def targets(self) -> dict[Target, Target]:
        """Give the polars method of each passthrough method of a typed class.

        Returns:
            The polars method, by the method of the typed class.

        Example:
            >>> methods = PassthroughMethods().targets()
            >>> methods[Target.of(TypedSeries, "round")].fullname
            'polars.series.series.Series.round'
            >>> import typing
            >>> methods = PassthroughMethods()
            >>> bases = methods.BASES.items()
            >>> pairs = [(base, name) for typed, base in bases for name in methods._names(typed)]
            >>> present = [getattr(base, name) for base, name in pairs if hasattr(base, name)]
            >>> any(typing.get_overloads(method) for method in present)
            False
        """
        found: dict[Target, Target] = {}
        for typed, base in self.BASES.items():
            for name in self._names(typed):
                found[Target.of(typed, name)] = Target.of(base, name)
        return found

    def _names(self, typed: type) -> list[str]:
        """Give the names of the methods that a typed class declares with `*args` and `**kwargs`.

        Args:
            typed: The typed class.

        Returns:
            The names, in the order of the class.

        Example:
            >>> "head" in PassthroughMethods()._names(TypedLazyFrame)
            True
            >>> "collect" in PassthroughMethods()._names(TypedLazyFrame)
            False
        """
        names: list[str] = []
        for name, member in vars(typed).items():
            if inspect.isfunction(member) and self._passes(member):
                names.append(name)
        return names

    def _passes(self, method: Callable[..., object]) -> bool:
        """Say whether a method takes only `self`, `*args` and `**kwargs`.

        Args:
            method: The method.

        Returns:
            Whether it does.

        Example:
            >>> PassthroughMethods()._passes(lambda self, *args, **kwargs: None)
            True
            >>> PassthroughMethods()._passes(lambda self, n: None)
            False
        """
        kinds = tuple(parameter.kind for parameter in inspect.signature(method).parameters.values())
        return kinds == self.PASSES


class PolarsSignatureHook:
    """Give a call of a passthrough method the parameters of the polars method that it calls.

    mypy checks the arguments of a call against the signature that a
    signature hook gives. This hook reads the signature of the polars
    method from the definitions that mypy holds, binds it to the receiver,
    and keeps the return type and the name of the typed method. So
    `lf.head(1, 2)` reports `Too many arguments`, as it does over a plain
    `pl.LazyFrame`, and the result keeps the typed frame. A method that the
    installed release of polars does not have reports the call
    (`check_release`).

    It sits in the plugin layer. It reads mypy definitions and builds a mypy
    signature, and it needs the lookup of the plugin to find the polars method.
    """

    def __init__(self, plugin: PluginLookup, polars: Target) -> None:
        """Keep the lookup of the plugin and the polars method.

        Args:
            plugin: The mypy plugin, which finds the definition of the polars method.
            polars: The polars method that the typed method calls.
        """
        self._plugin = plugin
        self._polars = polars

    def __call__(self, ctx: MethodSigContext) -> CallableType:
        """Give the signature of one call.

        Args:
            ctx: The mypy context of the signature at the call.

        Returns:
            The signature of the polars method, bound to the receiver, with the
            return type and the name of the typed method. The declared
            signature where mypy holds no plain signature of the polars method.
        """
        self.check_release(ctx)
        declared = self.declared()
        if declared is None:  # pragma: no cover
            # No polars method that a typed class passes on is overloaded, which a
            # doctest of `PassthroughMethods.targets` checks.
            return ctx.default_signature
        bound = bind_self(declared, ctx.type)
        default = ctx.default_signature
        return bound.copy_modified(ret_type=default.ret_type, name=default.name)

    def check_release(self, ctx: MethodSigContext) -> None:
        """Report the call where the installed release of polars does not have the method.

        Args:
            ctx: The mypy context of the signature at the call.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> polars = Target.of(pl.DataFrame, "gather")
            >>> hook = PolarsSignatureHook(new_plugin(new_type_fixture()), polars)
            >>> older, later = new_fail_context(), new_fail_context()
            >>> with PolarsRelease.pretend("1.40.1"):
            ...     hook.check_release(older)
            >>> with PolarsRelease.pretend("1.41.0"):
            ...     hook.check_release(later)
            >>> older.api.messages[0].endswith("DataFrame.gather from release 1.41.0")
            True
            >>> later.api.messages
            []
        """
        owner, _, name = self._polars.fullname.rpartition(".")
        first = PassthroughMethods.SINCE.get(name)
        if first is not None and PolarsRelease.current() < PolarsRelease.parse(first):
            method = f"{owner.rpartition('.')[2]}.{name}"
            InvalidCallShapeError.absent_method(method, first).fail(ctx)

    def declared(self) -> CallableType | None:
        """Give the signature that polars declares for its method, with `self`.

        A decorated method has the signature after its decorators, as mypy
        infers it.

        Returns:
            The signature, or `None` where the definition is not a plain or a
            decorated function.

        Example:
            >>> fixture = new_type_fixture()
            >>> polars = Target.of(pl.DataFrame, "head")
            >>> PolarsSignatureHook(new_plugin(fixture), polars).declared() is None
            True
        """
        node = self._plugin.lookup_fully_qualified(self._polars.fullname)
        definition = None if node is None else node.node
        if isinstance(definition, Decorator):
            typ = definition.var.type
        elif isinstance(definition, FuncDef):
            typ = definition.type
        else:  # pragma: no cover -- each target is one plain method of polars
            typ = None
        proper = get_proper_type(typ)
        return proper if isinstance(proper, CallableType) else None
