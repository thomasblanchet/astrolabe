"""The module that mypy loads when `plugins = ["astrolabe.plugin"]` names it.

mypy asks a plugin, for each function, method, attribute and type that it
meets, whether the plugin has a hook for it. This module answers those
questions. It holds no checking rule itself: it reads the settings, builds
the tables of hooks, and gives mypy the hook for each name.
"""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

from mypy.plugin import (
    AnalyzeTypeContext,
    AttributeContext,
    ClassDefContext,
    DynamicClassDefContext,
    FunctionContext,
    FunctionSigContext,
    MethodContext,
    MethodSigContext,
    Plugin,
    ReportConfigContext,
)

import astrolabe
from astrolabe._hooks.code_fingerprint import CodeFingerprint
from astrolabe._hooks.config import AstrolabeConfigLoader, ConfigWarner
from astrolabe._hooks.dispatch.attribute import AttributeHooks
from astrolabe._hooks.dispatch.class_def import ClassDefHooks
from astrolabe._hooks.dispatch.dynamic_class import DynamicClassHooks
from astrolabe._hooks.dispatch.recording import RecordingHooks
from astrolabe._hooks.dispatch.function import FunctionHooks
from astrolabe._hooks.fit import FitSignatureFactory
from astrolabe._hooks.dispatch.method import MethodHooks
from astrolabe._hooks.dispatch.type_analysis import TypeAnalyzeHooks
from astrolabe._hooks.dispatch.typed_base import TypedBaseRoute
from astrolabe._hooks.domain.unproven_gate import UnprovenGate
from astrolabe._hooks.inferred.accumulator import InferAccumulator
from astrolabe._hooks.inferred.call_evaluations import CallEvaluations
from astrolabe._hooks.inferred.return_hook import InferredReturnHookFactory
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.inferred.infer_hook import InferSignatureHook
from astrolabe._hooks.inferred.position import InferredRemainderGuard
from astrolabe._hooks.passthrough_signature import PassthroughMethods, PolarsSignatureHook
from astrolabe._hooks.runtime_schema import RuntimeSchemaSignatureHook
from astrolabe.declare import InferredExpr, infer
from astrolabe.expr import TypedExpr
from astrolabe.frame import TypedDataFrame, assume_schema, check_schema, force_schema
from astrolabe.remainder import Both

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.nodes import MypyFile
    from mypy.options import Options
    from mypy.types import CallableType
    from mypy.types import Type as MypyType


class AstrolabePlugin(Plugin):
    """Give mypy the astrolabe hook for each name that mypy asks about.

    mypy calls a `get_*_hook` method with the full name of a function, a
    method, an attribute or a type. The method gives back a hook, which is
    the code that mypy then runs at each use of that name, or None. Most
    names have a fixed hook: a polars method such as `DataFrame.select`,
    a marker type, or a declaration call such as `column(...)`. A table
    for each kind of hook holds these: `TypeAnalyzeHooks`, `MethodHooks`,
    `FunctionHooks`, `AttributeHooks`, `DynamicClassHooks` and
    `ClassDefHooks`.

    Two kinds of hook cannot live in a fixed table, because they serve
    functions that the user writes. `FitSignatureFactory` checks the
    arguments of a function whose parameters are declared expression
    types or open frames. `InferredReturnHookFactory` computes the return
    type of a function that returns `InferredExpr`, `InferredDataFrame`,
    `InferredLazyFrame`, or a frame type with an `InferredRemainder`.
    Each one looks at the definition of the called function, and gives a
    hook only where one applies.
    """

    _PACKAGE: Final[str] = astrolabe.__name__
    _OWN_MODULE_DEPS: Final[tuple[str, ...]] = (
        TypedDataFrame.__module__,
        Both.__module__,
        TypedExpr.__module__,
        InferredExpr.__module__,
    )
    _ORDINARY_IMPORT_PRIORITY: Final[int] = 10
    _INFER: Final[str] = Target.of(infer).fullname
    _RUNTIME_SCHEMA: Final[frozenset[str]] = frozenset(
        Target.of(function).fullname for function in (check_schema, force_schema, assume_schema)
    )
    _NO_KNOWN_LINE: Final[int] = -1

    def __init__(self, options: Options) -> None:
        """Read the `[tool.astrolabe]` settings and build every table of hooks.

        If mypy runs without `allow_redefinition`, a warning says so. The
        checker needs that option: a loop that rebinds a variable to a new
        expression type is an error without it.

        Args:
            options: The options of mypy for this run.
        """
        super().__init__(options)
        self.astrolabe_config = AstrolabeConfigLoader().load_for_options(options)
        if not options.allow_redefinition:
            ConfigWarner().redefinition_not_allowed()

        self._type_analyze_hooks = TypeAnalyzeHooks(self)
        infer_accumulator = InferAccumulator()
        self._function_hooks = FunctionHooks(
            self,
            self.astrolabe_config.allow_unproven,
            self.astrolabe_config.disallow_vertical_concat,
            infer_accumulator,
        )
        self._method_hooks = MethodHooks(
            self,
            self.astrolabe_config.allow_unproven,
            self.astrolabe_config.elevate_untyped_select,
        )
        self._attribute_hooks = AttributeHooks(self)
        self._typed_base = TypedBaseRoute(self)
        self._dynamic_class_hooks = DynamicClassHooks(self)
        self._class_def_hooks = ClassDefHooks()
        evaluations = CallEvaluations()
        self._inferred_return_hooks = InferredReturnHookFactory(
            self,
            infer_accumulator,
            self.astrolabe_config.disallow_implicit_inferred_fallback,
            evaluations,
        )
        self._recording = RecordingHooks(self)
        self._fit_hooks = FitSignatureFactory(
            self,
            UnprovenGate(self.astrolabe_config.allow_unproven),
            evaluations,
        )
        self._passthrough = PassthroughMethods().targets()

    def get_function_signature_hook(
        self,
        fullname: str,
    ) -> Callable[[FunctionSigContext], CallableType] | None:
        """Give the hook that checks each argument of a call against its declared type.

        Any function that a user writes can take a declared expression type
        or an open frame, that is, a frame type with a remainder of
        undeclared columns. So no fixed table lists these functions.
        `FitSignatureFactory` reads the definition of the called function
        at each name.

        Args:
            fullname: The full name of the called function.

        Returns:
            The hook, or None for a function with no such parameter.
            `infer` gets its own hook, `InferSignatureHook`, and so do the
            functions that take a frame type as a value,
            `RuntimeSchemaSignatureHook`.
        """
        if fullname == self._INFER:
            return InferSignatureHook()
        if fullname in self._RUNTIME_SCHEMA:
            return RuntimeSchemaSignatureHook()
        return self._fit_hooks.hook_for(fullname)

    def get_method_signature_hook(
        self,
        fullname: str,
    ) -> Callable[[MethodSigContext], CallableType] | None:
        """Give the hook that checks each argument of a method call against its declared type.

        A method that a user writes gets the same check as a function. mypy
        gives the hook the signature of the bound method, which has no
        `self` parameter. A method of a typed class that passes its
        arguments on to polars gets the parameters of the polars method
        (`PolarsSignatureHook`), and so does the same method of a class
        derived from a typed frame.

        Args:
            fullname: The full name of the called method.

        Returns:
            The hook, or None for a method with no open frame parameter
            and no declared expression parameter.
        """
        routed = self._typed_base.of(fullname) or fullname
        polars = self._passthrough.get(Target(routed))
        if polars is not None:
            return PolarsSignatureHook(self, polars)
        return self._fit_hooks.method_hook_for(fullname)

    def get_additional_deps(self, file: MypyFile) -> list[tuple[int, str, int]]:
        """Add the public astrolabe modules to every build, even if no file imports them.

        mypy analyzes only the modules that the code imports. The hooks
        build types from the classes in `astrolabe.frame`,
        `astrolabe.remainder`, `astrolabe.expr` and `astrolabe.declare`,
        and they look those classes up by name. A file that uses only
        polars still gets typed results, for example from `pl.col("a")` or
        from a frame built with `schema={...}`. If no file imports
        astrolabe, the lookup finds nothing, and those hooks cannot build
        their types. So each module that mypy analyzes depends on these
        four modules.

        Args:
            file: The module that mypy is about to analyze.

        Returns:
            One dependency for each of the four modules, in the form that
            mypy expects: the priority of an ordinary import, the module, and
            -1 for the line, because no line of source code holds the import.
            For the `astrolabe` package or a module in it, no dependency: its
            own imports already reach them.
        """
        package = self._PACKAGE
        if file.fullname == package or file.fullname.startswith(f"{package}."):
            return []
        priority = self._ORDINARY_IMPORT_PRIORITY
        return [(priority, module, self._NO_KNOWN_LINE) for module in self._OWN_MODULE_DEPS]

    def get_type_analyze_hook(
        self,
        fullname: str,
    ) -> Callable[[AnalyzeTypeContext], MypyType] | None:
        """Give the hook that reads a marker type in an annotation.

        The hook checks the arguments of a marker, for example
        `Minus[Rest, Literal["a"]]`, and reports a marker that it cannot read.

        Args:
            fullname: The full name of the marker type.

        Returns:
            The hook, or None if the name has none.
        """
        return self._type_analyze_hooks.dispatcher.get_hook(fullname)

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], MypyType] | None:
        """Give the hook that computes the result type of a method call.

        Frame methods such as `select` and `join`, and expression methods
        such as `sum` and `str.len_chars`, have one in the fixed table. If
        the table has none, the called method can be one that a user wrote
        with an inferred return type, for example `InferredExpr`.
        `InferredReturnHookFactory` then gives the hook that computes its
        return type from its body. A hook from the table runs behind
        `InferredRemainderGuard`, which reports a receiver whose remainder
        is `InferredRemainder`, a type written out of place.

        Args:
            fullname: The full name of the called method.

        Returns:
            The hook from the table, else the hook for an inferred return
            type, else None.
        """
        recorded = self._recording.method_hook(fullname)
        if recorded is not None:
            return recorded
        table = self._method_hooks.dispatcher.get_hook(fullname)
        registered = table or self._derived_method_hook(fullname)
        if registered is not None:
            return InferredRemainderGuard(registered)
        return self._inferred_return_hooks.build(fullname)

    def _derived_method_hook(self, fullname: str) -> Callable[[MethodContext], MypyType] | None:
        """Give the hook of the typed base for a method of a class derived from a typed frame.

        Args:
            fullname: The full name of the called method.

        Returns:
            The hook that the base registers for the method, or None.
        """
        routed = self._typed_base.of(fullname)
        if routed is None:
            return None
        return self._method_hooks.dispatcher.get_hook(routed)

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], MypyType] | None:
        """Give the hook that computes the result type of a function call.

        A polars function such as `pl.col` or `pl.concat` has a hook in the
        fixed table. If the table has none, the called function can be one
        that a user wrote with an inferred return type, for example
        `InferredExpr`. `InferredReturnHookFactory` then gives the hook
        that computes its return type from its body.

        Args:
            fullname: The full name of the called function.

        Returns:
            The hook from the table, else the hook for an inferred return
            type, else None.
        """
        registered = self._function_hooks.dispatcher.get_hook(fullname)
        recorded = self._recording.function_hook(fullname, registered)
        if recorded is not None:
            return recorded
        if registered is not None:
            return registered
        return self._inferred_return_hooks.build(fullname)

    def get_attribute_hook(self, fullname: str) -> Callable[[AttributeContext], MypyType] | None:
        """Give the hook that computes the type of an attribute.

        For example, `.schema` on a typed frame and `.str` on a typed
        expression have one.

        Args:
            fullname: The full name of the attribute.

        Returns:
            The hook, or None if the name has none. An attribute of a class
            derived from a typed frame gets the hook of its typed base.
        """
        recorded = self._recording.attribute_hook(fullname)
        if recorded is not None:
            return recorded
        registered = self._attribute_hooks.dispatcher.get_hook(fullname)
        if registered is not None:
            return registered
        routed = self._typed_base.of(fullname)
        return None if routed is None else self._attribute_hooks.dispatcher.get_hook(routed)

    def get_dynamic_class_hook(
        self,
        fullname: str,
    ) -> Callable[[DynamicClassDefContext], None] | None:
        """Give the hook that turns a declaration call into a named type.

        mypy calls this for an assignment such as
        `Price = column("price", pl.Float64)` or `Rest = other_columns()`.
        The hook reads the call and makes the name on the left a type that
        annotations can use.

        Args:
            fullname: The full name of the called function.

        Returns:
            The hook, or None if the name has none.
        """
        return self._dynamic_class_hooks.dispatcher.get_hook(fullname)

    def get_customize_class_mro_hook(
        self,
        fullname: str,
    ) -> Callable[[ClassDefContext], None] | None:
        """Give the hook that mypy runs when it reads the definition of a frame class.

        mypy calls this for each class, before it reads the body of the
        class. The hook of `TypedDataFrame` and `TypedLazyFrame` gives
        their remainder parameter the default `None`, so a frame type
        written as a value, as in `check_schema(raw, TypedDataFrame[Trades])`,
        takes one argument.

        Args:
            fullname: The full name of the class.

        Returns:
            The hook, or None if the class has none.
        """
        return self._class_def_hooks.dispatcher.get_hook(fullname)

    def report_config_data(self, ctx: ReportConfigContext) -> object:
        """Give mypy the value that decides when its cache of a module is out of date.

        mypy reuses the cached result of a module while this value stays
        the same. The value holds the `[tool.astrolabe]` settings, so a new
        setting makes mypy check every module again. It also holds the
        modules that `get_additional_deps` adds to each build. The cache
        records those modules as dependencies of every module, so a cache
        that names a different set of modules must not be used. It holds a
        fingerprint of the source of the plugin too (`CodeFingerprint`), so
        a change of a rule makes mypy check every module again.

        Args:
            ctx: The context that mypy gives for one module. The value does
                not depend on the module, so the method does not read it.

        Returns:
            A value that JSON can encode: the settings, the added modules and
            the fingerprint of the code.
        """
        return {
            "config": self.astrolabe_config.cache_key(),
            "forced_modules": list(self._OWN_MODULE_DEPS),
            "code": CodeFingerprint().digest(),
        }


def plugin(version: str) -> type[Plugin]:
    """Give mypy the plugin class. mypy calls this function to load the plugin.

    mypy looks for a function named `plugin` in each module that its
    `plugins = [...]` option names.

    Args:
        version: The version of mypy that runs. The plugin works the same
            for each version, so the function does not read it.

    Returns:
        `AstrolabePlugin`.
    """
    return AstrolabePlugin
