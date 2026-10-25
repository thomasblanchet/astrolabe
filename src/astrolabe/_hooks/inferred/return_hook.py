"""Give each call of a function its return type, built from the values that its body marks.

A user declares a function to return `InferredExpr`, `InferredDataFrame`
or `InferredLazyFrame`, and marks each returned value with `infer(...)`.
While mypy checks the body, another hook records the type of each
marked value. The hook here runs at each call of the function. It reads
those types, fits them to the arguments of the call, checks them, and
gives their union as the type of the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mypy.expandtype import expand_type
from mypy.nodes import FuncDef, TypeInfo
from mypy.plugin import MethodContext
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    TypeOfAny,
    TypeVarId,
    TypeVarType,
    get_proper_type,
)
from mypy.typeops import make_simplified_union

from astrolabe._hooks.errors import (
    ImplicitlyInferredFallbackError,
    InferMisuseError,
    WrongInferredVariantError,
)
from astrolabe._hooks.inferred.cached_variants import CachedVariants
from astrolabe._hooks.domain.deferral import CheckerDeferral
from astrolabe._hooks.inferred.families import FAMILIES, family_for
from astrolabe._hooks.inferred.walker import LiveInferCallWalker

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.plugin import FunctionContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.config import ImplicitInferredFallbackPolicy
    from astrolabe._hooks.domain.frame import PluginLookup
    from astrolabe._hooks.inferred.accumulator import InferAccumulator
    from astrolabe._hooks.inferred.call_evaluations import CallEvaluations
    from astrolabe._hooks.inferred.families import PlaceholderFamily

    # A call of a function, or a call of a method on an instance.
    type CallContext = FunctionContext | MethodContext


@dataclass(frozen=True)
class _SharedDependencies:
    """Hold the objects that the hook for every such function shares.

    Attributes:
        accumulator: The record of the type of each `infer(...)` call,
            by the function whose body holds it.
        fallback_policy: The setting that decides, for each placeholder,
            whether a plain return type is an error.
        plugin: The mypy plugin instance. It finds the `TypeInfo` of the
            plain polars class when the return type falls back to it.
        evaluations: The binding of each call whose open frame arguments
            the fit checked, which writes each value in the caller's terms.
    """

    accumulator: InferAccumulator
    fallback_policy: ImplicitInferredFallbackPolicy
    plugin: PluginLookup
    evaluations: CallEvaluations


class InferredReturnHookFactory:
    """Give mypy a hook for a call of a user function whose return type is a placeholder.

    mypy asks the plugin for a function hook by the fullname of the
    called function. The plugin keeps its other function hooks in a
    fixed table. It cannot list the user functions that return a
    placeholder, because the user writes them. So on a fullname that
    the table does not hold, the plugin asks this factory. The factory
    looks the function up, and builds a hook if the function returns
    `InferredExpr`, `InferredDataFrame` or `InferredLazyFrame`.
    """

    def __init__(
        self,
        plugin: Plugin,
        accumulator: InferAccumulator,
        fallback_policy: ImplicitInferredFallbackPolicy,
        evaluations: CallEvaluations,
    ) -> None:
        """Keep the plugin for lookups, and the objects that every hook it builds shares.

        Args:
            plugin: The mypy plugin instance. It looks up a function by
                its fullname.
            accumulator: The record of the type of each `infer(...)`
                call.
            fallback_policy: The setting that decides whether a plain
                return type is an error.
            evaluations: The binding of each call whose open frame
                arguments the fit checked.
        """
        self._plugin = plugin
        self._deps = _SharedDependencies(accumulator, fallback_policy, plugin, evaluations)

    def build(self, fullname: str) -> Callable[[CallContext], MypyType] | None:
        """Build the hook for one called function or method, if it returns a placeholder.

        A method that takes a decorator, for example `@staticmethod`, is
        not a plain `def`, so it has no hook. An overloaded function has
        no hook either.

        Args:
            fullname: The mypy fullname of the called function.

        Returns:
            The hook, or None if the name is not found, names something
            other than one plain `def`, or names a function with no
            callable type or with a return type that is not a
            placeholder.
        """
        node = self._plugin.lookup_fully_qualified(fullname)
        if node is None or not isinstance(node.node, FuncDef):
            return None
        funcdef = node.node
        signature = funcdef.type
        if not isinstance(signature, CallableType):
            return None
        family = family_for(signature.ret_type)
        if family is None:
            return None
        return _InferredReturnHook(funcdef, signature, family, self._deps)


class _InferredReturnHook:
    """Type each call of one function whose return type is a placeholder.

    At a call, the hook reads the variants of the function from the
    record. A variant is the type of one value that the body returns
    through `infer(...)`. The hook fits each variant to the arguments of
    this call, checks that each is a class that the placeholder accepts,
    and gives the union. If one variant is the plain polars class, the
    call gets the plain class, and the checker reports that if the
    setting asks for it.

    The hook also reports, once for each function, an `infer(...)` call
    that the body uses in a place where it does not work.

    If mypy has not checked the body yet, the record is empty. The hook
    then asks mypy to check the function that holds the call again, after
    the body. Where mypy cannot do that, the call keeps the type that mypy
    gives it, the placeholder class.
    """

    def __init__(
        self,
        funcdef: FuncDef,
        signature: CallableType,
        family: PlaceholderFamily,
        deps: _SharedDependencies,
    ) -> None:
        """Keep the function, its signature and its placeholder, and the shared objects.

        Args:
            funcdef: The function whose calls this hook types.
            signature: The declared signature of the function. It gives
                the type variables that a call binds.
            family: The placeholder that the function returns, with the
                classes that it accepts.
            deps: The objects that the hooks of every such function
                share.
        """
        self._funcdef = funcdef
        self._signature = signature
        self._family = family
        self._deps = deps

    def __call__(self, ctx: CallContext) -> MypyType:
        """Give one call of the function the type that its variants build.

        Args:
            ctx: The mypy context of the call.

        Returns:
            The type that `_resolved_type` builds from the variants in the
            record in memory or in the cache. Without a variant, the type
            that `_waiting_type` gives.
        """
        self._report_misplaced_calls(ctx)
        variants = self._deps.accumulator.resolve_variants(self._funcdef) or CachedVariants().read(
            self._deps.plugin,
            self._funcdef,
        )
        if variants:
            return self._resolved_type(ctx, variants)
        return self._waiting_type(ctx)

    def _waiting_type(self, ctx: CallContext) -> MypyType:
        """Give the type of a call that comes before the body gave a variant.

        If the body returns an `infer(...)` call, its variant is not
        recorded yet, and the body will give one. mypy then checks the
        function that holds this call again, after the body. The call
        meanwhile has a type that reports nothing.

        Args:
            ctx: The mypy context of the call.

        Returns:
            `Any` while mypy will check the call again, and else the type
            that mypy gives the call, the placeholder class.
        """
        walker = LiveInferCallWalker(only_returned=True)
        returns_infer = bool(walker.find(self._funcdef.body))
        if returns_infer and CheckerDeferral(ctx.api).deferred():
            return AnyType(TypeOfAny.special_form)
        return ctx.default_return_type

    def _resolved_type(self, ctx: CallContext, variants: list[MypyType]) -> MypyType:
        """Build the return type of one call from the variants of the function.

        The variants are fitted to the arguments of the call before they
        are checked. A generic function, for example one with a parameter
        `e: ExprT`, records `ExprT` as the variant of `return infer(e)`.
        Only the argument of the call turns that type variable into a
        class that the check can accept.

        The check reports a `[wrong-inferred-variant]` error for each
        variant that is neither a typed class of the placeholder nor its
        plain class.

        Args:
            ctx: The mypy context of the call, where the errors point.
            variants: The type of each live variant of the function.
                There is at least one.

        Returns:
            An `Any` from an error if a variant is a class that the
            placeholder does not accept. The plain polars class if a
            variant is that class. Otherwise, the union of the variants.
        """
        substitution = self._substitution(ctx)
        specialized = [expand_type(variant, substitution) for variant in variants]
        wrong = [variant for variant in specialized if not self._matches_family(variant)]
        for variant in wrong:
            self._wrong_variant(variant).fail(ctx)
        if wrong:
            return AnyType(TypeOfAny.from_error)
        if any(self._family.matches_plain(variant) for variant in specialized):
            return self._plain_fallback(ctx)
        return make_simplified_union(self._in_callers_terms(ctx, specialized))

    def _wrong_variant(self, variant: MypyType) -> WrongInferredVariantError:
        """Build the error for one variant that the placeholder does not accept.

        A variant of another family, for example an expression where the
        function declares a frame, is named by its kind. Its type holds forms
        of the plugin that a reader does not know. Any other variant is named
        by its type.

        Args:
            variant: The type of the variant, fitted to the call.

        Returns:
            The error.
        """
        for family in FAMILIES:
            if family.matches_typed(variant) or family.matches_plain(variant):
                return WrongInferredVariantError.of_kind(self._family.display_name, family.kind)
        return WrongInferredVariantError.of_type(self._family.display_name, str(variant))

    def _in_callers_terms(self, ctx: CallContext, variants: list[MypyType]) -> list[MypyType]:
        """Write each variant in the terms of the caller's own frames.

        A variant is written in the terms of the function: its remainder,
        the columns that its type does not name, is built over the
        remainder of a parameter, for example `Minus[Rest, Literal["x"]]`.
        The fit of the call bound `Rest` to the caller's own remainder, or
        to the caller's own columns. A call with no open frame argument has
        no binding, and its variants name no remainder of the function.

        Args:
            ctx: The mypy context of the call.
            variants: The variants, fitted to the call and checked.

        Returns:
            Each variant under the binding of the call.
        """
        evaluation = self._deps.evaluations.taken(ctx.context)
        if evaluation is None:
            return variants
        return [evaluation.evaluated(variant) for variant in variants]

    def _report_misplaced_calls(self, ctx: CallContext) -> None:
        """Report each `infer(...)` call that the body uses in a wrong place.

        The record gives these calls only at the first call of the
        function that mypy checks. Later calls report nothing again.

        Args:
            ctx: The mypy context of the call, where the error points.
        """
        for _ in self._deps.accumulator.misplaced_calls_once(self._funcdef):
            InferMisuseError.not_directly_returned().fail(ctx)

    def _matches_family(self, variant: MypyType) -> bool:
        """Answer whether one variant is a class that the placeholder accepts.

        Args:
            variant: The variant to look at.

        Returns:
            True if the variant is a typed class of the placeholder or
            its plain class.
        """
        return self._family.matches_typed(variant) or self._family.matches_plain(variant)

    def _plain_fallback(self, ctx: CallContext) -> MypyType:
        """Give the call the plain polars class, and report that if the setting asks for it.

        One plain variant makes the whole return type plain. Each typed
        class is a subclass of the plain class, so their union is the
        plain class, and every typed column is lost. The setting
        `disallow_implicit_inferred_fallback` decides whether this is an
        error.

        Args:
            ctx: The mypy context of the call, where the error points.

        Returns:
            The plain polars class of the placeholder, or the type that
            mypy gives the call if that class is not loaded.
        """
        if self._deps.fallback_policy.disallows(self._family.operation_key):
            ImplicitlyInferredFallbackError(self._family.display_name).fail(ctx)
        info = self._deps.plugin.lookup_fully_qualified(self._family.plain_target.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # A variant is already an instance of the plain class, so
            # mypy has loaded that class before this call is typed.
            return ctx.default_return_type
        return Instance(info.node, [])

    def _substitution(self, ctx: CallContext) -> dict[TypeVarId, MypyType]:
        """Give the type that this call binds to each type variable of the function.

        Only a parameter whose declared type is a bare type variable, for
        example `e: ExprT`, binds one, to the type of the first argument
        for that parameter. A type variable inside another type, for
        example `list[ExprT]`, stays unbound, and so does one whose
        parameter gets no argument. mypy lists the arguments of a call
        by parameter, whether the caller passed them by position or by
        name.

        Args:
            ctx: The mypy context of the call.

        Returns:
            The type of the argument, keyed by the id of the type
            variable of its parameter.
        """
        skipped = self._skipped(ctx)
        substitution: dict[TypeVarId, MypyType] = {}
        for index, declared in enumerate(self._signature.arg_types):
            proper = get_proper_type(declared)
            if not isinstance(proper, TypeVarType):
                continue
            listed = index - skipped
            if 0 <= listed < len(ctx.arg_types) and ctx.arg_types[listed]:
                substitution[proper.id] = ctx.arg_types[listed][0]
        return substitution

    def _skipped(self, ctx: CallContext) -> int:
        """Give how many parameters at the start of the signature this call does not list.

        A method called on an instance, `maker.make(e)`, does not list the
        instance, which fills `self`. A method called on its class,
        `Maker.make(maker, e)`, lists it, as a plain function call lists
        every argument.

        Args:
            ctx: The mypy context of the call.

        Returns:
            1 for a method called on an instance, else 0.
        """
        if not isinstance(ctx, MethodContext):
            return 0
        return 1 if isinstance(get_proper_type(ctx.type), Instance) else 0
