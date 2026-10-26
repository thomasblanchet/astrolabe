"""Mark a function whose return type the checker infers as part of the interface of its module.

mypy can check a module in two phases: first the code at the top of the
module and the signatures of its functions, then the bodies of the
functions. It checks a body in the first phase only where semantic analysis
marked the function as one whose body shapes the interface of the module.
A function that returns `InferredExpr`, `InferredDataFrame`,
`InferredLazyFrame`, or a frame with `InferredRemainder`, takes its return
type from its body. Without the mark, a call at the top of the module meets
the function before its body, and keeps the placeholder as its type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import FuncDef
from mypy.semanal import SemanticAnalyzer
from mypy.typeanal import TypeAnalyser
from mypy.types import AnyType, TypeOfAny

from astrolabe._hooks.errors import MarkerArityError

if TYPE_CHECKING:
    from mypy.plugin import AnalyzeTypeContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.dispatch.generic import Target


class EnclosingFunction:
    """Find the function whose signature or body semantic analysis reads now.

    A type-analysis hook gets the interface of the type analyser, which
    does not name the function that holds the annotation. The analyser
    keeps the semantic analyzer, and that one keeps the stack of the
    functions that it reads. Both are private objects of mypy, so this
    class is the one place that touches them.

    Example:
        >>> EnclosingFunction(object()).current() is None
        True
    """

    def __init__(self, api: object) -> None:
        """Keep the interface that mypy gives a type-analysis hook.

        Args:
            api: The `ctx.api` of the hook, a `TypeAnalyser`.
        """
        self._api = api

    def current(self) -> FuncDef | None:
        """Give the innermost function that semantic analysis reads now.

        Returns:
            The function, or None at the top of a module or of a class.
        """
        if not isinstance(self._api, TypeAnalyser):
            return None
        semantic = self._api.api
        if not isinstance(semantic, SemanticAnalyzer):  # pragma: no cover
            # mypy builds each type analyser with the semantic analyzer
            # that reads the module.
            return None
        functions = semantic.scope.functions
        function = functions[-1] if functions else None
        return function if isinstance(function, FuncDef) else None


class PlaceholderTypeHook:
    """Read a placeholder of an inferred return type, and mark the function that writes it.

    The placeholders are `InferredExpr`, `InferredDataFrame`,
    `InferredLazyFrame` and `InferredRemainder`. Each one says that the
    body of the function gives its return type. The hook marks that
    function (`def_or_infer_vars`), so that mypy checks its body with the
    signatures and the code at the top of the module, in the order of the
    source. A call after the function then finds the type that its body
    gives. The type of the placeholder stays as it is.

    It is a mypy type-analysis hook in the plugin layer. The plugin builds
    one for each placeholder.
    """

    def __init__(self, target: Target) -> None:
        """Keep the placeholder class of this hook.

        Args:
            target: The placeholder class.
        """
        self._target = target

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Mark the function that writes the placeholder, and give the placeholder type.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The placeholder type. `Any` after an error, for a placeholder
            with type arguments.
        """
        name = self._target.fullname.rsplit(".", 1)[-1]
        if ctx.type.args:
            MarkerArityError(f"{name} takes no type arguments").fail(ctx)
            return AnyType(TypeOfAny.from_error)
        function = EnclosingFunction(ctx.api).current()
        if function is not None:
            function.def_or_infer_vars = True
        return ctx.api.named_type(self._target.fullname, [])
