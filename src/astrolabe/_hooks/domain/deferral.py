"""Ask mypy to check a function again in a later pass, as it does for a type that is not ready.

mypy checks the function bodies of a module in order. A call to a
function whose return type comes from its body, for example one that
returns `InferredExpr`, can come before that body. The type of the call
is then not ready. mypy has a mechanism for such a case: it defers the
function that holds the call, checks the rest of the module, and checks
the deferred function again. The plugin interface does not expose it, so
this module reaches into the type checker, in the same way as mypy's own
`handle_cannot_determine_type`.
"""

from __future__ import annotations

from mypy.checker import TypeChecker
from mypy.nodes import FuncDef


class CheckerDeferral:
    """Defer the function that holds the current call, where mypy itself could.

    mypy defers only a function at the top of a module or of a class, and
    never a lambda. It can defer only before its last pass. A call at the
    top of a module has no function to defer. In each of these cases the
    deferral does nothing, and the caller keeps its own answer.

    The type checker is a private object of mypy, so this class is the
    one place that touches it.
    """

    def __init__(self, api: object) -> None:
        """Keep the API that mypy gives a hook during type checking.

        Args:
            api: The `ctx.api` of a hook. During type checking it is the
                `TypeChecker`.
        """
        self._api = api

    def deferred(self) -> bool:
        """Defer the function that holds the current call, if mypy allows it.

        The method defers the top-level function of the current scope. mypy then checks that
        function again in its next pass.

        Returns:
            True if mypy will check the function again. False in the last pass, or if the
            scope is not inside a function that mypy can defer.
        """
        checker = self._api if isinstance(self._api, TypeChecker) else None
        if checker is None or checker.pass_num >= checker.last_pass:
            return False
        node = checker.scope.top_level_function()
        if not isinstance(node, FuncDef):
            return False
        checker.defer_node(node, checker.scope.enclosing_class(node))
        checker.current_node_deferred = True
        return True

    def pending(self) -> bool:
        """Answer whether mypy will check the current function again.

        A hook that reports an error from a result that a deferred call
        left open would report it once in this pass and again in the next.
        So it asks this first.

        Returns:
            True during a pass whose function mypy has deferred.
        """
        return isinstance(self._api, TypeChecker) and self._api.current_node_deferred
