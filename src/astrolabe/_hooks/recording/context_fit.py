"""Check a recorded expression against the type that its position declares.

A user can declare the type of an expression, for example as the return type
of a function or on an annotated assignment. The type that the recorder builds
for the expression at that position is its code, which is almost never written
in the same way as the declaration, even when the two agree. If mypy compares
them, it compares them as text. It then rejects a correct body and shows both
long types in full.

mypy tells a hook the type that the position of its call expects: the declared
type of a `return` value, of an annotated assignment or of a call argument. It
gives this expected type only to the outermost call, and not to the calls nested
inside it. If it is one of the declarations of this project, the code here
decides the position by the fit, the check that an expression matches its
declaration. An expression that fits gets the declared type back. An expression
that does not fit gets one error that names the rule it breaks, and also the
declared type back, so that mypy adds no error of its own.

One case keeps the message of mypy instead. mypy checks a binary operator by
trying one side with its errors hidden. If it then uses the reflected method of
the other side, it drops those errors and the result of the first side. An error
from the fit under such an operator is lost, and mypy reports the position
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from mypy.checker import TypeChecker
from mypy.nodes import ComparisonExpr, OpExpr, UnaryExpr

from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.deferred.argument import DeferredArgument
from astrolabe._hooks.deferred.fit import DeferredFit
from astrolabe._hooks.encoding.expr_evaluation import ExpressionArgument
from astrolabe._hooks.encoding.selection_codec import SelectionReader
from astrolabe._hooks.errors import ExpressionFitError
from astrolabe._hooks.expression.fit import SliceDecl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.nodes import Context, Expression
    from mypy.plugin import FunctionContext, MethodContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.deferred.code import Code


@dataclass(frozen=True)
class _FitSite:
    """The place where an error from the fit is reported.

    An error class reports through an object with a `context` and an
    `api`, the same two attributes as a mypy hook context. This class
    holds the two values a codec keeps from its call.

    Attributes:
        context: The node of the call, where the error points.
        api: The mypy type checker, which reports the error.
    """

    context: Context
    api: TypeChecker


class _ContextReader:
    """Read the type that the position of one call expects, and report there.

    Both codecs in this module need the expected type and a way to
    report an error at the call. This class gives both from one mypy
    context.

    Args of the constructor:
        ctx: The context of the call.
    """

    def __init__(self, ctx: MethodContext | FunctionContext) -> None:
        """Keep the type checker and the place to report at.

        Args:
            ctx: The context of the call.
        """
        self._checker = checked_cast(ctx.api, TypeChecker)
        self._site = _FitSite(ctx.context, self._checker)

    def declared(self) -> MypyType | None:
        """Give the type that the position of the call expects.

        Returns:
            The expected type from mypy, or None if the position does
            not declare one.
        """
        return self._checker.type_context[-1]

    def refuse(self, refusal: str) -> None:
        """Report that the expression does not fit, at the node of the call.

        Args:
            refusal: The rule of the fit that the expression breaks,
                in the words of the engine.
        """
        ExpressionFitError.at_context(refusal).fail(self._site)


class DeferredContextFit:
    """Decide the position of a recorded call of the deferred model by the fit.

    In the deferred model, the recorder builds the type of each call on an
    expression, and the type holds the code. Where the position of the call
    declares an expression type, the call gets that declared type, after
    the fit of its code, as `ContextFittedCodec` does for a minted type.

    mypy checks an operator with its errors held back, and an error there
    turns the result into `Any`. So an operator that does not fit keeps the
    recorded type, and mypy reports that the type does not match the
    position.

    Args of the constructor:
        ctx: The context of the call.
    """

    OPERATORS: ClassVar[tuple[type[Expression], ...]] = (OpExpr, UnaryExpr, ComparisonExpr)

    def __init__(self, ctx: MethodContext | FunctionContext) -> None:
        """Read the position of the call.

        Args:
            ctx: The context of the call.
        """
        self._context = _ContextReader(ctx)
        self._operator = isinstance(ctx.context, self.OPERATORS)

    def fitted(self, recorded: MypyType) -> MypyType:
        """Give the type of a recorded call at its position.

        Args:
            recorded: The type that the recorder built.

        Returns:
            The declared type where the position declares an expression
            type, after the error of a code that does not fit it. Otherwise
            the recorded type.
        """
        spelled = self._context.declared()
        codes = DeferredArgument("row").codes(recorded)
        if spelled is None or codes is None:
            return recorded
        refusals = self._refusals(spelled, codes)
        if refusals is None:
            return recorded
        refusal = next((reason for reason in refusals if reason is not None), None)
        if refusal is not None and self._operator:
            return recorded
        if refusal is not None:
            self._context.refuse(refusal)
        return spelled

    def _refusals(
        self,
        spelled: MypyType,
        codes: Sequence[Code[MypyType]],
    ) -> list[str | None] | None:
        """Give the reason that each code does not fit the declared type, or `None` where it fits.

        Args:
            spelled: The declared type of the position.
            codes: The code of each member of the recorded type.

        Returns:
            One reason or `None` for each code, or `None` where the declared
            type is neither a selection nor an expression type.
        """
        argument = DeferredArgument("row")
        selection = SelectionReader().decoded(spelled)
        if selection is not None:
            return [argument.selection_verdict(code, selection) for code in codes]
        declaration = ExpressionArgument().decoded(spelled)
        if declaration is None:
            return None
        decl = SliceDecl.from_expression(declaration)
        return [DeferredFit().verdict(argument.declared(code), decl) for code in codes]
