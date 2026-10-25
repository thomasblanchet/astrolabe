"""Decide in one place whether a result that the checker cannot prove gets an error.

The checker gives up on a proof in many places. The user picks how much of that to accept with
the `allow_unproven` setting. This module holds the one class that applies the setting, so that
every place that gives up reports in the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.domain.deferral import CheckerDeferral
from astrolabe._hooks.propagate import Refusal
from astrolabe._hooks.unproven import Reason

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext, FunctionSigContext, MethodContext

    from astrolabe._hooks.errors import UnprovenError
    from astrolabe._hooks.unproven import UnprovenPolicy


class UnprovenGate:
    """Report one unproven result, unless the `allow_unproven` setting lets it pass.

    A hook that cannot prove its result gives up and names a reason: what the user wrote that
    stops the proof. Each reason has a level on one scale. The `allow_unproven` setting picks
    the highest level that passes, for all operations or for one operation. Every place that
    gives up shares this gate, so the setting means the same thing everywhere. A reported
    result gets a note that names the setting which lets it pass. The user then knows how to
    accept it without reading the plugin.

    The gate sits in the plugin layer. It reads the setting as an `UnprovenPolicy`, and it
    answers with an error through mypy, or with nothing.

    Example:
        >>> from astrolabe._hooks.errors import UnnamedReadError
        >>> from astrolabe._hooks.unproven import Level, UnprovenPolicy
        >>> gate = UnprovenGate(UnprovenPolicy(Level.STATED))
        >>> gate.report(None, UnnamedReadError("select")) is None
        True
    """

    def __init__(self, policy: UnprovenPolicy) -> None:
        """Keep the setting that every decision of this gate reads.

        Args:
            policy: The `allow_unproven` setting, already read from the configuration.
        """
        self._policy = policy

    def report(
        self,
        ctx: MethodContext | FunctionContext | FunctionSigContext,
        error: UnprovenError,
    ) -> Refusal | None:
        """Report one error and its note, unless the setting lets the error pass.

        A pass that mypy will repeat reports nothing: a call in it can hold a type that is not
        ready yet, and the next pass reports whatever error remains.

        Args:
            ctx: The mypy context of the call that gave up.
            error: The error to report. It names its reason and its operation.

        Returns:
            None if the setting lets the error pass. Otherwise, a refusal with the reason
            `REPORTED`. It tells the caller that the call already has an error, so that no
            later step reports the same call again.
        """
        if self._policy.allows(error.reason, error.operation):
            return None
        if CheckerDeferral(ctx.api).pending():
            return Refusal(Reason.REPORTED)
        error.fail_with_note(ctx, self._policy.allowance(error.reason, error.operation))
        return Refusal(Reason.REPORTED)
