"""Write a chain of steps that can each fail as flat code, with no `if` after each step.

A hook often builds its answer in several steps, and each step can fail.
For example, it reads the receiver frame, then a column name from the
call, then that column's data type. Written the usual way, each step is
followed by `if x is None: return None`. The method then holds one branch
for each step, and the guards hide the steps from the reader.

This module removes those branches. Decorate the method with a boundary,
`@propagates(default)` or `@explains(default_reason)`. Inside it, wrap
each value that can be missing in `not_none(x)` or `not_falsy(x)`. If the
value is present, the call gives it back with its type narrowed, so the
next line uses it directly. If the value is missing, the call raises a
private exception. The boundary catches that exception and returns its
failure result in place of the method's result. The method body reads as
a straight list of steps, with no `if` after each step.

The two boundaries differ in what a failure returns. `@propagates` gives
one fixed value, most often None, for code that needs only to know that
the chain failed. `@explains` gives a `Refusal` that holds a `Reason`,
for code that must tell the user why the checker gave up on a proof. A
step names its reason with `reason=`, and the boundary supplies a default
for a step that names none. `not_refused` lets one explaining method use
the result of another, so the inner reason reaches the outer caller
unchanged.

This module sits under the whole plugin. It imports only `Reason`, so
every layer can use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from astrolabe._hooks.unproven import Reason


class _Abort(Exception):
    """Carry a failed step out of a decorated method, up to its boundary.

    A step helper raises it, and only a boundary decorator catches it. It
    is private because no other code must catch it. A handler anywhere
    else can swallow it, and the chain then goes on after a failed step.

    Attributes:
        reason: Why the step failed, or None if the step names no reason.
    """

    def __init__(self, reason: Reason | None = None) -> None:
        """Keep the reason that the failed step gives.

        Args:
            reason: The step's reason, or None if the step names none.
        """
        super().__init__()
        self.reason = reason

    def reason_or(self, default: Reason) -> Reason:
        """Give the step's own reason, or the boundary's default if it has none.

        Args:
            default: The reason of the boundary, for a step that names none.

        Returns:
            The step's reason if it has one, else `default`.
        """
        if self.reason is None:
            return default
        return self.reason


@dataclass(frozen=True)
class Refusal:
    """Tell the caller of an `@explains` method that its chain failed, and why.

    The method returns this value in place of its normal result. The
    caller tests the result with `isinstance(result, Refusal)`, and it
    can then give the reason to the code that reports the unproven result.

    Attributes:
        reason: Why the chain failed.

    Example:
        >>> from astrolabe._hooks.unproven import Reason
        >>> Refusal(Reason.NO_SCHEMA).reason.wording
        'no schema gives the column types'
    """

    reason: Reason


def not_none[T](value: T | None, *, reason: Reason | None = None) -> T:
    """Give `value` back with its type narrowed, or stop the chain if it is None.

    Use it only inside a method with a boundary decorator. Outside one,
    the exception it raises reaches the caller.

    Args:
        value: The result of one step.
        reason: Why a missing value stops the proof, for an `@explains`
            method. If you omit it, the boundary gives its default reason.

    Returns:
        `value`, with None taken out of its type.

    Raises:
        _Abort: If `value` is None.

    Example:
        >>> not_none(5)
        5
    """
    if value is None:
        raise _Abort(reason)
    return value


def not_falsy[T](value: T | None, *, reason: Reason | None = None) -> T:
    """Give `value` back, or stop the chain if it is None or empty.

    Use it in place of `not_none` where an empty result is also a
    failure, for example an empty list, an empty string, zero or False.
    Use it only inside a method with a boundary decorator.

    Args:
        value: The result of one step.
        reason: Why a missing or empty value stops the proof, for an
            `@explains` method. If you omit it, the boundary gives its
            default reason.

    Returns:
        `value`, with None taken out of its type.

    Raises:
        _Abort: If `value` is None or another falsy value.

    Example:
        >>> not_falsy(5)
        5
    """
    if not value:
        raise _Abort(reason)
    return value


def not_refused[T](value: T | Refusal) -> T:
    """Give back the value of an inner `@explains` method, or stop with its refusal.

    An explaining method can call another one as one of its steps. If the
    inner method refused, this stops the outer chain with the inner
    reason, so the outer caller learns the real cause and not the outer
    default. Use it only inside a method with a boundary decorator.

    Args:
        value: The result of the inner method.

    Returns:
        `value`, with `Refusal` taken out of its type.

    Raises:
        _Abort: If `value` is a `Refusal`.

    Example:
        >>> not_refused([1, 2])
        [1, 2]
    """
    if isinstance(value, Refusal):
        raise _Abort(value.reason)
    return value


class propagates[D]:
    """Make a method return a fixed value if one of its steps fails.

    Use it for a chain whose caller needs only to know that it failed.
    `@propagates(None)` fits most methods. `@propagates(False)` fits a
    method that answers a boolean. The boundary ignores the reason that a
    failed step names. If the caller needs that reason, use `@explains`.

    Example:
        >>> @propagates(None)
        ... def first_positive(numbers: list[int]) -> int | None:
        ...     return not_none(next((n for n in numbers if n > 0), None))
        >>> first_positive([-1, -2, 3])
        3
        >>> first_positive([-1, -2])
    """

    def __init__(self, default: D) -> None:
        """Keep the value that the method returns on a failure.

        Args:
            default: The value to return if a step fails.
        """
        self._default = default

    def __call__[**P, T](self, f: Callable[P, T]) -> Callable[P, T | D]:
        """Wrap a method so that a failed step returns the default.

        Args:
            f: The method to wrap.

        Returns:
            A function with the signature of `f`. It returns the result
            of `f`, or the default if a step of `f` fails.
        """

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | D:
            try:
                return f(*args, **kwargs)
            except _Abort:
                return self._default

        return wrapper


class explains:
    """Make a method return a `Refusal` with a reason if one of its steps fails.

    Use it for a chain whose caller must say why the checker gave up on a
    proof, for example a hook that reports an unproven result and names
    the cause. The method declares only its value type. A failed step
    makes it return a `Refusal`. The refusal holds the reason that the
    failed step names, or the default reason given here if the step names
    none.

    Example:
        >>> from astrolabe._hooks.unproven import Reason
        >>> @explains(Reason.CHECKER_GAP)
        ... def column_names(names: list[str] | None, frame: str | None) -> list[str]:
        ...     not_none(frame)
        ...     return not_none(names, reason=Reason.DYNAMIC_NAME)
        >>> column_names(["a"], "trades")
        ['a']
        >>> column_names(None, "trades").reason.name
        'DYNAMIC_NAME'
        >>> column_names(["a"], None).reason.name
        'CHECKER_GAP'
    """

    def __init__(self, default: Reason) -> None:
        """Keep the reason to give for a failed step that names none.

        Args:
            default: The reason for a step with no reason of its own.
        """
        self._default = default

    def __call__[**P, T](self, f: Callable[P, T]) -> Callable[P, T | Refusal]:
        """Wrap a method so that a failed step returns a `Refusal`.

        Args:
            f: The method to wrap.

        Returns:
            A function with the signature of `f`. It returns the result
            of `f`, or a `Refusal` with the reason of the failed step.
        """

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | Refusal:
            try:
                return f(*args, **kwargs)
            except _Abort as abort:
                return Refusal(abort.reason_or(self._default))

        return wrapper
