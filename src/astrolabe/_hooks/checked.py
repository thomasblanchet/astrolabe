"""Stop at once when a fact that the plugin's own logic guarantees turns out false.

Sometimes the code knows more than mypy can see. For example, an earlier
step already made sure that a value is not `None`, but mypy cannot follow
that step and still types the value as optional. A bare `assert` narrows
the type, but `python -O` deletes it. The two functions here narrow the
type and raise `TypeError` when the fact is false.

A failure here means a bug in the plugin, not a user input that did not
match. A value that a user input can make absent or wrong is not an
invariant. Code that handles such a value uses the fallible chains in
`_hooks/propagate.py`, which give up and fall back without a crash.
"""

from __future__ import annotations


class _UnexpectedTypeError(TypeError):
    def __init__(self, value: object, cls: type) -> None:
        super().__init__(f"expected {cls.__name__}, got {type(value).__name__}")


class _UnexpectedNoneError(TypeError):
    def __init__(self) -> None:
        super().__init__("expected a value, got None")


def checked_cast[T](value: object, cls: type[T]) -> T:
    """Give back a value as an instance of a class that the code knows it has.

    Use it where earlier logic proves the class of the value and mypy
    cannot see the proof.

    Args:
        value: The value whose class the code already knows.
        cls: The class that the value must have.

    Returns:
        The same value, which mypy now types as an instance of `cls`.

    Raises:
        _UnexpectedTypeError: If the value is not an instance of `cls`.
            This is a bug in the plugin.

    Example:
        >>> checked_cast(5, int)
        5
        >>> checked_cast("not an int", int)
        Traceback (most recent call last):
            ...
        astrolabe._hooks.checked._UnexpectedTypeError: expected int, got str
    """
    if not isinstance(value, cls):
        raise _UnexpectedTypeError(value, cls)
    return value


def checked_not_none[T](value: T | None) -> T:
    """Give back an optional value that the code knows is present.

    Use it where earlier logic proves that the value is not `None` and
    mypy cannot see the proof.

    Args:
        value: The value that the code knows is not `None`.

    Returns:
        The same value, which mypy now types without `None`.

    Raises:
        _UnexpectedNoneError: If the value is `None`. This is a bug in
            the plugin.

    Example:
        >>> checked_not_none(5)
        5
        >>> checked_not_none(None)
        Traceback (most recent call last):
            ...
        astrolabe._hooks.checked._UnexpectedNoneError: expected a value, got None
    """
    if value is None:
        raise _UnexpectedNoneError
    return value
