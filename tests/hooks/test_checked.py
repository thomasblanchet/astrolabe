"""Check that the guards of the facts that the plugin guarantees raise when a fact is false.

`checked_cast` and `checked_not_none` narrow a type where an earlier step of
the plugin already proved the fact. A false fact is a bug in the plugin, so
each guard raises `TypeError` with a message that names what it expected.
"""

from __future__ import annotations

import pytest

from astrolabe._hooks.checked import checked_cast, checked_not_none


def test_a_value_of_another_class_raises() -> None:
    """Name the class that the code expected and the class of the value."""
    with pytest.raises(TypeError, match="expected int, got str"):
        checked_cast("a", int)


def test_a_missing_value_raises() -> None:
    """Say that the code expected a value."""
    with pytest.raises(TypeError, match="expected a value, got None"):
        checked_not_none(None)
