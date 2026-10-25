"""Check how a target of the plugin compares, where a hook registers by full name.

A `Target` names a function or a method of polars or of astrolabe by its
full name. The plugin looks a hook up by that name, so two targets are equal
exactly when their full names are, and a target is never equal to a value
of another class, not even to the text of its own full name.
"""

from __future__ import annotations

import polars as pl

from astrolabe._hooks.dispatch.generic import Target


def test_a_target_is_not_equal_to_its_full_name_as_text() -> None:
    """Python asks the other side, and the text then compares as unequal."""
    target = Target.of(pl.Int64)
    assert target == Target(target.fullname)
    assert target != target.fullname
