"""Check each override of the definitions: it has a reason, and its probe still shows the behavior.

An override is a rule written for one node, where polars does what no part
of the vocabulary states (`astrolabe._rules.definitions.overrides`). Each one
names the node, says why no part of the vocabulary gives the behavior, and
holds a probe: a Python expression over the module `pl` that shows the
behavior. The test runs each probe, so a new version of polars that changes
the behavior fails here, and the override goes away.
"""

from __future__ import annotations

import polars as pl
import pytest

from spec.tests.rule_probes import PolarsAnswer, Stopped
from astrolabe._rules.definitions.overrides import Override, Overrides


@pytest.mark.parametrize("override", Overrides.ALL, ids=lambda override: override.node)
def test_override_has_a_reason_and_a_probe(override: Override) -> None:
    """Each override names its node and gives a reason of more than a few words, and a probe."""
    assert override.node
    assert len(override.reason.split()) > 10
    assert override.probe.startswith("pl.")


@pytest.mark.parametrize("override", Overrides.ALL, ids=lambda override: override.node)
def test_probe_of_an_override_stops_as_its_reason_says(override: Override) -> None:
    """The probe of each override panics or raises in polars, as its reason says."""
    namespace = {"pl": pl}
    probe = override.probe
    answer = PolarsAnswer().answer(
        lambda: eval(probe, namespace).dtypes[-1],  # ruff: ignore[suspicious-eval-usage]
    )
    assert answer == Stopped(override.stop, override.error)
