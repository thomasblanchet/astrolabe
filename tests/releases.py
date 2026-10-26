"""Say whether the installed release of polars is older than a version, for a test of some releases.

The answer of some tests depends on the release of polars, because the
checker follows the rules of the installed release and polars adds methods.
Such a test holds for a range of releases. A `.yml` case leaves out the
other releases with its `skip:` key:

    skip: __import__("tests.releases").releases.polars_before("1.43.0")

pytest-mypy-plugins evaluates the key, and the runtime cross-check
evaluates it in the same way (`YamlCorpus`), so a case that a release
leaves out runs in neither.
"""

from __future__ import annotations

from astrolabe._rules.releases import PolarsRelease


def polars_before(version: str) -> bool:
    """Say whether the installed release of polars is older than a version.

    Args:
        version: The version, for example `"1.43.0"`.

    Returns:
        Whether the installed release is older.

    Example:
        >>> polars_before("1.0.0")
        False
    """
    return PolarsRelease.installed() < PolarsRelease.parse(version)
