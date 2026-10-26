"""Check the release that `astrolabe._rules.releases` reads against the installed polars.

The rules follow the installed release of polars, so the release that they
read must be the version that polars gives. A rule that changes between
releases must then run the variant of that release when no test sets another
one.
"""

from __future__ import annotations

import polars as pl

from astrolabe._rules.releases import PolarsRelease, polars_releases


class Suffix:
    """Give a suffix that changes at a release above every installed one, so each runs the variant.

    Example:
        >>> Suffix().of() in {"_newest", "_older"}
        True
    """

    @polars_releases
    def of(self) -> str:
        """Give the suffix of the newest releases.

        Returns:
            The suffix.

        Example:
            >>> with PolarsRelease.pretend("99.0.0"):
            ...     print(Suffix().of())
            _newest
        """
        return "_newest"

    @of.before("99.0.0")
    def _older(self) -> str:
        """Give the suffix of every release before 99.0.0.

        Returns:
            The suffix.

        Example:
            >>> Suffix()._older()
            '_older'
        """
        return "_older"


def test_the_installed_release_is_the_version_of_polars() -> None:
    """Read the version that polars gives."""
    assert str(PolarsRelease.installed()) == pl.__version__


def test_the_installed_release_runs_its_variant() -> None:
    """Run the variant of the installed release when no test sets another one."""
    assert PolarsRelease.current() == PolarsRelease.installed()
    assert Suffix().of() == "_older"
