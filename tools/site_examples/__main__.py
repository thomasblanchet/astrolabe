"""Let a maintainer check the examples of the website with `python -m tools.site_examples`.

The module holds no logic of its own. pytest with `--doctest-modules`
does not collect a file named `__main__.py`, so the command lives in
`command.py`, where its examples run with the other tests.
"""

from __future__ import annotations

from tools.site_examples.command import SiteExamplesCommand

__all__ = ["SiteExamplesCommand"]

if __name__ == "__main__":  # pragma: no cover  # pytest imports the module and never runs it
    raise SystemExit(SiteExamplesCommand.main())
