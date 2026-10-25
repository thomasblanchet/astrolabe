"""Check the code examples of the website with mypy, and keep their output for the pages.

Each example of the documentation is a Python file in `site/examples/`.
This tool runs mypy with the plugin over every file, and writes what mypy
reports into `site/src/data/examples.json`. The pages show that output:
the errors under their lines, and each revealed frame as a row of columns.
So every message on the site is a message that the checker gives.

Run it after a change to an example or to the checker:

    uv run python -m tools.site_examples

`--check` writes nothing. It fails when the file on disk differs from a
new run, for example after a change of a message of the checker.
"""

from __future__ import annotations
