"""Let a maintainer start the mutation harness with `python -m tools.mutation`.

The module holds no logic of its own. pytest with `--doctest-modules`
does not collect a file named `__main__.py`, so an example here never
runs. The command therefore lives in `command.py`, where its examples
run with the other tests.
"""

from __future__ import annotations

from tools.mutation.command import MutationCommand

__all__ = ["MutationCommand"]

if __name__ == "__main__":  # pragma: no cover  # pytest imports the module and never runs it
    raise SystemExit(MutationCommand.main())
