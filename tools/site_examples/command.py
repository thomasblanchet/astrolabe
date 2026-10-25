"""The command line of the example checker: write the catalog, or check that it is up to date.

uv run python -m tools.site_examples
uv run python -m tools.site_examples --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from tools.site_examples.catalog import ExampleCatalog

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO


class SiteExamplesCommand:
    """Read the command line, then write the catalog of examples or check it.

    Example:
        >>> SiteExamplesCommand(sys.stdout, sys.stderr).NAME
        'tools.site_examples'
    """

    NAME: ClassVar[str] = "tools.site_examples"
    EXAMPLES: ClassVar[str] = "site/examples"
    OUTPUT: ClassVar[str] = "site/src/data/examples.json"

    def __init__(self, out: TextIO, err: TextIO) -> None:
        """Keep the two streams that the command writes to.

        Args:
            out: The stream for the result.
            err: The stream for a failure.

        Example:
            >>> SiteExamplesCommand(sys.stdout, sys.stderr)._err is sys.stderr
            True
        """
        self._out = out
        self._err = err

    @classmethod
    def main(cls) -> int:
        """Run the command with the arguments of this process.

        Returns:
            The exit code.

        Example:
            >>> SiteExamplesCommand.main.__name__
            'main'
        """
        return cls(sys.stdout, sys.stderr).run(sys.argv[1:])

    def run(self, argv: Sequence[str]) -> int:
        """Write the catalog, or with `--check`, tell whether it is up to date.

        Args:
            argv: The command line, without the program name.

        Returns:
            Zero, or one when `--check` finds the catalog out of date.

        Example:
            >>> SiteExamplesCommand(sys.stdout, sys.stderr).run.__name__
            'run'
        """
        args = self.parser().parse_args(list(argv))
        catalog = ExampleCatalog(Path(args.examples), Path(args.output))
        if args.check:
            if catalog.stale():
                rerun = f"uv run python -m {self.NAME}"
                self._err.write(f"{catalog.output} is out of date: run `{rerun}`.\n")
                return 1
            self._out.write(f"{catalog.output} is up to date.\n")
            return 0
        catalog.output.parent.mkdir(parents=True, exist_ok=True)
        catalog.output.write_text(catalog.build(), encoding="utf-8")
        self._out.write(f"Wrote {catalog.output}.\n")
        return 0

    def parser(self) -> argparse.ArgumentParser:
        """Describe the command line.

        Returns:
            The parser.

        Example:
            >>> SiteExamplesCommand(sys.stdout, sys.stderr).parser().parse_args([]).check
            False
        """
        parser = argparse.ArgumentParser(prog=f"python -m {self.NAME}")
        parser.add_argument(
            "--check",
            action="store_true",
            help="fail when the catalog is out of date",
        )
        parser.add_argument(
            "--examples",
            default=self.EXAMPLES,
            help="the directory of the example files",
        )
        parser.add_argument(
            "--output",
            default=self.OUTPUT,
            help="the catalog that the pages read",
        )
        return parser
