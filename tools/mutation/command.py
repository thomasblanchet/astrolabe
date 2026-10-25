"""The command line of the mutation harness, which runs one audit from start to end.

The harness needs to know which tests reach each line of the source.
First run the test suite once with coverage contexts, which record the
test that ran each line:

    COVERAGE_FILE=.mutation/baseline.coverage uv run pytest --cov-context=test

Then run the harness with that coverage data:

    uv run python -m tools.mutation --data .mutation/baseline.coverage

A run writes the result of each mutant to `results.jsonl` in its home
directory, `.mutation` by default, as soon as the mutant is done. If a run stops, start it again
with `--resume`. It then reads that file and runs only the mutants that
are left.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from tools.mutation.coverage_map import CoverageMap
from tools.mutation.report import MutationReport
from tools.mutation.runner import MutantRunner, Worker
from tools.mutation.mutants import MutantCatalog
from tools.mutation.store import ResultStore, SourceDigests
from tools.mutation.sweep import Progress, Sweep

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO

    from tools.mutation.mutants import Mutant


class MutationCommand:
    """Read the command line, run every mutant of the scope, and write the report.

    The command joins the parts of the harness. It reads the coverage
    data into a map from each line to its tests, opens the file of
    results, lists the mutants, runs them over several workers, and
    writes the score. A surviving mutant is a finding and not a failure
    of the command, so the command exits with zero.

    The default count of workers is small on purpose. Each worker runs
    a full pytest process, and a worker that builds mypy holds hundreds
    of megabytes. The count follows the memory of one worker, and not
    the count of the cores. Measure what one worker holds on the machine
    that runs the audit before you raise the count.

    Example:
        >>> MutationCommand(sys.stdout, sys.stderr).NAME
        'tools.mutation'
    """

    NAME: ClassVar[str] = "tools.mutation"
    SCOPE: ClassVar[str] = "src/astrolabe"
    RESULTS: ClassVar[str] = "results.jsonl"
    WORKERS: ClassVar[int] = 4

    def __init__(self, out: TextIO, err: TextIO) -> None:
        """Keep the two streams that the command writes to.

        The report and the progress go to different streams, so that a
        maintainer can save the report to a file and still watch the run.

        Args:
            out: The stream for the report.
            err: The stream for the progress.

        Example:
            >>> MutationCommand(sys.stdout, sys.stderr)._err is sys.stderr
            True
        """
        self._out = out
        self._err = err

    @classmethod
    def main(cls) -> int:
        """Run the command with the arguments of this process, for `python -m tools.mutation`.

        Returns:
            The exit code.

        Example:
            >>> MutationCommand.main.__name__
            'main'
        """
        return cls(sys.stdout, sys.stderr).run(sys.argv[1:])

    def run(self, argv: Sequence[str]) -> int:
        """Run one audit, from the command line to the report.

        The run first reads the coverage data of the baseline run into a
        map from each line to its tests. It then opens the file of
        results. The options decide what an earlier run keeps:
        `--resume` keeps what it finished, `--retry-undecided` runs again
        each mutant whose tests ran out of time, and `--retry-survived`
        runs again each mutant that no yml case killed or reached.

        The file of results keeps a digest of each file that the run
        mutates. If a file changed before a resumed run, the run stops.
        A held result names its mutant by a count of positions in the
        file, and after a change that count can point at another change.

        Before the work starts, the run shows how much work it has on the
        progress stream.

        Args:
            argv: The command line, without the program name.

        Returns:
            Zero, because a surviving mutant is a finding and not a
            failure.

        Example:
            >>> MutationCommand(sys.stdout, sys.stderr).run.__name__
            'run'
        """
        args = self.parser().parse_args(list(argv))
        root = Path(args.root).resolve()
        home = Path(args.home).resolve()
        coverage_map = CoverageMap(every_test=bool(args.all_tests))
        coverage = coverage_map.read(Path(args.data), root)
        digests = SourceDigests().of(MutantCatalog().files(Path(args.scope)))
        store = ResultStore(home / self.RESULTS).begin(
            resume=bool(args.resume),
            retry=bool(args.retry_undecided),
            again=bool(args.retry_survived),
            limit=float(args.timeout),
            digests=digests,
        )
        sweep = Sweep(
            coverage,
            package=str(args.package),
            store=store,
            limit=float(args.timeout),
        )
        found = sweep.mutants(Path(args.scope))
        mutants = self.chosen(sweep, found, int(args.limit))
        pending = sweep.pending(mutants)
        workers = int(args.workers)
        print(f"{len(found)} mutants, {len(pending)} to run, {workers} workers", file=self._err)
        progress = Progress(self._err, len(pending))
        places = (root, home)
        results = sweep.execute(mutants, places, workers, progress)
        MutationReport(results).write(self._out)
        return 0

    def chosen(self, sweep: Sweep, mutants: Sequence[Mutant], limit: int) -> list[Mutant]:
        """Choose the mutants of this run: every decided one, and a spread of the rest.

        `--limit` lets a maintainer run a sample of the scope. The limit
        counts the mutants that are still to decide, and not all the
        mutants of the scope. A resumed run therefore takes its sample
        from what is left. Beside `--resume`, a sample of the whole scope
        picks mutants that are already done, and adds no work.

        Args:
            sweep: The sweep, which knows what an earlier run finished.
            mutants: Every mutant of the scope.
            limit: How many mutants to run, or zero for all of them.

        Returns:
            The mutants already decided, then the sample to run.

        Example:
            >>> MutationCommand(sys.stdout, sys.stderr).chosen(Sweep(None), [], 0)
            []
        """
        decided = [m for m in mutants if sweep.finished(m) is not None]
        return decided + self.spread(sweep.pending(mutants), limit)

    def spread(self, mutants: Sequence[Mutant], limit: int) -> list[Mutant]:
        """Take a sample spread evenly over a list, so that it reaches every file.

        The mutants come sorted by file. A sample from the head of the
        list covers the first files only.

        Args:
            mutants: The mutants to choose from.
            limit: How many to keep, or zero for all of them.

        Returns:
            The chosen mutants, in their order in the list.

        Example:
            >>> MutationCommand(sys.stdout, sys.stderr).spread([1, 2, 3, 4], 2)
            [1, 3]
        """
        if limit <= 0 or limit >= len(mutants):
            return list(mutants)
        step = len(mutants) // limit
        return list(mutants[::step][:limit])

    def parser(self) -> argparse.ArgumentParser:
        """Build the reader of the command line, with a default for each argument.

        Returns:
            The parser.

        Example:
            >>> args = MutationCommand(sys.stdout, sys.stderr).parser()
            >>> args.parse_args(["--data", "d"]).workers
            4
        """
        parser = argparse.ArgumentParser(prog=self.NAME, description="Run the mutation harness.")
        parser.add_argument("--data", required=True, help="the baseline's coverage data file")
        parser.add_argument("--scope", default=self.SCOPE, help="the directory to mutate")
        parser.add_argument("--root", default=".", help="the project directory")
        parser.add_argument("--home", default=".mutation", help="where the copies go")
        parser.add_argument(
            "--workers",
            type=int,
            default=self.WORKERS,
            help="how many copies run at the same time",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="run a spread of this many of the mutants still to decide",
        )
        parser.add_argument(
            "--package",
            default=Worker.PACKAGE,
            help="the package each copy must supply",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="keep what an earlier run in the same home finished",
        )
        parser.add_argument(
            "--all-tests",
            action="store_true",
            help="let any test kill a mutant, and not only a yml case",
        )
        parser.add_argument(
            "--retry-survived",
            action="store_true",
            help="run each mutant that no yml case killed or reached again, after new yml cases",
        )
        parser.add_argument(
            "--retry-undecided",
            action="store_true",
            help="run each mutant that ran out of time again",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=MutantRunner.LIMIT,
            help="how long one mutant's tests can run, in seconds",
        )
        return parser
