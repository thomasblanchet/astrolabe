"""Run every mutant of a scope over several workers, each with its own copy of the project.

One worker runs its mutants one after another, because each mutant
patches a file of that worker's copy. Several workers run at the same
time, each in its own copy, so nothing that they do collides. Every
worker takes its next mutant from one shared queue, so that no worker
sits idle while another still has work.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, ClassVar

from tools.mutation.mutants import Mutant, MutantCatalog
from tools.mutation.runner import MutantResult, MutantRunner, Outcome, Worker

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from typing import TextIO

    from tools.mutation.coverage_map import CoverageMap
    from tools.mutation.store import ResultStore


class Progress:
    """Count the finished mutants, and show every `STEP` of them how far the run is.

    Several workers finish mutants at the same time, so a lock guards
    the count.

    Example:
        >>> Progress(None, 0).finished
        0
    """

    STEP: ClassVar[int] = 25

    def __init__(self, out: TextIO | None, total: int) -> None:
        """Start the count at zero.

        Args:
            out: The stream to write to, or None to show nothing.
            total: The count of mutants that the run must decide.

        Example:
            >>> Progress(None, 4).total
            4
        """
        self.out = out
        self.total = total
        self.finished = 0
        self._lock = threading.Lock()

    def step(self) -> None:
        """Count one finished mutant, and show the count if it reached a step.

        The count shows only if it is a multiple of `STEP` and a stream
        is set.

        Example:
            >>> Progress(None, 1).step()
        """
        with self._lock:
            self.finished += 1
            if self.out is None or self.finished % self.STEP:
                return
            print(f"  {self.finished}/{self.total}", file=self.out, flush=True)


class MutantQueue:
    """Give each mutant to whichever worker is free.

    A fixed share for each worker balances how many mutants it runs, and
    not what they cost. One mutant takes two seconds and another takes
    the whole limit. The worker that draws the expensive share then runs
    long after the others stop, and the run ends with one worker busy
    and the rest idle. A shared queue keeps every worker busy to the
    last mutant.

    Example:
        >>> from pathlib import Path
        >>> MutantQueue([]).take() is None
        True
    """

    def __init__(self, mutants: Sequence[Mutant]) -> None:
        """Keep the mutants that are still to run, in the order to run them.

        Args:
            mutants: The mutants, in the order to run them.

        Example:
            >>> MutantQueue([]).take() is None
            True
        """
        self._left = list(reversed(mutants))
        self._lock = threading.Lock()

    def take(self) -> Mutant | None:
        """Give the next mutant to one worker, or None if none is left.

        A lock keeps two workers from taking the same mutant.

        Returns:
            One mutant that no other worker holds, or None.

        Example:
            >>> from pathlib import Path
            >>> queue = MutantQueue([Mutant(Path("a.py"), "core/AddNot", 0, 4)])
            >>> queue.take().line
            4
            >>> queue.take() is None
            True
        """
        with self._lock:
            if not self._left:
                return None
            return self._left.pop()

    def each(self) -> Iterator[Mutant]:
        """Give the mutants that are left, one at a time, as a worker takes them.

        Returns:
            An iterator that ends when the queue is empty.

        Example:
            >>> from pathlib import Path
            >>> queue = MutantQueue([Mutant(Path("a.py"), "core/AddNot", 0, 4)])
            >>> [m.line for m in queue.each()]
            [4]
        """
        return iter(self.take, None)


class Sweep:
    """List the mutants of a scope, and run the ones that are not decided yet.

    A sweep joins the parts of a run. The catalog lists the mutants, the
    coverage map gives the tests of each, the store gives what an
    earlier run finished, and the workers run the rest from one queue.
    Each result goes to the store as soon as it comes in.

    Example:
        >>> Sweep(None).pending([])
        []
    """

    def __init__(
        self,
        coverage: CoverageMap | None,
        catalog: MutantCatalog | None = None,
        package: str = Worker.PACKAGE,
        store: ResultStore | None = None,
        limit: float = MutantRunner.LIMIT,
    ) -> None:
        """Keep the parts that the sweep joins.

        Args:
            coverage: The map from each line to the tests that run it,
                or None to select no test.
            catalog: The object that lists the mutants. A new one if
                None.
            package: The package that the tests must import from each
                copy.
            store: The file of results, or None to keep nothing.
            limit: How long the tests of one mutant can run, in seconds.

        Example:
            >>> isinstance(Sweep(None).catalog, MutantCatalog)
            True
        """
        self.coverage = coverage
        self.catalog = catalog or MutantCatalog()
        self.package = package
        self.store = store
        self.limit = limit

    def mutants(self, scope: Path) -> list[Mutant]:
        """List the mutants of a scope.

        Args:
            scope: The directory to mutate.

        Returns:
            Every mutant that the filters keep.

        Example:
            >>> Sweep(None).mutants.__name__
            'mutants'
        """
        return self.catalog.mutants(scope)

    def execute(
        self,
        mutants: Sequence[Mutant],
        places: tuple[Path, Path],
        count: int,
        progress: Progress | None = None,
    ) -> list[MutantResult]:
        """Run every mutant that is not decided yet, over the workers, and give every result.

        If every mutant is decided, the sweep makes no copy at all.

        Args:
            mutants: The mutants of the run.
            places: The project directory beside the directory for the
                copies.
            count: How many workers to use.
            progress: The counter to step, or None.

        Returns:
            One result for each mutant: the held results first, then
            the new ones.

        Example:
            >>> Sweep(None).execute.__name__
            'execute'
        """
        held = self.held(mutants)
        pending = self.pending(mutants)
        if not pending:
            return held
        queue = MutantQueue(pending)
        workers = self._workers(places, count)
        with ThreadPoolExecutor(max_workers=count) as pool:
            done = pool.map(self._process, workers, [queue] * count, [progress] * count)
        found = list(held)
        for group in done:
            found.extend(group)
        return found

    def pending(self, mutants: Sequence[Mutant]) -> list[Mutant]:
        """List the mutants that no earlier run finished.

        Args:
            mutants: The mutants to look at.

        Returns:
            The mutants that are still to run, in their order.

        Example:
            >>> Sweep(None).pending([])
            []
        """
        return [mutant for mutant in mutants if self.finished(mutant) is None]

    def held(self, mutants: Sequence[Mutant]) -> list[MutantResult]:
        """List the results that an earlier run already wrote.

        Args:
            mutants: The mutants to look at.

        Returns:
            One result for each mutant that the store holds.

        Example:
            >>> Sweep(None).held([])
            []
        """
        found = [self.finished(mutant) for mutant in mutants]
        return [result for result in found if result is not None]

    def finished(self, mutant: Mutant) -> MutantResult | None:
        """Give the result that an earlier run wrote for one mutant.

        Args:
            mutant: The change to look for.

        Returns:
            The result, or None if the sweep has no store or no run
            finished the mutant.

        Example:
            >>> Sweep(None).finished(None) is None
            True
        """
        if self.store is None:
            return None
        return self.store.finished(mutant)

    def _process(
        self,
        worker: Worker,
        queue: MutantQueue,
        progress: Progress | None,
    ) -> list[MutantResult]:
        """Do the work of one worker: take mutants from the queue until none is left.

        Each worker has its own runner, so that no two threads share one.
        If only tests that cannot kill a mutant run its line, the mutant
        gets the outcome `UNREACHED` and does not run. Each result goes
        to the store at once, if the sweep has one, and steps the
        counter, if the sweep has one.

        Args:
            worker: The copy to patch.
            queue: The mutants that every worker shares.
            progress: The counter to step, or None.

        Returns:
            One result for each mutant that this worker took.

        Example:
            >>> Sweep(None)._process.__name__
            '_process'
        """
        runner = MutantRunner(limit=self.limit)
        found: list[MutantResult] = []
        for mutant in queue.each():
            if self._unreached(mutant):
                result = MutantResult(mutant, Outcome.UNREACHED, None, 0.0)
            else:
                result = runner.run(worker, mutant, self.tests(mutant))
            if self.store is not None:
                self.store.record(result)
            found.append(result)
            if progress is not None:
                progress.step()
        return found

    def _unreached(self, mutant: Mutant) -> bool:
        """Answer whether only tests that cannot kill a mutant run its line.

        Args:
            mutant: The change to make.

        Returns:
            True if the line runs only under such tests. False if the
            sweep has no coverage map.

        Example:
            >>> Sweep(None)._unreached(None)
            False
        """
        if self.coverage is None:
            return False
        return self.coverage.unreached(mutant.path, mutant.line)

    def tests(self, mutant: Mutant) -> tuple[str, ...]:
        """Find the tests that run the line of one mutant.

        Args:
            mutant: The change to make.

        Returns:
            The node ids, cheapest first. Empty if the sweep has no
            coverage map.

        Example:
            >>> Sweep(None).tests(None)
            ()
        """
        if self.coverage is None:
            return ()
        return self.coverage.tests_for(mutant.path, mutant.line)

    def _workers(self, places: tuple[Path, Path], count: int) -> list[Worker]:
        """Make one copy for each worker, and check each copy before any mutant runs.

        The check verifies that the tests of each copy import the package
        from that copy.

        Args:
            places: The project directory beside the directory for the
                copies.
            count: How many workers to make.

        Returns:
            The workers, each with a checked copy.

        Example:
            >>> Sweep(None)._workers.__name__
            '_workers'
        """
        root, home = places
        workers = [Worker(root, home / f"worker{index}", self.package) for index in range(count)]
        for worker in workers:
            worker.prepare()
            worker.verify()
        return workers
