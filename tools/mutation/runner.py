"""Apply one mutant to a private copy of the project, run its tests, and read the outcome.

The harness never changes the `src/` of the project itself. Several
workers run at the same time, and each one needs a source that holds
only its own mutant. A run that stops halfway through a patch must also
leave the project as it was. So each worker owns a copy of the project.
It patches one file of its copy, runs the tests, and puts the file back.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from cosmic_ray.mutating import mutate_code
from cosmic_ray.plugins import get_operator

# The doctests of `mutation` and `classify` build a `Mutant`, so the import must run at run time.
from tools.mutation.mutants import Mutant  # ruff: ignore[typing-only-first-party-import]
from tools.mutation.pytest_call import PytestCall

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class Outcome(Enum):
    """Name what happened to one mutant.

    Only `KILLED` and `SURVIVED` measure the tests. `TIMED_OUT` means
    that the tests ran out of time while they still made progress, and
    `HUNG` that they stopped in one test until the limit, which points
    at a loop with no exit. `UNSELECTED` means that no test reaches the
    line, and `UNREACHED` that only tests which cannot kill a mutant
    reach it, for example a doctest. `UNCHANGED` means that the operator
    made no change, and `ERRORED` that the operator raised an exception.

    Example:
        >>> Outcome.KILLED.value
        'killed'
    """

    KILLED = "killed"
    SURVIVED = "survived"
    TIMED_OUT = "timed out"
    HUNG = "hung"
    UNSELECTED = "unselected"
    UNREACHED = "unreached"
    UNCHANGED = "unchanged"
    ERRORED = "errored"


@dataclass(frozen=True)
class MutantResult:
    """Hold one mutant, its outcome, the test that killed it, and how long its tests ran.

    Example:
        >>> MutantResult(None, Outcome.KILLED, "t.py::a", 1.5).scored()
        True
    """

    mutant: Mutant | None
    outcome: Outcome
    killer: str | None
    seconds: float

    def scored(self) -> bool:
        """Answer whether the result counts towards a score.

        A mutant measures nothing if no test reaches it, if the operator
        made no change or raised, or if its tests ran out of time.

        Returns:
            True for a killed or a surviving mutant.

        Example:
            >>> MutantResult(None, Outcome.UNCHANGED, None, 0.0).scored()
            False
        """
        return self.outcome in {Outcome.KILLED, Outcome.SURVIVED}


@dataclass(frozen=True)
class Mutation:
    """Hold the text of one mutated file, or the outcome to record when there is none.

    Attributes:
        text: The mutated text of the file, or None if the operator
            gave no change.
        refusal: The outcome to record if `text` is None.

    Example:
        >>> Mutation("x = 2", Outcome.UNCHANGED).text
        'x = 2'
    """

    text: str | None
    refusal: Outcome


class Worker:
    """Own one private copy of the project, for one thread of the run.

    A worker patches files of its own copy, so several workers can run
    at the same time with no collision, and the real source never
    changes. The copy holds the source, the tests and the settings that
    the tests read. It leaves out the Lean build directory, which holds
    several gigabytes, and the caches. A mypy cache built from other
    source does not see a change to the body of a hook, so a warm cache
    lets a mutant survive.

    Example:
        >>> "src" in Worker.COPIED
        True
    """

    COPIED: ClassVar[tuple[str, ...]] = ("src", "tests", "spec", "conftest.py", "pyproject.toml")
    IGNORED: ClassVar[tuple[str, ...]] = ("__pycache__", ".mypy_cache", ".lake", "*.pyc")

    PACKAGE: ClassVar[str] = "astrolabe"

    def __init__(self, root: Path, home: Path, package: str = PACKAGE) -> None:
        """Name the directory of the copy. The copy itself comes later, in `prepare`.

        Args:
            root: The project directory to copy.
            home: The directory for the copy.
            package: The package that the tests must import from the
                copy.

        Example:
            >>> from pathlib import Path
            >>> Worker(Path("."), Path("/tmp/w")).source.name
            'src'
        """
        self.root = root
        self.home = home
        self.package = package
        self.source = home / "src"

    def prepare(self) -> None:
        """Make the copy, and delete an older copy in the same place first.

        The copy leaves out the caches, and it skips an entry of
        `COPIED` that the project does not have.

        Example:
            >>> Worker.prepare.__name__
            'prepare'
        """
        if self.home.exists():
            shutil.rmtree(self.home)
        self.home.mkdir(parents=True)
        ignored = shutil.ignore_patterns(*self.IGNORED)
        for name in self.COPIED:
            origin = self.root / name
            if not origin.exists():
                continue
            if origin.is_dir():
                shutil.copytree(origin, self.home / name, ignore=ignored)
            else:
                shutil.copy2(origin, self.home / name)

    def verify(self) -> None:
        """Check that the tests of this worker import the package from its copy.

        A worker that imports the real `src` of the project never sees a
        mutant. It then reports every mutant as survived, and the whole
        run says nothing.

        Raises:
            RuntimeError: If the import reaches another tree.

        Example:
            >>> Worker.verify.__name__
            'verify'
        """
        name = self.package
        script = f"import {name}, sys; sys.stdout.write({name}.__file__)"
        done = PytestCall().probe(script, self.home, self.source)
        if not done.startswith(str(self.source)):
            message = f"the worker imports {done}, not the copy under {self.source}"
            raise RuntimeError(message)

    def read(self, relative: Path) -> str:
        """Read one file of the copy.

        Args:
            relative: The path, relative to the root of the project.

        Returns:
            The text of the file.

        Example:
            >>> Worker.read.__name__
            'read'
        """
        return (self.home / relative).read_text(encoding="utf-8")

    def write(self, relative: Path, text: str) -> None:
        """Write one file of the copy, to patch it or to put it back.

        Args:
            relative: The path, relative to the root of the project.
            text: The text to write.

        Example:
            >>> Worker.write.__name__
            'write'
        """
        (self.home / relative).write_text(text, encoding="utf-8")


class MutantRunner:
    """Apply one mutant to a copy, run the tests that reach it, and give its outcome.

    One time limit bounds the tests of each mutant, and it is the same
    number for the whole run. A limit that grows with the slowest run so
    far lets one slow selection raise the cost of every mutant after it,
    and over a long audit that waste adds up.

    Example:
        >>> MutantRunner().LIMIT
        180.0
    """

    LIMIT: ClassVar[float] = 180.0

    def __init__(self, calls: PytestCall | None = None, limit: float = LIMIT) -> None:
        """Set the pytest call and the time limit of the runner.

        Args:
            calls: The object that runs pytest. A new one if None.
            limit: How long the tests of one mutant can run, in seconds.

        Example:
            >>> MutantRunner().timeout()
            180.0
        """
        self.calls = calls or PytestCall()
        self.limit = limit
        self._operators: dict[str, Any] = {}

    def timeout(self) -> float:
        """Give the time limit for the tests of one mutant.

        Returns:
            The limit in seconds.

        Example:
            >>> MutantRunner(limit=90.0).timeout()
            90.0
        """
        return self.limit

    def operator(self, name: str) -> Any:
        """Give the operator object for a name, and build it only the first time.

        Args:
            name: The name of the operator.

        Returns:
            The operator object.

        Example:
            >>> MutantRunner().operator("core/AddNot") is not None
            True
        """
        if name not in self._operators:
            self._operators[name] = get_operator(name)()
        return self._operators[name]

    def run(self, worker: Worker, mutant: Mutant, tests: Sequence[str]) -> MutantResult:
        """Run one mutant against its tests, or give the reason why it did not run.

        The call patches the copy, runs the tests, and puts the original
        file back, even if the run fails.

        Args:
            worker: The copy to patch.
            mutant: The change to make.
            tests: The node ids of the tests that run the line of the
                mutant.

        Returns:
            The outcome, the test that killed the mutant, and the time
            that the tests took.

        Example:
            >>> MutantRunner.run.__name__
            'run'
        """
        if not tests:
            return MutantResult(mutant, Outcome.UNSELECTED, None, 0.0)
        original = worker.read(mutant.path)
        mutation = self.mutation(original, mutant)
        if mutation.text is None:
            return MutantResult(mutant, mutation.refusal, None, 0.0)
        started = time.monotonic()
        worker.write(mutant.path, mutation.text)
        try:
            done = self.calls.run(tests, worker.home, worker.source, self.timeout())
        finally:
            worker.write(mutant.path, original)
        return self.classify(mutant, done, time.monotonic() - started)

    def mutation(self, original: str, mutant: Mutant) -> Mutation:
        """Build the text of the file with the mutant applied.

        The operators come from cosmic-ray, and one of them raises on a
        node shape that it does not read: `ExceptionReplacer` raises on
        a dotted name in an `except` tuple. A defect in one operator must
        not end a run of several hours, so the mutant gets the outcome
        `ERRORED` instead.

        Args:
            original: The text of the file.
            mutant: The change to make.

        Returns:
            The mutated text, or the outcome to record if there is none.

        Example:
            >>> from pathlib import Path
            >>> runner = MutantRunner()
            >>> mutant = Mutant(Path("m.py"), "core/AddNot", 9, 1)
            >>> runner.mutation("x = 1", mutant).refusal
            <Outcome.UNCHANGED: 'unchanged'>
        """
        try:
            text = mutate_code(original, self.operator(mutant.operator), mutant.occurrence)
        except Exception:
            return Mutation(None, Outcome.ERRORED)
        if text is None or text == original:
            return Mutation(None, Outcome.UNCHANGED)
        return Mutation(text, Outcome.UNCHANGED)

    def classify(self, mutant: Mutant, done: Any, seconds: float) -> MutantResult:
        """Turn the answer of one pytest run into the outcome of the mutant.

        A run that passes the limit gets an outcome of its own, and it
        does not count towards a score. A selection can hold hundreds of
        tests and take minutes, so a run out of time does not show that
        a test caught the mutant. That outcome is `HUNG` if the run was
        stuck in one test, else `TIMED_OUT`.

        Args:
            mutant: The change that the run tested.
            done: The answer of the pytest run.
            seconds: How long the run took.

        Returns:
            The outcome, and the test that killed the mutant.

        Example:
            >>> from pathlib import Path
            >>> from tools.mutation.pytest_call import CallResult
            >>> mutant = Mutant(Path("m.py"), "core/AddNot", 0, 1)
            >>> stuck = CallResult(-1, "", timed_out=True, stalled=True)
            >>> MutantRunner().classify(mutant, stuck, 1.0).outcome
            <Outcome.HUNG: 'hung'>
        """
        if done.timed_out:
            undecided = Outcome.HUNG if done.stalled else Outcome.TIMED_OUT
            return MutantResult(mutant, undecided, None, seconds)
        if not done.failed():
            return MutantResult(mutant, Outcome.SURVIVED, None, seconds)
        return MutantResult(mutant, Outcome.KILLED, self.killer(done.output), seconds)

    def killer(self, output: str) -> str | None:
        """Read the first test that failed from the short summary of pytest.

        Args:
            output: Everything that pytest wrote.

        Returns:
            The node id on the first `FAILED` or `ERROR` line, or None
            if the summary names no test.

        Example:
            >>> MutantRunner().killer("FAILED t.py::a - boom")
            't.py::a'
            >>> MutantRunner().killer("ERROR t.py::a")
            't.py::a'
            >>> MutantRunner().killer("all good") is None
            True
        """
        for line in output.splitlines():
            for mark in ("FAILED ", "ERROR "):
                if line.startswith(mark):
                    return line[len(mark) :].split(" - ", maxsplit=1)[0].strip()
        return None
