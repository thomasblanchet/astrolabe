"""Run pytest in a child process, with the options that a mutant run needs.

The `addopts` setting of the project turns on coverage, the gate that
fails a run under 100% coverage, and xdist. A mutant run needs none of
them. Coverage costs time. The gate fails every run of a few tests, so
every mutant then looks killed by the gate and not by a test. This
module builds the call without those options.
"""

from __future__ import annotations

import os
import time

# The harness runs pytest in a child process, and builds each argument itself.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import IO


@dataclass(frozen=True)
class CallResult:
    """Hold what one pytest run gave back: its exit code, its output, and whether it timed out.

    A run that timed out is stalled if it wrote nothing for a long time
    before the limit. pytest writes one mark for each test that ends, so
    a run that stopped writing is stuck in one test, and not only slow.

    Example:
        >>> CallResult(1, "F", timed_out=False).failed()
        True
    """

    code: int
    output: str
    timed_out: bool
    stalled: bool = False

    def failed(self) -> bool:
        """Answer whether the run did not pass, so that the mutant can count as killed.

        Returns:
            True if pytest did not exit with zero, or ran out of time.

        Example:
            >>> CallResult(0, "", timed_out=False).failed()
            False
        """
        return self.timed_out or self.code != 0


class PytestCall:
    """Run a selection of tests in a copy of the project, and give back what pytest answered.

    The call keeps the options that the tests need. A yml case reads the
    `pyproject.toml` of the project, and the node id of a doctest needs
    `--doctest-modules`. It leaves out coverage and xdist.

    It keeps `--mypy-same-process`, which runs the mypy build of each
    yml case inside the pytest process. The harness cannot work without
    it. mypy in a process of its own reads the plugin from the installed
    package, and not from the patched copy. Every mutant of a hook then
    survives the yml case that catches it.

    That option makes one process hold the state of every case that it
    builds, and a thousand cases reach tens of gigabytes. The call
    therefore runs the selection in batches of `BATCH` tests. Each batch
    is a process of its own, which gives its memory back when it ends.

    Each run starts with a lower priority, so that the machine stays
    usable during a long audit.

    Example:
        >>> "--mypy-same-process" in PytestCall().KEPT
        True
    """

    KEPT: ClassVar[tuple[str, ...]] = (
        "--mypy-pyproject-toml-file=pyproject.toml",
        "--doctest-modules",
        "--mypy-same-process",
    )
    BATCH: ClassVar[int] = 40
    NICE: ClassVar[tuple[str, ...]] = ("/usr/bin/nice", "-n", "10")
    # No single test of the project runs this long. A run that wrote
    # nothing for this time is stuck in one test.
    STALL: ClassVar[float] = 600.0
    PROBE_LIMIT: ClassVar[float] = 60.0
    QUIET: ClassVar[tuple[str, ...]] = (
        "-p",
        "no:cacheprovider",
        "--no-cov",
        "-q",
        "--tb=no",
        "-rfE",
    )

    def arguments(self, targets: Sequence[str], *, stop_early: bool) -> list[str]:
        """Build the arguments that follow the Python interpreter on the command line.

        The call replaces `addopts` with the options that it keeps, and
        adds options that make the output short.

        Args:
            targets: The node ids or paths to run.
            stop_early: If True, stop at the first failure.

        Returns:
            The arguments for pytest.

        Example:
            >>> PytestCall().arguments(["a.py"], stop_early=True)[:3]
            ['-m', 'pytest', '-o']
        """
        options = ["-o", f"addopts={' '.join(self.KEPT)}", *self.QUIET]
        early = ["-x"] if stop_early else []
        return ["-m", "pytest", *options, *early, *targets]

    def run(
        self,
        targets: Sequence[str],
        cwd: Path,
        source: Path,
        timeout: float,
        *,
        stop_early: bool = True,
    ) -> CallResult:
        """Run a selection in batches, and stop at the first batch that fails.

        One limit covers the whole selection, and not each batch. Each
        batch gets the time that the selection has left. If no time is
        left, the selection counts as a run out of time.

        Args:
            targets: The node ids to run.
            cwd: The directory to run in.
            source: The `src` directory to import the package from.
            timeout: The limit for the whole selection, in seconds.
            stop_early: If True, stop each batch at its first failure.

        Returns:
            The answer of the first batch that fails, a run out of time
            if the limit ends between two batches, or a passing answer.

        Example:
            >>> PytestCall().run.__name__
            'run'
        """
        deadline = time.monotonic() + timeout
        for batch in self.batches(targets):
            left = deadline - time.monotonic()
            if left <= 0:
                return CallResult(-1, "", timed_out=True)
            done = self._once(batch, cwd, source, left, stop_early=stop_early)
            if done.failed():
                return done
        return CallResult(0, "", timed_out=False)

    def batches(self, targets: Sequence[str]) -> list[Sequence[str]]:
        """Cut a selection into batches small enough for the memory of one process.

        Args:
            targets: The node ids to run.

        Returns:
            The node ids in groups of at most `BATCH`, in their order.

        Example:
            >>> PytestCall().batches(["a", "b"])
            [['a', 'b']]
        """
        size = self.BATCH
        return [list(targets[at : at + size]) for at in range(0, len(targets), size)]

    def _once(
        self,
        targets: Sequence[str],
        cwd: Path,
        source: Path,
        timeout: float,
        *,
        stop_early: bool,
    ) -> CallResult:
        """Run pytest once, with a lower priority, and read its answer.

        If the run passes its limit, the call stops the process. The
        run is stalled if it wrote nothing for `STALL` seconds before
        the limit.

        Args:
            targets: The node ids to run.
            cwd: The directory to run in.
            source: The `src` directory to import the package from.
            timeout: The limit in seconds.
            stop_early: If True, stop at the first failure.

        Returns:
            The exit code, the output, whether the limit ran out, and
            whether the run was stalled.

        Example:
            >>> PytestCall()._once.__name__
            '_once'
        """
        command = [*self.NICE, sys.executable, *self.arguments(targets, stop_early=stop_early)]
        # This class builds every argument. None comes from a user.
        process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            cwd=cwd,
            env=self.environment(source),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        watch = OutputWatch(checked_stream(process.stdout)).started()
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stalled = watch.idle() >= self.STALL
            process.kill()
            process.wait()
            return CallResult(-1, watch.finished(), timed_out=True, stalled=stalled)
        return CallResult(code, watch.finished(), timed_out=False)

    def probe(self, script: str, cwd: Path, source: Path) -> str:
        """Run one short script with the same import path as a test.

        A worker uses this before its first mutant, to find the tree
        that it imports the package from.

        Args:
            script: The Python source to run.
            cwd: The directory to run in.
            source: The `src` directory to import the package from.

        Returns:
            What the script wrote to its standard output.

        Example:
            >>> PytestCall().probe.__name__
            'probe'
        """
        # This class builds every argument. None comes from a user.
        done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, "-c", script],
            cwd=cwd,
            env=self.environment(source),
            capture_output=True,
            text=True,
            timeout=self.PROBE_LIMIT,
            check=False,
        )
        return done.stdout.strip()

    def environment(self, source: Path) -> dict[str, str]:
        """Build the environment that makes a child process import the package from one tree.

        The project is installed in editable mode, so the real `src`
        is on the import path of every process. `PYTHONPATH` comes
        before that entry, so the tree that it names wins. The
        environment also leaves out `COVERAGE_FILE`, and it sets
        `PYTHONUNBUFFERED`, so that each mark of pytest arrives when its
        test ends.

        Args:
            source: The `src` directory to import the package from.

        Returns:
            A copy of the environment of this process, with the path set.

        Example:
            >>> from pathlib import Path
            >>> PytestCall().environment(Path("/w/src"))["PYTHONPATH"]
            '/w/src'
        """
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(source)
        environment["PYTHONUNBUFFERED"] = "1"
        environment.pop("COVERAGE_FILE", None)
        return environment


def checked_stream(stream: IO[str] | None) -> IO[str]:
    """Give the output stream of a child process that was started with a pipe.

    Args:
        stream: The `stdout` of the process.

    Returns:
        The same stream.

    Raises:
        TypeError: If the process has no pipe.

    Example:
        >>> import io
        >>> checked_stream(io.StringIO("")).read()
        ''
    """
    if stream is None:
        msg = "the process was started without a pipe"
        raise TypeError(msg)
    return stream


class OutputWatch:
    """Read the output of a child process as it arrives, and note when the last text came.

    A reader thread takes the text in small pieces, so that the pipe
    never fills and the child never blocks on a write. The time of the
    last piece tells a stuck run from a slow one.

    Example:
        >>> import io
        >>> OutputWatch(io.StringIO("..")).started().finished()
        '..'
    """

    PIECE: ClassVar[int] = 1

    def __init__(self, stream: IO[str]) -> None:
        """Hold the stream to read. Nothing is read until `started`.

        Args:
            stream: The output of the child process.
        """
        self._stream = stream
        self._pieces: list[str] = []
        self._last = time.monotonic()
        self._thread = threading.Thread(target=self._pump, daemon=True)

    def started(self) -> OutputWatch:
        """Start the reader thread.

        Returns:
            This watch.

        Example:
            >>> import io
            >>> OutputWatch(io.StringIO("")).started().finished()
            ''
        """
        self._thread.start()
        return self

    def idle(self) -> float:
        """Give the time since the last text arrived.

        Returns:
            The time in seconds.

        Example:
            >>> import io
            >>> OutputWatch(io.StringIO("")).idle() < 60
            True
        """
        return time.monotonic() - self._last

    def finished(self) -> str:
        """Wait for the end of the stream, and give all the text that it held.

        Returns:
            The output of the child process.

        Example:
            >>> import io
            >>> OutputWatch(io.StringIO("a")).started().finished()
            'a'
        """
        self._thread.join()
        return "".join(self._pieces)

    def _pump(self) -> None:
        """Read the stream until it ends, and note the time of each piece.

        Example:
            >>> import io
            >>> watch = OutputWatch(io.StringIO("ab"))
            >>> watch._pump()
            >>> watch._pieces
            ['a', 'b']
        """
        for piece in iter(lambda: self._stream.read(self.PIECE), ""):
            self._pieces.append(piece)
            self._last = time.monotonic()
