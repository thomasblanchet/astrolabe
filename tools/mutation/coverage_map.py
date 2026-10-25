"""Map each line of the source to the tests that run it, so that a mutant runs only those.

The scope holds thousands of mutants, and a run of the whole suite for
each one takes far too long. The project keeps 100% branch coverage, so
one run of the suite with coverage contexts names at least one test for
every line that the harness can mutate. A coverage context records
which test ran a line. Each mutant then runs the few tests that reach
its line, and not the rest. This selection is what makes the audit
possible at all.

Turn on the contexts with the pytest-cov option `--cov-context=test`.
Do not set `dynamic_context` in `pyproject.toml`: pytest-cov refuses it
under xdist, and the project runs the suite with `-n auto`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from coverage import CoverageData

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class TestOrder:
    """Sort the tests of a mutant so that the cheapest run first.

    The runner stops at the first test that fails. The order therefore
    decides what a kill costs. It changes no outcome, because every
    selected test still runs if the mutant survives.

    A rank follows what a test must build. A doctest runs in the pytest
    process. A test of the engine builds nothing. A yml case builds one
    module against a warm cache of the libraries. An integration test
    runs a whole mypy build of its own, which takes seconds. The ranks
    go by the path of the test, and a mark in the name finds the tests
    that run a mypy build.

    Example:
        >>> TestOrder().rank("src/astrolabe/a.py::a.B")
        0
        >>> TestOrder().rank("tests/test_allow_unproven_integration.py::test_a")
        4
    """

    EXPENSIVE: ClassVar[tuple[str, ...]] = (
        "integration",
        "probes",
        "mypy_binder",
        "test_init.py",
    )
    RANKS: ClassVar[tuple[tuple[str, int], ...]] = (
        ("src/astrolabe/", 0),
        ("tests/engine/", 1),
        ("tests/hooks/", 1),
        ("spec/tests/", 1),
        ("tests/typing/", 2),
    )
    DEFAULT: ClassVar[int] = 3
    SLOWEST: ClassVar[int] = 4

    def arranged(self, node_ids: Iterable[str]) -> tuple[str, ...]:
        """Sort the tests of one line by what each costs.

        Args:
            node_ids: The tests that run a line.

        Returns:
            The same tests, cheapest first, and by name within one rank.

        Example:
            >>> TestOrder().arranged(["tests/typing/a.yml::c", "src/astrolabe/a.py::b"])
            ('src/astrolabe/a.py::b', 'tests/typing/a.yml::c')
        """
        return tuple(sorted(node_ids, key=lambda n: (self.rank(n), n)))

    def rank(self, node_id: str) -> int:
        """Rank one test by what it must build.

        A test that runs a mypy build of its own ranks last. Any other
        test ranks by the directory that holds it.

        Args:
            node_id: The pytest node id of the test.

        Returns:
            Zero for the cheapest, and four for a test that runs a full
            mypy build.

        Example:
            >>> TestOrder().rank("tests/typing/test_select.yml::a")
            2
            >>> TestOrder().rank("tests/hooks/test_a.py::test_b")
            1
            >>> TestOrder().rank("tests/test_config.py::test_a")
            3
        """
        if any(mark in node_id for mark in self.EXPENSIVE):
            return self.SLOWEST
        for prefix, rank in self.RANKS:
            if node_id.startswith(prefix):
                return rank
        return self.DEFAULT


class YamlCases:
    """Let only a yml case kill a mutant, because only a yml case checks what a user sees.

    A yml case runs the checker over a small program, and states the
    answer that a user reads: the revealed type, the error and its code.
    That answer is the contract of the checker. A doctest sits inside
    the code that it runs, so a change to that code carries its example
    along. A unit test or an integration run checks a class or a setting
    from inside the plugin. A mutant that only those tests catch is held
    by the code, and not by what the checker answers.

    By default a run holds every other test back. The option
    `--all-tests` lets them in, to measure how much of a score they
    carry.

    Example:
        >>> YamlCases().keeps("tests/typing/test_select.yml::a_case")
        True
        >>> YamlCases().keeps("tests/engine/test_a.py::test_b")
        False
    """

    MARK: ClassVar[str] = ".yml::"
    # A parity test runs the engine and the Python model of `spec/` on the
    # same input, and compares their answers. The model is the design
    # authority, so a parity test checks an answer of the checker too.
    PARITY: ClassVar[tuple[str, ...]] = (
        "tests/engine/test_expression_parity.py::",
        "tests/engine/test_remainder_parity.py::",
    )

    def keeps(self, node_id: str) -> bool:
        """Answer whether a test is a yml case or a parity test, and so can kill a mutant.

        Args:
            node_id: The pytest node id of the test.

        Returns:
            True for a yml case or a parity test.

        Example:
            >>> YamlCases().keeps("src/astrolabe/expr.py::astrolabe.expr.C")
            False
            >>> YamlCases().keeps("tests/engine/test_expression_parity.py::TestA::test_b")
            True
        """
        return self.MARK in node_id or node_id.startswith(self.PARITY)


class CoverageMap:
    """Answer which tests run one line of one file, from the coverage data of the baseline run.

    The map reads the data once, and keeps for each line the tests that
    can kill a mutant there, cheapest first. By default, only a yml case
    enters.

    Example:
        >>> CoverageMap.SEPARATOR
        '|'
    """

    SEPARATOR: ClassVar[str] = "|"

    def __init__(self, *, every_test: bool = False) -> None:
        """Build an empty map, with the rules that decide which tests it keeps.

        Args:
            every_test: If True, let any test kill a mutant, and not
                only a yml case.

        Example:
            >>> CoverageMap().files()
            ()
        """
        self.cases = YamlCases()
        self.every_test = every_test
        self.order = TestOrder()
        self._lines: dict[str, dict[int, tuple[str, ...]]] = {}
        self._whole: dict[str, tuple[str, ...]] = {}
        self._tested: dict[str, frozenset[int]] = {}

    def read(self, data_file: Path, root: Path) -> CoverageMap:
        """Read a coverage data file into the map.

        Args:
            data_file: The file that the baseline run wrote.
            root: The directory that the path of a mutant is relative
                to.

        Returns:
            This map, filled.

        Example:
            >>> CoverageMap().read.__name__
            'read'
        """
        data = CoverageData(basename=str(data_file))
        data.read()
        for measured in data.measured_files():
            self._add(data, measured, root)
        return self

    def files(self) -> tuple[str, ...]:
        """List the files that the map holds.

        Returns:
            Each path, relative to the root, sorted.

        Example:
            >>> CoverageMap().files()
            ()
        """
        return tuple(sorted(self._lines))

    def tests_for(self, path: Path, line: int) -> tuple[str, ...]:
        """Give the tests that run one line.

        A line that runs while pytest collects the modules, for example
        a class body or a default argument, records no test. For such a
        line, the map gives every test that runs any line of its file.
        That set holds every test that can catch the mutant, so no kill
        is lost. A line that some test runs, but no test that can kill a
        mutant, gets no test: `unreached` names that case.

        Args:
            path: The file, relative to the root.
            line: The line, counted from one.

        Returns:
            The node ids, cheapest first. Empty if the coverage data has
            no record of the file.

        Example:
            >>> from pathlib import Path
            >>> CoverageMap().tests_for(Path("a.py"), 1)
            ()
        """
        key = str(path)
        found = self._lines.get(key, {}).get(line, ())
        if found or self.unreached(path, line):
            return found
        return self._whole.get(key, ())

    def unreached(self, path: Path, line: int) -> bool:
        """Answer whether a test runs one line, and none of those tests can kill a mutant.

        A doctest or a unit test can run a line that no yml case runs.
        A mutant there has no test that can kill it. It must not fall
        back to every test of the file either, because none of them
        reaches the line, and the mutant then looks like a survivor.

        Args:
            path: The file, relative to the root.
            line: The line, counted from one.

        Returns:
            True if a test runs the line and the map keeps none of them.

        Example:
            >>> from pathlib import Path
            >>> CoverageMap().unreached(Path("a.py"), 1)
            False
        """
        key = str(path)
        if self._lines.get(key, {}).get(line):
            return False
        return line in self._tested.get(key, frozenset())

    def _add(self, data: CoverageData, measured: str, root: Path) -> None:
        """Read the contexts of one file into the map, by line and for the whole file.

        The call also notes each line that a test ran. A line that ran
        only while pytest collected the modules records the empty
        context, so it does not count.

        Args:
            data: The coverage data to read.
            measured: The absolute path that coverage recorded.
            root: The directory that the path of a mutant is relative
                to.

        Example:
            >>> CoverageMap()._add.__name__
            '_add'
        """
        key = self._relative(measured, root)
        if key is None:
            return
        by_line = data.contexts_by_lineno(measured)
        self._lines[key] = {n: self._node_ids(c) for n, c in by_line.items()}
        every_context: list[str] = []
        tested: set[int] = set()
        for line, contexts in by_line.items():
            every_context.extend(contexts)
            if any(self._node_id(context) for context in contexts):
                tested.add(line)
        self._whole[key] = self._node_ids(every_context)
        self._tested[key] = frozenset(tested)

    def _relative(self, measured: str, root: Path) -> str | None:
        """Give a path that coverage recorded in the form that a mutant uses, relative to the root.

        Args:
            measured: The absolute path that coverage recorded.
            root: The directory that the path of a mutant is relative
                to.

        Returns:
            The relative path, or None for a file outside the root.

        Example:
            >>> from pathlib import Path
            >>> CoverageMap()._relative("/a/b/c.py", Path("/x")) is None
            True
        """
        try:
            return str(root.joinpath(measured).resolve().relative_to(root.resolve()))
        except ValueError:
            return None

    def _node_ids(self, contexts: Iterable[str]) -> tuple[str, ...]:
        """Turn coverage contexts into the pytest node ids of the tests that can kill a mutant.

        pytest-cov writes one context for each phase of a test:
        `<node id>|setup`, `<node id>|run` and `<node id>|teardown`. A
        line that ran while pytest collected the modules records the
        empty context.

        Args:
            contexts: The contexts that coverage recorded.

        Returns:
            The node ids, with no repeats, cheapest first.

        Example:
            >>> CoverageMap()._node_ids(["t.yml::a|run", "t.yml::a|setup", ""])
            ('t.yml::a',)
        """
        found = {self._node_id(context) for context in contexts}
        return self.order.arranged(n for n in found if self._selectable(n))

    def _node_id(self, context: str) -> str:
        """Cut the phase that pytest-cov adds to a context, and keep the node id.

        Args:
            context: One context.

        Returns:
            The node id alone.

        Example:
            >>> CoverageMap()._node_id("t.py::a|run")
            't.py::a'
        """
        return context.rsplit(self.SEPARATOR, 1)[0]

    def _selectable(self, node_id: str) -> bool:
        """Answer whether a test can kill a mutant in this run.

        Args:
            node_id: The node id, empty for the collection of the
                modules.

        Returns:
            True for a real test that is a yml case, or for any real test
            where every test can kill.

        Example:
            >>> CoverageMap()._selectable("")
            False
            >>> CoverageMap()._selectable("tests/engine/test_a.py::test_b")
            False
        """
        if not node_id:
            return False
        return self.every_test or self.cases.keeps(node_id)
