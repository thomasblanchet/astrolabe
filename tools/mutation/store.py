"""Write down each finished mutant at once, so that a run that stops can go on later.

A full audit runs for hours. Without a store, a run that stops loses
every mutant that it finished. The store writes one line for each
result as soon as it comes in. A later run reads those lines, and runs
only the mutants that are left.

The file holds JSON lines, one object for each mutant, so a maintainer
can read the results of a run while it is still going.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from tools.mutation.mutants import Mutant
from tools.mutation.runner import MutantResult, Outcome

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class StaleStore(Exception):
    """Stop a resumed run, because the store holds results for source that has changed since.

    A result names its mutant by a path, an operator, and the index of
    the change among the positions of that operator in the file. One new
    position in a file moves every position after it. A held result then
    describes another change, and its line number names a line that has
    moved. Nothing in the file of results shows this, so a run that
    reuses it reports wrong results with no warning.
    """

    def __init__(self, changed: Sequence[str]) -> None:
        """Build the message, which names each file that changed.

        Args:
            changed: The paths whose content differs from the digest in
                the store.

        Example:
            >>> str(StaleStore(["a.py", "b.py"]))
            'the source changed since the store was written: a.py, b.py'
        """
        super().__init__(f"the source changed since the store was written: {', '.join(changed)}")


class SourceDigests:
    """Take a digest of each file that a run mutates, to find a change in the source later.

    A run writes these digests beside its results. A resumed run
    compares them with the source, and stops if a file has changed. A
    stopped audit can therefore go on only over the source that it
    measured.

    Example:
        >>> SourceDigests().changed({"a.py": "x"}, {"a.py": "x"})
        []
    """

    def of(self, paths: Iterable[Path]) -> dict[str, str]:
        """Take the digest of every file of a scope.

        Args:
            paths: The files that a run mutates.

        Returns:
            One digest for each path, keyed by the path as the caller
            wrote it.

        Example:
            >>> SourceDigests().of([])
            {}
        """
        return {str(path): self.digest(path) for path in sorted(set(paths))}

    def digest(self, path: Path) -> str:
        """Take the SHA-256 digest of the bytes of one file.

        Args:
            path: The file to read.

        Returns:
            The digest, as hexadecimal text.

        Example:
            >>> len(SourceDigests().digest(Path("pyproject.toml")))
            64
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def changed(self, held: Mapping[str, str], now: Mapping[str, str]) -> list[str]:
        """List the files whose content differs from the digest that the store holds.

        A file that the earlier run did not have counts as unchanged. A
        new file adds mutants, and it makes no held result wrong.

        Args:
            held: The digests that the earlier run wrote.
            now: The digests of the source as it is now.

        Returns:
            The paths that differ, sorted.

        Example:
            >>> SourceDigests().changed({"a.py": "x"}, {"a.py": "y"})
            ['a.py']
        """
        return sorted(name for name, mark in now.items() if held.get(name, mark) != mark)


class ResultStore:
    """Read and write the file that holds the result of each finished mutant.

    Several workers finish mutants at the same time, so a lock guards
    each write. The store keys each result by the path, the operator and
    the occurrence of its mutant, which name the same change in every
    run over the same source. The digests of the source go in a second
    file beside the results.

    Example:
        >>> ResultStore.SEPARATOR
        '|'
    """

    SEPARATOR: ClassVar[str] = "|"
    UNTESTED: ClassVar[frozenset[Outcome]] = frozenset({
        Outcome.SURVIVED,
        Outcome.UNSELECTED,
        Outcome.UNREACHED,
    })

    def __init__(self, path: Path) -> None:
        """Name the file of results. Nothing is read or written until `begin`.

        Args:
            path: The JSON lines file.

        Example:
            >>> from pathlib import Path
            >>> ResultStore(Path("r.jsonl")).results()
            []
        """
        self.path = path
        self._held: dict[str, MutantResult] = {}
        self._lock = threading.Lock()
        self._retry = False
        self._again = False
        self._limit: float | None = None

    def key(self, mutant: Mutant) -> str:
        """Give the key of one mutant, which is the same in every run over the same source.

        The line is not part of the key. Only the reader of a report
        needs it.

        Args:
            mutant: The change to name.

        Returns:
            The path, the operator and the occurrence, joined by
            `SEPARATOR`.

        Example:
            >>> from pathlib import Path
            >>> ResultStore(Path("r")).key(Mutant(Path("a.py"), "core/AddNot", 2, 9))
            'a.py|core/AddNot|2'
        """
        parts = (str(mutant.path), mutant.operator, str(mutant.occurrence))
        return self.SEPARATOR.join(parts)

    def begin(
        self,
        *,
        resume: bool,
        retry: bool = False,
        again: bool = False,
        limit: float | None = None,
        digests: Mapping[str, str] | None = None,
    ) -> ResultStore:
        """Read the results of an earlier run, or start an empty file.

        A resumed run first compares the digests, and stops if the source
        changed. A new run writes the digests of the source beside its
        empty file of results, unless it has no digests.

        Args:
            resume: If True, keep what an earlier run finished.
            retry: If True, read each mutant whose tests ran out of time
                as unfinished.
            again: If True, read each mutant that no yml case killed or
                reached as unfinished, so that new yml cases can decide it.
            limit: The time limit of this run, in seconds. A retry keeps
                a mutant that already ran out a limit at least this long.
                None reads every such mutant as unfinished.
            digests: The digest of each file of the scope, or None to
                skip the comparison.

        Returns:
            This store.

        Example:
            >>> ResultStore.begin.__name__
            'begin'
        """
        self._retry = retry
        self._again = again
        self._limit = limit
        if resume and self.path.exists():
            self._verified(digests)
            self._read()
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        if digests is not None:
            text = json.dumps(dict(digests), sort_keys=True)
            self._marks_path().write_text(text, encoding="utf-8")
        return self

    def _verified(self, digests: Mapping[str, str] | None) -> None:
        """Stop the run if the source differs from the source that the store measured.

        The earlier run wrote its digests into a file beside the results.
        If that file does not exist, no file counts as changed.

        Args:
            digests: The digest of each file of the scope now, or None
                to skip the comparison.

        Raises:
            StaleStore: If a file changed after the store was written.

        Example:
            >>> from pathlib import Path
            >>> ResultStore(Path("r"))._verified(None)
        """
        if digests is None:
            return
        beside = self._marks_path()
        held: dict[str, str] = {}
        if beside.exists():
            held = json.loads(beside.read_text(encoding="utf-8"))
        changed = SourceDigests().changed(held, digests)
        if changed:
            raise StaleStore(changed)

    def _marks_path(self) -> Path:
        """Give the path of the file of digests.

        Returns:
            A path in the directory of the results file, named after it.

        Example:
            >>> from pathlib import Path
            >>> ResultStore(Path("/tmp/results.jsonl"))._marks_path().name
            'results-digests.json'
        """
        return self.path.parent / f"{self.path.stem}-digests.json"

    def finished(self, mutant: Mutant) -> MutantResult | None:
        """Give the result that the store holds for one mutant.

        Args:
            mutant: The change to look for.

        Returns:
            The result, or None if no run finished the mutant.

        Example:
            >>> from pathlib import Path
            >>> ResultStore(Path("r")).finished(Mutant(Path("a.py"), "x", 0, 1)) is None
            True
        """
        return self._held.get(self.key(mutant))

    def record(self, result: MutantResult) -> None:
        """Add one result to the file, and keep it in memory.

        A result with no mutant is skipped, because it has no key.

        Args:
            result: The finished mutant.

        Example:
            >>> ResultStore.record.__name__
            'record'
        """
        if result.mutant is None:
            return
        line = json.dumps(self._encoded(result), sort_keys=True)
        with self._lock:
            self._held[self.key(result.mutant)] = result
            with self.path.open("a", encoding="utf-8") as out:
                out.write(f"{line}\n")

    def results(self) -> list[MutantResult]:
        """Give every result that the store holds.

        Returns:
            The results, in no fixed order.

        Example:
            >>> from pathlib import Path
            >>> ResultStore(Path("r")).results()
            []
        """
        return list(self._held.values())

    def _read(self) -> None:
        """Read every line of the file into memory.

        The call skips a blank line. It also skips a result that this
        run must decide again, so that the run reads that mutant as
        unfinished.

        Example:
            >>> ResultStore._read.__name__
            '_read'
        """
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = self._decoded(json.loads(line))
            if self._untested(result) or self._timed_out(result):
                continue
            self._held[self.key(not_none_mutant(result))] = result

    def _untested(self, result: MutantResult) -> bool:
        """Answer whether a run after new yml cases must decide one held result again.

        A mutant that survived, or that no yml case reached, can die
        under a yml case that the earlier run did not have. The source is
        the same, so every other result still holds, and the run decides
        only this set.

        Args:
            result: The result that the file holds.

        Returns:
            True if the run must decide the mutant again.

        Example:
            >>> from pathlib import Path
            >>> mutant = Mutant(Path("a.py"), "core/AddNot", 0, 3)
            >>> store = ResultStore(Path("r"))
            >>> store._untested(MutantResult(mutant, Outcome.SURVIVED, None, 0.0))
            False
        """
        return self._again and result.outcome in self.UNTESTED

    def _timed_out(self, result: MutantResult) -> bool:
        """Answer whether a retry run must decide a mutant that ran out of time again.

        A mutant whose tests ran out of time has no verdict. A retry run
        therefore reads it as unfinished, and runs it again under a
        longer limit. This decides that set without a run of every other
        mutant. A mutant that already ran out a limit at least as long as
        the limit of this run stays as it is, because a second run can
        only end the same way. A hung mutant stays as it is too, because
        a longer limit does not end a loop.

        Args:
            result: The result that the file holds.

        Returns:
            True if the run must decide the mutant again.

        Example:
            >>> from pathlib import Path
            >>> mutant = Mutant(Path("a.py"), "core/AddNot", 0, 3)
            >>> store = ResultStore(Path("r"))
            >>> store._timed_out(MutantResult(mutant, Outcome.KILLED, None, 0.0))
            False
        """
        if not self._retry or result.outcome is not Outcome.TIMED_OUT:
            return False
        return self._limit is None or result.seconds < self._limit

    def _encoded(self, result: MutantResult) -> dict[str, object]:
        """Turn one result into plain data for one line of JSON.

        Args:
            result: The finished mutant.

        Returns:
            The object to write.

        Example:
            >>> from pathlib import Path
            >>> mutant = Mutant(Path("a.py"), "core/AddNot", 0, 3)
            >>> store = ResultStore(Path("r"))
            >>> store._encoded(MutantResult(mutant, Outcome.KILLED, "t::a", 1.0))["line"]
            3
        """
        mutant = not_none_mutant(result)
        return {
            "path": str(mutant.path),
            "operator": mutant.operator,
            "occurrence": mutant.occurrence,
            "line": mutant.line,
            "outcome": result.outcome.value,
            "killer": result.killer,
            "seconds": round(result.seconds, 3),
        }

    def _decoded(self, held: dict[str, object]) -> MutantResult:
        """Turn one object from the file back into a result.

        Args:
            held: One object that the file holds.

        Returns:
            The result.

        Example:
            >>> ResultStore._decoded.__name__
            '_decoded'
        """
        mutant = Mutant(
            Path(str(held["path"])),
            str(held["operator"]),
            int(str(held["occurrence"])),
            int(str(held["line"])),
        )
        killer = held["killer"]
        return MutantResult(
            mutant,
            Outcome(str(held["outcome"])),
            None if killer is None else str(killer),
            float(str(held["seconds"])),
        )


def not_none_mutant(result: MutantResult) -> Mutant:
    """Give the mutant of a result that the store holds, which always has one.

    A `MutantResult` allows a result with no mutant, but the store never
    writes one. This function turns that fact into a type that mypy can
    check.

    Args:
        result: The result to read.

    Returns:
        The mutant.

    Raises:
        TypeError: If the result carries no mutant, which means a bug in
            the harness.

    Example:
        >>> from pathlib import Path
        >>> mutant = Mutant(Path("a.py"), "core/AddNot", 0, 3)
        >>> not_none_mutant(MutantResult(mutant, Outcome.KILLED, None, 0.0)).line
        3
    """
    if result.mutant is None:
        message = "the store holds no result without a mutant"
        raise TypeError(message)
    return result.mutant
