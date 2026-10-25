"""Join each example with the messages of mypy, and write the catalog that the pages read.

The catalog is one JSON object, with one entry for each example. An entry
holds the file name and the lines that the page shows, the messages on
those lines, and each revealed type, read as a frame where it is one.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from tools.site_examples.messages import MypyBatch
from tools.site_examples.revealed import RevealedType
from tools.site_examples.source import ExampleSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tools.site_examples.messages import MypyMessage
    from tools.site_examples.source import RevealRequest


class HiddenMessage(Exception):
    """Stop the build, because mypy reports a message on a line that the page does not show.

    A reader would see clean code where the checker reports an error, so
    the page would not tell the truth.
    """

    def __init__(self, name: str, message: MypyMessage) -> None:
        """Build the message, which names the example, the line and the text.

        Args:
            name: The name of the example.
            message: The message on the hidden line.

        Example:
            >>> from tools.site_examples.messages import MypyMessage
            >>> str(HiddenMessage("demo", MypyMessage("a.py", 2, "error", "Bad", "misc")))
            'demo: mypy reports on a hidden line 2: Bad'
        """
        super().__init__(f"{name}: mypy reports on a hidden line {message.line}: {message.text}")


class ExampleRecord:
    r"""The entry of one example in the catalog.

    Example:
        >>> record = ExampleRecord(ExampleSource("demo", "x = 1\n"), [])
        >>> record.to_json()["lines"]
        ['x = 1']
    """

    REVEALED: ClassVar[str] = "Revealed type is "

    def __init__(self, source: ExampleSource, messages: Sequence[MypyMessage]) -> None:
        """Keep the example and the messages that mypy reports in its file.

        Args:
            source: The example.
            messages: The messages of its file, in the order of the report.

        Example:
            >>> ExampleRecord(ExampleSource("demo", ""), []).to_json()["messages"]
            []
        """
        self._source = source
        self._messages = list(messages)

    def to_json(self) -> dict[str, object]:
        """Build the entry.

        Returns:
            The entry, ready for `json.dumps`.

        Raises:
            HiddenMessage: If mypy reports on a line that the page does not show.

        Example:
            >>> sorted(ExampleRecord(ExampleSource("demo", ""), []).to_json())
            ['file', 'lines', 'messages', 'results', 'start']
        """
        shown: list[dict[str, object]] = []
        revealed: dict[int, str] = {}
        for message in self._messages:
            at = self._source.shown_line(message.line)
            if at is not None:
                shown.append(self._message(at, message))
            elif self._is_reveal(message):
                revealed[message.line] = message.text.removeprefix(self.REVEALED).strip('"')
            else:
                raise HiddenMessage(self._source.name, message)
        results = [
            self._result(request, revealed.get(line, ""))
            for line, request in sorted(self._source.reveals.items())
        ]
        return {
            "file": self._source.file,
            "start": self._source.start,
            "lines": self._source.shown,
            "messages": shown,
            "results": results,
        }

    def _is_reveal(self, message: MypyMessage) -> bool:
        """Tell whether a message is the note of a `reveal_type(...)` line.

        Args:
            message: A message on a line that the page does not show.

        Returns:
            True for the note that gives a revealed type.

        Example:
            >>> from tools.site_examples.messages import MypyMessage
            >>> record = ExampleRecord(ExampleSource("demo", "reveal_type(1)"), [])
            >>> record._is_reveal(MypyMessage("a.py", 1, "note", 'Revealed type is "int"', None))
            True
        """
        return message.line in self._source.reveals and message.text.startswith(self.REVEALED)

    def _message(self, at: int, message: MypyMessage) -> dict[str, object]:
        """Give one message on a shown line.

        Args:
            at: The number of the line on the page.
            message: The message.

        Returns:
            The message as the page reads it.

        Example:
            >>> from tools.site_examples.messages import MypyMessage
            >>> record = ExampleRecord(ExampleSource("demo", ""), [])
            >>> record._message(4, MypyMessage("a.py", 2, "note", "Hi", None))
            {'line': 4, 'severity': 'note', 'text': 'Hi', 'code': None}
        """
        return {
            "line": at,
            "severity": message.severity,
            "text": message.text,
            "code": message.code,
        }

    def _result(self, request: RevealRequest, text: str) -> dict[str, object]:
        """Give one revealed type, with each frame that it holds, read as columns.

        A union of frames, for example the inferred return type of a
        function with two `return` statements, holds one frame for each
        member. A type that is not a typed frame holds no frame.

        Args:
            request: The `reveal_type(...)` line.
            text: The type that mypy revealed.

        Returns:
            The result as the page reads it.

        Example:
            >>> from tools.site_examples.source import RevealRequest
            >>> record = ExampleRecord(ExampleSource("demo", ""), [])
            >>> one = "TypedDataFrame[TypedDict({'a': Int64}, closed=True), None]"
            >>> result = record._result(RevealRequest("df"), f"{one} | int")
            >>> len(result["frames"])
            1
        """
        revealed = RevealedType(text)
        frames: list[dict[str, object]] = []
        for member in revealed.alternatives():
            columns = member.columns()
            if columns is not None:
                frames.append(self._frame(request, member, columns))
        return {"label": request.expression, "frames": frames, "revealed": revealed.text}

    def _frame(
        self,
        request: RevealRequest,
        member: RevealedType,
        found: list[tuple[str, str]],
    ) -> dict[str, object]:
        """Give one frame of a revealed type: its kind, its columns with their marks, its remainder.

        A mark can name a column that the frame no longer has, for
        example a column that a `select` dropped. The page then shows that
        column struck through, after the others.

        Args:
            request: The `reveal_type(...)` line, with the marks.
            member: The frame.
            found: The columns of the frame.

        Returns:
            The frame as the page reads it.

        Example:
            >>> from tools.site_examples.source import RevealRequest
            >>> record = ExampleRecord(ExampleSource("demo", ""), [])
            >>> frame = RevealedType("TypedDataFrame[TypedDict({'a': Int64}, closed=True), None]")
            >>> request = RevealRequest("df", {"b": "gone"})
            >>> record._frame(request, frame, [("a", "Int64")])["columns"]
            [['a', 'Int64', ''], ['b', '', 'gone']]
        """
        columns = [[name, dtype, request.marks.get(name, "")] for name, dtype in found]
        names = {name for name, _ in found}
        columns += [[name, "", mark] for name, mark in request.marks.items() if name not in names]
        return {"kind": member.kind(), "columns": columns, "remainder": member.remainder()}


class ExampleCatalog:
    """Every example of a directory, checked by mypy and written as one JSON catalog.

    Example:
        >>> ExampleCatalog(Path("site/examples"), Path("site/src/data/examples.json")).examples
        PosixPath('site/examples')
    """

    def __init__(self, examples: Path, output: Path) -> None:
        """Keep the directory of the examples and the path of the catalog.

        Args:
            examples: The directory of the example files.
            output: The JSON file that the pages read.

        Example:
            >>> ExampleCatalog(Path("e"), Path("o.json")).output
            PosixPath('o.json')
        """
        self.examples = examples
        self.output = output

    def build(self) -> str:
        """Run mypy over every example, and give the text of the catalog.

        The examples that share their settings run in one run of mypy.

        Returns:
            The JSON text, sorted by the name of the example.

        Example:
            >>> ExampleCatalog(Path("e"), Path("o.json")).build.__name__
            'build'
        """
        sources = [ExampleSource.read(path) for path in sorted(self.examples.glob("*.py"))]
        groups: dict[tuple[str, ...], list[ExampleSource]] = {}
        for source in sources:
            groups.setdefault(tuple(source.settings), []).append(source)
        entries: dict[str, object] = {}
        for settings, members in groups.items():
            entries.update(self._checked(settings, members))
        ordered = dict(sorted(entries.items()))
        return json.dumps(ordered, indent=1, ensure_ascii=False) + "\n"

    def stale(self) -> bool:
        """Tell whether the catalog on disk differs from a new run.

        Returns:
            True if the file is missing or differs.

        Example:
            >>> ExampleCatalog(Path("e"), Path("o.json")).stale.__name__
            'stale'
        """
        if not self.output.exists():
            return True
        return self.output.read_text(encoding="utf-8") != self.build()

    def _checked(
        self, settings: tuple[str, ...], members: Sequence[ExampleSource]
    ) -> dict[str, object]:
        """Run mypy over the examples of one set of settings, and build their entries.

        Args:
            settings: The settings that the examples share.
            members: The examples.

        Returns:
            The entry of each example, by its name.

        Example:
            >>> ExampleCatalog(Path("e"), Path("o.json"))._checked((), [])
            {}
        """
        if not members:
            return {}
        paths = [self.examples / f"{source.name}.py" for source in members]
        with tempfile.TemporaryDirectory() as workdir:
            messages = MypyBatch(settings).run(paths, Path(workdir))
        by_path: dict[str, list[MypyMessage]] = {}
        for message in messages:
            by_path.setdefault(message.path, []).append(message)
        entries: dict[str, object] = {}
        for source, path in zip(members, paths, strict=True):
            record = ExampleRecord(source, by_path.get(str(path.resolve()), []))
            entries[source.name] = record.to_json()
        return entries
