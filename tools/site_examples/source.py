"""Read one example file into the lines that the page shows and the lines that only mypy reads.

An example is a complete Python file, so mypy can check it. The page
shows only part of it. A header line names the file that the page shows
and the number of its first line. A line that ends with `# hide` gives
mypy a definition that the reader already knows from an earlier example.
A `reveal_type(...)` line does not show either: its revealed type becomes
the row of columns under the code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class RevealRequest:
    """One `reveal_type(...)` line: the expression, and how the page marks some of its columns.

    The comment after the call can mark columns, for example
    `# added: notional; remainder: venue; dropped: quantity`. The page
    then draws `notional` as a new column, `venue` as a column of the
    remainder, and `quantity` as a column that the step took away.

    Example:
        >>> RevealRequest("result", {"notional": "add"}).marks
        {'notional': 'add'}
    """

    expression: str
    marks: dict[str, str] = field(default_factory=dict)


class ExampleSource:
    r"""One example file, split into its header, its shown lines, its hidden lines and its reveals.

    The page numbers the shown lines from the `# start:` header, so a
    second example can go on where a first one stopped. A message of
    mypy on a hidden line cannot show on the page, so the reader of a
    source can ask which shown line a line of the file is.

    Example:
        >>> source = ExampleSource("demo", "# file: demo.py\nx = 1\nreveal_type(x)\n")
        >>> source.file, source.shown
        ('demo.py', ['x = 1'])
    """

    HEADER: ClassVar[re.Pattern[str]] = re.compile(r"^# (file|start|setting): (.*)$")
    REVEAL: ClassVar[re.Pattern[str]] = re.compile(r"^reveal_type\((.*)\)(?:\s*#\s*(.*))?$")
    HIDE: ClassVar[str] = "# hide"
    MARKS: ClassVar[dict[str, str]] = {"added": "add", "remainder": "rem", "dropped": "gone"}

    def __init__(self, name: str, text: str) -> None:
        r"""Split the text of the file.

        Args:
            name: The name of the example, which a page uses to show it.
            text: The content of the file.

        Example:
            >>> ExampleSource("demo", "# start: 5\ny = 2\n").start
            5
        """
        self.name = name
        self.file = f"{name}.py"
        self.start = 1
        self.settings: list[str] = []
        self.shown: list[str] = []
        self.reveals: dict[int, RevealRequest] = {}
        self._shown_at: dict[int, int] = {}
        for number, line in enumerate(text.splitlines(), 1):
            self._take(number, line)

    @classmethod
    def read(cls, path: Path) -> ExampleSource:
        """Read an example file, named after its stem.

        Args:
            path: The file.

        Returns:
            The source of the example.

        Example:
            >>> ExampleSource.read.__name__
            'read'
        """
        return cls(path.stem, path.read_text(encoding="utf-8"))

    def shown_line(self, number: int) -> int | None:
        r"""Give the number that the page shows for a line of the file.

        Args:
            number: The number of the line in the file, from 1.

        Returns:
            The number on the page, or None for a line that does not show.

        Example:
            >>> text = "# start: 10\nimport x  # hide\ny = 1\n"
            >>> source = ExampleSource("demo", text)
            >>> source.shown_line(3), source.shown_line(2)
            (10, None)
        """
        index = self._shown_at.get(number)
        if index is None:
            return None
        return self.start + index

    def _take(self, number: int, line: str) -> None:
        """Sort one line into the header, the reveals, the hidden lines or the shown lines.

        A blank line before the first shown line does not show: it only
        spaced out the hidden lines.

        Args:
            number: The number of the line in the file, from 1.
            line: The text of the line.

        Example:
            >>> source = ExampleSource("demo", "")
            >>> source._take(1, "# setting: allow_unproven = 'none'")
            >>> source._take(2, "")
            >>> source.settings, source.shown
            (["allow_unproven = 'none'"], [])
        """
        header = self.HEADER.match(line)
        if header is not None:
            self._header(header.group(1), header.group(2).strip())
            return
        reveal = self.REVEAL.match(line)
        if reveal is not None:
            marks = self._marks(reveal.group(2) or "")
            self.reveals[number] = RevealRequest(reveal.group(1), marks)
            return
        if line.rstrip().endswith(self.HIDE):
            return
        if not line.strip() and not self.shown:
            return
        self._shown_at[number] = len(self.shown)
        self.shown.append(line)

    def _header(self, key: str, value: str) -> None:
        """Keep one header value.

        Args:
            key: `file`, `start` or `setting`.
            value: The text after the key.

        Example:
            >>> source = ExampleSource("demo", "")
            >>> source._header("file", "pipeline.py")
            >>> source.file
            'pipeline.py'
        """
        if key == "file":
            self.file = value
        elif key == "start":
            self.start = int(value)
        else:
            self.settings.append(value)

    def _marks(self, comment: str) -> dict[str, str]:
        """Read the marks of a reveal from its comment.

        Args:
            comment: The comment after the call, without the `#`.

        Returns:
            The mark of each named column.

        Example:
            >>> ExampleSource("demo", "")._marks("added: a, b; remainder: c")
            {'a': 'add', 'b': 'add', 'c': 'rem'}
        """
        marks: dict[str, str] = {}
        parts = [piece.strip() for piece in comment.split(";")]
        for part in filter(None, parts):
            kind, _, names = part.partition(":")
            mark = self.MARKS[kind.strip()]
            for name in filter(None, (piece.strip() for piece in names.split(","))):
                marks[name] = mark
        return marks
