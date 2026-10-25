"""Run mypy with the plugin over a set of examples, and read what it reports on each line.

The examples run under the configuration that `astrolabe init` writes:
the plugin and `allow_redefinition`, and no strict mode. An example can
add settings of `[tool.astrolabe]`. The examples that share their
settings run together, in one run of mypy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from mypy import api

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class MypyFailure(Exception):
    """Stop the build, because mypy did not finish its check of the examples.

    mypy exits with the status 2 when it crashes or cannot read a file.
    It writes to its error stream when it cannot read its configuration,
    for example a setting of an example. Its report is then incomplete or
    wrong, so a page would show an example as clean while the checker never
    checked it as written.
    """

    def __init__(self, output: str) -> None:
        r"""Build the message from the last lines of the output of mypy.

        Args:
            output: The output of mypy, both streams.

        Example:
            >>> str(MypyFailure("a\nb\nINTERNAL ERROR"))
            'mypy did not finish: a | b | INTERNAL ERROR'
        """
        tail = " | ".join(output.strip().splitlines()[-6:])
        super().__init__(f"mypy did not finish: {tail}")


@dataclass(frozen=True)
class MypyMessage:
    """One line of the report of mypy: the file, the line, the severity, the text and the code.

    Example:
        >>> message = MypyMessage.parse('/x/a.py:3: error: Unknown column "v"  [unknown-column]')
        >>> message.line, message.text, message.code
        (3, 'Unknown column "v"', 'unknown-column')
    """

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(.*?\.py):(\d+): (error|note): (.*?)(?:  \[([a-z0-9-]+)\])?$",
    )

    path: str
    line: int
    severity: str
    text: str
    code: str | None

    @classmethod
    def parse(cls, line: str) -> MypyMessage | None:
        """Read one line of the report, or give None for a line that is not a message.

        Args:
            line: One line of the output of mypy.

        Returns:
            The message, or None.

        Example:
            >>> MypyMessage.parse("Success: no issues found in 1 source file") is None
            True
        """
        match = cls.PATTERN.match(line)
        if match is None:
            return None
        path, number, severity, text, code = match.groups()
        return cls(path, int(number), severity, text, code)


class MypyBatch:
    """Run mypy once over the examples that share one set of settings.

    The run uses its own configuration file and its own cache, in a
    directory that the caller gives, so it reads nothing of the project
    around it.

    Example:
        >>> print(MypyBatch(['allow_unproven = "none"']).config())
        [tool.mypy]
        plugins = ["astrolabe.plugin"]
        allow_redefinition = true
        <BLANKLINE>
        [tool.astrolabe]
        allow_unproven = "none"
        <BLANKLINE>
    """

    FAILED: ClassVar[int] = 2

    def __init__(self, settings: Sequence[str]) -> None:
        """Keep the settings of `[tool.astrolabe]` for the run.

        Args:
            settings: Lines of TOML, for example `allow_unproven = "none"`.

        Example:
            >>> MypyBatch([]).config().count("[tool.astrolabe]")
            0
        """
        self._settings = list(settings)

    def config(self) -> str:
        """Write the configuration file of the run.

        Returns:
            The text of a `pyproject.toml`.

        Example:
            >>> MypyBatch([]).config().splitlines()[0]
            '[tool.mypy]'
        """
        text = '[tool.mypy]\nplugins = ["astrolabe.plugin"]\nallow_redefinition = true\n'
        if self._settings:
            text += "\n[tool.astrolabe]\n" + "\n".join(self._settings) + "\n"
        return text

    def run(self, files: Sequence[Path], workdir: Path) -> list[MypyMessage]:
        """Run mypy over the files, and read its report.

        Args:
            files: The example files.
            workdir: A directory for the configuration file and the cache.

        Returns:
            Every message, in the order that mypy reports them.

        Raises:
            MypyFailure: If mypy crashes, cannot read a file, or cannot read
                the configuration.

        Example:
            >>> MypyBatch([]).run.__name__
            'run'
        """
        config = workdir / "pyproject.toml"
        config.write_text(self.config(), encoding="utf-8")
        arguments = [
            "--config-file",
            str(config),
            "--cache-dir",
            str(workdir / "cache"),
            "--show-absolute-path",
            "--hide-error-context",
            "--no-error-summary",
            "--no-pretty",
            "--no-color-output",
            *(str(path.resolve()) for path in files),
        ]
        stdout, stderr, status = api.run(arguments)
        if status == self.FAILED or stderr.strip():
            raise MypyFailure(stdout + stderr)
        messages = (MypyMessage.parse(line) for line in stdout.splitlines())
        return [message for message in messages if message is not None]
