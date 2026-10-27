"""Check that the instructions for coding agents cite only code and files that exist.

`CLAUDE.md`, the rule files in `.claude/rules/` and the agent files in
`.claude/agents/` cite classes, functions and files as examples. When
code is renamed, such an example points at nothing, and an agent then
looks for code that does not exist. These tests fail when a cited name
or path is not in the repository.
"""

from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import pytest


class AgentDocs:
    """Read the instruction files, find what they cite, and look for it in the repository.

    A citation is an inline code span in Markdown. A span that looks like
    a path must exist under one of a few directories. A span that looks
    like a dotted Python name must have each of its parts as a word
    somewhere in the code or in the tool settings.
    """

    ROOT: ClassVar[Path] = Path(__file__).resolve().parents[1]
    SPAN: ClassVar[re.Pattern[str]] = re.compile(r"`([^`\n]+)`")
    PATH: ClassVar[re.Pattern[str]] = re.compile(r"[\w./-]*[\w-]+(?:/|\.(?:py|md|toml|yml))")
    NAME: ClassVar[re.Pattern[str]] = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\(\))?")
    WORD: ClassVar[re.Pattern[str]] = re.compile(r"\w+")
    # Names that the rules use as placeholders, or as files never to create.
    PLACEHOLDERS: ClassVar[frozenset[str]] = frozenset({
        "SomeError",
        "SomeClass",
        "ClassName",
        "utils.py",
        "helpers.py",
    })

    SEARCH_ROOTS: ClassVar[tuple[str, ...]] = (
        "",
        "src/astrolabe",
        "src/astrolabe/_hooks",
        ".claude/rules",
        "spec/lean",
    )
    CODE_GLOBS: ClassVar[tuple[str, ...]] = (
        "src/**/*.py",
        "tests/**/*.py",
        "spec/tests/**/*.py",
        "conftest.py",
        "pyproject.toml",
        ".mdformat.toml",
        "spec/lean/*.lean",
        "spec/lean/Spec/**/*.lean",
        "spec/lean/lakefile.toml",
    )

    def documents(self) -> list[Path]:
        """List the instruction files to check.

        Returns:
            `CLAUDE.md`, then each rule file and each agent file, by name.
        """
        rules = sorted((self.ROOT / ".claude" / "rules").glob("*.md"))
        agents = sorted((self.ROOT / ".claude" / "agents").glob("*.md"))
        return [self.ROOT / "CLAUDE.md", *rules, *agents]

    def spans(self, document: Path) -> list[str]:
        """List the inline code spans of one instruction file.

        Args:
            document: The instruction file.

        Returns:
            The text of each span. A placeholder name is left out.
        """
        spans = self.SPAN.findall(document.read_text(encoding="utf-8"))
        return [span for span in spans if span not in self.PLACEHOLDERS]

    def missing_paths(self, document: Path) -> list[str]:
        """List the cited paths that exist under none of the search directories.

        The check skips a span that is not a path.

        Args:
            document: The instruction file.

        Returns:
            The paths that do not exist.
        """
        missing: list[str] = []
        for span in self.spans(document):
            if self.PATH.fullmatch(span) is None:
                continue
            if not self._exists(span):
                missing.append(span)
        return missing

    def _exists(self, path: str) -> bool:
        """Answer whether a cited path exists under one of the search directories.

        A rule can write a path from the root of the repository, or from
        a directory such as `src/astrolabe/_hooks`, so each of those is
        tried.

        Args:
            path: The cited path.

        Returns:
            True if the file or the directory exists.
        """
        return any((self.ROOT / root / path).exists() for root in self.SEARCH_ROOTS)

    def missing_names(self, document: Path) -> list[str]:
        """List the cited code names that have a part the code never uses.

        A code name is a dotted Python name, with or without `()` at its end.
        A span that is also a path is not a code name.

        Args:
            document: The instruction file.

        Returns:
            The names with a part that is not in the code.
        """
        missing: list[str] = []
        for span in self.spans(document):
            if not self.NAME.fullmatch(span) or self.PATH.fullmatch(span):
                continue
            if not self._known(span):
                missing.append(span)
        return missing

    def _known(self, name: str) -> bool:
        """Answer whether each part of a dotted name appears in the code.

        The test does not import the name. A word match is enough to catch
        a rename, and it also accepts a name that the rules cite from a
        settings file.

        Args:
            name: The cited name.

        Returns:
            True if each part is a word of the code.
        """
        parts = name.removesuffix("()").split(".")
        return all(part in self.words for part in parts)

    @cached_property
    def words(self) -> set[str]:
        """Collect each word of the code, the Lean proofs and the tool settings files.

        Returns:
            The words.
        """
        words: set[str] = set()
        for pattern in self.CODE_GLOBS:
            for path in self.ROOT.glob(pattern):
                words.update(self.WORD.findall(path.read_text(encoding="utf-8")))
        return words

    def listed_rule_files(self) -> set[str]:
        """List the rule files that the table in `CLAUDE.md` names.

        A test compares this set with the rule files on disk, so that the
        table lists each one.

        Returns:
            Each file name in the first column of the table.
        """
        text = (self.ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        return set(re.findall(r"^\| `([\w-]+\.md)` \|", text, re.MULTILINE))


DOCS = AgentDocs()


@pytest.mark.parametrize("document", DOCS.documents(), ids=lambda path: path.name)
def test_every_cited_path_exists(document: Path) -> None:
    assert not DOCS.missing_paths(document)


@pytest.mark.parametrize("document", DOCS.documents(), ids=lambda path: path.name)
def test_every_cited_name_appears_in_the_code(document: Path) -> None:
    assert not DOCS.missing_names(document)


def test_claude_md_lists_every_rule_file() -> None:
    rule_files = {path.name for path in (AgentDocs.ROOT / ".claude" / "rules").glob("*.md")}
    assert DOCS.listed_rule_files() == rule_files
