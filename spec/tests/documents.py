"""Reading the specification documents as data, so they can be checked.

A document that says "there are 61 of them" when there are 28, or cites a rule
by a name no section defines, is wrong in a way no reader reliably catches. Both
of those really happened here.

This module parses the documents into the pieces a check needs: the
identifiers each one defines, the identifiers each one cites, the tables, and
the counts each one asserts about itself. `test_documents.py` then states the
invariants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).resolve().parents[1]

IDENTIFIER = re.compile(r"\b([DRETMB])-(\d+[ab]?)\b")
RANGE = re.compile(r"`([DRETMB])-(\d+)`\s*(?:…|\.\.\.)\s*`\1-(\d+)`")


@dataclass(frozen=True)
class Document:
    """One of the specification documents, read as text and as structure.

    Example:
        >>> Document("OPERATIONS.md").name
        'OPERATIONS.md'
    """

    name: str

    @cached_property
    def text(self) -> str:
        """Give the document's whole content.

        Returns:
            The file's text.

        Example:
            >>> Document("OPERATIONS.md").text.startswith("# The grammar")
            True
        """
        return (SPEC / self.name).read_text()

    @cached_property
    def lines(self) -> list[str]:
        """Give the document's lines.

        Returns:
            One string per line, without the line separator.

        Example:
            >>> Document("OPERATIONS.md").lines[0]
            '# The grammar of partial schemas'
        """
        return self.text.split("\n")

    @cached_property
    def identifiers(self) -> set[str]:
        """Give every identifier the document mentions.

        Returns:
            Each identifier, as `D-1`, `T-12` and so on.

        Example:
            >>> "T-1" in Document("OPERATIONS.md").identifiers
            True
        """
        return {f"{kind}-{number}" for kind, number in IDENTIFIER.findall(self.text)}

    @cached_property
    def table_rows(self) -> list[list[str]]:
        """Give every table row, split into its cells.

        A separator row of dashes is not a row of content, so it is
        left out. Nor is a line inside a fenced code block, where a
        vertical bar means alternation in a grammar and not a cell
        boundary. Each cell is trimmed.

        Returns:
            One list of cell strings per row.

        Example:
            >>> rows = Document("OPERATIONS.md").table_rows
            >>> all(len(r) >= 2 for r in rows)
            True
        """
        rows = []
        for line in self.prose_lines:
            stripped = line.strip()
            if self._is_row(stripped):
                rows.append([cell.strip() for cell in stripped.strip("|").split("|")])
        return rows

    def _is_row(self, line: str) -> bool:
        """Answer whether a line is a table row of content.

        Args:
            line: One stripped line of the document.

        Returns:
            True for a row that is neither a separator nor prose.

        Example:
            >>> Document("OPERATIONS.md")._is_row("|---|---|")
            False
        """
        return line.startswith("|") and not set(line) <= set("|- :")

    @cached_property
    def prose_lines(self) -> list[str]:
        """Give the lines outside every fenced code block.

        Returns:
            One string per line, with fenced blocks removed.

        Example:
            >>> lines = Document("OPERATIONS.md").prose_lines
            >>> any("P  ::=" in line for line in lines)
            False
        """
        blocks = self.text.split("\n```")
        outside = blocks[::2]
        return "\n".join(outside).split("\n")

    def cited_tests(self) -> set[str]:
        """Give every test or class name the document cites as evidence.

        Returns:
            Each name written as `test_...` or `TestSomething`.

        Example:
            >>> "TestJoinNaming" in Document("OPERATION_CATALOG.md").cited_tests()
            True
        """
        found = re.findall(r"`(test_[a-z0-9_]+|Test[A-Za-z0-9]+)`", self.text)
        return set(found)

    def cited_lean(self) -> set[str]:
        """Give every Lean theorem the document cites.

        Returns:
            Each name written in backticks as `Spec.Something`.

        Example:
            >>> isinstance(Document("OPERATIONS.md").cited_lean(), set)
            True
        """
        return set(re.findall(r"`(Spec\.[A-Za-z0-9_.']+)`", self.text))

    def cited_lean_files(self) -> dict[str, str]:
        """Give the file each cited Lean theorem is said to live in.

        Returns:
            Each theorem written as `` `Spec.X` in `spec/lean/...lean` ``,
            mapped to that file.

        Example:
            >>> isinstance(Document("OPERATIONS.md").cited_lean_files(), dict)
            True
        """
        pattern = r"`(Spec\.[A-Za-z0-9_.']+)` in `(spec/lean/[A-Za-z0-9_/]+\.lean)`"
        return dict(re.findall(pattern, self.text))


class Ledger:
    """`OPERATIONS.md`'s claim ledger, read as the register of identifiers.

    The ledger is the one place every identifier is listed. Making it the
    register is what stops it becoming decoration: an identifier missing
    from it fails a test, and so does an entry nothing cites.

    Example:
        >>> "T-5" in Ledger().registered
        True
    """

    def __init__(self) -> None:
        """Read the ledger out of the section that holds it."""
        self.document = Document("OPERATIONS.md")

    @cached_property
    def section(self) -> str:
        """Give the text of the claim-ledger section.

        Returns:
            Everything from the ledger heading to the end of the file.

        Example:
            >>> Ledger().section.startswith("## 19. Claim ledger")
            True
        """
        marker = "## 19. Claim ledger"
        return self.document.text[self.document.text.index(marker) :]

    @cached_property
    def registered(self) -> set[str]:
        """Give every identifier the ledger registers, ranges expanded.

        A row reading `D-1` … `D-8` registers all eight.

        Returns:
            Each registered identifier.

        Example:
            >>> {"D-1", "D-5", "D-8"} <= Ledger().registered
            True
        """
        found = set(RANGE.sub(self._expand, self.section).split())
        return {t.strip("`|,.") for t in found if IDENTIFIER.fullmatch(t.strip("`|,."))}

    @cached_property
    def proved_rows(self) -> list[list[str]]:
        """Give each ledger row that tags its claim proved, split into cells.

        A row of the ledger has four cells, and its third cell holds
        the tag.

        Returns:
            The cells of each row whose third cell says **proved**.

        Example:
            >>> rows = Ledger().proved_rows
            >>> any(row[0] == "`T-5`" for row in rows)
            True
            >>> any(row[0] == "`T-6`" for row in rows)
            False
        """
        rows = []
        for line in self.section.splitlines():
            row = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
            if len(row) == 4 and "**proved**" in row[2]:
                rows.append(row)
        return rows

    def _expand(self, match: re.Match[str]) -> str:
        """Rewrite a range in the ledger into every identifier it covers.

        Args:
            match: A match of a `X-a` … `X-b` range.

        Returns:
            The identifiers, separated by spaces.

        Example:
            >>> import re
            >>> Ledger()._expand(RANGE.match("`D-1` … `D-3`"))
            'D-1 D-2 D-3'
        """
        kind, first, last = match.group(1), int(match.group(2)), int(match.group(3))
        return " ".join(f"{kind}-{n}" for n in range(first, last + 1))


class Glossary:
    """One document's glossary, read as a term-to-identifier map.

    Making the glossary machine-readable is what lets a check state the
    rule the document claims for itself: a term used with a technical
    meaning is defined, and a definition given is a definition used.

    Example:
        >>> "remainder" in Glossary().entries
        True
    """

    def __init__(
        self,
        document_name: str = "DATAFRAME_SPECIFICATION.md",
        heading: str = "## 13. Glossary",
    ) -> None:
        """Read the glossary out of the section that holds it.

        Args:
            document_name: The document holding the glossary.
            heading: The glossary section's heading line.
        """
        self.document = Document(document_name)
        self.heading = heading

    @cached_property
    def section(self) -> str:
        """Give the text of the glossary section.

        Returns:
            Everything from the glossary heading to the end of the file.

        Example:
            >>> Glossary().section.startswith("## 13. Glossary")
            True
        """
        return self.document.text[self.document.text.index(self.heading) :]

    @cached_property
    def entries(self) -> dict[str, str]:
        """Give each glossary term and the identifier it resolves to.

        Each line of the section splits on the cell separator. Only a
        term's row counts, with four cells.

        Returns:
            The term from each row's first cell, mapped to its last.

        Example:
            >>> Glossary().entries["column set"]
            '`D-5`'
            >>> "Term" in Glossary().entries
            False
        """
        entries = {}
        for line in self.section.split("\n"):
            row = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if self._is_entry(row):
                entries[row[0]] = row[3]
        return entries

    def _is_entry(self, row: list[str]) -> bool:
        """Answer whether a parsed row is a glossary entry.

        Args:
            row: One row's cells.

        Returns:
            True for a term's row, and False for a header or separator.

        Example:
            >>> Glossary()._is_entry(["Term", "Means", "Here", "Formally"])
            False
        """
        if len(row) != 4 or row[0] == "Term":
            return False
        return bool(row[0]) and not set(row[0]) <= set("- :")

    def body(self) -> str:
        """Give the document's text before the glossary.

        Returns:
            Everything a glossary term must be used in.

        Example:
            >>> Glossary().heading in Glossary().body()
            False
        """
        return self.document.text[: self.document.text.index(self.heading)]


class LeanScope:
    """Track the namespaces open at one point of a Lean file.

    A theorem written `theorem T2.closure` inside `namespace Spec` has the
    full name `Spec.T2.closure`. The scope resolves such a short name.

    Example:
        >>> scope = LeanScope()
        >>> scope.read("namespace Spec")
        >>> scope.full("T2.closure")
        'Spec.T2.closure'
        >>> scope.read("mutual")
        >>> scope.read("end")
        >>> scope.full("T2.closure")
        'Spec.T2.closure'
        >>> scope.read("end Spec")
        >>> scope.full("T2.closure")
        'T2.closure'
    """

    OPEN: ClassVar[re.Pattern[str]] = re.compile(r"^(namespace|section|mutual)\b\s*([\w.']*)")
    CLOSE: ClassVar[re.Pattern[str]] = re.compile(r"^end\b")

    def __init__(self) -> None:
        """Start with no namespace open."""
        self._stack: list[str] = []

    def read(self, line: str) -> None:
        """Update the scope for one line of Lean.

        A `namespace` line adds its name. A `section` or a `mutual`
        line adds an empty part, so that its `end` closes it.

        Args:
            line: The line, with its comments already removed.
        """
        opened = self.OPEN.match(line)
        if opened is not None:
            segment = opened.group(2) if opened.group(1) == "namespace" else ""
            self._stack.append(segment)
        elif self.CLOSE.match(line) and self._stack:
            self._stack.pop()

    def full(self, name: str) -> str:
        """Give the full name of a declaration in the current scope.

        Args:
            name: The name as the declaration writes it.

        Returns:
            The name with every open namespace in front.
        """
        if name.startswith("_root_."):
            return name.removeprefix("_root_.")
        return ".".join([part for part in self._stack if part] + [name])


class LeanSources:
    """The Lean formalization under `spec/lean`, read as text.

    Only the project's own files are read: `Spec.lean`, `Audit.lean` and
    `Spec/`. Mathlib's files under `.lake` never are.

    Example:
        >>> "Spec.Functional.mono" in LeanSources().declared
        True
    """

    DECLARATION: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable)\s+)*"
        r"(?:theorem|lemma|def|abbrev|structure|inductive|instance)\s+([\w.']+)"
    )
    AUDITED: ClassVar[re.Pattern[str]] = re.compile(
        r"^#assert_standard_axioms\s+([\w.']+)",
        re.MULTILINE,
    )
    FORBIDDEN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(sorry|admit|native_decide|implemented_by)\b|^axiom\b",
        re.MULTILINE,
    )

    def __init__(self) -> None:
        """Locate the Lean project."""
        self.root = SPEC / "lean"

    def files(self) -> list[Path]:
        """List the project's own Lean files.

        Returns:
            `Spec.lean`, then every file under `Spec/`, in sorted order.
        """
        return [self.root / "Spec.lean", *sorted((self.root / "Spec").rglob("*.lean"))]

    def code(self, path: Path) -> str:
        """Give one file's text with its comments removed.

        Args:
            path: The Lean file.

        Returns:
            The text, without block comments and line comments.
        """
        text = re.sub(r"/-.*?-/", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
        return re.sub(r"--[^\n]*", "", text)

    @cached_property
    def declared_in(self) -> dict[str, str]:
        """Map every declared full name to the file that declares it.

        Each name gets the namespaces open at its line in front.

        Returns:
            Each full name, mapped to its path relative to the repository.
        """
        found: dict[str, str] = {}
        for path in self.files():
            relative = str(path.relative_to(ROOT))
            scope = LeanScope()
            for line in self.code(path).split("\n"):
                scope.read(line)
                for name in self.DECLARATION.findall(line):
                    found[scope.full(name)] = relative
        return found

    @property
    def declared(self) -> set[str]:
        """Give every declared full name.

        Returns:
            The names.
        """
        return set(self.declared_in)

    @cached_property
    def audited(self) -> set[str]:
        """Give every theorem `Audit.lean` checks for non-standard axioms.

        Returns:
            Each name after `#assert_standard_axioms`.
        """
        return set(self.AUDITED.findall((self.root / "Audit.lean").read_text(encoding="utf-8")))

    def forbidden(self) -> dict[str, list[str]]:
        """Find the tokens that would let a proof skip its obligation.

        Returns:
            Each file with such a token, mapped to the tokens found, in
            order.
        """
        found: dict[str, list[str]] = {}
        for path in self.files():
            matches = self.FORBIDDEN.finditer(self.code(path))
            tokens = [match.group(0) for match in matches]
            if tokens:
                found[str(path.relative_to(ROOT))] = tokens
        return found
