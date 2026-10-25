"""The executable model of expression type maps: tables over the quotient.

A type map is a finite map from quotient cells to output types, with
an absent cell meaning error. This module builds those tables, reads
them, composes them, and spells them canonically, so the theorems the
specification states about them can be checked by exhaustion here.

Ground truth is the cell-to-output mapping; equality of tables is
equality of that mapping, which is the extensional identity the
specification defines. The canonical spelling is derived from the
outputs alone, so two builds of one function cannot spell apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spec.tests.model import EVERY_DTYPE

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

QUOTIENT = tuple(sorted(EVERY_DTYPE))


@dataclass(frozen=True)
class Rule:
    """One declared area of a table: a set of cells, one outcome.

    The outcome is spelled as the specification's grammar spells it:
    `same` for the cell's own type back, `const:T` for a fixed type,
    `wrap:C` for the cell's type inside a constructor.

    Example:
        >>> Rule(frozenset({"Boolean"}), "const:UInt32").output_at("Boolean")
        'UInt32'
    """

    cells: frozenset[str]
    outcome: str

    def output_at(self, cell: str) -> str | None:
        """Apply this rule's outcome at one cell.

        Args:
            cell: The input type.

        Returns:
            The output type, or None when the cell is outside the rule.

        Example:
            >>> Rule(frozenset({"Int64"}), "same").output_at("Int64")
            'Int64'
        """
        if cell not in self.cells:
            return None
        return self._applied(cell)

    def _applied(self, cell: str) -> str:
        """Evaluate the outcome form at a covered cell.

        Args:
            cell: The input type, inside this rule's cells.

        Returns:
            The output type.

        Example:
            >>> Rule(frozenset({"Int64"}), "wrap:List")._applied("Int64")
            'List(Int64)'
        """
        if self.outcome == "same":
            return cell
        kind, _, argument = self.outcome.partition(":")
        if kind == "wrap":
            return f"{argument}({cell})"
        return argument

    def outputs(self) -> dict[str, str]:
        """Give every covered cell mapped to its output.

        Returns:
            One entry per cell of this rule.

        Example:
            >>> Rule(frozenset({"Boolean"}), "const:UInt32").outputs()
            {'Boolean': 'UInt32'}
        """
        return {cell: self._applied(cell) for cell in self.cells}


@dataclass(frozen=True)
class Table:
    """One expression type map over the quotient, stored extensionally.

    Example:
        >>> Table.from_rules([Rule(frozenset({"Int64"}), "same")]).output_at("Int64")
        'Int64'
    """

    outputs: tuple[tuple[str, str], ...]

    @classmethod
    def from_outputs(cls, outputs: Mapping[str, str]) -> Table:
        """Build a table from measured cell-to-output pairs.

        Args:
            outputs: Each defined cell mapped to its output type.

        Returns:
            The table, cells sorted.

        Outputs are normalised to the quotient on the way in: a
        parametrised rendering of a cell, `Duration(time_unit='us')`,
        becomes its cell, while a genuine wrap, `List(Int64)`, keeps
        the cell it carries.

        Example:
            >>> Table.from_outputs({"Int64": "Int64"}).output_at("Date") is None
            True
        """
        normalised = {c: cls._normalised(o) for c, o in outputs.items()}
        return cls(tuple(sorted(normalised.items())))

    @classmethod
    def _normalised(cls, output: str) -> str:
        """Collapse a parametrised rendering to its quotient cell.

        A text is a cell with parameters when its head is a cell and
        no cell name follows the opening bracket.

        Args:
            output: An output type's text.

        Returns:
            The cell name where the text is one cell with parameters,
            else the text unchanged.

        Example:
            >>> Table._normalised("Datetime(time_unit='us', time_zone=None)")
            'Datetime'
            >>> Table._normalised("Duration(time_unit='us')")
            'Duration'
            >>> Table._normalised("List(Int64)")
            'List(Int64)'
        """
        head, bracket, inside = output.partition("(")
        if not bracket or head not in QUOTIENT:
            return output
        if any(cell in inside for cell in QUOTIENT):
            return output
        return head

    @classmethod
    def from_rules(cls, rules: Iterable[Rule]) -> Table:
        """Build a table from declared rules, error staying the default.

        Args:
            rules: Disjoint areas with one outcome each.

        Returns:
            The table the rules denote.

        Example:
            >>> t = Table.from_rules([Rule(frozenset(QUOTIENT), "const:UInt32")])
            >>> t.output_at("String")
            'UInt32'
        """
        outputs: dict[str, str] = {}
        for rule in rules:
            outputs.update(rule.outputs())
        return cls.from_outputs(outputs)

    def output_at(self, cell: str) -> str | None:
        """Read one cell.

        Args:
            cell: The input type.

        Returns:
            The output type, or None for an error cell.

        Example:
            >>> Table.from_outputs({"Boolean": "UInt32"}).output_at("Boolean")
            'UInt32'
        """
        return dict(self.outputs).get(cell)

    def spelling(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Give the canonical spelling: outcomes, each with its cells.

        The spelling is derived from the outputs alone, never from how
        the table was built, and classification prefers `same` over a
        constant where both describe a cell, so one function has one
        spelling.

        Returns:
            Sorted (outcome, sorted cells) pairs; error by absence.

        Example:
            >>> Table.from_outputs({"Int64": "Int64"}).spelling()
            (('same', ('Int64',)),)
        """
        regions: dict[str, list[str]] = {}
        for cell, output in self.outputs:
            regions.setdefault(self._classified(cell, output), []).append(cell)
        return tuple(sorted((k, tuple(sorted(v))) for k, v in regions.items()))

    def _classified(self, cell: str, output: str) -> str:
        """Express one cell's output relative to its input.

        Args:
            cell: The input type.
            output: The measured output type.

        Returns:
            "same", a wrap pattern, or the constant output.

        Example:
            >>> Table(())._classified("Int64", "List(Int64)")
            'wrap:List'
        """
        if output == cell or output.startswith(cell + "("):
            return "same"
        if f"({cell})" in output:
            return "wrap:" + output.split(f"({cell})", maxsplit=1)[0]
        return "const:" + output
