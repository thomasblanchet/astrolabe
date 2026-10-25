"""The character of a declared expression: how it counts rows.

Each polars expression has one of three characters. An elementwise
expression gives one value for each row, for example `pl.col("a") + 1`.
An aggregating expression combines the rows into one value, for example
`pl.col("a").sum()`. A literal expression is a fixed value, for example
`pl.lit(1)`. A declaration states the character of the expressions that
it accepts, and an argument fits it only with the same character.

This module holds the three characters, finds one from the label that a
declaration or an encoded type holds, and words each one for an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


class CharacterWording:
    """Say what a character means, without its internal name.

    An error message tells a user how an expression treats rows. The
    internal name of a character, for example "aggregating", says little
    to a user who does not know the design. So each character has a plain
    phrase that completes the sentence "this expression ...". Code that
    writes an error about a character uses this phrase.

    Example:
        >>> CharacterWording().phrase("aggregating")
        'combines rows into one value'
    """

    PHRASES: ClassVar[dict[str, str]] = {
        "elementwise": "works row by row",
        "aggregating": "combines rows into one value",
        "literal": "is a fixed value",
    }

    def phrase(self, label: str) -> str:
        """Give the plain phrase for one character.

        Args:
            label: The internal name of the character.

        Returns:
            The phrase, which completes the sentence "this expression
            ...".

        Example:
            >>> CharacterWording().phrase("literal")
            'is a fixed value'
        """
        return self.PHRASES[label]


@dataclass(frozen=True)
class Character:
    """One of the three characters of a declared expression.

    A declared expression type holds its character. The three instances
    are `LITERAL`, `AGGREGATING` and `ELEMENTWISE`, and code compares them
    by identity.

    Example:
        >>> ELEMENTWISE.label
        'elementwise'
    """

    label: str


LITERAL = Character("literal")
AGGREGATING = Character("aggregating")
ELEMENTWISE = Character("elementwise")


class CharacterRoster:
    """Find a character from its label.

    A declaration and an encoded type hold a character as its label, a
    plain string. Code that reads one of them back uses this class to get
    the one shared `Character` instance.

    Example:
        >>> CharacterRoster().named("literal") is LITERAL
        True
    """

    def named(self, label: str) -> Character:
        """Give the character that a label names.

        Args:
            label: "elementwise", "aggregating" or "literal".

        Returns:
            The character. An unknown label is a bug in the caller, and
            the lookup then fails with a `KeyError`.

        Example:
            >>> CharacterRoster().named("aggregating") is AGGREGATING
            True
        """
        by_label = {
            "literal": LITERAL,
            "aggregating": AGGREGATING,
            "elementwise": ELEMENTWISE,
        }
        return by_label[label]
