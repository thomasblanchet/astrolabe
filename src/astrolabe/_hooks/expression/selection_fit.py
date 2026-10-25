"""Say in plain words how a selection renames the columns that it picks.

A user can declare a function parameter as a selection, for example with
`typed_selection(...)`, and pass `cs.numeric().abs()` to it. The fit is the
check that the argument matches that declaration, and its rename is one part of
that check. An error message about a failed fit compares two renames, and this
module words each one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrolabe._hooks.expression.selection import Affix


class AffixWording:
    """Say in plain words what a selection adds to the names of the columns it picks.

    An error message about a failed fit compares two renames. A
    selection can add a prefix, a suffix, both, or nothing. The phrase
    completes a sentence that starts "this expression ...".

    Example:
        >>> from astrolabe._hooks.expression.selection import Affix
        >>> AffixWording().phrase(Affix(suffix="_r"))
        'adds the suffix "_r"'
    """

    def phrase(self, affix: Affix) -> str:
        """Give the plain phrase for one rename.

        Args:
            affix: The prefix and the suffix that the selection adds.

        Returns:
            The phrase, which completes a sentence about a selection.

        Example:
            >>> from astrolabe._hooks.expression.selection import Affix
            >>> AffixWording().phrase(Affix())
            'adds nothing to the column names'
            >>> AffixWording().phrase(Affix(prefix="x_", suffix="_r"))
            'adds the prefix "x_" and the suffix "_r"'
            >>> AffixWording().phrase(Affix(prefix="x_"))
            'adds the prefix "x_"'
        """
        written = (("prefix", affix.prefix), ("suffix", affix.suffix))
        parts = [f'the {half} "{text}"' for half, text in written if text]
        if not parts:
            return "adds nothing to the column names"
        return f"adds {' and '.join(parts)}"
