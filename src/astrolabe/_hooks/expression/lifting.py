"""The lifting: read a data type as a set of types, and show the result as one type.

The checker sometimes knows a column's name and not its data type. It
writes that data type as `Unchecked`. A rule of the engine must still
answer for such a column. The lifting makes the answer follow from two
plain rules, so no operation needs its own case for `Unchecked`.

- Each data type stands for a non-empty set of types. A known type is
  the set of that type alone, and `Unchecked` is the set of every type.
  The engine applies a rule to each member of the set, and the results
  where the rule is defined form the result set.
- The result set shows as one type. One member shows as itself.
  Several lists show as `List(Unchecked)`, and several other members
  show as `Unchecked`. An empty set means that no possible input
  works, which is an error.
"""

from __future__ import annotations

UNCHECKED = "Unchecked"


class Collapse:
    """Turn the set of possible results of a rule into the one type that a minted type shows.

    Code that applies a rule of the engine to each type an `Unchecked`
    column can have gets a set of answers back. A type that the checker
    mints, that is, builds for the user to see, holds one data type. This
    class gives that one data type, and it tells the error case apart.

    Example:
        >>> Collapse().collapsed(frozenset({"Int64"}))
        'Int64'
        >>> Collapse().collapsed(frozenset({"Int64", "Float64"}))
        'Unchecked'
    """

    def collapsed(self, members: frozenset[str]) -> str | None:
        """Show a set of possible results as one type.

        Several lists keep their kind: `List(Unchecked)` holds each of
        them, and it says more than `Unchecked`.

        Args:
            members: The data types that the rule can give.

        Returns:
            The one member, `List(Unchecked)` for several lists,
            `Unchecked` for several other members, or None for an empty
            set. None is the error case: no possible input works.

        Example:
            >>> Collapse().collapsed(frozenset()) is None
            True
            >>> Collapse().collapsed(frozenset({"List(Int8)", "List(String)"}))
            'List(Unchecked)'
            >>> Collapse().collapsed(frozenset({"List(Int8)", "Int8"}))
            'Unchecked'
        """
        if not members:
            return None
        if len(members) == 1:
            return next(iter(members))
        if all(member.startswith("List(") for member in members):
            return f"List({UNCHECKED})"
        return UNCHECKED

    def defined(self, answers: set[str | None]) -> frozenset[str]:
        """Keep the answers where the rule is defined.

        Args:
            answers: One result for each input type. None marks an input
                where the rule fails.

        Returns:
            The results that are not None.

        Example:
            >>> Collapse().defined({None, "Int64"})
            frozenset({'Int64'})
        """
        return frozenset(answer for answer in answers if answer is not None)
