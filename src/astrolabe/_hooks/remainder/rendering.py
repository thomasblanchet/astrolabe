"""Text for an engine value that is the same in every Python process.

The engine sorts some values to give them one fixed order, for
example the operands of a union of remainders or the two sides of a
selector conjunction. It sorts them by their text. That text must not
change between two runs of mypy, or one type gets two spellings. The
`repr` of a `frozenset` lists its members in hash order, and string
hashes change from one process to the next. This module therefore
writes every set with its members sorted.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass


class StableText:
    """Write an engine value as text that does not depend on hash order.

    Code that needs a fixed order for engine values uses this text as
    the sort key. It walks frozen dataclasses and sets, and it uses
    `repr` for every other value.

    Example:
        >>> StableText()(frozenset({"b", "a"}))
        "{'a','b'}"
    """

    def __call__(self, value: object) -> str:
        """Write one value as text.

        A set lists the text of its members, sorted, inside braces. A
        dataclass gives its class name and the text of each field that
        takes part in equality. A field marked `compare=False` holds a
        cache or a note for an error message, so the text leaves it
        out: two equal values must give the same text.

        Args:
            value: A set, a dataclass instance, or any value with a
                stable `repr`.

        Returns:
            Text that is the same in every process for equal values.

        Example:
            >>> StableText()("a")
            "'a'"
            >>> StableText()(frozenset({2, 1}))
            '{1,2}'
            >>> from dataclasses import dataclass
            >>> @dataclass(frozen=True)
            ... class Point:
            ...     x: int
            >>> StableText()(Point(1))
            'Point(1)'
        """
        if isinstance(value, frozenset):
            return "{" + ",".join(sorted(self(m) for m in value)) + "}"
        if is_dataclass(value) and not isinstance(value, type):
            parts = [self(getattr(value, f.name)) for f in fields(value) if f.compare]
            return f"{type(value).__name__}({','.join(parts)})"
        return repr(value)
