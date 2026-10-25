"""Read a type that mypy revealed, so that the page can show a frame as its columns.

mypy writes each type with its full module path, for example
`polars.datatypes.classes.Int64` for `Int64`. A reader wants the short
names, and the page wants the columns themselves: for a frame, each
declared column with its data type, and the remainder if there is one.
"""

from __future__ import annotations

import re
from typing import ClassVar


class RevealedType:
    """One revealed type, written without module paths, and read as a frame where it is one.

    A typed frame reveals as its kind, a `TypedDict` of its declared
    columns, and its remainder: `None` for a closed frame, or the
    remainder of an open frame. Any other type has no columns.

    Example:
        >>> frame = "astrolabe.frame.TypedDataFrame"
        >>> int64 = "polars.datatypes.classes.Int64"
        >>> revealed = RevealedType(f"{frame}[TypedDict({{'a': {int64}}}, closed=True), None]")
        >>> revealed.text
        "TypedDataFrame[TypedDict({'a': Int64}, closed=True), None]"
        >>> revealed.columns(), revealed.remainder()
        ([('a', 'Int64')], None)
    """

    QUALIFIER: ClassVar[re.Pattern[str]] = re.compile(r"\b(?:[a-z_][a-z0-9_]*\.)+(?=[A-Za-z_])")
    FRAME: ClassVar[re.Pattern[str]] = re.compile(
        r"^(TypedDataFrame|TypedLazyFrame)\[TypedDict\((?:\w+, )?\{(.*)\}, closed=True\), (.+)\]$",
    )
    OPENING: ClassVar[str] = "[({"
    CLOSING: ClassVar[str] = "])}"

    def __init__(self, text: str) -> None:
        """Keep the type without the module paths, and leave the text in quotes as it is.

        Args:
            text: The type as mypy reveals it.

        Example:
            >>> RevealedType("main.Rest").text
            'Rest'
        """
        pieces = text.split("'")
        for index in range(0, len(pieces), 2):
            pieces[index] = self.QUALIFIER.sub("", pieces[index])
        self.text = "'".join(pieces)

    def alternatives(self) -> list[RevealedType]:
        """Give each member of a union, or the type itself when it is not a union.

        A function with an inferred return type gives the union of the
        types of its returned values, for example two frames.

        Returns:
            One revealed type for each member.

        Example:
            >>> frame = "TypedDataFrame[TypedDict({'a': Int64}, closed=True), None]"
            >>> union = RevealedType(f"{frame} | int")
            >>> [member.text for member in union.alternatives()]
            ["TypedDataFrame[TypedDict({'a': Int64}, closed=True), None]", 'int']
        """
        return [RevealedType(part) for part in self._split(self.text, " | ")]

    def kind(self) -> str | None:
        """Give the kind of frame, or None for a type that is not a typed frame.

        Returns:
            `TypedDataFrame`, `TypedLazyFrame` or None.

        Example:
            >>> RevealedType("TypedLazyFrame[TypedDict({}, closed=True), None]").kind()
            'TypedLazyFrame'
            >>> RevealedType("builtins.int").kind() is None
            True
        """
        match = self.FRAME.match(self.text)
        return None if match is None else match.group(1)

    def columns(self) -> list[tuple[str, str]] | None:
        """Give the declared columns of the frame, each with its data type.

        Returns:
            The columns in the order that mypy reveals them, or None for a
            type that is not a typed frame.

        Example:
            >>> body = "{'t': Datetime[Literal['us'], None], 'v': Int64}"
            >>> RevealedType(f"TypedDataFrame[TypedDict({body}, closed=True), None]").columns()
            [('t', "Datetime[Literal['us'], None]"), ('v', 'Int64')]
        """
        match = self.FRAME.match(self.text)
        if match is None:
            return None
        columns: list[tuple[str, str]] = []
        for item in self._split(match.group(2), ","):
            name, _, dtype = item.partition(": ")
            columns.append((name.strip("'"), dtype))
        return columns

    def remainder(self) -> str | None:
        """Give the remainder of an open frame, or None for a closed frame or another type.

        Returns:
            The remainder as the type writes it, or None.

        Example:
            >>> remainder = "Minus[Rest, Literal['a']]"
            >>> frame = f"TypedDataFrame[TypedDict({{}}, closed=True), {remainder}]"
            >>> RevealedType(frame).remainder()
            "Minus[Rest, Literal['a']]"
        """
        match = self.FRAME.match(self.text)
        if match is None or match.group(3) == "None":
            return None
        return match.group(3)

    def _split(self, body: str, separator: str) -> list[str]:
        """Split a text at each separator that is not inside brackets or quotes.

        Args:
            body: The text, for example the body of a `TypedDict`.
            separator: The separator, for example `","` or `" | "`.

        Returns:
            The parts, without the empty ones.

        Example:
            >>> RevealedType("x")._split("'a': List[Int64], 'b, c': String", ",")
            ["'a': List[Int64]", "'b, c': String"]
            >>> RevealedType("x")._split("", ",")
            []
        """
        parts: list[str] = []
        depth = 0
        quoted = False
        start = 0
        for index, char in enumerate(body):
            if char == "'":
                quoted = not quoted
            elif not quoted:
                depth += (char in self.OPENING) - (char in self.CLOSING)
                if depth == 0 and body.startswith(separator, index):
                    parts.append(body[start:index].strip())
                    start = index + len(separator)
        tail = body[start:].strip()
        if tail:
            parts.append(tail)
        return parts
