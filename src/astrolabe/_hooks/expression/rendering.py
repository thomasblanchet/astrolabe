"""The full data type of one column, as the text that polars prints for it.

The checker types an expression with a table over kinds. A kind is a
data type without its parameters, for example `Datetime` for every
time unit. To give the exact output type, the checker needs more: the
full data type of each input column, for example
`Datetime(time_unit='us', time_zone=None)`. This module calls that text
a rendering, and reads its parts: the element type of a `List` or an
`Array`, the fields of a `Struct`, a time unit, a time zone.

Every outcome and every parameter rule of the engine reads data types
through this module. So this module imports no other part of the
engine, and any of them can import it. The engine is the half of the
plugin that knows nothing about mypy.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import ClassVar


class ArrayText:
    """Read and write the text of an `Array` data type: an element type and a shape.

    Polars writes an `Array` of `Array` with one shape tuple, the outer
    width first. An `Array` of three `Array` of two `Int8` prints as
    `Array(Int8, shape=(3, 2))`, not as an `Array` inside an `Array`.
    An `Array` of `List` keeps the `List` inside, because a `List` has
    no width. Code that reads an `Array` rendering and code that writes
    one both use this class, so the two sides agree on one form.

    Example:
        >>> ArrayText().folded("Int8", "3")
        'Array(Int8, shape=(3,))'
        >>> ArrayText().folded("Array(Int8, shape=(2,))", "3")
        'Array(Int8, shape=(3, 2))'
        >>> ArrayText().folded("List(Int64)", "3")
        'Array(List(Int64), shape=(3,))'
    """

    HEAD: ClassVar[str] = "Array("
    SHAPE: ClassVar[str] = ", shape=("

    def folded(self, element: str, width: str) -> str:
        """Write the `Array` of a given width around an element type.

        Args:
            element: The rendering of the element type.
            width: The width of the new outer dimension.

        Returns:
            The `Array` rendering. Where the element is an `Array`, its
            widths join the one shape after the new width.

        Example:
            >>> ArrayText().folded("Int8", "2")
            'Array(Int8, shape=(2,))'
        """
        inner = self.read(element)
        if inner is None:
            return f"{self.HEAD}{element}{self.SHAPE}{width},))"
        widths = ", ".join((width, *inner[1]))
        return f"{self.HEAD}{inner[0]}{self.SHAPE}{widths}))"

    def read(self, text: str) -> tuple[str, tuple[str, ...]] | None:
        """Read the innermost element type and every width of an `Array` rendering.

        Args:
            text: The full rendering.

        Returns:
            The rendering of the innermost element type and one width
            for each dimension, or None where the text names no
            `Array` shape.

        Example:
            >>> ArrayText().read("Array(Int8, shape=(3, 2))")
            ('Int8', ('3', '2'))
            >>> ArrayText().read("Array") is None
            True
            >>> ArrayText().read("Array(Int8)") is None
            True
        """
        if not text.startswith(self.HEAD):
            return None
        body = text.removeprefix(self.HEAD).removesuffix(")")
        element, marker, shape = body.rpartition(self.SHAPE)
        if not marker:
            return None
        return element, self.widths(shape.removesuffix(")"))

    def widths(self, shape: str) -> tuple[str, ...]:
        """Read the widths from the text of one shape.

        Args:
            shape: The text between the parentheses of the shape.

        Returns:
            One width for each dimension. The comma after the only
            width of a one-dimension shape gives no width.

        Example:
            >>> ArrayText().widths("3, 2")
            ('3', '2')
            >>> ArrayText().widths("2,")
            ('2',)
        """
        return tuple(filter(None, map(str.strip, shape.split(","))))


@dataclass(frozen=True)
class Rendered:
    """The full data type of one column: its kind, and the text that polars prints.

    An outcome in an expression's table gives its answer from these.
    `kind` is the data type without parameters, for example `Datetime`.
    `text` is the full rendering, for example
    `Datetime(time_unit='us', time_zone=None)`. Where the text equals
    the kind, the parameters are not known, and a rule that needs them
    gives the kind alone.

    Example:
        >>> Rendered("Datetime", "Datetime(time_unit='us', time_zone=None)").kind
        'Datetime'
    """

    kind: str
    text: str

    @classmethod
    def bare(cls, kind: str) -> Rendered:
        """Build a rendering that is only a kind, with no parameters.

        Code that works on kinds alone, for example a table read at
        one cell, uses this form.

        Args:
            kind: The name of the kind.

        Returns:
            The rendering whose text is the kind itself.

        Example:
            >>> Rendered.bare("Int64").text
            'Int64'
        """
        return cls(kind, kind)


class NameText:
    """Write and read one name inside a rendering, as polars prints it.

    A rendering holds names: the field names of a `Struct` and the
    categories of an `Enum`. polars prints each name as Python prints a
    string, in quotes: `'a'`, or `"a, 'b"` for a name that holds a single
    quote. A name can hold a comma or a quote, so code that splits a
    rendering must skip the text inside quotes, and code that writes a
    rendering must quote each name in the same way as polars.

    Example:
        >>> print(NameText().spelled("a, 'b"))
        "a, 'b"
        >>> NameText().names("['a, b', 'c']")
        ['a, b', 'c']
    """

    def spelled(self, name: str) -> str:
        """Write one name in quotes, as polars prints it.

        Args:
            name: The name.

        Returns:
            The quoted name.

        Example:
            >>> NameText().spelled("a")
            "'a'"
        """
        return repr(name)

    def listed(self, names: list[str]) -> str:
        """Write a list of names in brackets, as polars prints the categories of an `Enum`.

        Args:
            names: The names, in order.

        Returns:
            The list text.

        Example:
            >>> NameText().listed(["a", "b"])
            "['a', 'b']"
        """
        return "[" + ", ".join(self.spelled(name) for name in names) + "]"

    def names(self, listed: str) -> list[str]:
        """Read a list of quoted names back.

        Args:
            listed: The list text, brackets included.

        Returns:
            The names, in order.

        Example:
            >>> NameText().names("[]")
            []
        """
        return [self._unquoted(piece) for piece in TopLevelSplit().of(listed[1:-1]) if piece]

    def leading(self, piece: str) -> tuple[str, str]:
        """Split one piece of text into the quoted name at its start and the text after it.

        A backslash escapes the next character, so an escaped quote does
        not close the name.

        Args:
            piece: The text, which starts with a quoted name.

        Returns:
            The name, and the text after the name.

        Example:
            >>> NameText().leading("'a, b': Int8")
            ('a, b', ': Int8')
            >>> NameText().leading("'a" + chr(92) + "'b' x")
            ("a'b", ' x')
        """
        text = piece.strip()
        quote = text[0]
        end = 1
        while text[end] != quote:
            end += 2 if text[end] == "\\" else 1
        return self._unquoted(text[: end + 1]), text[end + 1 :]

    def _unquoted(self, quoted: str) -> str:
        """Read one quoted name.

        Args:
            quoted: The name in its quotes, with any space around it.

        Returns:
            The name.

        Example:
            >>> NameText()._unquoted(" 'a' ")
            'a'
        """
        return str(ast.literal_eval(quoted.strip()))


class TopLevelSplit:
    """Split a rendering at the commas outside every bracket and every quoted name.

    A field's data type can hold commas of its own, and so can a quoted
    name, so a plain split at each comma is wrong. This class reads the
    text one character at a time and splits only at a comma that no
    parenthesis, brace, bracket or quote encloses. One instance splits
    one text.

    Example:
        >>> TopLevelSplit().of("'a': List(X, Y), 'b, c': Z")
        ["'a': List(X, Y)", " 'b, c': Z"]
    """

    def __init__(self) -> None:
        """Start a split with no text read."""
        self._depth = 0
        self._quote: str | None = None
        self._escaped = False
        self._current = ""
        self._pieces: list[str] = []

    def of(self, text: str) -> list[str]:
        """Split one text.

        Args:
            text: The text.

        Returns:
            One piece of text for each top-level item.
        """
        for char in text:
            if self._quote is not None:
                self._read_quoted(char)
            elif char == "," and self._depth == 0:
                self._pieces.append(self._current)
                self._current = ""
            else:
                self._read_plain(char)
        self._pieces.append(self._current)
        return self._pieces

    def _read_quoted(self, char: str) -> None:
        """Read one character inside a quoted name.

        A backslash escapes the next character, so an escaped quote does
        not close the name.

        Args:
            char: The character.

        Example:
            >>> split = TopLevelSplit()
            >>> split._quote = "'"
            >>> split._read_quoted(chr(92))
            >>> split._read_quoted("'")
            >>> split._quote
            "'"
        """
        self._current += char
        if self._escaped:
            self._escaped = False
            return
        self._escaped = char == "\\"
        if char == self._quote:
            self._quote = None

    def _read_plain(self, char: str) -> None:
        """Read one character outside every quote.

        Args:
            char: The character.
        """
        self._current += char
        if char in "'\"":
            self._quote = char
            return
        self._depth += self._DEPTH.get(char, 0)

    # How each bracket changes the depth.
    _DEPTH: ClassVar[dict[str, int]] = {"(": 1, "{": 1, "[": 1, ")": -1, "}": -1, "]": -1}


class StructFields:
    """Split the fields of a `Struct` rendering into names and data types.

    Polars prints the fields of a `Struct` as a dict, for example
    `Struct({'a': List(Int64), 'b': String})`. A field's data type can
    hold commas of its own, and a name can hold a comma or a quote.
    This class splits at the top-level commas only, and reads each
    quoted name back.

    Example:
        >>> StructFields().of("'len': UInt32, 'value': Date")
        (('len', 'UInt32'), ('value', 'Date'))
    """

    def of(self, body: str) -> tuple[tuple[str, str], ...]:
        """Split the text inside one `Struct(...)` into its fields.

        Args:
            body: The text between `Struct(` and the last `)`. The
                braces of the dict are optional.

        Returns:
            The name and the rendering of each field, in order.

        Example:
            >>> StructFields().of("'a, b': Int8")
            (('a, b', 'Int8'),)
            >>> StructFields().of("{}")
            ()
        """
        inner = body.removeprefix("{").removesuffix("}")
        pieces = [piece for piece in TopLevelSplit().of(inner) if piece.strip()]
        fields: list[tuple[str, str]] = []
        for piece in pieces:
            name, rest = NameText().leading(piece)
            fields.append((name, rest.removeprefix(":").strip()))
        return tuple(fields)
