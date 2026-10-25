"""The answer in one cell of the table of a declared expression.

A user declares the type of an expression with `typed_expr(...)` and its
kin: the columns that it reads, and what it gives for the data types of
those columns. The checker holds that declaration as a table. The table
has one axis for each column that the expression reads, and one cell for
each combination of kinds on those axes. A kind is a data type without
its parameters: `Datetime` stands for every time unit and time zone.
Each cell holds an outcome, which says what data type the expression
gives for inputs of those kinds.

A declaration writes an outcome in three forms: a fixed data type, the
data type of a read column with its parameters (`same`), and a container
around another outcome (`wrap`). This module holds those three forms.
There is no form for an error: a table has no cell where polars raises.

An outcome gives its data type only when a frame supplies the full data
types of the columns. Each column arrives as a rendering: the text that
polars prints for the data type, parameters included, for example
`Datetime(time_unit='us', time_zone=None)`. A column that the frame does
not pin to a rendering evaluates to `Unchecked`, the data type of a column
whose data type is not known.

This module is part of the engine, which knows nothing about mypy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astrolabe._hooks.expression.rendering import Rendered

UNCHECKED = "Unchecked"


@dataclass(frozen=True)
class Outcome:
    """The answer in one table cell, and the base of every outcome form.

    Each form answers two questions. `evaluated` gives the output data
    type once a frame supplies the renderings of the columns. `spelling`
    gives one line of text that names the form and its parts. Two
    tables are the same type exactly when their spellings are equal,
    so the spelling must not depend on how the outcome was built.

    An outcome is frozen, so it hashes by its parts. Code that collects
    the answers of one cell keeps them in a set, which merges two
    equal answers that came by different paths.

    Example:
        >>> Const("Boolean").spelling()
        'const:Boolean'
        >>> len({Const("Boolean"), Const("Boolean"), Const("Int64")})
        2
    """

    def evaluated(self, assignment: Mapping[str, Rendered]) -> str:
        """Give the output data type for the data types that a frame supplies.

        Args:
            assignment: The rendering of each read column, as the
                frame declares it.

        Returns:
            The rendering of the output data type.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> from astrolabe._hooks.expression.rendering import Rendered
            >>> Axis("a").evaluated({"a": Rendered.bare("Int64")})
            'Int64'
        """
        raise NotImplementedError

    def spelling(self) -> str:
        """Give the one line of text that identifies this outcome.

        Returns:
            The form's name, then its parts.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Wrap("List", Axis("t")).spelling()
            'wrap:List:axis:t'
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Const(Outcome):
    """A fixed output data type, which does not depend on the input.

    For example, `is_null()` gives `Boolean` for every input.

    Example:
        >>> Const("UInt32").evaluated({})
        'UInt32'
    """

    dtype: str

    def evaluated(self, assignment: Mapping[str, Rendered]) -> str:
        """Give the fixed data type.

        Args:
            assignment: Not read. The answer does not depend on it.

        Returns:
            The rendering of the fixed data type.

        Example:
            >>> from astrolabe._hooks.expression.rendering import Rendered
            >>> Const("Boolean").evaluated({"a": Rendered.bare("Date")})
            'Boolean'
        """
        del assignment
        return self.dtype

    def spelling(self) -> str:
        """Give the one line of text that identifies this outcome.

        Returns:
            `const:`, then the data type.

        Example:
            >>> Const("Int64").spelling()
            'const:Int64'
        """
        return f"const:{self.dtype}"


@dataclass(frozen=True)
class Axis(Outcome):
    """The input column's own data type, parameters included.

    For example, `sort()` on a column gives that column's data type
    back. The axis gives the full rendering, so a `Datetime` keeps its
    time unit and time zone. If the frame does not pin the column to a
    rendering, the axis gives `Unchecked`.

    Example:
        >>> from astrolabe._hooks.expression.rendering import Rendered
        >>> Axis("price").evaluated({"price": Rendered.bare("Float64")})
        'Float64'
    """

    column: str

    def evaluated(self, assignment: Mapping[str, Rendered]) -> str:
        """Give the full rendering of the column that this axis names.

        Args:
            assignment: The rendering of each read column.

        Returns:
            The column's rendering, or `Unchecked` where the
            assignment does not hold the column.

        Example:
            >>> Axis("a").evaluated({})
            'Unchecked'
        """
        rendered = assignment.get(self.column)
        if rendered is None:
            return UNCHECKED
        return rendered.text

    def spelling(self) -> str:
        """Give the one line of text that identifies this outcome.

        Returns:
            `axis:`, then the column's name.

        Example:
            >>> Axis("price").spelling()
            'axis:price'
        """
        return f"axis:{self.column}"


@dataclass(frozen=True)
class Wrap(Outcome):
    """A container data type around another outcome's answer.

    For example, `implode()` on a column of type `t` gives `List(t)`.
    A wrap around `Unchecked` gives `List(Unchecked)`: the inner answer
    stands for any data type, and the wrap stands for every list of
    them.

    Example:
        >>> from astrolabe._hooks.expression.rendering import Rendered
        >>> Wrap("List", Axis("a")).evaluated({"a": Rendered.bare("Int64")})
        'List(Int64)'
    """

    constructor: str
    inner: Outcome

    def evaluated(self, assignment: Mapping[str, Rendered]) -> str:
        """Put the container around the inner outcome's answer.

        Args:
            assignment: The rendering of each read column.

        Returns:
            The container's rendering.

        Example:
            >>> Wrap("List", Axis("a")).evaluated({})
            'List(Unchecked)'
        """
        return f"{self.constructor}({self.inner.evaluated(assignment)})"

    def spelling(self) -> str:
        """Give the one line of text that identifies this outcome.

        Returns:
            `wrap:`, the container, then the inner spelling.

        Example:
            >>> Wrap("List", Axis("a")).spelling()
            'wrap:List:axis:a'
        """
        return f"wrap:{self.constructor}:{self.inner.spelling()}"
