"""Materialization: find the column that a declared expression gives on one frame.

The type of a declared expression does not depend on any frame. It holds the
columns that the expression needs, a table from the data types of those
columns to the data type it gives, an output name, and a character, that
is, how it counts rows. When the expression is an argument of an
operation such as `select`, the checker must know the one column that
it adds to that frame.

Materialization has four steps. It checks that the frame declares every
column that the expression needs, and it reads the table at the types
that the frame declares. It refuses a cell where the table has no
answer. Then it applies the rule of the context: inside `agg`, the
answer of an expression that works row by row becomes a `List`. A
column that the frame declares as `Unchecked` stands for every kind. So
the table is read at every kind on that axis, and the answers show as
one type.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.expression.context import ContextRule
from astrolabe._hooks.expression.lifting import UNCHECKED, Collapse
from astrolabe._hooks.expression.rendering import Rendered

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from astrolabe._hooks.expression.quadruple import ExpressionType
    from astrolabe._hooks.expression.table import Table


@dataclass(frozen=True)
class Materializer:
    """Place one typed expression into one frame, and give the column it produces.

    This class sits in the engine, and it works on plain values: the
    schema of the frame as text, and the type of the expression. Code
    that checks an argument of a frame operation calls it once for each
    argument. The answer is either the new column or a short text that
    says why there is none, which that code turns into an error.

    `members` holds the kinds at which an `Unchecked` column is read: the
    scalar kinds that a declaration can name one by one. A kind is a data
    type with its parameters erased, for example `Int64`. `nested` holds
    the other kinds, for example `List`. A column of one of those kinds
    is still possible. The two containers, `List` and `Array`, are always
    read. Where no kind gives an answer, the column is read at the other
    nested kinds too.

    Example:
        >>> from astrolabe._hooks.expression.character import ELEMENTWISE
        >>> from astrolabe._hooks.expression.quadruple import ExpressionType
        >>> from astrolabe._hooks.expression.table import Table
        >>> flag = Table.constant(("a",), ("Int64",), "Boolean")
        >>> expr = ExpressionType(frozenset({"a"}), flag, "a", ELEMENTWISE)
        >>> Materializer(("Int64",)).result({"a": "Int64"}, expr, "row")
        {'a': 'Boolean'}
    """

    members: tuple[str, ...]
    nested: tuple[str, ...] = ()

    # The nested kinds that every read covers, because a join point
    # answers them by the full type of the column.
    CONTAINERS: ClassVar[frozenset[str]] = frozenset({"Array", "List"})
    # The type that each container stands for in such a read: every
    # element, and for an `Array` every width.
    OPEN: ClassVar[dict[str, str]] = {
        "List": f"List({UNCHECKED})",
        "Array": f"Array({UNCHECKED}, shape=({UNCHECKED},))",
    }

    def result(
        self,
        frame: Mapping[str, str],
        expr: ExpressionType,
        context: str,
    ) -> dict[str, str] | str:
        """Give the column that one expression produces on one frame.

        Args:
            frame: The schema of the frame: each column name, with its
                declared data type as polars prints it.
            expr: The type of the expression.
            context: "row" for `select` and its kin, or "agg" for
                `group_by(...).agg(...)`.

        Returns:
            The output column as a schema with one entry, or a text that
            says what failed.

        Example:
            >>> from astrolabe._hooks.expression.character import ELEMENTWISE
            >>> from astrolabe._hooks.expression.quadruple import (
            ...     ExpressionType,
            ... )
            >>> from astrolabe._hooks.expression.table import Table
            >>> flag = Table.constant(("a",), ("Int64",), "Boolean")
            >>> expr = ExpressionType(frozenset({"a"}), flag, "a", ELEMENTWISE)
            >>> Materializer(("Int64", "String")).result({}, expr, "row")
            "the frame declares no column 'a'"
            >>> Materializer(("Int64", "String")).result({"a": "String"}, expr, "row")
            'the declaration gives no result at the data types of this frame'
        """
        missing = expr.needs - set(frame)
        if missing:
            return f"the frame declares no column '{min(missing)}'"
        answer = self.read(frame, expr.table)
        if answer is None:
            return "the declaration gives no result at the data types of this frame"
        applied = ContextRule().applied(answer, expr.character, context)
        return {expr.name: applied}

    def read(self, frame: Mapping[str, str], table: Table) -> str | None:
        """Read a table at the types that a frame declares.

        An axis that the frame declares as `Unchecked` is read at every
        scalar kind and at the containers among the nested kinds. The
        answers where the table is defined then show as one type. Where
        no such kind answers, the read tries the other nested kinds too.
        A table with no cell at all gives no answer at any kind, so it
        gets no such second read.

        Args:
            frame: The schema of the frame: each column name, with its
                declared data type as polars prints it.
            table: The table of the expression.

        Returns:
            The one answer, `Unchecked` for several different answers, or
            None if the table has no answer at any of the types.

        Example:
            >>> from astrolabe._hooks.expression.table import Table
            >>> count = Table.constant(("a",), ("String",), "UInt32")
            >>> Materializer(("String",)).read({"a": "Unchecked"}, count)
            'UInt32'
            >>> length = Table.constant(("a",), ("List",), "UInt32")
            >>> Materializer(("String",), ("List",)).read({"a": "Unchecked"}, length)
            'UInt32'
            >>> empty = Table.from_cells(("a",), {})
            >>> Materializer(("String",), ("Object",)).read({"a": "Unchecked"}, empty) is None
            True
            >>> from astrolabe._hooks.expression.outcome import Const
            >>> cells = {("String",): Const("Int8"), ("List",): Const("Int16")}
            >>> both = Table.from_cells(("a",), cells)
            >>> Materializer(("String",), ("List",)).read({"a": "Unchecked"}, both)
            'Unchecked'
        """
        containers = tuple(kind for kind in self.nested if kind in self.CONTAINERS)
        answer = self._read_over(frame, table, self.members + containers)
        if answer is not None or not self.nested or not table.cells:
            return answer
        return self._read_over(frame, table, self.members + self.nested)

    def _read_over(
        self,
        frame: Mapping[str, str],
        table: Table,
        kinds: tuple[str, ...],
    ) -> str | None:
        """Read a table at the types that a frame declares, with `Unchecked` over some kinds.

        Args:
            frame: The schema of the frame.
            table: The table of the expression.
            kinds: The kinds that an `Unchecked` column stands for here.

        Returns:
            The one answer, `Unchecked` for several different answers, or
            None if the table has no answer at any of the types.

        Example:
            >>> from astrolabe._hooks.expression.table import Table
            >>> length = Table.constant(("a",), ("List",), "UInt32")
            >>> Materializer(("Int64",))._read_over({"a": "Unchecked"}, length, ("List",))
            'UInt32'
        """
        widened = Materializer(kinds)
        assignments = widened._assignments(frame, table.columns)
        answers = {table.read_at(assignment) for assignment in assignments}
        return Collapse().collapsed(Collapse().defined(answers))

    def _assignments(
        self,
        frame: Mapping[str, str],
        columns: tuple[str, ...],
    ) -> Iterator[dict[str, Rendered]]:
        """List every combination of known types that a frame stands for.

        Args:
            frame: The schema of the frame. `Unchecked` stands for every
                kind.
            columns: The axes of the table.

        Yields:
            One full type for each axis, once for each combination.

        Example:
            >>> probe = Materializer(("Int64", "String"))
            >>> len(list(probe._assignments({"a": "Unchecked"}, ("a",))))
            2
        """
        sets = [self._axis_renderings(frame, name) for name in columns]
        for renderings in product(*sets):
            yield dict(zip(columns, renderings, strict=True))

    def _axis_renderings(self, frame: Mapping[str, str], name: str) -> tuple[Rendered, ...]:
        """Give the full types that one read of the map stands for on this frame.

        Args:
            frame: The schema of the frame. `Unchecked` stands for every
                kind.
            name: The name of the read.

        Returns:
            Every kind for `Unchecked`, the one declared type in other
            cases, or nothing for a name that the frame does not hold.

        Example:
            >>> probe = Materializer(("Int64",))
            >>> probe._axis_renderings({"a": "Int64"}, "a")
            (Rendered(kind='Int64', text='Int64'),)
            >>> probe._axis_renderings({"a": "Int64"}, "b")
            ()
        """
        text = frame.get(name)
        if text == UNCHECKED:
            return tuple(map(self._member, self.members))
        if text is None:
            return ()
        return (self._rendering(text),)

    def _rendering(self, text: str) -> Rendered:
        """Pair a declared type with its kind.

        Args:
            text: The declared type as polars prints it, with its
                parameters.

        Returns:
            The full type, with the part before the first parenthesis as
            its kind.

        Example:
            >>> Materializer(())._rendering("Datetime(time_unit='ns')").kind
            'Datetime'
        """
        return Rendered(text.split("(", maxsplit=1)[0], text)

    def _member(self, kind: str) -> Rendered:
        """Give the type that one kind stands for, where an `Unchecked` column takes it.

        A container keeps its element and its width unknown, so a member
        that reads the element answers `Unchecked` there. Every other
        kind is read bare, with no parameters.

        Args:
            kind: The kind.

        Returns:
            The type, with an `Unchecked` element for a container.

        Example:
            >>> Materializer(())._member("List").text
            'List(Unchecked)'
            >>> Materializer(())._member("Int64").text
            'Int64'
        """
        return Rendered(kind, self.OPEN.get(kind, kind))
