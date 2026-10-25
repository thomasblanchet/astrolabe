"""Check that the code of a deferred argument fits the expression type that its parameter declares.

A user can declare a parameter as an expression type, for example
`price: column("price", pl.Float64)`. In the deferred model, the type of an
argument holds its code, not its data types. So the check at the call
evaluates the code at each point of the declared domain: at each data type
that the declaration accepts for each column that it reads. The argument fits
where five things hold. It reads only the columns that the declaration reads.
Its output name is the declared name. It counts rows in the declared way: row
by row, as one value, or as a fixed value. At each point, it gives the declared
data type. And polars plans it at the data type that it runs, because a body
that reads the parameter reads one data type in every pass.
"""

from __future__ import annotations

import ast
import itertools
import math
from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.deferred.evaluation import DeferredEvaluation
from astrolabe._hooks.deferred.open_types import OpenType
from astrolabe._hooks.expression.character import CharacterWording
from astrolabe._hooks.expression.fit import CellRefusal, Fit
from astrolabe._hooks.expression.lifting import UNCHECKED
from astrolabe._hooks.recording.argument_code import DataTypeCode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from astrolabe._hooks.deferred.code import Code
    from astrolabe._hooks.deferred.evaluation import DeclaredLeaf, DeferredAnswer
    from astrolabe._hooks.expression.fit import Row, SliceDecl


class DeferredFit:
    """Give the first reason that the code of a deferred argument does not fit a declaration.

    Example:
        >>> from astrolabe._hooks.deferred.code import Code
        >>> from astrolabe._hooks.expression.fit import Read, Row, SliceDecl
        >>> from astrolabe._hooks.expression.outcome import Axis
        >>> decl = SliceDecl(
        ...     (Read("a", frozenset({"Int64"})),),
        ...     (Row((frozenset({"Int64"}),), Axis("a")),),
        ...     None, "elementwise",
        ... )
        >>> DeferredFit().verdict(Code("pl.col('a').abs()"), decl) is None
        True
        >>> DeferredFit().verdict(Code("pl.col('a').sum()"), decl)
        'it combines rows into one value, but the declared type works row by row'
    """

    LITERAL: ClassVar[str] = "literal"
    LIMIT: ClassVar[int] = 256

    def verdict(self, code: Code[DeclaredLeaf | None], decl: SliceDecl) -> str | None:
        """Check the code of one argument against one declaration.

        Args:
            code: The code of the argument.
            decl: The declaration.

        Returns:
            None where the argument fits. Otherwise the first reason that it
            does not, as the end of a sentence about the argument.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.expression.fit import Read, Row, SliceDecl
            >>> from astrolabe._hooks.expression.outcome import Const
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int64"})),),
            ...     (Row((frozenset({"Int64"}),), Const("Boolean")),),
            ...     None, "elementwise",
            ... )
            >>> DeferredFit().verdict(Code("pl.col('b') > 1"), decl)
            'it uses column "b", which the declared type does not allow'
            >>> DeferredFit().verdict(Code("pl.col('a') + 1"), decl)
            'it makes an Int64 column, but the declared type is Boolean'
        """
        reads = DeferredEvaluation("row").reads(code)
        outside = self._outside(reads, decl) or self._dynamic(code, decl)
        if outside is not None:
            return outside
        for place, (row, point) in enumerate(self._points(decl)):
            columns = self._columns(decl, point, reads)
            answer = DeferredEvaluation("row").answer(code, columns)
            declared = decl.outcome_at(row, point)
            refusal = self._shape(code, columns, answer, decl) if place == 0 else None
            refusal = refusal or self._cell(code, columns, answer, declared)
            if refusal is not None:
                return refusal
        return None

    def _points(self, decl: SliceDecl) -> list[tuple[Row, dict[str, str]]]:
        """Give each point of the declared domain, with the row that covers it.

        A point is one data type for each checked read. A declared cell
        holds kinds, and a kind such as `Datetime` stands for each of its
        units and zones. The argument must give the declared outcome at
        each of them, so each cell gives one point for each data type of
        the sample of its kinds. A cell whose samples are more than `LIMIT`
        gives the one point of its kinds.

        Args:
            decl: The declaration.

        Returns:
            The points, each with its row.

        Example:
            >>> from astrolabe._hooks.expression.fit import Read, Row, SliceDecl
            >>> from astrolabe._hooks.expression.outcome import Axis
            >>> decl = SliceDecl(
            ...     (Read("d", frozenset({"Decimal"})),),
            ...     (Row((frozenset({"Decimal"}),), Axis("d")),),
            ...     None, "elementwise",
            ... )
            >>> [point for _, point in DeferredFit()._points(decl)]
            [{'d': 'Decimal(precision=10, scale=2)'}, {'d': 'Decimal(precision=38, scale=0)'}]
        """
        points: list[tuple[Row, dict[str, str]]] = []
        for row in decl.rows:
            for cell in decl.declared_cells(row):
                points.extend((row, point) for point in self._sampled(cell))
        return points

    def _sampled(self, cell: Mapping[str, str]) -> list[dict[str, str]]:
        """Give each choice of one data type of the sample of each kind of a cell.

        Args:
            cell: The kind of each checked read, by its name.

        Returns:
            The choices, each with a data type for each read.

        Example:
            >>> len(DeferredFit()._sampled({"t": "Datetime", "u": "Duration"}))
            9
            >>> DeferredFit()._sampled({"a": "Int8"})
            [{'a': 'Int8'}]
            >>> DeferredFit()._sampled({"a": "List", "b": "List"})
            [{'a': 'List', 'b': 'List'}]
        """
        options = [OpenType().renderings(kind) for kind in cell.values()]
        if math.prod(len(samples) for samples in options) > self.LIMIT:
            return [dict(cell)]
        choices = itertools.product(*options)
        return [dict(zip(cell, choice, strict=True)) for choice in choices]

    def _outside(self, reads: Sequence[str], decl: SliceDecl) -> str | None:
        """Check that the argument reads only columns that the declaration reads.

        A blank read accepts the columns of the argument, because the call
        binds the blank to them.

        Args:
            reads: The columns that the argument reads.
            decl: The declaration.

        Returns:
            The reason that the argument does not fit, or None.

        Example:
            >>> from astrolabe._hooks.expression.fit import Read, SliceDecl
            >>> decl = SliceDecl((Read("a", frozenset({"Int64"})),), (), None, "elementwise")
            >>> DeferredFit()._outside(["a"], decl) is None
            True
        """
        allowed = set(decl.read_names())
        if decl.blank_count():
            allowed |= set(reads)
        outside = sorted(set(reads) - allowed)
        if not outside:
            return None
        quoted = ", ".join(f'"{name}"' for name in outside)
        noun = "column" if len(outside) == 1 else "columns"
        return f"it uses {noun} {quoted}, which the declared type does not allow"

    def _dynamic(self, code: Code[DeclaredLeaf | None], decl: SliceDecl) -> str | None:
        """Refuse a literal whose value decides its data type, where the declaration fixes one.

        Beside a column, polars joins `pl.lit(1)` by the range of its value:
        it gives `UInt8` beside a `UInt8` column, where a literal of the
        declared `Int32` gives `Int32`. So such a literal does not fit a
        literal of a fixed data type. A declared `Unchecked` accepts it.

        Args:
            code: The code of the argument.
            decl: The declaration.

        Returns:
            The reason that the argument does not fit, or None.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.expression.fit import Row, SliceDecl
            >>> from astrolabe._hooks.expression.outcome import Const
            >>> one = SliceDecl((), (Row((), Const("Int32")),), "literal", "literal")
            >>> reason = DeferredFit()._dynamic(Code("pl.lit(1)"), one)
            >>> reason.endswith("write pl.lit(..., dtype=pl.Int32)")
            True
            >>> DeferredFit()._dynamic(Code("pl.lit(1, dtype=pl.Int32)"), one) is None
            True
        """
        if decl.character != self.LITERAL or not decl.rows:
            return None
        declared = decl.outcome_at(decl.rows[0], {})
        if declared == UNCHECKED or not DeferredEvaluation("row").is_dynamic(code):
            return None
        written = ast.unparse(DataTypeCode().of(declared))
        return (
            "its value decides its data type beside a column, but the declared type is "
            f"{declared}: write pl.lit(..., dtype={written})"
        )

    def _columns(
        self,
        decl: SliceDecl,
        cell: Mapping[str, str],
        reads: Sequence[str],
    ) -> dict[str, str]:
        """Give the frame of one point of the declared domain: the data type of each declared read.

        A blank read stands for the one column that the argument reads. A
        read that the declaration leaves `Unchecked` takes every data type.

        Args:
            decl: The declaration.
            cell: The kind of each read at the point, by its name.
            reads: The columns that the argument reads.

        Returns:
            The rendering of the data type of each column, by its name.

        Example:
            >>> from astrolabe._hooks.expression.fit import Read, SliceDecl
            >>> decl = SliceDecl((Read(None, frozenset({"Int8"})),), (), None, "elementwise")
            >>> DeferredFit()._columns(decl, {"": "Int8"}, ["price"])
            {'price': 'Int8'}
        """
        columns: dict[str, str] = {}
        for read in decl.reads:
            names = [read.name] if read.name is not None else list(reads)
            for name in names:
                columns[name] = cell.get(read.name or "", UNCHECKED)
        return columns

    def _shape(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        answer: DeferredAnswer,
        decl: SliceDecl,
    ) -> str | None:
        """Check the output name and the way that the argument counts rows, at one point.

        Args:
            code: The code of the argument.
            columns: The frame of the point.
            answer: The answer of the argument in a `select` there.
            decl: The declaration.

        Returns:
            The reason that the argument does not fit, or None.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.deferred.evaluation import DeferredAnswer
            >>> from astrolabe._hooks.expression.fit import Read, SliceDecl
            >>> decl = SliceDecl((Read("a", frozenset({"Int64"})),), (), "a", "elementwise")
            >>> answer = DeferredAnswer("b", "Int64", reads=("a",))
            >>> DeferredFit()._shape(Code("pl.col('a').alias('b')"), {"a": "Int64"}, answer, decl)
            'it names the column "b", but the declared name is "a"'
        """
        if decl.blank_count():
            if set(answer.reads) != {answer.name}:
                return "it must give back the one column it uses, under the same name"
        elif answer.name != decl.output_name():
            return self._renamed(answer.name, decl.output_name())
        character = self._character(code, columns, answer)
        if character is not None and character != decl.character:
            wording = CharacterWording()
            made, declared = wording.phrase(character), wording.phrase(decl.character)
            return f"it {made}, but the declared type {declared}"
        return None

    def _renamed(self, name: str | None, declared: str | None) -> str:
        """Give the reason that an argument gives another output name than its declaration.

        A code that the checker cannot read gives no name.

        Args:
            name: The output name of the argument, or `None` where the checker cannot read it.
            declared: The declared name.

        Returns:
            The reason.

        Example:
            >>> DeferredFit()._renamed("b", "a")
            'it names the column "b", but the declared name is "a"'
            >>> DeferredFit()._renamed(None, "a")
            'the checker cannot read the name of its column, but the declared name is "a"'
        """
        tail = f'but the declared name is "{declared}"'
        if name is None:
            return f"the checker cannot read the name of its column, {tail}"
        return f'it names the column "{name}", {tail}'

    def _character(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        answer: DeferredAnswer,
    ) -> str | None:
        """Give the way that an argument counts rows, from its answers in a `select` and in `agg`.

        Args:
            code: The code of the argument.
            columns: The frame of the point.
            answer: The answer of the argument in a `select`.

        Returns:
            `literal` for a fixed value, `aggregating` for one value from
            several rows, `elementwise` for a value for each row, or None
            where the answers do not show it.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> answer = DeferredEvaluation("row").answer(Code("pl.col('a').sum()"), {"a": "Int8"})
            >>> DeferredFit()._character(Code("pl.col('a').sum()"), {"a": "Int8"}, answer)
            'aggregating'
        """
        if DeferredEvaluation("row").is_literal(code):
            return self.LITERAL
        grouped = DeferredEvaluation("agg").answer(code, columns).rendering
        if grouped == f"List({answer.rendering})":
            return "elementwise"
        if grouped == answer.rendering and UNCHECKED not in grouped:
            return "aggregating"
        return None

    def _cell(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        answer: DeferredAnswer,
        declared: str,
    ) -> str | None:
        """Check that the argument gives the declared type at one point, in every pass.

        Args:
            code: The code of the argument.
            columns: The frame of the point.
            answer: The answer of the argument in a `select` there.
            declared: The type that the declaration promises there.

        Returns:
            The reason that the argument does not fit, or None.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> code = Code("pl.col('d').cum_sum()")
            >>> columns = {"d": "Decimal(precision=10, scale=2)"}
            >>> answer = DeferredEvaluation("row").answer(code, columns)
            >>> wide = "Decimal(precision=38, scale=2)"
            >>> reason = DeferredFit()._cell(code, columns, answer, wide)
            >>> reason.startswith("it is planned as Decimal(precision=10, scale=2) and runs as")
            True

            A declared `Unchecked` agrees with every answer, a refusal included:

            >>> from astrolabe._hooks.deferred.evaluation import DeferredAnswer
            >>> refused = DeferredAnswer("d", "Unchecked", refused=True)
            >>> DeferredFit()._cell(code, columns, refused, "Unchecked") is None
            True
        """
        answered = None if answer.refused else answer.rendering
        if Fit().disagree(answered, declared):
            return CellRefusal().wording(answered, declared)
        if answered is None:
            return None
        planned = DeferredEvaluation("planner").answer(code, columns).rendering
        return self._two_types(planned, answered) if Fit().disagree(planned, answered) else None

    def _two_types(self, planned: str, answered: str) -> str:
        """Say that polars plans another type than it runs, with the cast that makes them one.

        Args:
            planned: The planner type.
            answered: The run type.

        Returns:
            The reason.

        Example:
            >>> DeferredFit()._two_types("Int8", "Int16")
            'it is planned as Int8 and runs as Int16: cast it with .cast(pl.Int16)'
        """
        cast = ast.unparse(DataTypeCode().of(answered))
        return f"it is planned as {planned} and runs as {answered}: cast it with .cast({cast})"
