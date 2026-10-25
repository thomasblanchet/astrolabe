"""Find an expression that polars refuses at every frame, so that its call reports the refusal.

Some expressions have no result whatever the frame that meets them, as
`.name.keep()` over an expression that reads no column. The checker reports
such an expression at the call that builds it, where the user writes it. The
evaluation gives each column that the expression reads every data type of its
sample, in a `select` and in `agg`. Where polars refuses at each one, it
refuses at every frame.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from astrolabe._hooks.deferred.code_reader import CodeReader, CodeReadError, HoleValues
from astrolabe._hooks.deferred.evaluation import DeferredEvaluation, FieldNodes
from astrolabe._hooks.deferred.remainder import DeferredRemainder
from itertools import starmap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._hooks.deferred.code import Code
    from astrolabe._hooks.deferred.evaluation import DeclaredLeaf, DeferredAnswer


class BodyReads:
    """Find a read of `pl.element()` or `pl.field()` outside the evaluation that gives it a meaning.

    `pl.element()` stands for the element of a list in `list.eval`, and
    `pl.field()` for a field of a struct in `struct.with_fields`. Polars
    refuses each of them outside such an evaluation, at every frame. But an
    expression that reads one can still become the body of an evaluation,
    so a frame is not the only place where it can run.

    The class reads the tree that polars writes for an expression. A column
    name or a literal that spells `Element` is not a read.

    Example:
        >>> BodyReads().found({"Agg": {"Mean": "Element"}})
        True
        >>> body = {"BinaryExpr": {"left": "Element", "op": "Plus", "right": {"Column": "a"}}}
        >>> BodyReads().found({"Eval": {"expr": {"Column": "l"}, "evaluation": body}})
        False
    """

    EVALUATIONS: ClassVar[frozenset[str]] = frozenset({"Eval", "StructEval"})
    VALUES: ClassVar[frozenset[str]] = frozenset({"Column", "Literal"})

    def found(self, node: object) -> bool:
        """Say whether a node of the tree reads an element or a field outside an evaluation.

        Args:
            node: The node.

        Returns:
            Whether it does.

        Example:
            >>> BodyReads().found([{"Field": ["a"]}]), BodyReads().found({"Column": "Element"})
            (True, False)
        """
        if node == "Element":
            return True
        if isinstance(node, list):
            return any(self.found(item) for item in node)
        if not isinstance(node, dict):
            return False
        return any(starmap(self._entry, node.items()))

    def _entry(self, key: str, value: object) -> bool:
        """Say whether one entry of a node reads an element or a field outside an evaluation.

        An evaluation reads its input outside its body, so only its input counts.

        Args:
            key: The key of the entry.
            value: Its value.

        Returns:
            Whether it does.

        Example:
            >>> BodyReads()._entry("Field", ["a"]), BodyReads()._entry("Literal", "Element")
            (True, False)
        """
        if key == "Field":
            return True
        if key in self.VALUES:
            return False
        if key in self.EVALUATIONS and isinstance(value, dict):
            return self.found(value.get("expr"))
        return self.found(value)


class EveryFrameRefusal:
    """Say whether polars refuses the code of an expression at every frame, and how.

    A call that builds such an expression can never run in a frame, so the
    checker reports it at the call, before any frame meets it. The check
    must not report an expression that some frame accepts. A lazy `select`
    that holds the expression alone can run it where the other contexts
    refuse it, because polars moves its slices into the plan. So the check
    tries that context too.

    Each column that the code reads takes every data type of its sample. A
    selector, as `cs.numeric()`, stands for one column of any data type, as
    in a frame where it picks one column. The sample does not hold every
    data type: a list of structs of lists, for example, is not in it. So a
    refusal at each data type of the sample counts only where polars gives
    the same error and the same message at each one. A refusal that names a
    data type differs from one data type to the next, and the frame reports
    it instead.

    The check does not report in these cases, where the frame reports:

    - A refusal that depends on the frame, as `.alias()` over a selector that
      picks several columns.
    - A code of several columns whose data types give too many choices.
    - A code with a hole of a value, which stands for values that the
      evaluation does not try.
    - A read of several fields of a struct, which gives several columns.
    - A read of `pl.element()` or `pl.field()` outside an evaluation, which
      can become the body of one.

    A few common data types come first, because most expressions have a
    result at one of them, and that answer ends the check.

    Example:
        >>> from astrolabe._hooks.deferred.code import Code
        >>> EveryFrameRefusal().refusal(Code("pl.lit(1).name.keep()"))
        '`name.keep_name` expected at least one column name, and polars raises ...'
        >>> EveryFrameRefusal().refusal(Code("pl.col('a').abs()")) is None
        True
    """

    PROBES: ClassVar[tuple[str, ...]] = ("Int64", "String", "Date", "List(Int64)")
    COLUMN: ClassVar[str] = DeferredRemainder.PROBES[0]
    CONTEXTS: ClassVar[tuple[str, ...]] = ("row", "agg", "lazy row alone")
    OPEN: ClassVar[str] = "Unchecked"

    def refusal(self, code: Code[DeclaredLeaf | None]) -> str | None:
        """Give how polars refuses a code at every frame, where it does.

        Args:
            code: The code, with a declared column or `None` for each leaf.

        Returns:
            The way that polars stops, the class of its error and its message,
            or `None` where a frame can accept the code, or where the
            evaluation does not decide.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> EveryFrameRefusal().refusal(Code("cs.numeric().alias('z')")) is None
            True
            >>> EveryFrameRefusal().refusal(Code("pl.col('i').cumulative_eval(pl.element() * 2)"))
            '`cumulative_eval` is not allowed with non-scalar output, and polars raises ...'
            >>> EveryFrameRefusal().refusal(Code("pl.col('x').head(2).over(pl.lit(0))")) is None
            True
        """
        if self._unsampled(code):
            return None
        evaluation = DeferredEvaluation("row")
        selector = evaluation.selector(code)
        names = list(evaluation.reads(code))
        if selector is not None:
            names.append(self.COLUMN)
        if names and self._answered(code, names, selector):
            return None
        reasons: list[str] = []
        for context in self.CONTEXTS:
            open_columns = dict.fromkeys(names, self.OPEN)
            answer = self._answer(code, open_columns, selector, context)
            if len(answer.refusals) != 1:
                return None
            reasons.append(answer.reason)
        return reasons[0]

    def _unsampled(self, code: Code[DeclaredLeaf | None]) -> bool:
        """Say whether the evaluation of a code cannot stand for every frame and every value.

        Args:
            code: The code.

        Returns:
            Whether the code holds a hole of a value, reads several fields of
            a struct, or reads `pl.element()` or `pl.field()` outside an
            evaluation. `False` where the reader cannot build the code.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> EveryFrameRefusal()._unsampled(Code("pl.element().mean()"))
            True
            >>> EveryFrameRefusal()._unsampled(Code("pl.col('s').struct.field('a', 'b')"))
            True
            >>> EveryFrameRefusal()._unsampled(Code("pl.lit(1).shift(hole('int'))"))
            True
            >>> EveryFrameRefusal()._unsampled(Code("pl.col("))
            False
        """
        if HoleValues().in_text(code.text):
            return True
        try:
            read = CodeReader().read(code)
        except CodeReadError:
            return False
        if not isinstance(read.expression, pl.Expr):
            return False
        if FieldNodes().of(read.expression) is not None:
            return True
        tree = json.loads(read.expression.meta.serialize(format="json"))
        return BodyReads().found(tree)

    def _answered(
        self,
        code: Code[DeclaredLeaf | None],
        names: Sequence[str],
        selector: object | None,
    ) -> bool:
        """Say whether the code has a result in a `select` where all its columns share a data type.

        Args:
            code: The code.
            names: The columns that the code reads.
            selector: The serialized selector of the code, or `None`.

        Returns:
            Whether it has one at one of the data types.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> EveryFrameRefusal()._answered(Code("pl.col('a').dt.year()"), ["a"], None)
            True
        """
        for probe in self.PROBES:
            answer = self._answer(code, dict.fromkeys(names, probe), selector, "row")
            if not answer.refused:
                return True
        return False

    def _answer(
        self,
        code: Code[DeclaredLeaf | None],
        columns: dict[str, str],
        selector: object | None,
        context: str,
    ) -> DeferredAnswer:
        """Give the answer of a code at some columns, with one column in the place of its selector.

        Args:
            code: The code.
            columns: The rendering of the data type of each column, by its name.
            selector: The serialized selector of the code, or `None`.
            context: `row` for a `select`, or `agg` for the aggregation context.

        Returns:
            The answer.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> columns = {EveryFrameRefusal.COLUMN: "Int8"}
            >>> numeric = {"Selector": {"ByDType": "Numeric"}}
            >>> EveryFrameRefusal()._answer(Code("cs.numeric()"), columns, numeric, "row").rendering
            'Int8'
        """
        evaluation = DeferredEvaluation(context)
        if selector is None:
            return evaluation.answer(code, columns)
        return evaluation.at_column(code, columns, selector, self.COLUMN)
