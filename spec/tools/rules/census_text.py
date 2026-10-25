"""Give the leaf of the rules for a census outcome, and an answer of the rules as census text.

The census vocabulary of `spec.tools.node_census` names each input of a node
by its outcome: a column of a census kind in a row form, or a literal of a
class, direct or derived. The tests of the definitions and the check of random
trees build the leaf of the rules for each outcome, and compare the answers of
the rules with those of polars as text: the rendering of a data type,
`raises`, `hangs` or `unchecked`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from spec.tools.node_census.kinds import CensusKinds
from spec.tools.node_census.outcomes import ColumnOutcome, LiteralOutcome
from astrolabe._rules.dtypes import DType
from astrolabe._rules.failures import Stop
from astrolabe._rules.node_evaluation import ColumnLeaf, LiteralLeaf, Refused
from astrolabe._rules.operators import Operator

if TYPE_CHECKING:
    from spec.tools.node_census.outcomes import Outcome
    from astrolabe._rules.node_evaluation import Answer, CellAnswers, Leaf


class CensusLeaves:
    """Give the leaf of the rules for one outcome of an input of the census, and its name.

    Example:
        >>> from spec.tools.node_census.outcomes import OutcomeSpace
        >>> from astrolabe._rules.options import RowForm
        >>> CensusLeaves().leaf(ColumnOutcome(11, RowForm.ONE_ROW)).dtype
        Decimal(precision=10, scale=2)
        >>> CensusLeaves().leaf(LiteralOutcome(OutcomeSpace.CLASSES[6], derived=True))
        LiteralLeaf(value=True, operator=<Operator.OR: 'Or'>, neutral=False)
    """

    OPERATORS: ClassVar[dict[str, Operator]] = {"+": Operator.PLUS, "|": Operator.OR}

    def leaf(self, outcome: Outcome) -> Leaf:
        """Give the leaf of an outcome.

        Args:
            outcome: A column of a census kind in a row form, or a literal of a class.

        Returns:
            The leaf.

        Raises:
            TypeError: The outcome is neither a column nor a literal.

        Example:
            >>> from spec.tools.node_census.outcomes import OutcomeSpace
            >>> CensusLeaves().leaf(LiteralOutcome(OutcomeSpace.CLASSES[2], derived=False))
            LiteralLeaf(value=300, operator=None, neutral=None)
        """
        if isinstance(outcome, ColumnOutcome):
            dtype = DType.of(CensusKinds().dtypes()[outcome.kind])
            return ColumnLeaf(dtype, outcome.form)
        if not isinstance(outcome, LiteralOutcome):
            message = f"no leaf for the outcome {outcome.label()}"
            raise TypeError(message)
        literal = outcome.literal
        value = self._value(literal.value)
        if not outcome.derived:
            return LiteralLeaf(value)
        return LiteralLeaf(value, self.OPERATORS[literal.operator], self._value(literal.neutral))

    def label(self, outcome: Outcome) -> str:
        """Give a short name of an outcome, with the name of its data type.

        Args:
            outcome: The outcome.

        Returns:
            The name.

        Example:
            >>> from spec.tools.node_census.outcomes import OutcomeSpace
            >>> from astrolabe._rules.options import RowForm
            >>> CensusLeaves().label(ColumnOutcome(0, RowForm.PER_ROW))
            'Int8/per row'
            >>> CensusLeaves().label(LiteralOutcome(OutcomeSpace.CLASSES[2], derived=False))
            'lit int 300/direct'
        """
        if isinstance(outcome, ColumnOutcome):
            dtype = DType.of(CensusKinds().dtypes()[outcome.kind])
            return f"{dtype!r}/{outcome.form.value}"
        return outcome.label()

    def _value(self, value: object) -> bool | int | float | str | None:
        """Read the value of a literal class with its Python type.

        Args:
            value: The value that the census class holds.

        Returns:
            The value.

        Raises:
            TypeError: The value is not of a type of a literal.

        Example:
            >>> CensusLeaves()._value(1.5)
            1.5
            >>> CensusLeaves()._value(b"x")
            Traceback (most recent call last):
            ...
            TypeError: no literal of the value b'x'
        """
        if value is None or isinstance(value, bool | int | float | str):
            return value
        message = f"no literal of the value {value!r}"
        raise TypeError(message)


class CensusText:
    """Give an answer of the rules in the text of the census.

    The text is the rendering of a data type, `raises` for a run that raises
    or panics, `hangs` for a run that ends the process or never returns, and
    `unchecked` for a question that the rules cannot answer.

    Example:
        >>> CensusText().of(Refused(Stop.PANICS))
        'raises'
        >>> CensusText().of(DType("List", (DType("Int8"),)))
        'List(Int8)'
    """

    LEFT_OUT: ClassVar[frozenset[Stop]] = frozenset({Stop.CRASHES, Stop.HANGS})

    def of(self, answer: Answer) -> str:
        """Give the text.

        Args:
            answer: The answer of the rules.

        Returns:
            The text.

        Example:
            >>> CensusText().of(Refused(Stop.CRASHES))
            'hangs'
        """
        if isinstance(answer, DType):
            return answer.render()
        if isinstance(answer, Refused):
            return "hangs" if answer.stop in self.LEFT_OUT else "raises"
        return "unchecked"

    def answers(self, answers: CellAnswers) -> list[str]:
        """Give the four answers of the rules in the order of the census.

        Args:
            answers: The answers.

        Returns:
            The planner types, then the run types, each in a `select` and then in `agg`.

        Example:
            >>> from astrolabe._rules.node_evaluation import CellAnswers
            >>> int8 = DType("Int8")
            >>> CensusText().answers(CellAnswers(int8, int8, int8, Refused(Stop.HANGS)))
            ['Int8', 'Int8', 'Int8', 'hangs']
        """
        ordered = (answers.planner_select, answers.planner_agg, answers.run_select, answers.run_agg)
        return [self.of(answer) for answer in ordered]
