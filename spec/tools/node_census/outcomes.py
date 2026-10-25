"""The outcomes that a test gives to each input of a node of the census.

An input of a node is a column of some kind in some row form, or a literal
of some class. Polars treats a literal that went through an operation, a
derived literal, apart from a direct one, so a literal outcome also says
which it is. A test puts each outcome at each input position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

import polars as pl

from astrolabe._rules.options import RowForm

if TYPE_CHECKING:
    from spec.tools.node_census.kinds import CensusKinds


class Outcome(Protocol):
    """Give the expression of one input outcome, and its name."""

    def expression(self, *, null: bool) -> pl.Expr:
        """Give the expression that stands for the outcome.

        Args:
            null: Whether the values must be all null.
        """

    def label(self) -> str:
        """Give a short name of the outcome."""

    def can_be_null(self) -> bool:
        """Say whether the census also runs the outcome with all values null."""

    def second(self) -> Outcome:
        """Give the outcome at a second value of its class, or the outcome itself."""


@dataclass(frozen=True)
class ColumnOutcome:
    """Stand for a column of one census kind in one row form.

    The column `k{i}` of a sample frame holds the values, and `n{i}` holds
    nulls of the same kind.

    Example:
        >>> ColumnOutcome(3, RowForm.ONE_VALUE).label()
        'k3/one value'
    """

    kind: int
    form: RowForm

    def expression(self, *, null: bool) -> pl.Expr:
        """Give the column, in its row form.

        Args:
            null: Whether to read the twin whose values are all null.

        Returns:
            The expression.

        Example:
            >>> str(ColumnOutcome(3, RowForm.OTHER_LENGTH).expression(null=True))
            'col("n3").slice(offset=dyn int: 0, length=dyn int: 2)'
        """
        prefix = "n" if null else "k"
        column = pl.col(f"{prefix}{self.kind}")
        if self.form is RowForm.ONE_VALUE:
            return column.first()
        if self.form is RowForm.OTHER_LENGTH:
            return column.head(2)
        if self.form is RowForm.ONE_ROW:
            return column.head(1)
        return column

    def label(self) -> str:
        """Give the column name and the row form.

        Returns:
            The name.

        Example:
            >>> ColumnOutcome(0, RowForm.PER_ROW).label()
            'k0/per row'
        """
        return f"k{self.kind}/{self.form.value}"

    def can_be_null(self) -> bool:
        """Say that a column can also be all null.

        Returns:
            Always true.

        Example:
            >>> ColumnOutcome(0, RowForm.PER_ROW).can_be_null()
            True
        """
        return True

    def second(self) -> ColumnOutcome:
        """Give the column itself, which has no value of its own.

        Returns:
            The outcome.

        Example:
            >>> ColumnOutcome(0, RowForm.ONE_ROW).second().label()
            'k0/one row'
        """
        return self


@dataclass(frozen=True)
class LiteralClass:
    """Hold one class of literal values: a representative, its neutral value and a second value.

    A derived literal of the class combines the value with its neutral value,
    so the planner sees a literal after an operation, and the value stays the
    same. The second value is another value of the class. A node whose length
    is the value of a literal count, such as `top_k`, shows with it that its
    length is the value and not the class.

    Example:
        >>> str(LiteralClass("int 300", 300, 0, "+", 301).derived())
        '[(dyn int: 300) + (dyn int: 0)]'
    """

    name: str
    value: object
    neutral: object
    operator: str
    other: object = None

    def at_second(self) -> LiteralClass:
        """Give the class with its second value as its representative.

        Returns:
            The class.

        Example:
            >>> LiteralClass("int 1", 1, 0, "+", 2).at_second().value
            2
        """
        return LiteralClass(self.name, self.other, self.neutral, self.operator, self.other)

    def direct(self) -> pl.Expr:
        """Give the literal itself.

        Returns:
            The literal.

        Example:
            >>> str(LiteralClass("str", "a", "", "+").direct())
            '"a"'
        """
        return pl.lit(self.value)

    def derived(self) -> pl.Expr:
        """Give the literal combined with its neutral value.

        Returns:
            The derived literal.

        Example:
            >>> str(LiteralClass("bool", True, False, "|").derived())
            '[(true) | (false)]'
        """
        left = pl.lit(self.value)
        right = pl.lit(self.neutral)
        if self.operator == "|":
            return left | right
        return left + right


@dataclass(frozen=True)
class LiteralOutcome:
    """Stand for a literal of one class, direct or derived.

    Example:
        >>> LiteralOutcome(LiteralClass("int 1", 1, 0, "+"), derived=True).label()
        'lit int 1/derived'
    """

    literal: LiteralClass
    derived: bool

    def expression(self, *, null: bool) -> pl.Expr:
        """Give the literal. A literal has no all-null twin, so `null` changes nothing.

        Args:
            null: Ignored.

        Returns:
            The expression.

        Example:
            >>> one = LiteralOutcome(LiteralClass("int 1", 1, 0, "+"), derived=False)
            >>> str(one.expression(null=False))
            'dyn int: 1'
        """
        del null
        if self.derived:
            return self.literal.derived()
        return self.literal.direct()

    def label(self) -> str:
        """Give the class and whether the literal is derived.

        Returns:
            The name.

        Example:
            >>> LiteralOutcome(LiteralClass("float", 1.5, 0.0, "+"), derived=False).label()
            'lit float/direct'
        """
        way = "derived" if self.derived else "direct"
        return f"lit {self.literal.name}/{way}"

    def can_be_null(self) -> bool:
        """Say that a literal has no all-null twin.

        Returns:
            Always false.

        Example:
            >>> null = LiteralOutcome(LiteralClass("null", None, None, "+"), derived=False)
            >>> null.can_be_null()
            False
        """
        return False

    def second(self) -> LiteralOutcome:
        """Give the literal at the second value of its class.

        Returns:
            The outcome.

        Example:
            >>> one = LiteralOutcome(LiteralClass("int 1", 1, 0, "+", 2), derived=True)
            >>> str(one.second().expression(null=False))
            '[(dyn int: 2) + (dyn int: 0)]'
        """
        return LiteralOutcome(self.literal.at_second(), self.derived)


class OutcomeSpace:
    """List the 124 outcomes of one input.

    They are 27 kinds in 4 row forms, and 8 literal classes, each direct or
    derived.

    Example:
        >>> from spec.tools.node_census.kinds import CensusKinds
        >>> space = OutcomeSpace(CensusKinds())
        >>> len(space.all()), len(space.literals())
        (124, 16)
    """

    CLASSES: ClassVar[tuple[LiteralClass, ...]] = (
        LiteralClass("int 1", 1, 0, "+", 2),
        LiteralClass("int -1", -1, 0, "+", -2),
        LiteralClass("int 300", 300, 0, "+", 301),
        LiteralClass("int 2**40", 2**40, 0, "+", 2**40 + 1),
        LiteralClass("float", 1.5, 0.0, "+", 2.5),
        LiteralClass("str", "a", "", "+", "ab"),
        LiteralClass("bool", True, False, "|", False),
        LiteralClass("null", None, None, "+", None),
    )

    def __init__(self, kinds: CensusKinds) -> None:
        """Keep the kinds.

        Args:
            kinds: The kinds of the census.

        Example:
            >>> from spec.tools.node_census.kinds import CensusKinds
            >>> len(OutcomeSpace(CensusKinds()).columns())
            108
        """
        self._kinds = kinds

    def columns(self) -> list[Outcome]:
        """Give the column outcomes.

        Returns:
            One outcome for each kind and row form.

        Example:
            >>> from spec.tools.node_census.kinds import CensusKinds
            >>> OutcomeSpace(CensusKinds()).columns()[1].label()
            'k0/one value'
        """
        count = len(self._kinds.dtypes())
        outcomes: list[Outcome] = []
        for kind in range(count):
            outcomes.extend(ColumnOutcome(kind, form) for form in RowForm)
        return outcomes

    def literals(self) -> list[Outcome]:
        """Give the literal outcomes.

        Returns:
            One outcome for each class, direct and derived.

        Example:
            >>> from spec.tools.node_census.kinds import CensusKinds
            >>> OutcomeSpace(CensusKinds()).literals()[1].label()
            'lit int 1/derived'
        """
        outcomes: list[Outcome] = []
        for literal in self.CLASSES:
            outcomes.extend(LiteralOutcome(literal, derived) for derived in (False, True))
        return outcomes

    def all(self) -> list[Outcome]:
        """Give every outcome: the columns, then the literals.

        Returns:
            The 124 outcomes.

        Example:
            >>> from spec.tools.node_census.kinds import CensusKinds
            >>> OutcomeSpace(CensusKinds()).all()[-1].label()
            'lit null/derived'
        """
        return self.columns() + self.literals()
