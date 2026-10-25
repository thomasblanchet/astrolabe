"""Give the value that a node computes over a literal, where the rules know it.

A rule of a later node can read the value of a literal input: the count of
`diff`, or the text that a strict cast parses. The code fixes that value
where the input is a node over a literal too, and the run meets it there.
Polars folds a binary node of two literals into a literal itself
(`SimplifyRule.folded`). Other nodes run over the one value of the literal.
Each class here gives the value that one kind of node computes
(`ValueRule`). A node with no such rule gives a column of no known value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from astrolabe._rules.cast_pass import CastFold
from astrolabe._rules.dtypes import Is
from astrolabe._rules.type_functions import Operand

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.dtypes import DType


class ValueRule(Protocol):
    """Give the literal that a node computes over literal inputs, or `None` where none is known."""

    def of(self, inputs: Sequence[Operand], dtype: DType) -> Operand | None:
        """Give the literal.

        Args:
            inputs: Each input as the run computes it.
            dtype: The type of the output.
        """


class LiteralValue:
    """Give a literal of a value at the type of the output, where the value converts plainly.

    A node can give its value at another type than its input, as a cast of
    an integer to a float does. The rules know the converted value only where
    the conversion is plain: an integer to an integer or a float, a float to
    a float, a text to a text, and a flag to a flag.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> LiteralValue().of(1, DType("Float64"))
        Operand(dtype=Float64, literal=True, value=1.0)
        >>> LiteralValue().of(1.5, DType("Int8")) is None
        True
    """

    def of(self, value: object, dtype: DType) -> Operand | None:
        """Give the literal.

        Args:
            value: The value before the conversion.
            dtype: The type of the output.

        Returns:
            The literal, or `None` where the conversion is not plain.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> LiteralValue().of(7, DType.dynamic_int(7)).value
            7
            >>> LiteralValue().of("a", DType("String")).value
            'a'
        """
        converted = self._converted(value, dtype)
        if converted is None:
            return None
        if Is.unknown(dtype):
            return Operand.of_value(converted)
        return Operand.typed_literal(converted, dtype)

    def _converted(self, value: object, dtype: DType) -> bool | int | float | str | None:
        """Give the value at the type of the output, or `None` where the conversion is not plain.

        Args:
            value: The value.
            dtype: The type of the output.

        Returns:
            The value.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> LiteralValue()._converted(True, DType("Boolean"))
            True
            >>> LiteralValue()._converted(True, DType("Int8")) is None
            True
        """
        if isinstance(value, bool):
            return value if dtype.name == "Boolean" else None
        if isinstance(value, int) and (Is.integer(dtype) or Is.dynamic_int(dtype)):
            return value
        if isinstance(value, int | float) and Is.float(dtype):
            return float(value)
        if isinstance(value, str) and dtype.name == "String":
            return value
        return None


@dataclass(frozen=True)
class KeptValue:
    """Give the value of the one literal at the first input, which the node keeps.

    Over one value, `first`, `max`, `unique`, `sort` or `str.join` give that
    value. A null literal gives what the node gives over a null, which the
    rules of nulls state, so this rule claims no value there.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> KeptValue().of([Operand.of_value("a")], DType("String"))
        Operand(dtype=String, literal=True, value='a')
        >>> KeptValue().of([Operand.of_value(None)], DType("Null")) is None
        True
    """

    def of(self, inputs: Sequence[Operand], dtype: DType) -> Operand | None:
        """Give the literal of the value of the first input.

        Args:
            inputs: Each input as the run computes it.
            dtype: The type of the output.

        Returns:
            The literal, or `None` where the first input is not a literal of a value.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> KeptValue().of([Operand.column(DType("Int8"))], DType("Int8")) is None
            True
        """
        first = inputs[0] if inputs else None
        if first is None or not first.literal or first.value is None:
            return None
        return LiteralValue().of(first.value, dtype)


@dataclass(frozen=True)
class CastValue:
    """Give the value of a literal after a cast that keeps it, as polars folds such a cast.

    Polars folds the cast of a literal into a literal of the target. A cast
    that does not keep the value raises, or gives a null, which other rules
    state. So this rule claims a value only where the cast keeps it.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> CastValue().of([Operand.of_value(300)], DType("Int16"))
        Operand(dtype=Int16, literal=True, value=300)
        >>> CastValue().of([Operand.of_value(300)], DType("Int8")) is None
        True
    """

    def of(self, inputs: Sequence[Operand], dtype: DType) -> Operand | None:
        """Give the literal of the cast value.

        Args:
            inputs: Each input as the run computes it.
            dtype: The type of the output, which is the target of the cast.

        Returns:
            The literal, or `None` where the cast does not keep a value.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> CastValue().of([Operand.of_value("a")], DType("Int8")) is None
            True
        """
        first = inputs[0] if inputs else None
        if first is None or not first.literal or not CastFold().keeps(first, dtype):
            return None
        return LiteralValue().of(first.value, dtype)
