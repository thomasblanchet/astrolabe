"""Report a row count of a frame method that polars cannot read, such as `df.head(2**64)`.

A frame method such as `head`, `slice` or `top_k` takes a count of rows.
Polars converts the count to an integer of a fixed width before it reads
any data, so a count outside that width raises at the call, whatever the
frame holds: `df.head(2**64)` raises `OverflowError`, and `lf.head(-1)`
raises `ValueError`. The width depends on the method and on the kind of
frame. A lazy frame reads most counts as the index type of polars. The
checker reports such a count where the caller writes it as a number.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from mypy.types import Instance, LiteralType, get_proper_type

from astrolabe._hooks.calls.constant_fold import ConstantFold
from astrolabe._hooks.errors import InvalidCallShapeError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mypy.nodes import Expression
    from mypy.plugin import MethodContext
    from mypy.types import Type as MypyType


@dataclass(frozen=True)
class CountRange:
    """The whole numbers that polars reads for one count, and the error past each end.

    Attributes:
        low: The smallest count, or `None` where every negative count runs.
        high: The largest count.
        below: The class of the error of polars below `low`.
        above: The class of the error of polars above `high`.
        zero: The class of the error of polars where the count is 0, or
            `None` where 0 runs. `gather_every(0)` is one such count.

    Example:
        >>> CountRange(0, 2**32 - 1).refusal(2**32)
        'OverflowError: polars reads it as a whole number from 0 to 4294967295'
        >>> CountRange(None, 2**64 - 1).refusal(-(2**70)) is None
        True
    """

    low: int | None
    high: int
    below: str = "OverflowError"
    above: str = "OverflowError"
    zero: str | None = None

    def refusal(self, value: int) -> str | None:
        """Give the error of polars for a count, or `None` where polars reads it.

        Args:
            value: The count.

        Returns:
            The class of the error and the range that polars reads.

        Example:
            >>> CountRange(0, 9, below="ValueError").refusal(-1)
            'ValueError: polars reads it as a whole number from 0 to 9'
            >>> CountRange(0, 9, zero="InvalidOperationError").refusal(0)
            'InvalidOperationError: polars reads it as a whole number from 1 to 9'
        """
        if value == 0 and self.zero is not None:
            return f"{self.zero}: {self._range()}"
        if self.low is not None and value < self.low:
            return f"{self.below}: {self._range()}"
        if value > self.high:
            return f"{self.above}: {self._range()}"
        return None

    def _range(self) -> str:
        """Give the words for the range that polars reads.

        Returns:
            The words.

        Example:
            >>> CountRange(None, 9)._range()
            'polars reads it as a whole number up to 9'
        """
        if self.low is None:
            return f"polars reads it as a whole number up to {self.high}"
        low = self.low + 1 if self.low == 0 and self.zero is not None else self.low
        return f"polars reads it as a whole number from {low} to {self.high}"


class RowCounts:
    """Give the range of each count of each frame method, by the kind of the receiver.

    The kinds are `DataFrame`, `LazyFrame`, `GroupBy` and `LazyGroupBy`.
    An eager frame reads `head`, `tail` and `limit` as an unsigned 64-bit
    integer and takes a negative count as a count from the end. A lazy
    frame reads them as the index type, and refuses a negative count.

    Example:
        >>> RowCounts().ranges("LazyFrame", "head")["n"].high
        4294967295
    """

    U64: ClassVar[int] = 2**64 - 1
    I64: ClassVar[int] = 2**63 - 1
    INDEX: ClassVar[int] = 2**32 - 1
    EAGER_ROWS: ClassVar[CountRange] = CountRange(None, U64)
    LAZY_ROWS: ClassVar[CountRange] = CountRange(0, INDEX, below="ValueError")
    OFFSET: ClassVar[CountRange] = CountRange(-I64 - 1, I64)
    UNSIGNED: ClassVar[CountRange] = CountRange(0, U64)
    STEP: ClassVar[CountRange] = CountRange(0, U64, zero="InvalidOperationError")
    INDEXED: ClassVar[CountRange] = CountRange(0, INDEX)
    ROW_INDEX: ClassVar[CountRange] = CountRange(0, INDEX, below="ValueError", above="ValueError")
    SHARED: ClassVar[dict[str, dict[str, CountRange]]] = {
        "gather_every": {"n": STEP, "offset": UNSIGNED},
        "top_k": {"k": INDEXED},
        "bottom_k": {"k": INDEXED},
        "with_row_index": {"offset": ROW_INDEX},
    }
    KINDS: ClassVar[dict[str, dict[str, dict[str, CountRange]]]] = {
        "DataFrame": {
            **SHARED,
            "head": {"n": EAGER_ROWS},
            "tail": {"n": EAGER_ROWS},
            "limit": {"n": EAGER_ROWS},
            "slice": {"offset": OFFSET, "length": CountRange(None, U64)},
        },
        "LazyFrame": {
            **SHARED,
            "head": {"n": LAZY_ROWS},
            "tail": {"n": INDEXED},
            "limit": {"n": LAZY_ROWS},
            "slice": {"offset": OFFSET, "length": LAZY_ROWS},
        },
        "GroupBy": {"head": {"n": UNSIGNED}, "tail": {"n": UNSIGNED}},
        "LazyGroupBy": {"head": {"n": UNSIGNED}, "tail": {"n": UNSIGNED}},
    }

    def ranges(self, kind: str, method: str) -> dict[str, CountRange]:
        """Give the range of each count of one method, by the name of its parameter.

        Args:
            kind: The kind of the receiver.
            method: The name of the method.

        Returns:
            The ranges, none for a method that takes no count.

        Example:
            >>> RowCounts().ranges("DataFrame", "select")
            {}
        """
        return self.KINDS.get(kind, {}).get(method, {})


class RowCountCheck:
    """Report the first count of one call that polars cannot read.

    The plugin declares some of these methods with `*args` and `**kwargs`,
    so the check maps each argument to the parameter of polars that it
    fills, by the signature of the polars method. It reads a count that
    the caller writes as a number, also as an expression of numbers such
    as `2**64`, or as a variable of a literal type.

    Example:
        >>> RowCountCheck("DataFrame", "head")._parameters()
        ['n']
    """

    CLASSES: ClassVar[dict[str, type]] = {
        "DataFrame": pl.DataFrame,
        "LazyFrame": pl.LazyFrame,
        "GroupBy": pl.dataframe.group_by.GroupBy,
        "LazyGroupBy": pl.lazyframe.group_by.LazyGroupBy,
    }

    RECEIVERS: ClassVar[dict[str, str]] = {
        "TypedDataFrame": "DataFrame",
        "TypedLazyFrame": "LazyFrame",
        "TypedGroupBy": "GroupBy",
        "TypedLazyGroupBy": "LazyGroupBy",
    }

    def __init__(self, kind: str, method: str) -> None:
        """Keep the kind of the receiver and the method.

        Args:
            kind: The kind of the receiver.
            method: The name of the method.
        """
        self._kind = kind
        self._method = method

    @classmethod
    def of_receiver(cls, typ: MypyType, method: str) -> RowCountCheck:
        """Build the check of a method from the type of its receiver.

        Args:
            typ: The type of the receiver, a typed frame or a typed group-by.
            method: The name of the method.

        Returns:
            The check.

        Example:
            >>> RowCountCheck.of_receiver(new_type_fixture().anyt, "head")._kind
            'DataFrame'
        """
        proper = get_proper_type(typ)
        name = proper.type.name if isinstance(proper, Instance) else ""
        return cls(cls.RECEIVERS.get(name, "DataFrame"), method)

    def reported(self, ctx: MethodContext) -> bool:
        """Report the first count of the call that polars cannot read.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            Whether the method reported an error.
        """
        ranges = RowCounts().ranges(self._kind, self._method)
        for name, node, typ in self._arguments(ctx):
            value = self._value(node, typ) if name in ranges else None
            refusal = None if value is None else ranges[name].refusal(value)
            if refusal is not None:
                message = f"{self._method}() with {name}={value} raises {refusal}"
                InvalidCallShapeError(message).fail(ctx)
                return True
        return False

    def _arguments(self, ctx: MethodContext) -> Iterator[tuple[str, Expression, MypyType]]:
        """Give the parameter, the node and the type of each argument of the call.

        Args:
            ctx: The context that mypy gives for the call.

        Yields:
            The name of the parameter of polars, the node and its type. A
            positional argument past the parameters of polars has the empty name.
        """
        parameters = iter(self._parameters())
        for nodes, names, types in zip(ctx.args, ctx.arg_names, ctx.arg_types, strict=True):
            for node, keyword, typ in zip(nodes, names, types, strict=True):
                yield keyword or next(parameters, ""), node, typ

    def _parameters(self) -> list[str]:
        """Give the parameters of the polars method that take an argument by position.

        Returns:
            The names, in order.

        Example:
            >>> RowCountCheck("LazyFrame", "slice")._parameters()
            ['offset', 'length']
        """
        signature = inspect.signature(getattr(self.CLASSES[self._kind], self._method))
        positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        return [
            parameter.name
            for parameter in list(signature.parameters.values())[1:]
            if parameter.kind in positional
        ]

    def _value(self, node: Expression, typ: MypyType) -> int | None:
        """Give the whole number that an argument holds, where the call states it.

        Args:
            node: The node of the argument.
            typ: The type of the argument.

        Returns:
            The number, or `None`.

        Example:
            >>> from mypy.nodes import IntExpr, OpExpr
            >>> check = RowCountCheck("DataFrame", "head")
            >>> check._value(OpExpr("**", IntExpr(2), IntExpr(64)), new_type_fixture().anyt)
            18446744073709551616
        """
        folded = ConstantFold().value(node)
        if isinstance(folded, int) and not isinstance(folded, bool):
            return folded
        proper = get_proper_type(typ)
        if isinstance(proper, LiteralType) and type(proper.value) is int:
            return proper.value
        last = getattr(proper, "last_known_value", None)
        if isinstance(last, LiteralType) and type(last.value) is int:
            return last.value
        return None


class RowCountHook:
    """Check the counts of a frame method that keeps the type of the frame, as `head`.

    It is a mypy method hook in the plugin layer. It reports a count that
    polars cannot read, and gives the type that the plugin declares.

    Example:
        >>> RowCountHook("DataFrame", "head")._check._method
        'head'
    """

    def __init__(self, kind: str, method: str) -> None:
        """Keep the check of the method.

        Args:
            kind: The kind of the receiver.
            method: The name of the method.
        """
        self._check = RowCountCheck(kind, method)

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Check one call, and give its declared type.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The type that the plugin declares for the method.
        """
        self._check.reported(ctx)
        return ctx.default_return_type
