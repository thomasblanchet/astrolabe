"""Give the data type of a `.cast(...)` on a typed series, or None where polars refuses it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ast

from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.deferred.code import Code
from astrolabe._hooks.deferred.evaluation import DeferredEvaluation
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeCodec, ConstDtypeReads
from astrolabe._hooks.recording.argument_code import DataTypeCode

if TYPE_CHECKING:
    from mypy.types import Type as MypyType

    from astrolabe._hooks.encoding.marker_minter import MarkerMinter


class CastCompatibility:
    """Give the data type of a `.cast(...)`, or None if polars always rejects the cast.

    A cast can fail in two ways. Some pairs of data types always fail, whatever the data. The
    checker reports those before the code runs. Other pairs fail only for some values, for
    example `String` to `Int64` for the text `"abc"`. The checker cannot see the data, so it
    accepts those pairs. Code that types a `.cast(...)` call on a typed series asks this class
    which case applies.

    The answer of a cast is not always its target. A `Struct` cast to `Float64` casts each
    field and stays a `Struct`. A `List` cast to a `List` of `Null` keeps its elements. This
    class reads both data types as the text that polars prints for them, and evaluates the
    cast of a column of the source type with the rules of polars. It is the same evaluation
    that the checker runs for `.cast()` on an expression, so a series and an expression agree.

    A source whose data type is not known, for example `Unchecked` or a type variable, gives
    the target. A cast that the rules do not know, as a cast of an `Object` column, gives
    `Unchecked`, as it does on an expression.

    Example:
        >>> fixture = new_type_fixture()
        >>> minter = new_marker_minter(fixture)
        >>> compat = CastCompatibility()
        >>> compat.resolve(int8_dtype(fixture), int64_dtype(fixture), minter)
        polars.datatypes.classes.Int64
        >>> compat.resolve(str_dtype(fixture), boolean_dtype(fixture), minter) is None
        True
        >>> compat.resolve(unchecked_instance(fixture), boolean_dtype(fixture), minter)
        polars.datatypes.classes.Boolean
    """

    def __init__(self) -> None:
        """Set up the reader of data types."""
        self._reads = ConstDtypeReads()

    def resolve(self, source: MypyType, target: MypyType, minter: MarkerMinter) -> MypyType | None:
        """Give the data type that a cast produces, or None if the cast always fails.

        Args:
            source: The data type of the series before the cast.
            target: The data type that the cast asks for.
            minter: The builder of marker types, which builds an answer that is not the
                target, such as a `Struct` with each field cast.

        Returns:
            The answer of the cast, or `target` where this class cannot read a data type.
            None if the cast always fails.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> field = column_ref(fixture, "a", int64_dtype(fixture))
            >>> struct = struct_instance(fixture, field)
            >>> recast = CastCompatibility().resolve(struct, int8_dtype(fixture), minter)
            >>> ConstDtypeReads().read_type(recast)
            "Struct({'a': Int8})"
            >>> items = list_dtype(fixture, str_dtype(fixture))
            >>> flags = list_dtype(fixture, boolean_dtype(fixture))
            >>> CastCompatibility().resolve(items, flags, minter) is None
            True
            >>> CastCompatibility().resolve(fixture.a, flags, minter)
            astrolabe.frame.List[polars.datatypes.classes.Boolean]
        """
        source_text = self._reads.read_type(source)
        target_text = self._reads.read_type(target)
        if source_text is None or target_text is None:
            return target
        answer = self.answer(source_text, target_text)
        if answer is None:
            return None
        if answer == target_text:
            return target
        # The answer holds only parts of the two types that this class read, so it has a type.
        return checked_not_none(ConstDtypeCodec(minter).built(answer))

    def answer(self, source: str, target: str) -> str | None:
        """Give the data type of a cast of a column of one data type to another.

        Args:
            source: The rendering of the data type of the column.
            target: The rendering of the target.

        Returns:
            The rendering of the answer, `Unchecked` where the rules do not
            know it, or None where polars refuses the cast whatever the data.

        Example:
            >>> CastCompatibility().answer("Int8", "Int64")
            'Int64'
            >>> CastCompatibility().answer("String", "Boolean") is None
            True
        """
        cast = ast.unparse(DataTypeCode().of(target))
        code: Code[None] = Code(f"pl.col('x').cast({cast})")
        answer = DeferredEvaluation("row").answer(code, {"x": source})
        return None if answer.refused else answer.rendering
