"""Describe the placeholder return types, and the classes that each one accepts.

A function can declare its return type as `InferredExpr`,
`InferredDataFrame` or `InferredLazyFrame`. Each placeholder stands for
one kind of value. Its function can return a typed value of that kind,
or the plain polars class. A frame type whose remainder is
`InferredRemainder`, for example `TypedDataFrame[WithTotal,
InferredRemainder]`, belongs to the family of its frame class. The code
that records `infer(...)` calls and the code that types a call of the
function both need that list, so it lives here once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TYPE_CHECKING

import polars as pl
from mypy.types import get_proper_type

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe.declare import (
    InferredDataFrame,
    InferredExpr,
    InferredLazyFrame,
    InferredRemainder,
)
from astrolabe.expr import TypedAgg, TypedColumn, TypedExpr, TypedLit, TypedThen
from astrolabe.frame import TypedDataFrame, TypedLazyFrame

if TYPE_CHECKING:
    from mypy.types import Type as MypyType


@dataclass(frozen=True)
class PlaceholderFamily:
    """Hold one placeholder return type, with the typed and the plain classes that it accepts.

    A variant is the type of one value that the function returns through
    `infer(...)`. Each variant must be exactly one of the typed classes
    of the placeholder, or exactly its plain polars class. Any other
    variant is an error, a subclass included. A plain variant makes the whole return type the plain
    class, because each typed class is a subclass of it.

    Attributes:
        placeholder_target: The placeholder class: `InferredExpr`,
            `InferredDataFrame` or `InferredLazyFrame`.
        typed_targets: The typed classes that a variant can be. For
            `InferredExpr` these are the typed expression classes, for
            example `TypedExpr` and `TypedColumn`. For a frame it is
            `TypedDataFrame` or `TypedLazyFrame`.
        plain_target: The plain polars class that a variant can be:
            `pl.Expr`, `pl.DataFrame` or `pl.LazyFrame`.
        operation_key: The name of this placeholder in the setting that
            decides whether a plain return type is an error: `expr`,
            `dataframe` or `lazyframe`.
        kind: The kind of value of the family, in words, for an error
            message: `an expression`, `a DataFrame` or `a LazyFrame`.
    """

    placeholder_target: Target
    typed_targets: tuple[Target, ...]
    plain_target: Target
    operation_key: str
    kind: str

    @property
    def display_name(self) -> str:
        """The short name of the placeholder class, as an error message shows it.

        Example:
            >>> FAMILIES[0].display_name
            'InferredExpr'
        """
        return self.placeholder_target.fullname.rsplit(".", 1)[-1]

    def matches_typed(self, proper: MypyType) -> bool:
        """Answer whether a variant is an instance of one of the typed classes.

        Args:
            proper: The type of the variant. An alias in it is expanded
                first.

        Returns:
            True if the type is an instance of a typed class of this
            placeholder, and not of a subclass.

        Example:
            >>> fixture = new_type_fixture()
            >>> FAMILIES[0].matches_typed(int_dtype(fixture))
            False
        """
        resolved = get_proper_type(proper)
        return any(target.matches(resolved) for target in self.typed_targets)

    def names_return(self, proper: MypyType, declared: Frame | None) -> bool:
        """Answer whether a declared return type belongs to this placeholder.

        Args:
            proper: The declared return type.
            declared: The declared frame if the return type is a frame
                whose remainder is `InferredRemainder`, else None.

        Returns:
            True for the placeholder class itself, or for a frame with an
            inferred remainder whose class is a typed class of this
            placeholder.

        Example:
            >>> fixture = new_type_fixture()
            >>> FAMILIES[0].names_return(int_dtype(fixture), None)
            False
        """
        if self.placeholder_target.matches(get_proper_type(proper)):
            return True
        return declared is not None and self.matches_typed(declared.as_instance())

    def matches_plain(self, proper: MypyType) -> bool:
        """Answer whether a variant is an instance of the plain polars class.

        Args:
            proper: The type of the variant. An alias in it is expanded
                first.

        Returns:
            True if the type is an instance of the plain class of this
            placeholder, for example `pl.Expr` for `InferredExpr`.
        """
        return self.plain_target.matches(get_proper_type(proper))


FAMILIES: Final[tuple[PlaceholderFamily, ...]] = (
    PlaceholderFamily(
        Target.of(InferredExpr),
        (
            Target.of(TypedExpr),
            Target.of(TypedThen),
            Target.of(TypedAgg),
            Target.of(TypedLit),
            Target.of(TypedColumn),
        ),
        Target.of(pl.Expr),
        "expr",
        "an expression",
    ),
    PlaceholderFamily(
        Target.of(InferredDataFrame),
        (Target.of(TypedDataFrame),),
        Target.of(pl.DataFrame),
        "dataframe",
        "a DataFrame",
    ),
    PlaceholderFamily(
        Target.of(InferredLazyFrame),
        (Target.of(TypedLazyFrame),),
        Target.of(pl.LazyFrame),
        "lazyframe",
        "a LazyFrame",
    ),
)


class InferredRemainderReturn:
    """Read a frame return type whose remainder the checker infers from the body.

    In `TypedDataFrame[WithTotal, InferredRemainder]`, the declared
    columns are a promise that each returned value must keep, and the
    remainder, that is, the columns that the type does not name, comes
    from each value. Code that checks a returned value against the
    declared columns, and code that gives the type of a call, read the
    declared frame through this class.

    Example:
        >>> fixture = new_type_fixture()
        >>> InferredRemainderReturn().declared(int_dtype(fixture)) is None
        True
    """

    _MARKER: Final[Target] = Target.of(InferredRemainder)

    @propagates(None)
    def declared(self, ret_type: MypyType) -> Frame | None:
        """Give the declared frame of a return type, if its remainder is `InferredRemainder`.

        Args:
            ret_type: The declared return type of a function.

        Returns:
            The frame, or None for any other type.
        """
        frame = not_none(Frame.from_proper_type(get_proper_type(ret_type)))
        rest = not_none(not_none(frame.known_columns).rest)
        not_falsy(self._MARKER.matches(get_proper_type(rest)))
        return frame


def family_for(ret_type: MypyType) -> PlaceholderFamily | None:
    """Give the placeholder that a declared return type names, if it names one.

    Code that meets a function or an `infer(...)` call asks this first,
    to know whether the function returns a placeholder at all.

    Args:
        ret_type: The declared return type of a function.

    Returns:
        The placeholder, or None if the type is not one of the three
        placeholder classes, or a frame type whose remainder is
        `InferredRemainder`.

    Example:
        >>> fixture = new_type_fixture()
        >>> plugin = new_plugin(fixture)
        >>> info = plugin.lookup_fully_qualified("astrolabe.declare.InferredExpr").node
        >>> family_for(Instance(info, [])) is not None
        True
        >>> family_for(int_dtype(fixture)) is None
        True
    """
    proper = get_proper_type(ret_type)
    declared = InferredRemainderReturn().declared(proper)
    for family in FAMILIES:
        if family.names_return(proper, declared):
            return family
    return None
