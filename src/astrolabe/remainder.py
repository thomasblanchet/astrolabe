"""The markers that describe the undeclared columns of a frame.

A frame type names some columns and can leave the others open. The columns
that it does not name are its remainder, and a frame with a remainder is an
open frame. `TypedDataFrame[Trades, Rest]` accepts any frame that holds the
columns of `Trades`, whatever other columns it holds. A function over such a
frame must say what it does to those other columns, and this module holds
the words for it.

`other_columns` declares a remainder and the promises a caller must keep
about it. The effect markers (`Minus`, `Only`, `Suffixed`, `Both` and the
others) state what an operation does to a remainder. The selector markers
(`StartsWith`, `OneOf`, `OfType` and the others) state which columns an
effect picks. The requirement markers (`Without`, `AllOf` and the others)
record the promises of a declared remainder.

No code creates an instance of these classes. Each class is a name that the
checker can read in a type annotation and write back in a result. mypy
shows the full name of a marker in every error and every `reveal_type`, so
the module is public.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


class Suffixed[NameExpr, Suffix]:
    """Add a literal suffix to the name of every column in a remainder.

    The remainder of a frame is the set of columns that its type does not
    name. `Suffixed[R, Literal["_right"]]` holds the columns of `R`, each
    renamed with `_right` at the end. A join writes this for the columns
    of its right frame, and `.name.suffix()` over a selection writes it too.

    Example:
        >>> from typing import Literal
        >>> Suffixed[Both[int, str], Literal["_right"]]
        astrolabe.remainder.Suffixed[astrolabe.remainder.Both[int, str], typing.Literal['_right']]
    """


class Prefixed[Prefix, NameExpr]:
    """Add a literal prefix to the name of every column in a remainder.

    The remainder of a frame is the set of columns that its type does not
    name. `Prefixed[Literal["src_"], R]` holds the columns of `R`, each
    renamed with `src_` at the start. The prefix comes first, as it does in
    the name.

    Example:
        >>> from typing import Literal
        >>> Prefixed[Literal["src_"], Both[int, str]]
        astrolabe.remainder.Prefixed[typing.Literal['src_'], astrolabe.remainder.Both[int, str]]
    """


class Both[L, R]:
    """Hold the columns of two remainders together.

    The remainder of a frame is the set of columns that its type does not
    name. `Both[A, B]` holds every column of `A` and every column of `B`.
    A horizontal concat or a join of two open frames gives this. The
    checker writes it only when `A` and `B` share no name.

    The order of the operands carries no meaning. The checker flattens a
    nested `Both`, sorts the operands and deletes a repeat, so two
    spellings of the same union are the same type.

    Example:
        >>> Both[int, str]
        astrolabe.remainder.Both[int, str]
    """


class Only[E, S]:
    """Keep only the columns of a remainder that a selector picks.

    The remainder of a frame is the set of columns that its type does not
    name. `Only[R, StartsWith[Literal["sec_"]]]` holds the columns of `R`
    whose names start with `sec_`. A `select` over a selector, for example
    `df.select(cs.starts_with("sec_"))`, gives this.

    Example:
        >>> Only[int, str]
        astrolabe.remainder.Only[int, str]
    """


class Except[E, S]:
    """Keep the columns of a remainder that a selector does not pick.

    The remainder of a frame is the set of columns that its type does not
    name. `Except[R, OfType[pl.String]]` holds the columns of `R` that are
    not `String`. It is the same set as `Only[R, SelNot[S]]`, written
    without the extra `SelNot`. A `with_columns` that changes the data type
    of the columns a selector picks gives this for the columns it leaves.

    Example:
        >>> Except[int, str]
        astrolabe.remainder.Except[int, str]
    """


class Minus[E, Names]:
    """Take named columns out of a remainder.

    The remainder of a frame is the set of columns that its type does not
    name. `Minus[R, Literal["z"]]` is `R` with no column named `z`, whether
    or not `R` held one. A function that writes the column `z` states this
    in its return type, because the new `z` replaces any `z` that the
    caller's remainder held.

    `Names` is one union of string literals, for example
    `Literal["a", "b"]`, and never a list of separate type arguments. A
    union has no order and no repeats, so each set of names has one
    spelling.

    Example:
        >>> from typing import Literal
        >>> Minus[int, Literal["a", "b"]].__args__[1]
        typing.Literal['a', 'b']
    """


class Retyped[E, D]:
    """Give every column of a remainder one new data type, and keep the names.

    The remainder of a frame is the set of columns that its type does not
    name. `Retyped[R, pl.UInt32]` holds the columns of `R`, each with the
    data type `UInt32`. A whole-frame reducer, for example `count()`, gives
    this. `Retyped[R, Unchecked]` keeps the names and says nothing about the
    data types, for example after `map_columns` runs a Python function over
    every column.

    Example:
        >>> Retyped[int, str]
        astrolabe.remainder.Retyped[int, str]
    """


class SplitColumns[*Parts]:
    """Divide the remainder of a frame into named parts.

    The remainder of a frame is the set of columns that its type does not
    name. Each part is a class that `other_columns(...)` declares. A column
    goes to the first part whose requirements accept it, so the order of
    the parts matters. `TypedDataFrame[Keys, SplitColumns[Numbers, Rest]]`
    with `Numbers = other_columns(all_of=NumericType)` puts every
    numeric column in `Numbers` and every other column in `Rest`.

    A function uses a split to treat each part in its own way. A part can
    only use a requirement that tests one column at a time: a requirement
    that relates two remainders cannot decide where a column goes.

    Example:
        >>> SplitColumns[int, str]
        astrolabe.remainder.SplitColumns[int, str]
    """


class StartsWith[S]:
    """Pick every column whose name starts with a literal string.

    Write it in a type annotation, for example
    `Only[R, StartsWith[Literal["sec_"]]]` for the columns of the remainder
    `R` whose names start with `sec_`.

    Example:
        >>> from typing import Literal
        >>> StartsWith[Literal["sec_"]].__args__[0]
        typing.Literal['sec_']
    """


class EndsWith[S]:
    """Pick every column whose name ends with a literal string.

    Write it in a type annotation, for example
    `Only[R, EndsWith[Literal["_prev"]]]` for the columns of the remainder
    `R` whose names end with `_prev`.

    Example:
        >>> from typing import Literal
        >>> EndsWith[Literal["_prev"]].__args__[0]
        typing.Literal['_prev']
    """


class Contains[S]:
    """Pick every column whose name contains a literal string.

    Write it in a type annotation, for example
    `Except[R, Contains[Literal["tmp"]]]` for the columns of the remainder
    `R` whose names do not contain `tmp`.

    Example:
        >>> from typing import Literal
        >>> Contains[Literal["tmp"]].__args__[0]
        typing.Literal['tmp']
    """


class OneOf[Names]:
    """Pick the columns whose names are in a list.

    `cs.by_name("a", "b")` gives the selector `OneOf[Literal["a", "b"]]`.
    `Names` is one union of string literals, as in `Minus`, so each set of
    names has one spelling.

    Example:
        >>> from typing import Literal
        >>> OneOf[Literal["a", "b"]].__args__[0]
        typing.Literal['a', 'b']
    """


class OfType[F]:
    """Pick every column whose data type belongs to a family.

    `F` is a polars data type class. It can be a family base, for example
    `NumericType`, or one data type, for example `pl.String`, which is
    a family of one. `cs.numeric()` gives the selector
    `OfType[NumericType]`.

    Example:
        >>> OfType[int]
        astrolabe.remainder.OfType[int]
    """


class SelAnd[A, B]:
    """Pick the columns that both selectors pick.

    `cs.numeric() & cs.string()` gives `SelAnd` over the two
    selectors.

    Example:
        >>> SelAnd[int, str]
        astrolabe.remainder.SelAnd[int, str]
    """


class SelOr[A, B]:
    """Pick the columns that one selector or the other picks.

    `cs.numeric() | cs.string()` gives `SelOr` over the two selectors.

    Example:
        >>> SelOr[int, str]
        astrolabe.remainder.SelOr[int, str]
    """


class SelNot[S]:
    """Pick the columns that a selector does not pick.

    `~cs.numeric()` gives `SelNot[OfType[NumericType]]`.

    Example:
        >>> SelNot[int]
        astrolabe.remainder.SelNot[int]
    """


class Matching[S]:
    """Pick the columns whose names match a rule that no other marker can spell.

    A selector that polars builds from a regular expression, for example
    `cs.matches("^px_")` or `cs.starts_with("px_")`, gives this. The
    checker also gives it where it derives a rule that no other marker
    can spell. The checker turns each rule into a finite-state machine that
    accepts the matching names, and makes that machine as small as it can.
    `S` is a string literal that holds the text of that smallest machine.
    Two rules that accept the same names have the same machine, so they
    are the same type, however each one was written.

    A user does not write this marker. Where possible, an error message
    shows the rule that the user wrote, in place of the machine.

    Example:
        >>> from typing import Literal
        >>> Matching[Literal["0F[.>0]"]].__args__[0]
        typing.Literal['0F[.>0]']
    """


class DeclaredRemainder:
    """Mark a class as a remainder that `other_columns(...)` declares.

    The checker accepts a class in the remainder position of a frame type
    only when the class has this base. So a stray type, for example `int`,
    in that position is an error, and the checker does not read it as a
    remainder. The class that `other_columns(...)` gives at run time has
    this base too.

    Example:
        >>> DeclaredRemainder
        <class 'astrolabe.remainder.DeclaredRemainder'>
    """


class Without[Names]:
    """Record that a remainder holds none of the listed names.

    `other_columns(without=["ticker"])` gives this requirement. The body
    of the function can then add a `ticker` column with no risk of a clash,
    and each caller must prove that its own undeclared columns have no
    `ticker`. `Names` is one union of string literals, as in `Minus`.

    Example:
        >>> from typing import Literal
        >>> Without[Literal["ticker"]].__args__[0]
        typing.Literal['ticker']
    """


class OnlyNames[Names]:
    """Record that a remainder holds no names other than the listed ones.

    `other_columns(only=["ticker"])` gives this requirement. It is the
    opposite of `Without`: every name outside the list is absent. A split
    can use it to send the listed names to one part.

    Example:
        >>> from typing import Literal
        >>> OnlyNames[Literal["ticker"]].__args__[0]
        typing.Literal['ticker']
    """


class WithoutPrefix[S]:
    """Record that a remainder holds no name that starts with a prefix.

    `other_columns(without_prefix="sec_")` gives this requirement.

    Example:
        >>> from typing import Literal
        >>> WithoutPrefix[Literal["sec_"]].__args__[0]
        typing.Literal['sec_']
    """


class OnlyPrefix[S]:
    """Record that every name in a remainder starts with a prefix.

    `other_columns(only_prefix="sec_")` gives this requirement.

    Example:
        >>> from typing import Literal
        >>> OnlyPrefix[Literal["sec_"]].__args__[0]
        typing.Literal['sec_']
    """


class WithoutSuffix[S]:
    """Record that a remainder holds no name that ends with a suffix.

    `other_columns(without_suffix="_prev")` gives this requirement.

    Example:
        >>> from typing import Literal
        >>> WithoutSuffix[Literal["_prev"]].__args__[0]
        typing.Literal['_prev']
    """


class OnlySuffix[S]:
    """Record that every name in a remainder ends with a suffix.

    `other_columns(only_suffix="_prev")` gives this requirement.

    Example:
        >>> from typing import Literal
        >>> OnlySuffix[Literal["_prev"]].__args__[0]
        typing.Literal['_prev']
    """


class AllOf[F]:
    """Record that every column of a remainder has a data type in a set.

    `other_columns(all_of=NumericType)` gives this requirement. `F` is
    a polars data type class, or a union of them that stands for all the
    data types that the members name together.

    Example:
        >>> AllOf[int]
        astrolabe.remainder.AllOf[int]
    """


class NoneOf[F]:
    """Record that no column of a remainder has a data type in a set.

    `other_columns(none_of=pl.String)` gives this requirement. `F` has the
    same form as in `AllOf`.

    Example:
        >>> NoneOf[int]
        astrolabe.remainder.NoneOf[int]
    """


class DisjointFrom[R]:
    """Record that a remainder shares no name with other remainders.

    `other_columns(disjoint_from=Left)` gives this requirement. A function
    that joins or concatenates two open frames needs it, because a name in
    both remainders makes the result depend on the caller. `R` is one
    declared remainder class, or a union of them. To share no name with
    the union is to share no name with each member.

    Example:
        >>> DisjointFrom[int]
        astrolabe.remainder.DisjointFrom[int]
    """


class Within[R]:
    """Record that every name of a remainder is also a name of other remainders.

    `other_columns(within=Left)` gives this requirement. `R` is one
    declared remainder class, or a union of them. The checker tests the
    requirement against the whole union, not member by member: a name can
    come from one member and the next name from another.

    Example:
        >>> Within[int]
        astrolabe.remainder.Within[int]
    """


# --- The declaration of a remainder ---------------------------------------
#
# `other_columns` is the one call in this module that a user writes. The
# checker reads its keywords and records each one as a requirement marker
# from above, on the class that the call declares.


def other_columns(  # ruff: ignore[too-many-arguments] -- each keyword is a separate requirement
    *,
    without: Sequence[str] = (),
    only: Sequence[str] = (),
    without_prefix: str | None = None,
    only_prefix: str | None = None,
    without_suffix: str | None = None,
    only_suffix: str | None = None,
    all_of: object = None,
    none_of: object = None,
    disjoint_from: object = None,
    within: object = None,
) -> Any:
    """Declare a remainder: the columns of a frame that its type does not name.

    Write `Rest = other_columns()` and then `TypedDataFrame[Trades, Rest]`.
    That type accepts any frame with the columns of `Trades`, whatever other
    columns it holds. At each call, the checker puts the caller's other
    columns in the place of `Rest`, so the return type of the function
    describes the caller's real frame.

    Each keyword states one promise that every caller must keep. The body
    of the function can rely on the promise, and the checker reports a
    call that cannot prove it. With no keyword, the remainder can hold any
    column.

    Args:
        without: Names that the remainder does not hold.
        only: The only names that the remainder can hold.
        without_prefix: A prefix that no name of the remainder starts with.
        only_prefix: A prefix that every name of the remainder starts with.
        without_suffix: A suffix that no name of the remainder ends with.
        only_suffix: A suffix that every name of the remainder ends with.
        all_of: The data types that every column of the remainder must
            have: a polars data type class, a data type with parameters,
            or a set of them.
        none_of: The data types that no column of the remainder can have,
            in the same forms as `all_of`.
        disjoint_from: Declared remainders that share no name with this
            one: one remainder, or a set of them.
        within: Declared remainders that hold every name of this one: one
            remainder, or a set of them.

    Returns:
        A new class that has `DeclaredRemainder` as its base. At run time
        the class holds nothing. The checker reads the call itself and
        knows the promises.

    Example:
        >>> Rest = other_columns(without=["ticker"])
        >>> from astrolabe.remainder import DeclaredRemainder
        >>> issubclass(Rest, DeclaredRemainder)
        True
    """
    del without, only, without_prefix, only_prefix
    del without_suffix, only_suffix, all_of, none_of, disjoint_from, within
    return type("OtherColumns", (DeclaredRemainder,), {})
