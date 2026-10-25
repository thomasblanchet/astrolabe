"""The types that the checker gives to polars expressions.

The type of an expression holds the code of the expression, as the user
wrote it. `pl.col("price") * 2` has this type:

```
TypedExpr[Deferred[Literal["pl.col('price') * 2"]]]
```

The checker computes nothing when it writes the code. Where the
expression meets a frame, the checker evaluates the code with the rules
of polars, at the data types of the frame. So it knows the data type of
each new column, as polars gives it.

The head class, here `TypedExpr`, stands for the class that polars gives
the value: an expression, a `when` before its `then`, or a selector. A
parameter that a declaration types, such as
`price: column("price", pl.Float64)`, is a leaf of the code, written
`$0`, and the type holds its declaration after the code.

A declaration has a type of its own, built from the other markers of
this module, for example `Reads`, `Row`, `Same` and `Wrap`. It states the
columns that an expression reads, the data type that it gives at each
data type of those columns, its name, and how it treats rows.

mypy shows these types in every error and every `reveal_type`, so the
module is public. No code creates an instance of these classes. Each head
class subclasses the polars class of the value that it stands for, so a
typed expression goes anywhere that polars accepts an expression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Concatenate, Self

import polars as pl
import polars.selectors as cs
from polars.expr.array import ExprArrayNameSpace
from polars.expr.binary import ExprBinaryNameSpace
from polars.expr.categorical import ExprCatNameSpace
from polars.expr.datetime import ExprDateTimeNameSpace
from polars.expr.list import ExprListNameSpace
from polars.expr.name import ExprNameNameSpace
from polars.expr.string import ExprStringNameSpace
from polars.expr.struct import ExprStructNameSpace
from polars.expr.whenthen import ChainedWhen, Then, When

if TYPE_CHECKING:
    from collections.abc import Callable


class TypedExpr[*Parts](pl.Expr):
    """The type of an expression that polars gives as an `Expr`.

    `pl.col("price") * 2` has the type
    `TypedExpr[Deferred[Literal["pl.col('price') * 2"]]]`: its one argument
    holds the code of the expression. The checker evaluates that code when
    the expression meets a frame or a declaration.

    A declaration gives this type another form. Its first argument is a
    `Reads[...]` of the columns that the expression reads, each with the data
    types that it accepts there. Its second argument is a
    `Column[name, outcome]`: the name of the result, and its data type at
    each data type of the reads. `typed_expr(...)` gives that form a name to
    use in a signature.

    Example:
        >>> TypedExpr[str, int]
        astrolabe.expr.TypedExpr[str, int]
    """

    def pipe[**P, T](
        self,
        function: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Let `.pipe()` pass a typed expression to a function that expects it.

        The signature of `pl.Expr.pipe` types the first parameter of
        `function` as a plain `pl.Expr`. A parameter type of a `Callable`
        is contravariant, so mypy then rejects a function whose first
        parameter is a typed expression. This signature takes the type of
        the receiver in its place. The method exists only for mypy: at run
        time, polars runs its own `pipe`.

        Args:
            function: The function to call. Its first parameter accepts the
                receiver.
            *args: More positional arguments for `function`, after the
                receiver.
            **kwargs: The keyword arguments for `function`.

        Returns:
            What `function` returns.
        """
        raise NotImplementedError


class TypedThen[*Parts](TypedExpr[*Parts]):
    """The type of a `when`/`then` chain that has no `.otherwise(...)` yet.

    `pl.when(pl.col("flag")).then(pl.col("a"))` has this type, with the code
    of the chain in one argument `Deferred[...]`. The chain is an expression
    already: where no branch holds, polars gives null. It can also take one
    more `.when(...)`, or end with `.otherwise(...)`.

    In polars, `Then` subclasses `Expr` and adds `.when(...)` and
    `.otherwise(...)`. This class does the same over `TypedExpr`, so that
    only a chain offers those two methods. Its methods exist only for mypy.
    The checker records each call, and at run time polars runs its own
    methods.

    Example:
        >>> TypedThen[str, int]
        astrolabe.expr.TypedThen[str, int]
    """

    def when(self, *predicates: Any, **constraints: Any) -> ChainedWhen:
        """Add one more condition to the chain.

        Args:
            predicates: The conditions. All of them must hold.
            constraints: Conditions written as `column=value`, each true
                where the column equals the value.

        Returns:
            The chain with its new condition, which waits for a
            `.then(...)`. If the checker cannot read the condition, the
            plain polars type.
        """
        raise NotImplementedError

    def then(self, statement: Any) -> Then:
        """Give the value for the last condition of the chain.

        Args:
            statement: The value where the condition holds.

        Returns:
            The chain with one more branch. If the checker cannot read the
            value, the plain polars type.
        """
        raise NotImplementedError

    def otherwise(self, statement: Any) -> pl.Expr:
        """End the chain with the value for the rows where no condition holds.

        Args:
            statement: The value where no condition holds.

        Returns:
            The finished expression. If the checker cannot read the value,
            the plain polars type.
        """
        raise NotImplementedError


class TypedWhen[*Parts](When):
    """The type of `pl.when(condition)` before its first `.then(...)`.

    `pl.when(pl.col("flag"))` has the type `TypedWhen[Deferred[...]]`, with
    the code of the condition. The next `.then(...)` adds its branch to that
    code. The condition is an input of the chain, so a frame must hold every
    column that it reads, and a condition that polars refuses fails the
    chain.

    Example:
        >>> TypedWhen[str, int]
        astrolabe.expr.TypedWhen[str, int]
    """


class TypedChainedWhen[*Parts](ChainedWhen):
    """The type of a chain after one more `.when(condition)`.

    `pl.when(a).then(x).when(b)` has the type `TypedChainedWhen[Deferred[...]]`,
    with the code of the chain so far. The next `.then(...)` adds its branch
    to that code.

    Example:
        >>> TypedChainedWhen[str, int, bytes]
        astrolabe.expr.TypedChainedWhen[str, int, bytes]
    """


class TypedAgg[Reads, Gives](pl.Expr):
    """The declared type of an expression that gives one value for each group.

    `typed_agg(...)` gives this type a name to use in a signature. In
    `group_by(...).agg(...)` such an expression gives one value for each
    group, and in `select` one value for the whole frame. The type arguments
    have the form of a declared `TypedExpr`.

    Example:
        >>> TypedAgg[str, int]
        astrolabe.expr.TypedAgg[str, int]
    """


class TypedLit[Reads, Gives](pl.Expr):
    """The declared type of an expression that gives one fixed value and reads no column.

    `typed_lit(...)` gives this type a name to use in a signature, with an
    empty `Reads[()]` and the output name `literal` unless the declaration
    names another. In `with_columns`, polars repeats the value for each row.

    Example:
        >>> TypedLit[str, int]
        astrolabe.expr.TypedLit[str, int]
    """


class TypedSelector[Selector](cs.Selector):
    """The type of a selector, or of an expression that polars gives as a `Selector`.

    `cs.numeric()` has the type `TypedSelector[Deferred[...]]`, with the code
    of the selector. A declaration such as `typed_selection(cs.numeric(),
    gives=same)` gives the form `TypedSelector[OfType[NumericType]]`, which
    states with the selector markers which columns it picks, each given back
    as it is. The class subclasses `cs.Selector`, so the operators `~`, `&`,
    `|` and `-` still combine two selectors.

    Example:
        >>> TypedSelector[str]
        astrolabe.expr.TypedSelector[str]
    """


class TypedSelection[Selector, Gives, Rename](pl.Expr):
    """The declared type of a selection that gives one value for each row.

    `typed_selection(...)` gives this type a name to use in a signature. It
    gives one result for each column that the selector picks. `Selector`
    states which columns it picks. `Gives` gives the data type of each
    result at the data type of its picked column. `Rename` is an `Affix` for
    the names of the results, or `None` where each result keeps the name of
    its column.

    Example:
        >>> TypedSelection[str, int, None]
        astrolabe.expr.TypedSelection[str, int, NoneType]
    """


class TypedAggSelection[Selector, Gives, Rename](pl.Expr):
    """The declared type of a selection that gives one value for each group.

    `typed_agg_selection(...)` gives this type a name to use in a signature.
    It gives one result for each column that the selector picks, and each
    result holds one value for each group. The type arguments have the form
    of `TypedSelection`.

    Example:
        >>> TypedAggSelection[str, int, None]
        astrolabe.expr.TypedAggSelection[str, int, NoneType]
    """


class TypedColumn[Name, Region](pl.Expr):
    """The declared type of a column that an expression reads and gives back unchanged.

    `column("price", pl.Float64)` gives the type
    `TypedColumn[Literal["price"], Float64]`. `Name` is the name of the
    column. `Region` is the set of data types that the declaration accepts
    there: `EveryDType`, one data type, a family such as `NumericType`, or a
    union. The result has the name and the data type of the column, with one
    value for each row. This is a short form of the declared `TypedExpr`
    that says the same thing.

    Example:
        >>> TypedColumn[str, int]
        astrolabe.expr.TypedColumn[str, int]
    """


class TypedExprStrNamespace[Expr](ExprStringNameSpace):
    """The type of `.str` on a typed expression.

    `pl.col("s").str` has the type
    `TypedExprStrNamespace[TypedExpr[Deferred[Literal["pl.col('s')"]]]]`.
    `Expr` is the type of the expression before `.str`. A member such as
    `.str.len_chars()` reads it back, so that the recorder can add the call
    to the code of that expression. The polars namespace class alone does
    not keep the type of its expression.

    Example:
        >>> TypedExprStrNamespace[str]
        astrolabe.expr.TypedExprStrNamespace[str]
    """


class TypedExprDtNamespace[Expr](ExprDateTimeNameSpace):
    """The type of `.dt` on a typed expression.

    `Expr` is the type of the expression before `.dt`. A method of the
    namespace reads it back, so that the recorder can add the call to the
    code of that expression.

    Example:
        >>> TypedExprDtNamespace[str]
        astrolabe.expr.TypedExprDtNamespace[str]
    """


class TypedExprBinNamespace[Expr](ExprBinaryNameSpace):
    """The type of `.bin` on a typed expression.

    `Expr` is the type of the expression before `.bin`. A method of the
    namespace reads it back, so that the recorder can add the call to the
    code of that expression.

    Example:
        >>> TypedExprBinNamespace[str]
        astrolabe.expr.TypedExprBinNamespace[str]
    """


class TypedExprListNamespace[Expr](ExprListNameSpace):
    """The type of `.list` on a typed expression.

    `Expr` is the type of the expression before `.list`. A method of the
    namespace reads it back, so that the recorder can add the call to the
    code of that expression.

    Example:
        >>> TypedExprListNamespace[str]
        astrolabe.expr.TypedExprListNamespace[str]
    """


class TypedExprArrNamespace[Expr](ExprArrayNameSpace):
    """The type of `.arr` on a typed expression.

    `Expr` is the type of the expression before `.arr`. A method of the
    namespace reads it back, so that the recorder can add the call to the
    code of that expression.

    Example:
        >>> TypedExprArrNamespace[str]
        astrolabe.expr.TypedExprArrNamespace[str]
    """


class TypedExprCatNamespace[Expr](ExprCatNameSpace):
    """The type of `.cat` on a typed expression.

    `Expr` is the type of the expression before `.cat`. A method of the
    namespace reads it back, so that the recorder can add the call to the
    code of that expression.

    Example:
        >>> TypedExprCatNamespace[str]
        astrolabe.expr.TypedExprCatNamespace[str]
    """


class TypedExprStructNamespace[Expr](ExprStructNameSpace):
    """The type of `.struct` on a typed expression.

    `Expr` is the type of the expression before `.struct`. A method of the
    namespace reads it back, so that the recorder can add the call to the
    code of that expression.

    Example:
        >>> TypedExprStructNamespace[str]
        astrolabe.expr.TypedExprStructNamespace[str]
    """


class TypedExprNameNamespace[Expr](ExprNameNameSpace):
    """The type of `.name` on a typed expression.

    `Expr` is the type of the expression before `.name`. A method of the
    namespace reads it back, so that the recorder can add the call to the
    code of that expression.

    Example:
        >>> TypedExprNameNamespace[str]
        astrolabe.expr.TypedExprNameNamespace[str]
    """


class Deferred[*Parts]:
    """The code of an expression, and the declared types that the code reads.

    `pl.col("price") * 0.9` has the type
    `TypedExpr[Deferred[Literal["pl.col('price') * 0.9"]]]`. The first
    argument is the code, written with the public API of polars, as one
    text. The code refers to the declared type at position `n` of the
    other arguments as `$n`, for example the type of a parameter that a
    function declares. A part that the code uses more than once is
    written once as `($k := (<code>))`, and as `$k` after that. The
    checker computes the data types of the expression from its code when
    the expression meets a frame or a declaration.

    Example:
        >>> Deferred[str]
        astrolabe.expr.Deferred[str]
    """


class Reads[*Cols]:
    """The columns that a declared expression reads.

    Each argument is a `Column[Literal[name], region]`, sorted by name. The
    region is the set of data types that the declaration accepts in that
    column. A frame operation reports a column that the frame does not
    have, or a data type outside the region.

    Example:
        >>> Reads[int, str]
        astrolabe.expr.Reads[int, str]
    """


class Rows[*Areas]:
    """The outcome of a declaration, given as a list of rows.

    Each argument is one `Row[...]`. A row covers a set of input data types
    and gives one result for all of them. No two rows cover the same input.
    If a frame gives an input that no row covers, the checker reports it at
    the frame operation. The rows come in one fixed order, so that one
    declaration always has one spelling.

    Example:
        >>> Rows[int, str]
        astrolabe.expr.Rows[int, str]
    """


class Row[*Cells]:
    """One row of a declaration: the input data types, then the result.

    The first arguments give one region for each column that the
    declaration reads, in the order of `Reads[...]`. A region is a set of
    data types, for example `Int8 | Int16` or `NumericType`. The last
    argument is the result for every input in those regions: a data type,
    or a marker such as `Same` that computes it from the input.

    Example:
        >>> Row[int, str]
        astrolabe.expr.Row[int, str]
    """


class Same[Col]:
    """The data type of a column that the declaration reads, unchanged.

    `column("price", NumericType)` gives `Same[Literal["price"]]` as its
    result: the data type of `price`. It keeps every parameter, for example
    the time unit and the time zone of a `Datetime`. With the name `Blank`,
    it stands for the column that a selection picks.

    Example:
        >>> from typing import Literal
        >>> Same[Literal["price"]]
        astrolabe.expr.Same[typing.Literal['price']]
    """


class Wrap[C, O]:
    """A nested data type around another result.

    `wrap(pl.List, same)` in a declaration gives `Wrap[List, Same[...]]`: a
    `List` of the data type of the column that it reads. `C` is the polars
    class of the outer data type, and `O` is the result inside it.

    Example:
        >>> Wrap[int, str]
        astrolabe.expr.Wrap[int, str]
    """


class Affix[Prefix, Suffix, Fixed]:
    """The rename that a declared selection applies to the name of every picked column.

    `typed_selection(cs.numeric(), gives=same, prefix="p_")` gives
    `Affix[Literal["p_"], Literal[""], None]`. `Prefix` and `Suffix` are the
    text added at each end. `Fixed` is `None` for a declaration.

    Example:
        >>> from typing import Literal
        >>> Affix[Literal["p_"], Literal[""], None]
        astrolabe.expr.Affix[typing.Literal['p_'], typing.Literal[''], NoneType]
    """


class Blank:
    """The empty name: a column whose name is not known yet.

    A declaration with `some_column(...)` reads one column of any name, and
    its type shows that name as `Blank`, for example
    `TypedColumn[Blank, FloatType]`. At each call, the checker puts the
    name of the argument's column in its place, so `Blank` never reaches a
    frame. In a declared selection, `Blank` names the column that the
    selector picks.

    Example:
        >>> Blank
        <class 'astrolabe.expr.Blank'>
    """


class EveryDType:
    """Every data type: the region of a declared read that accepts any data type.

    In a declaration, `some_column(EveryDType)` reads one column of any data
    type, and the body is checked at each of them.

    Example:
        >>> EveryDType
        <class 'astrolabe.expr.EveryDType'>
    """
