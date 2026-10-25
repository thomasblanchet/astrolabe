"""The calls that name an expression type, for use in a function signature.

The checker gives each polars expression a type, but that type is long and
hard to write by hand. A declaration is a call, assigned to a name, that
describes the type in a short form. The name can then stand in any type
annotation:

```python
Quantity = column("quantity", pl.Int64)
BuyFlag = typed_expr(Quantity, gives=pl.Boolean, named="is_buy")

def buy_flag(quantity: Quantity) -> BuyFlag:
    return (quantity > 0).alias("is_buy")
```

A declaration answers four questions. Which columns does the expression
read? Which data types does it accept in each of them? What data type does
it give, and under which name? Does it give one value for each row, one
value for each group, or one fixed value?

At run time, each call returns a plain record that holds its arguments.
The checker reads the call itself, when it analyzes the module, and gives
the name the type that the call describes. The checker reports a
declaration that breaks a rule, at the declaration. At each call of a
function with a declared parameter, it checks that the argument fits the
declaration.

The module also holds the inferred return types. A function whose return
type is `InferredExpr`, `InferredDataFrame` or `InferredLazyFrame` gets the
union of the types of the values that its `return` statements pass to
`infer`. A frame return type with the remainder `InferredRemainder` keeps
its declared columns, and gets the remainder of each value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BlankRead:
    """The runtime record of `some_column(region)`: one column of any name.

    `region` holds the data types that the read accepts.

    Example:
        >>> from polars.datatypes.classes import FloatType
        >>> BlankRead(FloatType).region
        FloatType
    """

    region: object


@dataclass(frozen=True)
class SchemaRead:
    """The runtime record of `column_of(...)` or `subset_of(...)`.

    It reads the named columns of a `Columns` class, each at the data type
    that the class declares for it.

    Example:
        >>> SchemaRead(object, ("a", "b")).names
        ('a', 'b')
    """

    schema: object
    names: tuple[str, ...]


@dataclass(frozen=True)
class SameOutcome:
    """The result `same`: the data type of a column that the expression reads.

    Write `gives=same` for an expression that gives back the data type of
    its one input column, with every parameter, for example the time unit
    of a `Datetime`. If the declaration reads several columns, write
    `same("name")` to say which one.

    Example:
        >>> same("quantity").read
        'quantity'
        >>> same.read is None
        True
    """

    read: str | None = None

    def __call__(self, read: str) -> SameOutcome:
        """Name the column whose data type the result takes.

        Args:
            read: The name of a column that the declaration reads.

        Returns:
            The result that takes the data type of that column.

        Example:
            >>> same("a")
            SameOutcome(read='a')
        """
        return SameOutcome(read)


same = SameOutcome()


@dataclass(frozen=True)
class WrapOutcome:
    """The runtime record of `wrap(...)`: a nested data type around a result.

    Example:
        >>> import polars as pl
        >>> wrap(pl.List, same).inner
        SameOutcome(read=None)
    """

    constructor: object
    inner: object


@dataclass(frozen=True)
class RowDeclaration:
    """The runtime record of `row(...)`: some input data types and their result.

    Example:
        >>> import polars as pl
        >>> row(column_of(object, "q"), gives=pl.Int64).gives
        Int64
    """

    reads: tuple[object, ...]
    gives: object


@dataclass(frozen=True)
class SliceDeclaration:
    """The runtime record of `typed_expr`, `typed_agg` or `typed_lit`.

    `character` says how the expression treats rows: `"elementwise"` for
    one value for each row, `"aggregating"` for one value for each group,
    or `"literal"` for one fixed value. `parts` holds the reads or the
    rows, `gives` the result, and `named` the name of the result.

    Example:
        >>> import polars as pl
        >>> typed_lit(gives=pl.Boolean).character
        'literal'
    """

    character: str
    parts: tuple[object, ...]
    gives: object
    named: str | None


def some_column(region: object) -> Any:
    """Read one column whose name each call chooses.

    `typed_expr(some_column(FloatType), gives=same)` is the type of any
    expression that reads one float column and gives its data type back,
    whatever the name of the column. At each call, the checker takes the
    name from the argument, and the result keeps that name. A declaration
    can hold only one such read, and no other read beside it.

    Args:
        region: The data types that the column can have: a data type, a
            polars family such as `FloatType`, a set of data types,
            `EveryDType`, or `Unchecked` to accept the column as it is.

    Returns:
        The read, to pass to a declaration.

    Example:
        >>> from polars.datatypes.classes import FloatType
        >>> some_column(FloatType)
        BlankRead(region=FloatType)
    """
    return BlankRead(region)


def column_of(schema: object, name: str) -> Any:
    """Read one column of a `Columns` class, at the data type that the class declares.

    `column_of(Trades, "price")` reads `price` at the data type that
    `Trades` gives it. The declaration then follows the class if its data
    type changes.

    Args:
        schema: The `Columns` class.
        name: The name of the column.

    Returns:
        The read, to pass to a declaration.

    Example:
        >>> column_of(object, "price").names
        ('price',)
    """
    return SchemaRead(schema, (name,))


def subset_of(schema: object, *names: str) -> Any:
    """Read several columns of a `Columns` class, each at its declared data type.

    Args:
        schema: The `Columns` class.
        *names: The names of the columns.

    Returns:
        The reads, to pass to a declaration.

    Example:
        >>> subset_of(object, "ask", "bid").names
        ('ask', 'bid')
    """
    return SchemaRead(schema, names)


def wrap(constructor: object, inner: object) -> Any:
    """Declare a nested data type around another result.

    `gives=wrap(pl.List, same)` declares a `List` of the data type of the
    column read, as `implode()` gives.

    Args:
        constructor: The polars class of the outer data type, for example
            `pl.List`.
        inner: The result inside it: a data type, `same`, or another
            `wrap(...)`.

    Returns:
        The result, to pass as `gives=`.

    Example:
        >>> import polars as pl
        >>> wrap(pl.List, pl.Int64).constructor
        List
    """
    return WrapOutcome(constructor, inner)


def row(*reads: object, gives: object) -> Any:
    """Declare one result for some of the input data types.

    One `gives=` is enough when the result is the same for every input.
    Where the result depends on the input, a declaration takes several rows
    in its place. For example, a sum gives `Int64` for a small integer
    column and the data type of the column for a larger one:

    ```python
    Total = typed_agg(
        row(column("quantity", {pl.Int8, pl.Int16}), gives=pl.Int64),
        row(column("quantity", {pl.Int64, pl.Float64}), gives=same),
    )
    ```

    Every row of one declaration must read the same columns, and no two
    rows can cover the same input data types. In a selection, a row takes
    one region for the picked column, for example `row(FloatType, gives=same)`.

    Args:
        *reads: The columns that the row reads, each with the data types
            that this row covers.
        gives: The result for every input that the row covers.

    Returns:
        The row, to pass to a declaration.

    Example:
        >>> import polars as pl
        >>> row(column_of(object, "q"), gives=same).gives
        SameOutcome(read=None)
    """
    return RowDeclaration(reads, gives)


def typed_expr(*parts: object, gives: object = None, named: str | None = None) -> Any:
    """Declare the type of an expression that gives one value for each row.

    This declaration is the type of an expression that reads `quantity` as
    `Int64`, and gives a `Boolean` column named `is_buy`:

    ```python
    BuyFlag = typed_expr(column("quantity", pl.Int64), gives=pl.Boolean, named="is_buy")
    ```

    Args:
        *parts: The columns that the expression reads, or the `row(...)`
            parts of the declaration. A declaration cannot mix the two.
        gives: The result, when one result covers every input. Leave it out
            when the parts are rows.
        named: The name of the result. If it is left out, the result takes
            the name of the one column that the expression reads.

    Returns:
        The declaration, to assign to a name.

    Example:
        >>> import polars as pl
        >>> typed_expr(column_of(object, "q"), gives=pl.Boolean, named="f").named
        'f'
    """
    return SliceDeclaration("elementwise", parts, gives, named)


def typed_agg(*parts: object, gives: object = None, named: str | None = None) -> Any:
    """Declare the type of an expression that gives one value for each group.

    `pl.col("price").sum()` is such an expression. A declaration with no
    read describes an aggregation such as `pl.len()`, which reads no
    column, and then `named=` is required.

    Args:
        *parts: The columns that the expression reads, or the `row(...)`
            parts of the declaration. A declaration cannot mix the two.
        gives: The result, when one result covers every input. Leave it out
            when the parts are rows.
        named: The name of the result. If it is left out, the result takes
            the name of the one column that the expression reads.

    Returns:
        The declaration, to assign to a name.

    Example:
        >>> import polars as pl
        >>> typed_agg(gives=pl.UInt32, named="len").character
        'aggregating'
    """
    return SliceDeclaration("aggregating", parts, gives, named)


@dataclass(frozen=True)
class SelectionDeclaration:
    """The runtime record of `typed_selection` or `typed_agg_selection`.

    It holds how the selection treats rows, the selector, the rows or the
    one result, and the prefix and the suffix for the names of the results.

    Example:
        >>> import polars.selectors as cs
        >>> typed_selection(cs.numeric(), gives=same, suffix="_r").suffix
        '_r'
    """

    character: str
    selector: object
    parts: tuple[object, ...]
    gives: object
    prefix: str | None
    suffix: str | None


def typed_selection(
    selector: object,
    *parts: object,
    gives: object = None,
    suffix: str | None = None,
    prefix: str | None = None,
) -> Any:
    """Declare the type of a selection that gives one value for each row.

    A selection is a selector with operations after it, for example
    `cs.numeric().round(2)`. It gives one result for each column that the
    selector picks. `typed_selection(cs.numeric(), gives=same)` is the type
    of any such selection that keeps the data type of each column.

    Args:
        selector: The selector, written with the `cs.*` functions of
            polars.
        *parts: The `row(region, gives=...)` parts, where the result depends
            on the data type of the picked column.
        gives: The result for every picked column, when one result covers
            them all.
        suffix: The text that every result adds at the end of its name.
        prefix: The text that every result adds at the start of its name.

    Returns:
        The declaration, to assign to a name.

    Example:
        >>> import polars as pl
        >>> import polars.selectors as cs
        >>> typed_selection(cs.numeric(), gives=pl.Float64).character
        'elementwise'
    """
    return SelectionDeclaration("elementwise", selector, parts, gives, prefix, suffix)


def typed_agg_selection(selector: object, *parts: object, gives: object = None) -> Any:
    """Declare the type of a selection that gives one value for each group.

    `cs.numeric().sum()` is such a selection. It gives one result for each
    column that the selector picks.

    Args:
        selector: The selector, written with the `cs.*` functions of
            polars.
        *parts: The `row(region, gives=...)` parts, where the result depends
            on the data type of the picked column.
        gives: The result for every picked column, when one result covers
            them all.

    Returns:
        The declaration, to assign to a name.

    Example:
        >>> import polars.selectors as cs
        >>> typed_agg_selection(cs.numeric(), gives=same).character
        'aggregating'
    """
    return SelectionDeclaration("aggregating", selector, parts, gives, None, None)


def typed_lit(*, gives: object, named: str | None = None) -> Any:
    """Declare the type of an expression that gives one fixed value and reads no column.

    `pl.lit(True)` is such an expression.

    Args:
        gives: The data type of the value.
        named: The name of the result. If it is left out, the name is
            `literal`, as polars gives it.

    Returns:
        The declaration, to assign to a name.

    Example:
        >>> import polars as pl
        >>> typed_lit(gives=pl.Boolean).gives
        Boolean
    """
    return SliceDeclaration("literal", (), gives, named)


class InferredExpr:
    """Let the checker compute the return type of a function that returns an expression.

    The type of an expression is long, and a function with several
    `return` statements can give several types. Write `InferredExpr` as the
    return type, and wrap each returned value in `infer(...)`:

    ```python
    def price_expr(currency: str) -> InferredExpr:
        if currency == "USD":
            return infer(pl.col("price_usd"))
        return infer(pl.col("price_eur").cast(pl.Float64))
    ```

    At each call of the function, the checker gives the union of the
    types of the `infer(...)` arguments. Here, that is the union of the two
    typed expressions, with no union written by hand.

    Each value in `infer(...)` must be a typed expression or a plain
    `pl.Expr`. If one of them is a plain `pl.Expr`, the whole result is a
    plain `pl.Expr`. The setting `disallow_implicit_inferred_fallback`
    makes that case an error. `InferredExpr` is correct only as a return
    type. For a parameter that accepts any expression, use a type variable
    bound to `pl.Expr`:

    ```python
    def normalize[ExprT: pl.Expr](e: ExprT, upper: bool) -> InferredExpr:
        if upper:
            return infer(e.str.to_uppercase())
        return infer(e)
    ```

    The type variable carries the type of the argument through both
    branches. So each call gets a result typed from its own argument, as
    with any generic function.

    Example:
        >>> InferredExpr
        <class 'astrolabe.declare.InferredExpr'>
    """


class InferredDataFrame:
    """Let the checker compute the return type of a function that returns a `DataFrame`.

    It works as `InferredExpr` does, for a function that returns a frame
    in place of an expression. Write it as the return type, and wrap each
    returned value in `infer(...)`. Each value must be a `TypedDataFrame`
    or a plain `pl.DataFrame`. At each call, the checker gives the union
    of their types, or a plain `pl.DataFrame` if one of them is plain.

    Example:
        >>> InferredDataFrame
        <class 'astrolabe.declare.InferredDataFrame'>
    """


class InferredLazyFrame:
    """Let the checker compute the return type of a function that returns a `LazyFrame`.

    It works as `InferredExpr` does, for a function that returns a lazy
    frame in place of an expression. Write it as the return type, and wrap
    each returned value in `infer(...)`. Each value must be a
    `TypedLazyFrame` or a plain `pl.LazyFrame`. At each call, the checker
    gives the union of their types, or a plain `pl.LazyFrame` if one of
    them is plain.

    Example:
        >>> InferredLazyFrame
        <class 'astrolabe.declare.InferredLazyFrame'>
    """


class InferredRemainder:
    """Let the checker compute the remainder of a returned frame, and keep its declared columns.

    The remainder of a frame is the set of columns that its type does not
    name. Write `InferredRemainder` in the remainder place of a return
    type, for example `TypedDataFrame[WithTotal, InferredRemainder]`, and
    wrap each returned value in `infer(...)`. Each value must hold every
    column that the return type declares, with a data type that agrees,
    and the checker reports a value that does not. At each call, the
    checker gives the union of the types of the values. A value keeps its
    other declared columns, and its remainder is written in the terms of
    the caller's own frame.

    Example:
        >>> InferredRemainder
        <class 'astrolabe.declare.InferredRemainder'>
    """


def infer[T](x: T) -> T:
    """Mark one value that a function returns, so that the checker adds its type.

    Use it in a function whose return type is `InferredExpr`,
    `InferredDataFrame`, `InferredLazyFrame`, or a frame type with the
    remainder `InferredRemainder`. The type of `x` becomes one member of
    the union that the function returns. At run time, `infer`
    gives `x` back unchanged.

    Write the call as the whole value of the `return` statement:
    `return infer(x)`. The checker does not count a call in any other
    place, for example `result = infer(x)` followed by `return result`. It
    reports such a call as an `[infer-misuse]` error. A call in a function with another
    return type is an error too.

    Args:
        x: The value to return.

    Returns:
        `x`, unchanged.

    Example:
        >>> infer(42)
        42
        >>> infer("hello")
        'hello'
    """
    return x
