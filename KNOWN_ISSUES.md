# Known issues

## A remainder column of the kind `Extension` reads as `Unchecked` under a selection

- Input: an open frame whose remainder holds an `Extension` column, and a selection with an operation that keeps the data type, for example `df.select(cs.all().fill_null(strategy="forward"))`.
- The checker: the remainder column takes `Unchecked`. The sample of data types that the checker tries for an open column holds no data type of the kind `Extension`.
- Polars: the column keeps its `Extension` data type.
- A selector with no operation keeps the column as it is. A sample member of the kind `Extension` closes the gap.

## The class `pl.Decimal` inside a nested data type reads as a `Decimal` of unknown precision

- Input: `pl.List(pl.Decimal)` or `pl.Struct({"a": pl.Decimal})`, with the class `pl.Decimal` and not an instance, in a cast or a typed literal.
- The checker: a `Decimal` of unknown precision, the same as an instance whose type names no precision, for example a parameter annotated `pl.Decimal`. A cast or a literal of such a nested type gets no error.
- Polars: a frame schema reads the class as `Decimal(38, 0)`, and an expression raises `TypeError`.
- The class at the top level is exact. A marker for the class closes the gap.

## An expression over two columns of unknown data types reads as `Unchecked`

- Input: `pl.concat_str(pl.col("a"), pl.col("b"))` over a plain frame.
- The checker: `Unchecked`. It tries each data type of its sample for a column whose data type it does not know, and two such columns give too many pairs to try.
- Polars: `String` at every pair of data types.
- One column of unknown data type keeps the whole sample.

## A null that the code decides is lost at a node that keeps some rows only

- Input: `pl.rolling_corr(pl.col("x").shift(1).head(1), pl.col("x"), window_size=2)`.
- The checker: accepts the call. The rules know that `shift(1)`, `diff`, `pct_change`, a rolling window and an exponentially weighted window that needs two samples hold a null wherever they hold a row, and an elementwise node keeps that fact. `head(1)` drops it, because the rules do not track where the null sits.
- Polars: panics.
- Without the `head(1)`, the checker reports the panic of polars.
