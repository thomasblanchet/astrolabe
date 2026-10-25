# file: features.py
import polars as pl

from astrolabe import Columns, InferredRemainder, TypedDataFrame, infer, other_columns


class Priced(Columns):
    price: pl.Float64
    quantity: pl.Int64


class WithNotional(Priced):
    notional: pl.Float64


Rest = other_columns()


def add_notional(
    df: TypedDataFrame[Priced, Rest],
) -> TypedDataFrame[WithNotional, InferredRemainder]:
    return infer(df.with_columns(notional=pl.col("price") * pl.col("quantity")))


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
result = add_notional(trades)
reveal_type(result)  # added: notional; remainder: symbol
