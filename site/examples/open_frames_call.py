# file: notional.py
import polars as pl  # hide

from astrolabe import Columns, InferredRemainder, TypedDataFrame, infer, other_columns  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


class WithNotional(Trades):
    notional: pl.Float64


Rest = other_columns()


def add_notional(
    df: TypedDataFrame[Trades, Rest],
) -> TypedDataFrame[WithNotional, InferredRemainder]:
    return infer(df.with_columns(notional=pl.col("price") * pl.col("quantity")))


class Blotter(Trades):
    venue: pl.String


blotter: TypedDataFrame[Blotter]
result = add_notional(blotter)
reveal_type(result)  # added: notional; remainder: venue
