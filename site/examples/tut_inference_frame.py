# file: features.py
import polars as pl  # hide
from astrolabe import Columns, InferredDataFrame, TypedDataFrame, infer  # hide
from astrolabe import other_columns  # hide


class Priced(Columns):
    price: pl.Float64
    quantity: pl.Int64


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


Rest = other_columns()
trades: TypedDataFrame[Trades]


def summary(df: TypedDataFrame[Priced, Rest], by_value: bool) -> InferredDataFrame:
    if by_value:
        return infer(df.select((pl.col("price") * pl.col("quantity")).sum().alias("value")))
    return infer(df.select(pl.col("quantity").sum()))


reveal_type(summary(trades, True))
