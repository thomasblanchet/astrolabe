# file: features.py
import polars as pl

from astrolabe import Columns, TypedDataFrame, column


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
Price = column("price", pl.Float64)


def discounted(price: Price) -> Price:
    return price * 0.9


trades.with_columns(discounted(pl.col("price")))
discounted(pl.col("quantity"))
discounted(pl.col("price").sum())
