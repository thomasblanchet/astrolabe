# file: pricing.py
import polars as pl  # hide

from astrolabe import Columns, TypedDataFrame, column  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


Price = column("price", pl.Float64)


def discounted(price: Price) -> Price:
    return price * 0.9


trades: TypedDataFrame[Trades]
reveal_type(trades.select(discounted(pl.col("price"))))
discounted(pl.col("quantity"))
discounted(pl.col("price").cast(pl.String))
discounted(pl.col("price").sum())
