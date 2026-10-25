# file: pricing.py
import polars as pl  # hide

from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64


def discounted(price: pl.Expr) -> pl.Expr:
    return price * 0.9


trades: TypedDataFrame[Trades]
reveal_type(trades.select(discounted(pl.col("price"))))
