# file: expressions.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
summary = trades.select(
    pl.col("price") * 2,
    pl.col("symbol").str.len_chars(),
    pl.when(pl.col("quantity") > 0).then(pl.col("price")).otherwise(0).alias("buy_price"),
)
reveal_type(summary)
