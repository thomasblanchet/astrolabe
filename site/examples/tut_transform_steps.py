# file: pipeline.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
priced = trades.with_columns(notional=pl.col("price") * pl.col("quantity"))
reveal_type(priced)  # added: notional
large = priced.filter(pl.col("notional") > 10_000)
by_symbol = large.group_by("symbol").agg(
    pl.col("notional").sum(),
    pl.len(),
)
reveal_type(by_symbol)  # added: len; dropped: price, quantity
