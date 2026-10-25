# file: operations.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
by_symbol = trades.group_by("symbol").agg(pl.col("price").mean(), pl.col("quantity").sum())
reveal_type(by_symbol)
counts = trades.group_by("symbol").len()
reveal_type(counts)  # added: len
