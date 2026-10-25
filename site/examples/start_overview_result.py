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
