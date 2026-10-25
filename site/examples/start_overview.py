# file: pipeline.py
import polars as pl

from astrolabe import Columns, TypedDataFrame


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
trades.select("venue")
trades.select(pl.col("symbol") - pl.col("price"))
trades.filter(pl.col("price"))
