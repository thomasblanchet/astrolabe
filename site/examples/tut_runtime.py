# file: pipeline.py
import polars as pl

from astrolabe import Columns, TypedDataFrame, check_schema


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


raw = pl.read_parquet("trades.parquet")
trades = check_schema(raw, TypedDataFrame[Trades])
reveal_type(trades)
