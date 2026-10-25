# file: pipeline.py
# setting: allow_unproven = "stated"
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
rounded = trades.with_columns(pl.col("price").map_elements(round, return_dtype=pl.Int64))
reveal_type(rounded)
