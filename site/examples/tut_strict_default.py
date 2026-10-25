# file: pipeline.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
rounded = trades.with_columns(pl.col("price").map_elements(round))
reveal_type(rounded)
