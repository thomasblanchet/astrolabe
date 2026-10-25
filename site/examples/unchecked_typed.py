# file: python.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
lowered = pl.col("symbol").map_elements(str.lower, return_dtype=pl.String)
labelled = trades.with_columns(label=lowered)
reveal_type(labelled)  # added: label
