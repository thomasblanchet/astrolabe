# file: python.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
labelled = trades.with_columns(label=pl.col("symbol").map_elements(str.lower))
reveal_type(labelled)  # added: label
