# file: example.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]


class Quotes(Columns):
    symbol: pl.String
    price: pl.Float64


quotes: TypedDataFrame[Quotes]
trades.vstack(quotes)
