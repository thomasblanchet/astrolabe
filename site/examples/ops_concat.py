# file: operations.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]


class Quotes(Columns):
    symbol: pl.String
    venue: pl.String


quotes: TypedDataFrame[Quotes]
diagonal = pl.concat([trades, quotes], how="diagonal")
reveal_type(diagonal)
vertical = pl.concat([trades, trades])
reveal_type(vertical)
