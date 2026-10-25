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
    price: pl.Float64
    venue: pl.String


quotes: TypedDataFrame[Quotes]
joined = trades.join(quotes, on="symbol")
reveal_type(joined)  # added: price_right, venue
