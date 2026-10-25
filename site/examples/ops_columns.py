# file: operations.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
trimmed = trades.drop("quantity").rename({"price": "px"})
reveal_type(trimmed)  # added: px; dropped: price, quantity
