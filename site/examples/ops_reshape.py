# file: operations.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
long = trades.unpivot(index="symbol")
reveal_type(long)  # added: variable, value; dropped: price, quantity
numbered = trades.with_row_index()
reveal_type(numbered)  # added: index
