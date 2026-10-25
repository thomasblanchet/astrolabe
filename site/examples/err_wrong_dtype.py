# file: example.py
import polars as pl

from astrolabe import Columns, TypedDataFrame, other_columns


class Trades(Columns):
    symbol: pl.String
    quantity: pl.Int64


class Rounded(Columns):
    symbol: pl.String
    quantity: pl.Float64


Rest = other_columns()


def total(df: TypedDataFrame[Trades, Rest]) -> None: ...


rounded: TypedDataFrame[Rounded]
total(rounded)
