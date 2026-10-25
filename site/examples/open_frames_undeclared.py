# file: venues.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame, other_columns  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


Rest = other_columns()


def by_venue(df: TypedDataFrame[Trades, Rest]) -> None:
    df.select("symbol", "venue")
