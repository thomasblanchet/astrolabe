# file: enrich.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame, other_columns  # hide


class Keys(Columns):
    symbol: pl.String


Left = other_columns()
Right = other_columns()


def enrich(left: TypedDataFrame[Keys, Left], right: TypedDataFrame[Keys, Right]) -> None:
    left.join(right, on="symbol")
