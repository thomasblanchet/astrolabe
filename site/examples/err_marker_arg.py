# file: example.py
import polars as pl

from astrolabe import Columns, TypedDataFrame


class Keys(Columns):
    symbol: pl.String


frame: TypedDataFrame[Keys, int]
