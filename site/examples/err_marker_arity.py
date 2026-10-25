# file: example.py
import polars as pl

from astrolabe import Columns, Prefixed, TypedDataFrame, other_columns


class Keys(Columns):
    symbol: pl.String


Rest = other_columns()
frame: TypedDataFrame[Keys, Prefixed[Rest]]
