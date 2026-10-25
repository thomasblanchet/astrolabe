# file: example.py
import polars as pl

from astrolabe import Columns, TypedDataFrame, other_columns


class Keys(Columns):
    symbol: pl.String


class Priced(Columns):
    symbol: pl.String
    price: pl.Float64


Rest = other_columns()


def stack(left: TypedDataFrame[Keys, Rest], right: TypedDataFrame[Priced]) -> None:
    pl.concat([left, right], how="diagonal")
