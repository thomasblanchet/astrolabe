# file: features.py
import polars as pl  # hide

from astrolabe import Columns, TypedDataFrame, other_columns  # hide


class Priced(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


Rest = other_columns()


def by_venue(df: TypedDataFrame[Priced, Rest]) -> None:
    df.group_by("venue").agg(pl.len())
