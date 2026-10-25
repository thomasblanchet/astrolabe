# file: features.py
import polars as pl

from astrolabe import Columns, TypedDataFrame, other_columns


class Priced(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


class Turnover(Columns):
    symbol: pl.String
    turnover: pl.Float64


Rest = other_columns()


def turnover(df: TypedDataFrame[Priced, Rest]) -> TypedDataFrame[Turnover]:
    value = (pl.col("price") * pl.col("quantity")).sum()
    return df.group_by("symbol").agg(value.alias("turnover"))
