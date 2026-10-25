# file: features.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame, other_columns  # hide


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


class Blotter(Priced):
    venue: pl.String


class Quotes(Columns):
    symbol: pl.String
    price: pl.Float64


blotter: TypedDataFrame[Blotter]
quotes: TypedDataFrame[Quotes]
turnover(blotter)
turnover(quotes)
