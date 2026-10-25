# file: scores.py
import polars as pl  # hide
from polars.datatypes.classes import FloatType  # hide

from astrolabe import Columns, TypedDataFrame, same, some_column, typed_expr  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64


class Readings(Columns):
    temp: pl.Float32


Standardised = typed_expr(some_column(FloatType), gives=same)


def zscore(e: Standardised) -> Standardised:
    return (e - e.mean()) / e.std()


trades: TypedDataFrame[Trades]
readings: TypedDataFrame[Readings]
reveal_type(trades.select(zscore(pl.col("price"))))
reveal_type(readings.select(zscore(pl.col("temp"))))
zscore(pl.col("price").alias("z"))
