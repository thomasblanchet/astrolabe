# file: boundary.py
import polars as pl

from astrolabe import Columns, TypedDataFrame, check_schema, other_columns


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64


Rest = other_columns()

raw = pl.read_parquet("trades.parquet")
trades = check_schema(raw, TypedDataFrame[Trades, Rest])
