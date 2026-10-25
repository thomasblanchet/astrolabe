# file: boundary.py
import polars as pl

from astrolabe import Columns, TypedDataFrame, TypedLazyFrame, check_schema, force_schema


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64


checked = check_schema(pl.read_parquet("trades.parquet"), TypedDataFrame[Trades])
cast = force_schema(pl.read_csv("trades.csv"), TypedDataFrame[Trades])
lazy = check_schema(pl.scan_parquet("trades.parquet"), TypedLazyFrame[Trades])
reveal_type(lazy)
