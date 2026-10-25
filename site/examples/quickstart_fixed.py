# file: quickstart.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64


def cheapest(trades: TypedDataFrame[Trades]) -> TypedDataFrame[Trades]:
    return trades.sort("price").head(1)


trades: TypedDataFrame[Trades]
reveal_type(trades.with_columns(doubled=pl.col("price") * 2))  # added: doubled
