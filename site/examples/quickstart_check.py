# file: quickstart.py
import polars as pl

from astrolabe import Columns, TypedDataFrame


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64


def cheapest(trades: TypedDataFrame[Trades]) -> TypedDataFrame[Trades]:
    return trades.sort("prices").head(1)
