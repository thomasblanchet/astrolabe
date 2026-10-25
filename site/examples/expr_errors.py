# file: expressions.py
from typing import Literal

import polars as pl

from astrolabe import Columns, Datetime, TypedDataFrame


class Fills(Columns):
    symbol: pl.String
    time: Datetime[Literal["us"], None]


fills: TypedDataFrame[Fills]
fills.select(pl.col("symbol").round(2))
fills.select(pl.col("time").cast(pl.Boolean))
fills.select(pl.col("symbol") + pl.col("time"))
