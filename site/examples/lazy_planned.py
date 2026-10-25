# file: lazy.py
from typing import Literal

import polars as pl

from astrolabe import Columns, Decimal, TypedLazyFrame


class Lots(Columns):
    price: Decimal[Literal[10], Literal[2]]


lots: TypedLazyFrame[Lots]
running = lots.with_columns(total=pl.col("price").cum_sum())
reveal_type(running)
reveal_type(running.collect())
