# file: schemas.py
from typing import Literal

import polars as pl

from astrolabe import Columns, Datetime, Decimal, Enum, List, TypedDataFrame


class Fills(Columns):
    time: Datetime[Literal["us"], None]
    cost: Decimal[Literal[10], Literal[2]]
    lots: List[pl.Int64]
    side: Enum[Literal["buy"], Literal["sell"]]


fills: TypedDataFrame[Fills]
reveal_type(fills)
