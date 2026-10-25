# file: schemas.py
import polars as pl

from astrolabe import TypedDataFrame, columns

Stamp = pl.Datetime("us")
Money = pl.Decimal(20, 4)
Trade = columns("Trade", {"time": Stamp, "cost": Money})

trade: TypedDataFrame[Trade]
reveal_type(trade)
