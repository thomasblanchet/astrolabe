# file: schemas.py
import polars as pl

from astrolabe import Columns, TypedDataFrame


class Loose(Columns, closed=False):
    symbol: pl.String


loose: TypedDataFrame[Loose]
