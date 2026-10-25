# file: schemas.py
import polars as pl

from astrolabe import Columns, TypedDataFrame


class Prices(Columns):
    symbol: pl.String
    price: float


class Partial(Columns, total=False):
    symbol: pl.String


prices: TypedDataFrame[Prices]
partial: TypedDataFrame[Partial]
