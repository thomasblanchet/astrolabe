# file: schemas.py
import polars as pl

from astrolabe import Columns, TypedDataFrame


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


class Blotter(Trades):
    venue: pl.String


blotter: TypedDataFrame[Blotter]
reveal_type(blotter)
