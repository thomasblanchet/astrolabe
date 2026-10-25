# file: operations.py
import polars as pl

from astrolabe import Columns, List, TypedDataFrame


class Baskets(Columns):
    basket: pl.String
    symbols: List[pl.String]


baskets: TypedDataFrame[Baskets]
exploded = baskets.explode("symbols")
reveal_type(exploded)
baskets.explode("basket")
