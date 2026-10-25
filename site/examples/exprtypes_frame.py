# file: pricing.py
import polars as pl  # hide

from astrolabe import Columns, TypedDataFrame, column  # hide


class Sizes(Columns):
    price: pl.Int32


Price = column("price", pl.Float64)


def discounted(price: Price) -> Price:
    return price * 0.9


sizes: TypedDataFrame[Sizes]
sizes.select(discounted(pl.col("price")))
