# file: flags.py
import polars as pl  # hide
from polars.datatypes.classes import NumericType  # hide

from astrolabe import Columns, TypedDataFrame, column, typed_expr  # hide


class Small(Columns):
    quantity: pl.Int8


class Texts(Columns):
    quantity: pl.String


Quantity = column("quantity", NumericType)
BuyFlag = typed_expr(Quantity, gives=pl.Boolean, named="is_buy")


def buy_flag(quantity: Quantity) -> BuyFlag:
    return (quantity > 0).alias("is_buy")


small: TypedDataFrame[Small]
texts: TypedDataFrame[Texts]
reveal_type(small.select(buy_flag(pl.col("quantity"))))
texts.select(buy_flag(pl.col("quantity")))
