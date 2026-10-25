# file: totals.py
import polars as pl  # hide

from astrolabe import Columns, TypedDataFrame, column, row, same, typed_agg  # hide


class Small(Columns):
    quantity: pl.Int8


class Wide(Columns):
    quantity: pl.Int32


Quantity = column("quantity", {pl.Int8, pl.Int16, pl.Int32, pl.Int64})
SameType = typed_agg(Quantity, gives=same)
Total = typed_agg(
    row(column("quantity", {pl.Int8, pl.Int16}), gives=pl.Int64),
    row(column("quantity", {pl.Int32, pl.Int64}), gives=same),
)


def wrong_total(quantity: Quantity) -> SameType:
    return quantity.sum()


def total(quantity: Quantity) -> Total:
    return quantity.sum()


small: TypedDataFrame[Small]
wide: TypedDataFrame[Wide]
reveal_type(small.select(total(pl.col("quantity"))))
reveal_type(wide.select(total(pl.col("quantity"))))
