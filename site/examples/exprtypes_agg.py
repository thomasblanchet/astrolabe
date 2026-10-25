# file: totals.py
import polars as pl  # hide

from astrolabe import Columns, TypedDataFrame, column, typed_agg  # hide


class Trades(Columns):
    symbol: pl.String
    quantity: pl.Int64


Quantity = column("quantity", pl.Int64)
TotalQuantity = typed_agg(Quantity, gives=pl.Int64)


def total_quantity(quantity: Quantity) -> TotalQuantity:
    return quantity.sum()


trades: TypedDataFrame[Trades]
reveal_type(trades.select(total_quantity(pl.col("quantity"))))
reveal_type(trades.group_by("symbol").agg(total_quantity(pl.col("quantity"))))
