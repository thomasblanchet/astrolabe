# file: expressions.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
one_row = trades.select(pl.col("quantity").sum())
reveal_type(one_row)
broadcast = trades.with_columns(total=pl.col("quantity").sum())
reveal_type(broadcast)  # added: total
per_symbol = trades.group_by("symbol").agg(pl.col("quantity").sum())
reveal_type(per_symbol)
