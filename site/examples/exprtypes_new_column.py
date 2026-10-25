# file: flags.py
import polars as pl  # hide

from astrolabe import Columns, TypedDataFrame, column, typed_expr  # hide


class Trades(Columns):
    symbol: pl.String
    quantity: pl.Int64


Quantity = column("quantity", pl.Int64)
BuyFlag = typed_expr(Quantity, gives=pl.Boolean, named="is_buy")


def buy_flag(quantity: Quantity) -> BuyFlag:
    return (quantity > 0).alias("is_buy")


trades: TypedDataFrame[Trades]
reveal_type(trades.with_columns(buy_flag(pl.col("quantity"))))  # added: is_buy
