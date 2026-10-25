# file: selections.py
import polars as pl  # hide
import polars.selectors as cs  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
doubled = trades.with_columns((cs.numeric() * 2).name.suffix("_x2"))
reveal_type(doubled)  # added: price_x2, quantity_x2
trades.select(cs.numeric().alias("number"))
