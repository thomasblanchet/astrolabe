# file: selections.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame, same, typed_selection  # hide
import polars.selectors as cs  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
Rounded = typed_selection(cs.numeric(), gives=same)


def rounded(columns: Rounded) -> Rounded:
    return columns.round(2)


tidy = trades.with_columns(rounded(cs.numeric()))
reveal_type(tidy)
