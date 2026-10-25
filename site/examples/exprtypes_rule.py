# file: flags.py
import polars as pl  # hide

from astrolabe import Columns, InferredRemainder, TypedDataFrame  # hide
from astrolabe import infer, other_columns, typed_expr  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


class Flagged(Trades):
    flag: pl.Boolean


Rest = other_columns()
FlagRule = typed_expr(Trades, gives=pl.Boolean, named="flag")


def flagged(
    df: TypedDataFrame[Trades, Rest],
    rule: FlagRule,
) -> TypedDataFrame[Flagged, InferredRemainder]:
    return infer(df.with_columns(rule))


class Blotter(Trades):
    venue: pl.String


blotter: TypedDataFrame[Blotter]
large = (pl.col("price") * pl.col("quantity") > 1e6).alias("flag")
result = flagged(blotter, large)
reveal_type(result)  # added: flag; remainder: venue
flagged(blotter, (pl.col("venue") == "X").alias("flag"))
flagged(blotter, pl.lit(True).alias("flag"))
