# file: prices.py
import polars as pl

from astrolabe import Columns, InferredExpr, TypedDataFrame, infer


class Quotes(Columns):
    price_usd: pl.Float64
    price_eur: pl.Float32


def price(currency: str) -> InferredExpr:
    if currency == "USD":
        return infer(pl.col("price_usd"))
    return infer(pl.col("price_eur").cast(pl.Float64))


quotes: TypedDataFrame[Quotes]
in_euros = quotes.select(price("EUR"))
reveal_type(in_euros)
