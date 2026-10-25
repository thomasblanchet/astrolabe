# file: enrich.py
import polars as pl

from astrolabe import Columns, InferredRemainder, TypedDataFrame, infer, other_columns


class Keys(Columns):
    symbol: pl.String


Left = other_columns()
Right = other_columns(disjoint_from=Left)


def enrich(
    left: TypedDataFrame[Keys, Left],
    right: TypedDataFrame[Keys, Right],
) -> TypedDataFrame[Keys, InferredRemainder]:
    return infer(left.join(right, on="symbol"))


class Quotes(Keys):
    bid: pl.Float64


class Venues(Keys):
    venue: pl.String


quotes: TypedDataFrame[Quotes]
venues: TypedDataFrame[Venues]
reveal_type(enrich(quotes, venues))  # remainder: bid, venue
