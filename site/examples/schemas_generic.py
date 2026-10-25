# file: schemas.py
import polars as pl

from astrolabe import Columns, TypedDataFrame


class Keyed[T: pl.DataType](Columns):
    key: pl.String
    value: T


def non_empty[T: pl.DataType](df: TypedDataFrame[Keyed[T]]) -> TypedDataFrame[Keyed[T]]:
    return df.filter(pl.col("key").str.len_chars() > 0)


counts: TypedDataFrame[Keyed[pl.UInt32]]
kept = non_empty(counts)
reveal_type(kept)
