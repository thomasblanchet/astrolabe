# file: prices.py
import polars as pl

from astrolabe import InferredDataFrame, infer


def doubled(price: pl.Expr) -> pl.Expr:
    return infer(price * 2)


def frame() -> InferredDataFrame:
    return infer(pl.col("price"))


frame()
