# file: example.py
# setting: disallow_implicit_inferred_fallback = true
import polars as pl

from astrolabe import InferredExpr, infer


def scaled(price: pl.Expr, factor: float) -> InferredExpr:
    return infer(price * factor)


scaled(pl.col("price"), 2.0)
