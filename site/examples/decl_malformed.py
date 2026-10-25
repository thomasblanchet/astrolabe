# file: features.py
import polars as pl

from astrolabe import column, typed_expr

NoRead = typed_expr(gives=pl.Int64, named="n")
Pair = typed_expr(column("price", pl.Float64), column("quantity", pl.Int64), gives=pl.Float64)
