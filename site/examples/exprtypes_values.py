# file: flags.py
import polars as pl  # hide

from astrolabe import column, typed_expr  # hide

Price = column("price", pl.Float64)
Expensive = typed_expr(Price, gives=pl.Boolean, named="expensive")

rule: Expensive = (pl.col("price") > 100.0).alias("expensive")
wrong: Expensive = pl.col("price").alias("expensive")


def expensive(price: Price) -> Expensive:
    return (price > 100.0).alias("expensive")


plain: pl.Expr
expensive(plain)
