# file: lazy.py
import polars as pl

trades = pl.scan_csv("trades.csv", schema={"symbol": pl.String, "price": pl.Float64})
large = trades.filter(pl.col("price") > 100).collect()
reveal_type(large)
