# file: load.py
import polars as pl

raw = pl.read_csv("trades.csv")
picked = raw.select("symbol", pl.col("price").cast(pl.Float64))
reveal_type(picked)
