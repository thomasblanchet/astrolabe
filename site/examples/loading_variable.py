# file: load.py
import polars as pl

schema = {"symbol": pl.String, "price": pl.Float64}
trades = pl.read_csv("trades.csv", schema=schema)
reveal_type(trades)
