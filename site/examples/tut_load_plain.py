# file: pipeline.py
import polars as pl

raw = pl.read_csv("trades.csv")
raw.select("symbol", "prcie")
reveal_type(raw)
