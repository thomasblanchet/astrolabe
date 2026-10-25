# file: load.py
import polars as pl

lazy = pl.scan_csv("trades.csv", schema={"symbol": pl.String, "price": pl.Float64})
reveal_type(lazy)
rows = pl.from_dicts(
    [{"symbol": "A", "price": 1.0}], schema={"symbol": pl.String, "price": pl.Float64}
)
reveal_type(rows)
