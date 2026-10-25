# file: pipeline.py
import polars as pl

trades = pl.read_csv(
    "trades.csv",
    schema={
        "symbol": pl.String,
        "price": pl.Float64,
        "quantity": pl.Int64,
    },
)
trades.select("symbol", "prcie")
