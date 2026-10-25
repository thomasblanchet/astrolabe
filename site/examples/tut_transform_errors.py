# file: pipeline.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
venues = pl.read_csv(
    "venues.csv",
    schema={"symbol": pl.Int64, "venue": pl.String},
)
trades.filter(pl.col("price"))
trades.with_columns(pl.col("symbol") - pl.col("price"))
trades.join(venues, on="symbol")
