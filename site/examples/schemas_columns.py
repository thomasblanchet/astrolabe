# file: schemas.py
import polars as pl

from astrolabe import TypedDataFrame, columns

Pivoted = columns("Pivoted", {"id": pl.Int64, "2024": pl.Float64, "v1|a": pl.Datetime("us")})

pivoted: TypedDataFrame[Pivoted]
picked = pivoted.select("id", "2024")
reveal_type(picked)
