# file: notional.py
import polars as pl  # hide
from astrolabe import Columns, TypedDataFrame, other_columns  # hide


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


Clean = other_columns(without=["notional"])


def add_notional(df: TypedDataFrame[Trades, Clean]) -> TypedDataFrame[Trades, Clean]:
    return df


class Booked(Trades):
    notional: pl.Float64


booked: TypedDataFrame[Booked]
add_notional(booked)
