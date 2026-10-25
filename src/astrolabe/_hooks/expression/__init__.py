"""The expression half of the engine: the rules that type a polars expression.

The checker gives each polars expression a type that says which columns
it reads, which data type it gives for each data type of those columns,
what it names its output, and how it counts rows. This package holds the
rules that build and combine those types. It works on plain Python
values: data-type names, tables of answers and the expression types
themselves.

Nothing here imports mypy. The encoding layer turns these values into
mypy types and back, and the plugin layer decides when to ask a
question. So each rule has one home, and a doctest can check it with no
type checker.
"""
