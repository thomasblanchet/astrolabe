"""The frame half of the engine: what the checker can prove about a remainder.

A function can declare only the columns it reads, for example
`df: TypedDataFrame[Trades, Rest]` with `Rest = other_columns()`.
`Rest` stands for every other column that a caller brings. These
columns are the remainder: the columns of a frame that its type does
not name. If a caller's frame also holds `venue`, the remainder of
that call is `venue`. The function never sees these names, so the
checker tracks what it can prove about them. It proves some names
absent, and it limits the data types that the columns can have.

This package holds that logic: rules over names, sets of data types,
selectors over whole columns, the effect of each operation on a
remainder, containment between two descriptions, one canonical form
for each remainder type, and the split of a remainder into parts.

Every class here works on plain Python values and never imports
mypy. The encoding layer translates between these values and the
types that mypy carries. So each rule is ordinary code with a
doctest.
"""
