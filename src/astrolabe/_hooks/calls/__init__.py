"""Read what a user wrote in a call, for hooks that need values and not only types.

mypy gives a hook the type of each argument. A check often needs more: the column name in
`select("price")`, the number in `head(3)`, the time unit in `pl.Datetime("ms")`. These modules
read such values from the abstract syntax tree (AST) of the call, and from the types that carry
a known value. `CallArgs` reads the arguments of a checked call by parameter name. `ColumnExpr`
reads a column name written as a string, or the last `.alias(...)` of a chain. The other modules
read literals and data types. The engine receives only the plain values that they answer.
"""
