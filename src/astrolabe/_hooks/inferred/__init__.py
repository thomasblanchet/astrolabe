"""Give a function the return type that its body builds, so that the user need not write it.

A declared expression type or frame type can be long. A user who does
not want to write one declares the return type as `InferredExpr`,
`InferredDataFrame` or `InferredLazyFrame`, and wraps each returned
value in `infer(...)`. These three classes are placeholders: they carry
no type of their own. At each call of the function, the checker gives
the union of the types of the values that `infer(...)` marks.

The work has two halves, which meet in one shared record. While mypy
checks the body of the function, a hook on `infer` records the type of
each marked value. When mypy later checks a call of the function,
another hook reads those records and builds the return type.
"""
