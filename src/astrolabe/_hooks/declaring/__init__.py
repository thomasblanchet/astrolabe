"""Read the types that a user declares, and give each one a name that mypy knows.

A user describes an expression type with a plain call assigned to a
name, for example `Price = column("price", pl.Float64)`. At run time the
call gives an ordinary value. The checker must read the same call as a
type, so that `Price` can annotate a parameter or a return value.

The code here reads the syntax of the call during semantic analysis,
the mypy phase that resolves names before any expression has a type. It
checks the rules that a declaration must obey, reports each broken rule
at the call, and registers the name as a type alias for the type that
the call describes. It sits in the plugin layer, and hands the values
it reads to the engine, the part of the project that knows nothing of
mypy.
"""
