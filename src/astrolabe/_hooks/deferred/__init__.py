"""The code of an expression, as a deferred type stores it, and the readers of that code.

A deferred type holds the code of a polars expression, as the user wrote it
with the public API of polars, beside the declared types that the code reads.
Nothing in this package refers to mypy: it composes the code of a call from
the codes of its parts, reads a code back into a polars expression on
placeholder inputs, and reads the tree that polars builds for it.
"""
