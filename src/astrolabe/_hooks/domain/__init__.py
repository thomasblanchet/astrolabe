"""Wrap the mypy types of frames, columns and series, and hold the rules that compare data types.

Every frame hook reads the columns of a typed frame and builds the typed frame that the
operation gives. `Frame` and `ColumnRef` do that work, so no hook reads the raw mypy types
itself. The other modules hold rules that several operations share: which data types match,
which common data type two columns become, and what the checker gives when it cannot prove a
result. They sit in the plugin layer, and the engine never sees them.
"""
