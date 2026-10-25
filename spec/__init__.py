"""The specification of astrolabe's type system, and the checks that hold it to account.

Four documents describe the system. `DATAFRAME_SPECIFICATION.md` and
`EXPRESSION_SPECIFICATION.md` say what it is and
why, `OPERATIONS.md` gives the rules and proves them, and
`OPERATION_CATALOG.md` applies them to every operation polars offers.

`tests/` holds the executable evidence. Every number the documents quote is
recomputed there, every theorem about the model is checked against a reference
implementation, and every claim about polars is a call that runs.
"""
