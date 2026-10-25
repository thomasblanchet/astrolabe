# Known issues

## `arr.dot` and `struct.drop` read as `Unchecked`

- Input: `pl.col("a").arr.dot(pl.col("b"))` or `pl.col("s").struct.drop(["x"])`, with polars 1.44.0 or a later release, which adds the two members.
- The checker: `Unchecked`, because no definition of the rules states the two nodes (`D-37`).
- Polars: `arr.dot` casts the two arrays to their supertype, refuses two arrays of other widths, and gives the type of a sum of the inner type. `struct.drop` gives the struct without the named fields, and with `strict=True` it refuses a name that the struct does not have, when it builds the plan.
- A definition for each node in `src/astrolabe/_rules/definitions/` closes the gap.
