# Known issues

## `arr.dot` and `struct.drop` read as `Unchecked`

- Input: `pl.col("a").arr.dot(pl.col("b"))` or `pl.col("s").struct.drop(["x"])`, with polars 1.44.0 or a later release, which adds the two members.
- The checker: `Unchecked`, because no definition of the rules states the two nodes (`D-37`).
- Polars: `arr.dot` casts the two arrays to their supertype, refuses two arrays of other widths, and gives the type of a sum of the inner type. `struct.drop` gives the struct without the named fields, and with `strict=True` it refuses a name that the struct does not have, when it builds the plan.
- A definition for each node in `src/astrolabe/_rules/definitions/` closes the gap.

## A lazy plan that polars types again after the cast pass, before polars 1.41.0

- Input: `pl.col("l").list.set_union(pl.col("s"))` over a `List(Int64)` column `l` and a `String` column `s`, in a lazy query, with polars 1.40 or an older release.
- The checker: the rules refuse the run with `InvalidOperationError`, as polars 1.41.0 and later releases do, because the cast pass casts `l` to `String` and the kernel needs a list.
- Polars: before 1.41.0, the projection pushdown of a lazy query types each `select` again after the cast pass. The type function of `set_union` refuses the `String` input there, so the query panics with `no valid schema can be derived for the query`.
- The rules do not type a lazy plan again after the cast pass. Where the run fails anyway, as here, only the way that polars stops differs. No case is known where the run of the later releases gives a type and the older releases panic, but the rules do not rule one out. A second typing of the tree after the cast pass, in a lazy query before polars 1.41.0, closes the gap.
