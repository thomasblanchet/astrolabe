# Known issues

## A remainder column of the kind `Extension` reads as `Unchecked` under a selection

- Input: an open frame whose remainder holds an `Extension` column, and a selection with an operation that keeps the data type, for example `df.select(cs.all().fill_null(strategy="forward"))`.
- The checker: the remainder column takes `Unchecked`. The sample of data types that the checker tries for an open column holds no data type of the kind `Extension`.
- Polars: the column keeps its `Extension` data type.
- A selector with no operation keeps the column as it is. A sample member of the kind `Extension` closes the gap.

## The class `pl.Decimal` inside a nested data type reads as a `Decimal` of unknown precision

- Input: `pl.List(pl.Decimal)` or `pl.Struct({"a": pl.Decimal})`, with the class `pl.Decimal` and not an instance, in a cast or a typed literal.
- The checker: a `Decimal` of unknown precision, the same as an instance whose type names no precision, for example a parameter annotated `pl.Decimal`. A cast or a literal of such a nested type gets no error.
- Polars: a frame schema reads the class as `Decimal(38, 0)`, and an expression raises `TypeError`.
- The class at the top level is exact. A marker for the class closes the gap.

## An expression over two columns of unknown data types reads as `Unchecked`

- Input: `pl.concat_str(pl.col("a"), pl.col("b"))` over a plain frame.
- The checker: `Unchecked`. It tries each data type of its sample for a column whose data type it does not know, and two such columns give too many pairs to try.
- Polars: `String` at every pair of data types.
- One column of unknown data type keeps the whole sample.

## A null that the code decides is lost at a node that keeps some rows only

- Input: `pl.rolling_corr(pl.col("x").shift(1).head(1), pl.col("x"), window_size=2)`.
- The checker: accepts the call. The rules know that `shift(1)`, `diff`, `pct_change`, a rolling window and an exponentially weighted window that needs two samples hold a null wherever they hold a row, and an elementwise node keeps that fact. `head(1)` drops it, because the rules do not track where the null sits.
- Polars: panics.
- Without the `head(1)`, the checker reports the panic of polars.

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

## The nodes that polars 2.0 adds read as `Unchecked`

- Input: with polars 2.0.0, `pl.col("x").erf()`, `erfc()`, `approx_quantile(...)`, `bin_intervals(...)`, `bin_quantiles(...)`, `bin_ranks(...)`, `pl.col("l").list.to_map()`, `pl.col("s").struct.eval(...)`, and each member of the `map` namespace.
- The checker: `Unchecked`, because no definition of the rules states these nodes (`D-37`).
- Polars: `erf` and `erfc` refuse a type that is not a number or a `Boolean`, and give a float, as `exp` does. `approx_quantile` gives a float, or a list of floats for a list of quantiles. The `bin_*` functions give a category of their labels, or a struct where `include_intervals=True`. `list.to_map` gives a `Map` of the key and value fields of a list of structs.
- A definition for each node in `src/astrolabe/_rules/definitions/` closes the gap.

## The `Map` data type of polars 2.0

- Input: a column of the type `pl.Map(pl.String, pl.Int64)`, with polars 2.0.0.
- The checker: the rules read the type as `Map`, with no key type and no value type (`DType.of`), and no kernel and no supertype knows the kind.
- Polars: a `Map` holds a key type and a value type. Its supertype beside another `Map` joins the value types where the key types agree, and the `map` namespace reads its keys and values.
- A kind `Map` in `DType`, with its two parameters, and the rules of the supertype and of the casts, close the gap.

## The streaming engine of `collect()`, from polars 2.0

- Input: `left.join(right, on="k", how="full", coalesce=True).collect()` over two lazy frames with no rows, a `UInt8` key on the left and a `UInt64` key on the right, with polars 2.0.0.
- The checker: `k: UInt64`, the type that the in-memory engine gives.
- Polars: from 2.0.0, `collect()` runs the streaming engine by default, and that engine keeps `k: UInt8` over frames with no rows. Over frames with rows, both engines give `UInt64`. `collect(engine="in-memory")` gives `UInt64` in both cases.
- The engines differ in the rows too. In a `select`, `pl.col("x").first().clip(pl.col("y"))` gives one row on the in-memory engine and one row for each row of the frame on the streaming engine. `pl.when(pl.col("x").first() > 1).then(pl.col("y")).otherwise(pl.col("x").head(2))` runs on the in-memory engine where the condition is true, and raises `ShapeError` on the streaming engine.
- The rules follow the in-memory engine. Rules for the streaming engine, chosen by the engine of the call, close the gap.

## `sort_by` in `agg` beside one list of each group, from polars 2.0

- Input: `pl.col("x").head(1).sort_by(pl.col("y").implode())` or `pl.col("x").implode().sort_by(pl.col("y").head(1))` in `group_by().agg()`, over groups of more than one row, with polars 2.0.0.
- The checker: `List(Int64)`, or `List(List(Int64))` for the second call. One value for each group whose values are lists passes, because the data decides the length of each list.
- Polars: the inputs do not share their groups, so `sort_by` reads each input as a list for each group and compares the lengths. The `implode` holds every row of the group, and the `head(1)` holds one row, so the call raises `ComputeError: series lengths don't match in 'sort_by' expression`.
- The length of the list that an `implode` gives is the size of the group. A rule that carries that length for an `implode` in `agg`, and compares it with the rows of the other inputs, closes the gap.

## The offset of `gather_every(1)` on an eager frame, from polars 2.0

- Input: `df.gather_every(1, offset=2**63)` on an eager frame, with polars 2.0.0.
- The checker: `[invalid-call-shape]`, because it reads each count of `gather_every` past $2^{63} - 1$ as a panic of the streaming engine.
- Polars: a step of 1 makes the query a slice, which reads the offset up to $2^{64} - 1$, so the call runs. A step of 2 or more with the same offset panics.
- The checker reads the range of each count alone. A range of the offset that depends on the step, where the call writes the step as a number, closes the gap.
