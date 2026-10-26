# Polars issues

This document lists the polars behaviors that the checker must work around: a call polars types as valid and then refuses, and a data type whose answer polars decides by a rule no statement captures. Each row gives the input, what polars does, and what the checker does about it.

A gap in the checker's own implementation is not here: `KNOWN_ISSUES.md` lists those, and that file exists only while there is one. A weakness the design accepts is not here either: the section "What this gives up" of `ARCHITECTURE.md` lists those.

The behaviors below were measured against polars 1.41.2, 1.42.1, 1.43.2 and 1.44.2. An entry that holds in some of these releases only says which.

## Calls polars types as valid and then refuses

- **A `datetime.timezone` as a zone.** `cs.datetime("us", timezone.utc)` raises `TypeError: 'timezone' object is not an instance of 'str'`. Polars declares the parameter as `str | timezone | Collection[str | timezone | None] | None`, so mypy accepts the call on polars' own signature. The checker reports it as `[invalid-call-shape]`, in the bare form and inside a list.

- **A `List` or an `Array` compared with a Python value.** `pl.col("l") == "x"` over a `List(String)` column raises `InvalidOperationError: cannot cast List type` once the column holds a row. `collect_schema()` gives `Boolean`, and a frame with no row gives `Boolean` too. The same holds for every comparison member, for a `List` and an `Array` of any element, and for every Python value except `None`. The checker reports `[unsupported-dtype-op]`.

- **Two integer literals of different standalone types in `coalesce`, `append` or `is_in`.** `pl.coalesce(pl.lit(1), pl.lit(2**40))` panics with `implementation error, cannot get ref Int32 from Int64`. The panic is a `PanicException`, which does not inherit from `Exception`. `pl.lit(1).append(pl.lit(2**40))` raises `SchemaError`, and `pl.lit(2**40).is_in(pl.lit(1))` panics with `cannot get ref Int64 from Int32`. The cast pass casts each literal to a supertype that is itself an untyped integer, and that cast keeps each literal at the type that it takes alone (`polars-core/src/series/mod.rs:392`). In the forms of `coalesce` and `append`, `collect_schema()` gives the type of the first literal, `Int32`. The checker reports the calls of `coalesce` and `append` as `[invalid-call-shape]`, and the rules in `src/astrolabe/_rules/` refuse all three.

- **`coalesce` of a `Struct` beside a number.** `pl.coalesce(pl.lit(1), pl.col("s"))` over a `Struct({'a': Int8})` column panics with `cannot get ref Int8 from Struct`, and `pl.coalesce(pl.col("s"), pl.lit(1))` raises `SchemaError`. The cast pass casts both to `Int8`, a cast of a struct to a number leaves a struct, and the kernel reads each later input as the type of the first input with no check (`polars-ops/src/series/ops/horizontal.rs:349-365`). `collect_schema()` gives `Int8`. The rules refuse both.

- **`extend_constant` with a categorical value.** `pl.col("x").extend_constant(pl.lit("a", dtype=pl.Enum(["a"])), 1)` panics with `not implemented: can not get dtype of Enum AnyValue` once the column holds a row, and a `Categorical` value does the same. The kernel reads the value as one value and asks for its type (`polars-core/src/series/ops/extend.rs:7`). A null value, an empty input, and an input of one value equal to the value run. `collect_schema()` gives the type of the input. The rules refuse it where the input cannot equal the value.

- **`extend_constant` over an input of no row with a value of another type.** `pl.col("i8").head(0).extend_constant(b"a", 3)` over an `Int8` column keeps the value as it is in a column of the type of the input, with no cast. `select` and `collect_schema()` give `Int8`, and polars panics with `unexpected value while building Series of type Int8` at the first read of the column, a print, a `sum` or a `+ 1`, and in the lists of `agg`. Its builder of a series takes a value of the kind of the type, an integer that fits, a narrower float, a text for a category, a decimal of no larger scale and a list, and refuses any other value. The rules refuse the call in both contexts, because no read of the column runs.

- **`sum_horizontal` over inputs of the type `Null` only.** Where every input has the type `Null`, the kernel returns the first input as it is (`polars-ops/src/series/ops/horizontal.rs:230-237`). So `pl.sum_horizontal(pl.lit(None), pl.col("n"))` over a column `n` of the type `Null` gives one row, and raises `InvalidOperationError` in `agg`. `collect_schema()` gives `Null`. The override `NullsGiveFirstInput` in `src/astrolabe/_rules/definitions/overrides.py` gives the output the rows of the first input, so the checker reports the call in `agg`. In a `select`, the rows of the frame decide whether the call runs beside other columns (`B-2`).

- **The value `2**63` in a chain, cast into `Int64`.** `pl.when(c).then(pl.col("d")).when(e).then(pl.col("u")).otherwise(2**63)` over a `Date` column `d` and a `UInt8` column `u` raises `InvalidOperationError: conversion from u64 to i64 failed` once the frame holds a row. The chain joins `u` with the literal into `UInt64`, and then joins `Date` with `UInt64` into `Int64`. `collect_schema()` gives `Int64`, and a frame with no row gives `Int64` too. A negative literal in place of `d` does the same. The checker gives `Int64`, as it does for a `UInt64` column that holds $2^{63}$ under a cast to `Int64`: the values of a column are not part of its type.

- **A `Struct` in a float function.** `exp`, `log1p` and `log` end the process with a segmentation fault over a `Struct` column once the column holds a row. Before polars 1.44.0, `sqrt`, `cbrt` and `pct_change` do too. From that release, they raise `InvalidOperationError`, because they check that their cast gives `Float64`. `log` does so with any base, and also with a `Struct` column as its base. In polars 1.43, `ewm_mean` and `ewm_sum` over a `Struct` column never return: the kernel calls itself with no cast on any input that is not a float (`polars-ops/src/series/ops/ewm.rs:42`). In polars 1.42, which has no `ewm_sum`, `ewm_mean` casts the input to `Float64` first and calls itself again, so the struct ends the process with a stack overflow. From polars 1.44.0, the cast pass refuses a `Struct` in every exponentially weighted function, with `InvalidOperationError`. `collect_schema()` gives `Float64` for `exp`, `log1p`, `pct_change` and `log`, and the `Struct` itself for `sqrt`, `cbrt` and `ewm_mean`. A frame with no row gives `Float64`. The checker reports `[unsupported-dtype-op]`.

- **`qcut` over text that does not parse, before polars 1.42.0.** `pl.lit("a").qcut([0.5])` panics with `called Option::unwrap() on a None value`. The kernel counts the nulls before its cast to `Float64`. The cast makes every value null, so the kernel finds no value for the breaks. `collect_schema()` gives `Categorical`. From polars 1.42.0, the kernel counts the nulls after the cast and gives a column of nulls. The rules refuse the literal as a panic. They keep the type of a text column, because its values decide whether it panics.

- **`index_of` of a `UInt128` element in a `Decimal` column.** `pl.col("d").index_of(pl.lit(1, dtype=pl.UInt128))` over a `Decimal(10, 2)` column panics in the planner with `called Option::unwrap() on a None value` (`polars-plan/src/plans/conversion/type_coercion/mod.rs:1720`). Every other element that polars cannot cast without loss raises an error. The checker reports `[unsupported-dtype-op]` for each of them.

- **A slice whose offset or length is a column.** `pl.col("a").slice(pl.col("o"), 2)` raises `ComputeError: invalid argument to slice; expected an offset literal` on an eager frame. On a lazy frame, the slice pushdown of `collect()` moves the slice below each place of its value, and the offset reads that same place, so the plan holds a cycle. The query never returns where the value is elementwise, as `pl.col("a").abs()`. Otherwise the slice leaves the tree, and the column comes back whole. `collect_schema()` gives the type of the value. The checker reports the eager call and the hang as `[unsupported-dtype-op]`, and gives the whole column where the slice leaves the tree.

- **A cast of a float to a `Decimal` with `strict=False`.** `pl.Series([1.0]).cast(pl.Decimal(3, 3), strict=False)` raises `ComputeError: decimal precision 3 can't fit values with 4 digits`, where a cast with `strict=False` gives null for every other value that does not fit. The value 0.5 fits, and the cast runs. `collect_schema()` gives `Decimal(3, 3)`. The checker gives `Decimal(3, 3)`, because the error depends on the values.

- **Arithmetic of a `Struct` of one field beside a `Struct` of more fields.** `pl.col("s1") + pl.col("s2")` over `s1: Struct({'a': Int8})` and `s2: Struct({'a': Int8, 'b': Int8})` raises `DuplicateError: multiple fields with name 'a' found` once the frame holds a row. Each field of the output takes the name of the one field of the left side (`polars-core/src/series/arithmetic/borrowed.rs:454-457`). `collect_schema()` gives `Struct({'a': Int8})`. The checker reports `[unsupported-dtype-op]`.

- **Arithmetic of a `List(Boolean)` beside a `List(Null)`.** `pl.col("lb") + pl.col("ln")` panics with `not implemented for dtype Boolean` once the frame holds a row. The leaves join into `Boolean`, and the list kernel has no arithmetic for a `Boolean` or a `Null` leaf (`polars-core/src/series/arithmetic/list.rs:436`). `collect_schema()` gives `List(Boolean)`. The checker reports `[unsupported-dtype-op]`.

- **A list of one row with one null leaf, beside a column of more rows.** From polars 1.43.0, `pl.col("l").first() + pl.col("i")` over a `List(Int64)` column `l` whose first value is `[null]`, and an `Int64` column `i`, panics with `called Option::unwrap() on a None value`. The broadcast rows share one buffer, and the kernel writes into it in place (`polars-core/src/series/arithmetic/list.rs:788`). A first value of `[1]` runs and gives `List(Int64)`, and so does a list of two leaves. `pl.lit` of the same list panics too. The checker gives `List(Int64)`, because the panic depends on the values.

- **`reverse` over an `Array` of values that are not numbers.** `pl.col("a").reverse()` over an `Array(List(Int64), 2)` column panics with `reverse for FixedSizeList with non-numeric dtypes not yet supported` once the column holds two rows. A scalar of one value and a run in `agg` reorder the rows with no kernel, and run. `collect_schema()` gives the `Array` itself. The rules in `src/astrolabe/_rules/` refuse the run.

- **A rolling function with weights over nulls.** `pl.col("x").rolling_mean(2, weights=[1.0, 1.0])` panics with `weights not yet supported on array with null values` where the column holds a null. The same holds for `rolling_min`, `rolling_max`, `rolling_sum`, `rolling_median`, `rolling_quantile`, `rolling_var` and `rolling_std`. A count of weights other than the window size panics too. `collect_schema()` gives the type of the function. The rules refuse a column of only nulls, and a literal text that the cast to a float makes null.

- **`agg_groups` outside a group.** `pl.col("x").agg_groups()` in a `select`, and over a literal in `agg`, panics with `internal error: entered unreachable code`. `collect_schema()` gives `List(UInt32)`. The rules refuse it.

- **`min_by` and `max_by` in `agg` over one value for each group.** `pl.col("x").first().max_by(pl.col("k"))` in `group_by().agg()` reads each input at the rows of the groups, with no check of bounds. An input of one value for each group holds fewer rows than the groups index, so the run reads past its end. Over groups of three and two rows of `Int64`, polars gives `[1, 0]` where the first values are `[1, 4]`, and text, binary data, nested and categorical values can end the process. A key of `sum()` does the same. `collect_schema()` gives the type of the value. The checker reports `[unsupported-dtype-op]` at every type.

- **`sort_by` in `agg` over a list that a node rebuilt, by one value for each group.** `pl.col("x").cum_sum().sort_by(pl.col("y").first())` in `group_by().agg()` panics with `invalid series dtype: expected List`. A function that runs group by group, as `cum_sum`, `shift`, `rank` or `rolling_min`, builds the list of each group anew, and `sort_by` then sorts each list by the list of its key, which a key of one value for each group does not give. A key of one value per row runs. `collect_schema()` gives `List(Int64)`. The checker reports `[unsupported-dtype-op]`.

- **`quantile` with a categorical quantile.** `pl.col("f").quantile(pl.lit("a", dtype=pl.Categorical))` in a `select` panics with `not implemented: can not get dtype of Categorical AnyValue`. `collect_schema()` gives `Float64`. The rules refuse it.

- **`pl.corr(method="spearman")` beside a null literal.** `pl.corr(pl.col("f"), pl.lit(None), method="spearman")` panics with `validity must be equal to the array's length`. `collect_schema()` gives `Float64`. The rules refuse it.

- **`is_empty(ignore_nulls=True)` over a null literal in `agg`.** `pl.lit(None).is_empty(ignore_nulls=True)` in `group_by().agg()` panics with an internal error. `collect_schema()` gives `Boolean`. The override `CountOfNullLiteral` in `src/astrolabe/_rules/definitions/overrides.py` refuses it.

- **`ewm_std` and `ewm_var` over a `Struct`.** Before polars 1.44.0, they end the process with a stack overflow once the column holds a row: the kernel casts the input to `Float64`, a struct stays a struct, and the kernel calls itself again. `collect_schema()` gives the `Struct` itself. The rules refuse it as a crash. From polars 1.44.0, the cast pass refuses the `Struct` first.

- **`rolling_corr` and `rolling_cov` over text beside one row.** `pl.rolling_corr(pl.col("s"), pl.lit(1), window_size=2)` over a `String` column panics with `validity must be equal to the array's length` where the text does not parse as a number: the cast to a float gives nulls, and the kernel of nulls panics beside an input of one row. Text that parses runs. The rules give `Float64`, because the panic depends on the values.

## Answers that the machine decides, not the program

- **`mean_horizontal` beside a `Decimal`.** The answer depends on the number of threads polars runs with, so the same source panics on one machine and answers on another.

  `mean_horizontal` is the one horizontal aggregate that does not cast its operands to their supertype first, so the raw data types reach the kernel and meet in pairs. Three or more operands meet in a balanced binary tree whose depth rayon takes from the thread count. Where a pair is an integer beside a `Decimal`, polars casts both to `Decimal` and then reaches an `unreachable!()` in its own owned-arithmetic fast path, which ends the process. The reverse pair, a `Decimal` beside an integer, takes the borrowed path and answers.

  With five `Int64` operands, one `Decimal` and one `Float64`:

  | operands | 1 thread | 2 threads or more |
  | -- | -- | -- |
  | `Decimal` fourth, `Float64` fifth | panic | `Float64` |
  | `Float64` third, `Decimal` fifth | `Float64` | panic |

  A `Decimal` beside a `Boolean` has no supertype and raises `SchemaError`, in both orders. A `Float64` operand rescues the pair only when the tree makes it absorb the `Decimal` first, which again depends on the thread count.

  `sum_horizontal` adds in the same order. The checker tries each order of the additions that keeps the operands in their places. Where the orders give one type, that is the answer. Where they differ, the answer is `Unchecked`. `TestThreadsDecideTheOrder` in `spec/tests/test_polars_facts.py` measures the first row of the table.

- **`clip` of a literal by a column.** `pl.lit(1).clip(pl.col("a"))` over an `Int64` column of five rows gives one row when the column holds one chunk. When the column holds two chunks, for example after `pl.concat(frames, rechunk=False)` or after `unique()`, polars panics with `expected arrays of the same length`. `collect_schema()` gives `Int32` in both cases. The checker gives `Int32`, the type of the run that answers. It reads the layout of the chunks as a fact of the data, like the values. `TestChunksDecideTheLength` in `spec/tests/test_polars_facts.py` measures this behavior.

- **Arithmetic between two dynamic literals.** A dynamic literal is `pl.lit` of an int or a float with no `dtype=`, or the plain value. Beside a column, polars joins it by the range of its value and not by its own data type. The sum of two such literals takes a different range in different runs.

  Input, over a frame with the `Boolean` column `b`, the group key `g` and the `UInt8` column `u`:

  ```python
  chain = pl.when(pl.col("b")).then(pl.col("u")).otherwise(pl.lit(200) + pl.lit(100))
  ```

  | run | data type |
  | -- | -- |
  | `df.select(chain)` | `UInt16` |
  | `df.lazy().select(chain).collect_schema()` | `UInt8` |
  | `df.lazy().select(chain).collect(optimizations=pl.QueryOptFlags.none())` | `UInt8` |
  | `df.group_by("g").agg(chain)` | `List(UInt8)` |

  An optimized `select` folds the constant to 300 and joins by that value. The other runs join by the range of the left literal, 200. In the `UInt8` runs the value 300 does not fit, and polars wraps it to 44 with no error.

  The rules follow the passes of each run: the fold of the optimizer where the run folds, and the range of the left literal where it does not. So the checker gives `UInt16` in an eager `select`, `List(UInt8)` in `agg`, and `Planned[UInt16, UInt8]` on a lazy frame.

- **A `Boolean` constant beside a `List` or an `Array`.** Polars refuses `|` and `&` between a carrier and a `Boolean` operand. But its optimizer folds a `Boolean` operand that holds a constant, and then it simplifies the operation.

  Input, over the `List(Int16)` column `l`:

  | expression | optimized `select` | unoptimized run and `agg` |
  | -- | -- | -- |
  | `pl.col("l") \| False` | `List(Int16)` | `InvalidOperationError` |
  | `pl.col("l") & True` | `List(Int16)` | `InvalidOperationError` |
  | `pl.col("l").first() \| True` | `Boolean` | `InvalidOperationError` |
  | `pl.col("l").first() & False` | `Boolean` | `InvalidOperationError` |
  | `True \| pl.col("l").first()` | `Boolean` | `SchemaError` |

  An `Array` column does the same. The constant can also be `pl.lit(False)` or an expression that the optimizer folds, for example `pl.col("b").first() & False`. `with_columns` and the streaming engine answer as the optimized `select` does. `collect_schema()` gives the type of the carrier in every form. `^`, a `Boolean` column, and the other sides of `&` raise in every run.

  The rules follow the optimizer of each run, so the checker gives the answer of each row of the table: `List(Int16)` for `pl.col("l") | False` in a `select`, and an error in `agg`. `pl.col("l") | pl.col("b")` is an error, because no run folds a column.

- **`len` over an operand that polars refuses.** An optimized lazy `select` that holds the `len` alone drops the operand and counts the rows. Every other run raises.

  Input, over the `String` column `s`, the `List(String)` column `l` and the `Int64` column `g`:

  | expression | optimized `select` of the `len` alone | other runs |
  | -- | -- | -- |
  | `pl.col("s").abs().len()` | `UInt32` | `InvalidOperationError` |
  | `pl.col("l").diff().len()` | `UInt32` | `InvalidOperationError` |
  | `pl.col("s").abs().sum().len()` | `InvalidOperationError` | `InvalidOperationError` |
  | `pl.col("s").abs().len() + 1` | `InvalidOperationError` | `InvalidOperationError` |

  The other runs are an eager `select`, an unoptimized run, `agg`, and an optimized `select` beside the column `g`. The operand of the first two rows keeps one value for each row. The operand of the third row combines the rows.

  The rules follow the projection pushdown, so the checker gives `UInt32` in a lazy `select` of the `len` alone, and reports the error in every other run.

## Declared data types that polars does not keep

The rules give the run type, which `collect()` gives, beside the planner type, which `collect_schema()` gives. An eager frame holds the run type. A lazy frame holds both, as `Planned[run, planner]`, and `collect()` gives the run type. The checker answers below are those of an eager frame.

- **`mean_horizontal` over one `Decimal` column.** `collect_schema()` gives `Float64` and `collect()` gives `Decimal(10, 2)`. One operand short-circuits before the cast and before the division. The checker gives the `Decimal`.

- **Four forms beside a dynamic literal.** Over the `UInt8` column `u` and the `Float32` column `f`, `collect_schema()` and `collect()` give different data types:

  | expression | `collect_schema()` | `collect()` |
  | -- | -- | -- |
  | `pl.concat_list(pl.col("u"), pl.lit(300))` | `List(UInt16)` | `List(Int32)` |
  | `pl.reduce(operator.add, [pl.col("u"), pl.lit(300)])` | `UInt16` | `Int32` |
  | `pl.col("u").shift(1, fill_value=300)` | `UInt8` | `UInt16` |
  | `pl.mean_horizontal(pl.col("f"), pl.lit(1))` | `Float32` | `Float64` |

  A chain of a `Date` column and `pl.lit(1)` gives `Unknown` in `collect_schema()` and `Int32` in `collect()`. The checker gives the `collect()` data type in each row.

- **`shift(fill_value=)` with a fill of another kind.** Over the `Int8` column `i` and the `Int16` column `j`, `pl.col("i").shift(1, fill_value=pl.col("j").first())` gives `Int8` in `collect_schema()` and `Int16` in `collect()`. `collect_schema()` gives the type of the receiver at every pair of kinds. `collect()` gives the supertype of the two kinds where they have one. The checker gives the `collect()` data type.

- **`mean_horizontal` over a `Float32` beside two `Boolean` columns.** `collect_schema()` gives `Float32` and `collect()` gives `Float64`. Two `Boolean` columns that meet as a pair sum to `UInt32`, and `Float32` beside `UInt32` widens to `Float64`, while the declared type folds the operands one at a time and keeps `Float32`. The order counts: `pl.mean_horizontal(f32, b)` and `pl.mean_horizontal(b, f32, b)` give `Float32` both ways. The checker gives `Float64` for the first shape and `Float32` for the other two.

- **`map_batches` with both flags inside `agg`.** Over the `Int64` column `a`, `pl.col("a").map_batches(f, return_dtype=pl.Int64, is_elementwise=True, returns_scalar=True)` inside `group_by("k").agg(...)` gives `Int64` in `collect_schema()` and `List(Int64)` in `collect()`. With `is_elementwise=True`, the run ignores `returns_scalar=True` and keeps the character of the operand. The checker gives `List(Int64)`.

- **A full join with `coalesce=True` on keys of two integer types.** Over a `UInt8` key on the left and a `UInt64` key on the right, `left.join(right, on="k", how="full", coalesce=True)` gives `k: UInt8` in `collect_schema()` and `k: UInt64` in `collect()` (`polars-plan/src/plans/conversion/dsl_to_ir/join.rs:262-317`). The other values of `how` give the same type both ways. The checker gives `UInt64`.

- **A cast of a flat column to a `List` of a logical type.** `pl.col("i").cast(pl.List(pl.Date))` over an `Int8` column gives `List(Date)` in `collect_schema()` and `List(Int32)` in `collect()`. The cast wraps each value at the physical type of the target (`polars-core/src/chunked_array/cast.rs:75-122`). The checker gives `List(Int32)`.

- **A slice of one list beside an offset or a length of several rows.** Over a `Date` column `d` and an `Int32` column `i`, `pl.col("d").implode().list.slice(pl.col("i"), 1)` gives `List(Date)` in `collect_schema()` and `List(Int32)` in `collect()`. The kernel collects the slices of the one list into a new list, which holds the physical type of the values, and `list.head` and `list.tail` slice the same way. In `agg`, the list of each group then has another type than the plan, and polars raises `ComputeError`. Over a frame of one row, or a group of one row, the offset has one row too, and the slice keeps `List(Date)`. So the number of rows decides the type, and the checker gives `Unchecked` in both contexts.

- **A cast of a `Struct` to a type that is not a `Struct` or text.** `pl.col("s").cast(pl.Int64)` over a `Struct({'a': Int64})` column gives `Int64` in `collect_schema()` and `Struct({'a': Int64})` in `collect()`. The cast casts each field to the target and keeps the struct (`polars-core/src/chunked_array/struct_/mod.rs:303-319`). A frame with no row gives the target type, `Int64`. The checker gives the `collect()` data type of a frame with rows. A cast of a column whose data type it does not know gives the target, because the checker takes the kept struct as an oddity of polars.

## Values that polars changes without an error

- **A dynamic literal whose value leaves its range.** A dynamic literal keeps the range of its first value through a member that changes the value, for example negation. Polars then writes a value that the joined data type cannot hold, and it reports no error.

  Over the `UInt8` column `u` with the values 1 and 2, `-pl.lit(1)` keeps the range of 1:

  | expression | data type | values |
  | -- | -- | -- |
  | `pl.when(pl.col("b")).then(pl.col("u")).otherwise(-pl.lit(1))` | `UInt8` | 1, 255 |
  | `pl.col("u") + -pl.lit(1)` | `UInt8` | null, null |

  The chain wraps -1 to 255, and the addition writes null. The checker gives `UInt8`, the data type that polars gives. A data type cannot tell a value that fits from one that does not, so the checker reports no error.

- **A shift fill that the receiver cannot hold.** Where the receiver and the fill of `shift(fill_value=)` have no supertype, polars casts the fill to the type of the receiver, and it reports no error. Over the `Int8` column `i`, the `Date` column `d` and the `Time` column `t`:

  | expression | data type | value of the fill |
  | -- | -- | -- |
  | `pl.col("i").shift(1, fill_value=pl.col("d").first())` | `Int8` | null for the date 2020-01-01 |
  | `pl.col("t").shift(1, fill_value=pl.col("d").first())` | `Time` | 00:00:00.000018262 for the same date |

  The cast reads the date as its number of days. The checker gives the data type that polars gives.

- **A strict cast of a flat value to a `List`.** `pl.Series(["a", "1"]).cast(pl.List(pl.Int64), strict=True)` gives `[[null], [1]]`. The strict cast checks the nulls of the outer list only, so a value that does not parse becomes an inner null with no error. The checker gives `List(Int64)`, the data type that polars gives.

- **An integer cast to categories inside a list or an array.** A cast of a list to an array, or of an array to a list, runs through the arrow kernel. That kernel casts an integer to the physical id of a category, and it does not check that the process holds the id. In a new process, `pl.Series([[5]], dtype=pl.Array(pl.Int64, 1)).cast(pl.List(pl.Categorical))` gives `List(Categorical)` with no error. A later read of the values, for example `to_list()`, then crashes the process. An `Enum` reads an index out of its range as the empty text. The checker gives the data type that polars gives.

- **`rolling_corr` and `rolling_cov` beside one row.** Where one input holds a null and the other input has one row, which polars spreads, polars panics with `validity must be equal to the array's length`: `pl.rolling_corr(pl.col("a"), pl.col("x").first(), window_size=2)` over a column `a` that holds a null. The same call runs where `a` holds no null. The nulls of the data decide the panic, so the checker gives `Float64`, the type of the run that does not panic. Where the code decides a null, as in the first row of `shift(1)` or of `rolling_mean(2)`, the checker reports the panic.

- **`sort` over one value for each group in `agg`.** `df.group_by("g").agg(pl.col("l").first().sort())` over a `List(Int64)` column panics with `capacity overflow`, and the process can then allocate memory with no bound. The sort reads the flat values of the input with the indices of the rows of each group, with no check of the bounds (`polars-expr/src/expressions/sort.rs:70-85`), and an aggregation holds one value for each group only. So the read is out of bounds. A list, an array of lists and a struct with a text field crash. A text and a `Categorical` crash in some processes, depending on the memory of the process. A number reads values that are not those of the column. The rules give the type of the input, and `spec/tests/test_definitions_node_kinds.py` leaves these cells out.

- **A `rolling` expression over an index of one value, in the `agg` of `rolling`.** `pl.col("x").sum().rolling(index_column=pl.col("x").first(), period="2i")` panics with `index out of bounds` in the `agg` of `group_by`, and with `range end index 2 out of range for slice of length 1` in the `agg` of `group_by_dynamic`. In the `agg` of `rolling`, the same call runs, and its windows read the rows of other windows: over `x = [10, 20, 30, 40, 50]` and the windows of `rolling("i", period="2i")`, the third window holds the values `20` and `30`, and gives `[30, 30]`, the sum of `10` and `20`. In that context, polars also accepts `period="2d"` over the integer index, which `select` refuses. `collect_schema()` gives `List(Int64)`. The rules refuse the call in each `agg`, as polars does in `group_by`.
