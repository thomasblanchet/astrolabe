# The polars operation catalog

This catalog states what every polars operation does under the rules of `OPERATIONS.md`. It covers the frame API of `DataFrame` and `LazyFrame`, `GroupBy`, the selectors, and the whole expression API. Read it after `OPERATIONS.md`, which follows `DATAFRAME_SPECIFICATION.md` and `EXPRESSION_SPECIFICATION.md`. `OPERATIONS.md` is authoritative: if a row here disagrees with it, the row is wrong.

## 0. How to read this

### 0.1 What this document is

`DATAFRAME_SPECIFICATION.md` and `EXPRESSION_SPECIFICATION.md` say what a typed frame and a typed expression are, and why. `OPERATIONS.md` gives the formal definitions, rules and proofs for both halves. This document applies those rules to the library, one operation at a time, and it is complete: every public method of `DataFrame` and `LazyFrame`, every method of `GroupBy`, every selector, every member of `pl.Expr` and its namespaces, and the expression-building and frame-level functions of `pl` appear somewhere below.

It describes the type system, not an implementation of it. Implementation status is outside the specification, so nothing here goes stale when code changes.

### 0.2 The columns of a Part I table

| Column | Content |
| -- | -- |
| **Call** | the variant, spelled as a user writes it |
| **Polars** | what the library does to the columns, measured |
| **Schema out** | the declared columns of the result, as a function of the input's |
| **Remainder out** | the effect on the remainder, as an `OPERATIONS.md` identifier |
| **Needs** | the precondition: a requirement identifier, a boundary identifier, or a short note; `none` if the row needs no promise |

The tables of 5.2 and 7.2 leave out the **Polars** column, because the text before each of them states what polars does.

A **boundary** row follows a table wherever an operation can fail for a reason no schema can see. Those are `B-1` … `B-6` of `OPERATIONS.md`, and they are never counted against the type.

### 0.3 The vocabulary

Every identifier is defined in `OPERATIONS.md`. The ones used most:

| Id | Effect on the remainder |
| -- | -- |
| `E-1` | unchanged |
| `E-2` | emptied: the result has no remainder |
| `E-3` | filtered by a selector; spelled `Minus`, `Only` or `Except` |
| `E-4` | the two operands' remainders together, spelled `Both` |
| `E-5a` / `E-5b` | every name gains a suffix, or a prefix |
| `E-6` | every type replaced, spelled `Retyped` |

| Id | Requirement |
| -- | -- |
| `R-1` | none |
| `R-2` … `R-9` | the eight per-column promises |
| `R-10`, `R-11` | `disjoint_from=`, `within=` |

**Every name an operation reads must be declared.** That obligation is universal and is not repeated in every row. A remainder never justifies a name reference, because the name can be absent for some caller. Where a row says `none` under **Needs**, it means no promise about the *remainder* is required beyond that.

### 0.4 How the Polars column was measured

Every behavior stated here was executed on **polars 1.41.2**, **polars 1.42.1**, **polars 1.43.2** and **polars 1.44.2**, and is asserted by a test under `spec/tests/`: the frame facts in `test_polars_facts.py`, the expression facts in `test_expression_facts.py`, and the rules that type Part III in `test_rules_*.py` and `test_definitions_*.py`. Run them with:

```
uv run pytest spec
```

A polars release that changes one of these behaviors fails the test that names it, so this document cannot quietly drift away from the library it describes. Where a behavior differs between releases, the row says which releases show it, and the rules apply the behavior of the installed release.

### 0.5 Scope, and why nothing is missing

In polars 1.44.2, the frame API is 152 methods across `DataFrame` and `LazyFrame`. Polars 1.42.0 adds `DataFrame.is_sorted`, so the frame API of polars 1.41 has 151 methods. Every one of them appears below, in Part I if it can change a schema and in Part II with a reason if it cannot, so a reader can look up any polars method and find it.

| Group | Count | Where |
| -- | -- | -- |
| frame methods that shape a schema | 75 | Part I |
| frame methods that cannot | 80 | Part II |
| named in both, having two meanings | 3 | `clear`, `n_unique`, `sql` |
| `GroupBy` and `LazyGroupBy` methods | 17 | Part I, section 7 |
| selector constructors, `cs.*` | 36 | Part I, section 9 |
| frame-level `pl.*` functions | 41 | 4 in Part I, 37 in Part II |
| `pl.Expr` methods | 212 | Part III, section 19 |
| `pl.Expr` namespace members | 223 | Part III, sections 20 and 21 |
| expression constructors, `pl.*` | 82 | Part III, section 18 |

The first three rows give $75 + 80 - 3 = 152$, so the partition is exact, and a test checks it. And Part I's content is replayed: `spec/tests/test_catalog_rows.py` executes every Part I operation over a frame holding declared columns and remainder stand-ins and asserts the full output schema, so a row whose stated behavior drifts from the library fails by name (`TestEveryRowReplays` in `spec/tests/test_catalog_rows.py`), and an operation the replay misses fails the coverage gate (`TestTheReplayCoversPartOne` in the same file). Three names appear twice. `n_unique` returns a number on a frame and a frame under a group-by. `clear` keeps a schema while emptying rows. `sql` is the method `DataFrame.sql` in Part I and the function `pl.sql` in Part II, two members that share a name.

`LazyFrame` differs from `DataFrame` in *when* work happens, not in what the columns become. Every Part I row holds for both unless it says otherwise.

______________________________________________________________________

# Part I. Operations that shape a schema

These operations change the columns of a frame, or can. Each row states what one call does to the declared columns and to the remainder, and what it needs.

______________________________________________________________________

## 1. Row-only operations

A row-only operation changes which rows survive, or in what order. It never changes the column mapping, so the schema and the remainder both pass through.

| Call | Polars | Schema out | Remainder out | Needs |
| -- | -- | -- | -- | -- |
| `.filter(pred)` | keeps every column; selects rows | unchanged | `E-1` | `none` |
| `.remove(pred)` | the complement of `.filter` | unchanged | `E-1` | `none` |
| `.sort(by, ...)` | reorders rows | unchanged | `E-1` | `none` |
| `.unique(subset=...)` | drops repeated rows | unchanged | `E-1` | `none` |
| `.drop_nulls(subset=...)` | drops rows holding a null | unchanged | `E-1` | `none` |
| `.drop_nans(subset=...)` | drops rows holding a NaN | unchanged | `E-1` | `none` |
| `.head(n)`, `.tail(n)`, `.limit(n)` | takes `n` rows | unchanged | `E-1` | `none` |
| `.first()`, `.last()` | takes one row | unchanged | `E-1` | `none` |
| `.slice(offset, length)` | takes a window of rows | unchanged | `E-1` | `none` |
| `.reverse()` | reverses the row order | unchanged | `E-1` | `none` |
| `.sample(n=..., fraction=...)` | takes rows at random | unchanged | `E-1` | `none` |
| `.gather(indices)` | takes rows by position | unchanged | `E-1` | `none` |
| `.gather_every(n, offset=0)` | takes every `n`th row | unchanged | `E-1` | `none` |
| `.top_k(k, by=...)`, `.bottom_k(k, by=...)` | takes the extreme rows | unchanged | `E-1` | `none` |
| `.shift(n)` | moves values between rows | unchanged | `E-1` | `none` |
| `.clear(n=0)` | keeps the schema, drops every row | unchanged | `E-1` | `none` |
| `.clone()` | copies the frame | unchanged | `E-1` | `none` |
| `.upsample(time_column=..., every=...)` | inserts rows on a time grid | unchanged | `E-1` | `none` |
| `.set_sorted(column)` | records a sort order as metadata | unchanged | `E-1` | `none` |

**Four obligations these carry beyond naming a declared column.** Each is statically knowable, and none is a name check.

| Obligation | Why |
| -- | -- |
| `.filter` and `.remove` need a **Boolean** predicate | `filter(pl.col("price"))` on a `Float64` raises `InvalidOperationError` at schema time, not from data |
| `.unique(subset=[])` is always an error | an empty key list raises `ComputeError` |
| `.sample(fraction=f)` needs $f \le 1$ without replacement | otherwise `ShapeError`, the same one `n` too large gives |
| a count of rows fits the integer that polars reads | polars converts `n`, `k`, `offset` and `length` to an integer of a fixed width before it reads any data, so `.head(2**64)` raises `OverflowError` and a lazy `.head(-1)` raises `ValueError` |

The width depends on the method and on the kind of frame. An eager frame reads the count of `.head`, `.tail` and `.limit`, and the `length` of `.slice`, as an unsigned 64-bit integer, and takes a negative count as a count from the end. A lazy frame reads them as the index type, from 0 to $2^{32} - 1$, and refuses a negative count. Both read the `offset` of `.slice` as a signed 64-bit integer, the counts of `.gather_every` and of the `head` and `tail` of a group-by as an unsigned 64-bit integer, a step `n` of 0 included, which raises `InvalidOperationError` when the query runs, and the `k` of `.top_k` and `.bottom_k` and the `offset` of `.with_row_index` as the index type. The checker reports a count outside its range where the call states it (measured by `TestRowCountWidths` in `spec/tests/test_polars_facts.py`).

> `B-1` boundaries here: `.upsample` needs a sorted time column, and every row-count argument (`n`, `k`, `fraction`) can exceed the frame's height. Both are outside the model.

______________________________________________________________________

## 2. Column shaping

These change which columns exist, or what they are called. They are where most of the grammar is exercised.

| Call | Polars | Schema out | Remainder out | Needs |
| -- | -- | -- | -- | -- |
| `.select(*names)` | keeps exactly the named columns | the named columns | `E-2` | `none` |
| `.select_seq(*names)` | the same, evaluated in order | the named columns | `E-2` | `none` |
| `.select(<name selector>)` | keeps what the selector picks | the picked declared columns | `E-3`, `Only` | `none` |
| `.select(<type selector>)` | keeps what the selector picks | the picked declared columns | `E-3`, `Only` | `none`; settled by `R-8`, `R-9` or a split (9.2) |
| `.select(pl.all().name.prefix(p))` | renames every column | every name prefixed | `E-5b` | `R-4` on `p` |
| `.select(pl.all().name.suffix(s))` | renames every column | every name suffixed | `E-5a` | `R-6` on `s` |
| `.with_columns(*exprs, **named)` | adds or replaces columns | plus the produced columns | `E-3`, `Minus` of the produced names | `none` |
| `.with_columns_seq(...)` | the same, evaluated in order | the same | the same | `none` |
| `.drop(*names, strict=True)` | removes the named columns | minus those names | `E-1` | `none` |
| `.drop(<selector>)` | removes what the selector picks | minus the picked columns | `E-3`, `Except` | as for `.select` |
| `.rename(mapping)` | renames the named columns | renamed | `E-1` | `R-2` on each destination |
| `.rename(callable)` | renames every column by a function | see 2.4 | see 2.4 | `B-6` |
| `.with_row_index(name="index")` | adds a counter at the front | plus `name : UInt32` | `E-3`, `Minus[name]` | `R-2` on `name` |
| `.with_row_count(name="row_nr")` | the deprecated spelling, different default name | plus `name : UInt32` | `E-3`, `Minus[name]` | `R-2` on `name` |
| `.insert_column(index, series)` | adds a column at a position | plus that column | `E-3`, `Minus` of its name | `R-2` on its name |
| `.replace_column(index, series)` | replaces the column at a position, name included | minus the old name, plus the series' name | `E-3`, `Minus` of the new name | `B-4` |
| `.map_columns(names, f)` | applies a function to whole columns, keeping names | those columns retyped to `Unchecked` | `E-1` | `none` |
| `.match_to_schema(schema)` | reorders and casts to a given schema | that schema | `E-2` | `none` |
| `.drop_in_place(name)` | removes one column and returns it | minus that name | `E-1` | `none` |

### 2.1 Why `.select` of a fixed list empties the remainder

The caller passed fifty columns and two come back. That is what `select` means, and the result is exact for every caller, which is why the row needs no promise.

### 2.2 Why `.with_columns` removes a name rather than adding one

Adding `notional` to a frame that already had a `notional` replaces it. Afterwards the schema holds `notional` and the remainder provably does not, or the column would be counted twice.

This is where `E-3` manufactures a guarantee. `Minus` adds the name to the absence rule unconditionally, whatever the input rule was, so a caller who promised nothing gets a result that satisfies a function demanding the name be absent.

A selection with an affix (`cs.numeric().name.suffix("_r")`) produces names the checker does not know one by one. Each one replaces a column of the same name, so the result is exact only when no produced name can be a remainder column or a declared column. The remainder's rule must exclude the affixed image (a remainder declared `without_suffix="_r"` does), and no declared name can lie in it. When either is unproven, the result depends on the caller, and the operation falls back with that reason.

### 2.3 Every collision these can produce

Measured, and each is a `DuplicateError` or a `ComputeError` rather than a silent replacement:

| Call | Raises when |
| -- | -- |
| `.rename({"a": "b"})` | `b` already exists |
| `.with_row_index("idx")` | `idx` already exists |
| `.select("a", "a")` | one name is projected twice |
| `.with_columns(z=..., **{"z": ...})` | one call produces `z` twice |

`.with_columns(a=...)` where `a` already exists is the exception: it **replaces** and does not raise.

> Measured by `TestNameCollisions` in `spec/tests/test_polars_facts.py`.

### 2.4 `.rename(callable)` cannot be typed in general

A callable is arbitrary code, so its effect on names is not readable from the signature. Two cases:

| The callable | Result |
| -- | -- |
| a constant affix, as `lambda n: "s_" + n` | not recognized as an affix, so the schema is unknown |
| anything else | the schema is unknown |

Nothing in the grammar reads a Python function body, so this call is `B-6` and is rejected over a remainder rather than guessed at. The mapping form is the spelling that can be typed, and `pl.all().name.prefix(...)` is the spelling for a whole-frame affix, which `E-5b` types exactly.

______________________________________________________________________

### 2.5 What `.drop` takes

`.drop` takes literal names, a column, a declared column and a selector, alone or in a list, and polars expands each to names before it drops them. `strict=` decides only for a literal name: a column, as `pl.col("x")`, and a name that a selector lists, as `cs.by_name("x")`, raise `ColumnNotFoundError` for a missing column whatever `strict=` says. Any other expression, as `pl.col("a") + 1`, `pl.col("a").alias("b")` or `pl.lit(1)`, raises `InvalidOperationError` at every frame, because no name follows from it, so the checker reports it. Where the data type of a column decides whether a selector picks it, the columns of the result depend on the data, as for `.select` (`D-44` of `OPERATIONS.md`). Measured by `TestDropTakesNamesAndSelectors` in `spec/tests/test_polars_facts.py`.

## 3. Reshaping

These change the *shape* of the frame, so several of them are where the model reaches a genuine boundary.

| Call | Polars | Schema out | Remainder out | Needs |
| -- | -- | -- | -- | -- |
| `.explode(*names)` | keeps every name; drops one nesting level from the named columns | those columns' inner type | `E-1` | `none` |
| `.unnest(*names)` | replaces a struct **in place** with its fields | minus the struct, plus its fields | `E-1` | `R-2` on each field name |
| `.unnest()` with no argument | unnests every struct column | the same, for every declared struct | `E-1` | `R-9` on `pl.Struct` |
| `.unpivot(on=..., index=...)` | turns every non-index column into rows | the index columns, plus `variable : String` and `value` | `E-2` | `none` |
| `.melt(...)` | the deprecated spelling of `.unpivot` | as `.unpivot` | `E-2` | `none` |
| `.pivot(on=..., index=..., values=...)` | output names come from **data** | the index columns, plus unknown names | `E-2` | `B-5` |
| `.pivot(on=..., values=...)`, no `index=` | uses **every remaining column** as the index | every other column, plus unknown names | `E-1` | `B-5` |
| `.pivot(on=..., on_columns=[...], index=..., values=...)` | output names come from the literal `on_columns` | the index columns, plus one column per listed value, typed by the aggregate function | `E-2` | `none` |
| `.pivot(on=..., on_columns=[...], values=...)`, no `index=` | uses every remaining column as the index | every other column, plus one column per listed value | `E-1` | `R-2` on each produced name |
| `.transpose(...)` | one output column per input **row** | not expressible | `E-2` | `B-2` |
| `.unstack(step=..., how=...)` | output names come from **positions** | not expressible | `E-2` | `B-2` |
| `.to_dummies(columns=...)` | output names come from **data** | not expressible | `E-2` | `B-5` |

### 3.1 `.pivot` without `index=` keeps columns

Omitting `index=` makes polars use every remaining column as the index, so a column outside the schema **survives**:

```
pivot(on="region", values="amount")          -> account, venue, EMEA, APAC
pivot(on="region", index="account", ...)     -> account, EMEA, APAC
```

The effect is therefore `E-1` when `index=` is omitted and `E-2` when it is given, which is the opposite of what a first reading suggests. The omitted form additionally needs the remainder to hold none of the produced names, and those names come from the data, so `B-5` applies in both directions.

> Measured by `TestReshape` in `spec/tests/test_polars_facts.py`.

A literal `on_columns=` names the output instead of the data. The listed values are exactly the new columns: a value with no data still gets a column, and a value not listed is dropped. With one values column each new column takes its value's name, and with several it takes `{values}{separator}{value}`. The aggregate function sets the data type, so `B-5` does not apply to this form. An expression in the list of `on_columns=` names no column: polars builds a series of the values, and raises `ComputeError` or `TypeError` at every frame (`TestReshape`). The checker reports it as `invalid-call-shape`.

### 3.2 `.unnest()` with no argument cannot use the remainder

Auto-detection would have to find every struct column, and the remainder can hold a struct the checker cannot see. So the no-argument form needs `none_of=pl.Struct` on the remainder (`R-9`), or it is rejected.

Narrowing on the declared structs alone would be **worse than rejecting**: it would produce a confident result that misses a struct in the remainder, which is the unsound direction.

### 3.3 Where the fields of an unnest land

The struct's fields take the struct's position, not the end of the frame. Position does not matter to the type (`D-5` makes a column set unordered), but name collision does: a field whose name is already taken raises `DuplicateError`, which is why the row demands `R-2` on each field name.

______________________________________________________________________

## 4. Data types

These keep every name and change types.

| Call | Polars | Schema out | Remainder out | Needs |
| -- | -- | -- | -- | -- |
| `.cast(dtype)` | casts every column to one type | every type replaced | `E-6` | `R-9` on `{pl.List, pl.Array}` |
| `.cast(mapping)` | casts the named columns | those types replaced | `E-1` | `none` |
| `.cast(<selector>: dtype)` | casts what the selector picks | as for `.select` of that selector | `E-6` on the picked part | as for `.select` |
| `.fill_null(value)`, an `int` or a `float` | fills the columns of the number kinds, `Int8` to `Int128`, `UInt8` to `UInt64`, `Float32`, `Float64` and `Decimal`, and every `Null` column. `UInt128` and `Float16` are not filled. With `matches_supertype=False`, it fills only `Int64` for an int, or `Float64` for a float, and `Null` | as for `.with_columns` of `pl.col(<the filled kinds>).fill_null(value)` | as for that `.with_columns`: each filled kind of the remainder retyped kind by kind (`T-20`) | `none` |
| `.fill_null(value)`, a `bool`, `str`, `date` or `time` | fills the columns of the value's own kinds, `Boolean`, `String` and `Categorical`, `Date`, or `Time`, and every `Null` column | as for `.with_columns` of `pl.col(<the filled kinds>).fill_null(value)`: unchanged, but a `Null` column takes the value's type | as for that `.with_columns`: the `Null` part retyped to the value's type | `none` |
| `.fill_null(value)`, a `timedelta` | fills every `Duration` column and every `Null` column | as for `.with_columns` of `(cs.duration() \| cs.by_dtype(pl.Null)).fill_null(value)`: a `Duration` takes the coarser of its unit and `us`, and `Null` gives `Duration('us')` | as for that `.with_columns` | `none` |
| `.fill_null(value)`, a `datetime` of zone $z$ | fills every `Datetime` column and every `Null` column, and raises on a `Datetime` of another zone | as for `.with_columns` of `(cs.datetime() \| cs.by_dtype(pl.Null)).fill_null(value)`: a `Datetime` of zone $z$ takes the coarser of its unit and `us`, and `Null` gives `Datetime('us', z)` | as for that `.with_columns` | `R-9` on `Datetime` |
| `.fill_null(value)`, an expression, a `Decimal` or `bytes` | fills every column, as `.select(pl.all().fill_null(value))` | as for that `.select` | as for that `.select` | `R-9` on each kind where the fill raises |
| `.fill_null(strategy="forward"/"backward")` | carries a value along the column | unchanged | `E-1` | `none` |
| `.fill_null(strategy="min"/"max")` | takes a value from the column | unchanged | `E-1` | `R-9` on `NestedType` |
| `.fill_null(strategy="zero")` | fills with zero | unchanged | `E-1` | `R-9` on `{NestedType, Null}`, and from polars 1.44.0 on `Categorical` and `Enum` too |
| `.fill_null(strategy="one")` | fills with one | unchanged | `E-1` | `R-9` on `{NestedType, String, Binary, Null}`, and from polars 1.44.0 on `Categorical` and `Enum` too |
| `.fill_null(strategy="mean")` | fills with the mean | unchanged | `E-1` | `R-9` on `{NestedType, String, Boolean, Binary}`, and from polars 1.44.0 on `Categorical` and `Enum` too |
| `.fill_nan(value)` | replaces NaN in the `Float16`, `Float32` and `Float64` columns, as `.with_columns(cs.float().fill_nan(value))`, with a value that is not an expression read as `pl.lit(value)` | as for that `.with_columns`: a number keeps each float column, and a `str` gives `String`. Every other column is unchanged | as for that `.with_columns`: each float kind of the remainder retyped kind by kind (`T-20`) | `R-9` on each float kind where the fill raises |
| `.interpolate()` | fills nulls between values; integer columns become `Float64` | integers to `Float64`, the rest unchanged | `E-6` to `Unchecked` | `none` |

### 4.1 A cast's domain is not a family, and is still expressible

Whole-frame `.cast(dtype)` raises for some source types and not others, and the accepted set is **not** a polars family in either direction. `Struct` casts to text. `List` and `Array` raise. So a gate written as `none_of=NestedType` would be wrong twice over: it excludes the one nested type that works and admits neither of the two that fail.

Naming the two types that fail is right, and it is writable because `R-9` takes any set of data types and not only a family:

```python
Castable = other_columns(none_of={pl.List, pl.Array})
```

This is the case that decides the shape of the vocabulary. Had `none_of=` accepted only a family, this operation would have had to be rejected over a remainder for want of a way to write its precondition, and that would have been a limit of the notation rather than of the model.

> Measured by `test_a_whole_frame_cast_rejects_two_of_the_three_nested_types` in `spec/tests/test_polars_facts.py`.

### 4.2 `.fill_null` splits by strategy, and by the value's type

Measured: `forward` and `backward` are total across every data type. `min`, `max`, `zero`, `one` and `mean` each raise on some type, once the column holds a null. So the two groups are different rows, and only the first needs no promise. `zero` and `one` also raise on a `Null` column, which has neither. No strategy changes a data type. A column of a refused type that holds no null keeps its type, so the refusal needs a row with a null. The checker reports a declared column of a refused type all the same, as it reports a `List` compared with a value, which polars refuses only once a row exists.

`.fill_null(value)` picks the filled columns from the Python type of the value, and runs `fill_null` as an expression on them alone. A column that the value does not pick keeps its type and its nulls, so the call is total for every kind it does not pick. The row of each value states the picked kinds. Every value but an expression also picks `Null`, and a `Null` column takes the type of the value.

- **A number** picks the number kinds of its row. `UInt128` and `Float16` are numeric, but polars does not list them, so a number never fills them: `.fill_null(1.5)` keeps `u : UInt128`, where a fill would give `Float64`. On each picked column the fill is the expression form `.fill_null(value)`, so the value is a literal beside the column (`D-34`): on `i : Int64`, `.fill_null(1.5)` gives `i : Float64`, on `u : UInt32`, `.fill_null(-1)` gives `u : Int64`, and on `u : UInt8`, `.fill_null(1)` keeps `u : UInt8`. A `Null` column takes the standalone type of the value. An int widens a `Decimal` to 38 digits and keeps its scale, and a float gives `Float64`. `matches_supertype=False` narrows the picked kinds to `Int64` for an int, or `Float64` for a float. A variable of the type `int` or `float` is a hole (`D-44`): a column keeps the answer that every value of the sample gives, and is `Unchecked` where they differ.
- **A `bool`, `str`, `date` or `time`** picks its own kinds, and each picked column keeps its type. A `str` picks `Categorical` but not `Enum`.
- **A `timedelta`** is a `Duration('us')`, and **a `datetime`** a `Datetime('us', z)` for its zone $z$. A picked column takes the coarser of its unit and `us`: `Duration('ns')` gives `Duration('us')`, and `Duration('ms')` stays. A `Datetime` of another zone than the value raises `SchemaError`.
- **An expression, `pl.lit(v)` included, a `Decimal` or `bytes`** picks every column, through `.select(pl.all().fill_null(value))`. So `pl.lit(1)` is not the plain `1` here: it fills a `Boolean` column, which gives `Int32` beside the literal.

Measured by `TestFrameFill` in `spec/tests/test_polars_facts.py`, the strategies there and by `TestOperationDomains`.

The checker records the fill as that expression and evaluates it at `.with_columns` or `.select` (section 17.2 of `OPERATIONS.md`), so the remainder answers kind by kind, as `T-20` states. Each picked kind $k$ of the remainder takes the answer that the samples of $k$ share (`D-44`): a `Decimal` beside an int gives `Decimal` with open parameters, and a `Duration` beside a `timedelta` gives `Duration` with an open unit. Where a kind of the remainder can raise, a `Datetime` of a zone that the remainder does not fix beside a `datetime`, or a refused kind beside a strategy, the result depends on the caller: the row needs `R-9` on those kinds, or the checker falls back with the reason `CALLER_DEPENDENT`. The same holds for the expression form over a selection, `cs.datetime().fill_null(value)` or `cs.datetime() > value`, where some data types of a picked kind refuse (`T-20`).

`.fill_null()` with neither a value nor a strategy, a value together with a strategy, and a `limit` with a strategy other than `forward` or `backward`, a value included, raise `ValueError` at the call, whatever the data.

______________________________________________________________________

## 5. Joins

Joining is the operation this system spends the most rules on, because it is the only one whose output **names** can depend on the caller.

### 5.1 What polars does when two names collide

The left frame keeps every name. A right column whose name collides takes a suffix, `_right` by default. This holds for every join type, including `how="right"`, where only the column order changes.

| Call | Result columns |
| -- | -- |
| `left.join(right, on="symbol")` | `symbol, price, sector, sector_right, country` |
| `left.join(right, on="symbol", how="right")` | `price, sector, symbol, sector_right, country` |
| `left.join(right, on="symbol", how="full")` | `symbol, price, sector, symbol_right, sector_right, country` |
| `left.join(right, on="symbol", suffix="_ref")` | `symbol, price, sector, sector_ref, country` |

for `left` of `symbol, price, sector` and `right` of `symbol, sector, country`.

**A suffix is applied once, never twice.** A renamed column that collides again raises `DuplicateError` rather than suffixing a second time. So the set of output names is decided in one pass, and a rule that iterated would describe an operation polars does not have.

**A collision is read against the left columns that survive.** Under `how="right"` with coalescing the left key leaves and the right key stays, so a right column named like the key collides with nothing:

```
left {k, v} joined to right {k, v} on "k"
  how="inner", coalesce=True   ->  k, v, v_right      the left key survives
  how="right", coalesce=True   ->  v, k, v_right      the RIGHT key survives
```

> Measured by `TestJoinNaming` and `TestJoinSurvivingColumns` in `spec/tests/test_polars_facts.py`.

### 5.2 The rows

| Call | Schema out | Remainder out | Needs |
| -- | -- | -- | -- |
| `.join(other, on=..., how="semi")` | the left schema | `E-1` | `none` |
| `.join(other, on=..., how="anti")` | the left schema | `E-1` | `none` |
| `.join(other, ...)`, right frame closed | left plus right, suffixed on collision | `E-1` | `R-2` or `R-4` naming the right's columns |
| `.join(other, ...)`, both open, nothing shared | left plus right | `E-4` | `R-10`, plus an `R-2` on each side |
| `.join(other, ...)`, both open, right contained in left | left plus right, all suffixed | `E-4` over `E-5a` | `R-11` plus `R-6` |
| `.join(other, ...)`, partial or unknown overlap | rejected | rejected | nothing that works |
| `.join(other, how="cross")` | left plus right, suffixed on collision | `E-4` | as for `on=` |
| `.join_where(other, *predicates)` | as `.join` | as `.join` | as `.join` |
| `.join_asof(other, on=..., by=...)` | as `.join`, except the key merge: the `on` pair merges only where `coalesce` is true and both keys have one name, and the `by` pairs always merge | as `.join` | as `.join`, plus `B-3` |

**Semi and anti need nothing.** They take no column from the right frame, so nothing can collide and the remainder passes straight through.

**Every join key must be a declared column.** A key in the remainder cannot be proven to exist for every caller, so the whole call would be unprovable. Since a declared column is never in the remainder, no part of a split can hold a key, and the surviving-left rule touches only the declared half.

**A join key holds no alias.** Polars refuses an `.alias(...)` anywhere in the tree of a key of `.join` or `.join_asof`, or of a predicate of `.join_where`, with `InvalidOperationError`, and the schema of the lazy join raises too (`TestJoinKeyTree`). A rename through `.name`, for example `.name.suffix(...)`, is not an alias and runs. The checker reports a key or a predicate whose code holds an alias as `invalid-call-shape`.

**A join key keeps the rows.** Polars moves a key of one value, as an aggregation or a literal, into a column first, and refuses any other key of `.join` or `.join_asof` that does not work value by value at each of its nodes, as `shift`, `sort`, `over`, `head` or a literal series, with `InvalidOperationError`. Only the literal of an `is_in` stays outside the check (`TestJoinKeyTree`). A predicate of `.join_where` is a filter, and polars runs it. Polars reads both facts of a call of a function of the user from the call itself: `map_batches` and `pl.map_groups` from `is_elementwise=` and `returns_scalar=`, both `False` by default, `pl.fold` and `pl.reduce` from `returns_scalar=` alone, as they never work value by value, while `map_elements` works value by value and `rolling_map` does not. `pl.cum_fold` and `pl.cum_reduce` name their result with an alias, so polars refuses them as a key. The checker reports such a key as `invalid-call-shape`, and leaves to the run a key with another function of the user, a flag that is not a literal, or a node that no definition states.

**A declared key leaves the join to the caller.** A declared expression, such as a parameter typed by `column(...)`, fixes what the expression computes, and not its tree. A join reads the tree: only a bare `pl.col(name)` key merges the keys, so `pl.col("a").cast(pl.Int64)` over an `Int64` column keeps both (`TestJoinKeyTree`), and an alias raises. So a key of `.join`, or a predicate of `.join_where`, that holds a declared expression gives a result that depends on the caller, and the checker falls back with the reason `CALLER_KEY`, at the level `possible`.

### 5.3 Why partial overlap is an error and not a vague result

For a caller whose frame has no `sector`, the result holds `sector`. For a caller whose frame has one, the result holds that caller's `sector` plus a `sector_right`. Two callers, two different result types, and one return type must be correct for both.

The error names the columns at risk and the remainder that puts them at risk, and it names both ways out: rule the collision out with a promise or a projection, or make it total with `within=`. Which one is right depends on what the code means, and the checker does not guess.

Reporting an error here has a real cost, since that line is legal polars and it runs. The alternative is worse: a checker that returns a plain frame teaches the user nothing at the point of the problem, and the lost information shows up several calls later as a column that cannot be found.

### 5.4 Failures no schema can see

| Id | Case |
| -- | -- |
| `B-3` | `.join_asof` needs both frames sorted on the key |
| `B-1` | a key present on both sides with incompatible types raises `SchemaError` |

The second is a schema-level failure for two **declared** keys, and is decided outright. It is a boundary only when a key's type is `Unchecked`.

______________________________________________________________________

## 6. Concatenation and stacking

These combine two or more frames. What each demands of the two column sets is what decides whether it can be typed over a remainder.

| Call | Polars | Schema out | Remainder out | Needs |
| -- | -- | -- | -- | -- |
| `pl.concat(how="vertical")` | needs identical schemas, in order | that schema | `E-1` | rejected over a remainder |
| `pl.concat(how="vertical_relaxed")` | the same, with types widened | that schema, widened | `E-1` | rejected over a remainder |
| `pl.concat(how="diagonal")` | the union; a shared name needs one type on both sides | the union | `E-4` | `R-10` |
| `pl.concat(how="diagonal_relaxed")` | the same, with a shared name's types widened | the union | `E-4` | `R-10` |
| `pl.concat(how="horizontal_extend")` | side by side; a shared name raises. Polars 1.42.0 adds this `how`, and an older release raises `ValueError` for it | the union | `E-4` | `R-10` |
| `pl.concat(how="align*")` | joins on **every** common column | the union | `E-4` | `R-10` |
| `.vstack(other)` | as `how="vertical"` | that schema | `E-1` | rejected over a remainder |
| `.extend(other)` | as `how="vertical"`, in place | that schema | `E-1` | rejected over a remainder |
| `.hstack(other)` | as `how="horizontal_extend"` | the union | `E-4` | `R-10` |
| `.merge_sorted(other, key=...)` | as `how="vertical"`, interleaved by key | that schema | `E-1` | rejected over a remainder |
| `.update(other, on=..., how=...)` | replaces values, keeping the left columns | the left schema, a shared column widened | `E-1` | `R-2` naming the right's columns |

The schema check of `how="vertical"`, `how="diagonal"`, `.vstack` and `.extend` reads every `Decimal` as one type, at any depth: a `Decimal(10, 2)` column beside a `Decimal(38, 3)` column runs, and the result keeps the type of the first frame. The relaxed forms widen two `Decimal` types to the larger precision and the larger scale. An align strategy and `.merge_sorted` refuse two different `Decimal` types (measured by `TestDecimalStacks` in `spec/tests/test_polars_facts.py`).

The relaxed forms and `.update` widen a column that two frames hold to the supertype of its two data types: an `Int32` column beside an `Int64` column gives `Int64`, and an `Int64` column beside a `String` column gives `String`, in either order. `unpivot` widens its `value` column the same way. A side of the data type `Unchecked` makes that supertype `Unchecked`, at any depth, because the unknown side decides it (`D-41`): `List(Unchecked)` beside `List(Int64)` gives `List(Unchecked)`. `how="diagonal"` needs one type on both sides, so it gives the known side instead, because a column of another type raises there. Two structs widen to the union of their fields: the fields of the longer struct, or of the first where both have as many, in their order, then each field of the other that it lacks, and a field of both at the supertype of its two types (measured by `TestWideningTakesTheSupertype` in `spec/tests/test_polars_facts.py`).

### 6.1 Why vertical concatenation is rejected over a remainder

It needs the two schemas to be equal **including column order**. A column set here is unordered by definition (`D-5`), chosen so that two frames with the same columns written in a different order are the same type. That choice makes column order unavailable, so the precondition cannot be stated at all, and the call is rejected rather than approximated. This is `B-4`.

Rejecting rather than erasing matters: a caller whose two frames really do match is turned away, and told why, instead of silently losing every downstream check.

### 6.2 Alignment keys need only be castable

Measured on `pl.align_frames`, which section 10 lists: two key columns align when a common type exists, not only when their types are equal.

| Left key | Right key | Aligned key |
| -- | -- | -- |
| `Int32` | `Int64` | `Int64` |
| `Int64` | `UInt64` | `Int128` |

`Int64` and `UInt64` have no common type of their own width, so polars widens to `Int128`. A rule demanding identical types would reject valid polars.

> Measured by `TestAlignment` in `spec/tests/test_polars_facts.py`.

### 6.3 Every combining strategy constrains a shared name

A first reading suggests that only `horizontal_extend` cares, because only it raises on a duplicate. Measured, all four care, and each in its own way:

| How | A name both frames hold |
| -- | -- |
| `diagonal` | must have **one** type on both sides; `Int32` beside `Int64` raises |
| `diagonal_relaxed` | is widened to the supertype of the two that polars gives |
| `horizontal_extend` | raises `DuplicateError`, from polars 1.42.0, which adds this `how` |
| `align*` | becomes a join key, so its types must be castable |

So the requirement is the same for all four: `R-10`, that the two remainders share no name. That is the requirement `disjoint_from=` states, and this group of operations is the one that consumes it.

The reasons differ and are worth keeping apart. For `horizontal_extend` a shared name is an error outright. For the other three it is not an error, and the result's **type** for that name depends on two types the checker cannot see. Under `diagonal_relaxed` the answer would be `Unchecked` under a name that is not known to be there at all, which is why the promise is demanded rather than the type weakened.

> Measured by `TestConcatenation` and `TestSharedColumnsWhenCombining` in `spec/tests/test_polars_facts.py`.

______________________________________________________________________

## 7. Grouping and aggregation

A group-by names its keys, so the keys are always declared. What varies is what happens to every other column.

### 7.1 The explicit form

| Call | Polars | Schema out | Remainder out | Needs |
| -- | -- | -- | -- | -- |
| `.group_by(*keys).agg(*exprs)` | the keys, then one column per expression | the keys plus the produced columns | `E-2` | `none` |
| `.group_by(*keys).agg(pl.all().<r>())` | the keys, then every other column reduced | the keys plus every other column, retyped by `r` | `E-6` | the domain of `r` |
| `.group_by_dynamic(index_column=..., ...).agg(...)` | the keys of `group_by=`, the bounds of each window where `include_boundaries=True`, the index column, then one column per expression over each window | the keys, `_lower_boundary` and `_upper_boundary` at the type of the index, `Datetime("us")` over a `Date` index, the index, then the produced columns | `E-2` | an index of `Int32`, `Int64`, `Date` or `Datetime`, and durations whose unit fits it |
| `.rolling(index_column=..., ...).agg(...)` | the keys of `group_by=`, the index column, then one column per expression over each window | the keys, the index, then the produced columns | `E-2` | an index of `Int32`, `Int64`, `UInt32`, `UInt64`, `Date` or `Datetime`, and durations whose unit fits it |
| `.partition_by(*keys, as_dict=...)` | a **list or dict** of frames | each has the input's schema | `E-1` each | `none` |

`.partition_by` must reject a key that is not declared, exactly as `.group_by` does: the two take the same argument and mean the same thing by it.

The windows of `.rolling` and `.group_by_dynamic` take the place of the groups in `agg`. A duration counts rows, as `2i`, over an integer index, and is a span of time, as `2d`, over a temporal index, and polars refuses either unit over the other index. The windows of `.rolling` always share rows. The windows of `.group_by_dynamic` share rows where `period` is longer than `every`, or as long with `closed="both"`. Over windows that share rows, a function of one input that runs window by window builds each list at the type of its data, where `group_by` checks it against the planned type, so `pl.col("t").rolling_mean(2)` over a `String` column `t` gives `List(Float64)` there, and a lazy frame holds `Planned[List(Float64), List(String)]` (`D-36`). `pl.all()` leaves out the keys and the index. `.head(n)` and `.tail(n)` on a lazy group-by over windows raise, because polars explodes the index.

### 7.2 The shorthand forms, which are more common than `.agg`

`GroupBy` and `LazyGroupBy` each carry seventeen methods: `.agg` of 7.1 and the sixteen shorthands below. Nine of the shorthands reduce every non-key column to one value per group, and `.all()` collects every non-key column into a list. In those ten the keys come first in the result.

| Call | Schema out | Remainder out | Needs |
| -- | -- | -- | -- |
| `.sum()`, `.mean()`, `.median()`, `.min()`, `.max()`, `.first()`, `.last()`, `.n_unique()`, `.quantile(q)` | as `.agg(pl.all().<reducer>())` (section 17.2 of `OPERATIONS.md`): the keys, then every other column retyped by the reducer in the aggregation context, so `.mean()` keeps a `String` column | as for that `.agg`: `E-6`, kind by kind (`T-20`) | the reducer's domain in the aggregation context |
| `.all()` | as `.agg(pl.all())`: the keys, then every other column as a list of its own type | `E-6`, to `List(Unchecked)`: the wrap of each data type is outside the effect language, per section 12.1 of `OPERATIONS.md` | `none` |
| `.head(n)`, `.tail(n)` | the input's schema, keys first | `E-1` | `none` |
| `.count()` | the keys plus one column, `count : UInt32`; a deprecated spelling of `.len()` | `E-2` | no key named `count` |
| `.len()` | the keys plus one column, `len : UInt32` | `E-2` | no key named `len` |
| `.having(pred)` | no frame: a group-by over the groups `pred` keeps, and the call after it decides the schema | as the call after it | `none` |
| `.map_groups(f)` | arbitrary Python; not expressible | `E-2` | `B-6` |

`.len()` and its deprecated spelling `.count()` are the two shorthands that do **not** keep the other columns. Each produces a new name, `len` or `count`, beside the keys alone, so only a key of that name clashes: a column of that name that is not a key is dropped with the rest, and the remainder needs no promise.

> Measured by `TestGroupByShorthands` in `spec/tests/test_polars_facts.py`.

### 7.3 A reducer behaves differently in each context, and on the whole frame

An expression is evaluated in one of two contexts (`D-28`): the aggregation context of `.agg()` and the row context of `.select()`. A whole-frame reducer such as `df.sum()` is a frame method, not a third context, but it has its own domain too. A rule written for one of the three is wrong for another.

| Reducer | In `.agg()`, the aggregation context | In `.select()`, the row context | The whole-frame method |
| -- | -- | -- | -- |
| `sum` on `String` | raises | raises | **passes through** unchanged |
| `sum` on `Date` | `Date` | **raises** | `Date` |
| `mean` on `String` | `String` | **raises** | `String` |
| `mean` on `Date` | `Datetime("us")` | `Datetime("us")` | `Datetime("us")` |
| `min`/`max` on `List` | `List(...)` | **raises** | `List(...)` |

So a reducer's domain must be read per context and, apart from both, for the whole-frame method, which is the most permissive of the three. A column whose data type is a type variable of a generic schema class has no rule on the whole frame: its answer depends on the caller, so it keeps its name as `Unchecked`, as the remainder does.

> Measured by `TestReducerContexts` in `spec/tests/test_polars_facts.py`.

### 7.4 A reducer's domain is not a family, and this recurs

`sum`'s accepted set spans numeric, Boolean and `Duration`, and rejects `String`, `Date`, `Datetime`, `Time` and `List` in a `select`. That is not a polars family in either direction, so no family requirement states it, exactly as 4.1 found for `.cast`.

Two consequences, both stated rather than approximated:

- The gate is written as an explicit set, since `R-9` takes one. For `sum` in a `select` that is `none_of={pl.String, pl.Date, pl.Datetime, pl.Time, pl.List}`, which is exact. Choosing the nearest family would accept programs that raise: `TemporalType` would exclude `Duration`, which `sum` accepts.
- An error message about such a gate must name the set the operation actually enforces. `var` rejects exactly `Duration` and accepts everything `std` accepts, so a message about it naming "numeric or Boolean" describes a different operation.

> Measured by `test_var_rejects_a_duration_where_std_accepts_it` in `spec/tests/test_polars_facts.py`.

______________________________________________________________________

## 8. Whole-frame reduction

Each of these keeps every column name, so the remainder keeps its names, and only its types change. Every one but `.corr()` produces a one-row frame. `.corr()` produces one row per input column.

| Call | Polars | Schema out | Remainder out | Needs |
| -- | -- | -- | -- | -- |
| `.min()`, `.max()` | identity on every data type | every type unchanged | `E-1` | `none` |
| `.sum()` | identity on every type but `Boolean`, which becomes `UInt32` | mapped per type | `E-1` under `R-9` on `Boolean`, else `E-6` to `Unchecked` | see 8.4 |
| `.mean()` | integers become `Float64`; `Date` becomes `Datetime("us")` | mapped per type | `E-6` to `Unchecked` | `none` |
| `.median()` | as `.mean()` | mapped per type | `E-6` to `Unchecked` | `none` |
| `.std()`, `.var()` | integers, decimals and `Boolean` become `Float64`; every other type passes through, `Duration` included | mapped per type | `E-6` to `Unchecked` | `none` |
| `.product()` | `Boolean` becomes `Int64`; text, temporals and the nested types collapse to `Null` | mapped per type | `E-6` to `Unchecked` | `none` |
| `.quantile(q)` | integers and decimals become `Float64`; `Date` becomes `Datetime("us")`; every other type passes through, text and `Boolean` included | mapped per type | `E-6` to `Unchecked` | `none` |
| `.count()` | every column becomes a count | every type `UInt32` | `E-6` to `UInt32` | `none` |
| `.null_count()` | every column becomes a count | every type `UInt32` | `E-6` to `UInt32` | `none` |
| `.corr(label=...)` | the correlation of every pair of columns, one row per column; `label=` adds a column of the names first | every type `Float64`, plus `label : String` where given | `E-6` to `Float64` | numeric or `Boolean` columns only; `R-2` on the label |
| `.describe()` | adds `statistic : String`; numeric columns become `Float64` | see below | `E-6` to `Unchecked` | `R-2` on `"statistic"` |

### 8.1 Why several of these retype to `Unchecked` rather than to a named type

`mean`, `median`, `std`, `var`, `product` and `quantile` each map an input type to an output type by a rule that is neither constant nor identity. Over a remainder the input types are not known column by column, so the output type cannot be named.

The **names** are still known, and `Unchecked` is what keeps them. The result records every name truthfully and marks the types `Unchecked`, rather than discarding the frame. Recovering the name matters more than recovering the type: an unknown name weakens every later lookup on the frame, and an unknown type weakens only operations reading that one column.

Where the input types **are** known, from `R-8` or from a split, the exact output type is computed instead and `Unchecked` does not appear.

### 8.2 `.count()` and `.null_count()` are the exact case

Both map every input type to `UInt32`, whatever it was. That is a constant function, so `E-6` names a real type and nothing is `Unchecked`. These two are the reason `E-6` exists in the grammar at all.

### 8.3 `.describe()` retypes selectively

Measured: `statistic` is `String`, a numeric column becomes `Float64`, and a text column stays `String`. So the rule is not "everything becomes text", and a remainder of unknown types retypes to `Unchecked`.

> Measured by `TestWholeFrameShapes` in `spec/tests/test_polars_facts.py`.

### 8.4 `.sum()` is the one gate the vocabulary can state

Measured, whole-frame `.sum()` keeps every data type except `Boolean`, which becomes `UInt32`. So it is the identity on types **under one condition**, and that condition is a single named type rather than a family boundary:

```python
NoBooleans = other_columns(none_of=pl.Boolean)
```

Under that promise the effect is `E-1` and the result is exact. Without it the names are still known and the types are not, so the effect is `E-6` to `Unchecked`.

This is the simplest case of a pattern 4.1, 4.2 and 7.4 all show: an operation's domain is often not a family, and is almost always a set of types that a promise can name. The vocabulary takes a set for exactly that reason.

> Measured by `TestWholeFrameReducers` in `spec/tests/test_polars_facts.py`.

______________________________________________________________________

## 9. Selectors

A selector picks columns. There are thirty-six constructors, in four groups, and the group decides how exactly a selector resolves over a remainder.

### 9.1 Name selectors: exact over a remainder

These read only the column's name, so they give the same answer for every caller. No promise is ever needed.

| Constructor | Picks |
| -- | -- |
| `cs.all()`, `pl.all()` | every column |
| `cs.empty()` | no column |
| `cs.by_name(*names, require_all=True)` | the named columns |
| `cs.starts_with(p)` | names beginning with `p` |
| `cs.ends_with(s)` | names ending in `s` |
| `cs.contains(sub)` | names holding `sub` |
| `cs.matches(pattern)`, `pl.col("^...$")` | names in which polars finds the pattern |
| `cs.exclude(*names)`, `pl.exclude(*names)` | every column but those |
| `cs.alpha()` | names that are alphabetic |
| `cs.digit()` | names that are numeric |
| `cs.alphanumeric()` | names that are alphanumeric |

**`cs.alpha`, `cs.digit` and `cs.alphanumeric` read the name, not the data type.** The names look like type tests and are not. On a frame of `n : Int64`, `s : String`, `d : Boolean`, `cs.alpha()` picks **all three**, because all three *names* are alphabetic.

> Measured by `TestSelectorsThatReadNames` in `spec/tests/test_polars_facts.py`.

**A pattern is searched, not matched.** `cs.matches("px")` picks `venue_px` as well as `px_bid`, and `.` does not match a newline. `pl.col` reads a name as a pattern only when it starts with `^` and ends with `$`, so `pl.col("^a|b$")` still picks `xb`. The checker reads the part of the dialect that `D-11` lists. `\d`, `\w`, a flag group such as `(?i)`, a counted repetition and a large pattern leave the selector plain.

> Measured by `TestPatternReader` in `spec/tests/test_pattern_reader.py`.

A name selector over a remainder gives `E-3`: the columns it rejects become provably absent, and the rest survive. The result is **not** empty, which is the trap: a remainder column can match the predicate and survive.

### 9.2 Type selectors: settled only under a promise or a split

These read only the data type.

| Constructor | Picks |
| -- | -- |
| `cs.numeric()`, `cs.integer()`, `cs.float()`, `cs.signed_integer()`, `cs.unsigned_integer()`, `cs.decimal()` | by numeric family |
| `cs.string()`, `cs.binary()`, `cs.boolean()` | by scalar family |
| `cs.temporal()`, `cs.date()`, `cs.datetime()`, `cs.duration()`, `cs.time()` | by temporal family |
| `cs.categorical()`, `cs.enum()` | by categorical family |
| `cs.nested()`, `cs.list()`, `cs.array()`, `cs.struct()` | by nested family |
| `cs.object()` | the `Object` type |
| `cs.by_dtype(*dtypes)` | an explicit set of data types |
| `cs.datetime(units, zones)`, `cs.duration(units)` | the `Datetime` data types of those units and zones, or the `Duration` data types of those units |

`cs.datetime` and `cs.duration` with arguments, and `cs.by_dtype` over a data type with parameters such as `pl.Datetime("us")`, name single data types (`D-3`). So does `cs.by_dtype` over a bare marker class of astrolabe such as `Datetime` or `List`, which polars reads as the default instance, `Datetime("us", None)` or `List(Null)`, and a marker with its arguments, `Unchecked` and the class `Decimal` of astrolabe are refused (`TestMarkerClassesAsDataTypes`). A subclass of a polars data type that user code defines reads as the default instance of the polars class of its name, and a name that no kind other than `Decimal` and `Extension` has is refused (`TestUserClassesAsDataTypes`). `cs.datetime("us", None)` picks the naive `us` columns only. `cs.datetime("us")` names no zone, which takes every zone of the unit, the naive columns included. That is one whole sector of the `Datetime` kind (`D-15`), and the type set holds it exactly: `Datetime[Literal["us"], AnyZone]` is how it spells, and a remainder of one `us` zone proves an `all_of=` requirement over it while a remainder of every `Datetime` does not. `cs.datetime("us", "*")` takes every **named** zone and leaves the naive columns out, which is the whole sector less one data type. No one family states that, so the selector is the whole sector met with the negation of the naive data type, which the algebra of `D-12` spells exactly. `cs.datetime("us", ["*", None])` is the whole sector again.

> Measured by `TestTemporalSelectors` in `spec/tests/test_polars_facts.py`.

Over a remainder the answer follows the record's type set. Every row is `T-20`'s effect `Only[R, C]` (`E-3`); a promise or a split settles what that effect keeps:

| The remainder is declared as | Result |
| -- | -- |
| `all_of=F` | every column qualifies, so `E-1` |
| `none_of=F` | none qualifies, so `E-2` |
| nothing at all | `Only[R, C]`: the columns the selector picks, whichever they turn out to be for each caller |
| `SplitColumns[InF, Rest]` | exactly the part, and exact |

**The rule is read against the selector as written, never the family it names.** `~cs.numeric()` under `all_of=NumericType` is emptied and under `none_of=NumericType` is unchanged, which is the exact opposite of the table.

### 9.3 Position selectors: never resolvable

| Constructor | Picks |
| -- | -- |
| `cs.first()` | the first column |
| `cs.last()` | the last column |
| `cs.by_index(*indices)` | columns by position |

A column set is unordered by definition (`D-5`), so position is not recorded and these cannot be resolved even for a fully declared schema. This is `B-4`, and it is a consequence of the choice that makes two frames with the same columns in a different order the same type.

### 9.4 Set operations on selectors

| Written | Means |
| -- | -- |
| `a \| b` | picks what either picks |
| `a & b` | picks what both pick |
| `a - b` | picks what `a` picks and `b` does not |
| `~a` | picks what `a` does not |
| `a ^ b` | picks what exactly one picks |

Each maps onto a combinator of the selector language (`D-12`), which is closed under all of them. A combination of name selectors stays exact; a combination holding a type selector is decided by 9.2's table, applied to the whole expression rather than to any one operand.

______________________________________________________________________

## 10. Frame-level functions

| Call | Polars | Schema out | Remainder out | Needs |
| -- | -- | -- | -- | -- |
| `pl.concat(items, how=...)` | see section 6 | see section 6 | see section 6 | see section 6 |
| `pl.align_frames(*frames, on=...)` | one aligned frame per input | each input's schema | `E-1` each | see 6.2 |
| `pl.collect_all(lazy_frames)` | executes each | each input's schema | `E-1` each | `none` |
| `DataFrame.sql(query)` | runs SQL over the frame | depends on the query text | `E-2` | `B-5` |
| `pl.select(*exprs, **named)` | evaluates expressions with no input frame | the expressions' outputs, by section 17.1 of `OPERATIONS.md` over an empty schema | no remainder | `none` |

`sql` is listed here rather than in Part II because it produces a frame. Its output schema comes from a string, so no signature can state it, and the call is rejected over a remainder.

**Reading a list variable rather than a literal list.** `pl.concat(frames, ...)` where `frames` is a variable is the same operation as `pl.concat([a, b], ...)`, and the type system must read both. A checker that reads one spelling and not the other does not meet this specification: the difference is one of spelling, not of what is statically known. The same holds for `rename(dict(a="b"))` beside `rename({"a": "b"})`, `select(["a", "b"])` beside `select("a", "b")`, and `cs.by_name(["a", "b"])` beside `cs.by_name("a", "b")`.

______________________________________________________________________

# Part II. The rest of the API

The type system computes no schema for these members. Each one keeps the columns as they are, produces no frame, reads a frame from outside the program, or hands the frame to user code. Each appears here with the rule that excludes it, so a reader can look up any polars method and find it.

The six exclusion rules are closed: every member below matches exactly one.

______________________________________________________________________

## 11. Writes to a sink

The frame is unchanged and the result is a file, a stream or `None`.

`sink_batches`, `sink_csv`, `sink_delta`, `sink_iceberg`, `sink_ipc`, `sink_ndjson`, `sink_parquet`, `write_avro`, `write_clipboard`, `write_csv`, `write_database`, `write_delta`, `write_excel`, `write_iceberg`, `write_ipc`, `write_ipc_stream`, `write_json`, `write_ndjson`, `write_parquet`

______________________________________________________________________

## 12. Converts out of polars

The result is not a polars frame, so it carries no schema this system describes.

`deserialize`, `serialize`, `to_arrow`, `to_dict`, `to_dicts`, `to_init_repr`, `to_jax`, `to_numpy`, `to_pandas`, `to_struct`, `to_torch`

A converted value can be converted back, and the frame that comes back is a new frame whose schema is declared where it enters the program.

______________________________________________________________________

## 13. Returns rows or scalars, not a frame

`fold`, `get_column`, `get_columns`, `hash_rows`, `item`, `iter_columns`, `iter_rows`, `iter_slices`, `max_horizontal`, `mean_horizontal`, `min_horizontal`, `row`, `rows`, `rows_by_key`, `sum_horizontal`, `to_series`

The four `*_horizontal` methods reduce across columns within each row, so the result is one `Series` and not a frame. Their input obligation is the same one every operation carries: each column they read must be declared.

`get_column` and `to_series` return a `Series`, which carries one name and one data type. Those are read from the declared schema and are exact, but the result is not a frame, so it leaves this document's scope.

______________________________________________________________________

## 14. Returns a bool or a number

`approx_n_unique`, `equals`, `estimated_size`, `get_column_index`, `is_duplicated`, `is_empty`, `is_sorted`, `is_unique`, `n_chunks`, `n_unique`

`DataFrame.n_unique()` returns an `int`, which is why it is here and `GroupBy.n_unique()` is in Part I: the two share a name and do different things.

______________________________________________________________________

## 15. Execution, display, a no-op on the schema, or user code

The frame that comes out has the same columns as the frame that went in, the call produces no frame at all, or the call hands the frame to user code whose result the type system does not compute.

`cache`, `clear` *(in Part I: it keeps the schema and is usable in a pipeline)*, `collect`, `collect_async`, `pl.collect_all_async`, `collect_batches`, `collect_schema`, `execute`, `explain`, `fetch`, `glimpse`, `inspect`, `lazy`, `map_batches`, `pipe`, `pipe_with_schema`, `profile`, `rechunk`, `remote`, `show`, `show_graph`, `shrink_to_fit`, `with_context`

`pipe(f)` calls `f` on the frame. The type system computes nothing for it: the result's type is the return type `f`'s own signature declares, which mypy reads there. `map_batches` and `map_rows` run arbitrary Python per batch or per row, so their output schema is not readable at all (`B-6`): those are rejected rather than guessed.

______________________________________________________________________

## 16. Reads a frame from outside the program

These mint a schema rather than transform one. The schema of the result is whatever the file, database or object holds, which no signature can state, so it is declared at the point it enters the program.

`pl.defer`, `pl.from_arrow`, `pl.from_dataframe`, `pl.from_dict`, `pl.from_dicts`, `pl.from_numpy`, `pl.from_pandas`, `pl.from_records`, `pl.from_repr`, `pl.from_torch`, `pl.json_normalize`, `pl.read_avro`, `pl.read_clipboard`, `pl.read_csv`, `pl.read_database`, `pl.read_database_uri`, `pl.read_delta`, `pl.read_excel`, `pl.read_ipc`, `pl.read_ipc_stream`, `pl.read_json`, `pl.read_lines`, `pl.read_ndjson`, `pl.read_ods`, `pl.read_parquet`, `pl.scan_arrow_c_stream`, `pl.scan_csv`, `pl.scan_delta`, `pl.scan_iceberg`, `pl.scan_ipc`, `pl.scan_lines`, `pl.scan_ndjson`, `pl.scan_parquet`, `pl.scan_pyarrow_dataset`, `pl.sql`, `pl.wrap_df`

A reader's result is given a schema by an annotation at the call site, and every later operation is checked against it. That is the boundary of the system on the input side, and it is where a program states what it believes the data to be.

______________________________________________________________________

# Part III. The expression API

Every public member of `pl.Expr`, its namespaces, and the expression constructors of `pl`. The vocabulary is the expression half of `OPERATIONS.md`: the type of an expression as its code (`D-33`), the tree of nodes (`D-34`), the evaluator, its passes and its rules (`D-35` … `D-37`), holes and open columns (`D-44`), and the `Unchecked` lifting (`D-41`).

______________________________________________________________________

## 17. How to read Part III

### 17.1 Every call is a node

Polars builds every call of the expression API as a node of its tree, and the checker types each node with its definition (`D-37`). The definition of a node names the parts of the vocabulary of polars that the node uses: its options, its type function, its casts and its kernel. Part III therefore lists each operation under the family of its definition, and states no answer of its own: the answer of a call at a frame is what the evaluator gives (`T-25`), and the tests of each family measure the definitions against polars at every kind in every row form (`M-21`).

A row names the family of the node that the call builds. A call with arguments of another form can build another node: `pl.col("x").fill_nan(v)` is a `when` that polars writes, and `pl.col("l").list.first()` is a `get` with the position `0` as an input. The tree that polars builds decides, so the checker keeps no list of which arguments are inputs.

### 17.2 The planner and the run

Polars answers schema questions in its planner, and runs data in its kernels, and the two disagree in measured ways (`D-36`). The binding layer for this catalog is the run: what `select` and `agg` do with real values. The planner can give a type where every run raises, as `.abs()` on `String` does, and the checker types by the run, so a call that the planner accepts and every run refuses is an error. Where the planner and the run give two types, a frame column takes the run type, and a declaration refuses the argument (`D-40`).

The run of a frame passes through the optimizer of polars, and the kind of frame decides which passes run. An eager `select` or `with_columns` simplifies the expressions: the optimizer can put an input or a literal in place of a node. An eager `group_by().agg()` runs no pass at all, so `pl.col("l") & True` over a `List` column raises there, where a lazy `agg` simplifies it to `l`, and `-pl.lit(2**31)` gives `Int64` there and `Int32` in a lazy `agg`. A lazy frame runs every pass when it collects, the slice pushdown included: a slice whose offset or length is a column hangs where the sliced value is elementwise, leaves the tree otherwise, and raises on an eager frame (`POLARS_ISSUES.md`). In a `select` of one expression, the pushdown moves the slices that cut every column into the plan, so the expression runs over the rows of the slice (`D-36` of `OPERATIONS.md`). The checker evaluates each expression with the passes of the frame that runs it (measured by `TestEngines` in `spec/tests/test_expression_facts.py`).

A lazy frame stores the planner type of each column, and plans each later step at it, while the data has the run type. Where the two differ, the column holds both, as `Planned[run, planner]`, `collect()` gives the run type, and a join checks its keys at both (`D-45` of `OPERATIONS.md`).

### 17.3 The counts

`pl.Expr` carries 10 namespaces. Its methods, the members of its namespaces and the constructors of section 18 depend on the release of polars. Polars 1.42.0 adds the method `is_sorted`. Polars 1.43.0 adds the methods `ewm_sum` and `ewm_sum_by`, the members `cat.physical` and `cat.to`, and the constructor `pl.list`. Polars 1.44.0 adds the members `arr.dot` and `struct.drop`.

| Polars | Methods | Methods that the census calls | Members of the namespaces | Methods and members | Constructors |
| -- | -- | -- | -- | -- | -- |
| 1.41.2 | 209 | 202 | 221 | 430 | 81 |
| 1.42.1 | 210 | 203 | 221 | 431 | 81 |
| 1.43.2 | 212 | 205 | 223 | 435 | 82 |
| 1.44.2 | 212 | 205 | 225 | 437 | 82 |

The methods that the census does not call are the 7 whose output type comes from code outside the call, so no rule holds it. The suite computes every one of these numbers from the installed polars, and fails when the table has no row for that release or when the row and the library disagree (`M-12`). Measured by `TestExpressionCensus` in `spec/tests/test_documents.py`.

______________________________________________________________________

## 18. Constructors: `pl.*`

The callables in `pl` that build expressions are counted in section 17.3. The census calls 73 of them, and leaves out the ones that work only inside another expression, the ones that run a function of the user, and the ones that build a data type.

| Family | Typed as |
| -- | -- |
| one value per row, typed by the type function and the kernel of each node | `pl.arctan2d`, `pl.date`, `pl.from_epoch`, `pl.time` |
| binary operators, under the cast pass of a binary operator | `pl.sql_expr` |
| aggregations: one value from the rows of their input | `pl.any`, `pl.approx_n_unique`, `pl.corr`, `pl.count`, `pl.cov`, `pl.first`, `pl.implode`, `pl.last`, `pl.max`, `pl.mean`, `pl.median`, `pl.min`, `pl.n_unique`, `pl.quantile`, `pl.std`, `pl.sum`, `pl.var` |
| functions over the whole column that keep its length: cumulative, rolling, ranking and window functions | `pl.cum_count`, `pl.cum_sum`, `pl.rolling_corr`, `pl.rolling_cov` |
| functions that give another length | `pl.arg_where`, `pl.repeat` |
| nodes of a kind of their own: a name, a filter, a sort, a gather, a slice, a window, an explode, a `when` | `pl.arg_sort_by`, `pl.head`, `pl.ones`, `pl.tail`, `pl.when`, `pl.zeros` |
| functions of several inputs, cast to a supertype or checked one by one | `pl.all_horizontal`, `pl.any_horizontal`, `pl.arctan2`, `pl.coalesce`, `pl.max_horizontal`, `pl.mean_horizontal`, `pl.min_horizontal`, `pl.struct`, `pl.sum_horizontal` |
| range builders | `pl.arange`, `pl.date_range`, `pl.date_ranges`, `pl.datetime_range`, `pl.datetime_ranges`, `pl.int_range`, `pl.int_ranges`, `pl.linear_space`, `pl.linear_spaces`, `pl.row_index`, `pl.time_range`, `pl.time_ranges` |
| text functions | `pl.concat_str`, `pl.format` |
| temporal functions and builders | `pl.business_day_count`, `pl.datetime`, `pl.duration` |
| list functions | `pl.concat_list` |
| array functions | `pl.concat_arr` |
| leaves of the tree (`D-34`) | `pl.col`, `pl.len`, `pl.lit` |
| a selector (`D-38`) | `pl.all`, `pl.exclude`, `pl.nth` |
| a function of the user: typed by `return_dtype=` where the call takes one, and `Unchecked` otherwise (`B-6`) | `pl.cum_fold`, `pl.cum_reduce`, `pl.cum_sum_horizontal`, `pl.fold`, `pl.map_batches`, `pl.map_groups`, `pl.reduce` |
| leaves bound by an enclosing call (`D-34`) | `pl.element`, `pl.field`, `pl.groups`, `pl.list` |
| builders of a data type, not of an expression | `pl.dtype_of`, `pl.self_dtype`, `pl.struct_with_fields` |

`pl.sum_horizontal` and `pl.mean_horizontal` add five inputs or more in an order that the number of threads of polars decides, and the order can change the type or make polars panic. The rules try each order that keeps the inputs in their places: where every order gives one type, that is the answer, and where the orders differ, the answer is `Unchecked` (measured by `TestThreadsDecideTheOrder` in `spec/tests/test_polars_facts.py`).

______________________________________________________________________

## 19. `pl.Expr` methods

| Family | Typed as |
| -- | -- |
| one value per row, typed by the type function and the kernel of each node | `abs`, `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctanh`, `bitwise_count_ones`, `bitwise_count_zeros`, `bitwise_leading_ones`, `bitwise_leading_zeros`, `bitwise_trailing_ones`, `bitwise_trailing_zeros`, `cast`, `cbrt`, `ceil`, `cos`, `cosh`, `cot`, `cut`, `degrees`, `exp`, `fill_null`, `floor`, `hash`, `is_finite`, `is_infinite`, `is_nan`, `is_not_nan`, `is_not_null`, `is_null`, `log`, `log10`, `log1p`, `neg`, `not_`, `radians`, `reinterpret`, `round`, `round_sig_figs`, `set_sorted`, `sign`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `to_physical`, `truncate` |
| binary operators, under the cast pass of a binary operator | `add`, `and_`, `eq`, `eq_missing`, `floordiv`, `ge`, `gt`, `le`, `lt`, `mod`, `mul`, `ne`, `ne_missing`, `or_`, `sub`, `truediv`, `xor` |
| aggregations: one value from the rows of their input | `agg_groups`, `all`, `any`, `approx_n_unique`, `arg_max`, `arg_min`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `count`, `dot`, `entropy`, `first`, `has_nulls`, `implode`, `index_of`, `is_empty`, `is_sorted`, `item`, `kurtosis`, `last`, `len`, `lower_bound`, `max`, `max_by`, `mean`, `median`, `min`, `min_by`, `n_unique`, `nan_max`, `nan_min`, `null_count`, `product`, `quantile`, `skew`, `std`, `sum`, `upper_bound`, `var` |
| functions over the whole column that keep its length: cumulative, rolling, ranking and window functions | `arg_sort`, `backward_fill`, `cum_count`, `cum_max`, `cum_min`, `cum_prod`, `cum_sum`, `diff`, `ewm_mean`, `ewm_mean_by`, `ewm_std`, `ewm_sum`, `ewm_sum_by`, `ewm_var`, `forward_fill`, `interpolate`, `interpolate_by`, `is_duplicated`, `is_first_distinct`, `is_last_distinct`, `is_unique`, `pct_change`, `peak_max`, `peak_min`, `qcut`, `rank`, `rechunk`, `reverse`, `rle_id`, `rolling_kurtosis`, `rolling_max`, `rolling_max_by`, `rolling_mean`, `rolling_mean_by`, `rolling_median`, `rolling_median_by`, `rolling_min`, `rolling_min_by`, `rolling_quantile`, `rolling_quantile_by`, `rolling_rank`, `rolling_rank_by`, `rolling_skew`, `rolling_std`, `rolling_std_by`, `rolling_sum`, `rolling_sum_by`, `rolling_var`, `rolling_var_by`, `shift` |
| functions that give another length | `append`, `arg_true`, `arg_unique`, `bottom_k`, `bottom_k_by`, `drop_nans`, `drop_nulls`, `extend_constant`, `gather_every`, `hist`, `mode`, `repeat_by`, `reshape`, `rle`, `sample`, `search_sorted`, `shuffle`, `top_k`, `top_k_by`, `unique`, `unique_counts`, `value_counts` |
| nodes of a kind of their own: a name, a filter, a sort, a gather, a slice, a window, an explode, a `when` | `alias`, `explode`, `fill_nan`, `filter`, `flatten`, `gather`, `get`, `head`, `limit`, `over`, `rolling`, `slice`, `sort`, `sort_by`, `tail`, `where` |
| functions of several inputs, cast to a supertype or checked one by one | `clip`, `is_between`, `is_close`, `is_in`, `pow`, `replace`, `replace_strict` |
| an inner expression evaluated at the element of each list or array, or at the fields of a struct | `cumulative_eval` |
| a selector (`D-38`) | `exclude` |
| gives its input unchanged | `shrink_dtype` |
| a function of the user: typed by `return_dtype=` where the call takes one, and `Unchecked` otherwise (`B-6`) | `deserialize`, `from_json`, `inspect`, `map_batches`, `map_elements`, `pipe`, `register_plugin`, `rolling_map` |

A method that takes an expression argument takes it as an input of its node, so every input runs and a failure there fails the call (`D-31`). A method that takes a Python value takes it as a literal input, whose value the node reads (`D-34`), or as an option of the node, which the node reads from the tree.

______________________________________________________________________

## 20. The namespaces

| Family | Typed as |
| -- | -- |
| nodes of a kind of their own: a name, a filter, a sort, a gather, a slice, a window, an explode, a `when` | `list.explode`, `str.explode` |
| text functions | `str.concat`, `str.contains`, `str.contains_any`, `str.count_matches`, `str.decode`, `str.encode`, `str.ends_with`, `str.escape_regex`, `str.extract`, `str.extract_all`, `str.extract_groups`, `str.extract_many`, `str.find`, `str.find_many`, `str.head`, `str.join`, `str.json_decode`, `str.json_path_match`, `str.len_bytes`, `str.len_chars`, `str.normalize`, `str.pad_end`, `str.pad_start`, `str.replace`, `str.replace_all`, `str.replace_many`, `str.reverse`, `str.slice`, `str.split`, `str.split_exact`, `str.splitn`, `str.starts_with`, `str.strip_chars`, `str.strip_chars_end`, `str.strip_chars_start`, `str.strip_prefix`, `str.strip_suffix`, `str.strptime`, `str.tail`, `str.to_date`, `str.to_datetime`, `str.to_decimal`, `str.to_integer`, `str.to_lowercase`, `str.to_time`, `str.to_titlecase`, `str.to_uppercase`, `str.zfill` |
| binary functions | `bin.contains`, `bin.decode`, `bin.encode`, `bin.ends_with`, `bin.get`, `bin.head`, `bin.reinterpret`, `bin.size`, `bin.slice`, `bin.starts_with`, `bin.tail` |
| category functions | `cat.ends_with`, `cat.get_categories`, `cat.len_bytes`, `cat.len_chars`, `cat.physical`, `cat.slice`, `cat.starts_with`, `cat.to` |
| temporal functions and builders | `dt.add_business_days`, `dt.base_utc_offset`, `dt.cast_time_unit`, `dt.century`, `dt.combine`, `dt.convert_time_zone`, `dt.date`, `dt.datetime`, `dt.day`, `dt.days_in_month`, `dt.dst_offset`, `dt.epoch`, `dt.hour`, `dt.is_business_day`, `dt.is_leap_year`, `dt.iso_year`, `dt.microsecond`, `dt.millennium`, `dt.millisecond`, `dt.minute`, `dt.month`, `dt.month_end`, `dt.month_start`, `dt.nanosecond`, `dt.offset_by`, `dt.ordinal_day`, `dt.quarter`, `dt.replace`, `dt.replace_time_zone`, `dt.round`, `dt.second`, `dt.strftime`, `dt.time`, `dt.timestamp`, `dt.to_string`, `dt.total_days`, `dt.total_hours`, `dt.total_microseconds`, `dt.total_milliseconds`, `dt.total_minutes`, `dt.total_nanoseconds`, `dt.total_seconds`, `dt.truncate`, `dt.week`, `dt.weekday`, `dt.with_time_unit`, `dt.year` |
| struct functions | `struct.field`, `struct.json_encode`, `struct.rename_fields`, `struct.unnest` |
| list functions | `arr.head`, `arr.slice`, `arr.tail`, `list.arg_max`, `list.arg_min`, `list.concat`, `list.contains`, `list.count_matches`, `list.diff`, `list.drop_nulls`, `list.first`, `list.gather`, `list.gather_every`, `list.get`, `list.head`, `list.join`, `list.last`, `list.len`, `list.max`, `list.mean`, `list.median`, `list.min`, `list.sample`, `list.set_difference`, `list.set_intersection`, `list.set_symmetric_difference`, `list.set_union`, `list.shift`, `list.slice`, `list.sort`, `list.std`, `list.sum`, `list.tail`, `list.to_array`, `list.to_struct`, `list.var` |
| array functions | `arr.arg_max`, `arr.arg_min`, `arr.contains`, `arr.count_matches`, `arr.explode`, `arr.first`, `arr.get`, `arr.join`, `arr.last`, `arr.len`, `arr.max`, `arr.mean`, `arr.median`, `arr.min`, `arr.shift`, `arr.slice(as_array=True)`, `arr.sort`, `arr.std`, `arr.sum`, `arr.to_list`, `arr.to_struct`, `arr.var` |
| an inner expression evaluated at the element of each list or array, or at the fields of a struct | `arr.agg`, `arr.all`, `arr.any`, `arr.eval`, `arr.n_unique`, `arr.reverse`, `arr.unique`, `list.agg`, `list.all`, `list.any`, `list.eval`, `list.filter`, `list.item`, `list.n_unique`, `list.reverse`, `list.unique`, `struct.with_fields` |
| rewrites of the output name (`D-32`) | `name.keep`, `name.prefix`, `name.prefix_fields`, `name.replace`, `name.suffix`, `name.suffix_fields`, `name.to_lowercase`, `name.to_uppercase` |
| a rewrite of the output name by a function of the user, so the name is unknown (`D-32`) | `name.map`, `name.map_fields` |
| reads the expression and builds no column (section 21) | `meta.as_expression`, `meta.as_selector`, `meta.eq`, `meta.has_multiple_outputs`, `meta.is_column`, `meta.is_column_selection`, `meta.is_literal`, `meta.is_regex_projection`, `meta.ne`, `meta.output_name`, `meta.pop`, `meta.root_names`, `meta.serialize`, `meta.show_graph`, `meta.tree_format`, `meta.undo_aliases`, `meta.write_json` |
| no definition states the node, so its answer is `Unchecked` (`D-37`) | `arr.dot` and `struct.drop` (from polars 1.44.0), `ext.storage`, `ext.to` |

A `list` or `arr` member that evaluates an inner expression, `list.eval(pl.element() * 2)`, evaluates it at the element of the list, with the element as the one column of the inner tree. The `name` members rewrite the output name and never a data type, so a selection admits only the two affixes (`T-3`, `M-13`).

______________________________________________________________________

## 21. `meta` (17 members)

The introspection namespace reads an expression instead of building one. None of its members produces a column, so none has a row above beside the introspection row.

The seventeen: `meta.as_expression`, `meta.as_selector`, `meta.eq`, `meta.has_multiple_outputs`, `meta.is_column`, `meta.is_column_selection`, `meta.is_literal`, `meta.is_regex_projection`, `meta.ne`, `meta.output_name`, `meta.pop`, `meta.root_names`, `meta.serialize`, `meta.show_graph`, `meta.tree_format`, `meta.undo_aliases`, `meta.write_json`.

______________________________________________________________________

## 22. What this document does not cover

| Not covered | Where it lives |
| -- | -- |
| the formal definitions, rules and proofs | `OPERATIONS.md` |
| what a typed frame and a typed expression are, and why | `DATAFRAME_SPECIFICATION.md`, `EXPRESSION_SPECIFICATION.md` |
| the plugin, its configuration and its implementation status | outside the specification |
