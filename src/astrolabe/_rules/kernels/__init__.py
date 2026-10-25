"""Give the run type of a node from the data types that reach its kernel, as polars dispatches.

Polars plans a node at the planner types of its inputs, and its cast pass
inserts casts. The kernel then runs on the data that arrives, and polars checks
nothing more than the dispatch of the kernel on the data type. So the run type
of a node is the answer of its kernel at the data types, and it can differ from
the planner type.

Each module translates one family of kernels of polars:

- `dispatch` holds the parts that every family uses: the data that reaches
  a kernel, the cast at run time, and the form in which a definition
  states the dispatch of a kernel of one input.
- `arithmetic` holds `+`, `-`, `*`, `/` and `%` of two series, with their
  own coercion, and the true division, the floor division and `pow`.
- `temporal` holds the arithmetic of `Date`, `Datetime`, `Duration` and `Time`.
- `nested` holds the arithmetic of `List`, `Array` and `Struct` values.
- `comparison` holds `==`, `!=`, `<`, `<=`, `>` and `>=`.
- `logic` holds `&`, `|`, `^`, the logical operators and the bitwise functions.
- `binary` maps each binary operator of an expression to its kernel.
- `floats` holds the functions that run a float kernel, such as `sqrt` and
  `log`, and the tests of a float, such as `is_nan`.
- `numeric` holds `abs`, `neg`, the roundings, `cut`, `cum_sum`,
  `rolling_mean`, `sum` and `mean`.
- `layout` holds the functions that read how the data is laid out, such as
  `is_null`, `hash`, `to_physical` and `reinterpret`.
- `fill` holds `fill_null`, which reads the nulls of its value.
- `reductions` holds the aggregations, such as `min`, `std`, `implode`,
  `quantile` and `corr`, with one kernel for a whole series and one for
  the groups of `agg`.
- `namespaces` holds the kernels of the `str` namespace that read an option
  of their node, such as the target of `str.strptime`.
- `dates` holds the kernels of the `dt` namespace, such as `dt.year`,
  `dt.truncate` and `dt.replace`, and of the functions of business days.
- `lists`, `arrays` and `structs` hold the kernels of the `list`, `arr` and
  `struct` namespaces, which work on the values inside each nested value.

A kernel takes plain values: a `Data` for each input. It gives a `DType`, or it
raises `PolarsRefusal` with the way that polars stops and, where polars raises,
the class of its error.
"""
