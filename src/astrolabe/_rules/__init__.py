"""Type polars expressions by the rules that polars declares, translated from its source.

Polars declares the type of each function in a small vocabulary: a type
function, a few options, the rules of its cast pass, and the dispatch of its
kernel on the data type. This package translates that vocabulary once, so that
the rules of a node are a short definition in it. Measurement does not supply a
rule here: the tests in `spec/tests` check each rule against polars. Each
module translates one mechanism of polars:

- `releases`: the release of polars whose rules apply, and the methods whose
  rule changes between releases.
- `dtypes`: the data types as plain values, the type tests of polars, and the
  types of an integer literal by its value. It also holds the range of each
  integer type, the fit of a value in a `Decimal`, and the time units.
- `failures`: the ways that polars stops, the Python classes of its errors,
  the refusal of a query and the answer that the rules cannot give.
- `operators`: the binary operators of an expression, and the arithmetic
  operations of the kernels of polars.
- `supertype`: the join of two data types, with its flags.
- `casts`: the check that a cast can run and the lossless check of the cast
  pass. It also holds the cast kernel of each source kind and the fate of each
  value of a cast.
- `options`: the `FunctionOptions` of a function, the projection height of
  the planner, the row form of a value, and the length rule of a kernel.
- `type_functions`: the planner type of a node, from its `FieldsMapper` or from
  the rule of its kind of node.
- `cast_pass`: the casts that the type coercion pass puts above the inputs
  of a node, and the plans that it refuses.
- `execution`: the run of a node in a `select` and in `group_by().agg()`,
  the fold of two literals, and the rewrites of the optimizer: the slice
  pushdown and the rewrites of `&` and `|`.
- `kernels`: the dispatch of each family of kernels on the data that reaches
  it at run time: arithmetic, comparison, logic, floats, numbers, time,
  nested types, the fill of nulls, the namespaces and the layout of a series.
- `definitions`: the definition of each node, which names the part of the
  vocabulary that the node uses at each place, and a registry that finds a
  definition by the identity of its node.
- `node_evaluation`: the four answers of one node whose inputs are leaves,
  from its definition: the planner type and the run type, in a `select` and
  in `group_by().agg()`.
- `tree` and `tree_evaluation`: the tree that polars builds for an
  expression, and the state of each node from the states of its inputs.

Every rule refuses in one way. Where polars refuses a query, the rule raises
`PolarsRefusal`. It holds the way that polars stops (it raises, panics, ends
the process or never returns), the Python class of the error where it raises,
and the message of the refusal. Some functions of polars give no answer in
place of an error, for example the join of two types and the cast check. A rule
that mirrors such a function gives `None`. Where the rules cannot answer,
because the state of the process decides or because the package does not
translate that part of polars, a rule raises `UndecidedError`. The fate of one
value of a cast is an `Outcome`, not a refusal.

The plugin gives an expression the type that these rules give its tree.
"""
