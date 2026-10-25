# Plan: close the testing gaps of `src/astrolabe`

## Goal

Every line of `src/astrolabe` runs in a test other than its own doctest. A doctest sits inside the code that it runs, so a change to the code carries its example along, and it does not check the code against polars or against what the checker answers.

The measure: run `uv run pytest tests spec tools -n 2 --cov-fail-under=0 --cov-report=json:<file>`. That run leaves out the doctests of `src/`, and each line that it does not reach is a gap.

## The rules for every gap

- A branch of the rules (`src/astrolabe/_rules/`) gets a case that compares the rules with polars on an input that reaches it: a tree in the four answers of `TreeFrame`, or the probe of the topical test file. A difference from polars is a bug: fix it, or record it in `KNOWN_ISSUES.md`.
- A branch of the plugin (`src/astrolabe/_hooks/`, `frame.py`, `plugin.py`) gets a yml case, or a test in `tests/` where the branch works on plain values.
- A branch that no input can reach is dead: delete it. A guard that a mypy contract or an earlier check rules out gets `# pragma: no cover` with its reason.
- A new case goes in the topical test file of its module, never in a file named after this work.
- Derive the input of a branch from the code and from the source of polars. Verify each fact of polars with a script.

## Status

- Measure: done. Without the doctests of `src/`, 636 lines and 410 branch arcs are not reached: about 400 lines in `_rules/`, and the rest in `_hooks/` and `frame.py`.
- Tools: `TreeFrame` takes columns of other data types (`TreeColumns`). A tree that polars writes only in its serialized form, as a leaf of fields that is not strict, is read with `pl.Expr.deserialize`.
- A `__repr__` exists only to show a value in a doctest, so it stays out of scope.
- Kernels: in progress (`KernelTrees` in `spec/tests/test_rules_kernels.py`).
- Type functions: in progress (`TypeFunctionTrees` in `spec/tests/test_rules_type_functions.py`).
- Bugs that the cases found: `pl.col("s").struct[0]` and `pl.col("l").list[0]` crashed the checker (fixed: the recorder writes a subscript, and the code reader reads it). A field by a missing place gives no error (`KNOWN_ISSUES.md`).
- Bugs that the search of random trees found, all fixed:
  - A nested value cast to a flat type crashed the value cast (`ArrowValues._fields`).
  - The rolling kernels and `list.slice` read a literal whose cast raises as a null (`CastFold.nulled`). A pair of types with no cast kernel gave the outcome `LOST`, not `RAISES` (`CastKernel.outcome`).
  - An `over` whose keys each hold one row compared the function with the groups of `agg`, not with one group of the whole frame. The mapping `join` checked the rows.
  - A lazy `select` of one expression did not move its slices into the plan (`SliceLift`). The check of a refusal at every frame now tries that context too.
- Found and recorded in `KNOWN_ISSUES.md`: `extend_constant` over an input of no row, and the stale cache of mypy after a change of the plugin.
- `TreeFrame` now evaluates the rules with the flags of a lazy `select` of one expression, as its polars side runs.
- Second measure (all tests, no doctests of `src/`): 547 lines and 351 branch arcs are not reached, 307 lines of them in `_rules/`.
- Rules done since then: the cast pass, the casts, the node options, the data types, the window keys and the lone `len` and slice forms, part of the kernels. Each reachable path has a case in its topical file (`CastPassTrees`, `CastTrees`, `KernelTrees`, `WIDE_CASES`, `LoneCounts`). Each guard on the format of the tree of polars, or that an earlier check rules out, has a pragma with its reason. The members `TEMPORAL`, `ARRAY` and `STRUCT` of `Namespace` were dead, and are deleted.
- More bugs that the cases found, fixed: `dt.replace` takes the rows of its first input, as `clip` does. `list.slice` read an offset of one row of text through the cast to `Int64`, so `"1.5"` gave a false error.
- Recorded in `KNOWN_ISSUES.md`: a date format with `%%` and no day, and a cast to an array with a shape of a tuple, which names a sample type in its error.
- Plugin: `frame.py` has runtime tests for each marker of another kind. Found, to fix after the measure: a method call on a hand-written `Deferred` type whose code does not parse crashes mypy in the recorder, and the fit of such an argument names the column "None".
- Third measure: 370 lines and 242 branch arcs are not reached, 184 lines of them in `_rules/`. All tests pass.
- Since the third measure:
  - Rules: kernel cases for appends whose data the plan does not name, nested struct arithmetic, a slice of one list by a null offset, linear spaces of a count of text and of `Float16`, rolling windows of no length, a replace of no occurrence by a column, and a parse of a time that is not exact. Each kernel guard that the type function, the cast pass, the coercion or the run rules out has a pragma with that reason.
  - Bug fixed: a strict parse of a literal time by a format with no hour or no minute was typed. Polars refuses it, and the spec (`OPERATIONS.md`) and the rules now say so.
  - Plugin: a callable of the user in a list, as `cs.by_dtype([int])`, gave a false error with the text of an internal object. The build check and the evaluation now read a refusal that a hole of a function can cause as not proven.
  - Plugin cases: the folding of `~` and of a wide product, the fit and the join of an unreadable code, a struct of two open fields, selections over a remainder, and the selector lists of a declaration.
  - Recorded in `KNOWN_ISSUES.md`: a bare `TypedDataFrame`, and a parse of a literal text by a format in a variable.
- Fourth measure: 267 lines and 166 branch arcs are not reached, 104 lines of them in `_rules/`.
- Since the fourth measure:
  - Rules: each reachable path has a case (`LiteralFirst`, `CASES`, `KernelTrees`, `TypeFunctionTrees`, `LoneCounts`, and the tree reader). Each guard that an earlier check or the format of polars rules out has a pragma with its reason. An arc that only one side of a condition reaches has `# pragma: no branch` with its reason. `Definitions.of`, `LengthClosures.index`, the join of two integer literals and the join of a text literal were dead, and are deleted or used.
  - Bugs fixed in the rules: an `over` key in `agg` with a `list.eval` or an `is_in` of a literal inside `struct.with_fields` was refused, because the test of an elementwise tree read a node with no definition as not elementwise. A `pl.field` inside `pl.struct` inside `struct.with_fields` crashed the reader of output names.
  - Bugs fixed in the plugin: a cast of a typed series gave the target where the rules answer `Unchecked`. `update`, a relaxed concat and `unpivot` gave the known side of an `Unchecked` column, where the supertype is unknown. A float constant that folds to an infinity was recorded as `1e309`, which the evaluation could not read. The reader of plain values read `pl.lit(x)` as `x`, so a pivot that polars refuses was typed. An expression over a declared column whose name the checker does not read leaked the reserved name `__leaf_0__`.
  - Dead in the plugin, and deleted: `CheckerLiteralTypeBuilder.boolean` and its test builder.
  - Recorded in `KNOWN_ISSUES.md`: the casts of `Object`, the array slice that keeps the array, a `cut` with a NaN break, a pivot with an expression in `on_columns`, the stored type that a declared selection drops, an opaque call over a column of the remainder, a declared column with a name of the type `str`, and a cast to a struct of a field type in a variable.
  - Recorded in `POLARS_ISSUES.md`: `clip` of a literal by a column, whose answer depends on the chunks of the column.
- Fifth measure: 132 lines and 81 branch arcs are not reached, 42 lines of them in `_rules/`, most of them `__repr__` methods.
- Since the fifth measure:
  - Plugin cases: Final names whose type keeps no value, instances of a user class named after a kind, a narrowed lazy frame in `pl.concat`, a series of a data type in a type variable, structs of other numbers of fields in a diagonal concat, a frame sum over an open `Decimal`, and a fit error that names a `UInt` type. The engine parity tests compare the selector entailment with its model, and denote the canonical forms, a carried union and a graft of no column.
  - Bug fixed: a cast of a typed series to a data type that the checker cannot read kept the old data type. It now gives `Unchecked`, through the `allow_unproven` gate.
  - Dead in the plugin, and deleted: `Outcome.rank` and `KindRoster.nested`.
  - Recorded in `KNOWN_ISSUES.md`: a join key with an alias, and a relaxed concat of two structs.
- Open in the rules: the `__repr__` methods and `Arguments.texts`, which only the reprs read, `ArrowCast._structs` for structs of other numbers of fields, and the `frame` rule of `KernelLength.length`.
- Open in the plugin: `OpenParameters.opened` for a struct and a list, `Generalization._parts` and `_pairs`, `DeferredEvaluation._stored` for a planner type `Unknown(any)`, `DeferredArgument._leaf` for a leaf that is no instance, `DeferredRemainder` (`effect`, `_node`, `cause`, `affix`), `DeferredSelectionFit._counted`, `ColumnRef.tuple_argument`, `Frame.schema_items`, the encoding readers (`expr_codec`, `join_remainder`, `marker_minter`, `remainder_edit`, `remainder_knowledge`, `rendering_codec`, `selection_codec`, `selector_codec`), the expression helpers (`ArrayText.read`, `TopLevelSplit._read_quoted`, `Collapse.collapsed`, `Materializer._axis_renderings`, `SelectionMaterializer`, `UnknownKind.answer_at`, `BlankBinding.outcome`), `_JoinKeyResolver._is_bare_column`, the readers of `ArgumentCode` (`_type_of`, `_float_call`, `_family`), and the remainder engine (`TemporalFamilies.duration`, `KindSectors`, `KindPart.within`, `ColumnSet.holds`).
