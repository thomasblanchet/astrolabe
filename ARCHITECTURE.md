# Architecture

How the system in `spec/` is built.

The specification says what the type system promises and how it decides things. This document says how the code is arranged to deliver that, and how it relates to the type checker it runs inside.

Read the specification first. This document assumes its vocabulary: schema, remainder, record, requirement, effect, the every-caller rule, kind, code, node, hole, character, selection, declaration, fit.

______________________________________________________________________

## 1. The one decision everything follows from

A type checker can be used in two ways. It can be asked to **derive** things, by encoding a problem so that its own rules produce the answer. Or it can be asked only to **carry** things, with every answer computed by us.

> **This system asks the type checker to carry, never to derive.**

Concretely, the checker is asked for three services:

- hold a type and pass it around unchanged,
- substitute a type variable when a function is called, and
- tell a hook which type an expression is expected to have.

It is never asked whether one remainder satisfies another's requirement, never asked to normalise a set of names, never asked whether one expression type fits a declared one, and never asked to infer how a generic parameter varies. Every rule in the specification's section 8 and in the operations catalogue is answered by our code.

### 1.1 Why

Encoding the problem into the checker's own rules is seductive, because the answers then come for free. It fails for three reasons, in increasing order of seriousness.

**Not everything fits.** A requirement relating two remainders, such as "these two remainders share no name", is not a property of either type alone. No arrangement of subtyping expresses it, so some requirements need our code regardless, and the system would carry two mechanisms.

**The fit distorts the design.** Making the checker's rules produce the right answer means choosing encodings for their side effects rather than their clarity: a set of names becomes a union because unions ignore order, a marker gains a member it never uses because that member steers variance inference.

**The failure mode is silent.** When an encoding is subtly wrong, nothing reports it. The checker keeps answering, and answers incorrectly. For example, a marker that inherits a variance that is correct for a constraint and wrong for a constructor accepts a function that claims to remove one column and removes two.

Computing the answers ourselves costs code. It buys the ability to be wrong **loudly**, in a place that can be tested directly.

### 1.2 What it costs

The checker enforces nothing about our types on its own. Wherever we do not intervene, it compares our types structurally, understanding nothing. That is acceptable only because of section 7: everywhere the checker decides alone, structural comparison coincides with our own notion of equality, so its verdict is right even though its reasoning is empty.

______________________________________________________________________

## 2. Three layers

```
   engine     pure. Knows nothing about any type checker.
     |          _hooks/remainder/   _hooks/expression/   _hooks/deferred/   _rules/
  encoding    translates engine values to and from checker types.
     |          _hooks/encoding/   _hooks/deferred/argument.py
   plugin     hooks. Decode, ask the engine, encode the answer.
                _hooks/*.py  _hooks/recording/  _hooks/declaring/  _hooks/dispatch/
```

The layering has one hard rule:

> **The engine must not reference the type checker in any way.**

Nothing under `_hooks/remainder/`, `_hooks/expression/`, `_hooks/deferred/` or `_rules/` imports mypy, apart from `deferred/argument.py`, which reads the types of mypy into the plain values of the evaluation and so belongs to the encoding. That is what makes the specification's rules into ordinary code with doctests: an entailment rule is a function from values to a boolean, the rule of a node is a function from the states of its inputs to its own, and each has a test with no type checker involved.

The encoding is the only layer that knows both worlds. The plugin is thin: it decides *when* a question is asked, not *what the answer is*.

Two facades keep mypy's own objects out of the plugin layer as well. `_hooks/domain/` wraps a frame instance (`Frame`, `KnownColumns`) and a column item (`ColumnRef`); `_hooks/calls/` wraps call-site argument access (`CallArgs`) and data-type expressions written in code. `ConstantFold` computes arithmetic of literals, such as `2**10`, because polars receives the value and not the text.

______________________________________________________________________

## 3. The frame half of the engine

Everything under `_hooks/remainder/` denotes a set of column sets. A *column set* (`record.py`'s `ColumnSet`) is one concrete answer: these names, with these data types. A real frame has exactly one. Everything the engine manipulates describes the column sets that are still possible.

| Value | Denotes |
| -- | -- |
| a remainder expression (`canonical.py`) | a **function** from variable bindings to one column set |
| a record (`record.py`) | the **set** of column sets still possible |
| a requirement (a marker on a declaration) | the **set** of column sets that satisfy it |
| an effect (`effects.py`) | a **function** on column sets, lifted to act on a whole set of them |

**A record is an over-approximation, and that is why losing a fact is safe.** Forgetting something makes the possible-set larger, and a larger set still contains the truth. Inventing a fact makes the set smaller, and it may no longer contain the truth. Every rule is written to record less rather than guess.

**Entailment is containment.** A requirement holds of a record exactly when the record's set is contained in the requirement's set.

### 3.1 Remainder expressions and their canonical form

`canonical.py` holds the expression grammar and its normal form.

```
Expr      = Ref(name)                        a variable
          | Combined(left, right)            the union of two remainders
          | Applied(effect, inner)           one effect over one remainder:
                Keep | Empty | Retype | Suffix | Prefix | Filter
          | Carried(form)                    an already canonical form, back in
                                             expression position
Canonical = (base, where, suffix, prefix, retype)
Base      = Variable(name) | Nothing | Union(forms)
```

An expression is syntax. `Expr.denote(bindings)` gives the column set it stands for once its variables are bound, and `Expr.normalise()` folds every effect into the five canonical slots. Reading a canonical form inward: a data type replacement, a prefix, a suffix, a filter, and a base. Two expressions are the same type exactly when their canonical forms are equal.

There is one filter case, not three. The specification's `Minus`, `Only` and `Except` are spellings of one `Filter` slot holding a selector, negated where the spelling excludes. There is no empty remainder value: an empty remainder carries no information and is represented by the absence of a remainder.

### 3.2 Absence rules, selectors, the name language

`absence.py` is the predicate language over a name:

```
Absence = Never | Always | OneOf(names)
        | StartsWith(prefix) | EndsWith(suffix) | Contains(part)
        | Matching(signature)
        | Not(rule) | And(a, b) | Or(a, b)
        | AfterStripping(suffix, rule) | BeforeStripping(prefix, rule)
```

`Absence.holds_of(name)` evaluates a rule at one name. Every constructor denotes a regular set of names, which is what makes entailment decidable: `name_language.py` wraps the `greenery` library behind one facade (`NameSet`), and `MachineSignature` is the minimized automaton's text, the one form two rules built from different spellings compare equal under. `Matching` is the only constructor a user cannot write directly. It is where a rule lands when no named constructor spells it. A type spells it by a regular expression of its names where one reads well, as `ab.*`, and by the text of the minimal automaton otherwise (section 5.3).

`selector.py` is the same idea over a whole column, name and data type together: `NameTest(rule)`, `OfType(family)`, and `SelNot`, `SelAnd`, `SelOr` over them. A selector picks columns; its negation is an absence rule. `Selector.surviving_types(types)` gives the data types a picked column can still have, which is what a selection's remainder effect reads. `SelectorEntailment` decides containment between two selectors. `type_set.py` holds the data-type half: `TypeSet`, and the `Family` roster derived from polars' own class hierarchy.

`entailment.py` decides whether one absence rule entails another, with the cheap paths first: two finite sets, a prefix against a prefix, and only then the automaton.

### 3.3 Records and knowing

`Record(absence, types)` is what is known about a remainder. It is never stored in a type. It is derived on demand by *knowing* a remainder expression under an environment of records, one rule per effect:

- a filter adds the negated selector's names to the absence rule and narrows the type set to the survivors,
- a retype keeps the names and replaces the type set,
- a suffix or prefix is the only case that invents names, which is why the absence language carries `AfterStripping` and `BeforeStripping`,
- a union is absent of a name only when both sides are.

`split.py` binds a `SplitColumns[...]` declaration: an ordered tuple of part predicates where the first match wins, each part's closed form an ordinary selector. `SplitPredicates` turns each declared part's record into its predicate, and the fit gives a split parameter one bound declaration per part: a closed argument's columns go to their first accepting part, and an open argument's remainder is narrowed to each part's closed form.

### 3.4 Requirements

A requirement is written by the user on a declaration: `other_columns(without=[...])`, `other_columns(disjoint_from=Other)`, the family forms. It is read back as markers on the declared class's bases (section 5.3), never from plugin memory. A unary requirement is also a source of knowledge: a body may assume what its caller promised, so the declaration's record comes from its own requirements. A relational requirement is a fact about a pair and is checked only at a call, where both sides are bound.

**Effects and requirements share no ancestry.** An effect transforms a remainder; a requirement tests one. If one inherited from the other so that subtyping derived entailment, the inheritance would carry the variance of one into the other, where it is wrong.

______________________________________________________________________

## 4. The expression half of the engine

The type of an expression is its code (`D-33`). The engine reads that code where it meets a frame or a declaration. It follows polars node by node to compute the column that the expression gives.

### 4.1 The rules of polars

`_rules/` is the evaluator. It is part of the engine, so it imports nothing of mypy.

- Each mechanism of polars has one module, translated from its source. The modules are the data types (`dtypes.py`), the supertype (`supertype.py`), the casts (`casts.py`) and the options of a node (`options.py`). Then come the type functions (`type_functions.py`), the cast pass (`cast_pass.py`), the run in a `select` and in `agg` (`execution.py`), and the kernels (`kernels/`).
- `definitions/` holds a definition for each node of polars, one class for each family. A definition names the part of the vocabulary at each place of its node, and says when its output holds only nulls: from the nulls of its inputs (`NullRule`), or from the few rows of its first input (`ShortNulls`). A later cast reads that fact. `registry.py` finds a definition by the identity of its node. `overrides.py` holds the few rules that one node needs and that no part of the vocabulary states.
- `node_evaluation.py` gives the four answers of one node from the states of its inputs. They are the planner type and the run type, in a `select` and in `agg`.
- `tree.py` reads the tree that polars builds for an expression. `tree_evaluation.py` gives the state of each node from the states of its inputs (`T-25`).

The tests of `_rules/` are in `spec/tests/`. Each module has a test against polars, and each node has a sample that runs against live polars (`M-21`). `spec/tools/rules/` checks random trees of the census nodes against polars (`M-20`).

### 4.2 The code and its evaluation

`_hooks/deferred/` reads the code of a type, on plain values.

- `code.py` holds the code as canonical text. `$n` refers to a declared type of the code. A part that the code uses twice is written once as `($k := (<code>))`. `CodeTree` composes the code of a call from the codes of its parts.
- `code_reader.py` builds the polars expression of a code again. It parses the text with `ast.parse` and interprets it with an allowlist, so it executes no text. A hole, a value that the checker cannot know, becomes a placeholder value that a rule recognizes. A call that takes a function of the user becomes a column with a reserved name. A function of `pl` that runs a function of the user over several expressions, as `pl.fold` or `pl.map_batches`, becomes such a column over a struct of its inputs, which names the output as polars names it (`GatheredCall`).
- Polars cannot write the tree of some calls. It refuses a `replace` over a list of expressions at every frame, so the reader states that refusal itself (`ObjectValues`). It cannot write the tree of a function that it builds around a function of Python of its own, as `pl.cum_sum_horizontal`. It still names the output and the columns that the function reads, so the evaluation keeps those under `Unchecked` (`UnwrittenTree`).
- `evaluation.py` evaluates a code at the columns of a frame. It builds the expression with the code reader, reads its tree, and evaluates the tree with `_rules/`. It reads each data type of the frame from its rendering (`Rendering` in `_rules/dtypes.py`). It answers the run type in a `select` or in `agg`, the planner type for a fit, the type that a lazy frame stores, a refusal of polars, or `Unchecked`.
- A lazy frame plans each step at the type that it stores for a column, and runs on the data (`D-45`). Where the two types differ, the column holds `Planned[run, planner]`. The leaf of such a column holds both (`ColumnLeaf.planned`), `collect()` gives the run type (`Frame.collected`), and a frame operation that checks a type reads the stored type (`PlannedType` in `_hooks/domain/planned.py`).
- A hole takes each value of a fixed sample of its type (`D-44`). The answer stands where every value that polars does not refuse gives it. A function of the user gives the type that `return_dtype=` names, or `Unchecked`.
- `open_types.py` reads a data type with holes: `Unchecked`, a family such as `NumericType`, a bare `Datetime`, or a nested type that holds one. The evaluation tries each data type of a fixed sample that the declaration allows. It drops the data types where polars refuses, and keeps what the other answers share (`Generalization`, `T-26`).
- `every_frame.py` evaluates a code with each column that it reads open, in a `select`, in `agg` and in a lazy `select` that holds the code alone. The recorder uses it to report an expression that polars refuses at every frame.
- `polars_selector.py` reads a selector of polars, in its serialized form, as a rule of `remainder/`. It also reads the names that the selector requires the frame to hold (`NameRequirement`).
- `remainder.py` gives what the code of a selection does to the remainder of an open frame (`T-20`). It evaluates the code at a probe column of each data type of the sample of each kind that the selector can pick and the record allows. Where polars refuses some of them and runs at the others, the result depends on the caller. It reads the rename from the output names of two probes.
- `fit.py` checks the code of an argument against a declared expression type (`D-40`). It evaluates the code at each data type of the sample of each cell of the declaration, so a unit cast does not fit `same` over `Datetime`. It compares the reads, the output name, the character, the type in each context, and the planner type with the run type. `selection_fit.py` checks a code against a declared selection.
- `argument.py` reads a deferred argument of a frame operation into those plain values. It gives the code of each member of a union, and the declared column of each leaf of a code. A leaf keeps its declaration, so a choice of data types where the declaration states no outcome refuses (section 17.1 of `spec/OPERATIONS.md`).

### 4.3 Declarations

A declaration is a declared map (`D-39`). `_hooks/expression/` holds its engine values.

- `fit.py` holds `SliceDecl`: the reads, rows, name and character of a declaration. `WellFormedness` checks the rules of a declaration on its own, for example that its regions partition and that a blank read binds once. `Fit.disagree` compares one answer with one declared outcome.
- `outcome.py` holds the three outcomes that a declaration states at a cell: `Const`, `Axis` (`same`) and `Wrap`.
- `table.py` holds the map from cells to outcomes of a declaration. `kind.py` gives the kinds that index a cell, from the class hierarchy of polars.
- `quadruple.py` holds `ExpressionType`: the reads, map, output name and character of a declaration, as the codec of section 5 writes and reads it.
- `selection.py` holds `SelectionType`, a declared selection, and `Affix`, the one rename that a selection admits. It also holds `RemainderEffect`, what a selection does to a remainder.
- `lifting.py` collapses several answers into one. `character.py` names the three characters, `context.py` the two contexts, and `rendering.py` the text that polars prints for a data type.

### 4.4 Materialization

The consumers of section 6.4 read an argument through `RequestResolver` in `expression_requests.py`. The resolver evaluates a deferred argument at the declared columns of the receiver. A column that the code reads and the frame does not declare is an error. So is an expression that polars refuses at the data types of the frame. The text of a refusal gives the reason in the words of the checker, then the class of the error of polars. A selection gives one column for each declared column that its selector picks. On an open frame it also gives an effect on the remainder, kind by kind. A union argument is evaluated member by member (`D-42`). `MemberSplit` gives a union of frames for an argument of two members, one frame for each member, and merges the answers of an argument of more members, which a loop can leave. A declared expression type, as a parameter that a body passes on unchanged, is read at its declared cells (`materialize.py`).

______________________________________________________________________

## 5. The encoding

`_hooks/encoding/` turns engine values into checker types and back. It has one contract:

> **Decoding an encoded value gives back the same value.**

It is a serialisation format, chosen for being simple and total, not for producing useful subtyping. And the rendered type *is* the encoding: what `reveal_type` prints is the canonical form, so an error message names what the user wrote whenever a named marker spells it.

### 5.1 The codecs

| Codec | Engine value | Spelling |
| -- | -- | -- |
| `remainder_codec.py` | `Canonical` | `Retyped[Prefixed[Suffixed[Only\|Except\|Minus[base, ...]]]]` over a declared class, `Both[...]`, or `SplitColumns[...]` |
| `selector_codec.py` | `Selector` | `OfType[Family]`, `OneOf[...]`, `StartsWith`, `Matching[Literal[spelling]]`, `SelAnd`, `SelOr`, `SelNot` |
| `recording/deferred_types.py` | a `Code` | `TypedExpr[Deferred[Literal[code], *leaves]]`, and the same code under the head of the class that polars gives the call, such as `TypedWhen` |
| `expr_codec.py` | `ExpressionType`, a declaration | `TypedExpr\|TypedAgg\|TypedLit[Reads[Column[name, region], ...], gives]`, the identity collapsed to `TypedColumn[name, region]` |
| `selection_codec.py` | `SelectionType`, a declared selection | `TypedSelector[C]`, `TypedSelection\|TypedAggSelection[C, Rows[...], Rename]` |
| `declaration_codec.py` | a declaration's requirements and record | the marker bases of the class `other_columns(...)` registered |

`marker_minter.py` builds and reads the marker instances every codec is made of, over a literal builder for the phase it runs in (semantic analysis or checking). The blank output name is the `Blank` marker. A read at every data type is `EveryDType`, a read at exactly one polars family's members is that family's base class, and any other read is the union of its region's kinds. The cells of a declaration that share one outcome spell as the rectangles of a fixed cover, so a declaration whose answers fall in bands costs a row per band rather than a row per cell.

### 5.2 The codec is the only mint

Every type that exists was produced by a codec from a canonical engine value. No hook assembles a marker instance by hand. This is what makes section 7 true: a hand-assembled type may be well-formed and not normalised, and it would compare unequal to its own normal form in a position where nothing is watching.

The recorder writes one canonical text for a code: `pl.col("a")` and `pl.col('a')`, or two orders of keyword arguments, give one code. A loop that rebinds its variable (`x = f(x)`) still grows its code on every pass, so mypy, with `allow_redefinition = true`, leaves the union of the versions of its first passes (`M-6`), and a frame operation reads each member (`D-42`).

### 5.3 Where the encoded form lives

In the type, always. Not in a table beside it. mypy's incremental cache reuses the serialised types of an unchanged module and never re-runs the code that populated a side table, so anything kept beside the type is present on a first run and missing on the second.

mypy reuses the cache of a module while the value that the plugin reports for it (`report_config_data`) stays the same. That value holds the settings, the modules that the plugin adds to each build, and a fingerprint of the source of the package (`_hooks/code_fingerprint.py`), because mypy hashes only the module that loads the plugin, and a changed rule must not keep the results of the earlier code.

- A remainder variable is the class `other_columns(...)` registered, and the requirements it declares are marker bases of that class, read off the `TypeInfo` alone.
- `Matching` carries a text that only its names fix (`NameSet.spelling`): a regular expression where the minimal automaton has no loop but a state that keeps every character, as `ab.*` or `a(b.*)?`, and the text of the automaton otherwise. The conversion of a loop grows fast, so it stops there. Two rules with the same names give one text, and equality is on the automaton.
- An expression type carries its whole code as a literal, and the declaration of each leaf as a type argument.
- The inferred return type of a function, the union of the values its body marks with `infer(...)`, lives beside the function under a hidden name in the symbol table that holds it (`_hooks/inferred/cached_variants.py`). mypy caches that table with the module, so a call from another module reads the type on a warm run too. The record in plugin memory serves the first run, before mypy has checked the body. A call that mypy checks before the body finds no record. If the body returns an `infer(...)` call, the hook asks mypy to check the function that holds the call again, after the body, by the deferral that mypy uses for a type that is not ready (`_hooks/domain/deferral.py`). mypy defers only a function at the top of a module or a class, and only before its last pass. A call outside a function, or in a body that calls itself, keeps the placeholder. The values of a function with an open frame parameter are written over its own remainder variables. So the fit of each call (`fit.py`) keeps its binding in `_hooks/inferred/call_evaluations.py`, and the hook that types the call uses it to write each value in the caller's terms. A frame return type whose remainder is `InferredRemainder` keeps its declared columns, and `InferHook` checks each value against them at its `infer(...)`.

The marker classes live in three public modules, grouped by what they encode. `remainder.py` holds `other_columns`, the remainder effects (`Only`, `Minus`, `Both` and the others), the name and type tests (`StartsWith`, `OneOf`, `OfType` and the others) and the requirement records. `expr.py` holds the expression heads, `Deferred`, and the markers of a declaration (`Reads`, `Row`, `Same` and the others). `declare.py` holds the declarations and the inferred return types. The modules are public because mypy prints the full name of a marker in every message and every `reveal_type`.

Every encoded type is invariant, declared explicitly. Invariance means "the same type or unrelated", which is the verdict section 7 wants.

______________________________________________________________________

## 6. The plugin layer

### 6.1 Dispatch

`plugin.py` is the entry point. `_hooks/dispatch/` registers every hook against the mypy fullname it handles: type-analysis hooks for the marker grammar, method hooks for frame operations and expression members on both frame classes and every expression head, function hooks for the `pl.*` constructors and selector roots, dynamic-class hooks for declarations, a signature hook for fit, an attribute hook for `.schema`, and a class hook for the two frame classes. The class hook gives the remainder parameter of a frame class the default `None`. So a frame type written as a value, as in `check_schema(raw, TypedDataFrame[Trades])`, takes one argument, as an annotation does. Every hook is a class constructed with its dependencies and called with a context; a fullname is derived from the real object, never typed by hand. mypy names a method or an attribute by the class that the call goes through, so a class that a user derives from a typed frame has names of its own. `TypedBaseRoute` sends such a name to the same member on the typed base, unless the derived class defines the member itself, and `Frame.from_proper_type` reads an instance of the derived class as its typed base.

A typed frame or series declares a method that changes no column, such as `head`, with `*args` and `**kwargs`, so that it follows polars at any version. `passthrough_signature.py` finds those methods, and its signature hook gives each call the parameters of the polars method, so mypy checks the arguments as polars declares them. A class derived from a typed frame gets the same hook through the route of `dispatch/typed_base.py`.

### 6.2 Frame hooks

Each frame operation is a thin shell: decode the receiver (`Frame.known_columns` for the declared half, `RemainderKnowledge` for the remainder's record), ask the engine, encode the result. `RemainderEdit` is the round trip one effect needs on the raw remainder slot: decode, apply, normalise, encode. It offers the effects by name (`minus`, `only`, `excepted`, `suffixed`, `prefixed`, `retyped`, `both`) and `with_effect`, the application of a selection's effect beside kept columns. `concat_remainder.py` and `join_remainder.py` decide the two binary operations' remainders, or name the column at risk.

`rolling` and `group_by_dynamic` give a group-by over windows of the rows of an index column. `group_by/windows.py` reads the index, the keys and the durations, checks them against the data type of the index, and mints one of four typed group-by classes of `frame.py`. The last argument of such a class names the windows: `rolling`, `dynamic` or `disjoint`, with ` bounds` where the call asks for the bounds of each window. `agg` gives that name to the evaluation, which takes a fixed set of windows in place of the groups (`DeferredEvaluation.WINDOWS`). `group_by/window_columns.py` gives the columns before the aggregations: the keys, the bounds and the index.

A frame hook that cannot resolve its call answers the plain `pl.DataFrame`/`pl.LazyFrame` through `domain/untyped_fallback.py`. A hook that knows a name but not its data type keeps the name under `Unchecked` through `domain/unchecked.py`. Both are an `UnprovenGate` (`domain/unproven_gate.py`), which is where `allow_unproven` is enforced. See 6.7.

### 6.3 Recording

`_hooks/recording/` writes the type of each expression. `dispatch/recording.py` registers a few generic hooks for every call that gives an expression. They take each method of a typed expression or of one of its namespaces, `pl.col`, each function of `pl` and each selector of `cs`. They also take the namespace attributes, such as `.str`, and a comparison whose literal stands on the left.

The hooks know nothing about any node. `RecordedCall` and `RecordedFunction` write the call as the user wrote it, from the codes of its parts (`ArgumentCode`). An argument that is an expression gives its code, and a literal gives its text. A value whose type is a `Literal` gives that value. Any other value is a hole that keeps its Python type. `DeferredMinter` writes the code in the type, under the head of the class that polars gives the call. Where a part is a union, as after an `if` and an `else`, the hook records the call for each member and gives the union.

The recorder reports at the call what polars refuses whatever the frame. That is a value that polars refuses while it builds the expression (`argument_check.py`), or an expression that polars refuses at every frame (`deferred/every_frame.py`). `RecordedShortForm` records the expression that a frame method runs for the user, as `pl.all().sum()` for `group_by(...).sum()`. The hook of the other operation then types it.

### 6.4 Consumers

`expression_requests.py`'s `RequestResolver` materializes one argument of a frame operation against the receiver: a deferred code gives the column of its evaluation (section 4.4), a selection gives the picked columns and a remainder effect, an unnest spreads the fields of the struct its input gives, a declared expression type gives the column it declares, and a bare literal or plain expression is salvaged by name and kept under `Unchecked`. `expr_evaluation.py` is the boundary the resolver reads through, and `schema_renderings.py` turns column items into the renderings the engine reads and back. `select`, `with_columns`, `group_by/agg.py`, `filter.py`, the keys of `sort` (`sort_keys.py`), the predicates of `join_where` and `pl.select` consume through the resolver. Each consumer tells the resolver what its result holds for a column that a selector does not pick (`Unpicked`). Where the data type of a column decides the pick, `select`, `agg` and `pl.select` give up, `with_columns` keeps the column with the type that both answers share, and keys and predicates change no column. `join.py` reads the code of an expression key only to tell a bare column from a key that changes its column. `join_key_tree.py` reads the tree of each key of a join: polars refuses an alias anywhere in it, and a declared expression leaves the tree to the caller. `pivot.py` reads literal names.

### 6.5 Declarations

`declare.py` in the public package holds the runtime identities: `column`, `typed_expr`, `typed_agg`, `typed_lit`, `row`, `some_column`, `same`, `wrap`, `column_of`, `subset_of`, `typed_selection`, `typed_agg_selection`. It also holds the inferred return types `InferredExpr`, `InferredDataFrame` and `InferredLazyFrame`, the remainder marker `InferredRemainder`, and `infer`, which marks one candidate return value. Each call is an ordinary value at run time, and a named value the plugin reads during semantic analysis through the dynamic-class hooks in `declaring/hooks.py`. `declaring/parser.py` reads a slice declaration into a `SliceDecl`, `regions.py` and `reads.py` and `outcomes.py` read its parts, `selectors.py` reads a selector off the call's syntax, and `selection_parser.py` reads a selection declaration. A broken rule is `[malformed-declaration]` at the declaration. An unresolved forward reference is skipped silently on that pass, never deferred: a dynamic-class hook that defers beside another one that re-registers its class on every pass makes mypy loop to its iteration limit.

### 6.6 Fit

Two hooks decide fit.

`fit.py` is the signature hook for a call whose parameter is `TypedDataFrame[Schema, Rest]` or a declared expression type. It serves a function call and a method call. A method call reads the bound signature, which has no `self`, so the hook builds its per-kind hooks again at each method call. A closed argument binds the remainder to its own leftover columns and the declaration's requirements are decided by inspection; an open argument binds it to its own remainder slot and the requirements are proven from that slot's record. A call that mixes closed and open arguments gets a return type in two halves: the columns that the closed arguments give join the declared ones, and the open arguments give the remainder (`T-24`). `expression_fit.py` runs first in the same chain and holds each declared expression parameter to the fit of `deferred/fit.py`, binding a blank read to the argument's name across parameter and return. Having decided, the hook sets each parameter type to the argument's own type and evaluates the declared return under the binding, so what it hands back mentions no remainder variable and the checker compares a type with itself.

`recording/context_fit.py` decides fit at a return statement and an annotated assignment: the recorder reads the declared type context mypy offers and answers the declared type whenever the recorded code fits it, so `return price * 0.9` under a declared `Price` type-checks without a call. A value that does not fit gets the refusal as one `[expression-fit]` error, and the declared type back so mypy adds nothing of its own. Mypy checks a binary operator by trying one side with its errors filtered, and where it then takes the other side it drops both those errors and that side's result, so a refusal under an operator that has a usable reflected form is lost and mypy reports the position itself.

`_hooks/inferred/position.py` reports an inferred return type that stands anywhere but the function's own return type. The check runs in the same signature hook at each call, and around each frame method hook for a receiver whose remainder is `InferredRemainder`. `infer` has its own signature hook, `InferSignatureHook`: it takes the type variable out of `infer[T](x: T) -> T`, so that the expected frame type of a `return` does not decide `T`.

### 6.7 Configuration and diagnostics

`config.py` reads `[tool.astrolabe]` and warns about a malformed setting. Every check-time error is a class in `errors.py` with a canonical code, built from structured arguments, and the code is what a `.yml` case asserts. The `Unchecked` policies report an introduction once, at the operation that introduced it, naming the fix; a consumer holding an answer already signed `Unchecked` owes no second report.

`unproven.py` holds the `allow_unproven` setting as data. Every site that gives up on a proof names a `Reason`, and each reason has a `Level`: `stated`, `omitted`, `possible` or `asserted`. Two reasons sit off the scale: a checker gap, reported only at `never` or under `report_checker_gaps`, and `REPORTED`, a path where a hard error already fired, which is always silent. `UnprovenPolicy` compares the level of the reason to the level of the operation, its override or else the global level. `UnprovenGate` reports what does not pass, then a `note:` that names the setting which lets it pass.

A plan builder carries its reason through `propagate.py`. A step writes `not_none(value, reason=...)`, and a boundary marked `explains(default)` returns a `Refusal` with that reason instead of `None`. The hook hands the reason to its fallback, and `not_refused` passes a refusal through a nested boundary unchanged. A step after a hard error names `REPORTED`, so one call reports once.

The checks that the remainder of an open frame decides report through the same gate: an undeclared name the remainder can hold (`RequestedColumns` and the request resolver), a join or concat whose result depends on the caller (`JoinRemainderStep`, `ConcatRemainderStep`), a requirement the caller cannot prove, and an untyped argument to a declaration (`fit.py`, `expression_fit.py`).

______________________________________________________________________

## 7. Where each question is answered

Every question has exactly one place it is asked.

| Question | Asked where | How |
| -- | -- | -- |
| may this frame operation run on this receiver | the operation's hook | decode, know the remainder, evaluate the precondition |
| what is a frame operation's result | the same hook | apply the effect in the engine, normalise, encode |
| what does this expression give | the recorder, at its outermost call | write the call as code, from the codes of its parts |
| what column does this argument produce on this frame | the consumer's request resolver | evaluate the code at the receiver's renderings |
| does this argument fit its declared expression type | the signature hook | evaluate the code at each data type of each declared cell |
| does this frame argument satisfy its parameter's requirement | the signature hook | bind, know, decide by containment |
| does a returned or assigned expression fit its declared type | the type-context fit of the recorder | the same clauses |
| does this remainder marker mean anything | the type-analysis hook | decode it; an undecodable spelling is `[marker-arg]` |
| what frame does a runtime schema call give | the function hook of `check_schema`, `force_schema` and `assume_schema` | close the schema of the frame type as an annotation does (`ClosedSchema`). A remainder is `[frame-type-not-closed]` |

A position with no hook is decided by the checker alone. Section 8 explains why that is safe, and section 11 records what it costs.

______________________________________________________________________

## 8. Normalisation, and why structural comparison is enough

> **Normalise before encoding. Every type that exists is in normal form.**

The remainder codec encodes a `Canonical`, and every mint site goes through `Expr.normalise()` first: a hook's result, an annotation the user writes (`Minus[Minus[R, "a"], "b"]` comes back as one filter), and a substitution at a call, which the signature hook performs itself because it evaluates the declared return under the binding rather than leaving the checker to substitute.

The expression codec accepts only canonical declarations: reads sorted by name, rows in the tie order of the declared map, a single-region read collapsed to the two-argument display, the identity collapsed to `TypedColumn`. The rectangle cover that spells the rows reads only the cells, taking them in sorted order and growing each along the reads in order, so one declaration always spells one way. A deferred type is its code, which the recorder writes as one canonical text. Two codes with one map but different text, as `a + b` and `b + a`, compare unequal, and the fit decides on the map wherever a declaration expects one (`D-33`, `D-40`).

Where the checker compares two of our types with no understanding of them, it compares structurally. Because both are in normal form, structural equality *is* engine equality for types we minted. The checker reaches the right verdict by accident, and the accident is guaranteed by construction.

### 8.1 What the checker mints on its own

The claim holds only for types a codec produced. The checker mints remainder-slot types itself:

| Where | What appears in the slot |
| -- | -- |
| `a if flag else b` over two frames | a union of two remainders |
| `[a, b]` with differing remainders | `object` |
| inference with nothing to constrain it | `Never` |
| `isinstance(x, Frame)` narrowing | `Any` |
| a call no hook sees (`functools.partial`, a decorated function, an unpacked `*args`) | an un-normalised nesting |

A union of remainders decodes to a `Union` base: a name is absent only when absent from every member, which is the specification's rule for a result that depends on which case holds. `object`, `Never` and `Any` carry no remainder: decoding yields nothing, and the frame is treated as having an unknown remainder rather than being trusted. The decoder must never assume it is looking at its own output, so decoding an arbitrary well-formed type must not crash: every read of a marker's argument is fallible, and an undecodable spelling is reported.

### 8.2 A resource limit

Where two operands of a `Both` share a subtree, mypy's own substitution copies rather than shares structure, and the encoded type doubles in size per level:

| nesting levels | time | memory |
| -- | -- | -- |
| 16 | 4 s | 0.4 GB |
| 18 | 12 s | 1.3 GB |
| 20 | 44 s | 4.9 GB |

Normalisation does not change these figures, so more careful normalising cannot fix the growth. Two things reduce the exposure: `Both` is the only binary case, and its operands must be provably disjoint, which rules out combining a frame with something derived from itself. Reaching a dozen levels takes deliberate effort.

______________________________________________________________________

## 9. Invariants

1. **The engine does not reference the type checker.** Enforceable by inspection of imports under `_hooks/remainder/`, `_hooks/expression/`, `_hooks/deferred/` (apart from `argument.py`) and `_rules/`.
2. **Decoding an encoded value returns the same value.** Every codec has the round trip as a doctest.
3. **Every type we mint is canonical, and the decoder is total over types we did not mint.** The codec is the only mint; every decode-side read is fallible.
4. **No behaviour depends on the checker's variance, subtyping, or normalisation of our types.**
5. **A signature hook returns a signature whose every parameter type is the corresponding argument's own type, and which contains no remainder variable.** The first clause is what delivers section 1; the second alone does not.
6. **Everything a declaration states is recoverable from its `TypeInfo` alone.** Warm cache included.
7. **A dynamic-class hook never defers.**
8. **A pass that mypy will repeat reports no unproven result.** A deferred call holds `Any` in that pass, and `UnprovenGate` waits for the pass that has the real type.

Invariant 5 is the load-bearing one. The argument that the checker's opinion cannot hurt us rests on never leaving it anything to have an opinion about at a call site.

______________________________________________________________________

## 10. The mutation harness

Coverage proves that a line ran. It does not prove that a test fails when the line is wrong. `tools/mutation/` measures that difference.

The harness changes one operator, constant or branch in `src/astrolabe`, runs the tests that cover that line, and records whether a test fails. A mutant that no test kills names a rule that nothing pins down. The harness is an offline audit, and it is not one of the project's verification commands.

| Class | Module | Work |
| -- | -- | -- |
| `CoverageMap` | `coverage_map.py` | maps each source line to the yml cases that run it |
| `MutantCatalog` | `mutants.py` | enumerates the changes, with cosmic-ray's operators over a parso tree |
| `MutantRunner` | `runner.py` | patches one worker's copy, runs the selected tests, restores the file |
| `MutantQueue` | `sweep.py` | hands each mutant to whichever worker is free |
| `ResultStore` | `store.py` | writes each finished mutant, so a stopped run can go on |
| `SourceDigests` | `store.py` | marks each mutated file, so a resume over changed source stops |
| `MutationCommand` | `command.py` | reads the command line and runs one sweep |
| `MutationReport` | `report.py` | scores each module and lists the survivors |

Three decisions carry the design.

**Test selection makes the audit possible.** A whole-suite run for each mutant is not possible, because the scope holds thousands of them. Coverage is at 100%, so one baseline run with contexts names a killer for every mutable line. Each mutant then runs the tests that reach it, and not the rest. The selection is ordered cheapest first, because the runner stops at the first failure.

**The project's own source never changes.** Each worker owns a copy of the project, and `PYTHONPATH` points the tests at that copy. Each worker reads `astrolabe.__file__` before it starts. A worker that reaches the real source reports every mutant as survived, and the run then says nothing.

**The mypy cache is cold for each mutant.** mypy detects a plugin change from the bytes of `plugin.py` and from its `__version__`, which astrolabe does not set. A mutant elsewhere in `_hooks/` leaves that snapshot the same. A body change also leaves a module's interface hash the same, so dependent modules come from the cache. A warm cache therefore invents survivors.

Eight filters keep the report readable. Each one drops a mutant that survives every test whatever the tests say, so each one removes noise and not evidence.

| Filter | Drops | Why it cannot change an answer |
| -- | -- | -- |
| `AnnotationSite` | a mutant inside a type annotation, and not in the value of an annotated assignment | every module imports `annotations` from `__future__`, so nothing evaluates an annotation |
| `TypeCheckingSite` | a mutant on an `if TYPE_CHECKING:` guard | the guard is False while the code runs, and the block holds imports alone |
| `PragmaSite` | a mutant on a line that carries `# pragma: no cover` | the project marks a line that cannot run this way |
| `IndexOffset` | the downward change of a number that indexes a sequence | a guard above holds the sequence to one member, so `[-1]` reads the member that `[0]` reads, and the upward change stays |
| `FrozenFlag` | `frozen=True` turned into `frozen=False` on a dataclass | no code relies on the immutability of these classes |
| `StrictZip` | `strict=True` turned into `strict=False` in a call of `zip` | each strict `zip` pairs two sequences of one length, so both forms give the same pairs |
| `SplitBound` | `maxsplit=1` raised to 2 in `split(...)[0]` or `rsplit(...)[-1]` | more cuts do not change the first piece of a split or the last piece of an rsplit |
| `KeywordMarker` | the bare `*` of a signature turned into `/` | every call passes the parameters after the `*` by keyword, and they bind the same values after a `/` |

cosmic-ray itself never offers a `|` inside an annotated assignment, in the value too. A value of that shape therefore gives no mutant, so the source keeps a `|` out of the direct value of an annotated assignment.

Every worker takes its next mutant from one queue. A fixed share for each worker balances how many mutants it runs, not what they cost, and the costs differ by two orders of magnitude. The worker that draws the expensive share then runs long after the others stop, and the run ends with one worker busy and the rest idle.

**The harness keeps `--mypy-same-process`, and holds the memory down by batching.** The flag runs each yml case's mypy build inside the pytest process. It is not optional here: mypy in a process of its own reads the plugin from the installed package and not from the patched copy, so every mutant of a hook lives through the yml case that would catch it. The flag also makes one process hold the state of every case it builds, and a thousand cases reach tens of gigabytes. The call therefore runs the selection in batches, and each batch is a process that gives its memory back when it ends. One worker then holds under two gigabytes over a selection of any size.

The worker count follows the memory of one worker, and not the count of the cores. Each worker runs a full pytest process. Measure what one worker holds before you raise the count. Each run also takes a lower priority, so a long audit leaves the machine usable.

One limit bounds each mutant's tests, and it is the same number for the whole run. A mutant that passes the limit gets its own outcome, and it counts towards no score, because a selection here can hold hundreds of tests and a slow selection does not say that a test caught the mutant. pytest writes one mark for each test that ends, and the harness reads that output as it arrives. A run that wrote nothing for ten minutes before the limit is `hung`: it is stuck in one test, which points at a loop with no exit. Any other run past the limit is `timed out`. A limit that grows with the slowest run seen so far is worse than a fixed one: one slow selection then raises the cost of every mutant after it, and the waste compounds.

An operator comes from cosmic-ray, and one of them raises on a node shape it does not read. `ExceptionReplacer` raises on a dotted name inside an `except` tuple. The runner catches that and reports the mutant as errored, because one operator's defect must not end a run of several hours.

A filter cannot catch every mutant that nothing can reach. A mutant that removes `@staticmethod` from a method which only the class calls is one, because Python binds a plain function reached through the class in the same way. To separate that from a real change needs the name resolution that a filter does not have. Triage therefore meets a few such mutants in each report.

A full audit runs for hours, so `ResultStore` writes one line for each mutant as it lands. A later run reads those lines and runs only what is left. The file is plain JSON lines, so a run can be read while it runs. A retry run reads a mutant that timed out as unfinished, which decides that set under a longer limit without running the rest again. It keeps a mutant that already ran out a limit at least as long, and a hung mutant, because a second run can only end the same way. A run with `--retry-survived` reads each mutant that survived, or that no yml case reached, as unfinished. After new yml cases and a new coverage map over the same source, it decides only that set.

The store also marks each mutated file by its content. A result names its mutant by a path, an operator and a count of that operator's positions in the file, so one new position moves every position after it. A resume over changed source would then attach a held verdict to another change. The store compares the marks and stops the run, and it names the file that changed.

**Only a yml case kills a mutant.** A yml case runs the checker over a small program and states the answer a user reads: the revealed type, the error and its code. That answer is the checker's contract. A doctest sits inside the code it runs, so a change to that code carries its example along. A unit test or an integration run checks a class or a setting from inside the plugin. A mutant that only those catch is pinned by the implementation, and not by what the checker answers. The parity tests of `tests/engine/` count as yml cases here: each one runs the engine and the Python model of `spec/` on one input and compares the answers, so it checks what the checker answers against the design authority. `YamlCases` holds every other test back from the selection, and `--all-tests` lets them in to measure how much of a score they carry. A line that only those other tests run is `unreached`: it has no test that can kill its mutant, and it counts towards no score. A line that runs only while pytest collects the modules names no test at all, so it falls back to every yml case of its file.

______________________________________________________________________

## 11. What this gives up

This section lists the weaknesses that the design accepts. `KNOWN_ISSUES.md` lists the bugs, and the gaps that a later change can fix, and it exists only while there is one. `POLARS_ISSUES.md` lists the polars behaviors the checker works around.

**A join that can rename is rejected, not described.** A join over a generic frame with no promise ruling out a collision is an error rather than a result. The user lifts it with one promise on the declaration.

**Weakening works at calls, not at returns.** A caller promising more than is demanded is accepted at a call site, where the hook decides. The same relationship is rejected at a return statement between two distinct remainder variables, because a checker relates two type variables by identity. Declare one requirement and use it in both positions.

**Every check costs a hook.** A position we do not intervene in gets structural comparison, which is sound but says nothing. Section 7 exists so that adding an operation is a deliberate step.

**A node that no definition states reads as `Unchecked`.** The rules cover every node that the census of the polars API meets. A node that polars adds in a later version gives `Unchecked` under the name of the expression. `allow_unproven` reports it only at `never` or under `report_checker_gaps`.

**An answer at a sample stands for every value of its type.** A hole, a value that the checker cannot know, is read at a few values of its Python type. A column of an open type is read at a few data types of each kind (`D-44`). Where every member of the sample gives one answer, the checker takes that answer for every value and every data type. The samples sit where the rules of polars change their answer: an integer width, a sign, a scale, a unit or a zone. A rule that changes its answer between two members of a sample gives a type that is wrong for some values.

**A loop whose first versions agree is taken to agree at every pass.** mypy stops its analysis of a loop after a few passes. The code of an expression grows on every pass, so the type holds the versions of the first passes only (`M-6`). Where they agree at a frame, the checker takes that answer for any number of passes (`T-18`). A seed declared with an `Unchecked` output gives the loop one type.

**A remainder kind that the checker has no sample for reads as `Unchecked`.** The sample holds no data type of the kind `Extension`. So a remainder column of that kind takes `Unchecked` under a selection with any operation, also one that keeps the data type, such as `fill_null(strategy="forward")`. A selector with no operation keeps the column as it is.

**A variable is not a type annotation.** `Stamp = pl.Datetime("us")` reaches every declaration that takes its data type as a call argument, because `DtypeAliasHook` registers the data type beside the variable at the assignment. A `Columns` field is a type annotation, and mypy refuses a `Variable` there before any plugin hook runs. Name the data type with its marker in a field: `Stamped = Datetime[Literal["us"], AnyZone]` is a type alias, and every declaration reads it.

**The class `pl.Decimal` inside a nested data type reads as a `Decimal` of unknown precision.** `pl.List(pl.Decimal)` and `pl.Struct({"a": pl.Decimal})` hold the class, and polars decides only where the data type is used: a frame schema reads it as `Decimal(38, 0)`, and an expression raises `TypeError`. The checker cannot tell that class from an instance whose type names no precision, for example a parameter annotated `pl.Decimal`, which polars accepts everywhere. So it reads both as a `Decimal` of unknown precision, and a cast or a literal of such a nested type gets no error. A marker for the class would close the gap, at the cost of one more type in every message and every reader of data types. The class at the top level is exact (`D-3`).

**A vertical stack compares columns by name.** A schema has no order, so `.vstack`, `.extend`, `.merge_sorted` and a vertical `pl.concat` check that two frames hold the same names and data types, whatever order each type writes them in. polars stacks by position, so two frames whose columns stand in different orders at run time still raise. The type cannot record an order that no operation keeps.

**An inferred return type does not survive mypy's parallel mode.** With `--num-workers`, mypy writes the cache of a module before it checks the function bodies, so the hidden name that holds an inferred return type is not in the cache. On a warm run, a call from another module then gets the placeholder `InferredExpr`, `InferredDataFrame` or `InferredLazyFrame`. A serial run caches it. The deferral of a call that comes before the body works inside one module, in either mode.

**A mutant the harness cannot decide stays undecided.** A mutant can sit in code that most of the suite reaches, and that suite takes over an hour to run in full. The limit stops its tests while they still pass, so no test killed it and the rest never ran. A verdict for such a mutant costs more machine time than the answer is worth.

**A line that runs only while pytest collects has no test to name.** The harness selects a mutant's tests from the coverage contexts of its line. A line in a module of plain declarations runs while pytest imports the modules, and never inside a test. If no yml case runs any other line of that module, the harness has no test to run, and the mutant is unselected.

**One cosmic-ray operator raises on a shape it does not read.** `ExceptionReplacer` writes to a field that a dotted name inside an `except` tuple does not hold, and it raises. The runner catches the raise and reports that mutant as errored, so one operator's defect cannot end a run of several hours. Such a mutant is never measured.

**A column of another length beside a column with one value per row passes.** `df.select(pl.col("b"), pl.col("a").head(2))` and `df.with_columns(pl.col("a").filter(mask))` set a column whose length can differ from the height of the frame. Polars raises `ShapeError` unless the two lengths agree, and they agree only by chance: on a frame of two rows for `head(2)`, or for a mask that keeps every row. The checker accepts the call, because a length that depends on the data is not a property of the types.

**A plain expression is salvaged by name.** A plain expression at a consumer keeps its name under `Unchecked` when an outermost `.alias(...)` or a keyword names it. With no name to keep, the consumer falls back to the plain frame, and `allow_unproven = "none"` or `"never"` reports it.

**A value too wide for a later cast is a fact of the data.** `pl.when(c).then(pl.col("d")).otherwise(pl.when(c).then(pl.col("u")).otherwise(pl.lit(2**63)))` over a `Date` column `d` and a `UInt8` column `u` gives `Int64`. The inner chain makes a `UInt64` column that holds $2^{63}$, and the outer chain casts it to `Int64`. Polars raises where a row takes the literal. The checker answers `Int64`, as it does for a `UInt64` column that holds $2^{63}$ under a cast to `Int64`: the values of a column are not part of its type.

**A cast of a `Struct` to a number answers as a frame with rows.** `pl.col("st").cast(pl.Int64)` over a `Struct` column gives the struct, with each field cast, on a frame that holds a row, and `Int64` on a frame with no row (`POLARS_ISSUES.md`). The type of a frame does not say whether it holds a row, so the checker gives the answer of a frame with rows.

**A cast of an `Unchecked` column gives its target.** `pl.col("u").cast(pl.Int8)` over a column whose data type the checker does not know gives `Int8`. Polars keeps a struct under such a cast, so a `Struct` column would give a struct. The checker takes that as an oddity of polars, and keeps the cast that recovers the type of a column of a plain frame.

**An expression over several open columns reads as `Unchecked`.** The checker tries each data type of its sample for a column whose data type it does not know. Two such columns give too many choices, so `pl.concat_str(pl.col("a"), pl.col("b"))` over a plain frame gives `Unchecked`, although polars gives `String` at every pair of data types. One open column keeps the whole sample.

**A null that the code decides survives only nodes that work value by value.** The rules know that `shift(1)`, `diff`, `pct_change`, a rolling window and an exponentially weighted window that needs two samples hold a null wherever they hold a row, and an elementwise node keeps that fact. So `pl.rolling_corr(pl.col("x").shift(1), pl.col("x").first(), window_size=2)` reports the panic of polars beside one row. A node that keeps some rows only, as `head(1)`, drops the fact, because the rules do not track the place of the null. The checker then accepts `pl.rolling_corr(pl.col("x").shift(1).head(1), pl.col("x"), window_size=2)`, which panics.

**The rules compute the value of few functions of a literal.** A later rule can read the value of a literal input, as the count of `diff` or the text of a strict cast. The rules know that value through a fold of two literals, a node that keeps the one value of a literal, as `first` or `str.join`, and a cast that keeps it. They compute no other function of a value, because their rules give types and not values. So `pl.lit("a").str.to_uppercase().cast(pl.Time)` gives `Time`, and polars raises at every run.
