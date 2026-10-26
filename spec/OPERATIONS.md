# The grammar of partial schemas

This document gives the formal definitions, rules and proofs of the type system, for both of its halves: sections 1 to 12 formalize typed frames, and sections 13 to 18 formalize typed expressions. Read it after `DATAFRAME_SPECIFICATION.md` and `EXPRESSION_SPECIFICATION.md`, which introduce the same system in prose with examples. This document is authoritative: where another document restates one of its definitions and disagrees with it, the restatement is wrong.

## 0. How to read this

### 0.1 What each of the four documents does

| Document | Answers | Read after |
| -- | -- | -- |
| `DATAFRAME_SPECIFICATION.md` | What is a typed frame, and why is it built this way? | nothing |
| `EXPRESSION_SPECIFICATION.md` | What is a typed expression, and why is it built this way? | `DATAFRAME_SPECIFICATION.md` |
| **this document** | What exactly are the rules for both halves, and why are they correct? | the two specifications |
| `OPERATION_CATALOG.md` | What does each polars operation do under these rules? | this document |

This document is the source of the vocabulary. The two specifications restate some of its definitions in prose and cite the identifier. If a restatement disagrees with this document, the restatement is wrong.

### 0.2 Identifiers

Every definition, requirement, effect, theorem, measurement and boundary carries an identifier. A cross-reference cites the identifier in backticks, for example `T-12`, because an identifier stays the same when a document is reorganized. A cross-reference to a section is written "section 6.4" in plain text, and to a section of another document "section 6.4 of `DATAFRAME_SPECIFICATION.md`".

| Prefix | Stands for |
| -- | -- |
| `D-n` | Definition |
| `R-n` | Requirement, written where a remainder is declared |
| `E-n` | Effect, written in a return type |
| `T-n` | Theorem |
| `M-n` | Measurement over a finite space |
| `B-n` | Boundary: something outside the model |

### 0.3 How a claim is supported

Every numbered claim that is not a definition carries a tag, and section 19 lists them all. A claim can carry two tags, proved and measured.

| Tag | Means |
| -- | -- |
| **proved** | Proved in Lean 4 with Mathlib. The claim names the theorem and its file under `spec/lean/`, and a short sketch of the proof follows. |
| **measured** | A test computes it. The test and its file are named at the claim. |
| **assumed** | Taken as given. Its source is named. |

A tag is written in bold with a period, for example **Tag: proved.**, and a qualifier follows it as plain text. A claim with no tag is a defect in this document.

Every **proved** claim is checked by `lake build --wfail` in `spec/lean/`. `spec/lean/Audit.lean` fails the build when a cited theorem uses `sorry` or an axiom other than Lean's three standard ones. A proof that rests on a measured fact takes the fact as a hypothesis of its theorem, such as the field `locality` of the structure `Polars` in `spec/lean/Spec/L6/Node.lean`, and the claim's tag names that fact.

Every **measured** claim is checked by a test under `spec/tests/`, run by `uv run pytest spec`. A measurement over a finite universe is evidence and not a proof: the universe is a special case of the infinite one, so a counterexample found there is real, while a clean run only fails to find one. Where a claim needs a proof it is proved, and the measurement is a guard against the Lean model drifting from the text.

### 0.4 Notation

| Written | Means |
| -- | -- |
| $N$ | the set of all column names, which is every finite string |
| $T$ | the set of all data types |
| $n + s$ | the name $n$ followed by the string $s$ |
| $\llbracket e \rrbracket$ | what the expression $e$ denotes |
| $\beta$ | a binding: one column set for each remainder variable |
| $\llbracket e \rrbracket_\beta$ | the column set $e$ denotes under the binding $\beta$ |
| $e_1 \equiv e_2$ | $e_1$ and $e_2$ have the same canonical form |
| $e_1 = e_2$ | $e_1$ and $e_2$ denote the same column set under every binding |

$\equiv$ and $=$ are different relations, and this document never conflates them. $\equiv$ implies $=$ (`T-8`, proved). The converse holds too (`T-10`, measured): the canonical form folds an effect into the operands of a union, so an effect over a union and the same effect over each operand have one form.

In sections 1 to 12, $e$ is a remainder expression (`D-18`). In sections 13 to 18, $e$ is a polars expression (`D-30`).

______________________________________________________________________

## 1. Objects

### D-1. Name

A **name** is a finite string. $N$ is the set of all of them, and it is infinite. Nothing in this system bounds the names a caller can bring, which is why section 2.1's language must describe infinite sets of names rather than list them.

### D-2. Data type

A **data type** is one of polars' types: `Int64`, `String`, `Datetime("us")`, and so on. $T$ is the set of all of them. $T$ is infinite, because data types nest without bound: `List(Int64)`, `List(List(Int64))`, and so on. The set of **kinds**, the data types with their parameters erased (`D-35`), is finite. Every question this system asks about a data type is a membership question against a data-type set (`D-3`), decided by a lookup, and finitely many such sets divide $T$ into finitely many classes (`T-9`).

### D-3. Family, and the data-type sets a requirement names

A **family** is a named subset of $T$. Polars groups its data types into families, and each family is an ordinary class: `NumericType` holds the integer, floating-point and decimal types, and `NestedType` holds lists, arrays and structs.

A requirement about data types names a **data-type set**: a finite union of kinds and single data types. A family is one convenient way to name one, because every family is a union of kinds. Three spellings all denote a data-type set, and a set of them denotes their union:

| Written | Denotes |
| -- | -- |
| `NumericType` | that family's members |
| `pl.String` | the one data type, as a set of one |
| `pl.Datetime("us")` | the one data type with those parameters, as a set of one |
| `{NestedType, pl.String, pl.Binary}` | the union of the three |

A bare class means two different things, and the place decides which. In a requirement or a selector, `pl.Datetime` names the whole kind, which is what `cs.by_dtype(pl.Datetime)` picks in polars. In a column's own declaration it names one data type, the one polars' own constructor builds with no argument: `pl.Datetime` is `Datetime("us", None)`, `pl.Duration` is `Duration("us")` and `pl.Decimal` is `Decimal(38, 0)`. A column has one data type, parameters included, so a declaration that named a whole kind would say less than the column knows.

Wherever a call takes one data type as a value, polars reads a bare class the same way: the target of a `cast`, a `dtype=` or `return_dtype=`, the values of a `schema=`, and the inner type of `pl.List(...)`, `pl.Array(...)` or a field of `pl.Struct({...})`, which polars reads as the default even inside a selector. There `pl.Enum` is `Enum([])`, `pl.List` is `List(Null)`, `pl.Array` is `Array(Null, 0)` and `pl.Struct` is `Struct({})`. `pl.Decimal` is the exception. A series and a frame built from Python data (`pl.Series`, `pl.from_dict`, `pl.from_dicts`, `pl.from_records`) read it as `Decimal(38, 0)`. Inside an expression, in a cast, in `pl.from_numpy` and in the schema of a function that reads a file, polars raises `TypeError` for it, so the checker reports it. Inside `pl.List(...)` or `pl.Struct({...})` polars keeps the class until the data type is used, and the checker reads it as the kind. Measured by the case `a_data_type_class_stands_for_its_default_instance` and its neighbours in `tests/typing/test_parametrized_dtype_calls.yml`, which the runtime cross-check runs against polars. `pl.Array(...)` needs a shape: a call with no shape, or the shape `None`, raises `TypeError` whatever the element, so the checker reports it, and a shape of several widths nests an array for each width after the first (`an_array_with_no_shape_is_an_error` and `an_array_of_a_shape_of_several_widths_nests_an_array_for_each` in the same file).

A marker class of astrolabe is a type for an annotation, not a data type that polars reads. Wherever a call takes one data type as a value, polars refuses a marker with its arguments, such as `List[pl.Int64]` or `Datetime[Literal["ms"], None]`, the class `Unchecked` and the class `Decimal` of astrolabe, whatever the data, so the checker reports each one. The same three raise in a selector, `cs.by_dtype(...)` and `.exclude(...)`, and a marker with its arguments also raises inside `pl.List(...)`, `pl.Array(...)`, `pl.Struct({...})` and `pl.Field(...)`, the field of a `pl.Struct([...])`, so the checker reports them there too. Inside a nested data type `Unchecked` and `Decimal` behave as `pl.Decimal` does: the data type builds, and each use of it raises. A bare marker class other than `Decimal`, that is `List`, `Array`, `Struct`, `Datetime`, `Duration` or `Enum` with no arguments, reads as the polars class of the same name wherever polars reads one data type, so it stands for the default instance of that class: `List` is `List(Null)` and `Datetime` is `Datetime("us", None)`. A selector reads it as that one instance too, not as the whole kind, because polars picks the whole kind only for its own class: `cs.by_dtype(List)` picks a `List(Null)` column and no other list. `pl.Series` refuses a bare marker for every data, so the checker reports it there. `pl.from_dict`, `pl.from_records` and `pl.from_dicts` refuse it for a list of values and read the default instance for a series or a row, so the data decides, and the checker gives the default instance. Measured by `TestMarkerClassesAsDataTypes` in `spec/tests/test_polars_facts.py`, over a cast, `dtype=`, `return_dtype=`, `schema=`, a selector and a nested data type. A subclass of a polars data type that user code defines, such as `class Tagged(pl.Int64)`, is no marker and no data type of polars either. In a cast, `dtype=`, `return_dtype=`, a nested data type, a selector and the schema of a file reader, polars reads it, or an instance of it, by its name alone: a class named after a kind other than `Decimal` and `Extension` reads as the polars class of that name, whatever it subclasses, so `class Float32(pl.String)` casts to `Float32`, and every other name raises, so the checker reports it there and reads a named class as the default instance of the polars class of its name. An instance of a named class whose kind has parameters holds parameters its type does not show, so the checker reads no data type from it. `pl.Series` refuses every subclass of a kind other than `List`, `Array` and `Struct`, whatever its name and its data, so the checker reports it, and ignores a subclass of those three, so the values decide and the series is `Unchecked`. `pl.from_dicts` reads it by its name in the same way for every data, a list of dicts or an empty list, so the checker reports a class whose name polars does not read there too. `pl.from_dict` and `pl.from_records` refuse or ignore such a class depending on the data, and the checker gives a class whose name polars does not read `Unchecked` in their schema. Measured by `TestUserClassesAsDataTypes` in `spec/tests/test_polars_facts.py`.

This matters more than it looks. Several real operations have a domain that is **not** a family: a whole-frame `cast` accepts `Struct` and rejects `List` and `Array`, so no family names its gate and only a set of two kinds does. A vocabulary restricted to families would have to reject those operations over a remainder, and that would be a limit of the notation rather than of the model.

Families nest, and a fact about a wider family does not settle a question about a narrower one. Every column being numeric does not say whether any is an integer.

$$
\begin{aligned}
\mathtt{NumericType} &= \{\mathtt{Int8}, \ldots, \mathtt{Int128}, \mathtt{UInt8}, \ldots, \mathtt{UInt128}, \mathtt{Float16}, \mathtt{Float32}, \mathtt{Float64}, \mathtt{Decimal}\} \\
\mathtt{TemporalType} &= \{\mathtt{Date}, \mathtt{Datetime}, \mathtt{Time}, \mathtt{Duration}\}
\end{aligned}
$$

### D-4. Column

A **column** is a pair of a name (`D-1`) and a data type (`D-2`), written $n : t$. A column has no other property this system records. It has no position, because `D-6` makes a schema unordered, and it has no length, because nothing here counts rows.

### D-5. Column set

A **column set** is a finite partial map from names to data types. Written out:

$$
\{\mathtt{symbol} : \mathtt{String},\ \mathtt{price} : \mathtt{Float64}\}
$$

A column set is the one concrete thing everything else describes. A frame that really exists has exactly one. Two column sets are equal when they map the same names to the same data types. The order they are written in is not part of them.

### D-6. Schema

A **schema** is a column set (`D-5`) whose every entry the program knows. It is written as a closed `TypedDict` class, usually one that inherits `Columns`:

```python
class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64
```

The set is closed. `Trades` means exactly those three columns.

A column name that is not a Python identifier cannot be a class attribute. `columns("Pivoted", {"1": pl.Int64, "v1|a": pl.String})` builds the schema class with those columns.

A schema class can take a type variable for the data type of a column, bound by `pl.DataType`:

```python
class Keyed[T: pl.DataType](Columns):
    k: pl.String
    v: T
```

`Keyed[pl.Int64]` is the schema $\{\mathtt{k} : \mathtt{String},\ \mathtt{v} : \mathtt{Int64}\}$. Inside a generic function, `Keyed[T]` is a schema whose `v` data type is fixed for each call and not known to the body. An operation that only carries `v` keeps `T`. An operation whose result depends on that data type cannot check it: it gives `Unchecked` for that column, or the plain frame where that data type decides the whole result.

### D-7. Remainder

A **remainder** stands for the columns of a frame that the schema does not name. It is not a column set: it is a *description* of the column sets that are still possible. `D-17` and `D-19` make this precise.

A remainder is introduced by a declaration, which gives it a name and states the requirements every caller must satisfy:

```python
Rest = other_columns()
```

### D-8. Frame type

A **frame type** is a schema (`D-6`) and at most one remainder (`D-7`):

```
TypedDataFrame[ Schema, Remainder? ]
TypedDataFrame[ Remainder ]
```

The second form is a frame with no declared columns: its schema is empty.

The schema and the remainder never overlap. A name in the schema is by definition not among "the columns other than the declared ones", so a declared name is automatically absent from the remainder and no promise is ever needed about it.

There is no third part. Every requirement lives on a remainder's declaration, never inside a frame type, so two frames with the same schema and the same remainder are the same type with no further condition to compare.

**Why at most one remainder.** Two reasons, and the second is decisive.

"The columns other than the declared ones" names one set, so a frame has one remainder. And in a **parameter** two remainders cannot be solved: given `TypedDataFrame[S, A, B]` and a caller with three columns outside `S`, nothing decides which belong to `A` and which to `B`. `E-4` is how two remainders become the one a frame can hold, and `D-24` is how one remainder can be described in parts without becoming two.

______________________________________________________________________

## 2. The three languages

The system reasons in three languages, and keeping them apart is what makes each rule statable. Sections 2.1, 2.2 and 2.3 give one language each.

| Language | Tests | Written | Section |
| -- | -- | -- | -- |
| absence rule $P$ | a **name** | never by a user directly | 2.1 |
| selector $C$ | a **column**: name and data type | in a call, as `cs.numeric()` and its kin | 2.2 |
| type set $\Theta$ | a set of **data types** | never: it is computed | 2.3 |

### 2.1 Absence rules

An **absence rule** is a predicate on names. Where it holds, no column of that name can be present, under any binding.

### D-9. The absence language

```
P  ::=  never                  no name is absent: the unconstrained remainder
     |  always                 every name is absent: the empty remainder
     |  oneOf(S)               absent for the names in the finite set S
     |  startsWith(s)          absent for every name beginning with s
     |  endsWith(s)            absent for every name ending in s
     |  contains(s)            absent for every name holding s anywhere
     |  matching(m)            absent for every name the name expression m accepts
     |  not P
     |  afterStripping(s, P)   n ends in s, and P holds of n with s removed
     |  beforeStripping(s, P)  n begins with s, and P holds of n with s removed
     |  P and P
     |  P or P
```

**Why the three affix constructors.** `startsWith`, `endsWith` and `contains` describe sets no finite list is equal to. "Every name beginning with `sec_`" is infinite. A language of lists alone would force a user joining a thirty-column reference table to write thirty names and to rewrite them whenever that table changed.

**Why the two stripping constructors.** They exist for the two renaming effects and for nothing else. `E-5a` and `E-5b` derive them. Neither is useful alone: each is read together with a negated affix test, and the pair says "this name cannot have come from adding an affix to a name the operand held". No requirement is written in terms of either.

**Why `matching` is last.** It holds an arbitrary name expression (`D-11`). It exists so that a rule derived by the checker always has somewhere to land, and so that a pattern selector has a rule. Any rule that can be built from the named constructors is, because the cheap rules of section 8.4 read those shapes.

`matching` is needed for more than a general selector. Moving a filter past a renaming, which `D-22` does, produces a test that is not always one of the named shapes: filtering on `startsWith("ab")` after appending `"b_"` also accepts a column named `a`, because the affix supplies the rest of the prefix.

> Measured (`M-1`) by `test_a_filter_through_an_affix_is_not_a_substitution` in `spec/tests/test_normal_form.py`, which checks that the moved test holds of `a`, and by `test_the_canonical_form_denotes_what_the_expression_denotes` in the same file, whose expression space holds the suffix `"b_"`. The quotient this case produces, `startsWith("ab") or oneOf({"a"})`, is computed by the machinery of `D-11`.

### D-10. What an absence rule denotes

$\llbracket P \rrbracket \subseteq N$ is the set of names the rule holds of, defined by structural induction: $\llbracket \mathtt{never} \rrbracket = \emptyset$, $\llbracket \mathtt{always} \rrbracket = N$, $\llbracket \mathtt{oneOf}(S) \rrbracket = S$, $\llbracket \mathtt{not}\ P \rrbracket = N \setminus \llbracket P \rrbracket$, $\llbracket P\ \mathtt{and}\ Q \rrbracket = \llbracket P \rrbracket \cap \llbracket Q \rrbracket$, $\llbracket P\ \mathtt{or}\ Q \rrbracket = \llbracket P \rrbracket \cup \llbracket Q \rrbracket$, $\llbracket \mathtt{afterStripping}(s, P) \rrbracket = \{\, n + s : n \in \llbracket P \rrbracket \,\}$, $\llbracket \mathtt{beforeStripping}(s, P) \rrbracket = \{\, s + n : n \in \llbracket P \rrbracket \,\}$, and each affix constructor and `matching` its evident set.

### D-11. Name expression

A **name expression** is a regular set of names, carried as the finite automaton that decides it. Every constructor of `D-9` denotes a regular set, and `matching` is restricted to one, so the whole language is regular.

That restriction is what the rest of this document rests on. Regular sets are closed under union, intersection and complement, and containment between two of them is decidable, so `T-9` and `T-12` are questions with answers rather than questions a checker approximates. Every absence rule denotes a regular set when each $\mathtt{matching}$ in it does. Proved in Lean: `Spec.D11.regular` in `spec/lean/Spec/L4/Regular.lean`.

A pattern selector (`cs.matches(p)`, or `pl.col(p)` with `p` starting with `^` and ending with `$`) produces a `matching` over the names that polars finds with `p`. Polars searches a name for `p`, so the set is $\Sigma^* L \Sigma^*$ for a branch $L$ with no anchor, with $\Sigma^*$ left out at an anchored edge, and a union of these over the top-level branches. Each is regular. The checker reads only a part of the dialect whose meaning is the same in polars and in the automaton: literals, escaped punctuation, `.` (every character but a newline), character classes, groups, alternation, `*`, `+` and `?`, with `^` and `$` only at the edges of a top-level branch. A `]` outside a class is the character itself in polars, so the checker reads it as escaped. Polars compiles the pattern only where a frame expands the selector, so a text that is no pattern of its dialect, as an unclosed class `^a[` or a look-around, raises `InvalidOperationError` at every frame, a frame of no column too. The same holds for a name of `pl.col` or of `exclude` that starts with `^` and ends with `$`, and the checker reports each such call (`test_polars_refuses_a_pattern_that_it_cannot_compile_at_every_frame` in `spec/tests/test_pattern_reader.py`). It also bounds the pattern's size, because the search can make the automaton exponential in the pattern. Any other pattern leaves the selector plain.

The other two producers of `matching` are the affix translations of `D-23`, and both preserve regularity. Appending a fixed string to every member of a regular set gives a regular set, and so does the quotient that undoes it. Proved in Lean: `Spec.D11.preimage_regular` in `spec/lean/Spec/L4/Regular.lean`.

**A type spells a `matching` by its names.** An automaton decides what a name test accepts, and only the automaton takes part in comparison, so two derived tests accepting the same names are the same test. A type spells a `matching` by a text that only its names fix: a regular expression of the names where the minimal automaton has no loop but a state that keeps every character, as `ab.*` or `a(b.*)?`, and the text of the automaton otherwise. The conversion of a loop into an expression grows fast, so it stops there. Measured by `TestATypeSpellsAMatchingByItsNames` in `spec/tests/test_normal_form.py`.

### 2.2 Selectors

A **selector** decides one whole column: its name and its data type together. A selector is what a user writes, as `cs.numeric()`, `cs.starts_with("tmp_")` or a plain list of names.

### D-12. The selector language

```
C  ::=  nameTest(P)            picks a column when P holds of its name
     |  ofType(F)              picks a column when its data type is in the data-type set F
     |  not C  |  C and C  |  C or C
```

**The selector language is strictly larger than the absence language.** Every absence rule becomes a selector through `nameTest`, and `ofType` is a selector that is no absence rule, because a data type cannot be read from a name. This is why `D-9` deliberately has no `ofType` constructor, and why `E-6` needs a separate rule to turn a type test into a statement about names.

### D-13. What a selector denotes

$\llbracket C \rrbracket$ is a set of columns (`D-4`): $\llbracket \mathtt{nameTest}(P) \rrbracket = \{\, n : t \mid n \in \llbracket P \rrbracket \,\}$, $\llbracket \mathtt{ofType}(F) \rrbracket = \{\, n : t \mid t \in F \,\}$, and the three combinators are complement, intersection and union.

### D-14. What a selector settles without seeing a column

An operation over a remainder cannot see a column, so it needs two weaker questions answered. Each returns an **absence rule**, given what is known about data types (`D-15`):

- $\mathrm{certainlyPicks}(C, \Theta)$ holds of the names whose every column is picked, whatever its data type.
- $\mathrm{certainlyRejects}(C, \Theta)$ holds of the names whose every column is rejected, whatever its data type.

They are defined by induction on $C$:

| $C$ | $\mathrm{certainlyPicks}(C, \Theta)$ | $\mathrm{certainlyRejects}(C, \Theta)$ |
| -- | -- | -- |
| $\mathtt{nameTest}(P)$ | $P$ | $\mathtt{not}\ P$ |
| $\mathtt{ofType}(F)$ | $\mathtt{always}$ if $\Theta \subseteq F$, else $\mathtt{never}$ | $\mathtt{always}$ if $\Theta \cap F = \emptyset$, else $\mathtt{never}$ |
| $\mathtt{not}\ C$ | $\mathrm{certainlyRejects}(C, \Theta)$ | $\mathrm{certainlyPicks}(C, \Theta)$ |
| $C\ \mathtt{and}\ D$ | $\mathrm{certainlyPicks}(C, \Theta)\ \mathtt{and}\ \mathrm{certainlyPicks}(D, \Theta)$ | $\mathrm{certainlyRejects}(C, \Theta)\ \mathtt{or}\ \mathrm{certainlyRejects}(D, \Theta)$ |
| $C\ \mathtt{or}\ D$ | $\mathrm{certainlyPicks}(C, \Theta)\ \mathtt{or}\ \mathrm{certainlyPicks}(D, \Theta)$ | $\mathrm{certainlyRejects}(C, \Theta)\ \mathtt{and}\ \mathrm{certainlyRejects}(D, \Theta)$ |

For a pure name test both are exact. For a type test both are all-or-nothing: either the recorded type set settles the question for every name at once, or it settles it for none.

Both are sound: a name that $\mathrm{certainlyPicks}(C, \Theta)$ holds of is picked with every data type in $\Theta$, and a name that $\mathrm{certainlyRejects}(C, \Theta)$ holds of is rejected with every such data type. Proved in Lean: `Spec.D14.certainly_picks_sound` and `Spec.D14.certainly_rejects_sound` in `spec/lean/Spec/L1/Settles.lean`.

**A type selector has three cases, and they are the three values of the $\mathtt{ofType}$ row.** "Every column passes", "no column passes" and "neither" are the three pairs the row can give. Writing them as two rules rather than three cases is what makes the combinators follow, so `~cs.numeric()` and `cs.numeric() & cs.starts_with("p")` need no new rule.

**The rule reads the selector as written, never the family it names.** Under `all_of=NumericType`, `cs.numeric()` picks everything and `~cs.numeric()` picks nothing. A checker that resolved the negation by looking up "numeric" and applying a table would be wrong in both directions, and wrong silently, since both answers are legal for some other program.

### 2.3 Type sets

The third language records which data types a remainder's columns can have.

### D-15. Type set

A **type set** $\Theta \subseteq T$ records the data types a remainder's columns can have. It is the second half of what is recorded about a remainder (`D-16`), and it is computed rather than written.

A type set is $T$ itself, $\emptyset$, a data-type set of `D-3`, or a singleton $\{U\}$, or a set built from these by union, intersection and difference. These are the only forms the rules of section 7 produce.

Every such set is held kind by kind, and each kind divides into finitely many **sectors**. A kind whose data types a finite list states has one sector, the whole kind. `Datetime` has one sector per time unit, because its zone is a name from an open set: `cs.datetime("us")` holds every zone of one unit, which no finite list and no finite exclusion states. Within a kind, a part holds some sectors whole, less a finite set of data types taken out of them, and a finite set of data types from the sectors it does not hold whole. Both sets are finite and the sectors are finitely many, so a part is closed under union, intersection and complement. So the checker holds a type set in that form and answers each question below exactly.

Three questions are asked of it, and each is a set operation:

| Question | Decides |
| -- | -- |
| $\Theta \subseteq F$ | the `all_of=F` requirement (`R-8`) |
| $\Theta \cap F = \emptyset$ | the `none_of=F` requirement (`R-9`) |
| $\Theta \cap F$ | what a type selector leaves (`T-2`) |

______________________________________________________________________

## 3. The record and its meaning

### D-16. Record

What is known about a remainder is a **record**: an absence rule (`D-9`) and a type set (`D-15`).

$$
\mathrm{Record} = (P, \Theta)
$$

That is the whole record. Three things are deliberately not in it, and each omission is forced rather than chosen.

**Which names are present.** Knowing a column's name is what puts it in the schema (`D-8`), so a name cannot be known and remain in the remainder. This is why section 5 has no "must contain" requirement: it would be written by declaring the column.

**Whether the remainder can be empty.** Empty means every name is absent, which the absence rule already says.

**Column order and column count.** No rule in this document asks, and section 11 of `DATAFRAME_SPECIFICATION.md` lists the operations that cannot be typed.

### D-17. What a record describes

A column set $X$ (`D-5`) is **described by** a record $(P, \Theta)$ when

$$
\text{no name of } X \text{ is in } \llbracket P \rrbracket \quad \text{and} \quad \text{every data type of } X \text{ is in } \Theta.
$$

Write $\mathrm{describes}((P, \Theta), X)$. A record therefore denotes a set of column sets: those it does not rule out.

### D-18. Remainder expression

A **remainder expression** is a variable, an effect applied to an expression, or two expressions combined:

```
e  ::=  R                    a variable, standing for what a caller supplies
     |  ε(e)                 an effect (section 6) applied to e
     |  both(e, e)           two remainders held together
```

An expression is syntax. Its meaning is fixed only when its variables are given column sets, which is why it is carried in a type and evaluated later rather than collapsed to a record on the spot.

### D-19. Denotation of an expression

A **binding** $\beta$ gives one column set to each variable. Then $\llbracket e \rrbracket_\beta$ is one definite column set, by induction: $\llbracket R \rrbracket_\beta = \beta(R)$, $\llbracket \varepsilon(e) \rrbracket_\beta = \varepsilon(\llbracket e \rrbracket_\beta)$ using the effect's own action on column sets (section 6), and $\llbracket \mathtt{both}(e_1, e_2) \rrbracket_\beta = \llbracket e_1 \rrbracket_\beta \cup \llbracket e_2 \rrbracket_\beta$.

A statement about a remainder means "true under every binding". This is the every-caller rule of `DATAFRAME_SPECIFICATION.md`, restated at the level of the machinery, and it is the soundness condition every rule below must respect.

### D-20. Knowledge of an expression

$\mathrm{know}(e)$ is the record (`D-16`) computed for an expression, by induction: $\mathrm{know}(R)$ is what $R$'s declaration states, $\mathrm{know}(\varepsilon(e))$ is the effect's record rule applied to $\mathrm{know}(e)$, and $\mathrm{know}(\mathtt{both}(e_1, e_2))$ is the `E-4` rule applied to $\mathrm{know}(e_1)$ and $\mathrm{know}(e_2)$ (section 7).

Knowledge must be computed through the **whole** expression. A checker reading only the outermost effect sees `Minus[Minus[R, "a"], "b"]` as proving only `b` absent and silently loses the guarantee about `a`, so composing two correct helpers produces a false error.

______________________________________________________________________

## 4. Soundness

### T-1. The record over-approximates

> For every expression $e$ and every binding $\beta$ that satisfies the declaration of each variable, $\mathrm{describes}(\mathrm{know}(e), \llbracket e \rrbracket_\beta)$.

**Tag: proved.** Proved in Lean: `Spec.T1.over_approximates` in `spec/lean/Spec/L2/Closure.lean`. Measured by `test_effect_transformation_is_sound` and `test_both_transformation_is_sound` in `spec/tests/test_absence_language.py`, which check every effect against every record of a finite family over a finite universe.

The proof is by induction on $e$. A variable is the base case: every caller is checked against the declaration at the call site (section 11), so the binding satisfies it, and the theorem takes that as its hypothesis. An effect or a `both` is one step of `T-2`.

**The direction matters and only one direction is safe.** Losing a fact makes the described set larger. A larger set still contains the truth, so the record says less and stays honest. Gaining a fact makes the set smaller, and a smaller set can no longer contain the truth. Every rule in section 7 records less than it can wherever the alternative is a guess.

______________________________________________________________________

## 5. Requirements

A **requirement** is a promise a caller makes about a remainder. It is written where the remainder is declared, as a keyword, and never inside a frame type.

### 5.1 The eight per-column requirements, and the empty one

`R-1` demands nothing. Each of the other eight tests one column on its own. That is what makes them usable both as a promise and as a part of a split (`D-24`).

| Id | Written | Holds when, for every column $n : t$ |
| -- | -- | -- |
| `R-1` | `other_columns()` | always. The default: promises nothing |
| `R-2` | `without=[a, b]` | $n \notin \{a, b\}$ |
| `R-3` | `only=[a, b]` | $n \in \{a, b\}$ |
| `R-4` | `without_prefix="p"` | $n$ does not begin with $p$ |
| `R-5` | `only_prefix="p"` | $n$ begins with $p$ |
| `R-6` | `without_suffix="s"` | $n$ does not end in $s$ |
| `R-7` | `only_suffix="s"` | $n$ ends in $s$ |
| `R-8` | `all_of=F` | $t \in F$, where $F$ is a data-type set (`D-3`) |
| `R-9` | `none_of=F` | $t \notin F$ |

The eight are four matched pairs: a name list, a prefix, a suffix and a data-type set, each in a negative and a positive form. The symmetry is not decoration. A split partitions its columns by each part's own predicate, so a part that takes a namespace needs the positive form, exactly as a part that avoids one needs the negative.

Keywords combine on one declaration, and the requirement is their conjunction:

```python
R = other_columns(without=["audit_id"], none_of=NestedType)
```

### D-21. The predicate of a requirement

For a per-column requirement $\rho$, $\mathrm{pred}(\rho)$ is the selector (`D-12`) in the table above: $\mathrm{pred}(\texttt{without=[a]}) = \mathtt{not}\ \mathtt{nameTest}(\mathtt{oneOf}(\{a\}))$, $\mathrm{pred}(\texttt{all\_of=F}) = \mathtt{ofType}(F)$, and so on. For the empty requirement, $\mathrm{pred}(\texttt{other\_columns()}) = \top$, where $\top$ is the selector $\mathtt{nameTest}(\mathtt{always})$ that picks every column.

Every per-column requirement is therefore a selector, which is what `D-24` uses.

### 5.2 The two relational requirements

These describe how **two** remainders relate. They are still written where a remainder is declared, and they name another remainder, which must already exist.

| Id | Written | Holds when |
| -- | -- | -- |
| `R-10` | `disjoint_from=A`, or `disjoint_from={A, B}` | no name of this remainder is a name of the named one, or of their union |
| `R-11` | `within=A`, or `within={A, B}` | every name of this remainder is a name of the named one, or of their union |

```python
L = other_columns()
R = other_columns(disjoint_from={L})
```

`disjoint_from=` takes a **set** of remainders. One is the common case and is what the join sections of `DATAFRAME_SPECIFICATION.md` write. Several are needed when the other side is described in parts (`D-24`), since a split has no single name standing for the whole remainder.

`disjoint_from=` is symmetric, so stating it on either remainder establishes it for both. `within=` is not symmetric: it says that the names of the declared remainder are among the names of the named one. Both are written on the second remainder, the only ordering Python allows, since the first must exist before it can be named.

**Neither relational requirement can appear in a split.** A relation is a fact about a pair of remainders, not a test one column passes or fails, so it cannot decide where a column goes. This is not a restriction imposed on splits: it follows from `D-24` needing a predicate, and `R-10` and `R-11` having none.

The closure of the record under every effect is `T-2`, in section 7. The effects come first.

______________________________________________________________________

## 6. Effects

An **effect** describes what an operation does to a remainder. It appears in a return type, because it describes an output rather than a demand. No effect carries a requirement and no requirement appears in a return type.

There are **seven effects**, written in **nine ways**. Each is a function on column sets (`D-5`), lifted to act on the set of column sets a record describes.

| Id | Effect | Written | Action on a column set $X$ |
| -- | -- | -- | -- |
| `E-1` | unchanged | the variable itself, for example `Rest` | $X$ |
| `E-2` | emptied | *(the remainder is omitted)* | $\emptyset$ |
| `E-3` | filtered | `Minus[R, S]`, `Only[R, C]`, `Except[R, C]` | $\{\, n : t \in X \mid n : t \in \llbracket C \rrbracket \,\}$ |
| `E-4` | combined | `Both[A, B]` | $X_A \cup X_B$ |
| `E-5a` | suffixed | `Suffixed[R, Literal["s"]]` | $\{\, n + s : t \mid n : t \in X \,\}$ |
| `E-5b` | prefixed | `Prefixed[Literal["p"], R]` | $\{\, p + n : t \mid n : t \in X \,\}$ |
| `E-6` | retyped | `Retyped[R, U]` | $\{\, n : U \mid n : t \in X \,\}$ |

### 6.1 Why `E-3` is one effect with three spellings

Removing a finite set of names $S$ is dropping whatever a name-list selector picks, so `Minus[R, S]` and `Except[R, oneOf(S)]` are one thing written twice.

| Written | Keeps |
| -- | -- |
| `Minus[R, S]` | everything except the names in $S$ |
| `Except[R, C]` | everything the selector $C$ does not pick |
| `Only[R, C]` | everything the selector $C$ does pick |

One node in the model, three spellings on the page. `Minus` has its own spelling because removing a list of names needs no selector.

This is why the rule that repeated removals collapse is not a special case: it is the general rule that two filters combine into one, read at a name-list selector.

### 6.2 Why `E-4` is a construct and not a second remainder slot

`D-8` gives the two reasons. `E-4` states no requirement of its own: a union is well defined only when nothing is counted twice. That condition is proved from the two records, when no name is allowed by both absence rules, or declared on an operand with `R-10`, where every requirement is declared. A `disjoint_from=` promise covers a remainder only while its names are the declared ones: under `Suffixed` or `Prefixed` it covers nothing. The union of two column sets is a column set exactly when no name carries two different data types across them. Proved in Lean: `Spec.E4.functional_union_iff` in `spec/lean/Spec/L2/Effects.lean`. A caller supplying two remainders that can overlap gets an error naming that declaration, not the return type.

Each operand carries its own effects, which is how a result with one side suffixed and the other not is written:

```python
Both[L, Suffixed[R, Literal["_prev"]]]
```

**An operation that merges a shared name is not `E-4`, and is absent.** Merging needs the two columns to agree on data type, and for two remainders no such agreement can be proven. The library has such operations, and they behave exactly as that predicts: `pl.concat(how="diagonal")` raises when a shared name holds `Int32` on one side and `Int64` on the other, and its relaxed form answers with a supertype of two data types the checker cannot see. A caller who wants merging declares `R-10` instead, which is stronger than merging needs but is something a caller can actually establish.

### 6.3 Why only the two renamings invent names

`E-5a` and `E-5b` are the only effects that produce a name none of their inputs had. Every other effect produces names an input already held. That matters twice: it is why `D-9` carries the two stripping constructors, which nothing else uses, and it is why `T-3` has to be checked for these two and is free for the rest.

### 6.4 An empty affix

`Suffixed[R, Literal[""]]` and `Prefixed[Literal[""], R]` rename nothing, so both denote `R` and both normalize to it (`D-22`). A filter through an empty affix is then the filter on `R` itself.

> Measured (`M-2`) by `TestAnEmptyAffixNormalizesAway` in `spec/tests/test_normal_form.py`: a filter through no affix, an empty suffix or an empty prefix gives the canonical form of the filter on `R`.

______________________________________________________________________

## 7. Closure

### T-2. The record language is closed under every effect

> For each effect $\varepsilon$ there is a rule $\mathrm{record}(\varepsilon, \cdot)$ on records such that $\mathrm{describes}(r, X)$ implies $\mathrm{describes}(\mathrm{record}(\varepsilon, r), \varepsilon(X))$, and $\mathrm{record}(\varepsilon, \cdot)$ produces a record in the language of `D-16`.

**Tag: proved.** Proved in Lean: `Spec.T2.closure` in `spec/lean/Spec/L2/Closure.lean`, and `Spec.T2.closure_both` in the same file for `E-4`. A record rule produces a record in the language of `D-16` by construction: its absence rule and its type set are built from the constructors of `D-9` and `D-15`. Measured by `test_effect_transformation_is_sound` and `test_both_transformation_is_sound` in `spec/tests/test_absence_language.py`: that file makes 855 checks over fourteen effects and fifty-six records, the starting rules including the stripping and machine-carried shapes the renaming effects themselves produce, over a sixty-four-column-set universe of three names and three data types, the universe's count computed by the suite itself.

Closure matters because a chain of operations would otherwise eventually produce a record that cannot be written down. Both halves must be closed: without the type half, the effect of a type selector on $\Theta$ is undefined, and every later type question on the result has no answer.

### 7.1 The two halves, in one table

Write $r = (P, \Theta)$.

| Effect | Absence rule out | Type set out |
| -- | -- | -- |
| `E-1` unchanged | $P$ | $\Theta$ |
| `E-2` emptied | $\mathtt{always}$ | $\emptyset$ |
| `E-3` filtered by $C$ | $P\ \mathtt{or}\ \mathrm{certainlyRejects}(C, \Theta)$ | $\mathrm{surviving}(C, \Theta)$ |
| `E-4` combined, $A$ with $B$ | $P_A\ \mathtt{and}\ P_B$ | $\Theta_A \cup \Theta_B$ |
| `E-5a` suffixed by $s$ | $\mathtt{not}\ \mathtt{endsWith}(s)\ \mathtt{or}\ \mathtt{afterStripping}(s, P)$ | $\Theta$ |
| `E-5b` prefixed by $p$ | $\mathtt{not}\ \mathtt{startsWith}(p)\ \mathtt{or}\ \mathtt{beforeStripping}(p, P)$ | $\Theta$ |
| `E-6` retyped to $U$ | $P$ | $\{U\}$ |

with

| $C$ | $\mathrm{surviving}(C, \Theta)$ |
| -- | -- |
| $\mathtt{nameTest}(P)$ | $\Theta$ |
| $\mathtt{ofType}(F)$ | $\Theta \cap F$ |
| $\mathtt{not}\ C$ | $\Theta \setminus \mathrm{surviving}(C, \Theta)$ if $C$ reads no name, else $\Theta$ |
| $C\ \mathtt{and}\ D$ | $\mathrm{surviving}(C, \Theta) \cap \mathrm{surviving}(D, \Theta)$ |
| $C\ \mathtt{or}\ D$ | $\mathrm{surviving}(C, \Theta) \cup \mathrm{surviving}(D, \Theta)$ |

The guard of the $\mathtt{not}$ row is syntactic: $C$ reads no name when it holds no $\mathtt{nameTest}$, even one that holds of every name or of none. With the guard, $\mathrm{surviving}(C, \Theta)$ holds the data type of every column in $\Theta$ that $C$ picks, and for a selector that reads no name it is exact. Proved in Lean: `Spec.D14.surviving_sound` and `Spec.D14.surviving_exact` in `spec/lean/Spec/L1/Settles.lean`.

### 7.2 The rows, derived

The proof takes one case per effect. `E-3` rests on the soundness of $\mathrm{certainlyRejects}$ (`D-14`) and of $\mathrm{surviving}$ (section 7.1). `E-5a` and `E-5b` rest on cancellation: $n + s = n' + s$ implies $n = n'$, and so does $s + n = s + n'$. The other rows follow from the definitions.

**`E-6` keeps $P$.** An operation that keeps every column and changes only data types loses nothing about which names exist, and the record must say so rather than collapsing to "anything is possible".

**`E-5a` needs both halves of its rule, and neither is a list.** A name $m$ is absent from the result when it does not end in $s$, or when the name under $s$ was absent before. "Does not end in `_prev`" is a fact about a name that no finite list of names states. A language restricted to lists would prove nothing absent after a suffixing rename, and every operation after one would lose all checking. `E-5b` is the same with the ends swapped.

**`E-3` needs the guard on the $\mathtt{not}$ row of $\mathrm{surviving}$.** For $\mathtt{not}\ \mathtt{nameTest}(P)$ a column of any data type can survive, and subtracting $\mathrm{surviving}(\mathtt{nameTest}(P), \Theta) = \Theta$ would give $\emptyset$: a fact invented rather than kept, the one direction `T-1` forbids.

### 7.3 An effect that manufactures a guarantee

`E-3` adds $\mathrm{certainlyRejects}(C, \Theta)$ **unconditionally**, whatever the input rule was. Dropping a name makes it absent whether or not it was ever there.

That one line is the entire explanation of a property the compatibility rules lean on: an effect can produce a guarantee its input did not have. An unconstrained remainder with `notional` removed satisfies a function demanding `notional` be absent, and its caller promised nothing. `Minus` is one of the spellings of `E-3`, so this is not a separate rule.

### T-3. When a new effect can be added

> An effect renames each column by a function $f$ on names. A name $m$ is absent from the result exactly when every input name $f$ sends to $m$ is absent:
>
> $$
> \mathrm{absentResult}(m) \Leftrightarrow \forall n \,.\, f(n) = m \Rightarrow \mathrm{absentInput}(n)
> $$
>
> A renaming $f$ is **admissible** when $f$ is injective and the preimage under $f$ of every absence rule is an absence rule. An admissible renaming has a sound record rule: the preimage of the input's absence rule, with the type set unchanged. An injective renaming keeps a column set a column set, and a renaming that is not injective sends some column set to a relation that gives one name two data types.

**Tag: proved.** Proved in Lean, all in `spec/lean/Spec/L2/Admissible.lean`: `Spec.T3.absent_iff` for the displayed equivalence, `Spec.T3.admissible_closed` for the record rule, and `Spec.T3.functional_rename` and `Spec.T3.collision` for the two directions of injectivity.

The equivalence is the definition of the image of a column set under a renaming, negated. The record rule is its preimage. The collision takes two names that $f$ sends to one, and gives them two different data types.

Two worked cases.

*`E-5a` passes.* The preimage of $m$ under $n \mapsto n + s$ is empty when $m$ does not end in $s$, and the single name $m$ with $s$ removed when it does. $\mathtt{not}\ \mathtt{endsWith}(s)$ and $\mathtt{afterStripping}(s, \cdot)$ are those two cases, and both are constructors. Proved in Lean: `Spec.T3.suffix_admissible`, and `Spec.T3.prefix_admissible` for `E-5b`, in `spec/lean/Spec/L2/Admissible.lean`.

*Lowercasing every name fails.* The preimage of `abc` is every capitalization of `abc`. That set is regular, so the first condition holds. The second does not: the operation sends `ABC` and `abc` to one name, so the result would depend on which the caller brought, and no single type covers both callers.

The second condition is the same one joins are subject to. An operation whose output name for a column depends on what else the caller brought has no single result type.

______________________________________________________________________

## 8. Canonical form

Two different questions are easy to conflate.

The first is **requirement checking**: knowledge must be computed through the whole expression (`D-20`).

The second is **type identity**: a body producing `Minus[Minus[R, "a"], "b"]` must satisfy a signature declaring `Minus[R, {"a", "b"}]`.

### 8.1 The form

`D-22` defines the canonical form, and `D-23` defines how a selector moves inward through it.

### D-22. Canonical form

The canonical form of an expression is a five-part record:

$$
\mathrm{canonical} = (\mathit{base},\ \mathit{filter}^?,\ \mathit{suffix},\ \mathit{prefix},\ \mathit{retype}^?)
$$

A slot marked $?$ can be empty (written $\mathtt{none}$). The suffix and the prefix are strings, and the empty string renames nothing. Read from the outermost slot inward, the form is

$$
\mathit{retype}^? \text{ over } \mathit{prefix} \text{ over } \mathit{suffix} \text{ over } \mathit{filter}^? \text{ over } \mathit{base}
$$

where a **base** is a variable, a sorted union of canonical forms, or nothing at all.

$\mathrm{normalize}(e)$ is defined by structural induction on $e$:

| Case | $e$ | $\mathrm{normalize}(e)$ |
| -- | -- | -- |
| variable | $R$ | $(\mathtt{Variable}(R), \mathtt{none}, \texttt{""}, \texttt{""}, \mathtt{none})$ |
| union | $\mathtt{both}(e_1, e_2)$ | $(\mathtt{Union}(\mathrm{sorted}(\mathrm{flatten}(\mathrm{normalize}(e_1)), \mathrm{flatten}(\mathrm{normalize}(e_2)))), \mathtt{none}, \texttt{""}, \texttt{""}, \mathtt{none})$ |
| `E-1` | $e'$ unchanged | $\mathrm{normalize}(e')$ |
| `E-2` | $e'$ emptied | $(\mathtt{Nothing}, \mathtt{none}, \texttt{""}, \texttt{""}, \mathtt{none})$ |
| `E-6` | $e'$ retyped to $U$ | $\mathrm{normalize}(e')$ with $\mathit{retype} := U$ |
| `E-5a` | $e'$ suffixed by $s$ | $\mathrm{normalize}(e')$ with $\mathit{suffix} := \mathit{suffix} + s$ |
| `E-5b` | $e'$ prefixed by $p$ | $\mathrm{normalize}(e')$ with $\mathit{prefix} := p + \mathit{prefix}$ |
| `E-3` | $e'$ filtered by $C$ | $\mathrm{normalize}(e')$ with $\mathit{filter} := \mathit{filter}\ \mathtt{and}\ \mathrm{translate}(C, \mathrm{normalize}(e'))$ |

An effect row applies to a form whose base is a variable or nothing. Over a union it applies to each operand, and the operands unite again: every effect acts on each column separately, so it distributes over a union, and a union form carries no effect of its own. Proved in Lean: `Spec.D22.distribute_denote` in `spec/lean/Spec/L3/Canonical.lean`. The union row flattens a nested union into its operands. Its operands are sorted, a repeated operand is removed, an operand that denotes nothing is dropped, and a union of one operand is that operand. A filter that picks every column is dropped. In Lean, `normalize` takes the name translation of `D-23`, the two tests that settle a filter, and the arrangement of a union's operands as parameters, each with the property it must have, so the theorems below hold for every implementation that meets those properties.

A filter that picks no column normalizes to the form `E-2` produces: $(\mathtt{Nothing}, \mathtt{none}, \texttt{""}, \texttt{""}, \mathtt{none})$. A provably empty remainder therefore has one spelling, and a frame whose remainder is provably empty is written with no remainder at all.

**This is a function, not a fixed point.** Normalization is not defined as rewriting until no rule applies. Such a definition owes two arguments: that the rewriting terminates, and that it ends in the same place whichever order the rules fire in. A definition by induction needs neither, because a function has exactly one value.

### T-4. Every expression has exactly one canonical form

**Tag: proved.** Proved in Lean: `Spec.T4.canonical_form_unique` in `spec/lean/Spec/L3/Canonical.lean`. Measured by `test_normalising_is_deterministic` and `test_the_canonical_form_has_the_one_declared_shape` in `spec/tests/test_normal_form.py`, each over one in eleven expressions of the space: the claim is structural, a function returning one value, so the sample guards transcription and the full space is covered by the denotation sweep of `T-5`.

Lean accepts `normalize` because it is defined by structural recursion, and such a definition is a total function with one value. No termination measure and no confluence argument are needed, because there is no rewriting.

### D-23. Translating a selector inward

$\mathrm{translate}(C, k)$ rewrites a selector that runs **after** everything the canonical form $k$ already did, into one that tests $k$'s base column. Three steps, outermost slot first:

1. **Past $\mathit{retype} = U$.** Every column now has data type $U$, so every type test in $C$ is settled outright: $\mathtt{ofType}(F)$ becomes $\mathtt{nameTest}(\mathtt{always})$ if $U \in F$, and $\mathtt{nameTest}(\mathtt{never})$ otherwise. The result reads no data type.
2. **Past $\mathit{prefix} = p$.** A test on $m$ becomes a test on $p + m$.
3. **Past $\mathit{suffix} = s$.** A test on $m$ becomes a test on $m + s$.

After steps 2 and 3 a test on the base name $n$ is a test on $p + n + s$, which is what $k$ produces.

**This is computed, not substituted, and that is the trap.** Filtering on `startsWith("ab")` after appending `"b_"` is not filtering on `startsWith("ab")` before it: a column named `a` becomes `ab_`, which begins with `ab`, so it survives a test its original name fails. The translated rule is `startsWith("ab") or oneOf({"a"})`, and in general the translation lands in `matching` (`D-9`), which is why that constructor exists.

Steps 2 and 3 are skipped when the affix is empty (section 6.4).

The translation is correct: $\mathrm{translate}(C, k)$ picks a base column exactly when $C$ picks the column that base column becomes under $k$. Proved in Lean: `Spec.D23.translate_correct` in `spec/lean/Spec/L3/Canonical.lean`.

### 8.2 What the form forgets, and why it must

**A union forgets order and grouping.** `Both[A, B]` and `Both[B, A]` describe the same frame, as do `Both[Both[A, B], C]` and `Both[A, Both[B, C]]`. A union's meaning has neither an order nor a grouping. A node has both. Flattening and sorting is how the node is made to forget a distinction its meaning never had.

The sort key must be **total**, **the same in every run**, and **derived only from meaning**. The third obligation rules out anything reaching for an object identity, and the second rules out anything built from a `frozenset`'s iteration order, which differs between processes.

> Measured (`M-3`) by `test_the_sort_key_does_not_depend_on_set_ordering` in `spec/tests/test_normal_form.py`.

**Like effects combine.** Two filters become one taking $\mathtt{and}$ of their selectors. Two retypings keep the outer data type. Two suffixes, or two prefixes, concatenate. Each row of `D-22` states this.

**Unlike effects commute.** Dropping a column and then counting nulls gives the same frame as counting nulls and then dropping the column. A user can write these in either order and both must be acceptable to the same function. A system that treated them as different types would reject one of two programs that do the same thing.

> Measured by `TestUnlikeEffectsCommute` in `spec/tests/test_normal_form.py`.

**Two orders that really differ stay different.** Removing `a` before suffixing removes `a_p` from the result. Removing `a` after suffixing removes nothing at all. The canonical form keeps these apart, and `TestUnlikeEffectsThatDoNotCommute` in `spec/tests/test_normal_form.py` checks that it does.

### 8.3 Adequacy

The canonical form is useful only if it keeps the meaning of the expression it replaces. `T-5` states that it does, and `T-6` … `T-9` build on it.

### T-5. The canonical form preserves meaning

> For every expression $e$ and every binding $\beta$, $\llbracket \mathrm{normalize}(e) \rrbracket_\beta = \llbracket e \rrbracket_\beta$.

**Tag: proved.** Proved in Lean: `Spec.T5.normalize_preserves_denotation` in `spec/lean/Spec/L3/Canonical.lean`. Measured by `test_the_canonical_form_denotes_what_the_expression_denotes` in `spec/tests/test_normal_form.py`, over every expression of depth two and all 81 bindings of a two-name, two-type universe.

The proof is by induction on $e$, one case per row of `D-22`. Two affixes concatenate because appending is associative, and a later retype overwrites an earlier one. The filter case rests on the correctness of `D-23`'s translation, and the union case on the fact that a union ignores the order, the grouping and the repeats of its operands, and drops an operand that denotes nothing.

### T-6. Substitution

> Passing a frame whose remainder is $e$ to a function whose remainder variable is $R$ replaces $R$ by $e$ throughout the callee's result, and the result is normalized.

**Tag: assumed.** Its source is the definition of how a call site composes two signatures. Its consequence is `T-7`.

### T-24. Denotation is additive in each variable

> For a remainder expression $e$, a binding $\sigma$, and its variables split into two sets $V_c$ and $V_o$: $\llbracket e \rrbracket_{\sigma} = \llbracket e \rrbracket_{\sigma[V_o \mapsto \emptyset]} \cup \llbracket e \rrbracket_{\sigma[V_c \mapsto \emptyset]}$, where $\sigma[V \mapsto \emptyset]$ binds each variable of $V$ to no column.

**Tag: measured.** Every effect acts on each column alone: a filter tests one column, a suffix, a prefix and a retype change one column, and a union keeps the columns of each operand. So the columns that one variable contributes do not depend on what another variable is bound to. Measured by `test_denotation_is_additive_in_each_variable` in `spec/tests/test_normal_form.py`, over every expression of depth two and every binding of its universe.

Its consequence is the evaluation of a return type at a call that **mixes** closed and open arguments. The closed arguments bind their variables $V_c$ to column sets, and the open ones bind $V_o$ to the callers' remainders. The return remainder then denotes the union of two halves: the columns that $V_c$ gives, with $V_o$ bound to nothing, which are known and join the declared columns, and the remainder that $V_o$ gives, by the substitution of `T-6` with $V_c$ bound to the empty remainder. A frame type cannot hold a declared name in its remainder, so the open half has the names of the closed half taken out. When every argument is closed the open half is empty and the result is closed. When every argument is open the closed half is empty and the result is the substitution of `T-6` alone.

### T-7. Every minted type is in canonical form

> Every type this system produces has passed through $\mathrm{normalize}$.

**Tag: assumed.** It is an obligation on a checker that implements this system, and it is the obligation the whole comparison rests on. There are three places a type is minted: a checker computing a result, a user writing an annotation, and **substitution** at a call site (`T-6`). Missing the third makes the composition of two correct helpers produce a nested form that compares unequal to the flattened one a signature declares.

### T-8. Equal canonical forms mean equal denotations

> $e_1 \equiv e_2$ implies $e_1 = e_2$.

**Tag: proved.** Proved in Lean: `Spec.T8.equal_forms_equal_denotations` in `spec/lean/Spec/L3/Canonical.lean`. Measured by `test_equal_canonical_forms_mean_equal_denotations` in `spec/tests/test_normal_form.py`.

It follows from `T-5`: each expression denotes what its canonical form denotes.

This is the direction the checker relies on, and it is the only one it needs.

### T-9. Containment between two selectors is decidable

> For any two selectors $C$ and $D$, whether every column $C$ picks is a column $D$ picks is decidable.

**Tag: proved.** Proved in Lean: `Spec.T9.decidable` in `spec/lean/Spec/L4/Decide.lean`. Measured by `TestSelectorContainment` in `spec/tests/test_requirements.py`, which checks every rule of the cheap path against the exact one, and checks every selector containment the cheap path establishes over a battery holding type tests, column by column.

The Lean theorem gives a finite list of data types and a bound on the length of names, and states that containment holds exactly when every name shorter than the bound that $C$ picks at a listed data type, $D$ picks too. Lean's characters are finite, so that is a finite check.

Three steps give it. Every data type is treated like one of finitely many witnesses: the data types the two selectors name, and one more data type of each kind (`Spec.T9.reduction`). At one data type, the names a selector picks form a regular set (`Spec.T9.regular`). Containment between two regular sets holds when it holds for every name shorter than the state count of the automaton for the difference, because a longer counterexample can be shortened by the pumping lemma (`Spec.T9.finite_check`).

### 8.4 Deciding cheaply, and deciding exactly

`T-9` says the question has an answer. Two things decide it, and the order matters.

**A table of cheap rules, tried first.** A longer prefix is contained in a shorter one, a name list is contained in whatever holds of each of its names, a conjunction is contained in whatever either half is, and so on. Each is a string comparison or a structural step. These settle the containments that arise from what a user writes.

**An automaton comparison, when no cheap rule fires.** This decides every remaining case, and it is exact.

The cheap rules are not an approximation of the exact one and never disagree with it. They exist because they are far faster.

### T-10. Equal denotations give equal canonical forms

> $e_1 = e_2$ implies $e_1 \equiv e_2$.

**Tag: measured.** Measured (`M-5`) by `TestTheConverseHolds` in `spec/tests/test_normal_form.py`, over 666 expressions and every binding of a four-name, two-type universe, with each candidate confirmed against a wider universe. No equality is left undecided. Measured again by `TestTheConverseHoldsOverUnionsOfEffects` in the same file, over a space that adds 484 unions of expressions that apply an effect, so that a union can sit above an effect as well as under it.

Two slots make it hold. The canonical form's filter slot holds a selector **keyed by the columns it picks**, not by the shape it was written in, and `T-9` makes that key exact. Two filters picking the same columns are therefore the same value, however each was reached. And a union carries no effect: an effect over a union folds into its operands (`D-22`), so $\mathtt{Suffixed}(\mathtt{both}(A, B), x)$ and $\mathtt{both}(\mathtt{Suffixed}(A, x), \mathtt{Suffixed}(B, x))$ are one form. The other slots are already canonical: an affix is a string, a data type is a name, and a union is flattened, sorted and free of repeats.

**Why the measurement is in two passes.** Grouping expressions by what they denote needs a universe to evaluate over, and a narrow universe makes two expressions look equal when a longer name would separate them. A coarse universe proposes every meaning the form appears to split, and a wider universe confirms whether each proposal is a real equality. Every proposal is refuted, which is what makes the count zero rather than vacuous.

### 8.5 What deciding costs

Containment between two regular sets is decided by building, complementing and intersecting automata, and the cost of that is not the cost of a lookup. The cheap path of section 8.4 keeps it off the common case: a containment between two prefix tests, which is what most user-written rules produce, never builds an automaton at all.

Two consequences for an implementation.

**Cache the machine.** A derived name test carries its automaton (`D-11`), so the machine is built once where the test is derived and not again at each comparison.

**Keep the cheap rules exhaustive over what users write.** Every requirement of section 5 produces a rule the cheap path settles. An automaton comparison is then reached only for a test the checker derived, which is where it is needed and where no error message depends on its shape.

______________________________________________________________________

## 9. Splitting a remainder

A remainder is one undifferentiated set of columns. Often a caller knows more: some of the remainder's columns are numbers, or all of a reference table's columns begin with `sec_`. A **split** records that structure.

### D-24. Split

```python
Numeric = other_columns(all_of=NumericType)
Rest = other_columns()


def totals(
    df: TypedDataFrame[Trades, SplitColumns[Numeric, Rest]],
) -> TypedDataFrame[TradeKey, Numeric]:
    return df.select("symbol", cs.numeric())
```

`SplitColumns[...]` with parts $A_1, \ldots, A_n$ declares that the remainder divides into $n$ parts. It fills **one** remainder slot, so the "at most one remainder" of `D-8` is untouched, and each part stays an ordinary declaration, which is what lets a return type name whichever parts survive.

`SplitColumns[...]` is the one spelling. There is no function form.

### D-25. The binding rule

Each part $A_i$ is declared with a per-column requirement (section 5.1), so it has a predicate $P_i = \mathrm{pred}(A_i)$ (`D-21`). **First match wins**: a column goes to the first part whose predicate accepts it. In closed form, part $i$ holds exactly the columns satisfying

$$
P_i\ \mathtt{and}\ \mathtt{not}\ P_1\ \mathtt{and}\ \cdots\ \mathtt{and}\ \mathtt{not}\ P_{i-1}
$$

So the parts are disjoint by construction and nothing needs to promise it. There is no search: one pass decides every column.

### T-11. A split adds no expression node

> Once bound, part $i$ is the ordinary expression that filters the whole remainder $G$ by the selector $P_i\ \mathtt{and}\ \mathtt{not}\ P_1\ \mathtt{and}\ \cdots\ \mathtt{and}\ \mathtt{not}\ P_{i-1}$ (`E-3`).

**Tag: proved.** Proved in Lean: `Spec.T11.split_is_filter` in `spec/lean/Spec/L5/Split.lean`. Measured by exhaustion on the executable binder, in `test_bound_parts_equal_the_closed_forms_everywhere` in `spec/tests/test_split_model.py`: over every column set of a three-name, three-type universe, first-match binding equals the closed forms part for part. The order-sensitivity control `test_first_match_order_carries_meaning` sits beside it.

First-match binding sends a column to part $i$ exactly when $P_i$ picks it and no earlier $P_j$ does, which is the displayed selector read one column at a time. That selector is in `D-12`, which is closed under $\mathtt{and}$ and $\mathtt{not}$, so the part is `E-3` applied to $G$.

So a split introduces no new node, no new absence constructor, no new closure row and no new canonical-form slot. It is a **binding form**: it adds a way to name parts of a remainder, and adds nothing to the language those names are written in.

An argument whose slot is a split therefore carries the record of the union $A_1 \cup \cdots \cup A_n$ of its parts (`D-19`): the parts partition the remainder, so a name is absent exactly when every part proves it absent.

### 9.1 What a split buys

**Each part after the first knows more than it was declared with.** `Rest` above was declared with no requirement, yet inside the body it provably holds no numeric column, because any numeric one would have gone to `Numeric`. That is the $\mathtt{not}\ P_1$ term, and it is why a split tells the checker something about **both** parts.

**It says something a single remainder cannot.** With one remainder a signature has two choices and neither is this one. `other_columns(all_of=NumericType)` demands that **every** column of the remainder be numeric, so a caller bringing a mix is turned away. `other_columns()` accepts that caller and then knows nothing about what arrived. A split is how a signature accepts the mixed caller **and** keeps track of which columns went where.

### M-4. What a split buys, counted

Take five columns outside the schema, of which two are numeric and three are not. A caller brings some subset, so there are 32 possible callers.

| The remainder is declared as | `select(cs.numeric())` gives | Callers settled |
| -- | -- | -- |
| `all_of=NumericType` | unchanged | 4 of 4 it admits |
| `none_of=NumericType` | emptied | 8 of 8 it admits |
| nothing at all | `Only[R, cs.numeric()]` (`T-20`) | 11 of 32 |
| `SplitColumns[Numeric, Rest]` | exactly `Numeric` | 32 of 32 |

The 11 are the shapes whose data types happen to be all numeric or all non-numeric: 4 plus 8, less the empty shape counted in both. The other **21** cannot be settled by any single family requirement: the result is sound, but it does not say which columns it holds.

> Measured by `test_a_family_requirement_answers_eleven_of_thirty_two` and `test_a_family_requirement_leaves_twenty_one_of_thirty_two` in `spec/tests/test_measurements.py`. The 11 and the 21 are complementary halves of the same 32. The split row's 32 of 32 is measured on the executable binder, and the per-part counts are derived from one split of the same five columns, in `TestTheSplitSettlesEveryCaller` in `spec/tests/test_split_model.py`.

### 9.2 The two calls that must be rejected

**A column no part will take has nowhere to go.** `SplitColumns[OnlyPrefixed, Numeric]` rejects a caller holding a plain `String` column, and the message can name it. Formally: the selector $P_1\ \mathtt{or}\ \cdots\ \mathtt{or}\ P_n$ must pick every column, or some caller has a column the split cannot place.

**A part an earlier part subsumes can never hold anything.** `SplitColumns[Rest, Numeric]`, with `Rest` unconstrained, leaves `Numeric` permanently empty, because $P_1 = \top$. Nothing would fail, and the user would never get the answer they wrote the split for. Formally: for each $i$, $P_i\ \mathtt{and}\ \mathtt{not}\ P_1\ \mathtt{and}\ \cdots\ \mathtt{and}\ \mathtt{not}\ P_{i-1}$ must pick some column.

Both conditions are satisfiability questions over `D-12`, decided by `T-9`.

> Measured by `TestTheTwoRejections` in `spec/tests/test_split_model.py`: the gapped split and the shadowed part are both caught over the finite universe.

The checker reports the shadowed part where the split is declared, because a part an earlier part subsumes is useless wherever it stands. It reports the gap at the **call**, where the caller is known: a closed argument holding a column no part takes is named in the message, and an argument whose own remainder promises nothing is reported without a name, since it can hold one. A caller whose record rules the missing shape out has no column to place, and nothing is reported there.

### 9.3 A part is its columns, not its name

After `cast(pl.String)` the part called `Numeric` holds no numeric column, so a later `select(cs.numeric())` correctly returns nothing from it. The declared requirement described the columns when they arrived. It is not a standing promise about them.

### 9.4 Why both parts are written out

A single name standing for the pair would read better, and it does not work. A type variable is bound by the checker from the annotation **as written**, so a part that appears only after a plugin expands a name is never bound at all.

| Written | Result |
| -- | -- |
| `SplitColumns[Numeric, Rest]` | both parts bind, and a return naming one is exact |
| a plain alias for the pair | the part resolves to `Never`, silently |
| a bare name a plugin expands | the part binds, and the *other* part errors as unbound |

**Tag: assumed.** Its source is an experiment against mypy that `spec/tests/` does not replay.

The middle row is the dangerous one. `Never` in a remainder slot reads as "this frame has no remainder", so it would claim a result is closed while it still carries the numeric columns. That is the unsound direction, and it comes with no diagnostic when the result is used directly.

______________________________________________________________________

## 10. Compatibility

### D-26. Satisfaction

A remainder expression $e$ **satisfies** a requirement $\rho$ when $\rho$ holds of $\llbracket e \rrbracket_\beta$ for **every** binding $\beta$. This is `D-19` again, one level up.

### T-12. Each requirement is decided from the record

**Tag: proved.** Proved in Lean: `Spec.T12.decide_iff`, `Spec.T12.disjoint_iff`, `Spec.T12.within_iff` and `Spec.T12.within_row_unsound` in `spec/lean/Spec/L5/Requirements.lean`. Measured for the three positive forms, the rows the parts of a split lean on, by `TestPositiveRequirements` in `spec/tests/test_requirements.py`: each is decided by entailing its shape's complement, and every decision is verified against every permitted name set.

Write $(P, \Theta)$ for the record of the remainder being checked, and $P_B$ for the absence rule of a remainder $B$ it names. A requirement holds of every column set the records permit exactly when its row holds, with one degenerate case: a record that permits no column, because $\Theta = \emptyset$ or $\llbracket P \rrbracket = N$, satisfies every per-column requirement, and a pair in which either record permits no column satisfies `R-10`.

| Requirement | Holds when |
| -- | -- |
| `R-1` none | always |
| `R-2` `without=[n, ...]` | $n \in \llbracket P \rrbracket$ for each $n$ |
| `R-3` `only=[n, ...]` | $\llbracket P \rrbracket \supseteq N \setminus \{n, \ldots\}$ |
| `R-4` `without_prefix=p` | $\llbracket P \rrbracket \supseteq \llbracket \mathtt{startsWith}(p) \rrbracket$ |
| `R-5` `only_prefix=p` | $\llbracket P \rrbracket \supseteq \llbracket \mathtt{not}\ \mathtt{startsWith}(p) \rrbracket$ |
| `R-6`, `R-7` | the same at the other end of the name |
| `R-8` `all_of=F` | $\Theta \subseteq F$ |
| `R-9` `none_of=F` | $\Theta \cap F = \emptyset$ |
| `R-10` `disjoint_from={B, ...}` | for each $B$: no name satisfies $\mathtt{not}\ P\ \mathtt{and}\ \mathtt{not}\ P_B$ |
| `R-11` `within=B` | the record permits no column; the records decide nothing more |

A one-column set is permitted whenever its column is, so a per-column requirement holds of every permitted column set exactly when every permitted column passes its selector. A column $(n, t)$ is permitted when $n \notin \llbracket P \rrbracket$ and $t \in \Theta$, so the test splits into the name half and the data-type half of the row. The same one-column sets on each side decide `R-10`.

**The records do not decide `within=`.** Two remainders are bound independently, so a caller can bind $B$ to no columns while the checked remainder holds one. Comparing $\mathtt{not}\ P$ with $\mathtt{not}\ P_B$ is therefore unsound: two unconstrained remainders pass that comparison and fail `within=`. From the records alone, `within=B` holds only when the checked remainder permits no column. Otherwise a closed caller discharges it by inspection, where both sides are known sets.

**Deciding one name is cheap.** Evaluating an absence rule at one name walks the rule once, so `R-2` costs one walk per name demanded.

**Deciding a relation is not.** `R-10` is a satisfiability question over the whole of `D-9`. Every constructor denotes a regular set (`D-11`), so it is decidable, by complementing and intersecting two automata and testing emptiness. That is not the cost of a lookup, which is why section 11 keeps it off the common path.

### T-13. Disjointness distributes over a split, and containment does not

> $R \cap (A \cup B) = \emptyset$ holds exactly when $R \cap A = \emptyset$ and $R \cap B = \emptyset$. $R \subseteq A \cup B$ follows from $R \subseteq A$ and from $R \subseteq B$, and implies **neither**.

**Tag: proved.** Proved in Lean: `Spec.T13.disjoint_union_iff`, `Spec.T13.subset_union_of_subset` and `Spec.T13.subset_union_not_imp` in `spec/lean/Spec/L5/Requirements.lean`. Measured by `TestRelationalDecisionsFromRules` in `spec/tests/test_requirements.py`, over every triple of subsets of a three-name universe, both as plain sets and as promises decided from a remainder's rule.

The first is distribution of intersection over union: $R \cap (A \cup B) = (R \cap A) \cup (R \cap B)$, and a union of two sets is empty exactly when both are. In the second, each part is contained in the union, and $R = \{a, b\}$, $A = \{a\}$, $B = \{b\}$ gives $R \subseteq A \cup B$ with $R \not\subseteq A$ and $R \not\subseteq B$.

The consequence is operational. A `disjoint_from=` promise about $n$ parts is $n$ independent questions, each the size of a question about one remainder. A `within=` promise about $n$ parts is one question about their union. Containment in one part proves it, but the promise can hold with no part containing the remainder, so the parts cannot replace the union.

### 10.1 Four consequences

**Checking reads knowledge, never syntax.** `Minus[Anything, "notional"]` satisfies `without=["notional"]` because the rule computed for it holds of `notional`, not because the two mention the same string. Any expression reaching the same rule satisfies the requirement equally.

**Weakening is free.** A remainder proving `a` and `b` absent satisfies a demand for `a`, because the test consults only the names demanded. No separate rule is needed.

**A promise satisfies a weaker promise, and the entailment runs one way.** A namespace promise satisfies a demand for any name inside that namespace, because excluding everything beginning with `sec_` excludes `sec_country` in particular. A list promise does not satisfy a namespace demand, however long the list, because $\llbracket \mathtt{startsWith}(p) \rrbracket$ is infinite and $\llbracket \mathtt{oneOf}(S) \rrbracket$ is finite. Between two namespace promises the shorter prefix is the stronger one.

**A requirement is never inferred from a body.** If a function's body calls something demanding `without=["sector"]` and the body's own remainder cannot prove it, that is an error in the body. The fix is to declare the requirement on the enclosing function, which pushes the obligation to its callers. A contract that changed when its body changed would silently break callers.

______________________________________________________________________

## 11. Deciding an operation

### 11.1 The procedure

For an operation applied to a frame with schema $M$ and remainder $G$:

1. Determine the operation's precondition. This comes from the operation's real behavior, not from the type system. `OPERATION_CATALOG.md` gives it, per operation, measured.
2. Decide it. The part concerning names in $M$ is decided outright, since $M$ is exact. The part concerning $G$ must be **proven** from $\mathrm{know}(G)$ by `T-12`.
3. If proven, compute the result's schema and remainder by section 7, and normalize by `D-22`.
4. If not proven, report an error.

### 11.2 A precondition is not always a conjunction

This is the subtlety that shapes the join rules. A join's true precondition is a **disjunction**: for each shared name, either the name is not shared after all, or its suffixed form is free. A record of absence rules cannot decide a disjunction of that shape over names it cannot enumerate.

This system does not approximate that disjunction. It demands the stronger condition that no rename can happen at all, which **is** a conjunction of per-name tests, so step 2 decides it exactly. A join that fails it is reported, not guessed at.

### 11.3 Where the expensive case is avoided

The `R-10` row of `T-12` is the only satisfiability question. Two cases avoid it:

**The caller is concrete.** The variable binds to a definite column set: the caller's actual columns, minus the callee's declared schema. Every requirement is then decided by inspection, and `R-10` and `R-11` are set operations on two known sets. This is also the one case that decides `R-11` for a remainder that can hold a column.

**The parts of a split.** A name-declared part and a complementary name-declared part cannot share a name, and that follows from the two declarations alone, with nothing checked at a call site.

______________________________________________________________________

## 12. Soundness and incompleteness

### T-14. The system is sound

> No operation this system accepts can fail on the column sets its record permits.

**Tag: proved.** As a corollary of `T-1`. Proved in Lean: `Spec.T14.sound` and `Spec.T14.requirement_sound` in `spec/lean/Spec/L5/Sound.lean`.

`T-1` says the record describes the column set under every binding that satisfies the declarations. Step 2 of section 11.1 accepts an operation only when a decision that is sound for records proves its precondition, as every row of `T-12` is. So the precondition holds under every such binding, which is the every-caller rule.

### 12.1 The system is incomplete, from four sources

Some correct programs are rejected. They are listed because a specification that hides its own limits is not trustworthy.

| Source | Example |
| -- | -- |
| **Strengthened preconditions** (section 11.2) | a join whose exact condition is a disjunction is refused unless the stronger conjunction holds, and diagonal concatenation likewise demands `R-10` disjointness rather than minting `Unchecked` under names not known to be present |
| **Approximation at case splits** | a data type differing between branches becomes `Unchecked`, even where a reader can see which branch is taken |
| **A relational fact between two written expressions** | concatenating a frame with itself after both have had the same column dropped is safe, and each side's guarantees are compared independently |
| **A wrap the effect language cannot spell** | `agg(pl.all())` wraps each remainder column in a `List` of its own data type: the record keeps the names and retypes to `Unchecked`, and the exact data types stay available by naming columns |

None of the four violates soundness. They are the price of never accepting a program that will crash.

### 12.2 Boundaries: what is outside the model entirely

These are not incompleteness. Nothing in `D-9`, `D-12` or `D-15` can express them, so nothing here pretends to.

| Id | Boundary | Example |
| -- | -- | -- |
| `B-1` | anything decided by **data** | `cast` failing on the row where the text is not a number |
| `B-2` | anything decided by **row count** | `hstack` needing two frames of equal height |
| `B-3` | anything decided by **row order** | `join_asof` needing a sorted key |
| `B-4` | anything decided by **column order** | vertical concatenation needing matching positions |
| `B-5` | anything decided by a **value** | `pivot` without a literal `on_columns=`, whose output names come from the data in a column |
| `B-6` | anything decided by **Python code** the checker cannot read | `map_elements` without `return_dtype=`, `.name.map(f)`, `rename(callable)` |

`B-4` is a consequence of a deliberate choice. `D-5` makes a column set unordered, so that two frames with the same columns written in a different order are the same type. That choice makes column order unavailable.

### 12.3 One sort of uncertainty that is permitted

An operation can keep a column's **name** while producing a **data type** that cannot be named. Summing a column of unknown data type is the usual case. The name is recorded in the schema and the data type is written `Unchecked`.

### D-27. `Unchecked`

`Unchecked` is the data type that stands for "this column exists under this name, and its data type is not checked". Formally it denotes the set of all data types, inside the reading `D-41` gives every data-type position, and its behavior under every operation follows from the lifting there: propagation where the operation's output varies with its input, recovery where it does not, and a type gate passing when some member passes. `D-27` and `D-41` define one marker: `D-27` for the columns of a frame, `D-41` for the data-type positions of an expression.

The two sorts of uncertainty are worth keeping apart. An **unknown name** weakens every later name lookup on the frame. An **unknown data type** weakens only the operations that read that one column. This system permits the second and rejects the first, which is why a fallback substitutes `Unchecked` under a known name rather than discarding the frame, wherever a name is recoverable at all.

______________________________________________________________________

## 13. Expressions

The system so far types frames. An operation's arguments are expressions: `pl.col("price") - pl.col("cost")`, a `when`/`then` chain, `cs.numeric().mean()`. This section and the five after it give expressions their type, the evaluation that reads the type at a frame, and the theorems that say what the evaluation gives.

An expression here is a computation a user writes. It is not a remainder expression (`D-18`), which describes the remainder and is built from effects. The two share a word and nothing else. From here on, $e$ is such a computation.

### D-28. Evaluation context

A frame operation evaluates each argument expression in an **evaluation context**, and there are two:

| Context | Fixed by |
| -- | -- |
| row | `select`, `with_columns`, `filter`, and every other row-shaped site |
| aggregation | `group_by(...).agg(...)`, and the interior of `.over(...)` |

### D-29. Row form and character

Polars gives every node of an expression a **row form**, one of four:

| Row form | Meaning | Examples |
| -- | -- | -- |
| one value per row | the height of the frame | `pl.col("a") + 1` |
| one value | one value that the planner knows to be a scalar | `pl.col("a").sum()`, `pl.lit(1)`, `pl.len()` |
| one row | exactly one row, which the planner does not know to be a scalar | `pl.col("a").head(1)` |
| another length | any other length | `pl.col("a").head(2)`, `pl.col("a").drop_nulls()` |

The row form of a node follows from the options of its node and the row forms of its inputs (`D-37`). A node that reduces its input gives one value, a node over inputs of one value gives one value, and a node over an input of one value per row gives one value per row. `.over(...)` gives one value per row whatever its input gives.

A declaration (`D-39`) states a **character**, one of three: **elementwise** for a value per row, **aggregating** for one value from the rows of a column, and **literal** for one value from no column. The character of an argument is read from its answers: it is literal where its code reads no column and polars builds it as a literal, elementwise where the aggregation context wraps the row answer in a `List`, and aggregating where the two contexts give one answer. A fresh root is not always literal: `pl.len()` reads no column and is aggregating, because it counts rows.

### T-15. The context rule

> Let $t$ be the data type an expression produces in the row context. In the aggregation context it produces `List(t)` when the expression gives one value per row, one row or another length, and $t$ when it gives one value.

**Tag: measured.** Measured by `TestContextRule` in `spec/tests/test_expression_facts.py`: `pl.col("a") + 1` inside `.agg(...)` gives `List(Int64)` where `pl.col("a").sum()` gives `Int64` and `pl.lit(9)` gives `Int32` unwrapped. The rules state it as the wrap of a node in `agg` (`D-36`), measured by `test_agg_wraps_a_node_that_the_planner_does_not_know_to_be_one_value` in `spec/tests/test_rules_options.py`.

The rule relates the two contexts for an expression whose answer is the same in both, and most expressions are such. Some nodes run in another way in each group, so their answer differs between the contexts in more than the wrap: `.mean()` on a `String` column raises in the row context and gives `String` in the aggregation context, and a node of a group-aware kind builds its list at its planned type. The evaluator follows the run in `agg` node by node (`D-36`), so these answers need no exception list. The interior of `.over(...)` is the aggregation context, so a window reads the answer of its input in `agg`.

### D-30. What an expression denotes

For each context $k$, an expression $e$ denotes a partial function $\llbracket e \rrbracket_k$ from an input column set (`D-5`) to a finite non-empty sequence of output columns. Partial, because polars raises on some inputs. Only names and data types are in the model. Values, lengths and row order are not, for the same reason `D-4` gives a column no length.

### D-31. Reads

The **reads** of an expression are the names of the columns it reads: every column leaf of its tree (`D-34`). Every input of a node runs, so a column that steers the evaluation is read like any other. The condition of a `when`, the mask of a `.filter(...)`, the key of a `.sort_by(...)` and the partition of an `.over(...)` each add their names, and a failure there makes the whole expression fail.

> Measured by `TestMetaIntrospection` in `spec/tests/test_expression_facts.py`: polars computes the reads itself, as `meta.root_names()`, and the column of a `when` condition is a root while the branches name the output.

### D-32. The output name

The **output name** of an expression is the name of the column it appends. Polars computes it from the tree, as `meta.output_name()`, and the checker takes it from there:

- a column reference gives its own name, and an operation over several columns gives the name of its first input that reads a column: `pl.col("a") + pl.col("b")` is named `a`,
- `pl.lit(...)` is named `literal` and `pl.len()` is named `len`,
- a `when`/`then` chain takes its first branch's output name,
- `.alias(n)` replaces the name, and `.name.suffix`, `.name.prefix`, `.name.to_lowercase`, `.name.keep` and their kin rewrite it, computably, because the code holds the name being rewritten.

A keyword argument of a frame operation replaces the output name outermost, an inner `.alias(...)` included, so `with_columns(z=e)` appends `z` whatever `e` says (`test_a_keyword_argument_replaces_the_output_name_outermost` in `spec/tests/test_expression_facts.py`).

`.name.map(f)` applies an arbitrary callable to the name. No checker can evaluate it, so the output name becomes unknown, and an unknown name is the sort of uncertainty this system rejects rather than tracks (`D-27`). `.name.map` is outside the typed fragment (`B-6`).

A selection expression names no single output. Its outputs take the matched columns' names, rewritten by its affix (`D-38`). One other member is multi-output, `.struct.unnest()`, and its outputs are the fields of its struct, the one output name that a data type decides (`T-21`).

### D-33. The type of an expression

The **type of an expression** is its code, in one of two forms:

- A **deferred** type holds the code of the expression, written with the public API of polars, as the user wrote it. It covers every input, and it computes nothing until the expression meets a frame or a declaration. `pl.col("price") * pl.col("qty")` has the type `TypedExpr[Deferred["pl.col('price') * pl.col('qty')"]]`.
- A **declared** type is a declaration that the user writes (`D-39`). It covers the cells that it declares, and it is undefined elsewhere.

A declared type can be a leaf of a deferred type, for a parameter of a function: the code refers to it as `$0`, `$1` and so on, and the type holds it as an argument. A call of a function keeps the declared leaf and does not copy its argument into the code, because two arguments that fit one declaration are alike for every body that reads it (`T-23`). A part that the code uses more than once is written once, at its first use, as `($1 := <code>)`, and as `$1` after that. The recorder writes one canonical text, so `pl.col("a")` and `pl.col('a')`, or two orders of keyword arguments, give one code.

What a type means is its **map**: the columns it reads, and the outcome of the evaluation (`D-36`) at each frame. So `a + b` and `b + a` have one map, although their code differs. The host checker compares types by how they are written, so the checker decides each comparison that matters on the map, by the fit of `D-40`, and never by the spelling. A type holds no fact of the passes of polars, so a new version of polars changes the evaluator and not the stored types.

The type decides materialization: the evaluation of `D-36` reads only the code of each argument and the schema of the frame, so two arguments with equal types give equal result frames at every site, in every context.

### D-34. Tree, leaf and hole

Polars stores an expression as a **tree of nodes**, which `meta.serialize()` shows. A **node** applies one operation of polars to its **inputs**, in order. Every sub-expression of a call is an input of its node, the receiver included: in `pl.col("x").clip(pl.col("lo"), pl.col("hi"))` the three columns are the three inputs of one node. A **leaf** brings a value in:

| Leaf | Reads | Gives |
| -- | -- | -- |
| column reference, `pl.col("price")` | the one named column | that column |
| literal, `pl.lit(v)`, a Python value | nothing | its value, with the type of its value |
| selector, `pl.col("a", "b")`, `cs.numeric()` | the matched columns | `D-38` |
| element and field, `pl.element()`, `pl.field(...)` | nothing of its own | the element of the enclosing `list.eval` or `arr.eval`, or a field inside `pl.struct` |

A column reference is written exactly as polars has it, and the code adds no second spelling: expression code is plain polars, always. A literal keeps its value in the code, because polars reads the value: `pl.col("u") + 1` over `u : UInt8` gives `UInt8`, and `pl.col("u") + 300` gives `UInt16`. An element or field leaf outside its binder is an error, polars' own (`test_the_element_root_is_bound_by_its_enclosing_call` in `spec/tests/test_expression_facts.py`).

A **hole** stands in the code for a value that the checker cannot know, such as `n` in `shift(n)` where `n` is an `int`. An infinity or a NaN that the call writes, as `float("inf")`, `float("nan")`, `math.inf` or `math.nan`, is a value of the code and no hole: `pl.lit(float("inf")).cast(pl.Int64)` raises at every run, and the evaluator reads it so, although the tree of polars writes such a float as null. A NaN that the code holds is a known NaN too, so `pl.col("x").cut([float("nan")])` plans a bin named `NaN` and raises at every run (`test_cut_over_a_nan_break_that_the_code_holds_gives_the_answers_of_polars` in `spec/tests/test_rules_definitions.py`). A hole keeps the Python type of its value: `bool`, `int`, `float`, `str`, a date or a time, a number of either sort, a value of the type `Any`, or an expression whose type holds no code. A function of the user, as in `map_batches(f)`, is a hole too, and only its `return_dtype=` gives its type (`B-6`). `D-44` states what the evaluation reads of a hole.

A reference whose name argument the checker cannot read, `pl.col(key)` over a string computed at run time, is the territory of `B-5`: the reads become unknown, and an unknown name is refused rather than tracked (`D-27`). A variable that holds a literal string is read like the literal, because the code records the value of a `Literal` type. A selector whose name or pattern comes from a variable of the type `str`, as `cs.matches(pattern)` or `cs.by_name(name)`, is the same: the data decides which columns it picks, so the call gives the plain frame for the same reason. A declared column whose name is a plain `str`, as a parameter `c: TypedColumn[str, pl.Int64]`, is a leaf of its declared data type whose name no code knows: `pl.col("a") + c` gives `Int64`, an output that `c` names has no name, and the frame need not hold the column that `c` reads, so the read is unproven at the level `stated`, as any name that is not a literal. A name that is a union of literals, as `Literal["a", "b"]`, makes the same leaf, and its read is proven where the frame declares each name of the union. A name of the type `Any` makes no leaf, so the expression is `Unchecked`.

______________________________________________________________________

## 14. The evaluator

The data type an expression produces is not one data type: it depends on the data types the frame supplies. The type holds the code, so the dependence is computed where the expression meets a frame, by an evaluator that follows polars node by node.

### D-35. The state of a node

The evaluator computes a **state** for each node of the tree. The state holds the facts that the passes of polars read:

| Fact | What it is | Who reads it |
| -- | -- | -- |
| planner type | the type that `collect_schema()` shows | the type function of a parent, the first visit of the cast pass to a parent, and a lazy frame, which stores it (`D-45`) |
| type after the first pass | the type that the cast pass gives the node | the second visit of the cast pass to a parent, and the root |
| run type | the type of the data | the kernel of a parent, and a frame operation |
| all values null | whether every value of the data is null | the kernels and the casts, which take a shortcut for such input |
| row form | `D-29` | the parent, the context and the frame |
| state in `agg` | the state that the run gives the node in each group | the run of a parent in `agg` |
| elementwise | whether every node below works value by value | `over` in `agg`, which needs an elementwise partition key |

The state exists only during one evaluation. The type never stores it (`D-33`).

A node is elementwise where polars marks it so: a column, a literal of one value, a cast, a binary operation, a `when`, an `eval` over each list, and a function whose options are elementwise. `is_in` is elementwise only where its set of values is a literal, so `pl.col("x").over(pl.col("k").is_in([1, 2]))` runs in `agg`, and `pl.col("x").over(pl.col("k").is_in(pl.col("x")))` raises there (`spec/tests/test_rules_tree_evaluation.py`). A key of `over` in `agg` must be elementwise at every node below it, so `pl.col("x").over(pl.col("k").sort())` raises there although `sort` keeps the rows (`TestElementwiseKeys` in `spec/tests/test_expression_facts.py`).

A literal adds three facts that polars reads: its value, whether it is direct, as `pl.lit(300)`, or derived, as `pl.lit(300).abs()`, and the value that a fold of literals computes. Beside `Int8` under `*`, with the literal on the left, the planner gives `Int8` for the first and `Int16` for the second. With the literal on the right, the first also gives `Int8` before polars 1.43.0, and `Int16` from that release.

### D-36. The three passes, and the outcome

Polars reads a tree in three passes, and each can give a node another type:

1. **The planner.** `collect_schema()` computes the type of each node with its **type function**, on the tree as the user wrote it. This is the **planner type**.
2. **The cast pass.** Polars changes the tree so that the inputs of each node agree: it casts an input, gives a literal a new type, implodes an input, or puts a new node in place of a comparison. It visits a node before its inputs and again after them, and each visit reads the types of the inputs at that moment. The root of a `select` has one visit, after its inputs, so it reads each node below it at its type after the first pass; a rename at the root, as `.alias()` or `.name.keep()`, leaves the node below it the root. Over a `Datetime` column `d` and a `UInt128` column `y`, `pl.col("d").ewm_var(alpha=0.5).search_sorted(pl.col("y"))` is planned at the root, where `ewm_var` reads as `Float64`, and refused under `.abs()` and in `agg`, where `search_sorted` also reads the planned `Datetime` (`test_a_rename_at_the_root_leaves_the_node_below_it_the_root` in `spec/tests/test_rules_tree_evaluation.py`). A node below the root, and a root that `agg` wraps, reads each node below it at its planner type at the first visit and at its type after its visits at the second, so over a `String` column `t`, `pl.col("t").ewm_std(alpha=0.5) - pl.col("t")` is refused in `agg` at the second visit, where `ewm_std` reads as `Float64`.
3. **The run.** The **kernel** of each node computes the data on the changed tree. The data type of the result is the **run type**.

So each node has a planner type and a run type, and the two can differ: `pl.col("d").cum_sum()` over `d : Decimal(10, 2)` is planned as `Decimal(10, 2)` and runs as `Decimal(38, 2)`. The evaluator repeats the three passes, each node from the states of its inputs, from the leaves up, and it sees the whole tree, so it knows which node is the root. In the aggregation context each input holds the values of one group, and the evaluator reads the kernel at the states in `agg`. It evaluates `agg` over two groups of different sizes, the normal case of a `group_by`, so a construct that runs only with one group is an error.

The `agg` of `rolling` and of `group_by_dynamic` runs over windows of the rows of an index column, and the evaluator takes such windows in place of the groups. Windows can share rows: those of `rolling` always do, and those of `group_by_dynamic` do where `period` is longer than `every`, or as long with `closed="both"`. Over windows that share rows, a group-aware node of one input that is not aggregated runs on each window and builds its lists at the type of their data, not at its planned type. So over a `String` column `t`, `pl.col("t").rolling_mean(2)` gives `List(Float64)` in the `agg` of `rolling`, and panics in the `agg` of a `group_by`. A slice keeps the windows of its input, and an aggregation ends them (`spec/tests/test_rules_windows.py`).

An `over` runs its function over the groups that its keys make. A key with the rows of the frame makes groups that its values decide, so the evaluator takes the groups of `agg`, and the rows of the function are a fact of the data. Where every key holds one row, the keys make one group of the whole frame. With the mapping `group_to_rows`, a function that does not give one value must then keep every row of the frame, so over the five rows of the evaluator `pl.col("x").head(3).over(pl.lit(0))` raises in a `select`, where the two groups of `agg` hold three rows and two. In `agg`, each group is the window, and the same check runs on each group. The mapping `join` gives each row a list of its group, so it checks no rows, and the mapping `explode` gives the rows of the groups one after the other (`test_an_over_of_keys_of_one_row_runs_over_one_group_of_the_frame` in `spec/tests/test_rules_execution.py`).

A lazy `select` that holds one expression runs the slice pushdown first. It reads a slice, `first` and `last` as slices, and moves each one down the nodes that work value by value to the deepest places that it reaches, so a slice above `pl.col("x") + 1` cuts the column too, and a slice that a node stops before every column stays where the node stops it. A slice over a slice joins into one slice. Where every column that the expression reads, and each `pl.len()`, then lies directly below a slice of literal bounds, and the slices join into one set of rows, slices from the start into the least offset and the greatest end and slices from the end alike, the pushdown puts that set of rows in the plan: a slice of a column as long as the plan's goes away, and any other slice moves its offset by the start of the plan. The expression then runs as the pushdown leaves it, over the rows of the plan, so `lf.select(pl.col("x").head(2).over(pl.lit(0)))` gives `Int64` where an eager `select` raises (`test_a_lone_select_runs_over_the_rows_of_its_moved_slices` in `spec/tests/test_rules_execution.py`). The projection pushdown then puts `pl.len()` in the place of a lone `len` of an input with the height of the frame, so the run never meets a refusal of the input. So over a `String` column `s`, `lf.select(pl.col("s").abs().len())` gives `UInt32`, where an eager `select` raises, `pl.col("s").abs().head(2).len()` counts too, and `pl.col("s").abs().sort().head(2).len()` raises. Polars counts the expressions after it expands a selector. A refusal of the planner or of the cast pass stands, and `count`, a second expression, `with_columns` and `agg` keep the input (`test_a_lone_len_in_a_lazy_select_counts_the_rows` in `spec/tests/test_rules_execution.py`). The rules translate the pushdown step by step (`SlicePushdown`). A slice whose offset or length is not a literal moves into that offset or length too, which then reads the slice itself, so the pushdown builds a cycle: where the expression reaches the cycle, the query never returns, as it does for the slice node alone (`test_the_slice_pushdown_hangs_where_a_cycle_reaches_the_value` in `spec/tests/test_rules_execution.py`). A node that no definition states, except an `eval`, leaves the move unknown where a slice stands in the expression, and there the run of a lone `len` is `Unchecked`, and so is a run that a check of the rows refuses.

The evaluation of an expression at a frame gives an **outcome** in each context, one of three:

- a type: the run type, with the row form and the output name,
- `Unchecked`: polars runs the expression, and the checker does not know its type,
- undefined: polars refuses, and the checker reports an error at the frame operation, where polars itself fails.

From polars 1.44.0, a `when` whose mask has one row runs only the branch that the value of the mask picks. Where its two branches have other lengths, neither of one row, the output has the rows of one branch, which the data decides, so its row form is another length. A mask of more rows that holds both values runs both branches, and polars raises where a branch does not fit the mask. A mask of more rows that holds one value only runs one branch and raises nothing. That `when` is an exception to the rule below on failures that the data decides: the evaluator gives it as undefined, as every release before 1.44.0 does for every mask. A mask that holds one value makes the `when` give one of its branches, which the code can state without a `when`, so the failure marks a fault in the code and not a fact of the data (`test_select_gives_the_rows_of_polars` in `spec/tests/test_rules_execution.py`).

A frame column takes the run type, because a frame operation runs its expression. A node fails in four ways: the planner refuses, the run refuses, the process crashes, or the process hangs. The evaluator gives the four as undefined where the source of polars shows them, and `POLARS_ISSUES.md` lists the crashes and the hangs.

A run can raise for some values and not others. Where it raises for every sample of values, the cell is undefined, a type error. Where it raises only for some, such as `pl.col("a").cast(pl.UInt8)` over an `Int64` column, the failure is a fact of the data, and the cell keeps the run type. The code holds the value of a literal, so a run over a literal meets that value every time: `pl.lit("a").str.to_date("%Y-%m-%d")` raises at every run, because no date has the shape of `"a"`, and `pl.lit("01").str.strptime(pl.Time, "%H")` raises too, because a time needs its hour and its minute, and the hour of a 12-hour clock its half of the day (`TestLiteralTextParse` in `spec/tests/test_expression_facts.py`). A date needs its year with its month and day, or with its day of the year, where `chrono` parses it. Polars first reads an exact format of fixed-width directives (`%Y`, `%y`, `%m`, `%d`, `%b`, `%B`, `%H`, `%M`, `%S` and the fractions) with a parser of its own, which gives a missing month or day the value 1, so `pl.lit("2024-01").str.to_date("%Y-%m")` gives `Date`, and `pl.lit("2024-1").str.to_date("%Y-%m")` and `pl.lit("2024%01").str.to_date("%Y%%%m")` raise. A format that no text fits raises wherever the text holds a value, over a column too, and polars refuses a format with an hour and no minute, a second and no hour, or a 12-hour clock and no half of the day at every frame of an exact parse into a date or a moment (`test_a_format_without_a_day_parses_only_what_the_fixed_width_parser_reads` and the two tests after it in `spec/tests/test_rules_kernels.py`). A binary node of two literals computes their fold wherever it stands, so the count of `pl.col("x").first().diff(pl.lit(1) + 0)` is the literal `1`, and the conversion folds such a node only where it is the root. A node that keeps the one value of a literal, as `first`, `max`, `unique`, `sort` or `str.join`, and a cast that keeps the value, which polars folds, give that value to the rule above them, so `pl.lit("a").str.join().cast(pl.Time)` raises at every run (`spec/tests/test_definitions_values.py`). Any other function of a literal, as `abs` or `str.to_uppercase`, gives a column of no known value. Polars also runs `str.strptime` as an elementwise function where its text reads no column, so in `agg` such a parse beside an `ambiguous` of one value for each group gives one row, which the groups refuse. The facts of the data that decide a run, such as the length of an input against the others or the values of a mapping, are in no type, and the checker accepts them. The run marks a length that the values decide, the length of a `filter`, a `drop_nulls`, a `unique` or an `explode`, and of each node that keeps such a length: a check of lengths passes it beside any other length, and `item` passes it, because the data can give the length that the check needs. So `pl.col("x").filter(pl.col("x") > 5).item()` runs where one row is left, and `item(allow_empty=True)` also runs over no row (`TestLengthsTheValuesDecide` in `spec/tests/test_expression_facts.py`). The code bounds other lengths: a literal and one value hold one row, and a slice of a literal length, as `head(2)`, at most that many on every frame. A literal index or count past such a bound fails at every run, so `pl.col("x").head(2).gather(300)`, `.get(300)` and `.sample(300)` are errors, where `pl.col("x").gather(300)` depends on the rows of the frame (`TestBoundedRows`). The run knows where an output holds only nulls: a node that keeps the nulls of its input, as a cast, a slice, a sort or `+`, gives only nulls where its input holds only nulls, so `pl.lit(None, dtype=pl.Decimal(10, 2)).head(2).dt.epoch()` runs, and `fill_null(1)` in place of `head(2)` raises (`TestNullsThroughANode`, and `spec/tests/test_definitions_nulls.py` for the rule of each node). A node also gives only nulls where the code bounds its input to few rows: `shift(1)`, `diff(1)` and `pct_change(1)` over one row, a rolling function or an exponentially weighted function over fewer rows than its `min_samples`, `std` and `var` over no more rows than `ddof`, and `ewm_std` and `ewm_var` with `bias=False` over one row. So `pl.col("x").diff(pl.lit(1.5).rolling_kurtosis(2))` raises, because its count is null, and `pl.col("x").first().std().cast(pl.List(pl.Date))` gives `List(Date)`, where a value gives `List(Int32)` (`ShortCalls` in `spec/tests/test_definitions_nulls.py`). The same rules say where an output holds a null wherever it holds a row, because the code decides it: a count other than zero leaves a null in place of the rows that `shift`, `diff` and `pct_change` move past, and a window that needs two samples leaves its first row null. A node that works value by value keeps that fact, so `pl.rolling_corr(pl.col("x").shift(1), pl.col("x").first(), window_size=2)` panics, as polars does where an input holds a null beside an input of one row (`test_a_null_that_the_code_decides_meets_one_row`). A slice of length zero holds no value, so the run reads it as values that are all null: polars casts it to any type with no kernel, and `pl.col("l").head(0) == pl.col("t").first()` runs over a list `l` and a text `t`. In `agg`, a strict cast reads the values of each group, and a cast that is not strict reads the whole column of an input that is not aggregated, the rows that a slice drops included (`TestSliceOfNoRow`). `extend_constant` over an input of no row keeps its value as it is, in a column of the type of the input, and polars panics where it builds that column from a value of another kind, at the first read of a `select` and in the lists of `agg`, so the evaluator refuses it in both, although `collect_schema()` gives the type of the input (`TestExtendOverNoRow`). A literal of a series holds its own number of rows in a `select`, whatever the height of the frame, and gives all its values to each group in `agg`. Where the values decide the run type itself, the cell is `Unchecked`: in `agg`, `pl.col("i").quantile(pl.col("q").first())` over an integer `i` gives `Float64` where `q` holds a value between 0 and 1, and keeps the type of `i` otherwise (`TestQuantileColumn` in `spec/tests/test_expression_facts.py`). Where the number of rows decides the run type, the cell is `Unchecked` too: over a `Date` column `d` and an `Int32` column `i`, `pl.col("d").implode().list.slice(pl.col("i"), 1)` gives `List(Date)` over one row and `List(Int32)` over more (`TestRowsDecideAType`).

### D-37. Rules, definitions and the registry

Polars declares the rules of each operation in a small vocabulary, and the evaluator follows the same vocabulary:

| Part | Holds |
| -- | -- |
| data types and the supertype | the data types, the tests on them, and the type two types join into |
| casts | the casts that exist, the type each gives, and the values that survive |
| options | how a node runs: elementwise, length preserving, group-aware or an aggregation, and whether the cast pass casts its inputs to a supertype |
| type functions | the planner type of each node from the planner types of its inputs |
| the cast pass | the casts of binary operators, of the nodes that cast to a supertype and of `when`, and the few rules of their own |
| kernels | the run type of each node from the run types of its inputs |
| the run | the state of each input in `agg`, the list each group builds, the fold of literals, the slice pushdown, and the count of the rows that a lazy `select` gives a `len` that stands alone |

The **definition** of a node names the parts of the vocabulary that the node uses. A behavior of polars that no part states, such as a bug of polars or a rare edge case of one node, is a measured fact of the **registry**, with its reason and the probe that measures it. Where it needs code, the registry holds an **override**, a rule written for one node that stands in one part of its definition.

Every rule covers every input. A rule is a function of the types, the row forms and the literal values of its inputs, so it covers every data type, every parameter and every literal value, a nested type of any depth and any number of inputs. Where no rule and no fact of the registry gives an answer, the answer is `Unchecked`, so a guess never becomes a type. A node that no definition states is `Unchecked` (`test_a_node_with_no_definition_is_unchecked` in `spec/tests/test_rules_tree_evaluation.py`).

### M-20. Polars types a node from the states of its inputs

Polars computes the state of a node only from the states of its inputs. Measured by the verification of random trees, `spec/tools/rules/tree_check.py`, and by `test_the_evaluation_of_a_tree_gives_the_answers_of_polars` in `spec/tests/test_rules_tree_evaluation.py`, over trees whose nodes meet in every row form and both contexts.

### M-21. Each definition gives the answers of polars

Each part of the vocabulary, and each definition, gives the state of polars at every state of its inputs that its tests sample: every kind in every row form at each input, with literals of each class, in a `select` and in `agg`. Measured by the tests of each part, from `spec/tests/test_rules_supertype.py` to `spec/tests/test_rules_execution.py`, and by the tests of each family of definitions, from `spec/tests/test_definitions_aggregations.py` to `spec/tests/test_definitions_temporal.py`. The census holds no `Object` kind, so the casts of `Object` are measured apart: the kernel casts a series of Python objects only to `Object`, and raises `ComputeError` for any other target, strict or not, where the series holds a value (`TestObjectCasts` in `spec/tests/test_rules_casts.py`).

### T-25. The evaluator gives the answer of polars

> Where each rule gives the state of polars at the states of its inputs, the evaluator gives the state of polars on the whole tree, at every frame.

**Tag: proved** from `M-20` and `M-21`. Proved in Lean: `Spec.Node.T25.evaluate_eq_answer` in `spec/lean/Spec/L6/Node.lean`.

By induction over the tree. A leaf's state is the rule of the leaf, which `M-21` makes polars' own. A node's state is the rule of the node at the states of its inputs, which the induction makes polars' own, and `M-20` says that polars computes the node's state from those states alone. The hypothesis `M-21` is a measurement over samples, not over every input, so the theorem transfers exactly the confidence the tests give each rule, and no more.

______________________________________________________________________

## 15. Values the checker cannot know

The code holds what the user wrote. Some of it the checker cannot know: a value in a variable of a plain type, a column of an open data type, and an expression built in a branch or a loop. This section states what the evaluation does with each.

### D-44. Holes and open columns

A hole (`D-34`) is evaluated at a fixed **sample** of values of its Python type: an `int` at $1$, $-1$, $2^{15}$, $2^{40}$ and $2^{63}$, a `float` at $0.5$ and $2^{100}$, a `str` at two texts, a `bool` at both values, and a date or a time with and without a zone. Where every choice of the sample gives one answer, that is the answer: the count of `shift` never changes a type, so `pl.col("a").shift(n)` keeps the data type of `a`. Where the answers differ, the value decides the answer, and the answer is `Unchecked`: `pl.col("u") + n` over `u : UInt8` gives `UInt8` for $1$ and `UInt16` for $2^{15}$. A value of the type `Any` and an expression whose type holds no code give `Unchecked`, with the name kept where polars computes it without them. A text in the place of a format, as in `pl.lit("2020-01-01").str.to_date(fmt)`, decides whether a literal text parses, and neither sample text need fit the literal, so where every choice refuses and a format is such a hole, the answer is `Unchecked`: a format that the sample does not hold, as `"%Y-%m-%d"`, can fit it.

A column whose data type the frame does not fully declare is **open**: `Unchecked`, a family such as `NumericType`, a kind with open parameters such as a bare `pl.Datetime`, or a nested type that holds such a part, as `List(Unchecked)`. Polars needs a full data type to type an expression, so the evaluation tries each data type of a fixed **sample** of the open type. The sample holds one data type for each kind that holds no other type, and several for each kind whose parameters change the answers of polars: the units and zones of a `Datetime`, the units of a `Duration`, the precision of a `Decimal`, the categories of an `Enum`, and a `List`, an `Array` of two widths and a `Struct` of each of those. The fields of an open struct are not known, so each `Struct` of the sample holds a field whose name no code writes, beside each field that the code reads, and an answer that keeps that field has fields that the open type does not name, so its struct is `Unchecked`. A hole of the kind `Struct` also stands for a struct of exactly the fields that the code reads, since polars casts a value that is not a struct only to a struct of one field (`TestStructCastNeedsOneField` in `spec/tests/test_polars_facts.py`), and so does the kind `Struct` of a remainder, whose columns a caller can bring with those fields: `cs.struct().struct.field("x")` then runs for some callers and raises for others, so the call is unproven. A choice where polars refuses drops out, so a gate passes where some member passes (`D-41`). A selector that tests the data type can pick an open column at some members of its sample and not at others, and then the data decides whether it picks the column. Where the arguments give the columns of the result, as in `select` and `agg`, the result holds the column for some data and not for other data, which no type says, so the result is unproven at the level `stated` and the checker gives the plain frame. Where the column stays either way under its own name, as in `with_columns`, its type is what its answer and its own data type share. A new name, as `.name.suffix()` gives, exists only for the data that the selector picks, so it is unproven too. A selector that picks the column at every member or at none is exact, and a selector of keys or predicates, as in `sort` and `filter`, leaves the columns of the result as they are. The answer is what the answers of the other choices share (`T-26`). Where every choice refuses, the outcome is undefined. A column that the code reads, which the frame does not declare and its remainder can hold, is unproven (`undeclared-column`), and where a caller holds it, it can have any data type, so the evaluation reads it as an open column: `pl.col("ask").map_elements(str, return_dtype=pl.String)` gives `String`, and `pl.col("ask") + 1` gives `Unchecked`.

An answer that holds at every member of a sample is not proved for every value or every data type the sample stands for. The samples are chosen where the rules of polars change their answer: a bucket of integer width, a sign, a scale, a unit, a zone, a nesting. That choice is an assumption of this system, the one that `T-26` and `D-41` rest on.

### T-26. Each answer fits what the answers share

> What several answers share, their **generalization**, is their common parts: a kind with its parameters and its inner data types where they agree, the kind alone where a parameter or the number of inner data types differs, and `Unchecked` where the kinds differ. Every answer fits the generalization of a list of answers that holds it.

**Tag: proved.** Proved in Lean: `Spec.Generalize.fits_join_left`, `Spec.Generalize.fits_join_right` and `Spec.Generalize.fits_generalize` in `spec/lean/Spec/L6/Generalize.lean`.

So `List(Int8)` and `List(String)` generalize to `List(Unchecked)`, and `Datetime("ms", None)` and `Datetime("us", None)` to `Datetime` with its parameters open. A generalization is an upper bound of each answer in the order of `D-41`, so the type it gives holds every answer of the sample.

### D-42. Branches and loops

An `if`/`else` that assigns an expression gives the variable a union of two types, and a conditional expression `a if c else b` gives the same. A loop that rebuilds an expression gives a union of its unrolled versions, because the host checker binds a rebound variable to the union it accumulates over a bounded re-analysis of the loop (`M-6`).

A frame operation evaluates each member of a union argument. An argument of two members, as an `if`/`else` gives, makes `select`, `with_columns` and `agg` give a union of frames, one for each choice of a member of each such argument, and a choice that gives the frame of another choice adds no member: `df.select(pl.col("a") if flag else pl.col("a").cast(pl.String))` over `a : Int64` gives one frame with `a : Int64` and one with `a : String`, and the host checker checks each later step once for each frame. An argument of more members can be the versions that a loop leaves, and a later pass can give an answer that none of them gives (`M-6`), so it keeps one frame, and so do an `if`/`elif`/`else` of three branches and a call whose choices are more than eight. There, where every member gives one output name and one data type, that is the output (`T-18`). Where the names agree and the data types differ, the output keeps the name under `Unchecked`, and the gate reports the loss. Where the names differ, the argument gives no column, and the operation falls back.

A join of two types inside the host checker, as in the list `[a, b]`, keeps the code where the two codes are equal. Where they differ, the host checker erases the code, also where the two maps are equal, and the elements are plain expressions.

### M-6. How mypy types a rebound loop variable

Three behaviors, each checked in `spec/tests/test_mypy_binder.py` against the installed mypy:

1. By default, an unannotated variable is pinned to its first assignment's type, and the loop's rebinding must fit it (`TestDefaultBinding`).
2. With `allow_redefinition`, and identically with `allow_redefinition_new` plus `local_partial_types`, the rebinding is legal and the variable's type after the loop is the union accumulated over a bounded re-analysis of the body (`TestRedefinitionBinding`).
3. The bound is real. A step whose result type strictly grows is cut off four levels deep under a ten-iteration loop, so the union does not contain the runtime value's type. A step that maps the union back into itself infers exactly `Seed | Step`.

A deferred type grows under every step that appends to its code, so a loop that rebuilds an expression meets behavior 3: the union holds the versions of the first passes and not the version of the last one.

### T-18. A loop whose versions agree

> Where every member of the union that a loop leaves gives one outcome at a frame, the loop gives that outcome at that frame for every number of passes.

**Tag: assumed.** It is the reading this system gives behavior 3 of `M-6`. A step whose versions agree on the first passes is taken to agree on the later ones, and a step whose versions differ gives `Unchecked` by `D-42`. The assumption fails only for a step whose outcome changes after the passes that the host checker unrolls. The typed way to remove the assumption is to annotate the seed of the loop with a declaration, so that the loop variable keeps one declared type and the loop rebinds nothing that grows.

______________________________________________________________________

## 16. Selections

### D-38. The type of a selection expression

A selection expression is rooted in a selector rather than in named columns, and polars marks the root itself: `meta.has_multiple_outputs()` holds, and the mark survives operations applied on top (`TestMetaIntrospection` in `spec/tests/test_expression_facts.py`). Its type is its code, like any expression's (`D-33`). At a frame, polars expands the selector into one column per matched column, and the code stands for each in turn: the evaluator puts the matched column in the place of the selector and evaluates the code (`D-36`). Four parts of the code decide what the selection does:

$$
(C,\ N,\ u,\ \rho)
$$

where $C$ is the selector (`D-12`), $N$ is the set of named columns the code reads besides the ones it picks (`D-31`), $u$ is the outcome of the code at one matched column, as a function of that column's data type and the data types of $N$, and $\rho$ is a name rewrite: the identity, a suffix, a prefix, or one fixed name. `cs.numeric() + pl.col("a")` has $N = \{a\}$: each matched column meets `a`, and each output keeps its matched column's name. Two selections of one selector pair column by column, so `cs.numeric() * cs.numeric()` reads the matched column on both sides.

**A column on the left of a selection names every output.** In `pl.col("a") + cs.numeric()` each output takes the left operand's name, `a`, so polars raises a duplicate error as soon as the selection picks two columns. The rewrite $\rho$ therefore holds a fourth form beside the three admissible ones, a **fixed name**, which sends every matched name to one name. A keyword argument and an outermost `.alias(...)` give every output the same fixed name. It is not a rename of the remainder, so a selection that carries it never fits a declared selection, and it is read only where the outputs are known: at the frame, no matched column gives no output, one gives that one output, and two or more give the duplicate error (`test_a_fixed_name_over_no_picked_column_gives_no_column` in `spec/tests/test_expression_facts.py`). Over an open frame, the fixed name meets the remainder. Where the record allows a column that the selector picks, the number of outputs depends on the caller, and so does the result: the call is unproven, a result the remainder decides (section 18.5). Where the record proves that the remainder holds no such column, the declared columns are the whole answer, and the result has no remainder from the selection.

### T-20. A selection materializes as the frame effects

> Evaluating a selection $(C, N, u, \rho)$ against `TypedDataFrame[S, R]` is: every name of $N$ must be a declared column of $S$; on the schema, keep the columns $C$ picks, read $u$ at each data type, the columns of $N$ at their declared data types, and apply $\rho$ to each name. On the remainder, `Only[R, C]` (`E-3`), then `Retyped` (`E-6`) as $u$ requires, then `Suffixed` or `Prefixed` (`E-5a`, `E-5b`) as $\rho$ requires. This is exact when $u$ gives each column back, with no `Retyped`, or gives one fixed data type $U$, with `Retyped` to $U$. For any other $u$ no single `Retyped` spells the result: the names still follow the effects, and the data types are `Unchecked`.

**Tag: proved.** By reduction to the frame effects. Proved in Lean: `Spec.T20.selection_as_effects`, `Spec.T20.selection_approx` and `Spec.T20.selection_beside_approx` in `spec/lean/Spec/L7/Selection.lean`. At one frame the columns of $N$ have fixed data types, so $u$ is one function of the matched column's data type, and the theorem holds for it as for any function.

A selection keeps each column $C$ picks, reads $u$ at its data type and applies $\rho$ to its name. When $u$ is the identity or a constant, that is the effect chain column for column. For any other $u$, each output column has a name the chain produces, so `T-2` applied to the chain gives the absence rule, and the type set is every data type. Closure is `T-2`, admissibility is `T-3`, and every containment question about $C$ is `T-9`.

The remainder holds columns of the kinds its record permits (`D-16`), and the checker reads $u$ there **kind by kind**, the same theorem applied to each part `Only[R, OfType[k]]`. For each kind $k$ that $C$ can pick in the remainder, the evaluator gives the matched column each data type of the sample of $k$ (`D-44`). Where every data type comes back unchanged, the columns of $k$ keep their data type. Where the answers share one data type, the columns of $k$ are retyped to it. Where they share less, they are retyped to their generalization (`T-26`), `Unchecked` at worst. Where every data type of $k$ refuses, the selection has no result type at a column the remainder can hold, and the call is an `unsupported-dtype-op` error, because the record does not exclude such a column. Where some data types of $k$ refuse and others answer, the result depends on the caller: polars raises for a caller whose remainder holds a refused data type, and runs for the others. The call is unproven, a result the remainder decides (section 18.5), as for a fixed name. The evaluator tries only the data types of the sample of $k$ that the record allows, so a record that excludes the refused ones proves the call. A remainder kind is not an open column of the frame: the some-member gate of `D-41` reads a column whose type the user left open, and a remainder kind stands for every column that each caller can bring. Measured by `TestRemainderKindSplit` in `spec/tests/test_expression_facts.py`: a zoned value raises beside a naive `Datetime` column and runs beside a zoned one, in a fill and in a comparison. A kind whose sample is empty, such as `Extension`, gives `Unchecked`. The effect writes the most common answer once, and every other answer beside it as a `Retyped` of its kinds. So `cs.numeric().n_unique()` retypes every numeric column of the remainder to `UInt32`, and `cs.numeric().cum_sum()` keeps the other numeric columns, retypes the `Int8`, `UInt8`, `Int16` and `UInt16` columns to `Int64`, and retypes the `Decimal` columns to `Decimal` with open parameters.

Two consequences, both inherited rather than decreed. Whether $u$ has an answer at every matched column is a type-set question, decided against the record by `T-12` and never by optimism. And the admissibility line falls exactly where `T-3` put it: `.name.suffix` on a selection is `E-5a` and passes. `.name.to_lowercase()` on a selection is the inadmissible renaming of `T-3` and is an error, even though the same rewrite on a single column expression passes, because there the checker computes the one resulting string.

> Measured (`M-13`) by `TestNameNamespace` in `spec/tests/test_expression_facts.py`: the `name` namespace holds exactly ten members, and they split along this line. The line's reason is measured by `TestNameRewriteAdmissibility` in the same file: an affix keeps distinct names distinct, where a whole-name function collides two known names into one, so no single result type can exist over the remainder. A named output colliding with a selection's outputs is the same duplicate error, so a checker over a remainder proves the name absent from what the selector picks, the provable-absence question the record already decides.

In the aggregation context the code runs in `agg` at each matched column, so the context rule (`T-15`) applies once per matched column: an elementwise selection implodes each matched column's own data type into a `List`, and a reducer applied to a selection stays unwrapped per column.

> Measured by `TestSelectionRoots` in `spec/tests/test_expression_facts.py`: `pl.all()` and `cs.numeric()` in an `agg` give one `List` per matched column, and `cs.numeric().sum()` gives each column's own data type unwrapped.

`.struct.unnest()` is multi-output without a selector root: its outputs are the fields of the struct its input gives, one column per field, each under the field's own name and data type. The input is any expression: a column read, a field of one, or a struct that `pl.struct` builds. The field names are the one output name that a data type decides: a struct type carries its field names, so the outputs are known exactly when the struct's data type is. A column declared `Unchecked` has no field list, which `D-27` refuses rather than tracks. In a `select` the fields spread into columns. In a `with_columns` they overlay beside the struct. An `.alias(...)` over the several outputs is a duplicate-name error. `.struct.field("x", "y")`, a list of names, or `"*"`, is the same unnest over the named fields, or over every field for `"*"`: the outputs keep the struct's order of its fields, and a name the struct does not hold errs, as polars does.

> Measured by `TestStructUnnest` in `spec/tests/test_expression_facts.py`: the spread, the overlay and the alias error.

The fit of `D-40` applies to a selection through one bridge. A selection argument fits a declared selection when its selector picks no column the declared selector rejects, decided by the containment of `T-9`, when its code answers the declared outcome at every cell of the declared rows, which partition the matched column's own data type, and at each data type of the sample of a cell's kind, as for an expression, and when the affixes and the characters are equal. `cs.numeric().sum()` is an aggregating selection and is declared `typed_agg_selection`. A code that reads a declared selection, as `texts.str.len_chars()` for a parameter `texts`, is a selection too: its selector is the declared one, and at each column that it picks, the leaf is the declared map there, with the declared type and the rename (section 17.1). A frame operation expands it as it expands a selector of polars, and gives the remainder the effect of `T-20` from the answers at each kind.

______________________________________________________________________

## 17. Materialization and declarations

### 17.1 The procedure

For a frame `TypedDataFrame[S, R]` evaluating expression $e$ in context $k$, extending the procedure of section 11:

1. **Resolve the reads.** Every name in the reads (`D-31`) must be a declared column of `S`. The remainder never justifies one, exactly as in section 0.3 of `OPERATION_CATALOG.md`. A selection is the exception: it reads by selector, not by name, and its remainder columns come through the effect of `T-20`, never through a read.
2. **Evaluate the code** (`D-36`) at the declared data types of the read columns, in context $k$. An open column is read at its sample (`D-44`), a declared leaf gives the outcome its declaration states at those data types (`D-39`), and a union argument is read member by member (`D-42`). An undefined outcome is an error at this site, polars' own. A declaration states no outcome outside its region, so a choice of the data types where a declared leaf's column lies outside it has no outcome: a frame whose column the declaration does not allow is an error here, and an open column drops the choices outside it, as it drops the choices polars refuses. An argument that fits a declaration can give any type, or none, outside the declared region: `pl.col("price").cast(pl.Float32)` fits `column("price", pl.Float32)` and gives `Float32` at an `Int64` column, where the declared `same` would give `Int64`.
3. **Assemble the frame.** A keyword argument replaces its expression's output name outermost, an inner `.alias(...)` included. `select` builds the result schema from the outputs and empties the remainder, except that a selection contributes its `T-20` effect on `R`. Two outputs under one name are an error. `with_columns` overlays outputs onto `S` and keeps `R` under the rules its catalog rows state. `agg` starts with the group keys and evaluates every argument in the aggregation context.

Only these three steps read an expression, and they read only its code, which is its type (`D-33`).

Everything is checked as soon as it is determined, and nothing later than the frame. A declaration's reads, output name and character are judged at the call that passes an argument to it, and so are its declared cells (`D-40`). The one question only a frame can answer, the data types of the frame's own columns, is settled here, at the operation where polars itself would fail.

### 17.2 Consumer sites

Every expression argument at every site materializes by section 17.1. The sites differ only in how they consume the output:

| Site | Uses the output |
| -- | -- |
| `select`, `with_columns`, `agg` | as the result's columns (step 3) |
| `filter`, `remove` | demands a `Boolean` answer and keeps the schema |
| `sort`, `sort_by`, `top_k` keys | discards it: the reads must resolve and the code must have an answer, and nothing lands |
| `group_by(expression)` keys | as key columns, under the output names |
| `join_where` predicates | demands a `Boolean` answer over the two frames' united columns |
| `map_columns` | takes a callable, not an expression: the fallback of section 18.3 |

A `filter` whose predicate's answer is not `Boolean` is an error at the site, polars' own schema-time behavior, and an aggregating predicate broadcasts, so the predicate's character is unconstrained.

> Measured by `TestConsumerSites` in `spec/tests/test_expression_facts.py`: the non-Boolean predicate raises at schema time, the aggregating predicate runs, a sort key runs at every data type, and a `group_by` expression key lands in the schema under its output name.

A frame method that polars runs as another operation over one expression is that operation: `group_by(...).mean()` is `agg(pl.all().mean())`, `fill_nan(v)` is `with_columns(cs.float().fill_nan(v))`, and `fill_null(v)` is `with_columns` over the columns that the value fills, or `select(pl.all().fill_null(v))` for a value that fills every column. The checker records the expression and evaluates it at that operation, so the method has no rule of its own.

### D-45. The stored type of a lazy frame

A lazy frame runs no step until `collect()`. Polars plans each step at the type that the frame **stores** for each column, which is the planner type of the expression that made the column (`D-36`), and runs the step on the data, which has the run type. Where the two differ, a column of a lazy frame holds both: `Planned[R, P]`, with the run type $R$ and the stored type $P$. The checker writes the marker only where $R \ne P$, and only at a lazy frame. An eager frame runs each step, so its columns hold their run types.

- **An expression** reads such a column as a leaf whose planner type is $P$ and whose data has the type $R$, as a parent reads an inner node with those two types (`D-35`). Its output takes the run type and the stored type that the evaluation gives in the lazy context. A declared map, as a declared selection at one column that it picks, plans at the type that it promises at the stored types of the columns that it reads, so `typed_selection(cs.numeric(), gives=same)` keeps `Planned[R, P]`.
- **A frame operation** that checks a data type plans at $P$. A join plans its keys at $P$ and checks them again at $R$ when it runs, so a key pair must agree at both. A reducer of the whole frame, such as `max()`, maps both types, and the result holds both where they still differ.
- **`collect()`** gives $R$, and `collect_schema()` gives $P$.

So `lf.with_columns(d=pl.col("i8") * pl.col("dec"))` over `i8 : Int8` and `dec : Decimal(10, 2)` gives `d : Planned[Decimal(38, 2), Decimal(10, 2)]`, and a later `pl.col("i8").fill_null(pl.col("d"))` casts `i8` to the stored type of `d` and gives `Decimal(10, 2)`, where the eager frame gives `Decimal(38, 2)`.

> Measured by `TestStoredType` in `spec/tests/test_expression_facts.py`: the stored type and the run type of the column, the later step at the stored type, and the join that checks its keys at both. `test_a_leaf_with_a_stored_type_gives_the_answer_of_polars` in `spec/tests/test_rules_tree_evaluation.py` evaluates chains of two steps with the leaf of `D-35`.

### D-39. Slice declarations

Inference computes every type above. A user writes one only on a function boundary, as a **slice declaration**: a named value, built by plain calls and assigned to a name that then appears in signatures. The name is the only spelling: the checker refuses a call expression in annotation position before any hook runs (measured by `TestAnnotationSurface` in `spec/tests/test_mypy_binder.py`). A declaration answers four questions: which columns are read, which cells are covered there, what outcome each covered area has, and how the expression counts rows.

```
declaration ::= column(name, region)
             |  typed_expr(rows)  |  typed_agg(rows)
             |  typed_lit(gives=dtype, named=name?)
             |  typed_selection(selector, rows,
                                suffix=s? | prefix=p?)
             |  typed_agg_selection(selector, rows)
             |  InferredExpr                    return position only

rows        ::= reads..., gives=outcome, named=name?  one area
             |  row(...), row(...), ...               several areas

reads       ::= column(name, region)    one column, by name
             |  column_of(Schema, name) one column, at the schema's
                                        declared type
             |  subset_of(Schema, name, ...)
                                        several, at their declared
                                        types
             |  Schema                  every declared column
             |  some_column(region)     one column, its name a blank

region      ::= dtype | family | {dtype, ...}
             |  EveryDType              every type, checked at each
             |  Unchecked               every type, checked at none
outcome     ::= dtype | same | same(name) | wrap(K, outcome)
name        ::= a plain string
```

Reads are positional. Only the surface's own keywords, `gives=` and `named=`, are keywords, so a column named `gives` collides with nothing and the surface can grow without colliding with anyone's data. A declaration is a **declared map**: for each read a region of kinds, for each **cell**, one kind per read, an outcome, and the output name and the character. Each production maps onto one piece of it:

| Production | Means |
| -- | -- |
| `column(n, r)` alone | reads $\{n\}$ over $r$, the outcome `same`, name $n$, elementwise |
| `typed_expr` / `typed_agg` / `typed_lit` | the character, nothing else |
| `column(n, r)` as a read | one read, restricted to the region's kinds |
| `column_of(Schema, n)`, `subset_of(Schema, ...)` | reads at the schema's declared data types |
| a schema class | one read per declared column, each a single data type |
| `some_column(r)` | one read under a name blank the call fills |
| `region` | a set of kinds: the vocabulary of `D-3` plus its two ends (`D-41`). A bare constructor, `pl.List`, names its kind, like the frame side's `none_of={pl.List}` |
| `gives=t` | the outcome $t$ at every declared cell |
| `gives=same` | that column's own data type back |
| `gives=wrap(K, o)` | the constructor $K$ around the outcome $o$ |
| `named=n` | the output name |
| `row(...)` | one more area of the same map |
| `typed_selection`, `typed_agg_selection` | the type of `D-38`: the rows partition the matched column's own data type, one implicit read, and the second form is the aggregating character |

The outcome `same` over an `Unchecked` read evaluates by the lifting: the read pins no data type, so `same` there gives `Unchecked` back, and a `wrap` there wraps every data type at once, which collapses to `List(Unchecked)`, the set of every list. `column(n, Unchecked)` therefore declares exactly what it reads as: the column given back, with its data type still `Unchecked`.

Defaults: `named=` defaults to the sole read's name, blank included. With two or more reads it is required, so no declaration inherits polars' leftmost-root convention silently. `typed_lit` is named `literal` unless `named=` says otherwise, matching `pl.lit` itself. `column(n, r)` alone is the one all-defaults form.

Well-formedness, checked at the declaration itself:

1. `typed_expr` reads at least one column. `typed_agg` with no reads is the rootless aggregation, the shape of `pl.len()`: it counts rows instead of reading columns, so `named=` is required and `gives=` is a data type. A declaration reading none and counting no rows is a literal and is written `typed_lit`.
2. Every `row(...)` of one declaration reads the same column names: rows partition the data-type space of one map, never the column set. Regions can differ per row. The names cannot.
3. Rows are disjoint: two rows claiming one cell is an error at the declaration, not later at some call.
4. `same` with no argument needs exactly one read. `same(n)` names one of the reads.
5. At most one `some_column` per declaration. The blank is a binder: an argument fits it when it reads exactly one column and gives it back under that name, the call binds the name, and a blank never reaches a frame, because `D-27` refuses a nameless column. A signature using one declaration for its parameter and its return shares the blank, so the output name is the input name, per call.
6. A region is never empty, and no column is read twice across the positional reads.
7. The rename of `typed_selection` takes only the two admissible affixes, `suffix=` or `prefix=` (`T-3`). Its selector is written with the ordinary `cs.*` constructors.
8. A blank read stands alone: no named read sits beside it, and no explicit `named=` renames it, because the binding gives the blank's own name back.

With several reads, a row's cells are the product of its reads' regions, one kind per read, and the disjointness of rule 3 is over these tuples.

The declaration surface is syntax, not a second sort of type: what a declaration builds is a declared map (`D-33`), and the checker shows and compares the expanded bracket forms, `TypedExpr[Reads[...], Column[Literal[...], ...]]` and its kin. A schema reference, `column_of(Trades, "price")`, is type-level only and never reaches polars. It is a call and not an attribute because a `Columns` class is a `TypedDict`, which carries no class attribute at run time or under the checker. Plain-Python spellings of the same slices stay legal and mean nothing more: an `@overload` per row, or a bounded type variable where an identity outcome must hold across a region.

`InferredExpr` is a return annotation and nothing else. It asks the checker to compute the function's return type from the function's own body: every candidate return value is wrapped in `infer(...)`, an identity at run time, and the return type is the union of what the wrapped values infer to. It carries no contract, because it follows the body. No parameter counterpart exists, deliberately: a parameter constrains what callers pass, and there is nothing in the callee to infer that from, so a parameter is the plain `pl.Expr` or a slice declaration:

| Parameter | Contract | Body checked against |
| -- | -- | -- |
| `pl.Expr` | none: the argument's type is forgotten | the plain type |
| a slice of regions | one answer per region | every cell of every row |
| a slice of single cells | exact | the declared cells |

### D-40. Fit

An argument **fits** a declared slice when:

- its reads are within the declared reads: the declaration promises the context, and reading less than the promise still evaluates,
- its output name is the declared one, a blank binding it by rule 5 of `D-39`,
- its character is the declared one, read from its answers in the two contexts (`D-29`), with no substitution between characters. `T-15` gives every pair of characters different types in some context, so a constant rule is declared `typed_lit`, and a boundary meaning to accept either character offers one declaration per character,
- at every declared cell, the evaluation of its code at a frame of that cell's data types (`D-36`) answers exactly the declared outcome, and polars plans it at the data type that it runs.

The last clause is the one that reads the code. The checker evaluates the argument at each declared cell, and a declared read that the argument does not make contributes nothing. A cell holds kinds, and a kind with open parameters stands for each data type of its sample (`D-44`), so the clause holds at each of those data types: the argument's answer there is compared with the declared outcome there, never the generalization of its answers with the generalization of the outcomes. `pl.col("t").dt.cast_time_unit("ms")` and `same` both generalize to `Datetime` with its parameters open over the sample of `Datetime`, and the cast still gives another unit than `same` at every member of the sample but one, so it does not fit `column("t", pl.Datetime)`. A region is a set of kinds (`D-39`), so `column("t", pl.Datetime("ms"))` declares every `Datetime`, and a fixed outcome keeps its parameters: `gives=pl.Decimal(38, 2)` is that one data type at each cell, not `same`.

> Measured by `TestFitAtEachDataType` in `spec/tests/test_expression_facts.py`: over the units of `Datetime`, the unit cast agrees with the column's own type at `ms` only, and a cast to `Float32` gives `Float32` at an `Int64` column. There is no subtyping between data types (`D-2`), so an argument answering `Int32` does not fit a declared `Int64`: evaluation inserts no coercion that would justify it. `Unchecked` is the one exception, and it is symmetric: a declared `Unchecked` covers every cell, so no answer can disagree with it, and an `Unchecked` answer fits any declared outcome, the trust having been signed where the marker was introduced (`D-41`, `T-22`). The relation is not transitive: `Int32` matches `Unchecked` and `Unchecked` matches `Int64`, and `Int32` still does not fit `Int64`.

A literal whose value decides its data type, as `pl.lit(1)`, `pl.lit(1.5)` or a plain number, joins a column by the range of its value (`D-37`): beside a `UInt8` column, `pl.lit(1)` gives `UInt8` and `pl.lit(300)` gives `UInt16`, where a literal of `Int32` gives `Int32`. So it fits no `typed_lit(gives=t)` of a fixed $t$, and the checker asks for `pl.lit(value, dtype=t)`. No declaration states such a literal: beside a column, a literal of unknown value gives `Unchecked` wherever its value decides the type, so the form would state little that `Unchecked` does not.

A body that reads a parameter reads one data type in every pass, so an argument whose planner type and run type differ fits no declaration: `pl.col("price").cum_sum()` over a `Decimal(10, 2)` column is planned as `Decimal(10, 2)` and runs as `Decimal(38, 2)`, and the checker refuses it and names the cast that makes the two one, `.cast(pl.Decimal(38, 2))`. A column of a frame holds one data type in every pass, so a column read always passes this clause.

Fit is a computed relation, never the host checker's own subtype order. Expression types declare no hierarchy: identity is the map (`D-33`), and the clauses above read names, answers and characters, which no variance of type arguments can express. The checker decides fit at each call of a function or a method (a static method and a class method included, and a method under any other decorator excluded, because the decorator can change the signature), from the argument's type, through the same per-call signature computation the frame side's requirement checks use. Ordinary subclassing supplies exactly one rung: every typed expression is a `pl.Expr`, so a plain `pl.Expr` parameter accepts every expression, and a plain `pl.Expr` argument fits no declared slice, because neither its name nor its answers are known.

> Measured by `TestFitIsNotSubtyping` in `spec/tests/test_mypy_binder.py`: with no fit computation, the checker refuses an identity expression against its own declared cell and still accepts the plain-parameter call. The positive half is measured by `TestHooksHostTheDesign` in the same file: a signature hook computing fit admits that same call.

The name, the character and the reads are judged at the call, and so are the declared cells: an argument's code is a complete object, so a cast's disagreement with a declared `Boolean` does not get to hide until a frame arrives. What no call knows is which frame the result will meet, so the frame's own data types are the one obligation that travels (section 17.1, step 2).

### T-23. Fit is sound

> Wherever a declared slice materializes, every argument that fits it materializes too, under the same name, and at each declared cell to the same data type, or to two data types of which one holds the other: `Unchecked` holds every data type, and in the aggregation context `List(Unchecked)` holds every list. Where neither answer is `Unchecked`, the two frame types are the same.

**Tag: proved.** Proved in Lean: `Spec.T23.fit_sound`, `Spec.T23.fit_sound_exact` and `Spec.T23.unchecked_counterexample` in `spec/lean/Spec/L7/Fit.lean`. Measured by exhaustion on the executable model, by `TestFitIsSound` in `spec/tests/test_fit_model.py`: over a battery of declarations, fitting arguments, every frame inside each slice and both contexts, the argument and the declaration materialize alike, and a refused argument really differs.

Step 1 of section 17.1 only shrinks: a subset of the declared reads resolves wherever the declared reads resolve. Step 2 evaluates at a declared cell, where fitting makes the answers agree up to `Unchecked`. The context rule followed by the collapse is monotone: it sends `Unchecked` to `Unchecked` in the row context, and to `List(Unchecked)`, which holds the declared answer's list, in the aggregation context. Step 3 reads only the outputs of the earlier steps. The clause of `D-40` that lets an `Unchecked` answer fit any declared outcome is why the result holds only up to `Unchecked`: an argument answering `Unchecked` fits a declared `Int64`, and the two materialize differently. At a call, the caller gets the declared type. Where the argument answers `Unchecked`, that declared type is trust, signed where the marker entered the program (`D-41`).

The fit evaluates a declared cell at the sample of its kinds (`D-44`), so the theorem holds at the data types of that sample.

______________________________________________________________________

### D-43. Inferred frame returns

`InferredDataFrame` and `InferredLazyFrame` do for a frame what `InferredExpr` does for an expression (`D-39`): the type of a call is the union of the types of the values that the body wraps in `infer(...)`, and one value of the plain polars class makes the type that class. A frame return type can instead keep its declared columns and leave only its remainder to the body: `TypedDataFrame[S, InferredRemainder]`, and the same for a lazy frame.

Then $S$ is a contract on each value $v$, checked at its `infer(...)`, by the clauses that `D-40` applies to the declared columns of a closed argument. A column of $S$ that $v$ does not declare is an error when $v$ is closed or its remainder proves the name absent, and unproven when its remainder can hold the name. A column of $S$ that $v$ declares with a data type that does not agree is an error, `Unchecked` agreeing with every data type. Each value keeps its other declared columns: nothing moves into the remainder, because a remainder expression cannot hold a named column (`D-18`). The values stay apart as members of the union, so their remainders are never joined.

At a call, each value is written in the terms of the caller by the substitution of `T-6`, the same evaluation that a declared open return type gets. Where every open frame argument is open, each remainder variable of the function becomes the caller's remainder. Where every one is closed, the value denotes a closed frame (`D-19`) that names the caller's own columns. Where they mix, the value splits into the two halves of `T-24`. This holds for `InferredDataFrame` and `InferredLazyFrame` too, whose values are as much written over the function's remainder variables.

`InferredExpr`, `InferredDataFrame`, `InferredLazyFrame` and `InferredRemainder` belong to the function's own return type and nowhere else. A parameter type that holds one, at any depth, and a return type that holds one below its top, are errors at each call of the function. A frame whose remainder is `InferredRemainder` is an error at each method call on it.

## 18. `Unchecked`

### D-41. Data-type sets, the lifting, and the two ends

Every data-type position in this system is read as a **non-empty set** of data types. Writing a concrete data type is writing its singleton, `Unchecked` is the whole set $T$, a family is every data type of its kinds, and a kind with open parameters, a bare `pl.Datetime`, is every data type of that kind. This is the same marker `D-27` defines for the columns of a frame, read inside the order `D-15` already reasons in. Two rules give the reading its arithmetic:

- **Lifting.** A tree whose columns hold sets is evaluated at every choice of one member of each set, and the defined results form the result set. The evaluation errs only when that set is empty: when every choice fails. A gate over data types passes when it passes for some member.
- **Collapse.** A minted type shows a singleton result set as its member, and a wider set as its generalization (`T-26`): `List(Unchecked)` for a set of lists, a kind with open parameters for a set of one kind, and `Unchecked` for a set of several kinds. Intermediate sets live only inside one check. The collapse keeps every member, and it is monotone. Proved in Lean: `Spec.subset_collapse` and `Spec.collapse_mono` in `spec/lean/Spec/L7/Fit.lean`, and the generalization by `T-26`.

The lifted set holds every state the tree gives at the members. Proved in Lean: `Spec.Node.D41.lift_sound` in `spec/lean/Spec/L6/Node.lean`. The evaluator computes the lifting over the sample of each open position (`D-44`), not over every member, and that sample is the assumption the computed lifting rests on.

Every `Unchecked` consequence follows from these two rules. No operation decides its own:

| On `x : Unchecked` | Result set | Minted |
| -- | -- | -- |
| `x.str.len_chars()` | $\{\mathtt{UInt32}\}$, defined only on `String` | `UInt32` |
| `x.cast(pl.Int64)` | $\{\mathtt{Int64}\}$, a fixed target | `Int64` |
| `x + 1` | one data type per numeric input | `Unchecked` |
| `x.implode()` | one list per member | `List(Unchecked)` |
| `pl.coalesce(pl.col("l"), x)`, `l : List(Int16)` | lists only, because every other member refuses | `List(Unchecked)` |

The first row is the point: an `Unchecked` input does not condemn an output the operation determines by itself. The third row derives the propagation `D-27` states, instead of decreeing it. And the last row is the derivation outrunning any decree: a `List` beside a column of another kind refuses, so the lifting keeps the lists alone, where a hand-written "the marker absorbs a join" rule would have lost the constructor.

A declared **region** (`D-39`) is the same vocabulary read the other way: a set of kinds a body is checked at, cell by cell, the data-type sets of `D-3` plus the two ends. `EveryDType` is the whole set as a region, every cell checked. `Unchecked` is the whole set under the lifting, no cell checked. And only declarations hold regions: a schema class column is one concrete data type, `Unchecked` or a type variable of a generic schema class (`D-6`), never `EveryDType`, because a region there would promise a per-cell check no single column can host. The quantifiers point in opposite directions: a region's gate passes when every member passes, and the lifting's gate passes when some member does. No fallback ever mints `EveryDType` (section 18.3).

### T-21. Containment

> Replacing one column's data type by `Unchecked` changes nothing but data types computed from that column, and the output names of `.struct.unnest()` over a struct computed from that column.

**Tag: proved.** Proved in Lean: `Spec.Node.T21.containment` in `spec/lean/Spec/L6/Node.lean`, and the exception by `Spec.T21.unnest_counterexample` in `spec/lean/Spec/L7/Context.lean`.

The evaluator reads a column only where the tree reads its name, so a column the tree does not read changes no state. Output names are computed from the tree and never from a data type (`D-32`), except the field names of `.struct.unnest()`, which are read from the data type of the struct its input gives. The record (`D-16`) never consults a declared column's data type.

### T-22. Losing precision never creates an error

> A program that checks with a concrete data type at some position checks with `Unchecked` at that position, when every gate that reads that position passes on some member, as the lifting of `D-41` reads a gate, and no unnest reads that position.

**Tag: proved.** Proved in Lean: `Spec.Node.T22.monotone` and `Spec.Node.T22.no_new_error` in `spec/lean/Spec/L6/Node.lean`.

Enlarging a set enlarges every result set the lifting computes, member by member, and the collapse is monotone too. An error is then the emptiness of such a set, and that is monotone: a tree with an answer at some choice keeps it once a column holds more members. A gate that demands every member, as a declared region does, is not monotone: it passes on $\{\mathtt{Int64}\}$ and fails on the set of every data type. So a gate on an `Unchecked` position is always read as a some-member gate. An unnest is the other exception: its output names are the fields of the struct it reads, `Unchecked` gives no field names, and `D-27` refuses an unknown name (`T-21`). So `.unnest()` and `.struct.unnest()` over an `Unchecked` column are refused.

The converse direction is the purpose: errors disappear under `Unchecked`, which is what an escape hatch is. This is what reconciles the marker with a system that rejects the unproven. A program with no `Unchecked` is checked with no optimism anywhere, and each `Unchecked` weakens exactly the checks its column reaches (`T-21`), never the rest.

### 18.3 Introduction is explicit, or flagged

`Unchecked` enters a program at two sorts of place: an annotation the user writes, or a fallback substituting it under a known name where the output data type cannot be known, per the standing fallback discipline. `map_elements` without `return_dtype=` is the canonical fallback: two lambdas with one signature produce two schemas (`M-14`, measured by `TestOpaquePython` in `spec/tests/test_expression_facts.py`), so the data type is decided by code no checker reads (`B-6`), and polars' own `Unknown` data type is the runtime shadow of the same gap. `map_columns(names, f)` on the frame side is the same fallback: it applies arbitrary Python to whole columns, so each named column is retyped to `Unchecked`, per its catalog row. Under `allow_unproven = "never"` every fallback introduction is an error naming the operation and the fix, so at that level every `Unchecked` in the program is one the user signed.

The two introduction paths do not make two meanings. A written `Unchecked` and a fallback's `Unchecked` are the same set under the same lifting, the same to every later operation. The setting above is what reduces the second path to the first. A pessimistic second marker, admitting only operations defined on every data type, is deliberately absent: it would refuse nearly everything, so a fallback minting it would no longer keep the column usable. The universally-checked reading exists on the declared side only, as `EveryDType`, the widest declared region, and no fallback mints it.

### 18.4 `Unchecked` is not a region

`Unchecked` stands in region position in the grammar of `D-39`, and it is not a region: it declares that no region is checked. The two tools relax data-type precision in opposite directions:

| Aspect | `Unchecked` | A declared region |
| -- | -- | -- |
| Meaning | this one value's data type is not checked | for every member, checked |
| Gates | pass when some data type passes | pass only when the whole region passes |
| Output | recomputed by the lifting, often `Unchecked` | the declared outcome, exactly |
| Relates positions | no: two `Unchecked`s share nothing | yes: `same` gives the read column's own data type back |
| Honest reading | trust me | works for all of them |

A reusable function wants the region: its body is checked at every cell, and frames get exact answers back. A bounded type variable declares the same slice where an identity outcome must hold across it, and means nothing different. `Unchecked` in a signature is not a spelling of generic: it is an abdication the lifting then propagates. Both are legitimate. Only one is a contract.

### 18.5 What the lenience setting relaxes, and what it never does

The checker rejects what it cannot prove. The `allow_unproven` setting lets a user accept some unproven results, and it can do so without changing what the rest of this section promises.

A result is unproven for one reason, and each reason has a level. The levels are cumulative: each one passes everything the level before it passes.

| Level | Passes |
| -- | -- |
| `never` | nothing, and a case the checker does not model is reported too |
| `none` | nothing that the program states. A case the checker does not model passes unless `report_checker_gaps` is set |
| `stated` | a construct that states the schema is dynamic: a column name that is not a literal string, a plain expression, a read with no schema, an opaque function that polars gives no type keyword for |
| `omitted` | an omission that a keyword closes: `return_dtype=`, `dtype=`, a bare `int` or `float` where a literal fixes the bucket, a data type not written as a literal |
| `possible` | a result the remainder decides: an undeclared name that the remainder can hold, a join or a concat whose result depends on the caller |
| `asserted` | a claim: a requirement that the caller's remainder cannot prove, and an argument with no expression type passed to a declaration |

A case the checker does not model is not on this scale. The program has nothing to fix there, so it passes at every level from `none` up, unless `report_checker_gaps` is set.

**Never relaxed.** A proof that the program fails is not a matter of lenience. These errors fire at every level: `unknown-column`, `duplicate-column`, `unsupported-dtype-op`, `wrong-column-shape`, `invalid-cast`, `join-key-dtype`, `schema-mismatch`, `invalid-call-shape`, `missing-column`, `wrong-dtype`, `forbidden-column` and `forbidden-dtype` against a concrete residual, `expression-fit` between two typed values, `selection-rename`, `blank-column`, `malformed-declaration`, and the errors for a type spelling the checker cannot read.

**What a pass gives.** Every level below `asserted` keeps one promise: a type the checker gives is true of every run in which the call succeeds. A passed result loses precision: a known name keeps its column under `Unchecked`, or the operation gives the plain frame. At `possible` the checker also accepts a call that fails for a caller whose remainder does not hold a name, and the type it gives stays true for every caller where the call succeeds. `T-22` is why losing precision never creates an error downstream.

`asserted` gives that promise up. The checker takes a declared type as true without a proof, so a caller that breaks the requirement gets a result whose type is wrong. It is the one level at which a clean check does not mean that every type is true.

______________________________________________________________________

## 19. Claim ledger

Every numbered claim, and how it is supported. A claim absent from this table, or present with no tag, is a defect.

| Id | Claim | Tag | Where |
| -- | -- | -- | -- |
| `D-1` … `D-8` | the objects | definition | section 1 |
| `D-9` | the absence language | definition | 2.1 |
| `D-10` | what an absence rule denotes | definition | 2.1 |
| `D-11` | name expressions are regular sets, carried as machines | definition | 2.1 |
| `D-12` | the selector language | definition | 2.2 |
| `D-13` | what a selector denotes | definition | 2.2 |
| `D-14` | what a selector settles from a record | definition | 2.2 |
| `D-15` | type sets | definition | 2.3 |
| `D-16` … `D-21` | the record, denotation, knowledge, the requirement predicate | definition | 3, 5 |
| `R-1` … `R-9` | the eight per-column requirements, and the empty one | definition | 5.1 |
| `R-10`, `R-11` | the two relational requirements | definition | 5.2 |
| `E-1` … `E-4`, `E-6` | five of the seven effects | definition | 6 |
| `E-5a`, `E-5b` | the two renaming effects | definition | 6 |
| `D-22` | canonical form | definition | 8.1 |
| `D-23` | translating a selector inward | definition | 8.1 |
| `D-24`, `D-25` | splits and their binding rule | definition | 9 |
| `D-26` | satisfaction | definition | 10 |
| `D-27` | `Unchecked`, for the columns of a frame | definition | 12.3 |
| `T-1` | the record over-approximates | **proved**, **measured** | 4, `Spec.T1.over_approximates` |
| `T-2` | closure, both halves | **proved**, **measured** | 7, `Spec.T2.closure` |
| `T-3` | admissibility of a new effect | **proved** | 7, `Spec.T3.admissible_closed` |
| `T-4` | one canonical form per expression | **proved**, **measured** | 8.1, `Spec.T4.canonical_form_unique` |
| `T-5` | the canonical form preserves meaning | **proved**, **measured** | 8.3, `Spec.T5.normalize_preserves_denotation` |
| `T-6` | substitution | **assumed** | 8.3 |
| `T-24` | denotation is additive in each variable | **measured** | 8.3 |
| `T-7` | every minted type is in canonical form | **assumed**, an obligation on an implementation | 8.3 |
| `T-8` | $\equiv$ implies $=$ | **proved**, **measured** | 8.3, `Spec.T8.equal_forms_equal_denotations` |
| `T-9` | containment between two selectors is decidable | **proved**, **measured** | 8.3, `Spec.T9.decidable` |
| `T-10` | $=$ implies $\equiv$ | **measured** | 8.4 |
| `T-11` | a split adds no expression node | **proved**, **measured** | 9, `Spec.T11.split_is_filter` |
| `T-12` | per-column requirements and `R-10` are decided from the record; `R-11` only for a remainder that permits no column | **proved**, **measured** | 10, `Spec.T12.decide_iff` |
| `T-13` | disjointness distributes over a split; containment does not | **proved**, **measured** | 10, `Spec.T13.disjoint_union_iff` |
| `T-14` | soundness | **proved**, as a corollary of `T-1` | 12, `Spec.T14.sound` |
| `M-1` | a filter through an affix is not a substitution | **measured** | 2.1 |
| `M-2` | an empty affix normalizes away | **measured** | 6.4 |
| `M-3` | the sort key must not read set ordering | **measured** | 8.2 |
| `M-4` | 11 of 32 caller shapes settled by a family, 21 unsettled | **measured** | 9.1 |
| `M-5` | no equality is left undecided over the measured space | **measured** | 8.4 |
| `D-28` … `D-34` | the expression objects: contexts, row forms, reads, names, the type as code, the tree | definition | 13 |
| `D-35` … `D-37` | the state of a node, the three passes and the outcome, the rules | definition | 14 |
| `D-38` | the type of a selection | definition | 16 |
| `D-39`, `D-40` | slice declarations, and fit | definition | 17 |
| `D-41` | data-type sets, the lifting, and the two ends: `Unchecked` for expressions | definition | 18 |
| `D-42` | branches and loops: a union of codes | definition | 15 |
| `D-43` | inferred frame returns, and `InferredRemainder` | definition | 17 |
| `D-44` | holes and open columns, read at their samples | definition | 15 |
| `D-45` | the stored type of a lazy frame | definition, **measured** | 17 |
| `T-15` | the context rule | **measured** | 13 |
| `T-18` | a loop whose unrolled versions agree gives their outcome | **assumed**, from `M-6` | 15 |
| `T-20` | a selection materializes as the effects; exactly for an identity or constant answer, up to `Unchecked` otherwise | **proved** | 16, `Spec.T20.selection_as_effects` |
| `T-21` | `Unchecked` touches only what reads it, and the names of `.struct.unnest()` | **proved** | 18, `Spec.Node.T21.containment` |
| `T-22` | losing precision never creates an error, where every gate is a some-member gate | **proved** | 18, `Spec.Node.T22.monotone` |
| `T-23` | fit is sound up to `Unchecked` | **proved**, **measured** | 17, `Spec.T23.fit_sound` |
| `T-25` | the evaluator gives the answer of polars | **proved** from `M-20` and `M-21` | 14, `Spec.Node.T25.evaluate_eq_answer` |
| `T-26` | each answer fits what the answers share | **proved** | 15, `Spec.Generalize.fits_generalize` |
| `M-6` | mypy's three loop-binding behaviors | **measured** | 15 |
| `M-12` | the size of the expression API | **measured** | `OPERATION_CATALOG.md` Part III |
| `M-13` | the `name` namespace splits on admissibility | **measured** | 16 |
| `M-14` | one signature, two schemas | **measured** | 18 |
| `M-20` | polars types a node from the states of its inputs | **measured** | 14 |
| `M-21` | each definition gives the answers of polars on its sample | **measured** | 14 |
| `B-1` … `B-6` | boundaries | **assumed**, from the library's behavior | 12.2 |
