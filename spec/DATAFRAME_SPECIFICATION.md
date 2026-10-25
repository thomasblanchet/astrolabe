# Typed frames with partial knowledge

This document answers what a typed frame is, and why, in prose with examples: how to describe a frame's schema in Python's type system when only part of the schema is known. Read it first: it assumes no other document. `OPERATIONS.md` gives the formal definitions, rules and proofs, and it is authoritative: where this document restates one of its definitions, it cites the identifier, and if the two disagree, `OPERATIONS.md` is right.

## 1. Abstract

A frame has columns, and each column has a name and a data type. When a program knows every column, a type checker can verify every operation on it: a misspelled name, a sum over text, a join that would collide, all become errors before the program runs.

Most real code does not know every column. A function written to be reused takes the columns it cares about and lets its caller bring anything else along. This specification describes how to type such a function: how to say "I need these columns, and you can bring others", how to state what those others must not contain, and how to describe what the function does to them.

The result is a small grammar with two halves. A **requirement** is something a caller must guarantee. An **effect** is something a function does. Everything else in this document follows from one rule about how they interact.

The computations that produce columns, polars *expressions*, have a type system of their own, described in `EXPRESSION_SPECIFICATION.md`. This document types the frames.

______________________________________________________________________

## 2. Motivation

### 2.1 The easy case

When a schema is fully known, typing it is straightforward. Write the columns down:

```python
class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
```

Now the checker knows everything. It can reject `trades.select("smbol")` because no such column exists. It can reject a sum over `symbol` because text does not sum. It can compute the exact type of `trades.drop("price")`.

### 2.2 The case that motivates this document

Reusable code does not work like that. Consider a helper every analytics codebase writes:

```python
def add_notional(df):
    return df.with_columns(notional=pl.col("price") * pl.col("quantity"))
```

This function needs exactly two columns. It must accept a frame with those two and **any** other columns whatsoever, and it must return that same frame with `notional` added. A trading desk passes a frame with fifty columns; a test passes one with three.

To type this, the signature must express three separate ideas:

1. **What it requires.** `price` and `quantity` must be present, with numeric data types.
2. **What it tolerates.** Any other columns can be present, and it does not know or care what they are.
3. **What it produces.** The same other columns, plus `notional`.

Point 2 is the interesting one. The other columns are not "no columns" and not "any type at all". They are a real but unnamed set that must be carried through the function and handed back to the caller intact. This document calls that set the **remainder**.

### 2.3 Why the remainder cannot simply be ignored

A tempting shortcut is to type the function as taking a frame with two columns and returning a frame with three, discarding whatever else the caller had. This is wrong in a way that gets worse the more the helper is used. The caller passes fifty columns and gets a type claiming three. Every subsequent operation on that result is checked against a lie: real columns look missing, and the checker rejects correct code.

The opposite shortcut, giving up and typing the result as "some frame", is also wrong. It is not unsound, but it is useless. Every operation downstream of one such helper loses all checking.

The remainder exists so that neither shortcut is necessary.

______________________________________________________________________

## 3. The model

### 3.1 A frame is known columns plus a remainder

Every typed frame has two parts:

- a **schema**: the columns whose names and data types are known, and
- an optional **remainder**: a stand-in for the columns that are not known.

A frame whose type has a remainder is an **open frame**. A frame whose type has none is a **closed frame**.

Written out, a frame type names both:

```python
Rest = other_columns()

def add_notional(
    df: TypedDataFrame[Trades, Rest],
) -> ...
```

`Trades` is the schema. `Rest` is the remainder. A frame with no declared columns is written `TypedDataFrame[Rest]`. `TypedDataFrame` alone names no columns, and polars runs a call on a frame of any columns, so no schema follows from it: the checker reads it as the plain frame `pl.DataFrame`, which it does not track. `TypedDataFrame[()]` is a frame of no column.

A schema class can also be generic in the data type of a column, `class Keyed[T: pl.DataType](Columns)`, so that one function accepts `Keyed[pl.Int64]` and `Keyed[pl.Float64]` alike (`D-6` of `OPERATIONS.md`).

The first line declares the remainder and gives it a name. `Rest` is not a column and not a type you could write out; it is a placeholder that each call fills in differently. For one caller it stands for the columns `venue` and `trader`, for another it stands for no column at all. Declaring it separately, rather than inline in the signature, is deliberate: a remainder usually carries a requirement, and a named requirement can be explained once and shared by every function that needs it.

The schema and the remainder never overlap. If a column is named in the schema, it is not in the remainder. This is a definition, not a coincidence: the remainder means "the columns *other than* the declared ones". It has a consequence used constantly below. **A declared column is automatically absent from the remainder, so no promise is ever needed about a name you have declared.**

### 3.2 The every-caller rule

This is the central idea of the specification, and everything else is derived from it.

> **An operation on a frame with a remainder is allowed if and only if it would succeed for every possible value of that remainder.**

Read it as a rule about *who is at fault*. A function that accepts an unknown remainder is making a promise to every possible caller. If there is any remainder that would make the function crash, the function is wrong. It is not the caller's job to avoid the bad case, because the function's signature invited them in.

This is what makes a reusable helper checkable on its own, without looking at a single call site.

Two consequences follow immediately, and both are worth stating separately because they are easy to get backwards.

**You cannot reference a column you did not declare.** Consider:

```python
Rest = other_columns()


def bad(df: TypedDataFrame[Trades, Rest]) -> ...:
    return df.select("venue")
```

Some caller's remainder contains `venue`; some caller's does not. The second caller crashes. So this is an error, even though it works for the first caller. The possibility that a column *can* be there is never grounds to use it. If a function needs `venue`, it must declare it.

**Giving up is not the same as being right.** If an operation would crash for some caller, the answer is an error, not a vague result type. A checker that responds to a difficult case by forgetting everything it knew has not been careful; it has been silent about a real bug. Vagueness is only correct when the answer is genuinely unknowable, never when the answer is known to be wrong.

### 3.3 What the rule does not cover

The rule quantifies over **sets of columns**. It says nothing about the data inside them.

Some operations fail for reasons no schema can predict. Joining on time order requires the data to be sorted. Stacking two frames side by side requires them to have the same number of rows. Converting text to a number fails on the row where the text is not a number.

These are outside the model (`B-1` … `B-3`). The specification neither proves nor rejects them. This is a boundary, not an exemption: nothing in the grammar below can express them, so nothing in it pretends to.

______________________________________________________________________

## 4. Requirements: what a caller must guarantee

A requirement is a promise the caller makes about the remainder. It is written where the remainder is introduced, and every caller must satisfy it.

The complete set of requirements is small, and each one exists because some real operation cannot be typed without it.

### 4.1 No requirement

The default. The remainder can be anything, including nothing.

```python
Rest = other_columns()

def add_notional(
    df: TypedDataFrame[Trades, Rest],
) -> ...
```

Most functions need nothing more than this. A function that only *adds* columns, or only *reads* declared ones, imposes no burden on its caller at all.

### 4.2 Absent names

Some operations fail if a particular name is already taken. Renaming a column to `ticker` fails if the frame already has a `ticker`. Adding a row-number column called `index` fails if `index` already exists.

The declared columns are visible, so a collision with one of those is caught without any help. The remainder is not visible, so the caller must promise:

```python
Free = other_columns(without=["ticker"])


def to_ticker(
    df: TypedDataFrame[Trades, Free],
) -> ...:
    return df.rename({"symbol": "ticker"})
```

`other_columns(without=["ticker"])` reads as "other columns, but not one called `ticker`". Several names are listed together:

```python
Free = other_columns(without=["ticker", "venue"])
```

The order does not matter and repeats do not matter, because this is a set of names.

Naming the requirement is worth a moment. `Free` is a value, so it can be declared once and used by every function that shares the requirement, and its name can say what it means. Requirements are written as calls throughout this specification for exactly that reason.

A caller who promises *more* than is required is always acceptable. A function demanding that `ticker` be absent is happy to receive a remainder that also excludes `venue`. Promising more is never a problem; promising less is.

**Every requirement has a positive twin.** `only=["a", "b"]` says the remainder holds nothing but those names, where `without=` says it holds neither of them:

```python
NoTicker = other_columns(without=["ticker"])  # no column called ticker
TickerOnly = other_columns(only=["ticker"])  # nothing but a ticker column
```

The positive form is rarely what a function wants on its own. It exists because section 4.6 divides a remainder into parts, and a part that *takes* a group of columns needs to say so. The same twin appears at every requirement below.

### 4.3 Absent namespaces

A list of names cannot say everything a caller needs to say. Some operations need a whole **set** of names to be free, not a fixed list of them.

The clearest case is a reference table. A function that brings in every column of a thirty-column table needs all thirty names free. Written as a list, that is thirty names in the signature, and the signature must change whenever the table changes.

A namespace requirement says it once:

```python
Room = other_columns(without_prefix="sec_")

def enrich(
    df: TypedDataFrame[Trades, Room],
) -> ...
```

`other_columns(without_prefix="sec_")` reads as "other columns, but none whose name begins with `sec_`". It admits a frame with `venue` and `trader`, and rejects one with `sec_country`.

Two properties make this more than shorthand.

**It describes a set no list can.** "Every name beginning with `sec_`" is infinite. No list of names is equal to it, so a specification with only lists cannot express this requirement at all.

**It is stable.** The promise does not name any column of the reference table, so that table can gain a column, lose one, or rename one, and no caller's signature changes. A list-based promise makes every such edit a breaking change for every caller.

The cost is on the other side, and it is real. The caller must arrange for the namespace to exist. Section 6.5 shows how, by renaming the reference table's columns before the operation that needs them.

Everything in section 4.2 still applies. Promising more is acceptable, so a remainder excluding both `sec_` and `ref_` satisfies a function demanding only `sec_`.

**A namespace can be at either end of a name.** `without_suffix="_prev"` promises that no name ends in `_prev`, exactly as `without_prefix` does for the start:

```python
Room = other_columns(without_suffix="_prev")
```

The two are the same requirement in mirror image, and section 6.7 needs the suffix form: an operation that renames by appending must know the names it will produce are free.

Both have the positive twin section 4.2 describes. `only_prefix="sec_"` says every name in the remainder begins with `sec_`, and `only_suffix="_prev"` says every name ends in `_prev`. Section 4.6 is what these are for.

### 4.4 Data-type families

Some operations work on some data types and not others. Summing every column in a frame fails if one of them is text. Over a known schema this is checkable. Over a remainder it is not, unless the caller says something about the data types involved.

Two requirements cover this, and they point in opposite directions:

```python
from astrolabe import NumericType, NestedType

Rest = other_columns(all_of=NumericType)  # every remainder column is numeric
Rest = other_columns(none_of=NestedType)  # no column is a list, array or struct
Rest = other_columns(all_of=pl.Datetime("us"))  # every column is a naive microsecond datetime
```

A data type with parameters names itself alone: `pl.Datetime("us")` is not `pl.Datetime("ms")`, and a set such as `{pl.Datetime("us"), pl.String}` names their union. Here, in a requirement, a bare `pl.Datetime` names the whole kind. In a column's own declaration, and wherever a call takes one data type, such as the target of a `cast` or a `schema=`, it names polars' own default, `Datetime("us", None)` (`D-3`). A constructor the checker cannot read exactly, for example `pl.List(pl.Int64)`, is refused rather than widened to its whole kind.

A note on where these names come from. Polars groups its data types into families, and a family is an ordinary class: `NumericType` is the parent of the integer, floating-point and decimal types, `NestedType` is the parent of lists, arrays and structs. Polars does not export these classes at the top level, so `pl.NumericType` does not exist. astrolabe re-exports them, and this document writes them as `NumericType` and `NestedType`. `pl.DataType`, the base of every class, names every kind in a requirement. A column declared with a family class, for example `anyd: pl.DataType` or a `FloatType` column, can hold any kind of that family, so a requirement admits the column only where it admits each of those kinds.

These two are the matched pair section 4.2 describes, at the level of data types rather than names. `all_of` is what lets an operation apply to the remainder at all. `none_of` is what lets an operation *skip* the remainder safely, and it turns out to be the one needed more often: a function that selects the numeric columns usually means "the numeric ones I declared", which is exactly a promise that the remainder contributes none.

A family is a group of related data types: all numbers, all integers, all nested types. Families nest, and a promise about a wider family does not settle a question about a narrower one. Promising that everything is numeric does not say whether anything is an integer, because a numeric column can be an integer or a decimal or a floating-point number.

**Both keywords take any set of data types, and a family is one way to name one.** A single data type is a set of one, and several written together mean their union:

```python
Rest = other_columns(none_of=NestedType)  # a family
Rest = other_columns(none_of=pl.String)  # one data type
Rest = other_columns(none_of={NestedType, pl.String})  # their union
```

The third form is not a convenience. Real operations have domains that are not families: casting a whole frame to text accepts `Struct` and rejects `List` and `Array`, so no family names its gate and only a set of two data types does. Without the set form such an operation would be rejected over a remainder for want of a way to write the promise, which would be a limit of the notation rather than of the system.

### 4.5 Requirements about two remainders together

The requirements above describe one remainder on its own. Some operations take two frames, and what matters is how their two remainders relate to each other.

A relational requirement is still written where a remainder is introduced, like every other requirement. It names the other remainder, which must already be declared:

```python
L = other_columns()
R = other_columns(disjoint_from=L)
```

`R = other_columns(disjoint_from=L)` says that `R` shares no column name with `L`. The fact is about the pair, and stating it on either one establishes it for both. Writing it on the second is the only ordering Python allows, since `L` must exist before it can be named.

Keywords combine, so one declaration carries as many requirements as it needs:

```python
R = other_columns(disjoint_from=L, without=["audit_id"])
```

A relational requirement behaves exactly like `without=`. The body of a function declaring it can assume it. Every call site is checked against it.

**Two relations exist, and they are opposite ends of the same axis.**

| Written | Says |
| -- | -- |
| `other_columns(disjoint_from=L)` | shares no name with `L` |
| `other_columns(within=L)` | every name is also a name of `L` |

Section 6 explains why these two, and no others. In short: an operation that renames on collision behaves determinately when nothing collides, and determinately when everything collides. Between those two it does not.

`disjoint_from=` names **one remainder, or a set of them**. One is the usual case, written bare as above. It has to accept several because section 4.6 lets a remainder be described in parts, and a set of parts has no single name to promise about:

```python
Right = other_columns(disjoint_from={LeftNumeric, LeftRest})
```

`within=` takes the same, and a set again means the union of the parts.

The two behave differently under that union, and the difference is worth knowing. "Shares no name with `A` and `B`" is the same as sharing no name with `A` and sharing no name with `B`, so a `disjoint_from=` promise can be checked one part at a time. "Every name is a name of `A` or of `B`" is **not** the same as being within `A` or within `B`, so a `within=` promise has to be checked against the whole union at once. Neither costs a user anything extra to write; it is the checker that does more work for the second. `T-13` proves both.

**Neither relation can divide a remainder into parts.** A relation is a fact about a pair of remainders, not a test that one column passes or fails, so it cannot decide where a column goes. Section 4.6 needs a per-column test, and only the requirements of sections 4.2, 4.3 and 4.4 are one.

Nothing about a relational requirement appears in a return type. A return type carries effects, and an effect never carries a requirement. That separation is what lets effects compose without a reader having to ask, at each level, which operands a buried requirement is talking about.

### 4.6 Splitting a remainder

A remainder is one undifferentiated set of columns. Often a caller knows more than that: some of the columns in the remainder are numbers, or all of the reference table's columns begin with `sec_`. A **split** records that structure.

```python
Numeric = other_columns(all_of=NumericType)
Rest = other_columns()


def totals(
    df: TypedDataFrame[Trades, SplitColumns[Numeric, Rest]],
) -> TypedDataFrame[TradeKey, Numeric]:
    return df.select("symbol", cs.numeric())
```

`SplitColumns[A, B]` declares that the columns outside the schema divide into `A` and `B`. It fills **one** remainder slot, so the rule of section 8.1 that a frame has at most one remainder is untouched, and the parts stay ordinary declarations, which is what lets a return type name whichever of them survives.

**A split is a binding form, so it belongs in a parameter.** `SplitColumns[...]` says how to divide the columns a caller brings, which is only a question where columns arrive. A return type names a part, or names the split again if it keeps both:

```python
def keep_numbers(
    df: TypedDataFrame[S, SplitColumns[Numeric, Rest]],
) -> TypedDataFrame[S, Numeric]: ...
def keep_both(
    df: TypedDataFrame[S, SplitColumns[Numeric, Rest]],
) -> TypedDataFrame[S, SplitColumns[Numeric, Rest]]: ...
```

**The rule is first match wins.** `A` takes every column its own requirement allows. `B` takes every column that is left and that its requirement allows. In closed form, with $P_i$ the predicate of part $i$, part $i$ holds exactly the columns satisfying

$$
P_i \land \lnot P_1 \land \dots \land \lnot P_{i-1}
$$

so the parts are disjoint by construction and nothing needs to promise it. There is no search: one pass decides every column.

**Each part after the first knows more than it was declared with.** `Rest` above was declared with no requirement, yet inside the body it provably holds no numeric column, because any numeric one would have gone to `Numeric`. That is the $\lnot P_1$ term, and it is why a split is worth declaring: it tells the checker something about *both* parts.

**Two calls must be rejected, and neither is obvious.**

A column that no part will take has nowhere to go. `SplitColumns[Namespaced, Numeric]`, where `Namespaced` takes only the `sec_` names, rejects a caller holding a plain `String` column called `venue`, and the message can name it.

A part that an earlier part subsumes can never hold anything. `SplitColumns[Rest, Numeric]`, with `Rest` unconstrained, leaves `Numeric` permanently empty. Nothing would fail, and the user would simply never get the answer they wrote the split for, so this is an error rather than a curiosity.

**A relational requirement cannot split.** Section 4.5 says why: `disjoint_from=` and `within=` are facts about a pair of remainders rather than tests a single column passes, so they cannot decide where a column goes. Only the requirements of sections 4.2, 4.3 and 4.4 can appear in a split, and that is what their positive twins are for.

**A split says something new, and costs nothing to describe.** Those are two different claims and it is worth keeping them apart.

What it *says* cannot be said otherwise. With a single remainder a signature has two choices, and neither is this one: `other_columns(all_of=NumericType)` demands that **every** column of the remainder be numeric, so a caller bringing a mix is turned away; `other_columns()` accepts that caller and then knows nothing about what arrived. A split is how a signature accepts the mixed caller *and* keeps track of which columns went where.

What it *costs* is nothing in the grammar. A split is a binding form, not an effect. Once bound, part $i$ is the ordinary filtered expression $\mathtt{Where}(G, P_i \land \lnot P_1 \land \dots \land \lnot P_{i-1})$ over the whole remainder $G$, so it introduces no new node, no new absence constructor, no new closure rule and no new canonical form. This is proved as `T-11` in `OPERATIONS.md`: each part's requirement is a selector, and the selector language is closed under `and` and `not`, so the expression a split produces is one the grammar already had.

**Every operation treats the parts one at a time.** Each effect is defined a column at a time, so it distributes across a partition, and the two renamings are injective, so the parts stay disjoint afterwards. That gives the whole behavior table in one line each:

| Operation | On a frame split into parts |
| -- | -- |
| row-only: `filter`, `sort`, `head` | identity on every part |
| `select` of a fixed name list | emptied, as for one remainder |
| `select`/`drop` by a **name** selector | applies to each part; every part survives, smaller |
| `select`/`drop` by a **type** selector | picks whole parts when the split divides on the data types the selector tests, and is then exact |
| `with_columns(z=...)` | `z` becomes declared, so it leaves every part |
| `cast(dtype)` | retypes every part |
| prefix or suffix rename | applies to every part; injective, so the partition survives |
| `group_by(k).agg(pl.all())` | keeps every part's names; the per-part list wrap is outside the effect language, so each part is retyped to `Unchecked` |
| `unpivot`, `pivot` with `index=` | emptied |
| `join` | each part is a separate collision source of section 6.4 |

Two rows repay a second look. A **type** selector picks whole parts rather than splitting them, which is the entire point: `select(cs.numeric())` on `SplitColumns[Numeric, Rest]` keeps `Numeric` and drops `Rest`, with nothing left uncertain.

And `cast` shows that **a part is its columns, not its name**. After `cast(pl.String)` the part called `Numeric` holds no numeric column, so a later `select(cs.numeric())` correctly returns nothing from it. The declared requirement described the columns when they arrived; it is not a standing promise about them.

**How finely to split is the caller's choice, and it is the whole tradeoff.** A coarse split keeps less: `SplitColumns[Numeric, Rest]` remembers that the non-numeric columns are non-numeric, and nothing more. A finer one keeps more. Section 5.7 shows what each granularity buys.

**Why the parts are written out rather than hidden behind one name.** A name standing for the pair would read better, and it does not work. A type variable is bound by the checker from the annotation as **written**, so a part that appears only after a checker plugin expands a name is never bound at all. Three spellings behave differently. This table restates section 9.4 of `OPERATIONS.md`, which tags it assumed:

| Written | Result |
| -- | -- |
| `SplitColumns[Numeric, Rest]` | both parts bind; a return naming one is exact |
| a plain alias for the pair | the part resolves to `Never`, silently |
| a bare name a plugin expands | the part binds, the *other* part errors as unbound |

The middle row is the dangerous one. `Never` in a remainder slot reads as "this frame has no remainder". It would claim a result is closed while it still carries the numeric columns, which is the unsound direction, and it comes with no diagnostic when the result is used directly.

Writing both parts avoids the question entirely: the checker sees them, so it binds them, and nothing has to reconstruct what it could not see.

______________________________________________________________________

## 5. Effects: what an operation does to a remainder

An effect describes how an operation transforms the remainder. It appears in the return type, because it describes an output rather than a demand.

There are seven, written in nine ways, and like the requirements each exists because a real operation needs it. Three of the spellings are one effect, which section 5.6 explains at the point where the third appears.

### 5.1 Unchanged

Filtering rows, sorting, taking the first hundred, dropping nulls, and reading a declared column all leave the remainder exactly as it was.

```python
Rest = other_columns()


def recent(
    df: TypedDataFrame[Trades, Rest],
) -> TypedDataFrame[Trades, Rest]:
    return df.filter(pl.col("price") > 0).sort("price").head(100)
```

The remainder goes in as `Rest` and comes out as `Rest`.

### 5.2 Emptied

Selecting a fixed list of columns discards everything not named, remainder included. The output has no remainder at all, so none is written:

```python
Rest = other_columns()


def key_only(
    df: TypedDataFrame[Trades, Rest],
) -> TypedDataFrame[TradeKey]:
    return df.select("symbol", "price")
```

The caller passed fifty columns; two come back. That is what `select` means.

Reshaping does the same thing for a different reason. `unpivot` turns every column outside its index into rows, so the columns of the remainder stop being columns at all:

```python
Rest = other_columns()


def long_form(
    df: TypedDataFrame[Trades, Rest],
) -> TypedDataFrame[TradesLong]:
    return df.unpivot(index="symbol")
```

The result is `symbol`, `variable` and `value`, whatever the caller brought. An operation with no remainder in its result is exact for every caller, which is why this one needs no promise.

### 5.3 Names removed

When a column moves out of the remainder and into the known schema, it must leave the remainder, or it would be counted twice.

Adding a column is the clearest case. Adding `notional` to a frame that already had a `notional` replaces it. Afterwards, the known schema has `notional` and the remainder provably does not:

```python
Rest = other_columns()


def add_notional(
    df: TypedDataFrame[Trades, Rest],
) -> TypedDataFrame[TradesWithNotional, Minus[Rest, Literal["notional"]]]:
    return df.with_columns(notional=pl.col("price") * pl.col("quantity"))
```

`Minus[Rest, Literal["notional"]]` reads as "the remainder, without `notional`". Section 5.6 shows that it is one spelling of a more general effect.

This effect has a property worth dwelling on, because a later section depends on it. **`Minus` manufactures a guarantee that its input did not have.** `Rest` alone promises nothing. `Minus[Rest, Literal["notional"]]` promises that `notional` is absent, whatever `Rest` was. Removing something makes it absent, regardless of whether it was ever there.

### 5.4 Two remainders combined

An operation that takes two frames and keeps the columns of both produces a remainder that holds both. `Both[L, R]` is that effect (`E-4`):

```python
L = other_columns()
R = other_columns(disjoint_from=L)


def side_by_side(
    left: TypedDataFrame[Prices, L],
    right: TypedDataFrame[Volumes, R],
) -> TypedDataFrame[PricesAndVolumes, Both[L, R]]: ...
```

The result's remainder holds every column of `L` and every column of `R`. A name is absent from it when it is absent from both.

**Why this is one construct and not two slots.** A frame type could name two remainders side by side instead. It does not, for two reasons. This restates `D-8` and section 6.2 of `OPERATIONS.md`.

A remainder means "the columns other than the declared ones". There is one such set, so a frame has one remainder. Two would be that set with a split drawn through it, and nothing in the frame says where the split falls or why it matters.

The second reason is what breaks. In a **parameter**, two remainders cannot be solved. Given `TypedDataFrame[S, A, B]` and a caller with three columns outside the schema, nothing decides which belong to `A` and which to `B`. One remainder in a parameter has exactly one answer, which is why a parameter never carries more than one.

`Both` states no requirement of its own. A union is only well defined when nothing is counted twice, but that condition is declared on `R`, where every requirement is declared. A caller supplying two remainders that can overlap gets an error naming that declaration, not the return type.

Each operand carries its own effects, which is how section 6.7 writes a result with one side suffixed and the other not:

```python
Both[L, Suffixed[R, Literal["_prev"]]]
```

Because `Both[L, R]` and `Both[R, L]` describe the same frame, the two are the same type. Section 8.7 says how, and it is the one rule this construct genuinely costs.

**One effect covers every operation that combines two remainders.** Placing two frames side by side, aligning them on shared keys, and the joins of section 6 all produce the same thing: a remainder holding both operands.

There is a version of this that would not be `Both`, and it is worth saying why it is absent. An operation could **merge** a shared name into one column instead of requiring the name not to be shared. That needs the two columns to agree on data type, and for two unknown remainders no such agreement can be proven. A caller who wants it declares `disjoint_from=` instead, which is stronger than merging needs but is something a caller can actually establish.

### 5.5 Names renamed by an affix

An operation that renames by adding to every name produces a remainder whose names are all changed. Two effects cover it, one at each end of a name:

```python
Suffixed[R, Literal["_prev"]]  # every name gains "_prev" at the end
Prefixed[Literal["sec_"], R]  # every name gains "sec_" at the start
```

Each holds one column for each column of `R`, under the changed name, with the same data type. An empty affix changes nothing, so `Suffixed[R, Literal[""]]` is simply `R`. In polars they are `name.suffix` and `name.prefix`:

```python
Rest = other_columns()


def namespaced(
    df: TypedDataFrame[SectorRef, Rest],
) -> TypedDataFrame[NamespacedRef, Prefixed[Literal["sec_"], Rest]]:
    return df.select(pl.all().name.prefix("sec_"))
```

`Prefixed` is what lets the namespacing advice of section 6.5 apply to a frame whose columns are not all known. Without it, a user could namespace a reference table only when its every column was declared.

These are the only effects that **invent names**. Every other effect produces names one of its inputs already had. That matters twice. It is why the absence language of section 8.4 carries the two stripping constructors, which nothing else uses, and it is why these effects are allowed only where the renaming is the same for every caller. Adding a constant affix always is, since it never sends two columns to one name.

### 5.6 Columns kept or dropped by name

Some operations take a **selector** rather than a list, and keep or drop whatever it picks. The remainder loses the columns the selector rejects, or the ones it accepts:

```python
Rest = other_columns()


def without_scratch(
    df: TypedDataFrame[Trades, Rest],
) -> TypedDataFrame[Trades, Except[Rest, StartsWith[Literal["tmp_"]]]]:
    return df.drop(cs.starts_with("tmp_"))


def only_previous(
    df: TypedDataFrame[Trades, Rest],
) -> TypedDataFrame[TradeKey, Only[Rest, EndsWith[Literal["_prev"]]]]:
    return df.select("symbol", cs.ends_with("_prev"))
```

`Only[R, C]` keeps the columns of `R` that `C` picks. `Except[R, C]` keeps the ones it does not.

**`Minus` is the same effect.** Removing a list of names is dropping whatever a name-list selector picks, so `Minus[R, Literal["a"]]` and `Except[R, OneOf[Literal["a"]]]` are the same thing written twice. The specification keeps `Minus` as the short spelling for removing a list of names, but there is one operation here, not three. This restates section 6.1 of `OPERATIONS.md` (`E-3`):

| Written | Keeps |
| -- | -- |
| `Minus[R, N]` | everything except the names in `N` |
| `Except[R, C]` | everything the selector `C` does not pick |
| `Only[R, C]` | everything the selector `C` does pick |

One node in the model, three spellings on the page. That is why the rule of section 7.2 about repeated removals collapsing is not a special case: it is the general rule that two filters combine into one, read at the list selector.

The selectors are the ones polars offers by name: a list of names, a prefix, a suffix, a substring, and a general expression. No promise is needed for any of them. A selector decides each column on its own name, so it gives the same answer for every caller, whatever the remainder turns out to hold.

**`Except` is worth noticing.** It produces a guarantee out of nothing, the way section 5.3 does. Dropping everything with a `tmp_` prefix means the result provably has no such column, whatever the input was. A function demanding `without_prefix="tmp_"` accepts the result of this one, and its caller promised nothing.

### 5.7 Columns kept or dropped by type

A selector can also test a column's data type. Whether that is decidable over a remainder depends on what the caller promised about data types, and there are three cases:

```python
AllNumeric = other_columns(all_of=NumericType)
NoNumeric = other_columns(none_of=NumericType)
Anything = other_columns()

df1: TypedDataFrame[Trades, AllNumeric]
df2: TypedDataFrame[Trades, NoNumeric]
df3: TypedDataFrame[Trades, Anything]

df1.select(cs.numeric())  # every column qualifies: the remainder is unchanged
df2.select(cs.numeric())  # none qualifies: the remainder is emptied
df3.select(cs.numeric())  # Only[Anything, C], with C the selector cs.numeric():
# the part the selector keeps, settled by the caller
```

Neither of the first two needs a new effect. They are the identity and the emptied effect, chosen by the requirement the caller already declared. The third is the filtered effect `Only[Anything, C]` (`T-20`): sound, and exact about data types, but it cannot say which names survive.

**A complement swaps the two.** `select(~cs.numeric())` keeps every column that is *not* numeric, so under `all_of=NumericType` it is emptied and under `none_of=NumericType` it is unchanged. That is the exact opposite of the table above:

```python
df1.select(~cs.numeric())  # every column qualifies as numeric: emptied
df2.select(~cs.numeric())  # none does: the remainder is unchanged
```

So the rule is read against **the selector as written**, never against the family it names. A checker that resolved `~cs.numeric()` by looking up "numeric" and applying the table would be wrong in both rows, and wrong silently, since both answers are legal shapes for some other program.

This is worth stating because it is the one place a **type** requirement decides a question about **names**. `all_of` and `none_of` exist in section 4.4 to make type operations checkable. Here they earn a second keep, by making a name-level operation exact that would otherwise stay imprecise.

**A split names the third case.** The table above leaves `df3`, whose caller promised nothing, with a filtered remainder whose names it cannot settle, and that is the common one: a caller with a mix of numeric and non-numeric columns outside the schema fits neither of the first two rows. Declaring the mix is exactly what section 4.6 is for:

```python
Numeric = other_columns(all_of=NumericType)
Rest = other_columns()


def totals(
    df: TypedDataFrame[Trades, SplitColumns[Numeric, Rest]],
) -> TypedDataFrame[TradeKey, Numeric]:
    return df.select("symbol", cs.numeric())
```

Every caller now matches the split. The numeric columns go to `Numeric`, everything else to `Rest`, and `select(cs.numeric())` keeps the first and drops the second.

Counted over every shape a five-column caller can bring, where two of the five are numeric: a single remainder with a family requirement settles **11 of 32**, and a split settles all 32. The 11 are the shapes whose data types happen to be all numeric (4 of them) or all non-numeric (8), less the empty shape counted in both. The remaining **21** cannot be settled by any single family requirement (`M-4`).

> Measured by `test_a_family_requirement_answers_eleven_of_thirty_two` and `test_a_family_requirement_leaves_twenty_one_of_thirty_two` in `spec/tests/test_measurements.py`.

So the three cases are not three sorts of caller. They are one sort of caller, described at three levels of detail:

| The remainder is declared as | `select(cs.numeric())` gives |
| -- | -- |
| `all_of=NumericType` | unchanged |
| `none_of=NumericType` | emptied |
| nothing at all | `Only[Anything, C]`, with `C` the selector `cs.numeric()`: the filtered effect |
| `SplitColumns[Numeric, Rest]` | exactly `Numeric` |

The first two are the split's degenerate cases, where one part would be empty for every caller. Section 4.6 rejects writing those as splits, because a part that can never hold anything is a mistake rather than a shorthand.

### 5.8 Data types replaced

Some operations keep every column and every name, and change only the data types. Counting the null entries of each column turns every column, whatever it was, into a count:

```python
Rest = other_columns()


def counts(
    df: TypedDataFrame[Trades, Rest],
) -> TypedDataFrame[TradeCounts, Retyped[Rest, pl.UInt32]]:
    return df.null_count()
```

`Retyped[Rest, pl.UInt32]` means "the same columns as `Rest`, all of data type `UInt32`". The names are unchanged, which matters: everything that was provably absent from `Rest` is still provably absent, so nothing is lost.

Where the resulting data type genuinely varies by input data type, and so cannot be named, the effect is `Retyped[Rest, Unchecked]`.

`Unchecked` is a data type meaning "this column exists under this name, and its data type is not checked". `D-27` defines it for frames and `D-41` for expressions, and the two are one marker. Operations read it through the lifting that `D-41` defines: an operation whose output varies with its input propagates `Unchecked`, an operation whose output is fixed recovers a concrete data type, and a type gate passes when some data type passes. It is not a way of saying "any type at all is allowed here", which would be a claim about the column; it is a way of saying nothing about the data type while continuing to say something true about the name.

Keeping the name is better than discarding the frame, and section 8.9 says why the two sorts of uncertainty are worth keeping apart.

______________________________________________________________________

## 6. Joins

Joining is the hardest operation in this specification, and it is the one users ask about first. This section covers it in full: what makes it hard, what the rule is, what a user must write, and what the user gets back.

Nothing here is a new principle. Everything follows from the every-caller rule of section 3.2.

### 6.1 What polars does when two names collide

A join places the columns of two frames side by side. When a name appears on both sides, polars keeps the left column under its own name. It renames the right column by adding a suffix. The default suffix is `_right`.

**The left side is never renamed.** This holds for every join type, including `how="right"`. Take a left frame of `symbol`, `price`, `sector` and a right frame of `symbol`, `sector`, `country`:

| Call | Result columns |
| -- | -- |
| `left.join(right, on="symbol")` | `symbol`, `price`, `sector`, `sector_right`, `country` |
| `left.join(right, on="symbol", how="right")` | `price`, `sector`, `symbol`, `sector_right`, `country` |
| `left.join(right, on="symbol", suffix="_ref")` | `symbol`, `price`, `sector`, `sector_ref`, `country` |
| `left.join(right, on="symbol", how="full")` | `symbol`, `price`, `sector`, `symbol_right`, `sector_right`, `country` |

In every row it is the right frame's `sector` that changed its name.

The last row shows one more fact. A full join that does not coalesce keeps both key columns, so the key itself can take the suffix.

**A suffix is applied once, never twice.** If the renamed column collides in turn, polars raises rather than suffixing again. Joining a left frame holding both `price` and `price_prev` to a right frame holding `price`, under suffix `_prev`, is an error and not a `price_prev_prev`. So the set of output names is decided in one pass, and a rule that tried to iterate would describe an operation polars does not have.

### 6.2 Why a remainder makes this hard

The rename depends on the **whole** left frame, not on its declared part. A right column takes the suffix when its name matches any left column at all.

A remainder is a set of columns the checker cannot name. So when the left frame has a remainder, the checker cannot tell which right columns will be renamed:

```python
Rest = other_columns()


def enrich(trades: TypedDataFrame[Trades, Rest]):
    return trades.join(SECTORS, on="symbol")  # SECTORS: symbol, sector, ctry
```

For a caller whose frame has no `sector`, the result holds `sector`. For a caller whose frame has one, the result holds that caller's `sector` plus a `sector_right` from the reference table. Two callers, two different result types.

The every-caller rule then decides the matter. One return type must be correct for every caller. No single type is correct for both of these callers.

### 6.3 The rule

> A join over a remainder is allowed when the same columns are renamed for every caller.

That is the whole rule, and it follows straight from section 3.2. The result type must be correct for every caller. A column that is renamed for one caller and not for another appears under two different names, so no single type covers both.

Note what the rule does **not** say. It does not say that renaming is forbidden. Renaming is fine when it is certain. What is forbidden is renaming that *depends* on the caller.

**Two situations are certain, and they are opposite ends of the same axis:**

| Situation | What happens | Result |
| -- | -- | -- |
| nothing collides | no column is renamed | right frame's columns keep their names |
| everything collides | every column is renamed | right frame's columns all take the suffix |

Anything between them is uncertain. If some names collide and others can or can not, the checker cannot say which columns come out under which name, and section 6.9 reports an error.

Sections 6.4 through 6.6 cover the first situation, which is the more common one. Section 6.7 covers the second.

### 6.4 The four places a collision can come from

This section and the two after it are about the first situation of section 6.3: making sure nothing collides.

A left frame is a declared schema and a remainder. So is a right frame. A collision is a shared name, and it can arise from four pairings:

| Names that can collide | Ruled out by |
| -- | -- |
| left declared, right declared | nothing to write. The checker sees both sets and compares them. |
| left **remainder**, right declared | a bound on the left remainder |
| left declared, right **remainder** | a bound on the right remainder |
| left **remainder**, right **remainder** | `disjoint_from=` on one of the two |

**The surviving-left rule.** "Left" in that table means the left columns that *survive* the join, not every column the left frame had. The two differ in one case, and it is not a corner: joining with `how="right"` and coalescing drops the left key, so a right column named like that key collides with nothing and is not renamed. Reading the rule against the left frame's full column set would reject a join polars performs happily.

```
left {k, v} joined to right {k, v}, on "k"

  how="inner", coalesce=True   ->  k, v, v_right      the left key survives
  how="right", coalesce=True   ->  v, k, v_right      the RIGHT key survives
```

> Measured by `TestJoinSurvivingColumns` in `spec/tests/test_polars_facts.py`: the symmetric shape above, an asymmetric shape where a non-colliding right column keeps its name, differing key names under `left_on`/`right_on`, and a two-key join whose coalesced keys move together.

The join is allowed when all four hold, and the checker verifies all four itself. It does not assume that a bound written in the signature covers the right names. A user who writes the wrong bound gets the error of section 6.9, not a wrong answer.

Most code satisfies three of the four rows for free, because most joins have a closed frame on at least one side.

### 6.5 Joining an open frame to a closed one

This is the common shape. The left frame is open and the right frame is a reference table with a known schema.

Rows one and three of section 6.4 are then free. The right frame has no remainder, so it holds nothing the checker cannot see. Only row two is left. The left remainder must not hold any of the right frame's names:

```python
NoSectorData = other_columns(without=["sector", "country"])


def enrich(
    trades: TypedDataFrame[Trades, NoSectorData],
) -> TypedDataFrame[EnrichedTrades, NoSectorData]:
    return trades.join(SECTORS, on="symbol")
```

The caller must supply a frame with no `sector` and no `country`. In return the caller gets an exact result, and its own remainder comes back untouched.

**A wide reference table costs one name for each of its columns.** A table of thirty columns needs thirty names in the bound. The bound is part of a public signature, so adding one column to a private reference table then becomes a breaking change for every caller.

Two forms avoid that cost.

**Take only the columns you need.** Narrowing the right frame first is often the better program anyway:

```python
NoSector = other_columns(without=["sector"])


def enrich(
    trades: TypedDataFrame[Trades, NoSector],
) -> TypedDataFrame[TradesWithSector, NoSector]:
    return trades.join(SECTORS.select("symbol", "sector"), on="symbol")
```

**Give the reference table its own namespace.** This is the requirement of section 4.3, and joining is what it exists for. The caller promises that a whole namespace of names is free, and the right frame is renamed into that namespace:

```python
NoSecPrefix = other_columns(without_prefix="sec_")


def enrich(
    trades: TypedDataFrame[Trades, NoSecPrefix],
) -> TypedDataFrame[EnrichedTrades, NoSecPrefix]:
    return trades.join(
        SECTORS.select(pl.all().name.prefix("sec_")),
        left_on="symbol",
        right_on="sec_symbol",
    )
```

One promise now covers the whole table, whatever its width. The reference table can gain a column and no signature has to change.

### 6.6 Joining two open frames

When both frames are open, all four rows of section 6.4 need attention. Three promises cover them:

```python
# each bound names the OTHER side's declared columns
L = other_columns(without=["sector", "country"])
R = other_columns(without=["price", "quantity"], disjoint_from=L)


def merge(
    left: TypedDataFrame[Trades, L],  # symbol, price, quantity
    right: TypedDataFrame[Sectors, R],  # symbol, sector, country
) -> TypedDataFrame[Enriched, Both[L, R]]:
    return left.join(right, on="symbol")
```

Each promise closes one row:

- `L`'s bound closes row two. The left remainder holds nothing the right frame declares.
- `R`'s `without=` closes row three. The right remainder holds nothing the left frame declares.
- `R`'s `disjoint_from=L` closes row four.
- Row one needs nothing written. `Trades` and `Sectors` are both declared, so the checker compares them itself. Their shared `symbol` is the join key, which is allowed.

`Both[L, R]` in the return type is only an effect. It names the result's remainder as the union of the two and demands nothing, because the demand is already on `R`'s declaration.

The two `without=` lists name the *other* side's declared columns, and those names appear in the same signature. That makes them easy to write and easy to keep correct.

This shape is more work than section 6.5. Two open frames joined together is a genuinely harder thing to make a promise about.

### 6.7 Joining a frame to one contained in it

This is the second situation of section 6.3, where **everything** collides.

The pattern is comparing a frame with another of the same shape. Today's trades against yesterday's, one run of a model against the previous run, a table against a corrected copy of itself:

```python
today.join(yesterday, on="symbol", suffix="_prev")
```

Every column of the right frame collides, because the two frames have the same columns. So every one is renamed, for every caller, and the result is exact:

```
['symbol', 'price', 'qty', 'venue', 'price_prev', 'qty_prev', 'venue_prev']
```

Two promises make this determinate:

```python
L = other_columns(without_suffix="_prev")
R = other_columns(within=L)


def compare(
    current: TypedDataFrame[Snapshot, L],
    previous: TypedDataFrame[Snapshot, R],
) -> TypedDataFrame[Compared, Both[L, Suffixed[R, Literal["_prev"]]]]:
    return current.join(previous, on="symbol", suffix="_prev")
```

- `within=L` says every name of `R` is also a name of `L`, so every column of `R` is certain to be renamed. Without it, some column of `R` can fail to collide, and that column's output name would depend on the caller.
- `without_suffix="_prev"` says `L` holds no name ending in `_prev`, so the renamed columns cannot collide with anything. This is not a formality. Without it, a left frame holding both `price` and `price_prev` makes polars raise `DuplicateError`, because the renamed `price` lands on a name already taken.

The result's remainder is `L` together with every column of `R` suffixed, which is what `Both[L, Suffixed[R, Literal["_prev"]]]` says.

`within=` is the exact opposite of `disjoint_from=`, and that is not a coincidence. One requires the overlap to be empty, the other requires it to be total. Those are the two ends where the outcome does not depend on the caller.

### 6.8 Joins that need no promise

**Semi and anti joins take no columns from the right frame.** They only filter the left frame's rows. Nothing can collide, so no promise applies and the remainder passes through:

```python
Rest = other_columns()


def only_known(
    trades: TypedDataFrame[Trades, Rest],
) -> TypedDataFrame[Trades, Rest]:
    return trades.join(SECTORS, on="symbol", how="semi")
```

**A join between two closed frames** is checked exactly, as it always was. Nothing in this section restricts it.

### 6.9 What the user sees when a promise is missing

A join that cannot rule out a rename is an error at the join. It is not a silent loss of type information:

```python
Rest = other_columns()


def enrich(trades: TypedDataFrame[Trades, Rest]):
    return trades.join(SECTORS, on="symbol")
    #      error: this join can rename "sector" and "country" of the right
    #      because the remainder Rest can contain those names.
```

The message names the columns at risk and the remainder that puts them at risk. It does not only report that something is wrong.

The uncertain case is the middle one of section 6.3, so the message can name both ways out. The user can rule the collision out, with a bound or a projection, or make it total, with `within=`. Which one is right depends on what the code means, and the checker does not guess.

Reporting an error here has a real cost. That line is legal polars and it runs. The alternative is worse. A checker that returns a plain frame teaches the user nothing at the point of the problem. The lost type information then shows up several calls later, as a column that cannot be found.

### 6.10 Joining a frame that is split

A split is extra information, so it can only help a join. Nothing that was provable about an undifferentiated remainder becomes unprovable once its parts are named. What changes is how much has to be *said*, and that depends on whether the parts were declared by name or by data type.

**A name-declared part settles its own pairings, with nothing promised.** A part declared `only_prefix="sec_"` and a part declared `without_prefix="sec_"` cannot share a name, and that follows from the two declarations alone. No `disjoint_from=` is written and nothing is checked at the call site. This is the best case the whole section has: a promise replaced by a fact.

**A type-declared part settles nothing, but costs nothing either.** A data type is a fact about one frame's column, not about a name, so it constrains no other frame. Two honestly-described parts can still collide:

```
'qty' is Int64 on the left  -> the left's numeric part
'qty' is String on the right -> the right's non-numeric part
```

So a type split leaves the disjointness question exactly where it was. It does not make it worse: **one promise about the two whole remainders discharges every pairing**, because each part is a subset of its frame's remainder.

> Measured by `TestSplitPairingsInheritDisjointness` in `spec/tests/test_requirements.py`, over every pair of frames a three-name universe allows: with `G` and `H` promised disjoint, no part-pairing collides.

**What a split does cost is a way to say it.** The checker has $(m+1)(n+1)$ pairings to verify for $m$ and $n$ parts, and a promise written with `disjoint_from=` needs a name for each. Before a split there was one name for the whole remainder; after it there is none, so the promise lists the parts, using the set form of section 4.5:

```python
Left = SplitColumns[LeftNum, LeftRest]
Right = other_columns(disjoint_from={LeftNum, LeftRest})
```

That is bookkeeping, not a weakening. It is called out because the alternative reading, that more pairings mean more to prove, is wrong and inviting.

**Two smaller consequences.**

The join key is unaffected. Every join key must be a declared column (section 5.2 of `OPERATION_CATALOG.md`), and a declared column is never in the remainder, so no part can hold a key. The surviving-left rule of section 6.4 therefore applies only to the declared half, and parts always survive a join whatever the `how=`.

The containment join of section 6.7 needs care in one direction only. A **part** cannot declare `within=`, because section 4.6 admits only per-column tests and a relation is not one. The frame a part is contained *in* can be split freely: the promise is then written `within={LeftNum, LeftRest}`, exactly as `disjoint_from=` is.

### 6.11 Summary

| Shape | What the user writes | Result |
| -- | -- | -- |
| semi or anti join | nothing | remainder unchanged |
| open left, closed right | `other_columns(without=[...])` | remainder unchanged |
| the same, wide right frame | `other_columns(without_prefix="...")` | remainder unchanged |
| open left, narrowed right | `select` on the right, then a short bound | remainder unchanged |
| two open frames, nothing shared | `disjoint_from=` plus two `without=` | remainder is the union |
| a frame and a copy of itself | `within=` plus `without_suffix=` | union, right side suffixed |
| partial or unknown overlap | nothing that works | error at the join |

In every allowed shape the result is exact. No join produces a partly known frame. Either no column is renamed or every column is, and which of the two it is can be read off the signature. That is what the rule of section 6.3 buys.

`join_asof` and `join_where` follow the same naming rule and the same requirements.

______________________________________________________________________

## 7. Compatibility

Requirements are stated on functions. Effects are produced by functions. The system is useful only if the two fit together, so that the output of one function can be the input of the next.

### 7.1 When a remainder satisfies a requirement

A remainder satisfies a requirement if the requirement holds for **every** possible value of that remainder (`D-26`). This is the every-caller rule again, applied one level up.

Three cases cover it.

**A promise satisfies a weaker promise.** A remainder declared to exclude `ticker` and `venue` can be passed where only `ticker` must be excluded.

This holds across the two forms of sections 4.2 and 4.3, and it runs one way only. A namespace promise satisfies a demand for any name inside that namespace, because excluding everything beginning with `sec_` excludes `sec_country` in particular. A list promise does not satisfy a namespace demand, however long the list, because the namespace is infinite and the list is not:

```python
Namespaced = other_columns(without_prefix="sec_")
Listed = other_columns(without=["sec_country", "sec_sector"])
OneName = other_columns(without=["sec_country"])


def needs_one_name(df: TypedDataFrame[Trades, OneName]) -> None: ...
def needs_namespace(df: TypedDataFrame[Trades, Namespaced]) -> None: ...


def caller(
    a: TypedDataFrame[Trades, Namespaced],
    b: TypedDataFrame[Trades, Listed],
) -> None:
    needs_one_name(a)  # accepted: the namespace covers sec_country
    needs_one_name(b)  # accepted: the list names it
    needs_namespace(a)  # accepted
    needs_namespace(b)  # rejected: a list cannot establish a namespace
```

Between two namespace promises the shorter prefix is the stronger one. Excluding everything beginning with `sec` also excludes everything beginning with `sec_`, so the first satisfies the second and not the reverse.

**An effect satisfies a promise it creates.** This is the important one. A completely unconstrained remainder, with `notional` removed, satisfies a function that demands `notional` be absent:

```python
Anything = other_columns()
NoNotional = other_columns(without=["notional"])


def needs_notional_free(
    df: TypedDataFrame[TradesWithNotional, NoNotional],
) -> None: ...


def pipeline(df: TypedDataFrame[Trades, Anything]) -> None:
    needs_notional_free(add_notional(df))  # accepted
    needs_notional_free(df)  # rejected: Anything promises nothing
```

`Anything` guarantees nothing at all. `Minus[Anything, Literal["notional"]]`, which is what `add_notional` returns, guarantees exactly what the callee needs. The effect did the work, and the caller had to promise nothing.

**Nothing else satisfies anything.** In particular, an unconstrained remainder satisfies only an unconstrained requirement.

### 7.2 Effects compose

An effect applies to whatever remainder its function receives. So applying two functions in sequence applies both effects, and the result is written as one effect wrapped around the other. It is read from the inside out.

```python
Rest = other_columns()


def pipeline(
    df: TypedDataFrame[Trades, Rest],
) -> TypedDataFrame[Final, Minus[Rest, Literal["notional", "rank"]]]:
    return add_rank(add_notional(df))
```

The first function removes `notional`, the second removes `rank`, and the result removes both.

That example hides something, because the two effects are the same effect. Written out in full, the body produces `Minus[Minus[Rest, Literal["notional"]], Literal["rank"]]` while the signature declares `Minus[Rest, Literal["notional", "rank"]]`. Those are two spellings of one thing, and the specification treats them as the same type. Section 8.7 says how.

**Which pairs are equal to a single effect.** Two applications of the same effect always combine, and the specification uses the shorter form:

| Applied in this order | Equal to | Because |
| -- | -- | -- |
| filter by $C$, then filter by $D$ | filter by $C\ \mathtt{and}\ D$ | each keeps a column only if its own test passes |
| retype to $U$, then retype to $V$ | retype to $V$ | the second replaces every data type the first set |
| suffix $s_1$, then suffix $s_2$ | suffix $s_1 + s_2$ | each appends to whatever name it was given |

The first row is the general form of a rule that looks like three. Removing two name lists in turn, dropping two namespaces in turn, and narrowing twice with `Only` are one rule read at three selectors, because the three spellings of section 5.6 are one effect.

**Different effects do not combine, but they do commute.** Dropping a column and then counting nulls gives the same frame as counting nulls and then dropping the column:

```python
df.drop("a").null_count()  # remove, then retype
df.null_count().drop("a")  # retype, then remove
```

A user can write those in either order, and both results must be acceptable to the same function. A specification that treated them as different types would reject one of two programs that do the same thing.

The same holds for every other pair of different effects, though a filter moved past a rename must have its selector translated, since it is now testing a name the rename has not yet changed. Section 8.7 gives the canonical form that settles all of them at once, and shows why that translation is computed rather than substituted.

**Which pairs do not collapse.** `Both` brings together two different remainders rather than narrowing one, so a chain of them produces a type whose shape mirrors the chain. An emptied remainder absorbs everything after it, because there is no remainder left for a later effect to apply to. `Suffixed` and `Minus` do not commute, because removing `a` before suffixing removes `a_prev` from the result, while removing `a` after suffixing removes nothing at all.

**What composition almost never does.** Only `Suffixed` and `Prefixed` introduce a name that none of their inputs had. Every other effect produces names an input already held. That is what keeps the absence language of section 8.4 small, and those two are the only place it has to pay for a name it invented.

### 7.3 Where the system is deliberately incomplete

Some correct programs are rejected. These are listed because a specification that hides its own limits is not trustworthy. Section 8.10 gives every source of incompleteness. Two of them are the ones a user meets first when composing functions:

**Facts that relate two separately-written expressions.** Concatenating a frame with itself after both have had the same column dropped is safe, but the checker compares each side's guarantees independently and cannot see that they came from the same place.

**Case-splitting on a frame's type.** If a variable holds one of two frame types, calling a method on it works, but passing it to a function with a remainder does not. This is one form of the approximation at case splits that section 8.10 lists.

Anything decided by data is not in this list. It is a boundary (section 3.3): the model does not describe it at all, so it is not incompleteness.

______________________________________________________________________

## 8. How it works

The sections above describe what the system promises. This one describes the machinery: what a remainder means, what is recorded about it, how the record changes, and how each sort of question is decided.

Nothing here describes a particular implementation. It is the logic any implementation has to follow.

**Where the proofs are.** This section states the machinery and says what each piece guarantees. `OPERATIONS.md` defines every piece formally and proves every claim, and each part below names the identifier to look up. `OPERATIONS.md` is authoritative: where a part below restates one of its definitions, a disagreement is an error in this document. A claim stated here without an identifier is a claim this document does not itself establish.

### 8.1 What a frame type is, exactly

A frame type has two parts, of which only the first is always present. This restates `D-8`:

```
TypedDataFrame[ Schema, Remainder? ]
```

- The **schema** is a column set (`D-5`): a finite map from column names to data types. Always present, possibly empty.
- The **remainder** stands for the columns not in the schema. There is **at most one**, and that is forced rather than chosen. "The columns other than the declared ones" is one set. Two slots would also be unsolvable in a parameter: given `TypedDataFrame[S, A, B]` and a caller with three columns outside the schema, nothing decides which belong to `A` and which to `B`. `Both` is how two remainders become the one a frame can hold.

There is no third part. A frame type says what columns a frame has, and nothing else. Every requirement, relational ones included, lives on the declaration of a remainder, never in a frame type. Two frames with the same schema and the same remainder are the same type, with no further condition to compare.

That is what section 4.5 buys by putting relational requirements at the declaration site. A frame type is a description of contents only, so a reader never has to ask whether some part of it is a demand rather than a fact.

**A split fills the one remainder slot.** `SplitColumns[A, B]` is not a second remainder. It is a way of naming parts of the single one, and section 8.8 gives the rule that binds it.

### 8.2 What a remainder denotes

Fix a **column set**: a finite partial map from names to data types (`D-5`).

A remainder expression contains variables (`D-18`). Give every variable a column set, and the expression denotes one definite column set (`D-19`). Such a choice is a **binding** $\beta$, and $\llbracket e \rrbracket_\beta$ is the column set $e$ denotes under it. Different bindings give different answers, and a remainder is exactly the set of answers over all bindings.

So a statement about a remainder means "true under every binding". This is the every-caller rule of section 3.2, restated at the level of the machinery, and it is the soundness condition every rule below must respect.

### 8.3 What is recorded

Two things, a **record** $(P, \Theta)$. This restates `D-16`.

**An absence rule** $P$. A predicate over names. Where it holds, that name is absent under every binding.

**A type set** $\Theta$. The data types the remainder's columns can have, under every binding (`D-15`).

That is the whole record. Three things are deliberately *not* recorded, and each omission is forced rather than chosen.

**Which names are present.** Knowing a column's name is what puts it in the schema. The two halves never overlap, so a name cannot be known and remain in the remainder. This is why section 4 has no "must contain" requirement: it would be written by declaring the column.

**Whether the remainder can be empty.** Empty means every name is absent, which the absence rule already says. The unconstrained remainder is the one whose rule holds of nothing; the empty one is the rule that holds of everything.

**Column order and column count.** No operation in the grammar asks, and section 11 lists the operations that consequently cannot be typed.

A remainder whose rule holds of every name carries no information, and is written as **no remainder at all**. This is part of the canonical form (`D-22`), not a convention: it keeps `TypedDataFrame[S]` and `TypedDataFrame[S, <provably empty>]` from being two spellings of one frame.

### 8.4 Three languages, not one

"A predicate over names" is too vague to build on, and there is more than one predicate here. The machinery uses three languages, and keeping them apart is what makes each rule statable.

| Language | Tests | Written by a user | Defined |
| -- | -- | -- | -- |
| **absence rule** | a name | never directly | `D-9` |
| **selector** | a whole column: name and data type | yes, as `cs.numeric()` and its kin | `D-12` |
| **type set** | which data types can occur | never; it is computed | `D-15` |

The absence language is the smallest of the three. This grammar restates `D-9`:

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

**`ofType(F)` is deliberately not in it.** This language describes names, and a data type is not readable from a name. `ofType` is a *selector*, usable wherever section 5.6 takes one, and section 8.5 turns it into a rule from this language by consulting the type set rather than by testing a name. So the selector language is strictly larger than the absence language, and `ofType` is the whole difference.

**The named constructors mirror the selectors a user can write.** `oneOf` is a list of names, `startsWith` and `endsWith` are the namespace bounds of section 4.3, and `contains` is the substring test.

`matching(m)` is the escape, and it is deliberately last. It holds a general name expression (`D-11`), and nothing a user writes produces one. It is needed for more than a general selector: moving a filter past a renaming, which section 8.7 does, produces a test that is not always one of the named shapes. Filtering on `startsWith("ab")` after appending `"b_"` also accepts a column named `a`, because the affix supplies the rest of the prefix. The translated rule is `startsWith("ab") or oneOf({"a"})`, and in general no named constructor describes it. Without `matching` there would be expressions the canonical form could not write down.

**Every constructor here denotes a regular set of names** (`D-11`), `matching` included. That is not a technicality. It is why `matching` can be added without breaking anything, why the language stays closed under the effects below, and why containment between two rules is decidable, so section 8.7 can decide type identity exactly and section 8.6 can decide `disjoint_from=` rather than approximate it.

A `matching` therefore carries the machine that decides it, which answers every question about it. A user states a requirement with a name, a prefix, a suffix or a substring. Only a rule the checker derived falls back to `matching`, and only that code pays for it. A type spells a `matching` by its names alone: a regular expression where one reads well, as `ab.*`, and the text of its automaton otherwise.

`startsWith`, `endsWith` and `contains` describe sets no list can. "Every name beginning with `sec_`" has no finite list of names equal to it. A language of lists alone would force a user joining a thirty-column reference table to write thirty names, and to rewrite them whenever that table changed. Section 6.5 shows what that costs.

**The language is closed under every effect** (`T-2`), which it must be, or a chain of operations would eventually produce a record that cannot be written down. Closure holds for **both** halves of the record, and section 8.5 gives the rules.

**The test for admitting a new effect** (`T-3`). An effect renames each column by some function $f$. A name $m$ is absent from the result exactly when every input name that $f$ sends to $m$ is absent:

$$
\mathrm{absentResult}(m) \Leftrightarrow \forall n.\ \bigl(f(n) = m \Rightarrow \mathrm{absentInput}(n)\bigr)
$$

So an effect is admissible exactly when **the preimage under $f$ is expressible in this language**, and $f$ sends no two columns to one name. That single question decides every effect anyone proposes later. `Suffixed` passes because the preimage of $m$ is empty when $m$ lacks the suffix and a single name when it has one, and both cases are constructors. An operation that lowercased every name fails the second condition: it sends two different columns to one name, which section 6.3 already rejects for joins and which is rejected here for the same reason.

Two properties follow, and both matter for a real checker.

**Deciding one name is cheap.** Evaluating $\mathrm{absent}(n)$ walks the rule once.

**Deciding disjointness is not.** `disjoint_from=` asks whether *any* name is possible in both of two remainders, which is a question about the whole language rather than one name. Every construct denotes a regular set, so the question is decided exactly (`T-9`), by intersecting two machines and asking whether the result is empty. That is not the cost of a lookup, and section 8.6 describes how the common cases avoid paying it.

### 8.5 What each effect does

Both halves of the record change together, and an effect that said nothing about one of them would leave a later question about it unanswerable. `T-2` gives the full table and proves it row by row. In prose, for an operand $e$ whose record is $(P, \Theta)$:

**Minus** adds names to the rule with `or`. The names are added **unconditionally**, whatever the input rule was. Removing a name makes it absent whether or not it was ever there. That one line is the entire explanation of the property sections 5.3 and 7.1 describe: an effect can produce a guarantee its input did not have, which is what lets an unconstrained remainder satisfy a demanding function.

**Retyped** leaves the rule untouched and replaces the type set. Leaving the rule untouched is the point: an operation that keeps every column and changes only data types loses nothing about which names exist, and the record must say so rather than collapsing to "anything is possible".

**Both** takes `and` on the rule and a union on the type set. A name is absent from the union only when it is absent from both inputs. Nothing is renamed, so the result holds no name that neither input held.

**Suffixed** is the one effect that invents names at the end, and its rule is worth deriving rather than asserting. $\mathtt{Suffixed}(e, s)$ holds $n + s$ for each $n$ that $e$ holds, and nothing else. So a name $m$ is **present** exactly when $m$ ends in $s$ and the name obtained by removing $s$ is present in $e$. Negating that gives absence:

$$
\begin{aligned}
\mathrm{absent}(m) &\Leftrightarrow (m \text{ does not end in } s) \lor (\text{the name under } s \text{ is absent from } e) \\
&\Leftrightarrow m \in \llbracket \mathtt{not}\ \mathtt{endsWith}(s)\ \mathtt{or}\ \mathtt{afterStripping}(s, P) \rrbracket
\end{aligned}
$$

Both halves are needed and neither is a list. "Does not end in `_prev`" is a perfectly good fact about a name, and no finite list of names states it. A language restricted to lists would prove **nothing** absent after a suffixing rename, and every operation after one would lose all checking. No data type changes, so the type set is untouched.

**Prefixed** is the same derivation with the ends swapped, giving `not startsWith(p) or beforeStripping(p, P)`. Nothing about the argument depends on which end of the name changes, which is why the two effects share section 5.5 and have two symmetric rules here.

**Where**, the filter behind the three spellings of section 5.6 (`E-3`), does not rename, so its rule needs no stripping. A name is absent from the result when it was absent before **or** the selector is certain to reject every column of that name. The three spellings differ only in the selector, so this one rule covers all of them. The type set narrows to the data types a surviving column can have.

**A type test cannot be read off a name, so `ofType` consults the type set instead.** $\mathtt{Where}(e, \mathtt{ofType}(F))$ has three cases, and only the last loses anything:

| The type set $\Theta$ of $e$ | The rule | Exact? |
| -- | -- | -- |
| contained in $F$ | $P$, and the type set unchanged | yes |
| disjoint from $F$ | `always`, and the type set empty | yes |
| neither | $P$, with the type set narrowed to $\Theta \cap F$ | for names, no |

The third row is sound, because dropping a disjunct only ever claims *less* is absent. Note that its **type** half is still exact: whatever survives is in $F$, so a later type question about the result is answerable even though a name question is not. It is also the case a split exists to remove. Over the same 32 caller shapes section 5.7 counts: an undifferentiated remainder is imprecise in **21** of them, a part whose data types are contained in $F$ in 0 of the 4 it admits, and a part whose data types are disjoint from $F$ in 0 of the 8 it admits.

So splitting once, on the data types a later test reads, makes that test exact on every part, which is why the feature does not trade one gap for another.

This is where an effect manufactures a guarantee, as `Minus` does, and for the same reason: `Minus` is one of the spellings. Every name the selector rejects becomes absent whatever the input rule was, which is what lets a frame that promised nothing satisfy a namespace requirement after a single `drop`.

### 8.6 Two ways a requirement is discharged

A requirement is checked at a call site, where the callee's remainder variables are matched against what the caller supplied. There are two cases, and they are decided differently (`T-12`).

**The caller is closed.** The variable binds to a definite column set: the caller's actual columns, minus the callee's declared schema. Every requirement is then decided exactly, by inspection. `disjoint_from=` and `within=` are simple set operations on two known sets, and the cost concern of section 8.4 does not arise.

**The caller is itself open.** The variable binds to another expression, so the requirement must be *proven* from the record computed for it. This is the harder case, and it is where the whole design earns its keep. Write $(P_A, \Theta)$ for the record of the caller's remainder $A$, and $P_B$ for the rule of a named remainder $B$. This table restates `T-12`:

| Requirement | Holds when |
| -- | -- |
| none (`R-1`) | always |
| `without=[n, ...]` (`R-2`) | $n \in \llbracket P_A \rrbracket$ for each $n$ |
| `only=[n, ...]` (`R-3`) | $\llbracket P_A \rrbracket \supseteq N \setminus \{n, \dots\}$ |
| `without_prefix=p` (`R-4`) | $\llbracket P_A \rrbracket \supseteq \llbracket \mathtt{startsWith}(p) \rrbracket$ |
| `only_prefix=p` (`R-5`) | $\llbracket P_A \rrbracket \supseteq \llbracket \mathtt{not}\ \mathtt{startsWith}(p) \rrbracket$ |
| `without_suffix=s` (`R-6`) | $\llbracket P_A \rrbracket \supseteq \llbracket \mathtt{endsWith}(s) \rrbracket$ |
| `only_suffix=s` (`R-7`) | $\llbracket P_A \rrbracket \supseteq \llbracket \mathtt{not}\ \mathtt{endsWith}(s) \rrbracket$ |
| `all_of=S` (`R-8`) | $\Theta \subseteq S$ |
| `none_of=S` (`R-9`) | $\Theta \cap S = \emptyset$ |
| `disjoint_from={B, ...}` (`R-10`) | for each $B$, no name is in $N \setminus \llbracket P_A \rrbracket$ and in $N \setminus \llbracket P_B \rrbracket$ |
| `within={B, ...}` (`R-11`) | $N \setminus \llbracket P_A \rrbracket = \emptyset$, or $\Theta = \emptyset$ |

Here $N$ is the set of all names, and $S$ in the two data-type rows is a set of data types. A record that permits no column satisfies every row.

`within=` is the exception. The two remainders are bound independently, so a caller can bind $B$ to no columns while $A$ holds one, and comparing the two rules proves nothing. An open caller therefore discharges `within=` only when its remainder permits no column. A closed caller discharges it by inspection.

Three consequences.

**Checking reads knowledge, never syntax.** `Minus[Anything, "notional"]` satisfies `without=["notional"]` because the rule computed for it holds of `notional`, not because the two mention the same string. Any expression reaching the same rule satisfies the requirement equally.

**Weakening is free.** A remainder proving `a` and `b` absent satisfies a demand for `a`, because the test consults only the names demanded. No separate rule is needed.

**An unprovable requirement propagates outward, and is not inferred.** If a function's body calls something demanding `without=["sector"]` and the body's own remainder cannot prove it, that is an error in the body. The fix is to declare the requirement on the enclosing function, which pushes the obligation to its callers. Requirements are never inferred from a body: a function's signature is its contract, and a contract that changed when its body changed would silently break callers.

### 8.7 Canonical form, and when two types are the same

Two different questions are easy to conflate, and conflating them makes composing two correct helpers fail.

The first is requirement checking, covered above: knowledge must be computed through the **whole** expression (`D-20`). A checker reading only the outermost effect sees `Minus[Minus[R, "a"], "b"]` as guaranteeing only that `b` is absent, and silently loses the guarantee about `a`. Composing two correct helpers then produces a false error.

The second is **type identity**. A body that produces `Minus[Minus[R, "a"], "b"]` must satisfy a signature declaring `Minus[R, {"a", "b"}]`, because those denote the same thing.

**Every expression has one canonical form, and it is computed rather than searched for** (`D-22`, `T-4`). This restates `D-22`. The form has five slots, written from the outermost on the left to the base on the right:

```
retype?  over  prefix?  over  suffix?  over  filter?  over  base
```

where a **base** is a variable, a sorted union of canonical forms, or nothing at all. Every effect an expression applies folds into one of those slots, by a rule given once per effect. Two suffixes concatenate; two filters take `and` of their selectors, and a filter that reads only names is then keyed by the set of names it accepts, so removing two names one at a time and removing both at once give one form; the outer retyping wins; a union flattens and sorts its operands, and drops a provably empty one, since `Both[Nothing, S]` and `S` denote the same frame; an emptying effect absorbs everything below it, leaving the base nothing at all.

Defining the form this way, rather than as the fixed point of a rewriting system, is deliberate. A fixed point owes two arguments: that the rewriting stops, and that it stops in the same place whichever order the rules fire in. A function defined by induction on the expression owes neither, because a function has exactly one value.

**Three folds are easy to get wrong, and each has a program that then breaks.**

*Ordering `Retyped`.* `Minus[Retyped[R, UInt32], "a"]` and `Retyped[Minus[R, "a"], UInt32]` describe the same columns. They are the two orders in which a user can drop a column and count nulls. Without the rule, one of two programs doing the same thing is rejected.

*Sorting `Both`.* `Both[A, B]` and `Both[B, A]` describe the same frame, as do `Both[Both[A, B], C]` and `Both[A, Both[B, C]]`. A union does not care about order or grouping; a node does. Flattening and sorting is how the node is made to forget a distinction its meaning never had. The sort key must be total, the same in every run, and derived only from meaning, which rules out anything built from an object's identity or from a hash set's iteration order.

*Translating a selector through an affix.* This one is not a substitution, which is the trap. Filtering on `startsWith("ab")` **after** appending `"b_"` is not filtering on `startsWith("ab")` before it. A column named `a` becomes `ab_`, which begins with `ab`, so it survives a test its original name fails. The translated selector is `startsWith("ab") or oneOf({"a"})`. In general the translation is computed, not rewritten (`D-23`), and `matching` from section 8.4 is where it lands when no named constructor describes the result. An affix that is empty renames nothing, so it must not translate at all.

**The canonical form decides type identity exactly.** Two expressions with the same canonical form denote the same frame for every caller (`T-8`), and two expressions that denote the same frame for every caller have the same canonical form (`T-10`). An effect over a union and the same effect over each operand are one form, because the form folds an effect into the operands of a union.

The direction that matters for soundness is the easy one: equal forms mean equal frames, which follows from the form preserving meaning. The other direction is what stops the system rejecting correct programs, and it holds because of how the filter slot is compared and because a union carries no effect.

**A filter is compared by the columns it picks, not by how it was written.** Consider two ways of writing one thing:

```python
Only[R, StartsWith["ab"]]
Only[Only[R, StartsWith["a"]], StartsWith["ab"]]
```

Every name beginning with `ab` begins with `a`, so the second test removes nothing the first kept, and the two keep the same columns for every caller. A comparison that looked at the shape would call them different types and reject a program that composes two correct helpers.

They are the same type here, because a name test denotes a **regular set of names**, and containment between two regular sets is decidable (`T-9`). Every constructor in section 8.4 denotes one, including `matching`, so the question always has an answer.

**Deciding it cheaply is a separate concern from deciding it.** A containment between two prefix tests is a string comparison, and every rule a user writes is settled that way. Only a test the checker derived, by translating a filter through a rename, needs the general comparison. That split keeps the common case fast.

**Substitution is what creates nesting**, and it is a third place a type is minted (`T-6`). Passing a frame whose remainder is $e$ to a function whose remainder variable is $R$ replaces $R$ with $e$ throughout the callee's result. That result must be put in canonical form like any other (`T-7`). A call can bind some variables to the known columns of a closed caller and others to an open caller's remainder. Every effect acts on each column alone (`T-24`), so the result splits into the known columns, which join the declared ones, and the substituted remainder of the open callers. Missing this one is what makes composing two correct helpers produce a nested form that compares unequal to the flattened form a signature declares.

### 8.8 Binding a split

`SplitColumns[...]` with parts $A_1, \dots, A_n$ fills the one remainder slot of section 8.1 and names parts of it. Binding is one pass, with no search. This restates `D-25`.

Each part is declared with a per-column requirement, so each has a predicate $P_i$ (`D-21`). **First match wins**: a column goes to the first part whose predicate accepts it. In closed form, part $i$ holds exactly the columns satisfying

$$
P_i \land \lnot P_1 \land \dots \land \lnot P_{i-1}
$$

so the parts are disjoint by construction and nothing needs to promise it.

A frame whose remainder is a split, passed as an argument, proves what every part proves. The parts together hold the whole remainder, so its record is the union of the parts' records: a name is absent when every part proves it absent, and a data type is possible when some part permits it.

**A split adds nothing to the grammar** (`T-11`). Each part's requirement is a selector, and the selector language is closed under `and` and `not`, so part $i$ is the ordinary expression $\mathtt{Where}(G, P_i \land \lnot P_1 \land \dots \land \lnot P_{i-1})$ over the one remainder $G$. No new node, no new absence constructor, no new closure rule, no new canonical form.

This is why section 4.5 excludes the relational requirements from a split. Binding needs a per-column predicate, and `disjoint_from=` and `within=` have none.

### 8.9 No result depends on the remainder

An operation whose output schema differed from one caller to the next would have no single type. The record for it would have to be whatever holds in every case, which is a real weakening. It also spreads, because every later operation then reads a less exact frame.

The joining rule of section 6.3 removes the only operation with that shape. A join whose result would differ between callers is an error rather than a weakened record. So every allowed operation in this specification produces one schema that is correct for every caller.

One sort of uncertainty does remain, and it is a different sort. An operation can keep a column's **name** while producing a **data type** that cannot be named. Summing a column of unknown data type is the usual case. The name is recorded in the schema and the data type is marked `Unchecked` (section 5.8, `D-27`).

The two are worth keeping apart. An unknown name weakens every later name lookup on the frame. An unknown data type weakens only the operations that read that one column. This specification permits the second and rejects the first.

### 8.10 Soundness, and where the system is knowingly incomplete

The record approximates the real column set, and must approximate in one direction only (`T-1`).

> **Every recorded fact holds under every binding.**

Losing a fact is safe: it can reject a correct program, never accept an incorrect one. Gaining a fact that does not hold is never safe. Every rule in section 8.5 records less than it can whenever the alternative is a guess.

The system is therefore **sound** (`T-14`) **but incomplete**. Four sources account for it, and this list restates section 12.1 of `OPERATIONS.md`:

**Strengthened preconditions.** Where an exact condition is not expressible, the system demands more than it needs, and a caller who could have met the exact condition is turned away. Section 8.11 is the case that shapes section 6. Diagonal concatenation likewise demands `disjoint_from=` rather than minting `Unchecked` under names not known to be present.

**Approximation at case splits.** A data type differing between branches becomes `Unchecked`, even where a reader can see which branch is taken.

**Facts relating two separately-written expressions.** Concatenating a frame with itself after both have had the same column dropped is safe, and the checker compares each side's guarantees independently.

**A wrap the effect language cannot spell.** `agg(pl.all())` wraps each column of the remainder in a `List` of its own data type. The record keeps the names and retypes to `Unchecked`, and the exact data types stay available by naming the columns.

None of the five violates section 3.2. They are the price of never accepting a program that will crash.

Anything decided by data, row count, row order, column order, a value or Python code the checker cannot read is not incompleteness. It is a boundary (`B-1` … `B-6`), outside the model, as section 3.3 says.

### 8.11 How an operation is checked

This procedure restates sections 11.1 and 11.2 of `OPERATIONS.md`. For an operation applied to a frame with schema $M$ and remainder $G$:

1. Determine the operation's precondition. This comes from the operation's real behavior, not from the type system: renaming needs its destination free, joining needs no possible rename. `OPERATION_CATALOG.md` gives it, per operation, measured against the library.
2. Decide it. The part concerning names in $M$ is decided outright, since $M$ is exact. The part concerning $G$ must be **proven** from $G$'s record (`T-12`).
3. If proven, compute the new schema and remainder by section 8.5, and put the result in canonical form by section 8.7.
4. If not proven, report an error.

Step 2 carries the subtlety that shapes section 6. **A precondition is not always a conjunction of independent per-name tests.** A join's true precondition is a disjunction. For each shared name, either the name is not shared after all, or its suffixed form is free. A record of absence rules cannot decide a disjunction of that shape over names it cannot enumerate.

This specification does not approximate that disjunction. It demands the stronger condition of section 6.3, that no rename can happen at all. That condition **is** a conjunction of per-name tests, so step 2 decides it exactly. A join that fails it is reported, not guessed at. Section 6.9 states what this costs the user and why the cost is worth paying.

______________________________________________________________________

## 9. Writing it down

A remainder is declared once, by name, and then used. There is one form:

```python
Anything = other_columns()
NoTicker = other_columns(without=["ticker"])
Numbers  = other_columns(without=["ticker"], all_of=NumericType)

def f(df: TypedDataFrame[Trades, NoTicker]) -> ...
```

Every requirement is a keyword. Adding one does not change the shape of the declaration or of any signature that uses it, and a signature never grows as requirements accumulate.

Naming the requirement is the point. `NoTicker` says what it means at every use site, it can be shared by a group of related functions, and when the requirement changes it changes in one place. A requirement is a value, so it behaves like one.

**Requirements are calls; effects are types.** A requirement is a thing you declare and reuse, so it is written as a call. An effect describes the shape of a return value, so it has to be written where a type goes, in square brackets:

```python
def add_notional(
    df: TypedDataFrame[Trades, Anything],
) -> TypedDataFrame[TradesWithNotional, Minus[Anything, Literal["notional"]]]: ...
```

`Literal["notional"]` is how a specific string is written in a type position. It appears in effects and never in requirements, because a requirement is a call and can take a plain string.

A return type can also leave its remainder to the body, as an inferred remainder. `InferredRemainder` in the remainder slot keeps the declared columns as a contract, and takes the remainder from each value that the body returns through `infer(...)`:

```python
def add_total(
    df: TypedDataFrame[Trades, Anything],
) -> TypedDataFrame[TradesWithTotal, InferredRemainder]:
    return infer(df.with_columns((pl.col("price") * pl.col("quantity")).alias("total")))
```

Each returned value must hold every column of `TradesWithTotal`. At a call, the result is the union of the values, each written in the caller's own terms: a closed caller gets closed frames, and an open caller gets its own remainder with the edits of the body (`D-43`).

Schemas are declared as classes throughout:

```python
class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64
```

Every column is required and the set is closed. A schema class means exactly these columns and no others, which is what makes the remainder the only source of openness.

______________________________________________________________________

## 10. Worked example

A pipeline that reads as ordinary code, with each signature explained.

```python
class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


class TradesWithNotional(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64
    notional: pl.Float64


class SectorRef(Columns):
    symbol: pl.String
    sector: pl.String


class Enriched(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64
    notional: pl.Float64
    sector: pl.String
```

**Step one: add a column. No requirement.**

```python
Anything = other_columns()


def add_notional(
    df: TypedDataFrame[Trades, Anything],
) -> TypedDataFrame[TradesWithNotional, Minus[Anything, Literal["notional"]]]:
    return df.with_columns(notional=pl.col("price") * pl.col("quantity"))
```

Any caller can use this. If their frame already had a `notional`, it is replaced, and the effect records that it is no longer in the remainder.

**Step two: join. One requirement.**

```python
NoSector = other_columns(without=["sector"])


def add_sector(
    df: TypedDataFrame[TradesWithNotional, NoSector],
    ref: TypedDataFrame[SectorRef],
) -> TypedDataFrame[Enriched, NoSector]:
    return df.join(ref, on="symbol")
```

The caller must promise that its remainder holds no `sector` column, because otherwise the join would rename the reference table's `sector` and the result would depend on the caller. With the promise, nothing collides and the remainder passes through untouched. Section 6 covers the other join shapes and what each one costs the caller.

**Step three: compose them.**

```python
def enrich(
    df: TypedDataFrame[Trades, NoSector],
) -> TypedDataFrame[Enriched, Minus[NoSector, Literal["notional"]]]:
    return add_sector(add_notional(df))
```

The requirement propagates outward: `enrich` needs whatever `add_sector` needs. The effect propagates too: `add_notional` removed `notional`, and nothing after it put anything back.

**Step four: call it.**

```python
class Blotter(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64
    venue: pl.String
    trader: pl.String


blotter: TypedDataFrame[Blotter]
result = enrich(blotter)
```

`NoSector` binds to the columns `venue : String` and `trader : String`. That satisfies the requirement, since neither is called `sector`. The result is a closed frame:

```
symbol, price, quantity, notional, sector, venue, trader
```

The caller's two other columns survived the whole pipeline, and every step in between was checked without knowing what they were.

**And a call that is correctly rejected:**

```python
class BlotterWithSector(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64
    sector: pl.String


blotter_with_sector: TypedDataFrame[BlotterWithSector]
enrich(blotter_with_sector)  # error: NoSector must not contain column "sector"
```

______________________________________________________________________

## 11. Limits

Operations this specification cannot type, and why. The first is a strengthened precondition that a user can lift. The last is typed in part. The others are boundaries: outside the model, not omissions.

**A join that can rename a column.** The result differs from caller to caller, so no single type describes it. This is the one limit in this list that a user can lift, by making one of the promises of section 6.

**Stacking frames vertically** (`B-4`). Appending one frame below another requires the two schemas to match exactly, including column order. A column set has no order (`D-5`), deliberately, so that two frames with the same columns written in a different order are the same type. That choice makes column order unavailable, and so makes this operation untypeable over a remainder. Over two closed frames the checker compares the columns by name only, with no order: `pl.concat` then gives the plain frame, and `.vstack`, `.extend` and `.merge_sorted` keep the receiver's type. A frame whose columns stand in another order at run time can still make polars raise there, and the type cannot show it.

**Transposing** (`B-2`). The output has one column per input row. Nothing in a schema knows how many rows there are.

**Pivoting without `on_columns=`** (`B-5`). The output column names come from the *values* in a column, not from any schema, so no type names them. A literal `on_columns=` names them instead, and that form is typed.

**Selecting by position** (`B-4`). Taking the first or last column requires knowing the order, which schemas do not record.

**Summing an unknown column.** Some operations map input data types to output data types in a way that is neither constant nor identity, so with an unknown input data type the output data type cannot be named. Where the *name* survives such an operation, the result records the name and marks the data type `Unchecked`, rather than discarding the frame.

______________________________________________________________________

## 12. Summary of the grammar

**Requirements**, written where a remainder is introduced. Each of the eight after the empty one tests one column on its own, which is what lets it divide a remainder into parts:

| Written | Means |
| -- | -- |
| `other_columns()` | anything |
| `other_columns(without=["a", "b"])` | holds neither `a` nor `b` |
| `other_columns(only=["a", "b"])` | holds nothing but `a` and `b` |
| `other_columns(without_prefix="p_")` | holds no name beginning with `p_` |
| `other_columns(only_prefix="p_")` | every name begins with `p_` |
| `other_columns(without_suffix="_p")` | holds no name ending in `_p` |
| `other_columns(only_suffix="_p")` | every name ends in `_p` |
| `other_columns(all_of=S)` | every column's data type is in the set `S` |
| `other_columns(none_of=S)` | no column's data type is in the set `S` |

The eight are four matched pairs: a name list, a prefix, a suffix and a set of data types, each in a negative and a positive form. `S` is any set of data types: a polars family, a single data type, or several written together, as section 4.4 describes.

Two more describe how **two** remainders relate. Neither tests a single column, so neither can divide a remainder into parts:

| Written | Means |
| -- | -- |
| `other_columns(disjoint_from=L)` | shares no name with `L` |
| `other_columns(disjoint_from={A, B})` | shares no name with `A` or with `B` |
| `other_columns(within=L)` | every name is also a name of `L` |
| `other_columns(within={A, B})` | every name is a name of `A` or of `B` |

**One binding form**, filling the single remainder slot:

| Written | Means |
| -- | -- |
| `SplitColumns[A, B, ...]` | the remainder divides into these parts, first match winning |

A split is not an effect and not a second remainder. It names parts of the one remainder a frame has, and section 8.8 gives the rule that binds it.

**Effects**, written in a return type:

| Written | Means |
| -- | -- |
| `Rest` | unchanged |
| *(omitted)* | emptied |
| `Minus[Rest, Literal["a"]]` | without `a` |
| `Both[A, B]` | `A` and `B` together |
| `Suffixed[Rest, Literal["s"]]` | same columns, every name with `s` appended |
| `Prefixed[Literal["p"], Rest]` | same columns, every name with `p` in front |
| `Only[Rest, C]` | only the columns the selector `C` picks |
| `Except[Rest, C]` | only the columns `C` does not pick |
| `Retyped[Rest, U]` | same names, every data type replaced by `U` |
| `InferredRemainder` | the remainder of each value that the body returns through `infer(...)`, in the caller's terms (`D-43`) |

A selector `C` is a name list, a prefix, a suffix, a substring, a general expression, or a set of data types, a family being one such set, matching what polars offers. Only the last is not a test on the name, and section 8.5 gives its rule separately.

`Minus`, `Only` and `Except` are three spellings of one effect: keep the columns a selector picks. `Minus[R, N]` is `Except[R, OneOf[N]]`, kept as the short spelling for a list of names. So there are seven effects written in nine ways.

No effect appears in the requirements table, and no requirement appears in a return type. A requirement is written where a remainder is declared, an effect where a result is described, and neither ever carries the other. Section 4.5 explains why relational requirements belong on the declaration even though they name two remainders at once.

Everything in this specification follows from the every-caller rule in section 3.2: an operation is allowed exactly when it works for every possible remainder. A requirement narrows what "every" means. An effect describes what comes out.

______________________________________________________________________

## 13. Glossary

Every term this document uses with a technical meaning, and where it is defined. A term not in this list does not appear in this document with a special meaning.

The last column names the identifier in `OPERATIONS.md` that defines the term formally. Where no single identifier defines it, the column names the section of `OPERATIONS.md` that does, or says `measured` for a fact a test measures.

| Term | Means | Here | Formally |
| -- | -- | -- | -- |
| absence rule | a predicate on names saying which are provably absent | 8.4 | `D-9` |
| binding | one column set given to each remainder variable | 8.2 | `D-19` |
| canonical form | the one shape every remainder expression is written in | 8.7 | `D-22` |
| closed frame | a frame whose type has no remainder | 3.1 | `D-8` |
| column | a name together with a data type | 3.1 | `D-4` |
| column set | a finite partial map from names to data types | 8.2 | `D-5` |
| data type | one of polars' types, such as `Int64` or `String` | 1 | `D-2` |
| effect | what an operation does to a remainder, written in a return type | 5 | section 6 |
| every-caller rule | an operation is allowed exactly when it works for every remainder | 3.2 | `D-19` |
| family | a named group of data types, such as every numeric type | 4.4 | `D-3` |
| frame type | a schema and at most one remainder | 8.1 | `D-8` |
| inferred remainder | a return remainder that each value of the body supplies, through `infer(...)` | 9 | `D-43` |
| name expression | a general name test, carried as the machine deciding it | 8.4 | `D-11` |
| open frame | a frame whose type has a remainder | 3.1 | `D-8` |
| part | one named piece of a split remainder | 4.6 | `D-24` |
| record | what is known about a remainder: an absence rule and a type set | 8.3 | `D-16` |
| regular set | a set of names a finite automaton decides, which every rule here denotes | 8.4 | `D-11` |
| remainder | a stand-in for the columns the schema does not name | 3.1 | `D-7` |
| remainder expression | a variable, an effect applied to a remainder expression, or two remainder expressions combined by `Both` | 8.2 | `D-18` |
| requirement | a promise the caller makes, written at a declaration | 4 | section 5 |
| schema | a column set the program fully knows, written as a class | 3.1 | `D-6` |
| selector | a test on a whole column, name and data type together | 5.6 | `D-12` |
| split | a division of the one remainder into named parts | 4.6 | `D-24` |
| surviving-left rule | a collision is read against the left columns a join keeps | 6.4 | measured |
| type set | which data types a remainder's columns can have | 8.3 | `D-15` |
| `Unchecked` | a data type meaning "the name is known, the data type is not checked"; one marker for frames and expressions | 5.8 | `D-27` |
