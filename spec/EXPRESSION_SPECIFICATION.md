# Typed expressions

This document answers what a typed expression is, and why, in prose with examples: what `pl.col("price") * 2` is to a type checker, and how a program says so. Read `DATAFRAME_SPECIFICATION.md` before it. `OPERATIONS.md` gives the formal definitions, rules and proofs, and it is authoritative: where this document restates one of its definitions, it cites the identifier, and if the two disagree, `OPERATIONS.md` is right.

## 1. Abstract

A polars expression computes columns from columns. `pl.col("price") * 2` reads one column and produces one, `pl.when(...).then(...)` chooses between several, and `cs.numeric().round(2)` transforms every column a selector picks. Expressions are the arguments of `select`, `with_columns` and `agg`, and they are how every derived column in a polars program is written.

This document gives expressions types. The type of an expression is its **code** (`D-33`): the expression, written with the public API of polars, exactly as the user wrote it. The type stores nothing about how polars computes. When the expression meets a frame, the checker rebuilds polars' own tree of the expression and evaluates it node by node, with the rules that polars declares for each node (`D-36`, `D-37`), so the data type it gives a column is the one polars gives (`T-25`). Where an expression crosses a function boundary, a **declaration** states what the boundary promises, and the checker evaluates each argument against it (`D-39`, `D-40`).

The companion `DATAFRAME_SPECIFICATION.md` types the frames these expressions run against, and this document reuses its machinery where the two meet, cited by identifier. The registry of identifiers, `D-n` for definitions, `T-n` for theorems, `M-n` for measurements and `B-n` for boundaries, is `OPERATIONS.md`'s. Every measured claim here names the test that measures it, under `spec/tests/`.

______________________________________________________________________

## 2. Motivation

Write one expression and hand it to three frames:

```python
margin = (pl.col("price") - pl.col("cost")).alias("margin")

frame_a: price : Float64, cost : Float64    ->  margin : Float64
frame_b: price : Datetime, cost : Datetime  ->  margin : Duration
frame_c: price : String, cost : Int64       ->  error: no such subtraction
```

Three frames, three answers, from the one expression. Any account of "the type of `margin`" has to carry all three: the output data type is a *function* of the input data types, and the error is one of the function's values. That function is what this document calls the expression's **map** (`D-33`).

The account also has to respect where polars itself draws the line. *Building* an expression never fails: `margin` above constructs happily, and so does an expression no frame could ever run. Failure happens at evaluation, when an actual frame supplies actual data types. A faithful type system checks in the same place, and never more strictly than the library it types. The rules that compute the map follow the source of polars, one mechanism at a time, and the suite measures them against polars at every kind of input (`M-21`), so the answers in this document are polars' own conduct, not a stricter house rule. Where a program wants strictness beyond polars, section 7 gives it ways to ask, and it is never imposed.

Two audiences read an expression's type. A frame operation reads it to answer "what schema comes out?". A function signature reads it to answer "which expressions do I accept?". The same object serves both, which is why this document spends its second half on function boundaries: reusable, checked expression logic is the payoff of typing expressions at all.

One name is worth defusing early. `DATAFRAME_SPECIFICATION.md` has a *remainder expression* (`D-18`): a description of the remainder, built from effects. An *expression* here is a computation a user writes, `pl.col("a") + 1`. The two share a word and nothing else.

______________________________________________________________________

## 3. What an expression is

Start from three expressions a user actually writes, over the running `Trades` frame of `DATAFRAME_SPECIFICATION.md` (`symbol : String`, `price : Float64`, `quantity : Int64`):

```python
trades.select(pl.col("price") * 1.1)
# price : Float64, one value per input row

trades.select(pl.col("price").sum())
# price : Float64, one value for the whole frame

trades.select(pl.lit(0.25))
# literal : Float64, one value, from no input at all
```

Three expressions, three relationships between input rows and output values. Polars tracks that relationship for every node of an expression as its **row form** (`D-29`): one value per row, one value, one row, or another length. A declaration states it more coarsely, as a **character**, and the three characters are the three expressions above.

**Elementwise** is the everyday case: a transformation applied to every row. `pl.col("price") * 1.1` produces exactly one output value for each input value, so the output column has the input column's length. Arithmetic, comparisons, `.round()`, `.abs()`, the string methods: most of what a user writes is elementwise.

**Aggregating** collapses a column into one value. `.sum()` reads every row and answers with a single number: one for the whole frame in a `select`, one per group under a `group_by`. The reducers live here: `sum`, `mean`, `min`, `max`, `count`, `quantile`, and their kin.

**Literal** brings a value in from outside instead of reading rows. `pl.lit(0.25)` mentions no column. Its one value is fixed by the code itself, and it broadcasts to whatever length the site needs.

Why the checker cares: put the same three expressions inside a `group_by`, and watch the character decide three different outcomes:

```python
trades.group_by("symbol").agg(pl.col("price") * 1.1)
# symbol : String, price : List(Float64)   <- imploded into a list!

trades.group_by("symbol").agg(pl.col("price").sum())
# symbol : String, price : Float64         <- one sum per group

trades.group_by("symbol").agg(pl.lit(0.25))
# symbol : String, literal : Float64       <- one value, unwrapped
```

The same expression answered differently in the two places, so the place is part of the story. The site an expression runs in is its **evaluation context** (`D-28`), and there are exactly two: the *row* context of `select`, `with_columns` and `filter`, and the *aggregation* context of `.agg(...)`. One rule relates them (`T-15`): **the aggregation context implodes a result of one value per row into a `List`, and leaves one value alone.**

The row form composes from node to node, so the checker always knows it:

```python
pl.col("price").sum().abs()  # still one value: abs of one value
pl.col("price").sum().over("symbol")
# one value per row again: broadcast per row
pl.col("price") - pl.col("price").mean()
# one value per row: the mean broadcasts
```

The rule is not the whole difference between the contexts. In the aggregation context polars runs each node in each group, and some nodes run in another way there:

```python
trades.select(pl.col("symbol").mean())
# error: mean does not work on String

trades.group_by("quantity").agg(pl.col("symbol").mean())
# runs, and gives quantity : Int64, symbol : String
```

The checker evaluates the aggregation context as polars runs it, node by node, over two groups of different sizes (`D-36`). So such a difference needs no list of exceptions: it is what the rules of the node give in each group, and a later step reads it, as `pl.col("symbol").mean().rank("dense")` inside an `agg` does.

______________________________________________________________________

## 4. The type of an expression is its code

What is the type of `(pl.col("price") - pl.col("cost")).alias("margin")`?

A frame receiving this expression cares which columns it reads, and what column comes back for each assignment of data types to those columns. A smaller summary of the expression would serve that question for most expressions. It fails for some, because polars types a node from facts of its inputs that no summary of the data types keeps. The cast pass of polars visits a node before its inputs and again after them, so the place of a node in the tree changes its answer:

```python
# n : Null, dec : Decimal(10, 2), i8 : Int8
(pl.col("n") / pl.col("dec")) + pl.col("i8")
# at the root of a select: Decimal(38, 2)
((pl.col("n") / pl.col("dec")) + pl.col("i8")).abs()
# under .abs(): Int8
```

The inner sum is the same, and its answer differs with what stands above it. So the type of an expression is the expression itself, as **code** (`D-33`):

```python
notional = pl.col("price") * pl.col("qty")
reveal_type(notional)
# TypedExpr[Deferred["pl.col('price') * pl.col('qty')"]]
```

This is a **deferred** type: it computes nothing until the expression meets a frame or a declaration. The code is written with the public API of polars, and the checker writes one canonical text, so `pl.col("a")` and `pl.col('a')` give one code. A part the code uses more than once is written once, as `($1 := <code>)`, and as `$1` after that, so a variable that an expression uses twice does not double its code:

```python
e = pl.col("a") + pl.col("b")
reveal_type(e * e)
# TypedExpr[Deferred["($1 := pl.col('a') + pl.col('b')) * $1"]]
```

A parameter of a function with a declared type (section 7) is a **declared leaf** of the code, written `$0`, and its declaration rides in the type beside the code:

```python
Price = column("price", pl.Float64)


def discounted(price: Price) -> InferredExpr:
    return infer(price * 0.9)


reveal_type(discounted(pl.col("price")))
# TypedExpr[Deferred["$0 * 0.9", TypedColumn['price', Float64]]]
```

The call keeps the leaf and does not copy its argument into the code: an argument that fits the declaration cannot be told apart from another argument that fits it, so the result of the call does not depend on which one it got (`T-23`).

What a type means is its **map**: the columns it reads, and its answer at each frame. So `a + b` and `b + a` have one map, although their code differs, and the same goes for `pl.col("price") - pl.col("cost")` and its `.abs()`: the two compute different *values*, and no frame sees a different column type from them. The host checker compares types by their spelling, so the checker decides each comparison that matters on the map, by the fit of section 7, and never by the spelling.

Two parts of the answer do not depend on input data types, and polars computes both from the tree. The **reads** (`D-31`) are the names of every column the tree reads, the condition of a `when`, the mask of a `filter`, the key of a `sort_by` and the partition of an `over` included, because every input of a node runs:

```python
e = pl.when(pl.col("quantity") > 0).then(pl.col("price")).otherwise(0.0)
e.meta.root_names()  # ['price', 'quantity']  <- the condition counts
e.meta.output_name()  # 'price'                <- but does not name it
```

The **output name** (`D-32`) follows rules a user can predict:

```python
pl.col("price") * 2  # price: the first column names it
pl.col("price").sum()  # price: a reducer keeps the name
(...).alias("margin")  # margin: an alias overrides everything
pl.when(c).then(pl.col("price")).otherwise(0.0)
# price: a chain takes its first branch
pl.lit(0.25)  # literal: the fixed name of a literal
```

______________________________________________________________________

## 5. How the checker reads the code

**When is the code read?** Never while the expression is being built. A column reference, `pl.col("price")`, is written exactly as polars has it, and this specification adds no second spelling: the code rides along, through every operation chained on top, and the frame that finally receives the expression supplies the data types to read it at. That place and that verdict are polars' own.

**What the reading does.** At a frame, the checker rebuilds the expression from its code, with a reader that executes no text: it parses the code and calls the real methods of polars on placeholder inputs, so polars builds its own tree and runs no data. Then the checker evaluates the tree, node by node, from the leaves up, with the rules of polars (`D-36`). A node of the tree applies one operation to its **inputs** (`D-34`), and every sub-expression of a call is an input: in `pl.col("x").clip(pl.col("lo"), pl.col("hi"))` the three columns are three inputs of one node.

Polars reads a tree in three passes, and the checker repeats them. The planner computes a type for each node, the cast pass changes the tree so that the inputs of each node agree, and the run computes the data. Each node therefore has a **planner type** and a **run type**, and the two can differ:

```python
pl.col("d").cum_sum()  # d : Decimal(10, 2)
# before polars 2.0.0: planned as Decimal(10, 2), runs as Decimal(38, 2)
```

From polars 2.0.0, the planner gives `Decimal(38, 2)` too, and `pl.col("i").shift(1, fill_value=pl.col("j").first())` over `i : Int8` and `j : Int16` shows the difference: it is planned as `Int8` and runs as `Int16` in every release.

A frame operation runs its expression, so a frame column takes the run type. A lazy frame runs nothing until `collect()`, and polars plans each later step at the planner type that it stores for the column. Where the two types differ, a column of a lazy frame holds both, as `Planned[run, planner]` (`D-45`):

```python
lf.with_columns(d=pl.col("d").cum_sum())
# before polars 2.0.0, d : Planned[Decimal(38, 2), Decimal(10, 2)]: a later step plans d at Decimal(10, 2)
```

**Where the rules come from.** Polars does not type each operation with code of its own. It declares the rules of each operation in a small vocabulary: the supertype of two data types, the casts, the options that say how a node runs, a type function, the rules of the cast pass, and a kernel. The checker translates that vocabulary once, and each node's definition names the parts it uses (`D-37`). A rule covers every input: every data type, every parameter, every literal value, any depth of nesting and any number of inputs. A behavior that follows no logic, such as a bug of polars, is a measured fact of the registry, with its reason. Where no rule and no fact gives an answer, the answer is `Unchecked`, so a guess never becomes a type.

Two facts make the evaluation correct. Each definition gives the answers of polars on the samples its tests measure (`M-21`), and polars types a node only from the states of its inputs (`M-20`). From these, the evaluator gives the answer of polars on the whole tree (`T-25`, proved).

**Literals keep their value.** A Python value in an expression is a literal input, and polars reads its value:

```python
pl.col("u") + 1  # u : UInt8, gives UInt8
pl.col("u") + 300  # u : UInt8, gives UInt16: 300 needs 16 bits
```

The code holds the value as written, so the checker reads it as polars does.

**Values the checker cannot know.** A value in a variable of a plain type is a **hole** (`D-34`, `D-44`): the code keeps its Python type and not its value. The checker evaluates the code at a fixed sample of values of that type, and keeps the answer where every value of the sample gives one:

```python
def shifted(n: int) -> None:
    trades.select(pl.col("price").shift(n))
    # price : Float64: the count never changes a data type

    sizes.select(pl.col("u") + n)
    # u : Unchecked: 1 keeps UInt8, and 2 ** 15 widens it
```

A column whose data type the frame does not fully declare is read the same way: `Unchecked`, a family such as `NumericType`, or a bare `pl.Datetime` stands for a sample of the data types it allows, and the answer is what the answers of the sample share (`T-26`). A choice where polars refuses drops out of the sample, so the reading errs only where every choice refuses (section 9).

**How errors land.** Where polars refuses an expression at a frame, the checker reports the error at that frame operation, where polars itself fails, and names the operation. A run that raises only for some values, such as a cast of a string that does not parse, is a fact of the data and not of the types, and the checker accepts it (`B-1`).

______________________________________________________________________

## 6. Building an expression in a loop

Expressions are built dynamically, and the motivating shape is a rule table of length unknown until run time:

```python
e = pl.when(rules[0].cond).then(rules[0].value)
for rule in rules[1:]:
    e = e.when(rule.cond).then(rule.value)
result = e.otherwise(fallback)
```

What does the checker infer for `e`? The variable is assigned twice, once before the loop and once inside it, so the answer depends on how mypy treats a rebound variable, and that is measured behavior, not something this system chooses (`M-6`). With `allow_redefinition` enabled, mypy analyzes the loop body again for as long as the type of `e` changes, within a fixed budget, and gives `e` the union of the types it saw. Every pass appends to the code of `e`, so the type changes on every pass and the budget ends the analysis: `e` holds the versions of the first few passes, and not the version of a long run.

The checker reads such a union member by member (`D-42`). A union of two members, as an `if`/`else` gives, gives one frame for each member at `select`, `with_columns` and `agg`. A loop leaves more members, and a later pass can give an answer that none of them gives, so there the answers merge. Where every member gives one output name and one data type at a frame, that is the output. Where the data types differ, the output keeps its name under `Unchecked`, and the gate reports the loss. So a loop whose versions agree types exactly, and a loop whose versions do not degrades to `Unchecked` under its name. The first case rests on an assumption: that versions which agree over the first passes agree over every later pass (`T-18`).

A loop whose type genuinely depends on the number of passes, such as one that nests a column deeper on every pass, can say so with a declaration on its seed:

```python
Nested = typed_expr(column("scores", pl.Int64), gives=Unchecked)

e: Nested = pl.col("scores")
for _ in range(depth):
    e = e.implode()  # each pass fits Nested, so e keeps one type

trades_with_scores.select(e)  # scores : Unchecked
```

Each assignment is checked against the declaration, and an argument whose answer the declaration leaves `Unchecked` fits it (`D-40`). So `e` keeps the one declared type through every pass, and the surrender of the data type is explicit, visible, and local to one column.

______________________________________________________________________

## 7. Writing it down

Everything so far was inferred, and the one annotation was the seed of section 6. Types are written on function boundaries, where Python programs already write them. This section is a walk through complete functions, from the fully spelled to the broadest, with every type written out at every step.

**Declarations are named values.** A slice declaration is written the way a remainder's requirements are written in `DATAFRAME_SPECIFICATION.md`: as a call, assigned to a name, and the name is what appears in signatures. The shape is forced, not chosen: the checker refuses a call expression in annotation position (measured by `TestAnnotationSurface` in `spec/tests/test_mypy_binder.py`). The call's arguments carry the column names as plain strings or keywords, so a declaration never spells `Literal[...]`; the expanded type it registers does, and so does a signature that writes an expanded type or a remainder effect directly, as the `flagged` example below does:

```python
Price = column("price", pl.Float64)  # one column, read and given
# back under its own name

Quantity = column("quantity", pl.Int64)

BuyFlag = typed_expr(  # one area of a declared map
    Quantity,  # the reads, as values
    gives=pl.Boolean,  # the outcome at those cells
    named="is_buy",  # the output name
)


def buy_flag(quantity: Quantity) -> BuyFlag: ...  # names in signatures
```

Naming the declaration is worth the same moment it was worth for remainders: a declaration is a value, declared once, reused by every function that shares the contract, and its name says what it means. The bracket forms, `TypedExpr[Reads[...], Column[Literal[...], ...]]` and its kin, remain the *expanded* types the checker shows and compares, and a declaration is the way they are written.

**The declaration grammar.** Every declaration answers four questions: *which columns* are read, *which cells* are covered there, *what outcome* each covered area has, and *how it counts rows*. The whole surface is one grammar, and this passage restates `D-39`:

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
outcome     ::= dtype | same | same(name) | wrap(C, outcome)
name        ::= a plain string
```

Reads are positional; only the surface's own keywords, `gives=` and `named=`, are keywords, so a column named `gives` collides with nothing and the surface can grow without colliding with anyone's data. A read built once is a value, reused anywhere a read goes:

```python
Quantity = column("quantity", pl.Int64)

BigTrade = typed_expr(Quantity, gives=pl.Boolean, named="is_big")


def big_trade(quantity: Quantity) -> BigTrade: ...
```

The `column_of(...)` and `subset_of(...)` spellings, each a schema reference, anchor a read to a schema class, so neither the name's spelling nor its data type is repeated:

```python
Legs = typed_expr(subset_of(Quotes, "ask", "bid"), gives=pl.Float64, named="spread")
```

These are type-level only. `column_of(Trades, "price")` appears inside declarations and nowhere else; it is not an expression, it never reaches polars, and section 5's commitment stands: expression code is plain polars, always. They are calls, not attributes of the schema class, because a `Columns` class is a `TypedDict`, which carries no class attribute at run time or under the checker.

A **region** is a declared set of kinds (`D-39`), the cells a body is checked at. A declaration is a **declared map**: for each read a region, for each **cell**, one kind per read, an outcome, and the output name and the character. Each production maps onto one piece of it, as the production table of `D-39` states, and the last two rows below add the region's two ends from `D-41`:

| Production | Means |
| -- | -- |
| `column(n, r)` alone | reads `{n}` over `r`, the outcome `same`, name `n`, elementwise |
| `typed_expr` / `typed_agg` / `typed_lit` | the character, nothing else |
| `column(n, r)` as a read | one read, restricted to the region's kinds |
| `column_of(Schema, n)`, `subset_of(Schema, ...)` | reads at the schema's declared data types |
| a schema class | one read per declared column, each a single data type |
| `some_column(r)` | one read under a name blank the call fills |
| `region` | a set of kinds: `D-3`'s vocabulary plus its two ends (`D-41`); a bare constructor, `pl.List`, names its kind, the frame side's `none_of={pl.List}` spelling |
| `gives=T` | the outcome `T` at every declared cell |
| `gives=same` | that column's own data type back |
| `gives=wrap(C, o)` | the constructor `C` around the outcome `o` |
| `named=n` | the output name, a string literal that is not empty |
| `row(...)` | one more area of the same map |
| `typed_selection`, `typed_agg_selection` | `D-38`'s type: the rows partition the matched column's own type, one implicit read, and the second form is the aggregating character (section 8) |
| `EveryDType` | the region of every kind: the body must answer at each of them |
| `Unchecked` | no region at all: the marker of section 9 |

**Defaults, so the common case stays short:**

- `named=` defaults to the sole read's name, blank included; with two or more reads it is required, so no declaration inherits polars' leftmost-root convention silently.
- `typed_lit` is named `literal` unless `named=` says otherwise, matching `pl.lit` itself.
- `column(n, r)` alone is the one all-defaults form: `gives=same`, `named=n`, elementwise.

**Well-formedness, checked at the declaration itself:**

1. `typed_expr` reads at least one column. `typed_agg` with no reads is the rootless aggregation, `pl.len()`'s shape: it counts rows instead of reading columns, so `named=` is required and `gives=` is a data type. A declaration reading none and counting no rows is a literal and is written `typed_lit`:

   ```python
   RowCount = typed_agg(gives=pl.UInt32, named="len")


   def row_count() -> RowCount:
       return pl.len()
   ```

2. Every `row(...)` of one declaration reads the *same column names*: rows partition the type space of one map, never the column set. Regions can differ per row; the names cannot.

3. Rows are disjoint: two rows claiming one cell is an error at the declaration, not later at some call.

4. `same` with no argument needs exactly one read; `same(n)` names one of the reads.

5. At most one `some_column` per declaration. The blank is a binder: an argument fits it when it reads exactly one column and gives it back under that name, the call binds the name, and a blank never reaches a frame, because `D-27` refuses a nameless column. A signature using one declaration for its parameter and its return shares the blank, so the output name is the input name, per call. An argument that renames does not fit; the caller renames the result instead.

6. A region is never empty, and no column is read twice across the positional reads.

7. `typed_selection`'s rename takes only the two admissible affixes, `suffix=` or `prefix=` (`T-3`); its selector is written with the ordinary `cs.*` constructors.

8. A blank read stands alone: no named read sits beside it, and no explicit `named=` renames it, because the binding gives the blank's own name back.

The outcome `same` over an `Unchecked` read evaluates by the lifting: the read pins no data type, so `same` there gives `Unchecked` back, and a `wrap` there wraps every kind at once, which collapses to `List(Unchecked)`, the set of every list. `column(n, Unchecked)` therefore declares exactly what it reads as: the column given back, its data type still `Unchecked`.

Two of the rules deserve their half-sentence of why. Rule 2 holds because a declared map has one set of reads, and a "row" that read different columns would be a different map wearing the same name. And rule 3 makes a declaration a *partition*, so what a contract says at a cell is never an accident of rule order. With several reads, a row's cells are the product of its reads' regions, one kind per read, and disjointness is over these tuples.

Broad areas compose from these pieces exactly as promised. `Total`, in the aggregating example below, writes `sum`'s measured behavior over the numeric kinds as one declaration of two rows.

**A first function, fully spelled.** Take the price column, return it discounted:

```python
Price = column("price", pl.Float64)


def discounted(price: Price) -> Price:
    return price * 0.9
```

Read the annotations first. The parameter says: I accept an expression that reads the one column `price : Float64` and gives it back under its own name, elementwise. The return says: I produce the same interface. `column(n, d)` registers `TypedColumn[Literal[n], d]`, the identity collapsed into one form:

```python
TypedColumn[Literal["price"], pl.Float64]
# the collapsed form of
# TypedExpr[
#     Reads[Column[Literal["price"], pl.Float64]],   # what it reads
#     Column[Literal["price"], pl.Float64],          # what it gives
# ]
```

That form is what error messages and revealed types show; the declaration is what a person writes.

Now the body, one step at a time:

```python
price  # TypedColumn[Literal["price"], pl.Float64]
# reads {price}, gives price : Float64, elementwise
price * 0.9  # reads unchanged: a literal reads no column
# gives price : Float64, the answer of the multiplication
# elementwise * literal = elementwise
```

The body's inferred type equals the declared return, so the function checks. The caller's side:

```python
discounted(pl.col("price"))
# fits: the argument, evaluated at the declared cell Float64,
# gives Float64 back, exactly as declared

trades.select(discounted(pl.col("price")))
# TypedDataFrame[TypedDict({'price': Float64}, closed=True), None]

sizes.select(discounted(pl.col("price")))  # sizes: price : Int32
# error, at the select: the slice covers Float64, and this frame
# supplies Int32

discounted(pl.col("quantity"))
# error, at the call: gives quantity, where price is declared

discounted(pl.col("price").cast(pl.String))
# error, at the call: the cast gives String at every cell,
# where Float64 is declared
```

Watch where each verdict lands, because the timing is the system's one rule about it: **everything is checked as soon as it is determined, and nothing later than the frame.** The output name and the character are part of the argument, so they are judged at the call. The argument's *code* is a complete object too, so the checker evaluates it at the declared cell right at the call: the column read agrees at `Float64`, the cast does not, and both verdicts are immediate. What no one knows at the call is which *frame* the result will meet, so the one remaining obligation, that the frame really supplies `price : Float64`, travels with the result and is settled at the `select`, which is exactly where polars would have settled it. The obligation travels inside a returned code too: a function `def bumped(price: Price) -> InferredExpr` returns code that reads `price` through its declaration, and a `select` over a frame whose `price` is `Int64` is an error, because an argument that fits `Price` can give anything at `Int64`.

**A function whose output is not its input.** The gives can change both name and data type while the reads stay put. `BuyFlag` and `buy_flag` are the ones declared at the top of this section:

```python
def buy_flag(quantity: Quantity) -> BuyFlag:
    return (quantity > 0).alias("is_buy")
```

The body again, step by step:

```python
quantity  # reads {quantity}
# gives quantity : Int64, elementwise
quantity > 0  # reads unchanged
# gives quantity : Boolean, a comparison's answer
(...).alias("is_buy")
# reads unchanged: alias reads nothing
# gives is_buy : Boolean
```

Note what the return declaration had to say that `column(...)` could not: the column given is not the column read, so the reads, the outcome and the new name each get their own argument. One area of a declared map, spelled in full.

**Why does `Reads` state a data type at all?** Because the gives is only a promise *at* it. `is_buy : Boolean` is the comparison's answer at the cell `Int64`. A declared type covers a slice of the map, and `Reads` is where the slice's cells are named. Three places consume that cell.

A frame evaluating the returned expression must supply `quantity : Int64`, exactly: a frame whose `quantity` is `List(Int64)` would make the comparison raise at run time, and the declared read's data type is what lets the checker refuse it. Inside the function, the body is checked *assuming* the declared reads: the `Int64` flowing from the parameter into `quantity > 0` is what gives the body a type at all. And at every call, the argument is evaluated exactly there, which is how a disagreeing cast is refused on the spot. Written with names alone, `Reads` would name a slice with no cells in it: no cell to check the body at, and nothing to hold a caller's frame to.

The pinned `Int64` is stricter than a comparison strictly needs, and that is the system's standing trade: exact types, explicit generalization. A flag meant for every numeric column widens its region instead, one word where a single cell stood:

```python
NumericQuantity = column("quantity", NumericType)
NumericFlag = typed_expr(NumericQuantity, gives=pl.Boolean, named="is_buy")


def numeric_buy_flag(quantity: NumericQuantity) -> NumericFlag:
    return (quantity > 0).alias("is_buy")
```

The body is now checked at every cell of the region, which is what a broad declaration costs: `> 0` must answer at every numeric kind, and does. A Python type variable with a bound declares the same slice when the identity outcome must be spelled across it, and means nothing different.

Notice that this region is *narrower than polars*. Measured (`TestComparisonAgainstZero` in `spec/tests/test_expression_facts.py`), polars executes `> 0` on every kind except `String`, `Binary`, `Enum`, `Categorical`, `List`, `Array` and `Struct`: a `Date` compares as its physical days-since-epoch and runs. The checker demands only that the body's operations are defined on the whole region, so `NumericType` here is the author refusing meanings polars would execute, on the judgment that "is this date after the epoch?" is not what `is_buy` means. A bound as wide as the measured domain is equally legal. The region is a contract, and how much of polars' permissiveness it passes through is the author's call.

Note what a region is *not* doing: it is not asking the caller to prove a data type. A call hands over an expression that is evaluated at every cell of the region, and the frame that eventually evaluates the result proves its own membership, exactly as with a single pinned cell. The region is the honest record of what the body relies on, and the only thing it changes at the call is how many cells the argument is evaluated at.

The region must be written, not inferred, and the reason is a rule shared with `DATAFRAME_SPECIFICATION.md`: a contract is never inferred from a body. A parameter typed "open" would accept exactly the data types its body happens to survive, so an edit to the body would silently change which callers break, far from the edit. With the region written, the same edit breaks the function itself, on its own line, and the author chooses knowingly whether to move it.

So every declared slice checks in the same two moments: arguments against the slice at the call, the frame against the slice's cells at the `select`. One spelling remains, for the author who wants no data type checked in either moment:

```python
TrustedQuantity = column("quantity", Unchecked)
TrustedFlag = typed_expr(TrustedQuantity, gives=pl.Boolean, named="is_buy")


def trusted_buy_flag(quantity: TrustedQuantity) -> TrustedFlag:
    return (quantity > 0).alias("is_buy")
```

`Unchecked` is the declared form of *trust*: the slice that is every cell at once, so no answer can disagree with it at the call and no frame can fall outside it at the `select`. A declared `Unchecked` position accepts an argument of every data type, which the set reading of section 9 makes it mean and `T-22` requires. The body is then checked by the lifting: the comparison answers `Boolean` for every input it accepts at all, so the gives is recovered concretely even here.

Be precise about what the marker defers, because it is less than "this parameter is not checked". The name stays checked: any frame the result meets must declare a `quantity`. The fit stays checked: an argument giving another name, or an aggregating one, is refused. The output stays exact: `is_buy : Boolean`, recovered by the lifting. The one deferred question is whether `quantity`'s actual data type supports what the body does with it. For a data type polars rejects, a `List` here, the failure is a loud error when polars evaluates. For a data type polars executes with a meaning nobody intended, a `Date` compared to zero, there is no failure at all: the marker bought polars' full permissiveness, and that is part of what was signed. That is `T-21`'s containment read as a promise: one column's one question moved to run time, everything else still proved. The practical difference from a region: with `NumericType` declared, the frame that supplies a `List` column is refused at the `select`, before anything runs; with `Unchecked`, nothing is refused anywhere and the comparison raises at run time, the annotation being where the trust was signed.

One ladder, then: pin a single cell when the function is for one column type, declare a region when it must be checked for a family, write `Unchecked` when you accept the risk in exchange for taking anything.

**An aggregating function.** The character moves into the constructor:

```python
Quantity = column("quantity", pl.Int64)
TotalQuantity = typed_agg(Quantity, gives=pl.Int64)


def total_quantity(quantity: Quantity) -> TotalQuantity:
    return quantity.sum()
```

`quantity.sum()` keeps the name, keeps `Int64` (`sum` is the identity on it), and flips the character to aggregating, which is why the return constructor is `TypedAgg` and nothing else fits.

**Why must this function pin a data type at all?** `sum` runs on any numeric column, so the pinned `Int64` invites the thought: let the reads stay unevaluated, let the gives say "whatever sum gives at whatever arrives", and resolve everything at the frame, the way the inline expression would. The system refuses that signature, and the refusal has content on both sides.

The principled side: deferred code travels along expression *flow*, where the checker follows the value to the frame that reads it, and a declared type *replaces* the value's history at the boundary. A signature holding unread code is not a summary but a postponed computation, and its reading point lies past every place the checker can see from the call. Ask "who checks the body, against what?" and the proposal collapses into forms that already exist: checked at every cell of a region is a declared slice; checked at none is `Unchecked`. There is no third regime for a boundary, which is why there is no third spelling.

The empirical side is better, because the obvious broad declaration is *wrong*, and the checker is what catches it:

```python
Broad = typed_agg(NumericQuantity, gives=same)


def total_quantity(quantity: NumericQuantity) -> Broad:
    return quantity.sum()
    # error: at the cell Int8, sum gives Int64, and the
    #        declaration says same
```

Measured (`TestSumWidensSmallIntegers` in `spec/tests/test_expression_facts.py`), `sum` widens `Int8`, `Int16`, `UInt8` and `UInt16` to `Int64` and is the identity on the numeric types from 32 bits up. "Sum works for anything numeric, so give my type back" is a false promise at `Int8`, and the rejected signature is polars' own behavior, enforced. And this is where declaring several areas at once earns its keep: the honest declaration is the answer of `sum` itself over the numeric kinds, row for row, in one named value:

```python
Total = typed_agg(
    row(column("quantity", {pl.Int8, pl.Int16, pl.UInt8, pl.UInt16}), gives=pl.Int64),
    row(
        column("quantity", {pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64}),
        gives=same,
    ),
)


def total_quantity(quantity: NumericQuantity) -> Total:
    return quantity.sum()
```

Callers with `Int8` columns get `Int64` back, callers with `Int32` get `Int32`, and both facts sit in one contract, checked against the body row by row. Plain-Python spellings of the same slices remain legal, an `@overload` per row or a bounded type variable for an identity region, and they mean nothing a `row(...)` does not. The character then plays out at every use site, by section 3's rule:

```python
trades.select(total_quantity(pl.col("quantity")))
# quantity : Int64, one row

trades.group_by("symbol").agg(total_quantity(pl.col("quantity")))
# symbol : String, quantity : Int64, one row per group
```

**A function over any column.** Standardizing works for any float column under any name, and one declaration says precisely that:

```python
Standardized = typed_expr(some_column(FloatType), gives=same)


def zscore(e: Standardized) -> Standardized:
    return (e - e.mean()) / e.std()
```

`some_column(FloatType)` reads one column of any name, its data type anywhere in the float region, and `gives=same` promises that column's own type back under its own name. Walk the body cell by cell over the region:

```python
e  # one column, call its cell's type F
e.mean()  # mean is the identity on every float type: still F,
#   aggregating
e - e.mean()  # elementwise minus aggregating: the mean broadcasts
#   still F, since F joined with F is F
(...) / e.std()
# std is the identity on floats too: still F
```

Every cell of the region answers `same`, which is the declared outcome, so the function checks, once, for the whole slice.

**How the blank resolves.** `some_column` does not mean "no name"; it means the name is a **name blank**, a binder the call fills, and two rules pin it down. First, one declaration is one blank, and a signature using the same named declaration for its parameter and its return *shares* it: `Standardized` appears twice above, so the output name is the input name, per call. Second, the fill happens at the call and never later: every actual argument has a concrete output name, names being structural, so `zscore(pl.col("price"))` returns the instantiated slice, reads `{price}` and gives `price` back, and the `select` that eventually receives it resolves an ordinary named expression. A blank never reaches a frame; the one thing a body must not do with a blank-named parameter is materialize it itself, `df.select(e)` inside `zscore`, because a frame's schema cannot hold a nameless column, and an unknown name is exactly the uncertainty this system refuses to track (`D-27`).

A call binds the name now and leaves the cell to the frame:

```python
zscore(pl.col("price"))  # the name binds; the slice rides along

trades.select(zscore(pl.col("price")))
# price : Float64: this frame's price sits inside the region

readings.select(zscore(pl.col("temp")))  # temp : Float32 there
# temp : Float32: the width survives

trades.select(zscore(pl.col("symbol")))
# error, at the select: String is not in FloatType

zscore(pl.col("price").sum())
# error, at the call: an aggregating expression where the
#        declaration is elementwise
```

One wrong turn is worth showing, because the checker catches it at the definition and the error teaches the system's rules. Suppose the gives had been declared `Float64`, on the thought that standardized values are "just floats":

```python
AsFloat64 = typed_expr(some_column(FloatType), gives=pl.Float64)


def zscore(e: Standardized) -> AsFloat64:
    return (e - e.mean()) / e.std()
    # error: at the cell Float32 the body gives Float32,
    #        not Float64. There is no subtyping between data
    #        types, so the function itself is rejected.
```

The parameter stays `Standardized`, the column read back at its own type: declared `AsFloat64` as well, `e` would give `Float64` at every cell by its own contract, and the body would fit.

The rejection is the fit rule (below) protecting *callers*: a caller passing a `Float32` column was promised `Float64` back, and the body does not deliver it.

**A function that consumes an expression.** So far every function produced expressions. The other direction is a function that takes one, and this is where `Reads[...]` earns its keep. Give it a frame and a rule to flag rows with, and let callers write the rule:

```python
class TradesWithFlag(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64
    flag: pl.Boolean


Rest = other_columns()
FlagRule = typed_expr(Trades, gives=pl.Boolean, named="flag")


def flagged(
    df: TypedDataFrame[Trades, Rest],
    rule: FlagRule,
) -> TypedDataFrame[TradesWithFlag, Minus[Rest, Literal["flag"]]]:
    return df.with_columns(rule)
```

`FlagRule` passes a schema class as its reads: a schema class *is* a column set, so the declaration reads as "any elementwise expression reading only `Trades`' columns, giving `flag : Boolean`". That restriction is the every-caller rule wearing its expression hat: the body evaluates `rule` against a frame that is known to hold `Trades` plus a remainder, so a rule reading any column beyond `Trades` could name a column some caller's frame lacks. The frame half of the signature is `DATAFRAME_SPECIFICATION.md`'s `Minus` idiom unchanged.

Callers now compose rules, and the checker holds the line:

```python
flagged(trades, (pl.col("price") > 100.0).alias("flag"))
# fits: reads {price}, within Trades; the comparison gives
# Boolean at Trades' declared cells

flagged(trades, (pl.col("price") * pl.col("quantity") > 1e6).alias("flag"))
# fits: reads {price, quantity}, still within Trades

flagged(trades, (pl.col("price") > 100.0).alias("suspect"))
# error, at the call: gives suspect, where flag is declared

flagged(trades, pl.col("price").alias("flag"))
# error, at the call: at Trades' declared cell, price : Float64,
#        this argument gives Float64, where Boolean is declared

flagged(trades, (pl.col("venue") == "X").alias("flag"))
# error, at the call: reads venue, which Trades does not declare,
#        so some caller's frame has no venue at all
```

The fourth call shows the declared reads earning their cells twice over: `Trades` names `price : Float64`, so the argument is evaluated *there*, at the call, and the disagreement with `Boolean` does not get to hide until a frame arrives.

**The return side's no-cost rung.** Every return type above was declared. `InferredExpr` is the alternative: it does not declare a type, it asks the checker to compute one, from the function's own body. Each candidate return value is wrapped in `infer(...)`, which returns its argument unchanged at run time, and the function's real return type is the union of what the wrapped values themselves infer to:

```python
def price_expr(currency: str) -> InferredExpr:
    if currency == "USD":
        return infer(pl.col("price_usd"))
    return infer(pl.col("price_eur").cast(pl.Float64))


# callers receive the union of the branches' own inferred types:
# the USD branch, its code still unevaluated, beside the EUR
# branch, whose cast already fixed Float64
```

This is inference from the *body*: the author skips spelling a type the checker can compute, and callers still get the full answer. What it does not give is a contract. The return type now follows the body, so an edit to the body silently changes what every caller receives; a declared form freezes the interface instead.

**The parameter side has no such marker, deliberately**: there is nothing to infer a parameter from, since the body must check against *every* argument a caller could pass. A parameter is either `pl.Expr`, the plain polars type, which accepts anything and forgets the argument's type, or a declared form, which states what the body can rely on.

**Fit, the rule behind every "fits" above.** This passage restates `D-40`. An argument fits a declared slice when three clauses hold, and each appeared in the walk:

- its reads are within the declared reads. An expression reading *less* than the declaration promises fits: the first `flagged` call reads one column where `Trades` declares three. The declared `Reads[...]` is a promise about the frame, not a minimum.
- the output matches at the declared cells: the name is the declared one, a blank binding it by rule 5, and at every declared cell the argument, evaluated at that cell, answers exactly the declared outcome, and polars plans it at the data type that it runs. A cell of a kind with open parameters, such as `Datetime`, is checked at each data type of its sample, so `pl.col("t").dt.cast_time_unit("ms")` does not fit `column("t", pl.Datetime)`: it gives `ms` where the column is `us`. `suspect` for `flag` failed on the name, the column read at `Float64` failed against `Boolean`, and the wrong `zscore` return failed the same clause from the inside. A declared read that the argument does not make contributes nothing. There is no subtyping between data types: an argument answering `Int32` does not fit a declared `Int64`. The one exception is `Unchecked`, and it is symmetric: a declared `Unchecked` covers every cell, so no answer can disagree with it, and an `Unchecked` answer fits any declared outcome, the trust having been signed where the marker was introduced (section 9). The relation is not transitive: `Int32` matches `Unchecked` and `Unchecked` matches `Int64`, and `Int32` still does not fit `Int64`.
- the characters are equal, with no substitution between them. `flagged(trades, pl.lit(True).alias("flag"))` is refused alongside `pl.col("price").sum()`, and the literal's refusal is the subtler verdict: one value does broadcast wherever one value per row is expected, in a `select`. In an `agg` the two part ways, an elementwise result imploding into a `List` while a literal stays unwrapped (section 3), so a literal smuggled through an elementwise contract would falsify the contract's own aggregation answer. A constant rule is declared `typed_lit`, and a function meaning to accept either offers one signature per character.

None of this is Python subtyping. The bracket forms' arguments would compare nominally and refuse nearly every call above, so the checker computes fit at each call, from the argument's own type (measured: without that computation, an identity expression is refused against its own declared cell, in `TestFitIsNotSubtyping` in `spec/tests/test_mypy_binder.py`). The one rung ordinary subtyping supplies is the plain `pl.Expr` parameter: every typed expression is a `pl.Expr`, so the no-contract parameter accepts everything, and a plain `pl.Expr` argument fits no declared slice, because neither its name nor its answers are known.

The ladder, one table per side of the signature. The parameter table restates the one under `D-39`:

| Parameter | Contract | Body checked against |
| -- | -- | -- |
| `pl.Expr` | none; the argument's type is forgotten | the plain type |
| a slice of regions | one answer per region | every cell of every row |
| a slice of single cells | exact | the declared cells |

| Return | Contract | Callers receive |
| -- | -- | -- |
| `InferredExpr`, with `infer(...)` | none; follows the body | the body's own inferred type |
| a declared form | exact | the declared slice |

One last walk, because one expression visits every corner of this section. Where `pl.col("price") + 1` travels, and what is checked where:

```python
e = pl.col("price") + 1
# e : reads {price}; the addition, unevaluated; elementwise

# 1. into a frame operation: the everyday case
trades.select(e)  # price : Float64 here, so price : Float64 out
sizes.select(e)  # price : Int32 here: the same e, another answer
names.select(e)  # error: no column "price" in this schema


# 2. out of an InferredExpr return: the code rides along
def some_price() -> InferredExpr:
    return infer(e)


# 3. into a declared parameter: name and character judged now,
#    the code evaluated at the declared cells now, the frame later
discounted(e)  # fits: the addition at Float64 gives Float64
zscore(e)  # fits: identity-shaped at every cell of the region
buy_flag(e)  # error, at the call: gives price, where
#        quantity is declared

# 4. into an Unchecked parameter: no data type is checked; the name
#    and the character still are
trusted_buy_flag(pl.col("quantity") * 3)
# fits: the slice that is every cell, so no
# answer can disagree
```

Cases 1 and 2 are the code traveling to the one place a frame exists, and being evaluated there. Case 3 is a contract doing everything a contract can do at its own site, and leaving the frame exactly one question. Case 4 is the signed escape. Nothing else exists: those four are every way an expression crosses a boundary in this system.

______________________________________________________________________

## 8. Selections

A selection is an expression rooted in a *selector* rather than in named columns, and everything `DATAFRAME_SPECIFICATION.md` built for selectors carries over. Concretely:

```python
class Quotes(Columns):
    symbol: pl.String
    bid: pl.Float64
    ask: pl.Float64


quotes.select(cs.numeric().round(2))
# bid : Float64, ask : Float64      <- two outputs from one expression
```

Its type is its code, like any expression's, and four parts of the code decide what it does (`D-38`): the selector, the columns it reads beside the ones it picks, its answer at one matched column, and one name rewrite. Materializing it is exactly the frame effects of `DATAFRAME_SPECIFICATION.md` (`T-20`): keep what the selector picks, retype by the answer, rename by the affix. A column on the left of a selection, `pl.col("a") + cs.numeric()`, is the one rewrite that is no affix: polars names every output after that column, so the selection carries that one fixed name, two matched columns are the duplicate error, and no matched column gives no output. The retype is exact when the answer gives each column back or gives one fixed data type. Over an open frame the remainder gets the same three steps, kind by kind: `Only[R, C]` keeps the remainder columns the selector picks, even when the remainder is unconstrained, and for each kind the selector can pick there, the answer at the sample of that kind retypes its columns, and the affix renames them.

```python
class QuoteLevels(Columns):
    bid: pl.Float64
    ask: pl.Float64


Numeric = other_columns(all_of=NumericType)


def rounded(
    df: TypedDataFrame[Quotes, Numeric],
) -> TypedDataFrame[QuoteLevels, Numeric]:
    return df.select(cs.numeric().round(2))
    # Quotes' numeric columns and the whole remainder, all rounded:
    # cs.numeric() picks every column Numeric admits, and rounding
    # keeps every numeric type
```

The two contexts of section 3 need no new rule either: in an `agg`, the context rule applies once per matched column, by the selection's character (measured by `TestSelectionRoots` in `spec/tests/test_expression_facts.py`):

```python
quotes.group_by("symbol").agg(cs.numeric())
# symbol : String, bid : List(Float64), ask : List(Float64)

quotes.group_by("symbol").agg(cs.numeric().sum())
# symbol : String, bid : Float64, ask : Float64
```

A member that answers other data types in an `agg` than in a `select` keeps that difference on a selection too, because the code of a selection runs in `agg` at each matched column, as an expression's does (`D-38`). `cs.string().mean()` is refused in a `select` and gives each `String` column back in an `agg` and under `.over(...)`.

A selection can read other columns beside the ones it picks. `cs.numeric() + pl.col("fee")` reads each matched column and `fee` together, and each output keeps its matched column's name. `cs.numeric().sum().over("desk")` steers by `desk`. So the code of a selection also reads those names, at their declared data types (`D-38`). A column on the left, `pl.col("fee") + cs.numeric()`, gives every output the name `fee`, which polars refuses as soon as two columns match. A value meets each matched column alone: `cs.numeric().fill_null(0)` and `cs.numeric().shift(1, fill_value=0)` fill each column with the value, as the expression forms do.

One member outside the selector roots shares this multi-output shape: `.struct.unnest()` gives one column per field of the struct its operand gives, named and typed by the struct type itself. The operand is any expression that gives a struct. The field names come from a *type*, which is the one exception to the rule of section 4 that polars names an output from the tree, and it is a safe one: the names are known exactly when the struct's type is, and a struct declared `Unchecked` has unknown field names, which the system refuses rather than tracks.

`.struct.field("p")` reads one field. The expression reads only `s`, and the evaluation takes the field's type out of the type of `s` at the frame, so later operations compose on the field's own type as they do on a column's. A field of the element of a container, `pl.col("l").list.first().struct.field("p")`, is read the same way: `pl.col("l").list.first().struct.field("p") + 1` has the type that `+ 1` gives over the field's type.

A declared selection has its own bridge to the fit of `D-40`. A selection argument fits a declared selection when its selector picks no column the declared selector rejects (`T-9`), when its code answers the declared outcome at every cell of the declared rows, which partition the matched column's own type, and at each unit, zone or other parameter that a cell's kind stands for, and when the affixes and the characters are equal. An aggregating selection, `cs.numeric().sum()`, is declared `typed_agg_selection(selector, rows)`.

Well-formedness rule 7 of section 7 admits only two affixes, and the reason surfaces here: it decides which name rewrites are usable on a selection:

```python
quotes.select(cs.numeric().name.suffix("_r"))
# bid_r, ask_r: the suffixing effect, exact for every caller

quotes.select(cs.numeric().name.to_lowercase())
# error over a remainder: two remainder columns can collide at one
# lowercased name, so no single result type exists

pl.col("price").name.to_lowercase()
# fine: one known name; the checker computes the string
```

`.name.suffix` on one column and on a selection are both fine; `.name.to_lowercase` is fine on a known name and inadmissible on a selection (`T-3`). The full split of the ten `name` members is measured in `M-13` and listed in the catalog.

______________________________________________________________________

## 9. `Unchecked`, precisely

`Unchecked` enters from the frame side as "name known, data type not checked" (`D-27`). `D-27` defines it for frames and `D-41` for expressions, and the two define one marker. Expressions force the question of what it *does*, operation by operation, and the answer is one rule rather than a list of choices.

First, where does one come from? Two places only. The user writes it, as in the declaration of section 6's seed. Or a fallback listed in the catalog substitutes it under a known name, where the output data type is genuinely unknowable:

```python
trades.with_columns(pl.col("price").map_elements(convert))
# the output type is whatever convert returns: no checker reads that
# fallback: price : Unchecked; with allow_unproven at "stated" or a
# stricter level, an error naming map_elements and the fix

trades.with_columns(
    pl.col("price").map_elements(convert, return_dtype=pl.String)
)  # price : String: declared, so nothing is Unchecked
```

`map_elements` really is unknowable, measured: two lambdas with one signature produce two different schemas (`M-14`), so the answer lives in the function body, which is `B-6`'s territory.

Now, what happens when an operation meets an `Unchecked` column? The temptation is a per-operation list of special cases, and the system refuses it. Instead, read the marker as a *set*: `Unchecked` stands for the set of every data type, exactly as `Int64` stands for the one-member set of itself. Then every question has a mechanical two-step answer. The two steps restate the lifting and the collapse of `D-41`:

1. **Lifting.** Apply the operation to every member of the set, and keep the results from the members where it works at all. The application errs only when no member works.
2. **Collapse.** Show what that produced: exactly one distinct answer, and the output *is* that answer. Several answers show as what they share (`T-26`): `List(Unchecked)` for several lists, and `Unchecked` for several kinds.

The checker runs the lifting over a sample of the data types the set holds, one for each kind and several where the parameters of a kind change the answers of polars (`D-44`).

Run the two steps by hand on three operations and the whole behavior falls out:

```python
x = pl.col("mystery")  # mystery : Unchecked

x.str.len_chars()  # UInt32:  defined only on String, and then UInt32,
#          so every surviving member agrees
x.cast(pl.Int64)  # Int64:   every member casts to Int64
x + 1  # Unchecked: many members, many answers
x.implode()  # List(Unchecked): one list per member
```

The first line is the surprise worth stating twice: an `Unchecked` input does not condemn the output. An operation with one possible answer *recovers* a concrete type from `Unchecked`, and no special case grants it, only the lifting.

Two theorems make the marker safe to permit at all. **Containment** (`T-21`): forgetting one column's type changes nothing but data types computed from that column. The one exception is `.struct.unnest()` over a struct computed from that column, whose output names are the struct's fields.

```python
class Mixed(Columns):
    symbol: pl.String
    payload: Unchecked


mixed.select(pl.col("symbol").str.to_uppercase())
# symbol : String: payload's uncertainty reaches nothing here
```

**And no new errors** (`T-22`): a program that checks with a concrete type still checks after that type is forgotten, because every gate on an `Unchecked` position passes when some member passes. A gate that demanded every member would break this, which is why the lifting never reads a gate that way. The marker only ever removes errors, so strictness is preserved as a global invariant: a program with no `Unchecked` is checked with no optimism anywhere, and each marker weakens exactly the checks its column reaches.

**One marker, two doors, one meaning.** The two origins can read as two concepts wearing one name. Written in a signature, `Unchecked` is a *choice*: accept anything here. Minted by a fallback, it is a *loss*: the checker no longer knows. The semantics refuses the distinction: both are the same set of every type, under the same lifting and the same two theorems, and no operation downstream can tell which door its input came through. One name is therefore a decision, not a necessity of grammar: a second name with the same behavior would be an alias, and this system folds two spellings of one meaning into one form on principle. What separates the two doors is only who signed the trust, and `allow_unproven = "never"` exists to erase exactly that difference: under it, every loss must be repaired or countersigned by an annotation, so in strict code the marker has one meaning, trust, explicitly given.

The alternative is a second marker for "genuinely unknown", checked pessimistically: only operations valid for *every* data type would pass. It is rejected on two grounds. Almost no operation is total over every type, so a pessimistic unknown would refuse nearly everything, and the fallback that mints it would stop paying for the name it keeps, which is its one job. The checked-for-every-type reading exists on the *declared* side only, where refusing nearly everything is a contract someone chose: it is the region at its widest, `EveryDType`, and no fallback ever mints it. Nor does a schema class hold it: a frame column is one concrete type, `Unchecked` or a type variable of a generic schema class (`D-6`), because a region promises a per-cell check no single column can host.

Last, the confusion to avoid. `Unchecked` is not the way to write a generic function, and the two read almost alike:

```python
def bad(e: TypedColumn[Literal["x"], Unchecked]) -> ...:
    return e.round(2)  # accepted: some type rounds. Trust, not proof.


def good[D: NumericType](e: TypedColumn[Literal["x"], D]) -> ...:
    return e.round(2)  # checked: every numeric type rounds
```

The table restates the one in section 18.4 of `OPERATIONS.md`:

| Aspect | `Unchecked` | A declared region |
| -- | -- | -- |
| meaning | this one value's data type is not checked | for every member, checked |
| gates | pass when some type passes | pass only when the whole region passes |
| output | recomputed by the lifting, often `Unchecked` | the declared outcome, exactly |
| relates positions | no: two `Unchecked`s share nothing | yes: `same` gives the read column's own type back |
| honest reading | trust me | works for all of them |

A reusable function wants the region: its body is checked at every cell, and frames get exact answers back. `Unchecked` in a signature is an abdication the lifting then propagates. Both are legitimate. Only one is a contract.

______________________________________________________________________

## 10. Limits

Boundaries of the model, each genuine rather than an omission.

**Renaming through arbitrary code.** `.name.map(f)` computes the output name by running `f`, so the name becomes unknown, which is the kind of uncertainty this system rejects rather than tracks (`D-27`, `B-6`). The computable rewrites, `.alias`, `.name.suffix`, `.name.prefix` and their kin, are the typed spellings.

**A function's output type.** `map_elements` without `return_dtype=` gives two different schemas for two lambdas with one signature (`M-14`), so the output type is decided by code no checker reads (`B-6`). With `return_dtype=` given, or under the `Unchecked` fallback, the call is typed.

**A loop whose type grows on every pass.** mypy stops its analysis of a loop after a few passes, so a loop that rebuilds an expression leaves the versions of the first passes (`M-6`). Where those versions agree, the checker takes their answer for every number of passes (`T-18`), and where they differ, the output is `Unchecked`. The declared seed of section 6 is the stated way to fix one type.

**A name computed at run time.** `pl.col(key)` over a string the checker cannot read has unknown reads, and an unknown name is refused rather than tracked (`B-5`). A variable holding a literal string is read like the literal.

**Values are invisible.** The evaluation reads data types, never data. An operation whose failure depends on what the rows hold, division by a zero that is present, a cast that meets a malformed string, is checked to its type-level domain and no further (`B-1`). A value in a variable is read at a sample of its type (`D-44`).

______________________________________________________________________

## 11. Summary of the forms

**Inferred types**, which the checker writes for every expression:

| Written | Means |
| -- | -- |
| `TypedExpr[Deferred["<code>"]]` | the code of an expression, evaluated where it meets a frame or a declaration |
| `TypedExpr[Deferred["<code>", <declaration>, ...]]` | code over declared leaves, `$0`, `$1` and so on, each a parameter's declaration |
| `($1 := <code>) ... $1` | a part of the code that it uses more than once, written once |
| `hole('int')` and its kin | a value the checker cannot know, read at a sample of its type |

**Slice declarations**, named values used in signatures, and the expanded forms they register:

| Declared | Expands to |
| -- | -- |
| `column(n, d)` | `TypedColumn[Literal[n], d]`, the identity collapsed: read and given back |
| `typed_expr(reads..., gives=<outcome>, named=n)` | `TypedExpr[Reads[...], Column[Literal[n], ...]]` |
| `typed_agg(...)`, `typed_lit(...)` | `TypedAgg[...]`, `TypedLit[...]` |
| `column_of(Schema, n)`, `subset_of(Schema, n, ...)` | reads at the schema's declared types |
| `typed_expr(row(...), row(...))` | several areas of one declared map, disjoint |
| `some_column(<region>)` | one read under a name blank a call fills |
| `same`, `wrap(C, ...)` | the read column's own data type, and a constructor around an outcome |
| `typed_selection(cs..., rows, suffix=...)` | `TypedSelection[...]`, holding the selector $C$, the answer at one matched column and the name rewrite $\rho$: rows over the matched column, applied per match |
| `typed_agg_selection(cs..., rows)` | the same, aggregating per matched column |
| `InferredExpr` | a return type computed from the function's own body |

A region is a set of kinds: one data type, a family, a set, or `EveryDType` for every kind, checked at each. `Unchecked` stands in the same position and declares that no region is checked.

The bracket forms on the right are what the checker shows and compares. On the frame side, `Retyped[R, T]` (`E-6`) retypes a remainder the way an answer retypes one column. Section 7 gives the fit rule that decides when an argument fits a declaration.

Everything in this document follows from the definition in section 4: the type of an expression is its code, what it means is its map, and every question a checker asks is an evaluation of it.

______________________________________________________________________

## 12. Glossary

Every term this document uses with a technical meaning, and where it is defined. A term not in this list does not appear in this document with a special meaning. The "Formally" column gives the identifier in `OPERATIONS.md`, or a section there where no single definition holds the term, or "none" for a spelling that only this document uses.

| Term | Means | Here | Formally |
| -- | -- | -- | -- |
| cell | one tuple of kinds, one per declared read, at which a declaration states an outcome | 7 | `D-39` |
| character | whether an expression is elementwise, aggregating or literal, as a declaration states it | 3 | `D-29` |
| code | the type of an expression: the expression written with the public API of polars | 4 | `D-33` |
| declared leaf | a parameter's declaration inside the code of an expression, written `$0` | 4 | `D-33` |
| declared map | a declaration read as a map: regions in, one outcome per cell | 7 | `D-39` |
| deferred | the form of a type that holds code and computes nothing until a frame or a declaration | 4 | `D-33` |
| evaluation context | where an expression runs: the row context or the aggregation context | 3 | `D-28` |
| `EveryDType` | the widest region: every kind, the body checked at each | 7 | `D-41` |
| expression | a computation a user writes, producing columns from columns | 1 | section 13 |
| fit | when an argument's type agrees with a declared slice | 7 | `D-40` |
| gives | the output column a declared slice promises | 7 | `D-39` |
| hole | a value in the code that the checker cannot know, read at a sample of its type | 5 | `D-44` |
| input | a sub-expression of a call, read by the node of the call | 5 | `D-34` |
| kind | a data type with its parameters erased: one per scalar type, one per parametric constructor | 7 | `D-3` |
| lifting | applying a rule to every member of a data-type set | 9 | `D-41` |
| map | what a type means: the columns it reads, and its answer at each frame | 2 | `D-33` |
| name blank | `some_column`'s unnamed read: a binder each call fills | 7 | `D-39` |
| outcome | what a declaration states at a cell: a data type, the read column's own type, or a constructor around an outcome | 7 | `D-39` |
| output name | the name of the column an expression appends | 4 | `D-32` |
| planner type | the type that polars plans for a node, before it runs | 5 | `D-36` |
| reads | the column names an expression reads, every input of every node included | 4 | `D-31` |
| region | a declared set of kinds, the cells a body is checked at | 7 | `D-39` |
| row form | how many values a node gives: one per row, one value, one row, or another length | 3 | `D-29` |
| run type | the data type of the data that polars computes for a node | 5 | `D-36` |
| sample | the values or data types at which the checker reads a hole or an open column | 5 | `D-44` |
| schema reference | `column_of(Trades, "a")`: a read at the schema's declared type | 7 | `D-39` |
| selection | an expression rooted in a selector, evaluated at every matched column | 8 | `D-38` |
| slice declaration | a named value declaring areas of a declared map: regions in, one outcome each | 7 | `D-39` |
| `TypedColumn` | the form `column(n, d)` registers: the identity collapsed | 7 | none |
| `Unchecked` | a data type meaning "the name is known, the data type is not checked", one marker for frames and for expressions (`D-41`) | 9 | `D-27` |
