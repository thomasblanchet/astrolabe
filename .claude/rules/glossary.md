---
paths:
  - '**/*.md'
  - '**/*.py'
---

# Glossary

Use one term for each concept. The right column lists the words to avoid for that concept. A code identifier keeps its own spelling, for example `dtype` in a parameter name.

## Frames and columns

| Term | Meaning | Do not write |
| -- | -- | -- |
| frame | a polars `DataFrame` or `LazyFrame` | dataframe, table, df |
| typed frame | a `TypedDataFrame` or a `TypedLazyFrame` | annotated frame |
| plain frame | a `pl.DataFrame` or `pl.LazyFrame` that the checker does not track | untyped frame, erased frame, bare frame |
| column | one named series of a frame | field, attribute |
| field | one member of a `Struct` column | column, key |
| data type | the polars type of a column, for example `Int64` | dtype (in prose), column type |
| schema | the columns of a frame, each with a name and a data type. A schema has no order. | layout, shape |
| declared columns | the columns that the type of a frame names | known columns, explicit columns |
| remainder | the columns of an open frame that its type does not name, written with `other_columns()` | rest, hidden columns, extra columns |
| open frame | a frame whose type has a remainder | generic frame, partial frame |
| closed frame | a frame whose type has no remainder | complete frame |
| `Unchecked` | the data type of a column whose name is known and whose data type is not | any dtype, unknown type |

In an error message for a user, write "the undeclared columns" for the remainder.

## Expressions

| Term | Meaning | Do not write |
| -- | -- | -- |
| expression | a polars `Expr` | expr |
| typed expression | an expression whose type the checker minted | annotated expression |
| plain expression | a `pl.Expr` that the checker does not type | untyped expression, raw expression |
| selection | a typed selector expression, for example `cs.numeric()` | selector type |
| declaration | a type that the user writes: `column(...)`, `typed_selection(...)`, a `Columns` class or `other_columns(...)` | annotation, spec |
| fit | the check that an argument matches its declaration | match, conformance |
| kind | a data type with its parameters taken away, for example `Datetime` for every time unit and time zone | base type, bare type |
| rendering | the text that polars prints for a data type, parameters included, for example `List(UInt32)` | dtype string, repr |

## The checker

| Term | Meaning | Do not write |
| -- | -- | -- |
| checker | mypy with the astrolabe plugin loaded | type checker, the tool |
| operation | a frame method or a polars function that the checker types, for example `select` or `pl.concat` | op, method call |
| hook | a class that the plugin registers with mypy for one operation | handler, callback |
| mint | build a type in a hook | create, synthesize, generate |
| report | give an error through mypy | raise, emit, flag, throw |
| error | a message that mypy prints with a code | issue, problem, diagnostic |
| fall back | give the plain frame instead of a typed result | erase, degrade, widen |
| setting | one key under `[tool.astrolabe]` | option, flag, config value |
| reason | why an operation gave up on a proof, a `Reason` | cause, motive |
| level | a value of `allow_unproven`, a `Level` | strictness, lenience |

## Verbs for the work

| Term | Meaning | Do not write |
| -- | -- | -- |
| check | what the checker, a test or a gate does to code | test (as a verb), lint |
| verify | confirm a fact yourself, with a script or a test | confirm, validate, ensure, make sure |
| run | start a command or a test | execute, invoke, launch |
| show | put text on the screen | display, render, print (in prose) |
| delete | take away a file, a line or a key for good | remove, drop, destroy |
