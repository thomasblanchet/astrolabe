# The specification

Four documents describe astrolabe's type system, and one test suite holds them to account. A fifth records the behaviors of polars that the rules work around.

| Document | Answers | Read after |
| -- | -- | -- |
| [DATAFRAME_SPECIFICATION.md](DATAFRAME_SPECIFICATION.md) | What is a typed frame, and why is it built this way? It explains in prose, with an example for each idea. | nothing |
| [EXPRESSION_SPECIFICATION.md](EXPRESSION_SPECIFICATION.md) | What is the type of an expression, and how is it used? It explains in prose, with an example for each idea. | `DATAFRAME_SPECIFICATION.md` |
| [OPERATIONS.md](OPERATIONS.md) | What exactly are the rules for frames and expressions, and why are they correct? It gives the definitions and the proofs. | the two specifications |
| [OPERATION_CATALOG.md](OPERATION_CATALOG.md) | What does one polars call do under these rules? It covers the frame API, `GroupBy`, the selectors and the expression API. | `OPERATIONS.md` |
| [POLARS_BEHAVIORS.md](POLARS_BEHAVIORS.md) | Which polars behaviors do the rules work around? It lists the calls that polars types as valid and then refuses, and the answers that polars decides by a rule no statement captures. | `OPERATIONS.md` |

A reader new to the system reads the two specifications in order. A reader who implements or attacks the rules reads `OPERATIONS.md`. A reader with one polars call in hand looks it up in the catalog.

## How a claim is supported

`OPERATIONS.md` is authoritative. It gives every definition, requirement, effect, theorem, measurement and boundary an identifier: `D-n`, `R-n`, `E-n`, `T-n`, `M-n` and `B-n`. Its last section, the claim ledger, tags each one. The other three documents cite those identifiers. The two specifications restate some definitions in prose for the reader. If a restatement disagrees with `OPERATIONS.md`, the restatement is wrong.

Every claim has one of three kinds of support:

- **proved**: proved in Lean 4 with Mathlib. The text names the theorem and its file under `spec/lean/`, and gives a short sketch of the proof.
- **measured**: the text names the test that computes it, in the form "Measured by `TestName` in `spec/tests/test_file.py`."
- **assumed**: the text names its source.

## Conventions

- A formula is written in LaTeX: `$...$` inline, `$$` around a display.
- A section is cited as "section 6.4", and a section of another document as "section 6.4 of `OPERATIONS.md`". A definition, theorem or measurement is cited by its identifier.
- The documents describe the type system, not the plugin that implements it.

## The proofs

`lean/` holds the Lean formalization of every proved claim. The build checks every proof:

```
cd spec/lean && lake build --wfail
```

A warning fails the build, and so does `sorry`. `Audit.lean` fails it when a cited theorem uses an axiom other than Lean's three standard ones. A fact that only a measurement gives, for example that polars types a node from the states of its inputs, enters a proof as a hypothesis of its theorem, and the hypothesis names the test that measures it. Before the first build, run `elan toolchain install "$(cat lean-toolchain)"` and `lake exe cache get` in `lean/`.

| File | Formalizes |
| -- | -- |
| `lean/Spec/L0/Objects.lean` | names, kinds, data types, column sets and data-type sets |
| `lean/Spec/L1/` | the absence and selector languages, and the settling functions of `D-14` |
| `lean/Spec/L2/` | effects, records and remainder expressions: `T-1`, `T-2`, `T-3` |
| `lean/Spec/L3/Canonical.lean` | the canonical form: `D-22`, `T-4`, `T-5`, `T-8` |
| `lean/Spec/L4/` | regular name sets and decidable containment: `D-11`, `T-9` |
| `lean/Spec/L5/` | requirements, splits and soundness: `T-11` … `T-14` |
| `lean/Spec/L6/Node.lean` | the evaluator of trees and its lifting to data-type sets: `T-25`, `D-41`, `T-21`, `T-22` |
| `lean/Spec/L6/Generalize.lean` | what several answers share: `T-26` |
| `lean/Spec/L7/` | the contexts, selections and fit: `T-15`, `T-20`, `T-21`, `T-23` |

## The evidence

`tests/` is the executable half. It holds a reference implementation of the model, and it checks the documents against that model and against the installed polars.

```
uv run pytest spec
```

| File | Establishes |
| -- | -- |
| `model.py` | the reference model: the three languages, the record, the effects |
| `normal_form.py` | remainder expressions, their meaning, and their canonical form |
| `universe.py` | a finite universe, so a claim can be checked by exhaustion |
| `measurements.py` | the counts the documents quote |
| `documents.py` | the documents read as data |
| `test_absence_language.py` | closure of the record under every effect |
| `test_normal_form.py` | the canonical form exists, is unique, and decides equality |
| `test_requirements.py` | containment between two name rules is decided exactly |
| `test_polars_facts.py` | what polars does, for every behavior the catalog asserts |
| `test_measurements.py` | every count the documents quote |
| `fit_model.py` | executable declarations, fit, and the materialization procedure |
| `table_model.py` | the declared map: cells, outcomes, and how a declaration reads them |
| `axis_model.py` | declared maps over several reads |
| `test_fit_model.py` | fit is sound by exhaustion, with its falsifiability controls |
| `split_model.py` | the split binder: first match wins |
| `test_split_model.py` | the split theorems and the recomputed precision counts |
| `test_grammar_corners.py` | the verdict for every pathological declaration |
| `test_catalog_rows.py` | every Part I catalog row, replayed against polars |
| `test_scenarios.py` | whole pipelines: model predictions against polars at every step |
| `test_expression_facts.py` | what polars does with an expression at a frame: contexts, names, consumer sites, selections |
| `rule_probes.py` | the probes that run polars in worker processes for the tests of the rules |
| `test_rules_*.py` | each part of the vocabulary of polars against polars: data types, the supertype, casts, options, type functions, the cast pass, kernels and the run |
| `test_definitions_*.py` | the definition of each node, family by family, against polars |
| `test_rules_tree.py`, `test_rules_tree_evaluation.py` | the tree of a polars expression, and its evaluation node by node |
| `test_pattern_reader.py` | the reader of name patterns |
| `test_mypy_binder.py` | how mypy types a rebound loop variable, and how a hook hosts fit |
| `test_declaration_probes.py` | a declaration class with requirement bases survives mypy's incremental cache |
| `test_documents.py` | the documents agree with each other, with the tests, and with polars |

`tools/node_census/` names the nodes of polars, the kinds of data, the input outcomes and the samples of values that the tests of the rules vary. `tools/rules/tree_check.py` builds random trees of those nodes, and compares the evaluation of each tree with polars. A maintainer runs it by hand at a new version of polars.

`test_documents.py` keeps the four documents consistent over time. It fails when an identifier is cited but not registered, when a quoted count disagrees with the test that computes it, when a cited test does not exist, when a cited Lean theorem does not exist or is not audited, or when a polars release adds an operation that the catalog does not name.

## What is not here

`../ARCHITECTURE.md` describes how the plugin is built. `../KNOWN_ISSUES.md` records where the plugin falls short of these documents. Neither is part of the specification.
