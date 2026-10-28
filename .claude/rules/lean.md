---
paths:
  - spec/lean/**
---

# Lean

These rules apply to the Lean 4 formalization in `spec/lean/`. It proves the theorems of `spec/OPERATIONS.md`, and the document cites each proof by its Lean name.

The rule in `CLAUDE.md` that keeps the plugin and `spec/` in step applies to every change here.

## Names

- Name a theorem that the specification cites `Spec.<Id>.<snake_case>`, where `<Id>` is the identifier it proves, for example `Spec.T2.<name>` for a proof of `T-2`.
- Name a lemma that a definition needs, but that the specification does not state, after that definition, for example `Spec.D14.<name>` for a lemma about `D-14`.
- Cite a theorem in a document as `` `Spec.<Id>.<name>` in `spec/lean/Spec/<layer>/<File>.lean` ``. A test in `spec/tests/test_documents.py` checks that the name exists in that file.

## Obligations

- Add every cited theorem to `Audit.lean` with `#assert_standard_axioms`. Why: the build then fails when the theorem uses `sorry` or an axiom other than `propext`, `Classical.choice` and `Quot.sound`.
- Do not use `sorry`, `admit`, `native_decide`, `implemented_by` or `axiom`. A test checks the sources for these tokens.
- Write a measured fact about polars or mypy as a field of a structure, never as an axiom, as the field `locality` of `Polars` in `Spec/L6/Node.lean` states one. A theorem that needs the fact takes the structure as an argument. The field's docstring names the test that measures the fact.
- If Lean shows a statement of the specification false, correct the statement to what Lean proves, and open a GitHub issue for the weakness. Do not weaken a Python test to match.

## Structure

- Put each layer in its own directory under `Spec/`, numbered from `Spec/L0/` up. A layer imports only the layers below it.
- Import only the Mathlib modules a file needs, for example the one for finite automata, never all of Mathlib. Why: the build stays fast.
- Represent a name as `List Char`, a column set as `Set (Name × DType)`, and a data-type set as a finite list of kinds and single data types. Nest an inductive through a list, never through Mathlib's finite sets or finite maps, which are quotients.
- Do not run `lake update`. The Mathlib version is pinned in `lakefile.toml` and `lake-manifest.json`.
