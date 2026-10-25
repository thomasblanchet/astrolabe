"""Executable evidence for the four specification documents.

The specification documents state facts of three kinds. A proved fact is
proved in Lean, in `spec/lean/`. This package holds the evidence for every
other fact, and checks the proved facts again on a finite model.

- A **measurement** is a count over a finite space. `test_measurements.py`
  recomputes each one, so a number in a document cannot drift.
- A **theorem** about the model is proved in Lean and checked again by
  exhaustion on a finite model. `test_absence_language.py` and `test_normal_form.py` do this
  against `model.py`, a reference implementation of the specification.
- A **polars fact** is what the real library does.
  `test_polars_facts.py` runs the call and asserts the columns it gives.

`test_documents.py` then checks that the documents agree with all three,
and that every Lean theorem they cite exists and is audited.
"""
