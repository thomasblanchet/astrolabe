---
paths:
  - tests/**
  - spec/tests/**
  - conftest.py
---

# Tests

## Coverage

- `uv run pytest` fails under 100% coverage. Ship every new function, branch and method with the test that covers it, in the same change.
- A branch that cannot run gets `# pragma: no cover` with a one-line or two-line reason beside it. Use this only when a mypy contract or an earlier check rules the branch out. Do not write a test that fakes an impossible input.
- Cover a path that needs a full mypy run with a `.yml` case. Cover a helper that works on plain values with a doctest.

## yml cases

- The files in `tests/typing/*.yml` run through `pytest-mypy-plugins`. Each expected error is a `# E: message  [code]` comment on its line. Each note is a `# N: text` comment after it, on the same line.
- An error from the `allow_unproven` gate has a note under it, so its line needs a `# N:` part too.
- A `mypy_config:` key must come before `main:` in a case.
- A yml case runs with a generated `pyproject.toml` that keeps only `[tool.mypy]`. To check a `[tool.astrolabe]` setting, give the case its own `pyproject.toml` in a `files:` entry. pytest-mypy-plugins writes that file after its own, so it replaces it, and it must repeat the `[tool.mypy]` keys: `plugins`, `strict` and `allow_redefinition`. See `tests/typing/test_pinned_top1.yml`.
- A test that needs a second run of mypy, for example over a warm cache, calls `mypy.api.run` against a real `pyproject.toml` in `tmp_path`. See `tests/test_incremental_cache_integration.py`.
- `--mypy-same-process` runs each case inside the pytest process. Why: coverage cannot see a subprocess.
- Where the answer of a case depends on the release of polars, keep one case for each range of releases. Give each one a `skip:` key that leaves out the other releases, for example `skip: __import__("tests.releases").releases.polars_before("1.43.0")`, and `skip: not ...` on its twin for the older releases. The runtime cross-check reads the same key. Where the older releases do not have the function that the cases call, the cases skip those releases, and one twin shows what the checker gives there. Why: the checker follows the rules of the installed release, and the CI runs the suite under each supported release.

## Runtime cross-check

`tests/runtime_cross_check.py` runs the `main:` of each yml case as real Python on sample data. It then compares the real schema to the type that the case asserts.

- Read the schema of a real `LazyFrame` with `.collect_schema()`, not `.schema`. Why: `.schema` can abort the process from a background thread.
- Catch `BaseException` around code that runs polars, and re-raise `KeyboardInterrupt`, `SystemExit` and `GeneratorExit`. Why: `pyo3_runtime.PanicException` does not inherit from `Exception`.

## spec tests

`spec/tests/` proves or measures every claim in `spec/`. `spec/tests/documents.py` parses the Markdown files for identifiers, table rows and counts. If you change the Markdown layout of `spec/`, run `uv run pytest spec`.
