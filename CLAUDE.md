# astrolabe

astrolabe is a mypy plugin. It checks the column names and the data types of polars frames and expressions before the code runs.

- `spec/` is the design authority. It states what each type means and proves the rules.
- `ARCHITECTURE.md` describes the layers and the modules.
- `site/` holds the user guide: the landing page and the documentation. `README.md` is a short introduction that links to it.
- `KNOWN_ISSUES.md` lists the bugs and the fixable gaps. It exists only while there is one. `POLARS_ISSUES.md` lists the polars behaviors the checker works around. The last section of `ARCHITECTURE.md` lists only the weaknesses that the design accepts.
- Documentation and code describe the project as it is now. Do not record history in them.

## Rule files

The topic rules are in `.claude/rules/`. A rule file loads when you read a file that matches its paths.

| File | Loads for | Covers |
| -- | -- | -- |
| `python.md` | every `.py` file | class structure, third-party wrappers, code smells, the complexity limit, fallible chains, doctests |
| `hooks.md` | `src/astrolabe/_hooks/` | error classes, `allow_unproven` and the gate, fallbacks, mypy facts |
| `tests.md` | `tests/`, `spec/tests/` | yml cases, integration runs, coverage |
| `writing.md` | every `.md` and `.py` file | Simplified Technical English, docstrings, Markdown format |
| `glossary.md` | every `.md` and `.py` file | the one term for each concept |
| `lean.md` | `spec/lean/` | the Lean formalization of the proofs in `spec/` |

The `simple-english` skill in `.claude/skills/` has the full writing rules. Load it before you write or rewrite any prose: a document, a docstring or a comment. One docstring needs it as much as a whole document.

## Verification

After each change that is not trivial, run every command below. A run passes only if every command passes and coverage stays at 100% in both runs of the suite.

```
uv sync
uv run pytest
uv run --exact --group numpy pytest && uv sync
uv run mypy src/astrolabe
uv run mypy tests
uv run mypy spec
uv run mypy tools
uv run ruff check .
uv run ruff format --check .
uv run complexipy . --failed
uv run mdformat --check .
uv run python -m tools.site_examples --check
cd spec/lean && lake build --wfail
```

- `uv run pytest` measures coverage and fails under 100%. It takes about 15 minutes.
- The suite runs twice: without numpy, and with it. The type stubs of polars read numpy, so some calls type differently in the two worlds, and the rules must hold in both. `uv sync` installs the `dev` group and removes numpy. `--group numpy` adds it for the second run, and the last `uv sync` removes it again, because `uv run` alone keeps a package that an earlier command installed.
- Do not edit a file in `src/` while a coverage run is in progress. The run then measures a mix of old and new code.
- Read the output of `complexipy` in full, or filter it with `grep -E "FAILED|\.py$"`. Do not pipe it through `tail -1`, because its banner then hides the failures.
- `lake build --wfail` checks every Lean proof in `spec/lean/`, and `Audit.lean` fails the build when a cited theorem uses `sorry` or an axiom other than Lean's three standard ones. Before the first run, run `source ~/.elan/env`, `elan toolchain install "$(cat lean-toolchain)"` and `lake exe cache get` in `spec/lean/`. The download needs the network and about 8 GB.
- `uv run ruff format .` formats every Python file: a call, a signature or a literal fits on one line, or it puts each item on its own line with a trailing comma. `uv run ruff format --check .` only reports a file that is not formatted.
- `uv run mdformat .` formats every Markdown file. `uv run mdformat --check .` only reports a file that is not formatted.
- `uv run python -m tools.site_examples --check` runs mypy over the examples of the documentation in `site/examples/`. It fails when `site/src/data/examples.json` differs from a new run. After a change that alters a message or a type, run it without `--check` to write the file, and read each page that shows an example that changed.

## Rules for every task

- Verify a fact about mypy or polars with a small script before you rely on it. Why: mypy and polars often behave differently from what their documentation suggests.
- Treat a plan as a hypothesis. Verify each grouping and each claim of a plan against the code before you implement it.
- Keep a change to the scope that the user asked for. If you find a problem outside that scope, report it.
- Keep the plugin and `spec/` in step. When you change what the plugin accepts, rejects or mints, change the rule in `spec/` that states it, in the same change: the `spec/OPERATIONS.md` definition or theorem, its restatement in the two specifications, and the catalog row. When you change a rule in `spec/`, change the plugin and its tests to match in the same change. Update the Python model in `spec/tests/` and the Lean proof in `spec/lean/` where the rule has one. Why: `spec/` is the design authority, and a plugin that disagrees with it is a bug on one side or the other.
- Keep the documentation in step with the plugin. When a change alters a message, a rule, a setting or the API, update the pages in `site/src/content/docs/` that describe it, and the examples in `site/examples/`. Why: the site is the user guide, and a page that disagrees with the checker misleads the user.
- Record each bug, gap or defect that you find and do not fix in the same change. This applies to the plugin, to `spec/` and to a disagreement between them. Add it to `KNOWN_ISSUES.md`, and create the file if it does not exist. Give the input, what the checker or the spec says, and what polars or the other side says. When a change fixes an entry, delete the entry in that change. When the file has no entry left, delete the file. Why: an issue that is not written down gets lost between sessions.
- This directory is not a git repository. Back up the files before a destructive edit, for example with `tar czf` into the scratchpad.
- Do not delete or weaken a test to make a gate pass. Fix the code, or report the conflict.
- Save a plan that outlives one session as a file in the repository. Why: a later plan can overwrite a plan-mode file.
