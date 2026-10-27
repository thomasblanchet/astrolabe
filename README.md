# astrolabe

A mypy plugin that checks Polars column names and data types statically, one step past the runtime schema Polars keeps for itself.

A frame's static type carries its schema: the columns it declares, and a *remainder* for the columns it does not. An expression's static type carries what it reads and what it gives. The plugin computes every answer itself and asks mypy only to hold the types and substitute them at calls.

The documentation is at <https://thomasblanchet.github.io/astrolabe/docs/>.

> **Pre-release.** The API, the messages and the supported versions can change in any release.

## Setup

Install astrolabe, then run this command in the project directory:

```
astrolabe init
```

The command adds the plugin to `[tool.mypy]` in `pyproject.toml` and sets `allow_redefinition = true`. It also writes a `[tool.astrolabe]` table with every setting at its default, each with a comment. The comments and the format of the file stay as they are, and a second run changes nothing.

- `--allow-unproven LEVEL` writes a different level. See [Configuration](https://thomasblanchet.github.io/astrolabe/docs/reference/configuration/).
- `--dry-run` prints the changes and writes nothing.
- `--path DIR` sets up the project in `DIR`.

If mypy reads its configuration from `mypy.ini`, `.mypy.ini` or the `[mypy]` section of `setup.cfg`, the command changes nothing. It prints the lines to add to that file instead.

To set up by hand, add these lines to `pyproject.toml`:

```toml
[tool.mypy]
plugins = ["astrolabe.plugin"]
allow_redefinition = true
```

`allow_redefinition = true` is required. An expression type converges in a loop that rebinds its variable only when mypy permits the rebinding. The plugin logs a warning when the setting is off.

## Example

```python
import polars as pl
from astrolabe import Columns, TypedDataFrame


class Trades(Columns):
    symbol: pl.String
    price: pl.Float64
    quantity: pl.Int64


trades: TypedDataFrame[Trades]
trades.select("symbol", pl.col("price") * pl.col("quantity"))
# TypedDataFrame[TypedDict({'symbol': String, 'price': Float64}, closed=True), None]

trades.select("venue")
# error: Unknown column "venue"  [unknown-column]

trades.select(pl.col("symbol") - pl.col("price"))
# error: "symbol" has no result type at this frame's column types: arithmetic on dtypes
#        String and Float64 is not allowed, and polars raises InvalidOperationError
#        [unsupported-dtype-op]

trades.filter(pl.col("price"))
# error: Column "price" is not Boolean, cannot .filter()  [wrong-column-shape]
```

## Documentation

The site holds the user guide:

- [Start here](https://thomasblanchet.github.io/astrolabe/docs/start/overview/): what the checker finds, the install and a first typed frame.
- [Tutorial](https://thomasblanchet.github.io/astrolabe/docs/tutorial/load/): one pipeline, typed step by step.
- [Concepts](https://thomasblanchet.github.io/astrolabe/docs/concepts/why/): the ideas behind the types, and what a type guarantees.
- [Guides](https://thomasblanchet.github.io/astrolabe/docs/guides/schemas/): one page for each feature.
- [Reference](https://thomasblanchet.github.io/astrolabe/docs/reference/api/): the API, the operations, the data types, the error codes, the settings and the command line.
- [Specification](https://thomasblanchet.github.io/astrolabe/docs/spec/how-to-read-it/): the documents of `spec/`.

Supported releases: Polars `1.39.3` to `2.0`.

## Documents

[`spec/`](spec/) is the type system: what it promises, what its rules are, and what every Polars operation does under it. Start with [`spec/DATAFRAME_SPECIFICATION.md`](spec/DATAFRAME_SPECIFICATION.md). Every claim there is proved on the page or recomputed by a test in [`spec/tests/`](spec/tests/).

[`ARCHITECTURE.md`](ARCHITECTURE.md) is how the plugin is built. [`POLARS_ISSUES.md`](POLARS_ISSUES.md) lists the polars behaviors the checker works around.

## Development

```bash
uv sync
uv run pytest                  # runtime tests, typing cases, doctests, the spec suite
uv run --exact --group numpy pytest && uv sync   # the same suite with numpy installed
uv run mypy src/astrolabe
uv run mypy tests
uv run mypy spec
uv run mypy tools
uv run ruff check .
uv run ruff format --check .
uv run complexipy . --failed
uv run mdformat --check .
(cd spec/lean && lake build --wfail)   # the Lean proofs of the spec
```

`uv run pytest spec` runs the specification's evidence on its own.

The suite runs with numpy and without it, because the type stubs of polars read numpy and some calls type differently. The last `uv sync` removes numpy again.

### The website

`site/` holds the landing page and the documentation, built with Astro and Starlight. The build writes the site to `docs/`, which GitHub Pages serves. See `site/README.md`.

Each example of the documentation is a Python file in `site/examples/`. A run of mypy with the plugin gives the errors and the revealed types that the pages show:

```bash
uv run python -m tools.site_examples           # write site/src/data/examples.json
uv run python -m tools.site_examples --check   # fail if the file differs from a new run
```

### Mutation testing

`tools/mutation/` measures how well the tests pin the checker's rules. It changes one operator, constant or branch in `src/astrolabe`, then runs the tests that cover that line. A mutant that no test kills names a rule that nothing pins down. This is an offline audit, and it is not one of the commands above.

Run the baseline once, with the coverage contexts turned on:

```bash
COVERAGE_FILE=.mutation/baseline.coverage uv run pytest --cov-context=test
```

Then run the harness against that map:

```bash
uv run python -m tools.mutation --data .mutation/baseline.coverage
```

Only a yml case counts as a killer, because it states the answer a user reads. `--all-tests` counts every test, which shows how much of a score the other tests carry. `--scope` mutates one directory instead of the whole package. `--limit` runs a spread of the mutants still to decide. `--workers` sets how many copies of the project run at the same time. `--timeout` sets how long one mutant's tests can run.

Raise `--workers` only after you measure what one worker holds. One worker holds about two gigabytes, whatever the size of the selection, because the run takes the selection in batches. Measure one with:

```bash
/usr/bin/time -l uv run python -m tools.mutation --data .mutation/baseline.coverage \
  --home .mutation/probe --workers 1 --limit 5
```

Each run writes every finished mutant to `results.jsonl` under its home. `--resume` reads that file and runs only what is left, so a run that stops loses nothing. A run also marks each mutated file by its content. If a file changed, a resume stops and names it, because the held results no longer describe the source.

A mutant whose tests pass the limit is undecided: no test killed it, and the rest of its tests did not run. Such a mutant sits in code that many tests reach. To decide them, run the harness again over the same home with a longer limit:

```bash
uv run python -m tools.mutation --data .mutation/baseline.coverage \
  --resume --retry-undecided --timeout 900
```

The run writes one score for each module, and then every surviving mutant. For each survivor, add the test that kills it, or record it in `KNOWN_ISSUES.md`.
