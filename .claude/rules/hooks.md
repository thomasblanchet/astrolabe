---
paths:
  - src/astrolabe/_hooks/**
  - src/astrolabe/plugin.py
---

# Plugin hooks

These rules apply to the mypy hooks in `src/astrolabe/_hooks/` and to `plugin.py`.

The rule in `CLAUDE.md` that keeps the plugin and `spec/` in step applies to every change here.

## Errors

- Report each check-time error through a class in `_hooks/errors.py`. Call `SomeError(...).fail(ctx)` at the report site. Do not call `ctx.api.fail` with an inline string. Why: each class carries one `CODE`, which a user can filter with `--disable-error-code` or silence with `# type: ignore[code]`.
- Build the message in the class's `__init__` from structured arguments, for example `UnknownColumnError(name)`.
- Give a new error its own code only for a new kind of check. Share a code across similar messages, the way `DuplicateColumnError(operation, name)` covers every duplicate. Each `type_analysis.py` message keeps its own wording under the shared `marker-arity` and `marker-arg` codes.
- `fail(ctx)` accepts any context with `.context` and `.api`: a `MethodContext`, `FunctionContext`, `FunctionSigContext` or `AnalyzeTypeContext`. They share no base class, so `_FailContext` is a `Protocol`.
- When you add or change an error, update each `.yml` case that asserts it. `pytest-mypy-plugins` compares the code in brackets as well as the text.

## Unproven results

- An operation that gives up on a proof names a `Reason` from `_hooks/unproven.py`. Each reason has a `Level`, and `allow_unproven` decides which levels pass. Why: the level depends on why the proof failed, not on which operation failed.
- Report through the gate, never with `.fail`. `UntypedFallback.resolve` gives the plain frame, `UncheckedFallback.substituted` gives an `Unchecked` column, and `UnprovenGate.report` reports any `UnprovenError`. The gate adds the note that names the setting which lets the error pass.
- A path that already reported a hard error gives up with `Reason.REPORTED`, which is always silent. Why: one call must report one error.
- Use `Reason.CHECKER_GAP` only where the user wrote nothing to fix and the plugin does not model the case.
- A new operation that can report gets a key in `UnprovenPolicy.KNOWN_OPERATIONS`. `tests/test_config.py` checks the roster.

## Fallbacks

Before a hook falls back to the plain frame or substitutes `Unchecked`, answer these questions. Also answer them before you write a test that expects a fallback.

1. Does the answer really depend on data that the checker cannot see, for example the body of a `map_elements` function? Or is the implementation incomplete, for example an argument form that the hook does not read yet? If it is incomplete, fix it or open a GitHub issue for it. Do not write a test that makes the gap look intended.
2. If the data type is unknowable, is the name still known from a keyword, an outermost `.alias(...)` or the receiver? Then keep the name under `Unchecked` and keep the rest of the schema. Fall back to the plain frame only when no name is known.
3. Do not treat the current output as the reference answer. Verify against the real polars behavior with a script.

A `[wrong-column-shape]` error names the family that the method really accepts. Verify the family with a script. Do not reuse the label of a nearby family.

## mypy facts

These facts hold for the mypy version that this project uses. Verify a new fact with a script before you rely on it.

- `get_method_hook` receives the fullname of the class that the call goes through, not the class that defines the method. A hook registered with `Target.of(TypedDataFrame, "method")` fires for an inherited method. Add an override on `TypedDataFrame` only when the fallback return type of the base stub is wrong.
- Do not call `ctx.api.defer()` from a dynamic-class hook.
- mypy filters the errors while it tries one side of a binary operator. If a hook reports an error there and the reflected form succeeds, mypy drops the error. There is no reliable way to detect the filter, because a filter is active at every `return` and every annotated assignment.
- mypy expands every type alias in a message, but it prints a named class by its name. In an error, a named `TypedDict` is its name only. In a `reveal_type` note it is the name and every item.
- A `TypeVar` bound cannot refer to another `TypeVar`. A relation between two remainders is therefore a schema item, not a bound.
- A decorator cannot change how mypy checks the body of the function that it decorates. Only the callers see the decorated type.
- The markers are covariant. To verify a variance claim, read `TypeInfo.defn.type_vars[i].variance`. Do not test it with two assignments in one scope, because the first assignment narrows the variable.
- `TypedDataFrame[T]` over a bare `TypeVar` is a `[marker-arg]` error. A generic frame uses `other_columns()`.
- Before you trust a scratch run, verify that the plugin loaded. Run mypy with the config file of this project.
- mypy's cache does not notice a change to a module of the plugin other than `plugin.py`. After you edit `src/`, run a scratch file again with `--no-incremental`, or its answer comes from the cache.
