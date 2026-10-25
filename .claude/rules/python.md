---
paths:
  - '**/*.py'
---

# Python code

These rules apply to every Python file: `src/`, `tests/` and `spec/tests/`.

## Structure

- Put each unit of behavior on a class. A free function is allowed only when it is tiny, pure and stateless. Why: a class names its state and its dependencies, and a set of free functions hides them.
- If two classes need the same function, put it on a small class that both use, and inject that class. Do not let two unrelated classes import the same free function.
- Do not use inheritance unless no other design works, for example where an API requires a subclass, as mypy requires for `Plugin`. Prefer composition, a `Protocol`, or even repeated code. Why: a subclass depends on every detail of its base, and a reader must read the whole chain to know what one method does.
- Do not create a `utils.py` or a `helpers.py` file. Why: a module of unrelated functions shows that a real abstraction is missing. Find or make the class that owns the function.
- Give each class one responsibility, and give each method one job. A job can have several steps. If a class has many unrelated methods, split it.
- Construct classes that do the same job in the same way, with the same call shape. A mypy hook is a class with `__call__(self, ctx)`, built with `ClassName()` or `ClassName(dependency)`, even when it needs no argument.
- Prefer dependency injection and a small class per variant (a strategy) to a long `if` chain or a closure.
- Use `@staticmethod` or `@classmethod` only for a method that code outside the instance calls, for example a factory such as `ColumnRef.from_name_and_dtype`. Keep a private helper that only `self` calls as a plain instance method, even if its body does not use `self`.

## Third-party APIs

- Wrap each third-party type in a project class, and keep the raw object private to that class. Do not pass a raw third-party type across more than one module. Examples: `Frame` and `ColumnRef` in `_hooks/domain/` wrap the mypy `Instance` and `TypeInfo`. `CallArgs` in `_hooks/calls/` wraps the call's argument nodes.
- If you already hold a `TypeInfo` or an `Instance`, build the new value from it. Do not look the same thing up again by name. Why: `api.named_generic_type` can crash mypy for a fullname that is not a builtin. Use `Plugin.lookup_fully_qualified` for a new lookup, and keep `named_generic_type` for builtins such as `str`, `object` and `tuple`.
- Derive a third-party fullname from the real object, for example `Target.of(pl.String).fullname`. Do not type the string by hand. Why: a typed string does not follow a rename.

## Code smells

Fix each of these when you find it.

- A long function that does more than one job. Move each job that has its own name and meaning into its own method.
- A chain of private methods that each have one caller and only pass the call on, for example a guard and then a call to the next method. Fold the chain into its caller. Why: the reader must read each docstring of the chain to follow one check.
- Many arguments, especially optional ones. Make a class, and move the arguments into its constructor or into a small parameter object.
- A group of methods that pass the same call-constant values through every level. Wrap that state in a small class for the call, read it as `self.*`, and build the class once for each call. Examples: `_JoinKeyResolver` in `join.py`, `_UnnestSplicer` in `unnest.py`, `_DiagonalColumnReconciler` in `concat.py`. A per-item loop variable stays a method parameter.
- A constructor that takes a full mypy `Context` only to reach `.api`. Take `api` instead. Why: a class that needs a real `Context` cannot have a doctest.
- An expression that must wrap in a complex or awkward way, for example a dense comprehension, a long conditional expression or a call with many nested arguments. Split it into named steps, a loop or a helper, so that each line holds one simple step.
- A deeply nested expression, for example a call inside a call inside a call. Give each inner step a name, on its own line.
- A module-level constant. Put it on the class that owns it as a `ClassVar`.
- Two things that do the same job in different ways. Make them look the same: the same construction, the same registration, the same argument access.
- A function that returns a tuple. If the parts are independent, fetch each one with its own call. A tuple is correct only when the parts are one fact that two or more callers use as a pair.
- A bare `assert` outside tests. `python -O` removes it. For an invariant that mypy cannot see, use `checked_cast` or `checked_not_none` from `_hooks/checked.py`. They raise `TypeError`, and a failure means a bug in this code, not an input that did not match.

## Style

- Prefer short lines and vertical code to long horizontal code, always.
- A call, a signature or a literal either fits on one line, or it puts each item on its own line and ends with a trailing comma. Do not let several items share one indented line between the brackets. `ruff format` keeps a bracket with a trailing comma vertical, so add the comma, then run `uv run ruff format .`.
- If an expression wraps and cannot take that vertical form, for example a comprehension, a long condition or a single argument that spans lines, split it into several statements. Give each part a name in an intermediate variable, on its own line.
- If a constructor call wraps, do not call a method on its result in the same expression. Put the object in a variable, then call the method on its own line: `error = SomeError(...)`, then `error.fail(ctx)`. A short call that fits on one line, such as `SomeError(name).fail(ctx)`, stays as it is.

## The complexity limit

- `complexipy` enforces a cognitive complexity of at most 10. It counts each `if`, loop, `except`, ternary, filtered comprehension and boolean chain, and it adds a cost for each level of nesting. This limit stops deep nesting and tangled conditions.
- `ruff` enforces a mccabe complexity of at most 10. It counts each branch with no cost for nesting. This limit stops a long flat `if` chain, where a strategy class is the better form.
- The limit is a ceiling, not a target. A method can hold a whole short algorithm, for example a loop with a nested guard and two early returns.
- Split a method when a part has its own name and meaning: a concept, a step that other code also uses, or a strategy. Do not split a method only to lower a count when the parts have no name of their own.
- If a function fails either limit, first make it flatter: use early returns, a fallible chain or a class for the state of one call. Do not count by hand: run the tool until it passes.

## Fallible chains

- Write a chain of steps that can each fail as one flat method with `not_none`, `not_falsy` and a boundary decorator from `_hooks/propagate.py`. `not_none(x)` narrows `T | None` to `T`. `not_falsy(x)` also rejects `False`, `0`, `""` and `[]`. Each one aborts the whole method on failure. Why: the steps read as a straight list, with no `if` after each step.
- Use `@propagates(default)` when the caller only needs a default value on failure, for example `@propagates(None)`.
- Use `@explains(default_reason)` when the caller must know why the chain failed. The method then returns its value or a `Refusal`. A step names its own reason with `not_none(x, reason=Reason.DYNAMIC_NAME)`. `not_refused(value)` passes a nested `Refusal` through unchanged. The caller branches on `isinstance(result, Refusal)`.
- Use the same pattern inside a loop that stops at the first failure: `result.append(not_none(step(item)))`.
- If a boolean condition must abort the chain, write `not_falsy(condition)` as one step of the chain.
- Keep a loop that threads more state than one result list as an explicit loop. Pass the state through parameters and return values.
- Keep a guard that does real work, for example one that reports an error before it returns, as a small ordinary private method.

## Doctests

- Give every class, method and function a doctest in its `Example:` section. Why: an example explains a method faster than more prose.
- Build example values with the functions in `tests/support.py`, for example `new_type_fixture()`, `column_ref(...)` and `frame(...)`. A doctest can call them with no import, because the root `conftest.py` puts them in the `doctest_namespace`. If an example needs a shape that `tests/support.py` does not build, add a builder there.
- Do not import `mypy.test.typefixture` in `src/astrolabe`. It is test scaffolding.
- Do not give a runnable example to the `__call__` of a hook that takes a full mypy context. The `.yml` cases cover it. Give doctests to its helpers that work on plain values.
- If a constructed value does not read clearly from its repr, print a short comment that says what it is.

## Types

- `[tool.mypy]` sets `strict = true` for `src/`, `tests/` and `spec/`. Every parameter needs a real annotation, and no function can return `Any` by accident.
- A test fixture parameter takes its type from `tests/hooks/conftest.py`, imported under `TYPE_CHECKING`.
- `tests/test_runtime.py` is the one module with `# mypy: ignore-errors`. It checks runtime behavior that mypy cannot express.
