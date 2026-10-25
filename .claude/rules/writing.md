---
paths:
  - '**/*.md'
  - '**/*.py'
---

# Writing

## Simplified Technical English

Write all prose in ASD-STE100 Simplified Technical English (STE). This covers code comments, docstrings, Markdown documents, error messages, setting warnings, command output and agent instructions.

These rules serve the reader. Apply them with judgment. Prose that obeys every rule and reads badly has failed. Where a rule and natural English pull apart, write the sentence that a person says. Then follow the rules as far as that sentence allows.

- `spec/` is exempt from the STE rules. Why: its proofs need defined terms and long chains of logic. The Markdown format rules below apply to `spec/` too.
- Before you write or rewrite a document, a docstring or a comment, load the `simple-english` skill. It has the full rule catalog and a self-check. One docstring needs it as much as a whole document.
- Apply these rules even when the skill is not loaded:
  - Give one fact or one instruction in each sentence.
  - Write at most 20 words in an instruction and at most 25 words in a description.
  - Use the active voice and the simple tenses.
  - Use only the modals can, will and must. Do not use should, may, might, could or would.
  - Where a rule bans a word, keep what that word carried. Use a word the rules allow. Do not drop the meaning to obey the rule. Why: "an argument arrives by name" states a law, and "an argument can arrive by name" states one case of several.
  - Put a condition first: "If the file is missing, create it."
  - Use one term for each concept. `glossary.md` lists the fixed terms.
  - Use the exact technical term where one exists, for example AST or coroutine. STE allows a technical noun. Give the full name once, then use the short form. Why: a vague phrase in place of an exact term makes the reader guess.
  - Do not use a semicolon. Write two sentences.
  - Do not use an em dash. Write two sentences, or use a comma, a colon or parentheses.
  - Write "for example" and "that is", not "e.g." and "i.e.".
  - Delete words that carry no fact, for example simply, just, robust, seamlessly and "in order to".
  - Where a simpler word makes the sentence odd, rewrite the sentence. Do not stop at a word swap. Why: "reading the AST takes care" came from "requires care", and it says something else.
- Write an error message in this order: what is wrong, then the cause if it is known, then the fix as an instruction.

## Comments and docstrings

Write each docstring for a reader who knows Python and knows nothing about this project. That reader has the file open and no other page. Give that reader every fact they need to decide whether this is the code they want.

### What a docstring gives

A name and a type already say what a thing is called and what it takes. A docstring that repeats them adds nothing.

Write about the place the code holds in the design. The list below shows the kind of fact that helps a reader. It is not a template, and it is not a set of headings. Take what fits the code in front of you, and leave out the rest.

- the job it does for the project, and what goes wrong without it
- the work it saves its user, or the thing its user cannot do alone
- the layer it belongs to, what it reads, and what it answers
- the kind of code that has this need, and the situation that creates the need

One clear paragraph often carries all of it. A large class can need two. Do not write a paragraph for each line of the list.

Write in plain words. Do not use a project term that the docstring does not explain. Do not expect the reader to have read another file.

A class carries this context. A method then says what its one step does, and how that step serves the class.

### An example

This is a real class of the plugin, as its docstring stood before this rule. Every sentence is true, and a reader who does not already know the type checker learns nothing.

```
class CallArgs:
    """Read a call's actual arguments by parameter name.

    `ctx.args` and `ctx.callee_arg_names` already index a call's
    arguments by parameter, not by call-site position, whether the
    caller used positional or keyword form. `CallArgs` reads the AST
    node written for each parameter, not `ctx.arg_types`, since a
    parameter's static type is often too general to carry the literal
    value a hook needs.
    """
```

It names three fields of an object the reader has never met. It does not say what the class is for, what needs it, or where it sits. The same class, written for a reader with no context:

```
class CallArgs:
    """Read the argument values a caller typed, for code that checks the call.

    A check often needs the value itself, and not its type. In the
    call `head(3)`, the abstract syntax tree (AST) holds the number 3.
    The type of that argument says only that it is a whole number. A
    check that must know the number therefore reads the AST.

    Reading the AST is not simple. An argument can arrive by name or
    by position. A literal value can sit inside a wrapper call. This
    class does that work once, and answers a plain Python value.

    It sits in the plugin layer, between the type checker and the
    engine that decides the answer. The engine never reads the AST.
    """
```

The second one is longer, and a reader who opens the file cold can use it.

### Rules

- Give every class, method and function a Google-style docstring: the summary and the context above, then `Args:`, `Returns:` and `Raises:` as needed, then `Example:`. `python.md` has the doctest rules.
- Do not repeat a type hint in an `Args:` line.
- Describe the need, not the caller. Write "code that reads the arguments of a checked call", and not the name of a class that calls this one today. Why: a class that lists its callers goes stale when one moves, and it makes a reader open those classes to understand this one.
- Name a class this one uses, because that class is its own concern. Do not name a class that uses it.
- Explain a project term in each docstring that uses it, in one clause, for example "a kind, that is, a data type with its parameters taken away". Why: the reader has only this docstring, and a term that one module defines is unknown in the next.
- Do not use the short name of a helper object as a plain noun, for example "the minter" or "the editor". Say what the object does, or what the code does not need: "it builds no type, so it needs no access to the plugin". Why: the short name means something only to a reader who knows the class.
- Keep the text self-contained. Do not refer to a document, for example `README.md` or a section of `spec/`, from a comment or a docstring. State the fact itself. A reference to another symbol, for example "see `UnprovenGate`", is correct.

### What a docstring must not hold

A docstring describes the design as it stands now. It is not a history of the work.

- Do not tell how the design came about, and do not list the approaches that did not work. If the current approach can look odd to a reader, give the reason in one short sentence.
- Do not describe experiments, spikes or checks that you ran. Do not write that something was "empirically verified" or "confirmed". A statement in the documentation is true, so it needs no such claim.
- Do not name a person, a date or a change, and do not write what the code did before.
- Do not list the technical problems you met while you wrote the code.

## Markdown format

- Do not hard-wrap Markdown. Write each paragraph and each list item on one line. Why: a wrapped paragraph makes each edit rewrap its neighbors, and the diff then hides the real change.
- Keep tables compact, with no padding to align the columns.
- In a table, escape a `|` inside a code span as `\|`. Why: GitHub-flavored Markdown ends a cell at each pipe that is not escaped, even inside a code span.
- Write every mathematical formula in LaTeX: `$...$` inline, and `$$` on its own line before and after a display formula. Do not write a formula in a code span or with Unicode math symbols such as `∈` or `⊆`. Why: one notation reads the same in every document, and GitHub renders it.
- Keep code in code spans, even inside a formula's sentence: an identifier such as `other_columns`, a type such as `pl.Int64`, a citation such as `T-22`.
- Inside a LaTeX formula in a table, write `\mid` for a vertical bar. Why: GitHub-flavored Markdown ends a cell at every `|`, even inside math.
- Run `uv run mdformat .` after you edit Markdown. `.mdformat.toml` sets these rules, and `uv run mdformat --check .` is a gate.
