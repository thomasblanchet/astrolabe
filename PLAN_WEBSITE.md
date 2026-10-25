# Plan: the website

## Goal

The site in `site/` holds the landing page of Astrolabe and its user documentation.

The landing page tells an engineer what the checker finds, shows it at work on one pipeline, and sends the reader to the documentation or to the install.

In the documentation, a reader can learn the checker from the start, look up one fact during daily work, and decide whether to trust the checker. Each kind of reader has one section. The documentation covers 66 public names, 29 error codes, 5 settings with their overrides, the `astrolabe init` command, about 30 frame operations, the typing rules for expressions, and the four documents of `spec/`.

## Decisions

- The landing page is the page `project/Main.dc.html` of the canvas artifact <https://claude.ai/artifact/6CYAGY52qLSPN5X8Ys49Kf>. Its design and its copy are final, and the port reproduces them.
- The landing page and the documentation share one file of design tokens and three fonts: Spectral for titles, Instrument Sans for text and JetBrains Mono for code. Instrument Sans replaces Inter.
- The documentation has six sections: Start here, Tutorial, Concepts, Guides, Reference and Specification.
- A guide is a topic chapter: one page for each feature, with headings that name tasks.
- The tutorial is one running project. A pipeline over trades data grows from chapter to chapter, as on the landing page.
- The site renders the documents of `spec/` at build time. There is one source and no copy. The pages need math rendering.
- The site holds the user guide. `README.md` keeps the pitch, the setup, one example, the section for developers and a link to the site. When the site covers the content of the README, slim the README and change the line of `CLAUDE.md` that calls the README the user guide.
- Versions of the documentation wait for the first release that breaks the code of users. Then each release gets its own frozen build, in its own folder of `docs/`.

## Landing page

The prototype is the only source of the design. The scripts that computed its sky did not survive, so the page itself holds the reference values: the tile of stars, the coordinates of the figures and their keyframes. Read it with the artifact tool.

| Part | What it does | Port |
| -- | -- | -- |
| Sky | A disc of stars turns around Polaris once in 10 minutes, with 120 twinkling stars. The stars fade out below the fold. The sky scrolls with the page. | One component. The tile of stars stays an inline SVG image. |
| Figures | Ursa Major, Ursa Minor, Cassiopeia, Cygnus and Lyra. Each figure is drawn star by star in turn, in a cycle of 40 seconds, and turns with the stars. Polaris sends out three rings. | The component computes the lines and the keyframes at build time, from the coordinates in the prototype. |
| Navigation and hero | The small icon, the links, the headline, and the buttons "Read the documentation" and "Install". | Static. |
| Value panels (`#why`) | Three panels: "Speed and type safety", "Guardrails and context", "Plain Polars, no overhead". | Static. |
| Walkthrough (`#tour`) | Seven steps over one pipeline. The code panel stays in view next to the text, and the step follows the scroll. A note opens next to a marked line when the pointer is on it. Chips show what the checker knows about the frame. Below 900 px, the notes show under the code. | A plain script in an Astro component. The steps are data in a TypeScript module. |
| Link to the documentation | "Want to go further?" with a button. | Static. |
| Features (`#features`) | Six cards. | Static. |
| Comparison (`#compare`) | Astrolabe, pandera, patito and dataframely. A row is highlighted under the pointer. Below 720 px, each row becomes a grid. | Static. |
| Install (`#install`) | Tabs for uv, pip and Poetry, then `astrolabe init`. | Radio inputs and CSS, with no script. |
| Footer | The name, the licence and the link to the repository. | Static. |

- Keep the copy of the prototype. Change it only when the user asks.
- Before the port, run mypy on each example of the walkthrough. Use its real errors and revealed types.
- Before the port, verify each claim of the comparison table against the current releases of pandera, patito and dataframely.
- When `prefers-reduced-motion` is set, the sky stands still, and the figures show without animation.
- Delete each old component that the new page does not use: `StarField`, `Hero`, `Features`, `CodeExample`, `Install`, `Footer` and `AstrolabeIcon`.
- Write the title and the description of the page with the name "Astrolabe".
- Verify the result with `npm run build` and with headless Chrome screenshots at 1440 px and 390 px, next to the prototype. Also verify the reduced motion and the use with a keyboard.

## Documentation structure

### Start here

| Page | Content | Sources |
| -- | -- | -- |
| Overview | What the checker finds, what it does not find, and how it compares with runtime validators. | The landing page, the README introduction |
| Install and set up | The install, `astrolabe init` and its options, the setup by hand, the case of `mypy.ini`, editors and CI. | README "Setup", `src/astrolabe/_cli.py` |
| Quickstart | One schema, one `select` and one error, in five minutes. | README "Schemas and frame operations" |

### Tutorial

One pipeline over trades data. Each chapter adds one idea to the code of the chapter before it.

1. Load data with a schema.
2. Transform the frame.
3. Write a function that accepts other columns.
4. Let the body decide the return type.
5. Declare reusable expressions.
6. Check the data at run time, at the boundary of the program.
7. Make the checker stricter.

### Concepts

| Page | Content | Sources |
| -- | -- | -- |
| Why type frames statically | The problem, and what a runtime validator checks and when. | The landing page, the introduction of `spec/DATAFRAME_SPECIFICATION.md` |
| Frames as types | Schema, declared columns, remainder, open and closed frames, `Unchecked`. | `spec/DATAFRAME_SPECIFICATION.md` |
| How the checker works | The type of an expression holds its code. The checker computes each result with the typing rules of Polars. mypy holds the types and substitutes them at calls. | README, `ARCHITECTURE.md` at the level of a user |
| Proof and trust | Unproven results, reasons, levels and checker gaps. | README "Configuration" |
| What a type guarantees | What a type proves, and what no schema can see: values, row counts and column order. | `ARCHITECTURE.md` "What this gives up", `spec/OPERATION_CATALOG.md` section 5.4, `POLARS_ISSUES.md` |

### Guides

| Page | Content | Sources |
| -- | -- | -- |
| Schemas | `Columns`, `columns()`, generic schemas, data types held in names. | README "Schemas and frame operations" |
| Loading data | The readers with `schema=`, and `elevate_untyped_select`. | README |
| Frame operations | Each operation and what it checks. | README, `spec/OPERATION_CATALOG.md` |
| Expressions and data types | How the checker computes the data type of a column, and aggregations against row-wise expressions. | README |
| Lazy frames | `TypedLazyFrame`, `Planned` and `collect()`. | README |
| Open frames | `other_columns()`, `InferredRemainder` and the promises at a call. | README "Functions over a partly known frame" |
| Inferred return types | `InferredDataFrame`, `infer()` and `InferredRemainder`. | README "Return types from the body" |
| Declare expression types | The types of the expressions that a function takes and returns: `column()`, `typed_expr()`, `typed_agg()`, `typed_lit()`, `row()`, `some_column()`, the fit at a call, the check at the frame, and the rules of a declaration. | README "Declarations", `spec/EXPRESSION_SPECIFICATION.md` section 7 |
| Selections | `typed_selection()` and renames over a selection. | README "Selections" |
| Python functions and `Unchecked` | The opaque members, and `return_dtype=`. | README "Unchecked" |
| Runtime checks | `assume_schema`, `check_schema` and `force_schema`. | README "The runtime half" |
| Adopting in an existing codebase | Plain frames stay untracked. Type the frames at the boundaries first, then raise the level step by step. | New. Verify each step with a run. |

### Reference

| Page | Content | Sources |
| -- | -- | -- |
| API | The 66 public names in groups: frame types, schemas, remainders, data-type markers, declarations, inferred returns, runtime functions, and last the markers that the checker writes in revealed types. | The docstrings in `src/astrolabe/` |
| Operations | One row for each operation: what it checks, what it gives and the reasons it can report. | README, `spec/OPERATION_CATALOG.md` |
| Data types | Each Polars data type and the marker that names it in a type, for example `Datetime[Literal["us"], None]`. | README, `spec/DATAFRAME_SPECIFICATION.md` |
| Remainders | `other_columns()` and its promises, then the markers that the checker writes in revealed types: `Minus`, `Only`, `Except`, `Suffixed`, `Prefixed`, `Retyped`, `Both`, `SplitColumns` and the selector markers. A program does not write the markers. | README, `spec/DATAFRAME_SPECIFICATION.md` section 5, `src/astrolabe/remainder.py` |
| Error codes | The 29 codes. Each one has its message, its cause, its fix and the level at which it passes. | `src/astrolabe/_hooks/errors.py` |
| Configuration | The 5 settings and their overrides. | README "Configuration", `src/astrolabe/_hooks/config.py` |
| CLI | `astrolabe init` and its options. | README "Setup", `src/astrolabe/_cli.py` |

### Specification

The site renders these pages from `spec/` at build time. They keep their own style, because `spec/` is exempt from the writing rules.

| Page | Source |
| -- | -- |
| How to read it | `spec/README.md` |
| Frames | `spec/DATAFRAME_SPECIFICATION.md` |
| Expressions | `spec/EXPRESSION_SPECIFICATION.md` |
| Rules and proofs | `spec/OPERATIONS.md` and `spec/lean/` |
| Operation catalog | `spec/OPERATION_CATALOG.md` |
| Polars behaviors | `POLARS_ISSUES.md` |

## Rules for every page

- Write each page for a software or data engineer who knows static type checking and does not know Astrolabe.
- Follow `.claude/rules/writing.md` and the terms of `.claude/rules/glossary.md`. The pages of the specification are exempt.
- Do not use the star or navigation metaphor in the text. The visual design carries it.
- Give each fact one home. A page links to the home of a fact and does not copy it.
- Make each example real: its errors, notes and revealed types are what mypy prints with the plugin.
- Follow the design boards of the canvas artifact <https://claude.ai/artifact/6CYAGY52qLSPN5X8Ys49Kf>: the home page, the guide page in its dark and light themes, the reference pages, and the tokens and components.

## Open items

- Versions of the documentation: see Decisions.

## Status

- Landing page: done. The port reproduces the prototype, with five additions from the open items: keyboard access to the notes of the walkthrough, a copy button on the install card, a link preview image, a link to the specification in the footer, and a line with the requirements under the hero.
- Documentation: done. Start here, the Tutorial, the Concepts, the Guides and the Reference are written. The Specification pages render `spec/` and `POLARS_ISSUES.md` at build time, with math.
- Checked examples: done. `tools/site_examples` runs mypy over `site/examples/`, and the gates of `CLAUDE.md` run its `--check`.
- README: slimmed, and `CLAUDE.md` names the site as the user guide.
