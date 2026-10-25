/**
 * The walkthrough of the landing page: seven steps over one pipeline.
 *
 * Each step has code, the errors that mypy reports on it, and notes. A note
 * belongs to a range of lines. When the reader points at those lines, the
 * note opens next to them, and the frame that the note describes shows as a
 * row of column chips. The errors and the column chips are what mypy prints
 * with the plugin, for the code exactly as shown.
 *
 * The functions below turn a step into markup. The server renders the first
 * step with them, and the script of the page renders every later state.
 */
import { escape, parts, tokens } from './python';

export { parts };

export interface Stage {
  label: string;
  schema: [string, string][];
  added: string[];
  removed: string[];
}

export interface Note {
  from: string;
  to?: string;
  text: string;
  stage?: Stage;
}

export interface Step {
  title: string;
  blurb: string;
  file: string;
  code: string;
  errors: [string, string][];
  stage: Stage | null;
  notes: Note[];
}

const READ = [
  'trades = pl.read_csv(',
  '    "trades.csv",',
  '    schema={',
  '        "symbol": pl.String,',
  '        "price": pl.Float64,',
  '        "quantity": pl.Int64,',
  '    },',
  ')',
].join('\n');
const AS_BEFORE = '# trades as in step 1';
const TRADES: [string, string][] = [
  ['symbol', 'String'],
  ['price', 'Float64'],
  ['quantity', 'Int64'],
];

const TRADES_STAGE: Stage = { label: 'what the checker knows about trades', schema: TRADES, added: [], removed: [] };
const RESULT_STAGE: Stage = {
  label: 'what the checker knows about result',
  schema: [
    ['price', 'Float64'],
    ['quantity', 'Int64'],
    ['notional', 'Float64'],
    ['symbol', 'String'],
  ],
  added: ['notional', 'symbol'],
  removed: [],
};

export const STEPS: Step[] = [
  {
    title: 'Declare the schema at load time',
    blurb: 'Astrolabe reads the schema that you pass to Polars, so the frame has typed columns from the start.',
    file: 'pipeline.py',
    code: 'import polars as pl\n\n' + READ,
    errors: [],
    stage: null,
    notes: [
      { from: 'trades = pl.read_csv(', text: 'A plain `read_csv` call. Nothing here comes from Astrolabe.' },
      {
        from: 'schema={',
        to: '},',
        text: 'The schema gives each column a data type. The checker reads it, so `trades` now has known columns.',
        stage: { label: 'what the checker knows about trades', schema: TRADES, added: ['symbol', 'price', 'quantity'], removed: [] },
      },
    ],
  },
  {
    title: 'Typos are caught',
    blurb: 'A reference to a column that does not exist is a type error in your editor, not an exception at run time.',
    file: 'pipeline.py',
    code: AS_BEFORE + '\ntrades.select("prcie")',
    errors: [['trades.select("prcie")', 'error: Unknown column "prcie"  [unknown-column]']],
    stage: TRADES_STAGE,
    notes: [
      {
        from: 'trades.select("prcie")',
        text: '`select` asks for a column named `prcie`. `trades` has `price`, not `prcie`, so mypy reports the line.',
      },
    ],
  },
  {
    title: 'Every step updates the schema',
    blurb: 'Each operation produces a frame with its own schema, and every later use is checked against it.',
    file: 'pipeline.py',
    code: AS_BEFORE + '\nsymbols = trades.select("symbol", "price")\n\n# ... later in the pipeline\nsymbols.filter(pl.col("quantity") > 0)',
    errors: [['symbols.filter(pl.col("quantity") > 0)', 'error: Unknown column "quantity"  [unknown-column]']],
    stage: TRADES_STAGE,
    notes: [
      {
        from: 'symbols = trades.select("symbol", "price")',
        text: '`select` keeps two columns. The checker knows that `symbols` has no `quantity`.',
        stage: { label: 'what the checker knows about symbols', schema: TRADES, added: [], removed: ['quantity'] },
      },
      {
        from: 'symbols.filter(pl.col("quantity") > 0)',
        text: 'Later, a filter reads `quantity` from `symbols`. The column is gone, and the error points at this line.',
      },
    ],
  },
  {
    title: 'Data types are checked too',
    blurb: 'Each operation needs columns of the right data type, and a column of the wrong type is an error, like a missing one.',
    file: 'pipeline.py',
    code:
      AS_BEFORE +
      '\nvenues = pl.read_csv(\n    "venues.csv",\n    schema={"symbol": pl.Int64, "venue": pl.String},\n)\n\ntrades.filter(pl.col("price"))\ntrades.join(venues, on="symbol")',
    errors: [
      ['trades.filter(pl.col("price"))', 'error: Column "price" is not Boolean, cannot .filter()  [wrong-column-shape]'],
      ['trades.join(venues, on="symbol")', 'error: Join key "symbol" is String on the left and Int64 on the right  [join-key-dtype]'],
    ],
    stage: TRADES_STAGE,
    notes: [
      { from: 'venues = pl.read_csv(', to: ')', text: 'A second file. Here, `symbol` is an `Int64` code, not a `String`.' },
      {
        from: 'trades.filter(pl.col("price"))',
        text: '`filter` keeps the rows where a condition is true, so it needs a `Boolean` column. `price` is a `Float64`.',
      },
      { from: 'trades.join(venues, on="symbol")', text: 'A join compares the keys on the two sides. A `String` key cannot match an `Int64` key.' },
    ],
  },
  {
    title: 'New columns are typed',
    blurb: 'The checker works out the data type of each new column, as Polars does at run time.',
    file: 'pipeline.py',
    code:
      AS_BEFORE +
      '\npriced = trades.with_columns(\n    (pl.col("price") * pl.col("quantity")).alias("notional"),\n)\nby_symbol = priced.group_by("symbol").agg(\n    pl.col("notional").sum(),\n    pl.len(),\n)',
    errors: [],
    stage: TRADES_STAGE,
    notes: [
      {
        from: 'priced = trades.with_columns(',
        to: ')',
        text: 'A `Float64` times an `Int64` gives a `Float64`. `priced` has a new column, `notional`.',
        stage: { label: 'what the checker knows about priced', schema: TRADES.concat([['notional', 'Float64']]), added: ['notional'], removed: [] },
      },
      {
        from: 'by_symbol = priced.group_by("symbol").agg(',
        to: ')',
        text: '`group_by` keeps the key and the aggregates. The sum stays a `Float64`, and `pl.len()` counts rows as a `UInt32`.',
        stage: {
          label: 'what the checker knows about by_symbol',
          schema: [
            ['symbol', 'String'],
            ['price', 'Float64'],
            ['quantity', 'Int64'],
            ['notional', 'Float64'],
            ['len', 'UInt32'],
          ],
          added: ['len'],
          removed: ['price', 'quantity'],
        },
      },
    ],
  },
  {
    title: 'Write generic functions',
    blurb:
      "Astrolabe's rich grammar of generics lets you work with partially known schemas. Annotate a parameter with only the columns that the function reads, and let `Rest` stand for any others.",
    file: 'features.py',
    code:
      'from astrolabe import Columns, TypedDataFrame, other_columns\n\nclass Priced(Columns):\n    price: pl.Float64\n    quantity: pl.Int64\n\nRest = other_columns()\n\ndef notional(df: TypedDataFrame[Priced, Rest]):\n    return df.select(\n        (pl.col("price") * pl.col("quantity")).alias("notional"),\n    )\n\nquotes = pl.read_csv(\n    "quotes.csv",\n    schema={"symbol": pl.String, "price": pl.Float64},\n)\nnotional(quotes)',
    errors: [['notional(quotes)', 'error: Missing required column "quantity"  [missing-column]']],
    stage: null,
    notes: [
      { from: 'class Priced(Columns):', to: 'quantity: pl.Int64', text: 'A `Columns` class names columns and their data types, like a dataclass.' },
      { from: 'Rest = other_columns()', text: '`Rest` stands for all the other columns, the ones that the function does not name.' },
      {
        from: 'def notional(df: TypedDataFrame[Priced, Rest]):',
        text: 'The function takes any frame with `price` and `quantity`. Because of `Rest`, the frame can have other columns too.',
      },
      {
        from: 'quotes = pl.read_csv(',
        to: ')',
        text: '`quotes` has `symbol` and `price`, but no `quantity`.',
        stage: {
          label: 'what the checker knows about quotes',
          schema: [
            ['symbol', 'String'],
            ['price', 'Float64'],
          ],
          added: [],
          removed: [],
        },
      },
      {
        from: 'notional(quotes)',
        text: 'The call gives the function a frame without a column that it needs. The error is at the call, not inside the function.',
      },
    ],
  },
  {
    title: 'Let inference track the rest',
    blurb:
      "The schema of each frame becomes part of the function's signature. Declare the columns that the function adds, and inference works out what the body does to the others. Each caller gets its own columns back.",
    file: 'features.py',
    code:
      'from astrolabe import InferredRemainder, infer\n\nclass WithNotional(Priced):\n    notional: pl.Float64\n\ndef add_notional(\n    df: TypedDataFrame[Priced, Rest],\n) -> TypedDataFrame[WithNotional, InferredRemainder]:\n    return infer(\n        df.with_columns(\n            notional=pl.col("price") * pl.col("quantity"),\n        ),\n    )\n\nresult = add_notional(trades)\nlarge = result.filter(pl.col("notional") > 1000)',
    errors: [],
    stage: null,
    notes: [
      { from: 'class WithNotional(Priced):', to: 'notional: pl.Float64', text: '`WithNotional` takes the columns of `Priced` and adds `notional`.' },
      {
        from: ') -> TypedDataFrame[WithNotional, InferredRemainder]:',
        text: 'The function promises the columns of `WithNotional`. `InferredRemainder` stands for the other columns: the checker reads them from the body.',
      },
      { from: 'return infer(', to: ')', text: '`infer(...)` marks the returned value. The checker makes sure that it holds `WithNotional`.' },
      {
        from: 'result = add_notional(trades)',
        text: '`trades` brings `symbol` as an other column. `result` keeps it, next to the new `notional`.',
        stage: RESULT_STAGE,
      },
      {
        from: 'large = result.filter(pl.col("notional") > 1000)',
        text: '`result` now has a `notional` column of type `Float64`, so later code can use it like any other column.',
        stage: RESULT_STAGE,
      },
    ],
  },
];

/** The first and the last line of a note, as indexes into the lines of the code. */
export function range(step: Step, note: Note): [number, number] {
  const texts = step.code.split('\n').map((text) => text.trim());
  const first = texts.indexOf(note.from);
  const last = note.to === undefined ? first : texts.indexOf(note.to, first);
  return [first, last];
}

/** The index of the note that owns a line, or -1. */
export function noteAt(step: Step, index: number): number {
  return step.notes.findIndex((note) => {
    const [first, last] = range(step, note);
    return index >= first && index <= last;
  });
}

export function status(step: Step): string {
  const count = step.errors.length;
  if (count === 0) return 'mypy · no errors';
  return `mypy · ${count} ${count === 1 ? 'error' : 'errors'}`;
}

/** The frame to show: the stage of the active note, or of the last note before it. */
export function stageAt(step: Step, active: number): Stage | null {
  const last = active < 0 ? step.notes.length - 1 : active;
  let found = step.stage;
  step.notes.slice(0, last + 1).forEach((note) => {
    if (note.stage !== undefined) found = note.stage;
  });
  return found;
}

export function chips(stage: Stage | null): string {
  if (stage === null) return '';
  return stage.schema
    .map(([name, dtype]) => {
      const removed = stage.removed.includes(name);
      const added = stage.added.includes(name);
      const look = removed ? ' removed' : added ? ' added' : '';
      return `<span class="chip${look}"><span class="chip-name">${escape(name)}</span> <span class="chip-type">${escape(dtype)}</span></span>`;
    })
    .join('');
}

function errorLine(text: string): string {
  return `<div class="tl err"><span class="tk-error">  ^ ${escape(text)}</span></div>`;
}

function errorsAfter(step: Step, text: string): string {
  return step.errors
    .filter(([line]) => line === text.trim())
    .map(([, message]) => errorLine(message))
    .join('');
}

/** The code of a step with its errors, for the version of the walkthrough without notes on hover. */
export function inlineLines(step: Step): string {
  return step.code
    .split('\n')
    .map((text) => `<div class="tl">${tokens(text)}</div>${errorsAfter(step, text)}`)
    .join('');
}

/** The code of a step for the panel: each line that has a note can take the pointer and the focus. */
export function panelLines(step: Step): string {
  return step.code
    .split('\n')
    .map((text, index) => {
      const owner = noteAt(step, index);
      const focus = owner >= 0 ? ` tabindex="0" aria-describedby="tour-pop"` : '';
      const hinted = owner >= 0 ? ' hinted' : '';
      return `<div class="tl${hinted}" data-line="${index}" data-owner="${owner}"${focus}>${tokens(text)}</div>${errorsAfter(step, text)}`;
    })
    .join('');
}
