import { readFile } from 'node:fs/promises';

import { defineCollection } from 'astro:content';
import type { Loader, LoaderContext } from 'astro/loaders';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

/**
 * The documents of the specification, read from the repository at build time.
 *
 * The specification lives in `spec/`, beside the code that it describes. The
 * site renders each document as a page of the documentation, so there is one
 * source and no copy. A link from one document to another becomes a link
 * between the pages. A link to any other file of the repository goes to
 * GitHub.
 */
const REPOSITORY = 'https://github.com/thomasblanchet/astrolabe';

const SPEC = [
  { file: 'spec/README.md', slug: 'how-to-read-it', title: 'How to read it' },
  { file: 'spec/DATAFRAME_SPECIFICATION.md', slug: 'frames', title: 'Frames' },
  { file: 'spec/EXPRESSION_SPECIFICATION.md', slug: 'expressions', title: 'Expressions' },
  { file: 'spec/OPERATIONS.md', slug: 'rules-and-proofs', title: 'Rules and proofs' },
  { file: 'spec/OPERATION_CATALOG.md', slug: 'operation-catalog', title: 'Operation catalog' },
  { file: 'spec/POLARS_BEHAVIORS.md', slug: 'polars-behaviors', title: 'Polars behaviors' },
];

const ROOT = new URL('../../', import.meta.url);

function resolve(from: string, target: string): string {
  const base = from.includes('/') ? from.slice(0, from.lastIndexOf('/') + 1) : '';
  const parts: string[] = [];
  for (const part of (base + target).split('/')) {
    if (part === '..') parts.pop();
    else if (part !== '.' && part !== '') parts.push(part);
  }
  return parts.join('/');
}

/**
 * Rewrite the links of a document, in its prose only.
 *
 * Code holds text such as `class Keyed[T: pl.DataType](Columns)`, which reads
 * like a link and is not one. So fenced code blocks and inline code stay as
 * they are. A split on a pattern with one group puts the code at the odd
 * places of the result.
 */
function relink(from: string, markdown: string): string {
  return markdown
    .split(/(^```[^\n]*\n[\s\S]*?^```[ \t]*$)/m)
    .map((block, index) => (index % 2 === 1 ? block : relinkProse(from, block)))
    .join('');
}

function relinkProse(from: string, text: string): string {
  return text
    .split(/(`[^`\n]*`)/)
    .map((part, index) => (index % 2 === 1 ? part : part.replace(/\]\(([^)\s]+)\)/g, (whole, target: string) => relinked(from, whole, target))))
    .join('');
}

function relinked(from: string, whole: string, target: string): string {
  if (/^[a-z]+:/.test(target) || target.startsWith('#')) return whole;
  const [path, anchor] = target.split('#');
  const file = resolve(from, path);
  const page = SPEC.find((doc) => doc.file === file);
  if (page !== undefined) return `](../${page.slug}/${anchor ? `#${anchor}` : ''})`;
  return `](${REPOSITORY}/blob/main/${file}${anchor ? `#${anchor}` : ''})`;
}

async function loadSpec(context: LoaderContext): Promise<void> {
  for (const [order, doc] of SPEC.entries()) {
    const text = await readFile(new URL(doc.file, ROOT), 'utf-8');
    const body = relink(doc.file, text.replace(/^# .*\n+/, ''));
    const id = `docs/spec/${doc.slug}`;
    const data = await context.parseData({
      id,
      data: {
        title: doc.title,
        sidebar: { order: order + 1 },
        editUrl: `${REPOSITORY}/edit/main/${doc.file}`,
      },
    });
    const rendered = await context.renderMarkdown(body);
    const filePath = `../${doc.file}`;
    context.store.set({ id, data, body, rendered, filePath, digest: context.generateDigest(text) });
  }
}

const docsAndSpec: Loader = {
  name: 'astrolabe-docs-and-spec',
  load: async (context) => {
    await docsLoader().load(context);
    await loadSpec(context);
  },
};

export const collections = {
  docs: defineCollection({ loader: docsAndSpec, schema: docsSchema() }),
};
