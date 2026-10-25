# The Astrolabe website

This directory holds the landing page and the documentation of Astrolabe, built with [Astro](https://astro.build) and [Starlight](https://starlight.astro.build). The build writes the site to `dist/`, which git ignores. The workflow `.github/workflows/pages.yml` builds the site and deploys it to GitHub Pages on each push to `main`.

```sh
npm install
npm run dev      # a local preview, at http://localhost:4321/astrolabe/
npm run build    # writes dist/
npm run preview  # serves the build in dist/
```

## Layout

- `src/pages/index.astro` is the landing page, and `src/components/landing/` holds its sections. `src/lib/sky.ts` computes the sky at build time, and `src/lib/tour.ts` holds the steps of the walkthrough.
- `src/content/docs/docs/` holds the pages of the documentation, one folder for each section of the sidebar.
- `src/content.config.ts` adds the documents of `../spec/` and `../POLARS_ISSUES.md` to the documentation as the Specification section. The site renders them from the repository, with no copy.
- `src/components/docs/` holds the components of the documentation, and the overrides of Starlight.
- `src/styles/tokens.css` holds the colors and the fonts of the whole site, in a dark and a light theme.

## Checked examples

Each example of the documentation is a Python file in `examples/`. A page shows it with `<Example name="..." />`. The errors, the notes and the revealed types come from a run of mypy with the plugin, which writes `src/data/examples.json`. Run it from the root of the repository after a change to an example or to the checker:

```sh
uv run python -m tools.site_examples
```

A line that ends with `# hide` does not show on the page. A `reveal_type(...)` line does not show either: its type becomes the row of columns under the code. A comment after it can mark columns, for example `# added: notional; remainder: venue; dropped: quantity`. The headers `# file:`, `# start:` and `# setting:` give the file name, the number of the first line, and a setting of `[tool.astrolabe]`.
