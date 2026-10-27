// @ts-check
import { readFileSync } from 'node:fs';

import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { ExpressiveCodeTheme } from '@astrojs/starlight/expressive-code';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

/* Plain code blocks use the colors of the checked examples, in both themes. */
const night = ExpressiveCodeTheme.fromJSONString(readFileSync(new URL('./src/styles/code-theme.json', import.meta.url), 'utf-8'));

/* One folder of src/content/docs/docs/ for each section of the sidebar. The
 * `sidebar.order` of a page sets its place in the section. */
const section = (label, directory) => ({ label, items: [{ autogenerate: { directory: `docs/${directory}` } }] });

// https://astro.build/config
export default defineConfig({
  site: 'https://thomasblanchet.github.io/astrolabe',
  base: '/astrolabe',
  /* The specification writes its formulas in LaTeX. */
  markdown: { remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] },
  integrations: [
    starlight({
      title: 'Astrolabe',
      description: 'Astrolabe is a mypy plugin. It checks the column names and the data types of Polars frames and expressions before the code runs.',
      favicon: '/favicon.svg',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/thomasblanchet/astrolabe' }],
      editLink: { baseUrl: 'https://github.com/thomasblanchet/astrolabe/edit/main/site/' },
      customCss: ['katex/dist/katex.min.css', './src/styles/starlight-theme.css'],
      components: {
        Banner: './src/components/docs/Banner.astro',
        SiteTitle: './src/components/docs/SiteTitle.astro',
        PageTitle: './src/components/docs/PageTitle.astro',
        SocialIcons: './src/components/docs/SocialIcons.astro',
      },
      expressiveCode: {
        themes: [night],
        useStarlightDarkModeSwitch: false,
        useStarlightUiThemeColors: false,
        styleOverrides: {
          borderRadius: '10px',
          borderColor: 'rgba(212, 175, 55, 0.22)',
          codeFontFamily: "'JetBrains Mono', ui-monospace, monospace",
          codeFontSize: '0.84rem',
          codeLineHeight: '1.75',
          uiFontFamily: "'JetBrains Mono', ui-monospace, monospace",
          frames: {
            editorTabBarBackground: '#10162a',
            editorActiveTabBackground: '#10162a',
            editorActiveTabForeground: '#9298ad',
            editorActiveTabIndicatorTopColor: 'transparent',
            editorActiveTabIndicatorBottomColor: 'transparent',
            editorTabBarBorderBottomColor: 'rgba(212, 175, 55, 0.13)',
            terminalTitlebarBackground: '#10162a',
            terminalTitlebarForeground: '#9298ad',
            terminalTitlebarBorderBottomColor: 'rgba(212, 175, 55, 0.13)',
            terminalTitlebarDotsOpacity: '0.25',
            frameBoxShadowCssValue: 'none',
            inlineButtonBorder: 'rgba(212, 175, 55, 0.3)',
          },
        },
      },
      sidebar: [
        section('Start here', 'start'),
        section('Tutorial', 'tutorial'),
        section('Concepts', 'concepts'),
        section('Guides', 'guides'),
        section('Reference', 'reference'),
        section('Specification', 'spec'),
      ],
    }),
  ],
});
