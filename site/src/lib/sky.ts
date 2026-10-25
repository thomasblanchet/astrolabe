/**
 * The sky of the landing page, computed at build time.
 *
 * The sky has three parts. A tile of still stars repeats over a large disc,
 * and the disc turns around Polaris. Twinkling stars sit on the same disc.
 * The figures turn with the disc, and each one is drawn star by star in its
 * turn. This module gives each part as markup or CSS. The page only places
 * them, so the whole sky is static HTML with CSS animations and no script.
 */
import { FIGURES, POLARIS } from './figures';
import { seeded } from './seeded';

/** The seconds that the whole sky takes for one turn around Polaris. */
export const TURN_SECONDS = 600;

/** The radius of the disc of stars: it covers screens up to about 3500 pixels wide. */
export const DISC_RADIUS = 2000;

const TILE_WIDTH = 1400;
const TILE_HEIGHT = 1000;
const TILE_STARS = 155;
/** Size range and opacity range of the stars at each of three depths. */
const DEPTHS: [[number, number], [number, number]][] = [
  [[1, 1.6], [0.15, 0.35]],
  [[1.4, 2.2], [0.3, 0.55]],
  [[1.8, 3], [0.45, 0.75]],
];
const TWINKLES = 120;
const TWINKLE_RADIUS = 1300;

/** The tile of still stars, as a data URI for `background-image`. */
export function tileUri(): string {
  let circles = '';
  for (let i = 0; i < TILE_STARS; i++) {
    const [[s0, s1], [o0, o1]] = DEPTHS[i % 3];
    const x = seeded(i * 2 + 1) * TILE_WIDTH;
    const y = seeded(i * 2 + 2) * TILE_HEIGHT;
    const size = s0 + seeded(i * 3 + 7) * (s1 - s0);
    const opacity = o0 + seeded(i * 5 + 3) * (o1 - o0);
    circles += `<circle cx='${x.toFixed(1)}' cy='${y.toFixed(1)}' r='${(size / 2).toFixed(2)}' fill='%23f5f3ec' fill-opacity='${opacity.toFixed(2)}'/>`;
  }
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${TILE_WIDTH}' height='${TILE_HEIGHT}'>${circles}</svg>`;
  return `data:image/svg+xml,${svg.replaceAll('<', '%3C').replaceAll('>', '%3E')}`;
}

export interface Twinkle {
  left: number;
  top: number;
  size: number;
  opacity: number;
  delay: number;
}

/** The twinkling stars, in pixels from the top left corner of the disc. */
export function twinkles(): Twinkle[] {
  const out: Twinkle[] = [];
  for (let i = 0; i < TWINKLES; i++) {
    const r = TWINKLE_RADIUS * Math.sqrt(seeded(i * 7 + 11));
    const a = 2 * Math.PI * seeded(i * 7 + 13);
    out.push({
      left: Math.round(DISC_RADIUS + r * Math.cos(a)),
      top: Math.round(DISC_RADIUS + r * Math.sin(a)),
      size: Number((1.4 + seeded(i * 7 + 17) * 1.4).toFixed(2)),
      opacity: Number((0.35 + seeded(i * 7 + 19) * 0.4).toFixed(2)),
      delay: Number(((i * 0.7) % 4).toFixed(1)),
    });
  }
  return out;
}

const SLOT = 8;
const CYCLE = SLOT * FIGURES.length;
const BASE = 0.55;
const PEAK = 1;
const LINE_PEAK = 0.42;
/** The first line of a figure starts this many seconds into its slot. */
const START = 0.3;
/** The seconds to draw one line. */
const STEP = 0.42;
/** The seconds for a star to brighten when its line reaches it. */
const LIGHT = 0.3;
/** The whole figure fades from OFF to GONE, in seconds from the start of its slot. */
const OFF = 5.6;
const GONE = 7.5;

const GOLD = '#d4af37';
const BRIGHT = '#f0c75e';
const WHITE = '#f5f3ec';

function fixed(x: number): string {
  return x.toFixed(2);
}

/** The figures: the markup inside the layer, and the keyframes that it needs. */
export function figureLayer(): { markup: string; keyframes: string } {
  const rules: string[] = [];
  const keyframes = (stops: [number, string][]): string => {
    const name = `cg-${rules.length + 1}`;
    const body = stops.map(([t, css]) => `${((100 * t) / CYCLE).toFixed(3)}%{${css}}`).join('');
    rules.push(`@keyframes ${name}{${body}}`);
    return name;
  };
  let markup = '';
  FIGURES.forEach((figure, i) => {
    const t0 = SLOT * i;
    const off = t0 + OFF;
    const gone = t0 + GONE;
    const lit = new Map<number, number>();
    figure.links.forEach(([a, b], j) => {
      const s = t0 + START + STEP * j;
      if (!lit.has(a)) lit.set(a, s);
      if (!lit.has(b)) lit.set(b, s + STEP);
      const [x1, y1] = figure.stars[a];
      const [x2, y2] = figure.stars[b];
      const length = fixed(Math.hypot(x2 - x1, y2 - y1));
      const name = keyframes([
        [0, `stroke-dashoffset:${length};opacity:0`],
        [s, `stroke-dashoffset:${length};opacity:${LINE_PEAK}`],
        [s + STEP, `stroke-dashoffset:0;opacity:${LINE_PEAK}`],
        [off, `stroke-dashoffset:0;opacity:${LINE_PEAK}`],
        [gone, 'stroke-dashoffset:0;opacity:0'],
        [CYCLE, `stroke-dashoffset:${length};opacity:0`],
      ]);
      markup += `<line class="cg-line" x1="${fixed(x1)}" y1="${fixed(y1)}" x2="${fixed(x2)}" y2="${fixed(y2)}" stroke="${GOLD}" stroke-width="0.8" stroke-dasharray="${length}" style="opacity: 0; animation: ${name} ${CYCLE}s ease-in-out infinite"></line>`;
    });
    figure.stars.forEach(([x, y, r], j) => {
      if (r === 0) return; // Polaris: the layer draws it on its own.
      const on = lit.get(j) ?? t0;
      const name = keyframes([
        [0, `opacity:${BASE}`],
        [on, `opacity:${BASE}`],
        [on + LIGHT, `opacity:${PEAK}`],
        [off, `opacity:${PEAK}`],
        [gone, `opacity:${BASE}`],
        [CYCLE, `opacity:${BASE}`],
      ]);
      markup += `<g style="opacity: ${BASE}; animation: ${name} ${CYCLE}s ease-in-out infinite"><circle cx="${fixed(x)}" cy="${fixed(y)}" r="${fixed(r * 3)}" fill="${WHITE}" opacity="0.08"></circle><circle cx="${fixed(x)}" cy="${fixed(y)}" r="${fixed(r)}" fill="${WHITE}"></circle></g>`;
    });
  });
  const [px, py] = POLARIS;
  const rings = [0, 1, 2]
    .map((k) => `<circle class="cg-pulse" cx="${px}" cy="${py}" r="7" fill="none" stroke="${BRIGHT}" stroke-width="0.9" style="animation-delay: ${-2 * k}s"></circle>`)
    .join('');
  const polaris = `<circle cx="${px}" cy="${py}" r="9" fill="${WHITE}" opacity="0.14"></circle><circle cx="${px}" cy="${py}" r="2.8" fill="${WHITE}"></circle>`;
  return { markup: `<g class="sky-turn">${markup}</g>${rings}${polaris}`, keyframes: rules.join('\n') };
}
