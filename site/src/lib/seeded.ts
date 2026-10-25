/**
 * A pseudo-random number in [0, 1) for each integer, the same on every build.
 *
 * The sky of the landing page places its stars with this function, so the
 * page does not change between two builds of the same source.
 */
export function seeded(n: number): number {
  const x = Math.sin(n * 9301 + 49297) * 233280;
  return x - Math.floor(x);
}
