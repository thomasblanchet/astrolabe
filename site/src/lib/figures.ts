/**
 * The five figures of the landing page, in the coordinates of its sky layer.
 *
 * The layer is 1500 by 800 pixels, centered on the content column. Polaris is
 * at (1030, 345). Each star is `[x, y, r]`, with `r` the radius of its core.
 * Each link joins two stars by their index, and the order of the links is the
 * order in which the figure is drawn. In Ursa Minor, star 0 is Polaris, which
 * the layer draws on its own.
 */
export interface Figure {
  name: string;
  stars: [number, number, number][];
  links: [number, number][];
}

export const POLARIS: [number, number] = [1030, 345];

export const FIGURES: Figure[] = [
  {
    name: 'Ursa Major',
    stars: [[1222.3, 420.55, 1.43], [1194.08, 396.06, 1.43], [1179.38, 368.74, 1.43], [1172.1, 336.22, 1.1], [1198.68, 310.97, 1.43], [1179.07, 267.17, 1.43], [1141.8, 286.63, 1.65]],
    links: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 3]],
  },
  {
    name: 'Ursa Minor',
    stars: [[1030, 345, 0], [1044.37, 365.97, 0.88], [1052.37, 386.49, 0.99], [1057.94, 409.49, 1.04], [1083.48, 416.16, 0.99], [1088.57, 440.63, 1.32], [1063.54, 437.39, 1.54]],
    links: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 3]],
  },
  {
    name: 'Cassiopeia',
    stars: [[858.37, 297.71, 1.43], [896.7, 311.5, 1.54], [890.61, 354, 1.54], [927.33, 368.91, 1.32], [922.85, 410.28, 1.32]],
    links: [[0, 1], [1, 2], [2, 3], [3, 4]],
  },
  {
    name: 'Cygnus',
    stars: [[1347.93, 122.65, 1.76], [1339.19, 163.74, 1.43], [1327.55, 218.51, 1.32], [1296.95, 160.48, 1.32], [1379.87, 160.93, 1.32], [1276.03, 164.62, 1.1], [1397.46, 158.95, 1.1]],
    links: [[0, 1], [1, 2], [5, 3], [3, 1], [1, 4], [4, 6]],
  },
  {
    name: 'Lyra',
    stars: [[1273.01, 572.89, 1.87], [1290.27, 597.93, 1.1], [1313.13, 589.57, 1.1], [1326.28, 622.06, 1.21], [1303.42, 630.42, 1.21]],
    links: [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [4, 1]],
  },
];
