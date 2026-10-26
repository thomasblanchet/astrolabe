/**
 * The versions that the site names: Astrolabe itself, and what it needs.
 *
 * Change them here when `pyproject.toml` changes. The header of the
 * documentation and the install pages read them. Astrolabe supports each
 * release of Polars from the oldest to the newest.
 */
export const ASTROLABE = '0.1.0';

/** Astrolabe has no stable release yet. The banner of the documentation and the hero say so. */
export const PRERELEASE = 'The API, the messages and the supported versions can change in any release.';
export const POLARS_OLDEST = '1.39.3';
export const POLARS_NEWEST = '2.0';
export const POLARS = `${POLARS_OLDEST} to ${POLARS_NEWEST}`;
export const MYPY = '1.13';
export const PYTHON = '3.12';
