"""Development tools that are not part of the astrolabe package.

A maintainer runs these by hand. The plugin never imports them, and a
user who installs astrolabe does not get them. `mutation/` holds the
mutation harness, which measures how well the tests pin down the rules
of the checker.
"""

from __future__ import annotations
