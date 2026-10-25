"""Tests of the mutation harness in `tools/mutation`.

The harness makes one small change, a mutant, to the source of the
plugin, runs the tests that reach the changed line, and records
whether a test fails. A mutant that no test catches shows a rule that
no test pins down. These tests run the harness on a small toy project,
so they finish fast and never touch the real source.
"""

from __future__ import annotations
