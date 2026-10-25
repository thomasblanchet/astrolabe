"""Measure how well the tests pin down the rules of the checker, by mutation testing.

Coverage shows that a line ran. It does not show that a test fails when
the line is wrong. This harness makes one small change, a mutant, to
`src/astrolabe`: it flips an operator, a constant or a branch. It then
runs the tests that reach that line, and records whether one of them
fails. A mutant that no test kills points at a rule that nothing checks.

A maintainer runs the harness by hand, as an audit that takes hours. It
is not one of the commands that check each change to the project.
"""

from __future__ import annotations
