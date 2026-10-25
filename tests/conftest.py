"""Hold the pytest fixtures that every test under `tests/` shares.

It defines no fixture. The `.yml` cases under `tests/typing/` need no
setup here: `pytest-mypy-plugins` registers itself with pytest and
finds those cases on its own. A fixture that tests in several
directories need belongs in this file.
"""
