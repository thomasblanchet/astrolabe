"""Root pytest configuration.

This file exposes `tests/support.py`'s builder functions to every
doctest in `src/astrolabe`, through pytest's `doctest_namespace`
fixture. A doctest can call `new_type_fixture()`, `column_ref(...)`, and
the other functions in `tests/support.py` without an `import` line.

It also gives every `pytest-mypy-plugins` `.yml` case its own, private
incremental cache directory. Left at its default, `--mypy-testing-base`
resolves to the whole machine's shared OS temp directory, so every
`.yml` case's own `--cache-dir` lands in one `.mypy_cache` folder there.
A single pytest process runs every case one at a time, so this default
is safe under plain `uv run pytest`. It stops being safe under
`pytest-xdist`, where several worker processes run cases at once: every
worker would read, write, and delete files in that same shared
`.mypy_cache` folder concurrently, corrupting it the same way a
concurrent `--cov` run corrupts a shared `.coverage` file. A directory
built fresh per test session, one per worker under `pytest-xdist`, gives
every worker its own cache with no shared state at all.

It also changes how `pytest-mypy-plugins` restores the module table
after each case. The plugin puts a new dictionary in `sys.modules`. The
import system of Python keeps its own reference to the old dictionary,
so the two tables then disagree. A later import of a package that
imports its submodules from C code, such as numpy, then fails with a
`KeyError`. `ModuleTableRestore` puts the saved entries back into the
same dictionary.

The restore also collects the garbage of the cases, after every tenth
case. mypy tunes the garbage collector for one run: it collects rarely,
and it moves the objects of a build to the oldest generation, which the
collector visits least. So the objects of each finished case stay in
memory, and a process that runs hundreds of cases grows by gigabytes. A
collection visits every object and takes a large part of a second, so it
does not run after each case.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING, ClassVar

import pytest
from pytest_mypy_plugins import utils

from tests import support

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping
    from types import ModuleType


class ModuleTableRestore:
    """Restore `sys.modules` after one `.yml` case, and keep the same dictionary.

    A case runs mypy in this process, and mypy imports the plugin and
    its modules. The restore takes away the modules that the case added,
    so that each case starts from the same state. It keeps the modules
    of `astrolabe`: the plugin reads its settings and builds its hooks
    again for each mypy run, and the values that its classes keep
    depend on neither, so a second import gives nothing new. To import
    the plugin again for each case costs about a third of a second, and
    each discarded copy stays in memory. The restore changes the
    entries of the dictionary and never replaces the dictionary,
    because the import system holds a reference to it. After every
    `COLLECT`-th case, it collects the garbage of the cases, which mypy
    leaves in the oldest generation of the garbage collector.

    A module that compiled mypy imports once and then expects to stay
    loaded must be loaded before the first case, so that the restore
    keeps it. `sqlite3`, which holds mypy's cache, is such a module.

    Example:
        >>> table = sys.modules
        >>> with ModuleTableRestore().restored():
        ...     sys.modules["astrolabe_example_module"] = sys
        ...     sys.modules["astrolabe.example_module"] = sys
        >>> sys.modules is table, "astrolabe_example_module" in sys.modules
        (True, False)
        >>> sys.modules.pop("astrolabe.example_module") is sys
        True
    """

    COLLECT: ClassVar[int] = 10
    cases: ClassVar[list[int]] = [0]

    @contextlib.contextmanager
    def restored(self) -> Iterator[None]:
        """Save the entries of `sys.modules`, and put them back at the end.

        Yields:
            Nothing. The body runs with the table as it is.
        """
        saved = sys.modules.copy()
        try:
            yield
        finally:
            kept = self._plugin_modules()
            sys.modules.clear()
            sys.modules.update(saved)
            sys.modules.update(kept)
            self.cases[0] += 1
            if self.cases[0] % self.COLLECT == 0:
                gc.collect()

    def _plugin_modules(self) -> dict[str, ModuleType]:
        """Give the modules of `astrolabe` that are loaded now.

        Returns:
            Each module of the package, by its name.

        Example:
            >>> "astrolabe" in ModuleTableRestore()._plugin_modules()
            True
        """
        return {
            name: module
            for name, module in sys.modules.items()
            if name == "astrolabe" or name.startswith("astrolabe.")
        }


@pytest.fixture(autouse=True)
def _add_support_to_doctest_namespace(
    doctest_namespace: MutableMapping[str, object],
) -> None:
    for name in dir(support):
        if not name.startswith("_"):
            doctest_namespace[name] = getattr(support, name)


def pytest_configure(config: pytest.Config) -> None:
    """Give this pytest process its own private `pytest-mypy-plugins` testing base.

    It also installs `ModuleTableRestore` in place of the restore of
    `sys.modules` that `pytest-mypy-plugins` runs after each case.

    Args:
        config: The pytest session configuration. Its
            `mypy_testing_base` option is overwritten with a fresh
            directory, private to this process.
    """
    importlib.import_module("sqlite3")
    # The package types its own restore with the context manager class
    # of the `decorator` library, which is not the class of `contextlib`.
    utils.temp_sys_modules = ModuleTableRestore().restored  # type: ignore[assignment]
    testing_base = tempfile.mkdtemp(prefix="astrolabe-mypy-testing-")
    config.option.mypy_testing_base = testing_base
    config.add_cleanup(lambda: shutil.rmtree(testing_base, ignore_errors=True))
