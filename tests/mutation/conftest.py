"""Give the tests of the mutation harness a small toy project to copy, change and test.

The harness copies a whole project for each worker, changes one line,
and runs pytest there. The real project is too large and too slow for
a unit test. The `toy` fixture writes a project of one small module
and one test file into a temporary directory, with the same layout as
the real project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

TOY_SOURCE = '''"""A toy module."""

from __future__ import annotations


class Toy:
    """A toy class."""

    def positive(self, value: int) -> bool:
        """Answer whether a value is positive.

        Example:
            >>> Toy().positive(1)
            True
        """
        return value > 0

    def clamp(self, value: int) -> int:
        """Hold a value under a ceiling.

        Example:
            >>> Toy().clamp(5)
            5
        """
        if value > 100:
            return 100
        return value
'''

TOY_TEST = """from __future__ import annotations

from toy import Toy


def test_positive() -> None:
    assert Toy().positive(1)
    assert not Toy().positive(0)


def test_clamp() -> None:
    assert Toy().clamp(5) == 5
"""


@pytest.fixture
def toy(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "spec").mkdir()
    (root / "src" / "toy.py").write_text(TOY_SOURCE, encoding="utf-8")
    (root / "tests" / "test_toy.py").write_text(TOY_TEST, encoding="utf-8")
    (root / "spec" / "__init__.py").write_text('"""Toy spec."""\n', encoding="utf-8")
    (root / "conftest.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[tool.mypy]\nstrict = false\n", encoding="utf-8")
    return root
