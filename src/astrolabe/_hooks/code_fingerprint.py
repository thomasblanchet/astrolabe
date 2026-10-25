"""A fingerprint of the source of the plugin, which tells mypy when its cache is out of date."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

import astrolabe


class CodeFingerprint:
    """Give one text that changes whenever a source file of the plugin changes.

    mypy reuses the cached result of a module while the value that the plugin
    reports for it stays the same. A change to a rule of the plugin changes
    its answers, so the value must change with the code. mypy itself hashes
    only the module that loads the plugin, and the rules live in other
    modules. This class hashes the name and the bytes of each Python file of
    the package, in a fixed order.

    Example:
        >>> len(CodeFingerprint().digest())
        64
    """

    def digest(self) -> str:
        """Give the fingerprint of the installed package.

        The package does not change while mypy runs, so the method reads the
        files once for each process.

        Returns:
            The hex digest.

        Example:
            >>> CodeFingerprint().digest() == CodeFingerprint().digest()
            True
        """
        return _package_digest()

    def of(self, root: Path) -> str:
        """Give the fingerprint of the Python files below a directory.

        Args:
            root: The directory.

        Returns:
            The hex digest of the relative path and the bytes of each file.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as folder:
            ...     before = CodeFingerprint().of(Path(folder))
            ...     _ = (Path(folder) / "rule.py").write_text("x = 1")
            ...     before != CodeFingerprint().of(Path(folder))
            True
        """
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()


@cache
def _package_digest() -> str:
    """Give the fingerprint of the installed package, once for each process.

    Returns:
        The hex digest.

    Example:
        >>> len(_package_digest())
        64
    """
    return CodeFingerprint().of(Path(astrolabe.__file__).parent)
