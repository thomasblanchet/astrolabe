"""The declaration-class encoding survives mypy's incremental cache.

The rewrite stores a remainder declaration's requirements on the
declared class itself: `other_columns(...)` registers a real class
whose extra bases are requirement markers carrying literal
arguments. That only works if a `TypeInfo` a plugin builds this way
serializes into the incremental cache and comes back with its bases
intact, because a warm run must recover every requirement from the
`TypeInfo` alone, never from plugin memory.

The probe builds such a class in a throwaway plugin, then reads the
requirement back out of the argument type's `TypeInfo` in a hook,
and runs mypy twice over one cache directory. The second run
re-checks only the reader module, so the declaration arrives through
deserialization, and the recovered requirement must not change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy import api

if TYPE_CHECKING:
    from pathlib import Path

DECLARATION_PLUGIN = """
from mypy.nodes import GDEF, SymbolTableNode, TypeInfo
from mypy.plugin import DynamicClassDefContext, FunctionContext, Plugin
from mypy.types import Instance, LiteralType


def declare(ctx: DynamicClassDefContext) -> None:
    obj = ctx.api.named_type("builtins.object")
    info = ctx.api.basic_new_typeinfo(ctx.name, obj, ctx.call.line)
    marker = ctx.api.named_type("decl.Without").type
    literal = LiteralType("ticker", ctx.api.named_type("builtins.str"))
    info.bases = [obj, Instance(marker, [literal])]
    info.mro = [info, marker, obj.type]
    ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, info))


def read_requirement(ctx: FunctionContext):
    if not ctx.arg_types or not ctx.arg_types[0]:
        return ctx.default_return_type
    argument = ctx.arg_types[0][0]
    if isinstance(argument, Instance):
        for base in argument.type.bases:
            for arg in base.args:
                if isinstance(arg, LiteralType):
                    ctx.api.fail(f"requirement: {arg.value}", ctx.context)
    return ctx.default_return_type


class DeclarationPlugin(Plugin):
    def get_dynamic_class_hook(self, fullname: str):
        if fullname == "decl.other_columns":
            return declare
        return None

    def get_function_hook(self, fullname: str):
        if fullname == "decl.probe_requirements":
            return read_requirement
        return None


def plugin(version: str):
    return DeclarationPlugin
"""

DECLARATION_MODULE = """
from typing import Generic, TypeVar

S = TypeVar("S")


class Without(Generic[S]): ...


def other_columns() -> type:
    raise NotImplementedError


def probe_requirements(value: object) -> None:
    raise NotImplementedError


R = other_columns()
"""

READER_MODULE = """
import decl


def held(value: "decl.R") -> None:
    decl.probe_requirements(value)
"""


class DeclarationProbe:
    """Two mypy runs over one cache, the declaring plugin loaded."""

    def reports(self, tmp_path: Path) -> tuple[str, str]:
        """Run cold and warm, and give both full reports.

        The warm run touches only the reader module, so the declared
        class comes back from the incremental cache.

        Args:
            tmp_path: A directory for the plugin, config, cache and
                modules.

        Returns:
            The cold run's report and the warm run's.
        """
        paths = self._written(tmp_path)
        cold = self._run(tmp_path, paths)
        reader = tmp_path / "reader.py"
        reader.write_text(reader.read_text() + "\n# touched\n")
        warm = self._run(tmp_path, paths)
        return cold, warm

    def _written(self, tmp_path: Path) -> list[str]:
        """Write the plugin, the config and the modules.

        Args:
            tmp_path: The probe's directory.

        Returns:
            The module paths, in a fixed order.
        """
        plugin_file = tmp_path / "declaration_probe.py"
        plugin_file.write_text(DECLARATION_PLUGIN)
        ini = tmp_path / "mypy.ini"
        ini.write_text(f"[mypy]\nplugins = {plugin_file}\n")
        (tmp_path / "decl.py").write_text(DECLARATION_MODULE)
        (tmp_path / "reader.py").write_text(READER_MODULE)
        return [str(tmp_path / "decl.py"), str(tmp_path / "reader.py")]

    def _run(self, tmp_path: Path, paths: list[str]) -> str:
        """Run mypy once over the modules, cache shared.

        Args:
            tmp_path: The probe's directory, holding config and
                cache.
            paths: The module paths to check.

        Returns:
            Everything mypy printed.
        """
        out, err, _ = api.run([
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-error-summary",
            "--config-file",
            str(tmp_path / "mypy.ini"),
            *paths,
        ])
        return out + err


class TestDeclarationSurvivesTheCache:
    """A plugin-built class keeps its requirement bases when cached."""

    def test_the_requirement_reads_back_cold_and_warm(self, tmp_path: Path) -> None:
        """Both runs recover the same requirement from the bases.

        The probing hook deliberately fails with the recovered
        literal, so the report carries the proof: the requirement
        was read out of the `TypeInfo` the argument type holds, in
        the warm run out of the deserialized one.
        """
        cold, warm = DeclarationProbe().reports(tmp_path)
        assert "requirement: ticker" in cold, cold
        assert "requirement: ticker" in warm, warm
        assert "cannot be resolved" not in warm, warm
