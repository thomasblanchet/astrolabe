"""How mypy types a variable a loop rebinds, measured.

The convergence theorem for dynamically built expressions rests on
three behaviours of the checker itself, not of polars. By default an
unannotated variable is pinned to its first assignment's type. With
redefinition allowed, the variable's type after a loop is the union
accumulated over a bounded re-analysis of the body. And that bound is
real: a step whose result type strictly grows is truncated, so the
union misses the runtime type of a longer loop. Each behaviour is
measured here against the installed mypy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy import api

if TYPE_CHECKING:
    from pathlib import Path

GROWING = """
from typing import Generic, TypeVar

T = TypeVar("T")

class Leaf: ...
class Wrap(Generic[T]): ...

def wrap(x: T) -> "Wrap[T]":
    return Wrap()

def build() -> None:
    e = Leaf()
    for _ in range(10):
        e = wrap(e)
    reveal_type(e)
"""

CONVERGING = """
class Leaf: ...
class Chain: ...

def step(x: "Leaf | Chain") -> "Chain":
    return Chain()

def build() -> None:
    e = Leaf()
    for _ in range(10):
        e = step(e)
    reveal_type(e)
"""


class BinderProbe:
    """One mypy run over one snippet, under chosen flags.

    Example:
        >>> BinderProbe().depth_of("Leaf | Wrap[Leaf | Wrap[Leaf]]")
        2
    """

    REDEFINITION_MODES: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("--allow-redefinition",),
        ("--allow-redefinition-new", "--local-partial-types"),
    )

    def report(self, source: str, tmp_path: Path, *flags: str) -> str:
        """Run mypy over a snippet and give its whole report.

        Args:
            source: The snippet to check.
            tmp_path: A directory for the snippet and mypy's cache.
            *flags: Extra mypy flags.

        Returns:
            Everything mypy printed.
        """
        snippet = tmp_path / "snippet.py"
        snippet.write_text(source)
        cache = ("--cache-dir", str(tmp_path / "cache"), "--no-error-summary")
        isolated = self._isolating_config(tmp_path, flags)
        out, err, _ = api.run(["--strict", *cache, *isolated, *flags, str(snippet)])
        return out + err

    def _isolating_config(self, tmp_path: Path, flags: tuple[str, ...]) -> tuple[str, ...]:
        """Give a config flag shielding a probe from the repository's own.

        The repository's `pyproject.toml` enables `allow_redefinition`,
        and a probe measuring the default binder must not inherit it,
        so a bare probe runs against an empty configuration. A caller
        supplying its own `--config-file` keeps it.

        Args:
            tmp_path: A directory for the empty configuration.
            flags: The caller's extra flags.

        Returns:
            The isolating flags, or nothing when the caller configures.
        """
        if "--config-file" in flags:
            return ()
        ini = tmp_path / "isolated.ini"
        ini.write_text("[mypy]\n")
        return ("--config-file", str(ini))

    def plugin_report(self, source: str, tmp_path: Path) -> str:
        """Run mypy with the astrolabe plugin over a snippet.

        Args:
            source: The snippet to check.
            tmp_path: A directory for the snippet, config and cache.

        Returns:
            Everything mypy printed.
        """
        ini = tmp_path / "mypy.ini"
        ini.write_text("[mypy]\nplugins = astrolabe.plugin\n")
        return self.report(source, tmp_path, "--config-file", str(ini))

    def revealed(self, report: str) -> str:
        """Give the revealed type a report holds.

        Args:
            report: A full mypy report holding one revealed type note.

        Returns:
            The revealed type's text.
        """
        marker = 'Revealed type is "'
        start = report.index(marker) + len(marker)
        return report[start : report.index('"', start)]

    def depth_of(self, revealed: str) -> int:
        """Give the deepest `Wrap[...]` nesting in a revealed type.

        Each opening bracket goes one level deeper, and each closing
        bracket comes one level back.

        Args:
            revealed: The revealed type's text.

        Returns:
            The nesting depth, 0 for no wrapping at all.

        Example:
            >>> BinderProbe().depth_of("Leaf")
            0
            >>> BinderProbe().depth_of("Wrap[Leaf] | Wrap[Leaf]")
            1
        """
        deepest = depth = 0
        for char in revealed.replace("Wrap[", "["):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            deepest = max(deepest, depth)
        return deepest


class TestDefaultBinding:
    """Without redefinition, the first assignment's type is the contract."""

    def test_a_growing_step_is_rejected_at_the_rebinding(self, tmp_path: Path) -> None:
        """The loop assignment must fit the seed's own type, and does not."""
        report = BinderProbe().report(GROWING, tmp_path)
        assert 'expression has type "Wrap[Leaf]", variable has type "Leaf"' in report


class TestRedefinitionBinding:
    """With redefinition allowed, the loop accumulates a bounded union."""

    def test_a_growing_step_is_truncated(self, tmp_path: Path) -> None:
        """The union stops four levels deep, under a ten-iteration loop.

        The runtime value's type is ten levels deep, so the union is a
        truncation: the inferred type does not contain the value. This
        is the failure the convergence requirement exists to make
        unreachable.
        """
        probe = BinderProbe()
        for flags in BinderProbe.REDEFINITION_MODES:
            report = probe.report(GROWING, tmp_path, *flags)
            assert "error" not in report
            assert probe.depth_of(probe.revealed(report)) == 4

    def test_an_idempotent_step_reaches_its_fixed_point(self, tmp_path: Path) -> None:
        """A step mapping the union back into itself infers exactly the union."""
        probe = BinderProbe()
        for flags in BinderProbe.REDEFINITION_MODES:
            report = probe.report(CONVERGING, tmp_path, *flags)
            assert "error" not in report
            assert probe.revealed(report) == "snippet.Leaf | snippet.Chain"


CALL_ANNOTATION = """
def make(name: str) -> object: ...
def f(x: make("q")) -> None: ...
"""


class TestAnnotationSurface:
    """What mypy admits in annotation position, measured.

    The declaration surface writes slice declarations as named values.
    This probe pins the constraint that forces that shape: a literal
    call in annotation position is refused before any plugin hook
    runs, so a name, a class subscription, or a string are the only
    spellings a surface can build on.
    """

    def test_a_call_expression_in_annotation_position_is_refused(self, tmp_path: Path) -> None:
        """The checker rejects `x: make("q")` with no plugin recourse."""
        report = BinderProbe().report(CALL_ANNOTATION, tmp_path)
        assert "Invalid type comment or annotation" in report


POWER = """
from typing import Final

SHIFT: Final = 9
reveal_type(2 ** (SHIFT + 1))
"""


class TestPowerOfLiterals:
    """mypy types a power whose exponent is not a literal as `Any`.

    The stub of `int.__pow__` answers `Any` for an `int` exponent that
    is not a literal, so `2 ** (SHIFT + 1)` is `Any`, and mypy calls no
    hook for an operation with an `Any` operand. The checker computes
    the value, so it gives the power the type of that value back.
    """

    def test_a_power_with_a_computed_exponent_is_any(self, tmp_path: Path) -> None:
        """Without the plugin, the power reveals `Any`."""
        probe = BinderProbe()
        assert probe.revealed(probe.report(POWER, tmp_path)) == "Any"

    def test_the_plugin_gives_the_power_the_type_of_its_value(self, tmp_path: Path) -> None:
        """With the plugin, the power reveals `int`."""
        probe = BinderProbe()
        assert probe.revealed(probe.plugin_report(POWER, tmp_path)) == "int"


FIT_GAP = """
from typing import Literal
import polars as pl
from astrolabe import TypedExpr

def declared(e: TypedExpr[Literal["price"], pl.Float64]) -> None: ...
def plain(e: pl.Expr) -> None: ...

declared(pl.col("price"))
plain(pl.col("price") + 1)
"""


class TestFitIsNotSubtyping:
    """The checker's own subtype order cannot decide fit.

    Expression types declare no subtype hierarchy: mypy compares the
    minted types' arguments nominally, so without a per-call fit
    computation even an identity expression is refused against its
    own declared point. The one rung ordinary subtyping supplies is
    the plain `pl.Expr` parameter, which every typed expression
    satisfies by subclassing.
    """

    def test_without_a_fit_hook_the_declared_call_is_refused(self, tmp_path: Path) -> None:
        """The declared call errors; the plain-parameter call passes."""
        report = BinderProbe().plugin_report(FIT_GAP, tmp_path)
        assert 'Argument 1 to "declared" has incompatible type' in report
        assert 'to "plain"' not in report


FIT_PLUGIN = """
from mypy.plugin import FunctionSigContext, Plugin
from mypy.types import Instance, LiteralType


def fit_hook(ctx: FunctionSigContext):
    sig = ctx.default_signature
    if not ctx.args[0]:
        return sig
    got = ctx.api.get_expression_type(ctx.args[0][0])
    want = sig.arg_types[0]
    if isinstance(got, Instance) and isinstance(want, Instance):
        give, declared = got.args[0], want.args[0]
        if isinstance(give, LiteralType) and isinstance(declared, LiteralType):
            if give.value == declared.value:
                return sig.copy_modified(arg_types=[got, *sig.arg_types[1:]])
            ctx.api.fail("fit: names disagree", ctx.context)
    return sig


class FitPlugin(Plugin):
    def get_function_signature_hook(self, fullname: str):
        if fullname == "snippet.declared":
            return fit_hook
        return None


def plugin(version: str):
    return FitPlugin
"""

FIT_SNIPPET = """
from typing import Generic, Literal, TypeVar

N = TypeVar("N")
D = TypeVar("D")


class Expr(Generic[N, D]): ...


def col(name: str) -> Expr[Literal["price"], Literal["lookup"]]:
    return Expr()


def declared(e: Expr[Literal["price"], Literal["Float64"]]) -> None: ...


declared(col("price"))
"""

CHAIN_PLUGIN = """
from mypy.plugin import MethodContext, Plugin
from mypy.types import Instance, LiteralType


def append_hook(ctx: MethodContext):
    receiver = ctx.type
    default = ctx.default_return_type
    if not isinstance(receiver, Instance) or not ctx.arg_types[0]:
        return default
    arg = ctx.arg_types[0][0]
    held = receiver.args[0]
    if isinstance(held, LiteralType) and isinstance(arg, Instance):
        parts = set(str(held.value).split("|")) | {arg.type.name}
        literal = LiteralType("|".join(sorted(parts)), held.fallback)
        return Instance(receiver.type, [literal])
    return default


class ChainPlugin(Plugin):
    def get_method_hook(self, fullname: str):
        if fullname == "snippet.Chain.append":
            return append_hook
        return None


def plugin(version: str):
    return ChainPlugin
"""

CHAIN_SNIPPET = """
from typing import Generic, Literal, TypeVar

S = TypeVar("S")


class Int64: ...


class Chain(Generic[S]):
    def append(self, branch: object) -> "Chain[S]":
        return self


def seed() -> Chain[Literal["seed"]]:
    return Chain()


e = seed()
for _ in range(10):
    e = e.append(Int64())
reveal_type(e)
"""


class TestHooksHostTheDesign:
    """The two hosting obligations, proven with throwaway plugins.

    Fit must be computable at the call, and a loop over minted types
    must converge. Each probe builds a small plugin, points mypy at
    it by file path, and asserts the behaviour the design requires.
    """

    def hosted_report(
        self,
        tmp_path: Path,
        stem: str,
        plugin_source: str,
        snippet: str,
        *flags: str,
    ) -> str:
        """Run mypy over a snippet with a throwaway plugin loaded.

        The plugin file's stem must differ between the probes: mypy
        imports a plugin under its file's module name, and a second
        plugin sharing a stem would be served the first one's module
        when two probes run in one process.

        Args:
            tmp_path: A directory for the plugin, config and cache.
            stem: The plugin module's own name.
            plugin_source: The plugin module's source.
            snippet: The program to check.
            *flags: Extra configuration lines for the mypy config.

        Returns:
            Everything mypy printed.
        """
        plugin_file = tmp_path / f"{stem}.py"
        plugin_file.write_text(plugin_source)
        ini = tmp_path / "mypy.ini"
        lines = ["[mypy]", f"plugins = {plugin_file}", *flags]
        ini.write_text("\n".join(lines) + "\n")
        return BinderProbe().report(snippet, tmp_path, "--config-file", str(ini))

    def test_a_signature_hook_computes_fit_and_admits_the_call(self, tmp_path: Path) -> None:
        """The call bare mypy refuses passes once a hook computes fit.

        This is the positive half beside `TestFitIsNotSubtyping`: the
        same shape of call, refused nominally, is admitted when a
        signature hook reads the argument's minted type and rewrites
        the expected parameter.
        """
        report = self.hosted_report(tmp_path, "fit_probe", FIT_PLUGIN, FIT_SNIPPET)
        assert "error" not in report

    def test_minted_absorbing_types_converge_in_a_loop(self, tmp_path: Path) -> None:
        """Canonical minting turns a ten-pass loop into two states.

        The plugin mints the chain's type as a sorted union spelling,
        so a repeated append mints the same type again, and the
        redefinition union stops at `Seed | Step`: the hosting proof
        of the convergence obligation, beside the measured truncation
        of a minting that grows.
        """
        report = self.hosted_report(
            tmp_path,
            "chain_probe",
            CHAIN_PLUGIN,
            CHAIN_SNIPPET,
            "allow_redefinition = True",
        )
        revealed = BinderProbe().revealed(report)
        states = set(revealed.split(" | "))
        assert states == {
            "snippet.Chain[Literal['seed']]",
            "snippet.Chain[Literal['Int64|seed']]",
        }
