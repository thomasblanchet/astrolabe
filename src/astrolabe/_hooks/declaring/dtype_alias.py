"""Remember the data type that a variable holds, so that a declaration can read it.

A user can name a data type once and reuse it: `Stamp = pl.Datetime("us")`,
then `column("t", Stamp)`. A declaration is read during semantic
analysis, the mypy phase that resolves names before any expression has a
type. At that point `Stamp` is only a variable, and nothing says which
time unit it holds. The declaration then loses the parameters of the
data type.

The hook here runs at the assignment itself, where the call is still in
view. It registers the data type under a second, hidden name beside the
variable, and a declaration looks that name up. No source file can spell
the hidden name, so it collides with nothing. The variable keeps its own
meaning, and mypy checks every use of it as a value in the usual way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import GDEF, SymbolTableNode, TypeAlias

from astrolabe._hooks.calls.dtype_constructor_readers import (
    SemanticAnalyzerLiteralTypeBuilder,
)
from astrolabe._hooks.calls.dtype_node import DtypeNodeReader, DtypeShadow

if TYPE_CHECKING:
    from mypy.plugin import DynamicClassDefContext


class DtypeAliasHook:
    """Register the data type that an assignment such as `Money = pl.Decimal(20, 4)` binds.

    mypy calls a dynamic-class hook for an assignment `Name = f(...)`,
    where the plugin has registered a hook for `f`. The plugin registers
    this one for every polars data type class. As a result,
    `Money = pl.Decimal(20, 4)` and `Counts = pl.List(pl.Int64)` give a
    declaration the same parameters as the call written in place.

    If an argument of the call is not a literal, the hook registers
    nothing. A declaration that uses the variable then reads the data
    type without its parameters.
    """

    def __init__(self) -> None:
        """Build the reader that turns a constructor call into a data type.

        The reader builds literal types in the way that semantic analysis
        allows, because the hook runs in that phase.
        """
        self._nodes = DtypeNodeReader(SemanticAnalyzerLiteralTypeBuilder())

    def __call__(self, ctx: DynamicClassDefContext) -> None:
        """Register the data type of one assignment under its hidden name.

        Args:
            ctx: The mypy context of the assignment. It holds the
                constructor call and the name of the variable.
        """
        instance = self._nodes.instance(ctx, ctx.call)
        if instance is None:
            return
        name = DtypeShadow().beside(ctx.name)
        alias = TypeAlias(
            instance,
            ctx.api.qualified_name(name),
            ctx.api.cur_mod_id,
            ctx.call.line,
            ctx.call.column,
        )
        ctx.api.add_symbol_table_node(name, SymbolTableNode(GDEF, alias))
