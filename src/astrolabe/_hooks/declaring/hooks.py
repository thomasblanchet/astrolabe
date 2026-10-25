"""Turn a declaration call into a type alias, so that its name can annotate code.

A user writes `Price = column("price", pl.Float64)` or
`BuyFlag = typed_expr(Quantity, gives=pl.Boolean, named="is_buy")`. At
run time each call gives a plain value. The checker needs `Price` and
`BuyFlag` to be types, because they annotate parameters and return
values.

mypy calls a dynamic-class hook for an assignment `Name = f(...)`, where
the plugin has registered a hook for `f`. The hook runs during semantic
analysis, the phase that resolves names. Each hook here reads one kind
of declaration call and registers the assigned name as an alias for the
type that the call describes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import GDEF, SymbolTableNode, TypeAlias

from astrolabe._hooks.calls.dtype_constructor_readers import (
    SemanticAnalyzerLiteralTypeBuilder,
)
from astrolabe._hooks.declaring.parser import DeclarationParser
from astrolabe._hooks.declaring.selection_parser import SelectionDeclarationParser
from astrolabe._hooks.declaring.regions import RegionParser
from astrolabe._hooks.declaring.site import CallShape, DeclarationCall, UnresolvedReferences
from astrolabe._hooks.encoding.expr_codec import QuadrupleCodec, RegionCodec
from astrolabe._hooks.encoding.marker_minter import MarkerMinter
from astrolabe._hooks.encoding.selection_codec import SelectionCodec
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe.expr import TypedColumn

if TYPE_CHECKING:
    from mypy.plugin import DynamicClassDefContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup


class DeclarationRegistrar:
    """Register the assigned name of a declaration as an alias for one type.

    A hook that reads an expression or a selection declaration ends with
    the same two steps. It builds its type with a minter, the object
    that assembles every encoded type from marker classes. It then puts
    the assigned name in the symbol table as a type alias for that type.
    This class holds those two steps, so that each hook does them in the
    same way.

    The minter builds literal types in the way that semantic analysis
    allows. A minter built for the later checking phase does not work
    here.
    """

    def __init__(self, plugin: PluginLookup) -> None:
        """Keep the plugin, which a minter needs to look up a class by name.

        Args:
            plugin: The mypy plugin instance.
        """
        self._plugin = plugin

    def minter(self, ctx: DynamicClassDefContext) -> MarkerMinter:
        """Build a minter for one declaration, for use during semantic analysis.

        Args:
            ctx: The mypy context of the declaration. Its semantic analyzer
                resolves the names that the minter looks up.

        Returns:
            The minter.
        """
        return MarkerMinter(self._plugin, SemanticAnalyzerLiteralTypeBuilder(), ctx.api)

    def register(self, ctx: DynamicClassDefContext, target: MypyType) -> None:
        """Put the assigned name in the symbol table as an alias for one type.

        Args:
            ctx: The mypy context of the declaration. It gives the name,
                the module and the position of the call.
            target: The type that the name stands for.
        """
        fullname = ctx.api.qualified_name(ctx.name)
        alias = TypeAlias(target, fullname, ctx.api.cur_mod_id, ctx.call.line, ctx.call.column)
        ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, alias))


class SliceDeclarationHook:
    """Give the name of a `typed_expr`, `typed_agg` or `typed_lit` declaration its type.

    These three calls declare an expression type: the columns that an
    expression reads, the data type that it gives, its output name, and
    how it counts rows. The hook reads the call into a `SliceDecl`, the
    value that the engine uses for a declaration. The parser checks the
    rules of a declaration and reports each broken rule at the call.

    The same codec encodes a declared type and a type that the checker
    builds for an expression. The two types therefore spell the same
    way, and mypy can compare them. If the declaration breaks a rule,
    the hook registers nothing, and mypy reports each use of the name as
    not valid as a type.
    """

    def __init__(self, plugin: PluginLookup, character: str) -> None:
        """Keep the plugin, and how the declared expression counts rows.

        Args:
            plugin: The mypy plugin instance.
            character: How the declared expression counts rows. It is
                `elementwise` for `typed_expr`, `aggregating` for
                `typed_agg` and `literal` for `typed_lit`.
        """
        self._registrar = DeclarationRegistrar(plugin)
        self._character = character

    @propagates(None)
    def __call__(self, ctx: DynamicClassDefContext) -> None:
        """Register the assigned name as the declared expression type.

        If the call uses a name that mypy has not resolved yet, the hook
        does nothing on this pass. mypy analyzes the statement again when
        the name resolves. If the declaration breaks a rule, the parser
        reports the error, and the hook registers nothing.

        Args:
            ctx: The mypy context of the declaration.
        """
        if UnresolvedReferences().found(ctx.call):
            return
        decl = not_none(DeclarationParser(ctx, self._character).parsed(ctx.call))
        codec = QuadrupleCodec(self._registrar.minter(ctx))
        spelled = not_none(codec.encoded(decl.as_expression()))
        self._registrar.register(ctx, spelled)


class ColumnDeclarationHook:
    """Give the name of a `column(name, dtype)` declaration its type.

    `Price = column("price", pl.Float64)` declares an expression that
    reads the column `price` and gives it back unchanged. A parameter
    annotated `Price` accepts `pl.col("price")`. The alias is
    `TypedColumn[Literal["price"], X]`.

    `X` is the region of the read, that is, the data types at which the
    column can be read. A single data type keeps its parameters as the
    user wrote them, so that the alias can also describe a field of a
    `Struct`. A polars family class, a set of data types or `EveryDType`
    gives a region of several data types, spelled by their names alone.
    If the call does not have this shape, the hook registers nothing,
    and mypy reports each use of the name as not valid as a type.
    """

    def __init__(self, plugin: PluginLookup) -> None:
        """Keep the plugin, for the alias and its minter.

        Args:
            plugin: The mypy plugin instance.
        """
        self._registrar = DeclarationRegistrar(plugin)
        self._shape = CallShape()

    @propagates(None)
    def __call__(self, ctx: DynamicClassDefContext) -> None:
        """Register the assigned name as the declared column.

        If the call uses a name that mypy has not resolved yet, the hook
        does nothing on this pass. mypy analyzes the statement again when
        the name resolves. The hook also registers nothing if the name is
        not a string literal, or if the second argument names no region.

        A region written as one data type keeps that data type, with its
        parameters. Any other region is spelled by its kinds, the data
        types with their parameters left out.

        Args:
            ctx: The mypy context of the declaration.
        """
        if UnresolvedReferences().found(ctx.call):
            return
        call = DeclarationCall(ctx.call)
        name = not_none(self._shape.literal(call.argument(0, "name")))
        region = not_none(RegionParser(ctx).parsed(not_none(call.argument(1, "dtype"))))
        minter = self._registrar.minter(ctx)
        spelled_region: MypyType | None = region.dtype
        if spelled_region is None:
            spelled_region = not_none(RegionCodec(minter).built(region.kinds))
        spelled = not_none(minter.marker(TypedColumn, [minter.string(name), spelled_region]))
        self._registrar.register(ctx, spelled)


class SelectionDeclarationHook:
    """Give the name of a `typed_selection` or `typed_agg_selection` declaration its type.

    These calls declare the type of a selector expression, for example
    `cs.numeric()` with a cast. A selection gives one output column for
    each column that its selector picks, so its declaration states the
    selector, the data type that each output has, and a rename. The hook
    reads the call into a `SelectionType`, the value that the engine
    uses, and encodes it with the selection codec.

    The parser checks the rows of the declaration with the rules of an
    expression declaration, over one read of a column whose name the
    selector fills. If the declaration breaks a rule, the hook registers
    nothing.
    """

    def __init__(self, plugin: PluginLookup, character: str) -> None:
        """Keep the plugin, and how the declared selection counts rows.

        Args:
            plugin: The mypy plugin instance.
            character: How the declared selection counts rows. It is
                `elementwise` for `typed_selection` and `aggregating`
                for `typed_agg_selection`.
        """
        self._registrar = DeclarationRegistrar(plugin)
        self._character = character

    @propagates(None)
    def __call__(self, ctx: DynamicClassDefContext) -> None:
        """Register the assigned name as the declared selection type.

        If the call uses a name that mypy has not resolved yet, the hook
        does nothing on this pass. mypy analyzes the statement again when
        the name resolves. If the declaration breaks a rule, the parser
        reports the error, and the hook registers nothing.

        Args:
            ctx: The mypy context of the declaration.
        """
        if UnresolvedReferences().found(ctx.call):
            return
        parser = SelectionDeclarationParser(ctx, self._character)
        selection = not_none(parser.parsed(ctx.call))
        spelled = not_none(SelectionCodec(self._registrar.minter(ctx)).encoded(selection))
        self._registrar.register(ctx, spelled)
