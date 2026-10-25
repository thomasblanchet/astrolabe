"""Read the data type in a declaration, such as `column("t", pl.Datetime("us"))`, from its AST.

A declaration can name a data type as a class (`pl.Int64`), an instance (`pl.Float64()`), a call
with parameters (`pl.Datetime("us")`), a nested type (`pl.List(pl.Int64)`), or a variable that
holds one of these. The checker reads declarations during semantic analysis, the phase of mypy
before it infers any type. So this module reads each form from the abstract syntax tree (AST)
and the symbol tables, not from an inferred type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl
from mypy.nodes import CallExpr, IntExpr, RefExpr, TypeAlias, TypeInfo, Var
from mypy.types import Instance
from mypy.types import get_proper_type
from mypy.types import Type as MypyType

from astrolabe import frame
from astrolabe._hooks.calls.dtype_constructor_args import DtypeConstructorArgs
from astrolabe._hooks.calls.dtype_constructor_readers import ParametrizedDtypeReaders
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeReads

if TYPE_CHECKING:
    from mypy.nodes import Expression, SymbolNode
    from mypy.plugin import DynamicClassDefContext

    from astrolabe._hooks.calls.dtype_constructor_readers import (
        LiteralTypeBuilder,
        _ParametrizedDtypeReader,
    )


class DtypeShadow:
    """Name the hidden entry that holds the data type of a variable such as `Stamp`.

    After `Stamp = pl.Datetime("us")`, mypy treats `Stamp` as a variable, not as a type, so the
    time unit is lost. A hook for that assignment therefore stores the data type as a type alias
    under a second, hidden name in the same module. This class makes that name, and finds the
    entry again when a declaration uses `Stamp`. The hidden name holds a character that no
    source file can contain, so it cannot clash with a real name.

    Example:
        >>> DtypeShadow().beside("Stamp") == "\uffffdtype\uffffStamp"
        True
    """

    MARK: ClassVar[str] = "\uffffdtype\uffff"

    def beside(self, name: str) -> str:
        """Give the hidden name for one variable.

        Args:
            name: The name of the variable.

        Returns:
            The hidden name.

        Example:
            >>> DtypeShadow().beside("Money")[-5:]
            'Money'
        """
        return self.MARK + name

    def found(self, ctx: DynamicClassDefContext, fullname: str) -> SymbolNode | None:
        """Give the node stored under the hidden name of one variable.

        The method reads the symbol table of the module directly. A lookup by full name is
        wrong here: for a name that does not exist, mypy notes an incomplete reference and runs
        semantic analysis again.

        Args:
            ctx: The mypy context of the dynamic class hook that reads the declaration.
            fullname: The full name of the variable, with its module.

        Returns:
            The stored node, or None if the variable has no hidden entry.

        Example:
            >>> ctx = new_dynamic_class_context()
            >>> DtypeShadow().found(ctx, "app.types.Stamp") is None
            True
        """
        module, _, name = fullname.rpartition(".")
        file = ctx.api.modules.get(module)
        if file is None:
            return None
        entry = file.names.get(self.beside(name))
        return entry.node if entry is not None else None


class DtypeNodeReader:
    """Turn the AST node of a data type in a declaration into a mypy instance type.

    A declaration states the data type of a column, and the checker holds that data type as a
    mypy instance. This class reads the node in each form that a user can write. A call to
    `pl.Datetime`, `pl.Duration`, `pl.Decimal` or `pl.Enum` gives a marker class of this project
    whose type arguments hold the parameters, for example `Datetime[Literal["us"], None]`.
    `pl.List(...)` and `pl.Array(...)` give a marker that holds the inner data type. Every other
    class gives its plain instance.

    If the parameters of a call are not literals, the reader gives the plain instance, for
    example `Datetime` with no unit. The declaration then keeps its column with less detail.

    It sits in the plugin layer and runs during semantic analysis.
    """

    def __init__(self, literal_type: LiteralTypeBuilder) -> None:
        """Keep the builder of literal types, and build the readers of parameters from it.

        Args:
            literal_type: The builder that turns a Python value into a `Literal[...]` type in
                the current phase of mypy.
        """
        self._readers = ParametrizedDtypeReaders(literal_type)
        self._literal_type = literal_type

    # The polars classes that hold an inner data type, each with the marker class of this
    # project that stands for it. The flag is True for pl.Array, which also holds a width.
    NESTED: ClassVar[dict[str, tuple[str, bool]]] = {
        Target.of(pl.List).fullname: (Target.of(frame.List).fullname, False),
        Target.of(pl.Array).fullname: (Target.of(frame.Array).fullname, True),
    }

    def instance(self, ctx: DynamicClassDefContext, node: Expression) -> Instance | None:
        """Give the data type that one node names.

        Args:
            ctx: The mypy context of the dynamic class hook that reads the declaration.
            node: The AST node of the data type.

        Returns:
            The instance, or None if the node names no data type.
        """
        if isinstance(node, CallExpr):
            return self._called(ctx, node)
        aliased = self.aliased(ctx, node)
        if aliased is not None:
            return aliased
        return self._plain(ctx, node)

    def aliased(self, ctx: DynamicClassDefContext, node: Expression) -> Instance | None:
        """Give the data type that a name stands for.

        After `Price = pl.Float64`, mypy treats `Price` as a type alias, and `Cost = Price` is one
        too. The method reads the target of the alias, which keeps any parameters, for example
        the inner type of `List[pl.Int64]`. A variable such as `Stamp = pl.Datetime("us")` is not
        an alias, so the method reads its hidden entry (see `DtypeShadow`).

        Args:
            ctx: The mypy context of the dynamic class hook that reads the declaration.
            node: The AST node of the data type.

        Returns:
            The instance that the name stands for. None if the node is not such a name, or if
            the name stands for something other than a data type.

        Example:
            >>> from astrolabe._hooks.calls.dtype_constructor_readers import (
            ...     SemanticAnalyzerLiteralTypeBuilder,
            ... )
            >>> reader = DtypeNodeReader(SemanticAnalyzerLiteralTypeBuilder())
            >>> reader.aliased(None, str_literal_node("x")) is None
            True
            >>> from mypy.nodes import Var
            >>> variable = NameExpr("x")
            >>> variable.node = Var("x")
            >>> reader.aliased(new_dynamic_class_context(), variable) is None
            True
        """
        target = node.node if isinstance(node, RefExpr) else None
        if isinstance(target, Var):
            target = DtypeShadow().found(ctx, target.fullname)
        if not isinstance(target, TypeAlias):
            return None
        return self._dtype_target(get_proper_type(target.target))

    def _dtype_target(self, target: MypyType) -> Instance | None:
        """Keep the target of an alias only if it is a data type.

        A name can stand for anything. `BarCol = column("bar", pl.Int64)`, for example, stands
        for an expression, not a data type. The method keeps only a target that the checker can
        read as a data type.

        Args:
            target: The type that the alias stands for.

        Returns:
            The instance, or None for any other target.

        Example:
            >>> from mypy.types import UninhabitedType
            >>> from astrolabe._hooks.calls.dtype_constructor_readers import (
            ...     SemanticAnalyzerLiteralTypeBuilder,
            ... )
            >>> reader = DtypeNodeReader(SemanticAnalyzerLiteralTypeBuilder())
            >>> reader._dtype_target(UninhabitedType()) is None
            True
        """
        if not isinstance(target, Instance):
            return None
        if ConstDtypeReads().read(target) is None:
            return None
        return target

    def _called(self, ctx: DynamicClassDefContext, node: CallExpr) -> Instance | None:
        """Give the data type that a call such as `pl.List(pl.Int64)` or `pl.Float64()` names.

        A call to a class with parameters gives the marker with parameters. If one parameter is
        not a literal, it gives the plain instance.

        Args:
            ctx: The mypy context of the dynamic class hook that reads the declaration.
            node: The AST node of the call.

        Returns:
            The instance, or None if the call names no data type.
        """
        nested = self.NESTED.get(getattr(node.callee, "fullname", "") or "")
        if nested is not None:
            return self._nested(ctx, node, nested)
        reader = self._readers.reader_for(node.callee)
        if reader is not None:
            parametrized = self._parametrized(ctx, node, reader)
            if parametrized is not None:
                return parametrized
        return self._plain(ctx, node)

    def _nested(
        self,
        ctx: DynamicClassDefContext,
        node: CallExpr,
        nested: tuple[str, bool],
    ) -> Instance | None:
        """Build the marker of a list or an array, with its inner data type as a type argument.

        The inner data type is the first argument of the call. It can be any form that
        `instance` reads.

        Args:
            ctx: The mypy context of the dynamic class hook that reads the declaration.
            node: The AST node of the call.
            nested: The full name of the marker class, and True if it also holds a width.

        Returns:
            The marker instance, or None if the inner data type or the width cannot be read.
        """
        marker, shaped = nested
        info = self.type_info(ctx, marker)
        element = self.instance(ctx, node.args[0]) if node.args else None
        width = self._width(ctx, node) if shaped else []
        if info is None or element is None or width is None:
            return None
        return Instance(info, [element, *width])

    def _width(self, ctx: DynamicClassDefContext, node: CallExpr) -> list[MypyType] | None:
        """Give the width of a `pl.Array(...)` call as a type argument.

        An argument can hold the width if it comes after the inner data type, and if it is
        positional or has the keyword `shape` or `width`.

        Args:
            ctx: The mypy context of the dynamic class hook that reads the declaration.
            node: The AST node of the call.

        Returns:
            A list with the width as a `Literal[...]` type. None if the call does not give one
            integer literal for the width.
        """
        named = zip(node.args[1:], node.arg_names[1:], strict=True)
        widths = [argument for argument, name in named if name in {None, "shape", "width"}]
        if len(widths) != 1 or not isinstance(widths[0], IntExpr):
            return None
        return [self._literal_type.integer(widths[0].value, ctx.api)]

    def _parametrized(
        self,
        ctx: DynamicClassDefContext,
        node: CallExpr,
        reader: _ParametrizedDtypeReader,
    ) -> Instance | None:
        """Build the marker instance whose type arguments hold the parameters of a call.

        Args:
            ctx: The mypy context of the dynamic class hook that reads the declaration.
            node: The AST node of the call.
            reader: The reader of the parameters of the called class.

        Returns:
            The marker instance, or None if one argument is not a literal.
        """
        constructor_args = DtypeConstructorArgs(node, reader.PARAM_NAMES)
        type_args = reader(constructor_args, ctx.api)
        if type_args is None:
            return None
        info = self.type_info(ctx, reader.MARKER_TARGET.fullname)
        if info is None:  # pragma: no cover
            # The marker classes live in astrolabe.frame. The plugin adds that module to the
            # dependencies of every build, so the lookup always finds them.
            return None
        return Instance(info, type_args)

    def _plain(self, ctx: DynamicClassDefContext, node: Expression) -> Instance | None:
        """Give the plain instance of the class that a node names, with no type arguments.

        A class reference names the class, and so does the callee of a call. For every data
        type class other than the ones with parameters or an inner type, the class alone is the
        whole data type. The arguments of such a call add nothing.

        Args:
            ctx: The mypy context of the dynamic class hook that reads the declaration.
            node: The AST node of the data type.

        Returns:
            The instance, or None if the node names no class.
        """
        fullname = self.fullname(node)
        if fullname is None:
            return None
        info = self.type_info(ctx, fullname)
        if info is None:
            return None
        return Instance(info, [])

    def fullname(self, node: Expression) -> str | None:
        """Give the full name of the class that a node names, or that a call calls.

        A local variable has a name with no module. mypy stops with a crash if code looks up
        such a name as a full name, so the method keeps only a name that includes a module.

        Args:
            node: A class reference, or a call to a class.

        Returns:
            The full name, or None for any other node, or for a name with no module.

        Example:
            >>> from astrolabe._hooks.calls.dtype_constructor_readers import (
            ...     SemanticAnalyzerLiteralTypeBuilder,
            ... )
            >>> reader = DtypeNodeReader(SemanticAnalyzerLiteralTypeBuilder())
            >>> reader.fullname(dtype_constructor_call([str_literal_node("us")]))
            'polars.datatypes.classes.Datetime'
            >>> reader.fullname(str_literal_node("x")) is None
            True
            >>> local = NameExpr("inner")
            >>> local.fullname = "inner"
            >>> reader.fullname(local) is None
            True
        """
        named = node.callee if isinstance(node, CallExpr) else node
        found: str | None = getattr(named, "fullname", None)
        if found is None or "." not in found:
            return None
        return found

    def type_info(self, ctx: DynamicClassDefContext, fullname: str) -> TypeInfo | None:
        """Find a class by its full name, with the lookup of the semantic analysis phase.

        Args:
            ctx: The mypy context of the dynamic class hook that reads the declaration.
            fullname: The full name of the class in mypy.

        Returns:
            The `TypeInfo` of the class, or None if mypy cannot find it.
        """
        symbol = ctx.api.lookup_fully_qualified_or_none(fullname)
        if symbol is None or not isinstance(symbol.node, TypeInfo):
            return None
        return symbol.node
