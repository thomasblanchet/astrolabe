"""The mypy hook that turns `Rest = other_columns(...)` into a class that the checker can read.

A remainder is the set of columns of a frame that its type does not
name. A user declares one with `Rest = other_columns(...)`, and each
keyword states a promise about those columns, for example
`without=["ticker"]`. The promises must reach the checker as types.

A plain call suits this better than a type written by hand. In a type
argument, mypy reads a bare string as a forward reference to a class,
and it checks that the class exists before any plugin hook can act. In
a call argument, a string is an ordinary string literal in the abstract
syntax tree (AST), and the hook reads it as it is. The declarations of
expressions, for example `column` and `typed_expr`, live in
`_hooks/declaring/`.
"""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

from mypy.nodes import (
    GDEF,
    CallExpr,
    ListExpr,
    RefExpr,
    SetExpr,
    StrExpr,
    SymbolTableNode,
    TypeInfo,
)
from mypy.types import Instance, UninhabitedType, UnionType

import astrolabe.remainder as remainder_markers
from astrolabe._hooks.calls.dtype_node import DtypeNodeReader
from astrolabe._hooks.calls.dtype_constructor_readers import (
    SemanticAnalyzerLiteralTypeBuilder,
)
from astrolabe._hooks.declaring.site import DeclarationSite
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.errors import MarkerArgumentError
from astrolabe.remainder import other_columns as _other_columns

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.nodes import Expression
    from mypy.plugin import DynamicClassDefContext
    from mypy.types import Type as MypyType


class DynamicClassHook:
    """The shared shape of a hook that mypy calls for `Name = some_function(...)`.

    mypy lets a plugin define a class from such an assignment, while it
    analyzes the module. This is a dynamic-class hook. Each such hook
    implements `__call__`, which receives the context of one assignment.
    """

    def __call__(self, ctx: DynamicClassDefContext) -> None:
        """Define the name on the left of the assignment as the class that the call describes.

        Args:
            ctx: The mypy context of the assignment, which holds the name and
                the call.
        """
        raise NotImplementedError


class RemainderDeclarationHook(DynamicClassHook):
    """Turn `Rest = other_columns(...)` into a real class that holds its promises as base classes.

    The hook defines `Rest` as a new class. Its first base is
    `DeclaredRemainder`, which tells the checker that the class is a
    remainder. Each keyword adds one more base, a marker that holds the
    promise as type arguments, for example `Without[Literal["ticker"]]`.

    The promises live in the class itself. mypy caches the class between
    runs, and it does not run this hook again for a module that did not
    change. If the plugin kept a fact in its own memory, the fact is
    missing on that second run. A base class is still there.

    If one keyword is wrong, the hook reports an error and leaves out that
    one promise. The class still exists, so each later use of `Rest` does
    not report a second error.

    The hook runs during semantic analysis, the phase in which mypy binds
    names, before it checks types.
    """

    TARGET: Final[Target] = Target.of(_other_columns)

    def __init__(self) -> None:
        """Build the readers for the arguments of the call."""
        self._literal_type = SemanticAnalyzerLiteralTypeBuilder()
        self._dtype_nodes = DtypeNodeReader(self._literal_type)

    def __call__(self, ctx: DynamicClassDefContext) -> None:
        """Define the declared class, with one base for each keyword that the hook can read.

        Every declared remainder has `DeclaredRemainder` as its first base.
        `other_columns` accepts keywords only, so the hook skips an argument
        that has no name. The hook then builds the new class and binds it to
        the name on the left of the assignment.

        Args:
            ctx: The mypy context of the assignment.
        """
        declared = Target.of(remainder_markers.DeclaredRemainder).fullname
        bases = [ctx.api.named_type(declared)]
        call = ctx.call
        for name, node in zip(call.arg_names, call.args, strict=True):
            if name is None:
                continue
            base = self._requirement_base(ctx, name, node)
            if base is not None:
                bases.append(base)
        obj = ctx.api.named_type("builtins.object")
        info = ctx.api.basic_new_typeinfo(ctx.name, obj, call.line)
        info.bases = [obj, *bases]
        info.mro = [info, *(base.type for base in bases), obj.type]
        ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, info))

    def _requirement_base(
        self,
        ctx: DynamicClassDefContext,
        name: str,
        node: Expression,
    ) -> Instance | None:
        """Build the base class for the promise of one keyword.

        Args:
            ctx: The mypy context of the assignment.
            name: The name of the keyword.
            node: The syntax node of the argument.

        Returns:
            The marker base, for example `Without[Literal["ticker"]]`. None if
            the keyword is unknown or its argument has the wrong form. The
            method then reported an error.
        """
        reader = self._readers().get(name)
        if reader is None:
            problem = MarkerArgumentError(f'other_columns() got an unexpected keyword "{name}"')
            problem.fail(DeclarationSite(ctx))
            return None
        argument = reader(ctx, node)
        if argument is None:
            self._malformed(ctx, name)
            return None
        marker = self._MARKERS[name]
        fullname = Target.of(marker).fullname
        return Instance(ctx.api.named_type(fullname).type, [argument])

    def _malformed(self, ctx: DynamicClassDefContext, name: str) -> None:
        """Report an argument that has the wrong form for its keyword, and name the right form.

        Args:
            ctx: The mypy context of the assignment.
            name: The name of the keyword.
        """
        shapes = {
            "without": 'a list of column names, e.g. without=["ticker"]',
            "only": 'a list of column names, e.g. only=["ticker"]',
            "without_prefix": 'a string, e.g. without_prefix="sec_"',
            "only_prefix": 'a string, e.g. only_prefix="sec_"',
            "without_suffix": 'a string, e.g. without_suffix="_prev"',
            "only_suffix": 'a string, e.g. only_suffix="_prev"',
            "all_of": self._DTYPE_SHAPE,
            "none_of": self._DTYPE_SHAPE,
            "disjoint_from": "a declared remainder, or a set of them",
            "within": "a declared remainder, or a set of them",
        }
        problem = MarkerArgumentError(f"other_columns() {name}= must be {shapes[name]}")
        problem.fail(DeclarationSite(ctx))

    _DTYPE_SHAPE: Final[str] = ", ".join([
        "a polars data type class",
        'a data type such as pl.Datetime("us")',
        "or a set of them",
    ])
    _MARKERS: Final[dict[str, type]] = {
        "without": remainder_markers.Without,
        "only": remainder_markers.OnlyNames,
        "without_prefix": remainder_markers.WithoutPrefix,
        "only_prefix": remainder_markers.OnlyPrefix,
        "without_suffix": remainder_markers.WithoutSuffix,
        "only_suffix": remainder_markers.OnlySuffix,
        "all_of": remainder_markers.AllOf,
        "none_of": remainder_markers.NoneOf,
        "disjoint_from": remainder_markers.DisjointFrom,
        "within": remainder_markers.Within,
    }

    def _readers(
        self,
    ) -> dict[
        str,
        Callable[[DynamicClassDefContext, Expression], MypyType | None],
    ]:
        """Give the reader for the argument of each keyword.

        Returns:
            One reader for each keyword that `other_columns` accepts.
        """
        return {
            "without": self._name_set,
            "only": self._name_set,
            "without_prefix": self._string,
            "only_prefix": self._string,
            "without_suffix": self._string,
            "only_suffix": self._string,
            "all_of": self._dtype_set,
            "none_of": self._dtype_set,
            "disjoint_from": self._class_set,
            "within": self._class_set,
        }

    def _name_set(self, ctx: DynamicClassDefContext, node: Expression) -> MypyType | None:
        """Read a list of column names as one union of string literal types.

        The names are sorted and each appears once, so two declarations with
        the same names in a different order give the same type. An empty list
        gives `Never`, and a list of one name gives that literal alone.

        Args:
            ctx: The mypy context of the assignment.
            node: The syntax node of the argument.

        Returns:
            The union of the names, or None if the argument is not a list of
            string literals written in place.
        """
        if not isinstance(node, ListExpr):
            return None
        names: set[str] = set()
        for item in node.items:
            if not isinstance(item, StrExpr):
                return None
            names.add(item.value)
        literals = [self._literal_type.string(name, ctx.api) for name in sorted(names)]
        if not literals:
            return UninhabitedType()
        if len(literals) == 1:
            return literals[0]
        return UnionType(list(literals))

    def _string(self, ctx: DynamicClassDefContext, node: Expression) -> MypyType | None:
        """Read a prefix or suffix argument as a string literal type.

        Args:
            ctx: The mypy context of the assignment.
            node: The syntax node of the argument.

        Returns:
            The literal type, or None if the argument is not a string literal.
        """
        if not isinstance(node, StrExpr):
            return None
        return self._literal_type.string(node.value, ctx.api)

    def _class_set(self, ctx: DynamicClassDefContext, node: Expression) -> MypyType | None:
        """Read a class, or a set of classes, as one instance or a union of instances.

        The relational keywords `disjoint_from` and `within` take this form. A
        set means each of its classes.

        Args:
            ctx: The mypy context of the assignment.
            node: The syntax node of the argument.

        Returns:
            One instance with no arguments, or a union of them. None for any
            other form.
        """
        if isinstance(node, SetExpr):
            return self._union_of(ctx, node, self._class_instance)
        return self._class_instance(ctx, node)

    def _dtype_set(self, ctx: DynamicClassDefContext, node: Expression) -> MypyType | None:
        """Read a data type argument: a class, one data type with parameters, or a set of them.

        A data type with parameters, for example `pl.Datetime("us")`, keeps its
        parameters in its marker. Any other constructor call, for example
        `pl.List(pl.Int64)`, is refused. The hook does not widen it to every
        `List`, because the wider set lets through callers that the promise
        refuses.

        Args:
            ctx: The mypy context of the assignment.
            node: The syntax node of the argument.

        Returns:
            One instance, or a union of them. None for any other form.
        """
        if isinstance(node, SetExpr):
            return self._union_of(ctx, node, self._dtype_instance)
        return self._dtype_instance(ctx, node)

    def _dtype_instance(self, ctx: DynamicClassDefContext, node: Expression) -> Instance | None:
        """Read one data type item: a name, or a constructor call with parameters.

        A name can be a class, for example `pl.Float64`. It can also be a
        variable bound to a data type, for example `Price = pl.Float64`.
        mypy treats such a variable as a type alias, and its target holds the
        data type with its parameters. A constructor call counts only if it
        gives a marker that holds its parameters as type arguments.

        Args:
            ctx: The mypy context of the assignment.
            node: The syntax node of the item.

        Returns:
            The instance, or None for any other form.
        """
        if not isinstance(node, CallExpr):
            direct = self._class_instance(ctx, node)
            if direct is not None:
                return direct
            return self._dtype_nodes.aliased(ctx, node)
        instance = self._dtype_nodes.instance(ctx, node)
        if instance is not None and instance.args:
            return instance
        return None

    def _union_of(
        self,
        ctx: DynamicClassDefContext,
        node: SetExpr,
        read_one: Callable[[DynamicClassDefContext, Expression], Instance | None],
    ) -> MypyType | None:
        """Read each item of a set, and join them into a union.

        Args:
            ctx: The mypy context of the assignment.
            node: The syntax node of the set.
            read_one: The reader for one item.

        Returns:
            The union, or None if one item cannot be read.
        """
        items: list[MypyType] = []
        for item in node.items:
            one = read_one(ctx, item)
            if one is None:
                return None
            items.append(one)
        return UnionType(items)

    def _class_instance(self, ctx: DynamicClassDefContext, node: Expression) -> Instance | None:
        """Read a reference to a class as an instance of that class with no arguments.

        The method accepts only a reference whose node is already a class
        (`TypeInfo`). It does not look up any other name by its full name.
        During semantic analysis, `named_type` fails with an assertion when the
        name is not a class, so an alias or an unbound name then crashes the
        whole mypy run. Here, such a name gives None, and the caller reports
        one wrong argument.

        Args:
            ctx: The mypy context of the assignment. The method does not read
                it.
            node: The syntax node of the reference.

        Returns:
            The instance, or None for anything that is not a class reference.
        """
        del ctx
        if not isinstance(node, RefExpr):
            return None
        if isinstance(node.node, TypeInfo):
            return Instance(node.node, [])
        return None
