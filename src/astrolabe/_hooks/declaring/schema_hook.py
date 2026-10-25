"""Turn a `columns("Name", {...})` call into a schema class, for names Python cannot spell.

A user declares the columns of a frame with a `Columns` class, one
annotated attribute for each column. An attribute name must be a Python
identifier, but a column name can be any text. After a pivot, for
example, the columns can be named `1` or `v1|a`. The call
`Pivoted = columns("Pivoted", {"1": pl.Int64, "v1|a": pl.Int64})`
declares such names. The hook here gives `Pivoted` the same schema that
a written `Columns` class gives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import polars as pl
from mypy.exprtotype import TypeTranslationError, expr_to_unanalyzed_type
from mypy.nodes import GDEF, DictExpr, StrExpr, SymbolTableNode, TypedDictData
from mypy.types import TypedDictType, get_proper_type

from astrolabe._hooks.declaring.regions import RegionParser
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.declaring.site import (
    DeclarationCall,
    DeclarationSite,
    UnresolvedReferences,
)
from astrolabe._hooks.errors import MarkerArgumentError
from astrolabe._hooks.propagate import not_falsy, not_none, propagates

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import DynamicClassDefContext
    from mypy.types import Type as MypyType


class SchemaDeclarationHook:
    """Register `Name = columns("Name", {column: data type, ...})` as a schema class.

    mypy calls a dynamic-class hook for an assignment `Name = f(...)`,
    where the plugin has registered a hook for `f`. It runs during
    semantic analysis, the phase that resolves names. To mypy, a
    `Columns` class is a closed `TypedDict`, that is, a `TypedDict` that
    allows no key it does not list. This hook builds that class from the
    dict of the call and adds it to the module. Code that reads a schema
    then cannot tell it from a written `Columns` class.

    Each value of the dict is one data type. The user can write it as
    polars writes it, for example `pl.Int64` or `pl.Datetime("us")`, as
    a type such as `List[pl.Int64]`, or as `Unchecked`. If the call does
    not have this shape, the hook reports one `[marker-arg]` error and
    registers nothing.
    """

    _DTYPE: Final[Target] = Target.of(pl.DataType)

    SHAPE: Final[str] = (
        "columns() takes the class name and a dict of column names to data types, "
        'for example columns("Pivoted", {"1": pl.Int64})'
    )

    def __call__(self, ctx: DynamicClassDefContext) -> None:
        """Register the assigned name as the schema class that the call describes.

        If the call uses a name that mypy has not resolved yet, the hook
        does nothing on this pass. mypy analyzes the statement again when
        the name resolves.

        Args:
            ctx: The mypy context of the declaration.
        """
        if UnresolvedReferences().found(ctx.call):
            return
        fields = self._fields(ctx)
        if fields is None:
            MarkerArgumentError(self.SHAPE).fail(DeclarationSite(ctx))
            return
        self._register(ctx, fields)

    @propagates(None)
    def _fields(self, ctx: DynamicClassDefContext) -> dict[str, MypyType] | None:
        """Read the columns that the call declares.

        As with `TypedDict("Name", {...})`, the class name in the call
        and the assigned name must agree. The columns must be written
        out in the call as a dict, because the hook reads the syntax and
        not a value. Each key of the dict is a column name.

        Args:
            ctx: The mypy context of the declaration.

        Returns:
            The data type of each column, keyed by column name. None if
            the first argument is not the assigned name, the second is
            not a dict written in the code, a key is not a string
            literal, or a value is not a data type.
        """
        call = DeclarationCall(ctx.call)
        class_name = call.argument(0, "name")
        not_falsy(isinstance(class_name, StrExpr) and class_name.value == ctx.name)
        items = call.argument(1, "fields")
        written = not_none(items if isinstance(items, DictExpr) else None)
        fields: dict[str, MypyType] = {}
        for key, value in written.items:
            name = not_none(key if isinstance(key, StrExpr) else None).value
            fields[name] = not_none(self._value_type(ctx, value))
        return fields

    def _value_type(self, ctx: DynamicClassDefContext, node: Expression) -> MypyType | None:
        """Give the data type that one value of the dict names.

        The code first reads the value as the region reader of an
        expression declaration does. That reader accepts a value written
        as polars writes it, for example `pl.Int64`, `pl.Datetime("us")`
        or `Unchecked`. A value in the form of a type, for example
        `List[pl.Int64]` or `Struct[...]`, is then read as a type.

        Args:
            ctx: The mypy context of the declaration.
            node: The value in the dict.

        Returns:
            The data type, or None for a value that is not exactly one
            data type. A polars family or a set of data types is a
            region of several kinds, and gives None.
        """
        region = RegionParser(ctx).parsed(node)
        if region is not None:
            return region.dtype
        return self._as_type(ctx, node)

    @propagates(None)
    def _as_type(self, ctx: DynamicClassDefContext, node: Expression) -> MypyType | None:
        """Read a value of the dict in the form of a type, for example `List[pl.Int64]`.

        The value is an expression in the code, so mypy must first turn
        it into a type, and then resolve the names in it.

        Args:
            ctx: The mypy context of the declaration.
            node: The value in the dict.

        Returns:
            The resolved type, or None for a value that is not a type or
            is a type that is not a polars data type.
        """
        try:
            unanalyzed = expr_to_unanalyzed_type(node, ctx.api.options)
        except TypeTranslationError:
            return None
        analyzed = not_none(ctx.api.anal_type(unanalyzed))
        not_falsy(self._DTYPE.is_base_of(get_proper_type(analyzed)))
        return analyzed

    def _register(self, ctx: DynamicClassDefContext, fields: dict[str, MypyType]) -> None:
        """Build the class as a closed `TypedDict`, and add it to the module.

        Every column of the `TypedDict` is required. Its base type is the
        `_TypedDict` instance that mypy uses for every `TypedDict`.

        Args:
            ctx: The mypy context of the declaration.
            fields: The data type of each column, keyed by column name.
        """
        fallback = ctx.api.named_type("typing._TypedDict")
        info = ctx.api.basic_new_typeinfo(ctx.name, fallback, ctx.call.line)
        typed_dict = TypedDictType(fields, set(fields), set(), fallback, is_closed=True)
        info.update_typeddict_type(typed_dict)
        info.typeddict_data = TypedDictData(True, [], {})
        ctx.api.add_symbol_table_node(ctx.name, SymbolTableNode(GDEF, info))
