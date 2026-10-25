"""Give a typed frame to a function that builds a frame with a `schema=` argument.

Functions such as `pl.read_csv` and `pl.from_dict` build a new frame from a
file or from Python data. polars types each one as a plain frame, so the
checker knows none of the columns. With `schema=`, polars builds the frame
with the columns and data types that the argument gives, so the checker can
know them too. This hook reads `schema=` in two shapes:

- `schema=Prices.schema`, where `Prices` is a module-level alias for a
  closed `TypedDataFrame[...]` or `TypedLazyFrame[...]`. The result has the
  columns of that alias.
- `schema={"price": pl.Float64, ...}`, a dict literal with literal string
  keys. The result has one column for each key, with the data type of its
  value.

`pl.from_dict({"price": [...], ...})` with no `schema=` also gives a typed
frame. The keys of the dict literal give the column names, and each column
gets the data type `Unchecked`, which means that its name is known and its
data type is not. The hook does not infer a data type from the values.

In every other case, the hook falls back through `UntypedFallback` to the
plain frame that polars declares, with the reason `NO_SCHEMA`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import DictExpr, MemberExpr, TypeInfo
from mypy.types import CallableType, Instance, get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.bare_literal import BareLiteralReader
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.calls.dtype_constructor_readers import CheckerLiteralTypeBuilder
from astrolabe._hooks.calls.dtype_expr import DtypeExpr
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.dtype_class_guard import DtypeClassGuard
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.encoding.marker_minter import MarkerMinter
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeCodec
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import FunctionContext, Plugin

    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback
    from astrolabe.frame import _TypedFrameBase


class SchemaArgFrameHook:
    """Give a frame-building function a typed frame from its `schema=` argument.

    It is a mypy function hook in the plugin layer, with one instance for
    each function. Each instance knows the kind of frame that its function
    gives: `TypedDataFrame` for a function that reads eagerly, such as
    `pl.read_csv`, and `TypedLazyFrame` for a function that scans, such as
    `pl.scan_csv`. It reads the argument from the abstract syntax tree
    (AST), because the type of a dict literal does not hold the data type
    of each value.
    """

    # The functions that read the class `pl.Decimal` as `Decimal(38, 0)`.
    # Every other function here raises for it.
    _DECIMAL_DEFAULTS: ClassVar[frozenset[str]] = frozenset({
        "from_dict",
        "from_dicts",
        "from_records",
    })
    # The function that reads a class of user code by its name for every data.
    _DICT_ROWS: ClassVar[str] = "from_dicts"

    def __init__(
        self,
        plugin: Plugin,
        expected_frame_type: type[_TypedFrameBase],
        operation: str,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Keep the frame kind of the function, its name, and the tools for each call.

        Args:
            plugin: The mypy plugin, which finds `Column` and the frame
                class by full name.
            expected_frame_type: The typed frame class that the function
                gives.
            operation: The name of the function, for example
                `"read_csv"`, for error messages and the `allow_unproven`
                override.
            untyped_fallback: Gives the plain frame for a call that the
                hook cannot type, and reports it unless the setting lets
                it pass.
            unchecked_fallback: Gives the data type `Unchecked` to a
                column whose data type the hook cannot read, and reports
                it unless the setting lets it pass.
        """
        self._plugin = plugin
        self._expected_target = Target.of(expected_frame_type)
        self._operation = operation
        self._untyped = untyped_fallback
        self._unchecked_fallback = unchecked_fallback
        self._bare_literal = BareLiteralReader()
        self._dtype_expr = DtypeExpr()

    def __call__(self, ctx: FunctionContext) -> MypyType:
        """Give the type of one call to a frame-building function.

        The hook tries each shape that it reads, in order, and gives the
        first result: `schema=` as a `.schema` read, `schema=` as a dict
        literal, then `data` as a dict literal.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The typed frame from `schema=`, or from the `data` dict literal
            of `pl.from_dict`. Else the plain `pl.DataFrame` or
            `pl.LazyFrame` of the kind that the function gives.
        """
        alias_frame = self._schema_alias_frame(ctx)
        if alias_frame is not None:
            return alias_frame.as_instance()
        minted = self._schema_dict_frame(ctx)
        if minted is not None:
            return minted
        minted = self._data_dict_frame(ctx)
        if minted is not None:
            return minted
        return self._untyped.resolve(ctx, Reason.NO_SCHEMA, self._operation, self._fresh_frame())

    def _fresh_frame(self) -> Frame | None:
        """Build an empty frame of the kind that the function gives, for the fallback.

        `UntypedFallback.resolve` reads a frame to choose between
        `pl.DataFrame` and `pl.LazyFrame`. A method hook gives it the
        receiver. This function has no frame argument, so the method
        builds a frame of the right kind with no columns. The fallback
        reads only its kind.

        Returns:
            A frame of the kind that the function gives, or None if mypy
            cannot find the class.

        Example:
            >>> fixture = new_type_fixture()
            >>> hook = SchemaArgFrameHook(
            ...     new_plugin(fixture),
            ...     typed_data_frame_class(),
            ...     "read_csv",
            ...     untyped_fallback(),
            ...     unchecked_fallback(),
            ... )
            >>> hook._fresh_frame().as_instance().type.fullname
            'astrolabe.frame.TypedDataFrame'
        """
        info = self._plugin.lookup_fully_qualified(self._expected_target.fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # The plugin makes every module depend on `astrolabe.frame`, so
            # mypy loads the typed frame classes even in a program that
            # never imports astrolabe. So the lookup never fails.
            return None
        return Frame(Instance(info.node, []))

    @propagates(None)
    def _schema_alias_frame(self, ctx: FunctionContext) -> Frame | None:
        """Build the result from `schema=Alias.schema`, with the columns of the alias.

        The result always has the kind that the function gives, whatever
        the kind of the alias. For example, `pl.scan_csv(...,
        schema=Prices.schema)` gives a `TypedLazyFrame` even if `Prices`
        is a `TypedDataFrame`. The alias gives only the columns.

        If the code uses a type alias such as `Prices` as a value, and not
        as an annotation, mypy gives it the type of its constructor. That
        is a `CallableType` whose return type is the frame. So the method
        reads the return type of a callable, and the type itself in any
        other case. The alias must be a closed frame, that is, its type
        names all of its columns.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            A typed frame with the columns of the alias, or None if the call
            does not pass `schema=` once, if `schema=` is not a `.schema`
            read, or if the alias is not a closed typed frame.
        """
        nodes = CallArgs(ctx).node_list("schema")
        node = nodes[0] if len(nodes) == 1 else None
        if not isinstance(node, MemberExpr) or node.name != "schema":
            return None
        receiver_type = get_proper_type(ctx.api.get_expression_type(node.expr))
        if isinstance(receiver_type, CallableType):
            frame = not_none(Frame.from_proper_type(receiver_type.ret_type))
        else:
            frame = not_none(Frame.from_proper_type(receiver_type))
        columns = not_none(frame.columns)
        return not_none(Frame.build_fresh(self._expected_target, columns, self._plugin, ctx.api))

    @propagates(None)
    def _schema_dict_frame(self, ctx: FunctionContext) -> MypyType | None:
        """Build the result from a `schema={...}` dict literal, with one column for each entry.

        A column gets the data type `Unchecked` if its value is not a data
        type that the hook can read.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            A new typed frame, or None if the call does not pass `schema=`
            once, if `schema=` is not a dict literal, or if a key is not a
            literal string or a string variable declared `Final`. None also
            if mypy cannot find a class that a column or the frame needs.
        """
        nodes = CallArgs(ctx).node_list("schema")
        dict_expr = nodes[0] if len(nodes) == 1 else None
        if not isinstance(dict_expr, DictExpr):
            return None
        columns: list[ColumnRef] = []
        for key_node, value_node in dict_expr.items:
            name = not_none(self._dict_key_name(ctx, key_node))
            dtype = not_none(self._dict_value_dtype(ctx, value_node, name))
            column = not_none(ColumnRef.build_fresh(name, dtype, ctx.api, self._plugin))
            columns.append(column)
        frame = not_none(Frame.build_fresh(self._expected_target, columns, self._plugin, ctx.api))
        return frame.as_instance()

    def _dict_value_dtype(
        self,
        ctx: FunctionContext,
        value_node: Expression,
        name: str,
    ) -> MypyType | None:
        """Read the data type of one column, or give `Unchecked` if the hook cannot.

        A class with parameters, such as `pl.Datetime`, gives the instance
        that polars builds for it. The columns of the dict literal are
        independent. One value that the hook cannot read changes only its
        own column, and the other columns keep their names and data types.

        A function that builds a frame from Python data reads the class
        `pl.Decimal` as `Decimal(38, 0)`. Every other function raises for it,
        so the method reports it there. Every function raises for a marker
        class of astrolabe that polars cannot read, such as `List[pl.Int64]`,
        so the method reports it everywhere.

        Args:
            ctx: The context that mypy gives for the call.
            value_node: The AST node of the value.
            name: The literal name of the column, for the error message.

        Returns:
            The data type, or a new `Unchecked` if the value is not a data
            type that the hook can read, or after a report of a value that
            the function refuses (see `_refused`). None if mypy cannot find
            `Unchecked`.
        """
        decimal_defaults = self._operation in self._DECIMAL_DEFAULTS
        if self._refused(ctx, value_node):
            return self._unchecked_fallback.substituted(ctx, Reason.REPORTED, self._operation, name)
        codec = ConstDtypeCodec(MarkerMinter(self._plugin, CheckerLiteralTypeBuilder(), ctx.api))
        if decimal_defaults:
            dtype = self._dtype_expr.schema_typed(ctx.api, value_node, codec)
        else:
            dtype = self._dtype_expr.typed(ctx.api, value_node, codec)
        if dtype is not None:
            return dtype
        return self._unchecked_fallback.substituted(
            ctx,
            Reason.NON_LITERAL_DTYPE,
            self._operation,
            name,
        )

    def _refused(self, ctx: FunctionContext, value_node: Expression) -> bool:
        """Report a value of `schema=` that the function refuses for every data.

        A function that builds a frame from Python data reads `pl.Decimal`
        as `Decimal(38, 0)`, so only a marker class of astrolabe is refused
        there. `pl.from_dicts` also refuses a class of user code whose
        name no polars data type has. `pl.from_dict` and `pl.from_records`
        accept such a class for some data, so they do not report it.

        Args:
            ctx: The context that mypy gives for the call.
            value_node: The AST node of the value.

        Returns:
            True if the method reported the value.
        """
        guard = DtypeClassGuard()
        if self._operation == self._DICT_ROWS:
            return guard.refused_in_dict_rows(ctx, value_node)
        if self._operation in self._DECIMAL_DEFAULTS:
            return guard.refused_marker(ctx, value_node)
        return guard.refused(ctx, value_node)

    def _dict_key_name(self, ctx: FunctionContext, key_node: Expression | None) -> str | None:
        """Read the key of one dict literal entry as a column name.

        Args:
            ctx: The context that mypy gives for the call.
            key_node: The AST node of the key, or None for a `**other`
                entry.

        Returns:
            The name, or None for a `**other` entry, or for a key that is
            not a literal string or a string variable declared `Final`.
        """
        if key_node is None:
            return None
        return self._bare_literal.string(key_node, ctx.api.get_expression_type(key_node))

    @propagates(None)
    def _data_dict_frame(self, ctx: FunctionContext) -> MypyType | None:
        """Build the result from the `data` dict literal of `pl.from_dict`.

        This runs only if `schema=` gives no frame. The keys give the
        column names. The values are lists, and the hook does not infer a
        data type from them, so each column gets `Unchecked`. The reason for
        `Unchecked` is `NO_SCHEMA`, and the fallback reports it unless the
        `allow_unproven` setting lets it pass.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            A new typed frame, or None if the call does not pass `data` once,
            if `data` is not a dict literal, or if a key is not a literal
            string or a string variable declared `Final`. None also if mypy
            cannot find `Unchecked`, `Column` or the frame class.
        """
        nodes = CallArgs(ctx).node_list("data")
        dict_expr = nodes[0] if len(nodes) == 1 else None
        if not isinstance(dict_expr, DictExpr):
            return None
        columns: list[ColumnRef] = []
        for key_node, _value_node in dict_expr.items:
            name = not_none(self._dict_key_name(ctx, key_node))
            substituted = self._unchecked_fallback.substituted(
                ctx,
                Reason.NO_SCHEMA,
                self._operation,
                name,
            )
            dtype = not_none(substituted)
            column = not_none(ColumnRef.build_fresh(name, dtype, ctx.api, self._plugin))
            columns.append(column)
        frame = not_none(Frame.build_fresh(self._expected_target, columns, self._plugin, ctx.api))
        return frame.as_instance()
