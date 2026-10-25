"""Register the hooks that mypy calls for a call of a module-level function.

mypy calls a function hook for a call of a function or of a class,
by its full name. This module registers the hooks for these calls:

- `pl.concat`, `pl.select` and `infer`.
- The functions that read or build a frame from a `schema=` argument,
  such as `pl.read_csv` and `pl.from_dict`. They share one hook class.
- `pl.Series`, and the data type classes with parameters, such as
  `pl.Datetime`, which keep their parameters in the type.
- `check_schema`, `force_schema` and `assume_schema`, which take a frame
  type as a value.

The recorder takes each function that builds an expression, such as
`pl.lit`, `pl.when` or `cs.numeric()` (`RecordingHooks`). A call of such a
function that gives no expression, as `pl.repeat(1, 3, eager=True)`, keeps
the type of its stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from mypy.plugin import FunctionContext
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.dtype_constructor_readers import (
    CheckerLiteralTypeBuilder,
    ParametrizedDtypeReaders,
)
from astrolabe._hooks.concat import ConcatHook
from astrolabe._hooks.select_function import SelectFunctionHook
from astrolabe._hooks.dispatch.generic import HookDispatcher, Target
from astrolabe._hooks.domain.unchecked import UncheckedFallback
from astrolabe._hooks.domain.untyped_fallback import UntypedFallback
from astrolabe._hooks.domain.unproven_gate import UnprovenGate
from astrolabe._hooks.inferred.infer_hook import InferHook
from astrolabe._hooks.read_schema import SchemaArgFrameHook
from astrolabe.declare import infer
from astrolabe._hooks.typed_dtype_constructor import (
    DtypeConstructorHook,
    FieldConstructorHook,
    NestedDtypeConstructorHook,
    StructDtypeConstructorHook,
)
from astrolabe._hooks.typed_series_constructor import SeriesConstructorHook
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe._hooks.runtime_schema import RuntimeSchemaHook
from astrolabe._hooks.type_analysis import BareDtypeDefaults, ClosedSchema, ImplicitDecimalPrecision
from astrolabe.frame import (
    TypedDataFrame,
    TypedLazyFrame,
    assume_schema,
    check_schema,
    force_schema,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.plugin import Plugin

    from astrolabe._hooks.unproven import UnprovenPolicy
    from astrolabe._hooks.inferred.accumulator import InferAccumulator
    from astrolabe.frame import _TypedFrameBase


class FunctionHooks:
    """Build the function hooks, and register each one by the full name of its function.

    The plugin asks this class for the hook of a function when mypy
    checks a call of it. Several hooks share the objects that apply the
    `allow_unproven` setting, so that every function reports in the
    same way.
    """

    def __init__(
        self,
        plugin: Plugin,
        allow_unproven: UnprovenPolicy,
        disallow_vertical_concat: bool,
        infer_accumulator: InferAccumulator,
    ) -> None:
        """Build every function hook, and register it in a new dispatcher.

        Args:
            plugin: The mypy plugin, which the hooks use to look up a
                class by its full name.
            allow_unproven: The resolved `allow_unproven` setting. The
                hooks that can fall back share one `UntypedFallback`
                and one `UncheckedFallback` built from it.
            disallow_vertical_concat: The setting that `ConcatHook`
                reads. See `AstrolabeConfig`.
            infer_accumulator: The store that `InferHook` fills. Code
                that types a call of a function with an inferred return
                type reads the same store.
        """
        self._dispatcher = HookDispatcher[FunctionContext, MypyType]()
        untyped_fallback = UntypedFallback(allow_unproven, plugin)
        unchecked_fallback = UncheckedFallback(allow_unproven, plugin)
        self._dispatcher.register(
            Target.of(pl.concat),
            ConcatHook(untyped_fallback, disallow_vertical_concat, plugin, unchecked_fallback),
        )
        infer_hook = InferHook(infer_accumulator, plugin, UnprovenGate(allow_unproven))
        self._dispatcher.register(Target.of(infer), infer_hook)
        self._dispatcher.register(
            Target.of(pl.select),
            SelectFunctionHook(plugin, untyped_fallback, unchecked_fallback),
        )
        self._register_schema_arg_hooks(plugin, untyped_fallback, unchecked_fallback)
        self._register_interop_constructor_hooks(plugin, untyped_fallback, unchecked_fallback)
        self._register_dtype_constructor_hooks(plugin)
        self._dispatcher.register(
            Target.of(pl.Series),
            SeriesConstructorHook(plugin, unchecked_fallback),
        )
        self._register_runtime_schema_hooks(plugin)

    def _register_runtime_schema_hooks(self, plugin: Plugin) -> None:
        """Register a `RuntimeSchemaHook` for each function that takes a frame type as a value.

        The hooks close the schema of the frame type as an annotation does.
        They build the default parameters of a bare data type class with the
        builders of the type checking phase of mypy.

        Args:
            plugin: The mypy plugin, which looks up each marker class by its
                full name.
        """
        literal_type = CheckerLiteralTypeBuilder()
        implicit_precision = ImplicitDecimalPrecision(literal_type)
        bare_defaults = BareDtypeDefaults(implicit_precision, literal_type, FreshTypeInfo(plugin))
        closed_schema = ClosedSchema(bare_defaults)
        for function in (check_schema, force_schema, assume_schema):
            hook = RuntimeSchemaHook(function.__name__, closed_schema)
            self._dispatcher.register(Target.of(function), hook)

    def _register_dtype_constructor_hooks(self, plugin: Plugin) -> None:
        """Register the hooks of the data type classes with parameters, such as `pl.Datetime`.

        The hook replaces the return type of the call with a type that
        holds the parameters of the data type. The list of classes
        comes from `ParametrizedDtypeReaders`, which also reads these
        data types elsewhere, so the list is written in one place.
        A hook of `pl.Field` reports a data type that polars refuses
        there.

        Args:
            plugin: The mypy plugin, which looks up each marker class
                by its full name.
        """
        readers = ParametrizedDtypeReaders(CheckerLiteralTypeBuilder())
        for target, reader in readers.targets_and_readers():
            self._dispatcher.register(target, DtypeConstructorHook(reader, plugin))
        lists = NestedDtypeConstructorHook(plugin, array=False)
        self._dispatcher.register(Target.of(pl.List), lists)
        arrays = NestedDtypeConstructorHook(plugin, array=True)
        self._dispatcher.register(Target.of(pl.Array), arrays)
        self._dispatcher.register(Target.of(pl.Struct), StructDtypeConstructorHook(plugin))
        self._dispatcher.register(Target.of(pl.Field), FieldConstructorHook())

    def _register_schema_arg_hooks(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Register a `SchemaArgFrameHook` for each function that reads a file with a schema.

        Each function accepts a `schema=` argument, and the hook reads
        the frame type from it. A function that reads eagerly gives a
        `TypedDataFrame`, and a function that scans gives a
        `TypedLazyFrame`.

        Args:
            plugin: The mypy plugin, which looks up `Column` and the
                frame class by full name, for a `schema={...}` literal.
            untyped_fallback: The object that falls back to the plain
                frame for a call that the checker cannot read.
            unchecked_fallback: The object that gives `Unchecked` to a
                column whose data type the checker cannot read.
        """
        frame_type_pairs: tuple[
            tuple[Callable[..., object], type[_TypedFrameBase]],
            ...,
        ] = (
            (pl.read_csv, TypedDataFrame),
            (pl.scan_csv, TypedLazyFrame),
            (pl.read_parquet, TypedDataFrame),
            (pl.scan_parquet, TypedLazyFrame),
            (pl.read_json, TypedDataFrame),
            (pl.read_ndjson, TypedDataFrame),
            (pl.scan_ndjson, TypedLazyFrame),
        )
        for function, frame_type in frame_type_pairs:
            hook = SchemaArgFrameHook(
                plugin,
                frame_type,
                function.__name__,
                untyped_fallback,
                unchecked_fallback,
            )
            self._dispatcher.register(Target.of(function), hook)

    def _register_interop_constructor_hooks(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Register a `SchemaArgFrameHook` for each function that builds a frame from Python data.

        `from_numpy`, `from_dict`, `from_dicts` and `from_records` accept
        a `schema=` argument of the same form as the file readers, so
        each one needs only a registration. Each gives a
        `TypedDataFrame`. If `pl.from_dict` has no `schema=`, the hook
        also reads the column names from a dict literal in `data`, with
        the data type `Unchecked`. The `data` of the other three
        functions is never a dict literal.

        `from_pandas`, `from_arrow` and `from_repr` are not registered.
        Each one can return a `Series` instead of a `DataFrame`,
        depending on its input, so a typed frame is not a safe answer.

        Args:
            plugin: The mypy plugin, which looks up `Column` and
                `TypedDataFrame` by full name, for a `schema={...}`
                literal.
            untyped_fallback: The object that falls back to the plain
                frame for a call that the checker cannot read.
            unchecked_fallback: The object that gives `Unchecked` to a
                column whose data type the checker cannot read.
        """
        for function in (pl.from_numpy, pl.from_dict, pl.from_dicts, pl.from_records):
            hook = SchemaArgFrameHook(
                plugin,
                TypedDataFrame,
                function.__name__,
                untyped_fallback,
                unchecked_fallback,
            )
            self._dispatcher.register(Target.of(function), hook)

    @property
    def dispatcher(self) -> HookDispatcher[FunctionContext, MypyType]:
        """The dispatcher that holds every registered function hook."""
        return self._dispatcher
