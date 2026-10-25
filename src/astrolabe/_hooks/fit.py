"""The check at each call that a frame argument fits a parameter with a declared remainder.

A function can take `TypedDataFrame[Trades, Rest]`, where
`Rest = other_columns(...)`. The remainder `Rest` stands for the columns
of the caller's frame that the type does not name, and it acts like a
type variable: each call puts the caller's own columns in its place.
mypy compares types by class, so it can never match such a parameter to
a real argument. The remainder class of the argument always differs
from `Rest`.

So this signature hook makes the check itself, at each call. First it
binds `Rest`. For a closed argument, a frame whose type names all of its
columns, `Rest` becomes the columns left over after the declared ones.
The hook checks each promise of the declaration, for example
`without=["ticker"]`, on those columns directly. For an open argument,
`Rest` becomes the argument's own remainder, and the hook proves each
promise from what that remainder is known to hold.

Then the hook works out the declared return type under that binding. A
closed caller gets back a frame with every column known. An open caller
gets back its own remainder, with the effects that the return type
states, for example a new suffix.

The same hook also runs the fit of declared expression parameters. See
`ExpressionFitHook`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mypy.nodes import Decorator, FuncDef, SymbolNode, TypeInfo
from mypy.subtypes import is_same_type
from mypy.types import CallableType, Instance, get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.calls.dtype_constructor_readers import (
    CheckerLiteralTypeBuilder,
)
from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.declared_columns import DeclaredColumnsFit
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.encoding.declaration_codec import DeclarationCodec
from astrolabe._hooks.encoding.marker_minter import MarkerMinter
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeCodec, ConstDtypeReads
from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.encoding.remainder_knowledge import (
    RemainderKnowledge,
    SlotRelations,
)
from astrolabe._hooks.errors import (
    UnplacedColumnError,
    UnplacedSplitError,
    ForbiddenColumnError,
    ForbiddenDtypeError,
    MarkerArgumentError,
    UnprovenRequirementError,
)
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe._hooks.remainder.entailment import AbsenceEntailment
from astrolabe._hooks.remainder.record import ColumnSet, Record
from astrolabe._hooks.remainder.split import Split, SplitPredicates
from astrolabe._hooks.remainder.type_set import FamilyRoster

from mypy.plugin import FunctionSigContext, MethodSigContext

from astrolabe._hooks.encoding.expr_evaluation import ExpressionArgument
from astrolabe._hooks.encoding.selection_codec import SelectionReader
from astrolabe._hooks.expression_fit import ExpressionFitHook
from astrolabe._hooks.inferred.families import family_for
from astrolabe._hooks.inferred.position import InferredMarkers, MisplacedInferredHook

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.plugin import Plugin

    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.unproven_gate import UnprovenGate
    from astrolabe._hooks.inferred.call_evaluations import CallEvaluations


class _Binding:
    """What one declared remainder stands for at one call."""


@dataclass(frozen=True)
class _ConcreteBinding(_Binding):
    """The columns of a closed argument that are left over after the declared ones.

    Attributes:
        residual: The columns of the argument that the parameter does not
            declare, in the order of the argument.
    """

    residual: tuple[ColumnRef, ...]

    def names(self) -> frozenset[str]:
        """List the names of the leftover columns.

        Returns:
            The name of each leftover column. A column with no literal name is
            left out.
        """
        return frozenset(column.name for column in self.residual if column.name is not None)


@dataclass(frozen=True)
class _SlotBinding(_Binding):
    """The remainder of an open argument, which the declared remainder stands for at the call.

    Attributes:
        rest: The remainder of the argument, in its encoded form.
    """

    rest: MypyType


@dataclass(frozen=True)
class _Bound:
    """One declared remainder of a parameter, matched with the argument at one call.

    Attributes:
        index: The position of the parameter in the signature.
        declaration: The full name of the declared remainder class.
        schema: The columns that the parameter declares.
        actual: The frame of the argument.
        binding: What the declared remainder stands for at this call.
    """

    index: int
    declaration: str
    schema: list[ColumnRef]
    actual: Frame
    binding: _Binding


class FitSignatureFactory:
    """Build the signature hook for a function or method that has declared parameters.

    mypy asks the plugin for a hook by the full name of each function that
    the code calls. Any user function can declare a parameter with a
    remainder or an expression type, so no fixed table can list them. The
    factory looks up the definition of each called function, and builds a
    hook only when a parameter needs the fit. It skips the modules of the
    standard library, polars, mypy and the plugin itself.
    """

    _SKIPPED_PREFIXES: tuple[str, ...] = (
        "builtins.",
        "typing.",
        "polars.",
        "astrolabe.",
        "mypy.",
    )

    def __init__(self, plugin: Plugin, gate: UnprovenGate, evaluations: CallEvaluations) -> None:
        """Keep the plugin, which looks up each definition, the gate and the store of bindings.

        Args:
            plugin: The mypy plugin.
            gate: The gate that decides, under the `allow_unproven` setting,
                whether to report a promise or an argument that the fit cannot
                prove.
            evaluations: The store that passes the binding of a call to the
                hook that gives a function with an inferred return type
                its type at that call.
        """
        self._plugin = plugin
        self._gate = gate
        self._evaluations = evaluations

    @propagates(None)
    def hook_for(self, fullname: str) -> Callable[[FunctionSigContext], CallableType] | None:
        """Give the fit hook for one function, if it needs one.

        The method looks up the declared signature of the function, and
        builds a hook only if a parameter needs the fit.

        Args:
            fullname: The full name of the called function.

        Returns:
            The hook, or None for a function with no declared expression
            parameter and no open frame parameter. None also if the name is
            not a plain function with a callable type.
        """
        if fullname.startswith(self._SKIPPED_PREFIXES):
            return None
        node = self._plugin.lookup_fully_qualified(fullname)
        if node is None or not isinstance(node.node, FuncDef):
            return None
        signature = node.node.type
        if not isinstance(signature, CallableType):
            return None
        return SignatureFitHook(not_falsy(self.hooks_for(signature)))

    @propagates(None)
    def method_hook_for(self, fullname: str) -> Callable[[MethodSigContext], CallableType] | None:
        """Give the fit hook for one method, if it needs one.

        The method looks up the declared signature of the method, and builds
        a hook only if a parameter needs the fit. The declared signature of a
        method holds `self`, and the signature at a call does not. The
        positions of the parameters therefore differ, so the hook builds its
        checks again from the signature of each call.

        Args:
            fullname: The full name of the called method, for example
                `mod.Cls.method`.

        Returns:
            The hook, or None for a method with no open frame parameter and no
            declared expression parameter. None also if the name is not a
            method, or the method has a decorator other than `staticmethod` or
            `classmethod`.
        """
        if fullname.startswith(self._SKIPPED_PREFIXES):
            return None
        node = self._plugin.lookup_fully_qualified(fullname)
        function = self._undecorated(node.node if node is not None else None)
        if function is None or not isinstance(function.type, CallableType):
            return None
        not_falsy(self.hooks_for(function.type))
        return MethodFitHook(self)

    def hooks_for(
        self,
        signature: CallableType,
    ) -> tuple[Callable[[FunctionSigContext], CallableType], ...]:
        """Build the fit checks that one signature needs, one for each kind of declared parameter.

        A check that reports an inferred return type out of place runs first.
        It runs only if a parameter type, or the return type below its top,
        holds an inferred return type. The expression fit runs next, for each
        parameter whose type is a declared expression or selection. The
        remainder fit then runs for each parameter whose type is a frame with
        a remainder. It reads a signature in which the expression parameters
        already have the argument types.

        Args:
            signature: The declared signature of the function.

        Returns:
            The checks, in the order they run.
        """
        hooks: list[Callable[[FunctionSigContext], CallableType]] = []
        misplaced = InferredMarkers().misplaced(signature)
        if misplaced:
            hooks.append(MisplacedInferredHook(misplaced))
        declared: list[int] = []
        open_frames: list[int] = []
        for index, arg_type in enumerate(signature.arg_types):
            if self._is_declared(arg_type):
                declared.append(index)
            if self._is_open_frame(arg_type):
                open_frames.append(index)
        if declared:
            hooks.append(ExpressionFitHook(self._plugin, self._gate, tuple(declared)))
        if open_frames:
            remainder_fit = RemainderFitHook(
                self._plugin,
                self._gate,
                tuple(open_frames),
                self._evaluations,
            )
            hooks.append(remainder_fit)
        return tuple(hooks)

    def _is_declared(self, declared: MypyType) -> bool:
        """Tell whether one parameter has a declared expression type or selection type.

        Args:
            declared: The declared type of the parameter.

        Returns:
            True for an expression type or a selection.
        """
        if ExpressionArgument().decoded(declared) is not None:
            return True
        return SelectionReader().decoded(declared) is not None

    def _undecorated(self, node: SymbolNode | None) -> FuncDef | None:
        """Give the function of a method, if no decorator changes its type.

        mypy takes `staticmethod` and `classmethod` out of the list of
        decorators, so an empty list means that the declared type applies. Any
        other decorator can change the signature.

        Args:
            node: The node that the name of the method points to.

        Returns:
            The function, or None for an overload, a method with a decorator, or
            a node that is not a method.
        """
        if isinstance(node, Decorator) and not node.decorators:
            return node.func
        return node if isinstance(node, FuncDef) else None

    def _is_open_frame(self, declared: MypyType) -> bool:
        """Tell whether one parameter is a frame with a remainder.

        Args:
            declared: The declared type of the parameter.

        Returns:
            True if the type is a typed frame that has a remainder.
        """
        frame = Frame.from_proper_type(get_proper_type(declared))
        if frame is None:
            return False
        known = frame.known_columns
        return known is not None and known.rest is not None


class SignatureFitHook:
    """Run each fit check in turn on the signature of one function call.

    Each check reads the signature that the check before it gave.
    """

    def __init__(self, hooks: tuple[Callable[[FunctionSigContext], CallableType], ...]) -> None:
        """Keep the checks.

        Args:
            hooks: The checks, in the order they run.
        """
        self._hooks = hooks

    def __call__(self, ctx: FunctionSigContext) -> CallableType:
        """Give the signature of one call after every check.

        Args:
            ctx: The mypy context of the function signature at the call.

        Returns:
            The signature after each check changed it in turn.
        """
        signature = ctx.default_signature
        for hook in self._hooks:
            signature = hook(FunctionSigContext(ctx.args, signature, ctx.context, ctx.api))
        return signature


class MethodFitHook:
    """Run the fit on one method call, from the signature of that call.

    The positions of the parameters are known only at the call, where
    `self` is already bound, so this hook builds the checks from the
    signature of each call.
    """

    def __init__(self, factory: FitSignatureFactory) -> None:
        """Keep the factory that builds the checks.

        Args:
            factory: The factory that builds the checks for one signature.
        """
        self._factory = factory

    def __call__(self, ctx: MethodSigContext) -> CallableType:
        """Give the signature of one method call after every check that it needs.

        Args:
            ctx: The mypy context of the method signature at the call.

        Returns:
            The signature of the call after each check changed it.
        """
        signature = ctx.default_signature
        hooks = SignatureFitHook(self._factory.hooks_for(signature))
        return hooks(FunctionSigContext(ctx.args, signature, ctx.context, ctx.api))


class RemainderFitHook:
    """The fit check for the open frame parameters of one function.

    The factory builds one for each function with such parameters, with
    their positions. The work for one call happens in a fresh `_CallFit`.
    """

    def __init__(
        self,
        plugin: Plugin,
        gate: UnprovenGate,
        indices: tuple[int, ...],
        evaluations: CallEvaluations,
    ) -> None:
        """Keep the plugin, the gate, the positions of the open parameters and the store.

        Args:
            plugin: The mypy plugin.
            gate: The gate that decides whether to report a promise that the
                fit cannot prove.
            indices: The position of each open frame parameter.
            evaluations: The store that passes the binding of a call to the
                hook that types a function with an inferred return type.
        """
        self._plugin = plugin
        self._gate = gate
        self._indices = indices
        self._evaluations = evaluations

    def __call__(self, ctx: FunctionSigContext) -> CallableType:
        """Give the signature of one call after the fit.

        Args:
            ctx: The mypy context of the function signature at the call.

        Returns:
            The signature, with each open parameter replaced by the type of its
            argument, and the return type worked out under the binding. The
            declared signature as it is, if the call does not have a form that
            the fit can read.
        """
        call_fit = _CallFit(ctx, self._plugin, self._gate, self._indices)
        fitted = call_fit.adjusted(self._evaluations)
        if fitted is None:
            return ctx.default_signature
        return fitted


class _CallFit:
    """The fit of every open frame parameter for one call.

    It holds the values that stay the same for the whole call: the context,
    the declared signature, the plugin and the positions of the open
    parameters. Each step reads them from the instance.

    Several steps read a record. A record is what the engine knows about a
    remainder: the names that it cannot hold, and the data types that its
    columns can have. A declaration has a record that states its promises,
    and an open argument has a record that states what its own
    declaration promises.
    """

    def __init__(
        self,
        ctx: FunctionSigContext,
        plugin: Plugin,
        gate: UnprovenGate,
        indices: tuple[int, ...],
    ) -> None:
        """Keep the values that stay the same for the whole call.

        Args:
            ctx: The mypy context of the function signature at the call.
            plugin: The mypy plugin.
            gate: The gate that decides whether to report a promise that the
                fit cannot prove.
            indices: The position of each open frame parameter.
        """
        self._ctx = ctx
        self._sig = ctx.default_signature
        self._plugin = plugin
        self._gate = gate
        self._indices = indices
        self._knowledge = RemainderKnowledge()
        self._declarations = DeclarationCodec()
        self._unchecked = UncheckedCompatibility()

    @propagates(None)
    def adjusted(self, evaluations: CallEvaluations) -> CallableType | None:
        """Build the signature of the call after the fit.

        A function with an inferred return type gets its type at the call
        from the values that its body returns. Each value is written in the
        terms of the function, so the fit keeps its binding in
        `evaluations` for the hook that types the call.

        Args:
            evaluations: The store of the binding of each such call.

        Returns:
            The new signature, or None if the call does not have a form that the
            fit can read. The declared signature then applies.
        """
        bounds = [one for index in self._indices for one in not_none(self._bounds(index))]
        self._check_requirements(bounds)
        if family_for(self._sig.ret_type) is not None:
            evaluations.record(self._ctx.context, _BoundEvaluation(self, bounds))
        return self._sig.copy_modified(
            arg_types=self._replaced_arg_types(bounds),
            ret_type=self.evaluated(self._sig.ret_type, bounds),
        )

    @propagates(None)
    def _bounds(self, index: int) -> list[_Bound] | None:
        """Match one open parameter with its argument, one declared remainder at a time.

        A parameter can declare its remainder as `SplitColumns[...]`, a list of
        parts where each part is its own declared remainder. Such a parameter
        gives one match for each part, and every later step reads the matches
        in the same way.

        Args:
            index: The position of the parameter.

        Returns:
            One match for each declared part, or None if the parameter or the
            argument does not have a form that the fit can read. None also if
            the declaration of a part cannot be read, or the parts cannot
            divide the binding.
        """
        declared = not_none(self._declared_known(index))
        names = not_none(self._knowledge.part_declarations(not_none(declared.rest)))
        actual = not_none(self._actual_frame(index))
        actual_known = not_none(actual.known_columns)
        binding = not_none(self._binding(declared, actual_known))
        if len(names) == 1:
            return [_Bound(index, names[0], declared.columns, actual, binding)]
        records = tuple(not_none(self._record_named(name)) for name in names)
        split = not_none(SplitPredicates().split(records))
        bindings = not_none(self._part_bindings(split, binding))
        return [
            _Bound(index, name, declared.columns, actual, part)
            for name, part in zip(names, bindings, strict=True)
        ]

    def _part_bindings(self, split: Split, binding: _Binding) -> list[_Binding] | None:
        """Divide the binding of one parameter among the parts of its split.

        Args:
            split: The declared split.
            binding: The binding of the whole parameter.

        Returns:
            One binding for each part, or None for a binding that the split
            cannot divide.
        """
        if isinstance(binding, _ConcreteBinding):
            return self._concrete_parts(split, binding)
        if isinstance(binding, _SlotBinding):
            return self._slot_parts(split, binding)
        return None  # pragma: no cover -- a binding is always one of the two kinds

    def _concrete_parts(self, split: Split, binding: _ConcreteBinding) -> list[_Binding]:
        """Put each leftover column of a closed argument into the first part that accepts it.

        The method reports each column of the caller that no part of the split
        accepts. `Split.bound` leaves out such a column, so without the report
        the call gives a smaller result in silence.

        Args:
            split: The declared split.
            binding: The binding of the parameter.

        Returns:
            One binding for each part, each in the column order of the argument.
        """
        columns = _ResidualColumns(binding.residual).column_set()
        for name in split.unplaced(columns):
            UnplacedColumnError(name).fail(self._ctx)
        parts: list[_Binding] = []
        for part in split.bound(columns):
            names = set(part.names())
            kept = tuple(column for column in binding.residual if column.name in names)
            parts.append(_ConcreteBinding(kept))
        return parts

    def _slot_parts(self, split: Split, binding: _SlotBinding) -> list[_Binding]:
        """Narrow the remainder of an open argument to the columns of each part.

        Part `i` gets the caller's remainder, narrowed to the columns that its
        rule accepts and that no earlier part accepts. That is the closed form
        of the part.

        The method also reports a split that cannot place every column that an
        open caller can hold. An open argument names no columns, so the
        message names none either. The method reports only for a caller whose
        remainder promises nothing. If the caller's own declaration rules out
        the missing columns, there is no column to place, and a report is
        wrong.

        Args:
            split: The declared split.
            binding: The binding of the parameter.

        Returns:
            One binding for each part.
        """
        caller = self._knowledge.record(binding.rest)
        if caller == Record.unconstrained() and split.has_gap():
            UnplacedSplitError().fail(self._ctx)
        editor = RemainderEdit.for_checker(self._plugin, self._ctx.api)
        return [
            _SlotBinding(editor.narrowed(binding.rest, split.closed_form(index)))
            for index in range(len(split.parts))
        ]

    def _record_named(self, fullname: str) -> Record | None:
        """Read what one declaration promises, from the full name of its class.

        The engine calls this a record: which names the remainder cannot hold,
        and which data types its columns can have.

        Args:
            fullname: The full name of the declaration class.

        Returns:
            The record, or None if the class is not known.
        """
        info = self._plugin.lookup_fully_qualified(fullname)
        if info is None or not isinstance(info.node, TypeInfo):  # pragma: no cover
            # The plugin accepts a split in an annotation only when each
            # part is a declared remainder class, so the lookup always
            # finds one.
            return None
        return self._declarations.record_of(info.node)

    def _declared_known(self, index: int) -> KnownColumns | None:
        """Read the declared columns and the declared remainder of one parameter.

        Args:
            index: The position of the parameter.

        Returns:
            The declared columns and remainder, or None if the parameter is not
            a typed open frame.
        """
        frame = Frame.from_proper_type(get_proper_type(self._sig.arg_types[index]))
        if frame is None:  # pragma: no cover
            # The factory builds this hook only after it found that the
            # parameter is a typed open frame.
            return None
        known = frame.known_columns
        if known is None or known.rest is None:  # pragma: no cover
            # The same test in the factory covers the remainder.
            return None
        return known

    def _actual_frame(self, index: int) -> Frame | None:
        """Read the type of one argument as a frame.

        Args:
            index: The position of the parameter.

        Returns:
            The frame of the argument, or None if the call does not pass one
            expression there, or the expression is not a typed frame.
        """
        if index >= len(self._ctx.args) or len(self._ctx.args[index]) != 1:
            return None
        actual = self._ctx.api.get_expression_type(self._ctx.args[index][0])
        return Frame.from_proper_type(get_proper_type(actual))

    def _binding(
        self,
        declared: KnownColumns,
        actual: KnownColumns,
    ) -> _Binding | None:
        """Bind the declared remainder of one parameter at this call.

        Args:
            declared: The declared columns and remainder of the parameter.
            actual: The columns and remainder of the argument.

        Returns:
            The binding. None if the argument is open and its declared columns
            differ from those of the parameter, a case that the fit cannot
            express.
        """
        if actual.rest is None:
            return self._concrete_binding(declared, actual)
        return self._slot_binding(declared, actual)

    def _concrete_binding(
        self,
        declared: KnownColumns,
        actual: KnownColumns,
    ) -> _ConcreteBinding:
        """Bind the declared remainder to the leftover columns of a closed argument.

        The method also reports a declared column that the argument lacks or
        has with another data type. The call is wrong for this caller in any
        case, and a report here names the exact column.

        Args:
            declared: The declared columns and remainder of the parameter.
            actual: The columns of the argument.

        Returns:
            The binding to the leftover columns: the columns of the argument
            that the parameter does not declare, in the order of the argument.
        """
        DeclaredColumnsFit(self._gate, "fit").check(self._ctx, declared.columns, actual)
        declared_names = {column.name for column in declared.columns if column.name is not None}
        residual = tuple(column for column in actual.columns if column.name not in declared_names)
        return _ConcreteBinding(residual)

    def _slot_binding(self, declared: KnownColumns, actual: KnownColumns) -> _SlotBinding | None:
        """Bind the declared remainder to the remainder of an open argument.

        The declared columns of the argument must be exactly those of the
        parameter. A remainder expression cannot take in an extra known column,
        so for such an argument the method gives None, and mypy compares the
        types by class. Each pair of columns with one name must have the same
        data type, or one side must be `Unchecked`.

        Args:
            declared: The declared columns and remainder of the parameter.
            actual: The columns and remainder of the argument.

        Returns:
            The binding, or None if the declared columns or their data types
            differ.
        """
        declared_by_name = {column.name: column for column in declared.columns}
        actual_by_name = actual.by_name()
        if set(declared_by_name) != set(actual_by_name):
            return None
        for name, column in actual_by_name.items():
            declared_dtype = declared_by_name[name].dtype
            if self._unchecked.resolved(declared_dtype, column.dtype) is None:
                return None
        return _SlotBinding(not_none(actual.rest))

    def _check_requirements(self, bounds: list[_Bound]) -> None:
        """Check the promises of each bound declaration, and report each one that fails.

        A promise about the columns of one remainder is checked on its own.
        A promise that relates two remainders, `disjoint_from=` or `within=`,
        goes to `_RelationFit`.

        Args:
            bounds: Every match of the call.
        """
        infos = {bound.declaration: self._info_of(bound) for bound in bounds}
        for bound in bounds:
            self._check_record(bound, infos[bound.declaration])
        _RelationFit(self._ctx, self._gate, bounds, infos).check()

    def _info_of(self, bound: _Bound) -> TypeInfo | None:
        """Read the class of one bound declaration from the declared remainder of the parameter.

        Args:
            bound: The match.

        Returns:
            The declaration class, or None if the remainder of the parameter
            cannot be read.
        """
        known = self._declared_known(bound.index)
        if known is None or known.rest is None:  # pragma: no cover
            # An earlier step of the same call already read this
            # parameter as an open frame, so a second read cannot fail.
            return None
        return self._knowledge.declarations(known.rest).get(bound.declaration)

    def _check_record(self, bound: _Bound, info: TypeInfo | None) -> None:
        """Check the promises of one declaration about its own columns, and report a failure.

        The record of a declaration holds its promises: the names that it
        excludes and the data types that it allows. A closed argument is
        checked on its leftover columns, and an open argument on its known
        remainder.

        Args:
            bound: The match.
            info: The declaration class, or None.
        """
        # `_info_of` gives None only on a branch that never runs.
        demanded = None if info is None else self._declarations.record_of(info)
        if demanded is None:  # pragma: no cover
            # The declaration hook leaves out a wrong promise at the
            # declaration, so the record of a declared class can always
            # be read.
            self._gate.report(self._ctx, UnprovenRequirementError(self._short(bound.declaration)))
            return
        if isinstance(bound.binding, _ConcreteBinding):
            self._check_concrete(bound, demanded, bound.binding)
            return
        self._check_slot(bound, demanded, checked_cast(bound.binding, _SlotBinding))

    def _check_concrete(self, bound: _Bound, demanded: Record, binding: _ConcreteBinding) -> None:
        """Check the promises of a declaration on the leftover columns of a closed argument.

        The method reports the first excluded name, in name order. If no name
        is excluded, it reports the first data type, in text order, that the
        promise about data types does not admit. An `Unchecked` column can
        have any data type, and it passes a promise if some data type passes.
        So `Unchecked` fails only a promise that no data type meets.

        Args:
            bound: The match.
            demanded: The record of the declaration.
            binding: The leftover columns.
        """
        short = self._short(bound.declaration)
        columns = _ResidualColumns(binding.residual).column_set()
        for name in sorted(columns.names()):
            if demanded.absence.holds_of(name):
                ForbiddenColumnError(short, name).fail(self._ctx)
                return
        checked = columns.type_set().members
        if not demanded.types.is_empty():
            checked -= {"Unchecked"}
        refused = [rendering for rendering in checked if not self._admits(demanded, rendering)]
        outside = sorted(refused)
        if outside:
            ForbiddenDtypeError(short, outside[0]).fail(self._ctx)

    def _admits(self, demanded: Record, rendering: str) -> bool:
        """Answer whether a promise about data types admits one leftover column.

        A column of an abstract polars class, for example `pl.FloatType` or
        `pl.DataType`, can hold any kind of its family, so the promise must
        admit each of them.

        Args:
            demanded: The record of the declaration.
            rendering: The data type of the column, in the words of the
                engine.

        Returns:
            True where the promise admits every data type that the column
            can hold.
        """
        family = FamilyRoster().required(rendering)
        members = family.members if family is not None else frozenset({rendering})
        return all(demanded.types.holds(member) for member in members)

    def _check_slot(self, bound: _Bound, demanded: Record, binding: _SlotBinding) -> None:
        """Prove the promises of a declaration from the known remainder of an open argument.

        Args:
            bound: The match.
            demanded: The record of the declaration.
            binding: The bound remainder.
        """
        caller = self._knowledge.record(binding.rest)
        if caller is None or not self._slot_proven(demanded, caller):
            self._gate.report(self._ctx, UnprovenRequirementError(self._short(bound.declaration)))

    def _slot_proven(self, demanded: Record, caller: Record) -> bool:
        """Tell whether the record of an open argument proves the promises of the declaration.

        A record that allows no column proves every promise.

        Args:
            demanded: The record of the declaration.
            caller: The record of the argument.

        Returns:
            True if every name that the declaration excludes, the argument
            excludes too, and every data type that the argument allows, the
            declaration allows too.
        """
        names_proven = AbsenceEntailment()(demanded.absence, caller.absence)
        types_proven = caller.types.within(demanded.types)
        return caller.permits_nothing() or (names_proven and types_proven)

    def _short(self, fullname: str) -> str:
        """Give the short name of a declaration, for a message.

        Args:
            fullname: The full name of the declaration.

        Returns:
            The part of the name after the last dot.
        """
        return fullname.rsplit(".", 1)[-1]

    def _replaced_arg_types(self, bounds: list[_Bound]) -> list[MypyType]:
        """Replace the type of each open parameter with the type of its argument.

        mypy then compares each argument with its own type, and adds no error
        of its own. The fit already made the real check.

        Args:
            bounds: Every match of the call.

        Returns:
            The parameter types of the signature, with each open position set
            to the type of its argument.
        """
        arg_types = list(self._sig.arg_types)
        for bound in bounds:
            arg_types[bound.index] = bound.actual.as_instance()
        return arg_types

    def evaluated(self, frame_type: MypyType, bounds: list[_Bound]) -> MypyType:
        """Give a frame type of the function, worked out under the binding of the call.

        The declared return type is one such type. A value that the body of
        a function with an inferred return type returns is another.

        Args:
            frame_type: The frame type, in the terms of the function.
            bounds: Every match of the call.

        Returns:
            The new frame type. The same type if it is not an open frame, or
            its remainder names a variable that no argument binds.
        """
        evaluated = self._evaluated_return(frame_type, bounds)
        if evaluated is None:
            return frame_type
        return evaluated

    @propagates(None)
    def _evaluated_return(self, frame_type: MypyType, bounds: list[_Bound]) -> MypyType | None:
        """Work out the remainder of a frame type under the binding.

        Args:
            frame_type: The frame type, in the terms of the function.
            bounds: Every match of the call.

        Returns:
            The new frame, or None if the type is not an open frame.
        """
        frame = not_none(Frame.from_proper_type(get_proper_type(frame_type)))
        known = not_none(frame.known_columns)
        rest = not_none(known.rest)
        kinds = {type(bound.binding) for bound in bounds}
        if kinds <= {_SlotBinding}:
            return self._substituted_return(bounds, frame, known, rest)
        if kinds == {_ConcreteBinding}:
            return self._denoted_return(bounds, frame, known, rest)
        return self._mixed_return(bounds, frame, known, rest)

    def _substituted_return(
        self,
        bounds: list[_Bound],
        frame: Frame,
        known: KnownColumns,
        rest: MypyType,
    ) -> MypyType | None:
        """Put the callers' remainders into the remainder of the return type.

        Args:
            bounds: Every match of the call. Each one binds a remainder.
            frame: The declared return frame.
            known: The declared columns and remainder of the return frame.
            rest: The declared remainder of the return frame, in its encoded
                form.

        Returns:
            The return frame with the new remainder. None if nothing changes,
            so that the declared return type, with its own names, applies.
        """
        editor = RemainderEdit.for_checker(self._plugin, self._ctx.api)
        pairs = [(bound.declaration, self._slot_of(bound)) for bound in bounds]
        slots: dict[str, MypyType | None] = dict(pairs)
        substituted = editor.substituted(rest, slots)
        if substituted is None or is_same_type(substituted, rest):
            return None
        rebuilt = frame.rebuilt(known.columns, self._ctx.api, rest=substituted)
        return rebuilt.as_instance()

    def _slot_of(self, bound: _Bound) -> MypyType | None:
        """Read the remainder that one match binds.

        Args:
            bound: The match.

        Returns:
            The bound remainder, or None, the empty remainder, for a match
            that binds leftover columns.
        """
        if isinstance(bound.binding, _SlotBinding):
            return bound.binding.rest
        return None

    @propagates(None)
    def _denoted_return(
        self,
        bounds: list[_Bound],
        frame: Frame,
        known: KnownColumns,
        rest: MypyType,
    ) -> MypyType | None:
        """Work out the returned remainder as real columns, and add them after the declared ones.

        Every argument is closed, so the checker knows each column that each
        declared remainder stands for. The return type then becomes a closed
        frame.

        Args:
            bounds: Every match of the call. Each one binds leftover columns.
            frame: The declared return frame.
            known: The declared columns and remainder of the return frame.
            rest: The declared remainder of the return frame, in its encoded
                form.

        Returns:
            The closed return frame, with the new columns after the declared
            ones.
        """
        editor = RemainderEdit.for_checker(self._plugin, self._ctx.api)
        spliced = not_none(self._closed_half(editor, bounds, rest))
        rebuilt = frame.rebuilt([*known.columns, *spliced], self._ctx.api, rest=None)
        return rebuilt.as_instance()

    @propagates(None)
    def _mixed_return(
        self,
        bounds: list[_Bound],
        frame: Frame,
        known: KnownColumns,
        rest: MypyType,
    ) -> MypyType | None:
        """Work out the returned remainder when some arguments are closed and some are open.

        Every effect on a remainder acts on each column alone, so the
        remainder splits into the columns that the closed arguments give and
        the columns that the open arguments give. The closed half is known
        columns, which join the declared ones. The open half stays a
        remainder, written in the terms of the callers' own remainders. A
        frame type cannot have a remainder that holds a declared name, so the
        open half has the names of the closed half taken out.

        Args:
            bounds: Every match of the call. Some bind leftover columns, and
                some bind a remainder.
            frame: The declared return frame.
            known: The declared columns and remainder of the return frame.
            rest: The declared remainder of the return frame, in its encoded
                form.

        Returns:
            The return frame, with the closed half as declared columns and the
            open half as its remainder. None if the remainder names a variable
            that no argument binds.
        """
        editor = RemainderEdit.for_checker(self._plugin, self._ctx.api)
        spliced = not_none(self._closed_half(editor, bounds, rest))
        pairs = [(bound.declaration, self._slot_of(bound)) for bound in bounds]
        open_half = editor.substituted(rest, dict(pairs))
        names = [column.name for column in spliced if column.name is not None]
        remainder = editor.minus(open_half, names)
        rebuilt = frame.rebuilt([*known.columns, *spliced], self._ctx.api, rest=remainder)
        return rebuilt.as_instance()

    @propagates(None)
    def _closed_half(
        self,
        editor: RemainderEdit,
        bounds: list[_Bound],
        rest: MypyType,
    ) -> list[ColumnRef] | None:
        """Give the columns of a remainder that the closed arguments give.

        Each closed argument binds its leftover columns, and each open one
        binds no column. Every effect acts on each column alone, so this is
        exactly the part of the remainder that comes from the closed
        arguments.

        Args:
            editor: The editor of remainder types for this call.
            bounds: Every match of the call.
            rest: The declared remainder, in its encoded form.

        Returns:
            The columns, in the order of the worked-out remainder. None if the
            remainder names a variable that no argument binds.
        """
        residuals = _ResidualColumns.merged(bounds)
        column_sets = {
            bound.declaration: _ResidualColumns(self._residual_of(bound)).column_set()
            for bound in bounds
        }
        denoted = not_none(editor.denoted(rest, column_sets))
        return [not_none(self._spliced_column(entry, residuals)) for entry in denoted.entries]

    def _residual_of(self, bound: _Bound) -> tuple[ColumnRef, ...]:
        """Read the leftover columns that one match binds.

        Args:
            bound: The match.

        Returns:
            The leftover columns, or no column for a match that binds a
            remainder.
        """
        if isinstance(bound.binding, _ConcreteBinding):
            return bound.binding.residual
        return ()

    def _spliced_column(
        self,
        entry: tuple[str, str],
        residuals: dict[str, ColumnRef],
    ) -> ColumnRef | None:
        """Build the column for one name and data type of the worked-out remainder.

        If a leftover column already has that name and data type, the method
        keeps it with its exact data type. A name or a data type that an effect
        changed gets a new column.

        Args:
            entry: One pair of a name and a data type, in the words of the
                engine.
            residuals: The leftover columns of the callers, by name.

        Returns:
            The column, or None if the words of the engine name no data type
            that the checker knows.
        """
        name, kind = entry
        existing = residuals.get(name)
        if existing is not None and _ResidualColumns.kind_text(existing.dtype) == kind:
            return existing
        minter = MarkerMinter(self._plugin, CheckerLiteralTypeBuilder(), self._ctx.api)
        dtype = ConstDtypeCodec(minter).built(kind)
        if dtype is None:  # pragma: no cover
            # The data type of a typed column is always a polars class or
            # a marker, and its text always gives the same type back.
            return None
        return ColumnRef.build_fresh(name, dtype, self._ctx.api, self._plugin)


@dataclass(frozen=True)
class _BoundEvaluation:
    """Work out a frame type of a function under the binding of one call.

    The fit of a call binds each declared remainder of the function. A
    function with an inferred return type gives its call the types of the
    values that its body returns, and these are written in the terms of
    the function. This object applies the binding of the call to each of
    them, as the fit applies it to a declared return type.

    Attributes:
        fit: The fit of the call.
        bounds: Every match of the call.
    """

    fit: _CallFit
    bounds: list[_Bound]

    def evaluated(self, frame_type: MypyType) -> MypyType:
        """Give one frame type in the terms of the call.

        Args:
            frame_type: The frame type, in the terms of the function.

        Returns:
            The frame type under the binding of the call.
        """
        return self.fit.evaluated(frame_type, self.bounds)


class _ResidualColumns:
    """Read the leftover columns of a closed argument as the plain values that the engine reads.

    The engine knows nothing of mypy. It reads a set of columns as pairs of
    a name and a data type written as text, for example `"Int64"` or
    `"Datetime('us')"`.
    """

    def __init__(self, residual: tuple[ColumnRef, ...]) -> None:
        """Keep the leftover columns.

        Args:
            residual: The leftover columns of one binding.
        """
        self._residual = residual

    def column_set(self) -> ColumnSet:
        """Give the leftover columns as pairs of a name and a data type.

        Returns:
            The set of columns that the records and the effects of the engine
            read.
        """
        return ColumnSet.holding(
            (column.name, self.kind_text(column.dtype))
            for column in self._residual
            if column.name is not None
        )

    @staticmethod
    def kind_text(dtype: MypyType) -> str:
        """Write the data type of one column as the engine writes it.

        A data type with parameters keeps them. So a promise about one data
        type, for example `all_of=pl.Datetime("us")`, is checked on the
        parameters of the column.

        Args:
            dtype: The data type of the column.

        Returns:
            The data type as text. The bare class name for a class that the
            engine has no text for. `Unchecked` for a type that is not a class
            instance.

        Example:
            >>> from mypy.types import UninhabitedType
            >>> fixture = new_type_fixture()
            >>> _ResidualColumns.kind_text(int_dtype(fixture))
            'Int64'
            >>> _ResidualColumns.kind_text(UninhabitedType())
            'Unchecked'
        """
        proper = get_proper_type(dtype)
        if not isinstance(proper, Instance):
            return "Unchecked"
        return ConstDtypeReads().read(proper) or proper.type.name.rsplit(".", 1)[-1]

    @staticmethod
    def merged(bounds: list[_Bound]) -> dict[str, ColumnRef]:
        """Merge the leftover columns of every closed binding, by name.

        Args:
            bounds: Every match of the call.

        Returns:
            Each leftover column by name. If two bindings share a name, the
            later one wins.
        """
        merged: dict[str, ColumnRef] = {}
        for bound in bounds:
            merged.update(_ResidualColumns._named(bound))
        return merged

    @staticmethod
    def _named(bound: _Bound) -> dict[str, ColumnRef]:
        """Read the leftover columns of one binding, by name.

        Args:
            bound: The match.

        Returns:
            The leftover columns by name. An empty dictionary for a binding to
            a remainder.
        """
        if not isinstance(bound.binding, _ConcreteBinding):  # pragma: no cover
            # `merged` runs only when every binding is a set of columns.
            return {}
        return {column.name: column for column in bound.binding.residual if column.name is not None}


class _RelationFit:
    """Prove the promises that relate one declared remainder to another.

    `disjoint_from=` promises that two remainders share no column name, and
    `within=` promises that every name of one remainder is also in others.
    Such a promise names other declarations, so the proof reads what each
    of those stands for at this call.

    A remainder bound to nothing but its own declaration keeps its
    promises by definition. A value that only holds fewer columns than its
    declaration still keeps every promise of that declaration.
    """

    def __init__(
        self,
        ctx: FunctionSigContext,
        gate: UnprovenGate,
        bounds: list[_Bound],
        infos: dict[str, TypeInfo | None],
    ) -> None:
        """Keep the matches of the call and the readers that the proof needs.

        Args:
            ctx: The mypy context of the function signature at the call.
            gate: The gate that decides whether to report a promise that the
                fit cannot prove.
            bounds: Every match of the call.
            infos: The class of each bound declaration, by full name.
        """
        self._ctx = ctx
        self._gate = gate
        self._knowledge = RemainderKnowledge()
        self._declarations = DeclarationCodec()
        self._relations = SlotRelations()
        self._by_declaration = {bound.declaration: bound for bound in bounds}
        self._bounds = bounds
        self._infos = infos

    def check(self) -> None:
        """Prove the relations of every bound declaration, and report each one that fails.

        A binding that is the declaration itself keeps its own promises, so
        the method skips it.
        """
        for bound in self._bounds:
            if self._own_contract(bound):
                continue
            info = self._infos.get(bound.declaration)
            if info is None:  # pragma: no cover
                # The remainder of the parameter always holds its
                # declaration, and an earlier step reported a missing one.
                continue
            self._check_relations(bound, info)

    def _check_relations(self, bound: _Bound, info: TypeInfo) -> None:
        """Prove each relation that one declaration states, and report each one that fails.

        The method proves each `disjoint_from=` promise, then each `within=`
        promise.

        Args:
            bound: The match.
            info: The declaration class.
        """
        relations = self._declarations.relations_of(info)
        if relations is None:
            # A user can write a relational base by hand, for example
            # `DisjointFrom[None]`, which names no remainder.
            error = MarkerArgumentError(
                f'"{info.name}" states a relation that names no remainder: DisjointFrom[...] '
                "and Within[...] take classes declared with other_columns(...)"
            )
            error.fail(self._ctx)
            return
        for group in relations.disjoint_from:
            for member in group:
                if not self._disjoint_proven(bound, member):
                    self._gate.report(self._ctx, UnprovenRequirementError(self._short(bound)))
        for group in relations.within:
            self._check_within(bound, group)

    def _own_contract(self, bound: _Bound) -> bool:
        """Tell whether the binding is the declaration itself, which keeps its own promises.

        Args:
            bound: The match.

        Returns:
            True if the bound remainder refers to no declaration other than the
            one that states the promise.
        """
        if not isinstance(bound.binding, _SlotBinding):
            return False
        referenced = set(self._knowledge.declarations(bound.binding.rest))
        return referenced <= {bound.declaration}

    def _disjoint_proven(self, bound: _Bound, member: str) -> bool:
        """Tell whether one `disjoint_from=` promise is proven.

        A binding that allows no column shares no name with anything.

        Args:
            bound: The match.
            member: The full name of the other declaration.

        Returns:
            True if the two cannot share a name.
        """
        if self._permits_nothing(bound):
            return True
        other = self._by_declaration.get(member)
        if other is None:
            return False
        return self._permits_nothing(other) or self._pair_disjoint(bound, other)

    def _permits_nothing(self, bound: _Bound) -> bool:
        """Tell whether one binding holds no column for any caller.

        Args:
            bound: The match.

        Returns:
            True for an empty set of leftover columns, or a remainder whose
            record allows no column.
        """
        if isinstance(bound.binding, _ConcreteBinding):
            return not bound.binding.names()
        record = self._knowledge.record(self._rest_of(bound))
        return record is not None and record.permits_nothing()

    def _pair_disjoint(self, bound: _Bound, other: _Bound) -> bool:
        """Tell whether two bindings surely share no name.

        Args:
            bound: The match whose declaration states the promise.
            other: The match of the other declaration.

        Returns:
            True if the two cannot share a name.
        """
        if isinstance(bound.binding, _ConcreteBinding):
            return self._concrete_disjoint(bound.binding, other)
        if isinstance(other.binding, _ConcreteBinding):
            return self._concrete_disjoint(other.binding, bound)
        return not self._relations.can_share(self._rest_of(bound), self._rest_of(other))

    def _rest_of(self, bound: _Bound) -> MypyType:
        """Read the remainder that one match binds.

        Each caller of this method has already ruled out a binding to leftover
        columns, so the cast states a fact that the code cannot show to mypy.

        Args:
            bound: The match.

        Returns:
            The bound remainder.
        """
        return checked_cast(bound.binding, _SlotBinding).rest

    def _concrete_disjoint(self, binding: _ConcreteBinding, other: _Bound) -> bool:
        """Tell whether a set of leftover columns shares no name with another binding.

        Args:
            binding: The leftover columns.
            other: The match of the other declaration.

        Returns:
            True if no leftover name can appear in the other binding.
        """
        if isinstance(other.binding, _ConcreteBinding):
            return not (binding.names() & other.binding.names())
        return not self._relations.unprovable(self._rest_of(other), binding.names())

    def _check_within(self, bound: _Bound, group: frozenset[str]) -> None:
        """Prove one `within=` promise, and report a failure.

        The method decides it only when this binding and every other one are
        leftover columns. A record gives the names that a remainder can hold at
        most, and an upper bound cannot prove that one set is inside another.
        So a binding to a remainder gives a report through the gate. A
        leftover name outside the group gives a `ForbiddenColumnError`.

        Args:
            bound: The match.
            group: The full names of the declarations that must hold every name.
        """
        if self._permits_nothing(bound):
            return
        names = self._concrete_names(bound)
        cover = self._group_names(group)
        if names is None or cover is None:
            self._gate.report(self._ctx, UnprovenRequirementError(self._short(bound)))
            return
        if not names <= cover:
            outside = min(names - cover)
            ForbiddenColumnError(self._short(bound), outside).fail(self._ctx)

    def _concrete_names(self, bound: _Bound) -> frozenset[str] | None:
        """Read the names of one binding, if it is a set of leftover columns.

        Args:
            bound: The match.

        Returns:
            The names, or None for a binding to a remainder.
        """
        if isinstance(bound.binding, _ConcreteBinding):
            return bound.binding.names()
        return None

    @propagates(None)
    def _group_names(self, group: frozenset[str]) -> frozenset[str] | None:
        """Join the names of every declaration in a `within=` group.

        Args:
            group: The full names of the declarations in the group.

        Returns:
            All their names, or None if a declaration of the group is not bound
            at this call, or is bound to a remainder.
        """
        names: frozenset[str] = frozenset()
        for member in group:
            other = not_none(self._by_declaration.get(member))
            names |= not_none(self._concrete_names(other))
        return names

    def _short(self, bound: _Bound) -> str:
        """Give the short name of one bound declaration, for a message.

        Args:
            bound: The match.

        Returns:
            The part of the name after the last dot.
        """
        return bound.declaration.rsplit(".", 1)[-1]
