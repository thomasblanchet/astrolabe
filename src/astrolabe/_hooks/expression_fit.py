"""The check at each call that an expression argument fits the type that its parameter declares.

A user can declare a parameter as an expression type, for example
`price: column("price", pl.Float64)`. The checker gives each expression
a type that holds the columns it reads, the data type it gives for each
combination of input data types (its table), its output name, and how it
treats rows (row by row, as an aggregation, or as a literal). The
argument fits when three things hold. It reads only the columns that
the declaration reads. Its output name and its data type at each
declared combination agree with the declaration. It treats rows in the
same way as the declaration.

mypy compares types by class and cannot make this check itself. So the
hook runs on the signature of each call. It checks each argument,
reports the first part that fails as one error, and then replaces the
parameter type with the argument's own type, so that mypy sees two equal
types and adds no error of its own.

A declaration can leave the name of the column open, with a blank read
(`some_column`). If the return type uses the same blank, the call binds
it to the name of the argument, and the return type gets that name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from astrolabe._hooks.calls.dtype_constructor_readers import CheckerLiteralTypeBuilder
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.deferred.argument import DeferredArgument
from astrolabe._hooks.deferred.fit import DeferredFit
from astrolabe._hooks.deferred.code import Code
from astrolabe._hooks.encoding.expr_codec import QuadrupleCodec
from astrolabe._hooks.encoding.expr_evaluation import ExpressionArgument
from astrolabe._hooks.encoding.marker_minter import MarkerMinter, ProperReads
from astrolabe._hooks.errors import ExpressionFitError, UntypedArgumentError
from astrolabe._hooks.encoding.selection_codec import SelectionReader
from astrolabe._hooks.expression.fit import SliceDecl
from astrolabe._hooks.expression.outcome import Axis, Outcome, Wrap
from astrolabe._hooks.expression.quadruple import ExpressionType
from astrolabe._hooks.expression.table import Table
from astrolabe._hooks.propagate import not_falsy, not_none, propagates

if TYPE_CHECKING:
    from astrolabe._hooks.expression.selection import SelectionType
    from collections.abc import Sequence

    from mypy.plugin import FunctionSigContext, Plugin

    from astrolabe._hooks.domain.unproven_gate import UnprovenGate
    from mypy.types import CallableType
    from mypy.types import Type as MypyType


class ExpressionFitHook:
    """The signature hook that checks the declared expression parameters of one function.

    The plugin builds one instance for each function that has such
    parameters, with the positions of those parameters. At each call, the
    hook gives mypy a new signature for that call. The work for one call
    happens in a fresh `_ExpressionCallFit`.
    """

    def __init__(self, plugin: Plugin, gate: UnprovenGate, indices: tuple[int, ...]) -> None:
        """Keep the plugin, the gate and the positions of the declared parameters.

        Args:
            plugin: The mypy plugin, which mints the bound return type.
            gate: The gate that decides, under the `allow_unproven` setting,
                whether to report an argument that is not typed.
            indices: The position of each declared expression parameter.
        """
        self._plugin = plugin
        self._gate = gate
        self._indices = indices

    def __call__(self, ctx: FunctionSigContext) -> CallableType:
        """Give the signature of one call, after the fit of each declared parameter.

        Args:
            ctx: The mypy context of the function signature at the call.

        Returns:
            The signature, with each declared parameter replaced by the type of
            its argument, and a blank in the return type bound to a name.
        """
        return _ExpressionCallFit(ctx, self._plugin, self._gate, self._indices).adjusted()


class _ExpressionCallFit:
    """The fit of every declared expression parameter for one call.

    It holds the values that stay the same for the whole call, and the
    name that a blank read binds to, which a parameter sets and the return
    type reads.
    """

    PLAIN: str = "this value is not a typed expression, so its columns and types are unknown"
    NOT_SELECTION: str = "this value is not a selection of columns"

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
            gate: The gate that decides whether to report an argument that is
                not typed.
            indices: The position of each declared expression parameter.
        """
        self._ctx = ctx
        self._sig = ctx.default_signature
        self._plugin = plugin
        self._gate = gate
        self._indices = indices
        self._argument = ExpressionArgument()
        self._binding: str | None = None

    def adjusted(self) -> CallableType:
        """Build the signature of the call after the fit.

        Returns:
            The signature, with each declared parameter replaced by the type of
            its argument. The return type has the blank bound where an
            argument bound it, and is the declared one otherwise.
        """
        arg_types = list(self._sig.arg_types)
        for index in self._indices:
            arg_types[index] = self._fitted(index)
        bound = self._bound_return()
        ret_type = self._sig.ret_type if bound is None else bound
        return self._sig.copy_modified(arg_types=arg_types, ret_type=ret_type)

    def _fitted(self, index: int) -> MypyType:
        """Check the argument of one parameter, and give the type that the parameter takes.

        If the argument type is a union, each member of the union gets the
        check.

        Args:
            index: The position of the parameter.

        Returns:
            The type of the argument after the check. The declared type if the
            call passes no argument there.
        """
        nodes = self._ctx.args[index]
        if not nodes:
            return self._sig.arg_types[index]
        actual = self._ctx.api.get_expression_type(nodes[0])
        for item in ProperReads().union_items(actual):
            self._judge(index, item, self._sig.arg_types[index])
        return actual

    def _judge(self, index: int, item: MypyType, declared: MypyType) -> None:
        """Check one member of the argument type, and report the fault if it has one.

        If the argument is not typed, the gate reports it, unless the
        `allow_unproven` setting lets it pass. If the argument is typed and
        does not fit, the error is always reported.

        Args:
            index: The position of the parameter, for the message.
            item: One member of the argument type.
            declared: The declared type of the parameter.
        """
        name = self._sig.arg_names[index] or str(index)
        codes = DeferredArgument("row").codes(item)
        if codes is None:
            untyped = self._untyped(item, declared)
            if untyped is not None:
                self._gate.report(self._ctx, UntypedArgumentError(name, untyped))
                return
            # A declared value is the code of its one leaf.
            codes = [Code("$0", (item,))]
        self._judge_deferred(name, codes, declared)

    def _judge_deferred(
        self,
        name: str,
        codes: Sequence[Code[MypyType]],
        declared: MypyType,
    ) -> None:
        """Check an argument whose type holds its code, the deferred model, against its declaration.

        A declared selection goes through the fit for a selection. Every
        other parameter that reaches this check declares an expression type.

        Args:
            name: The name of the parameter, for the message.
            codes: The code of each member of the argument type.
            declared: The declared type of the parameter.
        """
        selection = SelectionReader().decoded(declared)
        if selection is not None:
            self._judge_deferred_selection(name, codes, selection)
            return
        decl = checked_not_none(self._argument.decoded(declared))
        argument = DeferredArgument("row")
        sliced = SliceDecl.from_expression(decl)
        for code in codes:
            refusal = DeferredFit().verdict(argument.declared(code), sliced)
            if refusal is not None:
                ExpressionFitError.at_argument(name, refusal).fail(self._ctx)
                return
        reads = argument.reads_of(codes[0])
        if "" in decl.needs and len(reads) == 1:
            self._binding = reads[0]

    def _judge_deferred_selection(
        self,
        name: str,
        codes: Sequence[Code[MypyType]],
        selection: SelectionType,
    ) -> None:
        """Check an argument of the deferred model against a declared selection.

        A code that holds no selector is not a selection, which the gate
        reports as an argument that is not typed.

        Args:
            name: The name of the parameter, for the message.
            codes: The code of each member of the argument type.
            selection: The declared selection.
        """
        argument = DeferredArgument("row")
        for code in codes:
            refusal = argument.selection_verdict(code, selection)
            if refusal == self.NOT_SELECTION:
                self._gate.report(self._ctx, UntypedArgumentError(name, refusal))
                return
            if refusal is not None:
                ExpressionFitError.at_argument(name, refusal).fail(self._ctx)
                return

    def _untyped(self, item: MypyType, declared: MypyType) -> str | None:
        """Say what the argument is not, if it is not of the kind that the parameter declares.

        Args:
            item: The argument type.
            declared: The declared type of the parameter: an expression type or
                a selection.

        Returns:
            `NOT_SELECTION` for a value that is not a selection when the
            parameter declares one. `PLAIN` for a value that is not a typed
            expression when the parameter declares an expression type.
            Otherwise None.
        """
        if SelectionReader().decoded(declared) is not None:
            return self.NOT_SELECTION if SelectionReader().decoded(item) is None else None
        if self._argument.decoded(item) is None:
            return self.PLAIN
        return None

    @propagates(None)
    def _bound_return(self) -> MypyType | None:
        """Put the bound name in place of the blank in the declared return type.

        Returns:
            The new return type. None if the return type has no blank read, or
            no argument bound the blank.
        """
        binding = not_none(self._binding)
        declared = not_none(self._argument.decoded(self._sig.ret_type))
        not_falsy("" in declared.needs)
        minter = MarkerMinter(self._plugin, CheckerLiteralTypeBuilder(), self._ctx.api)
        return QuadrupleCodec(minter).encoded(BlankBinding(binding).bound(declared))


@dataclass(frozen=True)
class BlankBinding:
    """Put one column name in place of the blank read in an expression type.

    A blank read, written `some_column` in a declaration, is a column whose
    name each call chooses. Inside the engine the blank is the empty name.
    When a call binds it, this class writes the bound name everywhere the
    empty name appears: in the columns that the expression reads, in its
    output name, and in each cell of its tables.

    Attributes:
        name: The bound name.

    Example:
        >>> from astrolabe._hooks.expression.character import ELEMENTWISE
        >>> blank = Table.from_cells(("",), {("Float64",): Axis("")})
        >>> decl = ExpressionType(frozenset({""}), blank, "", ELEMENTWISE)
        >>> bound = BlankBinding("price").bound(decl)
        >>> bound.name, sorted(bound.needs), bound.table.cells
        ('price', ['price'], ((('Float64',), Axis(column='price')),))
    """

    name: str

    def bound(self, expression: ExpressionType) -> ExpressionType:
        """Put the bound name in place of the blank in each part of an expression type.

        The table for an aggregation context gets the bound name too,
        where the expression has one.

        Args:
            expression: The declared expression type.

        Returns:
            The same expression type, with the bound name in place of the blank.

        Example:
            >>> from astrolabe._hooks.expression.character import AGGREGATING
            >>> count = Table.constant((), (), "UInt32")
            >>> rootless = ExpressionType(frozenset(), count, "len", AGGREGATING)
            >>> BlankBinding("x").bound(rootless).name
            'len'
            >>> blank = Table.from_cells(("",), {("Int64",): Axis("")})
            >>> some = ExpressionType(frozenset({""}), blank, "", AGGREGATING)
            >>> BlankBinding("q").bound(some).table.columns
            ('q',)
        """
        return ExpressionType(
            frozenset(self._renamed(need) for need in expression.needs),
            self.table(expression.table),
            self._renamed(expression.name),
            expression.character,
        )

    def table(self, table: Table) -> Table:
        """Rename the blank column of a table, and each answer that names it.

        Args:
            table: The table.

        Returns:
            The table with the bound name in place of the blank.

        Example:
            >>> blank = Table.from_cells(("",), {("Int64",): Axis("")})
            >>> BlankBinding("q").table(blank).columns
            ('q',)
        """
        columns = tuple(self._renamed(column) for column in table.columns)
        cells = {cell: self.outcome(outcome) for cell, outcome in table.cells}
        return Table.from_cells(columns, cells)

    def outcome(self, outcome: Outcome) -> Outcome:
        """Rename the blank inside one answer of a table.

        An answer that wraps another renames the answer inside. A constant
        answer names no column, so it stays as it is.

        Args:
            outcome: The answer.

        Returns:
            The answer with the bound name in place of the blank.

        Example:
            >>> BlankBinding("q").outcome(Wrap("List", Axis("")))
            Wrap(constructor='List', inner=Axis(column='q'))
            >>> from astrolabe._hooks.expression.outcome import Const
            >>> BlankBinding("q").outcome(Const("Int64"))
            Const(dtype='Int64')
        """
        if isinstance(outcome, Axis):
            return replace(outcome, column=self._renamed(outcome.column))
        if isinstance(outcome, Wrap):
            return replace(outcome, inner=self.outcome(outcome.inner))
        return outcome

    def _renamed(self, column: str) -> str:
        """Give one column name, with the bound name in place of the blank.

        Args:
            column: The name. The empty name is the blank.

        Returns:
            The bound name for the blank. Otherwise the name as it is.

        Example:
            >>> BlankBinding("q")._renamed("")
            'q'
            >>> BlankBinding("q")._renamed("a")
            'a'
        """
        return column or self.name
