"""The hooks that record each call on an expression as code, in a deferred type.

`RecordedCall` records a method, an operator or a member of a namespace, called
on a typed expression. `RecordedFunction` records a function of `pl`, a
selector of `cs`, and `pl.col`. `RecordedAccessor` wraps the type of an
expression when the code reads a namespace, such as `.str`, so that the next
member reads its code. `RecordedLiteralComparison` records a comparison whose
literal stands on the left, as in `1 == pl.col("a")`. `RecordedShortForm`
records the expression that a method of a frame runs for the user, as
`pl.all().sum()` for `group_by(...).sum()`.

Each hook writes the call as the user wrote it, from the codes of its parts
(`ArgumentCode`), and gives the deferred type whose head stands for the class
that polars gives the call (`DeferredHeads`). A call that gives no expression,
such as `.meta.output_name()`, keeps the type of its stub. Where a part is a
union of types, as after an `if` and an `else`, the hook records the call for
each member, and gives the union of the deferred types.
"""

from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import CallExpr, ComparisonExpr, IndexExpr, OpExpr, TempNode
from mypy.plugin import MethodContext
from mypy.types import Instance, UnionType, get_proper_type

from astrolabe._hooks.calls.dtype_constructor_readers import CheckerLiteralTypeBuilder
from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.deferred.code import Code, CodeTree
from astrolabe._hooks.deferred.code_reader import (
    ArgumentMismatch,
    BuildRefusal,
    CodeReader,
    CodeReadError,
    HoleValues,
)
from astrolabe._hooks.encoding.marker_minter import MarkerMinter
from astrolabe._hooks.deferred.argument import DeferredArgument
from astrolabe._hooks.deferred.every_frame import EveryFrameRefusal
from astrolabe._hooks.errors import AstrolabeError, InvalidCallShapeError
from astrolabe._hooks.recording.context_fit import DeferredContextFit
from astrolabe._hooks.recording.argument_check import ArgumentCheck
from astrolabe._hooks.recording.argument_code import ArgumentCode
from astrolabe._hooks.recording.deferred_types import (
    DeferredHeads,
    DeferredMinter,
    DeferredReader,
)
from astrolabe.expr import TypedExpr

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from mypy.errors import ErrorInfo, ErrorWatcher
    from mypy.nodes import Expression
    from mypy.plugin import AttributeContext, FunctionContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup


class _Variants:
    """Give each choice of one member for each part of a call whose type is a union.

    A part is the receiver of a call or one of its arguments. A part whose
    type is a union of expression types takes each member in turn. The
    choices multiply, so above `LIMIT` choices the parts keep their union,
    and the code writes each such part as a hole.

    Example:
        >>> fixture = new_type_fixture()
        >>> list(_Variants().choices([], fixture.anyt))
        [(Any, {})]
    """

    LIMIT: ClassVar[int] = 64

    def choices(
        self,
        nodes: Sequence[tuple[Expression, MypyType]],
        receiver: MypyType,
    ) -> Iterator[tuple[MypyType, dict[int, MypyType]]]:
        """Give each choice of a member of the receiver and of each argument.

        Args:
            nodes: Each argument node with its type.
            receiver: The type of the receiver.

        Yields:
            The member of the receiver, and the member of each argument, by
            the identity of its node.

        Example:
            >>> fixture = new_type_fixture()
            >>> union = UnionType([fixture.a, fixture.b])
            >>> len(list(_Variants().choices([], union)))
            2
        """
        reader = DeferredReader()
        receivers = reader.members(receiver)
        parts = [(node, self._members(typ)) for node, typ in nodes]
        count = len(receivers)
        for _, members in parts:
            count *= len(members)
        if count > self.LIMIT:
            receivers = receivers[:1]
            parts = [(node, [typ]) for node, typ in nodes]
        for member in receivers:
            for picked in itertools.product(*(members for _, members in parts)):
                yield member, {id(node): typ for (node, _), typ in zip(parts, picked, strict=True)}

    def _members(self, typ: MypyType) -> list[MypyType]:
        """Give the members of the type of an argument that a choice takes in turn.

        Only a union whose members are typed expressions splits. Any other
        type stays whole.

        Args:
            typ: The type.

        Returns:
            The members.

        Example:
            >>> fixture = new_type_fixture()
            >>> union = UnionType([fixture.a, fixture.b])
            >>> _Variants()._members(union) == [union]
            True
        """
        reader = DeferredReader()
        members = reader.members(typ)
        typed = [member for member in members if reader.code(member) or reader.declared(member)]
        if len(members) < 2 or len(typed) != len(members):
            return [typ]
        return members


class _Recording:
    """Hold what one recorded call needs: the minter of deferred types and the choices of its parts.

    Example:
        >>> isinstance(_Recording.__name__, str)
        True
    """

    def __init__(self, plugin: PluginLookup, api: object) -> None:
        """Build the minter of the deferred types for the phase of the check.

        Args:
            plugin: The plugin, which finds a class by its full name.
            api: The checker.
        """
        self._api = api
        self._minter = DeferredMinter(MarkerMinter(plugin, CheckerLiteralTypeBuilder(), api))

    def recorded(
        self,
        head: type,
        nodes: Sequence[tuple[Expression, MypyType]],
        receiver: MypyType,
        build: Callable[[ArgumentCode, MypyType], ast.expr],
    ) -> MypyType | None:
        """Give the deferred type of a call, or the union over the members of its parts.

        Args:
            head: The head of the deferred type.
            nodes: Each argument node with its type.
            receiver: The type of the receiver, or of the call for a function.
            build: The tree of the call, from the writer of its arguments and
                the member of its receiver.

        Returns:
            The type, or `None` where mypy did not load the module of the markers.
        """
        minted: list[MypyType] = []
        for member, types in _Variants().choices(nodes, receiver):
            tree: CodeTree[MypyType] = CodeTree()
            known = {id(node): typ for node, typ in nodes}
            writer = ArgumentCode(self._api, tree, {**known, **types})
            typ = self._minter.minted(head, tree.code(build(writer, member)))
            if typ is None:  # pragma: no cover
                # The plugin adds the module of the markers to every build.
                return None
            minted.append(typ)
        distinct = [typ for index, typ in enumerate(minted) if typ not in minted[:index]]
        return UnionType.make_union(distinct)

    def wrapped(self, namespace: str, typ: MypyType) -> MypyType | None:
        """Give the wrapper of a namespace over the type of an expression.

        Args:
            namespace: The namespace.
            typ: The type of the expression.

        Returns:
            The wrapper.
        """
        return self._minter.wrapped(namespace, typ)


class _BuildCheck:
    """Report a recorded call that polars refuses while it builds the expression, or at every frame.

    Polars refuses some calls before any frame meets them, as `pl.lit` of an
    integer that no data type holds. The code of such a call cannot become
    an expression, so no frame can type it. The check builds the code once,
    and reports the error of polars where the call raises one. A hole is a
    value that the checker cannot know, and its placeholder can make polars
    refuse, so a code with a hole of a value is not checked.

    Polars builds other expressions and refuses them at every frame, as
    `.name.keep()` over an expression that reads no column. The check
    reports them at the call too, because no frame can run them.

    Example:
        >>> _BuildCheck().failure(Code("pl.lit(1)")) is None
        True
    """

    def refused(self, ctx: FunctionContext | MethodContext, typ: MypyType) -> bool:
        """Find the first member of a recorded type that no frame can run, and report a refusal.

        A member gives no expression where polars refuses to build it, or
        where the arguments of a call do not fit its function, which mypy
        reports. Where mypy already reported an error of the arguments of
        the call, the refusal of polars gives no second error.

        Args:
            ctx: The context of the call.
            typ: The recorded type.

        Returns:
            Whether a member gives no expression, or one that polars refuses
            at every frame.
        """
        reader = DeferredReader()
        for member in reader.members(typ):
            code = reader.code(member)
            if code is not None and self._member_refused(ctx, code):
                return True
        return False

    def _member_refused(self, ctx: FunctionContext | MethodContext, code: Code[MypyType]) -> bool:
        """Say whether no frame can run the code of one member, and report the refusal of polars.

        Args:
            ctx: The context of the call.
            code: The code of the member.

        Returns:
            Whether no frame can run it.
        """
        failure = self.failure(code)
        if isinstance(failure, ArgumentMismatch):
            return True
        error = self._error(code, failure)
        if error is None:
            return False
        if not _ArgumentErrors().reported(ctx):
            error.fail(ctx)
        return True

    def _error(self, code: Code[MypyType], failure: BuildRefusal | None) -> AstrolabeError | None:
        """Give the error of a code that polars refuses while it builds it, or at every frame.

        Args:
            code: The code.
            failure: The refusal of polars while it builds the code, or `None`.

        Returns:
            The error, or `None` where a frame can run the code.

        Example:
            >>> code = Code("pl.lit(1).name.keep()")
            >>> _BuildCheck()._error(code, None)._message
            'This call has no result at any frame: `name.keep_name` expected at least one ...'
            >>> _BuildCheck()._error(Code("pl.col('a').abs()"), None) is None
            True
        """
        if failure is not None:
            return InvalidCallShapeError.refused_by_polars(failure.error, str(failure))
        refusal = EveryFrameRefusal().refusal(DeferredArgument("row").declared(code))
        if refusal is None:
            return None
        return InvalidCallShapeError.refused_at_every_frame(refusal)

    def failure(self, code: Code[MypyType]) -> BuildRefusal | ArgumentMismatch | None:
        """Give why the code of a call gives no expression, where it gives none.

        Args:
            code: The code.

        Returns:
            The refusal of polars, or the mismatch of the arguments of a call.
            `None` where polars builds the expression, where the reader cannot
            read the code, or where a refusal can come from the placeholder of
            a hole, of a value or of a function.

        Example:
            >>> _BuildCheck().failure(Code("pl.lit(1, dtype=pl.Int8).cast(pl.Struct({}))")) is None
            True
            >>> type(_BuildCheck().failure(Code("pl.col('a').cast()"))).__name__
            'ArgumentMismatch'
            >>> _BuildCheck().failure(Code("cs.by_dtype([hole('function')])")) is None
            True
        """
        try:
            CodeReader().read(code)
        except ArgumentMismatch as mismatch:
            return mismatch
        except BuildRefusal as refusal:
            return None if HoleValues().any_in_text(code.text) else refusal
        except CodeReadError:
            return None
        return None


class _ArgumentErrors:
    """Say whether mypy already reported an error of the arguments of a call.

    mypy checks the arguments of a call before it calls the hook, so an
    argument of the wrong type, or a missing argument, is already an error
    on the line of the call. Polars refuses such an argument too, while it
    builds the expression, and one call reports one error. mypy can hold the
    error in a watcher of the errors while it checks an enclosing
    expression, and record it later, so the watchers count too.

    Example:
        >>> sorted(_ArgumentErrors.CODES)
        ['arg-type', 'call-arg', 'call-overload']
    """

    CODES: ClassVar[frozenset[str]] = frozenset({"arg-type", "call-arg", "call-overload"})

    def reported(self, ctx: FunctionContext | MethodContext) -> bool:
        """Say whether mypy reported an error of the arguments on the line of the call.

        Args:
            ctx: The context of the call.

        Returns:
            Whether it did.
        """
        errors = ctx.api.msg.errors
        infos = list(errors.error_info_map.get(errors.file, []))
        for watcher in errors.get_watchers():
            infos.extend(self._held(watcher))
        line = ctx.context.line
        codes = {info.code.code for info in infos if info.line == line and info.code is not None}
        return bool(codes & self.CODES)

    def _held(self, watcher: ErrorWatcher) -> list[ErrorInfo]:
        """Give the errors that a watcher holds, where it keeps them.

        Args:
            watcher: The watcher.

        Returns:
            The errors, or none where the watcher keeps no errors.
        """
        try:
            held = watcher.filtered_errors()
        except AssertionError:
            return []
        return list(held or [])


class _Arguments:
    """Pair each argument node of a call with the type that mypy gives it.

    Example:
        >>> _Arguments().of([], [])
        []
    """

    def of(
        self,
        args: Sequence[Sequence[Expression]],
        types: Sequence[Sequence[MypyType]],
    ) -> list[tuple[Expression, MypyType]]:
        """Give each argument node with its type.

        Args:
            args: The argument nodes of each parameter.
            types: The types of the argument nodes of each parameter.

        Returns:
            The pairs.

        Example:
            >>> fixture = new_type_fixture()
            >>> from mypy.nodes import IntExpr
            >>> node = IntExpr(1)
            >>> _Arguments().of([[node]], [[fixture.a]]) == [(node, fixture.a)]
            True
        """
        pairs: list[tuple[Expression, MypyType]] = []
        for nodes, typs in zip(args, types, strict=True):
            pairs.extend(zip(nodes, typs, strict=True))
        return pairs


class RecordedCall:
    """Record a method, an operator or a member of a namespace called on a typed expression.

    The receiver is the expression before the dot, or the left side of an
    operator, or the right side of a reflected operator such as `1 + e`. A
    member of a namespace reads the expression that the namespace wraps.

    Example:
        >>> RecordedCall(None, "abs")._name
        'abs'
    """

    OPERATORS: ClassVar[dict[str, type[ast.operator]]] = {
        "__add__": ast.Add,
        "__sub__": ast.Sub,
        "__mul__": ast.Mult,
        "__truediv__": ast.Div,
        "__floordiv__": ast.FloorDiv,
        "__mod__": ast.Mod,
        "__pow__": ast.Pow,
        "__and__": ast.BitAnd,
        "__or__": ast.BitOr,
        "__xor__": ast.BitXor,
    }
    COMPARISONS: ClassVar[dict[str, type[ast.cmpop]]] = {
        "__eq__": ast.Eq,
        "__ne__": ast.NotEq,
        "__lt__": ast.Lt,
        "__le__": ast.LtE,
        "__gt__": ast.Gt,
        "__ge__": ast.GtE,
    }
    UNARY: ClassVar[dict[str, type[ast.unaryop]]] = {
        "__neg__": ast.USub,
        "__pos__": ast.UAdd,
        "__invert__": ast.Invert,
    }

    def __init__(self, plugin: PluginLookup, name: str) -> None:
        """Keep the plugin and the name of the method.

        Args:
            plugin: The plugin, which finds a class by its full name.
            name: The name of the method, such as `abs` or `__add__`.
        """
        self._plugin = plugin
        self._name = name

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the deferred type of the call.

        Args:
            ctx: The context of the method call.

        Returns:
            The deferred type, or the type of the stub where the call gives no
            expression or cannot be written.
        """
        head = DeferredHeads().head_for(ctx.default_return_type)
        if head is None or DeferredReader().code(ctx.default_return_type) is not None:
            return ctx.default_return_type
        if ArgumentCheck().refused(ctx):
            return ctx.default_return_type
        proper = get_proper_type(ctx.type)
        owner = proper.type.fullname if isinstance(proper, Instance) else ""
        namespace = DeferredHeads().namespace_of(owner)
        receiver = DeferredReader().unwrapped(ctx.type)
        nodes = _Arguments().of(ctx.args, ctx.arg_types)

        def build(writer: ArgumentCode, member: MypyType) -> ast.expr:
            part = writer.expression(member) or writer.hole("expr")
            if namespace is not None:
                part = ast.Attribute(part, namespace, ast.Load())
            return self._node(ctx, writer, part)

        recorded = _Recording(self._plugin, ctx.api).recorded(head, nodes, receiver, build)
        if recorded is None or self._refused(ctx, recorded):
            return ctx.default_return_type
        return DeferredContextFit(ctx).fitted(recorded)

    def _refused(self, ctx: MethodContext, recorded: MypyType) -> bool:
        """Say whether polars refuses to build the recorded call, after the error is reported.

        The reflected side of an operator reports nothing, because mypy hides
        its errors.

        Args:
            ctx: The context that mypy gives for the call.
            recorded: The deferred type of the call.

        Returns:
            Whether polars refuses it.
        """
        return not self._reflected(ctx) and _BuildCheck().refused(ctx, recorded)

    def _reflected(self, ctx: MethodContext) -> bool:
        """Say whether mypy checks the call as the reflected side of an operator, as `1 + e`.

        mypy hides an error of that side, and then reports that the operator
        does not apply. So the check of the build does not report there. A
        frame that meets the code reports the refusal.

        Args:
            ctx: The context of the method call.

        Returns:
            Whether it does.
        """
        return self._name.startswith("__r") and isinstance(ctx.context, OpExpr)

    def _node(self, ctx: MethodContext, writer: ArgumentCode, receiver: ast.expr) -> ast.expr:
        """Give the tree of the call: a method call, or the operator that the method stands for.

        mypy checks an operator, an augmented assignment and a comparison
        with the method of its operator, so the name of the method decides
        the form. A call of that method that the user wrote, as
        `e.__radd__(1)`, runs the same method, so it takes the same form,
        where it has the arguments of the operator. mypy checks an item of a
        namespace, as `pl.col("s").struct[0]`, with `__getitem__`, so that
        method gives the subscript. Any other method is a call that the user
        wrote.

        Args:
            ctx: The context of the method call.
            writer: The writer of the arguments.
            receiver: The part of the receiver.

        Returns:
            The tree.
        """
        if self._name == "__getitem__" and isinstance(ctx.context, IndexExpr):
            return ast.Subscript(receiver, writer.part(ctx.args[0][0]), ast.Load())
        operator = self._operator(ctx, writer, receiver)
        if operator is not None:
            return operator
        function = ast.Attribute(receiver, self._name, ast.Load())
        return writer.call(function, checked_cast(ctx.context, CallExpr))

    def _operator(
        self,
        ctx: MethodContext,
        writer: ArgumentCode,
        receiver: ast.expr,
    ) -> ast.expr | None:
        """Give the operator that the method stands for, where the call has the arguments of it.

        Args:
            ctx: The context of the method call.
            writer: The writer of the arguments.
            receiver: The part of the receiver.

        Returns:
            The tree of the operator, or `None` where the method is not an
            operator, or where the call has other arguments.
        """
        name = self._name
        operator = not isinstance(ctx.context, CallExpr)
        count = sum(len(nodes) for nodes in ctx.args)
        if name in self._binary_names() and (operator or count == 1):
            return self._binary(receiver, writer.part(self._other(ctx)))
        if name in self.COMPARISONS and (operator or count == 1):
            comparison = self.COMPARISONS[name]()
            return ast.Compare(receiver, [comparison], [writer.part(ctx.args[0][0])])
        if name in self.UNARY and (operator or count == 0):
            return ast.UnaryOp(self.UNARY[name](), receiver)
        return None

    def _other(self, ctx: MethodContext) -> Expression:
        """Give the node of the other side of a binary operator, as the user wrote it.

        mypy checks a reflected operator, as `Decimal("1.5") * pl.col("u")`,
        with a temporary node that holds only the type of the left side. The
        node of the operator still holds that side, with its value.

        Args:
            ctx: The context of the method call.

        Returns:
            The node.
        """
        node = ctx.args[0][0]
        context = ctx.context
        if not isinstance(node, TempNode) or not isinstance(context, OpExpr):
            return node
        return context.left if self._name.startswith("__r") else context.right

    def _binary_names(self) -> set[str]:
        """Give the names of the binary operators and of their reflected forms.

        Returns:
            The names.

        Example:
            >>> "__rsub__" in RecordedCall(None, "abs")._binary_names()
            True
        """
        reflected = {f"__r{name[2:]}" for name in self.OPERATORS}
        return set(self.OPERATORS) | reflected

    def _binary(self, receiver: ast.expr, argument: ast.expr) -> ast.expr:
        """Give the tree of a binary operator, with the receiver on the right for a reflected one.

        Args:
            receiver: The part of the receiver.
            argument: The part of the other side.

        Returns:
            The tree.

        Example:
            >>> receiver, other = ast.Name("e", ast.Load()), ast.Constant(1)
            >>> ast.unparse(RecordedCall(None, "__radd__")._binary(receiver, other))
            '1 + e'
        """
        if self._name in self.OPERATORS:
            return ast.BinOp(receiver, self.OPERATORS[self._name](), argument)
        operator = self.OPERATORS[f"__{self._name[3:]}"]
        return ast.BinOp(argument, operator(), receiver)


class RecordedFunction:
    """Record a call of a function of `pl`, of a selector of `cs`, or of `pl.col`.

    The function takes its public name under its module, such as `pl.lit` or
    `cs.numeric`. A function that gives no expression keeps its own hook, if
    the checker has one, as `pl.concat` does.

    Example:
        >>> RecordedFunction(None, "pl.lit", None)._function
        'pl.lit'
    """

    def __init__(
        self,
        plugin: PluginLookup,
        function: str,
        fallback: Callable[[FunctionContext], MypyType] | None,
    ) -> None:
        """Keep the plugin, the code of the function, and the hook of the function for a frame.

        Args:
            plugin: The plugin, which finds a class by its full name.
            function: The code of the function, such as `pl.lit`, or `abs`.
            fallback: The hook of the function where it gives no expression.
        """
        self._plugin = plugin
        self._function = function
        self._fallback = fallback

    def __call__(self, ctx: FunctionContext | MethodContext) -> MypyType:
        """Give the deferred type of the call.

        Args:
            ctx: The context of the call. `pl.col` is an object, so mypy
                checks its call as a method.

        Returns:
            The deferred type, the type of the fallback hook, or the type of
            the stub.
        """
        head = DeferredHeads().head_for(ctx.default_return_type)
        if head is None or not isinstance(ctx.context, CallExpr):
            return self._unrecorded(ctx)
        call = ctx.context
        nodes = _Arguments().of(ctx.args, ctx.arg_types)
        if self._function == "abs" and not self._has_expression(nodes):
            return ctx.default_return_type
        if ArgumentCheck().refused(ctx):
            return ctx.default_return_type
        function = ast.parse(self._function, mode="eval").body

        def build(writer: ArgumentCode, member: MypyType) -> ast.expr:
            del member
            return writer.call(function, call)

        recorded = _Recording(self._plugin, ctx.api).recorded(
            head, nodes, ctx.default_return_type, build
        )
        if recorded is None or _BuildCheck().refused(ctx, recorded):
            return ctx.default_return_type
        return DeferredContextFit(ctx).fitted(recorded)

    def _unrecorded(self, ctx: FunctionContext | MethodContext) -> MypyType:
        """Give the type of a call that gives no expression: from its hook, or from its stub.

        Args:
            ctx: The context of the call.

        Returns:
            The type.
        """
        if self._fallback is not None and not isinstance(ctx, MethodContext):
            return self._fallback(ctx)
        return ctx.default_return_type

    def _has_expression(self, nodes: Sequence[tuple[Expression, MypyType]]) -> bool:
        """Say whether a typed expression is among the arguments.

        Args:
            nodes: Each argument node with its type.

        Returns:
            Whether one is.

        Example:
            >>> RecordedFunction(None, "abs", None)._has_expression([])
            False
        """
        reader = DeferredReader()
        return any(reader.code(typ) or reader.declared(typ) for _, typ in nodes)


@dataclass(frozen=True)
class ShortForm:
    """The expression that a method of a frame runs, and the parameters that it passes on.

    Polars runs some methods as another operation over one expression.
    `gb.quantile(0.5)` runs `gb.agg(pl.all().quantile(0.5))`, so its short
    form is the call `pl.all().quantile()`, which takes the parameter
    `quantile` of the method. The outermost call of the expression takes each
    of these parameters that the user passes, by keyword. A parameter that the
    user does not pass keeps the default of polars.

    Attributes:
        expression: The code of the expression. It ends with a call.
        parameters: The parameters of the method that the outermost call takes,
            under the same names.

    Example:
        >>> ShortForm("pl.all().sum()").parameters
        ()
    """

    expression: str
    parameters: tuple[str, ...] = ()


class RecordedShortForm:
    """Record the expression that a method runs for the user, with the arguments of the call.

    The hook of such a method types it as the other operation, over the
    deferred type that this class gives. A literal argument keeps its value in
    the code, and an argument that the checker cannot know is a hole, as in
    any recorded call.

    Example:
        >>> RecordedShortForm(None, ShortForm("pl.all()"))._form.expression
        'pl.all()'
    """

    def __init__(self, plugin: PluginLookup, form: ShortForm) -> None:
        """Keep the plugin and the short form.

        Args:
            plugin: The plugin, which finds a class by its full name.
            form: The expression that the method runs.
        """
        self._plugin = plugin
        self._form = form

    def recorded(self, ctx: MethodContext) -> MypyType | None:
        """Give the deferred type of the expression that one call runs.

        Args:
            ctx: The context of the call of the method.

        Returns:
            The deferred type. `None` where polars refuses to build the
            expression, after the error, or where mypy did not load the
            module of the markers.
        """
        passed = self._passed(ctx.callee_arg_names, ctx.args, ctx.arg_types)
        nodes = [(node, typ) for _, node, typ in passed]

        def build(writer: ArgumentCode, member: MypyType) -> ast.expr:
            del member
            call = checked_cast(ast.parse(self._form.expression, mode="eval").body, ast.Call)
            for name, node, _ in passed:
                call.keywords.append(ast.keyword(name, writer.part(node)))
            return call

        recording = _Recording(self._plugin, ctx.api)
        recorded = recording.recorded(TypedExpr, nodes, ctx.default_return_type, build)
        if recorded is None or _BuildCheck().refused(ctx, recorded):
            return None
        return recorded

    def _passed(
        self,
        names: Sequence[str | None],
        args: Sequence[Sequence[Expression]],
        types: Sequence[Sequence[MypyType]],
    ) -> list[tuple[str, Expression, MypyType]]:
        """Give each parameter of the short form that a call passes, with its node and its type.

        Args:
            names: The name of each parameter of the method.
            args: The argument nodes of each parameter.
            types: The types of the argument nodes of each parameter.

        Returns:
            The name, the node and the type of each passed parameter, in the
            order of the method.

        Example:
            >>> fixture = new_type_fixture()
            >>> from mypy.nodes import IntExpr
            >>> form = RecordedShortForm(None, ShortForm("pl.len().alias()", ("name",)))
            >>> [name for name, _, _ in form._passed(["name"], [[IntExpr(1)]], [[fixture.a]])]
            ['name']
            >>> form._passed(["name"], [[]], [[]])
            []
        """
        passed: list[tuple[str, Expression, MypyType]] = []
        for name, nodes, typs in zip(names, args, types, strict=True):
            if name in self._form.parameters and nodes:
                passed.append((name, nodes[0], typs[0]))
        return passed


class RecordedAccessor:
    """Wrap the type of a typed expression when the code reads one of its namespaces.

    Example:
        >>> RecordedAccessor(None, "str")._namespace
        'str'
    """

    def __init__(self, plugin: PluginLookup, namespace: str) -> None:
        """Keep the plugin and the namespace.

        Args:
            plugin: The plugin, which finds a class by its full name.
            namespace: The namespace, such as `str`.
        """
        self._plugin = plugin
        self._namespace = namespace

    def __call__(self, ctx: AttributeContext) -> MypyType:
        """Give the wrapper over the type of the expression.

        Args:
            ctx: The context of the attribute.

        Returns:
            The wrapper, or the type of the stub for an expression that is not typed.
        """
        reader = DeferredReader()
        members = reader.members(ctx.type)
        if not all(reader.code(member) or reader.declared(member) for member in members):
            return ctx.default_attr_type
        wrapped = _Recording(self._plugin, ctx.api).wrapped(self._namespace, ctx.type)
        return ctx.default_attr_type if wrapped is None else wrapped


class RecordedLiteralComparison:
    """Record a comparison whose literal stands on the left, as in `1 == pl.col("a")`.

    mypy checks `1 == e` with `int.__eq__`, which takes any object and gives a
    `bool`. Polars gives an expression, because `int.__eq__` refuses an
    expression and Python then calls `Expr.__eq__`.

    Example:
        >>> RecordedLiteralComparison(None, "__eq__")._name
        '__eq__'
    """

    def __init__(self, plugin: PluginLookup, name: str) -> None:
        """Keep the plugin and the name of the comparison.

        Args:
            plugin: The plugin, which finds a class by its full name.
            name: `__eq__` or `__ne__`.
        """
        self._plugin = plugin
        self._name = name

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the deferred type of the comparison where its right side is a typed expression.

        Args:
            ctx: The context of the comparison.

        Returns:
            The deferred type, or the type of the stub.
        """
        nodes = _Arguments().of(ctx.args, ctx.arg_types)
        if len(nodes) != 1 or not isinstance(ctx.context, ComparisonExpr):
            return ctx.default_return_type
        reader = DeferredReader()
        right = nodes[0][1]
        if not all(
            reader.code(member) or reader.declared(member) for member in reader.members(right)
        ):
            return ctx.default_return_type
        literal = ctx.context.operands[0]
        comparison = RecordedCall.COMPARISONS[self._name]()

        def build(writer: ArgumentCode, member: MypyType) -> ast.expr:
            del member
            return ast.Compare(writer.part(literal), [comparison], [writer.part(nodes[0][0])])

        recorded = _Recording(self._plugin, ctx.api).recorded(TypedExpr, nodes, ctx.type, build)
        return ctx.default_return_type if recorded is None else recorded
