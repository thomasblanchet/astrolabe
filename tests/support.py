"""Build the mypy objects that a test or a doctest of the plugin needs.

The plugin reads mypy's own types: a `TypeInfo` for a class, an
`Instance` for a use of that class, and the nodes of the abstract
syntax tree (AST) for the code that the user wrote. A unit test of one
plugin class runs no mypy build, so it has none of these objects. This
module builds small, valid ones by hand, over mypy's own test helper
`TypeFixture`, and gives plain stand-ins for the parts of mypy that the
plugin asks questions of.

Read this module first when you write a new test or a new doctest. Most
builders make one shape that the plugin meets in a real build, for
example a polars data type, a column or a frame. Other builders make a
plugin class that needs mypy objects, for example a codec, which turns
a value of the plugin into a mypy type and back. If an example needs a
shape that no builder makes, add a builder here.

A doctest in `src/astrolabe` calls these functions with no import line,
because the root `conftest.py` puts every public name of this module in
the doctest namespace. A pytest test imports them, or takes them as the
fixtures in `tests/hooks/conftest.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import (
    ARG_NAMED,
    ARG_POS,
    GDEF,
    ArgKind,
    Block,
    CallExpr,
    Context,
    FloatExpr,
    FuncDef,
    IntExpr,
    ListExpr,
    MemberExpr,
    MypyFile,
    NameExpr,
    ReturnStmt,
    StrExpr,
    SymbolTableNode,
)
from mypy.test.typefixture import TypeFixture
from mypy.types import (
    CallableType,
    Instance,
    LiteralType,
    NoneType,
    TupleType,
    TypedDictType,
    TypeVarId,
    TypeVarType,
    UnionType,
)
from mypy.types import Type as MypyType

import astrolabe.remainder as remainder_markers
from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.calls.dtype_constructor_readers import CheckerLiteralTypeBuilder
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.dtype_families import DtypeFamilies
from astrolabe._hooks.domain.dtype_promotion import DTypePromotion
from astrolabe._hooks.domain.frame import Frame, KnownColumns, PluginLookup
from astrolabe._hooks.domain.join_key_compatibility import JoinKeyCompatibility
from astrolabe._hooks.domain.reducer_promotion import ReducerDtype
from astrolabe._hooks.domain.schema_equality import SchemaEquality
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility, UncheckedFallback
from astrolabe._hooks.domain.untyped_fallback import UntypedFallback
from astrolabe._hooks.encoding.concat_remainder import ConcatRemainderResolver
from astrolabe._hooks.encoding.declaration_codec import DeclarationCodec
from astrolabe._hooks.encoding.expr_codec import QuadrupleCodec
from astrolabe._hooks.encoding.join_remainder import JoinRemainderResolver
from astrolabe._hooks.encoding.marker_minter import MarkerMinter as _EncodingMinter
from astrolabe._hooks.encoding.marker_minter import NameSetCodec
from astrolabe._hooks.encoding.remainder_codec import RemainderCodec
from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.encoding.selection_codec import SelectionCodec
from astrolabe._hooks.encoding.selector_codec import SelectorCodec
from astrolabe._hooks.unproven import Level, UnprovenPolicy
from astrolabe._hooks.update import _UpdatePlan
from astrolabe.declare import infer
from astrolabe.expr import TypedColumn as _TypedColumn
from astrolabe.frame import Array as _ArrayMarker
from astrolabe.frame import Datetime as _DatetimeMarker
from astrolabe.frame import Decimal as _DecimalMarker
from astrolabe.frame import Enum as _EnumMarker
from astrolabe.frame import List as _ListMarker
from astrolabe.frame import Struct as _StructMarker
from astrolabe.frame import TypedDataFrame as _TypedDataFrame
from astrolabe.frame import TypedLazyFrame as _TypedLazyFrame
from astrolabe.frame import Unchecked as _UncheckedMarker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.errorcodes import ErrorCode
    from mypy.nodes import Expression, Statement, TypeInfo

    from astrolabe._hooks.expression.quadruple import ExpressionType


class _FakeCallContext:
    """Stand in for the mypy context of one call, so that `CallArgs` can read it.

    When mypy asks a hook about a call, it passes a context object. That
    object holds the name of each parameter, the AST nodes that the
    caller wrote for it, and the plugin API. `CallArgs` reads only those
    parts. A real context exists only inside a mypy build, so a test of
    argument reading builds this stand-in instead.

    The API is read only when a node is a name, for example a `Final`
    variable whose value mypy tracks in its type. A literal written in
    place never needs the API, so most tests leave it as None.
    """

    def __init__(
        self,
        names: Sequence[str],
        nodes: Sequence[list[Expression]],
        arg_names: Sequence[list[str | None]] | None = None,
        api: object = None,
    ) -> None:
        """Hold the parts of the call that `CallArgs` reads.

        Args:
            names: The name of each parameter, in the same order as `nodes`.
            nodes: The AST nodes that the caller wrote for each parameter.
                An empty list stands for a parameter that the caller left out.
            arg_names: The keyword that the caller wrote before each node.
                A `**kwargs` parameter needs it, because its keywords are not
                parameter names. The default is None for every node, as in a
                call that passes every argument by position.
            api: The mypy plugin API, which a name lookup reads. The default
                is None, which is safe when every node is a literal.
        """
        self.callee_arg_names = list(names)
        self.args = list(nodes)
        if arg_names is None:
            arg_names = [[None] * len(group) for group in nodes]
        self.arg_names = list(arg_names)
        self.api = api


class NamedDeclarations:
    """Stand in for the declaration gate: accept the classes in a fixed set of full names.

    The plugin asks mypy whether a class is a declared remainder, one that
    `other_columns(...)` made. A test has no such mypy state, so it names the
    declared classes itself.

    Example:
        >>> fixture = new_type_fixture()
        >>> minter = new_marker_minter(fixture)
        >>> one = minter.resolved("kept.R")
        >>> NamedDeclarations(frozenset({"kept.R"})).is_declaration(one)
        True
    """

    def __init__(self, fullnames: frozenset[str]) -> None:
        """Keep the full names to accept.

        Args:
            fullnames: The full names of the declared remainder classes.
        """
        self._fullnames = fullnames

    def is_declaration(self, instance: Instance) -> bool:
        """Tell whether the class of one instance is in the set.

        Args:
            instance: The plain instance to test.

        Returns:
            True if the full name of the class is in the set.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> one = minter.resolved("kept.R")
            >>> NamedDeclarations(frozenset()).is_declaration(one)
            False
        """
        return instance.type.fullname in self._fullnames


class FakeApi:
    """Stand in for the mypy checker API, which a hook reaches through its context.

    Plugin code asks this API two questions. It builds an instance of a
    builtin class, for example `str` or `tuple`, when it mints a type.
    It also asks for the type that mypy inferred for an AST node. A real
    API exists only inside a mypy build, so a test uses this stand-in.

    The builtin classes come from the `TypeFixture`. The inferred types
    are whatever the test puts in `expression_types`, keyed by the `id`
    of the node.

    Example:
        >>> fixture = new_type_fixture()
        >>> api = new_api(fixture)
        >>> api.named_generic_type("builtins.str", []).type.fullname
        'builtins.str'
    """

    def __init__(self, fixture: TypeFixture) -> None:
        """Start with the builtin classes of the fixture and no inferred types.

        Args:
            fixture: The source of the builtin classes. A class that the
                fixture does not hold is made on first use.
        """
        self._fixture = fixture
        self._known: dict[str, TypeInfo] = {
            "builtins.str": fixture.str_type_info,
            "builtins.object": fixture.oi,
            "builtins.tuple": fixture.std_tuplei,
        }
        self.expression_types: dict[int, MypyType] = {}

    def named_generic_type(self, fullname: str, args: list[MypyType]) -> Instance:
        """Build an instance of a builtin class, as the real API does.

        The same full name always gives the same `TypeInfo`, so two
        instances built here compare equal when their arguments do.

        Args:
            fullname: The full name of the class, for example `builtins.str`.
            args: The type arguments of the instance.

        Returns:
            The instance of that class.
        """
        info = self._known.setdefault(fullname, self._fixture.make_type_info(fullname))
        return Instance(info, args)

    def get_expression_type(self, node: object) -> MypyType:
        """Give the type that the test put in `expression_types` for this node.

        This is the answer that mypy gives in a real build, after it infers
        the type of the node. A node with no entry raises `KeyError`, which
        shows that the code under test read a node the test did not expect.

        Args:
            node: The AST node.

        Returns:
            The type that the test stored for the node.
        """
        return self.expression_types[id(node)]


class FakePlugin:
    """Stand in for the mypy `Plugin` object, which finds a class by its full name.

    Plugin code that mints a new type often needs a class it does not
    hold yet, for example `polars.Float64` for the result of a division.
    It asks the plugin for that class with `lookup_fully_qualified`. This
    stand-in answers that one question, so that such code can run in a
    test with no mypy build.

    Example:
        >>> fixture = new_type_fixture()
        >>> plugin = new_plugin(fixture)
        >>> plugin.lookup_fully_qualified("builtins.str").node.fullname
        'builtins.str'
    """

    def __init__(self, fixture: TypeFixture) -> None:
        """Start with no class known. Each lookup makes its class once.

        Args:
            fixture: The `TypeFixture` that makes each class.
        """
        self._fixture = fixture
        self._known: dict[str, TypeInfo] = {}

    def lookup_fully_qualified(self, fullname: str) -> SymbolTableNode:
        """Find a class by its full name, as the real plugin does.

        The first lookup of a name makes a `TypeInfo` for it. Each later
        lookup of that name gives the same `TypeInfo`, so the types built
        from it compare equal.

        Args:
            fullname: The full name of the class, for example
                `polars.datatypes.classes.Float64`.

        Returns:
            The symbol table entry that holds the class, in the form that
            mypy gives.
        """
        info = self._known.setdefault(fullname, self._fixture.make_type_info(fullname))
        return SymbolTableNode(GDEF, info)


class FakeSemanticApi:
    """Stand in for the API of mypy's semantic analyzer, with no module loaded.

    The semantic analyzer is the mypy phase that binds names before any
    type is checked. Some plugin code in that phase looks a name up in
    the symbol table of a module, for example to find the data type that
    a variable such as `Stamp = pl.Datetime("us")` holds. That code reads
    only the table of modules, so this stand-in holds that table and
    nothing else. The table is empty, so every lookup finds nothing.

    Example:
        >>> FakeSemanticApi().modules
        {}
    """

    def __init__(self) -> None:
        """Start with no module in the table."""
        self.modules: dict[str, MypyFile] = {}


class FakeDynamicClassContext:
    """Stand in for the context that mypy gives a dynamic-class hook.

    mypy calls a dynamic-class hook for an assignment such as
    `Price = column("price", pl.Int64)`, where a call on the right makes
    a new name. The hook reaches the semantic analyzer through the `api`
    of its context. This stand-in gives it a `FakeSemanticApi`, so that
    code which runs in such a hook can have a doctest.

    Example:
        >>> FakeDynamicClassContext().api.modules
        {}
    """

    def __init__(self) -> None:
        """Give the context an API with no module loaded."""
        self.api = FakeSemanticApi()


def new_dynamic_class_context() -> FakeDynamicClassContext:
    """Build the context of a dynamic-class hook, in which no name is defined.

    Use it for a doctest of code that looks a name up during semantic
    analysis. Every lookup finds nothing.

    Returns:
        A new context.

    Example:
        >>> new_dynamic_class_context().api.modules
        {}
    """
    return FakeDynamicClassContext()


class FakeFailApi:
    """Stand in for the part of the mypy API that reports an error, and keep each message.

    Example:
        >>> api = FakeFailApi()
        >>> api.fail("no result", Context())
        >>> api.messages
        ['no result']
    """

    def __init__(self) -> None:
        """Start with no message."""
        self.messages: list[str] = []

    def fail(self, msg: str, ctx: Context, /, *, code: ErrorCode | None = None) -> None:
        """Keep the message of one error.

        Args:
            msg: The message.
            ctx: The place of the error, which the stand-in does not read.
            code: The code of the error, which the stand-in does not read.
        """
        del ctx, code
        self.messages.append(msg)


class FakeFailContext:
    """Stand in for the context of a hook, at which code reports an error.

    A hook reports through the `api` of its context, at its `context`. This
    stand-in keeps each message, so that code which reports only in some
    releases of polars can have a doctest.

    Example:
        >>> FakeFailContext().api.messages
        []
    """

    def __init__(self) -> None:
        """Give the context an API that keeps the messages, and a place."""
        self.api = FakeFailApi()
        self.context = Context()


def new_fail_context() -> FakeFailContext:
    """Build the context of a hook that keeps each error it reports.

    Returns:
        A new context, with no message.

    Example:
        >>> new_fail_context().api.messages
        []
    """
    return FakeFailContext()


def new_type_fixture() -> TypeFixture:
    """Build a new `TypeFixture`, the starting point of almost every other builder.

    `TypeFixture` is mypy's own helper for its unit tests. It holds the
    `TypeInfo` of `object`, `str`, `tuple` and a few other builtins, and
    it makes a new `TypeInfo` for any full name on request. Most builders
    here take a fixture as their first argument.

    Returns:
        A new fixture.

    Example:
        >>> fixture = new_type_fixture()
        >>> fixture.str_type_info.fullname
        'builtins.str'
    """
    return TypeFixture()


def new_api(fixture: TypeFixture) -> FakeApi:
    """Build the stand-in for the mypy checker API.

    Use it where the code under test takes an `api`. See `FakeApi`.

    Args:
        fixture: The source of the builtin classes.

    Returns:
        The stand-in API.
    """
    return FakeApi(fixture)


def new_plugin(fixture: TypeFixture) -> FakePlugin:
    """Build the stand-in for the mypy `Plugin` object.

    Use it where the code under test takes a `plugin` to find a class by
    its full name. See `FakePlugin`.

    Args:
        fixture: The `TypeFixture` that makes each class.

    Returns:
        The stand-in plugin.
    """
    return FakePlugin(fixture)


def new_remainder_edit(fixture: TypeFixture, declared: set[str]) -> RemainderEdit:
    """Build the editor that applies one change to the remainder of a frame type.

    The remainder of a frame is the set of its columns that the type does
    not name. A hook for an operation such as `drop` or `rename` must
    change that part of the type as well as the named columns. The editor
    reads the remainder out of the type, applies the change, and writes
    the result back as a type. Use this builder for a test or a doctest
    of such a hook.

    Args:
        fixture: The `TypeFixture` that makes each class.
        declared: The full names of the classes that stand for a remainder
            the user declared. In a real build these are the classes that
            `other_columns(...)` registered. Any other class does not read
            as a remainder.

    Returns:
        The editor, over a fake plugin and a fake API.

    Example:
        >>> edit = new_remainder_edit(new_type_fixture(), {"kept.R"})
        >>> edit.minus(None, ["z"]) is None
        True
    """
    return RemainderEdit(
        RemainderCodec(new_marker_minter(fixture), NamedDeclarations(frozenset(declared))),
    )


def new_join_remainder_resolver(fixture: TypeFixture) -> JoinRemainderResolver:
    """Build the class that decides the remainder of a join result.

    A join of two open frames can give a result whose columns depend on
    columns that neither type names. The resolver decides that part of
    the result, or names the column that can collide.

    Three class names read as declared remainders here. `decl.Rest` is
    the default name of `declaration_info`, and `decl.S` is a second
    one, so an example can join two different declarations. `kept.R`
    carries no requirement, for an example about a remainder that
    promises nothing.

    Args:
        fixture: The `TypeFixture` that makes each class.

    Returns:
        The resolver, over a fake plugin and a fake API.

    Example:
        >>> fixture = new_type_fixture()
        >>> resolver = new_join_remainder_resolver(fixture)
        >>> resolver._names_unchanged(declared_rest(fixture))
        True
    """
    return JoinRemainderResolver(new_remainder_edit(fixture, {"decl.Rest", "decl.S", "kept.R"}))


def new_expr_codec(fixture: TypeFixture) -> QuadrupleCodec:
    """Build the codec that writes the type of an expression as a mypy type, and reads it back.

    The engine describes an expression with four parts: the columns it
    reads, a table from their data types to the data type it gives, the
    name of its output, and how it counts rows. The codec turns that
    value into marker types such as `TypedExpr[...]`, and turns such a
    type back into the value. Use it for a doctest that needs a typed
    expression.

    Args:
        fixture: The `TypeFixture` that makes each class.

    Returns:
        The codec, over a fake plugin and a fake API.

    Example:
        >>> from astrolabe._hooks.expression.roots import LiteralRoot
        >>> codec = new_expr_codec(new_type_fixture())
        >>> codec.encoded(LiteralRoot().of("Int32")) is not None
        True
    """
    return QuadrupleCodec(new_marker_minter(fixture))


def new_selection_codec(fixture: TypeFixture) -> SelectionCodec:
    """Build the codec for the type of a selection, such as `cs.numeric()`.

    A selection picks columns by a rule, and can then apply operations to
    every column it picks. The codec writes that value as a marker type,
    and reads it back. Use it for a doctest that needs a typed selection.

    Args:
        fixture: The `TypeFixture` that makes each class.

    Returns:
        The codec, over a fake plugin and a fake API.

    Example:
        >>> from astrolabe._hooks.expression.selection import SelectionType
        >>> from astrolabe._hooks.remainder.selector import OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> codec = new_selection_codec(new_type_fixture())
        >>> codec.encoded(SelectionType.identity(OfType(Family.numeric()))) is not None
        True
    """
    return SelectionCodec(new_marker_minter(fixture))


def new_concat_remainder_resolver(fixture: TypeFixture) -> ConcatRemainderResolver:
    """Build the class that decides the remainder of a `pl.concat` result.

    A concat of frames with a remainder must also account for the
    columns that no type names. The resolver decides whether the
    remainders of the frames agree, and what the result holds. The same
    three class names as in `new_join_remainder_resolver` read as
    declared remainders.

    Args:
        fixture: The `TypeFixture` that makes each class.

    Returns:
        The resolver, over a fake plugin and a fake API.

    Example:
        >>> resolver = new_concat_remainder_resolver(new_type_fixture())
        >>> resolver.outcome([]).allowed
        True
    """
    return ConcatRemainderResolver(new_remainder_edit(fixture, {"decl.Rest", "decl.S", "kept.R"}))


def new_declaration_codec(fixture: TypeFixture) -> DeclarationCodec:
    """Build the codec that reads the requirements of a declared remainder.

    A user declares a remainder with `other_columns(...)`, and can add
    requirements to it, for example that it holds no column named `z`.
    The plugin stores each requirement as a base class of the declared
    class. The codec reads those bases back. It needs no mypy object, so
    the fixture is accepted only to keep the same call as the other
    codec builders.

    Args:
        fixture: Not used.

    Returns:
        The codec.

    Example:
        >>> codec = new_declaration_codec(new_type_fixture())
        >>> codec.relations_of(declaration_info(new_type_fixture()))
        Relations(disjoint_from=(), within=())
    """
    del fixture
    return DeclarationCodec()


def declaration_info(
    fixture: TypeFixture,
    *bases: Instance,
    without: tuple[str, ...] = (),
    name: str = "decl.Rest",
) -> TypeInfo:
    """Build the class that `other_columns(...)` registers for a declared remainder.

    In a real build, the user writes `Rest = other_columns(...)` and the
    plugin registers a class for `Rest`. That class has the base
    `DeclaredRemainder`, which marks it as a remainder, and one more base
    for each requirement. This builder makes the same class by hand.

    Args:
        fixture: The `TypeFixture` that makes each class.
        *bases: More requirement bases, already built.
        without: The names that the remainder must not hold. If the tuple
            is not empty, the class gets a `Without` base for them. Most
            examples need this requirement, so the builder makes it.
        name: The full name of the class.

    Returns:
        The class. Its bases are `object`, `DeclaredRemainder`, the
        `Without` base if any, and then `bases`.

    Example:
        >>> info = declaration_info(new_type_fixture(), without=("z",))
        >>> len(info.bases)
        3
    """
    minter = new_marker_minter(fixture)
    declared = checked_not_none(minter.marker(remainder_markers.DeclaredRemainder, []))
    marked = [Instance(fixture.oi, []), declared]
    if without:
        names = NameSetCodec(minter).built(frozenset(without))
        marked.append(checked_not_none(minter.marker(remainder_markers.Without, [names])))
    marked.extend(bases)
    return fixture.make_type_info(name, bases=marked, mro=[declared.type, fixture.oi])


def declared_rest(
    fixture: TypeFixture,
    *,
    without: tuple[str, ...] = (),
    name: str = "decl.Rest",
) -> Instance:
    """Build the remainder part of a frame type that holds one declared remainder.

    A frame type has two type arguments: the named columns, and the
    remainder. When the user writes a declared remainder there with no
    change on it, that argument is an instance of the declared class.
    This builder makes that argument.

    Args:
        fixture: The `TypeFixture` that makes each class.
        without: The names that the remainder must not hold, if any.
        name: The full name of the declared class.

    Returns:
        An instance of the declared class.

    Example:
        >>> slot = declared_rest(new_type_fixture(), without=("z",))
        >>> slot.type.fullname
        'decl.Rest'
    """
    return Instance(declaration_info(fixture, without=without, name=name), [])


def new_remainder_codec(fixture: TypeFixture, declared: set[str]) -> RemainderCodec:
    """Build the codec that writes a remainder as a mypy type, and reads it back.

    The engine holds a remainder as a plain value: a declared remainder
    with the changes that operations made to it, for example a dropped
    name or a suffix. The codec turns that value into marker types such
    as `Minus[...]`, and turns such a type back into the value.

    Args:
        fixture: The `TypeFixture` that makes each class.
        declared: The full names of the classes that read as declared
            remainders. Any other class does not read as a remainder.

    Returns:
        The codec, over a fake plugin and a fake API.

    Example:
        >>> from mypy.types import UninhabitedType
        >>> codec = new_remainder_codec(new_type_fixture(), {"kept.R"})
        >>> codec.decoded(UninhabitedType()) is None
        True
    """
    return RemainderCodec(new_marker_minter(fixture), NamedDeclarations(frozenset(declared)))


def new_selector_codec(fixture: TypeFixture) -> SelectorCodec:
    """Build the codec for a selector, the rule that picks columns by name and data type.

    A selector such as `cs.starts_with("x") & cs.numeric()` is written as
    marker types, for example `SelAnd[...]`. The codec builds that type
    from the engine value, and reads it back.

    Args:
        fixture: The `TypeFixture` that makes each class.

    Returns:
        The codec, over a fake plugin and a fake API.

    Example:
        >>> new_selector_codec(new_type_fixture()).decoded(None) is None
        True
    """
    return SelectorCodec(new_marker_minter(fixture))


def new_marker_minter(fixture: TypeFixture) -> _EncodingMinter:
    """Build the minter that every codec uses to make a marker type or a literal type.

    A marker is a small class in a public module of astrolabe, and a
    codec writes each value as instances of such classes. The minter
    here works as in the type-checking phase of mypy. Use it for a
    doctest that builds a marker type by hand.

    Args:
        fixture: The `TypeFixture` that makes each class.

    Returns:
        The minter, over a fake plugin and a fake API.

    Example:
        >>> new_marker_minter(new_type_fixture()).string("a")
        Literal['a']
    """
    return _EncodingMinter(new_plugin(fixture), CheckerLiteralTypeBuilder(), new_api(fixture))


def int_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Int64`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Int64`.

    Example:
        >>> int_dtype(new_type_fixture())
        polars.datatypes.classes.Int64
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Int64"), [])


def str_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `String`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.String`.

    Example:
        >>> str_dtype(new_type_fixture())
        polars.datatypes.classes.String
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.String"), [])


def boolean_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Boolean`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Boolean`.

    Example:
        >>> boolean_dtype(new_type_fixture())
        polars.datatypes.classes.Boolean
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Boolean"), [])


def unchecked_instance(fixture: TypeFixture) -> Instance:
    """Build the type `Unchecked`, as the data type of a column.

    `Unchecked` is the data type that the checker gives a column whose
    name it knows and whose data type it cannot prove. A test of how
    plugin code treats such a column starts here.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `astrolabe.frame.Unchecked`.

    Example:
        >>> unchecked_instance(new_type_fixture())
        astrolabe.frame.Unchecked
    """
    return Instance(fixture.make_type_info(Target.of(_UncheckedMarker).fullname), [])


def int16_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Int16`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Int16`.

    Example:
        >>> int16_dtype(new_type_fixture())
        polars.datatypes.classes.Int16
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Int16"), [])


def uint32_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `UInt32`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.UInt32`.

    Example:
        >>> uint32_dtype(new_type_fixture())
        polars.datatypes.classes.UInt32
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.UInt32"), [])


def uint64_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `UInt64`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.UInt64`.

    Example:
        >>> uint64_dtype(new_type_fixture())
        polars.datatypes.classes.UInt64
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.UInt64"), [])


def int64_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Int64`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Int64`.

    Example:
        >>> int64_dtype(new_type_fixture())
        polars.datatypes.classes.Int64
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Int64"), [])


def float64_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Float64`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Float64`.

    Example:
        >>> float64_dtype(new_type_fixture())
        polars.datatypes.classes.Float64
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Float64"), [])


def int8_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Int8`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Int8`.

    Example:
        >>> int8_dtype(new_type_fixture())
        polars.datatypes.classes.Int8
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Int8"), [])


def int32_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Int32`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Int32`.

    Example:
        >>> int32_dtype(new_type_fixture())
        polars.datatypes.classes.Int32
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Int32"), [])


def uint8_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `UInt8`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.UInt8`.

    Example:
        >>> uint8_dtype(new_type_fixture())
        polars.datatypes.classes.UInt8
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.UInt8"), [])


def float32_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Float32`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Float32`.

    Example:
        >>> float32_dtype(new_type_fixture())
        polars.datatypes.classes.Float32
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Float32"), [])


def datetime_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Datetime`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Datetime`, with no time
        unit and no time zone.

    Example:
        >>> datetime_dtype(new_type_fixture())
        polars.datatypes.classes.Datetime
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Datetime"), [])


def parametrized_datetime_dtype(fixture: TypeFixture, unit: str, zone: str | None) -> Instance:
    """Build a `Datetime` data type that carries its time unit and time zone.

    polars stores a time unit and a time zone in each `Datetime` data
    type, and the checker treats them as part of the type. astrolabe
    writes them as the literal type arguments of its own `Datetime`
    marker, which is a subclass of the polars class. This builder makes
    that marker type, for a test that needs the unit or the zone.

    Args:
        fixture: The `TypeFixture` that makes each class.
        unit: The time unit, for example `"us"`.
        zone: The time zone, for example `"UTC"`, or None for no zone.

    Returns:
        An instance of `astrolabe.frame.Datetime`, with the unit and the
        zone as literal type arguments.

    Example:
        >>> parametrized_datetime_dtype(new_type_fixture(), "ns", "UTC")
        astrolabe.frame.Datetime[Literal['ns'], Literal['UTC']]
        >>> parametrized_datetime_dtype(new_type_fixture(), "us", None)
        astrolabe.frame.Datetime[Literal['us'], None]
    """
    plain_datetime_info = fixture.make_type_info("polars.datatypes.classes.Datetime")
    plain_datetime_base = Instance(plain_datetime_info, [])
    datetime_info = fixture.make_type_info(
        Target.of(_DatetimeMarker).fullname,
        bases=[plain_datetime_base],
        mro=[plain_datetime_info],
    )
    zone_type = _datetime_zone_type(fixture, zone)
    return Instance(datetime_info, [literal_str_type(fixture, unit), zone_type])


def _datetime_zone_type(fixture: TypeFixture, zone: str | None) -> MypyType:
    """Build the type argument that holds the time zone of a `Datetime`.

    Args:
        fixture: The `TypeFixture` that makes the `str` class of the literal.
        zone: The time zone, or None for no zone.

    Returns:
        `Literal[zone]`. If `zone` is None, the type `None`.
    """
    if zone is None:
        return NoneType()
    return literal_str_type(fixture, zone)


def duration_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Duration`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Duration`, with no time
        unit.

    Example:
        >>> duration_dtype(new_type_fixture())
        polars.datatypes.classes.Duration
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Duration"), [])


def decimal_dtype(fixture: TypeFixture, precision: int | None, scale: int) -> Instance:
    """Build a `Decimal` data type that carries its precision and scale.

    The precision is the count of digits in all, and the scale is the
    count of digits after the point. The checker treats both as part of
    the type, and astrolabe writes them as the literal type arguments of
    its own `Decimal` marker, a subclass of the polars class.

    Args:
        fixture: The `TypeFixture` that makes each class.
        precision: The precision, or None when polars sets no precision.
            None is the polars default.
        scale: The scale.

    Returns:
        An instance of `astrolabe.frame.Decimal`, with the precision and
        the scale as type arguments.

    Example:
        >>> decimal_dtype(new_type_fixture(), 10, 2)
        astrolabe.frame.Decimal[Literal[10], Literal[2]]
        >>> decimal_dtype(new_type_fixture(), None, 2)
        astrolabe.frame.Decimal[None, Literal[2]]
    """
    plain_decimal_info = fixture.make_type_info("polars.datatypes.classes.Decimal")
    plain_decimal_base = Instance(plain_decimal_info, [])
    decimal_info = fixture.make_type_info(
        Target.of(_DecimalMarker).fullname,
        bases=[plain_decimal_base],
        mro=[plain_decimal_info],
    )
    precision_type = _decimal_precision_type(fixture, precision)
    return Instance(decimal_info, [precision_type, literal_int_type(fixture, scale)])


def _decimal_precision_type(fixture: TypeFixture, precision: int | None) -> MypyType:
    """Build the type argument that holds the precision of a `Decimal`.

    Args:
        fixture: The `TypeFixture` that makes the `int` class of the literal.
        precision: The precision, or None for no precision.

    Returns:
        `Literal[precision]`. If `precision` is None, the type `None`.
    """
    if precision is None:
        return NoneType()
    return literal_int_type(fixture, precision)


def enum_dtype(fixture: TypeFixture, *categories: str) -> Instance:
    """Build an `Enum` data type that carries its categories.

    A polars `Enum` holds a fixed list of text values. The checker treats
    that list as part of the type, and astrolabe writes each category as
    a literal type argument of its own `Enum` marker.

    Args:
        fixture: The `TypeFixture` that makes each class.
        *categories: The categories, in order.

    Returns:
        An instance of `astrolabe.frame.Enum`, with one literal type
        argument for each category.

    Example:
        >>> enum_dtype(new_type_fixture(), "a", "b")
        astrolabe.frame.Enum[Literal['a'], Literal['b']]
    """
    plain_enum_info = fixture.make_type_info("polars.datatypes.classes.Enum")
    plain_enum_base = Instance(plain_enum_info, [])
    enum_info = fixture.make_type_info(
        Target.of(_EnumMarker).fullname,
        bases=[plain_enum_base],
        mro=[plain_enum_info],
    )
    category_types: list[MypyType] = [literal_str_type(fixture, name) for name in categories]
    return Instance(enum_info, category_types)


def binary_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Binary`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Binary`.

    Example:
        >>> binary_dtype(new_type_fixture())
        polars.datatypes.classes.Binary
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Binary"), [])


def categorical_dtype(fixture: TypeFixture) -> Instance:
    """Build the mypy type of the polars data type `Categorical`, as the data type of a column.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `polars.datatypes.classes.Categorical`.

    Example:
        >>> categorical_dtype(new_type_fixture())
        polars.datatypes.classes.Categorical
    """
    return Instance(fixture.make_type_info("polars.datatypes.classes.Categorical"), [])


def list_dtype(fixture: TypeFixture, inner: Instance) -> Instance:
    """Build a `List` data type over the data type of its elements.

    Each call makes a new `TypeInfo` for `List`, so two results of two
    calls never compare equal. Use `list_dtype_pair` where a test
    compares two lists.

    Args:
        fixture: The `TypeFixture` that makes the class.
        inner: The data type of each element.

    Returns:
        An instance of `astrolabe.frame.List` over `inner`.

    Example:
        >>> list_dtype(new_type_fixture(), int_dtype(new_type_fixture()))
        astrolabe.frame.List[polars.datatypes.classes.Int64]
    """
    list_info = fixture.make_type_info(Target.of(_ListMarker).fullname)
    return Instance(list_info, [inner])


def list_dtype_pair(
    fixture: TypeFixture,
    left_inner: Instance,
    right_inner: Instance,
) -> list[Instance]:
    """Build two `List` data types over one `TypeInfo`, so that they can compare equal.

    mypy compares two instances by their `TypeInfo` object, and the
    fixture makes a new `TypeInfo` on each request. Two calls of
    `list_dtype` therefore give lists that never compare equal, even over
    the same element type. A test that compares two lists, for example
    one that reconciles the element types of two columns, needs this
    builder instead.

    Args:
        fixture: The `TypeFixture` that makes the class.
        left_inner: The element type of the first list.
        right_inner: The element type of the second list.

    Returns:
        The two lists, in that order.

    Example:
        >>> fixture = new_type_fixture()
        >>> left, right = list_dtype_pair(fixture, int_dtype(fixture), int_dtype(fixture))
        >>> left.type == right.type
        True
    """
    list_info = fixture.make_type_info(Target.of(_ListMarker).fullname)
    return [Instance(list_info, [left_inner]), Instance(list_info, [right_inner])]


def array_dtype(fixture: TypeFixture, inner: Instance) -> Instance:
    """Build an `Array` data type of width 3 over the data type of its elements.

    A polars `Array` holds the same count of elements in every row, and
    the checker keeps that width in the type. Every test that uses this
    builder works with a width of 3, so the width is fixed.

    Args:
        fixture: The `TypeFixture` that makes each class.
        inner: The data type of each element.

    Returns:
        An instance of `astrolabe.frame.Array` over `inner` and
        `Literal[3]`.

    Example:
        >>> array_dtype(new_type_fixture(), int_dtype(new_type_fixture()))
        astrolabe.frame.Array[polars.datatypes.classes.Int64, Literal[3]]
    """
    array_info = fixture.make_type_info(Target.of(_ArrayMarker).fullname)
    return Instance(array_info, [inner, literal_int_type(fixture, 3)])


def class_object(
    fixture: TypeFixture, instance: Instance, *, generic: bool = False
) -> CallableType:
    """Build the mypy type of a class used as a value, such as the argument of `.cast(List)`.

    mypy types a class in a value position as a callable that builds an
    instance. A generic class with no type arguments keeps its type
    variables on that callable. A class with its type arguments, such
    as `List[pl.Int64]`, has none.

    Args:
        fixture: The `TypeFixture` that gives the `type` class.
        instance: The instance that the callable builds.
        generic: True for a generic class with no type arguments.

    Returns:
        The callable type of the class.

    Example:
        >>> fixture = new_type_fixture()
        >>> class_object(fixture, int_dtype(fixture)).is_type_obj()
        True
    """
    variables = [plain_typevar(fixture)] if generic else []
    return CallableType(
        [],
        [],
        [],
        instance,
        Instance(fixture.type_typei, []),
        variables=variables,
    )


def user_dtype(fixture: TypeFixture, name: str, base: str = "Int64") -> Instance:
    """Build an instance of a class that user code defines as a subclass of a polars data type.

    Such a class, for example `class Tagged(pl.Int64)`, looks like a data
    type to mypy, but polars does not read it as one in most calls. A test
    of how plugin code treats it starts here.

    Args:
        fixture: The `TypeFixture` that makes the classes.
        name: The short name of the class, in the module `user`.
        base: The short name of the polars data type that it subclasses.

    Returns:
        An instance of `user.<name>`, whose MRO holds the polars class and
        `pl.DataType`.

    Example:
        >>> tagged = user_dtype(new_type_fixture(), "Tagged")
        >>> [info.fullname for info in tagged.type.mro]
        ['user.Tagged', 'polars.datatypes.classes.Int64', 'polars.datatypes.classes.DataType']
    """
    data_type = fixture.make_type_info("polars.datatypes.classes.DataType")
    polars_class = fixture.make_type_info(f"polars.datatypes.classes.{base}", mro=[data_type])
    info = fixture.make_type_info(f"user.{name}", mro=[polars_class, data_type])
    return Instance(info, [])


def struct_instance(fixture: TypeFixture, *fields: ColumnRef) -> Instance:
    """Build a `Struct` data type over its fields.

    A polars `Struct` column holds named fields in each row, and each
    field has its own data type. astrolabe writes the fields as a tuple
    of `Column[name, data type]` types inside its `Struct` marker. Build
    each field with `column_ref`.

    Args:
        fixture: The `TypeFixture` that makes each class.
        *fields: The fields, in order.

    Returns:
        An instance of `astrolabe.frame.Struct` over the fields.

    Example:
        >>> fixture = new_type_fixture()
        >>> price = column_ref(fixture, "price", int_dtype(fixture))
        >>> struct = struct_instance(fixture, price)  # Struct[Column["price", Int64]]
        >>> struct.type.fullname
        'astrolabe.frame.Struct'
    """
    struct_info = fixture.make_type_info(Target.of(_StructMarker).fullname)
    fields_tuple = columns_tuple_type(fixture, [field.as_instance() for field in fields])
    return Instance(struct_info, [fields_tuple])


def literal_str_type(fixture: TypeFixture, value: str) -> LiteralType:
    """Build the literal type of one `str` value, as mypy writes it.

    A literal type such as `Literal[...]` holds one exact value. mypy
    stores it with a fallback, the plain `str` class that the value
    belongs to.

    Args:
        fixture: The `TypeFixture` that makes the `str` class.
        value: The value.

    Returns:
        The literal type of `value`.

    Example:
        >>> literal_str_type(new_type_fixture(), "price")
        Literal['price']
    """
    str_fallback = Instance(fixture.str_type_info, [])
    return LiteralType(value, str_fallback)


def literal_int_type(fixture: TypeFixture, value: int) -> LiteralType:
    """Build the literal type of one `int` value, as mypy writes it.

    A literal type such as `Literal[...]` holds one exact value. mypy
    stores it with a fallback, the plain `int` class that the value
    belongs to.

    Args:
        fixture: The `TypeFixture` that makes the `int` class.
        value: The value.

    Returns:
        The literal type of `value`.

    Example:
        >>> literal_int_type(new_type_fixture(), 3)
        Literal[3]
    """
    int_fallback = Instance(fixture.make_type_info("builtins.int"), [])
    return LiteralType(value, int_fallback)


def literal_bool_type(fixture: TypeFixture, value: bool) -> LiteralType:
    """Build the literal type of one `bool` value, as mypy writes it.

    A literal type such as `Literal[...]` holds one exact value. mypy
    stores it with a fallback, the plain `bool` class that the value
    belongs to.

    Args:
        fixture: The `TypeFixture` that makes the `bool` class.
        value: The value.

    Returns:
        The literal type of `value`.

    Example:
        >>> literal_bool_type(new_type_fixture(), True)
        Literal[True]
    """
    bool_fallback = Instance(fixture.make_type_info("builtins.bool"), [])
    return LiteralType(value, bool_fallback)


def literal_receiver_instance(fixture: TypeFixture, value: int | float | str | bool) -> Instance:
    """Build the type that mypy gives a `Final` variable, which carries the known value.

    For `NAME: Final = "a"`, mypy gives `NAME` the type `str`, and keeps
    the value `"a"` in the `last_known_value` of that type. Plugin code
    that reads an argument value reads it from there when the caller
    passes such a name in place of a literal. This builder makes that
    type, for a test of such code.

    Args:
        fixture: The `TypeFixture` that makes the builtin class.
        value: The known value. Its exact class picks the builtin, so
            `True` gives `builtins.bool` and not `builtins.int`.

    Returns:
        An instance of `int`, `float`, `str` or `bool`, whose
        `last_known_value` is the literal type of `value`.

    Example:
        >>> literal_receiver_instance(new_type_fixture(), 5).last_known_value
        Literal[5]
        >>> literal_receiver_instance(new_type_fixture(), "a").last_known_value
        Literal['a']
    """
    fullnames = {
        bool: "builtins.bool",
        int: "builtins.int",
        float: "builtins.float",
        str: "builtins.str",
    }
    info = fixture.make_type_info(fullnames[type(value)])
    fallback = Instance(info, [])
    return Instance(info, [], last_known_value=LiteralType(value=value, fallback=fallback))


def builtin_instance(fixture: TypeFixture, fullname: str) -> Instance:
    """Build the type that mypy gives an ordinary variable of a builtin class, with no known value.

    For `share: float = 1`, mypy gives `share` the type `float` and keeps no value. Plugin code
    that reads a number from its type has only the class to go on. This builder makes that type,
    for a test of such code.

    Args:
        fixture: The `TypeFixture` that makes the builtin class.
        fullname: The full name of the class, for example `builtins.float`.

    Returns:
        An instance of the class, with no `last_known_value`.

    Example:
        >>> builtin_instance(new_type_fixture(), "builtins.float")
        builtins.float
    """
    return Instance(fixture.make_type_info(fullname), [])


def column_instance(fixture: TypeFixture, name: str, dtype: Instance) -> Instance:
    """Build the mypy type of one column, `Column[Literal[name], data type]`.

    astrolabe writes a column as a `Column` marker whose two type
    arguments are the name, as a literal, and the data type. This is the
    raw type. `column_ref` wraps it in the class that plugin code reads.

    Args:
        fixture: The `TypeFixture` that makes each class.
        name: The name of the column.
        dtype: The data type of the column, from a builder such as
            `int_dtype`.

    Returns:
        An instance of `astrolabe.frame.Column`.

    Example:
        >>> fixture = new_type_fixture()
        >>> dtype = int_dtype(fixture)
        >>> column_instance(fixture, "price", dtype)
        astrolabe.frame.Column[Literal['price'], polars.datatypes.classes.Int64]
    """
    column_info = fixture.make_type_info(ColumnRef.TARGET.fullname)
    return Instance(column_info, [literal_str_type(fixture, name), dtype])


def typed_column_instance(fixture: TypeFixture, name: str, dtype: Instance) -> Instance:
    """Build the mypy type of a declared column, `TypedColumn[Literal[name], data type]`.

    A declaration such as `Price = column("price", pl.Float64)` gives this
    type. A code of the deferred model reads it as a leaf.

    Args:
        fixture: The `TypeFixture` that makes each class.
        name: The name of the column.
        dtype: The data type of the column, from a builder such as `int_dtype`.

    Returns:
        An instance of `astrolabe.expr.TypedColumn`.

    Example:
        >>> fixture = new_type_fixture()
        >>> typed_column_instance(fixture, "price", int_dtype(fixture))
        astrolabe.expr.TypedColumn[Literal['price'], polars.datatypes.classes.Int64]
    """
    column_info = fixture.make_type_info(Target.of(_TypedColumn).fullname)
    return Instance(column_info, [literal_str_type(fixture, name), dtype])


def column_ref(fixture: TypeFixture, name: str, dtype: Instance) -> ColumnRef:
    """Build one column as the plugin reads it, with a name and a data type.

    `ColumnRef` is the plugin class that wraps the mypy type of a column
    and answers its name and its data type. It is the usual input of a
    test about columns and frames. Call `as_instance()` on the result to
    get the raw mypy type back, for example to build a frame with
    `frame`.

    Args:
        fixture: The `TypeFixture` that makes each class.
        name: The name of the column.
        dtype: The data type of the column, from a builder such as
            `int_dtype`.

    Returns:
        The column.

    Example:
        >>> fixture = new_type_fixture()
        >>> column = column_ref(fixture, "price", int_dtype(fixture))
        >>> column.name
        'price'
    """
    return ColumnRef(column_instance(fixture, name, dtype))


def int_literal_node(value: int) -> IntExpr:
    """Build the AST node of a whole number that the caller wrote in place, such as `200`.

    Plugin code that reads an argument value, for example the `n` of
    `head(n)`, reads such a node.

    Args:
        value: The number.

    Returns:
        The node.

    Example:
        >>> int_literal_node(200).value
        200
    """
    return IntExpr(value)


def float_literal_node(value: float) -> FloatExpr:
    """Build the AST node of a float that the caller wrote in place, such as `2.5`.

    Args:
        value: The number.

    Returns:
        The node.

    Example:
        >>> float_literal_node(2.5).value
        2.5
    """
    return FloatExpr(value)


def str_literal_node(value: str) -> StrExpr:
    """Build the AST node of a string that the caller wrote in place, such as `"price"`.

    A column name in `pl.col("price")` or `drop("price")` reaches plugin
    code as such a node.

    Args:
        value: The text of the string.

    Returns:
        The node.

    Example:
        >>> str_literal_node("price").value
        'price'
    """
    return StrExpr(value)


def str_list_literal_node(*values: str) -> ListExpr:
    """Build the AST node of a list of strings, such as `["a", "b"]`.

    A list of column names, as in `drop(["a", "b"])`, reaches plugin code
    as such a node.

    Args:
        values: The strings, in order.

    Returns:
        The node, with one string node for each value.

    Example:
        >>> [item.value for item in str_list_literal_node("a", "b").items]
        ['a', 'b']
    """
    return ListExpr([StrExpr(value) for value in values])


def bool_literal_node(value: bool) -> NameExpr:
    """Build the AST node of `True` or `False`, as the caller wrote it.

    In the mypy AST, `True` and `False` are names, not literal nodes.
    Each one is a name node whose full name is `builtins.True` or
    `builtins.False`, and this builder sets that full name.

    Args:
        value: The value.

    Returns:
        The name node.

    Example:
        >>> bool_literal_node(True).fullname
        'builtins.True'
    """
    node = NameExpr("True" if value else "False")
    node.fullname = "builtins.True" if value else "builtins.False"
    return node


def named_call_node(
    fullname: str,
    args: Sequence[Expression] = (),
    arg_names: Sequence[str | None] = (),
) -> CallExpr:
    """Build the AST node of a call of a named function or class, such as `datetime(2020, 1, 1)`.

    Plugin code that recognizes a call by the full name of its callee,
    for example a `datetime` constructor, reads such a node. The callee
    is a name node with the full name set as mypy sets it.

    Args:
        fullname: The full name of the callee, for example
            `datetime.datetime`.
        args: The AST nodes of the arguments, in the order of the call.
        arg_names: The keyword of each argument, or None at a position
            where the argument came by position. A shorter list leaves
            the last arguments positional.

    Returns:
        The call node.

    Example:
        >>> call = named_call_node("datetime.datetime", [int_literal_node(2020)])
        >>> call.callee.fullname, call.arg_names
        ('datetime.datetime', [None])
    """
    callee = NameExpr(fullname.rsplit(".", maxsplit=1)[-1])
    callee.fullname = fullname
    names = [*arg_names, *([None] * (len(args) - len(arg_names)))]
    kinds = [ARG_POS if name is None else ARG_NAMED for name in names]
    return CallExpr(callee, list(args), kinds, names)


def named_reference_node(fullname: str) -> NameExpr:
    """Build the AST node of a name that refers to a known object, such as `UTC`.

    Args:
        fullname: The full name of the object, for example
            `datetime.UTC`.

    Returns:
        The name node, with the full name set as mypy sets it.

    Example:
        >>> named_reference_node("datetime.UTC").name
        'UTC'
    """
    node = NameExpr(fullname.rsplit(".", maxsplit=1)[-1])
    node.fullname = fullname
    return node


def infer_call_node(arg: Expression | None = None) -> CallExpr:
    """Build the AST node of a call `infer(arg)`.

    A user writes `return infer(value)` to mark the value that a function
    with an inferred return type gives. Plugin code finds these calls in
    the body of the function. This builder makes one such call, with the
    full name of `infer` set as mypy sets it.

    Args:
        arg: The AST node of the argument. If it is None, the argument
            is the number 1.

    Returns:
        The call node.

    Example:
        >>> infer_call_node().callee.fullname
        'astrolabe.declare.infer'
    """
    callee = NameExpr("infer")
    callee.fullname = Target.of(infer).fullname
    return CallExpr(callee, [arg if arg is not None else int_literal_node(1)], [ARG_POS], [None])


def func_def_with_body(*stmts: Statement, name: str = "some_func") -> FuncDef:
    """Build the AST node of a function definition with a given body.

    The function has no parameters and no type annotations. It serves a
    test of code that walks the statements of a function body, for
    example to find each `return infer(...)`.

    Args:
        stmts: The statements of the body, in order.
        name: The name of the function.

    Returns:
        The function definition node.

    Example:
        >>> func_def = func_def_with_body(*return_block(infer_call_node()).body)
        >>> len(func_def.body.body)
        1
        >>> func_def.name
        'some_func'
    """
    return FuncDef(name, [], Block(list(stmts)))


def return_block(*exprs: Expression) -> Block:
    """Build a block of `return` statements, one for each expression.

    Pass its `body` to `func_def_with_body` to build a function that
    returns in several places.

    Args:
        exprs: The returned expressions, in order.

    Returns:
        The block.

    Example:
        >>> len(return_block(infer_call_node()).body)
        1
    """
    return Block([ReturnStmt(expr) for expr in exprs])


def series_call_node(name: str) -> CallExpr:
    """Build the AST node of a call `pl.Series("name")`.

    The full name of the callee is set as mypy sets it, so plugin code
    that recognizes a call by its callee knows this call as `pl.Series`.
    The one argument is the name of the series.

    Args:
        name: The name of the series.

    Returns:
        The call node, with the name as its one positional argument.

    Example:
        >>> series_call_node("price").callee.fullname
        'polars.series.series.Series'
        >>> series_call_node("price").args[0].value
        'price'
    """
    callee = MemberExpr(NameExpr("pl"), "Series")
    callee.fullname = "polars.series.series.Series"
    return CallExpr(callee, [StrExpr(name)], [ARG_POS], [None])


def dtype_constructor_call(
    args: Sequence[Expression],
    arg_names: Sequence[str | None] | None = None,
) -> CallExpr:
    """Build the AST node of a data type call with parameters, such as `pl.Datetime("us")`.

    A user can write a data type as a call, for example in
    `cast(pl.Datetime("us", "UTC"))`. Plugin code reads the parameters
    from the arguments of the call. The callee is always `pl.Datetime`,
    because that code reads only the arguments.

    Args:
        args: The AST nodes of the arguments, in the order of the call.
        arg_names: The keyword of each argument, in the same order. None
            at a position means that the argument came by position. If
            the whole value is None, every argument came by position.

    Returns:
        The call node. Each argument is marked as named or positional,
        as mypy marks it.

    Example:
        >>> call = dtype_constructor_call([str_literal_node("us")])
        >>> call.args[0].value
        'us'
    """
    names = _dtype_call_arg_names(args, arg_names)
    kinds = [_dtype_call_arg_kind(name) for name in names]
    callee = MemberExpr(NameExpr("pl"), "Datetime")
    callee.fullname = "polars.datatypes.classes.Datetime"
    return CallExpr(callee, list(args), kinds, names)


def _dtype_call_arg_names(
    args: Sequence[Expression],
    arg_names: Sequence[str | None] | None,
) -> list[str | None]:
    """Give the keyword of each argument of `dtype_constructor_call`.

    Args:
        args: The AST nodes of the arguments.
        arg_names: The keyword of each argument, or None if every
            argument came by position.

    Returns:
        `arg_names` as a list. If it is None, one None for each argument.
    """
    if arg_names is not None:
        return list(arg_names)
    return [None] * len(args)


def _dtype_call_arg_kind(name: str | None) -> ArgKind:
    """Give the mypy kind of one argument: named or positional.

    Args:
        name: The keyword of the argument, or None if it came by position.

    Returns:
        `ARG_NAMED` for a keyword, and `ARG_POS` otherwise.
    """
    if name is None:
        return ARG_POS
    return ARG_NAMED


def union_type(*items: MypyType) -> UnionType:
    """Build the union of several types, such as `Int64 | String`.

    mypy can put a union where plugin code expects one type, for
    example after `x = a if flag else b`. A test of how code reads such
    a type starts here.

    Args:
        items: The members of the union.

    Returns:
        The union.

    Example:
        >>> fixture = new_type_fixture()
        >>> union_type(int_dtype(fixture), str_dtype(fixture))
        polars.datatypes.classes.Int64 | polars.datatypes.classes.String
    """
    return UnionType(list(items))


def columns_tuple_type(fixture: TypeFixture, items: list[MypyType]) -> TupleType:
    """Build a tuple type over some items, as astrolabe writes a list of columns.

    astrolabe writes the fields of a `Struct` as a tuple of `Column`
    types, and a test can put several items in the remainder part of a
    frame as a tuple. Both use this shape.

    Args:
        fixture: The `TypeFixture` that gives the `tuple` class.
        items: The items of the tuple, in order.

    Returns:
        The tuple type.

    Example:
        >>> fixture = new_type_fixture()
        >>> price = column_ref(fixture, "price", int_dtype(fixture))
        >>> len(columns_tuple_type(fixture, [price.as_instance()]).items)
        1
    """
    tuple_fallback = Instance(fixture.std_tuplei, [Instance(fixture.oi, [])])
    return TupleType(items, fallback=tuple_fallback)


def columns_typeddict_type(fixture: TypeFixture, items: list[MypyType]) -> TypedDictType:
    """Build the `TypedDict` type that holds the declared columns of a frame.

    The first type argument of a frame type is a closed `TypedDict` with
    no name. Each key is the name of a column, and each value is its data
    type. The key already holds the name, so the value is the data type
    alone and not a `Column` type. Closed means that the `TypedDict`
    holds no key that it does not list.

    Args:
        fixture: The `TypeFixture` that gives the `TypedDict` class.
        items: The columns, each a `Column` type as `column_instance`
            builds it. Its name becomes the key, and its data type the
            value.

    Returns:
        The `TypedDict` type, with every key required.

    Example:
        >>> fixture = new_type_fixture()
        >>> price = column_ref(fixture, "price", int_dtype(fixture))
        >>> sorted(columns_typeddict_type(fixture, [price.as_instance()]).items)
        ['price']
    """
    fallback = Instance(fixture.make_type_info("typing._TypedDict"), [])
    keyed: dict[str, MypyType] = {}
    for item in items:
        column = checked_not_none(ColumnRef.from_proper_type(item))
        keyed[checked_not_none(column.name)] = column.dtype
    return TypedDictType(
        keyed,
        required_keys=set(keyed),
        readonly_keys=set(),
        fallback=fallback,
        is_closed=True,
    )


def frame(fixture: TypeFixture, items: list[MypyType]) -> Frame:
    """Build a typed frame, as the plugin reads it, from a flat list of items.

    A typed frame is written `TypedDataFrame[Columns, Rest]`. The first
    argument holds the declared columns, the columns that the type names.
    The second holds the remainder, the columns that the type does not
    name. `Frame` is the plugin class that wraps such a type and answers
    questions about it.

    The list mixes both parts, so a test can write a frame in one line.
    Each `Column` item goes into the first argument. Every other item,
    for example a type variable for the remainder, goes into the second.
    If there is no other item, the frame is closed: it holds only its
    declared columns.

    Args:
        fixture: The `TypeFixture` that makes each class.
        items: The items. Build a column with `column_ref(...)` and then
            `.as_instance()`.

    Returns:
        The frame.

    Example:
        >>> fixture = new_type_fixture()
        >>> price = column_ref(fixture, "price", int_dtype(fixture))
        >>> built = frame(fixture, [price.as_instance()])
        >>> [column.name for column in built.columns]
        ['price']
    """
    columns, extras = _partitioned(items)
    columns_dict = columns_typeddict_type(fixture, columns)
    rest_type = _rest_slot(fixture, extras)
    frame_info = fixture.make_type_info(Target.of(_TypedDataFrame).fullname)
    return Frame(Instance(frame_info, [columns_dict, rest_type]))


def _partitioned(items: list[MypyType]) -> tuple[list[MypyType], list[MypyType]]:
    """Split the items of `frame` into the columns and the other items.

    Args:
        items: The items that `frame` received.

    Returns:
        The `Column` items, and then the other items. Each list keeps the
        order of `items`.
    """
    columns: list[MypyType] = []
    extras: list[MypyType] = []
    for item in items:
        _accumulate_by_kind(item, columns, extras)
    return columns, extras


def _accumulate_by_kind(
    item: MypyType,
    columns: list[MypyType],
    extras: list[MypyType],
) -> None:
    """Add one item of `frame` to the list of columns or to the list of other items.

    Args:
        item: The item.
        columns: The `Column` items found so far. If `item` is a
            `Column`, it goes here.
        extras: The other items found so far. Any other `item` goes here.
    """
    if ColumnRef.from_proper_type(item) is not None:
        columns.append(item)
        return
    extras.append(item)


def _rest_slot(fixture: TypeFixture, extras: list[MypyType]) -> MypyType:
    """Build the remainder argument of a frame type from the items that are not columns.

    Args:
        fixture: The `TypeFixture` that gives the `tuple` class, when
            there is more than one item.
        extras: The items that are not columns, in order.

    Returns:
        `Frame.NO_REST`, the type `None` that marks a closed frame, if
        there is no item. The item itself, if there is one. Otherwise a
        tuple of the items, in order.
    """
    if not extras:
        return Frame.NO_REST
    if len(extras) == 1:
        return extras[0]
    return columns_tuple_type(fixture, extras)


def known_columns(
    columns: list[ColumnRef],
    rest: MypyType | None = None,
) -> KnownColumns:
    """Build the columns of a frame as the plugin reads them, with no frame type around them.

    `KnownColumns` holds the declared columns of a frame and, apart, the
    remainder argument of its type, which it does not read. Plugin code
    that works on columns takes this value, so a test can build it
    straight from a list of columns.

    Args:
        columns: The declared columns.
        rest: The remainder argument of the frame type, or None for a
            closed frame.

    Returns:
        The columns.

    Example:
        >>> fixture = new_type_fixture()
        >>> price = column_ref(fixture, "price", int_dtype(fixture))
        >>> list(known_columns([price]).by_name())
        ['price']
    """
    return KnownColumns(columns, rest)


def bare_frame_instance(fixture: TypeFixture) -> Instance:
    """Build a `TypedDataFrame` type with no type arguments at all.

    A real mypy build never gives a frame type this shape. Plugin code
    still must not crash on a type it did not mint, so a test gives it
    this frame, which carries no column, to check that the code gives up
    safely.

    Args:
        fixture: The `TypeFixture` that makes the class.

    Returns:
        An instance of `TypedDataFrame` with no arguments.

    Example:
        >>> bare_frame_instance(new_type_fixture()).args
        ()
    """
    frame_info = fixture.make_type_info(Target.of(_TypedDataFrame).fullname)
    return Instance(frame_info, [])


def untyped_fallback(level: Level = Level.OMITTED, report_gaps: bool = False) -> UntypedFallback:
    """Build the fallback that a frame hook uses when it cannot type its result.

    When a hook cannot prove the columns of a result, it falls back: it
    gives the plain polars frame, which the checker does not track. The
    `allow_unproven` setting decides whether the checker also reports an
    error there. Use this builder for a doctest of a hook that can fall
    back.

    Args:
        level: The level of `allow_unproven`, that is, how much the user
            lets pass without a report. The default is the default of the
            plugin, which most doctests need.
        report_gaps: The value of the `report_checker_gaps` setting,
            which reports the places where the checker itself lacks a
            rule.

    Returns:
        The fallback, with that setting.

    Example:
        >>> untyped_fallback()._first_resolved(None) is None
        True
    """
    policy = UnprovenPolicy(level, report_gaps=report_gaps)
    return UntypedFallback(policy, FakePlugin(new_type_fixture()))


def unchecked_compatibility() -> UncheckedCompatibility:
    """Build the class that compares two data types when one of them can be `Unchecked`.

    `Unchecked` is the data type of a column whose name is known and
    whose data type is not. Such a column must not fail a comparison
    with a real data type, so plugin code that compares data types asks
    this class first.

    Returns:
        A new instance.

    Example:
        >>> unchecked_compatibility() is not None
        True
    """
    return UncheckedCompatibility()


def dtype_promotion(plugin: PluginLookup) -> DTypePromotion:
    """Build the class that finds the common data type of two data types.

    Some operations put the values of two columns into one column, for
    example a concat that aligns frames by name. polars then gives the
    column a common supertype, such as `Int64` for `Int32` and `Int64`.
    This class models a part of that rule.

    Args:
        plugin: The stand-in plugin, which finds the class of a data type
            that the class builds.

    Returns:
        A new instance.

    Example:
        >>> dtype_promotion(new_plugin(new_type_fixture())) is not None
        True
    """
    return DTypePromotion(UncheckedCompatibility(), plugin)


def schema_equality(plugin: PluginLookup) -> SchemaEquality:
    """Build the class that answers whether two lists of columns agree, column by column.

    Two lists agree when they have the same length, and the columns at
    each position have the same name and a data type that matches. In
    the relaxed mode, two data types that have a common supertype also
    match.

    Args:
        plugin: The stand-in plugin, which the inner promotion rule uses
            to find the class of a data type.

    Returns:
        A new instance.

    Example:
        >>> schema_equality(new_plugin(new_type_fixture())) is not None
        True
    """
    return SchemaEquality(
        UncheckedCompatibility(),
        DTypePromotion(UncheckedCompatibility(), plugin),
    )


def join_key_compatibility(plugin: PluginLookup) -> JoinKeyCompatibility:
    """Build the class that decides whether two keys can join, and the key type after a join.

    polars joins two key columns only when their data types agree well
    enough, for example any two integer types. A full join that
    coalesces the keys also gives the key column a new common data type.
    This class holds both rules.

    Args:
        plugin: The stand-in plugin, which finds the class of the common
            data type.

    Returns:
        A new instance.

    Example:
        >>> join_key_compatibility(new_plugin(new_type_fixture())) is not None
        True
    """
    return JoinKeyCompatibility(UncheckedCompatibility(), plugin)


def reducer_dtype(plugin: PluginLookup) -> ReducerDtype:
    """Build the class that gives the data type of a column after a reduction, such as `sum`.

    A reduction turns a column into one value, and its data type can
    differ from the data type of the column. For example, the mean of an
    integer column is a float. Plugin code that types `df.sum()` or the
    aggregate of a `pivot` asks this class.

    Args:
        plugin: The stand-in plugin, which finds the class of each result
            data type.

    Returns:
        A new instance.

    Example:
        >>> reducer_dtype(new_plugin(new_type_fixture())) is not None
        True
    """
    return ReducerDtype(plugin, DtypeFamilies(plugin))


def dtype_families(plugin: PluginLookup) -> DtypeFamilies:
    """Build the class that answers which family a data type belongs to.

    A family is a group of data types that an operation treats alike,
    for example every numeric type, or every float type. Plugin code asks
    this class before it applies a rule that holds for a whole family.

    Args:
        plugin: The stand-in plugin, which finds the class of a data type
            that the class builds, for example a wider `Decimal`.

    Returns:
        A new instance.

    Example:
        >>> dtype_families(new_plugin(new_type_fixture())) is not None
        True
    """
    return DtypeFamilies(plugin)


def unchecked_fallback(
    level: Level = Level.OMITTED,
    report_gaps: bool = False,
) -> UncheckedFallback:
    """Build the fallback that gives one column the data type `Unchecked`.

    When a hook knows the name of a result column but cannot prove its
    data type, it keeps the column under `Unchecked` and does not drop
    the whole frame type. The `allow_unproven` setting decides whether
    the checker also reports an error there. Use this builder for a
    doctest of a hook that can do this.

    Args:
        level: The level of `allow_unproven`. The default is the default
            of the plugin, which most doctests need.
        report_gaps: The value of the `report_checker_gaps` setting.

    Returns:
        The fallback, with that setting.

    Example:
        >>> unchecked_fallback() is not None
        True
    """
    policy = UnprovenPolicy(level, report_gaps=report_gaps)
    return UncheckedFallback(policy, FakePlugin(new_type_fixture()))


def typed_data_frame_class() -> type[_TypedDataFrame]:
    """Give the Python class `TypedDataFrame` itself.

    Some hooks take the frame class that they give in their constructor.
    For example, one hook class serves `pl.read_csv`, which gives an
    eager frame, and `pl.scan_csv`, which gives a lazy one. A doctest of
    such a hook gets the class here, with no import line.

    Returns:
        The class `TypedDataFrame`.

    Example:
        >>> typed_data_frame_class() is _TypedDataFrame
        True
    """
    return _TypedDataFrame


def typed_lazy_frame_class() -> type[_TypedLazyFrame]:
    """Give the Python class `TypedLazyFrame` itself.

    See `typed_data_frame_class`, which does the same for the eager
    frame class.

    Returns:
        The class `TypedLazyFrame`.

    Example:
        >>> typed_lazy_frame_class() is _TypedLazyFrame
        True
    """
    return _TypedLazyFrame


def plain_typevar(fixture: TypeFixture) -> TypeVarType:
    """Build a type variable `T` whose upper bound is `object`.

    A type variable can reach plugin code where it expects a data type,
    for example in a generic function. It is not a data type and not a
    remainder, so the code must answer that it cannot decide. A test of
    that case needs this value.

    Args:
        fixture: The `TypeFixture` that gives the `object` type.

    Returns:
        The type variable.

    Example:
        >>> plain_typevar(new_type_fixture()).name
        'T'
    """
    object_instance = Instance(fixture.oi, [])
    return TypeVarType("T", "test.T", TypeVarId(2), [], object_instance, object_instance)


def update_plan(
    fixture: TypeFixture,
    left_columns: Sequence[ColumnRef],
    right_columns: Sequence[ColumnRef],
    right_keys: Sequence[str],
) -> _UpdatePlan:
    """Build the inputs that the hook for `df.update(other)` reads, with no mypy context.

    `update` writes the values of `other` into the matching columns of
    the frame, and joins the two frames on key columns. The hook first
    gathers the columns of both sides and the names of the keys in one
    plan. A real plan comes from the context of a call, so a doctest
    builds it here.

    Args:
        fixture: The `TypeFixture` that makes each class.
        left_columns: The columns of the frame that `update` is called on.
        right_columns: The columns of `other`.
        right_keys: The names of the key columns of `other`. A key column
            is never a source of new values.

    Returns:
        The plan.

    Example:
        >>> fixture = new_type_fixture()
        >>> key = column_ref(fixture, "k", int_dtype(fixture))
        >>> update_plan(fixture, [key], [key], ["k"]).right_keys
        frozenset({'k'})
    """
    receiver = frame(fixture, [column.as_instance() for column in left_columns])
    by_name = {checked_not_none(column.name): column for column in right_columns}
    return _UpdatePlan(receiver, list(left_columns), by_name, frozenset(right_keys))


def call_args(
    names: Sequence[str],
    nodes: Sequence[list[Expression]],
    arg_names: Sequence[list[str | None]] | None = None,
    api: object = None,
) -> CallArgs:
    """Build the reader of the argument values of one call, with no mypy build.

    `CallArgs` is the plugin class that reads the value a caller wrote
    for a parameter, for example the `3` of `head(3)`. It reads the call
    from the context that mypy gives a hook. This builder gives it a
    stand-in context instead, so a test can describe a call as lists of
    AST nodes.

    Args:
        names: The name of each parameter, in the same order as `nodes`.
        nodes: The AST nodes that the caller wrote for each parameter.
            An empty list stands for a parameter that the caller left out.
        arg_names: The keyword that the caller wrote before each node.
            A `**kwargs` parameter needs it, because its keywords are not
            parameter names. The default is None for every node, as in a
            call that passes every argument by position.
        api: The mypy checker API, which a name lookup reads. The default
            is None, which is safe when every node is a literal.

    Returns:
        The reader.

    Example:
        >>> call_args(["separator"], [[]]).is_provided("separator")
        False
    """
    return CallArgs(_FakeCallContext(names, nodes, arg_names, api))  # type: ignore[arg-type]


class TypeComparison:
    """Compare two minted types by their parts, for a test that decodes what it encoded.

    An engine test encodes a type into a mypy type and decodes it again.
    The two objects differ, and the test must read their parts. Two tables
    compare by their canonical spelling, because one set of cells can
    arrive as two table objects.

    Example:
        >>> from astrolabe._hooks.expression.character import ELEMENTWISE
        >>> from astrolabe._hooks.expression.outcome import Axis
        >>> from astrolabe._hooks.expression.quadruple import ExpressionType
        >>> from astrolabe._hooks.expression.table import Table
        >>> one = Table.from_cells(("a",), {("Int64",): Axis("a")})
        >>> expr = ExpressionType(frozenset({"a"}), one, "a", ELEMENTWISE)
        >>> SAME.expressions(expr, expr)
        True
    """

    def expressions(self, one: ExpressionType, other: ExpressionType) -> bool:
        """Answer whether two expression types have the same needs, tables, name and character.

        Args:
            one: The type of one expression.
            other: The type of the other expression.

        Returns:
            True where every part agrees.
        """
        return (
            one.needs == other.needs
            and one.table.spelling() == other.table.spelling()
            and one.name == other.name
            and one.character == other.character
        )


SAME = TypeComparison()
