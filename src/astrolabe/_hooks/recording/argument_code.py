"""Write each argument of a recorded call as a part of its code.

The recorder writes a call as the user wrote it. An argument that is an
expression gives its own code, and a declared expression type gives a leaf of
the code. A literal gives its value, from the syntax of the call or from a
`Literal` type. A data type gives the constructor of polars that builds it, and
a date or a time gives the constructor of Python that builds it. Any other
value is a hole: the checker cannot know it, so the code keeps only its kind
(`HoleValues.KINDS`).

This module reads the syntax of the call and the types that mypy gives its
parts, so it belongs to the plugin. It gives each part as a node of a Python
syntax tree, which `CodeTree` turns into the canonical text of the code.
"""

from __future__ import annotations

import ast
import math
import tokenize
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import polars.datatypes.classes as polars_classes
from mypy.nodes import (
    ARG_NAMED,
    ARG_STAR,
    ARG_STAR2,
    BytesExpr,
    CallExpr,
    DictExpr,
    FloatExpr,
    IntExpr,
    ListExpr,
    NameExpr,
    OpExpr,
    RefExpr,
    SetExpr,
    StrExpr,
    TupleExpr,
    UnaryExpr,
)
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    LiteralType,
    NoneType,
    Overloaded,
    TypeType,
    UnionType,
    get_proper_type,
)

from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.calls.constant_fold import ConstantFold
from astrolabe._hooks.calls.dtype_expr import DtypeExpr
from astrolabe._hooks.calls.temporal_value import TemporalValueReader
from astrolabe._hooks.deferred.code_reader import HoleValues
from astrolabe._hooks.recording.deferred_types import DeferredReader
from astrolabe._rules.dtypes import Rendering, RenderingError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mypy.nodes import Expression
    from mypy.types import ProperType
    from mypy.types import Type as MypyType

    from astrolabe._hooks.deferred.code import CodeTree


class ArgumentCode:
    """Write the arguments of one recorded call as parts of its code.

    `types` gives the type of each argument node that mypy checked with the
    call, by the identity of the node. A node inside a list or a dict, which
    mypy checks on its own, takes its type from the checker.

    Example:
        >>> from astrolabe._hooks.deferred.code import CodeTree
        >>> writer = ArgumentCode(None, CodeTree(), {})
        >>> ast.unparse(writer.part(IntExpr(3)))
        '3'
    """

    DATA_TYPES: ClassVar[str] = pl.DataType.__module__
    DATA_TYPE: ClassVar[str] = f"{pl.DataType.__module__}.{pl.DataType.__name__}"
    NAMED: ClassVar[dict[str, bool | None]] = {"True": True, "False": False, "None": None}
    TEMPORAL: ClassVar[dict[str, str]] = {
        "datetime.date": "date",
        "datetime.datetime": "datetime",
        "datetime.time": "time",
        "datetime.timedelta": "timedelta",
        "decimal.Decimal": "Decimal",
    }
    PLAIN_EXPRESSIONS: ClassVar[frozenset[str]] = frozenset({
        "polars.expr.expr.Expr",
        "polars.expr.whenthen.When",
        "polars.expr.whenthen.ChainedWhen",
    })
    HOLE_KINDS: ClassVar[dict[str, str]] = {
        "builtins.bool": "bool",
        "builtins.int": "int",
        "builtins.float": "float",
        "builtins.str": "str",
        "datetime.datetime": "datetime",
        "datetime.date": "date",
        "datetime.time": "time",
        "datetime.timedelta": "timedelta",
        "decimal.Decimal": "decimal",
    }
    FLOAT: ClassVar[str] = "builtins.float"
    CONSTANTS: ClassVar[dict[str, str]] = {
        "datetime.UTC": "UTC",
        "datetime.timezone.utc": "UTC",
    }
    FLOATS: ClassVar[dict[str, str]] = {"math.inf": "inf", "math.nan": "nan"}

    def __init__(
        self, api: object, tree: CodeTree[MypyType], types: Mapping[int, MypyType]
    ) -> None:
        """Keep the checker, the tree of the code and the types of the argument nodes.

        Args:
            api: The checker, which gives the type of a node that the call did
                not check with it.
            tree: The tree of the code, which numbers the declared types.
            types: The type of each argument node, by the identity of the node.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> ArgumentCode(None, CodeTree(), {})._types
            {}
        """
        self._api = api
        self._tree = tree
        self._types = types

    def call(self, function: ast.expr, call: CallExpr) -> ast.Call:
        """Give the call of a function with the arguments of a call of the user.

        Args:
            function: The function, as a part of the code.
            call: The call that the user wrote.

        Returns:
            The call, as a part of the code.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> arguments = [IntExpr(1), StrExpr("a")]
            >>> call = CallExpr(NameExpr("f"), arguments, [ARG_POS, ARG_NAMED], [None, "b"])
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> ast.unparse(writer.call(ast.Name("f", ast.Load()), call))
            "f(1, b='a')"
        """
        positional: list[ast.expr] = []
        keywords: list[ast.keyword] = []
        arguments = zip(call.args, call.arg_kinds, call.arg_names, strict=True)
        for node, kind, name in arguments:
            part = self.part(node)
            if kind == ARG_STAR:
                positional.append(ast.Starred(part, ast.Load()))
            elif kind == ARG_STAR2:
                keywords.append(ast.keyword(None, part))
            elif kind == ARG_NAMED:
                keywords.append(ast.keyword(name, part))
            else:
                positional.append(part)
        return ast.Call(function, positional, keywords)

    def part(self, node: Expression) -> ast.expr:
        """Give the part of the code that an argument writes.

        Args:
            node: The argument, as the user wrote it.

        Returns:
            The part.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> ast.unparse(writer.part(ListExpr([StrExpr("a"), IntExpr(2)])))
            "['a', 2]"
        """
        typ = self._type_of(node)
        expression = None if typ is None else self.expression(typ)
        if expression is not None:
            return expression
        written = self._written(node)
        if written is not None:
            return written
        valued = None if typ is None else self._valued(typ)
        if valued is not None:
            return valued
        built = self._temporal(node) or self._data_class(typ) or self._data_type(node, typ)
        if built is not None:
            return built
        return self._hole(typ, node)

    def _type_of(self, node: Expression) -> MypyType | None:
        """Give the type of an argument node.

        Args:
            node: The node.

        Returns:
            Its type, or `None` where the checker gives none.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> ArgumentCode(None, CodeTree(), {})._type_of(IntExpr(1)) is None
            True
        """
        known = self._types.get(id(node))
        if known is not None:
            return known
        reader = getattr(self._api, "get_expression_type", None)
        if reader is None:
            return None
        typ: MypyType = reader(node)
        return typ

    def expression(self, typ: MypyType) -> ast.expr | None:
        """Give the part of an expression: its code, or the leaf of its declared type.

        The checker writes code that parses. A type that the user writes by
        hand can hold any text, and code that does not parse gives no part.

        Args:
            typ: The type of the argument.

        Returns:
            The part, or `None` where the argument is not a typed expression,
            or where its code does not parse.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code, CodeTree
            >>> from astrolabe._hooks.recording.deferred_types import DeferredMinter
            >>> from astrolabe.expr import TypedExpr
            >>> fixture = new_type_fixture()
            >>> ArgumentCode(None, CodeTree(), {}).expression(fixture.anyt) is None
            True
            >>> minter = DeferredMinter(new_marker_minter(fixture))
            >>> by_hand = minter.minted(TypedExpr, Code("pl.col("))
            >>> ArgumentCode(None, CodeTree(), {}).expression(by_hand) is None
            True
        """
        reader = DeferredReader()
        code = reader.code(typ)
        if code is not None:
            try:
                return self._tree.part(code)
            except (SyntaxError, tokenize.TokenError):
                return None
        if reader.declared(typ):
            return self._tree.leaf(get_proper_type(typ))
        return None

    def _written(self, node: Expression) -> ast.expr | None:
        """Give the part of a literal that the call writes: a constant, a container or a zone.

        Args:
            node: The node.

        Returns:
            The part, or `None` where the node is not a literal.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> ast.unparse(writer._written(UnaryExpr("-", IntExpr(1))))
            '-1'
        """
        constant = self._constant(node)
        if constant is not None:
            return constant
        return self._container(node) or self._zone(node)

    def _zone(self, node: Expression) -> ast.expr | None:
        """Give the part of a time zone: `UTC`, or a `ZoneInfo` of a literal key.

        Args:
            node: The node.

        Returns:
            The part, or `None` where the node is not a zone that the checker reads.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> from mypy.nodes import MemberExpr
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> utc = MemberExpr(named_reference_node("datetime.timezone"), "utc")
            >>> ast.unparse(writer._zone(utc))
            'UTC'
            >>> paris = named_call_node("zoneinfo.ZoneInfo", [str_literal_node("Europe/Paris")])
            >>> ast.unparse(writer._zone(paris))
            "ZoneInfo('Europe/Paris')"
        """
        zone = TemporalValueReader().zone(node)
        if zone is None:
            return None
        if not isinstance(node, CallExpr):
            return ast.Name("UTC", ast.Load())
        return ast.Call(ast.Name("ZoneInfo", ast.Load()), [ast.Constant(zone)], [])

    def _constant(self, node: Expression) -> ast.expr | None:
        """Give the part of a constant, or of arithmetic over constants, that the call writes.

        Python computes such arithmetic before polars sees the value, as in
        `pl.col("u") + 2**10`, so the code holds the value, which
        `ConstantFold` computes. A value with too many bits to compute, or an
        operation that Python refuses, as `1 // 0`, gives no constant.

        Args:
            node: The node.

        Returns:
            The part, or `None` where the node is not a constant.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> ast.unparse(writer._constant(OpExpr("**", IntExpr(2), IntExpr(10))))
            '1024'
            >>> writer._constant(OpExpr("//", IntExpr(1), IntExpr(0))) is None
            True
            >>> writer._constant(OpExpr("+", IntExpr(2), NameExpr("x"))) is None
            True
            >>> writer._constant(BytesExpr("x")).value
            b'x'
        """
        if isinstance(node, IntExpr | FloatExpr | StrExpr):
            return self._scalar(node.value)
        if isinstance(node, BytesExpr):
            # mypy keeps a bytes literal as the text that `repr` gives it, with its escapes.
            text = node.value.encode("ascii").decode("unicode_escape")
            return ast.Constant(text.encode("latin-1"))
        if isinstance(node, RefExpr) and node.fullname in self.CONSTANTS:
            return ast.Name(self.CONSTANTS[node.fullname], ast.Load())
        if isinstance(node, NameExpr) and node.name in self.NAMED:
            return ast.Constant(self.NAMED[node.name])
        written = self._float(node)
        if written is not None or not isinstance(node, UnaryExpr | OpExpr):
            return written
        value = ConstantFold().value(node)
        return None if value is None else self._scalar(value)

    def _scalar(self, value: int | float | str) -> ast.expr:
        """Give the part of a number or a text, with `float` of a text for an infinity or a NaN.

        `ast.unparse` writes an infinity as `1e309`. The tree of polars writes
        that float as null, so the code holds `float('inf')` instead, which the
        reader of the code reads as a stand-in that the tree carries.

        Args:
            value: The value, as the call gives it or as `ConstantFold` computes it.

        Returns:
            The part.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> ast.unparse(writer._scalar(1e308 * 10)), ast.unparse(writer._scalar(1.5))
            ("float('inf')", '1.5')
        """
        if isinstance(value, float) and not math.isfinite(value):
            return ast.Call(ast.Name("float", ast.Load()), [ast.Constant(str(value))], [])
        return ast.Constant(value)

    def _float(self, node: Expression) -> ast.expr | None:
        """Give the part of a float that the call names: an infinity, a NaN, or `float` of a text.

        Python writes an infinity and a NaN as `float("inf")`, `float("nan")`,
        `math.inf` or `math.nan`, and the code holds them so. A sign before
        one of them stays in the code.

        Args:
            node: The node.

        Returns:
            The part, or `None` where the node names no such float.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> infinity = named_call_node("builtins.float", [str_literal_node("inf")])
            >>> ast.unparse(writer._float(UnaryExpr("-", infinity)))
            "-float('inf')"
        """
        if isinstance(node, UnaryExpr) and node.op in {"-", "+"}:
            operand = self._float(node.expr)
            operator = ast.USub() if node.op == "-" else ast.UAdd()
            return None if operand is None else ast.UnaryOp(operator, operand)
        text = self.FLOATS.get(node.fullname) if isinstance(node, RefExpr) else None
        if self._float_call(node):
            text = checked_cast(checked_cast(node, CallExpr).args[0], StrExpr).value
        if text is None:
            return None
        return ast.Call(ast.Name("float", ast.Load()), [ast.Constant(text)], [])

    def _float_call(self, node: Expression) -> bool:
        """Say whether a node is a call of `float` over one text that reads as a float.

        Args:
            node: The node.

        Returns:
            Whether it is.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> writer._float_call(named_call_node("builtins.float", [str_literal_node("x")]))
            False
            >>> writer._float_call(named_call_node("builtins.float", [int_literal_node(1)]))
            False
        """
        if not isinstance(node, CallExpr) or not isinstance(node.callee, RefExpr):
            return False
        if node.callee.fullname != "builtins.float" or len(node.args) != 1:
            return False
        text = node.args[0]
        if not isinstance(text, StrExpr):
            return False
        try:
            float(text.value)
        except ValueError:
            return False
        return True

    def _container(self, node: Expression) -> ast.expr | None:
        """Give the part of a list, a tuple, a set or a dict that the call writes.

        Args:
            node: The node.

        Returns:
            The part, or `None` where the node is not a container.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> ast.unparse(writer._container(DictExpr([(StrExpr("a"), IntExpr(1))])))
            "{'a': 1}"
        """
        if isinstance(node, DictExpr):
            return self._dict(node)
        if not isinstance(node, ListExpr | TupleExpr | SetExpr):
            return None
        items = [self.part(item) for item in node.items]
        if isinstance(node, SetExpr):
            return ast.Set(items)
        if isinstance(node, TupleExpr):
            return ast.Tuple(items, ast.Load())
        return ast.List(items, ast.Load())

    def _dict(self, node: DictExpr) -> ast.Dict:
        """Give the part of a dict that the call writes.

        A key of `None` stands for a `**` entry that unpacks another dict.

        Args:
            node: The node of the dict.

        Returns:
            The part.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> ast.unparse(writer._dict(DictExpr([(None, NameExpr("d"))])))
            "{**hole('value')}"
        """
        keys = [None if key is None else self.part(key) for key, _ in node.items]
        values = [self.part(value) for _, value in node.items]
        return ast.Dict(keys, values)

    def _valued(self, typ: MypyType) -> ast.expr | None:
        """Give the part of an argument whose type holds its value, as `Literal[3]` does.

        A union of literal values, as `Literal["average", "dense"]`, holds
        one value of a known set, so the part is a hole that lists the set.

        Args:
            typ: The type of the argument.

        Returns:
            The constant or the hole, or `None` where the type holds no value.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> fixture = new_type_fixture()
            >>> literal = LiteralType(3, fixture.a)
            >>> ast.unparse(ArgumentCode(None, CodeTree(), {})._valued(literal))
            '3'
        """
        proper = get_proper_type(typ)
        if isinstance(proper, NoneType):
            return ast.Constant(None)
        if isinstance(proper, UnionType):
            return self._one_of(proper)
        value = self._literal(proper)
        return None if value is None else ast.Constant(value)

    def _one_of(self, union: UnionType) -> ast.expr | None:
        """Give the hole of a value whose type is a union of literal values, with the values.

        Args:
            union: The type of the value.

        Returns:
            The hole, or `None` where a member of the union is not a literal value.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> fixture = new_type_fixture()
            >>> union = UnionType([LiteralType("average", fixture.a), LiteralType(2, fixture.a)])
            >>> ast.unparse(ArgumentCode(None, CodeTree(), {})._one_of(union))
            "hole('one of', 'average', 2)"
        """
        values: list[int | str | float] = []
        for item in union.items:
            value = self._literal(get_proper_type(item))
            if value is None:
                return None
            values.append(value)
        return self.hole(HoleValues.ONE_OF, *values)

    def _literal(self, proper: ProperType) -> int | str | float | None:
        """Give the value that a literal type holds.

        The literal of a member of an `Enum` holds the name of the member,
        which is not the value that polars reads, so it gives no value.

        Args:
            proper: The type.

        Returns:
            The value, or `None` where the type holds no value.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> fixture = new_type_fixture()
            >>> ArgumentCode(None, CodeTree(), {})._literal(LiteralType("a", fixture.a))
            'a'
        """
        if isinstance(proper, Instance) and proper.last_known_value is not None:
            proper = proper.last_known_value
        if not isinstance(proper, LiteralType) or proper.is_enum_literal():
            return None
        value = proper.value
        return value if isinstance(value, int | str | float) else None

    def _data_class(self, typ: MypyType | None) -> ast.expr | None:
        """Give the part of an argument that is a data type class of polars, such as `pl.Datetime`.

        A class and its default instance differ in a selector:
        `cs.by_dtype(pl.Datetime)` picks every `Datetime`, and
        `cs.by_dtype(pl.Datetime())` picks one. So the part keeps the class,
        and polars reads it as it reads the class that the user wrote.

        Args:
            typ: The type of the argument.

        Returns:
            The part, or `None` where the argument is not such a class.

        Example:
            >>> from mypy.types import TypeType
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> ast.unparse(writer._data_class(TypeType(int8_dtype(new_type_fixture()))))
            'pl.Int8'
            >>> writer._data_class(int8_dtype(new_type_fixture())) is None
            True

        A variable of the type `type[...]` of a family holds a class of the
        family that the checker does not know, so its part is a hole.
        """
        proper = None if typ is None else get_proper_type(typ)
        variable = isinstance(proper, TypeType) and isinstance(proper.item, Instance)
        if isinstance(proper, TypeType) and isinstance(proper.item, Instance):
            fullname = proper.item.type.fullname
        elif isinstance(proper, CallableType | Overloaded) and proper.is_type_obj():
            fullname = proper.type_object().fullname
        else:
            return None
        module, _, name = fullname.rpartition(".")
        if module != self.DATA_TYPES:
            return None
        if variable and self._family(name):
            return self.hole(f"{HoleValues.DTYPE}{name}")
        return ast.Attribute(ast.Name("pl", ast.Load()), name, ast.Load())

    def _family(self, name: str) -> bool:
        """Say whether a class of polars is a family, which other classes of polars subclass.

        A value of the type `type[pl.NumericType]` can be any class of the
        family, so the code holds a hole of the family.

        Args:
            name: The name of the class.

        Returns:
            Whether it is.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> writer._family("NumericType"), writer._family("Int8")
            (True, False)
            >>> writer._family("Unchecked")
            False
        """
        known = getattr(polars_classes, name, None)
        if not isinstance(known, type):
            return False
        return any(child.__module__.startswith("polars.") for child in known.__subclasses__())

    def _data_type(self, node: Expression, typ: MypyType | None) -> ast.expr | None:
        """Give the part of an argument that names a data type of polars, as its constructor.

        The checker reads the data type from the type that mypy gives the
        argument, so a variable that holds a data type works too. Polars
        writes the data type as the call of its class, for example
        `Datetime(time_unit='ms', time_zone=None)`, and the part puts `pl.`
        before each class. A data type whose parameters the type does not
        give, as a variable of the type `pl.Datetime`, is a hole of that
        rendering.

        Args:
            node: The node.
            typ: The type of the node. Only a data type of polars, or a
                type that mypy does not give, names a data type: the
                `Decimal` of Python is not the `Decimal` of polars.

        Returns:
            The part, or `None` where the node names no data type.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> ArgumentCode(None, CodeTree(), {})._data_type(IntExpr(1), None) is None
            True
        """
        proper = None if typ is None else get_proper_type(typ)
        foreign = isinstance(proper, Instance) and not proper.type.has_base(self.DATA_TYPE)
        if self._api is None or foreign:
            return None
        rendering = DtypeExpr().rendering(self._api, node)
        if rendering is None:
            return None
        try:
            Rendering().parse(rendering)
        except RenderingError:
            return self.hole(f"{HoleValues.DTYPE}{rendering}")
        return DataTypeCode().of(rendering)

    def _temporal(self, node: Expression) -> ast.expr | None:
        """Give the part of a date, a time, a duration or a decimal that the call builds.

        Args:
            node: The node.

        Returns:
            The call of the constructor, or `None` where the node is no such call.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> callee = NameExpr("date")
            >>> callee.fullname = "datetime.date"
            >>> numbers = [IntExpr(2020), IntExpr(1), IntExpr(2)]
            >>> call = CallExpr(callee, numbers, [ARG_POS] * 3, [None] * 3)
            >>> ast.unparse(ArgumentCode(None, CodeTree(), {})._temporal(call))
            'date(2020, 1, 2)'
        """
        if not isinstance(node, CallExpr) or not isinstance(node.callee, RefExpr):
            return None
        name = self.TEMPORAL.get(node.callee.fullname)
        if name is None:
            return None
        return self.call(ast.Name(name, ast.Load()), node)

    def hole(self, kind: str, *values: float | str) -> ast.expr:
        """Give the hole of a value of one kind.

        Args:
            kind: One of `HoleValues.KINDS`.
            values: The values of a hole of a value from a known set.

        Returns:
            The call `hole('<kind>')`, with the values after the kind.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> ast.unparse(ArgumentCode(None, CodeTree(), {}).hole("expr"))
            "hole('expr')"
        """
        arguments: list[ast.expr] = [ast.Constant(kind)]
        arguments.extend(ast.Constant(value) for value in values)
        return ast.Call(ast.Name("hole", ast.Load()), arguments, [])

    def _hole(self, typ: MypyType | None, node: Expression) -> ast.expr:
        """Give the hole of an argument that the checker cannot know, with its kind.

        mypy gives the type `Any` to some arithmetic, as `2**n`. Python
        computes a number there before polars sees it, so the hole is of
        any value, and not of a value that the user can give a type. The type
        `date` also covers a `datetime`, except where a call of the class
        `date` builds the value, as `date.today()`: that hole is a `day`.

        Args:
            typ: The type of the argument.
            node: The argument, as the user wrote it.

        Returns:
            The call `hole('<kind>')`.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> writer = ArgumentCode(None, CodeTree(), {})
            >>> ast.unparse(writer._hole(None, NameExpr("x")))
            "hole('value')"
            >>> any_type = new_type_fixture().anyt
            >>> ast.unparse(writer._hole(any_type, NameExpr("x")))
            "hole('any')"
            >>> ast.unparse(writer._hole(any_type, OpExpr("**", IntExpr(2), NameExpr("n"))))
            "hole('value')"
        """
        kind = self._kind(typ)
        if kind == "any" and isinstance(node, OpExpr | UnaryExpr):
            kind = "value"
        if kind == "date" and TemporalValueReader().exact_date(node):
            kind = "day"
        return self.hole(kind)

    def _kind(self, typ: MypyType | None) -> str:
        """Give the kind of a hole from the type of its value.

        mypy accepts an `int` where the type is `float`, so a value of the
        type `float` is a `number`: an `int` or a `float`. A value of a
        subclass of `float` is always a `float`. A value of the type `Any`
        is a hole of its own kind, because the user can give it a type.

        Args:
            typ: The type of the value.

        Returns:
            One of `HoleValues.KINDS`.

        Example:
            >>> from astrolabe._hooks.deferred.code import CodeTree
            >>> fixture = new_type_fixture()
            >>> ArgumentCode(None, CodeTree(), {})._kind(fixture.callable(fixture.a, fixture.a))
            'function'
            >>> ArgumentCode(None, CodeTree(), {})._kind(fixture.anyt)
            'any'
        """
        proper = None if typ is None else get_proper_type(typ)
        if isinstance(proper, AnyType):
            return "any"
        if isinstance(proper, CallableType | Overloaded | TypeType):
            return "function"
        if not isinstance(proper, Instance):
            return "value"
        names = [info.fullname for info in proper.type.mro]
        if set(names) & self.PLAIN_EXPRESSIONS:
            return "expr"
        if names[0] == self.FLOAT:
            return "number"
        kinds = [self.HOLE_KINDS[name] for name in names if name in self.HOLE_KINDS]
        return kinds[0] if kinds else "value"


class DataTypeCode:
    """Turn the text of a data type of polars into the code that builds it.

    Example:
        >>> ast.unparse(DataTypeCode().of("List(Datetime(time_unit='ms', time_zone=None))"))
        "pl.List(pl.Datetime(time_unit='ms', time_zone=None))"
    """

    def of(self, rendering: str) -> ast.expr:
        """Give the code of a data type, with `pl.` before each class of polars.

        Args:
            rendering: The text of the data type, as polars writes it.

        Returns:
            The code.

        Example:
            >>> ast.unparse(DataTypeCode().of("Int64"))
            'pl.Int64'
        """
        parsed = ast.parse(rendering, mode="eval").body
        prefixed: ast.expr = _Prefixer().visit(parsed)
        return prefixed


class _Prefixer(ast.NodeTransformer):
    """Put `pl.` before each name of a class of polars.

    Example:
        >>> ast.unparse(_Prefixer().visit(ast.parse("Struct({'a': Int8})", mode="eval").body))
        "pl.Struct({'a': pl.Int8})"
    """

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Give the name of a class of polars under `pl`.

        Args:
            node: A name of the text.

        Returns:
            `pl.<name>`: polars writes each parameter of a data type as a
            plain value, so each name in the text is a class.

        Example:
            >>> ast.unparse(_Prefixer().visit(ast.Name("Date", ast.Load())))
            'pl.Date'
        """
        return ast.Attribute(ast.Name("pl", ast.Load()), node.id, ast.Load())
