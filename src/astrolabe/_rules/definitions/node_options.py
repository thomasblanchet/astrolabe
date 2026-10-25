"""Read the identity and the options of a node from the tree that polars serializes.

Polars serializes an expression as a tree in JSON, with `meta.serialize()`.
Each node of the tree has a kind, for example a function or a binary
operator. A function node also names its function. That pair is the identity
of the node, and one definition states the rules of one identity.

Calls of one node can differ by an option of the node, for example the target
type of a cast or the time unit of `dt.cast_time_unit`. Polars keeps such an
option in the function of the node, and its rules read it there. A definition
names the option by its place in the serialized node (`NodeOption`), and it
gives the option to the part of the vocabulary that reads it. So one
definition covers every value of the option. Code that evaluates a node first
reads the options of that node, as polars builds the function of a node from
its options.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import polars as pl

from astrolabe._hooks.checked import checked_cast
from astrolabe._rules.cast_pass import CastOption
from astrolabe._rules.dtypes import DType
from astrolabe._rules.failures import Undecided, UndecidedError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


@dataclass(frozen=True)
class NodeIdentity:
    """Name one node of polars by the root of its serialized tree: its shape and its function.

    `Expr.meta.serialize(format="json")` gives the tree of an expression. Its
    root has a shape, such as `Function`, `BinaryExpr` or `Agg`, and a
    function: the function of a `Function`, the operator of a `BinaryExpr`,
    or the aggregation of an `Agg`. A function of polars is a variant of one
    enum. Some variants are namespaces, such as `StringExpr` or `Pow`, and
    they hold a variant of an enum of functions of their own. So the function
    of an identity is the variant, and for a namespace also the variant in it:
    `sqrt` is `Pow.Sqrt`, and `dt.cast_time_unit("ms")` is
    `TemporalExpr.CastTimeUnit`. The rest of the root is the inputs and the
    options of the node. So `cum_sum(reverse=True)` is `CumSum`, as
    `cum_sum()` is, and a definition reads an option where it changes the
    rules (`NodeOption`).

    Example:
        >>> NodeIdentity.of({"Function": {"input": [], "function": {"Pow": "Sqrt"}}})
        NodeIdentity(shape='Function', function='Pow.Sqrt')
    """

    shape: str
    function: str

    NAMESPACES: ClassVar[frozenset[str]] = frozenset({
        "ArrayExpr",
        "BinaryExpr",
        "Bitwise",
        "Boolean",
        "Business",
        "Categorical",
        "Extension",
        "ListExpr",
        "Pow",
        "Range",
        "RollingExpr",
        "RollingExprBy",
        "StringExpr",
        "StructExpr",
        "TemporalExpr",
        "Trigonometry",
    })
    NAMED: ClassVar[dict[str, str]] = {"RollingExpr": "function", "RollingExprBy": "function_by"}

    @classmethod
    def of(cls, tree: Mapping[str, object]) -> NodeIdentity:
        """Read the identity of the root of a serialized tree.

        Args:
            tree: The tree, as `json.loads` gives it.

        Returns:
            The identity.

        Example:
            >>> NodeIdentity.of({"Agg": {"Sum": {"Column": "x"}}})
            NodeIdentity(shape='Agg', function='Sum')
            >>> NodeIdentity.of({"BinaryExpr": {"left": {}, "op": "Plus", "right": {}}}).function
            'Plus'
            >>> NodeIdentity.of({"Cast": {"expr": {}, "dtype": {}, "options": "Strict"}})
            NodeIdentity(shape='Cast', function='')
        """
        shape = next(iter(tree))
        body = tree[shape]
        if not isinstance(body, dict):
            return cls(shape, "")
        keys = {"Function": "function", "BinaryExpr": "op"}
        if shape in keys:
            return cls(shape, cls._name(body[keys[shape]]))
        if shape == "Agg":
            return cls(shape, cls._variant(body))
        return cls(shape, "")

    @classmethod
    def _name(cls, function: object) -> str:
        """Give the variant of a function, and for a namespace the variant in it, joined with a dot.

        A rolling function holds its variant under the field `function`, or
        `function_by` where a key orders its windows.

        Args:
            function: The function as the tree holds it: a name, or a map of
                one name to what the variant holds.

        Returns:
            The names, or an empty text where the part holds no name.

        Example:
            >>> NodeIdentity._name({"CumSum": {"reverse": False}})
            'CumSum'
            >>> NodeIdentity._name({"StringExpr": {"Contains": {"literal": False, "strict": True}}})
            'StringExpr.Contains'
            >>> NodeIdentity._name({"TemporalExpr": {"CastTimeUnit": "Milliseconds"}})
            'TemporalExpr.CastTimeUnit'
            >>> NodeIdentity._name({"RollingExpr": {"function": "Mean", "options": {}}})
            'RollingExpr.Mean'
            >>> NodeIdentity._name({"RollingExprBy": {"function_by": "MeanBy", "options": {}}})
            'RollingExprBy.MeanBy'
            >>> NodeIdentity._name({"Diff": "Ignore"})
            'Diff'
        """
        variant = cls._variant(function)
        if variant not in cls.NAMESPACES or not isinstance(function, dict):
            return variant
        inner = function[variant]
        if variant in cls.NAMED and isinstance(inner, dict):
            inner = inner.get(cls.NAMED[variant])
        return f"{variant}.{cls._variant(inner)}"

    @classmethod
    def _variant(cls, part: object) -> str:
        """Give the name of the variant of an enum, as polars serializes the variant.

        Polars writes a variant with no data as its name, and a variant with
        data as a map of its name to the data.

        Args:
            part: The variant.

        Returns:
            The name, or an empty text where the part is not a variant.

        Example:
            >>> NodeIdentity._variant("Abs"), NodeIdentity._variant({"Unique": False})
            ('Abs', 'Unique')
            >>> NodeIdentity._variant(None)
            ''
        """
        if isinstance(part, str):
            return part
        if isinstance(part, dict) and len(part) == 1:
            return str(next(iter(part)))
        return ""  # pragma: no cover -- polars names a variant by a text or a map of one key


class NoOptionError(LookupError):
    """Report a path that the serialized root of a node does not hold.

    A fault of a definition causes it: the definition names an option that
    its node does not have.

    Example:
        >>> str(NoOptionError(("dtype",)))
        "the node holds no option at ('dtype',)"
    """

    def __init__(self, path: Sequence[str | int]) -> None:  # pragma: no cover -- a guard raises it
        """Build the message.

        Args:
            path: The path of the option.

        Example:
            >>> NoOptionError(("function", 0)).args
            ("the node holds no option at ('function', 0)",)
        """
        super().__init__(f"the node holds no option at {tuple(path)!r}")


class UnreadOptionError(TypeError):
    """Report a rule that names an option of its node that nobody read.

    A fault of the code that evaluates a node causes it: that code must read
    the options of the node (`Definition.at`) before it calls a rule.

    Example:
        >>> str(UnreadOptionError("the target"))
        'the target is an option of the node: read the node first'
    """

    def __init__(self, part: str) -> None:  # pragma: no cover -- a guard raises it
        """Build the message.

        Args:
            part: The part of the rule that is an option of the node.

        Example:
            >>> UnreadOptionError("the kernel").args
            ('the kernel is an option of the node: read the node first',)
        """
        super().__init__(f"{part} is an option of the node: read the node first")


class NodeOptions:
    """Hold the serialized root of one node, so that a rule can read an option of that node.

    Polars serializes an expression as a tree of JSON, and the root of the tree
    holds the inputs of the node beside its options: for example, the target
    type and the strictness of a cast. Calls of one node can differ by an
    option, and some options change the rules. A definition names such an
    option by its path in the body of the root (`NodeOption`), and this class
    gives the part of the body at a path.

    Example:
        >>> cast = {"Cast": {"expr": {}, "dtype": {"Literal": "Int32"}, "options": "Strict"}}
        >>> NodeOptions(cast).value(("options",))
        'Strict'
    """

    def __init__(self, tree: Mapping[str, object]) -> None:
        """Keep the body of the root.

        Args:
            tree: The tree, as `json.loads` gives it.

        Example:
            >>> NodeOptions({"Function": {"function": "Abs"}}).value(("function",))
            'Abs'
        """
        self._body = tree[next(iter(tree))]

    def value(self, path: Sequence[str | int]) -> object:
        """Give the part of the body at a path.

        Args:
            path: The keys of the maps and the indices of the lists that lead to
                the part, from the body of the root.

        Returns:
            The part, as `json.loads` gives it.

        Raises:
            NoOptionError: The body holds nothing at the path.

        Example:
            >>> strptime = {"Strptime": ["Date", {"format": None}]}
            >>> node = NodeOptions({"Function": {"function": {"StringExpr": strptime}}})
            >>> node.value(("function", "StringExpr", "Strptime", 1, "format")) is None
            True
            >>> node.value(("function", "Abs"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.definitions.node_options.NoOptionError: the node holds no option at ...
        """
        part: Any = self._body
        for step in path:
            mapped = isinstance(part, dict) and step in part
            listed = isinstance(part, list) and isinstance(step, int) and 0 <= step < len(part)
            if not mapped and not listed:  # pragma: no cover -- polars holds each option read
                raise NoOptionError(path)
            part = part[step]
        return part


class SerializedDType:
    """Read a data type in the form that polars serializes it, as a `DType`.

    Polars writes a data type with no parameter as its name, for example
    `"Int8"`. It writes a data type with parameters as a map of its name to
    them, for example `{"Decimal": [10, 2]}` or `{"List": "Int8"}`. A time
    unit is the name of its variant, such as `Milliseconds`, and a zone is
    `{"inner": "UTC"}`. The categories of an `Enum` are an Arrow stream of
    text, which polars reads back.

    Example:
        >>> SerializedDType().of({"Datetime": ["Milliseconds", {"inner": "UTC"}]})
        Datetime(time_unit='ms', time_zone='UTC')
        >>> SerializedDType().of({"Array": [{"List": "Date"}, 2]})
        Array(List(Date), shape=(2,))
    """

    UNITS: ClassVar[dict[str, str]] = {
        "Nanoseconds": "ns",
        "Microseconds": "us",
        "Milliseconds": "ms",
    }

    def of(self, part: object) -> DType:
        """Read a data type.

        Args:
            part: The data type, as `json.loads` gives it.

        Returns:
            The data type.

        Raises:
            TypeError: The part is not a data type.
            UndecidedError: This module does not read the kind of the data type.

        Example:
            >>> SerializedDType().of({"Struct": [{"name": "a", "dtype": {"Decimal": [10, 2]}}]})
            Struct({'a': Decimal(precision=10, scale=2)})
            >>> SerializedDType().of({"Extension": {}})
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not translated: the data type Extension
            >>> SerializedDType().of(1)
            Traceback (most recent call last):
            ...
            TypeError: no data type in 1
        """
        if isinstance(part, str):
            return DType(part)
        if (  # pragma: no cover -- polars writes a data type here
            not isinstance(part, dict) or len(part) != 1
        ):
            message = f"no data type in {part!r}"
            raise TypeError(message)
        name, data = next(iter(part.items()))
        readers: dict[str, Callable[[Any], DType]] = {
            "Decimal": lambda d: DType("Decimal", (d[0], d[1])),
            "Datetime": lambda d: DType("Datetime", (self.unit(d[0]), self._zone(d[1]))),
            "Duration": lambda d: DType("Duration", (self.unit(d),)),
            "List": lambda d: DType("List", (self.of(d),)),
            "Array": lambda d: DType("Array", (self.of(d[0]), d[1])),
            "Struct": lambda d: DType("Struct", (tuple(self._field(field) for field in d),)),
            "Enum": lambda d: DType("Enum", (self._categories(d["strings"]),)),
            "Categorical": lambda _: DType("Categorical"),
            "Object": lambda _: DType("Object"),
            "Unknown": lambda _: DType.unknown("any"),
        }
        if name not in readers:  # pragma: no cover -- polars writes no other such type
            raise UndecidedError(Undecided.NOT_TRANSLATED, f"the data type {name}")
        return readers[name](data)

    def unit(self, part: object) -> str:
        """Read a time unit.

        Args:
            part: The name of the unit, for example `Milliseconds`.

        Returns:
            The unit: `ns`, `us` or `ms`.

        Raises:
            TypeError: The part is not a time unit.

        Example:
            >>> SerializedDType().unit("Nanoseconds")
            'ns'
            >>> SerializedDType().unit(1)
            Traceback (most recent call last):
            ...
            TypeError: no time unit in 1
        """
        if (  # pragma: no cover -- polars writes a time unit here
            not isinstance(part, str) or part not in self.UNITS
        ):
            message = f"no time unit in {part!r}"
            raise TypeError(message)
        return self.UNITS[part]

    def _zone(self, part: Any) -> str | None:
        """Read a time zone, or its absence.

        Args:
            part: The zone, as `{"inner": name}`, or `None`.

        Returns:
            The name of the zone, or `None`.

        Example:
            >>> SerializedDType()._zone({"inner": "Europe/Paris"})
            'Europe/Paris'
        """
        return None if part is None else str(part["inner"])

    def _field(self, part: Any) -> tuple[str, DType]:
        """Read one field of a `Struct`.

        Args:
            part: The field, as `{"name": name, "dtype": data type}`.

        Returns:
            The name and the type of the field.

        Example:
            >>> SerializedDType()._field({"name": "a", "dtype": "Int8"})
            ('a', Int8)
        """
        return str(part["name"]), self.of(part["dtype"])

    def _categories(self, part: Any) -> tuple[str, ...]:
        """Read the categories of an `Enum` from the Arrow stream that polars writes.

        Args:
            part: The bytes of the stream, as a list of integers.

        Returns:
            The categories, in order.

        Example:
            >>> import json
            >>> text = pl.col("x").cast(pl.Enum(["a", "b"])).meta.serialize(format="json")
            >>> tree = json.loads(text)
            >>> SerializedDType()._categories(tree["Cast"]["dtype"]["Literal"]["Enum"]["strings"])
            ('a', 'b')
        """
        return tuple(str(category) for category in self.stream(part).to_list())

    def stream(self, part: Any) -> pl.Series:
        """Read the series that polars writes as an Arrow stream, in a data type or a literal.

        Args:
            part: The bytes of the stream, as a list of integers.

        Returns:
            The series.

        Example:
            >>> import json
            >>> tree = json.loads(pl.lit([1, 2]).meta.serialize(format="json"))
            >>> SerializedDType().stream(tree["Literal"]["Scalar"]["List"]).dtype
            Int64
        """
        return pl.read_ipc_stream(io.BytesIO(bytes(part))).to_series()


class TemporalOptions:
    """Read the options of `pl.datetime` and `pl.duration` as the type that each node builds.

    Example:
        >>> TemporalOptions().duration("Microseconds")
        Duration(time_unit='us')
    """

    def datetime(self, part: object) -> DType:
        """Read the type that `pl.datetime` builds, from its unit and its zone.

        Args:
            part: The options, a map of `time_unit` and `time_zone`.

        Returns:
            The `Datetime`.

        Example:
            >>> TemporalOptions().datetime({"time_unit": "Milliseconds", "time_zone": None})
            Datetime(time_unit='ms', time_zone=None)
        """
        options = part if isinstance(part, dict) else {}
        unit = OptionReader().unit(options.get("time_unit"))
        return DType("Datetime", (unit, OptionReader().zone(options.get("time_zone"))))

    def duration(self, part: object) -> DType:
        """Read the type that `pl.duration` builds, from its unit.

        Args:
            part: The unit.

        Returns:
            The `Duration`.

        Example:
            >>> TemporalOptions().duration("Nanoseconds")
            Duration(time_unit='ns')
        """
        return DType("Duration", (OptionReader().unit(part),))


class OptionReader:
    """Read one option of a node, as polars serializes it, as a value of the vocabulary.

    Each method reads one kind of option: a data type, a time unit, a time
    zone, the option of a cast, a text, a flag, a number, or a list of numbers,
    texts or flags. A part of the wrong form is a fault of the definition that
    names it, or a new form of polars.

    Example:
        >>> OptionReader().cast_option("NonStrict")
        <CastOption.NON_STRICT: 1>
    """

    CAST_OPTIONS: ClassVar[dict[str, CastOption]] = {
        "Strict": CastOption.STRICT,
        "NonStrict": CastOption.NON_STRICT,
        "Overflowing": CastOption.OVERFLOWING,
    }

    def names(self, part: object) -> tuple[str, ...]:
        """Read the output name of each input of a node, as `meta.output_name()` gives it.

        Polars names the field of a struct after its input: a column by its
        name, an alias by the alias, and a literal `literal`.

        Args:
            part: The list of the trees of the inputs.

        Returns:
            The names. Polars raises `ComputeError` where an input has no
            output name, as `pl.all()`.

        Example:
            >>> OptionReader().names([{"Column": "a"}, {"Literal": {"Dyn": {"Int": 1}}}])
            ('a', 'literal')
            >>> OptionReader().names([{"KeepName": {"Alias": [{"Column": "a"}, "b"]}}])
            ('a',)
        """
        trees = part if isinstance(part, list) else [part]
        names: list[str] = []
        for tree in trees:
            text = io.StringIO(json.dumps(self._kept(tree)))
            names.append(pl.Expr.deserialize(text, format="json").meta.output_name())
        return tuple(names)

    def _kept(self, node: object) -> object:
        """Give a tree where each `name.keep` is an alias to the first column that its input reads.

        Polars asks the tree for a name, but the tree refuses where it holds
        `name.keep` or a field of a struct, as `pl.field("a")`: polars
        resolves those nodes only when it expands the expression at a frame.
        The expansion names `name.keep` after the first column that its input
        reads, in the order of `meta.root_names()`, and a field after its name,
        as a column of that name.

        Args:
            node: The serialized tree.

        Returns:
            The new tree. A `name.keep` whose input reads no column stays as
            it is, because polars refuses it.

        Example:
            >>> OptionReader()._kept({"KeepName": {"Column": "a"}})
            {'Alias': [{'Column': 'a'}, 'a']}
            >>> OptionReader()._kept({"Field": ["a"]})
            {'Column': 'a'}
        """
        if isinstance(node, list):
            return [self._kept(value) for value in node]
        if not isinstance(node, dict):
            return node
        kept = {key: self._kept(value) for key, value in node.items()}
        field = kept.get("Field")
        if isinstance(field, list) and len(field) == 1:
            return {"Column": field[0]}
        if set(kept) != {"KeepName"}:
            return kept
        text = io.StringIO(json.dumps(kept["KeepName"]))
        roots = pl.Expr.deserialize(text, format="json").meta.root_names()
        return {"Alias": [kept["KeepName"], roots[0]]} if roots else kept

    def first_name(self, part: object) -> str:
        """Read the output name of the first input of a node.

        Args:
            part: The list of the trees of the inputs.

        Returns:
            The name.

        Example:
            >>> OptionReader().first_name([{"Column": "a"}])
            'a'
        """
        return self.names(part)[0]

    def dtype_or_none(self, part: object) -> DType | None:
        """Read a data type that an option can hold, or `None` where the option is null.

        Args:
            part: The option.

        Returns:
            The data type, or `None`.

        Example:
            >>> OptionReader().dtype_or_none(None) is None
            True
        """
        return None if part is None else self.dtype(part)

    def dtype(self, part: object) -> DType:
        """Read a data type that a node holds as an option, such as the target of a cast.

        Polars writes it as `{"Literal": data type}`. It can also hold a data
        type that it computes from an expression, which this module does not
        read.

        Args:
            part: The option.

        Returns:
            The data type.

        Raises:
            UndecidedError: Polars computes the data type from an expression.

        Example:
            >>> OptionReader().dtype({"Literal": {"Duration": "Milliseconds"}})
            Duration(time_unit='ms')
            >>> OptionReader().dtype({"OfExpr": {}})
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not translated: a data type of an expression
        """
        if not isinstance(part, dict) or set(part) != {"Literal"}:
            raise UndecidedError(Undecided.NOT_TRANSLATED, "a data type of an expression")
        return SerializedDType().of(part["Literal"])

    def unit(self, part: object) -> str:
        """Read a time unit.

        Args:
            part: The option, for example `Milliseconds`.

        Returns:
            The unit: `ns`, `us` or `ms`.

        Example:
            >>> OptionReader().unit("Microseconds")
            'us'
        """
        return SerializedDType().unit(part)

    def unit_or_none(self, part: object) -> str | None:
        """Read a time unit, or its absence, such as the unit of `datetime_range`.

        Args:
            part: The option, or null.

        Returns:
            The unit, or `None`.

        Example:
            >>> OptionReader().unit_or_none(None), OptionReader().unit_or_none("Nanoseconds")
            (None, 'ns')
        """
        return None if part is None else self.unit(part)

    def cast_option(self, part: object) -> CastOption:
        """Read the option of a cast: strict, not strict, or overflowing.

        Args:
            part: The option, for example `Strict`.

        Returns:
            The option.

        Raises:
            TypeError: The part is not the option of a cast.

        Example:
            >>> OptionReader().cast_option("Overflowing")
            <CastOption.OVERFLOWING: 2>
            >>> OptionReader().cast_option("Loose")
            Traceback (most recent call last):
            ...
            TypeError: no option of a cast in 'Loose'
        """
        if (  # pragma: no cover -- polars writes an option of a cast here
            not isinstance(part, str) or part not in self.CAST_OPTIONS
        ):
            message = f"no option of a cast in {part!r}"
            raise TypeError(message)
        return self.CAST_OPTIONS[part]

    def text(self, part: object) -> str | None:
        """Read a text, such as a name or a format, or its absence.

        Args:
            part: The option.

        Returns:
            The text, or `None`.

        Raises:
            TypeError: The part is not a text.

        Example:
            >>> OptionReader().text("%Y"), OptionReader().text(None)
            ('%Y', None)
            >>> OptionReader().text(1)
            Traceback (most recent call last):
            ...
            TypeError: no text in 1
        """
        if part is not None and not isinstance(part, str):  # pragma: no cover -- polars writes text
            message = f"no text in {part!r}"
            raise TypeError(message)
        return part

    def zone(self, part: object) -> str | None:
        """Read a time zone, which polars writes as `{"inner": name}`, or its absence.

        Args:
            part: The option.

        Returns:
            The name of the zone, or `None`.

        Raises:
            TypeError: The part is not a time zone.

        Example:
            >>> OptionReader().zone({"inner": "Asia/Tokyo"}), OptionReader().zone(None)
            ('Asia/Tokyo', None)
            >>> OptionReader().zone(1)
            Traceback (most recent call last):
            ...
            TypeError: no time zone in 1
        """
        if part is None:
            return None
        if (  # pragma: no cover -- polars writes a time zone here
            not isinstance(part, dict) or not isinstance(part.get("inner"), str)
        ):
            message = f"no time zone in {part!r}"
            raise TypeError(message)
        return str(part["inner"])

    def flags(self, part: object) -> tuple[bool, ...]:
        """Read a list of flags, such as the business days of a week.

        Args:
            part: The option.

        Returns:
            The flags, in order.

        Raises:
            TypeError: The part is not a list of flags.

        Example:
            >>> OptionReader().flags([True, False])
            (True, False)
            >>> OptionReader().flags([1])
            Traceback (most recent call last):
            ...
            TypeError: no list of flags in [1]
        """
        if (  # pragma: no cover -- polars writes a list of flags here
            not isinstance(part, list) or not all(isinstance(item, bool) for item in part)
        ):
            message = f"no list of flags in {part!r}"
            raise TypeError(message)
        return tuple(bool(item) for item in part)

    def flag(self, part: object) -> bool:
        """Read a flag.

        Args:
            part: The option.

        Returns:
            The flag.

        Raises:
            TypeError: The part is not a flag.

        Example:
            >>> OptionReader().flag(True)
            True
            >>> OptionReader().flag(1)
            Traceback (most recent call last):
            ...
            TypeError: no flag in 1
        """
        if not isinstance(part, bool):  # pragma: no cover -- polars writes a flag here
            message = f"no flag in {part!r}"
            raise TypeError(message)
        return part

    def flag_or_none(self, part: object) -> bool | None:
        """Read a flag, or its absence.

        Args:
            part: The option.

        Returns:
            The flag, or `None`.

        Example:
            >>> OptionReader().flag_or_none(None), OptionReader().flag_or_none(False)
            (None, False)
        """
        return None if part is None else self.flag(part)

    def data_type(self, part: object) -> DType | None:
        """Read a data type that a function holds as it is, or its absence.

        A function such as `reinterpret` holds a plain data type, and not the
        form `{"Literal": data type}` of an option that an expression can
        give (`dtype`).

        Args:
            part: The option.

        Returns:
            The data type, or `None`.

        Example:
            >>> OptionReader().data_type("Float32"), OptionReader().data_type(None)
            (Float32, None)
        """
        return None if part is None else SerializedDType().of(part)

    def integer(self, part: object) -> int:
        """Read an integer, such as a number of digits.

        Args:
            part: The option.

        Returns:
            The integer.

        Raises:
            TypeError: The part is not an integer.

        Example:
            >>> OptionReader().integer(2)
            2
            >>> OptionReader().integer(True)
            Traceback (most recent call last):
            ...
            TypeError: no integer in True
        """
        if (  # pragma: no cover -- polars writes an integer here
            not isinstance(part, int) or isinstance(part, bool)
        ):
            message = f"no integer in {part!r}"
            raise TypeError(message)
        return part

    def integer_or_none(self, part: object) -> int | None:
        """Read an integer, or its absence, such as the width of `linear_spaces`.

        Args:
            part: The option, or null.

        Returns:
            The integer, or `None`.

        Example:
            >>> OptionReader().integer_or_none(None), OptionReader().integer_or_none(3)
            (None, 3)
        """
        return None if part is None else self.integer(part)

    def floats(self, part: object) -> tuple[float, ...]:
        """Read a list of floats, such as the breaks of `cut`.

        JSON has no NaN and no infinity, so polars writes each of them as
        null. The reader gives NaN for a null: a float that stands for NaN,
        `inf` and `-inf`, which a rule cannot tell apart (`CutBins.names`). A
        NaN that the code holds stays a `KnownNan`.

        Args:
            part: The option.

        Returns:
            The floats, in order.

        Raises:
            TypeError: The part is not a list of numbers and nulls.

        Example:
            >>> OptionReader().floats([0.0, 2, None])
            (0.0, 2.0, nan)
            >>> OptionReader().floats("0")
            Traceback (most recent call last):
            ...
            TypeError: no list of floats in '0'
        """
        if (  # pragma: no cover -- polars writes a list of floats here
            not isinstance(part, list) or not all(self._float(item) for item in part)
        ):
            message = f"no list of floats in {part!r}"
            raise TypeError(message)
        return tuple(self._floated(item) for item in part)

    def _floated(self, item: object) -> float:
        """Give one float of a list, NaN for a null, and a float of the code as it is.

        A float of the code keeps its class, so a `KnownNan` stays one.

        Args:
            item: A number, or null.

        Returns:
            The float.

        Example:
            >>> OptionReader()._floated(2), OptionReader()._floated(None)
            (2.0, nan)
        """
        if item is None:
            return math.nan
        return item if isinstance(item, float) else float(checked_cast(item, int))

    def _float(self, part: object) -> bool:
        """Say whether a part is a float as polars writes it in JSON: a number, or null.

        Args:
            part: The part.

        Returns:
            Whether it is.

        Example:
            >>> OptionReader()._float(None), OptionReader()._float(True), OptionReader()._float(1)
            (True, False, True)
        """
        number = isinstance(part, int | float) and not isinstance(part, bool)
        return number or part is None

    def texts(self, part: object) -> tuple[str, ...] | None:
        """Read a list of texts, such as the labels of `cut`, or its absence.

        Args:
            part: The option.

        Returns:
            The texts, in order, or `None`.

        Raises:
            TypeError: The part is not a list of texts.

        Example:
            >>> OptionReader().texts(["a", "b"]), OptionReader().texts(None)
            (('a', 'b'), None)
            >>> OptionReader().texts([1])
            Traceback (most recent call last):
            ...
            TypeError: no list of texts in [1]
        """
        if part is None:
            return None
        if (  # pragma: no cover -- polars writes a list of texts here
            not isinstance(part, list) or not all(isinstance(item, str) for item in part)
        ):
            message = f"no list of texts in {part!r}"
            raise TypeError(message)
        return tuple(str(item) for item in part)

    def variant(self, part: object) -> str:
        """Read the name of a variant of an enum, as polars writes it with or without its data.

        Polars writes a variant with no data as its name, and a variant with
        data as a map of its name to the data, as the strategies of
        `fill_null` show: `"Min"`, or `{"Forward": null}`.

        Args:
            part: The option.

        Returns:
            The name of the variant.

        Raises:
            TypeError: The part is not a variant.

        Example:
            >>> OptionReader().variant("Min"), OptionReader().variant({"Forward": 2})
            ('Min', 'Forward')
            >>> OptionReader().variant(1)
            Traceback (most recent call last):
            ...
            TypeError: no variant in 1
        """
        if isinstance(part, str):
            return part
        if (  # pragma: no cover -- polars writes a variant here
            not isinstance(part, dict) or len(part) != 1
        ):
            message = f"no variant in {part!r}"
            raise TypeError(message)
        return str(next(iter(part)))

    def number(self, part: object) -> float:
        """Read a number, such as the `alpha` of `ewm_mean`, as a float.

        JSON has no NaN and no infinity, so polars writes each of them as
        null, and the reader gives NaN for a null (`floats`).

        Args:
            part: The option.

        Returns:
            The number. A part that is not a number raises `TypeError`.

        Example:
            >>> OptionReader().number(0.5), OptionReader().number(1)
            (0.5, 1.0)
        """
        return self.floats([part])[0]

    def floats_or_none(self, part: object) -> tuple[float, ...] | None:
        """Read a list of floats, such as the weights of `rolling_mean`, or its absence.

        Args:
            part: The option.

        Returns:
            The floats, in order, or `None`.

        Example:
            >>> OptionReader().floats_or_none([1, 2.5]), OptionReader().floats_or_none(None)
            ((1.0, 2.5), None)
        """
        return None if part is None else self.floats(part)


@runtime_checkable
class OptionValue[T](Protocol):
    """Give a value that a rule reads from the options of its node."""

    def read(self, node: NodeOptions) -> T:
        """Give the value for one node.

        Args:
            node: The serialized root of the node.
        """


@dataclass(frozen=True)
class NodeOption[T]:
    """Name one option of a node by its path in the serialized root, and read its value.

    The path starts at the body of the root. For a `Function`, the options of
    the function sit under the names of its identity, for example
    `("function", "StructExpr", "FieldByName")` for the name of `struct.field`.
    The factories name the kind of value: a data type, a time unit, a time
    zone, the option of a cast, a text, a flag, a number, or a list
    (`OptionReader`).

    Example:
        >>> target = NodeOption.dtype("dtype")
        >>> target
        NodeOption.dtype('dtype')
        >>> cast = {"Cast": {"expr": {}, "dtype": {"Literal": "Int32"}, "options": "Strict"}}
        >>> target.read(NodeOptions(cast))
        Int32
    """

    path: tuple[str | int, ...]
    reader: Callable[[object], T]

    def __repr__(self) -> str:
        """Give the factory and the path.

        Returns:
            The text.

        Example:
            >>> NodeOption.flag("function", "CumSum", "reverse")
            NodeOption.flag('function', 'CumSum', 'reverse')
        """
        path = ", ".join(repr(step) for step in self.path)
        return f"NodeOption.{self.reader.__name__}({path})"

    @classmethod
    def dtype(cls, *path: str | int) -> NodeOption[DType]:
        """Name an option that is a data type.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.dtype("function", "StringExpr", "Strptime", 0).path[-1]
            0
        """
        return NodeOption(path, OptionReader().dtype)

    @classmethod
    def names(cls, *path: str | int) -> NodeOption[tuple[str, ...]]:
        """Name the inputs of a node, read as the output name of each.

        Args:
            *path: The path of the inputs.

        Returns:
            The option.

        Example:
            >>> NodeOption.names("input").path
            ('input',)
        """
        return NodeOption(path, OptionReader().names)

    @classmethod
    def first_name(cls, *path: str | int) -> NodeOption[str]:
        """Name the first input of a node, read as its output name.

        Args:
            *path: The path of the inputs.

        Returns:
            The option.

        Example:
            >>> NodeOption.first_name("input").path
            ('input',)
        """
        return NodeOption(path, OptionReader().first_name)

    @classmethod
    def dtype_or_none(cls, *path: str | int) -> NodeOption[DType | None]:
        """Name an option that is a data type, or null.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.dtype_or_none("function", "ReplaceStrict", "return_dtype").path[-1]
            'return_dtype'
        """
        return NodeOption(path, OptionReader().dtype_or_none)

    @classmethod
    def unit(cls, *path: str | int) -> NodeOption[str]:
        """Name an option that is a time unit.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.unit("function", "TemporalExpr", "CastTimeUnit")
            NodeOption.unit('function', 'TemporalExpr', 'CastTimeUnit')
        """
        return NodeOption(path, OptionReader().unit)

    @classmethod
    def cast_option(cls, *path: str | int) -> NodeOption[CastOption]:
        """Name an option that is the option of a cast.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.cast_option("options")
            NodeOption.cast_option('options')
        """
        return NodeOption(path, OptionReader().cast_option)

    @classmethod
    def text(cls, *path: str | int) -> NodeOption[str | None]:
        """Name an option that is a text, or its absence.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.text("function", "StructExpr", "FieldByName")
            NodeOption.text('function', 'StructExpr', 'FieldByName')
        """
        return NodeOption(path, OptionReader().text)

    @classmethod
    def zone(cls, *path: str | int) -> NodeOption[str | None]:
        """Name an option that is a time zone, or its absence.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.zone("function", "TemporalExpr", "ReplaceTimeZone", 0)
            NodeOption.zone('function', 'TemporalExpr', 'ReplaceTimeZone', 0)
        """
        return NodeOption(path, OptionReader().zone)

    @classmethod
    def flags(cls, *path: str | int) -> NodeOption[tuple[bool, ...]]:
        """Name an option that is a list of flags.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.flags("function", "Business", "IsBusinessDay", "week_mask")
            NodeOption.flags('function', 'Business', 'IsBusinessDay', 'week_mask')
        """
        return NodeOption(path, OptionReader().flags)

    @classmethod
    def flag(cls, *path: str | int) -> NodeOption[bool]:
        """Name an option that is a flag.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.flag("function", "CumSum", "reverse").reader.__name__
            'flag'
        """
        return NodeOption(path, OptionReader().flag)

    @classmethod
    def flag_or_none(cls, *path: str | int) -> NodeOption[bool | None]:
        """Name an option that is a flag, or its absence.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.flag_or_none("function", "Reinterpret", 0)
            NodeOption.flag_or_none('function', 'Reinterpret', 0)
        """
        return NodeOption(path, OptionReader().flag_or_none)

    @classmethod
    def data_type(cls, *path: str | int) -> NodeOption[DType | None]:
        """Name an option that is a plain data type, or its absence.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.data_type("function", "Reinterpret", 1)
            NodeOption.data_type('function', 'Reinterpret', 1)
        """
        return NodeOption(path, OptionReader().data_type)

    @classmethod
    def integer(cls, *path: str | int) -> NodeOption[int]:
        """Name an option that is an integer.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.integer("function", "RoundSF", "digits")
            NodeOption.integer('function', 'RoundSF', 'digits')
        """
        return NodeOption(path, OptionReader().integer)

    @classmethod
    def floats(cls, *path: str | int) -> NodeOption[tuple[float, ...]]:
        """Name an option that is a list of floats.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.floats("function", "Cut", "breaks")
            NodeOption.floats('function', 'Cut', 'breaks')
        """
        return NodeOption(path, OptionReader().floats)

    @classmethod
    def texts(cls, *path: str | int) -> NodeOption[tuple[str, ...] | None]:
        """Name an option that is a list of texts, or its absence.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.texts("function", "Cut", "labels")
            NodeOption.texts('function', 'Cut', 'labels')
        """
        return NodeOption(path, OptionReader().texts)

    @classmethod
    def variant(cls, *path: str | int) -> NodeOption[str]:
        """Name an option that is a variant of an enum, read as its name.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.variant("function", "FillNullWithStrategy")
            NodeOption.variant('function', 'FillNullWithStrategy')
        """
        return NodeOption(path, OptionReader().variant)

    @classmethod
    def number(cls, *path: str | int) -> NodeOption[float]:
        """Name an option that is a number, read as a float.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.number("function", "EwmMean", "options", "alpha")
            NodeOption.number('function', 'EwmMean', 'options', 'alpha')
        """
        return NodeOption(path, OptionReader().number)

    @classmethod
    def floats_or_none(cls, *path: str | int) -> NodeOption[tuple[float, ...] | None]:
        """Name an option that is a list of floats, or its absence.

        Args:
            *path: The path of the option.

        Returns:
            The option.

        Example:
            >>> NodeOption.floats_or_none("function", "RollingExpr", "options", "weights")
            NodeOption.floats_or_none('function', 'RollingExpr', 'options', 'weights')
        """
        return NodeOption(path, OptionReader().floats_or_none)

    def read(self, node: NodeOptions) -> T:
        """Give the value of the option for one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The value.

        Example:
            >>> node = NodeOptions({"Function": {"function": {"CumSum": {"reverse": True}}}})
            >>> NodeOption.flag("function", "CumSum", "reverse").read(node)
            True
        """
        return self.reader(node.value(self.path))


class FromOptions[T]:
    """Build a part of the vocabulary from options of the node, as polars builds a closure.

    Polars keeps an option in the function of a node, and a closure of its
    type function or its kernel captures it, for example the time unit of
    `dt.cast_time_unit`. This class holds the function of the vocabulary that
    builds the part, and the options that it takes, in order or by name. A
    part can be the shape of a node too, as for a gather that gives one value.

    Example:
        >>> from astrolabe._rules.kernels.dates import DateKernels
        >>> unit = NodeOption.unit("function", "TemporalExpr", "CastTimeUnit")
        >>> kernel = FromOptions(DateKernels().cast_time_unit, unit)
        >>> kernel
        FromOptions(cast_time_unit, NodeOption.unit('function', 'TemporalExpr', 'CastTimeUnit'))
        >>> function = {"TemporalExpr": {"CastTimeUnit": "Nanoseconds"}}
        >>> node = NodeOptions({"Function": {"function": function}})
        >>> kernel.read(node).name
        'cast_time_unit'
    """

    def __init__(
        self,
        function: Callable[..., T],
        *options: OptionValue[object],
        **named: OptionValue[object],
    ) -> None:
        """Keep the function and the options.

        Args:
            function: The function of the vocabulary that builds the part.
            *options: The options that it takes, in order.
            **named: The options that it takes by name.

        Example:
            >>> from astrolabe._rules.type_functions import StructClosures
            >>> len(FromOptions(StructClosures().field_by_name, NodeOption.text("x"))._options)
            1
        """
        self._function = function
        self._options = options
        self._named = named

    def __repr__(self) -> str:
        """Give the name of the function and the options.

        Returns:
            The text.

        Example:
            >>> from astrolabe._rules.options import Shape
            >>> from astrolabe._rules.type_functions import StructClosures
            >>> FromOptions(StructClosures().field_by_name, NodeOption.text("x"))
            FromOptions(field_by_name, NodeOption.text('x'))
            >>> FromOptions(Shape.gather, returns_scalar=NodeOption.flag("returns_scalar"))
            FromOptions(gather, returns_scalar=NodeOption.flag('returns_scalar'))
        """
        parts = [self._function.__name__, *(repr(option) for option in self._options)]
        parts.extend(f"{name}={option!r}" for name, option in self._named.items())
        return f"FromOptions({', '.join(parts)})"

    def read(self, node: NodeOptions) -> T:
        """Build the part for one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The part.

        Example:
            >>> from astrolabe._rules.options import Shape
            >>> name = FromOptions(str.upper, NodeOption.text("function", "FieldByName"))
            >>> name.read(NodeOptions({"Function": {"function": {"FieldByName": "a"}}}))
            'A'
            >>> get = FromOptions(Shape.gather, returns_scalar=NodeOption.flag("returns_scalar"))
            >>> get.read(NodeOptions({"Gather": {"returns_scalar": True}})).returns_scalar
            True
        """
        values = [option.read(node) for option in self._options]
        named = {name: option.read(node) for name, option in self._named.items()}
        return self._function(*values, **named)


class WhenSet[T]:
    """Pick one of two values by whether an option of the node holds a value.

    Polars matches some options as present or absent. For example, it gives
    `str.strptime` the options of an elementwise function where the node has
    a format, and other options where polars must infer the format.

    Example:
        >>> pattern = NodeOption.text("function", "format")
        >>> choice = WhenSet(pattern, "elementwise", "length preserving")
        >>> choice.read(NodeOptions({"Function": {"function": {"format": None}}}))
        'length preserving'
    """

    def __init__(self, option: OptionValue[object], present: T, absent: T) -> None:
        """Keep the option and the two values.

        Args:
            option: The option.
            present: The value where the option holds a value.
            absent: The value where the option holds `None`.

        Example:
            >>> WhenSet(NodeOption.text("x"), 1, 2)._absent
            2
        """
        self._option = option
        self._present = present
        self._absent = absent

    def __repr__(self) -> str:
        """Give the option and the two values.

        Returns:
            The text.

        Example:
            >>> WhenSet(NodeOption.text("x"), 1, 2)
            WhenSet(NodeOption.text('x'), 1, 2)
        """
        return f"WhenSet({self._option!r}, {self._present!r}, {self._absent!r})"

    def read(self, node: NodeOptions) -> T:
        """Give the value for one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The value.

        Example:
            >>> choice = WhenSet(NodeOption.text("function", "format"), 1, 2)
            >>> choice.read(NodeOptions({"Function": {"function": {"format": "%Y"}}}))
            1
        """
        return self._absent if self._option.read(node) is None else self._present


class WhenNoColumn[T]:
    """Pick one of two values by whether an input of the node reads no column of the frame.

    Polars gives `str.strptime` the options of an elementwise function where
    its text reads no column, as for a literal text, whatever its format.
    An input reads a column where its tree holds a column, `pl.len()`, or a
    field of a struct.

    Example:
        >>> choice = WhenNoColumn(0, "elementwise", "length preserving")
        >>> text = {"Literal": {"Scalar": {"String": "2020"}}}
        >>> choice.read(NodeOptions({"Function": {"input": [text]}}))
        'elementwise'
    """

    READS: ClassVar[frozenset[str]] = frozenset({"Column", "Len", "Field", "FieldByName"})

    def __init__(self, index: int, free: T, other: T | OptionValue[T]) -> None:
        """Keep the place of the input and the two values.

        Args:
            index: The place of the input among the inputs of the node.
            free: The value where the input reads no column.
            other: The value otherwise, or an option that gives it.

        Example:
            >>> WhenNoColumn(0, 1, 2)._index
            0
        """
        self._index = index
        self._free = free
        self._other = other

    def __repr__(self) -> str:
        """Give the place and the two values.

        Returns:
            The text.

        Example:
            >>> WhenNoColumn(0, 1, 2)
            WhenNoColumn(0, 1, 2)
        """
        return f"WhenNoColumn({self._index!r}, {self._free!r}, {self._other!r})"

    def read(self, node: NodeOptions) -> T:
        """Give the value for one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The value.

        Example:
            >>> choice = WhenNoColumn(0, 1, 2)
            >>> choice.read(NodeOptions({"Function": {"input": [{"Column": "s"}]}}))
            2
        """
        inputs = node.value(("input",))
        tree = inputs[self._index] if isinstance(inputs, list) else None
        if tree is not None and not self._reads(tree):
            return self._free
        return OptionValues().known(OptionValues().at(self._other, node), "the value")

    def _reads(self, tree: object) -> bool:
        """Say whether a serialized tree reads a column of the frame.

        Args:
            tree: The tree, or a part of it.

        Returns:
            Whether it does.

        Example:
            >>> WhenNoColumn(0, 1, 2)._reads({"Agg": {"First": {"Column": "s"}}})
            True
            >>> WhenNoColumn(0, 1, 2)._reads("Len")
            True
            >>> WhenNoColumn(0, 1, 2)._reads([1, {"Column": "s"}])
            True
        """
        if isinstance(tree, str):
            return tree in self.READS
        if isinstance(tree, dict):
            pairs = tree.items()
            return any(key in self.READS or self._reads(value) for key, value in pairs)
        if isinstance(tree, list):
            return any(self._reads(part) for part in tree)
        return False


class WhenIn[T]:
    """Pick one of two values by whether an option of the node is one of some values.

    Polars matches some options against a few of their values. For example,
    it gives `fill_null` the options of an elementwise function where the
    strategy is `zero` or `one`, and the options of a function that keeps the
    length of its input for the other strategies.

    Example:
        >>> strategy = NodeOption.variant("function", "FillNullWithStrategy")
        >>> choice = WhenIn(strategy, ("Zero", "One"), "elementwise", "length preserving")
        >>> choice.read(NodeOptions({"Function": {"function": {"FillNullWithStrategy": "One"}}}))
        'elementwise'
    """

    def __init__(
        self,
        option: OptionValue[object],
        values: Sequence[object],
        present: T,
        absent: T,
    ) -> None:
        """Keep the option, the values that it matches, and the two values that it picks.

        Args:
            option: The option.
            values: The values of the option that pick `present`.
            present: The value where the option is one of `values`.
            absent: The value where it is not.

        Example:
            >>> WhenIn(NodeOption.text("x"), ("a",), 1, 2)._values
            ('a',)
        """
        self._option = option
        self._values = tuple(values)
        self._present = present
        self._absent = absent

    def __repr__(self) -> str:
        """Give the option, the values that it matches and the two values.

        Returns:
            The text.

        Example:
            >>> WhenIn(NodeOption.text("x"), ("a",), 1, 2)
            WhenIn(NodeOption.text('x'), ('a',), 1, 2)
        """
        return f"WhenIn({self._option!r}, {self._values!r}, {self._present!r}, {self._absent!r})"

    def read(self, node: NodeOptions) -> T:
        """Give the value for one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The value.

        Example:
            >>> choice = WhenIn(NodeOption.text("function", "Diff"), ("Drop",), 1, 2)
            >>> choice.read(NodeOptions({"Function": {"function": {"Diff": "Ignore"}}}))
            2
        """
        return self._present if self._option.read(node) in self._values else self._absent


class OptionValues:
    """Give the value of a part of a rule that can be an option of its node.

    A part of a rule is a value of the vocabulary, or an option of the node
    that gives such a value (`OptionValue`). Code that reads a node turns the
    second into the first (`at`). A rule that runs needs the first (`known`).

    Example:
        >>> node = NodeOptions({"Cast": {"dtype": {"Literal": "Int8"}}})
        >>> OptionValues().at(NodeOption.dtype("dtype"), node), OptionValues().at(1, node)
        (Int8, 1)
    """

    def at[T](self, value: T | OptionValue[T], node: NodeOptions) -> T:
        """Give the value of a part for one node.

        Args:
            value: The part.
            node: The serialized root of the node.

        Returns:
            The value of the option at the node, or the part itself.

        Example:
            >>> OptionValues().at("a", NodeOptions({"Cast": {}}))
            'a'
        """
        if isinstance(value, OptionValue):
            read: T = value.read(node)
            return read
        return value

    def known[T](self, value: T | OptionValue[T], part: str) -> T:
        """Give a part that is not an option of the node any more.

        Args:
            value: The part.
            part: What the part is, for the message of the error.

        Returns:
            The part.

        Raises:
            UnreadOptionError: The part is still an option of the node.

        Example:
            >>> OptionValues().known(NodeOption.text("x"), "the name")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.definitions.node_options.UnreadOptionError: the name is an option ...
        """
        if isinstance(value, OptionValue):  # pragma: no cover -- options are read first
            raise UnreadOptionError(part)
        return value


class Arguments:
    """Hold the arguments that a rule passes to a part of the vocabulary, some of them options.

    Example:
        >>> arguments = Arguments({"dtype": NodeOption.dtype("dtype"), "strict": True})
        >>> arguments.at(NodeOptions({"Cast": {"dtype": {"Literal": "Date"}}})).known()
        {'dtype': Date, 'strict': True}
    """

    def __init__(self, values: Mapping[str, object]) -> None:
        """Keep the arguments by name.

        Args:
            values: The arguments.

        Example:
            >>> Arguments({"func": len})._values
            {'func': <built-in function len>}
        """
        self._values = dict(values)

    def texts(self) -> list[str]:
        """Give each argument as a text, with its name.

        Returns:
            The texts.

        Example:
            >>> Arguments({"func": len, "coerce": True}).texts()
            ['func=len', 'coerce=True']
        """
        texts: list[str] = []
        for name, value in self._values.items():
            shown = getattr(value, "__name__", repr(value))
            texts.append(f"{name}={shown}")
        return texts

    def at(self, node: NodeOptions) -> Arguments:
        """Give the arguments for one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The arguments, with each option read.

        Example:
            >>> Arguments({"n": 1}).at(NodeOptions({"Cast": {}})).known()
            {'n': 1}
        """
        values = OptionValues()
        return Arguments({name: values.at(value, node) for name, value in self._values.items()})

    def known(self) -> dict[str, object]:
        """Give the arguments, where none is an option of the node any more.

        Returns:
            The arguments.

        Example:
            >>> Arguments({"name": NodeOption.text("x")}).known()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.definitions.node_options.UnreadOptionError: the argument name is ...
        """
        values = OptionValues()
        return {
            name: values.known(value, f"the argument {name}")
            for name, value in self._values.items()
        }
