"""Turn the text form of a data type into a mypy type, and back.

The engine holds each data type as its rendering: the text that polars shows for
it, for example `Int64`, `List(UInt32)` or
`Datetime(time_unit='us', time_zone=None)`. Text is easy to compare and to store
in a table, and it needs no mypy. A frame column, however, carries its data type
as a mypy type, for example `pl.Int64` or `Datetime[Literal['us'], None]`.

This module crosses between the two forms. A data type with no parameters
becomes the polars class of the same name. A data type with parameters becomes
the marker that the user writes in a frame declaration, from `astrolabe.frame`.
A type that the checker computes is then equal to the declared type of a column,
and it shows the same way in a message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.types import Instance, NoneType, TupleType, get_proper_type

from astrolabe import frame
from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.marker_minter import MarkerMinter, ProperReads
from astrolabe._hooks.expression.kind import KindRoster
from astrolabe._hooks.expression.rendering import ArrayText, NameText, StructFields
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe.frame import Column
from astrolabe.frame import Unchecked as _UncheckedMarker

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.types import Type as MypyType


class DtypeSpelling:
    """Find the polars class for the name of a kind.

    A kind is a data type with its parameters taken away, for example `List` or
    `Datetime`. The expression tables of the engine name kinds as plain text,
    for example `UInt32`, and they use `Unchecked` for a data type that the
    checker does not know. A type that holds a kind holds the polars class of
    the same name, or the `Unchecked` marker. This class maps the name to the
    class.

    The map comes from the polars classes that are installed, so it cannot go
    out of step with polars.

    Example:
        >>> DtypeSpelling().fullname("UInt32")
        'polars.datatypes.classes.UInt32'
    """

    def fullname(self, text: str) -> str | None:
        """Give the full name of the class that stands for the name of a kind.

        Args:
            text: The name of the kind, for example `Boolean`, or `Unchecked`.

        Returns:
            The full name of the polars class, or of the `Unchecked` marker.
            None for a name that is not a known kind.

        Example:
            >>> DtypeSpelling().fullname("Boolean")
            'polars.datatypes.classes.Boolean'
            >>> DtypeSpelling().fullname("Unchecked")
            'astrolabe.frame.Unchecked'
            >>> DtypeSpelling().fullname("NoSuchKind") is None
            True
        """
        if text == "Unchecked":
            return Target.of(_UncheckedMarker).fullname
        for cls in KindRoster().classes():
            if cls.__name__ == text:
                return f"{cls.__module__}.{cls.__qualname__}"
        return None


class ConstDtypeCodec:
    """Build the mypy type for one data type rendering.

    The engine answers the data type of a column as a rendering, and a hook
    needs a mypy type to put in the column of a frame. This class builds that
    type. It handles each shape that a rendering can have: a bare data type, a
    `List` of another, an `Array` with a shape, a `Struct` of named fields, a
    `Decimal` with its digits, an `Enum` with its categories, and a `Duration`
    or a `Datetime` with a time unit. Each shape becomes the parametrized marker
    that a user writes in a frame declaration.

    Example:
        >>> fixture = new_type_fixture()
        >>> ConstDtypeCodec(new_marker_minter(fixture)).built("List(UInt32)")
        astrolabe.frame.List[polars.datatypes.classes.UInt32]
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the minter that builds the types.

        Args:
            minter: The marker minter for the current phase.
        """
        self._minter = minter
        self._spelling = DtypeSpelling()

    @propagates(None)
    def built(self, text: str) -> MypyType | None:
        """Build the type for one rendering.

        A rendering with no parenthesis is the name of a kind, and it becomes
        the polars class of that name. For a rendering with parameters, the name
        before the parenthesis selects the builder. A name with no builder of
        its own is a data type with a time unit.

        Args:
            text: The rendering.

        Returns:
            The type, or None if the text is not a rendering that this class
            knows.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = ConstDtypeCodec(new_marker_minter(fixture))
            >>> codec.built("Int64")
            polars.datatypes.classes.Int64
            >>> codec.built("Nope(1)") is None
            True
        """
        if "(" not in text:
            fullname = not_none(self._spelling.fullname(text))
            return self._minter.resolved(fullname)
        head, _, rest = text.partition("(")
        body = rest.removesuffix(")")
        builders: dict[str, Callable[[str], MypyType | None]] = {
            "List": self._list,
            "Struct": self._struct,
            "Array": self._array,
            "Decimal": self._decimal,
            "Enum": self._enum,
        }
        builder = builders.get(head)
        if builder is not None:
            return builder(body)
        return self._timed(head, body)

    @propagates(None)
    def planned(self, run: str, planner: str) -> MypyType | None:
        """Build the type of a column of a lazy frame whose planner type differs from its run type.

        A lazy frame plans each step at the type that it stores for a column,
        and the data of the column can have another type. The column then
        holds both, as `Planned[run, planner]`.

        Args:
            run: The rendering of the type of the data.
            planner: The rendering of the type that polars stores.

        Returns:
            The type, or None if either text is not a rendering that this
            class knows.

        Example:
            >>> fixture = new_type_fixture()
            >>> planned = ConstDtypeCodec(new_marker_minter(fixture)).planned("Float64", "Float32")
            >>> planned.type.fullname, [ConstDtypeReads().read_type(arg) for arg in planned.args]
            ('astrolabe.frame.Planned', ['Float64', 'Float32'])
        """
        types = [not_none(self.built(run)), not_none(self.built(planner))]
        return self._minter.marker(frame.Planned, types)

    @propagates(None)
    def _list(self, body: str) -> MypyType | None:
        """Build the `List[...]` marker around the type of the element.

        Args:
            body: The text between the parentheses: the rendering of the
                element.

        Returns:
            The `List[element]` type, or None if the element has no type.
        """
        return self._minter.marker(frame.List, [not_none(self.built(body))])

    @propagates(None)
    def _array(self, body: str) -> MypyType | None:
        """Build the `Array[...]` markers for an array rendering and its whole shape.

        polars shows an array of arrays as one array with a shape of several
        dimensions. The `Array[DType, Shape]` marker holds one width, so a shape
        of several dimensions needs one marker for each dimension. The
        outermost width becomes the outermost marker.

        Args:
            body: The text between the parentheses: the rendering of the
                element, then `shape=(n,)`.

        Returns:
            The nested `Array[...]` type, or None if the text does not read as
            an array, the shape is empty or a width is not a number.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = ConstDtypeCodec(new_marker_minter(fixture))
            >>> nested = codec.built("Array(Int64, shape=(2, 3))")
            >>> nested.args[1], nested.args[0].args[1]  # the outer width, then the inner
            (Literal[2], Literal[3])
        """
        element, widths = not_none(ArrayText().read(f"Array({body})"))
        not_falsy(element)
        built = not_none(self.built(element))
        not_falsy(widths)
        for width in reversed(widths):
            not_falsy(width.isdigit())
            array = self._minter.marker(frame.Array, [built, self._minter.integer(int(width))])
            built = not_none(array)
        return built

    @propagates(None)
    def _decimal(self, body: str) -> MypyType | None:
        """Build the `Decimal[...]` marker for a decimal rendering.

        Args:
            body: The text between the parentheses: `precision=p, scale=s`.

        Returns:
            The `Decimal[Literal[p], Literal[s]]` type, or None if a digit count
            is not a number.
        """
        precision, _, scale = body.partition(", ")
        digits = not_none(self._digits(precision.removeprefix("precision=")))
        places = not_none(self._digits(scale.removeprefix("scale=")))
        return self._minter.marker(
            frame.Decimal,
            [self._minter.integer(digits), self._minter.integer(places)],
        )

    def _digits(self, text: str) -> int | None:
        """Read one digit count of a decimal rendering.

        Args:
            text: The text of the count.

        Returns:
            The number, or None if the text is not a number.

        Example:
            >>> fixture = new_type_fixture()
            >>> ConstDtypeCodec(new_marker_minter(fixture))._digits("x") is None
            True
        """
        if not text.isdigit():
            return None
        return int(text)

    @propagates(None)
    def _enum(self, body: str) -> MypyType | None:
        """Build the `Enum[...]` marker for an enum rendering.

        The categories keep the order of the rendering, because the order of the
        categories is part of an enum data type.

        Args:
            body: The text between the parentheses: `categories=['a', 'b']`.

        Returns:
            The `Enum[...]` type over a tuple of string literals, or None if the
            text does not list categories.
        """
        listed = body.removeprefix("categories=")
        not_falsy(listed != body)
        categories = [self._minter.string(name) for name in NameText().names(listed)]
        return self._minter.marker(frame.Enum, [not_none(self._minter.tuple_of(categories))])

    @propagates(None)
    def _struct(self, body: str) -> MypyType | None:
        """Build the `Struct[...]` marker for a struct rendering.

        The fields keep the order of the rendering. Each field becomes a
        `Column[name, dtype]` item of one tuple, which is the form that mypy
        gives a `Struct[...]` declaration.

        Args:
            body: The text between the parentheses: a dict display that maps
                each field name to its rendering.

        Returns:
            The `Struct[...]` type, or None if a field has no type.
        """
        fields: list[MypyType] = []
        for name, rendering in StructFields().of(body):
            dtype = not_none(self.built(rendering))
            fields.append(not_none(self._minter.marker(Column, [self._minter.string(name), dtype])))
        return self._minter.marker(frame.Struct, [not_none(self._minter.tuple_of(fields))])

    @propagates(None)
    def _timed(self, head: str, body: str) -> MypyType | None:
        """Build the marker for a rendering that carries a time unit.

        A `Duration` has a unit only. A `Datetime` has a unit and a time zone. A
        `Datetime` rendering with no `time_zone` stands for a datetime in any
        zone, and it becomes the `AnyZone` marker. A zone of `None` stands for a
        datetime with no zone.

        Args:
            head: The name before the parenthesis.
            body: The text between the parentheses.

        Returns:
            The `Duration[...]` or `Datetime[...]` type, or None if the text is
            not in the form that polars shows.
        """
        text = _ConstText()
        unit = not_none(text.unit_of(body))
        literal = self._minter.string(unit)
        if head == "Duration":
            not_falsy(body == f"time_unit='{unit}'")
            return self._minter.marker(frame.Duration, [literal])
        not_falsy(head == "Datetime")
        if body == f"time_unit='{unit}'":
            every = self._minter.resolved(Target.of(frame.AnyZone).fullname)
            return self._minter.marker(frame.Datetime, [literal, not_none(every)])
        zone = text.zone_of(body)
        not_falsy(body == f"time_unit='{unit}', time_zone={text.zone_text(zone)}")
        zone_type = NoneType() if zone is None else self._minter.string(zone)
        return self._minter.marker(frame.Datetime, [literal, zone_type])


class ConstDtypeReads:
    """Read the mypy type of a data type back into its rendering.

    This is the reading half of `ConstDtypeCodec`. The engine reads the data
    types of a frame as renderings, so a hook reads each declared column through
    this class before it asks the engine. It builds no type, so it needs no
    access to the plugin.

    Example:
        >>> from mypy.types import UninhabitedType
        >>> ConstDtypeReads().read_type(UninhabitedType()) is None
        True
    """

    def __init__(self) -> None:
        """Set up the reader of instances."""
        self._reads = ProperReads()

    def planner(self, typ: MypyType) -> str | None:
        """Read the planner type of a column of a lazy frame, where it differs from its run type.

        A column of a lazy frame holds `Planned[run, planner]` where polars
        stores another type than the type of its data. Every other reading of
        such a column gives its run type.

        Args:
            typ: The data type of the column.

        Returns:
            The rendering of the planner type, or None for a type that is not
            `Planned[...]`.

        Example:
            >>> fixture = new_type_fixture()
            >>> planned = ConstDtypeCodec(new_marker_minter(fixture)).planned("Float64", "Float32")
            >>> ConstDtypeReads().planner(planned), ConstDtypeReads().read_type(planned)
            ('Float32', 'Float64')
        """
        instance = self._reads.instance(typ)
        if instance is None or not self._reads.matches(instance, frame.Planned):
            return None
        return self.read_type(instance.args[1])

    def read_type(self, typ: MypyType) -> str | None:
        """Read any type as a data type rendering.

        Args:
            typ: The type to read.

        Returns:
            The rendering, or None if the type does not stand for a data type.
        """
        instance = self._reads.instance(typ)
        if instance is None:
            return None
        return self.read(instance)

    def read(self, instance: Instance) -> str | None:
        """Read an instance as a data type rendering.

        A parametrized marker reads back into its rendering. A class with no
        parameters reads as the name of its data type.

        Args:
            instance: The instance to read.

        Returns:
            The rendering, or None if the class does not stand for a data type.

        Example:
            >>> from mypy.types import Instance
            >>> fixture = new_type_fixture()
            >>> ConstDtypeReads().read(Instance(fixture.oi, [])) is None
            True
        """
        structured = self._structured(instance)
        if structured is not None:
            return structured
        kind = instance.type.name.rsplit(".", 1)[-1]
        if kind not in KindRoster().names() and kind != "Unchecked":
            return None
        return kind

    @propagates(None)
    def _structured(self, instance: Instance) -> str | None:
        """Read a parametrized marker back into its rendering.

        The class of the marker selects the reader.

        Args:
            instance: The instance to read.

        Returns:
            The rendering, or None for a class with no parameters or a marker in
            a wrong form.
        """
        readers = (
            (frame.Planned, self._run),
            (frame.List, self._list),
            (frame.Struct, self._struct),
            (frame.Array, self._array),
            (frame.Decimal, self._decimal),
            (frame.Enum, self._enum),
        )
        for marker, reader in readers:
            if self._reads.matches(instance, marker):
                return reader(instance)
        return self._timed(instance)

    @propagates(None)
    def _run(self, instance: Instance) -> str | None:
        """Read a `Planned[...]` marker as its run type, the type of the data.

        Args:
            instance: The marker.

        Returns:
            The rendering of the run type, or None if the marker is in a wrong form.
        """
        not_falsy(len(instance.args) == 2)
        return self.read_type(instance.args[0])

    @propagates(None)
    def _list(self, instance: Instance) -> str | None:
        """Read a `List[...]` marker back into its rendering.

        Args:
            instance: The list marker.

        Returns:
            The rendering of the list around the rendering of its element, or
            None if the marker is in a wrong form.
        """
        not_falsy(len(instance.args) == 1)
        return f"List({not_none(self.read_type(instance.args[0]))})"

    @propagates(None)
    def _array(self, instance: Instance) -> str | None:
        """Read an `Array[...]` marker back into its rendering.

        An array whose element is also an array shows as one array with a shape
        of several dimensions, the way that polars shows it.

        Args:
            instance: The array marker.

        Returns:
            The rendering, or None if the marker is in a wrong form.
        """
        not_falsy(len(instance.args) == 2)
        element = not_none(self.read_type(instance.args[0]))
        width = not_none(self._reads.int_literal(instance.args[1]))
        return ArrayText().folded(element, str(width))

    @propagates(None)
    def _decimal(self, instance: Instance) -> str | None:
        """Read a `Decimal[...]` marker back into its rendering.

        Args:
            instance: The decimal marker.

        Returns:
            The rendering with its precision and scale, or None if the marker is
            in a wrong form.
        """
        not_falsy(len(instance.args) == 2)
        precision = not_none(self._reads.int_literal(instance.args[0]))
        scale = not_none(self._reads.int_literal(instance.args[1]))
        return f"Decimal(precision={precision}, scale={scale})"

    @propagates(None)
    def _enum(self, instance: Instance) -> str | None:
        """Read an `Enum[...]` marker back into its rendering.

        The marker holds its categories in one of two forms. A type annotation,
        `Enum[Literal["a"], Literal["b"]]`, holds them in one tuple, which is
        how mypy stores the arguments of a class with a variadic type
        parameter. A call such as `pl.Enum(["a"])` gives them as separate type
        arguments.

        Args:
            instance: The enum marker.

        Returns:
            The rendering, with the categories in the order of the marker, or
            None if the marker is in a wrong form.
        """
        items = list(instance.args)
        packed = get_proper_type(instance.args[0]) if len(items) == 1 else None
        if isinstance(packed, TupleType):
            items = list(packed.items)
        names = [not_none(self._reads.string_literal(item)) for item in items]
        return f"Enum(categories={NameText().listed(names)})"

    @propagates(None)
    def _struct(self, instance: Instance) -> str | None:
        """Read a `Struct[...]` marker back into its rendering.

        The marker holds one tuple of `Column[name, dtype]` fields. Each field
        becomes one `'name': rendering` entry.

        Args:
            instance: The struct marker.

        Returns:
            The rendering, with the fields in the order of the marker, or None
            if the marker is in a wrong form.
        """
        not_falsy(len(instance.args) == 1)
        fields_tuple = get_proper_type(instance.args[0])
        not_falsy(isinstance(fields_tuple, TupleType))
        fields: list[str] = []
        for argument in checked_cast(fields_tuple, TupleType).items:
            column = not_none(self._reads.instance(argument))
            not_falsy(self._reads.matches(column, Column))
            not_falsy(len(column.args) == 2)
            name = not_none(self._reads.string_literal(column.args[0]))
            rendering = not_none(self.read_type(column.args[1]))
            fields.append(f"{NameText().spelled(name)}: {rendering}")
        return "Struct({" + ", ".join(fields) + "})"

    @propagates(None)
    def _timed(self, instance: Instance) -> str | None:
        """Read a `Duration[...]` or `Datetime[...]` marker back into its rendering.

        A `Datetime` with the `AnyZone` marker gives a rendering with no
        `time_zone`. A zone argument of the type of `None` gives
        `time_zone=None`, and a string literal gives the zone in quotes.

        Args:
            instance: The instance to read.

        Returns:
            The rendering, or None for a class that carries no time unit or a
            marker in a wrong form.
        """
        if self._reads.matches(instance, frame.Duration):
            not_falsy(len(instance.args) == 1)
            unit = not_none(self._reads.string_literal(instance.args[0]))
            return f"Duration(time_unit='{unit}')"
        not_falsy(self._reads.matches(instance, frame.Datetime))
        not_falsy(len(instance.args) == 2)
        unit = not_none(self._reads.string_literal(instance.args[0]))
        zone_part = instance.args[1]
        zone_marker = self._reads.instance(zone_part)
        if zone_marker is not None and self._reads.matches(zone_marker, frame.AnyZone):
            return f"Datetime(time_unit='{unit}')"
        zone: str | None = None
        if not isinstance(get_proper_type(zone_part), NoneType):
            zone = not_none(self._reads.string_literal(zone_part))
        return f"Datetime(time_unit='{unit}', time_zone={_ConstText().zone_text(zone)})"


class _ConstText:
    """Read the time unit and the time zone out of the text of a rendering.

    polars shows a unit or a zone as a keyword argument in quotes, for example
    `time_unit='ns'`. The codec and the reader both need that text, in both
    directions.

    Example:
        >>> _ConstText().unit_of("time_unit='ns'")
        'ns'
    """

    def unit_of(self, text: str) -> str | None:
        """Give the time unit that a rendering names.

        Args:
            text: The rendering, or only the text between its parentheses.

        Returns:
            The unit, or None if the text names no unit.

        Example:
            >>> _ConstText().unit_of("Duration") is None
            True
        """
        return self._quoted_after(text, "time_unit='")

    def zone_of(self, text: str) -> str | None:
        """Give the time zone that a rendering names.

        Args:
            text: The rendering, or only the text between its parentheses.

        Returns:
            The zone, or None if the zone is `None` or the text names no zone.

        Example:
            >>> _ConstText().zone_of("time_unit='us', time_zone='UTC'")
            'UTC'
            >>> _ConstText().zone_of("time_unit='us', time_zone=None") is None
            True
        """
        return self._quoted_after(text, "time_zone='")

    def zone_text(self, zone: str | None) -> str:
        """Write a time zone the way that polars shows it in a rendering.

        Args:
            zone: The zone, or None.

        Returns:
            The zone in quotes, or `None`.

        Example:
            >>> _ConstText().zone_text("UTC"), _ConstText().zone_text(None)
            ("'UTC'", 'None')
        """
        if zone is None:
            return "None"
        return f"'{zone}'"

    def _quoted_after(self, text: str, marker: str) -> str | None:
        """Read the value in quotes that follows one keyword.

        Args:
            text: The text of the rendering.
            marker: The keyword, with its equals sign and its opening quote.

        Returns:
            The text up to the closing quote, or None if the keyword is not in
            the text.

        Example:
            >>> _ConstText()._quoted_after("a='x'", "a='")
            'x'
        """
        start = text.find(marker)
        if start < 0:
            return None
        end = text.index("'", start + len(marker))
        return text[start + len(marker) : end]
