"""Read a data type with holes as the data types that it stands for, and generalize answers.

A frame can declare a column whose data type the checker does not fully know:
`Unchecked`, a family of polars such as `NumericType`, or a kind with open
parameters such as a bare `pl.Datetime`. A nested type can hold such a part,
as `List(Unchecked)`. Polars needs a full data type to type an expression, so
the evaluation of a deferred expression tries each data type of a fixed
sample that the declaration allows. It then drops the data types where polars
refuses, and keeps what the other answers share: `List(Int8)` and
`List(String)` share `List(Unchecked)`.

The sample holds one data type for each kind that holds no other type, and a
few for the kinds whose parameters change the answers of polars: the unit and
the zone of a `Datetime`, the unit of a `Duration`, the precision of a
`Decimal` and the categories of an `Enum`. A `List`, an `Array` and a `Struct`
hold each of those, and a list of two kinds. An `Array` takes two widths,
because the width of an array can change an answer, as the width that
`concat_arr` gives. A struct of the sample also holds
each field that the code reads by name, so a read of a field meets the field.
An answer that holds at every data type of the sample is not proved for every
data type, in the same way as the values that the evaluation tries for a hole.
"""

from __future__ import annotations

import ast
import itertools
import math
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import polars.datatypes.classes as polars_classes

from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe._rules.dtypes import DType, Rendering, RenderingError
from astrolabe._rules.node_evaluation import ColumnLeaf
from astrolabe._rules.options import RowForm

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence


class OpenType:
    """Give the data types that a rendering with holes stands for.

    A hole is a bare name in the rendering that is not a full data type:
    `Unchecked` stands for every data type of the sample, a family such as
    `NumericType` for its members, and a kind with open parameters such as
    `Datetime` for the data types of that kind.

    Example:
        >>> [dtype.render() for dtype in OpenType().samples("Duration")]
        ["Duration(time_unit='ns')", "Duration(time_unit='us')", "Duration(time_unit='ms')"]
        >>> OpenType().samples("Int8")
        (Int8,)
    """

    UNCHECKED: ClassVar[str] = "Unchecked"
    # A struct of the sample holds this field beside the fields that the code
    # reads. No code writes the name, so an answer that keeps it holds a field
    # that the open struct does not name.
    FIELD: ClassVar[str] = "__field__"
    FLAT: ClassVar[tuple[str, ...]] = (
        "Null",
        "Boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Int128",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "UInt128",
        "Float16",
        "Float32",
        "Float64",
        "Decimal(precision=10, scale=2)",
        "Decimal(precision=38, scale=0)",
        "String",
        "Binary",
        "Date",
        "Time",
        "Datetime(time_unit='ns', time_zone=None)",
        "Datetime(time_unit='us', time_zone='UTC')",
        "Datetime(time_unit='ms', time_zone=None)",
        "Duration(time_unit='ns')",
        "Duration(time_unit='us')",
        "Duration(time_unit='ms')",
        "Categorical",
        "Enum(categories=['a'])",
        "Enum(categories=['a', 'b'])",
    )
    LISTS: ClassVar[tuple[str, ...]] = ("List(Int64)", "List(String)")
    WIDTHS: ClassVar[tuple[int, int]] = (2, 3)
    WIDE_ENUM: ClassVar[str] = "Enum(categories=['a', 'b'])"
    LIMIT: ClassVar[int] = 256

    def __init__(self, fields: Sequence[str] = (), texts: Sequence[str] = ()) -> None:
        """Keep the names of the fields and the texts that the code reads.

        Each struct of the sample holds the fields. The wider `Enum` of the
        sample holds the texts as categories: the categories of an `Enum`
        with open parameters can hold any text, and a text that the code
        compares or writes must not make every sample refuse.

        Args:
            fields: The names of the fields.
            texts: The texts of the code.

        Example:
            >>> OpenType(["f"])._fields
            ('f',)
        """
        self._fields = tuple(fields)
        self._texts = tuple(texts)

    def samples(self, rendering: str) -> tuple[DType, ...]:
        """Give the data types that a rendering stands for.

        Args:
            rendering: The rendering, with holes or without.

        Returns:
            One data type for a rendering with no hole. No data type where
            the rendering does not parse, or where its holes give more than
            `LIMIT` data types.

        Example:
            >>> len(OpenType().samples("List(Unchecked)"))
            158
            >>> OpenType().samples("Struct({'a': Unchecked, 'b': Unchecked})")
            ()
            >>> OpenType().samples("List(")
            ()

            A member that fills a hole can give a rendering that polars does
            not write, here an array whose shape is not a tuple:

            >>> OpenType().samples("Array(Unchecked, 2)")
            ()
        """
        holes = self._holes(rendering)
        if holes is None:
            return ()
        options = [self._members(hole.id) for hole in holes]
        if math.prod(len(members) for members in options) > self.LIMIT:
            return ()
        dtypes: list[DType] = []
        for choice in itertools.product(*options):
            try:
                dtypes.append(Rendering().parse(self._filled(rendering, holes, choice)))
            except RenderingError:
                return ()
        return tuple(dtypes)

    def renderings(self, rendering: str) -> list[str]:
        """Give the rendering of each data type that a rendering stands for.

        A check that must hold at each data type of a kind, as the fit of
        an argument at a declared kind, reads the kind through these.

        Args:
            rendering: The rendering, with holes or without.

        Returns:
            The renderings, or the rendering alone where it stands for no
            data type that the sample holds.

        Example:
            >>> OpenType().renderings("Duration")
            ["Duration(time_unit='ns')", "Duration(time_unit='us')", "Duration(time_unit='ms')"]
            >>> OpenType().renderings("List(")
            ['List(']
        """
        return [sample.render() for sample in self.samples(rendering)] or [rendering]

    def _holes(self, rendering: str) -> list[ast.Name] | None:
        """Give the holes of a rendering, each with its place in the text.

        Args:
            rendering: The rendering.

        Returns:
            The holes, or `None` where the rendering does not parse.

        Example:
            >>> [hole.id for hole in OpenType()._holes("Struct({'a': Unchecked})")]
            ['Unchecked']
            >>> OpenType()._holes("List(") is None
            True
        """
        try:
            tree = ast.parse(rendering, mode="eval")
        except SyntaxError:
            return None
        names = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
        return [name for name in names if self._open(name, tree)]

    def _filled(self, rendering: str, holes: Sequence[ast.Name], choice: Sequence[str]) -> str:
        """Put the rendering of a data type in the place of each hole of a rendering.

        Args:
            rendering: The rendering with holes.
            holes: The name of each hole, with its place in the text.
            choice: The rendering that each hole takes.

        Returns:
            The rendering with no hole.

        Example:
            >>> tree = ast.parse("List(Unchecked)", mode="eval")
            >>> hole = [node for node in ast.walk(tree) if isinstance(node, ast.Name)][1]
            >>> OpenType()._filled("List(Unchecked)", [hole], ["Int8"])
            'List(Int8)'
        """
        text = rendering.encode()
        pairs = sorted(zip(holes, choice, strict=True), key=lambda pair: -pair[0].col_offset)
        for hole, member in pairs:
            end = checked_not_none(hole.end_col_offset)
            text = text[: hole.col_offset] + member.encode() + text[end:]
        return text.decode()

    def _open(self, node: ast.Name, tree: ast.Expression) -> bool:
        """Say whether a name of a rendering is a hole: a bare name that is not a data type.

        Args:
            node: The name.
            tree: The whole rendering, where a name can be the function of a call.

        Returns:
            Whether the name is a hole.

        Example:
            >>> tree = ast.parse("List(Unchecked)", mode="eval")
            >>> names = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
            >>> [name.id for name in names if OpenType()._open(name, tree)]
            ['Unchecked']
        """
        called = any(isinstance(call, ast.Call) and call.func is node for call in ast.walk(tree))
        if called:
            return False
        try:
            Rendering().parse(node.id)
        except RenderingError:
            return True
        return False

    def _members(self, name: str) -> list[str]:
        """Give the renderings of the sample that a hole stands for.

        Args:
            name: The name of the hole.

        Returns:
            The renderings. No rendering for a name that is not a class of
            polars. A `Struct` also stands for a struct that holds exactly
            the fields that the code reads, as polars casts a value that is
            not a struct only to a struct of one field.

        Example:
            >>> OpenType()._members("FloatType")
            ['Float16', 'Float32', 'Float64']
            >>> OpenType()._members("Nothing")
            []
        """
        sample = self._sample()
        if name == self.UNCHECKED:
            return sample
        family = getattr(polars_classes, name, None)
        if not isinstance(family, type) or not issubclass(family, pl.DataType):
            return []
        members = [text for text in sample if isinstance(Rendering().parse(text).polars(), family)]
        if family is pl.Struct and self._fields:
            members.extend(self._read_struct(element) for element in (*self._flat(), *self.LISTS))
        return members

    def _read_struct(self, inner: str) -> str:
        """Give the rendering of a struct that holds exactly the fields that the code reads.

        Args:
            inner: The rendering of the data type of every field.

        Returns:
            The rendering.

        Example:
            >>> OpenType(["f", "g"])._read_struct("Int8")
            "Struct({'f': Int8, 'g': Int8})"
        """
        fields = ", ".join(f"{name!r}: {inner}" for name in self._fields)
        return f"Struct({{{fields}}})"

    def _sample(self) -> list[str]:
        """Give the renderings of every data type of the sample.

        Where the code reads fields, a list and an array of each struct of
        the sample stand for the containers of structs that it can read.

        Returns:
            The renderings: each kind that holds no other type, and each
            nested kind over each of those and over two lists.

        Example:
            >>> sample = OpenType(["f"])._sample()
            >>> len(sample), sample[-1]
            (254, "Array(Struct({'__field__': List(String), 'f': List(String)}), shape=(3,))")
            >>> len(OpenType()._sample())
            158
        """
        flat = self._flat()
        inner = (*flat, *self.LISTS)
        lists = [f"List({element})" for element in inner]
        arrays = self._arrays(inner)
        structs = [self._struct(element) for element in inner]
        if not self._fields:
            return [*flat, *lists, *arrays, *structs]
        listed = [f"List({struct})" for struct in structs]
        return [*flat, *lists, *arrays, *structs, *listed, *self._arrays(structs)]

    def _arrays(self, inner: Sequence[str]) -> list[str]:
        """Give the renderings of an array of each data type, at each width of the sample.

        Args:
            inner: The renderings of the data types of the elements.

        Returns:
            The renderings.

        Example:
            >>> OpenType()._arrays(["Int8"])
            ['Array(Int8, shape=(2,))', 'Array(Int8, shape=(3,))']
        """
        return [f"Array({element}, shape=({width},))" for width in self.WIDTHS for element in inner]

    def _flat(self) -> tuple[str, ...]:
        """Give the renderings of each kind that holds no other type, with the texts of the code.

        Returns:
            The renderings.

        Example:
            >>> OpenType(texts=["z"])._flat()[-1]
            "Enum(categories=['a', 'b', 'z'])"
        """
        if not self._texts:
            return self.FLAT
        categories = list(dict.fromkeys(["a", "b", *self._texts]))
        wide = f"Enum(categories={categories!r})"
        return tuple(wide if text == self.WIDE_ENUM else text for text in self.FLAT)

    def _struct(self, inner: str) -> str:
        """Give the rendering of a struct of the sample, with each field that the code reads.

        The first field has a name that no code writes (`FIELD`), so the struct
        is never empty and an answer can show that it keeps a field that the
        open struct does not name.

        Args:
            inner: The rendering of the data type of every field.

        Returns:
            The rendering.

        Example:
            >>> OpenType(["f"])._struct("Int8")
            "Struct({'__field__': Int8, 'f': Int8})"
        """
        fields = "".join(f", {name!r}: {inner}" for name in self._fields)
        return f"Struct({{{self.FIELD!r}: {inner}{fields}}})"


class OpenParameters:
    """Leave open each parameter of a data type that is not a data type itself.

    The answer of a code at a sample of a kind with open parameters, as
    `Decimal(38, 0)` for a bare `Decimal`, can depend on the parameters. The
    sample holds a few of them only. Where some refuse and some answer, the
    answer holds its kinds and leaves the other parameters open.

    Example:
        >>> OpenParameters().opened(DType("List", (DType("Decimal", (38, 0)),))).render()
        'List(Decimal)'
    """

    def kind(self, dtype: DType) -> str:
        """Give the kinds of a data type, as a text: the data type with its parameters taken away.

        Args:
            dtype: The data type.

        Returns:
            The text.

        Example:
            >>> OpenParameters().kind(DType("Array", (DType("Datetime", ("ms", None)), 2)))
            'Array(Datetime)'
        """
        inner = [self.kind(part) for part in self._parts(dtype)]
        return dtype.name if not inner else f"{dtype.name}({', '.join(inner)})"

    def opened(self, dtype: DType) -> DType:
        """Give a data type with each parameter that is not a data type left open.

        Args:
            dtype: The data type.

        Returns:
            The data type: a kind of other data types keeps them, opened in
            turn, and any other kind with parameters is the kind alone.

        Example:
            >>> OpenParameters().opened(DType("Decimal", (38, 0)))
            Decimal
            >>> OpenParameters().opened(DType("Struct", ((("a", DType("Decimal", (38, 0))),),)))
            Struct({'a': Decimal})
        """
        if dtype.name == "Struct":
            fields = tuple((name, self.opened(inner)) for name, inner in dtype.fields())
            return DType("Struct", (fields,))
        if all(isinstance(part, DType) for part in dtype.args):
            return DType(dtype.name, tuple(self.opened(part) for part in self._parts(dtype)))
        return DType(dtype.name)

    def _parts(self, dtype: DType) -> list[DType]:
        """Give the data types that a data type holds: its inner type or the types of its fields.

        Args:
            dtype: The data type.

        Returns:
            The data types.

        Example:
            >>> OpenParameters()._parts(DType("Struct", ((("a", DType("Int8")),),)))
            [Int8]
        """
        if dtype.name == "Struct":
            return [inner for _, inner in dtype.fields()]
        return [part for part in dtype.args if isinstance(part, DType)]


class Generalization:
    """Give what several data types share, with `Unchecked` where they differ.

    Example:
        >>> lists = [DType("List", (DType("Int8"),)), DType("List", (DType("String"),))]
        >>> Generalization().of(lists).render()
        'List(Unchecked)'
    """

    UNCHECKED: ClassVar[DType] = DType("Unchecked")

    def of(self, dtypes: Sequence[DType]) -> DType:
        """Give the data type that every data type of a list fits.

        Two data types of one kind share the parts where they agree. A part
        that is a data type, as the inner type of a `List` or a field of a
        `Struct`, takes the data type that the parts share. Any other part that
        differs, as the unit of a `Datetime`, leaves the parameters of the kind
        open: the shared data type is the kind alone, such as `Datetime`. Two
        kinds share only `Unchecked`. A struct that holds the field of a sample
        struct (`OpenType.FIELD`) holds fields that the code does not name, so
        it is `Unchecked` too.

        Args:
            dtypes: The data types, at least one.

        Returns:
            The shared data type.

        Example:
            >>> Generalization().of([DType("Int8"), DType("Int8")])
            Int8
            >>> durations = [DType("Duration", ("ms",)), DType("Duration", ("us",))]
            >>> Generalization().of(durations)
            Duration
            >>> Generalization().of([DType("Int8"), DType("String")])
            Unchecked
            >>> sampled = DType("Struct", (((OpenType.FIELD, DType("Int8")),),))
            >>> Generalization().of([DType("List", (sampled,))])
            List(Unchecked)
        """
        return self.named(self.shared(dtypes))

    def shared(self, dtypes: Sequence[DType]) -> DType:
        """Give the data type that every data type of a list fits, the fields of a sample kept.

        Code that compares an answer with a sample of its own needs the
        fields of the sample, so it names them later with `named`.

        Args:
            dtypes: The data types, at least one.

        Returns:
            The shared data type.

        Example:
            >>> Generalization().shared([DType("Int8"), DType("Int16")])
            Unchecked
        """
        first = dtypes[0]
        if all(dtype == first for dtype in dtypes):
            return first
        if any(dtype.name != first.name for dtype in dtypes):
            return self.UNCHECKED
        parts = self._parts(dtypes)
        return DType(first.name) if parts is None else DType(first.name, parts)

    def named(self, dtype: DType) -> DType:
        """Give a data type with `Unchecked` in place of each struct that holds a sample field.

        The field of a sample stands for fields that the code does not name,
        so such a struct is no type that a column can hold.

        Args:
            dtype: The data type.

        Returns:
            The data type, with its other parts as they are.

        Example:
            >>> sampled = DType("Struct", (((OpenType.FIELD, DType("Int8")),),))
            >>> Generalization().named(sampled)
            Unchecked
            >>> Generalization().named(DType("Struct", ((("x", sampled),),)))
            Struct({'x': Unchecked})
        """
        if dtype.name == "Struct":
            fields = dtype.fields()
            if any(name == OpenType.FIELD for name, _ in fields):
                return self.UNCHECKED
            return DType("Struct", (tuple((name, self.named(inner)) for name, inner in fields),))
        args = tuple(self.named(part) if isinstance(part, DType) else part for part in dtype.args)
        return DType(dtype.name, args)

    def _parts(self, dtypes: Sequence[DType]) -> tuple[object, ...] | None:
        """Give what the parts of several data types of one kind share, place by place.

        Args:
            dtypes: The data types, of one kind.

        Returns:
            The shared parts, or `None` where the data types have other
            numbers of parts, or where two parts differ and are not data
            types or fields.

        Example:
            >>> lists = [DType("List", (DType("Int8"),)), DType("List", (DType("Date"),))]
            >>> Generalization()._parts(lists)
            (Unchecked,)
            >>> stamps = [DType("Datetime", ("ms", None)), DType("Datetime")]
            >>> Generalization()._parts(stamps) is None
            True
        """
        if len({len(dtype.args) for dtype in dtypes}) != 1:
            return None
        parts: list[object] = []
        for place in range(len(dtypes[0].args)):
            part = self._part([dtype.args[place] for dtype in dtypes])
            if part is None:
                return None
            parts.append(part)
        return tuple(parts)

    def _part(self, parts: Sequence[object]) -> object:
        """Give what several parts of one place share.

        Args:
            parts: The part of each data type at one place.

        Returns:
            The shared part, or `None` where the parts differ and are not
            data types or fields.

        Example:
            >>> Generalization()._part(["ms", "us"]) is None
            True
            >>> Generalization()._part([DType("Int8"), DType("Int16")])
            Unchecked
        """
        first = parts[0]
        if all(part == first for part in parts):
            return first
        if all(isinstance(part, DType) for part in parts):
            return self.shared([part for part in parts if isinstance(part, DType)])
        return self._fields(parts)

    @propagates(None)
    def _fields(self, parts: Sequence[object]) -> object:
        """Give what the fields of several structs share, where the names of the fields agree.

        Args:
            parts: The fields of each struct, as pairs of a name and a data type.

        Returns:
            The shared fields, or `None` where the parts are not fields or
            where the names differ.

        Example:
            >>> one = (("a", DType("Int8")),)
            >>> two = (("a", DType("String")),)
            >>> Generalization()._fields([one, two])
            (('a', Unchecked),)
            >>> Generalization()._fields([one, (("b", DType("Int8")),)]) is None
            True
        """
        structs = [not_none(self._pairs(part)) for part in parts]
        names = {tuple(name for name, _ in struct) for struct in structs}
        not_falsy(len(names) == 1)
        fields: list[tuple[str, DType]] = []
        for place, name in enumerate(names.pop()):
            fields.append((name, self.shared([struct[place][1] for struct in structs])))
        return tuple(fields)

    def _pairs(self, part: object) -> tuple[tuple[str, DType], ...] | None:
        """Read the fields of a struct: pairs of a name and a data type.

        Args:
            part: The part of a data type.

        Returns:
            The fields, or `None` where the part is not a tuple of fields.

        Example:
            >>> Generalization()._pairs((("a", DType("Int8")),))
            (('a', Int8),)
            >>> Generalization()._pairs("ms") is None
            True
            >>> Generalization()._pairs((("a",),)) is None
            True
            >>> Generalization()._pairs((("a", "ms"),)) is None
            True
        """
        if not isinstance(part, tuple):
            return None
        pairs: list[tuple[str, DType]] = []
        for pair in part:
            if not isinstance(pair, tuple) or len(pair) != len(("name", "type")):
                return None
            name, dtype = pair
            if not isinstance(dtype, DType):
                return None
            pairs.append((str(name), dtype))
        return tuple(pairs)


class LeafChoices:
    """Give each choice of the data types of the columns that a code reads.

    A column of a full data type takes it in every choice. A column with holes
    that the code reads takes each data type that its rendering stands for. A
    column with holes that the code does not read, or whose rendering stands
    for no data type, has no leaf, so the rules give `Unchecked` where the
    code reads it. A column of a lazy frame whose stored type differs from
    the type of its data takes both.

    Example:
        >>> choices = LeafChoices().of({"a": "Int8", "u": "Unchecked"}, {"a", "u"})
        >>> len(choices), choices[0]["a"].dtype
        (158, Int8)
    """

    LIMIT: ClassVar[int] = 256

    def __init__(self, planned: Mapping[str, str] | None = None) -> None:
        """Keep the stored type of each column of a lazy frame that has one of its own.

        Args:
            planned: The rendering of the stored type of each column of a
                lazy frame where it differs from the type of its data, by its name.

        Example:
            >>> (choice,) = LeafChoices({"a": "Float32"}).of({"a": "Float64"}, ())
            >>> choice["a"].planned
            Float32
        """
        self._planned = dict(planned or {})

    def of(
        self,
        renderings: Mapping[str, str],
        reads: Collection[str],
        fields: Sequence[str] = (),
        restricted: Mapping[str, tuple[DType, ...]] | None = None,
        texts: Sequence[str] = (),
    ) -> list[dict[str, ColumnLeaf]]:
        """Give each choice of the leaves of the columns.

        Args:
            renderings: The rendering of the data type of each column, by its name.
            reads: The names of the columns that the code reads.
            fields: The names of the fields of a struct that the code reads.
            restricted: The data types that a column with holes takes in
                place of its whole sample, by its name, as the data types of
                it that a selector picks.
            texts: The texts of the code, which the wider `Enum` of the
                sample holds as categories.

        Returns:
            The choices, each with a leaf for each column, by its name. No
            choice where the choices are more than `LIMIT`.

        Example:
            >>> LeafChoices().of({"a": "Int8"}, set())[0]["a"]
            ColumnLeaf(dtype=Int8, form=<RowForm.PER_ROW: 'per row'>, nulls=False, planned=None)
            >>> LeafChoices().of({"u": "Unchecked", "v": "Unchecked"}, {"u", "v"})
            []
        """
        fixed: dict[str, ColumnLeaf] = {}
        options: dict[str, tuple[DType, ...]] = {}
        for name, rendering in renderings.items():
            closed = self.closed(rendering)
            samples = OpenType(fields, texts).samples(rendering) if name in reads else ()
            samples = (restricted or {}).get(name, samples)
            stored = self._planned.get(name)
            planned = None if stored is None else self.closed(stored)
            if closed is not None:
                fixed[name] = ColumnLeaf(closed, RowForm.PER_ROW, planned=planned)
            elif samples:
                options[name] = samples
        if math.prod(len(samples) for samples in options.values()) > self.LIMIT:
            return []
        return self._chosen(fixed, options)

    def _chosen(
        self,
        fixed: Mapping[str, ColumnLeaf],
        options: Mapping[str, tuple[DType, ...]],
    ) -> list[dict[str, ColumnLeaf]]:
        """Give each choice of one data type for each column with holes, beside the other leaves.

        Args:
            fixed: The leaf of each column of a full data type.
            options: The data types of each column with holes.

        Returns:
            The choices.

        Example:
            >>> options = {"d": (DType("Date"), DType("Time"))}
            >>> [choice["d"].dtype for choice in LeafChoices()._chosen({}, options)]
            [Date, Time]
        """
        choices: list[dict[str, ColumnLeaf]] = []
        for choice in itertools.product(*options.values()):
            chosen = dict(fixed)
            for name, dtype in zip(options, choice, strict=True):
                chosen[name] = ColumnLeaf(dtype, RowForm.PER_ROW)
            choices.append(chosen)
        return choices

    def closed(self, rendering: str) -> DType | None:
        """Give the data type of a rendering with no hole.

        Args:
            rendering: The rendering.

        Returns:
            The data type, or `None` where the rendering has a hole.

        Example:
            >>> LeafChoices().closed("Int8"), LeafChoices().closed("Unchecked")
            (Int8, None)
        """
        try:
            return Rendering().parse(rendering)
        except RenderingError:
            return None
