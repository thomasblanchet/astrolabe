"""Write the type of a declared expression as a mypy type, and read it back.

A user declares the type of an expression that a function takes or returns,
for example with `column("price", pl.Float64)` or `typed_expr(...)`. The engine
holds such a declaration as four parts: the columns that it reads, its map, its
output name and its character. The map gives, for the kinds of the read
columns, the data type of the result. A kind is a data type with its parameters
taken away, for example `List` or `Int64`. The character says how the
expression treats rows: row by row, one value for each group, or one fixed
value. The declared type lives only in its mypy type, so the whole map must
travel inside it.

The head class holds the character: `TypedExpr`, `TypedAgg` or `TypedLit`. Its
first argument, `Reads[...]`, holds one `Column[name, region]` for each read
column. A region is a set of kinds. The second argument, `Column[name, body]`,
holds the output name and the map. The body is one outcome if one outcome covers
the whole map, and `Rows[...]` otherwise. An outcome is the rule that gives the
data type at one cell.

mypy compares two types by their form alone. This codec therefore writes one
map in one way only. The cells that share one outcome become `Row[...]` entries,
each with one region for each read. If the regions of the group form one exact
rectangle, the group is one row. Otherwise a fixed method covers the cells with
rectangles, and it reads only the cells.

A region is written by what it holds. Every kind is `EveryDType`. The exact
members of one polars family are the base class of that family, for example
`NumericType`, because that is the name that polars itself gives the group. One
kind is its polars class, and any other region is the sorted union of those
classes. The declaration that gives back one column unchanged is written in
short form, as `TypedColumn[name, region]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from itertools import product

from mypy.types import Instance, UnionType, get_proper_type

import astrolabe.expr as markers
from astrolabe.expr import TypedAgg, TypedColumn, TypedExpr, TypedLit
from astrolabe.frame import Column
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.marker_minter import MarkerMinter, ProperReads
from astrolabe._hooks.encoding.rendering_codec import (
    ConstDtypeCodec,
    ConstDtypeReads,
    DtypeSpelling,
)
from astrolabe._hooks.expression.character import (
    AGGREGATING,
    ELEMENTWISE,
    LITERAL,
    Character,
)
from astrolabe._hooks.expression.kind import KindRoster
from astrolabe._hooks.expression.outcome import Axis, Const, Outcome, Wrap
from astrolabe._hooks.expression.quadruple import ExpressionType
from astrolabe._hooks.expression.table import Table
from astrolabe._hooks.propagate import not_falsy, not_none, propagates

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from mypy.types import Type as MypyType

    from astrolabe._hooks.expression.table import Cell


@dataclass(frozen=True)
class TableArea:
    """One rectangle of a table: a region on each axis, and the one outcome at every cell inside it.

    The codec writes each area as one `Row[...]` entry. The reader turns each
    row back into the cells of its area.

    Attributes:
        regions: One set of kinds for each read column, in the axis order of the
            table.
        outcome: The outcome at every cell that the regions cover.

    Example:
        >>> TableArea((frozenset({"Int64"}),), Axis("a")).outcome
        Axis(column='a')
    """

    regions: tuple[frozenset[str], ...]
    outcome: Outcome


class _RectangleCover:
    """Cover the cells of one outcome with rectangles, in the same way every time.

    A rectangle has one set of kinds on each axis, and every cell of their
    product must be in the group. The cover takes the lowest cell that is not
    yet covered as a seed. It grows the seed along each axis in turn, and it
    adds each kind in sorted order if the rectangle stays inside the group. The
    same cells therefore always give the same cover, whatever code built the
    table. Two rectangles can overlap. The shared cells have the same outcome in
    both, and the overlap lets a cross of cells take two rows and not one row
    for each cell.

    The cover is not always the smallest cover. It must be short, and it must
    depend on the cells alone.

    Example:
        >>> cross = frozenset({("Int64", "String"), ("String", "String"),
        ...                    ("String", "Int64")})
        >>> len(_RectangleCover(cross).regions())
        2
    """

    def __init__(self, cells: frozenset[Cell]) -> None:
        """Keep the cells to cover.

        Args:
            cells: The cells that share one outcome. There is at least one.
        """
        self._cells = cells
        self._axes = tuple(range(len(next(iter(cells)))))

    def regions(self) -> list[tuple[frozenset[str], ...]]:
        """Give the rectangles of the cover.

        Returns:
            The rectangles, each one as a set of kinds for each axis, in the
            order in which the seeds were taken.

        Example:
            >>> block = frozenset({("Int64", "Int64"), ("Int64", "String")})
            >>> [sorted(sorted(r) for r in one) for one in
            ...  _RectangleCover(block).regions()]
            [[['Int64'], ['Int64', 'String']]]
        """
        remaining = set(self._cells)
        found: list[tuple[frozenset[str], ...]] = []
        while remaining:
            grown = self._grown(min(remaining))
            found.append(grown)
            remaining -= set(product(*grown))
        return found

    def _grown(self, seed: Cell) -> tuple[frozenset[str], ...]:
        """Grow one seed cell into the largest rectangle that this method finds inside the group.

        The method widens one axis at a time, in axis order. On each axis, it
        tries each kind that the group uses there, in sorted order. If the group
        holds every cell of the wider rectangle, the method adds the kind.

        Args:
            seed: The cell that the rectangle starts from.

        Returns:
            One set of kinds for each axis.

        Example:
            >>> block = frozenset({("Int64", "Int64"), ("Int64", "String")})
            >>> sorted(sorted(r) for r in _RectangleCover(block)._grown(("Int64", "Int64")))
            [['Int64'], ['Int64', 'String']]

            A diagonal holds no rectangle wider than one cell:

            >>> diagonal = frozenset({("Int64", "Int64"), ("String", "String")})
            >>> sorted(sorted(r) for r in _RectangleCover(diagonal)._grown(("Int64", "Int64")))
            [['Int64'], ['Int64']]
        """
        regions = [{kind} for kind in seed]
        for axis in self._axes:
            used = {cell[axis] for cell in self._cells}
            for kind in sorted(used - regions[axis]):
                if self._holds(regions, axis, kind):
                    regions[axis].add(kind)
        return tuple(frozenset(region) for region in regions)

    def _holds(self, regions: list[set[str]], axis: int, kind: str) -> bool:
        """Tell whether the rectangle stays inside the group if one axis gets one more kind.

        Args:
            regions: The rectangle so far.
            axis: The axis to widen.
            kind: The kind to add.

        Returns:
            True if the group holds every cell of the wider rectangle.

        Example:
            >>> block = frozenset({("Int64", "Int64"), ("Int64", "String")})
            >>> _RectangleCover(block)._holds([{"Int64"}, {"Int64"}], 1, "String")
            True
            >>> _RectangleCover(block)._holds([{"Int64"}, {"Int64"}], 0, "String")
            False
        """
        trial = [set(region) for region in regions]
        trial[axis].add(kind)
        return all(cell in self._cells for cell in product(*trial))


class TableAreas:
    """Split a table into the areas that its type writes, in one fixed way.

    A table can have many cells, and one `Row[...]` for each cell makes a long
    type. The cells of a table usually fall in bands that share one outcome, so
    this class writes each band as few rows as possible. It groups the cells by
    outcome, in the order that `Table.spelling` gives. A group whose kinds on
    each axis form an exact product becomes one area. Any other group becomes
    the rectangles of `_RectangleCover`. The split depends only on the cells, so
    equal tables always give equal areas.

    Example:
        >>> table = Table.from_cells(("a",), {("Int64",): Axis("a")})
        >>> TableAreas().of(table)
        (TableArea(regions=(frozenset({'Int64'}),), outcome=Axis(column='a')),)
    """

    def of(self, table: Table) -> tuple[TableArea, ...]:
        """Split one table into its areas.

        Args:
            table: The table to split.

        Returns:
            The areas, in the order that `Table.spelling` gives. Together, they
            cover exactly the cells of the table.
        """
        by_spelling = {outcome.spelling(): outcome for _, outcome in table.cells}
        areas: list[TableArea] = []
        for label, cells in table.spelling():
            areas.extend(self._grouped(by_spelling[label], cells))
        return tuple(areas)

    def _grouped(self, outcome: Outcome, cells: tuple[Cell, ...]) -> list[TableArea]:
        """Split the cells of one outcome into areas.

        The method first takes the kinds that the group uses on each axis. The
        product of these sets always holds every cell of the group. If it has
        as many cells as the group, it holds no other cell, and the group is
        one area. Otherwise the rectangles of `_RectangleCover` give the areas.

        Args:
            outcome: The outcome of the group.
            cells: The cells of the group, sorted.

        Returns:
            One area if the cells form an exact product, or one area for each
            rectangle of the cover otherwise.

        Example:
            >>> band = (("Int64",), ("String",))
            >>> [sorted(r) for r in TableAreas()._grouped(Axis("a"), band)[0].regions]
            [['Int64', 'String']]

            A diagonal is not an exact product, so each cell is one area:

            >>> diagonal = (("Int64", "Int64"), ("String", "String"))
            >>> len(TableAreas()._grouped(Axis("a"), diagonal))
            2
        """
        width = len(cells[0])
        projections = tuple(frozenset(cell[axis] for cell in cells) for axis in range(width))
        size = 1
        for projection in projections:
            size *= len(projection)
        if size == len(cells):
            return [TableArea(projections, outcome)]
        cover = _RectangleCover(frozenset(cells))
        return [TableArea(regions, outcome) for regions in cover.regions()]


class RegionCodec:
    """Build the mypy type for one region, a set of kinds.

    A table names a region on each axis of each row, and the region of each read
    column also shows in `Reads[...]`. The type must be short, and one region
    must always give the same type. The whole set of kinds is `EveryDType`. The
    exact members of one polars family, for example every float kind, are the
    base class of that family, here `FloatType`. One kind is its own polars
    class. Any other set is the sorted union of the classes of its kinds.

    Example:
        >>> fixture = new_type_fixture()
        >>> codec = new_expr_codec(fixture)
        >>> codec._regions.built(frozenset({"Int64"}))
        polars.datatypes.classes.Int64
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the minter that builds the types.

        Args:
            minter: The marker minter for the current phase.
        """
        self._minter = minter
        self._spelling = DtypeSpelling()

    @propagates(None)
    def built(self, region: frozenset[str]) -> MypyType | None:
        """Build the type for one region.

        The method tries the names in order: `EveryDType` for every kind, then
        the base class of a family, then the class of each kind.

        Args:
            region: The kinds. The set is not empty.

        Returns:
            `EveryDType`, the base class of a family, the class of one kind, or
            the sorted union of the classes. None if a kind has no polars class.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_expr_codec(fixture)
            >>> codec._regions.built(frozenset({"Int64", "String"}))
            polars.datatypes.classes.Int64 | polars.datatypes.classes.String
            >>> codec._regions.built(frozenset({"Nope"})) is None
            True
        """
        if region == frozenset(KindRoster().names()):
            return self._minter.marker(markers.EveryDType, [])
        family = self._family(region)
        if family is not None:
            return family
        kinds: list[MypyType] = []
        for kind in sorted(region):
            kinds.append(not_none(self._kind(kind)))
        if len(kinds) == 1:
            return kinds[0]
        return UnionType(kinds)

    @propagates(None)
    def _kind(self, kind: str) -> MypyType | None:
        """Build the type of one kind of a region.

        Args:
            kind: A kind.

        Returns:
            The class of the kind, or None if the kind has no polars class.

        Example:
            >>> codec = new_expr_codec(new_type_fixture())
            >>> codec._regions._kind("Int8")
            polars.datatypes.classes.Int8
        """
        fullname = not_none(self._spelling.fullname(kind))
        return self._minter.resolved(fullname)

    def _family(self, region: frozenset[str]) -> MypyType | None:
        """Build the base class of the family whose members are exactly the region.

        A region of one kind is left to the class of that kind. That class
        names the same set, and it is the name that a reader expects.

        Args:
            region: The kinds.

        Returns:
            The base class of the family, or None if the region has one kind,
            or if no family has exactly these members.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_expr_codec(fixture)
            >>> codec._regions._family(frozenset({"Float16", "Float32", "Float64"}))
            polars.datatypes.classes.FloatType
            >>> codec._regions._family(frozenset({"Float64"})) is None
            True
        """
        if len(region) < 2:
            return None
        for base in sorted(KindRoster().group_bases(), key=lambda one: one.__name__):
            if frozenset(KindRoster().family(base)) == region:
                return self._minter.resolved(Target.of(base).fullname)
        return None


class RegionReads:
    """Read the type of a region back into its set of kinds.

    This is the reading half of `RegionCodec`. It builds no type, so it needs no
    access to the plugin.

    Example:
        >>> from mypy.types import UninhabitedType
        >>> RegionReads().read(UninhabitedType()) is None
        True
    """

    def __init__(self) -> None:
        """Set up the reader of instances."""
        self._reads = ProperReads()

    def read(self, typ: MypyType) -> frozenset[str] | None:
        """Read one type as a region.

        A union reads as the kinds of all its items together.

        Args:
            typ: The type to read.

        Returns:
            The kinds, or None if the type, or one item of the union, is not a
            region.

        Example:
            >>> from mypy.types import UninhabitedType
            >>> fixture = new_type_fixture()
            >>> mixed = UnionType([
            ...     new_marker_minter(fixture).resolved(
            ...         "polars.datatypes.classes.Int64"
            ...     ),
            ...     UninhabitedType(),
            ... ])
            >>> RegionReads().read(mixed) is None
            True
        """
        proper = get_proper_type(typ)
        if not isinstance(proper, UnionType):
            return self._single(proper)
        kinds: frozenset[str] = frozenset()
        for item in proper.items:
            one = self._single(get_proper_type(item))
            if one is None:
                return None
            kinds |= one
        return kinds

    def _single(self, proper: MypyType) -> frozenset[str] | None:
        """Read one type that is not a union as a region.

        The method reads an instance of a class as the kinds that the class
        names. It tries `EveryDType`, then a family, then one kind.

        Args:
            proper: The type to read, already expanded.

        Returns:
            The kinds, or None for a type that is not a region.

        Example:
            >>> from mypy.types import Instance
            >>> fixture = new_type_fixture()
            >>> RegionReads()._single(Instance(fixture.oi, [])) is None
            True
        """
        instance = self._reads.instance(proper)
        if instance is None:
            return None
        readers = (self._every, self._family, self._one_kind)
        for reader in readers:
            kinds = reader(instance)
            if kinds is not None:
                return kinds
        return None

    def _every(self, instance: Instance) -> frozenset[str] | None:
        """Read the `EveryDType` marker as every kind that polars defines.

        Args:
            instance: The instance to read.

        Returns:
            Every kind, or None for another class.

        Example:
            >>> fixture = new_type_fixture()
            >>> RegionReads()._every(
            ...     new_marker_minter(fixture).resolved(
            ...         "polars.datatypes.classes.Int64"
            ...     )
            ... ) is None
            True
            >>> every = new_marker_minter(fixture).resolved("astrolabe.expr.EveryDType")
            >>> sorted(RegionReads()._every(every))[:3]
            ['Array', 'Binary', 'Boolean']
        """
        if self._reads.matches(instance, markers.EveryDType):
            return frozenset(KindRoster().names())
        return None

    def _family(self, instance: Instance) -> frozenset[str] | None:
        """Read the base class of a family back into its members.

        A family of one kind does not read here. The class of that kind names
        it.

        Args:
            instance: The instance to read.

        Returns:
            The kinds of the family, or None for a class that does not name a
            family of more than one kind.

        Example:
            >>> minter = new_marker_minter(new_type_fixture())
            >>> floats = minter.resolved("polars.datatypes.classes.FloatType")
            >>> sorted(RegionReads()._family(floats))
            ['Float16', 'Float32', 'Float64']
            >>> objects = minter.resolved("polars.datatypes.classes.ObjectType")
            >>> RegionReads()._family(objects) is None
            True
            >>> RegionReads()._family(
            ...     minter.resolved("polars.datatypes.classes.Int64")
            ... ) is None
            True
        """
        name = instance.type.name.rsplit(".", 1)[-1]
        for base in KindRoster().group_bases():
            if base.__name__ != name:
                continue
            members = frozenset(KindRoster().family(base))
            if len(members) < 2:
                return None
            return members
        return None

    def _one_kind(self, instance: Instance) -> frozenset[str] | None:
        """Read the class of one kind.

        The tables come from measurements of polars over some of its kinds. A
        cell can still sit at any kind that polars defines, for example `List`,
        `Array` or `Struct`. `Unchecked` is also a kind here: it is the data
        type of a column whose name is known and whose data type is not.

        Args:
            instance: The instance to read.

        Returns:
            A set with the one kind, or None for a class that is not a kind.

        Example:
            >>> from mypy.types import Instance
            >>> fixture = new_type_fixture()
            >>> RegionReads()._one_kind(Instance(fixture.oi, [])) is None
            True
        """
        kind = instance.type.name.rsplit(".", 1)[-1]
        if kind not in KindRoster().names() and kind != "Unchecked":
            return None
        return frozenset({kind})


class OutcomeCodec:
    """Build the mypy type for one outcome, and for a whole table.

    An outcome is the rule that gives the data type at one cell of a table. It
    can be a fixed data type, the data type of a read column, or a rule over
    other outcomes. Each form has its own marker, for example `Same[...]` for
    the data type of a read column, or `Wrap[...]` for a `List` around another
    outcome. Some outcomes hold a whole table of their own, so this class also
    writes tables: one outcome where one outcome covers the table, or
    `Rows[...]` with one `Row[...]` for each area.

    Example:
        >>> fixture = new_type_fixture()
        >>> codec = new_expr_codec(fixture)
        >>> codec._outcomes.built(Axis("a"))
        astrolabe.expr.Same[Literal['a']]
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the minter and the codec for regions.

        Args:
            minter: The marker minter for the current phase.
        """
        self._minter = minter
        self._spelling = DtypeSpelling()
        self._regions = RegionCodec(minter)
        self._areas = TableAreas()

    @propagates(None)
    def body(self, table: Table) -> MypyType | None:
        """Build the type of a table in its shortest form: one outcome, or rows.

        A table in the output of an expression and a table inside an outcome are
        written in the same way. This method is therefore here, beside the
        outcome codec that it calls for each row.

        Args:
            table: The table to write.

        Returns:
            The one outcome if the table has one area, and `Rows[...]`
            otherwise. None if a part has no type.
        """
        areas = self._areas.of(table)
        if len(areas) == 1:
            return not_none(self.built(areas[0].outcome))
        return not_none(self.rows(table))

    @propagates(None)
    def rows(self, table: Table) -> MypyType | None:
        """Build the type of a table as rows, even if one outcome covers it.

        A table inside an outcome, the table for the aggregation context and the
        entries of an open chain have no `Reads[...]` part that gives their
        regions. So each of their rows names its own regions.

        Args:
            table: The table to write.

        Returns:
            The `Rows[...]` type, each row with its own regions. None if a part
            has no type.
        """
        rows: list[MypyType] = []
        for area in self._areas.of(table):
            parts = [not_none(self._regions.built(region)) for region in area.regions]
            parts.append(not_none(self.built(area.outcome)))
            rows.append(not_none(self._minter.marker(markers.Row, parts)))
        return self._minter.marker(markers.Rows, rows)

    def built(self, outcome: Outcome) -> MypyType | None:
        """Build the type of one outcome.

        The class of the outcome selects the builder.

        Args:
            outcome: The outcome from the engine.

        Returns:
            The marker type, or None for a data type that has no polars class.
        """
        builders: dict[type[Outcome], Callable[[Any], MypyType | None]] = {
            Const: self._built_const,
            Axis: self._built_axis,
            Wrap: self._built_wrap,
        }
        return builders[type(outcome)](outcome)

    def _built_const(self, outcome: Const) -> MypyType | None:
        """Build the type of a fixed data type.

        Args:
            outcome: The fixed outcome.

        Returns:
            The type of the data type, or None for a rendering that this codec
            does not know. A rendering is the text that polars shows for a data
            type.
        """
        return self._dtype(outcome.dtype)

    def _built_axis(self, outcome: Axis) -> MypyType | None:
        """Build the type of the outcome that gives the data type of one read column back.

        Args:
            outcome: The outcome that names the column.

        Returns:
            The `Same[column]` type.
        """
        return self._minter.marker(markers.Same, [self._minter.name(outcome.column)])

    @propagates(None)
    def _built_wrap(self, outcome: Wrap) -> MypyType | None:
        """Build the type of a data type around another outcome, for example a `List` of it.

        Args:
            outcome: The outcome to write.

        Returns:
            The `Wrap[constructor, inner]` type, or None if a part has no type.
        """
        inner = not_none(self.built(outcome.inner))
        constructor = not_none(self._dtype(outcome.constructor))
        return self._minter.marker(markers.Wrap, [constructor, inner])

    def _dtype(self, text: str) -> MypyType | None:
        """Build the type of one data type rendering.

        Args:
            text: The rendering: a bare kind, `Unchecked`, or a data type with
                parameters.

        Returns:
            The type, or None for a rendering that this codec does not know.
        """
        return ConstDtypeCodec(self._minter).built(text)


class OutcomeReads:
    """Read the type of an outcome, or of a table, back into the engine value.

    This is the reading half of `OutcomeCodec`. The full name of the marker
    class selects the reader. A class that is not an outcome marker reads as a
    fixed data type. The reader builds no type, so it needs no access to the plugin.

    Example:
        >>> from mypy.types import UninhabitedType
        >>> OutcomeReads().read(UninhabitedType()) is None
        True
    """

    def __init__(self) -> None:
        """Set up the readers of instances and regions."""
        self._reads = ProperReads()
        self._regions = RegionReads()

    @propagates(None)
    def rows_table(self, axes: tuple[str, ...], instance: Instance) -> Table | None:
        """Read a `Rows[...]` type as a table over the given axes.

        Args:
            axes: The axes of the table, in axis order.
            instance: The `Rows[...]` instance.

        Returns:
            The table with the cells of every row, or None if a row is in a
            wrong form.
        """
        cells: dict[Cell, Outcome] = {}
        for argument in instance.args:
            row = not_none(self._reads.instance(argument))
            not_falsy(self._reads.matches(row, markers.Row))
            not_falsy(len(row.args) == len(axes) + 1)
            regions = tuple(not_none(self._regions.read(region)) for region in row.args[:-1])
            outcome = not_none(self.read(row.args[-1]))
            cells.update(self.product_cells(regions, outcome))
        return Table.from_cells(axes, cells)

    def product_cells(
        self,
        regions: tuple[frozenset[str], ...],
        outcome: Outcome,
    ) -> Iterable[tuple[Cell, Outcome]]:
        """Expand the regions of one area into its cells.

        Args:
            regions: One set of kinds for each axis.
            outcome: The outcome at every cell of the area.

        Returns:
            Each cell of the product, with the outcome.

        Example:
            >>> pairs = OutcomeReads().product_cells(
            ...     (frozenset({"Int64"}),), Axis("a")
            ... )
            >>> sorted(pairs)
            [(('Int64',), Axis(column='a'))]
        """
        ordered = [sorted(region) for region in regions]
        return ((tuple(cell), outcome) for cell in product(*ordered))

    def read(self, typ: MypyType) -> Outcome | None:
        """Read one type as an outcome.

        The full name of the marker class selects the reader. A class that is
        not an outcome marker reads as a fixed data type. A fixed data type can
        be any kind that polars defines, not only a kind that the measurements
        used as input. For example, `add` on a `Boolean` column gives `UInt32`.
        A marker with parameters, for example `Datetime[...]`, reads back as its
        rendering.

        Args:
            typ: The type to read.

        Returns:
            The outcome, or None for a type that is not an outcome.

        Example:
            >>> from mypy.types import Instance
            >>> fixture = new_type_fixture()
            >>> OutcomeReads().read(Instance(fixture.oi, [])) is None
            True
        """
        instance = self._reads.instance(typ)
        if instance is None:
            return None
        readers: dict[str, Callable[[Instance], Outcome | None]] = {
            Target.of(markers.Same).fullname: self._same,
            Target.of(markers.Wrap).fullname: self._wrap,
        }
        reader = readers.get(instance.type.fullname)
        if reader is not None:
            return reader(instance)
        text = ConstDtypeReads().read(instance)
        if text is None:
            return None
        return Const(text)

    @propagates(None)
    def _same(self, instance: Instance) -> Outcome | None:
        """Read a `Same[...]` type.

        Args:
            instance: The marker instance.

        Returns:
            The outcome that gives the data type of the named column back, or
            None if the marker is in a wrong form.
        """
        not_falsy(len(instance.args) == 1)
        return Axis(not_none(self._reads.name(instance.args[0])))

    @propagates(None)
    def _wrap(self, instance: Instance) -> Outcome | None:
        """Read a `Wrap[...]` type.

        Args:
            instance: The marker instance.

        Returns:
            The wrap outcome, or None if the marker is in a wrong form.
        """
        not_falsy(len(instance.args) == 2)
        constructor = not_none(self._reads.instance(instance.args[0]))
        inner = not_none(self.read(instance.args[1]))
        return Wrap(constructor.type.name.rsplit(".", 1)[-1], inner)


class HeadRoster:
    """Map each character of an expression to the head class that holds it, and back.

    The head class of an expression type tells how the expression treats rows.
    `TypedExpr` gives one value for each row, `TypedAgg` gives one value for
    each group, and `TypedLit` is one fixed value. The codec writes the
    character as the head, and the reader finds the character from the head.

    Example:
        >>> HeadRoster().head_of(AGGREGATING)
        <class 'astrolabe.expr.TypedAgg'>
    """

    _HEADS: tuple[tuple[type, Character], ...] = (
        (TypedExpr, ELEMENTWISE),
        (TypedAgg, AGGREGATING),
        (TypedLit, LITERAL),
    )

    def head_of(self, character: Character) -> type:
        """Give the head class for one character.

        Args:
            character: The character of the expression.

        Returns:
            The head class.
        """
        by_character = {held: head for head, held in self._HEADS}
        return by_character[character]

    def character_of(self, fullname: str) -> Character | None:
        """Give the character that one head class holds.

        Args:
            fullname: The full name of the class.

        Returns:
            The character, or None for a class that is not a head.

        Example:
            >>> HeadRoster().character_of("builtins.int") is None
            True
        """
        by_head = {Target.of(head).fullname: character for head, character in self._HEADS}
        return by_head.get(fullname)


class QuadrupleCodec:
    """Write the type of an expression as one mypy type, and read it back.

    Every hook that types an expression operation, for example `pl.col`,
    `.sum()` or `+`, returns its answer through this codec. No hook builds an
    expression type by hand. The codec writes only the normal form, so an
    expression type is equal to another exactly when the engine values are
    equal, and mypy can compare them without help. It uses `RegionCodec` for the
    regions and `OutcomeCodec` for the tables. The reading half is
    `QuadrupleReader`.

    Example:
        >>> fixture = new_type_fixture()
        >>> codec = new_expr_codec(fixture)
        >>> identity = Table.from_cells(("a",), {("Int64",): Axis("a")})
        >>> expr = ExpressionType(
        ...     frozenset({"a"}), identity, "a", ELEMENTWISE
        ... )
        >>> codec.encoded(expr)
        astrolabe.expr.TypedColumn[Literal['a'], polars.datatypes.classes.Int64]
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the minter and the codecs for the parts.

        Args:
            minter: The marker minter for the current phase.
        """
        self._minter = minter
        self._regions = RegionCodec(minter)
        self._outcomes = OutcomeCodec(minter)
        self._heads = HeadRoster()

    @propagates(None)
    def encoded(self, expression: ExpressionType) -> MypyType | None:
        """Write one expression type as its mypy type.

        The expression that gives back one column unchanged takes the short form
        `TypedColumn[...]`. Any other takes the full form with a head class.

        Args:
            expression: The expression type from the engine.

        Returns:
            The mypy type, or None if a part has no type.
        """
        collapsed = self._collapsed_column(expression)
        if collapsed is not None:
            return collapsed
        head = self._heads.head_of(expression.character)
        reads = not_none(self.reads_for(expression.table))
        gives = not_none(self._built_gives(expression))
        return self._minter.marker(head, [reads, gives])

    @propagates(None)
    def _collapsed_column(self, expression: ExpressionType) -> MypyType | None:
        """Write an expression that gives back one column unchanged in the short form.

        That expression is `pl.col(name)`, the most common one. It reads one
        column, gives it back with its own data type at every kind, works row by
        row, and gives the same data type in both contexts.

        Args:
            expression: The expression type from the engine.

        Returns:
            `TypedColumn[Literal[name], region]`, or None for any other
            expression.
        """
        name = expression.name
        not_falsy(expression.character == ELEMENTWISE)
        not_falsy(expression.needs == frozenset({name}))
        not_falsy(expression.table.columns == (name,))
        cells = expression.table.cells
        not_falsy(all(outcome == Axis(name) for _, outcome in cells))
        region = not_none(self._regions.built(frozenset(cell[0] for cell, _ in cells)))
        return self._minter.marker(TypedColumn, [self._minter.name(name), region])

    @propagates(None)
    def reads_for(self, table: Table) -> MypyType | None:
        """Build the `Reads[...]` part: one column for each read.

        Each read is a `Column[name, region]` entry. The region holds only the
        kinds where the map has a cell, because the declaration says nothing
        at a kind outside it.

        Args:
            table: The map whose reads, and the kinds of each read, the part
                names.

        Returns:
            The `Reads[...]` type, or None if a region has no type.
        """
        columns: list[MypyType] = []
        for name in table.columns:
            region = not_none(self._regions.built(table.projection(name)))
            column = self._minter.marker(Column, [self._minter.name(name), region])
            columns.append(not_none(column))
        return self._minter.marker(markers.Reads, columns)

    @propagates(None)
    def _built_gives(self, expression: ExpressionType) -> MypyType | None:
        """Build the output part: the output name with the map.

        Args:
            expression: The declared expression type.

        Returns:
            The `Column[name, body]` type, or None if a part has no type.
        """
        body = not_none(self.body_for(expression.table))
        return self._minter.marker(Column, [self._minter.name(expression.name), body])

    @propagates(None)
    def body_for(self, table: Table) -> MypyType | None:
        """Build the type of one table in its shortest form.

        Args:
            table: The table to write.

        Returns:
            The one outcome if the table has one area, and `Rows[...]`
            otherwise. None if a part has no type.
        """
        return self._outcomes.body(table)


class QuadrupleReader:
    """Read the type of an expression back into the engine value, with no access to the plugin.

    This is the reading half of `QuadrupleCodec`. Code that only reads
    expression types, for example code that finds the column that an argument
    gives on a frame, uses it alone. It also offers the reads of the parts,
    which other codecs reuse for tables of their own.

    Example:
        >>> from mypy.types import UninhabitedType
        >>> QuadrupleReader().decoded(UninhabitedType()) is None
        True
    """

    def __init__(self) -> None:
        """Set up the readers for the parts."""
        self._reads = ProperReads()
        self._regions = RegionReads()
        self._outcomes = OutcomeReads()

    def decoded(self, typ: MypyType) -> ExpressionType | None:
        """Read one type as an expression type.

        Args:
            typ: The type to read.

        Returns:
            The expression type, or None for a type that is not an expression
            type.
        """
        instance = self._reads.instance(typ)
        if instance is None:
            return None
        if self._reads.matches(instance, TypedColumn):
            return self._column(instance)
        return self._headed(instance)

    @propagates(None)
    def _column(self, instance: Instance) -> ExpressionType | None:
        """Expand the short form `TypedColumn[...]` back into the full expression type.

        Args:
            instance: The `TypedColumn[...]` instance.

        Returns:
            The expression type that gives back one column unchanged, or None if
            the marker is in a wrong form.
        """
        not_falsy(len(instance.args) == 2)
        name = not_none(self._reads.name(instance.args[0]))
        region = not_none(self._regions.read(instance.args[1]))
        cells: dict[Cell, Outcome] = {(kind,): Axis(name) for kind in region}
        table = Table.from_cells((name,), cells)
        return ExpressionType(frozenset({name}), table, name, ELEMENTWISE)

    @propagates(None)
    def _headed(self, instance: Instance) -> ExpressionType | None:
        """Read the full form, with a head class, as an expression type.

        The head class gives the character, and the output body holds the map.

        Args:
            instance: The instance of the head class.

        Returns:
            The expression type, or None if a part is in a wrong form.
        """
        character = not_none(HeadRoster().character_of(instance.type.fullname))
        not_falsy(len(instance.args) == 2)
        reads = not_none(self.read_reads(instance.args[0]))
        gives = not_none(self._reads.instance(instance.args[1]))
        not_falsy(len(gives.args) == 2)
        name = not_none(self._reads.name(gives.args[0]))
        table = not_none(self.read_table(reads, gives.args[1]))
        return ExpressionType(frozenset(reads.axes), table, name, character)

    @propagates(None)
    def read_reads(self, typ: MypyType) -> DecodedReads | None:
        """Read the `Reads[...]` part: one `Column[name, region]` for each read.

        Args:
            typ: The type to read.

        Returns:
            The names of the reads with their regions. None if the part is in a
            wrong form.
        """
        instance = not_none(self._reads.instance(typ))
        not_falsy(self._reads.matches(instance, markers.Reads))
        axes: list[str] = []
        regions: list[frozenset[str]] = []
        for argument in instance.args:
            entry = not_none(self._reads.instance(argument))
            not_falsy(self._reads.matches(entry, Column))
            not_falsy(len(entry.args) == 2)
            axes.append(not_none(self._reads.name(entry.args[0])))
            regions.append(not_none(self._regions.read(entry.args[1])))
        return DecodedReads(tuple(axes), tuple(regions))

    @propagates(None)
    def read_table(self, reads: DecodedReads, body: MypyType) -> Table | None:
        """Read the table from the output body.

        Args:
            reads: The reads, already read.
            body: The body: one outcome, or `Rows[...]`.

        Returns:
            The table over the axes of the reads. One outcome covers the product
            of the regions in the reads. None if the body is in a wrong form.
        """
        instance = self._reads.instance(body)
        if instance is not None and self._reads.matches(instance, markers.Rows):
            return self.rows_table(reads.axes, instance)
        outcome = not_none(self._outcomes.read(body))
        cells = dict(self._outcomes.product_cells(reads.regions, outcome))
        return Table.from_cells(reads.axes, cells)

    def rows_table(self, axes: tuple[str, ...], instance: Instance) -> Table | None:
        """Read a `Rows[...]` type as a table over the given axes.

        Args:
            axes: The axes of the table, in axis order.
            instance: The `Rows[...]` instance.

        Returns:
            The table with the cells of every row, or None if a row is in a
            wrong form.
        """
        return self._outcomes.rows_table(axes, instance)


@dataclass(frozen=True)
class DecodedReads:
    """The `Reads[...]` part of an expression type, read back into plain values.

    Code that reads a table from a type needs the axes and their regions first.
    A table given as one outcome covers exactly the product of these regions.

    Attributes:
        axes: The column names of the reads, in the order of the type.
        regions: The region of each read, at the same index.
    """

    axes: tuple[str, ...]
    regions: tuple[frozenset[str], ...]
