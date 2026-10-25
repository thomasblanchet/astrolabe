"""Read which columns a declaration reads, and at which data types.

An expression declaration lists the columns that the expression reads.
A user can write a read in several ways: inline, as a name declared
earlier, or through a `Columns` class. Each way ends as the same engine
value, a `Read`: a column name, and the region of the read. The region
is the set of kinds at which the column is read. A kind is a data type
with its parameters left out, for example `Datetime` for
`Datetime("us")`.
"""

from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING

from mypy.nodes import RefExpr, TypeAlias, TypeInfo

from astrolabe._hooks.declaring.regions import RegionParser
from astrolabe._hooks.declaring.site import CallShape, DeclarationCall
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.encoding.expr_codec import RegionReads
from astrolabe._hooks.encoding.marker_minter import ProperReads
from astrolabe._hooks.encoding.schema_renderings import SchemaRenderings
from astrolabe._hooks.expression.fit import Read
from astrolabe._hooks.expression.lifting import UNCHECKED
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe.declare import column_of, some_column, subset_of
from astrolabe.expr import TypedColumn
from astrolabe.frame import column

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import DynamicClassDefContext

    from astrolabe._hooks.declaring.regions import ParsedRegion


class ReadParser:
    """Read one argument of a declaration as the column reads that it stands for.

    The parser of a declaration passes each argument here, and does not
    need to know the forms that a read can take. There are six:

    - `column(name, dtype)` reads one named column.
    - `some_column(region)` reads one column whose name comes from the
      argument at the call. This is a blank read, and its name is None.
    - A name bound earlier to `column(...)` reads that column.
    - A `Columns` class reads every column that it declares.
    - `column_of(Schema, name)` and `subset_of(Schema, *names)` read some
      columns of a `Columns` class, at their declared data types.

    The code reads the syntax of the call during semantic analysis, the
    mypy phase that resolves names before any expression has a type.
    """

    def __init__(self, ctx: DynamicClassDefContext) -> None:
        """Keep the context of the declaration, and build the readers that this class uses.

        Args:
            ctx: The mypy context of the declaration. The region reader
                resolves data types through it.
        """
        self._ctx = ctx
        self._regions = RegionParser(ctx)
        self._shape = CallShape()
        self._reads = ProperReads()

    def parsed(self, node: Expression) -> list[Read] | None:
        """Read one argument of the declaration, in whichever form the user wrote it.

        Args:
            node: The node of the argument.

        Returns:
            The reads, or None if the node has none of the six forms.
        """
        for reader in (self._column_call, self._blank_call, self._schema_call):
            reads = reader(node)
            if reads is not None:
                return reads
        return self._reference(node)

    @propagates(None)
    def _column_call(self, node: Expression) -> list[Read] | None:
        """Read a `column(name, dtype)` call written inside the declaration.

        Args:
            node: The node.

        Returns:
            The one read, or None for any other node.
        """
        call = DeclarationCall(not_none(self._shape.call(node, column)))
        name = not_none(self._shape.literal(call.argument(0, "name")))
        region = not_none(self._regions.parsed(not_none(call.argument(1, "dtype"))))
        return [self._read(name, region)]

    @propagates(None)
    def _blank_call(self, node: Expression) -> list[Read] | None:
        """Read a `some_column(region)` call as a read with no name.

        Args:
            node: The node.

        Returns:
            The one blank read, or None for any other node.
        """
        call = DeclarationCall(not_none(self._shape.call(node, some_column)))
        region = not_none(self._regions.parsed(not_none(call.argument(0, "region"))))
        return [self._read(None, region)]

    @propagates(None)
    def _schema_call(self, node: Expression) -> list[Read] | None:
        """Read a `column_of(...)` or `subset_of(...)` call as reads at the declared data types.

        The two calls read alike. The first argument names a `Columns`
        class, and each later argument names one of its columns.

        Args:
            node: The node.

        Returns:
            One read for each name, or None for any other node, a first
            argument that is not a `Columns` class, or a call with no
            name.
        """
        shaped = self._shape.call(node, column_of)
        if shaped is None:
            shaped = self._shape.call(node, subset_of)
        call = DeclarationCall(not_none(shaped))
        positional = call.positional()
        first = not_falsy(positional)[0]
        target = first.node if isinstance(first, RefExpr) else None
        info = not_none(target if isinstance(target, TypeInfo) else None)
        schema = not_none(self._schema_items_of(info))
        names = [not_none(self._shape.literal(item)) for item in positional[1:]]
        return [self._schema_read(schema, name) for name in not_falsy(names)]

    def _reference(self, node: Expression) -> list[Read] | None:
        """Read a bare name as a column declared earlier, or as a `Columns` class.

        Args:
            node: The node.

        Returns:
            The reads, or None if the name resolves to anything else.
        """
        target = node.node if isinstance(node, RefExpr) else None
        if isinstance(target, TypeAlias):
            return self._declared_column(target)
        if isinstance(target, TypeInfo):
            return self._whole_schema(target)
        return None

    @propagates(None)
    def _declared_column(self, alias: TypeAlias) -> list[Read] | None:
        """Read a name bound earlier to a `column(...)` declaration.

        The declaration hook made that name an alias for
        `TypedColumn[name, region]`. This step decodes the alias back into
        a read.

        Args:
            alias: The alias that the name resolves to.

        Returns:
            The one read, or None for an alias to any other type.
        """
        instance = not_none(self._reads.instance(alias.target))
        not_falsy(self._reads.matches(instance, TypedColumn))
        not_falsy(len(instance.args) == 2)
        name = not_none(self._reads.name(instance.args[0]))
        kinds = not_none(RegionReads().read(instance.args[1]))
        if kinds == {UNCHECKED}:
            return [Read(name or None, frozenset(), unchecked=True)]
        return [Read(name or None, kinds)]

    def _whole_schema(self, info: TypeInfo) -> list[Read] | None:
        """Read every column that a `Columns` class declares.

        Args:
            info: The class.

        Returns:
            One read for each column, sorted by name, or None for a
            class that is not a `Columns` class. The order is fixed
            because a schema has no order of its own.
        """
        items = self._schema_items_of(info)
        if items is None:
            return None
        return [self._schema_read(items, name) for name in sorted(items)]

    def _schema_items_of(self, info: TypeInfo) -> dict[str, str] | None:
        """Give the data type of each column that a `Columns` class declares.

        A `Columns` class is a `TypedDict` to mypy, so its columns are
        the items of that `TypedDict`.

        Args:
            info: The class.

        Returns:
            The data type of each column as text, for example
            `Datetime(time_unit='us', time_zone=None)`, keyed by column
            name. None for a class that is not a `TypedDict`.
        """
        if info.typeddict_type is None:
            return None
        columns = starmap(ColumnRef.from_name_and_dtype, info.typeddict_type.items.items())
        return SchemaRenderings().of(list(columns))

    def _schema_read(self, items: dict[str, str], name: str) -> Read:
        """Read one column of a `Columns` class at its declared data type.

        Args:
            items: The data type of each column of the class, as text.
            name: The name of the column.

        Returns:
            The read, at the kind of the declared data type. A column
            declared `Unchecked`, whose data type the checker does not
            know, gives a read with no region to check.
            A name that the class does not declare gives an empty
            region, and the rules of a declaration report that later.

        Example:
            >>> parser = ReadParser.__new__(ReadParser)
            >>> parser._schema_read({"t": "Datetime(time_unit='us', time_zone=None)"}, "t")
            Read(name='t', region=frozenset({'Datetime'}), unchecked=False)
            >>> parser._schema_read({"p": "Unchecked"}, "p")
            Read(name='p', region=frozenset(), unchecked=True)
            >>> parser._schema_read({}, "zz")
            Read(name='zz', region=frozenset(), unchecked=False)
        """
        rendering = items.get(name)
        if rendering is None:
            return Read(name, frozenset())
        kind = rendering.split("(", maxsplit=1)[0]
        if kind == UNCHECKED:
            return Read(name, frozenset(), unchecked=True)
        return Read(name, frozenset({kind}))

    def _read(self, name: str | None, region: ParsedRegion) -> Read:
        """Build one read from a column name and the region that the parser read.

        Args:
            name: The column name, or None for a blank read.
            region: The region as the region parser read it.

        Returns:
            The read.

        Example:
            >>> from astrolabe._hooks.declaring.regions import ParsedRegion
            >>> parser = ReadParser.__new__(ReadParser)
            >>> parser._read("q", ParsedRegion(frozenset(), unchecked=True))
            Read(name='q', region=frozenset(), unchecked=True)
        """
        return Read(name, region.kinds, region.unchecked)
