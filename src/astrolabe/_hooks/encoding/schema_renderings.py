"""Give the engine the columns of a frame as text, and turn its answer back into a column.

The engine finds the column that an expression gives on a frame. It reads the
frame as a plain dict that maps each column name to the rendering of its data
type. A rendering is the text that polars shows for a data type, for example
`List(UInt32)`. The engine answers the new column in the same form. A hook,
however, holds each column as a `ColumnRef`, whose data type is a mypy type.

This module does both crossings. `SchemaRenderings` turns the declared columns
of a frame into the dict, and `MaterializedColumn` turns one answer back into a
`ColumnRef`. Every hook that asks the engine for the columns of an expression
goes through these two classes, so all of them feed the engine and read its
answer in the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.planned import PlannedType
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeCodec, ConstDtypeReads
from astrolabe._hooks.expression.lifting import UNCHECKED
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._hooks.encoding.marker_minter import MarkerMinter


class SchemaRenderings:
    """Turn the declared columns of a frame into the dict of renderings that the engine reads.

    The declared columns are the columns that the type of a frame names. The
    engine cannot read a mypy type, so this class reads the data type of each
    column through `ConstDtypeReads`.

    Example:
        >>> fixture = new_type_fixture()
        >>> price = column_ref(fixture, "price", int64_dtype(fixture))
        >>> SchemaRenderings().of([price])
        {'price': 'Int64'}
    """

    def __init__(self) -> None:
        """Set up the reader of data types."""
        self._reads = ConstDtypeReads()

    def of(self, columns: Sequence[ColumnRef]) -> dict[str, str]:
        """Turn a list of declared columns into the dict of renderings.

        A column with no literal name cannot be found by name, so the dict
        leaves it out. A data type that the engine cannot read becomes
        `Unchecked`, the data type of a column whose name is known and whose
        data type is not. The engine knows nothing about such a data type,
        and `Unchecked` states exactly that.

        Args:
            columns: The declared columns of the frame.

        Returns:
            A dict that maps each named column to the rendering of its data
            type.

        Example:
            >>> fixture = new_type_fixture()
            >>> odd = column_ref(fixture, "odd", new_marker_minter(fixture).string("x"))
            >>> SchemaRenderings().of([odd])
            {'odd': 'Unchecked'}
        """
        return self._rendered(columns, stored=False)

    def stored(self, columns: Sequence[ColumnRef]) -> dict[str, str]:
        """Turn the declared columns of a lazy frame into the dict of the renderings that it stores.

        A lazy frame plans each step at the type that it stores for a column,
        which `Planned[run, planner]` holds where it differs from the type of
        the data.

        Args:
            columns: The declared columns of the frame.

        Returns:
            A dict that maps each named column to the rendering of its stored type.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int64_dtype(fixture))
            >>> SchemaRenderings().stored([price])
            {'price': 'Int64'}
        """
        return self._rendered(columns, stored=True)

    def _rendered(self, columns: Sequence[ColumnRef], *, stored: bool) -> dict[str, str]:
        """Give the rendering of the data type of each named column, or of its stored type.

        Args:
            columns: The declared columns of the frame.
            stored: Whether to read the stored type of a column of a lazy frame.

        Returns:
            A dict that maps each named column to the rendering.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int64_dtype(fixture))
            >>> SchemaRenderings()._rendered([price], stored=True)
            {'price': 'Int64'}
        """
        rendered: dict[str, str] = {}
        for column in columns:
            if column.name is None:
                continue
            dtype = PlannedType().stored(column.dtype) if stored else column.dtype
            rendering = self._reads.read_type(dtype)
            rendered[column.name] = UNCHECKED if rendering is None else rendering
        return rendered


class MaterializedColumn:
    """Turn one column that the engine answers back into a `ColumnRef` for a frame type.

    The engine answers a column as a name and the rendering of its data type. A
    hook needs a `ColumnRef` to put the column into the frame that it returns.
    This class builds the mypy type of the data type through `ConstDtypeCodec`.

    Example:
        >>> fixture = new_type_fixture()
        >>> boundary = MaterializedColumn(new_marker_minter(fixture))
        >>> spliced = boundary.built("total", "Int64")
        >>> spliced.name, spliced.dtype
        ('total', polars.datatypes.classes.Int64)
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the builder of data types.

        Args:
            minter: The marker minter for the current phase.
        """
        self._codec = ConstDtypeCodec(minter)

    @propagates(None)
    def built(self, name: str, rendering: str) -> ColumnRef | None:
        """Build the column for one answer of the engine.

        Args:
            name: The name of the new column.
            rendering: The rendering of its data type.

        Returns:
            The column, or None for a rendering that has no type.
        """
        return ColumnRef.from_name_and_dtype(name, not_none(self._codec.built(rendering)))

    @propagates(None)
    def planned(self, name: str, rendering: str, stored: str) -> ColumnRef | None:
        """Build the column of a lazy frame whose stored type differs from the type of its data.

        Args:
            name: The name of the new column.
            rendering: The rendering of the type of its data.
            stored: The rendering of the type that the lazy frame stores.

        Returns:
            The column, of the type `Planned[run, planner]`, or None for a
            rendering that has no type.

        Example:
            >>> fixture = new_type_fixture()
            >>> boundary = MaterializedColumn(new_marker_minter(fixture))
            >>> boundary.planned("d", "Float64", "Float32").dtype.type.fullname
            'astrolabe.frame.Planned'
        """
        dtype = not_none(self._codec.planned(rendering, stored))
        return ColumnRef.from_name_and_dtype(name, dtype)
