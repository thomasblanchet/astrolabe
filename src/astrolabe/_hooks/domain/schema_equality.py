"""Tell whether two frames have the same columns, for operations that stack one on the other.

`pl.concat` stacks frames vertically, and `.vstack()`, `.extend()` and `.merge_sorted()` add
the rows of one frame to another. Each needs the two frames to have the same column names and
data types. A schema has no order, so the code for each of them sorts both column lists by name
before it compares them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from mypy.types import get_proper_type

from astrolabe._hooks.encoding.rendering_codec import ConstDtypeReads

if TYPE_CHECKING:
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.domain.dtype_promotion import DTypePromotion


class DecimalParameters:
    """Tell whether two data types differ only in the precision and the scale of a `Decimal`.

    The schema check of a vertical stack in polars reads every `Decimal` as
    one type, at any depth: `pl.concat` of a `Decimal(10, 2)` column and a
    `Decimal(38, 2)` column runs, and keeps the type of the first frame.
    `List(Decimal(10, 2))` and `List(Decimal(38, 3))` match the same way.
    The class compares the renderings of the two types with the parameters
    of each `Decimal` left out. It builds no type, so it needs no access to
    the plugin.

    Example:
        >>> fixture = new_type_fixture()
        >>> DecimalParameters().same(decimal_dtype(fixture, 10, 2), decimal_dtype(fixture, 38, 3))
        True
        >>> DecimalParameters().same(int_dtype(fixture), decimal_dtype(fixture, 38, 3))
        False
    """

    PARAMETERS: ClassVar[re.Pattern[str]] = re.compile(r"Decimal\(precision=\w+, scale=\d+\)")

    def same(self, left: MypyType, right: MypyType) -> bool:
        """Tell whether two data types are equal once each `Decimal` loses its parameters.

        Args:
            left: The first data type.
            right: The second data type.

        Returns:
            Whether they are, where both read as a data type.

        Example:
            >>> fixture = new_type_fixture()
            >>> DecimalParameters().same(fixture.anyt, fixture.anyt)
            False
        """
        reads = ConstDtypeReads()
        renderings = [reads.read_type(left), reads.read_type(right)]
        if None in renderings or "Decimal" not in str(renderings[0]):
            return False
        bare = {self.PARAMETERS.sub("Decimal", str(rendering)) for rendering in renderings}
        return len(bare) == 1


class SchemaEquality:
    """Compare two lists of columns, pair by pair, by name and by data type.

    Code that checks a vertical stack of frames uses this to decide whether to report a schema
    mismatch. A column with the data type `Unchecked`, which means a known name and an unknown
    data type, matches any data type. Two data types that differ only in the parameters of a
    `Decimal` match, as polars reads them (`DecimalParameters`). In relaxed mode, as for
    `how="vertical_relaxed"`, two data types also match if polars can cast both to one common
    data type (see `DTypePromotion`).

    Example:
        >>> fixture = new_type_fixture()
        >>> dtype = int_dtype(fixture)
        >>> price = column_ref(fixture, "price", dtype)
        >>> other_price = column_ref(fixture, "price", dtype)
        >>> name = column_ref(fixture, "name", str_dtype(fixture))
        >>> equality = schema_equality(new_plugin(fixture))
        >>> equality.columns_match([price], [other_price], relaxed=False)
        True
        >>> equality.columns_match([price], [name], relaxed=False)
        False
    """

    def __init__(self, unchecked: UncheckedCompatibility, promotion: DTypePromotion) -> None:
        """Keep the two rules that decide whether two data types match.

        Args:
            unchecked: The equality of data types that lets `Unchecked` match anything. It
                applies in both modes.
            promotion: The rule for a common data type. It applies only in relaxed mode.
        """
        self._unchecked = unchecked
        self._promotion = promotion

    def columns_match(
        self,
        left_columns: list[ColumnRef],
        right_columns: list[ColumnRef],
        *,
        relaxed: bool,
    ) -> bool:
        """Tell whether two lists of columns match, pair by pair in list order.

        The lists match if they have the same length, and each pair has the same name and
        matching data types.

        Args:
            left_columns: The first list.
            right_columns: The second list.
            relaxed: True if two data types with a common data type also match.

        Returns:
            True if the lists match.
        """
        if len(left_columns) != len(right_columns):
            return False
        pairs = zip(left_columns, right_columns, strict=True)
        return all(self._column_matches(left, right, relaxed) for left, right in pairs)

    def _column_matches(self, left: ColumnRef, right: ColumnRef, relaxed: bool) -> bool:
        """Tell whether two columns have the same name and matching data types.

        Args:
            left: The first column.
            right: The second column.
            relaxed: True if two data types with a common data type also match.

        Returns:
            True if the names are equal, and the data types are equal or one of them is
            `Unchecked`. In relaxed mode, the data types also match if they have a common
            data type.
        """
        if left.name != right.name:
            return False
        left_dtype = get_proper_type(left.dtype)
        right_dtype = get_proper_type(right.dtype)
        if self._unchecked.compatible(left_dtype, right_dtype):
            return True
        if DecimalParameters().same(left_dtype, right_dtype):
            return True
        return relaxed and self._promotion.compatible(left_dtype, right_dtype)

    def sorted_by_name(self, columns: list[ColumnRef]) -> list[ColumnRef]:
        """Sort a list of columns by name, so that two lists compare with no regard to order.

        Args:
            columns: The columns to sort.

        Returns:
            The columns sorted by name. A column with no literal name sorts as if its name were
            the empty string.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> name = column_ref(fixture, "name", str_dtype(fixture))
            >>> equality = schema_equality(new_plugin(fixture))
            >>> [c.name for c in equality.sorted_by_name([price, name])]
            ['name', 'price']
        """
        return sorted(columns, key=lambda c: c.name or "")
