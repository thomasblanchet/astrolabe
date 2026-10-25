"""The check of a zip of two series of one type, which `coalesce` and `shift` with a fill run.

Polars zips two series with a mask: each row takes the value of one series
or of the other. The zip of polars reads the second series as the type of the
first, so the second series must have the type of the first. The cast pass
usually brings the two series to one type. A cast can still keep a series
apart, as a struct stays a struct, and the zip then refuses or panics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop

if TYPE_CHECKING:
    from astrolabe._rules.dtypes import DType


@dataclass(frozen=True)
class SameTypeZip:
    """Refuse a second series that the zip cannot read as the type of the first series.

    The zip reads the second series by the kind of the first series:

    - `Null` takes any second series, and stays `Null`;
    - a struct or a `Decimal` needs a second series of its kind, or raises a
      `SchemaError`;
    - a category needs the same type, or raises a `SchemaError`;
    - a temporal type reads the physical values of the second series, and
      panics where the physical types differ;
    - a list or an array reads any list or any array;
    - every other type needs the same type, and polars panics otherwise.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> SameTypeZip().check(DType("Int32"), DType("Int64"))
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: cannot get ref Int32 from Int64
    """

    OWN_KIND: ClassVar[frozenset[str]] = frozenset({"Struct", "Decimal"})
    CATEGORIES: ClassVar[frozenset[str]] = frozenset({"Categorical", "Enum"})
    TEMPORAL: ClassVar[frozenset[str]] = frozenset({"Date", "Datetime", "Duration", "Time"})
    ANY_INNER: ClassVar[frozenset[str]] = frozenset({"List", "Array"})

    def check(self, first: DType, other: DType) -> None:
        """Refuse a second series that the zip cannot read as the type of the first series.

        Args:
            first: The type of the first series.
            other: The type of the second series.

        Raises:
            PolarsRefusal: The zip cannot read the second series.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> SameTypeZip().check(struct, DType("Int8"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: invalid series dtype: ...
            >>> SameTypeZip().check(DType("Date"), DType("Int32")) is None
            True
            >>> SameTypeZip().check(struct, DType("Struct", ((("b", DType("Int8")),),))) is None
            True
            >>> SameTypeZip().check(DType("Categorical"), DType("String"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: expected 'Categorical' found ...
        """
        name = first.name
        if other.same(first) or name == "Null":
            return
        if name in self.OWN_KIND:
            if other.name != name:
                message = f"invalid series dtype: expected `{name}`, got `{other!r}`"
                raise PolarsRefusal(ErrorClass.SCHEMA, message)
            return
        if name in self.CATEGORIES:
            raise PolarsRefusal(ErrorClass.SCHEMA, f"expected '{first!r}' found '{other!r}'")
        if name in self.TEMPORAL:  # pragma: no cover -- the cast pass casts to one unit
            readable = other.physical().same(first.physical())
        else:
            readable = name in self.ANY_INNER and other.name == name
        if not readable:  # pragma: no branch -- the casts give two lists one type
            raise PolarsRefusal(Stop.PANICS, f"cannot get ref {first!r} from {other!r}")
