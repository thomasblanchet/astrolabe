"""The kinds of data type, read from the class hierarchy of polars.

The engine types an expression with a table from the data types of the
columns it reads to the data type it gives. A table cannot list every
data type, because a type such as `List` or `Datetime` has parameters
and so has no end of forms. So the table lists kinds. A kind is a data
type with its parameters erased: `Int64` and `String` are kinds, and
`List` stands for every list type, whatever its element type.

The list of kinds comes from the installed polars at run time, and
nobody types it by hand. If a polars release adds a data type, the list
changes with no edit here.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
from polars.datatypes.classes import NumericType, TemporalType


class KindRoster:
    """List the kinds of data type that the installed polars defines, and group them.

    A kind is a leaf class of the `pl.DataType` hierarchy that polars
    itself defines, that is, a class that no other polars data-type class
    subclasses. A base class that groups kinds, for example `NumericType`
    or `TemporalType`, is a family and not a kind. An alias name, for
    example `Utf8` for `String`, is the same class and so the same kind.
    A subclass that another package defines is not a kind. This includes
    the marker classes that this plugin uses to spell its types. Code that
    builds a table, reads a selector or checks a family requirement asks
    this class for the kinds.

    Example:
        >>> "Int64" in KindRoster().names()
        True
        >>> "NumericType" in KindRoster().names()
        False
    """

    _CLASSES: ClassVar[list[tuple[type[pl.DataType], ...]]] = []
    _GROUPS: ClassVar[list[tuple[type[pl.DataType], ...]]] = []
    _NAMES: ClassVar[list[tuple[str, ...]]] = []
    _FAMILIES: ClassVar[dict[type[pl.DataType], tuple[str, ...]]] = {}

    def classes(self) -> tuple[type[pl.DataType], ...]:
        """Give the class of every kind, sorted by name.

        The walk over the class hierarchy runs once in each process,
        because the answer depends only on the installed polars. The
        result is kept on the class, and so are the names and the
        families that come from it. A kind is a leaf class that polars
        itself defines.

        Returns:
            One class for each kind. An alias adds no second entry.

        Example:
            >>> pl.String in KindRoster().classes()
            True
            >>> from polars.datatypes.classes import NumericType
            >>> NumericType in KindRoster().classes()
            False
        """
        if not self._CLASSES:
            leaves = {
                candidate
                for candidate in self._datatype_classes()
                if not self._is_group(candidate) and self._is_polars_own(candidate)
            }
            self._CLASSES.append(tuple(sorted(leaves, key=lambda cls: cls.__name__)))
        return self._CLASSES[0]

    def _is_polars_own(self, candidate: type[pl.DataType]) -> bool:
        """Answer whether polars itself defines a class.

        Args:
            candidate: The class to check.

        Returns:
            True if the module of the class is inside the polars package.

        Example:
            >>> KindRoster()._is_polars_own(pl.Int64)
            True
        """
        return candidate.__module__.startswith("polars.")

    def names(self) -> tuple[str, ...]:
        """Give the name of every kind, sorted.

        Returns:
            The class names, one for each kind.

        Example:
            >>> names = KindRoster().names()
            >>> ("List" in names, "Datetime" in names)
            (True, True)
        """
        if not self._NAMES:
            self._NAMES.append(tuple(cls.__name__ for cls in self.classes()))
        return self._NAMES[0]

    def family(self, base: type[pl.DataType]) -> tuple[str, ...]:
        """Give the kinds that one polars family base covers.

        Args:
            base: A polars family class, for example `NumericType`.

        Returns:
            The names of the kinds that subclass the base, sorted.

        Example:
            >>> from polars.datatypes.classes import NumericType
            >>> "Float64" in KindRoster().family(NumericType)
            True
            >>> KindRoster().family(pl.Int64)
            ('Int64',)
        """
        if base not in self._FAMILIES:
            covered = tuple(cls.__name__ for cls in self.classes() if issubclass(cls, base))
            self._FAMILIES[base] = covered
        return self._FAMILIES[base]

    def group_bases(self) -> tuple[type[pl.DataType], ...]:
        """Give the family bases that polars defines.

        A family base is a data-type class that other polars data-type
        classes subclass, for example `NumericType`. These are the groups
        that a requirement or a selector can name. Like the kinds, the
        bases depend only on the installed polars, so the walk runs once
        in each process and the result is kept on the class.

        Returns:
            The bases, sorted by name.

        Example:
            >>> names = [cls.__name__ for cls in KindRoster().group_bases()]
            >>> "NumericType" in names and "Int64" not in names
            True
        """
        if not self._GROUPS:
            own = [cls for cls in self._datatype_classes() if self._is_polars_own(cls)]
            groups = [cls for cls in own if self._is_group(cls)]
            self._GROUPS.append(tuple(sorted(groups, key=lambda cls: cls.__name__)))
        return self._GROUPS[0]

    def numeric(self) -> tuple[str, ...]:
        """Give the numeric kinds, that is, the kinds under `NumericType`.

        Returns:
            The names of the numeric kinds, sorted.

        Example:
            >>> "Int64" in KindRoster().numeric()
            True
        """
        return self.family(NumericType)

    def temporal(self) -> tuple[str, ...]:
        """Give the temporal kinds, that is, the kinds under `TemporalType`.

        Returns:
            The names of the temporal kinds, sorted.

        Example:
            >>> "Datetime" in KindRoster().temporal()
            True
        """
        return self.family(TemporalType)

    def _datatype_classes(self) -> set[type[pl.DataType]]:
        """Collect every class in the data-type hierarchy.

        The walk reads each class once. A class that another package
        defines is collected too, because its subclasses can matter to
        the walk. It never becomes a kind, because a later step keeps
        only the classes of polars.

        Returns:
            Every direct or indirect subclass of `pl.DataType`, from any
            package.

        Example:
            >>> found = KindRoster()._datatype_classes()
            >>> pl.Boolean in found
            True
            >>> from polars.datatypes.classes import NumericType
            >>> NumericType in found
            True
        """
        found: set[type[pl.DataType]] = set()
        frontier = [pl.DataType]
        while frontier:
            current = frontier.pop()
            new = set(current.__subclasses__()) - found
            found |= new
            frontier.extend(new)
        return found

    def _is_group(self, candidate: type[pl.DataType]) -> bool:
        """Answer whether a class is a family base and not a kind.

        Only a subclass that polars defines counts. The markers of this
        plugin subclass `pl.Datetime` and other kinds, and a marker must
        not turn a kind into a family.

        Args:
            candidate: The class to classify.

        Returns:
            True if a data-type class that polars defines subclasses it.

        Example:
            >>> from polars.datatypes.classes import NumericType
            >>> KindRoster()._is_group(NumericType)
            True
            >>> KindRoster()._is_group(pl.Int64)
            False
        """
        return any(self._is_polars_own(sub) for sub in candidate.__subclasses__())
