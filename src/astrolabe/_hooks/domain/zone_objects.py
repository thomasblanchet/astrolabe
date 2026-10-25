"""Find a `datetime.timezone` object where polars needs the name of a zone.

The signature of polars says that a `time_zone` argument accepts a
`datetime.timezone`. At run time, polars raises
`TypeError: 'timezone' object is not an instance of 'str'`, so a call that
passes one can never run. mypy accepts such a call because of the signature.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import ListExpr, SetExpr, TupleExpr
from mypy.types import Instance, get_proper_type

from astrolabe._hooks.dispatch.generic import Target

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.types import Type as MypyType


class ZoneObjects:
    """Find a `datetime.timezone` object in the value of a zone argument.

    A type selector with a `time_zone` argument fails at run time if a
    zone is a `datetime.timezone` object and not a string. This class
    finds such an object in the argument, or in the items of a list,
    a tuple or a set written in the call.

    Args of the constructor:
        api: The mypy API of the type-checking phase, which types each
            item of a list, a tuple or a set.

    Example:
        >>> from mypy.nodes import StrExpr
        >>> ZoneObjects(None).among(StrExpr("UTC"), str_dtype(new_type_fixture()))
        False
    """

    TARGET: ClassVar[Target] = Target.of(datetime.timezone)

    def __init__(self, api: object) -> None:
        """Keep the mypy API.

        Args:
            api: The mypy API of the type-checking phase.
        """
        self._api = api

    def among(self, node: Expression, typ: MypyType) -> bool:
        """Answer whether one argument holds a `datetime.timezone` object.

        Args:
            node: The node of the argument.
            typ: The type of the argument.

        Returns:
            True for a timezone object, alone or as an item of a list,
            a tuple or a set.

        Example:
            >>> from mypy.nodes import IntExpr
            >>> ZoneObjects(None).among(IntExpr(3), str_dtype(new_type_fixture()))
            False
        """
        if isinstance(node, (ListExpr, TupleExpr, SetExpr)):
            # mypy gives a type for the argument, but not for the items
            # inside it, so ask for the type of each item.
            item_type = self._api.get_expression_type  # type: ignore[attr-defined]
            return any(self._typed(item_type(item)) for item in node.items)
        return self._typed(typ)

    def _typed(self, typ: MypyType) -> bool:
        """Answer whether a type is `datetime.timezone`.

        Args:
            typ: The type of the argument.

        Returns:
            True for an instance of the `datetime.timezone` class.

        Example:
            >>> ZoneObjects(None)._typed(str_dtype(new_type_fixture()))
            False
        """
        proper = get_proper_type(typ)
        if not isinstance(proper, Instance):
            return False
        return proper.type.fullname == self.TARGET.fullname
