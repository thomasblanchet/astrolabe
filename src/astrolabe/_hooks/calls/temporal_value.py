"""Read the polars data type of a `date` or a `datetime` value from the call that builds it.

polars stores a Python `datetime` as a `Datetime('us')` with the time zone
of the value, and a `date` as a `Date`. The type that mypy gives the value
does not carry either fact. The type `datetime` says nothing about the time
zone. The type `date` also covers a `datetime`, because `datetime` is a
subclass of `date`, and polars reads the two differently. A call that
builds the value in place, such as `datetime(2020, 1, 1, tzinfo=UTC)`,
states both facts in its abstract syntax tree (AST). This module reads
them there.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, ClassVar
from zoneinfo import ZoneInfo

from mypy.nodes import CallExpr, MemberExpr, NameExpr, RefExpr, StrExpr

from astrolabe._hooks.dispatch.generic import Target

if TYPE_CHECKING:
    from mypy.nodes import Expression


class TemporalValueReader:
    """Read what a call that builds a `date` or a `datetime` tells about its polars data type.

    Code that types an operation with a temporal value needs the data type
    that polars gives the value. For a `datetime`, that data type holds the
    time zone, and polars refuses to put a value of one zone in a column of
    another. The reader knows the zone where the call states it: no
    `tzinfo`, the zone `UTC`, or `ZoneInfo("...")` with a literal key. For
    a `date`, the reader tells whether the value is a `date` and not a
    `datetime`, which only a call of the class `date` itself proves.

    It sits in the plugin layer and reads only AST nodes.

    Example:
        >>> call = named_call_node("datetime.datetime", [int_literal_node(2020)])
        >>> TemporalValueReader().datetime_rendering(call)
        "Datetime(time_unit='us', time_zone=None)"
    """

    DATETIME: ClassVar[str] = Target.of(datetime).fullname
    DATE: ClassVar[str] = Target.of(date).fullname
    ZONE_INFO: ClassVar[str] = Target.of(ZoneInfo).fullname
    TIMEZONE: ClassVar[str] = Target.of(timezone).fullname
    # `UTC` is an instance, not a class, so its full name has no object to come from.
    UTC: ClassVar[frozenset[str]] = frozenset({"datetime.UTC", f"{TIMEZONE}.utc"})

    # The position of `tzinfo` among the positional arguments of `datetime(...)`.
    TZINFO_POSITION: ClassVar[int] = 7

    def exact_date(self, node: Expression) -> bool:
        """Tell whether a node builds a value of the class `date` and of no subclass.

        A call of the class, such as `date(2020, 1, 1)`, or of one of its
        class methods, such as `date.today()`, gives a `date`. A variable of
        the type `date` can hold a `datetime`.

        Args:
            node: The AST node of the value.

        Returns:
            True for a call of `date` or of a class method of `date`.

        Example:
            >>> reader = TemporalValueReader()
            >>> reader.exact_date(named_call_node("datetime.date"))
            True
            >>> reader.exact_date(named_reference_node("main.cutoff"))
            False
        """
        if not isinstance(node, CallExpr):
            return False
        callee = node.callee
        if isinstance(callee, MemberExpr):
            callee = callee.expr
        return isinstance(callee, RefExpr) and callee.fullname == self.DATE

    def datetime_rendering(self, node: Expression) -> str | None:
        """Give the data type that polars gives a `datetime` that a constructor call builds.

        Args:
            node: The AST node of the value.

        Returns:
            The rendering, with the time zone of the value, or None where
            the node is not a call of the constructor or its time zone
            is not written in a form that the reader knows.

        Example:
            >>> utc = named_reference_node("datetime.UTC")
            >>> call = named_call_node("datetime.datetime", [utc], ["tzinfo"])
            >>> TemporalValueReader().datetime_rendering(call)
            "Datetime(time_unit='us', time_zone='UTC')"
            >>> TemporalValueReader().datetime_rendering(utc) is None
            True
            >>> zone = named_reference_node("main.zone")
            >>> unread = named_call_node("datetime.datetime", [zone], ["tzinfo"])
            >>> TemporalValueReader().datetime_rendering(unread) is None
            True
        """
        if not isinstance(node, CallExpr) or not isinstance(node.callee, RefExpr):
            return None
        if node.callee.fullname != self.DATETIME:
            return None
        zone = self._tzinfo(node)
        if zone is None:
            return "Datetime(time_unit='us', time_zone=None)"
        name = self.zone(zone)
        if name is None:
            return None
        return f"Datetime(time_unit='us', time_zone='{name}')"

    def _tzinfo(self, call: CallExpr) -> Expression | None:
        """Give the `tzinfo` argument of a `datetime(...)` call.

        Args:
            call: The call of the constructor.

        Returns:
            The node of the argument, by keyword or by position, or None
            where the call gives no `tzinfo` or gives the literal None.

        Example:
            >>> utc = named_reference_node("datetime.UTC")
            >>> years = [int_literal_node(2020)] * 7
            >>> call = named_call_node("datetime.datetime", [*years, utc])
            >>> TemporalValueReader()._tzinfo(call) is utc
            True
        """
        for position, (argument, name) in enumerate(zip(call.args, call.arg_names, strict=True)):
            if name == "tzinfo" or (name is None and position == self.TZINFO_POSITION):
                return self._unless_none(argument)
        return None

    def _unless_none(self, argument: Expression) -> Expression | None:
        """Give an argument, or None where it is the literal None.

        Args:
            argument: The AST node of the argument.

        Returns:
            The node, or None for the literal None.

        Example:
            >>> TemporalValueReader()._unless_none(named_reference_node("builtins.None")) is None
            True
        """
        if isinstance(argument, NameExpr) and argument.fullname == "builtins.None":
            return None
        return argument

    def zone(self, tzinfo: Expression) -> str | None:
        """Read the name of a time zone that a `tzinfo` argument gives.

        Args:
            tzinfo: The AST node of the argument.

        Returns:
            `UTC` for `UTC` or `timezone.utc`, the key of a `ZoneInfo`
            call with a literal key, or None for any other form.

        Example:
            >>> paris = named_call_node("zoneinfo.ZoneInfo", [str_literal_node("Europe/Paris")])
            >>> TemporalValueReader().zone(paris)
            'Europe/Paris'
            >>> TemporalValueReader().zone(named_reference_node("main.zone")) is None
            True
        """
        if isinstance(tzinfo, CallExpr):
            return self._zone_info_key(tzinfo)
        if self._is_utc(tzinfo):
            return "UTC"
        return None

    def _is_utc(self, tzinfo: Expression) -> bool:
        """Tell whether a `tzinfo` argument is the zone `UTC`.

        Args:
            tzinfo: The AST node of the argument.

        Returns:
            True for `UTC` and for `timezone.utc`.

        Example:
            >>> from mypy.nodes import MemberExpr
            >>> utc = MemberExpr(named_reference_node("datetime.timezone"), "utc")
            >>> TemporalValueReader()._is_utc(utc)
            True
        """
        if isinstance(tzinfo, RefExpr) and tzinfo.fullname in self.UTC:
            return True
        if not isinstance(tzinfo, MemberExpr) or tzinfo.name != "utc":
            return False
        owner = tzinfo.expr
        return isinstance(owner, RefExpr) and owner.fullname == self.TIMEZONE

    def _zone_info_key(self, call: CallExpr) -> str | None:
        """Read the key of a `ZoneInfo("...")` call.

        Args:
            call: The call.

        Returns:
            The key, or None where the call is not a `ZoneInfo` call with
            one literal string.

        Example:
            >>> TemporalValueReader()._zone_info_key(named_call_node("main.zone")) is None
            True
            >>> keyed = named_call_node("zoneinfo.ZoneInfo", [named_reference_node("main.key")])
            >>> TemporalValueReader()._zone_info_key(keyed) is None
            True
        """
        callee = call.callee
        if not isinstance(callee, RefExpr) or callee.fullname != self.ZONE_INFO:
            return None
        if len(call.args) != 1 or not isinstance(call.args[0], StrExpr):
            return None
        return call.args[0].value
