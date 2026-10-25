"""Read a selector of polars, as the tree of polars writes it, as a rule for the remainder.

A selector such as `cs.numeric()` picks columns by their names and their data
types. Over an open frame, the checker cannot list the columns of the
remainder, the columns that the type of the frame does not name, so it keeps
a rule instead: which names and which data types the selector picks. The
code of a deferred expression holds the selector as a call of polars, and
polars serializes it as a small algebra: a test of the data type, a regular
expression or a list of names, joined by union, intersection and difference.
This module reads that algebra into the rules of `astrolabe._hooks.remainder`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from astrolabe._hooks.deferred.code_reader import CodeReader, CodeReadError, HoleValues
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe._hooks.remainder.absence import Always, Matching, Never, OneOf
from astrolabe._hooks.remainder.name_language import PatternReader
from astrolabe._hooks.remainder.selector import NameTest, OfType, SelAnd, Selector, SelNot, SelOr
from astrolabe._hooks.remainder.temporal_family import TemporalFamilies
from astrolabe._hooks.remainder.type_set import Family, FamilyRoster
from astrolabe._rules.definitions.node_options import SerializedDType

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from astrolabe._hooks.deferred.code import Code


class PolarsSelector:
    """Read the serialized form of a selector of polars as a rule for the remainder.

    A selector that picks columns by their position, such as `cs.first()`,
    has no rule, because the order of the remainder is not known. So does a
    regular expression that the checker cannot read, and a test of the type
    of the elements of a list.

    Example:
        >>> PolarsSelector().of({"ByDType": "Numeric"})
        OfType(family=NumericType)
        >>> PolarsSelector().of({"ByIndex": {"indices": [0], "strict": True}}) is None
        True
    """

    FAMILIES: ClassVar[dict[str, str]] = {
        "Numeric": "NumericType",
        "Integer": "IntegerType",
        "SignedInteger": "SignedIntegerType",
        "UnsignedInteger": "UnsignedIntegerType",
        "Float": "FloatType",
        "Temporal": "TemporalType",
        "Nested": "NestedType",
        "Struct": "Struct",
        "Decimal": "Decimal",
        "Enum": "Enum",
        "Categorical": "Categorical",
        "Object": "Object",
    }
    UNITS: ClassVar[dict[str, str]] = {
        "NANO_SECONDS": "ns",
        "MICRO_SECONDS": "us",
        "MILLI_SECONDS": "ms",
    }
    NAMED: ClassVar[dict[str, Selector]] = {
        "Wildcard": NameTest(Always()),
        "Empty": NameTest(Never()),
    }
    SET_OPERATIONS: ClassVar[frozenset[str]] = frozenset({
        "Union",
        "Intersect",
        "Difference",
        "ExclusiveOr",
    })
    ZONES: ClassVar[dict[str, list[str | None]]] = {
        "Any": [TemporalFamilies.ANY_ZONE, None],
        "AnySet": [TemporalFamilies.ANY_ZONE],
        "Unset": [None],
    }

    def of(self, node: object) -> Selector | None:
        """Read one selector.

        Args:
            node: The body of a `Selector` node of the tree, as `json.loads` gives it.

        Returns:
            The rule, or `None` where the selector has none.

        Example:
            >>> PolarsSelector().of("Wildcard")
            NameTest(rule=Always())
            >>> PolarsSelector().of({"Difference": ["Wildcard", {"ByDType": "Numeric"}]})
            SelNot(inner=OfType(family=NumericType))
            >>> PolarsSelector().of({"ByDType": "Numeric", "ByName": ["a"]}) is None
            True
        """
        if isinstance(node, str):
            return self.NAMED.get(node)
        if (  # pragma: no cover -- polars writes a node of one key
            not isinstance(node, dict) or len(node) != 1
        ):
            return None
        [(shape, body)] = node.items()
        if shape in self.SET_OPERATIONS:
            return self._joined(shape, body)
        return self._leaf(shape, body)

    @propagates(None)
    def _leaf(self, shape: str, body: object) -> Selector | None:
        """Read a selector that is not a set operation: a test of the names or of the data type.

        A name or a pattern from a variable of the type `str` is a value
        that no code knows, so the test has no rule.

        Args:
            shape: `Matches`, `ByName` or `ByDType`.
            body: The test.

        Returns:
            The rule, or `None` where the test has none.

        Example:
            >>> PolarsSelector()._leaf("ByName", {"names": ["a"], "strict": True})
            NameTest(rule=OneOf(names=frozenset({'a'})))
            >>> PolarsSelector()._leaf("Matches", "(?i)a") is None
            True
        """
        not_falsy(not HoleValues().held(body))
        if shape == "Matches":
            return NameTest(Matching.of(not_none(PatternReader().language(str(body)))))
        if shape == "ByName":
            names = body.get("names", []) if isinstance(body, dict) else []
            return NameTest(OneOf(frozenset(str(name) for name in names)))
        not_falsy(shape == "ByDType")
        return self._typed(body)

    @propagates(None)
    def _joined(self, shape: str, body: object) -> Selector | None:
        """Read a set operation on two selectors.

        Args:
            shape: `Union`, `Intersect`, `Difference` or `ExclusiveOr`.
            body: The two selectors.

        Returns:
            The rule, or `None` where a side has none.

        Example:
            >>> sides = [{"ByDType": "Numeric"}, {"ByDType": "Integer"}]
            >>> PolarsSelector()._joined("Difference", sides)
            SelAnd(left=OfType(family=NumericType), right=SelNot(inner=OfType(family=IntegerType)))
            >>> PolarsSelector()._joined("Union", sides[:1]) is None
            True
        """
        if (  # pragma: no cover -- polars writes two sides
            not isinstance(body, list) or len(body) != len(("left", "right"))
        ):
            return None
        sides = body
        right = not_none(self.of(sides[1]))
        if shape == "Difference" and sides[0] == "Wildcard":
            return SelNot(right)
        left = not_none(self.of(sides[0]))
        if shape == "Union":
            return SelOr(left, right)
        if shape == "Intersect":
            return SelAnd(left, right)
        if shape == "Difference":
            return SelAnd(left, SelNot(right))
        return SelOr(SelAnd(left, SelNot(right)), SelAnd(SelNot(left), right))

    @propagates(None)
    def _typed(self, body: object) -> Selector | None:
        """Read a test of the data type.

        Args:
            body: The test: the name of a family, a list of data types, or
                the units and zones of a `Datetime` or a `Duration`.

        Returns:
            The rule, or `None` where the test has none.

        Example:
            >>> PolarsSelector()._typed({"AnyOf": ["Int8", "String"]})
            OfType(family=Int8|String)
            >>> PolarsSelector()._typed({"List": "Integer"}) is None
            True
            >>> PolarsSelector()._typed(["Int8"]) is None
            True
        """
        if isinstance(body, str):
            return OfType(not_none(FamilyRoster().named(not_none(self.FAMILIES.get(body)))))
        if (  # pragma: no cover -- polars writes a body of one key
            not isinstance(body, dict) or len(body) != 1
        ):
            return None
        [(kind, test)] = body.items()
        if kind == "AnyOf":
            texts = [SerializedDType().of(dtype).render() for dtype in test]
            return OfType(Family.of_members(frozenset(texts)))
        if kind == "Duration":
            return OfType(not_none(TemporalFamilies().duration(self._units(test))))
        if kind == "Datetime":
            units, zones = test
            return TemporalFamilies().datetime_selector(self._units(units), self._zones(zones))
        not_falsy(kind in {"List", "Array"} and test in (None, [None, None]))
        return OfType(not_none(FamilyRoster().named(kind)))

    def _units(self, text: str) -> list[str] | None:
        """Read the time units of a test of a `Datetime` or a `Duration`.

        Args:
            text: The units, joined by `|`.

        Returns:
            The units, or `None` where the test takes every unit.

        Example:
            >>> PolarsSelector()._units("MICRO_SECONDS | MILLI_SECONDS")
            ['us', 'ms']
            >>> PolarsSelector()._units("NANO_SECONDS | MICRO_SECONDS | MILLI_SECONDS") is None
            True
        """
        units = [self.UNITS[unit.strip()] for unit in text.split("|")]
        return None if len(units) == len(self.UNITS) else units

    def _zones(self, test: object) -> list[str | None]:
        """Read the time zones of a test of a `Datetime`, where `None` stands for no zone.

        Args:
            test: The zones: `Any`, `AnySet`, `Unset`, or a list of zones that
                can also take no zone.

        Returns:
            The zones.

        Example:
            >>> PolarsSelector()._zones({"UnsetOrAnyOf": [{"inner": "UTC"}]})
            [None, 'UTC']
            >>> PolarsSelector()._zones("AnySet")
            ['*']
        """
        if isinstance(test, str):
            return list(self.ZONES[test])
        listed: dict[str, Sequence[object]] = test if isinstance(test, dict) else {}
        named = listed.get("AnyOf", listed.get("UnsetOrAnyOf", []))
        zones: list[str | None] = [None] if "UnsetOrAnyOf" in listed else []
        for zone in named:
            zones.append(str(zone["inner"]) if isinstance(zone, dict) else str(zone))
        return zones


class NameRequirement:
    """Read the names that a selector of polars requires the frame to hold.

    `cs.by_name("a")` requires its names by default: polars raises
    `ColumnNotFoundError` where the frame has no column of one of them. This
    holds at each place in the selector, also inside a difference or a
    complement. Over an open frame, a missing name can be a column of the
    remainder, the columns that the type of the frame does not name. So the
    expansion over the declared columns reads each list of names as one that
    requires nothing.

    Example:
        >>> NameRequirement().required({"ByName": {"names": ["a"], "strict": True}})
        ['a']
    """

    def required(self, node: object) -> list[str]:
        """Give each name that a serialized selector requires, in the order of the selector.

        A name from a variable of the type `str` is a value that no code
        knows, so the data decides whether the frame holds it.

        Args:
            node: The serialized selector.

        Returns:
            The names.

        Example:
            >>> strict = {"ByName": {"names": ["a"], "strict": True}}
            >>> lenient = {"ByName": {"names": ["b"], "strict": False}}
            >>> NameRequirement().required({"Difference": [lenient, strict]})
            ['a']
        """
        if isinstance(node, list):
            return [name for item in node for name in self.required(item)]
        if not isinstance(node, dict):
            return []
        inner = [name for value in node.values() for name in self.required(value)]
        return self._own(node) + inner

    def _own(self, node: Mapping[str, object]) -> list[str]:
        """Give the names that a node of a list of names requires itself.

        Args:
            node: A node of the serialized selector.

        Returns:
            The names of a strict list, with no name from a variable.

        Example:
            >>> strict = {"ByName": {"names": ["a", HoleValues.STR], "strict": True}}
            >>> NameRequirement()._own(strict), NameRequirement()._own({"ByDType": "Numeric"})
            (['a'], [])
        """
        body = node.get("ByName")
        if not isinstance(body, dict) or not body.get("strict"):
            return []
        return [name for name in body["names"] if not HoleValues().held(name)]

    def relaxed(self, node: object) -> object:
        """Give a serialized selector whose lists of names require nothing.

        Args:
            node: The serialized selector.

        Returns:
            The new selector.

        Example:
            >>> NameRequirement().relaxed({"ByName": {"names": ["a"], "strict": True}})
            {'ByName': {'names': ['a'], 'strict': False}}
        """
        if isinstance(node, list):
            return [self.relaxed(item) for item in node]
        if not isinstance(node, dict):
            return node
        relaxed = {key: self.relaxed(value) for key, value in node.items()}
        body = relaxed.get("ByName")
        if isinstance(body, dict):
            relaxed["ByName"] = {**body, "strict": False}
        return relaxed


class SelectionRoot:
    """Read an argument that polars expands to the names of columns, as an argument of `drop`.

    Polars takes a column or a selector at the root of such an argument. It
    refuses any other expression, as `pl.col("a") + 1` or
    `pl.col("a").alias("b")`, with `InvalidOperationError` at every frame,
    because no name follows from it.

    Example:
        >>> from astrolabe._hooks.deferred.code import Code
        >>> root = SelectionRoot()
        >>> root.refused(Code("pl.col('a') + 1")), root.column(Code("pl.col('a')"))
        (True, True)
    """

    ROOTS: ClassVar[frozenset[str]] = frozenset({"Column", "Selector"})

    def refused(self, code: Code[object]) -> bool:
        """Say whether a code holds no column and no selector at its root, which polars refuses.

        Args:
            code: The code.

        Returns:
            Whether it holds none. A code that the reader cannot read is not refused.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> root = SelectionRoot()
            >>> root.refused(Code("cs.numeric()")), root.refused(Code("pl.col("))
            (False, False)
        """
        tree = self._tree(code)
        return isinstance(tree, dict) and not self.ROOTS & tree.keys()

    def column(self, code: Code[object]) -> bool:
        """Say whether a code is one column, which polars requires the frame to hold.

        Args:
            code: The code.

        Returns:
            Whether it is.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> SelectionRoot().column(Code("cs.numeric()"))
            False
        """
        tree = self._tree(code)
        return isinstance(tree, dict) and "Column" in tree

    def _tree(self, code: Code[object]) -> object:
        """Give the serialized tree of the expression of a code.

        Args:
            code: The code.

        Returns:
            The tree, or `None` where the code gives no expression.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> SelectionRoot()._tree(Code("pl.col('a')"))
            {'Column': 'a'}
            >>> SelectionRoot()._tree(Code("1")) is None
            True
        """
        try:
            expression = CodeReader().read(code).expression
        except CodeReadError:
            return None
        if not isinstance(expression, pl.Expr):
            return None
        return json.loads(expression.meta.serialize(format="json"))
