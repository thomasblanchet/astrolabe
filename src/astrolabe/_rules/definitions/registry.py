"""Find the definition of a node by the identity of its node.

Code that reads a tree of polars meets each node by its identity: the kind of
the node, and its function. The registry holds the definitions of every
family by that identity. A node that no definition states has no entry, so
code that evaluates a tree knows that the rules cannot type that node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.definitions.aggregations import Aggregations
from astrolabe._rules.definitions.arrays import Arrays
from astrolabe._rules.definitions.binaries import Binaries
from astrolabe._rules.definitions.binary_operators import BinaryOperators
from astrolabe._rules.definitions.categories import Categories
from astrolabe._rules.definitions.definition import Definition
from astrolabe._rules.definitions.elementwise import Elementwise
from astrolabe._rules.definitions.length_preserving import LengthPreserving
from astrolabe._rules.definitions.lists import Lists
from astrolabe._rules.definitions.many_inputs import ManyInputs
from astrolabe._rules.definitions.node_kinds import NodeKinds
from astrolabe._rules.definitions.other_lengths import OtherLengths
from astrolabe._rules.definitions.ranges import Ranges
from astrolabe._rules.definitions.strings import Strings
from astrolabe._rules.definitions.structs import Structs
from astrolabe._rules.definitions.temporal import Temporal

if TYPE_CHECKING:
    from astrolabe._rules.definitions.node_options import NodeIdentity


class Definitions:
    """Hold the definition of each node, and find a definition by the identity of its node.

    The definitions stand in one class for each family of nodes. The registry
    reads every definition of each family, so a new definition needs no entry
    here.

    Example:
        >>> from astrolabe._rules.definitions.node_options import NodeIdentity
        >>> from astrolabe._rules.definitions.aggregations import Aggregations
        >>> Definitions.of(NodeIdentity("Agg", "Sum")) is Aggregations.SUM
        True
        >>> Definitions.of(NodeIdentity("Function", "Unknown")) is None
        True
    """

    FAMILIES: ClassVar[tuple[type, ...]] = (
        Elementwise,
        BinaryOperators,
        Aggregations,
        LengthPreserving,
        Strings,
        Binaries,
        Categories,
        Temporal,
        Structs,
        Lists,
        Arrays,
        OtherLengths,
        NodeKinds,
        ManyInputs,
        Ranges,
    )
    BY_IDENTITY: ClassVar[dict[NodeIdentity, Definition]] = {
        definition.identity: definition
        for family in FAMILIES
        for definition in vars(family).values()
        if isinstance(definition, Definition)
    }

    @classmethod
    def of(cls, identity: NodeIdentity) -> Definition | None:
        """Give the definition of a node.

        Args:
            identity: The identity of the node.

        Returns:
            The definition, or `None` where no definition states the node.

        Example:
            >>> from astrolabe._rules.definitions.node_options import NodeIdentity
            >>> Definitions.of(NodeIdentity("BinaryExpr", "Plus")).casts
            OperatorCasts(operator=<Operator.PLUS: 'Plus'>)
        """
        return cls.BY_IDENTITY.get(identity)
