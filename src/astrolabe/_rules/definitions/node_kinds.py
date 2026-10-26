"""The definitions of the nodes of a kind of their own: `when`, `filter`, the sorts and more.

Polars builds these nodes from a method that is not a function: a `when`
chain, `filter`, `sort`, `sort_by`, `slice` and its forms `head` and `tail`,
`gather` and `get`, `explode`, `alias`, and the renames of `.name`, such as
`.name.keep()` and `.name.suffix(...)`. Each kind has its own rule for its
type, the shape of its output and its run, which the vocabulary holds.
Polars drops an alias and a rename when it converts the expression, so each
runs as its input.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.cast_pass import SliceCasts, WhenCasts
from astrolabe._rules.definitions.cast_adapters import NoCasts, OwnRuleCasts
from astrolabe._rules.definitions.value_adapters import KeptValue
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import (
    AllInputs,
    ImplodedRun,
    LiteralInputs,
    Exploded,
    OneInput,
)
from astrolabe._rules.definitions.node_options import FromOptions, NodeIdentity, NodeOption
from astrolabe._rules.definitions.type_function_adapters import ExprField, ImplodedField, Mapper
from astrolabe._rules.kernels.dispatch import Dispatch
from astrolabe._rules.kernels.lengths import LengthKernels, Mapped
from astrolabe._rules.kernels.nodes import (
    Conditional,
    Filtered,
    Gathered,
    GatheredInGroups,
    Kept,
    SliceArguments,
)
from astrolabe._rules.kernels.windows import RollingIndex, WindowDuration
from astrolabe._rules.options import KernelLength, NodeKind, Shape
from astrolabe._rules.type_functions import ExprFields, FieldsMapper


class SliceShape:
    """Give the shape of a slice from its offset and its length, where both are integer literals.

    The run of a slice with a literal offset and length knows the rows that it
    keeps (`KernelLength.slice`). A null length slices to the end: polars
    reads it as the largest count of rows (`TO_THE_END`). Any other offset
    or length leaves the rows to the values.

    Example:
        >>> start, count = {"Literal": {"Dyn": {"Int": 0}}}, {"Literal": {"Dyn": {"Int": 2}}}
        >>> shape = SliceShape.of(start, count)
        >>> shape.length
        KernelLength(rule='slice', count=2, offset=0)
    """

    # The tree of a null literal, and the count that polars reads for a null length.
    NULL: ClassVar[dict[str, object]] = {"Literal": {"Scalar": {"Null": "Null"}}}
    TO_THE_END: ClassVar[int] = 2**64 - 1

    @classmethod
    def of(cls, offset: object, length: object) -> Shape:
        """Give the shape of a slice from its serialized offset and length.

        Args:
            offset: The tree of the offset.
            length: The tree of the length.

        Returns:
            The shape.

        Example:
            >>> SliceShape.of({"Column": "o"}, {"Literal": {"Dyn": {"Int": 2}}}).length is None
            True
            >>> end = {"Literal": {"Scalar": {"Null": "Null"}}}
            >>> SliceShape.of({"Literal": {"Dyn": {"Int": 1}}}, end).length.length([5])
            4
        """
        start = cls._integer(offset)
        count = cls.TO_THE_END if length == cls.NULL else cls._integer(length)
        if start is None or count is None:
            return Shape.of(NodeKind.SLICE)
        return Shape(NodeKind.SLICE, length=KernelLength.slice(start, count))

    @classmethod
    def raw(cls, part: object) -> object:
        """Give a part of the tree as it is, for `of` to read.

        Args:
            part: The part.

        Returns:
            The same part.

        Example:
            >>> SliceShape.raw({"Column": "o"})
            {'Column': 'o'}
        """
        return part

    @classmethod
    def _integer(cls, part: object) -> int | None:
        """Give the value of an integer literal of the tree, or `None` for any other part.

        Args:
            part: The tree of an argument.

        Returns:
            The integer, or `None`.

        Example:
            >>> SliceShape._integer({"Literal": {"Scalar": {"UInt64": 3}}})
            3
        """
        literal = part.get("Literal") if isinstance(part, dict) else None
        holder = next(iter(literal.values()), None) if isinstance(literal, dict) else None
        value = next(iter(holder.values()), None) if isinstance(holder, dict) else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None


class OverShape:
    """Read the mapping of an `over` from its node: the shape of the node, and whether it joins.

    `GroupsToRows` gives one row for each row of the frame, `Explode` gives
    the rows of each group one after the other, and `Join` gives one list for
    each row where the function does not give one value for each group.

    Example:
        >>> OverShape.joined("GroupsToRows")
        False
    """

    @classmethod
    def shape(cls, mapping: object) -> Shape:
        """Give the shape of an `over` from its mapping.

        Args:
            mapping: The mapping, as the tree writes it.

        Returns:
            The shape.

        Example:
            >>> OverShape.shape("Explode").explode, OverShape.shape("Join").joined
            (True, True)
        """
        return Shape.over(explode=mapping == "Explode", joined=cls.joined(mapping))

    @classmethod
    def joined(cls, mapping: object) -> bool:
        """Say whether the mapping of an `over` puts the values of each group in a list.

        Args:
            mapping: The mapping, as the tree writes it.

        Returns:
            Whether it is `Join`.

        Example:
            >>> OverShape.joined("Join")
            True
        """
        return mapping == "Join"


class NodeKinds:
    """Hold the definitions of the nodes of a kind of their own.

    Example:
        >>> NodeKinds.WHEN.identity
        NodeIdentity(shape='Ternary', function='')
    """

    WHEN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Ternary", ""),
        type_function=ExprField(ExprFields.when),
        options=Shape.of(NodeKind.TERNARY),
        casts=OwnRuleCasts(WhenCasts()),
        kernel=AllInputs(Conditional()),
    )
    FILTER: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Filter", ""),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.of(NodeKind.FILTER),
        casts=NoCasts(),
        kernel=AllInputs(Filtered()),
    )
    SORT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Sort", ""),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.of(NodeKind.SORT),
        casts=NoCasts(),
        kernel=AllInputs(Kept(), own_lengths=False),
    )
    SORT_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("SortBy", ""),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.of(NodeKind.SORT_BY),
        casts=NoCasts(),
        kernel=AllInputs(Kept(), own_lengths=False),
    )
    SLICE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Slice", ""),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=FromOptions(
            SliceShape.of,
            NodeOption(("offset",), SliceShape.raw),
            NodeOption(("length",), SliceShape.raw),
        ),
        casts=OwnRuleCasts(SliceCasts()),
        kernel=LiteralInputs(SliceArguments(), own_lengths=False),
    )
    GATHER: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Gather", ""),
        nulls=NullRule.ANY,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=FromOptions(
            Shape.gather,
            returns_scalar=NodeOption.flag("returns_scalar"),
        ),
        casts=NoCasts(),
        kernel=LiteralInputs(
            FromOptions(Gathered, null_on_oob=NodeOption.flag("null_on_oob")),
            own_lengths=False,
            groups=FromOptions(
                GatheredInGroups,
                returns_scalar=NodeOption.flag("returns_scalar"),
                null_on_oob=NodeOption.flag("null_on_oob"),
            ),
        ),
    )
    EXPLODE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Explode", ""),
        nulls=NullRule.FIRST,
        type_function=ExprField(ExprFields.explode),
        options=Shape.of(NodeKind.EXPLODE),
        casts=NoCasts(),
        kernel=Exploded(OneInput(Dispatch("explode", (), Mapped(ExprFields().explode)))),
    )
    OVER: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Over", ""),
        type_function=ImplodedField(
            FromOptions(OverShape.joined, NodeOption(("mapping",), SliceShape.raw))
        ),
        options=FromOptions(OverShape.shape, NodeOption(("mapping",), SliceShape.raw)),
        casts=NoCasts(),
        kernel=ImplodedRun(FromOptions(OverShape.joined, NodeOption(("mapping",), SliceShape.raw))),
    )
    ROLLING: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Rolling", ""),
        type_function=ImplodedField(True),
        options=Shape.of(NodeKind.ROLLING),
        casts=NoCasts(),
        kernel=ImplodedRun(
            True,
            FromOptions(
                RollingIndex,
                NodeOption(("period",), WindowDuration.of),
                NodeOption(("offset",), WindowDuration.of),
            ),
        ),
    )
    ALIAS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Alias", ""),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.of(NodeKind.CAST),
        casts=NoCasts(),
        kernel=OneInput(LengthKernels.SAME),
    )
    KEEP_NAME: ClassVar[Definition] = Definition(
        identity=NodeIdentity("KeepName", ""),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.of(NodeKind.CAST),
        casts=NoCasts(),
        kernel=OneInput(LengthKernels.SAME),
    )
    RENAME_ALIAS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("RenameAlias", ""),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.of(NodeKind.CAST),
        casts=NoCasts(),
        kernel=OneInput(LengthKernels.SAME),
    )
