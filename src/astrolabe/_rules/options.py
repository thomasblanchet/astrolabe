"""The options that polars declares for each node, and the shape that they give its output.

Polars declares for each function a set of `FunctionOptions`: flags that say
whether the function works value by value, keeps the length of its input,
or gives one value, and the rules of its cast pass. The other kinds of node
in a plan, such as a binary operation, a cast or a slice, have no options:
their kind alone decides their shape. Two consequences of these facts reach
the user. In a `select`, they decide the row form of the output: one value
per row, one value, one row, or another length. In `group_by().agg()`, they
decide whether polars wraps the values of each group in a list.

This module translates the flags and the named constructors of polars, so that
a definition names a constructor exactly as polars does for that function. It
also translates the projection height of the planner, which decides the list
wrap, and gives the row form of an output from the row forms of the inputs. A
kernel that refuses the lengths of its inputs raises `PolarsRefusal`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.failures import ErrorClass, PolarsRefusal
from astrolabe._rules.supertype import SupertypeFlag

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class FunctionFlag(enum.Flag):
    """Name the flags that polars sets on a function.

    Polars keeps the flags in one bit set. The values here are the bits of
    polars, so a set of flags reads the same as the bit set of polars. The
    flags that decide the shape of the output are `ROW_SEPARABLE`,
    `LENGTH_PRESERVING`, `RETURNS_SCALAR` and `RANGE`. The others decide names,
    wildcards and the order of rows.

    Example:
        >>> FunctionFlag.ROW_SEPARABLE | FunctionFlag.LENGTH_PRESERVING
        <FunctionFlag.ROW_SEPARABLE|LENGTH_PRESERVING: 192>
    """

    ALLOW_RENAME = 1 << 0
    PASS_NAME_TO_APPLY = 1 << 1
    INPUT_WILDCARD_EXPANSION = 1 << 2
    RETURNS_SCALAR = 1 << 3
    OPTIONAL_RE_ENTRANT = 1 << 4
    ALLOW_EMPTY_INPUTS = 1 << 5
    ROW_SEPARABLE = 1 << 6
    LENGTH_PRESERVING = 1 << 7
    PRESERVES_NULL_FIRST_INPUT = 1 << 8
    PRESERVES_NULL_ALL_INPUTS = 1 << 9
    NON_ORDER_OBSERVING = 1 << 10
    TERMINATES_INPUT_ORDER = 1 << 11
    NON_ORDER_PRODUCING = 1 << 12
    RANGE = 1 << 13


@dataclass(frozen=True)
class CastingRules:
    """Say how the cast pass casts the inputs of a function.

    Polars has two rules. The rule of the first input (`first_arg_lossless`)
    casts the other inputs to the type of the first input where no value is
    lost. The supertype rule (`supertype`) casts every input to the supertype
    of all the inputs, with the flags of that join (`SupertypeFlag`). The value
    holds the flags of the supertype, or `None` for the first rule.

    Example:
        >>> CastingRules.cast_to_supertypes().supertype_flags is None
        False
        >>> CastingRules.first_arg_lossless().supertype_flags is None
        True
    """

    supertype_flags: SupertypeFlag | None

    @classmethod
    def first_arg_lossless(cls) -> CastingRules:
        """Give the rule that casts the other inputs to the type of the first input.

        Returns:
            The rule.

        Example:
            >>> CastingRules.first_arg_lossless().supertype_flags is None
            True
        """
        return cls(None)

    @classmethod
    def supertype(cls, flags: SupertypeFlag) -> CastingRules:
        """Give the rule that casts every input to the supertype, with the flags of the join.

        Args:
            flags: The flags of the supertype.

        Returns:
            The rule.

        Example:
            >>> CastingRules.supertype(SupertypeFlag(0)).supertype_flags == SupertypeFlag(0)
            True
        """
        return cls(flags)

    @classmethod
    def cast_to_supertypes(cls) -> CastingRules:
        """Give the supertype rule with the default flags.

        Returns:
            The rule.

        Example:
            >>> CastingRules.cast_to_supertypes().supertype_flags == SupertypeFlag.default()
            True
        """
        return cls(SupertypeFlag.default())


@dataclass(frozen=True)
class FunctionOptions:
    """Hold the options of one function, as `FunctionOptions` of polars holds them.

    Polars gives each function its options with a few named constructors and
    builder methods. A definition of a node names the same constructor and the
    same builders, for example
    `FunctionOptions.elementwise().with_supertyping(...)` for `fill_null`.
    `check_lengths` is the flag that makes the run compare the length of the
    output of a function with the length of its input.

    Example:
        >>> FunctionOptions.aggregation().returns_scalar()
        True
        >>> FunctionOptions.length_preserving().is_elementwise()
        False
    """

    flags: FunctionFlag = field(default_factory=lambda: FunctionFlag(0))
    cast_options: CastingRules | None = None
    check_lengths: bool = True

    @classmethod
    def elementwise(cls) -> FunctionOptions:
        """Give the options of a function that works value by value.

        Returns:
            The options with the flags `ROW_SEPARABLE` and `LENGTH_PRESERVING`.

        Example:
            >>> FunctionOptions.elementwise().is_elementwise()
            True
        """
        both = FunctionFlag.ROW_SEPARABLE | FunctionFlag.LENGTH_PRESERVING
        return cls().with_flags(lambda flags: flags | both)

    @classmethod
    def elementwise_with_infer(cls) -> FunctionOptions:
        """Give the options of `elementwise_with_infer()`, which is `length_preserving()`.

        Returns:
            The options with the flag `LENGTH_PRESERVING`.

        Example:
            >>> FunctionOptions.elementwise_with_infer() == FunctionOptions.length_preserving()
            True
        """
        return cls.length_preserving()

    @classmethod
    def row_separable(cls) -> FunctionOptions:
        """Give the options of a function that works on any split of its rows.

        `drop_nulls` is an example: it changes the length, and each part of
        the rows gives its own part of the output.

        Returns:
            The options with the flag `ROW_SEPARABLE`.

        Example:
            >>> FunctionOptions.row_separable().is_length_preserving()
            False
        """
        return cls().flag(FunctionFlag.ROW_SEPARABLE)

    @classmethod
    def length_preserving(cls) -> FunctionOptions:
        """Give the options of a function that keeps the length of its input.

        `cum_sum`, `rank` and `shift` are examples: each output row can read
        every input row, and the output has the length of the input.

        Returns:
            The options with the flag `LENGTH_PRESERVING`.

        Example:
            >>> FunctionOptions.length_preserving().is_length_preserving()
            True
        """
        return cls().flag(FunctionFlag.LENGTH_PRESERVING)

    @classmethod
    def groupwise(cls) -> FunctionOptions:
        """Give the options of a function with no shape flag.

        The output can have any length, and in `agg` the function runs once
        for each group: `unique` and `gather_every` are examples.

        Returns:
            The options with no flag.

        Example:
            >>> FunctionOptions.groupwise().flags
            <FunctionFlag: 0>
        """
        return cls()

    @classmethod
    def aggregation(cls) -> FunctionOptions:
        """Give the options of a function that gives one value.

        Returns:
            The options of `groupwise()` with the flag `RETURNS_SCALAR`.

        Example:
            >>> FunctionOptions.aggregation().flags
            <FunctionFlag.RETURNS_SCALAR: 8>
        """
        return cls.groupwise().flag(FunctionFlag.RETURNS_SCALAR)

    def with_supertyping(self, flags: SupertypeFlag) -> FunctionOptions:
        """Make the cast pass cast each input to the supertype of the inputs.

        `fill_null` passes the default flags. `max_horizontal` passes them
        without `ALLOW_PRIMITIVE_TO_STRING`.

        Args:
            flags: The flags of the supertype.

        Returns:
            The options with the rule `Supertype`.

        Example:
            >>> options = FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default())
            >>> options.cast_options.supertype_flags == SupertypeFlag.default()
            True
        """
        return self.with_casting_rules(CastingRules.supertype(flags))

    def with_casting_rules(self, rules: CastingRules) -> FunctionOptions:
        """Set the rule of the cast pass.

        Args:
            rules: The rule.

        Returns:
            The options with that rule.

        Example:
            >>> lossless = CastingRules.first_arg_lossless()
            >>> FunctionOptions.aggregation().with_casting_rules(lossless).cast_options == lossless
            True
        """
        return replace(self, cast_options=rules)

    def flag(self, flags: FunctionFlag) -> FunctionOptions:
        """Add flags.

        Args:
            flags: The flags to add.

        Returns:
            The options with the flags added.

        Example:
            >>> FunctionOptions.groupwise().flag(FunctionFlag.ALLOW_RENAME).flags
            <FunctionFlag.ALLOW_RENAME: 1>
        """
        return replace(self, flags=self.flags | flags)

    def with_flags(self, change: Callable[[FunctionFlag], FunctionFlag]) -> FunctionOptions:
        """Change the flags with a function of the flags.

        A definition writes `with_flags(lambda f: f | FunctionFlag.X)`, as
        polars does with its own flags.

        Args:
            change: The function that gives the new flags from the old flags.

        Returns:
            The options with the new flags.

        Example:
            >>> wild = FunctionFlag.INPUT_WILDCARD_EXPANSION
            >>> FunctionOptions.elementwise().with_flags(lambda f: f | wild).is_elementwise()
            True
        """
        return replace(self, flags=change(self.flags))

    def is_elementwise(self) -> bool:
        """Say whether the function works value by value.

        Returns:
            Whether both `ROW_SEPARABLE` and `LENGTH_PRESERVING` are set.

        Example:
            >>> FunctionOptions.row_separable().is_elementwise()
            False
        """
        both = FunctionFlag.ROW_SEPARABLE | FunctionFlag.LENGTH_PRESERVING
        return both in self.flags

    def is_length_preserving(self) -> bool:
        """Say whether the output has the length of the input.

        Returns:
            Whether `LENGTH_PRESERVING` is set.

        Example:
            >>> FunctionOptions.groupwise().is_length_preserving()
            False
        """
        return FunctionFlag.LENGTH_PRESERVING in self.flags

    def returns_scalar(self) -> bool:
        """Say whether the function gives one value.

        Returns:
            Whether `RETURNS_SCALAR` is set.

        Example:
            >>> FunctionOptions.elementwise().returns_scalar()
            False
        """
        return FunctionFlag.RETURNS_SCALAR in self.flags

    def is_range(self) -> bool:
        """Say whether the function gives a range, as `int_range` does.

        Returns:
            Whether `RANGE` is set.

        Example:
            >>> FunctionOptions.groupwise().flag(FunctionFlag.RANGE).is_range()
            True
        """
        return FunctionFlag.RANGE in self.flags


class Height(enum.Enum):
    """Name the projection height that the planner gives an expression.

    The planner of polars gives each expression one of four heights: the height
    of the frame, one value, a range of a length of its own, or a length that
    it does not know. A `group_by().agg()` wraps the values of each group in a
    list exactly where the height is not one value.

    Example:
        >>> Height.SCALAR.zip_with(Height.COLUMN)
        <Height.COLUMN: 'column'>
    """

    COLUMN = "column"
    SCALAR = "scalar"
    RANGE = "range"
    UNKNOWN = "unknown"

    def zip_with(self, other: Height) -> Height:
        """Give the height of two inputs side by side.

        One value takes the height of the other input. A range beside the
        height of the frame gives the height of the frame. Otherwise a
        range wins, then an unknown length.

        Args:
            other: The height of the other input.

        Returns:
            The height.

        Example:
            >>> Height.RANGE.zip_with(Height.UNKNOWN), Height.COLUMN.zip_with(Height.UNKNOWN)
            (<Height.RANGE: 'range'>, <Height.UNKNOWN: 'unknown'>)
            >>> Height.RANGE.zip_with(Height.COLUMN)
            <Height.COLUMN: 'column'>
        """
        if self is Height.SCALAR:
            return other
        if other is Height.SCALAR:
            return self
        heights = {self, other}
        if heights == {Height.RANGE, Height.COLUMN}:
            return Height.COLUMN
        for height in (Height.RANGE, Height.UNKNOWN):
            if height in heights:
                return height
        return Height.COLUMN

    @classmethod
    def zipped(cls, heights: Sequence[Height]) -> Height:
        """Give the height of several inputs side by side.

        Polars reads a node of no input, as `pl.format` of a text with no
        placeholder, as one value, because each of its inputs is one value.

        Args:
            heights: The heights of the inputs.

        Returns:
            The heights folded from the left with `zip_with`, or `SCALAR`
            where there is no input.

        Example:
            >>> Height.zipped([Height.SCALAR, Height.COLUMN]), Height.zipped([])
            (<Height.COLUMN: 'column'>, <Height.SCALAR: 'scalar'>)
        """
        if not heights:
            return cls.SCALAR
        height = heights[0]
        for other in heights[1:]:
            height = height.zip_with(other)
        return height


class RowForm(enum.Enum):
    """Name the four row forms of a value: the shape of its rows that the checker tracks.

    A value of one value per row has the height of the frame. One value is
    a value that the planner knows to be a scalar, such as `sum()`, `.first()`
    or a literal. One row is a value of exactly one row that the planner does
    not know to be a scalar, such as `head(1)`: a node that broadcasts an
    input of length 1 spreads it, but `agg` wraps it in a list. Another length
    is any other length, such as `head(2)` or `drop_nulls()`.

    Example:
        >>> RowForm.ONE_ROW.height()
        <Height.UNKNOWN: 'unknown'>
    """

    PER_ROW = "per row"
    ONE_VALUE = "one value"
    OTHER_LENGTH = "other length"
    ONE_ROW = "one row"

    def height(self) -> Height:
        """Give the projection height that the planner gives a value of this row form.

        The planner knows the height of the frame and one value. One row and
        another length both come from a node whose length the planner does not
        know.

        Returns:
            The height.

        Example:
            >>> RowForm.PER_ROW.height(), RowForm.ONE_VALUE.height()
            (<Height.COLUMN: 'column'>, <Height.SCALAR: 'scalar'>)
        """
        if self is RowForm.PER_ROW:
            return Height.COLUMN
        if self is RowForm.ONE_VALUE:
            return Height.SCALAR
        return Height.UNKNOWN

    def rank(self) -> int:
        """Give the place of the row form in the broadcast order of polars.

        A node that broadcasts its inputs gives the longest of them: an input
        of length 1 spreads to the length of the others. One value spreads
        first, then one row, then another length. One value per row spreads no
        further.

        Returns:
            The place, from 0 for one value to 3 for one value per row.

        Example:
            >>> RowForm.ONE_ROW.rank() < RowForm.OTHER_LENGTH.rank()
            True
        """
        order = (RowForm.ONE_VALUE, RowForm.ONE_ROW, RowForm.OTHER_LENGTH, RowForm.PER_ROW)
        return order.index(self)

    def is_single(self) -> bool:
        """Say whether a value of this row form has exactly one row.

        Returns:
            Whether the row form is one value or one row.

        Example:
            >>> RowForm.ONE_VALUE.is_single(), RowForm.PER_ROW.is_single()
            (True, False)
        """
        return self in {RowForm.ONE_VALUE, RowForm.ONE_ROW}


@dataclass(frozen=True)
class KernelLength:
    """Say how the kernel of a node gives the length of its output from the lengths of its inputs.

    The planner knows only whether a node gives one value. The length of
    any other output comes from its kernel. Polars has a few rules:

    - `broadcast`: the inputs must have one length, and an input of length 1
      spreads to it.
    - `equal`: the inputs must have one length, with no broadcast, as the keys
      of `sort_by`.
    - `first`: the length of the first input, where the inputs pass the check
      of `broadcast`, as `clip` and the functions that keep the length of their
      input. A first input of length 1 beside longer inputs gives one row.
    - `one`: one value, as an aggregation gives.
    - `shrinks`: at most the length of the first input, and one row from one
      row, as `unique`, `mode` and `gather_every`.
    - `values`: a length that the values decide, as `drop_nulls`, `explode`
      and `top_k`. The run checks no length: the kernel checks its inputs.
    - `masked`: a length that a mask decides, as `filter` gives. The inputs
      pass the check of `broadcast`: the mask spreads, and so does a value of
      one value beside a longer mask.
    - `at_most`: at most `count` rows of the first input, as `top_k` with a
      literal count.
    - `of_input`: the length of the input at `count`, as the `element` input of
      `search_sorted`, which gives one index for each value of it.
    - `slice`: the rows of a slice with a literal `offset` and `count`.
    - `joined`: the rows of all the inputs one after the other, as `append`
      gives them, with no check of the lengths.
    - `frame`: the height of the frame, as `over` gives it.

    A definition names the rule of its kernel. A node whose kind fixes the
    rule, such as a binary operation, gets it from `Shape.kernel_length`.

    The values decide the length of the output of `values`, `masked` and
    `shrinks`, and of a rule that keeps such a length of an input. Such a
    length is the most rows that the node gives, and the data can give any
    other length. A check of lengths passes it beside any length, because
    the data can give the length that the check needs (`decided`).

    The code bounds the length of a literal, of one value, and of a slice
    with a literal length: no frame gives more rows. The rows of a column
    of the frame are a fact of the data. A kernel that reads an index or a
    count refuses one past a bound at every run (`bounded`).

    Example:
        >>> KernelLength.shrinks().form([RowForm.ONE_VALUE])
        <RowForm.ONE_ROW: 'one row'>
    """

    rule: str
    count: int = 0
    offset: int = 0

    CHECKS: ClassVar[frozenset[str]] = frozenset({"broadcast", "equal", "first", "masked"})
    DECIDED: ClassVar[frozenset[str]] = frozenset({"values", "masked", "shrinks"})

    @classmethod
    def broadcast(cls) -> KernelLength:
        """Give the rule of a kernel that broadcasts its inputs of length 1.

        Returns:
            The rule.

        Example:
            >>> KernelLength.broadcast().form([RowForm.ONE_ROW, RowForm.PER_ROW])
            <RowForm.PER_ROW: 'per row'>
        """
        return cls("broadcast")

    @classmethod
    def equal(cls) -> KernelLength:
        """Give the rule of a kernel that needs inputs of one length, with no broadcast.

        Returns:
            The rule.

        Example:
            >>> KernelLength.equal().form([RowForm.OTHER_LENGTH, RowForm.OTHER_LENGTH])
            <RowForm.OTHER_LENGTH: 'other length'>
        """
        return cls("equal")

    @classmethod
    def first(cls) -> KernelLength:
        """Give the rule of a kernel whose output has the length of its first input.

        Returns:
            The rule.

        Example:
            >>> KernelLength.first().form([RowForm.ONE_VALUE, RowForm.PER_ROW])
            <RowForm.ONE_ROW: 'one row'>
        """
        return cls("first")

    @classmethod
    def one(cls) -> KernelLength:
        """Give the rule of a kernel that gives one value.

        Returns:
            The rule.

        Example:
            >>> KernelLength.one().form([RowForm.PER_ROW])
            <RowForm.ONE_ROW: 'one row'>
        """
        return cls("one")

    @classmethod
    def shrinks(cls) -> KernelLength:
        """Give the rule of a kernel that gives at most the rows of its first input.

        Returns:
            The rule.

        Example:
            >>> KernelLength.shrinks().form([RowForm.PER_ROW])
            <RowForm.OTHER_LENGTH: 'other length'>
        """
        return cls("shrinks")

    @classmethod
    def values(cls) -> KernelLength:
        """Give the rule of a kernel whose length the values decide.

        Returns:
            The rule.

        Example:
            >>> KernelLength.values().form([RowForm.ONE_ROW])
            <RowForm.OTHER_LENGTH: 'other length'>
        """
        return cls("values")

    @classmethod
    def masked(cls) -> KernelLength:
        """Give the rule of a kernel whose length a mask decides, after a check of the lengths.

        Returns:
            The rule.

        Example:
            >>> KernelLength.masked().length([3, 1])
            3
        """
        return cls("masked")

    @classmethod
    def at_most(cls, count: int) -> KernelLength:
        """Give the rule of a kernel that keeps at most `count` rows of its first input.

        Args:
            count: The most rows.

        Returns:
            The rule.

        Example:
            >>> KernelLength.at_most(1).form([RowForm.PER_ROW])
            <RowForm.ONE_ROW: 'one row'>
        """
        return cls("at_most", count)

    @classmethod
    def of_input(cls, index: int) -> KernelLength:
        """Give the rule of a kernel whose output has the length of one input.

        Args:
            index: The position of that input.

        Returns:
            The rule.

        Example:
            >>> KernelLength.of_input(1).form([RowForm.PER_ROW, RowForm.OTHER_LENGTH])
            <RowForm.OTHER_LENGTH: 'other length'>
        """
        return cls("of_input", index)

    @classmethod
    def slice(cls, offset: int, length: int) -> KernelLength:
        """Give the rule of a slice with a literal offset and a literal length.

        `head(n)` is `slice(0, n)`, and `tail(n)` is `slice(-n, n)`.

        Args:
            offset: The offset. A negative offset counts from the end.
            length: The length.

        Returns:
            The rule.

        Example:
            >>> KernelLength.slice(0, 2).form([RowForm.PER_ROW])
            <RowForm.OTHER_LENGTH: 'other length'>
        """
        return cls("slice", length, offset)

    @classmethod
    def joined(cls) -> KernelLength:
        """Give the rule of a kernel that puts the rows of its inputs one after the other.

        Returns:
            The rule.

        Example:
            >>> KernelLength.joined().length([2, 3])
            5
        """
        return cls("joined")

    @classmethod
    def frame(cls) -> KernelLength:
        """Give the rule of a kernel whose output has the height of the frame.

        Returns:
            The rule.

        Example:
            >>> KernelLength.frame().form([RowForm.ONE_VALUE])
            <RowForm.PER_ROW: 'per row'>
        """
        return cls("frame")

    def form(self, inputs: Sequence[RowForm]) -> RowForm:
        """Give the row form of an output that the planner does not know to be one value.

        Where the rule needs inputs of one length, the form assumes that
        their lengths agree. The run checks the lengths on the data.

        Args:
            inputs: The row forms of the inputs.

        Returns:
            The row form. It is never one value, because the planner decides one value.

        Example:
            >>> KernelLength.broadcast().form([RowForm.ONE_VALUE, RowForm.ONE_ROW])
            <RowForm.ONE_ROW: 'one row'>
        """
        if self.rule in {"broadcast", "equal"}:
            longest = max(inputs, key=RowForm.rank)
            return RowForm.ONE_ROW if longest is RowForm.ONE_VALUE else longest
        if self.rule == "frame":
            return RowForm.PER_ROW
        if self.rule in {"first", "of_input"}:
            form = inputs[self.count if self.rule == "of_input" else 0]
            return RowForm.ONE_ROW if form is RowForm.ONE_VALUE else form
        return self._kept(inputs[0])

    def _kept(self, first: RowForm) -> RowForm:
        """Give the row form of a kernel that keeps some rows of its first input.

        Args:
            first: The row form of the first input.

        Returns:
            One row where the kernel keeps exactly one row, and another length otherwise.

        Example:
            >>> KernelLength.slice(-1, 1)._kept(RowForm.OTHER_LENGTH)
            <RowForm.ONE_ROW: 'one row'>
        """
        single = first.is_single()
        kept = {
            "one": True,
            "shrinks": single,
            "values": False,
            "masked": False,
            "joined": False,
            "at_most": self.count == 1 or (single and self.count > 0),
            "slice": self._window(1) == 1 if single else self.count == 1,
        }
        return RowForm.ONE_ROW if kept[self.rule] else RowForm.OTHER_LENGTH

    def length(self, lengths: Sequence[int], decided: Sequence[bool] = ()) -> int:
        """Give the number of rows of the output from the number of rows of each input.

        Where the values decide the length, this gives the length where
        each value gives one row: the common length of the inputs. `unique`
        and `mode` give at most that many rows, `filter` gives that many
        where its mask holds only `true`, and `explode` gives that many
        where each list holds one value.

        It raises `PolarsRefusal` where the inputs do not have lengths that
        the kernel accepts (`_checked`).

        Args:
            lengths: The number of rows of each input.
            decided: Whether the values decide the length of each input, or
                none where no input has such a length.

        Returns:
            The number of rows.

        Raises:
            NoLengthError: The rule is `frame`, whose length is not a length of the inputs.

        Example:
            >>> KernelLength.broadcast().length([1, 3, 3]), KernelLength.slice(-2, 5).length([3])
            (3, 2)
            >>> KernelLength.frame().length([3])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.options.NoLengthError: the rule frame gives the height of the ...
        """
        if self.rule == "frame":
            raise NoLengthError
        if self.rule in self.CHECKS:
            return self._checked(lengths, decided)
        counts = {
            "one": 1,
            "joined": sum(lengths),
            "of_input": lengths[min(self.count, len(lengths) - 1)],
            "at_most": min(self.count, lengths[0]),
            "slice": self._window(lengths[0]),
        }
        return counts.get(self.rule, lengths[0])

    def _checked(self, lengths: Sequence[int], decided: Sequence[bool] = ()) -> int:
        """Give the length of a kernel that checks that its inputs have one length.

        An input whose length the values decide passes the check: the data
        can give the length of the others.

        Args:
            lengths: The number of rows of each input.
            decided: Whether the values decide the length of each input.

        Returns:
            The length of the first input for the rule `first`. Otherwise the
            common length of the other inputs, or the longest input where
            those have one row.

        Raises:
            PolarsRefusal: Two inputs have other lengths. Polars raises a `ShapeError`.

        Example:
            >>> KernelLength.first()._checked([1, 3])
            1
            >>> KernelLength.broadcast()._checked([2, 3])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
            >>> KernelLength.broadcast()._checked([2, 3], [False, True])
            2
        """
        flags = list(decided) or [False] * len(lengths)
        fixed = [length for length, free in zip(lengths, flags, strict=True) if not free]
        common = self._common(fixed)
        if common is None:
            message = "the inputs have different lengths"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        if self.rule == "first":
            return lengths[0]
        return common if common != 1 or len(fixed) == len(lengths) else max(lengths)

    def decided(self, lengths: Sequence[int], decided: Sequence[bool]) -> bool:
        """Say whether the values decide the length of the output.

        Args:
            lengths: The number of rows of each input.
            decided: Whether the values decide the length of each input.

        Returns:
            Whether they do: always for `values`, `masked` and `shrinks`, and
            where the output keeps the length of such an input.

        Example:
            >>> KernelLength.masked().decided([3], [False])
            True
            >>> KernelLength.broadcast().decided([3, 3], [True, False])
            False
            >>> KernelLength.of_input(1).decided([3, 2], [False, True])
            True
            >>> KernelLength.first().decided([3, 1], [True, False])
            True
        """
        if self.rule in self.DECIDED or (self.rule == "joined" and any(decided)):
            return True
        if not any(decided) or self.rule in {"one", "frame"}:
            return False
        if self.rule == "of_input":
            return decided[min(self.count, len(decided) - 1)]
        if self.rule not in {"broadcast", "equal"}:
            return decided[0]
        pairs = zip(lengths, decided, strict=True)
        return not any(length != 1 and not free for length, free in pairs)

    def bounded(self, lengths: Sequence[int], bounded: Sequence[bool], length: int) -> bool:
        """Say whether the code bounds the length of the output: no frame gives more rows.

        Args:
            lengths: The number of rows of each input.
            bounded: Whether the code bounds the length of each input.
            length: The number of rows of the output.

        Returns:
            Whether it does: always for `one`, `slice` and `at_most`, and where
            the output takes its length from such an input.

        Example:
            >>> KernelLength.slice(0, 2).bounded([5], [False], 2)
            True
            >>> KernelLength.broadcast().bounded([5, 1], [False, True], 5)
            False
        """
        if self.rule in {"one", "slice", "at_most"}:
            return True
        if self.rule == "frame" or not bounded:
            return False
        if self.rule == "joined":
            return all(bounded)
        if self.rule == "of_input":
            return bounded[min(self.count, len(bounded) - 1)]
        if self.rule in {"first", "values", "shrinks"}:
            return bounded[0]
        pairs = zip(lengths, bounded, strict=True)
        return any(free for size, free in pairs if size == length)

    def _common(self, lengths: Sequence[int]) -> int | None:
        """Give the one length of inputs that must have one length.

        A kernel that broadcasts spreads an input of length 1 to the length of
        the others, and a kernel of the rules `first` and `values` checks its
        inputs in the same way. A kernel of the rule `equal` spreads nothing.

        Args:
            lengths: The number of rows of each input.

        Returns:
            The length, or `None` where two inputs have other lengths.

        Example:
            >>> KernelLength.equal()._common([1, 3]), KernelLength.broadcast()._common([1, 3])
            (None, 3)
        """
        spread = set() if self.rule == "equal" else {1}
        others = {length for length in lengths if length not in spread}
        if len(others) > 1:
            return None
        return next(iter(others), 1)

    def _window(self, rows: int) -> int:
        """Give the number of rows that a literal slice keeps of `rows` rows.

        A negative offset counts from the end, and the start and the stop are
        clamped to the rows.

        Args:
            rows: The number of rows of the input.

        Returns:
            The number of rows of the slice.

        Example:
            >>> KernelLength.slice(1, 1)._window(1), KernelLength.slice(-1, 1)._window(4)
            (0, 1)
        """
        start = self.offset + rows if self.offset < 0 else self.offset
        stop = start + self.count
        clamped_start = max(0, min(start, rows))
        clamped_stop = max(0, min(stop, rows))
        return clamped_stop - clamped_start


class NoLengthError(TypeError):
    """Report a request for the concrete length of a rule that does not read the inputs.

    A bug in the caller causes it: the rule `frame` gives the height of the
    frame, which the lengths of the inputs do not hold.

    Example:
        >>> str(NoLengthError())
        'the rule frame gives the height of the frame, not a length of the inputs'
    """

    def __init__(self) -> None:
        """Build the message.

        Example:
            >>> NoLengthError().args[0].startswith("the rule frame")
            True
        """
        super().__init__("the rule frame gives the height of the frame, not a length of the inputs")


class NodeKind(enum.Enum):
    """Name the kinds of node of a plan of polars that this vocabulary covers.

    Each value is the name of a kind of node in the plan of polars. A function,
    such as `abs` or `cum_sum`, is a `Function` with its options. The other
    kinds have no options. `sum` and `first` are `Agg`, `head` is `Slice`, and
    `a + b` is `BinaryExpr`.

    Example:
        >>> NodeKind.SLICE.value
        'Slice'
    """

    COLUMN = "Column"
    LITERAL = "Literal"
    LEN = "Len"
    ELEMENT = "Element"
    AGG = "Agg"
    BINARY = "BinaryExpr"
    TERNARY = "Ternary"
    CAST = "Cast"
    SORT = "Sort"
    SORT_BY = "SortBy"
    FILTER = "Filter"
    SLICE = "Slice"
    GATHER = "Gather"
    EXPLODE = "Explode"
    OVER = "Over"
    ROLLING = "Rolling"
    STRUCT_FIELD = "StructField"
    STRUCT_EVAL = "StructEval"
    FUNCTION = "Function"


@dataclass(frozen=True)
class ScalarInput:
    """Name an input of a function that the conversion of polars requires to give one value.

    Polars refuses a column at such an input, as the count of `shift`, with
    `ShapeError`. Its message names the parameter in one of two ways: most
    functions name the parameter alone, and the ranges and `index_of` name
    the parameter and the function.

    Example:
        >>> ScalarInput.named(1, "n").message
        "'n' must be a scalar value"
        >>> ScalarInput.passed(0, "start", "int_range").message
        'non-scalar start passed to `int_range`'
    """

    place: int
    message: str

    @classmethod
    def named(cls, place: int, parameter: str) -> ScalarInput:
        """Give an input whose refusal names the parameter alone.

        Args:
            place: The place of the input, from 0.
            parameter: The name of the parameter in polars.

        Returns:
            The input.

        Example:
            >>> ScalarInput.named(2, "fill_value").place
            2
        """
        return cls(place, f"'{parameter}' must be a scalar value")

    @classmethod
    def passed(cls, place: int, parameter: str, function: str) -> ScalarInput:
        """Give an input whose refusal names the parameter and the function.

        Args:
            place: The place of the input, from 0.
            parameter: The name of the parameter in polars.
            function: The name of the function in polars.

        Returns:
            The input.

        Example:
            >>> ScalarInput.passed(1, "value", "index_of").message
            'non-scalar value passed to `index_of`'
        """
        return cls(place, f"non-scalar {parameter} passed to `{function}`")


@dataclass(frozen=True)
class Shape:
    """Hold the facts of one node of a plan that decide the shape of its output.

    A function carries its options. A literal can be a scalar, or a series or
    a range of values. A gather can give one value, as `get` does. An `over`
    can explode its groups, or join each group into a list for each of its
    rows. Each other kind of node has no facts of its own.
    `length` is the rule of the kernel of a function, where it is not the
    default of its options (`kernel_length`). `scalars` names the inputs
    that the conversion of polars requires to give one value, as the count of
    `shift` (`check_inputs`).

    The shape gives three facts that later code reads:

    - the projection height of the node, which decides the list wrap in `agg`,
    - whether the node works value by value, which an `over` in `agg` needs of its key,
    - the row form of its output in a `select`.

    Example:
        >>> abs_ = Shape.function(FunctionOptions.elementwise())
        >>> abs_.form([RowForm.ONE_ROW]), abs_.wraps_in_agg([Height.SCALAR])
        (<RowForm.ONE_ROW: 'one row'>, False)
    """

    kind: NodeKind
    options: FunctionOptions = FunctionOptions()
    length: KernelLength | None = None
    scalar: bool = True
    returns_scalar: bool = False
    explode: bool = False
    joined: bool = False
    scalars: tuple[ScalarInput, ...] = ()

    FIXED_HEIGHTS: ClassVar[dict[NodeKind, Height]] = {
        NodeKind.COLUMN: Height.COLUMN,
        NodeKind.ELEMENT: Height.COLUMN,
        NodeKind.ROLLING: Height.COLUMN,
        NodeKind.STRUCT_FIELD: Height.UNKNOWN,
        NodeKind.FILTER: Height.UNKNOWN,
        NodeKind.SLICE: Height.UNKNOWN,
        NodeKind.EXPLODE: Height.UNKNOWN,
        NodeKind.AGG: Height.SCALAR,
        NodeKind.LEN: Height.SCALAR,
    }
    ZIPPED: ClassVar[frozenset[NodeKind]] = frozenset({
        NodeKind.BINARY,
        NodeKind.TERNARY,
        NodeKind.SORT_BY,
    })
    FIRST_HEIGHT: ClassVar[frozenset[NodeKind]] = frozenset({
        NodeKind.CAST,
        NodeKind.SORT,
        NodeKind.STRUCT_EVAL,
    })
    ELEMENTWISE: ClassVar[frozenset[NodeKind]] = frozenset({
        NodeKind.ELEMENT,
        NodeKind.BINARY,
        NodeKind.COLUMN,
        NodeKind.TERNARY,
        NodeKind.CAST,
        NodeKind.STRUCT_EVAL,
        NodeKind.STRUCT_FIELD,
    })
    KERNEL_LENGTHS: ClassVar[dict[NodeKind, KernelLength]] = {
        NodeKind.BINARY: KernelLength.broadcast(),
        NodeKind.TERNARY: KernelLength.broadcast(),
        NodeKind.SORT_BY: KernelLength.equal(),
        NodeKind.CAST: KernelLength.first(),
        NodeKind.SORT: KernelLength.first(),
        NodeKind.STRUCT_EVAL: KernelLength.first(),
        NodeKind.FILTER: KernelLength.masked(),
        NodeKind.EXPLODE: KernelLength.values(),
        NodeKind.STRUCT_FIELD: KernelLength.first(),
        NodeKind.GATHER: KernelLength.of_input(1),
        NodeKind.SLICE: KernelLength.values(),
        NodeKind.AGG: KernelLength.one(),
        NodeKind.LEN: KernelLength.one(),
        NodeKind.LITERAL: KernelLength.one(),
        NodeKind.COLUMN: KernelLength.frame(),
        NodeKind.ELEMENT: KernelLength.frame(),
        NodeKind.ROLLING: KernelLength.frame(),
        NodeKind.OVER: KernelLength.frame(),
    }

    @classmethod
    def of(cls, kind: NodeKind) -> Shape:
        """Give the shape of a kind of node that has no facts of its own.

        Args:
            kind: The kind, for example `NodeKind.BINARY` for `a + b`.

        Returns:
            The shape.

        Example:
            >>> Shape.of(NodeKind.AGG).height([Height.COLUMN])
            <Height.SCALAR: 'scalar'>
        """
        return cls(kind)

    @classmethod
    def function(cls, options: FunctionOptions, length: KernelLength | None = None) -> Shape:
        """Give the shape of a function with its options and the rule of its kernel.

        Args:
            options: The options that polars declares for the function.
            length: The rule of its kernel, or `None` for the default of its options.

        Returns:
            The shape.

        Example:
            >>> clip = Shape.function(FunctionOptions.elementwise(), KernelLength.first())
            >>> clip.form([RowForm.ONE_VALUE, RowForm.PER_ROW])
            <RowForm.ONE_ROW: 'one row'>
        """
        return cls(NodeKind.FUNCTION, options=options, length=length)

    def with_scalar_inputs(self, *inputs: ScalarInput) -> Shape:
        """Give the shape with inputs that the conversion of polars requires to give one value.

        Args:
            *inputs: Those inputs.

        Returns:
            The shape.

        Example:
            >>> count = ScalarInput.named(1, "n")
            >>> keeps = Shape.function(FunctionOptions.length_preserving())
            >>> shift = keeps.with_scalar_inputs(count)
            >>> [scalar.place for scalar in shift.scalars]
            [1]
        """
        return replace(self, scalars=inputs)

    def check_inputs(self, inputs: Sequence[Height]) -> None:
        """Refuse an input that must give one value, where its projection height is another.

        Polars converts the inputs of some functions first, and then checks
        that some of them give one value, as the count of `shift` and of
        `diff`. A column, or a slice such as `head(1)`, does not pass: its
        height is not one value. So the planner refuses the node, in a
        `select` and in `agg`, before any type function.

        Args:
            inputs: The heights of the inputs, before the cast pass.

        Raises:
            PolarsRefusal: An input of `scalars` does not give one value.

        Example:
            >>> count = ScalarInput.named(1, "n")
            >>> keeps = Shape.function(FunctionOptions.length_preserving())
            >>> shift = keeps.with_scalar_inputs(count)
            >>> shift.check_inputs([Height.COLUMN, Height.SCALAR])
            >>> shift.check_inputs([Height.COLUMN, Height.UNKNOWN])
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: 'n' must be a scalar value
        """
        for scalar in self.scalars:
            if inputs[scalar.place] is not Height.SCALAR:
                raise PolarsRefusal(ErrorClass.SHAPE, scalar.message)

    @classmethod
    def literal(cls, *, scalar: bool = True) -> Shape:
        """Give the shape of a literal: a scalar, or a series or a range of values.

        Args:
            scalar: Whether the literal is one value.

        Returns:
            The shape.

        Example:
            >>> Shape.literal(scalar=False).height([])
            <Height.RANGE: 'range'>
        """
        return cls(NodeKind.LITERAL, scalar=scalar)

    @classmethod
    def gather(cls, *, returns_scalar: bool) -> Shape:
        """Give the shape of a gather: `get` gives one value, and `gather` gives rows.

        Args:
            returns_scalar: Whether the gather gives one value.

        Returns:
            The shape.

        Example:
            >>> Shape.gather(returns_scalar=True).height([Height.COLUMN, Height.COLUMN])
            <Height.SCALAR: 'scalar'>
        """
        return cls(NodeKind.GATHER, returns_scalar=returns_scalar)

    @classmethod
    def over(cls, *, explode: bool = False, joined: bool = False) -> Shape:
        """Give the shape of an `over`, which can explode its groups or join each into a list.

        Args:
            explode: Whether the mapping strategy is `explode`.
            joined: Whether the mapping strategy is `join`.

        Returns:
            The shape.

        Example:
            >>> Shape.over(explode=True).height([Height.SCALAR])
            <Height.UNKNOWN: 'unknown'>
            >>> Shape.over(joined=True).joined
            True
        """
        return cls(NodeKind.OVER, explode=explode, joined=joined)

    def height(self, inputs: Sequence[Height]) -> Height:
        """Give the projection height of the node from the heights of its inputs.

        Args:
            inputs: The heights of the inputs, in the order of polars.

        Returns:
            The height.

        Example:
            >>> cum_sum = Shape.function(FunctionOptions.length_preserving())
            >>> cum_sum.height([Height.SCALAR]), cum_sum.height([Height.UNKNOWN])
            (<Height.SCALAR: 'scalar'>, <Height.UNKNOWN: 'unknown'>)
        """
        if self.kind in self.FIXED_HEIGHTS:
            return self.FIXED_HEIGHTS[self.kind]
        if self.kind in self.ZIPPED:
            return Height.zipped(inputs)
        if self.kind in self.FIRST_HEIGHT:
            return inputs[0]
        if self.kind is NodeKind.FUNCTION:
            return self._function_height(inputs)
        return self._own_height(inputs)

    def _function_height(self, inputs: Sequence[Height]) -> Height:
        """Give the height of a function from its flags.

        Args:
            inputs: The heights of the inputs.

        Returns:
            One value for an aggregation, the heights of the inputs side by
            side for a function that keeps their length, a range for a range,
            and an unknown length otherwise.

        Example:
            >>> Shape.function(FunctionOptions.groupwise())._function_height([Height.SCALAR])
            <Height.UNKNOWN: 'unknown'>
        """
        if self.options.returns_scalar():
            return Height.SCALAR
        if self.options.is_length_preserving():
            return Height.zipped(inputs)
        if self.options.is_range():
            return Height.RANGE
        return Height.UNKNOWN

    def _own_height(self, inputs: Sequence[Height]) -> Height:
        """Give the height of a literal, a gather or an `over`.

        A gather that does not give one value takes the height of its
        indices, where they have the height of the frame or a range.

        Args:
            inputs: The heights of the inputs.

        Returns:
            The height.

        Example:
            >>> Shape.gather(returns_scalar=False)._own_height([Height.COLUMN, Height.SCALAR])
            <Height.UNKNOWN: 'unknown'>
        """
        if self.kind is NodeKind.LITERAL:
            return Height.SCALAR if self.scalar else Height.RANGE
        if self.kind is NodeKind.OVER:
            return Height.UNKNOWN if self.explode else Height.COLUMN
        if self.returns_scalar:
            return Height.SCALAR
        indices = inputs[1]
        return indices if indices in {Height.COLUMN, Height.RANGE} else Height.UNKNOWN

    def wraps_in_agg(self, inputs: Sequence[Height]) -> bool:
        """Say whether `agg` wraps the values of each group of the node in a list.

        The planner adds an `implode` above each expression of `agg` whose
        height is not one value. It reads the expression before the cast pass
        changes it, so the heights of the inputs are those before the cast
        pass.

        Args:
            inputs: The heights of the inputs before the cast pass.

        Returns:
            Whether the node gives a list for each group.

        Example:
            >>> Shape.of(NodeKind.BINARY).wraps_in_agg([Height.SCALAR, Height.COLUMN])
            True
        """
        return self.height(inputs) is not Height.SCALAR

    def is_elementwise(self) -> bool:
        """Say whether the node itself works value by value.

        A tree is elementwise where each of its nodes is. An `over` in `agg`
        needs an elementwise partition key.

        Returns:
            Whether the node is elementwise, without its inputs.

        Example:
            >>> Shape.of(NodeKind.SORT).is_elementwise(), Shape.of(NodeKind.CAST).is_elementwise()
            (False, True)
        """
        if self.kind is NodeKind.FUNCTION:
            return self.options.is_elementwise()
        if self.kind is NodeKind.LITERAL:
            return self.scalar
        return self.kind in self.ELEMENTWISE

    def kernel_length(self) -> KernelLength:
        """Give the rule of the length of the kernel of the node.

        A function takes the rule of its definition, or else the rule of its
        options: one value for an aggregation, the broadcast for an
        elementwise function, the first input for a function that keeps the
        length of its input, and the values for any other function.

        Returns:
            The rule.

        Example:
            >>> Shape.function(FunctionOptions.length_preserving()).kernel_length().rule
            'first'
        """
        if self.length is not None:
            return self.length
        if self.kind is not NodeKind.FUNCTION:
            return self.KERNEL_LENGTHS[self.kind]
        if self.options.returns_scalar():
            return KernelLength.one()
        if self.options.is_elementwise():
            return KernelLength.broadcast()
        if self.options.is_length_preserving():
            return KernelLength.first()
        return KernelLength.values()

    def form(self, inputs: Sequence[RowForm]) -> RowForm:
        """Give the row form of the output of the node in a `select`.

        The planner decides one value (`height`). The kernel decides any
        other row form (`kernel_length`). Where the kernel needs inputs of
        one length, the row form assumes that their lengths agree.

        Args:
            inputs: The row forms of the inputs.

        Returns:
            The row form.

        Example:
            >>> head = Shape(NodeKind.SLICE, length=KernelLength.slice(0, 1))
            >>> head.form([RowForm.PER_ROW, RowForm.ONE_VALUE, RowForm.ONE_VALUE])
            <RowForm.ONE_ROW: 'one row'>
        """
        heights = [form.height() for form in inputs]
        if self.height(heights) is Height.SCALAR:
            return RowForm.ONE_VALUE
        return self.kernel_length().form(inputs)
