"""The overrides: rules for one node, where polars does what no part of the vocabulary states.

A definition names parts of the vocabulary, and each part translates one
mechanism of polars. A few behaviors of polars belong to one node only, for
example an internal error of polars in one rare case. Such a behavior has an
override: a rule for that one node, which stands in one field of its
definition. Each override states its condition in code, the reason that no
part of the vocabulary gives the behavior, and a probe that shows the
behavior in polars. A test runs each probe, so a new version of polars that
changes the behavior fails that test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.definitions.node_options import OptionValue, OptionValues
from astrolabe._rules.failures import PolarsRefusal, Stop

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.definitions.kernel_adapters import KernelContext, KernelRule
    from astrolabe._rules.definitions.node_options import NodeOptions
    from astrolabe._rules.dtypes import DType
    from astrolabe._rules.kernels.dispatch import Data


@dataclass(frozen=True)
class Override:
    """Name an override, and hold its reason and its probe.

    `node` names the call of polars whose rule the override gives. `reason`
    says why no part of the vocabulary gives the behavior. `probe` is a Python
    expression, over the module `pl` of polars, that shows the behavior: it
    raises or panics in the way that the override says. `stop` says how the
    probe stops, and `error` names the class of the error where it raises.
    `since` is the first release of polars that has the call, where an older
    release that the checker supports does not.

    Example:
        >>> Overrides.ALL[0].node
        'is_empty(ignore_nulls=True)'
    """

    node: str
    reason: str
    probe: str
    stop: Stop = Stop.PANICS
    error: str | None = None
    since: str | None = None


@dataclass(frozen=True)
class CountOfNullLiteral:
    """Give the kernel of `is_empty`, where it panics over a literal of nulls in `agg`.

    In `agg`, `is_empty` counts the values of each group of its input. With
    `ignore_nulls`, it counts the values that are not null, and a literal that
    holds a null reaches a branch of polars that states that it never runs:
    the process panics. A literal of a value, and a column of nulls, run. In a
    `select`, and without `ignore_nulls`, the kernel of the vocabulary runs.

    Example:
        >>> from astrolabe._rules.definitions.kernel_adapters import KernelContext, Reduction
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> from astrolabe._rules.kernels.reductions import ReductionKernels
        >>> boolean = ReductionKernels.BOOLEAN
        >>> kernel = CountOfNullLiteral(Reduction(boolean, boolean), ignore_nulls=True)
        >>> literal = KernelContext(grouped=True, scalars=(True,))
        >>> kernel.output([Data(DType("Null"))], literal)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: is_empty counts ...
    """

    OVERRIDE: ClassVar[Override] = Override(
        node="is_empty(ignore_nulls=True)",
        reason=(
            "In agg, is_empty counts the values of each group. Without nulls, the"
            " count of a literal of nulls reaches a branch of polars that is marked"
            " as never reached, and polars panics with an internal error. No other"
            " node counts a literal on that path."
        ),
        probe=(
            "pl.DataFrame({'g': [1, 2]}).group_by('g')"
            ".agg(pl.lit(None).is_empty(ignore_nulls=True))"
        ),
        since="1.41.0",
    )

    kernel: KernelRule
    ignore_nulls: bool | OptionValue[bool]

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output, or the panic over a literal of nulls in `agg`.

        Args:
            inputs: The data of the input.
            context: How the data reaches the kernel.

        Returns:
            The type of the kernel of the vocabulary.

        Raises:
            PolarsRefusal: The literal holds nulls, and the count leaves nulls out.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import KernelContext, Reduction
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.kernels.reductions import ReductionKernels
            >>> boolean = ReductionKernels.BOOLEAN
            >>> kernel = CountOfNullLiteral(Reduction(boolean, boolean), ignore_nulls=True)
            >>> kernel.output([Data(DType("Int8"))], KernelContext(grouped=True, scalars=(True,)))
            Boolean
        """
        ignore_nulls = OptionValues().known(self.ignore_nulls, "the flag ignore_nulls")
        literal = context.grouped and context.scalars[0]
        if ignore_nulls and literal and inputs[0].nulls:
            message = "is_empty counts the values of a literal of nulls"
            raise PolarsRefusal(Stop.PANICS, message)
        return self.kernel.output(inputs, context)

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say whether the kernel of the vocabulary can fail on one value.

        Args:
            inputs: The data of the input.

        Returns:
            Whether it can.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import Reduction
            >>> from astrolabe._rules.kernels.reductions import ReductionKernels
            >>> boolean = ReductionKernels.BOOLEAN
            >>> CountOfNullLiteral(Reduction(boolean, boolean), ignore_nulls=True).fallible([])
            False
        """
        return self.kernel.fallible(inputs)

    def checks_lengths(self) -> bool:
        """Say whether the kernel of the vocabulary checks the lengths of its inputs.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import Reduction
            >>> from astrolabe._rules.kernels.reductions import ReductionKernels
            >>> boolean = ReductionKernels.BOOLEAN
            >>> CountOfNullLiteral(Reduction(boolean, boolean), ignore_nulls=True).checks_lengths()
            False
        """
        return self.kernel.checks_lengths()

    def checks_group_lengths(self) -> bool:
        """Say whether the kernel of the vocabulary checks the lengths of a group.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import Reduction
            >>> from astrolabe._rules.kernels.reductions import ReductionKernels
            >>> boolean = ReductionKernels.BOOLEAN
            >>> kernel = CountOfNullLiteral(Reduction(boolean, boolean), ignore_nulls=True)
            >>> kernel.checks_group_lengths()
            False
        """
        return self.kernel.checks_group_lengths()

    def at(self, node: NodeOptions) -> CountOfNullLiteral:
        """Give the rule at the options of one node: the flag `ignore_nulls`.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import Reduction
            >>> from astrolabe._rules.definitions.node_options import NodeOption, NodeOptions
            >>> from astrolabe._rules.kernels.reductions import ReductionKernels
            >>> boolean = ReductionKernels.BOOLEAN
            >>> flag = NodeOption.flag("ignore_nulls")
            >>> kernel = CountOfNullLiteral(Reduction(boolean, boolean), ignore_nulls=flag)
            >>> kernel.at(NodeOptions({"Function": {"ignore_nulls": True}})).ignore_nulls
            True
        """
        ignore_nulls = OptionValues().at(self.ignore_nulls, node)
        return CountOfNullLiteral(self.kernel.at(node), ignore_nulls)


@dataclass(frozen=True)
class NullsGiveFirstInput:
    """Give the kernel of `sum_horizontal`, which leaves out its inputs of the type `Null`.

    The kernel of `sum_horizontal` adds only the inputs that reach it with
    another type than `Null`. Where every input reaches it as the type
    `Null`, it adds nothing and gives its first input back. The output then
    has the rows of the first input, and not the rows of the longest input
    that the rule of the node gives. A null literal first gives one row: a
    `select` beside a column of more rows raises, and `agg` raises where the
    groups hold more rows. From polars 1.40.0, the cast pass casts an input
    of the type `Null` beside another type, so only inputs that are all
    `Null` reach the kernel as `Null`. Before that release, a literal beside
    a column of the type `Null` gives the one row of the literal.

    Example:
        >>> from astrolabe._rules.definitions.kernel_adapters import AllInputs
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> from astrolabe._rules.kernels.horizontal import HorizontalSum
        >>> kernel = NullsGiveFirstInput(AllInputs(HorizontalSum(), own_lengths=False))
        >>> kernel.output_length([Data(DType("Null")), Data(DType("Null"))], (1, 3))
        1
    """

    OVERRIDE: ClassVar[Override] = Override(
        node="sum_horizontal over inputs of the type Null only",
        reason=(
            "The kernel of sum_horizontal leaves out the inputs of the type Null."
            " Where every input has the type Null, it returns its first input as"
            " it is. The output takes the rows of the inputs that it adds, or of"
            " the first input, where every other elementwise function takes the"
            " rows of its longest input."
        ),
        probe=(
            "pl.DataFrame({'g': [1, 1], 'n': [None, None]}).group_by('g')"
            ".agg(pl.sum_horizontal(pl.lit(None), pl.col('n')))"
        ),
        stop=Stop.RAISES,
        error="InvalidOperationError",
    )

    kernel: KernelRule

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output: the type of the kernel of the vocabulary.

        Args:
            inputs: The data of each input.
            context: How the data reaches the kernel.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import AllInputs, KernelContext
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.kernels.horizontal import HorizontalSum
            >>> kernel = NullsGiveFirstInput(AllInputs(HorizontalSum(), own_lengths=False))
            >>> context = KernelContext(grouped=False, scalars=(False,))
            >>> kernel.output([Data(DType("Null"))], context)
            Null
        """
        return self.kernel.output(inputs, context)

    def output_length(self, inputs: Sequence[Data], lengths: Sequence[int]) -> int | None:
        """Give the rows of the output where an input is of the type `Null`.

        Args:
            inputs: The data of each input, after the cast pass.
            lengths: The rows of each input.

        Returns:
            The rows of the first input where every input is `Null`, the rows of
            the longest input that is not `Null` where some are, and `None`
            where no input is `Null`.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import AllInputs
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.kernels.horizontal import HorizontalSum
            >>> kernel = NullsGiveFirstInput(AllInputs(HorizontalSum(), own_lengths=False))
            >>> kernel.output_length([Data(DType("Null")), Data(DType("Int8"))], (3, 1))
            1
            >>> kernel.output_length([Data(DType("Int8")), Data(DType("Int8"))], (3, 1)) is None
            True
            >>> kernel.output_length([Data(DType("Null"))], ()) is None
            True
        """
        if not lengths:
            return None
        added = [
            length
            for data, length in zip(inputs, lengths, strict=True)
            if data.dtype.name != "Null"
        ]
        if not added:
            return lengths[0]
        return max(added) if len(added) < len(inputs) else None

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say whether the kernel of the vocabulary can fail on one value.

        Args:
            inputs: The data of the inputs.

        Returns:
            Whether it can.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import AllInputs
            >>> from astrolabe._rules.kernels.horizontal import HorizontalSum
            >>> NullsGiveFirstInput(AllInputs(HorizontalSum(), own_lengths=False)).fallible([])
            False
        """
        return self.kernel.fallible(inputs)

    def checks_lengths(self) -> bool:
        """Say whether the kernel of the vocabulary checks the lengths of its inputs.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import AllInputs
            >>> from astrolabe._rules.kernels.horizontal import HorizontalSum
            >>> rule = NullsGiveFirstInput(AllInputs(HorizontalSum(), own_lengths=False))
            >>> rule.checks_lengths()
            False
        """
        return self.kernel.checks_lengths()

    def checks_group_lengths(self) -> bool:
        """Say whether the kernel of the vocabulary checks the lengths of a group.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import AllInputs
            >>> from astrolabe._rules.kernels.horizontal import HorizontalSum
            >>> rule = NullsGiveFirstInput(AllInputs(HorizontalSum(), own_lengths=False))
            >>> rule.checks_group_lengths()
            False
        """
        return self.kernel.checks_group_lengths()

    def at(self, node: NodeOptions) -> NullsGiveFirstInput:
        """Give the rule at the options of one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> from astrolabe._rules.definitions.kernel_adapters import AllInputs
            >>> from astrolabe._rules.definitions.node_options import NodeOptions
            >>> from astrolabe._rules.kernels.horizontal import HorizontalSum
            >>> rule = NullsGiveFirstInput(AllInputs(HorizontalSum(), own_lengths=False))
            >>> rule.at(NodeOptions({"Function": {}})).kernel.own_lengths
            False
        """
        return NullsGiveFirstInput(self.kernel.at(node))


class Overrides:
    """List every override, so that a test can run the probe of each.

    Example:
        >>> [override.node for override in Overrides.ALL]
        ['is_empty(ignore_nulls=True)', 'sum_horizontal over inputs of the type Null only']
    """

    ALL: ClassVar[tuple[Override, ...]] = (
        CountOfNullLiteral.OVERRIDE,
        NullsGiveFirstInput.OVERRIDE,
    )
