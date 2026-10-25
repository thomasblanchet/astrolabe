"""Give the type of the data that the kernel of a node gives when the query runs.

The planner type of a node is a promise. When the query runs, the kernel of
the node reads the data that reaches it, and dispatches on the type of that
data. That type can differ from the planner type, and the kernel can refuse
it. Kernels take their inputs in different ways, for example one input, every
input, or the groups of an aggregation. Each class here adapts one way to the
one call that code that evaluates a node makes (`KernelRule`). It reads the
data and the row counts of the inputs from a `KernelContext`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from astrolabe._rules.definitions.node_options import (
    NodeOptions,
    OptionValue,
    OptionValues,
)
from astrolabe._rules.kernels.binary import OperatorKernel
from astrolabe._rules.operators import Operator

from astrolabe._rules.dtypes import DType
from astrolabe._rules.failures import ErrorClass, PolarsRefusal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from astrolabe._rules.kernels.dispatch import (
        BinaryKernel,
        Data,
        Dispatch,
        InputsKernel,
        LiteralsKernel,
    )
    from astrolabe._rules.kernels.windows import RollingIndex
    from astrolabe._rules.type_functions import Operand


@dataclass(frozen=True)
class KernelContext:
    """Say how the data reaches the kernel of a node.

    `grouped` says that the kernel runs in `agg`, on the values of each group.
    `scalars` says, for each input, whether the input arrives as one value, as
    a literal or an aggregation does, and not as a column of rows. Some
    kernels of polars take another path there. `lengths` gives the number of
    rows of each input at the call, where the code that evaluates the node
    knows them. A kernel that checks the lengths of its inputs reads them.
    `planned` gives the type of the node at the planner types of its inputs.
    Polars computes it when it converts the expression, and some functions
    keep it as an option that their kernel reads (`PlannedInput`).
    `computed` gives each input as the run computes it, after the cast pass:
    a literal with its value, or a column. A kernel that reads the value of a
    literal input reads it there (`LiteralInputs`). `decided` says, for each
    input, that the values decide its length, as for a `filter`: the data
    can then give any number of rows. `lists` says that the run in `agg`
    reads the inputs group by group, or as lists of the groups, and not as
    flat values.

    Example:
        >>> KernelContext(grouped=False, scalars=(False, True)).scalars
        (False, True)
        >>> KernelContext(grouped=False, scalars=(False,), lengths=(5,)).lengths
        (5,)
    """

    grouped: bool
    scalars: tuple[bool, ...]
    lengths: tuple[int, ...] = ()
    planned: DType | None = None
    computed: tuple[Operand, ...] = ()
    decided: tuple[bool, ...] = ()
    lists: bool = False

    def fitted(self) -> list[tuple[int, ...]]:
        """Give the lengths that the data can give an input whose length the values decide.

        Such an input passes a check of lengths beside any other length. A
        kernel that checks the lengths itself can then meet one row there, or
        the rows of the first input whose length the values do not decide.

        Returns:
            The lengths of the inputs for each of the two, or none where the
            values decide no length that the context knows.

        Example:
            >>> context = KernelContext(
            ...     grouped=False,
            ...     scalars=(False, False),
            ...     lengths=(5, 3),
            ...     decided=(False, True),
            ... )
            >>> context.fitted()
            [(5, 1), (5, 5)]
        """
        if not self.lengths or not any(self.decided):
            return []
        pairs = list(zip(self.lengths, self.decided, strict=True))
        fixed = [length for length, free in pairs if not free]
        sizes = dict.fromkeys([1, *fixed[:1]])
        return [tuple(size if free else length for length, free in pairs) for size in sizes]


@runtime_checkable
class OwnLength(Protocol):
    """Give the rows of the output of a kernel where the types of its inputs decide them.

    The rule of a node gives the length of its output from the lengths of
    its inputs (`KernelLength`). A few kernels of polars give another length
    for some types, and such a kernel rule also has this method. Such a
    kernel gives the rows of its first input, so the output takes whether
    the values decide those rows.
    """

    def output_length(self, inputs: Sequence[Data], lengths: Sequence[int]) -> int | None:
        """Give the rows of the output, or `None` where the rule of the node gives them.

        Args:
            inputs: The data of each input, after the cast pass.
            lengths: The rows of each input.
        """


class NoPlannedTypeError(TypeError):
    """Report a kernel that reads the planned type of its node, where the context gives none.

    A fault of the code that evaluates a node causes it: that code must give
    the planned type in the context of the kernel (`KernelContext.planned`).

    Example:
        >>> str(NoPlannedTypeError("reinterpret"))
        'the kernel reinterpret reads the planned type of the node, and the context gives none'
    """

    def __init__(self, kernel: str) -> None:  # pragma: no cover -- a fault of the evaluation
        """Build the message.

        Args:
            kernel: The name of the kernel.

        Example:
            >>> NoPlannedTypeError("cut").args[0].startswith("the kernel cut")
            True
        """
        message = (
            f"the kernel {kernel} reads the planned type of the node, and the context gives none"
        )
        super().__init__(message)


class KernelRule(Protocol):
    """Give the run type of a node from the data that reaches its kernel."""

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output, or raise `PolarsRefusal` where the kernel fails.

        Args:
            inputs: The data of each input.
            context: How the data reaches the kernel.
        """

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say whether the kernel can fail on one value, which changes its run in `agg`.

        Args:
            inputs: The data of each input.
        """

    def checks_lengths(self) -> bool:
        """Say whether the kernel checks the lengths of its inputs itself, with its own error."""

    def checks_group_lengths(self) -> bool:
        """Say whether the kernel checks the lengths of a group itself, on the path of groups."""

    def at(self, node: NodeOptions) -> KernelRule:
        """Give the rule at the options of one node.

        Args:
            node: The serialized root of the node.
        """


@dataclass(frozen=True)
class Exploded:
    """Give the kernel of `explode`: the values of each list, or the input as it is.

    Over a list or an array, `explode` gives the values of each one, so the
    values decide the rows of its output. Over a flat input, it gives the
    input as it is, with its rows (`OwnLength`).

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> from astrolabe._rules.kernels.numeric import NumericKernels
        >>> Exploded(OneInput(NumericKernels.ABS)).output_length([Data(DType("Int8"))], (5,))
        5
    """

    kernel: OneInput

    NESTED: ClassVar[frozenset[str]] = frozenset({"List", "Array"})

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output: the type of the kernel of one input.

        Args:
            inputs: The data of the input.
            context: How the data reaches the kernel.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> context = KernelContext(grouped=False, scalars=(False,))
            >>> Exploded(OneInput(NumericKernels.ABS)).output([Data(DType("Int8"))], context)
            Int8
        """
        return self.kernel.output(inputs, context)

    def output_length(self, inputs: Sequence[Data], lengths: Sequence[int]) -> int | None:
        """Give the rows of a flat input, or `None` where the values of the lists decide them.

        Args:
            inputs: The data of the input, after the cast pass.
            lengths: The rows of the input.

        Returns:
            The rows, or `None`.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> listed = Data(DType("List", (DType("Int8"),)))
            >>> Exploded(OneInput(NumericKernels.ABS)).output_length([listed], (5,)) is None
            True
        """
        nested = inputs[0].dtype.name in self.NESTED
        return None if nested or not lengths else lengths[0]

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say that the kernel does not fail on one value.

        Args:
            inputs: The data of the input.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> Exploded(OneInput(NumericKernels.ABS)).fallible([])
            False
        """
        return self.kernel.fallible(inputs)

    def checks_lengths(self) -> bool:
        """Say that the kernel checks no lengths: it has one input.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> Exploded(OneInput(NumericKernels.ABS)).checks_lengths()
            False
        """
        return self.kernel.checks_lengths()

    def checks_group_lengths(self) -> bool:
        """Say that the run of the node checks no lengths of a group before the kernel.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> Exploded(OneInput(NumericKernels.ABS)).checks_group_lengths()
            False
        """
        return self.kernel.checks_group_lengths()

    def at(self, node: NodeOptions) -> Exploded:
        """Give the kernel of one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The kernel.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> kernel = Exploded(OneInput(NumericKernels.ABS))
            >>> kernel.at(NodeOptions({"Explode": {}})) == kernel
            True
        """
        return Exploded(self.kernel.at(node))


@dataclass(frozen=True)
class OneInput:
    """Give the kernel of a function of one input: one dispatch on the data type.

    The dispatch can be built from an option of the node, as the kernel of
    `dt.cast_time_unit` is built from its unit.

    Example:
        >>> from astrolabe._rules.kernels.numeric import NumericKernels
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> context = KernelContext(grouped=False, scalars=(False,))
        >>> OneInput(NumericKernels.ABS).output([Data(DType("Int8"))], context)
        Int8
    """

    dispatch: Dispatch | OptionValue[Dispatch]

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the input.
            context: How the data reaches the kernel, which the dispatch does not read.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.kernels.floats import FloatKernels
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> context = KernelContext(grouped=True, scalars=(False,))
            >>> OneInput(FloatKernels.SQRT).output([Data(DType("String"))], context)
            Float64
        """
        del context
        dispatch = OptionValues().known(self.dispatch, "the kernel")
        return dispatch.output(inputs[0])

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say that the kernel does not fail on one value.

        Args:
            inputs: The data of the input.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> OneInput(NumericKernels.ABS).fallible([])
            False
        """
        del inputs
        return False

    def checks_lengths(self) -> bool:
        """Say that the kernel checks no lengths: it has one input.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> OneInput(NumericKernels.ABS).checks_lengths()
            False
        """
        return False

    def checks_group_lengths(self) -> bool:
        """Say that the run of the node checks the lengths of a group before the kernel.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> OneInput(NumericKernels.ABS).checks_group_lengths()
            False
        """
        return False

    def at(self, node: NodeOptions) -> OneInput:
        """Give the kernel of one node, where the dispatch is built from its options.

        Args:
            node: The serialized root of the node.

        Returns:
            The kernel.

        Example:
            >>> from astrolabe._rules.definitions.node_options import FromOptions, NodeOption
            >>> from astrolabe._rules.kernels.dates import DateKernels
            >>> unit = NodeOption.unit("function", "TemporalExpr", "CastTimeUnit")
            >>> kernel = OneInput(FromOptions(DateKernels().cast_time_unit, unit))
            >>> function = {"TemporalExpr": {"CastTimeUnit": "Milliseconds"}}
            >>> kernel.at(NodeOptions({"Function": {"function": function}})).dispatch.name
            'cast_time_unit'
        """
        return OneInput(OptionValues().at(self.dispatch, node))


@dataclass(frozen=True)
class TwoInputs:
    """Give the kernel of a function of two inputs, such as the fill of `fill_null`.

    The kernel can be built from options of the node (`FromOptions`).

    Example:
        >>> from astrolabe._rules.kernels.fill import FillNull
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> context = KernelContext(grouped=False, scalars=(False, False))
        >>> TwoInputs(FillNull()).output([Data(DType("Null")), Data(DType("Int8"))], context)
        Int8
    """

    kernel: BinaryKernel | OptionValue[BinaryKernel]

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the two inputs.
            context: How the data reaches the kernel, which this kernel does not read.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.kernels.fill import FillNull
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> context = KernelContext(grouped=True, scalars=(False, False))
            >>> TwoInputs(FillNull()).output([Data(DType("Int8"))] * 2, context)
            Int8
        """
        del context
        kernel = OptionValues().known(self.kernel, "the kernel")
        return kernel.output(inputs[0], inputs[1])

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say that the kernel does not fail on one value.

        Args:
            inputs: The data of the two inputs.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.fill import FillNull
            >>> TwoInputs(FillNull()).fallible([])
            False
        """
        del inputs
        return False

    def checks_lengths(self) -> bool:
        """Say that the run of the node checks the lengths before the kernel.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.fill import FillNull
            >>> TwoInputs(FillNull()).checks_lengths()
            False
        """
        return False

    def checks_group_lengths(self) -> bool:
        """Say that the run of the node checks the lengths of a group before the kernel.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.fill import FillNull
            >>> TwoInputs(FillNull()).checks_group_lengths()
            False
        """
        return False

    def at(self, node: NodeOptions) -> TwoInputs:
        """Give the kernel of one node, where the kernel is built from its options.

        Args:
            node: The serialized root of the node.

        Returns:
            The kernel.

        Example:
            >>> from astrolabe._rules.kernels.fill import FillNull
            >>> type(TwoInputs(FillNull()).at(NodeOptions({"Function": {}})).kernel).__name__
            'FillNull'
        """
        return TwoInputs(OptionValues().at(self.kernel, node))


@dataclass(frozen=True)
class AllInputs:
    """Give the kernel of a function that reads every input and the number of rows of each.

    Such a kernel checks the lengths of its inputs itself, after it checks
    their types, as the parse of `str.strptime` does. The kernel can be built
    from options of the node. `can_fail` says that the kernel can fail on one
    value, as a strict parse of text can, and it can be an option of the node,
    as for `list.get`. `group_lengths` says that, where the
    kernel runs once for each group, polars calls it with no check of the
    lengths before it, as for `rolling_mean_by`. `own_lengths` says that the
    kernel checks the lengths of its inputs itself. Where it does not, as for
    a `filter`, the run of the node checks them first. An input whose length
    the values decide passes the check of the kernel, because the data can
    give the length that the check needs (`KernelContext.fitted`).

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> from astrolabe._rules.kernels.floats import Log
        >>> log = AllInputs(Log())
        >>> context = KernelContext(grouped=False, scalars=(False, True), lengths=(5, 1))
        >>> log.output([Data(DType("Float32")), Data(DType("Float64"))], context)
        Float32
    """

    kernel: InputsKernel | OptionValue[InputsKernel]
    can_fail: bool | OptionValue[bool] = False
    group_lengths: bool = False
    own_lengths: bool = True

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            context: How the data reaches the kernel: the kernel reads the lengths.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The kernel refuses the inputs at every length that
                the data can give them.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.kernels.horizontal import HorizontalSum
            >>> from astrolabe._rules.kernels.windows import PercentChange
            >>> total = AllInputs(HorizontalSum())
            >>> context = KernelContext(grouped=True, scalars=(False, False))
            >>> total.output([Data(DType("Int8")), Data(DType("Null"))], context)
            Int8
            >>> unique = KernelContext(
            ...     grouped=False,
            ...     scalars=(False, False),
            ...     lengths=(5, 3),
            ...     decided=(False, True),
            ... )
            >>> AllInputs(PercentChange()).output([Data(DType("Int8"))] * 2, unique)
            Float64
        """
        kernel = OptionValues().known(self.kernel, "the kernel")
        try:
            return kernel.output(inputs, context.lengths)
        except PolarsRefusal:
            for lengths in context.fitted():
                with contextlib.suppress(PolarsRefusal):
                    return kernel.output(inputs, lengths)
            raise

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say whether the kernel can fail on one value.

        Args:
            inputs: The data of the inputs, which this answer does not read.

        Returns:
            Whether it can.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.floats import Log
            >>> AllInputs(Log(), can_fail=True).fallible([])
            True
        """
        del inputs
        return OptionValues().known(self.can_fail, "whether the kernel can fail")

    def checks_lengths(self) -> bool:
        """Say whether the kernel checks the lengths of its inputs itself (`own_lengths`).

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.kernels.floats import Log
            >>> AllInputs(Log()).checks_lengths()
            True
        """
        return self.own_lengths

    def checks_group_lengths(self) -> bool:
        """Say whether the kernel checks the lengths of a group itself (`group_lengths`).

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.kernels.floats import Log
            >>> AllInputs(Log()).checks_group_lengths()
            False
        """
        return self.group_lengths

    def at(self, node: NodeOptions) -> AllInputs:
        """Give the kernel of one node, where the kernel is built from its options.

        Args:
            node: The serialized root of the node.

        Returns:
            The kernel.

        Example:
            >>> from astrolabe._rules.definitions.node_options import FromOptions, NodeOption
            >>> from astrolabe._rules.kernels.horizontal import HorizontalSum
            >>> total = AllInputs(FromOptions(HorizontalSum, NodeOption.flag("ignore_nulls")))
            >>> total.at(NodeOptions({"Function": {"ignore_nulls": False}})).kernel
            HorizontalSum(ignore_nulls=False)
        """
        values = OptionValues()
        kernel = values.at(self.kernel, node)
        can_fail = values.at(self.can_fail, node)
        return AllInputs(kernel, can_fail, self.group_lengths, self.own_lengths)


@dataclass(frozen=True)
class LiteralInputs:
    """Give the kernel of a function that reads every input, its rows, and the value of a literal.

    The code holds the value of a literal input, and some kernels read it:
    `dt.truncate` parses its literal duration, and `dt.replace` casts each
    literal part strictly. So a value that the kernel refuses refuses every
    run. The kernel checks the lengths of its inputs itself, as a kernel of
    `AllInputs` does, and it can be built from options of the node.
    `can_fail` says that the kernel can fail on one value, as `arr.get` can
    where `null_on_oob` is off, and it can be an option of the node.
    `group_lengths` says that, where the kernel runs once for each group,
    polars calls it with no check of the lengths before it, as for `diff`.
    `groups` names another kernel for `agg`, where polars runs another
    physical path there, as a gather does.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> from astrolabe._rules.kernels.dates import EveryKernel
        >>> from astrolabe._rules.type_functions import Operand
        >>> moment, text = Data(DType("Date")), Data(DType("String"))
        >>> computed = (Operand.column(DType("Date")), Operand.of_value("1mo"))
        >>> context = KernelContext(grouped=False, scalars=(False, True), computed=computed)
        >>> LiteralInputs(EveryKernel("truncate")).output([moment, text], context)
        Date
    """

    kernel: LiteralsKernel | OptionValue[LiteralsKernel]
    can_fail: bool | OptionValue[bool] = False
    group_lengths: bool = False
    own_lengths: bool = True
    groups: LiteralsKernel | OptionValue[LiteralsKernel] | None = None

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            context: How the data reaches the kernel: the kernel reads the
                lengths and the computed inputs.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.kernels.dates import EveryKernel
            >>> from astrolabe._rules.type_functions import Operand
            >>> moment, text = Data(DType("Date")), Data(DType("String"))
            >>> computed = (Operand.column(DType("Date")), Operand.of_value("x"))
            >>> context = KernelContext(grouped=True, scalars=(False, True), computed=computed)
            >>> LiteralInputs(EveryKernel("round")).output([moment, text], context)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        kernel = OptionValues().known(self.kernel, "the kernel")
        groups = OptionValues().known(self.groups, "the kernel in `agg`")
        if context.grouped and groups is not None:
            kernel = groups
        return kernel.output(inputs, context.lengths, context.computed)

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say whether the kernel can fail on one value in the sense of polars (`can_fail`).

        Polars marks only a few functions as able to fail on a value that a
        group leaves out, such as `arr.get`.

        Args:
            inputs: The data of the inputs, which this answer does not read.

        Returns:
            Whether it can.

        Example:
            >>> from astrolabe._rules.kernels.dates import EveryKernel
            >>> LiteralInputs(EveryKernel("round")).fallible([])
            False
        """
        del inputs
        return OptionValues().known(self.can_fail, "whether the kernel can fail")

    def checks_lengths(self) -> bool:
        """Say whether the kernel checks the lengths of its inputs itself (`own_lengths`).

        Where it does not, as for a gather, the run of the node checks them first.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.kernels.dates import EveryKernel
            >>> LiteralInputs(EveryKernel("round")).checks_lengths()
            True
        """
        return self.own_lengths

    def checks_group_lengths(self) -> bool:
        """Say whether the kernel checks the lengths of a group itself (`group_lengths`).

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.kernels.dates import EveryKernel
            >>> LiteralInputs(EveryKernel("round")).checks_group_lengths()
            False
        """
        return self.group_lengths

    def at(self, node: NodeOptions) -> LiteralInputs:
        """Give the kernel of one node, where the kernel is built from its options.

        Args:
            node: The serialized root of the node.

        Returns:
            The kernel.

        Example:
            >>> from astrolabe._rules.kernels.dates import EveryKernel
            >>> kernel = LiteralInputs(EveryKernel("round"))
            >>> kernel.at(NodeOptions({"Function": {}})).kernel
            EveryKernel(name='round')
        """
        values = OptionValues()
        kernel = values.at(self.kernel, node)
        can_fail = values.at(self.can_fail, node)
        groups = values.at(self.groups, node)
        return LiteralInputs(kernel, can_fail, self.group_lengths, self.own_lengths, groups)


@dataclass(frozen=True)
class LiteralReduction:
    """Give the kernel of an aggregation that reads a literal input, with a kernel for each context.

    `quantile` reads the value of its literal quantile, and runs one kernel in
    a `select` (`whole`) and another in `agg` (`groups`). The kernel of the
    groups runs over every input, a literal included, and reads whether the
    input is a literal itself. Each kernel checks the lengths of its inputs.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> from astrolabe._rules.kernels.reductions import GroupQuantile, Quantile
        >>> from astrolabe._rules.type_functions import Operand
        >>> computed = (Operand.column(DType("Int8")), Operand.of_value(1.5))
        >>> context = KernelContext(grouped=True, scalars=(False, True), computed=computed)
        >>> inputs = [Data(DType("Int8")), Data(DType("Float64"))]
        >>> LiteralReduction(Quantile(), GroupQuantile()).output(inputs, context)
        Int8
    """

    whole: LiteralsKernel
    groups: LiteralsKernel

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            context: How the data reaches the kernel.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.kernels.reductions import GroupQuantile, Quantile
            >>> from astrolabe._rules.type_functions import Operand
            >>> computed = (Operand.column(DType("Int8")), Operand.of_value(0.5))
            >>> context = KernelContext(grouped=False, scalars=(False, True), computed=computed)
            >>> inputs = [Data(DType("Int8")), Data(DType("Float64"))]
            >>> LiteralReduction(Quantile(), GroupQuantile()).output(inputs, context)
            Float64
        """
        kernel = self.groups if context.grouped else self.whole
        return kernel.output(inputs, context.lengths, context.computed)

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say that the kernel does not fail on one value in the sense of polars.

        Args:
            inputs: The data of the inputs, which this answer does not read.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.reductions import GroupQuantile, Quantile
            >>> LiteralReduction(Quantile(), GroupQuantile()).fallible([])
            False
        """
        del inputs
        return False

    def checks_lengths(self) -> bool:
        """Say that the kernel checks the lengths of its inputs itself.

        Returns:
            True.

        Example:
            >>> from astrolabe._rules.kernels.reductions import GroupQuantile, Quantile
            >>> LiteralReduction(Quantile(), GroupQuantile()).checks_lengths()
            True
        """
        return True

    def checks_group_lengths(self) -> bool:
        """Say that the run of the node checks the lengths of a group before the kernel.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.reductions import GroupQuantile, Quantile
            >>> LiteralReduction(Quantile(), GroupQuantile()).checks_group_lengths()
            False
        """
        return False

    def at(self, node: NodeOptions) -> LiteralReduction:
        """Give the kernel, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same kernel.

        Example:
            >>> from astrolabe._rules.kernels.reductions import GroupQuantile, Quantile
            >>> kernel = LiteralReduction(Quantile(), GroupQuantile())
            >>> kernel.at(NodeOptions({"Function": {}})) is kernel
            True
        """
        del node
        return self


@dataclass(frozen=True)
class Reduction:
    """Give the kernel of an aggregation: one kernel for a whole input, another for each group.

    In a `select`, an aggregation reduces its input to one value (`whole`). In
    `agg`, it runs a kernel of its own on the values of each group (`groups`),
    and the two kernels do not accept the same types. A literal in `agg` is
    one value for all the groups, so it takes the kernel of a whole input.

    Example:
        >>> from astrolabe._rules.kernels.numeric import NumericKernels
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> sum_ = Reduction(NumericKernels.SUM, NumericKernels.SUM_GROUPS)
        >>> sum_.output([Data(DType("Date"))], KernelContext(grouped=True, scalars=(False,)))
        Date
    """

    whole: Dispatch
    groups: Dispatch

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the input.
            context: How the data reaches the kernel.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> sum_ = Reduction(NumericKernels.SUM, NumericKernels.SUM_GROUPS)
            >>> literal = KernelContext(grouped=True, scalars=(True,))
            >>> sum_.output([Data(DType("Boolean"))], literal)
            UInt32
        """
        grouped = context.grouped and not context.scalars[0]
        dispatch = self.groups if grouped else self.whole
        return dispatch.output(inputs[0])

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say that the kernel does not fail on one value.

        Args:
            inputs: The data of the input.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> Reduction(NumericKernels.SUM, NumericKernels.SUM_GROUPS).fallible([])
            False
        """
        del inputs
        return False

    def checks_lengths(self) -> bool:
        """Say that the kernel checks no lengths: it has one input.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> Reduction(NumericKernels.SUM, NumericKernels.SUM_GROUPS).checks_lengths()
            False
        """
        return False

    def checks_group_lengths(self) -> bool:
        """Say that the run of the node checks the lengths of a group before the kernel.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> Reduction(NumericKernels.SUM, NumericKernels.SUM_GROUPS).checks_group_lengths()
            False
        """
        return False

    def at(self, node: NodeOptions) -> Reduction:
        """Give the kernel, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same kernel.

        Example:
            >>> from astrolabe._rules.kernels.numeric import NumericKernels
            >>> sum_ = Reduction(NumericKernels.SUM, NumericKernels.SUM_GROUPS)
            >>> sum_.at(NodeOptions({"Agg": {}})) is sum_
            True
        """
        del node
        return self


@dataclass(frozen=True)
class ItemReduction:
    """Give the kernel of `item`, which needs exactly one row, or no row with `allow_empty`.

    `item` gives the one value of its input, and raises over any other
    number of rows. With `allow_empty=True`, an input of no row gives a null.
    Where the values decide the length of the input, as after a `filter`,
    the data can give one row, so the run passes. Otherwise the kernel of
    the vocabulary checks the rows (`Single`).

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> from astrolabe._rules.kernels.reductions import ReductionKernels
        >>> item = ReductionKernels.ITEM
        >>> kernel = ItemReduction(Reduction(item, item), allow_empty=True)
        >>> empty = KernelContext(grouped=False, scalars=(False,), lengths=(0,))
        >>> kernel.output([Data(DType("Int8"))], empty)
        Int8
    """

    kernel: Reduction
    allow_empty: bool | OptionValue[bool]

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output, or the refusal of a number of rows other than one.

        Args:
            inputs: The data of the input.
            context: How the data reaches the kernel.

        Returns:
            The type of the input.

        Raises:
            PolarsRefusal: The input has no row, and `allow_empty` is not set.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.kernels.reductions import ReductionKernels
            >>> item = ReductionKernels.ITEM
            >>> kernel = ItemReduction(Reduction(item, item), allow_empty=False)
            >>> filtered = KernelContext(grouped=False, scalars=(False,), decided=(True,))
            >>> kernel.output([Data(DType("Int8"))], filtered)
            Int8
        """
        allow_empty = OptionValues().known(self.allow_empty, "the flag allow_empty")
        empty = context.lengths[:1] == (0,)
        if (empty and allow_empty) or any(context.decided[:1]):
            return inputs[0].dtype
        if empty:
            raise PolarsRefusal(ErrorClass.COMPUTE, "item expected a single value, got none")
        return self.kernel.output(inputs, context)

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say that the kernel does not fail on one value.

        Args:
            inputs: The data of the input.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.reductions import ReductionKernels
            >>> item = ReductionKernels.ITEM
            >>> ItemReduction(Reduction(item, item), allow_empty=False).fallible([])
            False
        """
        return self.kernel.fallible(inputs)

    def checks_lengths(self) -> bool:
        """Say that the kernel checks no lengths: it has one input.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.reductions import ReductionKernels
            >>> item = ReductionKernels.ITEM
            >>> ItemReduction(Reduction(item, item), allow_empty=False).checks_lengths()
            False
        """
        return self.kernel.checks_lengths()

    def checks_group_lengths(self) -> bool:
        """Say that the run of the node checks the lengths of a group before the kernel.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.reductions import ReductionKernels
            >>> item = ReductionKernels.ITEM
            >>> ItemReduction(Reduction(item, item), allow_empty=False).checks_group_lengths()
            False
        """
        return self.kernel.checks_group_lengths()

    def at(self, node: NodeOptions) -> ItemReduction:
        """Give the kernel of one node, with its flag `allow_empty`.

        Args:
            node: The serialized root of the node.

        Returns:
            The kernel.

        Example:
            >>> from astrolabe._rules.definitions.node_options import NodeOption
            >>> from astrolabe._rules.kernels.reductions import ReductionKernels
            >>> item = ReductionKernels.ITEM
            >>> flag = NodeOption.flag("Item", "allow_empty")
            >>> node = NodeOptions({"Agg": {"Item": {"allow_empty": True}}})
            >>> ItemReduction(Reduction(item, item), flag).at(node).allow_empty
            True
        """
        return ItemReduction(self.kernel, OptionValues().at(self.allow_empty, node))


@dataclass(frozen=True)
class OperatorRun:
    """Give the kernel of a binary operator, on the path that polars takes for its inputs.

    Two inputs that are columns of rows take the in-place path of `+`, `-`
    and `*`. An input that arrives as one value makes the operator run the
    kernel of two series (`OperatorKernel`), and so does a run in `agg` over
    lists or group by group (`KernelContext.lists`). Arithmetic on a
    `Decimal` can fail on one value.

    Example:
        >>> from astrolabe._rules.operators import Operator
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> context = KernelContext(grouped=False, scalars=(False, False))
        >>> pair = [Data(DType("Int8")), Data(DType("Decimal", (10, 2)))]
        >>> OperatorRun(Operator.PLUS).output(pair, context)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: ...
    """

    OWN_LENGTHS: ClassVar[frozenset[Operator]] = frozenset({
        Operator.FLOOR_DIVIDE,
        Operator.LOGICAL_AND,
        Operator.LOGICAL_OR,
    })

    operator: Operator

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the left input and of the right input.
            context: How the data reaches the kernel.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> context = KernelContext(grouped=False, scalars=(False, True))
            >>> pair = [Data(DType("Int8")), Data(DType("Decimal", (10, 2)))]
            >>> OperatorRun(Operator.PLUS).output(pair, context)
            Decimal(precision=38, scale=2)
        """
        owned = not any(context.scalars) and not context.lists
        kernel = OperatorKernel(owned=owned)
        return kernel.output(self.operator, inputs[0], inputs[1])

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say whether the operator can fail on one value: arithmetic on a `Decimal`.

        Args:
            inputs: The data of the two inputs.

        Returns:
            Whether it can.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> OperatorRun(Operator.PLUS).fallible([Data(DType("Decimal", (10, 2)))])
            True
        """
        decimal = any(data.dtype.name == "Decimal" for data in inputs)
        return decimal and self.operator.is_arithmetic()

    def checks_lengths(self) -> bool:
        """Say that the run of the node checks the lengths before the kernel.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> OperatorRun(Operator.PLUS).checks_lengths()
            False
        """
        return False

    def checks_group_lengths(self) -> bool:
        """Say whether the kernel checks the lengths of a group itself, as `//` and `and` do.

        Where the operator runs once for each group, polars computes `+`,
        `-`, `*`, `/`, `%`, `&`, `|` and `^` with the arithmetic of a column,
        which checks the lengths of the two sides first. It calls the kernel of
        `//` with no check before it, and it casts both sides of a logical
        `and` and `or` to `Boolean` before the check, so the checks of the
        types come first.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> OperatorRun(Operator.PLUS).checks_group_lengths()
            False
            >>> OperatorRun(Operator.FLOOR_DIVIDE).checks_group_lengths()
            True
        """
        return self.operator in self.OWN_LENGTHS

    def at(self, node: NodeOptions) -> OperatorRun:
        """Give the kernel, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same kernel.

        Example:
            >>> from astrolabe._rules.operators import Operator
            >>> OperatorRun(Operator.PLUS).at(NodeOptions({"BinaryExpr": {}})).operator
            <Operator.PLUS: 'Plus'>
        """
        del node
        return self


@dataclass(frozen=True)
class StepData:
    """Give the data of the one input of a node whose work is a step of the cast pass, as a cast is.

    The work of a cast node is the cast that the cast pass gives its input
    (`CastNode`). That step runs with the cast kernel of polars. So the node
    gives the data of its input after the step.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> context = KernelContext(grouped=False, scalars=(False,))
        >>> StepData().output([Data(DType("Int16"))], context)
        Int16
    """

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the data of the input.

        Args:
            inputs: The data of the input, after its steps.
            context: How the data reaches the node, which this rule does not read.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> context = KernelContext(grouped=True, scalars=(True,))
            >>> StepData().output([Data(DType("Date"))], context)
            Date
        """
        del context
        return inputs[0].dtype

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say that the node does not fail on one value: its step does, where it is a strict cast.

        Args:
            inputs: The data of the input.

        Returns:
            False.

        Example:
            >>> StepData().fallible([])
            False
        """
        del inputs
        return False

    def checks_lengths(self) -> bool:
        """Say that the node checks no lengths: it has one input.

        Returns:
            False.

        Example:
            >>> StepData().checks_lengths()
            False
        """
        return False

    def checks_group_lengths(self) -> bool:
        """Say that the run of the node checks the lengths of a group before the kernel.

        Returns:
            False.

        Example:
            >>> StepData().checks_group_lengths()
            False
        """
        return False

    def at(self, node: NodeOptions) -> StepData:
        """Give the rule, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same rule.

        Example:
            >>> StepData().at(NodeOptions({"Cast": {}}))
            StepData()
        """
        del node
        return self


class PlannedInput:
    """Give the kernel of a function of one input that polars builds from the planned type.

    Polars computes some options of a function when it converts the
    expression, from the planner types of the inputs, and keeps them in the
    function. `reinterpret` keeps the type that its kernel reads the bits of
    each value as. So this kernel is built from the planned type of the node
    (`KernelContext.planned`), and then it dispatches on the data type of its
    input. A literal with no data type has a planner type that is not the
    type of its data, so such a kernel can refuse it.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.layout import LayoutKernels
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> context = KernelContext(grouped=False, scalars=(False,), planned=DType("Int8"))
        >>> PlannedInput(LayoutKernels().reinterpret).output([Data(DType("UInt8"))], context)
        Int8
    """

    def __init__(self, build: Callable[[DType], Dispatch]) -> None:
        """Keep the function of the vocabulary that builds the dispatch from the planned type.

        Args:
            build: The function.

        Example:
            >>> from astrolabe._rules.kernels.layout import LayoutKernels
            >>> PlannedInput(LayoutKernels().reinterpret)
            PlannedInput(reinterpret)
        """
        self._build = build

    def __repr__(self) -> str:
        """Give the name of the function that builds the dispatch.

        Returns:
            The text.

        Example:
            >>> from astrolabe._rules.kernels.layout import LayoutKernels
            >>> repr(PlannedInput(LayoutKernels().reinterpret))
            'PlannedInput(reinterpret)'
        """
        return f"PlannedInput({self._build.__name__})"

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the input.
            context: How the data reaches the kernel, with the planned type of the node.

        Returns:
            The type.

        Raises:
            NoPlannedTypeError: The context gives no planned type.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.layout import LayoutKernels
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> context = KernelContext(grouped=True, scalars=(False,))
            >>> PlannedInput(LayoutKernels().reinterpret).output([Data(DType("Int8"))], context)
            Traceback (most recent call last):
            ...
            astrolabe._rules.definitions.kernel_adapters.NoPlannedTypeError: the kernel ...
        """
        if context.planned is None:  # pragma: no cover -- the evaluation gives the planned type
            raise NoPlannedTypeError(self._build.__name__)
        return self._build(context.planned).output(inputs[0])

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say that the kernel does not fail on one value.

        Args:
            inputs: The data of the input.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.layout import LayoutKernels
            >>> PlannedInput(LayoutKernels().reinterpret).fallible([])
            False
        """
        del inputs
        return False

    def checks_lengths(self) -> bool:
        """Say that the kernel checks no lengths: it has one input.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.layout import LayoutKernels
            >>> PlannedInput(LayoutKernels().reinterpret).checks_lengths()
            False
        """
        return False

    def checks_group_lengths(self) -> bool:
        """Say that the run of the node checks the lengths of a group before the kernel.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.layout import LayoutKernels
            >>> PlannedInput(LayoutKernels().reinterpret).checks_group_lengths()
            False
        """
        return False

    def at(self, node: NodeOptions) -> PlannedInput:
        """Give the kernel, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same kernel.

        Example:
            >>> from astrolabe._rules.kernels.layout import LayoutKernels
            >>> kernel = PlannedInput(LayoutKernels().reinterpret)
            >>> kernel.at(NodeOptions({"Function": {}})) is kernel
            True
        """
        del node
        return self


class PlannedKernel(Protocol):
    """Give the type of a kernel that reads every input and the planned type of its node."""

    def output(
        self,
        inputs: Sequence[Data],
        planned: DType,
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            planned: The type of the node at the planner types of its inputs.
            computed: Each input as the run computes it, or none.
        """


@dataclass(frozen=True)
class PlannedInputs:
    """Give the kernel of a function of several inputs that reads the planned type of its node.

    Polars casts the inputs of `append` to their supertype when it converts the
    expression, at the planner types. So the kernel casts the data of each
    input to the planned type (`KernelContext.planned`). `own_lengths` says
    that the kernel decides the lengths of its inputs itself: `append` takes
    inputs of any lengths. Where it does not, the run of the node checks them
    first.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> from astrolabe._rules.kernels.lengths import Appended
        >>> context = KernelContext(grouped=False, scalars=(False, True), planned=DType("Int16"))
        >>> PlannedInputs(Appended()).output([Data(DType("Int8")), Data(DType("Int32"))], context)
        Int16
    """

    kernel: PlannedKernel
    own_lengths: bool = True

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            context: How the data reaches the kernel, with the planned type.

        Returns:
            The type.

        Raises:
            NoPlannedTypeError: The context has no planned type, which is a bug
                in the code that calls the kernel.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.kernels.lengths import Appended
            >>> context = KernelContext(grouped=False, scalars=(False,))
            >>> PlannedInputs(Appended()).output([Data(DType("Int8"))], context)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.definitions.kernel_adapters.NoPlannedTypeError: the kernel ...
        """
        if context.planned is None:  # pragma: no cover -- the evaluation gives the planned type
            raise NoPlannedTypeError(type(self.kernel).__name__)
        return self.kernel.output(inputs, context.planned, context.computed)

    def fallible(self, inputs: Sequence[Data]) -> bool:
        """Say that the kernel does not fail on one value.

        Args:
            inputs: The data of the inputs.

        Returns:
            False.

        Example:
            >>> from astrolabe._rules.kernels.lengths import Appended
            >>> PlannedInputs(Appended()).fallible([])
            False
        """
        del inputs
        return False

    def checks_lengths(self) -> bool:
        """Say whether the kernel decides the lengths of its inputs itself (`own_lengths`).

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.kernels.lengths import Appended
            >>> PlannedInputs(Appended()).checks_lengths()
            True
        """
        return self.own_lengths

    def checks_group_lengths(self) -> bool:
        """Say whether the kernel decides the lengths of a group itself (`own_lengths`).

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.kernels.lengths import Appended
            >>> PlannedInputs(Appended()).checks_group_lengths()
            True
        """
        return self.own_lengths

    def at(self, node: NodeOptions) -> PlannedInputs:
        """Give the kernel, which reads no option of the node.

        Args:
            node: The serialized root of the node.

        Returns:
            The same kernel.

        Example:
            >>> from astrolabe._rules.kernels.lengths import Appended
            >>> kernel = PlannedInputs(Appended())
            >>> kernel.at(NodeOptions({"Function": {}})) is kernel
            True
        """
        del node
        return self


@dataclass(frozen=True)
class ImplodedRun:
    """Give the data of an `over` or a `rolling`: its function's data, or a list of it.

    A function that gives one value for each group or window gives that
    value, and any other function gives a list where `join` holds. A
    `rolling` also checks its index, the second input (`index`).

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> context = KernelContext(grouped=False, scalars=(False, False))
        >>> ImplodedRun(join=True).output([Data(DType("Int8")), Data(DType("Int8"))], context)
        List(Int8)
    """

    join: bool | OptionValue[bool]
    index: RollingIndex | OptionValue[RollingIndex] | None = None

    def output(self, inputs: Sequence[Data], context: KernelContext) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the check of the index refuses it.

        Args:
            inputs: The data of the function, then of the other inputs.
            context: How the data reaches the node: whether the function gives one value.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> context = KernelContext(grouped=False, scalars=(True,))
            >>> ImplodedRun(join=True).output([Data(DType("Int8"))], context)
            Int8
        """
        join = OptionValues().known(self.join, "whether the node puts a group in a list")
        index = OptionValues().known(self.index, "the index of the window")
        if index is not None:
            index.check(inputs[1], grouped=context.grouped)
        scalar = bool(context.scalars) and context.scalars[0]
        dtype = inputs[0].dtype
        return DType("List", (dtype,)) if join and not scalar else dtype

    def fallible(self, inputs: Sequence[Data]) -> bool:  # pragma: no cover -- no window reads it
        """Say that the node does not fail on one value.

        Args:
            inputs: The data of the inputs.

        Returns:
            False.

        Example:
            >>> ImplodedRun(join=False).fallible([])
            False
        """
        del inputs
        return False

    def checks_lengths(self) -> bool:  # pragma: no cover -- no window reads it
        """Say that the run of the node checks the lengths of its inputs.

        Returns:
            False.

        Example:
            >>> ImplodedRun(join=False).checks_lengths()
            False
        """
        return False

    def checks_group_lengths(self) -> bool:  # pragma: no cover -- no window reads it
        """Say that the run of the node checks the lengths of a group.

        Returns:
            False.

        Example:
            >>> ImplodedRun(join=False).checks_group_lengths()
            False
        """
        return False

    def at(self, node: NodeOptions) -> ImplodedRun:
        """Give the rule, with the option that it reads at one node.

        Args:
            node: The serialized root of the node.

        Returns:
            The rule.

        Example:
            >>> ImplodedRun(join=True).at(NodeOptions({"Rolling": {}})).join
            True
        """
        return ImplodedRun(OptionValues().at(self.join, node), OptionValues().at(self.index, node))
