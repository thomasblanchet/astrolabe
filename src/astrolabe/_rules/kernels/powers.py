"""The kernels of `pow` and `arctan2`, which read a number from each of two inputs.

Both kernels check the lengths of their inputs first: two inputs of more
than one row must have one length. `pow` needs a number on each side, and
reads the exponent as the base allows: a float exponent turns an integer base
into that float, an integer exponent must be a `UInt32`, and a float base
reads the exponent as its own type. `arctan2` reads both sides as the float
type of `y`, `Float64` where `y` is not a float.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.casts import SeriesCast
from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.dispatch import RunCast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.kernels.dispatch import Data
    from astrolabe._rules.type_functions import Operand


@dataclass(frozen=True)
class PairLengths:
    """Check that two inputs of more than one row have one length, with the error of a kernel.

    Example:
        >>> PairLengths("pow").check((3, 2))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: pow: the shapes ...
    """

    name: str

    def check(self, lengths: Sequence[int]) -> None:
        """Refuse two lengths of more than one row that differ.

        Args:
            lengths: The rows of the two inputs, or none where they are not known.

        Raises:
            PolarsRefusal: The lengths differ.

        Example:
            >>> PairLengths("arctan2").check((3, 1))
        """
        if len(lengths) < 2 or 1 in lengths[:2] or lengths[0] == lengths[1]:
            return
        message = f"{self.name}: the shapes {lengths[0]} and {lengths[1]} do not match"
        raise PolarsRefusal(ErrorClass.COMPUTE, message)


@dataclass(frozen=True)
class Power:
    """Give the type of `pow`, from the type of the base and the type of the exponent.

    Example:
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> Power().output([Data(DType("Int8")), Data(DType("Float32"))], (3, 3), ())
        Float32
    """

    INDEX: ClassVar[DType] = DType("UInt32")

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the base and of the exponent.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The lengths differ, a side is not a number, or the
                exponent does not cast.

        Example:
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.type_functions import Operand
            >>> exponent = Operand.of_value(-1)
            >>> Power().output([Data(DType("Int8")), Data(DType("Int32"))], (), ("", exponent))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        PairLengths("pow").check(lengths)
        base, exponent = inputs[0].dtype, inputs[1].dtype
        for side, dtype in (("base", base), ("exponent", exponent)):
            if not Is.primitive_numeric(dtype):
                message = f"`pow` operation not supported for dtype `{dtype!r}` as {side}"
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if Is.integer(base) and Is.float(exponent):
            return exponent
        target = self.INDEX if Is.integer(base) else base
        literal = computed[1] if len(computed) > 1 and computed[1].literal else None
        if literal is not None:
            cast = SeriesCast(literal.dtype.materialized(), target, strict=True)
            cast.output([literal.value])
        return base


@dataclass(frozen=True)
class ArcTangent:
    """Give the type of `arctan2`: the float type of `y`, or `Float64`.

    Example:
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> ArcTangent().output([Data(DType("Int8")), Data(DType("Float32"))], (3, 3))
        Float64
    """

    FLOAT64: ClassVar[DType] = DType("Float64")

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of `y` and of `x`.
            lengths: The rows of each input, or none where they are not known.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The lengths differ, or a cast fails. A `y` that stays
                apart from a float after its cast, such as a struct, makes the
                kernel call itself without end. An `x` that stays apart from
                the float type after its cast panics.

        Example:
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> ArcTangent().output([Data(DType("Float32")), Data(struct)], ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        PairLengths("arctan2").check(lengths)
        y = inputs[0]
        if not Is.float(y.dtype):
            y = RunCast().of(y, self.FLOAT64)
        target = y.dtype
        if not Is.float(target):
            message = f"arctan2 casts {target!r} to a float and calls itself without end"
            raise PolarsRefusal(Stop.CRASHES, message)
        x = RunCast().of(inputs[1], target)
        if not x.dtype.same(target):
            message = f"arctan2 reads {x.dtype!r} as {target!r}"
            raise PolarsRefusal(Stop.PANICS, message)
        return target
