"""The refusals of polars, and the answers that the rules cannot give.

Polars refuses a query in four ways. It raises an error that Python can catch,
it panics, it ends the process, or it never returns. A rule of this package
that finds a refusal raises `PolarsRefusal`. The refusal holds the way that
polars stops, the Python class of the error where polars raises, and its
message. Some functions of polars give no answer in place of an error, for
example the join of two types that have no supertype. A rule that mirrors such
a function gives `None` instead. A value that one cast loses is not a refusal
of the query, so the outcome of one value stays an outcome.

A rule can also meet a question that it cannot answer: a fact of the state of
the process, or a part of polars that the package does not translate. It then
raises `UndecidedError`. Code that types a query maps a refusal to an error for
the user, and an undecided answer to `Unchecked`, the data type of a column
whose data type the checker does not know.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from astrolabe._rules.releases import polars_releases

if TYPE_CHECKING:
    from collections.abc import Sequence


class Stop(enum.Enum):
    """Name the way that polars stops on a query that it refuses.

    `RAISES` is an error of polars that Python catches as an exception.
    `PANICS` is a panic of Rust: the Python binding raises it as a
    `PanicException`, which derives from `BaseException` and not from
    `Exception`. `CRASHES` ends the process, for example with a stack
    overflow. `HANGS` never returns.

    Example:
        >>> Stop.PANICS.value
        'panics'
    """

    RAISES = "raises"
    PANICS = "panics"
    CRASHES = "crashes"
    HANGS = "hangs"


class ErrorClass(enum.Enum):
    """Name the Python class of an error that polars raises.

    The Python binding of polars maps each kind of error to one Python class,
    for example a mismatch of two schemas to `SchemaError`. The value is the
    name of that class.

    Example:
        >>> ErrorClass("SchemaError")
        <ErrorClass.SCHEMA: 'SchemaError'>
    """

    COLUMN_NOT_FOUND = "ColumnNotFoundError"
    COMPUTE = "ComputeError"
    DUPLICATE = "DuplicateError"
    INVALID_OPERATION = "InvalidOperationError"
    OUT_OF_BOUNDS = "OutOfBoundsError"
    SCHEMA = "SchemaError"
    SHAPE = "ShapeError"
    STRUCT_FIELD_NOT_FOUND = "StructFieldNotFoundError"


class PolarsRefusal(Exception):
    """Report that polars refuses a query, as one rule of this package finds it.

    Every rule raises this exception where polars refuses: a type function that
    rejects its inputs, a kernel that fails on its data, a cast pass that loops
    with no end. The first argument says how polars stops: the Python class of
    the error where polars raises, and the way it stops otherwise. So an error
    class goes with a raise, and only with a raise. `message` says what polars
    checks. The text of the refusal names the class of the error where polars
    raises, and the way that it stops otherwise.

    Example:
        >>> refusal = PolarsRefusal(ErrorClass.SCHEMA, "no join")
        >>> refusal.stop, refusal.error
        (<Stop.RAISES: 'raises'>, <ErrorClass.SCHEMA: 'SchemaError'>)
        >>> str(PolarsRefusal(Stop.PANICS, "no supertype of Int8 and Date"))
        'panics: no supertype of Int8 and Date'
    """

    def __init__(self, how: ErrorClass | Stop, message: str) -> None:
        """Keep the facts of the refusal and build the text.

        Args:
            how: The class of the error where polars raises, or the way that it
                stops where it does not raise.
            message: What polars checks.

        Raises:
            ValueError: `how` is `Stop.RAISES`, which needs its error class.

        Example:
            >>> str(PolarsRefusal(ErrorClass.SHAPE, "lengths"))
            'ShapeError: lengths'
        """
        if how is Stop.RAISES:
            text = "a refusal that raises needs the class of its error"
            raise ValueError(text)
        self.stop = Stop.RAISES if isinstance(how, ErrorClass) else how
        self.error = how if isinstance(how, ErrorClass) else None
        self.message = message
        kind = self.stop.value if self.error is None else self.error.value
        super().__init__(f"{kind}: {message}")


class UnequalLengths:
    """Give the refusal of polars for inputs whose lengths do not broadcast.

    Two lengths broadcast where they are equal, or where one of them is one
    row. From polars 1.44.0, a binary operation on two columns and many
    kernels check the lengths of their inputs with one shared check, which
    raises `ShapeError`. Before that release, each of them raised its own
    `InvalidOperationError`, so the code that checks the lengths gives the
    message of the older releases.

    Example:
        >>> from astrolabe._rules.releases import PolarsRelease
        >>> with PolarsRelease.pretend("1.44.0"):
        ...     print(UnequalLengths().refusal([3, 1, 2], "older"))
        ShapeError: can't compute broadcast length, found incompatible lengths 2 and 3
    """

    def check(self, lengths: Sequence[int], older: str) -> None:
        """Refuse the lengths of the inputs where two of them do not broadcast.

        It raises the `PolarsRefusal` of `refusal` where two inputs have other
        lengths, and neither has one row.

        Args:
            lengths: The number of rows of each input, or none where they are not known.
            older: The message of the releases before the shared check.

        Example:
            >>> UnequalLengths().check([5, 1, 5], "older")
        """
        if len({length for length in lengths if length != 1}) > 1:
            raise self.refusal(lengths, older)

    @polars_releases
    def refusal(self, lengths: Sequence[int], older: str) -> PolarsRefusal:
        """Give the refusal of the shared check of the lengths.

        Args:
            lengths: The number of rows of each input, two of which do not broadcast.
            older: The message of the releases before the shared check.

        Returns:
            The refusal.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.44.0"):
            ...     UnequalLengths().refusal([5, 2], "older").error
            <ErrorClass.SHAPE: 'ShapeError'>
        """
        del older
        long = [length for length in lengths if length != 1]
        other = next(length for length in long if length != long[0])
        message = (
            f"can't compute broadcast length, found incompatible lengths {other} and {long[0]}"
        )
        return PolarsRefusal(ErrorClass.SHAPE, message)

    @refusal.before("1.44.0")
    def _own_refusal(self, lengths: Sequence[int], older: str) -> PolarsRefusal:
        """Give the refusal of the check of one operation, as the older releases raise it.

        Args:
            lengths: The number of rows of each input.
            older: The message of the check.

        Returns:
            The refusal.

        Example:
            >>> print(UnequalLengths()._own_refusal([5, 2], "different lengths"))
            InvalidOperationError: different lengths
        """
        del lengths
        return PolarsRefusal(ErrorClass.INVALID_OPERATION, older)


class Undecided(enum.Enum):
    """Name why the rules cannot answer a question.

    `PROCESS` is a fact of the state of the process, for example the
    categories that the process holds. `NOT_TRANSLATED` is a part of polars
    that this package does not translate. `NOT_SERIALIZED` is a value that
    the JSON form of the tree of polars does not hold, for example an
    infinite float, which polars writes as null. `VALUES` is a fact of the
    values of a column, which the code does not fix, for example a quantile
    that a column holds. `ROWS` is a fact of the number of rows of the frame,
    which the code does not fix either.

    Example:
        >>> Undecided.PROCESS.value
        'process'
    """

    PROCESS = "process"
    NOT_TRANSLATED = "not translated"
    NOT_SERIALIZED = "not serialized"
    VALUES = "values"
    ROWS = "rows"


class UndecidedError(Exception):
    """Report a question that the rules cannot answer, so that later code can give `Unchecked`.

    It is not a refusal of polars: polars can accept the query or refuse it,
    and the rules do not know which. `cause` says why the rules do not know,
    and `question` says what they do not know.

    Example:
        >>> str(UndecidedError(Undecided.NOT_TRANSLATED, "the run of Filter"))
        'not translated: the run of Filter'
    """

    def __init__(self, cause: Undecided, question: str) -> None:
        """Keep the cause and build the text.

        Args:
            cause: Why the rules cannot answer.
            question: What the rules cannot answer.

        Example:
            >>> UndecidedError(Undecided.PROCESS, "a cast").cause
            <Undecided.PROCESS: 'process'>
        """
        super().__init__(f"{cause.value}: {question}")
        self.cause = cause
        self.question = question
