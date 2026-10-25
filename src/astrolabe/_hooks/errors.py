"""Every error that the checker reports, with one class for each kind of error.

mypy prints each error with a code in brackets, for example
`[unknown-column]`. A user can turn a code off with
`--disable-error-code`, or silence one line with
`# type: ignore[unknown-column]`. The code is the stable name of a kind
of check, so it must be the same wherever the plugin reports that kind.

Each class here holds one `CODE` and builds its message from plain
values, for example a column name. A hook builds the error and calls
`fail(ctx)` at the place of the fault. It never writes the message
text itself. So the wording of each error lives in one place, and the
tests can compare the text and the code of each message.

Some errors are unproven results, not faults. The operation gave up on a
proof, and the `allow_unproven` setting decides whether the checker
reports it. Those classes derive from `UnprovenError`, and code reports
them through a gate that reads the setting. See `UnprovenGate`.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, TYPE_CHECKING, cast

from mypy.errorcodes import ErrorCode

from astrolabe._hooks.unproven import Reason

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.nodes import Context


class _FailApi(Protocol):
    def fail(self, msg: str, ctx: Context, /, *, code: ErrorCode | None = None) -> object: ...


class _NoteApi(Protocol):
    def note(self, msg: str, ctx: Context, /, *, code: ErrorCode | None = None) -> object: ...


class _FailContext(Protocol):
    @property
    def context(self) -> Context: ...
    @property
    def api(self) -> _FailApi: ...


class AstrolabeError:
    """The base of every error that the checker reports.

    A subclass sets `CODE` and builds its message in `__init__` or in a
    factory class method. It then gives the message to this class, which
    reports it through mypy with the code.
    """

    CODE: ClassVar[ErrorCode]

    def __init__(self, message: str) -> None:
        """Keep the finished message.

        Args:
            message: The text that `fail` reports.
        """
        self._message = message

    def fail(self, ctx: _FailContext) -> None:
        """Report the error through mypy, at the place of the call.

        Args:
            ctx: The mypy context to report at. It can be a `MethodContext`, a
                `FunctionContext`, a `FunctionSigContext`, an
                `AnalyzeTypeContext`, or any other object with `.context` and
                `.api`. These mypy classes share no base class, so the parameter
                is a protocol.
        """
        ctx.api.fail(self._message, ctx.context, code=self.CODE)

    def fail_with_note(self, ctx: _FailContext, note: str) -> None:
        """Report the error, then one note under it.

        mypy reports a note through the same object as an error, but its plugin
        interface does not declare the `note` method. The method therefore reads
        that object as one that has `note`. The note carries the code of this
        error, so the same filters apply to both.

        Args:
            ctx: The mypy context to report at.
            note: The text of the note.
        """
        self.fail(ctx)
        cast("_NoteApi", ctx.api).note(note, ctx.context, code=self.CODE)


class UnprovenError(AstrolabeError):
    """An error that the `allow_unproven` setting decides whether to report.

    The operation gave up on a proof for one reason, for example a column
    name that is not a literal. Each reason has a level. The setting
    compares the level of the reason to the level that the user allows for
    the operation. If the setting does not let it pass, the gate reports the
    error with a note that names the setting which lets it pass. See
    `UnprovenPolicy`.

    Attributes:
        reason: Why the operation gave up.
        operation: The name of the operation, which is also its key in the
            overrides of the setting.

    Example:
        >>> from astrolabe._hooks.unproven import Reason
        >>> ImplicitlyUntypedError(Reason.NO_SCHEMA, "read_csv").operation
        'read_csv'
    """

    def __init__(self, message: str, reason: Reason, operation: str) -> None:
        """Keep the message, the reason and the operation.

        Args:
            message: The finished message.
            reason: Why the operation gave up.
            operation: The name of the operation.
        """
        super().__init__(message)
        self.reason = reason
        self.operation = operation


class UnknownColumnError(AstrolabeError):
    """A name that the code uses is surely not a column of the frame.

    The checker reports this only when it proves that the frame cannot hold
    the name. The frame is closed, or its remainder, the columns that its
    type does not name, cannot hold the name. If the remainder can hold it,
    the error is `UndeclaredColumnError` instead.
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "unknown-column",
        "Check that a referenced column is declared",
        "astrolabe",
    )

    def __init__(self, name: str, *, frame: str | None = None) -> None:
        """Build the message for an unknown column.

        Args:
            name: The name of the unknown column.
            frame: For an operation on two frames, such as `.join()`, the side
                that lacks the column, for example `"left"`. Leave it out for an
                operation on one frame.
        """
        location = f" in {frame} frame" if frame is not None else ""
        super().__init__(f'Unknown column "{name}"{location}')


class BlankColumnError(AstrolabeError):
    """A blank column reached a frame before a call gave it a name.

    A declaration can read a column whose name is left open, written with
    `some_column`. Each call of the function then binds the name. Inside
    the body of the function, the name is still open, so the body cannot
    read that column from a frame. The error shares the code
    `unknown-column`, because the frame has no column with that name.

    Example:
        >>> BlankColumnError().CODE.code
        'unknown-column'
    """

    CODE: ClassVar[ErrorCode] = UnknownColumnError.CODE

    def __init__(self) -> None:
        """Build the message."""
        super().__init__("A blank read (some_column) has no name until a call binds it")


class UndeclaredColumnError(UnprovenError):
    """The checker cannot prove that a column exists.

    The frame does not declare the name, and it has a remainder that can
    hold the name or not, depending on the caller. The operation then fails
    at run time for at least one caller. `UnknownColumnError` is for a name
    that the checker proves is missing. The `allow_unproven` setting
    decides whether the checker reports this error.

    Example:
        >>> UndeclaredColumnError("price", "select").reason.level.spelling
        'possible'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "undeclared-column",
        "Check for columns that cannot be proven to exist",
        "astrolabe",
    )

    def __init__(self, name: str, operation: str) -> None:
        """Build the message for a column that the checker cannot prove.

        Args:
            name: The name of the column.
            operation: The name of the operation.
        """
        super().__init__(
            f'Column "{name}" is not declared and cannot be proven to exist',
            Reason.REMAINDER_MAY_HOLD,
            operation,
        )


class UnnamedReadError(UnprovenError):
    """A declared column reads a column whose name is not one literal.

    A parameter such as `c: TypedColumn[str, pl.Int64]` gives its data type,
    but its name is a value that no code knows. So the checker cannot prove
    that the frame holds the column that it reads, and the operation can fail
    at run time. A name that is a union of literals reads one of its names,
    and the checker reports it where the frame does not declare each name.
    The `allow_unproven` setting decides whether the checker reports this
    error, as for any column name that is not a literal.

    Example:
        >>> UnnamedReadError("select").reason.level.spelling
        'stated'
    """

    CODE: ClassVar[ErrorCode] = UndeclaredColumnError.CODE

    def __init__(self, operation: str) -> None:
        """Build the message for the read of a column whose name the code does not know.

        Args:
            operation: The name of the operation.
        """
        super().__init__(
            "A declared column whose name is not one literal reads a column that cannot be "
            "proven to exist",
            Reason.DYNAMIC_NAME,
            operation,
        )


class DuplicateColumnError(AstrolabeError):
    """An operation gives two columns with the same name.

    Polars refuses such a result at run time. One code covers every
    operation that can make a duplicate.

    Example:
        >>> DuplicateColumnError("select", "price").CODE.code
        'duplicate-column'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "duplicate-column",
        "Check that an operation doesn't produce duplicate columns",
        "astrolabe",
    )

    def __init__(self, operation: str, name: str) -> None:
        """Build the message for a repeated column name.

        Args:
            operation: The operation that makes the duplicate.
            name: The repeated name.
        """
        super().__init__(f'{operation} would produce duplicate column "{name}"')


class WrongColumnShapeError(AstrolabeError):
    """A column, or a typed series, is not in the family of data types that an operation needs.

    For example, `.explode()` needs a `List` or an `Array`, `.unnest()`
    needs a `Struct`, `.round()` needs a number, and `sum()` needs a number
    or a `Boolean`. The message names the family that the operation
    accepts. There is one factory for a frame column, which has a name, and
    one for a typed series, which has none.

    Example:
        >>> WrongColumnShapeError.on_column("x", expected="numeric", operation=".round()").CODE.code
        'wrong-column-shape'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "wrong-column-shape",
        "Check a column's dtype family for the operation it is used in",
        "astrolabe",
    )

    @classmethod
    def on_column(cls, name: str, *, expected: str, operation: str) -> WrongColumnShapeError:
        """Build the message for a frame column with the wrong family of data type.

        Args:
            name: The name of the column.
            expected: The family that the operation needs, in words.
            operation: The operation that needs the family.

        Returns:
            The error.

        Example: see class docstring.
        """
        return cls(f'Column "{name}" is not {expected}, cannot {operation}')

    @classmethod
    def on_series(cls, *, expected: str, operation: str) -> WrongColumnShapeError:
        """Build the message for a typed series with the wrong family of data type.

        Args:
            expected: The family that the operation needs, in words.
            operation: The operation that needs the family.

        Returns:
            The error.

        Example:
            >>> WrongColumnShapeError.on_series(expected="numeric", operation=".round()").CODE.code
            'wrong-column-shape'
        """
        return cls(f"This Series is not {expected}, cannot {operation}")


class UnsupportedDtypeOperationError(AstrolabeError):
    """An operator or an expression method has no result for the data types of its operands.

    Some pairs of data types have no result under an operator, for example
    `Boolean - Boolean` or `String + Int64`. Some data types have no result
    under a unary operator, for example `-` on a `UInt8`. Polars raises at
    run time in each case. This differs from `WrongColumnShapeError`, which
    checks the family of one column.
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "unsupported-dtype-op",
        "Check that an operator supports its own operand dtypes",
        "astrolabe",
    )

    @classmethod
    def at_frame(cls, name: str, refusal: str = "") -> UnsupportedDtypeOperationError:
        """Build the message for an expression that has no result at the column types of a frame.

        An expression can have a result at some data types and not at others.
        This error applies when the frame gives the columns that it reads data
        types with no result.

        Args:
            name: The output name of the expression.
            refusal: Why polars refuses, and how: the reason in the words of
                the checker, the way that polars stops and the class of its
                error. Empty where it is not known.

        Returns:
            The error.

        Example:
            >>> UnsupportedDtypeOperationError.at_frame("total").CODE.code
            'unsupported-dtype-op'
            >>> refusal = "the lengths differ, and polars raises ShapeError"
            >>> error = UnsupportedDtypeOperationError.at_frame("t", refusal)
            >>> error._message.endswith("types: the lengths differ, and polars raises ShapeError")
            True
        """
        text = f'"{name}" has no result type at this frame\'s column types'
        if refusal:
            return cls(f"{text}: {refusal}")
        return cls(text)

    @classmethod
    def unnamed_at_frame(cls, refusal: str) -> UnsupportedDtypeOperationError:
        """Build the message for an expression with no result and no output name at a frame.

        A node can take its output name from its input, as a field of a
        struct by its place takes the name of that field. Where the node has
        no result, it has no name either, so the message names the expression.

        Args:
            refusal: Why polars refuses, and how.

        Returns:
            The error.

        Example:
            >>> refusal = "no field at 5, and polars raises ColumnNotFoundError"
            >>> UnsupportedDtypeOperationError.unnamed_at_frame(refusal)._message[:42]
            'This expression has no result type at this'
        """
        return cls(f"This expression has no result type at this frame's column types: {refusal}")

    @classmethod
    def over_remainder(cls) -> UnsupportedDtypeOperationError:
        """Build the message for a selection that has no result at a data type of the remainder.

        A selection picks columns by a rule, for example `cs.numeric()`, so it
        can pick columns of the remainder, the columns that the type of the
        frame does not name. If one data type that the remainder can hold has
        no result, the operation can fail for some caller.

        Returns:
            The error.

        Example:
            >>> UnsupportedDtypeOperationError.over_remainder().CODE.code
            'unsupported-dtype-op'
        """
        return cls("The selection has no result type at a type the remainder can hold")


class UncastableDtypeError(AstrolabeError):
    """A `.cast(target)` call can never succeed from the data type of its source.

    A cast is not one operation that needs one family of data types. It is
    a pair of a source and a target, and `CastCompatibility` decides which
    pairs polars accepts. There is one factory for each place where the
    fault can occur: a typed series, which has no column name
    (`on_series`), and an expression with no source column, such as
    `pl.lit(...)` or a horizontal fold (`on_expression`).
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "invalid-cast",
        "Check that a cast is possible",
        "astrolabe",
    )

    @classmethod
    def on_series(cls, *, source: str, target: str) -> UncastableDtypeError:
        """Build the message for a typed series that cannot cast to the target.

        Args:
            source: The data type of the series, in words.
            target: The target data type, in words.

        Returns:
            The error.

        Example:
            >>> UncastableDtypeError.on_series(source="String", target="Boolean").CODE.code
            'invalid-cast'
        """
        return cls(f"This Series cannot cast from {source} to {target}")

    @classmethod
    def on_expression(cls, *, source: str, target: str) -> UncastableDtypeError:
        """Build the message for an expression with no source column that cannot cast to the target.

        This applies when the source of the cast has no column name, for
        example `pl.lit(...)` or a horizontal fold.

        Args:
            source: The data type of the expression, in words.
            target: The target data type, in words.

        Returns:
            The error.

        Example:
            >>> UncastableDtypeError.on_expression(source="String", target="Boolean").CODE.code
            'invalid-cast'
        """
        return cls(f"This expression cannot cast from {source} to {target}")


class JoinRenameAmbiguityError(UnprovenError):
    """A join renames a column for some callers and not for others.

    A join adds a suffix to a right column whose name is also a left
    column. If the remainder of one side can hold a declared name of the
    other side, that rename depends on the caller, so the checker cannot
    name the columns of the result. The message names the columns at risk
    and the declarations that fix the fault.

    Example:
        >>> JoinRenameAmbiguityError(["sector"], "join").reason.level.spelling
        'possible'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "join-rename",
        "Check that a join renames the same columns for every caller",
        "astrolabe",
    )

    def __init__(self, at_risk: Sequence[str], operation: str) -> None:
        """Build the message that names the columns at risk.

        Args:
            at_risk: The names in the result whose presence depends on the
                caller.
            operation: The name of the join operation.
        """
        names = ", ".join(f'"{name}"' for name in at_risk)
        super().__init__(
            f"This join renames columns for some callers but not others: "
            f"the remainder can hold {names}. Exclude the names with "
            f"without=/without_prefix=, or make the overlap total with within=",
            Reason.CALLER_DEPENDENT,
            operation,
        )


class JoinRemainderOverlapError(UnprovenError):
    """A join whose two remainders can hold the same name.

    If both remainders can hold one name, a caller can pass two frames that
    both hold it. The join then renames the right column, and the result
    differs from caller to caller. Declaring `disjoint_from=` on one
    remainder removes the fault.

    Example:
        >>> JoinRemainderOverlapError("join_asof").operation
        'join_asof'
    """

    CODE: ClassVar[ErrorCode] = JoinRenameAmbiguityError.CODE

    def __init__(self, operation: str) -> None:
        """Build the message for two remainders that can share a name.

        Args:
            operation: The name of the join operation.
        """
        super().__init__(
            "This join renames columns for some callers but not others: the "
            "two remainders can share a name. Declare disjoint_from= on one "
            "of them",
            Reason.CALLER_DEPENDENT,
            operation,
        )


class JoinSuffixTakenError(UnprovenError):
    """A join where the left frame can already hold a name with the suffix of the join.

    When every name of the right remainder is also a left name, the join
    renames every right column with the suffix. If the left frame can
    already hold a name that ends with the suffix, a renamed column can
    collide with it, and polars raises. Declaring `without_suffix=` on the
    left remainder removes the fault.

    Example:
        >>> JoinSuffixTakenError("_right", "join").reason.level.spelling
        'possible'
    """

    CODE: ClassVar[ErrorCode] = JoinRenameAmbiguityError.CODE

    def __init__(self, suffix: str, operation: str) -> None:
        """Build the message for a suffix that the left frame can already hold.

        Args:
            suffix: The suffix of the join.
            operation: The name of the join operation.
        """
        super().__init__(
            f'This join renames every right column with "{suffix}", but the '
            f"left frame can already hold a name ending in it, and real "
            f"polars raises rather than suffixing twice. Declare "
            f'without_suffix="{suffix}" on the left remainder',
            Reason.CALLER_DEPENDENT,
            operation,
        )


class MissingRequiredColumnError(AstrolabeError):
    """A frame argument lacks a column that its declared parameter needs."""

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "missing-column",
        "Check that a required column is present",
        "astrolabe",
    )

    def __init__(self, name: str) -> None:
        """Build the message for a missing column.

        Args:
            name: The name of the missing column.
        """
        super().__init__(f'Missing required column "{name}"')


class WrongDtypeError(AstrolabeError):
    """A frame argument has a column that its parameter declares, with a different data type."""

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "wrong-dtype",
        "Check a required column's dtype",
        "astrolabe",
    )

    def __init__(self, name: str) -> None:
        """Build the message for a column with the wrong data type.

        Args:
            name: The name of the column.
        """
        super().__init__(f'Column "{name}" has the wrong dtype')


class ForbiddenColumnError(AstrolabeError):
    """A frame argument holds a column that the declared remainder of its parameter excludes.

    A declaration such as `other_columns(without=["ticker"])` promises that
    the remainder never holds `ticker`. This error reports a caller whose
    frame breaks that promise.
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "forbidden-column",
        "Check a call-site remainder against its declaration",
        "astrolabe",
    )

    def __init__(self, remainder: str, name: str) -> None:
        """Build the message for a column that the remainder must not hold.

        Args:
            remainder: The short name of the declared remainder.
            name: The name of the column that breaks the promise.
        """
        super().__init__(f'Remainder "{remainder}" must not contain column "{name}"')


class UnplacedColumnError(AstrolabeError):
    """A frame argument holds a column that no part of a split accepts.

    A split declares a remainder as an ordered list of parts, and each
    column goes to the first part that accepts it. If no part accepts a
    column of the caller, the call has nowhere to put it, so the checker
    refuses the call. The message names the column.
    """

    CODE: ClassVar[ErrorCode] = ForbiddenColumnError.CODE

    def __init__(self, name: str) -> None:
        """Build the message for a column with no part.

        Args:
            name: The column that no part accepts.
        """
        super().__init__(f'No part of the split accepts column "{name}"')


class UnplacedSplitError(AstrolabeError):
    """A split cannot place every column that a caller with an open frame can hold.

    The checker does not know the columns of such a caller, so the message
    cannot name one. It names the fix: the last part must accept every
    column that the earlier parts leave.
    """

    CODE: ClassVar[ErrorCode] = ForbiddenColumnError.CODE

    def __init__(self) -> None:
        """Build the message for a split with a gap."""
        super().__init__(
            "The split leaves a column in no part, and this argument's "
            "remainder can hold one. Give the last part other_columns()"
        )


class ForbiddenDtypeError(AstrolabeError):
    """A frame argument holds a column of a data type that its declared remainder excludes."""

    CODE: ClassVar[ErrorCode] = ForbiddenColumnError.CODE

    def __init__(self, remainder: str, kind: str) -> None:
        """Build the message for a data type that the remainder must not hold.

        Args:
            remainder: The short name of the declared remainder.
            kind: The name of the data type, without its parameters.
        """
        super().__init__(f'Remainder "{remainder}" must not contain a column of type "{kind}"')


class UnprovenRequirementError(UnprovenError):
    """The remainder of a frame argument does not prove a promise that the parameter declares.

    The argument is open, so the checker knows only what its own remainder
    declares. If that declaration does not imply the promise of the
    parameter, the call can break it for some caller. If the setting lets
    the call pass, the call is checked as declared, and the declared type
    is a claim that the checker did not prove.

    Example:
        >>> error = UnprovenRequirementError("NoSector")
        >>> error.operation, error.reason.level.spelling
        ('fit', 'asserted')
    """

    CODE: ClassVar[ErrorCode] = ForbiddenColumnError.CODE

    def __init__(self, remainder: str) -> None:
        """Build the message for an unproven promise.

        Args:
            remainder: The short name of the declared remainder of the
                parameter.
        """
        super().__init__(
            f"The argument's own remainder does not prove the requirements "
            f'of "{remainder}". Declare the missing requirement on the '
            f"argument's own remainder",
            Reason.UNPROVEN_REQUIREMENT,
            "fit",
        )


class ConcatHiddenCollisionError(UnprovenError):
    """A concat whose shared column names depend on the caller.

    A diagonal concat lines up the columns of each frame by name. If the
    remainder of one frame can hold a name that another frame declares, the
    set of shared names depends on the caller, and the checker cannot name
    the columns of the result.

    Example:
        >>> ConcatHiddenCollisionError(["bar"]).operation
        'concat'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "concat-collision",
        "Check that a concat combines the same columns for every caller",
        "astrolabe",
    )

    def __init__(self, at_risk: Sequence[str], operation: str = "concat") -> None:
        """Build the message that names the columns at risk.

        Args:
            at_risk: The declared names that the remainder of another frame can
                also hold.
            operation: The operation that puts the frames side by side, for
                example `concat` or `hstack`.
        """
        names = ", ".join(f'"{name}"' for name in at_risk)
        super().__init__(
            f"This {operation} combines columns differently for some callers: "
            f"another frame's remainder can hold {names}. Exclude the names "
            f"with without=/without_prefix=",
            Reason.CALLER_DEPENDENT,
            operation,
        )


class ConcatRemainderOverlapError(UnprovenError):
    """A concat where the remainders of two frames can hold the same name.

    Example:
        >>> ConcatRemainderOverlapError().reason.level.spelling
        'possible'
    """

    CODE: ClassVar[ErrorCode] = ConcatHiddenCollisionError.CODE

    def __init__(self, operation: str = "concat") -> None:
        """Build the message for two remainders that can share a name.

        Args:
            operation: The operation that puts the frames side by side, for
                example `concat` or `hstack`.
        """
        super().__init__(
            f"This {operation} combines columns differently for some callers: two "
            "remainders can share a name. Declare disjoint_from= on one of "
            "them",
            Reason.CALLER_DEPENDENT,
            operation,
        )


class JoinKeyDtypeMismatchError(AstrolabeError):
    """The two sides of a join key have data types that polars does not join.

    Example:
        >>> JoinKeyDtypeMismatchError("id", left="Int64", right="String").CODE.code
        'join-key-dtype'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "join-key-dtype",
        "Check that a join key's dtypes reconcile on both sides",
        "astrolabe",
    )

    def __init__(self, name: str, *, left: str, right: str) -> None:
        """Build the message for a join key whose data types do not agree.

        Args:
            name: The name of the join key.
            left: The data type on the left side, in words.
            right: The data type on the right side, in words.
        """
        super().__init__(f'Join key "{name}" is {left} on the left and {right} on the right')


class SchemaMismatchError(AstrolabeError):
    """The schemas of two or more frames do not agree for a stack or a concat.

    There is one factory for each kind of call: a whole frame that differs
    in `pl.concat(how="vertical")` or `how="vertical_relaxed"`
    (`concat_frame`), one shared column whose data types do not agree in a
    diagonal concat (`concat_column`), and a schema that differs in
    `.vstack()`, `.extend()` or `.merge_sorted()` (`stacked`).

    Example:
        >>> SchemaMismatchError.stacked(".vstack()").CODE.code
        'schema-mismatch'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "schema-mismatch",
        "Check that frames share a schema for stacking or concatenation",
        "astrolabe",
    )

    @classmethod
    def concat_frame(cls, how: str, index: int) -> SchemaMismatchError:
        """Build the message for a frame whose schema differs in a vertical `pl.concat()`.

        Args:
            how: `"vertical"` or `"vertical_relaxed"`.
            index: The position of the first frame whose schema differs from
                the schema of the first frame.

        Returns:
            The error.

        Example:
            >>> SchemaMismatchError.concat_frame("vertical", 2).CODE.code
            'schema-mismatch'
        """
        return cls(
            f'pl.concat(how="{how}") requires identical schemas: '
            f"frame {index} does not match frame 0",
        )

    @classmethod
    def concat_column(cls, how: str, name: str, left: str, right: str) -> SchemaMismatchError:
        """Build the message for a shared column whose data types do not agree in a diagonal concat.

        Args:
            how: `"diagonal"` or `"diagonal_relaxed"`.
            name: The name of the shared column.
            left: The data type of the column in the first frame.
            right: The data type of the column in a later frame.

        Returns:
            The error.

        Example:
            >>> SchemaMismatchError.concat_column("diagonal", "x", "Int64", "String").CODE.code
            'schema-mismatch'
        """
        return cls(
            f'pl.concat(how="{how}") column "{name}" does not reconcile: {left} vs. {right}',
        )

    @classmethod
    def stacked(cls, operation: str) -> SchemaMismatchError:
        """Build the message for a schema that differs in a stack or a sorted merge.

        Args:
            operation: `".vstack()"`, `".extend()"` or `".merge_sorted()"`.

        Returns:
            The error.

        Example:
            >>> SchemaMismatchError.stacked(".vstack()").CODE.code
            'schema-mismatch'
        """
        return cls(f"{operation} requires an identical schema")


class InvalidCallShapeError(AstrolabeError):
    """A call that polars refuses whatever the data, because of the form of its arguments.

    An example is `.explode()` with no columns. Each place that reports
    this error writes its own message, because the fault differs for each
    call. Only the code is shared.

    Example:
        >>> InvalidCallShapeError("rename() with no arguments always raises").CODE.code
        'invalid-call-shape'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "invalid-call-shape",
        "Check that a call's shape can run at all",
        "astrolabe",
    )
    TOO_LARGE: ClassVar[str] = "integer too large for Polars"

    @classmethod
    def refused_by_polars(cls, error: str, message: str) -> InvalidCallShapeError:
        """Build the message for a call that polars refuses while it builds the expression.

        Args:
            error: The name of the class of the error of polars.
            message: The first line of the message of polars.

        Returns:
            The error.

        An integer that no data type holds gets the message of
        `integer_too_large`, which says what the value is.

        Example:
            >>> InvalidCallShapeError.refused_by_polars("ComputeError", "no")._message
            'This call raises ComputeError before any frame meets it: no'
            >>> too_large = "integer too large for Polars: OverflowError"
            >>> error = InvalidCallShapeError.refused_by_polars("InvalidOperationError", too_large)
            >>> error._message.startswith("an integer that no polars data type holds")
            True
        """
        if cls.TOO_LARGE in message:
            return cls.integer_too_large()
        return cls(f"This call raises {error} before any frame meets it: {message}")

    @classmethod
    def refused_at_every_frame(cls, refusal: str) -> InvalidCallShapeError:
        """Build the message for a call whose expression polars refuses at every frame.

        Polars builds such an expression, and refuses it at whatever data
        types a frame gives its columns, as `.name.keep()` over an expression
        that reads no column.

        Args:
            refusal: Why polars refuses, and how: the reason in the words of
                the checker, the way that polars stops and the class of its
                error.

        Returns:
            The error.

        Example:
            >>> refusal = "the lengths differ, and polars raises ShapeError"
            >>> InvalidCallShapeError.refused_at_every_frame(refusal)._message
            'This call has no result at any frame: the lengths differ, and polars raises ShapeError'
        """
        return cls(f"This call has no result at any frame: {refusal}")

    @classmethod
    def integer_too_large(cls) -> InvalidCallShapeError:
        """Build the message for a Python integer that no polars data type holds.

        polars raises when it turns such an integer into a value, for
        example in `pl.lit(2**127)` written out in digits, or in a fill.

        Returns:
            The error.

        Example:
            >>> InvalidCallShapeError.integer_too_large().CODE.code
            'invalid-call-shape'
        """
        return cls(
            "an integer that no polars data type holds raises InvalidOperationError: "
            "integer too large for Polars",
        )

    @classmethod
    def alias_in_join_key(cls, operation: str) -> InvalidCallShapeError:
        """Build the message for a join key that holds `.alias()` anywhere in its tree.

        polars refuses such a key in `.join()` and `.join_asof()`, and such a
        predicate in `.join_where()`, before it reads a row. A rename through
        `.name`, such as `.name.suffix()`, is no alias.

        Args:
            operation: The name of the join method.

        Returns:
            The error.

        Example:
            >>> InvalidCallShapeError.alias_in_join_key("join")._message[:42]
            'join() with an alias in a join key raises '
        """
        return cls(
            f"{operation}() with an alias in a join key raises InvalidOperationError: "
            "'alias' is not allowed in a join key, use 'with_columns' first",
        )

    @classmethod
    def not_a_selector(cls, operation: str) -> InvalidCallShapeError:
        """Build the message for an argument that is no name, no column and no selector.

        polars expands such an argument to the names of columns before the
        operation runs, and no name follows from an expression such as
        `pl.col("a") + 1`.

        Args:
            operation: The name of the operation.

        Returns:
            The error.

        Example:
            >>> InvalidCallShapeError.not_a_selector("drop")._message[:38]
            'drop() with an expression that is not '
        """
        return cls(
            f"{operation}() with an expression that is not a selector raises "
            "InvalidOperationError: polars takes only the columns that a name or a selector "
            "picks. Pass a column name, pl.col(...) or a selector",
        )

    @classmethod
    def key_changes_rows(cls, operation: str) -> InvalidCallShapeError:
        """Build the message for a join key that does not work value by value.

        polars moves a key of one value into a column first, and refuses any
        other key that changes the rows, as `shift`, `sort` or `over`, before
        it reads a row.

        Args:
            operation: The name of the join method.

        Returns:
            The error.

        Example:
            >>> InvalidCallShapeError.key_changes_rows("join")._message[:40]
            'join() with a join key that changes the '
        """
        return cls(
            f"{operation}() with a join key that changes the rows raises InvalidOperationError: "
            "all join key expressions must be elementwise",
        )

    @classmethod
    def zone_object(cls) -> InvalidCallShapeError:
        """Build the message for a `datetime.timezone` object where polars needs the name of a zone.

        The signature of polars accepts the object, and polars raises
        `TypeError` at run time, so the call can never run.

        Returns:
            The error.

        Example:
            >>> InvalidCallShapeError.zone_object().CODE.code
            'invalid-call-shape'
        """
        return cls(
            "polars refuses a datetime.timezone here and raises at run time. "
            'Name the zone, for example "UTC"',
        )

    @classmethod
    def marker_class(cls, marker: str) -> InvalidCallShapeError:
        """Build the message for a marker class of astrolabe where a call needs a data type.

        A marker class, such as `List[pl.Int64]`, writes a data type in a
        type annotation. polars cannot read it as a data type, and raises
        in a cast, a `dtype=`, a `return_dtype=` and a `schema=`.

        Args:
            marker: The name of the class, with `[...]` where it has type
                arguments, for example `List[...]` or `Unchecked`.

        Returns:
            The error. Its message names the polars form to pass.

        Example:
            >>> InvalidCallShapeError.marker_class("List[...]").CODE.code
            'invalid-call-shape'
        """
        fixes = {
            "Unchecked": "a data type of polars",
            "Decimal": "pl.Decimal(precision, scale)",
        }
        fix = fixes.get(marker, f"pl.{marker.removesuffix('[...]')}(...)")
        return cls(
            f"{marker} is a type annotation, not a data type of polars, "
            f"and polars raises: pass {fix}",
        )

    @classmethod
    def user_class(cls, name: str) -> InvalidCallShapeError:
        """Build the message for a subclass of a polars data type that user code defines.

        polars does not read such a class as a data type where a call
        takes one, for example in `pl.Series(..., dtype=...)` or a cast.

        Args:
            name: The short name of the class.

        Returns:
            The error.

        Example:
            >>> InvalidCallShapeError.user_class("Tagged").CODE.code
            'invalid-call-shape'
        """
        return cls(
            f"{name} subclasses a data type of polars, but polars does not read it "
            "as a data type here, and raises: pass a data type of polars",
        )

    @classmethod
    def array_without_shape(cls) -> InvalidCallShapeError:
        """Build the message for a call of `pl.Array` that passes no shape.

        polars needs the width of the array, and raises `TypeError` for a
        call with no shape, whatever the element.

        Returns:
            The error.

        Example:
            >>> InvalidCallShapeError.array_without_shape().CODE.code
            'invalid-call-shape'
        """
        return cls(
            "pl.Array() with no shape always raises TypeError: polars needs the width "
            "of the array. Pass the shape, for example pl.Array(pl.Int64, shape=2)",
        )


class VerticalConcatDisallowedError(AstrolabeError):
    """A vertical `pl.concat()` when the setting `disallow_vertical_concat` is on.

    A vertical concat stacks frames by column position. The checker never
    types its result, because it cannot prove the order of the columns that
    a remainder holds. By default the call then falls back to the plain
    frame type. With this setting, each vertical concat is an error, both
    `"vertical"` and `"vertical_relaxed"`. See `AstrolabeConfig`.

    Example:
        >>> VerticalConcatDisallowedError("vertical").CODE.code
        'vertical-concat-disallowed'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "vertical-concat-disallowed",
        'Check pl.concat(how="vertical"/"vertical_relaxed") is not disallowed by config',
        "astrolabe",
    )

    def __init__(self, how: str) -> None:
        """Build the message for a vertical concat that the setting refuses.

        Args:
            how: The value of `how` in the call, `"vertical"` or
                `"vertical_relaxed"`.
        """
        super().__init__(
            f'pl.concat(how="{how}") is disallowed: column order cannot be verified for a '
            "schema derived through a Rest-generic function (set disallow_vertical_concat = "
            "false in [tool.astrolabe] to allow it as an untyped result instead)",
        )


class ImplicitlyUntypedError(UnprovenError):
    """An operation gives a frame whose schema the checker does not know.

    The operation cannot prove the schema of its result, so it gives the
    plain `pl.DataFrame` or `pl.LazyFrame`. The message states the reason.
    The `allow_unproven` setting decides whether the checker reports it.

    Example:
        >>> from astrolabe._hooks.unproven import Reason
        >>> ImplicitlyUntypedError(Reason.DYNAMIC_NAME, "select").CODE.code
        'implicitly-untyped'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "implicitly-untyped",
        "Check for an operation whose result schema the plugin cannot verify",
        "astrolabe",
    )

    def __init__(self, reason: Reason, operation: str) -> None:
        """Build the message for an operation that gives a plain frame.

        Args:
            reason: Why the operation gave up.
            operation: The name of the operation.
        """
        super().__init__(
            f'The result of "{operation}" has no known schema: {reason.wording}',
            reason,
            operation,
        )


class ImplicitlyUncheckedError(UnprovenError):
    """One column of the result of an operation has the data type `Unchecked`.

    The operation knows the name of the column but cannot prove its data
    type. Unlike `ImplicitlyUntypedError`, the rest of the result keeps its
    schema. The `allow_unproven` setting decides whether the checker reports
    it.

    Example:
        >>> from astrolabe._hooks.unproven import Reason
        >>> ImplicitlyUncheckedError(Reason.BARE_INT, "select", "x").CODE.code
        'implicitly-unchecked'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "implicitly-unchecked",
        "Check for a column whose dtype the plugin cannot verify",
        "astrolabe",
    )

    def __init__(self, reason: Reason, operation: str, name: str) -> None:
        """Build the message for one column that becomes `Unchecked`.

        Args:
            reason: Why the operation gave up.
            operation: The operation that gives the column.
            name: The name of the column.
        """
        super().__init__(
            f'Column "{name}" of "{operation}" is Unchecked: {reason.wording}',
            reason,
            operation,
        )


class WrongInferredVariantError(AstrolabeError):
    """A value marked with `infer(...)` does not have a type of the kind that the function declares.

    A function whose return type is `InferredExpr` must return expressions:
    typed expressions or plain `pl.Expr`. A function whose return type is
    `InferredDataFrame` or `InferredLazyFrame` must return a frame of that
    kind, typed or plain.

    There is one factory for each form of the value: a value of another
    family, named in words (`of_kind`), and any other value, named by its
    type (`of_type`).

    Example:
        >>> WrongInferredVariantError.of_type("InferredExpr", "builtins.str").CODE.code
        'wrong-inferred-variant'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "wrong-inferred-variant",
        "Check that every infer(...) variant matches its placeholder's declared family",
        "astrolabe",
    )

    @classmethod
    def of_type(cls, placeholder_name: str, actual_type: str) -> WrongInferredVariantError:
        """Build the message for a value that no family accepts, named by its type.

        Args:
            placeholder_name: The declared return type: `"InferredExpr"`,
                `"InferredDataFrame"` or `"InferredLazyFrame"`.
            actual_type: The type of the value, for the message.

        Returns:
            The error.

        Example:
            >>> error = WrongInferredVariantError.of_type("InferredExpr", "Literal[1]?")
            >>> error._message[:41]
            'infer(...) variant has type "Literal[1]?"'
        """
        message = f'infer(...) variant has type "{actual_type}", which is not a valid variant'
        return cls(f"{message} of {placeholder_name}")

    @classmethod
    def of_kind(cls, placeholder_name: str, kind: str) -> WrongInferredVariantError:
        """Build the message for a value of another family, named in words.

        The type of a typed value is long, and it holds forms of the plugin
        that a reader does not know. The kind of the value names the fault.

        Args:
            placeholder_name: The declared return type.
            kind: The kind of the value, for example `"an expression"`.

        Returns:
            The error.

        Example:
            >>> error = WrongInferredVariantError.of_kind("InferredDataFrame", "an expression")
            >>> error._message[:35]
            'infer(...) variant is an expression'
        """
        message = f"infer(...) variant is {kind}, which is not a valid variant"
        return cls(f"{message} of {placeholder_name}")


class InferMisuseError(AstrolabeError):
    """`infer()`, or one of the inferred return types, used where it has no effect.

    There is one factory for each fault: `infer()` in a function whose
    return type is not an inferred one (`outside_inferred_function`),
    `infer()` that is not the whole expression of a `return`
    (`not_directly_returned`), and an inferred type in a place other than
    the return type of a function (`wrong_position`).

    Example:
        >>> InferMisuseError.not_directly_returned().CODE.code
        'infer-misuse'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "infer-misuse",
        "Check that infer() and Inferred* placeholders are used correctly",
        "astrolabe",
    )

    @classmethod
    def outside_inferred_function(cls) -> InferMisuseError:
        """Build the message for `infer()` in a function whose return type is not an inferred one.

        Returns:
            The error.

        Example:
            >>> InferMisuseError.outside_inferred_function().CODE.code
            'infer-misuse'
        """
        return cls(
            "infer() has no effect here: the enclosing function's return type is not "
            "InferredExpr/InferredDataFrame/InferredLazyFrame",
        )

    @classmethod
    def not_directly_returned(cls) -> InferMisuseError:
        """Build the message for an `infer(...)` that is not the whole `return` expression.

        Returns:
            The error.

        Example: see class docstring.
        """
        return cls("infer(...) must be the return statement's own expression")

    # The fix that the message of a misplaced inferred return type gives.
    _TYPEVAR_FIX: ClassVar[str] = "use an explicit TypeVar for a generic parameter instead"
    _POSITION_FIXES: ClassVar[dict[str, str]] = {
        "InferredRemainder": "declare the remainder of a parameter with other_columns() instead",
    }

    @classmethod
    def wrong_position(cls, placeholder_name: str) -> InferMisuseError:
        """Build the message for an inferred type outside the return type of a function.

        Args:
            placeholder_name: The name of the inferred type.

        Returns:
            The error.

        Example:
            >>> InferMisuseError.wrong_position("InferredDataFrame").CODE.code
            'infer-misuse'
        """
        fix = cls._POSITION_FIXES.get(placeholder_name, cls._TYPEVAR_FIX)
        return cls(f"{placeholder_name} can only be used as a function's own return type: {fix}")


class ImplicitlyInferredFallbackError(AstrolabeError):
    """An inferred return type falls back to the plain type, and the setting refuses it.

    If one value marked with `infer(...)` is a plain type that the checker
    does not track, the whole return type of the function falls back to the
    plain type, not only that one value. By default, that happens in
    silence. With `disallow_implicit_inferred_fallback`, it is an error. See
    `AstrolabeConfig`.

    Example:
        >>> ImplicitlyInferredFallbackError("InferredExpr").CODE.code
        'implicitly-inferred-fallback'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "implicitly-inferred-fallback",
        "Check for an Inferred*-returning function with an untyped infer(...) variant",
        "astrolabe",
    )

    def __init__(self, placeholder_name: str) -> None:
        """Build the message for a return type that falls back to the plain type.

        Args:
            placeholder_name: The declared inferred return type.
        """
        super().__init__(
            f"One of this function's infer(...) variants is untyped, so its whole return type "
            f"falls back to the plain, erased type instead of {placeholder_name}'s own union "
            "(set disallow_implicit_inferred_fallback = false in [tool.astrolabe] to allow this, "
            "or set [tool.astrolabe.disallow_implicit_inferred_fallback_overrides] "
            f"{placeholder_name.removeprefix('Inferred').lower()} = false to allow just this "
            "placeholder)",
        )


class MarkerArityError(AstrolabeError):
    """A marker type has the wrong number of type arguments.

    The plugin defines marker types that a user writes in annotations, for
    example `TypedDataFrame[...]`. Each place that reports this error writes
    its own message, because each marker has its own usage. Only the code
    is shared.
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "marker-arity",
        "Check a type-level marker's argument count",
        "astrolabe",
    )


class MarkerArgumentError(AstrolabeError):
    """A type argument of a marker type does not have the form that the marker needs.

    Each place that reports this error writes its own message, because each
    marker has its own usage. Only the code is shared.
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "marker-arg",
        "Check a type-level marker's argument shape",
        "astrolabe",
    )


class ColumnsNotTotalError(AstrolabeError):
    """A `Columns` class declares `total=False`, `Required[...]` or `NotRequired[...]`.

    A polars column is never optional: a frame holds it, or it does not. An
    optional field has no meaning for a frame.

    Example:
        >>> ColumnsNotTotalError("Legislators").CODE.code
        'columns-not-total'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "columns-not-total",
        "Check that a Columns subclass declares every field required",
        "astrolabe",
    )

    def __init__(self, name: str) -> None:
        """Build the message for a `Columns` class that is not total.

        Args:
            name: The name of the `Columns` class.
        """
        super().__init__(
            f'Columns subclass "{name}" must not use total=False, Required[...], or '
            "NotRequired[...]: every column is required",
        )


class ColumnsNotClosedError(AstrolabeError):
    """A `Columns` class declares `closed=False`.

    The declared columns of a frame are complete by definition. To declare
    a frame that can hold more columns, a user writes a remainder with
    `other_columns()`, and passes it as the second argument of the frame
    type.

    Example:
        >>> ColumnsNotClosedError("Legislators").CODE.code
        'columns-not-closed'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "columns-not-closed",
        "Check that a Columns subclass is closed",
        "astrolabe",
    )

    def __init__(self, name: str) -> None:
        """Build the message for a `Columns` class that is not closed.

        Args:
            name: The name of the `Columns` class.
        """
        super().__init__(
            f'Columns subclass "{name}" must not use closed=False. Declare a remainder '
            "with other_columns() and write it as the frame's second argument, for "
            f"example TypedDataFrame[{name}, Rest]",
        )


class ColumnsFieldTypeError(AstrolabeError):
    """A `Columns` class declares a field whose type is not a polars data type.

    Example:
        >>> ColumnsFieldTypeError("price").CODE.code
        'columns-field-type'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "columns-field-type",
        "Check that every Columns field is a polars dtype",
        "astrolabe",
    )

    def __init__(self, field_name: str) -> None:
        """Build the message for a field whose type is not a data type.

        Args:
            field_name: The name of the field.
        """
        super().__init__(
            f'Column "{field_name}" must be a polars dtype or an astrolabe dtype marker',
        )


class FrameTypeNotClosedError(AstrolabeError):
    """A runtime schema call gets a frame type that does not declare every column.

    `check_schema`, `force_schema` and `assume_schema` give a frame the type
    that their second argument names. That type must be a closed frame type:
    a schema class and no remainder. The undeclared columns of an open frame
    have no data type, so the call cannot check or cast them. A frame type
    with no schema class declares no column at all.

    There is one factory for each fault: a frame type with a remainder
    (`with_remainder`), and a frame type with no schema class
    (`without_schema`).

    Example:
        >>> FrameTypeNotClosedError.with_remainder("check_schema", "TypedDataFrame").CODE.code
        'frame-type-not-closed'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "frame-type-not-closed",
        "Check that a runtime schema call gets a closed frame type",
        "astrolabe",
    )
    MESSAGE: ClassVar[str] = (
        "{function}() needs a closed frame type, and this frame type {fault}. Pass a frame "
        "type with a schema class and no remainder, for example {frame}[MyColumns]"
    )

    @classmethod
    def with_remainder(cls, function: str, frame: str) -> FrameTypeNotClosedError:
        """Build the message for a frame type with a remainder.

        Args:
            function: The name of the called function, for example `"check_schema"`.
            frame: The name of the frame class, for example `"TypedDataFrame"`.

        Returns:
            The error.

        Example:
            >>> error = FrameTypeNotClosedError.with_remainder("check_schema", "TypedDataFrame")
            >>> error._message[:69]
            'check_schema() needs a closed frame type, and this frame type has und'
        """
        fault = "has undeclared columns"
        return cls(cls.MESSAGE.format(function=function, fault=fault, frame=frame))

    @classmethod
    def without_schema(cls, function: str, frame: str) -> FrameTypeNotClosedError:
        """Build the message for a frame type with no schema class.

        Args:
            function: The name of the called function.
            frame: The name of the frame class.

        Returns:
            The error.

        Example:
            >>> error = FrameTypeNotClosedError.without_schema("force_schema", "TypedLazyFrame")
            >>> error._message[-25:]
            'TypedLazyFrame[MyColumns]'
        """
        fault = "has no schema class"
        return cls(cls.MESSAGE.format(function=function, fault=fault, frame=frame))


class MalformedDeclarationError(AstrolabeError):
    """A declaration of an expression breaks one of its rules of form.

    The checker checks these rules at the declaration, never later at a
    call. Examples: a `typed_expr` that reads no column, rows over different
    column names, rows that overlap, a `same` that names no column that the
    declaration reads, two blank reads, an empty region, a column read two
    times, and a blank read next to a named read or under `named=`.

    Example:
        >>> MalformedDeclarationError("a column is read twice").CODE.code
        'malformed-declaration'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "malformed-declaration",
        "Check a slice declaration against its well-formedness rules",
        "astrolabe",
    )

    def __init__(self, problem: str) -> None:
        """Build the message for one broken rule.

        Args:
            problem: The broken rule, in the words of the engine.
        """
        super().__init__(f"Malformed declaration: {problem}")


class ExpressionFitError(AstrolabeError):
    """An expression does not fit the type that its parameter, or its position, declares.

    The check of fit has three parts. The argument reads only the columns
    that the declaration reads. Its output name and its data type at each
    declared combination of inputs agree with the declaration. It treats
    rows in the same way: row by row, as an aggregation, or as a literal.
    The message names the first part that fails.

    Example:
        >>> ExpressionFitError.at_argument("rule", "it has no result").CODE.code
        'expression-fit'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "expression-fit",
        "Check an expression argument against its declared parameter",
        "astrolabe",
    )

    @classmethod
    def at_argument(cls, parameter: str, refusal: str) -> ExpressionFitError:
        """Build the message for one call argument that does not fit.

        Args:
            parameter: The name of the parameter.
            refusal: The part that fails, in the words of the engine.

        Returns:
            The error.

        Example:
            >>> ExpressionFitError.at_argument("rule", "it has no result").CODE.code
            'expression-fit'
        """
        return cls(f'Argument "{parameter}" does not match its declared type: {refusal}')

    @classmethod
    def at_context(cls, refusal: str) -> ExpressionFitError:
        """Build the message for a value that does not fit the type declared at its position.

        A value in a `return` or in an annotated assignment gets the same check
        as a call argument. The declared type is the one written at that
        position, not the type of a parameter.

        Args:
            refusal: The part that fails, in the words of the engine.

        Returns:
            The error.

        Example:
            >>> ExpressionFitError.at_context("it has no result").CODE.code
            'expression-fit'
        """
        return cls(f"This expression does not match its declared type: {refusal}")


class UntypedArgumentError(UnprovenError):
    """A value that is not a typed expression goes to a declared expression parameter.

    The checker cannot read the columns and the data types of such a value,
    so it cannot check the fit. If the setting lets it pass, the call is
    checked as declared, and the declared type is a claim that the checker
    did not prove.

    Example:
        >>> error = UntypedArgumentError("price", "this value is not a selection of columns")
        >>> error.CODE.code, error.operation, error.reason.level.spelling
        ('expression-fit', 'fit', 'asserted')
    """

    CODE: ClassVar[ErrorCode] = ExpressionFitError.CODE

    def __init__(self, parameter: str, wording: str) -> None:
        """Build the message for one argument that is not typed.

        Args:
            parameter: The name of the parameter.
            wording: What the argument is not, in the words of the fit check.
        """
        super().__init__(
            f'Argument "{parameter}" does not match its declared type: {wording}',
            Reason.UNTYPED_ARGUMENT,
            "fit",
        )


class SelectionRenameError(AstrolabeError):
    """A rename that a selection cannot take, because it has many outputs.

    A selection names no single output, so `.alias(...)` cannot name it. A
    rename of the whole name, such as `.name.to_lowercase()`, can turn two
    different columns into one name. Only a prefix or a suffix, with
    `.name.prefix` or `.name.suffix`, keeps different names different.

    Example:
        >>> SelectionRenameError("alias").CODE.code
        'selection-rename'
    """

    CODE: ClassVar[ErrorCode] = ErrorCode(
        "selection-rename",
        "Check that a rename over a selection keeps every output distinct",
        "astrolabe",
    )

    def __init__(self, member: str) -> None:
        """Build the message for one rename that a selection cannot take.

        Args:
            member: The name of the rename method.
        """
        super().__init__(
            f"A selection has several outputs, which .{member}() cannot rename: "
            "use .name.suffix() or .name.prefix()"
        )
