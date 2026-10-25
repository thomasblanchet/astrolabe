"""The arithmetic of nested values: `List`, `Array` and `Struct`, as polars runs it.

Polars runs `+`, `-`, `*`, `/`, `%` and `//` on a list or an array leaf by
leaf, and on a struct field by field. A list or an array needs leaves of a few
types and the same nesting on both sides, and the output keeps the nesting with
the supertype of the leaves. A struct pairs its fields with the fields of the
other struct, and runs the flat kernel on each pair. `**`, `&`, `|` and `^`
have no nested kernel, so a nested input raises there. The operation of each
kernel is a `NumericOp`.
"""

from __future__ import annotations

import enum
from typing import Protocol

from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.dispatch import Data  # ruff: ignore[typing-only-first-party-import]
from astrolabe._rules.operators import NumericOp  # ruff: ignore[typing-only-first-party-import]


class ListInput:
    """Say whether a data type can take part in the arithmetic of a list or an array.

    A leaf must be a type that list arithmetic takes (`Is.list_arithmetic_input`):
    an integer, a float, `Boolean` or `Null`. A nested side must hold the
    same kind at every level down to the leaf: a list of lists, but not a
    list of arrays.

    Example:
        >>> nested = DType("List", (DType("Array", (DType("Int8"), 2)),))
        >>> ListInput().all_levels(nested, "List")
        False
    """

    def all_levels(self, dtype: DType, kind: str) -> bool:
        """Say whether the type is the given kind at every level above a supported leaf.

        Args:
            dtype: The type.
            kind: `List` or `Array`.

        Returns:
            Whether it is.

        Example:
            >>> ListInput().all_levels(DType("List", (DType("Int8"),)), "List")
            True
        """
        if dtype.name == kind:
            return self.all_levels(dtype.inner(), kind)
        return Is.list_arithmetic_input(dtype)


class ApplyType(enum.Enum):
    """Name which sides of a nested kernel are nested.

    Example:
        >>> ApplyType.PRIMITIVE_TO_LIST.value
        'PrimitiveToList'
    """

    LIST_TO_LIST = "ListToList"
    LIST_TO_PRIMITIVE = "ListToPrimitive"
    PRIMITIVE_TO_LIST = "PrimitiveToList"

    def nested(self, left: Data, right: Data) -> DType:
        """Give the type of the nested side, whose nesting the output keeps.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The type of the right input for `PRIMITIVE_TO_LIST`, else of the left input.

        Example:
            >>> ApplyType.PRIMITIVE_TO_LIST.nested(Data(DType("Int8")), Data(DType("Null")))
            Null
        """
        return right.dtype if self is ApplyType.PRIMITIVE_TO_LIST else left.dtype

    def all_null(self, left: Data, right: Data) -> bool:
        """Say whether a nested side holds only nulls, so that the kernel returns nulls at once.

        A nested kernel reads the outer validity of each nested side, and
        returns a series of nulls of the output type where one holds no value.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            Whether a nested side holds only nulls.

        Example:
            >>> ApplyType.LIST_TO_PRIMITIVE.all_null(Data(DType("Int8")), Data(DType("Null")))
            False
        """
        left_nested = self is not ApplyType.PRIMITIVE_TO_LIST
        right_nested = self is not ApplyType.LIST_TO_PRIMITIVE
        return (left_nested and left.nulls) or (right_nested and right.nulls)


class LeafKernel:
    """Check that a nested kernel has a numeric kernel for the type of the output leaves.

    The kernels of lists and arrays cast the leaves to the output leaf type and
    pick a numeric kernel for it. The choice of the kernel panics for a type
    that is not an integer or a float, such as `Boolean` beside `Null`.

    Example:
        >>> LeafKernel().check(DType("Float64")) is None
        True
    """

    def check(self, leaf: DType) -> None:
        """Raise a panic where the leaf type has no numeric kernel.

        Args:
            leaf: The leaf type of the output.

        Raises:
            PolarsRefusal: The leaf type is not an integer or a float.

        Example:
            >>> LeafKernel().check(DType("Boolean"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: no numeric kernel ...
        """
        if not Is.primitive_numeric(leaf):
            raise PolarsRefusal(Stop.PANICS, f"no numeric kernel for {leaf!r}")


class ListArithmetic:
    """Give the type of an arithmetic operation where one side is a `List`.

    The kernel joins the leaf types first. A list beside a list needs a flat
    inner type on both sides. A list beside a flat value needs a list at
    every level down to the leaf. The output is the nested side with its leaf
    replaced by the joined leaf type.

    Example:
        >>> lists = DType("List", (DType("Int8"),))
        >>> ListArithmetic(NumericOp.ADD).output(Data(lists), Data(DType("Float32")))
        List(Float32)
    """

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation.

        Example:
            >>> ListArithmetic(NumericOp.REM)._op
            <NumericOp.REM: 'rem'>
        """
        self._op = op
        self._inputs = ListInput()

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the types do not fit, or the leaf type has no kernel.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The type of the output.

        Example:
            >>> booleans = DType("List", (DType("Boolean"),))
            >>> ListArithmetic(NumericOp.MUL).output(Data(booleans), Data(booleans))
            List(UInt32)
        """
        leaf = self._op.leaf_supertype(left.dtype.leaf(), right.dtype.leaf())
        apply = self._apply_type(left.dtype, right.dtype)
        output = apply.nested(left, right).with_leaf(leaf)
        if apply.all_null(left, right):
            return output
        LeafKernel().check(leaf)
        return output

    def _apply_type(self, left: DType, right: DType) -> ApplyType:
        """Give which sides are lists, or raise where the types do not fit.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The sides that are lists.

        Raises:
            PolarsRefusal: The types do not fit.

        Example:
            >>> lists = DType("List", (DType("Int8"),))
            >>> ListArithmetic(NumericOp.ADD)._apply_type(DType("Null"), lists)
            <ApplyType.PRIMITIVE_TO_LIST: 'PrimitiveToList'>
        """
        if left.name == right.name == "List":
            inners = (left.inner(), right.inner())
            if not all(Is.list_arithmetic_input(inner) for inner in inners):
                message = f"cannot {self._op.value} two list columns with non-numeric inner types"
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
            return ApplyType.LIST_TO_LIST
        if left.name == "List" and Is.list_arithmetic_input(right):
            self._ensure_levels(left)
            return ApplyType.LIST_TO_PRIMITIVE
        if right.name == "List" and Is.list_arithmetic_input(left):
            self._ensure_levels(right)
            return ApplyType.PRIMITIVE_TO_LIST
        message = f"{self._op.value} operation not supported for {left!r} and {right!r}"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def _ensure_levels(self, dtype: DType) -> None:
        """Raise where the list side is not a list at every level.

        Args:
            dtype: The type of the list side.

        Raises:
            PolarsRefusal: It is not.

        Example:
            >>> lists = DType("List", (DType("Int8"),))
            >>> ListArithmetic(NumericOp.ADD)._ensure_levels(lists) is None
            True
        """
        if not self._inputs.all_levels(dtype, "List"):
            message = f"{dtype!r} is not a list at every level"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


class ArrayArithmetic:
    """Give the type of an arithmetic operation where one side is an `Array`.

    The kernel needs leaves that it accepts on both sides, joins them, and
    needs an array at every level down to the leaf. An array beside an array
    needs the same type on both sides once the leaves are joined, so the widths
    must agree. The output is the array side with its leaf replaced.

    Example:
        >>> arrays = DType("Array", (DType("Int8"), 2))
        >>> ArrayArithmetic(NumericOp.DIV).output(Data(arrays), Data(arrays))
        Array(Float64, shape=(2,))
    """

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation.

        Example:
            >>> ArrayArithmetic(NumericOp.SUB)._op
            <NumericOp.SUB: 'sub'>
        """
        self._op = op
        self._inputs = ListInput()

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        An array beside an array of the same length runs at once, with no check
        of nulls. Every other case returns nulls where its array side holds
        only nulls.

        It raises `PolarsRefusal` where the types do not fit, or the leaf type has no kernel.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The type of the output.

        Example:
            >>> arrays = DType("Array", (DType("Boolean"), 2))
            >>> nulls = Data(DType("Null"))
            >>> ArrayArithmetic(NumericOp.ADD).output(Data(arrays), nulls)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        self._check_leaves(left.dtype.leaf(), right.dtype.leaf())
        leaf = self._op.leaf_supertype(left.dtype.leaf(), right.dtype.leaf())
        apply = self._apply_type(left.dtype, right.dtype, leaf)
        output = apply.nested(left, right).with_leaf(leaf)
        direct = apply is ApplyType.LIST_TO_LIST and left.one_row == right.one_row
        if direct:
            LeafKernel().check(leaf)
            return output
        if apply.all_null(left, right):
            return output
        LeafKernel().check(leaf)
        return output

    def _check_leaves(self, left: DType, right: DType) -> None:
        """Raise on a leaf type that the kernels refuse.

        Args:
            left: The left leaf type.
            right: The right leaf type.

        Raises:
            PolarsRefusal: A leaf type is not accepted.

        Example:
            >>> ArrayArithmetic(NumericOp.ADD)._check_leaves(DType("Int8"), DType("Null")) is None
            True
        """
        if not (Is.list_arithmetic_input(left) and Is.list_arithmetic_input(right)):
            message = f"cannot {self._op.value} non-numeric inner dtypes: {left!r} and {right!r}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def _apply_type(self, left: DType, right: DType, leaf: DType) -> ApplyType:
        """Give which sides are arrays, or raise where the types do not fit.

        An array beside an array needs one type once the leaves are joined. An
        array beside a flat value needs a leaf that the kernels accept.

        Args:
            left: The left type.
            right: The right type.
            leaf: The leaf type of the output.

        Returns:
            The sides that are arrays.

        Raises:
            PolarsRefusal: The types do not fit.

        Example:
            >>> two = DType("Array", (DType("Int8"), 2))
            >>> three = DType("Array", (DType("Int8"), 3))
            >>> kernel = ArrayArithmetic(NumericOp.ADD)
            >>> kernel._apply_type(two, three, DType("Int8"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> kernel._apply_type(two, DType("String"), DType("Int8"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot add dtypes: ...
        """
        if left.name == right.name == "Array":
            if not left.with_leaf(leaf).same(right.with_leaf(leaf)):
                message = f"cannot {self._op.value} columns: differing dtypes"
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
            self._ensure_levels(left)
            return ApplyType.LIST_TO_LIST
        if left.name == "Array" and Is.list_arithmetic_input(right):
            self._ensure_levels(left)
            return ApplyType.LIST_TO_PRIMITIVE
        if right.name != "Array" or not Is.list_arithmetic_input(left):  # pragma: no cover
            # The type function refuses an array beside a type of no arithmetic first.
            message = f"cannot {self._op.value} dtypes: {left!r} != {right!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        self._ensure_levels(right)
        return ApplyType.PRIMITIVE_TO_LIST

    def _ensure_levels(self, dtype: DType) -> None:
        """Raise where the array side is not an array at every level.

        Args:
            dtype: The type of the array side.

        Raises:
            PolarsRefusal: It is not.

        Example:
            >>> inner = DType("List", (DType("Int8"),))
            >>> kernel = ArrayArithmetic(NumericOp.ADD)
            >>> kernel._ensure_levels(DType("Array", (inner, 2)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if not self._inputs.all_levels(dtype, "Array"):
            message = f"{dtype!r} is not an array at every level"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


class FieldKernel(Protocol):
    """Give the type and the name of a flat kernel that runs on a pair of fields of two structs.

    A struct takes the name of each output field from the series that the
    flat kernel gives. Most kernels name it after the left input, and a few
    paths of broadcast name it after the right input.
    """

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The left field.
            right: The right field.
        """

    def right_named(self, left: Data, right: Data) -> bool:
        """Say whether the output takes the name of the right input.

        Args:
            left: The left field.
            right: The right field.
        """


class StructArithmetic:
    """Give the type of an arithmetic operation of two structs, field by field.

    The kernel pairs the fields and runs the flat kernel of the operation on
    each pair. A struct of one field on the right meets every field on the
    left. A struct of one field on the left meets every field on the right, and
    each result takes the name that the flat kernel gives it. That is mostly
    the name of the one left field, so two fields or more on the right give a
    duplicate name, which raises. Otherwise the fields pair by position: the
    output keeps the names of the left, an extra field on the left stays as it
    is, and an extra field on the right drops out.

    Example:
        >>> from astrolabe._rules.kernels.arithmetic import SeriesArithmetic
        >>> left = DType("Struct", ((("a", DType("Int8")), ("b", DType("Float32"))),))
        >>> right = DType("Struct", ((("x", DType("Int16")),),))
        >>> fields = SeriesArithmetic(NumericOp.ADD)
        >>> StructArithmetic().output(Data(left), Data(right), fields)
        Struct({'a': Int16, 'b': Float32})
    """

    def output(self, left: Data, right: Data, fields: FieldKernel) -> DType:
        """Give the type of the output struct.

        A field of a struct of only nulls holds only nulls, so each pair
        reaches the flat kernel with that fact.

        It raises `PolarsRefusal` where a pair of fields fails, or where the
        fields of the output do not fit.

        Args:
            left: The left struct.
            right: The right struct.
            fields: The flat kernel that runs on each pair of fields.

        Returns:
            The struct type of the output.

        Example:
            >>> from astrolabe._rules.kernels.arithmetic import SeriesArithmetic
            >>> one = DType("Struct", ((("a", DType("Int8")),),))
            >>> two = DType("Struct", ((("x", DType("Int8")), ("y", DType("Int8"))),))
            >>> fields = SeriesArithmetic(NumericOp.ADD)
            >>> StructArithmetic().output(Data(one), Data(two), fields)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: DuplicateError: ...
        """
        mine, theirs = left.dtype.fields(), right.dtype.fields()
        if len(theirs) == 1:
            other = right.field(theirs[0][1])
            paired = [(name, fields.output(left.field(dtype), other)) for name, dtype in mine]
            self._check_length(left, right)
            return DType("Struct", (tuple(paired),))
        if len(mine) == 1:
            return self._one_left(left, right, fields)
        return self._by_position(left, right, fields)

    def _one_left(self, left: Data, right: Data, fields: FieldKernel) -> DType:
        """Give the output where the left struct has one field.

        The struct checks the length of its fields before their names.

        Args:
            left: The left struct, of one field.
            right: The right struct.
            fields: The flat kernel.

        Returns:
            The struct type of the output.

        Raises:
            PolarsRefusal: Two fields of the output have the same name.

        Example:
            >>> from astrolabe._rules.kernels.arithmetic import SeriesArithmetic
            >>> one = DType("Struct", ((("a", DType("Int8")),),))
            >>> none = DType("Struct", ((),))
            >>> StructArithmetic()._one_left(Data(one), Data(none), SeriesArithmetic(NumericOp.ADD))
            Struct({})
        """
        name, dtype = left.dtype.fields()[0]
        only = left.field(dtype)
        paired = []
        for other_name, other in right.dtype.fields():
            data = right.field(other)
            output = fields.output(only, data)
            paired.append((other_name if fields.right_named(only, data) else name, output))
        self._check_length(right, left)
        names = [field_name for field_name, _ in paired]
        if len(set(names)) < len(names):
            message = "two fields have the same name"
            raise PolarsRefusal(ErrorClass.DUPLICATE, message)
        return DType("Struct", (tuple(paired),))

    def right_named(self, left: Data, right: Data) -> bool:
        """Say whether the output struct takes the name of the right struct.

        Where the left struct has one field and the right struct does not, the
        kernel builds the output from the fields of the right struct, so the
        output keeps its name.

        Args:
            left: The left struct.
            right: The right struct.

        Returns:
            Whether the output takes the name of the right struct.

        Example:
            >>> one = DType("Struct", ((("a", DType("Int8")),),))
            >>> StructArithmetic().right_named(Data(one), Data(one))
            False
        """
        return len(left.dtype.fields()) == 1 and len(right.dtype.fields()) != 1

    def _by_position(self, left: Data, right: Data, fields: FieldKernel) -> DType:
        """Give the output where the fields pair by position.

        Each side of one row repeats to the length of the other first.

        It raises `PolarsRefusal` where a pair fails.

        Args:
            left: The left struct.
            right: The right struct.
            fields: The flat kernel.

        Returns:
            The struct type of the output, with the names of the left.

        Example:
            >>> from astrolabe._rules.kernels.arithmetic import SeriesArithmetic
            >>> left = DType("Struct", ((("a", DType("Int8")), ("b", DType("Int8"))),))
            >>> three = (("x", DType("Float32")), ("y", DType("Int8")), ("z", DType("Int8")))
            >>> right = DType("Struct", (three,))
            >>> fields = SeriesArithmetic(NumericOp.ADD)
            >>> StructArithmetic()._by_position(Data(left), Data(right), fields)
            Struct({'a': Float32, 'b': Int8})
        """
        theirs = right.dtype.fields()
        paired = []
        for index, (name, dtype) in enumerate(left.dtype.fields()):
            if index >= len(theirs):
                paired.append((name, dtype))
                continue
            other = right.field(theirs[index][1])
            paired.append((name, fields.output(left.field(dtype), other)))
        return DType("Struct", (tuple(paired),))

    def _check_length(self, outer: Data, other: Data) -> None:
        """Raise where the struct that keeps its fields has one row and the other side has more.

        Each field of the output then has the length of the other side, and a
        struct refuses a field of another length than its own.

        Args:
            outer: The struct whose fields the output keeps.
            other: The other struct.

        Raises:
            PolarsRefusal: The lengths do not fit.

        Example:
            >>> one = Data(DType("Struct", ((),)), one_row=True)
            >>> empty = Data(DType("Struct", ((),)))
            >>> StructArithmetic()._check_length(one, empty)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
        """
        if outer.one_row and not other.one_row:
            message = "a field has another length than the struct"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
