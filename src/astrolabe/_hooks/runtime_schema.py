"""Type `check_schema`, `force_schema` and `assume_schema`, which take a frame type as a value.

A call such as `check_schema(raw, TypedDataFrame[Trades])` gives a frame the
type that its second argument names. In an annotation, mypy reads
`TypedDataFrame[Trades]` through the type-analysis hook of the frame class,
which closes the schema and gives each bare data type class its default
parameters. Written as a value, the same frame type reaches mypy with no such
hook, so it is not the type of the annotation. The hooks of this module give
the value that type. A runtime schema call needs a closed frame type, because
the undeclared columns of an open frame have no data type to check or cast.
"""

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    NoneType,
    TypedDictType,
    TypeOfAny,
    TypeType,
    get_proper_type,
)

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.errors import FrameTypeNotClosedError
from astrolabe.frame import TypedDataFrame, TypedLazyFrame

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext, FunctionSigContext
    from mypy.types import ProperType
    from mypy.types import Type as MypyType

    from astrolabe._hooks.type_analysis import ClosedSchema


class RuntimeSchemaSignatureHook:
    """Give a runtime schema call a signature with no type variable to solve.

    The three functions are declared as `check_schema[F](df, frame_type:
    type[F]) -> F`. mypy solves `F` from the type that the place of the call
    expects, as well as from the argument. In `use(check_schema(raw,
    TypedDataFrame[Trades]))`, the parameter of `use` expects the frame of
    the annotation, so mypy solves `F` as that frame, and then reports the
    argument, whose schema is not yet closed. With this signature the call
    has nothing to solve: its result is the frame that the argument names,
    until `RuntimeSchemaHook` closes the schema.
    """

    def __call__(self, ctx: FunctionSigContext) -> CallableType:
        """Give the signature of one runtime schema call.

        Args:
            ctx: The mypy context of the signature at the call.

        Returns:
            The signature `(df, frame_type: A) -> I`, where `A` is the type
            of the frame type argument and `I` the type of its instances.
            The declared signature where the call passes no single frame
            type, or the argument is not a class.
        """
        if len(ctx.args) < 2 or len(ctx.args[1]) != 1:
            return ctx.default_signature
        actual = ctx.api.get_expression_type(ctx.args[1][0])
        instance = self._instance(get_proper_type(actual))
        if instance is None:
            return ctx.default_signature
        arg_types = list(ctx.default_signature.arg_types)
        arg_types[1] = actual
        return ctx.default_signature.copy_modified(
            arg_types=arg_types,
            ret_type=instance,
            variables=[],
        )

    def _instance(self, actual: ProperType) -> MypyType | None:
        """Give the type of the instances of a class that a call passes as a value.

        A class written as a value, as `TypedDataFrame[Trades]`, has the type
        of its constructor. A parameter of the type `type[F]` holds a class
        whose instances are `F`.

        Args:
            actual: The type of the argument.

        Returns:
            The type of the instances, `Any` for an argument of the type
            `Any`, or None for any other argument.

        Example:
            >>> hook = RuntimeSchemaSignatureHook()
            >>> hook._instance(AnyType(TypeOfAny.special_form))
            Any
            >>> hook._instance(NoneType()) is None
            True
        """
        if isinstance(actual, CallableType) and actual.is_type_obj():
            return actual.ret_type
        if isinstance(actual, TypeType):
            return actual.item
        if isinstance(actual, AnyType):
            return actual
        return None


class RuntimeSchemaHook:
    """Give a runtime schema call the frame type that an annotation of the same frame gives.

    Without this hook, `check_schema(raw, TypedDataFrame[Trades])` gives a
    frame whose schema is not closed, and whose bare data type classes have
    no parameters. That frame is not the frame of the annotation
    `TypedDataFrame[Trades]`, so a call that passes it to a function of that
    parameter is an error. The hook closes the schema with `ClosedSchema`,
    the same step that an annotation takes.

    A frame type with a remainder, or with no schema class, is an error:
    the call needs a frame type that declares every column. A frame type
    that the call reads through a type variable, or a class derived from a
    frame class, keeps the type that mypy gives.

    It is a mypy function hook in the plugin layer. The plugin builds one
    for each of the three functions.
    """

    FRAMES: ClassVar[frozenset[str]] = frozenset(
        {Target.of(TypedDataFrame).fullname, Target.of(TypedLazyFrame).fullname},
    )

    def __init__(self, function: str, closed_schema: ClosedSchema) -> None:
        """Keep the name of the function, and the step that closes a schema.

        Args:
            function: The name of the function, for the message of an error.
            closed_schema: Checks a schema class and closes its schema, with
                the builders of the type checking phase of mypy.
        """
        self._function = function
        self._closed_schema = closed_schema

    def __call__(self, ctx: FunctionContext) -> MypyType:
        """Give the frame type of the call.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The frame with a closed schema and no remainder. `Any` after an
            error. The type that mypy gives, where it is not a frame class.
        """
        frame = get_proper_type(ctx.default_return_type)
        if not isinstance(frame, Instance) or frame.type.fullname not in self.FRAMES:
            return ctx.default_return_type
        schema = get_proper_type(frame.args[0])
        if not isinstance(schema, TypedDictType):
            FrameTypeNotClosedError.without_schema(self._function, frame.type.name).fail(ctx)
            return AnyType(TypeOfAny.from_error)
        if not isinstance(get_proper_type(frame.args[1]), NoneType):
            FrameTypeNotClosedError.with_remainder(self._function, frame.type.name).fail(ctx)
            return AnyType(TypeOfAny.from_error)
        closed = self._closed_schema.closed(ctx, schema)
        if closed is None:
            return AnyType(TypeOfAny.from_error)
        return frame.copy_modified(args=[closed, NoneType()])
