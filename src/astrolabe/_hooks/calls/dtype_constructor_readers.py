"""Turn the arguments of a call such as `pl.Datetime("ms")` into type arguments.

These four data types have parameters: a time unit and a time zone, a precision and a scale, or
a list of categories. The parameters are part of the data type. The checker carries them as the
type arguments of a marker class of this project, for example `Datetime[Literal["ms"], None]`.
Each reader here reads the literal arguments of one call and gives those type arguments.

The checker reads these calls in two phases of mypy: during semantic analysis, for a
declaration such as `column("t", pl.Datetime("ms"))`, and during type checking. The two phases
build a literal type through different mypy APIs. A `LiteralTypeBuilder` does that one step for
one phase, and the code that builds the readers picks the builder for its phase.
"""

from __future__ import annotations

from typing import Final, Protocol, TYPE_CHECKING

import polars as pl
from mypy.types import LiteralType, NoneType
from mypy.types import Type as MypyType

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe.frame import Datetime as _DatetimeMarker
from astrolabe.frame import Decimal as _DecimalMarker
from astrolabe.frame import Duration as _DurationMarker
from astrolabe.frame import Enum as _EnumMarker

if TYPE_CHECKING:
    from collections.abc import ItemsView

    from mypy.nodes import Expression

    from astrolabe._hooks.calls.dtype_constructor_args import DtypeConstructorArgs


class LiteralTypeBuilder(Protocol):
    """Build a mypy `Literal[...]` type from a plain Python value, in one phase of mypy.

    A literal type needs a fallback type, such as `builtins.str`. The API of the type checking
    phase and the API of the semantic analysis phase find that type with different methods.
    `CheckerLiteralTypeBuilder` works in the first phase and `SemanticAnalyzerLiteralTypeBuilder`
    in the second. Code that builds literal types takes this protocol, and so works in either.
    """

    def string(self, value: str, api: object) -> LiteralType: ...

    def integer(self, value: int, api: object) -> LiteralType: ...


class CheckerLiteralTypeBuilder:
    """Build a `Literal[...]` type during type checking, through `named_generic_type`."""

    def string(self, value: str, api: object) -> LiteralType:
        """Build the type `Literal[value]` for a string.

        Args:
            value: The string.
            api: The mypy plugin API of this phase, which finds `builtins.str`.

        Returns:
            The literal type.
        """
        fallback = api.named_generic_type("builtins.str", [])  # type: ignore[attr-defined]
        return LiteralType(value=value, fallback=fallback)

    def integer(self, value: int, api: object) -> LiteralType:
        """Build the type `Literal[value]` for an integer.

        Args:
            value: The integer.
            api: The mypy plugin API of this phase, which finds `builtins.int`.

        Returns:
            The literal type.
        """
        fallback = api.named_generic_type("builtins.int", [])  # type: ignore[attr-defined]
        return LiteralType(value=value, fallback=fallback)


class SemanticAnalyzerLiteralTypeBuilder:
    """Build a `Literal[...]` type during semantic analysis, through `named_type`."""

    def string(self, value: str, api: object) -> LiteralType:
        """Build the type `Literal[value]` for a string.

        Args:
            value: The string.
            api: The mypy plugin API of this phase, which finds `builtins.str`.

        Returns:
            The literal type.
        """
        fallback = api.named_type("builtins.str")  # type: ignore[attr-defined]
        return LiteralType(value=value, fallback=fallback)

    def integer(self, value: int, api: object) -> LiteralType:
        """Build the type `Literal[value]` for an integer.

        Args:
            value: The integer.
            api: The mypy plugin API of this phase, which finds `builtins.int`.

        Returns:
            The literal type.
        """
        fallback = api.named_type("builtins.int")  # type: ignore[attr-defined]
        return LiteralType(value=value, fallback=fallback)


class _ParametrizedDtypeReader(Protocol):
    """What each reader for one data type class gives: its marker, its parameters, a reader.

    `MARKER_TARGET` names the marker class of this project whose type arguments the reader
    builds. `PARAM_NAMES` lists the parameters of the polars class in declared order.
    `DtypeConstructorArgs` needs that order to match a positional argument to its parameter.
    """

    @property
    def MARKER_TARGET(self) -> Target: ...

    @property
    def PARAM_NAMES(self) -> tuple[str, ...]: ...

    def __call__(self, args: DtypeConstructorArgs, api: object) -> list[MypyType] | None: ...


class LiteralStringArgs(Protocol):
    """The two reads that a reader of `time_unit` and `time_zone` needs from its arguments.

    `DtypeConstructorArgs` reads the AST of a call such as `pl.Datetime("us")`. `CallArgs`
    reads the arguments of the call that mypy checks now. Both have these two methods, so the
    reader for `pl.Datetime` works with either.
    """

    def is_omitted_or_none(self, name: str) -> bool: ...

    def literal_str(self, name: str, default: str | None = None) -> str | None: ...


class _DatetimeConstructorReader:
    """Read the time unit and the time zone of a `pl.Datetime(...)` call."""

    MARKER_TARGET: Final[Target] = Target.of(_DatetimeMarker)
    PARAM_NAMES: Final[tuple[str, ...]] = ("time_unit", "time_zone")

    def __init__(self, literal_type: LiteralTypeBuilder) -> None:
        """Keep the builder of literal types for the phase of mypy that uses this reader.

        Args:
            literal_type: The builder that turns a Python value into a `Literal[...]` type.
        """
        self._literal_type = literal_type

    @propagates(None)
    def __call__(self, args: LiteralStringArgs, api: object) -> list[MypyType] | None:
        """Give the two type arguments of `Datetime[Unit, Zone]` for one call.

        A missing `time_unit` is `"us"`, the default of polars. A missing or `None`
        `time_zone` means no time zone, and gives the `None` type.

        Args:
            args: The arguments of the call.
            api: The mypy plugin API, which the literal types need.

        Returns:
            `[Unit, Zone]`. None if `time_unit` is not a string literal, or if `time_zone` is
            neither missing, `None` nor a string literal.
        """
        unit = not_none(args.literal_str("time_unit", default="us"))
        unit_type = self._literal_type.string(unit, api)
        if args.is_omitted_or_none("time_zone"):
            return [unit_type, NoneType()]
        zone = not_none(args.literal_str("time_zone"))
        return [unit_type, self._literal_type.string(zone, api)]


class _DurationConstructorReader:
    """Read the time unit of a `pl.Duration(...)` call."""

    MARKER_TARGET: Final[Target] = Target.of(_DurationMarker)
    PARAM_NAMES: Final[tuple[str, ...]] = ("time_unit",)

    def __init__(self, literal_type: LiteralTypeBuilder) -> None:
        """Keep the builder of literal types for the phase of mypy that uses this reader.

        Args:
            literal_type: The builder that turns a Python value into a `Literal[...]` type.
        """
        self._literal_type = literal_type

    @propagates(None)
    def __call__(self, args: DtypeConstructorArgs, api: object) -> list[MypyType] | None:
        """Give the type argument of `Duration[Unit]` for one call.

        A missing `time_unit` is `"us"`, the default of polars.

        Args:
            args: The arguments of the call.
            api: The mypy plugin API, which the literal type needs.

        Returns:
            `[Unit]`, or None if `time_unit` is not a string literal.
        """
        unit = not_none(args.literal_str("time_unit", default="us"))
        return [self._literal_type.string(unit, api)]


class _DecimalConstructorReader:
    """Read the precision and the scale of a `pl.Decimal(...)` call."""

    MARKER_TARGET: Final[Target] = Target.of(_DecimalMarker)
    PARAM_NAMES: Final[tuple[str, ...]] = ("precision", "scale")
    # If the precision is missing or None, polars sets it to 38. pl.Decimal(None, 2) and
    # pl.Decimal(38, 2) are therefore the same data type, and they must give the same marker.
    _IMPLICIT_PRECISION: Final[int] = 38

    def __init__(self, literal_type: LiteralTypeBuilder) -> None:
        """Keep the builder of literal types for the phase of mypy that uses this reader.

        Args:
            literal_type: The builder that turns a Python value into a `Literal[...]` type.
        """
        self._literal_type = literal_type

    @propagates(None)
    def __call__(self, args: DtypeConstructorArgs, api: object) -> list[MypyType] | None:
        """Give the two type arguments of `Decimal[Precision, Scale]` for one call.

        A missing `scale` is 0, the default of polars. A missing or `None` `precision` gives
        `Literal[38]`, because polars then uses 38.

        Args:
            args: The arguments of the call.
            api: The mypy plugin API, which the literal types need.

        Returns:
            `[Precision, Scale]`. None if `precision` is neither missing, `None` nor an integer
            literal, or if `scale` is not an integer literal.
        """
        precision = self._IMPLICIT_PRECISION
        if not args.is_omitted_or_none("precision"):
            precision = not_none(args.literal_int("precision"))
        scale = not_none(args.literal_int("scale", default=0))
        precision_type = self._literal_type.integer(precision, api)
        return [precision_type, self._literal_type.integer(scale, api)]


class _EnumConstructorReader:
    """Read the categories of a `pl.Enum(...)` call."""

    MARKER_TARGET: Final[Target] = Target.of(_EnumMarker)
    PARAM_NAMES: Final[tuple[str, ...]] = ("categories",)

    def __init__(self, literal_type: LiteralTypeBuilder) -> None:
        """Keep the builder of literal types for the phase of mypy that uses this reader.

        Args:
            literal_type: The builder that turns a Python value into a `Literal[...]` type.
        """
        self._literal_type = literal_type

    @propagates(None)
    def __call__(self, args: DtypeConstructorArgs, api: object) -> list[MypyType] | None:
        """Give the type arguments of `Enum[*Categories]` for one call, one for each category.

        The order of the categories is part of an enum, so the type arguments keep it.

        Args:
            args: The arguments of the call.
            api: The mypy plugin API, which the literal types need.

        Returns:
            One string literal type for each category, in order. None if `categories` is not a
            list literal of string literals.
        """
        categories = not_none(args.literal_str_list("categories"))
        return [self._literal_type.string(category, api) for category in categories]


class ParametrizedDtypeReaders:
    """Find the reader for a call to one of the four polars data types that have parameters.

    Code that meets a call such as `pl.Datetime("ms")` asks this class for the reader of the
    called class. The four readers share one `LiteralTypeBuilder`, which the code that builds
    this object picks for its phase of mypy.
    """

    def __init__(self, literal_type: LiteralTypeBuilder) -> None:
        """Build the four readers, keyed by the mypy full name of their polars class.

        Args:
            literal_type: The builder that turns a Python value into a `Literal[...]` type. All
                four readers use it.
        """
        self._by_target: Final[dict[Target, _ParametrizedDtypeReader]] = {
            Target.of(pl.Datetime): _DatetimeConstructorReader(literal_type),
            Target.of(pl.Duration): _DurationConstructorReader(literal_type),
            Target.of(pl.Decimal): _DecimalConstructorReader(literal_type),
            Target.of(pl.Enum): _EnumConstructorReader(literal_type),
        }

    def reader_for(self, callee: Expression) -> _ParametrizedDtypeReader | None:
        """Find the reader for the class that a call calls.

        Args:
            callee: The AST node of the thing that the call calls, for example `pl.Datetime`.

        Returns:
            The reader, or None if `callee` is not `pl.Datetime`, `pl.Duration`, `pl.Decimal`
            or `pl.Enum`.
        """
        fullname = getattr(callee, "fullname", None)
        if fullname is None:
            return None
        return self._by_target.get(Target(fullname))

    def targets_and_readers(self) -> ItemsView[Target, _ParametrizedDtypeReader]:
        """Give each of the four polars classes with its reader.

        Code that registers one mypy hook for each of these classes reads the list here. The
        four classes are then listed in one place only.

        Returns:
            Each `(Target, reader)` pair.

        Example:
            >>> fixture = new_type_fixture()
            >>> readers = ParametrizedDtypeReaders(CheckerLiteralTypeBuilder())
            >>> len(readers.targets_and_readers())
            4
        """
        return self._by_target.items()
