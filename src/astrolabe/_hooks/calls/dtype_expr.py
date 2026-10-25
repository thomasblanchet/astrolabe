"""Turn the type of an argument that names a polars data type into one form: an instance."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl
from mypy.types import CallableType, Instance, get_proper_type
from mypy.types import Type as MypyType

from astrolabe import frame
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeReads
from astrolabe._hooks.expression.kind import KindRoster

if TYPE_CHECKING:
    from mypy.nodes import Expression, TypeInfo

    from astrolabe._hooks.encoding.rendering_codec import ConstDtypeCodec


class DtypeExpr:
    """Read the polars data type that an argument names, from the type that mypy gives it.

    Many calls take a data type as an argument, for example `.cast(pl.Float64)`, a
    `schema={...}` dict or `pl.lit(1, dtype=pl.Int8)`. A user can write the class, `pl.Float64`,
    or an instance, `pl.Float64()`. mypy types the class as a callable type object and the
    instance as an instance. This class turns both into the instance form, so that code which
    reads a data type handles one form only.

    A data type with parameters, such as `pl.Datetime("ms")`, needs no special case here. The
    checker already gives such a call a type that holds its parameters, for example
    `Datetime[Literal["ms"], None]`. That type reaches this class unchanged, whether the call
    is in the argument or in a variable.

    A class with parameters, such as `pl.Datetime`, stands for one data type where a call
    takes one data type, for example in `.cast(...)` or in `dtype=`. polars then builds the
    instance with its default parameters: `pl.Datetime` is `Datetime("us", None)`. A selector
    such as `cs.by_dtype(pl.Datetime)` reads the class as every data type of its kind, so
    `resolve` keeps the class as its kind, and `rendering` gives the default. The class
    `pl.Decimal` has a default only in a frame schema and in a series, where it is
    `Decimal(38, 0)`. polars refuses it in an expression, so `rendering` keeps it as its kind.

    It sits in the plugin layer. It reads the type of an AST node through the mypy API.
    """

    # The rendering of the instance that polars builds for a class that has parameters.
    _DEFAULTS: ClassVar[dict[str, str]] = {
        Target.of(pl.Datetime).fullname: "Datetime(time_unit='us', time_zone=None)",
        Target.of(pl.Duration).fullname: "Duration(time_unit='us')",
        Target.of(pl.Enum).fullname: "Enum(categories=[])",
        Target.of(pl.List).fullname: "List(Null)",
        Target.of(pl.Array).fullname: "Array(Null, shape=(0,))",
        Target.of(pl.Struct).fullname: "Struct({})",
    }
    _DECIMAL: ClassVar[str] = Target.of(pl.Decimal).fullname
    _DECIMAL_DEFAULT: ClassVar[str] = "Decimal(precision=38, scale=0)"
    # The marker classes of this project that take type arguments in an annotation.
    _MARKERS: ClassVar[frozenset[str]] = frozenset({
        Target.of(frame.List).fullname,
        Target.of(frame.Array).fullname,
        Target.of(frame.Struct).fullname,
        Target.of(frame.Datetime).fullname,
        Target.of(frame.Duration).fullname,
        Target.of(frame.Decimal).fullname,
        Target.of(frame.Enum).fullname,
    })
    # The polars class that each marker class of this project subclasses or
    # stands for. polars reads the bare marker class as this class.
    _POLARS_CLASSES: ClassVar[dict[str, str]] = {
        Target.of(frame.List).fullname: Target.of(pl.List).fullname,
        Target.of(frame.Array).fullname: Target.of(pl.Array).fullname,
        Target.of(frame.Struct).fullname: Target.of(pl.Struct).fullname,
        Target.of(frame.Datetime).fullname: Target.of(pl.Datetime).fullname,
        Target.of(frame.Duration).fullname: Target.of(pl.Duration).fullname,
        Target.of(frame.Decimal).fullname: Target.of(pl.Decimal).fullname,
        Target.of(frame.Enum).fullname: Target.of(pl.Enum).fullname,
    }
    # The marker classes that polars refuses even with no type arguments.
    _BARE_REFUSED: ClassVar[frozenset[str]] = frozenset({
        Target.of(frame.Unchecked).fullname,
        Target.of(frame.Decimal).fullname,
    })
    _DATA_TYPE: ClassVar[str] = Target.of(pl.DataType).fullname
    # The module that defines the polars class of each kind.
    _POLARS_MODULE: ClassVar[str] = _DATA_TYPE.rsplit(".", maxsplit=1)[0]
    # The top-level packages whose subclasses of `pl.DataType` are not user code.
    _OWNERS: ClassVar[frozenset[str]] = frozenset({
        pl.DataType.__module__.split(".", maxsplit=1)[0],
        frame.__name__.split(".", maxsplit=1)[0],
    })
    # The polars classes whose subclasses `pl.Series` reads as no data type at all.
    _NESTED: ClassVar[frozenset[str]] = frozenset({
        Target.of(pl.List).fullname,
        Target.of(pl.Array).fullname,
        Target.of(pl.Struct).fullname,
    })
    # The kinds whose name polars refuses on a class of user code, as it
    # refuses the class `pl.Decimal`.
    _UNREAD_NAMES: ClassVar[frozenset[str]] = frozenset({"Decimal", "Extension"})

    def resolve(self, api: object, node: Expression) -> Instance | None:
        """Give the data type that one argument names, as an instance type.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            The data type as an instance, or None if the type of `node` is neither a class nor
            an instance.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = int_dtype(fixture)
            >>> DtypeExpr().resolve(api, node) is not None
            True
        """
        proper = get_proper_type(api.get_expression_type(node))  # type: ignore[attr-defined]
        if isinstance(proper, CallableType) and proper.is_type_obj():
            return Instance(proper.type_object(), [])
        if isinstance(proper, Instance):
            return proper
        return None

    def rendering(self, api: object, node: Expression) -> str | None:
        """Give the rendering of the one data type that an argument stands for.

        A class with parameters gives the instance that polars builds for it. The class
        `pl.Decimal` gives its kind.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            The rendering, or None if the argument names no data type.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = int_dtype(fixture)
            >>> DtypeExpr().rendering(api, node)
            'Int64'
        """
        if self._user_class(api, node) is not None:
            return self._user_rendering(api, node)
        default = self._DEFAULTS.get(self.class_name(api, node) or "")
        if default is not None:
            return default
        resolved = self.resolve(api, node)
        return None if resolved is None else ConstDtypeReads().read_type(resolved)

    def typed(self, api: object, node: Expression, codec: ConstDtypeCodec) -> MypyType | None:
        """Give the one data type that an argument stands for, as an instance type.

        It is `rendering` as a type: a class with parameters gives the instance that polars
        builds for it, and the class `pl.Decimal` gives its kind.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.
            codec: Builds the type of a rendering.

        Returns:
            The data type, or None if the argument names no data type.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = int_dtype(fixture)
            >>> DtypeExpr().typed(api, node, None)
            polars.datatypes.classes.Int64
        """
        if self._user_class(api, node) is not None:
            rendering = self._user_rendering(api, node)
            return None if rendering is None else codec.built(rendering)
        default = self._DEFAULTS.get(self.class_name(api, node) or "")
        if default is not None:
            return codec.built(default)
        return self.resolve(api, node)

    def schema_typed(
        self,
        api: object,
        node: Expression,
        codec: ConstDtypeCodec,
    ) -> MypyType | None:
        """Give the data type that one column of a frame schema or a series takes, as a type.

        It is `typed`, except that the class `pl.Decimal` gives `Decimal(38, 0)`.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.
            codec: Builds the type of a rendering.

        Returns:
            The data type, or None if the argument names no data type.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = int_dtype(fixture)
            >>> DtypeExpr().schema_typed(api, node, None)
            polars.datatypes.classes.Int64
        """
        if self.names_decimal_class(api, node):
            return codec.built(self._DECIMAL_DEFAULT)
        return self.typed(api, node, codec)

    def names_decimal_class(self, api: object, node: Expression) -> bool:
        """Answer whether an argument is the class `pl.Decimal` itself.

        polars refuses that class inside an expression and in a cast, because a `Decimal`
        there needs its precision. A frame schema and a series read it as
        `Decimal(38, 0)`.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            True for the class, and False for an instance or any other argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = int_dtype(fixture)
            >>> DtypeExpr().names_decimal_class(api, node)
            False
        """
        return self.class_name(api, node) == self._DECIMAL

    def refused_marker(self, api: object, node: Expression) -> str | None:
        """Name a marker class of this project where an argument must be a data type.

        A marker class, such as `List[pl.Int64]`, writes a data type in a
        type annotation. It is not a data type that polars can read. A
        marker with its type arguments is a generic alias at run time,
        `Unchecked` is no data type of polars, and the bare class
        `Decimal` has no precision. polars raises for each of them
        wherever a call takes one data type. A bare marker class with no
        type arguments, such as `List`, subclasses a polars class, and
        this method does not name it.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            The name of the class, with `[...]` where it has type
            arguments, or None for any other argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> listed = list_dtype(fixture, int_dtype(fixture))
            >>> api.expression_types[id(node)] = class_object(fixture, listed)
            >>> DtypeExpr().refused_marker(api, node)
            'List[...]'
            >>> api.expression_types[id(node)] = class_object(fixture, listed, generic=True)
            >>> DtypeExpr().refused_marker(api, node) is None
            True
            >>> api.expression_types[id(node)] = class_object(fixture, unchecked_instance(fixture))
            >>> DtypeExpr().refused_marker(api, node)
            'Unchecked'
            >>> api.expression_types[id(node)] = int_dtype(fixture)
            >>> DtypeExpr().refused_marker(api, node) is None
            True
        """
        with_arguments = self.marker_with_arguments(api, node)
        if with_arguments is not None:
            return with_arguments
        proper = self._class_object(api, node)
        if proper is None:
            return None
        fullname = proper.type_object().fullname
        if fullname in self._BARE_REFUSED:
            return fullname.rsplit(".", maxsplit=1)[-1]
        return None

    def marker_with_arguments(self, api: object, node: Expression) -> str | None:
        """Name a marker class of this project that an argument names with its type arguments.

        A marker with its type arguments, such as `List[pl.Int64]`, is a
        generic alias at run time. polars raises for it wherever it reads a
        data type, also inside `pl.List(...)`, where it accepts `Unchecked`
        until the data type is used.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            The name of the class with `[...]`, or None for any other
            argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> listed = list_dtype(fixture, int_dtype(fixture))
            >>> api.expression_types[id(node)] = class_object(fixture, listed)
            >>> DtypeExpr().marker_with_arguments(api, node)
            'List[...]'
            >>> api.expression_types[id(node)] = class_object(fixture, unchecked_instance(fixture))
            >>> DtypeExpr().marker_with_arguments(api, node) is None
            True
        """
        proper = self._class_object(api, node)
        if proper is None:
            return None
        fullname = proper.type_object().fullname
        if fullname not in self._MARKERS or proper.variables:
            return None
        return fullname.rsplit(".", maxsplit=1)[-1] + "[...]"

    def class_name(self, api: object, node: Expression) -> str | None:
        """Give the full name of the class that an argument names, if it is a class.

        A bare marker class of this project, such as `List` with no type
        arguments, gives the polars class that polars reads it as.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            The full name, or None for an instance or any other argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = int_dtype(fixture)
            >>> DtypeExpr().class_name(api, node) is None
            True
            >>> listed = list_dtype(fixture, int_dtype(fixture))
            >>> api.expression_types[id(node)] = class_object(fixture, listed, generic=True)
            >>> DtypeExpr().class_name(api, node)
            'polars.datatypes.classes.List'
        """
        proper = self._class_object(api, node)
        if proper is None:
            return None
        fullname = proper.type_object().fullname
        if proper.variables:
            return self._POLARS_CLASSES.get(fullname, fullname)
        return fullname

    def bare_marker(self, api: object, node: Expression) -> str | None:
        """Name a marker class of this project that an argument names with no type arguments.

        polars reads a bare marker class, such as `List`, as the polars class
        of the same name in most calls. `pl.Series` refuses it for every
        input, so code that checks a series asks this method.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            The short name of the class, or None for any other argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> listed = list_dtype(fixture, int_dtype(fixture))
            >>> api.expression_types[id(node)] = class_object(fixture, listed, generic=True)
            >>> DtypeExpr().bare_marker(api, node)
            'List'
            >>> api.expression_types[id(node)] = class_object(fixture, listed)
            >>> DtypeExpr().bare_marker(api, node) is None
            True
        """
        proper = self._class_object(api, node)
        if proper is None or not proper.variables:
            return None
        fullname = proper.type_object().fullname
        if fullname not in self._POLARS_CLASSES:
            return None
        return fullname.rsplit(".", maxsplit=1)[-1]

    def names_data_type(self, api: object, node: Expression) -> bool:
        """Answer whether an argument is a data type class or a data type instance.

        mypy reports an argument of another class where a call takes a data
        type. The class then names no data type, and code that reads one
        gives up on it.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            True for a subclass of `pl.DataType` or an instance of one.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = user_dtype(fixture, "Tagged")
            >>> DtypeExpr().names_data_type(api, node)
            True
            >>> api.expression_types[id(node)] = fixture.o  # the class `object`
            >>> DtypeExpr().names_data_type(api, node)
            False
        """
        resolved = self.resolve(api, node)
        return resolved is not None and resolved.type.has_base(self._DATA_TYPE)

    def unread_user_class(self, api: object, node: Expression) -> str | None:
        """Name a class of user code that polars reads as no data type, where a call takes one.

        User code can subclass a polars data type, for example
        `class Tagged(pl.Int64)`. In a cast, a `dtype=`, a `return_dtype=`
        or a nested data type, polars reads such a class, or an instance of
        it, by its name alone. A class named after a kind of polars, such
        as `Int64`, reads as the polars class of that name. polars raises
        for any other name, and for the names `Decimal` and `Extension`.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            The short name of the class, or None for any other argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = user_dtype(fixture, "Tagged")
            >>> DtypeExpr().unread_user_class(api, node)
            'Tagged'
            >>> api.expression_types[id(node)] = user_dtype(fixture, "Int64")
            >>> DtypeExpr().unread_user_class(api, node) is None
            True
            >>> api.expression_types[id(node)] = int_dtype(fixture)
            >>> DtypeExpr().unread_user_class(api, node) is None
            True
        """
        info = self._user_class(api, node)
        if info is None:
            return None
        name = info.fullname.rsplit(".", maxsplit=1)[-1]
        if name in KindRoster().names() and name not in self._UNREAD_NAMES:
            return None
        return name

    def scalar_user_class(self, api: object, node: Expression) -> str | None:
        """Name a class of user code that subclasses a polars data type other than a nested one.

        `pl.Series` refuses such a class, whatever its name and whatever
        the data. A nested data type is `List`, `Array` or `Struct`.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            The short name of the class, or None for any other argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = user_dtype(fixture, "Int64")
            >>> DtypeExpr().scalar_user_class(api, node)
            'Int64'
            >>> api.expression_types[id(node)] = user_dtype(fixture, "Rows", "List")
            >>> DtypeExpr().scalar_user_class(api, node) is None
            True
        """
        info = self._user_class(api, node)
        if info is None or self._nested(info):
            return None
        return info.fullname.rsplit(".", maxsplit=1)[-1]

    def nested_user_class(self, api: object, node: Expression) -> bool:
        """Answer whether an argument is a class of user code that subclasses a nested data type.

        `pl.Series` reads no data type from such a class: the values decide
        the data type, and polars raises for nested values. A nested data
        type is `List`, `Array` or `Struct`.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            True for a subclass of user code of `pl.List`, `pl.Array` or
            `pl.Struct`, or an instance of one.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = user_dtype(fixture, "Rows", "List")
            >>> DtypeExpr().nested_user_class(api, node)
            True
            >>> api.expression_types[id(node)] = user_dtype(fixture, "Tagged")
            >>> DtypeExpr().nested_user_class(api, node)
            False
        """
        info = self._user_class(api, node)
        return info is not None and self._nested(info)

    def _user_class(self, api: object, node: Expression) -> TypeInfo | None:
        """Give the class of an argument if user code defines it as a subclass of a data type.

        A class that polars or this project defines is not user code.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            The class, or None for any other argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = unchecked_instance(fixture)
            >>> DtypeExpr()._user_class(api, node) is None
            True
        """
        resolved = self.resolve(api, node)
        if resolved is None:
            return None
        info = resolved.type
        if info.module_name.split(".", maxsplit=1)[0] in self._OWNERS:
            return None
        if not info.has_base(self._DATA_TYPE):
            return None
        return info

    def _user_rendering(self, api: object, node: Expression) -> str | None:
        """Give the rendering of the data type that polars reads from a class of user code.

        polars reads such a class by its name, as the polars class of that
        name, so a class gives the default instance of that polars class.
        An instance of a class whose polars class has parameters, such as
        `Datetime`, holds parameters that its type does not show.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument, a class of user code.

        Returns:
            The rendering, or None where polars reads no data type from the
            name, or where the parameters are not known.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> stamp = user_dtype(fixture, "Datetime", "Datetime")
            >>> api.expression_types[id(node)] = class_object(fixture, stamp)
            >>> DtypeExpr()._user_rendering(api, node)
            "Datetime(time_unit='us', time_zone=None)"
            >>> api.expression_types[id(node)] = stamp
            >>> DtypeExpr()._user_rendering(api, node) is None
            True
            >>> api.expression_types[id(node)] = user_dtype(fixture, "Int64")
            >>> DtypeExpr()._user_rendering(api, node)
            'Int64'
            >>> api.expression_types[id(node)] = user_dtype(fixture, "Tagged")
            >>> DtypeExpr()._user_rendering(api, node) is None
            True
        """
        if (  # pragma: no cover -- each caller reports a user class first
            self.unread_user_class(api, node) is not None
        ):
            return None
        name = checked_not_none(self.resolve(api, node)).type.fullname.rsplit(".", maxsplit=1)[-1]
        default = self._DEFAULTS.get(f"{self._POLARS_MODULE}.{name}")
        if default is None:
            return name
        if self._class_object(api, node) is None:
            return None
        return default

    def _nested(self, info: TypeInfo) -> bool:
        """Answer whether a class subclasses `pl.List`, `pl.Array` or `pl.Struct`.

        Args:
            info: The class.

        Returns:
            True if one of these classes is in its MRO.

        Example:
            >>> DtypeExpr()._nested(user_dtype(new_type_fixture(), "Rows", "Array").type)
            True
        """
        return any(base.fullname in self._NESTED for base in info.mro)

    def _class_object(self, api: object, node: Expression) -> CallableType | None:
        """Give the type of an argument if the argument is a class, and not an instance.

        mypy types a class in a value position as a callable that builds an
        instance. A generic class with no type arguments keeps its type
        variables on that callable.

        Args:
            api: The mypy plugin API, which gives the type of `node`.
            node: The AST node of the argument.

        Returns:
            The callable type of the class, or None for any other argument.

        Example:
            >>> fixture = new_type_fixture()
            >>> api = new_api(fixture)
            >>> node = object()
            >>> api.expression_types[id(node)] = int_dtype(fixture)
            >>> DtypeExpr()._class_object(api, node) is None
            True
        """
        proper = get_proper_type(api.get_expression_type(node))  # type: ignore[attr-defined]
        if isinstance(proper, CallableType) and proper.is_type_obj():
            return proper
        return None
