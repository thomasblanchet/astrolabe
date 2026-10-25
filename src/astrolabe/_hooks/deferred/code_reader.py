"""Read the code of a deferred type back into a polars expression, without executing its text.

The evaluator needs the tree that polars builds for an expression: polars
decides which arguments are inputs, and how it rewrites a method into other
nodes. So this module builds the expression again from its code. It parses the
text with `ast.parse`, which runs nothing, and interprets the syntax tree with
a small interpreter that accepts only the forms that the recorder writes: the
names `pl` and `cs`, attribute access and calls, constants and containers,
operators, shared parts and references. Each call goes to the real method of
polars, which builds an expression and runs no data. Anything else is a fault
of the recorder, and the reader refuses it.

A declared type of the code becomes a column with a reserved name, so that the
evaluator finds it as a leaf of the tree. A hole, which is a value that the
checker cannot know, becomes a placeholder value of its kind. The evaluator
then recognizes the placeholder wherever polars puts it. A function of the
user cannot reach a tree, so a call that takes one becomes a column with a
reserved name, and the reader keeps the call apart (`OpaqueCall`).
"""

from __future__ import annotations

import ast
import datetime
import decimal
import inspect
import json
import operator
import re
import tokenize
import warnings
import zoneinfo
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import polars.selectors as cs

from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.deferred.code import CodeRefs
from astrolabe._rules.tree import NonFinite

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from astrolabe._hooks.deferred.code import Code


class CodeReadError(ValueError):
    """Say that the reader cannot build the expression of a code.

    The code holds a form that the recorder does not write, or polars refuses
    the call when it builds the expression.

    Example:
        >>> str(CodeReadError("the name 'eval' is not allowed"))
        "the name 'eval' is not allowed"
    """


class ArgumentMismatch(CodeReadError):
    """Say that the arguments of a call of the code do not fit the signature of its function.

    mypy reports such a call itself, so the call is no refusal of polars.

    Example:
        >>> isinstance(ArgumentMismatch("x"), CodeReadError)
        True
    """


class BuildRefusal(CodeReadError):
    """Say that polars refuses a call while it builds the expression, whatever the data.

    The call raises one of the errors of polars before any frame meets it,
    as `pl.lit` of an integer that no data type holds. `holes` gives the kind
    of each hole that the reader met before the call, so that a reader of the
    refusal can try other values of those holes.

    Example:
        >>> refusal = BuildRefusal("InvalidOperationError", "integer too large for Polars")
        >>> refusal.error, str(refusal)
        ('InvalidOperationError', 'integer too large for Polars')
    """

    def __init__(self, error: str, message: str, holes: tuple[str, ...] = ()) -> None:
        """Keep the class of the error of polars, its message and the holes before the call.

        Args:
            error: The name of the class of the error.
            message: The first line of its message.
            holes: The kind of each hole that the reader met before the call.

        Example:
            >>> BuildRefusal("ComputeError", "x", ("str",)).holes
            ('str',)
        """
        super().__init__(message)
        self.error = error
        self.holes = holes


class RunRefusal(CodeReadError):
    """Say that polars builds an expression and refuses it at every frame, before any tree exists.

    The rules read the tree that polars writes for an expression. For some
    expressions polars cannot write that tree, and refuses them whenever a
    frame runs them. The reader then states the refusal itself.

    Example:
        >>> refusal = RunRefusal("InvalidOperationError", "no objects", "a")
        >>> refusal.error, refusal.name, str(refusal)
        ('InvalidOperationError', 'a', 'no objects')
    """

    def __init__(self, error: str, message: str, name: str | None) -> None:
        """Keep the class of the error of polars, its message and the output name.

        Args:
            error: The name of the class of the error.
            message: Its message.
            name: The output name of the expression, or `None` where polars gives none.

        Example:
            >>> RunRefusal("ComputeError", "x", None).name is None
            True
        """
        super().__init__(message)
        self.error = error
        self.name = name


class UnwrittenTree(CodeReadError):
    """Say that polars builds an expression but cannot write its tree, so no rule can read it.

    Polars builds some functions around a function of Python of its own, as
    `pl.cum_sum_horizontal`, and it cannot write the tree of such a node. It
    still names the output of the expression and the columns that it reads,
    so the reader keeps the expression for them.

    Example:
        >>> unwritten = UnwrittenTree("no tree", pl.cum_sum_horizontal("a", "b"))
        >>> unwritten.expression.meta.output_name()
        'cum_sum'
    """

    def __init__(self, message: str, expression: pl.Expr) -> None:
        """Keep the expression.

        Args:
            message: Why polars cannot write the tree.
            expression: The expression.

        Example:
            >>> UnwrittenTree("x", pl.col("a")).expression.meta.output_name()
            'a'
        """
        super().__init__(message)
        self.expression = expression


class ObjectValues:
    """Find a call that polars builds over values of the `Object` type and refuses at every frame.

    `replace` turns a list or a tuple of old values or of new values into a
    series. A list of expressions becomes a series of the `Object` type.
    Polars builds the call, but it cannot write the tree of such a series,
    and `replace` refuses the series when a frame runs the call.

    Example:
        >>> ObjectValues().refusal(pl.col("a").replace, [[1], [pl.lit(2)]], {})
        '`replace` does not support `old`/`new` values of object dtype'
    """

    ERROR: ClassVar[str] = "InvalidOperationError"
    REFUSED: ClassVar[dict[str, str]] = {
        "replace": "`replace` does not support `old`/`new` values of object dtype",
    }

    def refusal(
        self,
        function: Callable[..., object],
        positional: Sequence[object],
        keywords: Mapping[str, object],
    ) -> str | None:
        """Give the message of polars for a call that takes values of the `Object` type.

        Args:
            function: The function of the call.
            positional: The positional arguments.
            keywords: The keyword arguments.

        Returns:
            The message, or `None` where polars does not refuse the values.

        Example:
            >>> ObjectValues().refusal(pl.col("a").replace, [[1], [2]], {}) is None
            True
            >>> ObjectValues().refusal(pl.col("a").shift, [[pl.lit(2)]], {}) is None
            True
        """
        if not inspect.ismethod(function) or not isinstance(function.__self__, pl.Expr):
            return None
        message = self.REFUSED.get(function.__name__)
        if message is None:
            return None
        values = [*positional, *keywords.values()]
        return message if any(self._objects(value) for value in values) else None

    def _objects(self, value: object) -> bool:
        """Say whether polars turns a value into a series of the `Object` type.

        Polars gives that type to a list or a tuple of expressions.

        Args:
            value: The value of an argument.

        Returns:
            Whether it does.

        Example:
            >>> ObjectValues()._objects((pl.lit(1),)), ObjectValues()._objects([])
            (True, False)
        """
        if not isinstance(value, list | tuple) or not value:
            return False
        return all(isinstance(item, pl.Expr) for item in value)


class PatternValues:
    """Find a selector whose pattern polars cannot compile, which it refuses at every frame.

    `cs.matches`, a name of `pl.col` that starts with `^` and ends with `$`,
    and a name of `exclude` of that form read a text as a regular expression.
    Polars builds the call, and compiles the text only where a frame expands
    the selector. So for a text that is no regular expression of its dialect,
    as `^a[`, or a look-around, it raises `InvalidOperationError` at every
    frame, a frame with no column too.

    Example:
        >>> PatternValues().refusal(cs.matches, ["^a["], {})
        "invalid regex in selector '^a['"
        >>> PatternValues().refusal(pl.col, ["^a[$", "b"], {})
        "invalid regex in selector '^a[$'"
        >>> PatternValues().refusal(pl.col, ["a["], {}) is None
        True
    """

    def refusal(
        self,
        function: Callable[..., object],
        positional: Sequence[object],
        keywords: Mapping[str, object],
    ) -> str | None:
        """Give the message of polars for a call with a pattern that it cannot compile.

        Args:
            function: The function of the call.
            positional: The positional arguments.
            keywords: The keyword arguments.

        Returns:
            The message, or `None` where each pattern of the call compiles.

        Example:
            >>> PatternValues().refusal(cs.matches, ["^a"], {}) is None
            True
        """
        for pattern in self._patterns(function, [*positional, *keywords.values()]):
            try:
                cs.expand_selector({}, cs.matches(pattern))
            except pl.exceptions.InvalidOperationError as error:
                return str(error).splitlines()[0]
        return None

    def _patterns(self, function: Callable[..., object], values: Sequence[object]) -> list[str]:
        """Give the texts that a call reads as regular expressions.

        Args:
            function: The function of the call.
            values: The values of its arguments.

        Returns:
            The texts.

        Example:
            >>> PatternValues()._patterns(pl.all().exclude, [["^a$", "b"]])
            ['^a$']
            >>> PatternValues()._patterns(pl.lit, ["^a$"])
            []
        """
        texts = [
            text
            for value in values
            for text in (value if isinstance(value, list | tuple) else [value])
            if isinstance(text, str)
        ]
        if function is cs.matches:
            return texts
        named = function is pl.col or getattr(function, "__name__", None) == "exclude"
        anchored = [text for text in texts if text.startswith("^") and text.endswith("$")]
        return anchored if named else []


class HoleValues:
    """Give the placeholder value of a hole of each kind, and recognize it.

    A hole of a number or of a text takes a fixed value that the user code
    does not hold in practice, so the evaluator recognizes it in the tree. A
    hole of the kind `number` is a value of the type `float`, which can hold
    an `int` too, so the evaluator tries integers for it as well. A
    hole of a `bool` takes each of its two values in turn, so the evaluator
    reads both answers. A hole of a time or a duration takes a value of its
    type, because the value does not change the data type. A hole of a date,
    a moment or a decimal takes chosen values too: a variable of the type
    `date` can hold a moment, and the zone of a moment and the scale of a
    decimal change the data type. A hole of an expression becomes a column
    with a reserved name. A hole of a data type whose parameters the checker does not
    know, as `dtype:Datetime`, takes each data type that its rendering stands
    for, and `Null` where the evaluator chose none. A hole of a value from a
    known set, as `hole('one of', 'average', 'dense')`, takes each value of
    the set.

    Example:
        >>> HoleValues().in_text("pl.col('a').shift(hole('int'))")
        True
    """

    INT: ClassVar[int] = 2**62 - 57
    FLOAT: ClassVar[float] = 2.0**-1000
    STR: ClassVar[str] = "\x00hole\x00"
    KINDS: ClassVar[tuple[str, ...]] = (
        "int",
        "float",
        "number",
        "str",
        "bool",
        "date",
        "day",
        "time",
        "timedelta",
        "datetime",
        "decimal",
        "expr",
        "function",
        "one of",
        "value",
        "any",
    )
    DAY: ClassVar[datetime.date] = datetime.date(1970, 1, 1)
    TEMPORAL: ClassVar[dict[str, object]] = {
        "time": datetime.time(0, 0),
        "timedelta": datetime.timedelta(0),
    }
    MOMENT: ClassVar[datetime.datetime] = datetime.datetime(1970, 1, 1)
    DECIMAL: ClassVar[decimal.Decimal] = decimal.Decimal(0)
    DTYPE: ClassVar[str] = "dtype:"
    VALUE_HOLE: ClassVar[re.Pattern[str]] = re.compile(r"hole\('(?!function')")
    ANY_HOLE: ClassVar[re.Pattern[str]] = re.compile(r"hole\('")
    EXPR_COLUMN: ClassVar[str] = "__hole_{}__"
    EXPR_NAME: ClassVar[re.Pattern[str]] = re.compile(r"__hole_\d+__")
    ONE_OF: ClassVar[str] = "one of"

    def of(self, kind: str, number: int, choice: object) -> object:
        """Give the placeholder value of a hole, or the value that the evaluator chose for it.

        The evaluator can try a hole of a `bool`, an `int`, a `float` or a
        `str` at chosen values, to learn whether its value decides the answer.

        Args:
            kind: The kind of the hole, one of `KINDS`.
            number: The place of the hole in the code, counted from 0.
            choice: The value that the evaluator chose for the hole, or `None`.

        Returns:
            The value.

        Raises:
            CodeReadError: The kind of the hole has no placeholder value.

        Example:
            >>> HoleValues().of("str", 0, None) == HoleValues.STR
            True
            >>> HoleValues().of("int", 0, 300)
            300
            >>> HoleValues().of("expr", 2, None).meta.output_name()
            '__hole_2__'
            >>> HoleValues().of("dtype:Datetime", 0, None)
            Null
            >>> HoleValues().of(HoleValues().listing(("average", "dense")), 0, None)
            'average'
        """
        listed = self.listed(kind)
        if listed is not None:
            return listed[0] if choice is None else choice
        chosen = {
            "int": self.INT,
            "float": self.FLOAT,
            "number": self.FLOAT,
            "str": self.STR,
            "datetime": self.MOMENT,
            "decimal": self.DECIMAL,
            "date": self.DAY,
            "day": self.DAY,
        }
        if choice is not None and kind in chosen:
            return choice
        if kind.startswith(self.DTYPE):
            return pl.Null() if choice is None else choice
        values: dict[str, object] = {
            **chosen,
            "value": None,
            "any": None,
            "bool": bool(choice),
            "expr": pl.col(self.EXPR_COLUMN.format(number)),
            "function": OpaqueFunction(),
            **self.TEMPORAL,
        }
        if kind not in values:
            message = f"a hole of the kind {kind!r} has no placeholder value"
            raise CodeReadError(message)
        return values[kind]

    def listing(self, values: Sequence[object]) -> str:
        """Give the kind of a hole of a value from a known set, with the values in it.

        A value whose type is a union of literal values, as a variable of the
        type `Literal["average", "dense"]`, is one value of that set. The code
        writes its hole as `hole('one of', 'average', 'dense')`. The kind holds
        the values in JSON, so that the evaluator can try each one.

        Args:
            values: The values of the set.

        Returns:
            The kind.

        Raises:
            CodeReadError: A value is not a number, a text or a `bool`.

        Example:
            >>> HoleValues().listing(("average", 2))
            'one of:["average", 2]'
            >>> try:
            ...     HoleValues().listing((None,))
            ... except CodeReadError as error:
            ...     print(error)
            a hole of a known set takes only numbers, texts and bools
        """
        if not all(isinstance(value, int | str | float) for value in values):
            message = "a hole of a known set takes only numbers, texts and bools"
            raise CodeReadError(message)
        return f"{self.ONE_OF}:{json.dumps(list(values))}"

    def listed(self, kind: str) -> tuple[object, ...] | None:
        """Give the values that a hole of a value from a known set can take.

        Args:
            kind: The kind of the hole.

        Returns:
            The values, or `None` where the hole is not of a known set.

        Example:
            >>> HoleValues().listed(HoleValues().listing(("average", "dense")))
            ('average', 'dense')
            >>> HoleValues().listed("int") is None
            True
        """
        prefix = f"{self.ONE_OF}:"
        if not kind.startswith(prefix):
            return None
        values: list[object] = json.loads(kind.removeprefix(prefix))
        return tuple(values)

    def in_text(self, text: str) -> bool:
        """Say whether the text of a code holds a hole of a value, which polars can refuse.

        A hole of a function is not a value: the reader keeps its call apart.

        Args:
            text: The text of the code.

        Returns:
            Whether it does.

        Example:
            >>> HoleValues().in_text("pl.col('a').shift(hole('int'))")
            True
            >>> HoleValues().in_text("pl.col('a').map_batches(hole('function'))")
            False
        """
        return self.VALUE_HOLE.search(text) is not None

    def held(self, value: object) -> bool:
        """Say whether a value holds the placeholder of a text anywhere, as a selector does.

        A name or a pattern of a selector that comes from a variable of the
        type `str` is the placeholder, which picks no column. The data
        decides the name, so code that expands the selector must not read it.

        Args:
            value: A value, as `json.loads` gives the tree of an expression.

        Returns:
            Whether a text of the value holds the placeholder.

        Example:
            >>> HoleValues().held({"Matches": HoleValues.STR}), HoleValues().held({"Matches": "^a"})
            (True, False)
        """
        if isinstance(value, str):
            return self.STR in value
        if isinstance(value, dict):
            return any(self.held(item) for item in value.values())
        return isinstance(value, list) and any(self.held(item) for item in value)

    def any_in_text(self, text: str) -> bool:
        """Say whether the text of a code holds a hole of any kind, a function too.

        The recorder writes a callable of the user, as the class `int`, as a
        hole of a function. Polars can read its placeholder as a value: in
        `cs.by_dtype([int])`, as a data type, which it refuses.

        Args:
            text: The text of the code.

        Returns:
            Whether it does.

        Example:
            >>> HoleValues().any_in_text("cs.by_dtype([hole('function')])")
            True
            >>> HoleValues().any_in_text("pl.col('a').abs()")
            False
        """
        return self.ANY_HOLE.search(text) is not None


class OpaqueFunction:
    """Stand for a function of the user, which a call of polars receives.

    Example:
        >>> isinstance(OpaqueFunction(), OpaqueFunction)
        True
    """


@dataclass(frozen=True)
class OpaqueCall:
    """Hold a call that takes a function of the user, which no tree can hold.

    `receiver` is the expression of the call, `method` its name, and
    `keywords` its keyword arguments, such as `return_dtype=`. The evaluator
    gives the type that `return_dtype=` names, or `Unchecked`. `inputs` holds
    the expressions of a function of `pl` as the code writes them, where the
    receiver gathers them into a struct of its own.

    Example:
        >>> call = OpaqueCall(pl.col("a"), "map_batches", {"return_dtype": pl.Int64})
        >>> call.keywords["return_dtype"]
        Int64
    """

    receiver: pl.Expr
    method: str
    keywords: Mapping[str, object] = field(default_factory=dict)
    inputs: tuple[pl.Expr, ...] = ()

    def written(self) -> tuple[pl.Expr, ...]:
        """Give the expressions that the call reads, as the code writes them.

        Code that reads the tree of the code needs them: the receiver of a
        method, or the inputs of a function of `pl`, without the struct and
        the names that the reader gives them.

        Returns:
            The expressions.

        Example:
            >>> [str(e) for e in OpaqueCall(pl.col("a"), "map_batches").written()]
            ['col("a")']
            >>> gathered = OpaqueCall(pl.struct("a"), "fold", inputs=(pl.col("b"),))
            >>> [str(e) for e in gathered.written()]
            ['col("b")']
        """
        return self.inputs or (self.receiver,)


class GatheredCall:
    """Read a function of `pl` that runs a function of the user over several expressions.

    `pl.fold`, `pl.reduce`, `pl.cum_fold`, `pl.cum_reduce`, `pl.map_batches`
    and `pl.map_groups` take a list of expressions and a function of the user,
    which no tree can hold. The reader keeps such a call as an opaque call
    over a struct of its inputs. The struct reads each input, and polars
    names it after its first input, as it names the call: the accumulator of
    a fold, and the first expression of the others. `cum_fold` and
    `cum_reduce` give a struct of one field for each expression, each of the
    type that `return_dtype=` names, under the name of the function. With
    `include_init=True`, the run gives the accumulator its own type in a
    field of its own, so `return_dtype=` names no type of the result.

    Example:
        >>> call = GatheredCall().of(pl.reduce, [OpaqueFunction(), ["a", "b"]], {})
        >>> call.receiver.meta.output_name(), call.method
        ('a', 'reduce')
    """

    FUNCTIONS: ClassVar[dict[str, Callable[..., pl.Expr]]] = {
        "fold": pl.fold,
        "reduce": pl.reduce,
        "cum_fold": pl.cum_fold,
        "cum_reduce": pl.cum_reduce,
        "map_batches": pl.map_batches,
        "map_groups": pl.map_groups,
    }
    KEYWORDS: ClassVar[tuple[str, ...]] = ("return_dtype", "returns_scalar", "is_elementwise")

    def of(
        self,
        function: object,
        positional: Sequence[object],
        keywords: Mapping[str, object],
    ) -> OpaqueCall | None:
        """Give the opaque call of a function of `pl` over several expressions.

        Args:
            function: The function.
            positional: The positional arguments.
            keywords: The keyword arguments.

        Returns:
            The call, or `None` for another function.

        Example:
            >>> GatheredCall().of(pl.col, ["a"], {}) is None
            True
        """
        names = [name for name, known in self.FUNCTIONS.items() if known is function]
        if not names:
            return None
        name = names[0]
        signature = inspect.signature(self.FUNCTIONS[name])
        bound = signature.bind(*positional, **keywords).arguments
        fields = self._listed(bound["exprs"])
        inputs = [*self._listed(bound["acc"]), *fields] if "acc" in bound else fields
        receiver = self._struct(inputs)
        options = {key: bound[key] for key in self.KEYWORDS if key in bound}
        if bound.get("include_init") is True:
            options.pop("return_dtype", None)
        if name.startswith("cum_"):
            receiver = receiver.alias(name)
            options = self._fielded(options, fields)
        return OpaqueCall(receiver, name, options, tuple(inputs))

    def _listed(self, value: object) -> list[pl.Expr]:
        """Give the expressions of an argument: one expression, a column name, or a list of them.

        Args:
            value: The argument.

        Returns:
            The expressions.

        Example:
            >>> [str(expression) for expression in GatheredCall()._listed(["a", pl.lit(1)])]
            ['col("a")', 'dyn int: 1']
        """
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        return [self._expression(item) for item in values]

    def _expression(self, item: object) -> pl.Expr:
        """Give the expression of one item of an argument, as polars reads it.

        Args:
            item: An expression, a column name, or a value.

        Returns:
            The expression: a column for a name, and a literal for a value.

        Example:
            >>> str(GatheredCall()._expression("a"))
            'col("a")'
            >>> str(GatheredCall()._expression(1))
            'dyn int: 1'
        """
        if isinstance(item, pl.Expr):
            return item
        if isinstance(item, str):
            return pl.col(item)
        return pl.lit(item)

    def _struct(self, inputs: Sequence[pl.Expr]) -> pl.Expr:
        """Give a struct that reads every input and takes the name of the first.

        Args:
            inputs: The inputs.

        Returns:
            The struct.

        Example:
            >>> GatheredCall()._struct([pl.col("a"), pl.col("a")]).meta.output_name()
            'a'
        """
        others = [item.alias(f"__input_{place}__") for place, item in enumerate(inputs[1:])]
        return pl.struct(inputs[0], *others)

    def _fielded(
        self,
        options: Mapping[str, object],
        fields: Sequence[pl.Expr],
    ) -> dict[str, object]:
        """Give the options of a cumulative fold, whose type is a struct of a field for each input.

        Args:
            options: The options of the call.
            fields: The inputs that give a field.

        Returns:
            The options, with the struct in place of `return_dtype=`.

        Example:
            >>> options = GatheredCall()._fielded({"return_dtype": pl.Int8}, [pl.col("a")])
            >>> options["return_dtype"]
            Struct({'a': Int8})
        """
        named = options.get("return_dtype")
        if not isinstance(named, (pl.DataType, type)):
            return dict(options)
        dtype = checked_cast(named() if isinstance(named, type) else named, pl.DataType)
        names = [item.meta.output_name() for item in fields]
        struct = pl.Struct(dict.fromkeys(names, dtype))
        return {**options, "return_dtype": struct}


@dataclass(frozen=True)
class ReadCode:
    """Hold the expression that the reader built from a code, and what it met on the way.

    `holes` gives the kind of each hole, in the order of the text. `opaque`
    gives the call that each reserved column `__opaque_n__` stands for.
    `choice` gives the values that the reader took for the holes, by their
    places, where they are not the placeholders.

    Example:
        >>> ReadCode(pl.col("a")).holes
        ()
    """

    expression: object
    holes: tuple[str, ...] = ()
    opaque: tuple[OpaqueCall, ...] = ()
    choice: Mapping[int, object] = field(default_factory=dict)


class CodeReader:
    """Build the expression of a code, with placeholders for its declared types and its holes.

    The declared type at position `n` becomes the column `__leaf_n__`.

    Example:
        >>> from astrolabe._hooks.deferred.code import Code
        >>> read = CodeReader().read(Code("$0 * 0.9", ("Price",)))
        >>> str(read.expression)
        '[(col("__leaf_0__")) * (dyn float: 0.9)]'
    """

    def read(self, code: Code[object], choices: Mapping[int, object] | None = None) -> ReadCode:
        """Give the expression of a code.

        Args:
            code: The code.
            choices: The value that the evaluator chose for each hole, by
                the place of the hole. A hole with no choice takes its
                placeholder value.

        Returns:
            The expression, with the holes and the opaque calls that it met.

        Raises:
            CodeReadError: The code holds a form that the recorder does not
                write, or polars refuses a call.
            BuildRefusal: Polars refuses a call while it builds the expression.
            RunRefusal: Polars refuses the expression at every frame.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> CodeReader().read(Code("pl.col('a').shift(hole('int'))")).holes
            ('int',)
            >>> CodeReader().read(Code("pl.col('a').replace([1], [pl.lit(2)])"))
            Traceback (most recent call last):
            ...
            astrolabe._hooks.deferred.code_reader.RunRefusal: `replace` does not support ...
        """
        try:
            parsed = ast.parse(CodeRefs().parseable(code.text), mode="eval").body
        except (SyntaxError, tokenize.TokenError) as error:
            raise CodeReadError(str(error)) from error
        interpreter = _Interpreter(len(code.leaves), choices or {})
        with warnings.catch_warnings():
            # Polars warns about some calls, such as a comparison with None, where it builds them.
            warnings.simplefilter("ignore")
            try:
                expression = interpreter.visit(parsed)
            except BuildRefusal as refusal:
                holes = tuple(interpreter.holes)
                raise BuildRefusal(refusal.error, str(refusal), holes) from refusal
        if isinstance(expression, pl.Expr) and interpreter.refusals:
            name = expression.meta.output_name(raise_if_undetermined=False)
            raise RunRefusal(ObjectValues.ERROR, interpreter.refusals[0], name)
        if isinstance(expression, pl.Expr):
            self._written(expression)
        holes, opaque = tuple(interpreter.holes), tuple(interpreter.opaque)
        return ReadCode(expression, holes, opaque, dict(choices or {}))

    def _written(self, expression: pl.Expr) -> None:
        """Check that polars can write the tree of an expression, which every later step reads.

        Polars builds some functions, as `pl.cum_sum_horizontal`, around a
        function of Python, and it cannot write the tree of such a node.

        `pl.Expr()` with no argument builds an expression that holds no tree
        at all, and each method of it raises `AttributeError`. It has no name
        and reads no column, so the reader cannot keep it as a tree that
        polars cannot write: it cannot read the code at all.

        Args:
            expression: The expression.

        Raises:
            UnwrittenTree: Polars cannot write the tree.
            CodeReadError: The expression holds no tree.

        Example:
            >>> CodeReader()._written(pl.cum_sum_horizontal("a", "b"))
            Traceback (most recent call last):
            ...
            astrolabe._hooks.deferred.code_reader.UnwrittenTree: polars cannot write the tree ...
            >>> CodeReader()._written(pl.Expr())
            Traceback (most recent call last):
            ...
            astrolabe._hooks.deferred.code_reader.CodeReadError: pl.Expr() holds no tree...
        """
        try:
            expression.meta.serialize(format="json")
        except pl.exceptions.PolarsError as error:
            message = f"polars cannot write the tree of the expression: {error}"
            raise UnwrittenTree(message, expression) from error
        except AttributeError as error:
            message = "pl.Expr() holds no tree, because it was built with no argument"
            raise CodeReadError(message) from error


class _Interpreter(ast.NodeVisitor):
    """Interpret the syntax tree of a code, with only the forms that the recorder writes.

    Example:
        >>> parsed = ast.parse("pl.col('a') + 1", mode="eval").body
        >>> _Interpreter(0, {}).visit(parsed).meta.root_names()
        ['a']
    """

    # A panic of polars while it builds the expression is a refusal too, and it
    # derives from `BaseException`, so a plain `except Exception` misses it.
    REFUSALS: ClassVar[tuple[type[BaseException], ...]] = (
        pl.exceptions.PolarsError,
        pl.exceptions.PanicException,
        TypeError,
        ValueError,
    )
    BUILTINS: ClassVar[dict[str, object]] = {
        "abs": abs,
        "float": NonFinite().of,
        "date": datetime.date,
        "datetime": datetime.datetime,
        "time": datetime.time,
        "timedelta": datetime.timedelta,
        "Decimal": decimal.Decimal,
        "UTC": datetime.UTC,
        "ZoneInfo": zoneinfo.ZoneInfo,
    }
    BINARY: ClassVar[dict[type[ast.operator], Callable[[Any, Any], Any]]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
    }
    UNARY: ClassVar[dict[type[ast.unaryop], Callable[[Any], Any]]] = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Invert: operator.invert,
    }
    COMPARE: ClassVar[dict[type[ast.cmpop], Callable[[Any, Any], Any]]] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    def __init__(self, leaves: int, choices: Mapping[int, object]) -> None:
        """Keep the number of declared types and the chosen values of the holes.

        Args:
            leaves: The number of declared types of the code.
            choices: The value that the evaluator chose for each hole, by its place.

        Example:
            >>> _Interpreter(2, {}).holes
            []
        """
        self._leaves = leaves
        self._choices = choices
        self._shared: dict[int, object] = {}
        self.holes: list[str] = []
        self.opaque: list[OpaqueCall] = []
        self.refusals: list[str] = []

    def generic_visit(self, node: ast.AST) -> object:
        """Refuse a form that the recorder does not write.

        Args:
            node: The node.

        Raises:
            CodeReadError: Always.

        Example:
            >>> _Interpreter(0, {}).visit(ast.parse("lambda: 1", mode="eval").body)
            Traceback (most recent call last):
            ...
            astrolabe._hooks.deferred.code_reader.CodeReadError: the form Lambda is not allowed
        """
        message = f"the form {type(node).__name__} is not allowed"
        raise CodeReadError(message)

    def visit_Constant(self, node: ast.Constant) -> object:
        """Give a constant.

        Args:
            node: The constant.

        Returns:
            Its value.

        Example:
            >>> _Interpreter(0, {}).visit(ast.Constant("a"))
            'a'
        """
        return node.value

    def visit_Name(self, node: ast.Name) -> object:
        """Give the value of a name: a module, an allowed builtin, a declared type or a shared part.

        Args:
            node: The name.

        Returns:
            The value.

        Raises:
            CodeReadError: The name is not allowed.

        Example:
            >>> _Interpreter(1, {}).visit(ast.Name("__ref_0__", ast.Load())).meta.output_name()
            '__leaf_0__'
        """
        modules: dict[str, object] = {"pl": pl, "cs": cs, **self.BUILTINS}
        if node.id in modules:
            return modules[node.id]
        if node.id == "hole":
            return self._hole
        number = CodeRefs().number(node.id)
        if number is not None and number < self._leaves:
            return pl.col(f"__leaf_{number}__")
        if number is not None and number in self._shared:
            return self._shared[number]
        message = f"the name {node.id!r} is not allowed"
        raise CodeReadError(message)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> object:
        """Give the value of a shared part, and keep it for its references.

        Args:
            node: The definition `($k := value)`.

        Returns:
            The value.

        Raises:
            CodeReadError: The target of the definition is not a shared part.

        Example:
            >>> parsed = ast.parse("[(__ref_0__ := 2), __ref_0__]", mode="eval").body
            >>> _Interpreter(0, {}).visit(parsed)
            [2, 2]
        """
        value = self.visit(node.value)
        number = CodeRefs().number(node.target.id)
        if number is None:
            message = f"the name {node.target.id!r} is not a shared part"
            raise CodeReadError(message)
        self._shared[number] = value
        return value

    def visit_Attribute(self, node: ast.Attribute) -> object:
        """Give a public attribute of a value.

        Args:
            node: The attribute access.

        Returns:
            The attribute.

        Raises:
            CodeReadError: The name of the attribute starts with an underscore.

        Example:
            >>> _Interpreter(0, {}).visit(ast.parse("pl.Int64", mode="eval").body)
            Int64
        """
        if node.attr.startswith("_"):
            message = f"the attribute {node.attr!r} is not allowed"
            raise CodeReadError(message)
        owner = self.visit(node.value)
        try:
            return getattr(owner, node.attr)
        except AttributeError as error:
            raise CodeReadError(str(error)) from error

    def visit_Subscript(self, node: ast.Subscript) -> object:
        """Give an item of the `struct` or the `list` namespace of an expression.

        Polars gives `pl.col("s").struct[0]` the first field of a struct, and
        `pl.col("l").list[0]` the first value of each list. A field takes a
        place or a name, and a value of a list a place. The reader takes no
        other subscript.

        Args:
            node: The subscript.

        Returns:
            The item.

        Raises:
            CodeReadError: The value is not such a namespace, or the item is not
                a number or a text.

        Example:
            >>> item = ast.parse("pl.col('s').struct[0]", mode="eval").body
            >>> isinstance(_Interpreter(0, {}).visit(item), pl.Expr)
            True
            >>> _Interpreter(0, {}).visit(ast.parse("pl.Int64[0]", mode="eval").body)
            Traceback (most recent call last):
            ...
            astrolabe._hooks.deferred.code_reader.CodeReadError: no item of Int64
        """
        owner = self.visit(node.value)
        index = self.visit(node.slice)
        if isinstance(owner, pl.expr.struct.ExprStructNameSpace) and isinstance(index, int | str):
            return owner[index]
        if isinstance(owner, pl.expr.list.ExprListNameSpace) and isinstance(index, int):
            return owner[index]
        message = f"no item of {owner!r}"
        raise CodeReadError(message)

    def visit_Call(self, node: ast.Call) -> object:
        """Call a function of polars, or an allowed builtin, with its arguments.

        A method that takes a function of the user gives the reserved column
        of an opaque call instead, and so does a function of `pl` that runs a
        function of the user over several expressions (`GatheredCall`).

        Args:
            node: The call.

        Returns:
            The value of the call.

        Example:
            >>> parsed = ast.parse("pl.col('a').alias(name='b')", mode="eval").body
            >>> _Interpreter(0, {}).visit(parsed).meta.output_name()
            'b'
        """
        function = self.visit(node.func)
        positional = self._positional(node.args)
        keywords = self._keywords(node.keywords)
        opaque = [*positional, *keywords.values()]
        if not any(isinstance(value, OpaqueFunction) for value in opaque):
            return self._called(function, positional, keywords)
        gathered = GatheredCall().of(function, positional, keywords)
        if gathered is None:
            return self._opaque(node, keywords)
        self.opaque.append(gathered)
        return pl.col(f"__opaque_{len(self.opaque) - 1}__")

    def _called(
        self,
        function: object,
        positional: Sequence[object],
        keywords: Mapping[str, object],
    ) -> object:
        """Call a function with its arguments, and turn a refusal of polars into a read error.

        Args:
            function: The function.
            positional: The positional arguments.
            keywords: The keyword arguments.

        Returns:
            The value of the call.

        Raises:
            BuildRefusal: Polars raises one of its errors, panics, or refuses
                an argument with a `TypeError` or a `ValueError`.
            CodeReadError: The value is not callable, or the call raises
                another error.

        Example:
            >>> _Interpreter(0, {})._called(pl.lit, [1], {}).meta.output_name()
            'literal'
            >>> def panics() -> None:
            ...     raise pl.exceptions.PanicException("boom")
            >>> _Interpreter(0, {})._called(panics, [], {})
            Traceback (most recent call last):
            ...
            astrolabe._hooks.deferred.code_reader.BuildRefusal: boom
        """
        if not callable(function):
            message = "the call of a value that is not callable"
            raise CodeReadError(message)
        self._bound(function, positional, keywords)
        try:
            value = function(*positional, **keywords)
        except CodeReadError:
            raise
        except self.REFUSALS as error:
            raise BuildRefusal(type(error).__name__, str(error).splitlines()[0]) from error
        except Exception as error:
            message = f"polars refuses the call: {error}"
            raise CodeReadError(message) from error
        refusal = ObjectValues().refusal(function, positional, keywords)
        refusal = refusal or PatternValues().refusal(function, positional, keywords)
        if refusal is not None:
            self.refusals.append(refusal)
        return value

    def _bound(
        self,
        function: Callable[..., object],
        positional: Sequence[object],
        keywords: Mapping[str, object],
    ) -> None:
        """Check that the arguments of a call fit the signature of its function.

        mypy reports a call whose arguments do not fit, so the reader does
        not read such a call as one that polars refuses. A function whose
        signature Python cannot read gets no check.

        Args:
            function: The function.
            positional: The positional arguments.
            keywords: The keyword arguments.

        Raises:
            ArgumentMismatch: The arguments do not fit.

        Example:
            >>> _Interpreter(0, {})._bound(pl.lit, [1], {})
            >>> _Interpreter(0, {})._bound(pl.col("a").cast, [], {})
            Traceback (most recent call last):
            ...
            astrolabe._hooks.deferred.code_reader.ArgumentMismatch: the arguments do not fit ...
        """
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return
        try:
            signature.bind(*positional, **keywords)
        except TypeError as error:
            message = f"the arguments do not fit the call: {error}"
            raise ArgumentMismatch(message) from error

    def visit_BinOp(self, node: ast.BinOp) -> object:
        """Apply a binary operator.

        Args:
            node: The operation.

        Returns:
            Its value.

        Example:
            >>> _Interpreter(0, {}).visit(ast.parse("2 ** 3", mode="eval").body)
            8
        """
        rule = self.BINARY[type(node.op)]
        return self._applied(rule, self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> object:
        """Apply a unary operator.

        Args:
            node: The operation.

        Returns:
            Its value.

        Raises:
            CodeReadError: The operator is `not`.

        Example:
            >>> _Interpreter(0, {}).visit(ast.parse("-1", mode="eval").body)
            -1
        """
        rule = self.UNARY.get(type(node.op))
        if rule is None:
            message = "the operator not is not allowed"
            raise CodeReadError(message)
        return self._applied(rule, self.visit(node.operand))

    def visit_Compare(self, node: ast.Compare) -> object:
        """Apply a comparison of two values.

        Args:
            node: The comparison.

        Returns:
            Its value.

        Raises:
            CodeReadError: The comparison chains several operators, or uses `is` or `in`.

        Example:
            >>> str(_Interpreter(0, {}).visit(ast.parse("pl.col('a') > 1", mode="eval").body))
            '[(col("a")) > (dyn int: 1)]'
        """
        rule = self.COMPARE.get(type(node.ops[0])) if len(node.ops) == 1 else None
        if rule is None:
            message = "only one comparison of two values is allowed"
            raise CodeReadError(message)
        return self._applied(rule, self.visit(node.left), self.visit(node.comparators[0]))

    def visit_List(self, node: ast.List) -> object:
        """Give a list.

        Args:
            node: The list.

        Returns:
            Its value.

        Example:
            >>> _Interpreter(0, {}).visit(ast.parse("[1, 'a']", mode="eval").body)
            [1, 'a']
        """
        return self._positional(node.elts)

    def visit_Tuple(self, node: ast.Tuple) -> object:
        """Give a tuple.

        Args:
            node: The tuple.

        Returns:
            Its value.

        Example:
            >>> _Interpreter(0, {}).visit(ast.parse("(1, 2)", mode="eval").body)
            (1, 2)
        """
        return tuple(self._positional(node.elts))

    def visit_Dict(self, node: ast.Dict) -> object:
        """Give a dict.

        Args:
            node: The dict.

        Returns:
            Its value.

        Raises:
            CodeReadError: The dict unpacks another dict.

        Example:
            >>> _Interpreter(0, {}).visit(ast.parse("{'a': 1}", mode="eval").body)
            {'a': 1}
        """
        value: dict[object, object] = {}
        for key, item in zip(node.keys, node.values, strict=True):
            if key is None:
                message = "a dict that unpacks a dict is not allowed"
                raise CodeReadError(message)
            value[self.visit(key)] = self.visit(item)
        return value

    def _positional(self, nodes: list[ast.expr]) -> list[object]:
        """Give the values of positional arguments, with each starred argument unpacked.

        Args:
            nodes: The arguments.

        Returns:
            The values.

        Example:
            >>> call = ast.parse("f(1, *[2, 3])", mode="eval").body
            >>> _Interpreter(0, {})._positional(call.args)
            [1, 2, 3]
        """
        values: list[object] = []
        for node in nodes:
            if isinstance(node, ast.Starred):
                values.extend(self._iterable(self.visit(node.value)))
            else:
                values.append(self.visit(node))
        return values

    def _keywords(self, nodes: list[ast.keyword]) -> dict[str, object]:
        """Give the values of keyword arguments, with each `**` argument unpacked.

        Args:
            nodes: The keyword arguments.

        Returns:
            The values, by name.

        Raises:
            CodeReadError: A `**` argument is not a dict.

        Example:
            >>> call = ast.parse("f(a=1, **{'b': 2})", mode="eval").body
            >>> _Interpreter(0, {})._keywords(call.keywords)
            {'a': 1, 'b': 2}
        """
        values: dict[str, object] = {}
        for node in nodes:
            value = self.visit(node.value)
            if node.arg is not None:
                values[node.arg] = value
            elif isinstance(value, dict):
                values.update({str(key): item for key, item in value.items()})
            else:
                message = "a ** argument must be a dict"
                raise CodeReadError(message)
        return values

    def _iterable(self, value: object) -> list[object]:
        """Give the items of a starred argument.

        Args:
            value: The value after `*`.

        Returns:
            Its items.

        Raises:
            CodeReadError: The value is not a list or a tuple.

        Example:
            >>> _Interpreter(0, {})._iterable((1, 2))
            [1, 2]
        """
        if not isinstance(value, list | tuple):
            message = "a * argument must be a list or a tuple"
            raise CodeReadError(message)
        return list(value)

    def _hole(self, kind: str, *values: object) -> object:
        """Give the placeholder value of the next hole.

        Args:
            kind: The kind of the hole.
            values: The values of a hole of the kind `one of`.

        Returns:
            The value.

        Raises:
            CodeReadError: A hole of another kind lists values.

        Example:
            >>> _Interpreter(0, {1: True})._hole("int") == HoleValues.INT
            True
            >>> _Interpreter(0, {})._hole("one of", "a", "b")
            'a'
            >>> try:
            ...     _Interpreter(0, {})._hole("int", 1)
            ... except CodeReadError as error:
            ...     print(error)
            a hole of the kind 'int' lists no values
        """
        if values and kind != HoleValues.ONE_OF:
            message = f"a hole of the kind {kind!r} lists no values"
            raise CodeReadError(message)
        if values:
            kind = HoleValues().listing(values)
        number = len(self.holes)
        self.holes.append(kind)
        return HoleValues().of(kind, number, self._choices.get(number))

    def _opaque(self, node: ast.Call, keywords: dict[str, object]) -> object:
        """Give the reserved column of a call that takes a function of the user, and keep the call.

        Args:
            node: The call, whose function is an attribute of its receiver.
            keywords: The keyword arguments of the call.

        Returns:
            The reserved column.

        Raises:
            CodeReadError: The call is not a method of an expression.

        Example:
            >>> parsed = ast.parse("pl.col('a').map_batches(hole('function'))", mode="eval").body
            >>> interpreter = _Interpreter(0, {})
            >>> interpreter.visit(parsed).meta.output_name(), interpreter.opaque[0].method
            ('__opaque_0__', 'map_batches')
        """
        if not isinstance(node.func, ast.Attribute):
            message = "a function of the user reaches only a method of an expression"
            raise CodeReadError(message)
        receiver = self.visit(node.func.value)
        if not isinstance(receiver, pl.Expr):
            message = "a function of the user reaches only a method of an expression"
            raise CodeReadError(message)
        self.opaque.append(OpaqueCall(receiver, node.func.attr, keywords))
        return pl.col(f"__opaque_{len(self.opaque) - 1}__")

    def _applied(self, rule: Callable[..., Any], *values: object) -> object:
        """Apply an operator, and turn a refusal of polars into a read error.

        Args:
            rule: The operator.
            *values: Its operands.

        Returns:
            The value.

        Raises:
            BuildRefusal: Polars raises one of its errors, panics, or refuses
                an operand with a `TypeError` or a `ValueError`.
            CodeReadError: The operation raises another error.

        Example:
            >>> _Interpreter(0, {})._applied(operator.add, 1, 2)
            3
        """
        try:
            return rule(*values)
        except self.REFUSALS as error:
            raise BuildRefusal(type(error).__name__, str(error).splitlines()[0]) from error
        except Exception as error:
            message = f"polars refuses the operation: {error}"
            raise CodeReadError(message) from error
