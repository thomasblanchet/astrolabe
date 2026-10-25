"""The definitions of the functions of the `str` namespace, of `concat_str` and of `format`.

The `str` namespace holds the functions of text data. The identity of such a
node names the namespace, for example `StringExpr.Strptime`. `pl.concat_str`
and `pl.format` build nodes of the same namespace, `StringExpr.ConcatHorizontal`
and `StringExpr.Format`, so their definitions are here too.

The type function of a function of `str` does not check its first input: it
gives a fixed type, or the type of the first input. A function that casts its
inputs to their supertype checks in its cast pass that the first input is a
`String`. Every kernel reads its text as `String` data, so the run refuses
any other type.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.cast_pass import (
    Namespace,
    PadLengthCasts,
    PatternListCasts,
    TextInputCasts,
)
from astrolabe._rules.definitions.cast_adapters import FunctionCasting, OwnRuleCasts
from astrolabe._rules.definitions.value_adapters import KeptValue
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import LiteralInputs, OneInput
from astrolabe._rules.definitions.node_options import (
    FromOptions,
    NodeIdentity,
    NodeOption,
    NodeOptions,
    WhenNoColumn,
    WhenSet,
)
from astrolabe._rules.definitions.type_function_adapters import Mapper
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.namespaces import StringKernels
from astrolabe._rules.kernels.strings import TextKernels
from astrolabe._rules.options import FunctionFlag, FunctionOptions, KernelLength, Shape
from astrolabe._rules.supertype import SupertypeFlag
from astrolabe._rules.type_functions import (
    FieldsMapper,
    StringClosures,
)


class StrptimeFallible:
    """Say whether a parse of text can fail on one value, as polars marks it.

    Polars marks `str.strptime` as fallible where its text is not a literal,
    and where the parse is strict or `ambiguous` is not a literal
    `earliest`, `latest` or `null`. A run in `agg` reads the mark: a
    fallible function whose groups miss some values runs group by group.

    Example:
        >>> text = {"Literal": {"Scalar": {"String": "2020"}}}
        >>> options = {"StringExpr": {"Strptime": [{}, {"strict": True}]}}
        >>> node = NodeOptions({"Function": {"input": [text], "function": options}})
        >>> StrptimeFallible().read(node)
        False
    """

    SAFE: ClassVar[frozenset[str]] = frozenset({"earliest", "latest", "null"})
    STRICT: ClassVar[NodeOption[bool]] = NodeOption.flag(
        "function",
        "StringExpr",
        "Strptime",
        1,
        "strict",
    )

    def __repr__(self) -> str:
        """Give the name of the rule.

        Returns:
            The text.

        Example:
            >>> StrptimeFallible()
            StrptimeFallible()
        """
        return "StrptimeFallible()"

    def read(self, node: NodeOptions) -> bool:
        """Give whether the parse of one node can fail on one value.

        Args:
            node: The serialized root of the node.

        Returns:
            Whether it can.

        Example:
            >>> text = {"Column": "s"}
            >>> raises = {"Literal": {"Scalar": {"String": "raise"}}}
            >>> options = {"StringExpr": {"Strptime": [{}, {"strict": False}]}}
            >>> body = {"input": [text, raises], "function": options}
            >>> StrptimeFallible().read(NodeOptions({"Function": body}))
            True
        """
        inputs = node.value(("input",))
        parts = inputs if isinstance(inputs, list) else []
        if not parts or self._literal(parts[0]) is not None:
            return False
        ambiguous = self._literal(parts[1]) if len(parts) > 1 else None
        return self.STRICT.read(node) or ambiguous not in self.SAFE

    def _literal(self, tree: object) -> object | None:
        """Give the value of a literal input, or `None` where the input is not a literal.

        Args:
            tree: The serialized input.

        Returns:
            The text of a literal text, `True` for a literal of another value,
            or `None`.

        Example:
            >>> StrptimeFallible()._literal({"Literal": {"Scalar": {"String": "latest"}}})
            'latest'
        """
        if not isinstance(tree, dict) or "Literal" not in tree:
            return None
        scalar = tree["Literal"].get("Scalar") if isinstance(tree["Literal"], dict) else None
        text = scalar.get("String") if isinstance(scalar, dict) else None
        return text if isinstance(text, str) else True


class Strings:
    """Hold the definitions of the functions of the `str` namespace.

    Example:
        >>> Strings.STRPTIME.identity
        NodeIdentity(shape='Function', function='StringExpr.Strptime')
        >>> Strings.CONTAINS.casts
        FunctionCasting(namespace=<Namespace.STRING: 'str'>)
    """

    STRPTIME: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Strptime"),
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(
                StringClosures().strptime,
                NodeOption.dtype("function", "StringExpr", "Strptime", 0),
                NodeOption.text("function", "StringExpr", "Strptime", 1, "format"),
            ),
        ),
        options=WhenNoColumn(
            0,
            Shape.function(FunctionOptions.elementwise(), KernelLength.of_input(0)),
            WhenSet(
                NodeOption.text("function", "StringExpr", "Strptime", 1, "format"),
                Shape.function(FunctionOptions.elementwise(), KernelLength.of_input(0)),
                Shape.function(FunctionOptions.elementwise_with_infer(), KernelLength.of_input(0)),
            ),
        ),
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                StringKernels().strptime,
                FromOptions(
                    StringClosures().strptime,
                    NodeOption.dtype("function", "StringExpr", "Strptime", 0),
                    NodeOption.text("function", "StringExpr", "Strptime", 1, "format"),
                ),
                NodeOption.flag("function", "StringExpr", "Strptime", 1, "exact"),
                NodeOption.text("function", "StringExpr", "Strptime", 1, "format"),
                NodeOption.flag("function", "StringExpr", "Strptime", 1, "strict"),
            ),
            can_fail=StrptimeFallible(),
        ),
    )
    CONCAT_HORIZONTAL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.ConcatHorizontal"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("String")),
        options=Shape.function(
            FunctionOptions.elementwise().flag(FunctionFlag.INPUT_WILDCARD_EXPANSION),
        ),
        casts=OwnRuleCasts(TextInputCasts()),
        kernel=LiteralInputs(TextKernels.CONCAT_STR),
    )
    CONCAT_VERTICAL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.ConcatVertical"),
        computes=KeptValue(),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("String")),
        options=Shape.function(FunctionOptions.aggregation()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.JOIN),
    )
    FORMAT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Format"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("String")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.FORMAT),
    )
    CONTAINS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Contains"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(
            FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.CONTAINS),
    )
    CONTAINS_ANY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.ContainsAny"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(PatternListCasts((1,))),
        kernel=LiteralInputs(TextKernels.CONTAINS_ANY),
    )
    COUNT_MATCHES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.CountMatches"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("UInt32")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.COUNT_MATCHES),
    )
    ENDS_WITH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.EndsWith"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(
            FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.AFFIX),
    )
    STARTS_WITH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.StartsWith"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(
            FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.AFFIX),
    )
    ESCAPE_REGEX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.EscapeRegex"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(TextKernels.TO_TEXT),
    )
    EXTRACT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Extract"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(
            FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.EXTRACT),
    )
    EXTRACT_ALL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.ExtractAll"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("List", (DType("String"),))),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.EXTRACT_ALL),
    )
    EXTRACT_GROUPS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.ExtractGroups"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=NodeOption.data_type("function", "StringExpr", "ExtractGroups", "dtype"),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(
            FromOptions(
                TextKernels().parse,
                NodeOption.data_type("function", "StringExpr", "ExtractGroups", "dtype"),
            ),
        ),
    )
    EXTRACT_MANY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.ExtractMany"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("List", (DType("String"),))),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(PatternListCasts((1,))),
        kernel=LiteralInputs(TextKernels.EXTRACT_MANY),
    )
    FIND: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Find"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("UInt32")),
        options=Shape.function(
            FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.FIND),
    )
    FIND_MANY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.FindMany"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("List", (DType("UInt32"),))),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(PatternListCasts((1,))),
        kernel=LiteralInputs(TextKernels.FIND_MANY),
    )
    HEAD: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Head"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.HEAD),
    )
    TAIL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Tail"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.HEAD),
    )
    HEX_DECODE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.HexDecode"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Binary")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(TextKernels.HEX_DECODE),
    )
    HEX_ENCODE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.HexEncode"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(TextKernels.TO_TEXT),
    )
    JSON_DECODE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.JsonDecode"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=NodeOption.dtype("function", "StringExpr", "JsonDecode"),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(
            FromOptions(
                TextKernels().parse,
                NodeOption.dtype("function", "StringExpr", "JsonDecode"),
            ),
        ),
    )
    JSON_PATH_MATCH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.JsonPathMatch"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("String")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.JSON_PATH_MATCH),
    )
    LEN_BYTES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.LenBytes"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("UInt32")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(TextKernels.LEN_BYTES),
    )
    LEN_CHARS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.LenChars"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("UInt32")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(TextKernels.LEN_CHARS),
    )
    LOWERCASE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Lowercase"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(TextKernels.TO_TEXT),
    )
    UPPERCASE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Uppercase"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(TextKernels.TO_TEXT),
    )
    TITLECASE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Titlecase"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(TextKernels.TO_TEXT),
    )
    NORMALIZE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Normalize"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(TextKernels.TO_TEXT),
    )
    REVERSE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Reverse"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(TextKernels.TO_TEXT),
    )
    PAD_START: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.PadStart"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(PadLengthCasts()),
        kernel=LiteralInputs(TextKernels.PAD),
    )
    PAD_END: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.PadEnd"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(PadLengthCasts()),
        kernel=LiteralInputs(TextKernels.PAD),
    )
    ZFILL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.ZFill"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(PadLengthCasts()),
        kernel=LiteralInputs(TextKernels.PAD),
    )
    REPLACE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Replace"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(
            FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(
            FromOptions(
                TextKernels().replace,
                NodeOption.integer("function", "StringExpr", "Replace", "n"),
            ),
        ),
    )
    REPLACE_MANY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.ReplaceMany"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(PatternListCasts((1, 2))),
        kernel=LiteralInputs(TextKernels.REPLACE_MANY),
    )
    SLICE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Slice"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.SLICE),
    )
    SPLIT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.Split"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("List", (DType("String"),))),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.SPLIT),
    )
    SPLIT_EXACT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.SplitExact"),
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(
                StringClosures().split_exact,
                NodeOption.integer("function", "StringExpr", "SplitExact", "n"),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise(), KernelLength.of_input(0)),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(
            FromOptions(
                TextKernels().split,
                FromOptions(
                    StringClosures().split_exact,
                    NodeOption.integer("function", "StringExpr", "SplitExact", "n"),
                ),
            ),
        ),
    )
    SPLIT_N: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.SplitN"),
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(
                StringClosures().splitn,
                NodeOption.integer("function", "StringExpr", "SplitN"),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise(), KernelLength.of_input(0)),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(
            FromOptions(
                TextKernels().split,
                FromOptions(
                    StringClosures().splitn,
                    NodeOption.integer("function", "StringExpr", "SplitN"),
                ),
            ),
        ),
    )
    STRIP_CHARS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.StripChars"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.STRIP_CHARS),
    )
    STRIP_CHARS_START: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.StripCharsStart"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.STRIP_CHARS),
    )
    STRIP_CHARS_END: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.StripCharsEnd"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.STRIP_CHARS),
    )
    STRIP_PREFIX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.StripPrefix"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.STRIP_AFFIX),
    )
    STRIP_SUFFIX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.StripSuffix"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(TextKernels.STRIP_AFFIX),
    )
    TO_DECIMAL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.ToDecimal"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(
                StringClosures().to_decimal,
                NodeOption.integer("function", "StringExpr", "ToDecimal", "scale"),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=OneInput(
            FromOptions(
                TextKernels().parse,
                FromOptions(
                    StringClosures().to_decimal,
                    NodeOption.integer("function", "StringExpr", "ToDecimal", "scale"),
                ),
            ),
        ),
    )
    TO_INTEGER: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StringExpr.ToInteger"),
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(
                StringClosures().to_integer,
                NodeOption.data_type("function", "StringExpr", "ToInteger", "dtype"),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.STRING),
        kernel=LiteralInputs(
            FromOptions(
                TextKernels().to_integer,
                NodeOption.data_type("function", "StringExpr", "ToInteger", "dtype"),
            ),
        ),
    )
