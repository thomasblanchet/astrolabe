"""Check the definitions of the text family against polars, node by node.

The text family holds the functions of the `str`, `bin` and `cat` namespaces,
and `pl.concat_str` and `pl.format`. Each test builds the node of polars over
the inputs of one cell, and compares the four answers of polars with the
four answers of the rules (`CellCheck`): the planner type and the run type,
in a `select` and in `agg` over two groups of other sizes.

A node of one input takes every input (`NodeSample`). A node of two inputs
takes about 500 pairs beside a typical input at each place (`PairSample`),
and a node of three inputs about 500 cells (`TripleSample`). A call that
differs from the main call of its node only by an option takes a smaller
sample. A node that parses text, such as `str.decode`, takes a literal text
that parses: the rules read the type of a text and not its value, and a text
that does not parse is a fact of the data. The frames hold text that parses
as a number, as a date and as a time (`LiveFrames`).

Each test names the node and the cell, so a failure names both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import (
    CellCheck,
    Input,
    Inputs,
    NodeSample,
    PairSample,
    TemplateCall,
    TripleSample,
    Values,
)
from astrolabe._rules.definitions.binaries import Binaries
from astrolabe._rules.definitions.categories import Categories
from astrolabe._rules.definitions.strings import Strings
from astrolabe._rules.node_evaluation import LiteralLeaf

if TYPE_CHECKING:
    from astrolabe._rules.definitions.definition import Definition


class TextCall:
    """Hold one node of the text family: its definition, its call, and the values of its sample.

    `text` is the text of the literal text inputs, a text that the node
    parses. `refused` is the sample that a run of polars reads where the rules
    refuse the cell: a node that casts an input with a strict cast, or reads
    a number as an index or a base, reads the sample whose values fit there,
    so that the error is the error of the types (`Values.TEN`). `left_out`
    names the inputs that the sample of the node leaves out, each for a
    reason that the docstring of `TextNodes` gives.

    Example:
        >>> TextCall(Strings.LEN_CHARS, "pl.col('x').str.len_chars()").text
        'a'
    """

    def __init__(
        self,
        definition: Definition,
        code: str,
        text: str = "a",
        *,
        refused: Values = Values.TYPICAL,
        left_out: tuple[str, ...] = (),
    ) -> None:
        """Keep the definition, the call and the values of the sample.

        Args:
            definition: The definition of the node.
            code: The call, over the column `x`, as the census writes it.
            text: The text of the literal text inputs.
            refused: The sample that a run reads where the rules refuse it.
            left_out: The labels of the inputs that the sample leaves out.

        Example:
            >>> TextCall(Strings.HEX_DECODE, "pl.col('x').str.decode('hex')", "61")._code
            "pl.col('x').str.decode('hex')"
        """
        self._definition = definition
        self._code = code
        self.text = text
        self._refused = refused
        self._left_out = left_out

    def check(self) -> CellCheck:
        """Give the comparison of the rules with polars for this node.

        Returns:
            The comparison.

        Example:
            >>> TextCall(Strings.LEN_CHARS, "pl.col('x').str.len_chars()").check().label(())
            ''
        """
        return CellCheck(self._definition, TemplateCall(self._code), self._refused)

    def inputs(self) -> list[Input]:
        """Give every input, with the literal text of the node, less the inputs left out.

        Returns:
            The inputs.

        Example:
            >>> call = TextCall(Strings.HEX_DECODE, "pl.col('x').str.decode('hex')", "61")
            >>> str(call.inputs()[-6].expression)
            '"61"'
        """
        inputs = Inputs().with_text(self.text)
        return [entry for entry in inputs if entry.label not in self._left_out]


class TextNodes:
    """Hold the nodes of the tests, each with its call and the typical inputs of its sample.

    The typical text is a `String` column, and the typical bytes a `Binary`
    column. The typical pattern of a function of `str` is the literal `a`,
    and the typical count, offset, length or index is the literal `1`.
    `str.strip_chars` takes the literal null as its typical characters, as
    in `str.strip_chars()`, and `str.to_integer` the literal `10` as its
    typical base. The functions of `bin` take a `Binary` column as their
    typical pattern, because a literal of bytes is not an input of the
    tests, and `bin.get` takes the index `-1`, the last byte.

    `str.pad_start`, `str.pad_end` and `str.zfill` leave out each input whose
    values give a width of more than 100,000 characters (`WIDE`): the literal
    `2**40`, the `Datetime`, `Duration` and `Time` columns, whose physical
    values are large, and the derived literal `-1`, which a cast that wraps
    turns into a width of about `2**64`. Polars builds the padded text, so
    such a width takes more memory than the machine holds. The rules read only the type of
    the width. Where the rules refuse a pad, polars reads text that parses as a
    number, because a width of text casts with a strict cast.

    Some inputs hold values that decide the answer, so a node leaves them
    out. A regular expression of `str.contains`, `str.find` and
    `str.extract` leaves out a `Struct` pattern (`STRUCTS`): the text of a
    struct, such as `{1}`, is not a valid expression. `str.json_path_match`
    takes the path `$`, and leaves out a struct and a text (`TEXTS`), whose
    values are not paths. `bin.get` leaves out the indices 300 and `2**40`
    (`FAR`), past the end of every value of bytes. `str.replace_many` leaves
    out a text of one value for each row as its replacements: the cast pass
    makes one replacement of each row, and the count of rows must match the
    count of patterns. `str.to_integer` leaves out a `Categorical` base
    (`CATEGORICAL`): the ids of its categories decide the base, so the rules
    give `Unchecked` there.

    Example:
        >>> sorted(TextNodes.ONE_INPUT)[:2]
        ['bin.decode', 'bin.encode']
    """

    TEXT: ClassVar[Input] = Inputs().labelled("String/per row")
    BYTES: ClassVar[Input] = Inputs().labelled("Binary/per row")
    PATTERN: ClassVar[Input] = Inputs().labelled("lit str/direct")
    COUNT: ClassVar[Input] = Inputs().labelled("lit int 1/direct")
    NULL: ClassVar[Input] = Inputs().labelled("lit null/direct")
    LAST: ClassVar[Input] = Inputs().labelled("lit int -1/direct")
    BASE: ClassVar[Input] = Input("lit int 10/direct", LiteralLeaf(10), pl.lit(10))
    WIDE: ClassVar[tuple[str, ...]] = (
        "lit int 2**40/direct",
        "lit int 2**40/derived",
        "lit int -1/derived",
        *(
            entry.label
            for entry in Inputs().all()
            if entry.label.startswith(("Datetime(", "Duration(", "Time/"))
        ),
    )
    FAR: ClassVar[tuple[str, ...]] = (
        "lit int 300/direct",
        "lit int 300/derived",
        "lit int 2**40/direct",
        "lit int 2**40/derived",
    )
    STRUCTS: ClassVar[tuple[str, ...]] = tuple(
        entry.label for entry in Inputs().all() if entry.label.startswith("Struct(")
    )
    TEXTS: ClassVar[tuple[str, ...]] = (
        *(f"String/{form}" for form in ("per row", "one value", "other length", "one row")),
        "lit str/direct",
        "lit str/derived",
    )
    CATEGORICAL: ClassVar[tuple[str, ...]] = tuple(
        f"Categorical/{form}" for form in ("per row", "one value", "other length", "one row")
    )

    ONE_INPUT: ClassVar[dict[str, TextCall]] = {
        "bin.decode": TextCall(Binaries.HEX_DECODE, "pl.col('x').bin.decode('hex')"),
        "bin.encode": TextCall(Binaries.HEX_ENCODE, "pl.col('x').bin.encode('hex')"),
        "bin.reinterpret": TextCall(
            Binaries.REINTERPRET,
            "pl.col('x').bin.reinterpret(dtype=pl.UInt8)",
        ),
        "bin.size": TextCall(Binaries.SIZE, "pl.col('x').bin.size()"),
        "cat.ends_with": TextCall(Categories.ENDS_WITH, "pl.col('x').cat.ends_with('a')"),
        "cat.get_categories": TextCall(
            Categories.GET_CATEGORIES,
            "pl.col('x').cat.get_categories()",
        ),
        "cat.len_bytes": TextCall(Categories.LEN_BYTES, "pl.col('x').cat.len_bytes()"),
        "cat.len_chars": TextCall(Categories.LEN_CHARS, "pl.col('x').cat.len_chars()"),
        "cat.physical": TextCall(Categories.PHYSICAL, "pl.col('x').cat.physical()"),
        "cat.slice": TextCall(Categories.SLICE, "pl.col('x').cat.slice(0, 1)"),
        "cat.starts_with": TextCall(Categories.STARTS_WITH, "pl.col('x').cat.starts_with('a')"),
        "cat.to": TextCall(Categories.TO, "pl.col('x').cat.to(pl.String)"),
        "pl.format": TextCall(Strings.FORMAT, "pl.format('{}', pl.col('x'))"),
        "str.concat": TextCall(Strings.CONCAT_VERTICAL, "pl.col('x').str.concat()"),
        "str.decode": TextCall(Strings.HEX_DECODE, "pl.col('x').str.decode('hex')", "61"),
        "str.encode": TextCall(Strings.HEX_ENCODE, "pl.col('x').str.encode('hex')"),
        "str.escape_regex": TextCall(Strings.ESCAPE_REGEX, "pl.col('x').str.escape_regex()"),
        "str.extract_groups": TextCall(
            Strings.EXTRACT_GROUPS,
            "pl.col('x').str.extract_groups(r'(?<g>a)')",
        ),
        "str.join": TextCall(Strings.CONCAT_VERTICAL, "pl.col('x').str.join()"),
        "str.json_decode": TextCall(
            Strings.JSON_DECODE,
            "pl.col('x').str.json_decode(pl.Int64)",
            "1",
        ),
        "str.len_bytes": TextCall(Strings.LEN_BYTES, "pl.col('x').str.len_bytes()"),
        "str.len_chars": TextCall(Strings.LEN_CHARS, "pl.col('x').str.len_chars()"),
        "str.normalize": TextCall(Strings.NORMALIZE, "pl.col('x').str.normalize()"),
        "str.reverse": TextCall(Strings.REVERSE, "pl.col('x').str.reverse()"),
        "str.to_decimal": TextCall(Strings.TO_DECIMAL, "pl.col('x').str.to_decimal(scale=2)", "1"),
        "str.to_lowercase": TextCall(Strings.LOWERCASE, "pl.col('x').str.to_lowercase()"),
        "str.to_titlecase": TextCall(Strings.TITLECASE, "pl.col('x').str.to_titlecase()"),
        "str.to_uppercase": TextCall(Strings.UPPERCASE, "pl.col('x').str.to_uppercase()"),
    }
    ONE_INPUT_OPTIONS: ClassVar[dict[str, TextCall]] = {
        "bin.reinterpret Int16 big endian": TextCall(
            Binaries.REINTERPRET,
            "pl.col('x').bin.reinterpret(dtype=pl.Int16, endianness='big')",
        ),
        "cat.to Enum": TextCall(Categories.TO, "pl.col('x').cat.to(pl.Enum(['a', 'b']))"),
        "cat.to Categorical": TextCall(Categories.TO, "pl.col('x').cat.to(pl.Categorical())"),
        "pl.format two parts": TextCall(Strings.FORMAT, "pl.format('<{}>', pl.col('x'))"),
        "str.join with nulls": TextCall(
            Strings.CONCAT_VERTICAL,
            "pl.col('x').str.join('-', ignore_nulls=False)",
        ),
        "str.extract_groups two groups": TextCall(
            Strings.EXTRACT_GROUPS,
            "pl.col('x').str.extract_groups(r'(?<g>a)(?<h>b)?')",
        ),
        "str.json_decode Float32": TextCall(
            Strings.JSON_DECODE,
            "pl.col('x').str.json_decode(pl.Float32)",
            "1",
        ),
        "str.to_decimal scale 0": TextCall(
            Strings.TO_DECIMAL,
            "pl.col('x').str.to_decimal(scale=0)",
            "1",
        ),
    }
    TWO_INPUTS: ClassVar[dict[str, tuple[TextCall, tuple[Input, Input]]]] = {
        "bin.contains": (
            TextCall(Binaries.CONTAINS, "pl.col('x').bin.contains(b'a')"),
            (BYTES, BYTES),
        ),
        "bin.ends_with": (
            TextCall(Binaries.ENDS_WITH, "pl.col('x').bin.ends_with(b'a')"),
            (BYTES, BYTES),
        ),
        "bin.get": (
            TextCall(Binaries.GET, "pl.col('x').bin.get(0)", refused=Values.TEN, left_out=FAR),
            (BYTES, LAST),
        ),
        "bin.head": (
            TextCall(Binaries.HEAD, "pl.col('x').bin.head()", refused=Values.TEN),
            (BYTES, COUNT),
        ),
        "bin.starts_with": (
            TextCall(Binaries.STARTS_WITH, "pl.col('x').bin.starts_with(b'a')"),
            (BYTES, BYTES),
        ),
        "bin.tail": (
            TextCall(Binaries.TAIL, "pl.col('x').bin.tail()", refused=Values.TEN),
            (BYTES, COUNT),
        ),
        "pl.concat_str": (
            TextCall(Strings.CONCAT_HORIZONTAL, "pl.concat_str(pl.col('x'), pl.col('y'))"),
            (TEXT, PATTERN),
        ),
        "str.contains": (
            TextCall(Strings.CONTAINS, "pl.col('x').str.contains('a')", left_out=STRUCTS),
            (TEXT, PATTERN),
        ),
        "str.contains_any": (
            TextCall(Strings.CONTAINS_ANY, "pl.col('x').str.contains_any(['a'])"),
            (TEXT, PATTERN),
        ),
        "str.count_matches": (
            TextCall(Strings.COUNT_MATCHES, "pl.col('x').str.count_matches('a')"),
            (TEXT, PATTERN),
        ),
        "str.ends_with": (
            TextCall(Strings.ENDS_WITH, "pl.col('x').str.ends_with('a')"),
            (TEXT, PATTERN),
        ),
        "str.extract": (
            TextCall(Strings.EXTRACT, "pl.col('x').str.extract(r'(a)')", left_out=STRUCTS),
            (TEXT, PATTERN),
        ),
        "str.extract_all": (
            TextCall(Strings.EXTRACT_ALL, "pl.col('x').str.extract_all('a')"),
            (TEXT, PATTERN),
        ),
        "str.extract_many": (
            TextCall(Strings.EXTRACT_MANY, "pl.col('x').str.extract_many(['a'])"),
            (TEXT, PATTERN),
        ),
        "str.find": (
            TextCall(Strings.FIND, "pl.col('x').str.find('a')", left_out=STRUCTS),
            (TEXT, PATTERN),
        ),
        "str.find_many": (
            TextCall(Strings.FIND_MANY, "pl.col('x').str.find_many(['a'])"),
            (TEXT, PATTERN),
        ),
        "str.head": (
            TextCall(Strings.HEAD, "pl.col('x').str.head(1)", refused=Values.TEN),
            (TEXT, COUNT),
        ),
        "str.json_path_match": (
            TextCall(
                Strings.JSON_PATH_MATCH,
                "pl.col('x').str.json_path_match('$')",
                "$",
                left_out=(*STRUCTS, *TEXTS),
            ),
            (TEXT, Input("lit text/direct", LiteralLeaf("$"), pl.lit("$"))),
        ),
        "str.pad_end": (
            TextCall(
                Strings.PAD_END,
                "pl.col('x').str.pad_end(3)",
                refused=Values.TEN,
                left_out=WIDE,
            ),
            (TEXT, COUNT),
        ),
        "str.pad_start": (
            TextCall(
                Strings.PAD_START,
                "pl.col('x').str.pad_start(3)",
                refused=Values.TEN,
                left_out=WIDE,
            ),
            (TEXT, COUNT),
        ),
        "str.split": (TextCall(Strings.SPLIT, "pl.col('x').str.split('a')"), (TEXT, PATTERN)),
        "str.split_exact": (
            TextCall(Strings.SPLIT_EXACT, "pl.col('x').str.split_exact('a', 1)"),
            (TEXT, PATTERN),
        ),
        "str.splitn": (
            TextCall(Strings.SPLIT_N, "pl.col('x').str.splitn('a', 2)"),
            (TEXT, PATTERN),
        ),
        "str.starts_with": (
            TextCall(Strings.STARTS_WITH, "pl.col('x').str.starts_with('a')"),
            (TEXT, PATTERN),
        ),
        "str.strip_chars": (
            TextCall(Strings.STRIP_CHARS, "pl.col('x').str.strip_chars()"),
            (TEXT, NULL),
        ),
        "str.strip_chars_end": (
            TextCall(Strings.STRIP_CHARS_END, "pl.col('x').str.strip_chars_end()"),
            (TEXT, NULL),
        ),
        "str.strip_chars_start": (
            TextCall(Strings.STRIP_CHARS_START, "pl.col('x').str.strip_chars_start()"),
            (TEXT, NULL),
        ),
        "str.strip_prefix": (
            TextCall(Strings.STRIP_PREFIX, "pl.col('x').str.strip_prefix('a')"),
            (TEXT, PATTERN),
        ),
        "str.strip_suffix": (
            TextCall(Strings.STRIP_SUFFIX, "pl.col('x').str.strip_suffix('a')"),
            (TEXT, PATTERN),
        ),
        "str.tail": (
            TextCall(Strings.TAIL, "pl.col('x').str.tail(1)", refused=Values.TEN),
            (TEXT, COUNT),
        ),
        "str.to_integer": (
            TextCall(
                Strings.TO_INTEGER,
                "pl.col('x').str.to_integer()",
                "1",
                refused=Values.TEN,
                left_out=CATEGORICAL,
            ),
            (TEXT, BASE),
        ),
        "str.zfill": (
            TextCall(Strings.ZFILL, "pl.col('x').str.zfill(3)", refused=Values.TEN, left_out=WIDE),
            (TEXT, COUNT),
        ),
    }
    TWO_INPUT_OPTIONS: ClassVar[dict[str, tuple[TextCall, tuple[Input, Input]]]] = {
        "bin.get null on out of bounds": (
            TextCall(
                Binaries.GET,
                "pl.col('x').bin.get(0, null_on_oob=True)",
                refused=Values.TEN,
            ),
            (BYTES, COUNT),
        ),
        "pl.concat_str ignore nulls": (
            TextCall(
                Strings.CONCAT_HORIZONTAL,
                "pl.concat_str(pl.col('x'), pl.col('y'), separator='-', ignore_nulls=True)",
            ),
            (TEXT, PATTERN),
        ),
        "str.contains literal": (
            TextCall(Strings.CONTAINS, "pl.col('x').str.contains('a', literal=True)"),
            (TEXT, PATTERN),
        ),
        "str.count_matches literal": (
            TextCall(Strings.COUNT_MATCHES, "pl.col('x').str.count_matches('a', literal=True)"),
            (TEXT, PATTERN),
        ),
        "str.extract group 0": (
            TextCall(Strings.EXTRACT, "pl.col('x').str.extract(r'(a)', 0)", left_out=STRUCTS),
            (TEXT, PATTERN),
        ),
        "str.split inclusive": (
            TextCall(Strings.SPLIT, "pl.col('x').str.split('a', inclusive=True)"),
            (TEXT, PATTERN),
        ),
        "str.split_exact 2 inclusive": (
            TextCall(Strings.SPLIT_EXACT, "pl.col('x').str.split_exact('a', 2, inclusive=True)"),
            (TEXT, PATTERN),
        ),
        "str.splitn 1": (
            TextCall(Strings.SPLIT_N, "pl.col('x').str.splitn('a', 1)"),
            (TEXT, PATTERN),
        ),
        "str.to_integer UInt8 not strict": (
            TextCall(
                Strings.TO_INTEGER,
                "pl.col('x').str.to_integer(dtype=pl.UInt8, strict=False)",
                "1",
                refused=Values.TEN,
                left_out=CATEGORICAL,
            ),
            (TEXT, BASE),
        ),
    }
    THREE_INPUTS: ClassVar[dict[str, tuple[TextCall, tuple[Input, Input, Input]]]] = {
        "bin.slice": (
            TextCall(Binaries.SLICE, "pl.col('x').bin.slice(0, 1)", refused=Values.TEN),
            (BYTES, COUNT, COUNT),
        ),
        "str.replace": (
            TextCall(Strings.REPLACE, "pl.col('x').str.replace('a', 'b')"),
            (TEXT, PATTERN, PATTERN),
        ),
        "str.replace_all": (
            TextCall(Strings.REPLACE, "pl.col('x').str.replace_all('a', 'b')"),
            (TEXT, PATTERN, PATTERN),
        ),
        "str.replace_many": (
            TextCall(
                Strings.REPLACE_MANY,
                "pl.col('x').str.replace_many(['a'], ['b'])",
                left_out=("String/per row",),
            ),
            (TEXT, PATTERN, PATTERN),
        ),
        "str.slice": (
            TextCall(Strings.SLICE, "pl.col('x').str.slice(0, 1)", refused=Values.TEN),
            (TEXT, COUNT, COUNT),
        ),
    }

    @classmethod
    def one_input_cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node of one input beside each cell of its sample.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> cells = TextNodes.one_input_cells()
            >>> len(cells) == 236 * len(TextNodes.ONE_INPUT) + 63 * len(TextNodes.ONE_INPUT_OPTIONS)
            True
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, call in cls.ONE_INPUT.items():
            cells.extend((name, cell) for cell in NodeSample(call.inputs()).cells(1))
        for name, call in cls.ONE_INPUT_OPTIONS.items():
            cells.extend((name, cell) for cell in NodeSample(call.inputs()).options())
        return cells

    @classmethod
    def two_input_cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node of two inputs beside each pair of its sample.

        A call at another value of an option takes the typical input at one
        place beside each input of the smaller sample at the other place.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> cells = TextNodes.two_input_cells()
            >>> cells[0][0], len(cells)
            ('bin.contains', 16670)
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (call, typical) in cls.TWO_INPUTS.items():
            sample = PairSample(call.inputs(), typical[0], typical[1])
            cells.extend((name, pair) for pair in sample.pairs())
        for name, (call, (left, right)) in cls.TWO_INPUT_OPTIONS.items():
            options = [cell[0] for cell in NodeSample(call.inputs()).options()]
            cells.extend((name, (entry, right)) for entry in options)
            cells.extend((name, (left, entry)) for entry in options)
        return cells

    @classmethod
    def three_input_cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node of three inputs beside each cell of its sample.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> len(TextNodes.three_input_cells())
            2497
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (call, typical) in cls.THREE_INPUTS.items():
            sample = TripleSample(call.inputs(), typical)
            cells.extend((name, cell) for cell in sample.cells())
        return cells

    @classmethod
    def call(cls, name: str) -> TextCall:
        """Give the node of a name, of any number of inputs.

        Args:
            name: The name of the node.

        Returns:
            The node.

        Example:
            >>> TextNodes.call("str.slice").text
            'a'
        """
        calls = {**cls.ONE_INPUT, **cls.ONE_INPUT_OPTIONS}
        for nodes in (cls.TWO_INPUTS, cls.TWO_INPUT_OPTIONS, cls.THREE_INPUTS):
            calls.update({key: call for key, (call, _) in nodes.items()})
        return calls[name]

    @classmethod
    def ids(cls, cells: list[tuple[str, tuple[Input, ...]]]) -> list[str]:
        """Give the identifier of each test: the node and the cell.

        Args:
            cells: The name of the node and the cell, for each cell.

        Returns:
            The identifiers.

        Example:
            >>> TextNodes.ids([("str.find", (TextNodes.TEXT, TextNodes.PATTERN))])
            ['str.find-String/per row, lit str/direct']
        """
        return [f"{name}-{CellCheck.label(cell)}" for name, cell in cells]


@pytest.mark.parametrize(
    ("node", "cell"),
    TextNodes.one_input_cells(),
    ids=TextNodes.ids(TextNodes.one_input_cells()),
)
def test_text_node_of_one_input_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a text function of one input match polars, at one input."""
    problems = TextNodes.call(node).check().problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("node", "cell"),
    TextNodes.two_input_cells(),
    ids=TextNodes.ids(TextNodes.two_input_cells()),
)
def test_text_node_of_two_inputs_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a text function of two inputs match polars, at one pair of its sample."""
    problems = TextNodes.call(node).check().problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("node", "cell"),
    TextNodes.three_input_cells(),
    ids=TextNodes.ids(TextNodes.three_input_cells()),
)
def test_text_node_of_three_inputs_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a text function of three inputs match polars, at one cell."""
    problems = TextNodes.call(node).check().problems(list(cell))
    assert not problems, "\n".join(problems)
