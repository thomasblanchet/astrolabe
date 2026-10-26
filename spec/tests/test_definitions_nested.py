"""Check the definitions of the `list`, `arr` and `struct` namespaces against polars, node by node.

A definition names the parts of the vocabulary that the rules of one node
use, and `NodeEvaluation` gives the four answers of the node from it: the
planner type and the run type, in a `select` and in `group_by().agg()`. Each
test here builds the node of polars over the inputs of one cell, and compares
the four answers of polars with the four answers of the rules (`CellCheck`).
The frame has two groups of different sizes, and samples of values that fit
and that do not fit (`LiveFrames`).

A node of one input takes each input of `Inputs`: each type of `RuleKinds` in
each row form, and each literal class. Its calls at other values of an option
take a smaller sample (`NodeSample.options`). A node of two inputs takes about
500 pairs (`PairSample`), and a node of three inputs takes each input at each
place beside the typical inputs at the other places, and a few cells spread
over the product (`TripleSample`).

Two facts of the values shape the samples. The lists of the samples hold two
values, so an index or a count of 300 lies outside every list of every
sample, and polars raises where the rules give the type of the run. The
nodes that read an index or a count take a literal that fits in its place
(`FittingCounts`). And a strict cast in a kernel raises on a value that does
not fit before the kernel checks the lengths of its inputs. Where the rules
refuse the run, the test then accepts the refusal of polars on any sample of
values, because a sample whose values fit shows the refusal of the types
(`NestedCheck`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import (
    CellCheck,
    Input,
    Inputs,
    LiveFrames,
    NodeSample,
    PairSample,
    PolarsAnswer,
    PolarsCell,
    Stopped,
    Values,
)
from astrolabe._rules.definitions.arrays import Arrays
from astrolabe._rules.definitions.lists import Lists
from astrolabe._rules.definitions.node_options import NodeOptions
from astrolabe._rules.definitions.structs import Structs
from astrolabe._rules.dtypes import DType
from astrolabe._rules.failures import Stop
from astrolabe._rules.node_evaluation import LiteralLeaf, NodeEvaluation, Refused, Unchecked
from astrolabe._rules.operators import Operator
from astrolabe._rules.releases import PolarsRelease

if TYPE_CHECKING:
    from collections.abc import Callable

    from astrolabe._rules.definitions.definition import Definition


class FittingCounts:
    """Give the inputs of a place that reads an index or a count, with literals that fit the lists.

    The lists of the samples hold two values. A literal index or count of 300
    or of 2**40 lies outside them, which is a fact of the data, and polars
    raises at every sample. In its place, the inputs take the literal 0,
    direct and derived. The literal -1 stays, because it reads the last value.

    Example:
        >>> labels = [entry.label for entry in FittingCounts().inputs((300,))]
        >>> "lit int 300/direct" in labels, "lit int 0/direct" in labels
        (False, True)
    """

    def inputs(self, outside: tuple[int, ...]) -> list[Input]:
        """Give every input, with the literal ints of the values outside the lists replaced by 0.

        Args:
            outside: The values of the literal classes that lie outside the lists.

        Returns:
            The inputs.

        Example:
            >>> len(FittingCounts().inputs((300, 2**40))) == len(Inputs().all())
            True
        """
        return [self.fitted(entry, outside) for entry in Inputs().all()]

    def fitted(self, entry: Input, outside: tuple[int, ...]) -> Input:
        """Give an input, or the literal 0 in place of a literal int outside the lists.

        Args:
            entry: The input.
            outside: The values of the literal classes that lie outside the lists.

        Returns:
            The input, or the literal 0, direct or derived as the input is.

        Example:
            >>> large = Inputs().labelled("lit int 300/direct")
            >>> FittingCounts().fitted(large, (300,)).label, FittingCounts().fitted(large, ()).label
            ('lit int 0/direct', 'lit int 300/direct')
        """
        names = {"int 300": 300, "int 2**40": 2**40}
        for name, value in names.items():
            if value not in outside:
                continue
            if entry.label == f"lit {name}/direct":
                return Input("lit int 0/direct", LiteralLeaf(0), pl.lit(0))
            if entry.label == f"lit {name}/derived":
                leaf = LiteralLeaf(0, Operator.PLUS, 0)
                return Input("lit int 0/derived", leaf, pl.lit(0) + pl.lit(0))
        return entry


class NestedCheck:
    """Compare the rules with polars for one cell, and accept an answer that a sample shows.

    A kernel can cast an input with a strict cast before it checks the
    lengths of its inputs. On the typical sample, a value that does not fit
    the cast then raises first, with the class of the cast. A sample whose
    values fit shows the refusal of the lengths, which is the refusal of the
    types. A cast that is not strict turns text that does not parse into
    nulls, and `list.slice` then gives a list of `Null`. So where the run of
    the rules and the run of polars differ, the check runs each sample of
    values, and accepts the answer of the rules where one sample gives it.

    Example:
        >>> left = Inputs().labelled("List(Int64)/per row")
        >>> NestedCheck(Lists.LENGTH, lambda x: x.list.len()).problems([left])
        []
    """

    NAMES: ClassVar[tuple[str, ...]] = ("planner select", "planner agg", "run select", "run agg")

    def __init__(self, definition: Definition, build: Callable[..., pl.Expr]) -> None:
        """Keep the definition and the call of polars that builds the node.

        Args:
            definition: The definition.
            build: The call, with one expression for each input.

        Example:
            >>> NestedCheck(Lists.SUM, lambda x: x.list.sum())._definition.identity.function
            'ListExpr.Sum'
        """
        self._definition = definition
        self._build = build
        self._check = CellCheck(definition, build)

    def problems(self, inputs: list[Input]) -> list[str]:
        """Give each answer where polars and the rules differ, past a refusal that a sample shows.

        Args:
            inputs: One input for each input of the node.

        Returns:
            A description of each answer that differs.

        Example:
            >>> text = Inputs().labelled("String/per row")
            >>> NestedCheck(Lists.SUM, lambda x: x.list.sum()).problems([text])
            []
        """
        problems = self._check.problems(inputs)
        if not problems:
            return []
        node = self._build(*(entry.expression for entry in inputs))
        tree = json.loads(node.meta.serialize(format="json"))
        leaves = [entry.leaf for entry in inputs]
        rules = NodeEvaluation(self._definition.at(NodeOptions(tree)), leaves, CellCheck.GROUPS)
        answers = rules.answers()
        shown = {
            "run select": self._shown(node, agg=False, wanted=answers.run_select),
            "run agg": self._shown(node, agg=True, wanted=answers.run_agg),
        }
        return [problem for problem in problems if not self._accepted(problem, shown)]

    def _accepted(self, problem: str, shown: dict[str, bool]) -> bool:
        """Say whether a sample of values shows the answer of the rules in the answer of a problem.

        Args:
            problem: The description of the answer that differs.
            shown: For each run answer, whether a sample shows the answer of the rules.

        Returns:
            Whether the problem goes.

        Example:
            >>> check = NestedCheck(Lists.SUM, lambda x: x.list.sum())
            >>> check._accepted("a cell: run agg: polars raises, rules panics", {"run agg": True})
            True
        """
        return any(f": {name}: " in problem and seen for name, seen in shown.items())

    def _shown(self, node: pl.Expr, *, agg: bool, wanted: object) -> bool:
        """Say whether a sample of values gives the answer of the rules for a run.

        Args:
            node: The node in polars.
            agg: Whether the run is in `agg`.
            wanted: The answer of the rules for the run.

        Returns:
            Whether a sample gives the type of the rules, or refuses as the rules do.

        Example:
            >>> check = NestedCheck(Lists.SUM, lambda x: x.list.sum())
            >>> check._shown(pl.col("c0").list.sum(), agg=False, wanted=Unchecked("a"))
            False
        """
        expected: object = wanted
        if isinstance(wanted, Refused):
            error = None if wanted.error is None else wanted.error.value
            expected = Stopped(wanted.stop, error)
        elif not isinstance(wanted, DType):
            return False
        return any(self._run(node, values, agg=agg) == expected for values in Values)

    def _run(self, node: pl.Expr, values: Values, *, agg: bool) -> object:
        """Give the run answer of polars on one sample of values.

        Args:
            node: The node in polars.
            values: The sample.
            agg: Whether the run is in `agg`.

        Returns:
            The data type, or how the run stops.

        Example:
            >>> check = NestedCheck(Lists.SUM, lambda x: x.list.sum())
            >>> check._run(pl.col("c22").list.sum(), Values.TYPICAL, agg=False)
            Int64
        """
        frame = LiveFrames().frame(values).lazy()
        named = node.alias("o")
        query = frame.group_by("g", maintain_order=True).agg(named) if agg else frame.select(named)
        eager = pl.QueryOptFlags._eager()  # ruff: ignore[private-member-access]
        return PolarsAnswer().answer(lambda: query.collect(optimizations=eager)["o"].dtype)


class TripleSample:
    """Give a sample of the cells of a node of three inputs.

    The full product of the inputs holds more than 13 million cells. The sample
    puts each input at each place beside the typical inputs at the other
    places, then adds cells spread over the whole product, at fixed places so
    that each run checks the same cells.

    Example:
        >>> inputs = Inputs().all()
        >>> typical = (inputs[88], inputs[220], inputs[220])
        >>> len(TripleSample(inputs, typical).cells())
        736
    """

    SPREAD: ClassVar[int] = 28

    def __init__(self, inputs: list[Input], typical: tuple[Input, Input, Input]) -> None:
        """Keep the inputs and the typical input at each place.

        Args:
            inputs: The inputs that each place takes.
            typical: The typical input at each place.

        Example:
            >>> inputs = Inputs().all()
            >>> TripleSample(inputs, (inputs[0], inputs[1], inputs[2]))._typical[1].label
            'Int8/one value'
        """
        self._inputs = inputs
        self._typical = typical

    def cells(self) -> list[tuple[Input, ...]]:
        """Give the cells.

        Returns:
            The cells, each one input for each place.

        Example:
            >>> inputs = Inputs().all()
            >>> cells = TripleSample(inputs, (inputs[0], inputs[1], inputs[2])).cells()
            >>> [entry.label for entry in cells[0]]
            ['Int8/per row', 'Int8/one value', 'Int8/other length']
        """
        cells: list[tuple[Input, ...]] = []
        for place in range(3):
            for entry in self._inputs:
                cell = list(self._typical)
                cell[place] = entry
                cells.append(tuple(cell))
        count = len(self._inputs)
        for index in range(self.SPREAD):
            first = self._inputs[(index * 37 + 5) % count]
            second = self._inputs[(index * 101 + 11) % count]
            third = self._inputs[(index * 53 + 17) % count]
            cells.append((first, second, third))
        return cells


class NestedNodes:
    """Hold the nodes of the tests: each definition with the call of polars that builds its node.

    `arr.head`, `arr.slice` and `arr.tail` build the node of `list.slice` over
    the array as a list, so the node of `list.slice` stands for them. With
    `as_array=True` they build the node `ArrayExpr.Slice`.
    `concat_list` builds the node of `list.concat`, and `concat_arr` builds the
    node `ArrayExpr.Concat`, with any number of inputs.

    `list.sample` with replacement builds a list of the length of its count,
    whatever the length of the list. Its count leaves out each input whose
    values are large numbers (`LARGE`): the literal `2**40`, and the
    `Datetime`, `Duration` and `Time` columns, whose physical values are
    large. Such a count takes more memory than the machine holds, and the
    rules read only its type.

    Before polars 1.40.0, `list.sample` does not check the range of its
    fraction (`UNCHECKED`). A fraction of 300 or of 2**40 then asks for more
    values than a list holds, which is a fact of the data, so these cells
    take the literal 0 in its place (`FittingCounts`).

    Example:
        >>> sorted(NestedNodes.ONE_INPUT)[:3]
        ['arr.arg_max', 'arr.arg_min', 'arr.explode']
    """

    ALLOCATING: ClassVar[frozenset[str]] = frozenset({"list.sample with replacement"})
    UNCHECKED: ClassVar[frozenset[str]] = frozenset({"list.sample fraction"})
    LARGE: ClassVar[frozenset[str]] = frozenset(
        entry.label
        for entry in Inputs().all()
        if entry.label.startswith(("Datetime(", "Duration(", "Time/", "lit int 2**40"))
    )
    ONE_INPUT: ClassVar[dict[str, NestedCheck]] = {
        "list.len": NestedCheck(Lists.LENGTH, lambda x: x.list.len()),
        "list.sum": NestedCheck(Lists.SUM, lambda x: x.list.sum()),
        "list.mean": NestedCheck(Lists.MEAN, lambda x: x.list.mean()),
        "list.median": NestedCheck(Lists.MEDIAN, lambda x: x.list.median()),
        "list.std": NestedCheck(Lists.STD, lambda x: x.list.std()),
        "list.var": NestedCheck(Lists.VAR, lambda x: x.list.var()),
        "list.min": NestedCheck(Lists.MIN, lambda x: x.list.min()),
        "list.max": NestedCheck(Lists.MAX, lambda x: x.list.max()),
        "list.arg_min": NestedCheck(Lists.ARG_MIN, lambda x: x.list.arg_min()),
        "list.arg_max": NestedCheck(Lists.ARG_MAX, lambda x: x.list.arg_max()),
        "list.diff": NestedCheck(Lists.DIFF, lambda x: x.list.diff()),
        "list.sort": NestedCheck(Lists.SORT, lambda x: x.list.sort()),
        "list.drop_nulls": NestedCheck(Lists.DROP_NULLS, lambda x: x.list.drop_nulls()),
        "list.to_array": NestedCheck(Lists.TO_ARRAY, lambda x: x.list.to_array(2)),
        "list.to_struct": NestedCheck(Lists.TO_STRUCT, lambda x: x.list.to_struct(upper_bound=2)),
        "concat_list": NestedCheck(Lists.CONCAT, pl.concat_list),
        "arr.len": NestedCheck(Arrays.LENGTH, lambda x: x.arr.len()),
        "arr.min": NestedCheck(Arrays.MIN, lambda x: x.arr.min()),
        "arr.max": NestedCheck(Arrays.MAX, lambda x: x.arr.max()),
        "arr.sum": NestedCheck(Arrays.SUM, lambda x: x.arr.sum()),
        "arr.to_list": NestedCheck(Arrays.TO_LIST, lambda x: x.arr.to_list()),
        "arr.slice as an array": NestedCheck(
            Arrays.SLICE,
            lambda x: x.arr.slice(-1, 5, as_array=True),
        ),
        "arr.head as an array": NestedCheck(Arrays.SLICE, lambda x: x.arr.head(3, as_array=True)),
        "arr.std": NestedCheck(Arrays.STD, lambda x: x.arr.std()),
        "arr.var": NestedCheck(Arrays.VAR, lambda x: x.arr.var()),
        "arr.mean": NestedCheck(Arrays.MEAN, lambda x: x.arr.mean()),
        "arr.median": NestedCheck(Arrays.MEDIAN, lambda x: x.arr.median()),
        "arr.sort": NestedCheck(Arrays.SORT, lambda x: x.arr.sort()),
        "arr.arg_min": NestedCheck(Arrays.ARG_MIN, lambda x: x.arr.arg_min()),
        "arr.arg_max": NestedCheck(Arrays.ARG_MAX, lambda x: x.arr.arg_max()),
        "arr.explode": NestedCheck(Arrays.EXPLODE, lambda x: x.arr.explode()),
        "arr.to_struct": NestedCheck(Arrays.TO_STRUCT, lambda x: x.arr.to_struct()),
        "concat_arr": NestedCheck(Arrays.CONCAT, pl.concat_arr),
        "struct.json_encode": NestedCheck(Structs.JSON_ENCODE, lambda x: x.struct.json_encode()),
        "struct.rename_fields": NestedCheck(
            Structs.RENAME_FIELDS,
            lambda x: x.struct.rename_fields(["x"]),
        ),
        "struct.unnest": NestedCheck(Structs.SELECT_FIELDS, lambda x: x.struct.unnest()),
    }
    OPTIONS: ClassVar[dict[str, NestedCheck]] = {
        "list.std ddof 0": NestedCheck(Lists.STD, lambda x: x.list.std(ddof=0)),
        "list.var ddof 0": NestedCheck(Lists.VAR, lambda x: x.list.var(ddof=0)),
        "list.diff 2 drop": NestedCheck(Lists.DIFF, lambda x: x.list.diff(2, null_behavior="drop")),
        "list.sort descending": NestedCheck(Lists.SORT, lambda x: x.list.sort(descending=True)),
        "list.to_struct fields": NestedCheck(
            Lists.TO_STRUCT,
            lambda x: x.list.to_struct(fields=["a", "b", "c"]),
        ),
        "arr.std ddof 0": NestedCheck(Arrays.STD, lambda x: x.arr.std(ddof=0)),
        "arr.sort descending": NestedCheck(Arrays.SORT, lambda x: x.arr.sort(descending=True)),
        "arr.explode not empty as null": NestedCheck(
            Arrays.EXPLODE,
            lambda x: x.arr.explode(empty_as_null=False),
        ),
        "struct.rename_fields three": NestedCheck(
            Structs.RENAME_FIELDS,
            lambda x: x.struct.rename_fields(["x", "y", "z"]),
        ),
        "struct.field a b": NestedCheck(Structs.SELECT_FIELDS, lambda x: x.struct.field("a", "b")),
    }
    LISTS: ClassVar[str] = "List(Int64)/per row"
    ARRAYS: ClassVar[str] = "Array(Int64, shape=(2,))/per row"
    ONE: ClassVar[str] = "lit int 1/direct"
    LAST: ClassVar[str] = "lit int -1/direct"
    TWO_INPUTS: ClassVar[dict[str, tuple[NestedCheck, tuple[str, str], tuple[int, ...]]]] = {
        "list.get": (
            NestedCheck(Lists.GET, lambda x, y: x.list.get(y)),
            (LISTS, LAST),
            (300, 2**40),
        ),
        "list.shift": (NestedCheck(Lists.SHIFT, lambda x, y: x.list.shift(y)), (LISTS, ONE), ()),
        "list.join": (
            NestedCheck(Lists.JOIN, lambda x, y: x.list.join(y)),
            ("List(String)/per row", "lit str/direct"),
            (),
        ),
        "list.contains": (
            NestedCheck(Lists.CONTAINS, lambda x, y: x.list.contains(y)),
            (LISTS, ONE),
            (),
        ),
        "list.count_matches": (
            NestedCheck(Lists.COUNT_MATCHES, lambda x, y: x.list.count_matches(y)),
            (LISTS, ONE),
            (),
        ),
        "list.gather": (
            NestedCheck(Lists.GATHER, lambda x, y: x.list.gather(y)),
            (LISTS, LAST),
            (300, 2**40),
        ),
        "list.sample": (
            NestedCheck(Lists.SAMPLE, lambda x, y: x.list.sample(y)),
            (LISTS, ONE),
            (300,),
        ),
        "list.set_union": (
            NestedCheck(Lists.SET_OPERATION, lambda x, y: x.list.set_union(y)),
            (LISTS, LISTS),
            (),
        ),
        "list.concat": (
            NestedCheck(Lists.CONCAT, lambda x, y: x.list.concat(y)),
            (LISTS, ONE),
            (),
        ),
        "arr.get": (NestedCheck(Arrays.GET, lambda x, y: x.arr.get(y)), (ARRAYS, LAST), ()),
        "arr.shift": (NestedCheck(Arrays.SHIFT, lambda x, y: x.arr.shift(y)), (ARRAYS, ONE), ()),
        "arr.join": (
            NestedCheck(Arrays.JOIN, lambda x, y: x.arr.join(y)),
            (ARRAYS, "lit str/direct"),
            (),
        ),
        "arr.contains": (
            NestedCheck(Arrays.CONTAINS, lambda x, y: x.arr.contains(y)),
            (ARRAYS, ONE),
            (),
        ),
        "arr.count_matches": (
            NestedCheck(Arrays.COUNT_MATCHES, lambda x, y: x.arr.count_matches(y)),
            (ARRAYS, ONE),
            (),
        ),
        "concat_arr": (
            NestedCheck(Arrays.CONCAT, pl.concat_arr),
            (ARRAYS, "Int64/per row"),
            (),
        ),
    }
    TWO_OPTIONS: ClassVar[dict[str, tuple[NestedCheck, tuple[str, str]]]] = {
        "list.first": (
            NestedCheck(Lists.GET, lambda x, y: x.list.get(y, null_on_oob=True)),
            (LISTS, LAST),
        ),
        "list.gather null on oob": (
            NestedCheck(Lists.GATHER, lambda x, y: x.list.gather(y, null_on_oob=True)),
            (LISTS, LAST),
        ),
        "list.sample fraction": (
            NestedCheck(Lists.SAMPLE, lambda x, y: x.list.sample(fraction=y)),
            (LISTS, "lit float/direct"),
        ),
        "list.sample with replacement": (
            NestedCheck(Lists.SAMPLE, lambda x, y: x.list.sample(y, with_replacement=True)),
            (LISTS, ONE),
        ),
        "list.join keep nulls": (
            NestedCheck(Lists.JOIN, lambda x, y: x.list.join(y, ignore_nulls=False)),
            ("List(String)/per row", "lit str/direct"),
        ),
        "list.contains nulls apart": (
            NestedCheck(Lists.CONTAINS, lambda x, y: x.list.contains(y, nulls_equal=False)),
            (LISTS, ONE),
        ),
        "list.set_difference": (
            NestedCheck(Lists.SET_OPERATION, lambda x, y: x.list.set_difference(y)),
            (LISTS, LISTS),
        ),
        "list.set_intersection": (
            NestedCheck(Lists.SET_OPERATION, lambda x, y: x.list.set_intersection(y)),
            (LISTS, LISTS),
        ),
        "list.set_symmetric_difference": (
            NestedCheck(Lists.SET_OPERATION, lambda x, y: x.list.set_symmetric_difference(y)),
            (LISTS, LISTS),
        ),
        "arr.first": (
            NestedCheck(Arrays.GET, lambda x, y: x.arr.get(y, null_on_oob=True)),
            (ARRAYS, LAST),
        ),
        "arr.join keep nulls": (
            NestedCheck(Arrays.JOIN, lambda x, y: x.arr.join(y, ignore_nulls=False)),
            (ARRAYS, "lit str/direct"),
        ),
    }
    THREE_INPUTS: ClassVar[dict[str, tuple[NestedCheck, tuple[str, str, str]]]] = {
        "list.slice": (
            NestedCheck(Lists.SLICE, lambda x, y, z: x.list.slice(y, z)),
            (LISTS, ONE, ONE),
        ),
        "list.gather_every": (
            NestedCheck(Lists.GATHER_EVERY, lambda x, y, z: x.list.gather_every(y, z)),
            (LISTS, ONE, ONE),
        ),
    }

    @classmethod
    def pairs(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give the cells of the nodes of two inputs: each node beside each pair of its sample.

        A node at another value of an option takes the pairs where one input
        is a column in one value per row or a direct literal, beside the
        typical input at the other place.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> cells = NestedNodes.pairs()
            >>> cells[0][0], len(cells[0][1])
            ('list.get', 2)
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (_, labels, outside) in cls.TWO_INPUTS.items():
            inputs = FittingCounts().inputs(outside)
            left, right = Inputs().labelled(labels[0]), Inputs().labelled(labels[1])
            cells.extend((name, pair) for pair in PairSample(inputs, left, right).pairs())
        for name, (_, labels) in cls.TWO_OPTIONS.items():
            left, right = Inputs().labelled(labels[0]), Inputs().labelled(labels[1])
            outside = cls.outside(name)
            for (entry,) in NodeSample().options():
                cells.append((name, (entry, right)))
                if name not in cls.ALLOCATING or entry.label not in cls.LARGE:
                    cells.append((name, (left, FittingCounts().fitted(entry, outside))))
        return cells

    @classmethod
    def outside(cls, name: str) -> tuple[int, ...]:
        """Give the values of the literal classes outside the lists, for a node of an option.

        Args:
            name: The name of the node.

        Returns:
            The values: 300 and 2**40 for a fraction that the installed
            release does not check, and none otherwise.

        Example:
            >>> NestedNodes.outside("list.first")
            ()
        """
        older = PolarsRelease.installed() < PolarsRelease.parse("1.40.0")
        return (300, 2**40) if older and name in cls.UNCHECKED else ()

    @classmethod
    def triples(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give the cells of the nodes of three inputs: each node beside each cell of its sample.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> cells = NestedNodes.triples()
            >>> len(cells), cells[0][0]
            (1472, 'list.slice')
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        inputs = Inputs().all()
        for name, (_, labels) in cls.THREE_INPUTS.items():
            first, second, third = (Inputs().labelled(label) for label in labels)
            sample = TripleSample(inputs, (first, second, third))
            cells.extend((name, cell) for cell in sample.cells())
        return cells

    @classmethod
    def check(cls, name: str) -> NestedCheck:
        """Give the check of a node of two or three inputs.

        Args:
            name: The name of the node.

        Returns:
            The check.

        Example:
            >>> type(NestedNodes.check("list.slice")).__name__
            'NestedCheck'
        """
        if name in cls.TWO_INPUTS:
            return cls.TWO_INPUTS[name][0]
        if name in cls.TWO_OPTIONS:
            return cls.TWO_OPTIONS[name][0]
        return cls.THREE_INPUTS[name][0]


@pytest.mark.parametrize("node", list(NestedNodes.ONE_INPUT))
@pytest.mark.parametrize("cell", NodeSample().cells(1), ids=CellCheck.label)
def test_nested_node_of_one_input_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a nested function of one input match polars, at one input."""
    problems = NestedNodes.ONE_INPUT[node].problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("node", list(NestedNodes.OPTIONS))
@pytest.mark.parametrize("cell", NodeSample().options(), ids=CellCheck.label)
def test_nested_option_gives_the_answers_of_polars(node: str, cell: tuple[Input, ...]) -> None:
    """The four answers of a nested function at another value of an option match polars."""
    problems = NestedNodes.OPTIONS[node].problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("node", "cell"),
    NestedNodes.pairs(),
    ids=[f"{node}-{CellCheck.label(cell)}" for node, cell in NestedNodes.pairs()],
)
def test_nested_node_of_two_inputs_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a nested function of two inputs match polars, at one cell."""
    problems = NestedNodes.check(node).problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("node", "cell"),
    NestedNodes.triples(),
    ids=[f"{node}-{CellCheck.label(cell)}" for node, cell in NestedNodes.triples()],
)
def test_nested_node_of_three_inputs_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a nested function of three inputs match polars, at one cell."""
    problems = NestedNodes.check(node).problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("label", ["Struct({'a': Int8})/per row", "Struct({'a': Date})/one value"])
def test_unnest_that_picks_no_field_is_unchecked(label: str) -> None:
    """The rules give `Unchecked` for a `struct.field` whose selector picks no field.

    Polars expands such a node into no column, so the frame has no column of
    the name of the node.
    """
    entry = Inputs().labelled(label)
    node = entry.expression.struct.field("^z$")
    tree = json.loads(node.meta.serialize(format="json"))
    definition = Structs.SELECT_FIELDS.at(NodeOptions(tree))
    answers = NodeEvaluation(definition, [entry.leaf], CellCheck.GROUPS).answers()
    every = [answers.planner_select, answers.planner_agg, answers.run_select, answers.run_agg]
    assert all(isinstance(answer, Unchecked) for answer in every), every
    planned = PolarsCell(node.alias("o")).planner(agg=False)
    assert planned == Stopped(Stop.RAISES, "KeyError")
