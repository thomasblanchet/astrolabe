"""Check the definitions of the temporal family against polars, node by node.

The family holds the functions of the `dt` namespace, such as `dt.year`,
`dt.truncate` and `dt.replace`, the functions of business days:
`dt.add_business_days`, `dt.is_business_day` and `pl.business_day_count`, and
`pl.datetime` and `pl.duration`, which build a moment or a duration from its
parts. Polars builds `pl.datetime` and `pl.duration` of only literals as one
literal, so their samples leave out those cells.
Each test builds the node of polars over the inputs of one cell, and compares
the four answers of polars with the four answers of the rules (`CellCheck`):
the planner type and the run type, in a `select` and in `agg` over two groups
of other sizes.

A node of one input takes every input (`NodeSample`). A node of two inputs
takes about 500 pairs beside a typical input at each place (`PairSample`). A
node of three inputs or more takes about 500 cells of its own sample
(`SpreadSample`). A call that differs from the main call of its node only by
an option takes a smaller sample. A few cells check a literal whose value the
kernel reads, such as the duration of `dt.truncate` or a part of `dt.replace`.

The duration of `dt.truncate`, `dt.round` and `dt.offset_by` is text. The
literal text of their sample is `1d`, and the frames hold a sample whose text
is `1d` in every row (`Values.DURATION_TEXT`). The text `ambiguous` of
`dt.replace_time_zone` and `dt.replace` is `raise` in the same way
(`Values.AMBIGUOUS_TEXT`). The holidays of a function of business days are a
`List(Date)` column.

Each test names the node and the cell, so a failure names both.
"""

from __future__ import annotations

import datetime
from typing import ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import (
    ApiGate,
    CellCheck,
    Input,
    Inputs,
    NodeSample,
    PairSample,
    PolarsCell,
    TemplateCall,
    TreeFrame,
)
from astrolabe._rules.definitions.temporal import Temporal
from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.node_evaluation import ColumnLeaf, LiteralLeaf, NodeEvaluation


class SpreadSample:
    """Give a sample of about 500 cells of a node of three inputs or more.

    The full product of the inputs grows with the power of the number of
    inputs, so the sample takes three parts. The first place takes every
    input, beside the typical input at each other place. Each other place
    takes each input of a smaller set, beside the typical inputs. Then the
    sample adds cells spread over the whole product, at fixed places, so that
    each run checks the same cells.

    Example:
        >>> inputs = Inputs().all()
        >>> sample = SpreadSample(inputs, inputs[:3], NodeSample().options())
        >>> len(sample.cells(138))
        500
    """

    STEPS: ClassVar[tuple[int, ...]] = (37, 101, 61, 89, 53, 73, 97, 43, 67)
    OFFSETS: ClassVar[tuple[int, ...]] = (5, 11, 17, 23, 29, 31, 41, 47, 59)

    def __init__(
        self,
        inputs: list[Input],
        typical: list[Input] | tuple[Input, ...],
        others: list[tuple[Input]],
    ) -> None:
        """Keep the inputs, the typical input of each place, and the smaller set.

        Args:
            inputs: The inputs that the first place and the spread cells take.
            typical: The typical input of each place.
            others: The smaller set of each other place, one input in each tuple.

        Example:
            >>> inputs = Inputs().all()
            >>> len(SpreadSample(inputs, inputs[:3], [])._typical)
            3
        """
        self._inputs = inputs
        self._typical = tuple(typical)
        self._others = [entry for (entry,) in others]

    def cells(self, spread: int) -> list[tuple[Input, ...]]:
        """Give the cells.

        Args:
            spread: The number of cells spread over the whole product.

        Returns:
            The cells, each with one input for each place.

        Example:
            >>> inputs = Inputs().all()
            >>> cells = SpreadSample(inputs, inputs[:3], []).cells(0)
            >>> len(cells), cells[1][0].label
            (236, 'Int8/one value')
        """
        cells = [self._at(0, entry) for entry in self._inputs]
        for place in range(1, len(self._typical)):
            cells.extend(self._at(place, entry) for entry in self._others)
        count = len(self._inputs)
        for index in range(spread):
            cell = []
            for place in range(len(self._typical)):
                step, offset = self.STEPS[place], self.OFFSETS[place]
                cell.append(self._inputs[(index * step + offset) % count])
            cells.append(tuple(cell))
        return cells

    def _at(self, place: int, entry: Input) -> tuple[Input, ...]:
        """Give the cell with one input at one place and the typical input at each other place.

        Args:
            place: The place.
            entry: The input at that place.

        Returns:
            The cell.

        Example:
            >>> inputs = Inputs().all()
            >>> [item.label for item in SpreadSample(inputs, inputs[:2], [])._at(1, inputs[5])]
            ['Int8/per row', 'UInt8/one value']
        """
        cell = list(self._typical)
        cell[place] = entry
        return tuple(cell)


class TemporalNodes:
    """Hold the nodes of the tests, each with its call and the typical inputs of its sample.

    A function of one input takes every input. The duration of `truncate`,
    `round` and `offset_by` takes the literal `1d` beside a `Datetime` column
    in milliseconds. `replace_time_zone` takes a column in the zone
    `Europe/Paris` beside the literal `raise`, so that its kernel reads
    `ambiguous`. `combine` takes a `Date` column and a `Time` column. The
    functions of business days take a `Date` column, the literal `1` as `n`,
    and a `List(Date)` column of holidays. `replace` takes a `Datetime`
    column, a null at each part, and the literal `raise`, as `dt.replace()`
    does: its other places take each column of a census kind and each literal.

    Example:
        >>> sorted(TemporalNodes.ONE_INPUT)[:3]
        ['base_utc_offset', 'century', 'convert_time_zone']
        >>> TemporalNodes.DAY.label
        'lit 1d/direct'
    """

    DATES: ClassVar[Input] = Inputs().labelled("Date/per row")
    MOMENTS: ClassVar[Input] = Inputs().labelled("Datetime(time_unit='ms', time_zone=None)/per row")
    ZONED: ClassVar[Input] = Inputs().labelled(
        "Datetime(time_unit='ns', time_zone='Europe/Paris')/per row",
    )
    TIMES: ClassVar[Input] = Inputs().labelled("Time/per row")
    HOLIDAYS: ClassVar[Input] = Inputs().labelled("List(Date)/per row")
    ONE: ClassVar[Input] = Inputs().labelled("lit int 1/direct")
    NULL: ClassVar[Input] = Inputs().labelled("lit null/direct")
    DAY: ClassVar[Input] = Input("lit 1d/direct", LiteralLeaf("1d"), pl.lit("1d"))
    RAISE: ClassVar[Input] = Input("lit raise/direct", LiteralLeaf("raise"), pl.lit("raise"))
    REPLACE: ClassVar[str] = "pl.col('x').dt.replace()"
    # The kinds besides the temporal ones whose cast to a part of `replace`
    # reads integers that the sample does not choose.
    BY_VALUES: ClassVar[frozenset[str]] = frozenset({"Struct", "Categorical"})
    INTEGERS: ClassVar[Input] = Inputs().labelled("Int64/per row")

    ONE_INPUT: ClassVar[dict[str, CellCheck]] = ApiGate(prefix="dt.").kept({
        "millennium": CellCheck(Temporal.MILLENNIUM, lambda x: x.dt.millennium()),
        "century": CellCheck(Temporal.CENTURY, lambda x: x.dt.century()),
        "year": CellCheck(Temporal.YEAR, lambda x: x.dt.year()),
        "iso_year": CellCheck(Temporal.ISO_YEAR, lambda x: x.dt.iso_year()),
        "is_leap_year": CellCheck(Temporal.IS_LEAP_YEAR, lambda x: x.dt.is_leap_year()),
        "quarter": CellCheck(Temporal.QUARTER, lambda x: x.dt.quarter()),
        "month": CellCheck(Temporal.MONTH, lambda x: x.dt.month()),
        "days_in_month": CellCheck(Temporal.DAYS_IN_MONTH, lambda x: x.dt.days_in_month()),
        "week": CellCheck(Temporal.WEEK, lambda x: x.dt.week()),
        "weekday": CellCheck(Temporal.WEEKDAY, lambda x: x.dt.weekday()),
        "day": CellCheck(Temporal.DAY, lambda x: x.dt.day()),
        "ordinal_day": CellCheck(Temporal.ORDINAL_DAY, lambda x: x.dt.ordinal_day()),
        "hour": CellCheck(Temporal.HOUR, lambda x: x.dt.hour()),
        "minute": CellCheck(Temporal.MINUTE, lambda x: x.dt.minute()),
        "second": CellCheck(Temporal.SECOND, lambda x: x.dt.second()),
        "millisecond": CellCheck(Temporal.MILLISECOND, lambda x: x.dt.millisecond()),
        "microsecond": CellCheck(Temporal.MICROSECOND, lambda x: x.dt.microsecond()),
        "nanosecond": CellCheck(Temporal.NANOSECOND, lambda x: x.dt.nanosecond()),
        "time": CellCheck(Temporal.TIME, lambda x: x.dt.time()),
        "date": CellCheck(Temporal.DATE, lambda x: x.dt.date()),
        "datetime": CellCheck(Temporal.DATETIME, lambda x: x.dt.datetime()),
        "month_start": CellCheck(Temporal.MONTH_START, lambda x: x.dt.month_start()),
        "month_end": CellCheck(Temporal.MONTH_END, lambda x: x.dt.month_end()),
        "base_utc_offset": CellCheck(Temporal.BASE_UTC_OFFSET, lambda x: x.dt.base_utc_offset()),
        "dst_offset": CellCheck(Temporal.DST_OFFSET, lambda x: x.dt.dst_offset()),
        "timestamp": CellCheck(Temporal.TIMESTAMP, lambda x: x.dt.timestamp()),
        "epoch": CellCheck(Temporal.TIMESTAMP, lambda x: x.dt.epoch()),
        "to_string": CellCheck(Temporal.TO_STRING, lambda x: x.dt.to_string()),
        "strftime": CellCheck(Temporal.TO_STRING, lambda x: x.dt.strftime("%Y")),
        "total_days": CellCheck(Temporal.TOTAL_DAYS, lambda x: x.dt.total_days()),
        "total_hours": CellCheck(Temporal.TOTAL_HOURS, lambda x: x.dt.total_hours()),
        "total_minutes": CellCheck(Temporal.TOTAL_MINUTES, lambda x: x.dt.total_minutes()),
        "total_seconds": CellCheck(Temporal.TOTAL_SECONDS, lambda x: x.dt.total_seconds()),
        "total_milliseconds": CellCheck(
            Temporal.TOTAL_MILLISECONDS,
            lambda x: x.dt.total_milliseconds(),
        ),
        "total_microseconds": CellCheck(
            Temporal.TOTAL_MICROSECONDS,
            lambda x: x.dt.total_microseconds(),
        ),
        "total_nanoseconds": CellCheck(
            Temporal.TOTAL_NANOSECONDS,
            lambda x: x.dt.total_nanoseconds(),
        ),
        "with_time_unit": CellCheck(Temporal.WITH_TIME_UNIT, lambda x: x.dt.with_time_unit("ms")),
        "convert_time_zone": CellCheck(
            Temporal.CONVERT_TIME_ZONE,
            lambda x: x.dt.convert_time_zone("UTC"),
        ),
    })
    ONE_INPUT_OPTIONS: ClassVar[dict[str, CellCheck]] = ApiGate(prefix="dt.").kept({
        "timestamp ms": CellCheck(Temporal.TIMESTAMP, lambda x: x.dt.timestamp("ms")),
        "epoch ns": CellCheck(Temporal.TIMESTAMP, lambda x: x.dt.epoch("ns")),
        "to_string iso:strict": CellCheck(
            Temporal.TO_STRING,
            lambda x: x.dt.to_string("iso:strict"),
        ),
        "to_string polars": CellCheck(Temporal.TO_STRING, lambda x: x.dt.to_string("polars")),
        "strftime %H:%M": CellCheck(Temporal.TO_STRING, lambda x: x.dt.strftime("%H:%M")),
        "strftime %s %z": CellCheck(Temporal.TO_STRING, lambda x: x.dt.strftime("%s %z")),
        "strftime %Q": CellCheck(Temporal.TO_STRING, lambda x: x.dt.strftime("%Q")),
        "strftime %-d %%": CellCheck(Temporal.TO_STRING, lambda x: x.dt.strftime("%-d %%")),
        "total_days fractional": CellCheck(
            Temporal.TOTAL_DAYS,
            lambda x: x.dt.total_days(fractional=True),
        ),
        "total_seconds fractional": CellCheck(
            Temporal.TOTAL_SECONDS,
            lambda x: x.dt.total_seconds(fractional=True),
        ),
        "with_time_unit ns": CellCheck(
            Temporal.WITH_TIME_UNIT,
            lambda x: x.dt.with_time_unit("ns"),
        ),
        "convert_time_zone Asia/Tokyo": CellCheck(
            Temporal.CONVERT_TIME_ZONE,
            lambda x: x.dt.convert_time_zone("Asia/Tokyo"),
        ),
    })
    TWO_INPUTS: ClassVar[dict[str, tuple[CellCheck, tuple[Input, Input], str]]] = {
        "truncate": (
            CellCheck(Temporal.TRUNCATE, lambda x, y: x.dt.truncate(y)),
            (MOMENTS, DAY),
            "1d",
        ),
        "round": (CellCheck(Temporal.ROUND, lambda x, y: x.dt.round(y)), (MOMENTS, DAY), "1d"),
        "offset_by": (
            CellCheck(Temporal.OFFSET_BY, lambda x, y: x.dt.offset_by(y)),
            (MOMENTS, DAY),
            "1d",
        ),
        "replace_time_zone": (
            CellCheck(
                Temporal.REPLACE_TIME_ZONE,
                lambda x, y: x.dt.replace_time_zone("UTC", ambiguous=y),
            ),
            (ZONED, RAISE),
            "raise",
        ),
        "combine": (
            CellCheck(Temporal.COMBINE, lambda x, y: x.dt.combine(y)),
            (DATES, TIMES),
            "a",
        ),
        "is_business_day": (
            CellCheck(Temporal.IS_BUSINESS_DAY, lambda x, y: x.dt.is_business_day(holidays=y)),
            (DATES, HOLIDAYS),
            "a",
        ),
    }
    TWO_INPUT_OPTIONS: ClassVar[dict[str, tuple[CellCheck, tuple[Input, Input], str]]] = {
        "replace_time_zone to no zone": (
            CellCheck(
                Temporal.REPLACE_TIME_ZONE,
                lambda x, y: x.dt.replace_time_zone(None, ambiguous=y),
            ),
            (ZONED, RAISE),
            "raise",
        ),
        "replace_time_zone Asia/Tokyo null": (
            CellCheck(
                Temporal.REPLACE_TIME_ZONE,
                lambda x, y: x.dt.replace_time_zone("Asia/Tokyo", ambiguous=y, non_existent="null"),
            ),
            (MOMENTS, RAISE),
            "raise",
        ),
        "combine ns": (
            CellCheck(Temporal.COMBINE, lambda x, y: x.dt.combine(y, time_unit="ns")),
            (ZONED, TIMES),
            "a",
        ),
        "is_business_day weekend": (
            CellCheck(
                Temporal.IS_BUSINESS_DAY,
                lambda x, y: x.dt.is_business_day(week_mask=[False] * 5 + [True] * 2, holidays=y),
            ),
            (MOMENTS, HOLIDAYS),
            "a",
        ),
    }
    MANY_INPUTS: ClassVar[dict[str, tuple[CellCheck, tuple[Input, ...]]]] = {
        "add_business_days": (
            CellCheck(
                Temporal.ADD_BUSINESS_DAYS,
                lambda x, n, h: x.dt.add_business_days(n, holidays=h, roll="forward"),
            ),
            (DATES, ONE, HOLIDAYS),
        ),
        "business_day_count": (
            CellCheck(
                Temporal.BUSINESS_DAY_COUNT,
                lambda x, y, h: pl.business_day_count(x, y, holidays=h),
            ),
            (DATES, DATES, HOLIDAYS),
        ),
        "replace": (
            CellCheck(Temporal.REPLACE, TemplateCall(REPLACE)),
            (MOMENTS, NULL, NULL, NULL, NULL, NULL, NULL, NULL, RAISE),
        ),
        "pl.datetime": (
            CellCheck(
                Temporal.DATETIME_FUNCTION,
                lambda *parts: pl.datetime(*parts[:7], ambiguous=parts[7]).meta.undo_aliases(),
            ),
            (INTEGERS, ONE, ONE, ONE, ONE, ONE, ONE, RAISE),
        ),
        "pl.duration": (
            CellCheck(
                Temporal.DURATION,
                lambda w, d, h, m, s, ms, us, ns: pl.duration(
                    weeks=w,
                    days=d,
                    hours=h,
                    minutes=m,
                    seconds=s,
                    milliseconds=ms,
                    microseconds=us,
                    nanoseconds=ns,
                ),
            ),
            (ONE, INTEGERS, ONE, ONE, ONE, ONE, ONE, ONE),
        ),
    }

    @classmethod
    def one_input_cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node of one input beside each cell of its sample.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> one, options = len(NodeSample().cells(1)), len(NodeSample().options())
            >>> nodes = len(TemporalNodes.ONE_INPUT), len(TemporalNodes.ONE_INPUT_OPTIONS)
            >>> len(TemporalNodes.one_input_cells()) == one * nodes[0] + options * nodes[1]
            True
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name in cls.ONE_INPUT:
            cells.extend((name, cell) for cell in NodeSample().cells(1))
        for name in cls.ONE_INPUT_OPTIONS:
            cells.extend((name, cell) for cell in NodeSample().options())
        return cells

    @classmethod
    def two_input_cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node of two inputs beside each cell of its sample.

        The sample of a node puts its text in place of the text literal of the
        census. A call at another value of an option takes the typical input
        at one place beside each input of the smaller sample at the other
        place.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> len(TemporalNodes.two_input_cells())
            3504
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (_, typical, text) in cls.TWO_INPUTS.items():
            inputs = Inputs().with_text(text)
            cells.extend((name, cell) for cell in PairSample(inputs, *typical).pairs())
        for name, (_, (left, right), text) in cls.TWO_INPUT_OPTIONS.items():
            options = [cell[0] for cell in NodeSample(Inputs().with_text(text)).options()]
            cells.extend((name, (entry, right)) for entry in options)
            cells.extend((name, (left, entry)) for entry in options)
        return cells

    @classmethod
    def many_input_cells(cls, *, values: bool = False) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node of three inputs or more beside each cell of its sample.

        The other places of a node of three inputs take each column of one
        value per row and each direct literal. The eight parts of `replace`
        take each column of a census kind and each direct literal, so that its
        sample stays near 500 cells, and 60 cells spread over the product. A
        temporal column at a part of `replace` casts its physical integers, a
        `Categorical` column casts the ids of its categories, and a struct
        casts each of its fields. No sample of values holds integers that name
        a part there, and the ids of the categories depend on what the process
        did before, so such a cell runs on no sample, or on some samples only.
        The first error of a cell that the rules refuse is the error of a
        value. These cells are a fact of the values, and have a test of their
        own.

        Args:
            values: Whether to give only the cells whose run the values decide.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> len(TemporalNodes.many_input_cells())
            2540
            >>> len(TemporalNodes.many_input_cells(values=True))
            94
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (_, typical) in cls.MANY_INPUTS.items():
            cells.extend((name, cell) for cell in cls._node_cells(name, typical))
        kept = [(name, cell) for name, cell in cells if not cls._folded(name, cell)]
        return [(name, cell) for name, cell in kept if cls._by_values(name, cell) is values]

    @classmethod
    def _folded(cls, name: str, cell: tuple[Input, ...]) -> bool:
        """Say whether polars builds a literal in the place of the node of a cell.

        `pl.datetime` and `pl.duration` of only literals build their value as
        one literal, so the cell has no node of its function.

        Args:
            name: The name of the node.
            cell: The cell.

        Returns:
            Whether the cell has no node.

        Example:
            >>> TemporalNodes._folded("pl.duration", (TemporalNodes.ONE,) * 8)
            True
        """
        literals = all(entry.label.startswith("lit ") for entry in cell)
        return name in {"pl.datetime", "pl.duration"} and literals

    @classmethod
    def _node_cells(cls, name: str, typical: tuple[Input, ...]) -> list[tuple[Input, ...]]:
        """Give the cells of the sample of one node of three inputs or more.

        Args:
            name: The name of the node.
            typical: The typical input at each place.

        Returns:
            The cells.

        Example:
            >>> _, typical = TemporalNodes.MANY_INPUTS["replace"]
            >>> len(TemporalNodes._node_cells("replace", typical)) > 500
            True
        """
        inputs = Inputs().with_text("raise" if name == "replace" else "a")
        if len(typical) > 3:
            others = [cell for cell in NodeSample().options() if cls._census_kind(cell[0])]
        else:
            others = NodeSample(inputs).options()
        spread = max(500 - len(inputs) - len(others) * (len(typical) - 1), 60)
        return SpreadSample(inputs, typical, others).cells(spread)

    @classmethod
    def _by_values(cls, name: str, cell: tuple[Input, ...]) -> bool:
        """Say whether a part of a cell of `replace` is a temporal, `Categorical` or struct column.

        Args:
            name: The name of the node.
            cell: The cell.

        Returns:
            Whether it does.

        Example:
            >>> _, typical = TemporalNodes.MANY_INPUTS["replace"]
            >>> cell = (typical[0], typical[1], TemporalNodes.DATES, *typical[3:])
            >>> TemporalNodes._by_values("replace", cell)
            True
        """
        if name != "replace":
            return False
        leaves = [entry.leaf for entry in cell[1:8]]
        kinds = [leaf.dtype for leaf in leaves if isinstance(leaf, ColumnLeaf)]
        return any(Is.temporal(dtype) or dtype.name in cls.BY_VALUES for dtype in kinds)

    @classmethod
    def _census_kind(cls, entry: Input) -> bool:
        """Say whether an input is a literal, or a column of one of the 27 kinds of the census.

        Args:
            entry: The input.

        Returns:
            Whether it is.

        Example:
            >>> TemporalNodes._census_kind(TemporalNodes.HOLIDAYS)
            False
        """
        expression = str(entry.expression)
        if not expression.startswith('col("c'):
            return True
        return int(expression[6:].split('"', maxsplit=1)[0]) < 27

    @classmethod
    def check(cls, name: str) -> CellCheck:
        """Give the comparison of the rules with polars for the node of a name.

        Args:
            name: The name of the node.

        Returns:
            The comparison.

        Example:
            >>> TemporalNodes.check("replace").label(())
            ''
        """
        checks = {**cls.ONE_INPUT, **cls.ONE_INPUT_OPTIONS}
        for nodes in (cls.TWO_INPUTS, cls.TWO_INPUT_OPTIONS):
            checks.update({key: check for key, (check, _, _) in nodes.items()})
        checks.update({key: check for key, (check, _) in cls.MANY_INPUTS.items()})
        return checks[name]

    @classmethod
    def ids(cls, cells: list[tuple[str, tuple[Input, ...]]]) -> list[str]:
        """Give the identifier of each test: the node and the cell.

        Args:
            cells: The name of the node and the cell, for each cell.

        Returns:
            The identifiers.

        Example:
            >>> TemporalNodes.ids([("truncate", (TemporalNodes.DATES, TemporalNodes.DAY))])
            ['truncate-Date/per row, lit 1d/direct']
        """
        return [f"{name}-{CellCheck.label(cell)}" for name, cell in cells]


class LiteralCells:
    """Hold the cells whose answer depends on the value of a literal input, which the code holds.

    The kernels of `truncate`, `round` and `offset_by` parse a literal
    duration at every run. The kernel of `replace_time_zone` reads a literal
    `ambiguous`, and the kernel of `replace` casts each literal part strictly
    and checks that it names a moment. Each cell names the node, its literal
    and the moment beside it.

    Example:
        >>> len(LiteralCells.cells()) > 50
        True
    """

    MOMENTS: ClassVar[tuple[str, ...]] = (
        "Date/per row",
        "Datetime(time_unit='ms', time_zone=None)/per row",
        "Datetime(time_unit='ms', time_zone='UTC')/per row",
        "Datetime(time_unit='ns', time_zone='Europe/Paris')/per row",
        "Datetime(time_unit='ns', time_zone=None)/per row",
    )
    DURATIONS: ClassVar[tuple[str, ...]] = ("", "0d", "1ns", "-1d", "1mo", "1d1h", "+2w", "1x")
    AMBIGUOUS: ClassVar[tuple[str, ...]] = ("earliest", "null", "a")
    PARTS: ClassVar[tuple[tuple[int, int | float | bool], ...]] = (
        (1, 300),
        (1, 1_000_000),
        (2, 13),
        (2, 0),
        (3, 31),
        (3, 1.5),
        (4, 24),
        (5, True),
        (6, -1),
        (7, 999_999),
        (7, 2_000_000),
    )

    @classmethod
    def cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each cell: the name of the node and the inputs.

        Returns:
            The cells.

        Example:
            >>> LiteralCells.cells()[0][0]
            'truncate'
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for label in cls.MOMENTS:
            cells.extend(cls._beside(Inputs().labelled(label)))
        return cells

    @classmethod
    def _beside(cls, moment: Input) -> list[tuple[str, tuple[Input, ...]]]:
        """Give the cells of one moment: each literal of each node beside it.

        Args:
            moment: The moment.

        Returns:
            The cells.

        Example:
            >>> cells = LiteralCells._beside(TemporalNodes.DATES)
            >>> cells[-1][0], len(cells[-1][1])
            ('replace', 9)
        """
        durations = [
            (name, (moment, cls._text(text)))
            for name in ("truncate", "round", "offset_by")
            for text in cls.DURATIONS
        ]
        ambiguous = [("replace_time_zone", (moment, cls._text(text))) for text in cls.AMBIGUOUS]
        parts = [("replace", cls._replaced(moment, place, value)) for place, value in cls.PARTS]
        return durations + ambiguous + parts

    @classmethod
    def _text(cls, text: str) -> Input:
        """Give the input of a literal text.

        Args:
            text: The text.

        Returns:
            The input.

        Example:
            >>> LiteralCells._text("1d").label
            "lit '1d'/direct"
        """
        return Input(f"lit {text!r}/direct", LiteralLeaf(text), pl.lit(text))

    @classmethod
    def _replaced(cls, moment: Input, place: int, value: int | float | bool) -> tuple[Input, ...]:
        """Give the inputs of `replace` with one literal part.

        Args:
            moment: The moment.
            place: The place of the part.
            value: The value of the part.

        Returns:
            The inputs: the moment, the parts, and the literal `raise`.

        Example:
            >>> cell = LiteralCells._replaced(TemporalNodes.DATES, 2, 13)
            >>> [entry.label for entry in cell][1:3]
            ['lit null/direct', 'lit 13/direct']
        """
        _, typical = TemporalNodes.MANY_INPUTS["replace"]
        cell = [moment, *typical[1:]]
        cell[place] = Input(f"lit {value!r}/direct", LiteralLeaf(value), pl.lit(value))
        return tuple(cell)


@pytest.mark.filterwarnings("ignore:`dt.(datetime|with_time_unit)` is deprecated")
@pytest.mark.parametrize(
    ("node", "cell"),
    TemporalNodes.one_input_cells(),
    ids=TemporalNodes.ids(TemporalNodes.one_input_cells()),
)
def test_temporal_function_of_one_input_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a function of the `dt` namespace of one input match polars."""
    problems = TemporalNodes.check(node).problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("node", "cell"),
    TemporalNodes.two_input_cells(),
    ids=TemporalNodes.ids(TemporalNodes.two_input_cells()),
)
def test_temporal_function_of_two_inputs_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a temporal function of two inputs match polars, at one pair."""
    problems = TemporalNodes.check(node).problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("node", "cell"),
    TemporalNodes.many_input_cells(),
    ids=TemporalNodes.ids(TemporalNodes.many_input_cells()),
)
def test_temporal_function_of_many_inputs_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a temporal function of three inputs or more match polars, at one cell."""
    problems = TemporalNodes.check(node).problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("node", "cell"),
    LiteralCells.cells(),
    ids=TemporalNodes.ids(LiteralCells.cells()),
)
def test_temporal_kernel_reads_the_value_of_a_literal_as_polars_does(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers match polars where the kernel reads the value of a literal input."""
    problems = TemporalNodes.check(node).problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "cell",
    [cell for _, cell in TemporalNodes.many_input_cells(values=True)],
    ids=CellCheck.label,
)
def test_replace_with_a_temporal_part_runs_where_its_values_fit(cell: tuple[Input, ...]) -> None:
    """A temporal, `Categorical` or struct column at a part of `replace` runs where its values fit.

    The strict cast of each part runs on the physical integers of a temporal
    column, on the ids of the categories of a `Categorical` column, and on
    each field of a struct. Few samples hold integers that name
    a part there. So where the rules give a type, polars gives that type on a
    sample that names the part, or refuses each sample with the error of a
    value, never with an error of the types. Where the rules refuse, polars
    refuses too, and the values decide which error comes first. The planner
    types agree.
    """
    leaves = [entry.leaf for entry in cell]
    rules = NodeEvaluation(Temporal.REPLACE, leaves, CellCheck.GROUPS).answers()
    replaced = TemplateCall(TemporalNodes.REPLACE)(*(entry.expression for entry in cell))
    polars = PolarsCell(replaced.alias("o"))
    assert polars.planner(agg=False) == rules.planner_select
    assert polars.planner(agg=True) == rules.planner_agg
    for agg, wanted in ((False, rules.run_select), (True, rules.run_agg)):
        answer = polars.run(agg=agg, every=True)
        if isinstance(answer, DType):
            assert answer == wanted
        elif isinstance(wanted, DType):
            assert answer.error in {"InvalidOperationError", "ComputeError"}, answer


@pytest.mark.parametrize(
    "part",
    [pl.col("x"), pl.col("x").first(), pl.col("x").head(2)],
    ids=["a column", "one value for each group", "a slice"],
)
def test_replace_of_a_literal_gives_the_rows_of_the_literal(part: pl.Expr) -> None:
    """`dt.replace` gives the rows of its first input, so in `agg` a literal beside a part raises.

    Polars reads only the first row of the part beside a literal of one row,
    and in `agg` it refuses the output of one row beside the rows of the part.
    """
    expression = pl.lit(datetime.date(2024, 1, 1)).dt.replace(year=part)
    assert TreeFrame().rules(expression) == TreeFrame().polars(expression)
