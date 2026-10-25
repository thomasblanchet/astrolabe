"""The kernels of the `str` namespace that read an option of their node.

A function of a namespace of polars often holds an option, such as the target
type of `str.strptime`. Its kernel reads that option, and then it dispatches
on the data type of its input as any kernel does. So each kernel here is built
from the value of the option, and code that evaluates a node builds it once
the option of that node is known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType  # ruff: ignore[typing-only-first-party-import]
from astrolabe._rules.failures import ErrorClass, PolarsRefusal
from astrolabe._rules.kernels.dispatch import Data  # ruff: ignore[typing-only-first-party-import]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.type_functions import Operand


class TextFormat:
    """Say whether a text can fit a format of polars, by the shape of each part of the format.

    Polars parses text with the directives of the Rust library `chrono`. A
    numeric directive such as `%Y` or `%m` reads digits, a name such as
    `%b` reads letters, and any other character of the format must stand in
    the text. A text that does not have this shape fails the parse. A text
    of the right shape can still fail, as the month 13 does, so a match
    proves nothing, and only a text that cannot match is refused. A format
    with a directive that this class does not read gives no answer.

    Example:
        >>> TextFormat().refuses("a", "%Y-%m-%d"), TextFormat().refuses("2020-1-5", "%Y-%m-%d")
        (True, False)
    """

    NUMBERS: ClassVar[dict[str, str]] = {
        "Y": r"[+-]?\d+",
        "C": r"\d{1,2}",
        "y": r"\d{1,2}",
        "m": r"\d{1,2}",
        "d": r"\d{1,2}",
        "e": r"\d{1,2}",
        "H": r"\d{1,2}",
        "k": r"\d{1,2}",
        "I": r"\d{1,2}",
        "l": r"\d{1,2}",
        "M": r"\d{1,2}",
        "S": r"\d{1,2}",
        "j": r"\d{1,3}",
    }
    NAMES: ClassVar[dict[str, str]] = {
        "b": r"[A-Za-z]+",
        "B": r"[A-Za-z]+",
        "h": r"[A-Za-z]+",
        "a": r"[A-Za-z]+",
        "A": r"[A-Za-z]+",
        "p": r"[AaPp][Mm]",
        "P": r"[AaPp][Mm]",
    }
    SHORT: ClassVar[dict[str, str]] = {
        "F": "%Y-%m-%d",
        "T": "%H:%M:%S",
        "X": "%H:%M:%S",
        "D": "%m/%d/%y",
        "R": "%H:%M",
    }
    HOURS: ClassVar[frozenset[str]] = frozenset({"H", "k"})
    HALF_HOURS: ClassVar[frozenset[str]] = frozenset({"I", "l"})
    HALVES: ClassVar[frozenset[str]] = frozenset({"p", "P"})
    YEARS: ClassVar[frozenset[str]] = frozenset({"Y", "y"})
    MONTHS: ClassVar[frozenset[str]] = frozenset({"m", "b", "B", "h"})
    DAYS: ClassVar[frozenset[str]] = frozenset({"d", "e"})

    def refuses(self, text: str, form: str) -> bool | None:
        """Say whether a text cannot fit a format.

        Args:
            text: The text.
            form: The format.

        Returns:
            Whether the text cannot fit, or `None` where the format holds a
            directive that this class does not read.

        Example:
            >>> TextFormat().refuses("10:20", "%H:%M:%S.%f") is None
            True
        """
        pattern = self._pattern(form)
        return None if pattern is None else re.fullmatch(pattern, text) is None

    def lacks_time(self, form: str) -> bool:
        """Say whether a format lacks the hour or the minute, so that no text fits it as a time.

        `chrono` builds a time from its hour and its minute, and the hour of a
        12-hour clock also needs its half of the day, `%p`. A format with a
        directive that this class does not read gives no answer, so `False`.

        Args:
            form: The format.

        Returns:
            Whether it lacks them.

        Example:
            >>> TextFormat().lacks_time("%H"), TextFormat().lacks_time("%I:%M %p")
            (True, False)
            >>> TextFormat().lacks_time("%I:%M"), TextFormat().lacks_time("%R")
            (True, False)
            >>> TextFormat().lacks_time("%H:%M:%S%.f")
            False
        """
        if self._pattern(form) is None:
            return False
        directives = self._directives(form)
        half_day = bool(directives & self.HALF_HOURS) and bool(directives & self.HALVES)
        hour = bool(directives & self.HOURS) or half_day
        return not hour or "M" not in directives

    def lacks_date(self, form: str) -> bool:
        """Say whether a format lacks a part of a date, so that `chrono` builds no date from it.

        `chrono` builds a date from its year and either its month and its day,
        or its day of the year. A format with a directive that this class does
        not read gives no answer, so `False`.

        Args:
            form: The format.

        Returns:
            Whether it lacks one.

        Example:
            >>> TextFormat().lacks_date("%Y-%m"), TextFormat().lacks_date("%Y-%j")
            (True, False)
            >>> TextFormat().lacks_date("%b %d %y"), TextFormat().lacks_date("%m-%d")
            (False, True)
        """
        if self._pattern(form) is None:
            return False
        directives = self._directives(form)
        whole = bool(directives & self.MONTHS) and bool(directives & self.DAYS)
        return not (directives & self.YEARS) or not (whole or "j" in directives)

    def _directives(self, form: str) -> set[str]:
        """Give the directives of a format, each short directive written out.

        Args:
            form: The format.

        Returns:
            The letter of each directive.

        Example:
            >>> sorted(TextFormat()._directives("%F"))
            ['Y', 'd', 'm']
        """
        return set(re.findall(r"%(.)", self.expanded(form), flags=re.DOTALL))

    def expanded(self, form: str) -> str:
        """Give a format with each short directive written out, as `%R` for `%H:%M`.

        Args:
            form: The format.

        Returns:
            The format.

        Example:
            >>> TextFormat().expanded("%R")
            '%H:%M'
        """
        return re.sub(r"%([FTXDR])", lambda found: self.SHORT[found.group(1)], form)

    def _pattern(self, form: str) -> str | None:
        """Give the regular expression of the shapes that a format reads.

        Args:
            form: The format.

        Returns:
            The expression, or `None` where the format holds a directive that
            this class does not read.

        Example:
            >>> TextFormat()._pattern("%f") is None
            True
        """
        parts: list[str] = []
        for token in re.findall(r"%.|%$|.", self.expanded(form), flags=re.DOTALL):
            part = self._part(token)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)

    def _part(self, token: str) -> str | None:
        """Give the regular expression of one directive or one character of a format.

        A number may follow spaces, and a space of the format reads any
        number of spaces.

        Args:
            token: The directive, such as `%Y`, or one character.

        Returns:
            The expression, or `None` for a directive that this class does not read.

        Example:
            >>> TextFormat()._part("%f") is None
            True
            >>> TextFormat()._part("%%")
            '%'
        """
        if token.isspace():
            return r"\s*"
        if not token.startswith("%"):
            return re.escape(token)
        directive = token[1:]
        if directive == "%":
            return "%"
        if directive in self.NUMBERS:
            return r"\s*" + self.NUMBERS[directive]
        return self.NAMES.get(directive)


class FixedWidthParse:
    """Say whether the fixed-width parser of polars reads a text by a format.

    Polars parses a date or a moment by an exact format with its own parser
    first, where the format holds only the directives `%Y`, `%y`, `%m`,
    `%d`, `%b`, `%B`, `%H`, `%M`, `%S` and `%9f`, `%6f` or `%3f`. That parser
    reads each number at a fixed width of digits, gives a missing month or day
    the value 1, and needs the whole text. Where it does not read the text,
    polars parses it with `chrono`. The class gives a match of the shape, so
    a month 13 matches, as for `TextFormat`.

    Example:
        >>> FixedWidthParse().reads("2024-01", "%Y-%m"), FixedWidthParse().reads("2024-1", "%Y-%m")
        (True, False)
    """

    WIDTHS: ClassVar[dict[str, int]] = {"Y": 4, "m": 2, "d": 2, "H": 2, "M": 2, "S": 2}
    FRACTIONS: ClassVar[dict[str, int]] = {"9f": 9, "6f": 6, "3f": 3}
    NAMES: ClassVar[frozenset[str]] = frozenset({"b", "B"})
    MONTHS: ClassVar[tuple[str, ...]] = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    TOKENS: ClassVar[re.Pattern[str]] = re.compile(r"%[963]f|%.|%$|.", flags=re.DOTALL)

    def supports(self, form: str) -> bool:
        """Say whether the parser reads a format, after polars writes out its short directives.

        Args:
            form: The format.

        Returns:
            Whether every directive is one that the parser reads.

        Example:
            >>> FixedWidthParse().supports("%Y-%b"), FixedWidthParse().supports("%Y%%%m")
            (True, False)
        """
        tokens = self.TOKENS.findall(TextFormat().expanded(form))
        return all(self._supported(token) for token in tokens)

    def reads(self, text: str, form: str) -> bool:
        """Say whether the parser reads a text by a format that it supports.

        Args:
            text: The text.
            form: The format.

        Returns:
            Whether every part of the format matches the text, and nothing is left.

        Example:
            >>> FixedWidthParse().reads("2024-January", "%Y-%B")
            True
            >>> FixedWidthParse().reads("-2024", "%Y"), FixedWidthParse().reads("+2024", "%Y")
            (True, False)
        """
        expanded = TextFormat().expanded(form)
        value = text.encode()
        offset: int | None = 1 if value.startswith(b"-") and expanded.startswith("%Y") else 0
        for token in self.TOKENS.findall(expanded):
            offset = None if offset is None else self._step(value, offset, token)
        return offset == len(value)

    def _supported(self, token: str) -> bool:
        """Say whether the parser reads one token of a format.

        Args:
            token: A directive, such as `%Y`, or one character.

        Returns:
            Whether it reads it.

        Example:
            >>> FixedWidthParse()._supported("%j"), FixedWidthParse()._supported("-")
            (False, True)
        """
        if not token.startswith("%"):
            return True
        directive = token[1:]
        known = {*self.WIDTHS, *self.FRACTIONS, *self.NAMES, "y"}
        return directive in known

    def _step(self, value: bytes, offset: int, token: str) -> int | None:
        """Read one token of a format at a place of a text.

        Args:
            value: The text, as bytes.
            offset: The place.
            token: A directive, or one character.

        Returns:
            The place after the token, or `None` where the token does not match.

        Example:
            >>> FixedWidthParse()._step(b"-", 0, "-"), FixedWidthParse()._step(b"Jan", 0, "%b")
            (1, 3)
        """
        if not token.startswith("%"):
            return offset + 1 if value[offset : offset + 1] == token.encode() else None
        directive = token[1:]
        if directive in self.NAMES:
            return self._month(value, offset, full=directive == "B")
        return self._number(value, offset, directive)

    def _number(self, value: bytes, offset: int, directive: str) -> int | None:
        """Read a number of a fixed width at a place of a text.

        A year of two digits, `%y`, needs only its first digit.

        Args:
            value: The text, as bytes.
            offset: The place.
            directive: The directive, as `Y` or `6f`.

        Returns:
            The place after the number, or `None` where it does not match.

        Example:
            >>> FixedWidthParse()._number(b"2x", 0, "y"), FixedWidthParse()._number(b"x2", 0, "y")
            (2, None)
        """
        if directive == "y":
            field = value[offset : offset + 2]
            return offset + 2 if len(field) == 2 and field[:1].isdigit() else None
        width = self.WIDTHS.get(directive) or self.FRACTIONS[directive]
        field = value[offset : offset + width]
        return offset + width if len(field) == width and field.isdigit() else None

    def _month(self, value: bytes, offset: int, *, full: bool) -> int | None:
        """Read the name of a month, its three first letters or, for `%B`, its whole name.

        Args:
            value: The text, as bytes.
            offset: The place.
            full: Whether the whole name can stand in place of the three letters.

        Returns:
            The place after the name, or `None` where no month starts there.

        Example:
            >>> FixedWidthParse()._month(b"Mayday", 0, full=True)
            3
        """
        names = [name for name in self.MONTHS if value[offset:].startswith(name[:3].encode())]
        if not names:
            return None
        rest = names[0][3:].encode()
        end = offset + 3
        return end + len(rest) if full and value[end:].startswith(rest) else end


class FormatFault:
    """Give the fault of a format that polars refuses before it parses a date or a moment.

    Polars checks the format of an exact parse into a `Date` or a `Datetime`
    before it reads any text, so it refuses such a format at every frame,
    with a `ComputeError`. It looks for the directives anywhere in the format
    as the user writes it.

    Example:
        >>> FormatFault().of("%Y-%m-%d %H")
        'Please either specify both hour and minute, or neither'
        >>> FormatFault().of("%Y-%m-%d") is None
        True
    """

    HOUR: ClassVar[re.Pattern[str]] = re.compile(r"%[_-]?[HkIl]")
    MINUTE: ClassVar[re.Pattern[str]] = re.compile(r"%[_-]?M")
    SECOND: ClassVar[re.Pattern[str]] = re.compile(r"%[_-]?S")
    TWELVE_HOUR: ClassVar[re.Pattern[str]] = re.compile(r"%[_-]?[Il]")
    MERIDIEM: ClassVar[re.Pattern[str]] = re.compile(r"%[_-]?[pP]")

    def of(self, form: str) -> str | None:
        """Give the message of the fault of a format, or `None` where polars accepts it.

        Args:
            form: The format.

        Returns:
            The message, or `None`.

        Example:
            >>> FormatFault().of("%M:%S")
            'Please either specify both hour and minute, or neither'
            >>> FormatFault().of("%Y %S")
            'Found seconds directive, but no hours directive'
            >>> FormatFault().of("%I:%M")
            'Please either specify both 12-hour directive and meridiem directive, or neither'
        """
        hour = self.HOUR.search(form) is not None
        if hour != (self.MINUTE.search(form) is not None):
            return "Please either specify both hour and minute, or neither"
        if self.SECOND.search(form) is not None and not hour:
            return "Found seconds directive, but no hours directive"
        if (self.TWELVE_HOUR.search(form) is None) != (self.MERIDIEM.search(form) is None):
            return "Please either specify both 12-hour directive and meridiem directive, or neither"
        return None


@dataclass(frozen=True)
class Strptime:
    """Give the type of the kernel of `str.strptime`, which parses text into a date or a time.

    The kernel picks its work by the target type, which is an option of the
    node. Its first input holds the text, and it must be `String` data. A
    `Date` or a `Time` target reads only that input, so its second input,
    `ambiguous`, can hold any type and any number of rows. A `Datetime`
    target also reads `ambiguous` as `String` data, and then checks that the
    two inputs have one length, or that one of them has one row. A `Time`
    target parses only an exact format. The kernel gives the target, with the
    zone that the type function of the node gives it.

    Whether a text fits the format is a fact of the data. A strict parse raises
    on a text that does not fit, and a parse that is not strict gives a null.
    The code holds the value of a literal text, so a strict parse of a
    literal that cannot fit an exact format raises at every run
    (`TextFormat`).

    Example:
        >>> date = Strptime(DType("Date"), exact=True)
        >>> date.output([Data(DType("String")), Data(DType("Int8"))], [5, 2], ())
        Date
    """

    target: DType
    exact: bool
    form: str | None = None
    strict: bool = True

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the text and of `ambiguous`.
            lengths: The number of rows of each input, or none where they are
                not known.
            computed: Each input as the run computes it: the value of a
                literal text, or a column.

        Returns:
            The target type.

        Raises:
            PolarsRefusal: An input is not `String` data where the kernel reads
                it, the inputs of a `Datetime` target have other lengths, the
                target is a `Time` and the format is not exact, or a literal
                text cannot fit the format of a strict parse.

        Example:
            >>> moment = Strptime(DType("Datetime", ("us", None)), exact=True)
            >>> text = Data(DType("String"))
            >>> moment.output([text, Data(DType("Int8"))], [5, 5], ())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
            >>> moment.output([text, text], [5, 2], ())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
            >>> time = Strptime(DType("Time"), exact=False)
            >>> time.output([text, text], [], ())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        self._text(inputs[0])
        if self.target.name == "Datetime":
            self._text(inputs[1])
            self._lengths(lengths)
        if self.target.name == "Time" and not self.exact:
            message = "non-exact not implemented for Time data type"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        self._format(inputs[0])
        self._literal(computed)
        return self.target

    def _format(self, text: Data) -> None:
        """Refuse a format that polars refuses, or by which a strict parse fails for every text.

        Polars checks the format of an exact parse into a date or a moment
        first, whatever the data (`FormatFault`). A strict parse then raises
        on a text that does not fit, so a format that no text fits raises
        wherever the text holds a value: a time needs its hour and its
        minute, and `chrono` builds a date only from its year with its month
        and day, or with its day of the year. The fixed-width parser of polars
        gives a missing month or day the value 1, so a format that it reads
        fits some text without a day.

        Args:
            text: The data of the text.

        Raises:
            PolarsRefusal: Polars refuses the format, or no text fits it.

        Example:
            >>> month = Strptime(DType("Date"), exact=True, form="%Y%%%m")
            >>> month._format(Data(DType("String")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: conversion ...
            >>> month._format(Data(DType("String"), nulls=True))
            >>> Strptime(DType("Date"), exact=True, form="%Y-%m")._format(Data(DType("String")))
        """
        if self.form is None:
            return
        checked = self.exact and self.target.name != "Time"
        fault = FormatFault().of(self.form) if checked else None
        if fault is not None:
            raise PolarsRefusal(ErrorClass.COMPUTE, f"Invalid format string: {fault}")
        if self.strict and not text.nulls and self._unparsable(self.form):
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, self._failed())

    def _unparsable(self, form: str) -> bool:
        """Say whether no text fits a format, so that every value of a strict parse fails.

        Args:
            form: The format.

        Returns:
            Whether no text fits it.

        Example:
            >>> Strptime(DType("Time"), exact=True)._unparsable("%H")
            True
            >>> Strptime(DType("Date"), exact=False)._unparsable("%Y-%m")
            True
        """
        if self.target.name == "Time":
            return TextFormat().lacks_time(form)
        fixed = self.exact and FixedWidthParse().supports(form)
        return not fixed and TextFormat().lacks_date(form)

    def _failed(self) -> str:
        """Give the message of polars for a text that does not fit the format.

        Returns:
            The message.

        Example:
            >>> Strptime(DType("Date"), exact=True)._failed()
            'conversion from `str` to `date` failed'
        """
        return f"conversion from `str` to `{self.target.render().lower()}` failed"

    def _literal(self, computed: Sequence[Operand]) -> None:
        """Refuse a strict parse of a literal text that cannot fit an exact format.

        Where the fixed-width parser of polars reads the text, it fits. Else
        `chrono` parses it, which needs the shape of the format and, for a
        date or a moment, a whole date.

        Args:
            computed: Each input as the run computes it.

        Raises:
            PolarsRefusal: The literal text cannot fit the format.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> date = Strptime(DType("Date"), exact=True, form="%Y-%m-%d")
            >>> date._literal([Operand.of_value("a")])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: conversion ...
            >>> month = Strptime(DType("Date"), exact=True, form="%Y-%m")
            >>> month._literal([Operand.of_value("2024-1")])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: conversion ...
            >>> month._literal([Operand.of_value("2024-01")])
        """
        text = self._checked_text(computed)
        form = self.form
        if text is None or form is None:
            return
        dated = self.target.name != "Time"
        if dated and self._fixed(text, form):
            return
        lacks = dated and TextFormat().lacks_date(form)
        if lacks or TextFormat().refuses(text, form):
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, self._failed())

    def _checked_text(self, computed: Sequence[Operand]) -> str | None:
        """Give the literal text of a strict parse by an exact format, which every run meets.

        Args:
            computed: Each input as the run computes it.

        Returns:
            The text, or `None` where the text is no literal, or the parse is
            not strict or not exact.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> lenient = Strptime(DType("Date"), exact=True, form="%Y", strict=False)
            >>> lenient._checked_text([Operand.of_value("a")]) is None
            True
        """
        text = computed[0].value if computed and computed[0].literal else None
        checked = self.strict and self.exact
        return text if isinstance(text, str) and checked else None

    def _fixed(self, text: str, form: str) -> bool:
        """Say whether the fixed-width parser of polars reads a text by a format.

        Args:
            text: The text.
            form: The format.

        Returns:
            Whether the parser supports the format and reads the text.

        Example:
            >>> Strptime(DType("Date"), exact=True)._fixed("2024-01", "%Y-%m")
            True
        """
        parser = FixedWidthParse()
        return parser.supports(form) and parser.reads(text, form)

    def _text(self, data: Data) -> None:
        """Refuse data that is not text.

        Args:
            data: The data of one input.

        Raises:
            PolarsRefusal: The data is not `String` data.

        Example:
            >>> Strptime(DType("Date"), exact=True)._text(Data(DType("String")))
        """
        if data.dtype.name != "String":
            message = f"invalid series dtype: expected `String`, got `{data.dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)

    def _lengths(self, lengths: Sequence[int]) -> None:
        """Refuse a text and an `ambiguous` of other lengths, where neither has one row.

        Args:
            lengths: The number of rows of each input, or none where they are not known.

        Raises:
            PolarsRefusal: The two inputs have other lengths.

        Example:
            >>> Strptime(DType("Datetime", ("us", None)), exact=True)._lengths([1, 5])
        """
        if len(lengths) < 2 or 1 in lengths[:2] or lengths[0] == lengths[1]:
            return
        message = (
            f"arguments for `str.strptime` have different lengths ({lengths[0]} != {lengths[1]})"
        )
        raise PolarsRefusal(ErrorClass.SHAPE, message)


class StringKernels:
    """Build the kernels of the `str` namespace that read an option of their node.

    Example:
        >>> StringKernels().strptime(DType("Date"), exact=True)
        Strptime(target=Date, exact=True, form=None, strict=True)
    """

    def strptime(
        self,
        target: DType,
        exact: bool,
        form: str | None = None,
        strict: bool = True,
    ) -> Strptime:
        """Give the kernel of `str.strptime`, `str.to_date`, `str.to_datetime` and `str.to_time`.

        Args:
            target: The target type, with the zone of the type function.
            exact: Whether the format must match the whole text.
            form: The format, or `None` where polars infers it.
            strict: Whether a text that does not fit raises.

        Returns:
            The kernel.

        Example:
            >>> StringKernels().strptime(DType("Time"), exact=False).exact
            False
        """
        return Strptime(target, exact=exact, form=form, strict=strict)
