"""Run the code of each `.yml` case with real polars, and compare the result with its type.

A `.yml` case under `tests/typing/` states the types that the checker
must give a small program. `pytest-mypy-plugins` only checks that
program with mypy, and never runs it. The case and the plugin can
therefore share the same wrong belief about polars, for example a
wrong rule for the data type of a sum, and no test finds it.

The classes here close that gap. They run the `main:` code of each
case, statement by statement, on a one-row frame built from the
declared columns. They then compare each real result with the type
that the case states. A case states a type in two ways: an annotated
assignment such as `x: TypedDataFrame[...] = expr`, or a
`reveal_type(expr)` call with a `# N: Revealed type is "..."` comment.
Both are read as they stand, so no `.yml` file needs a change.
"""

from __future__ import annotations

import ast
import datetime
import decimal
import os
import platform
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import astrolabe
import polars as pl
import polars.datatypes.classes
import pytest
import yaml
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe.frame import Unchecked, DTypeLike, TypedDataFrame, TypedLazyFrame, _TypedFrameBase
from polars.lazyframe.in_process import InProcessQuery

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

_SAFE_EXCEPTIONS = (KeyboardInterrupt, SystemExit, GeneratorExit)


def _safely(action: Callable[[], Any]) -> tuple[bool, Any]:
    """Run a function, and catch every failure, a panic inside polars included.

    polars is written in Rust. A Rust panic reaches Python as
    `pyo3_runtime.PanicException`, which inherits from `BaseException`
    and not from `Exception`. An `except Exception` does not catch it, so
    this function catches `BaseException`. It still lets
    `KeyboardInterrupt`, `SystemExit` and `GeneratorExit` through, so
    that a user can stop the run.

    Args:
        action: The function to run, with no argument.

    Returns:
        `(True, result)` if the function returns, and `(False, error)`
        if it fails.

    Example:
        >>> _safely(lambda: 1 + 1)
        (True, 2)
        >>> succeeded, error = _safely(lambda: 1 / 0)
        >>> succeeded, type(error)
        (False, <class 'ZeroDivisionError'>)
    """
    try:
        return True, action()
    except _SAFE_EXCEPTIONS:
        raise
    except BaseException as error:
        return False, error


class DTypeSampleFactory:
    """Build one Python value that polars stores as a given data type.

    The cross-check fills each declared column with one such value, so
    that polars makes a real one-row frame with the declared schema. A
    container data type, such as a `List` or a `Struct`, gets a value
    built from its inner data types. A column of the data type
    `Unchecked` gets a whole number.

    Example:
        >>> DTypeSampleFactory().built(pl.Int64())
        1
        >>> DTypeSampleFactory().built(pl.List(pl.String()))
        ['x']
    """

    _SCALAR_SAMPLES: ClassVar[dict[DTypeLike, object]] = {
        pl.Int8(): 1,
        pl.Int16(): 1,
        pl.Int32(): 1,
        pl.Int64(): 1,
        pl.UInt8(): 1,
        pl.UInt16(): 1,
        pl.UInt32(): 1,
        pl.UInt64(): 1,
        pl.Float32(): 1.0,
        pl.Float64(): 1.0,
        pl.Boolean(): True,
        pl.String(): "x",
        pl.Binary(): b"x",
        pl.Date(): datetime.date(2020, 1, 1),
        pl.Time(): datetime.time(1, 2, 3),
        pl.Duration(): datetime.timedelta(seconds=1),
        pl.Datetime(): datetime.datetime(2020, 1, 1),
        pl.Categorical(): "x",
        pl.Decimal(): decimal.Decimal("1.00"),
        pl.Null(): None,
        Unchecked: 1,
    }

    def _list_sample(self, dtype: pl.List) -> Any:
        return [self.built(dtype.inner)]

    def _array_sample(self, dtype: pl.Array) -> Any:
        return [self.built(dtype.inner)] * (dtype.size or 1)

    def _struct_sample(self, dtype: pl.Struct) -> Any:
        return {field.name: self.built(field.dtype) for field in dtype.fields}

    def _enum_sample(self, dtype: pl.Enum) -> Any:
        return dtype.categories[0]

    _CONTAINER_SAMPLERS: ClassVar[dict[type, Callable[[DTypeSampleFactory, Any], Any]]] = {
        pl.List: _list_sample,
        pl.Array: _array_sample,
        pl.Struct: _struct_sample,
        pl.Enum: _enum_sample,
    }

    def built(self, dtype: Any) -> Any:
        """Build one value of a data type.

        Args:
            dtype: A polars data type, as an instance with its parameters.

        Returns:
            A Python value that `pl.Series([value], dtype=dtype)` accepts.

        Example: see class docstring.
        """
        sampler = self._CONTAINER_SAMPLERS.get(type(dtype))
        if sampler is not None:
            return sampler(self, dtype)
        for candidate, value in self._SCALAR_SAMPLES.items():
            if dtype == candidate:
                return value
        return 1


class SampleFrameBuilder:
    """Build a real one-row polars frame with the schema that a typed frame declares.

    A `.yml` case often starts with a bare declaration such as
    `df: Frame`, with no value. The cross-check needs a real frame there,
    so it builds one from the declared columns.

    Example:
        >>> from astrolabe import columns
        >>> X = columns("X", {"x": pl.Int64})
        >>> Frame = TypedDataFrame[X]
        >>> SampleFrameBuilder(DTypeSampleFactory()).built(Frame)
        shape: (1, 1)
        ┌─────┐
        │ x   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        └─────┘
    """

    def __init__(self, dtype_sample_factory: DTypeSampleFactory) -> None:
        """Keep the builder of the value in each column.

        Args:
            dtype_sample_factory: The builder of one value for each data type.
        """
        self._dtype_sample_factory = dtype_sample_factory

    def built(self, frame_type: type[_TypedFrameBase]) -> Any:
        """Build a real frame with one row and exactly the declared data types.

        A column of the data type `Unchecked` gets the data type `Int64`.

        Args:
            frame_type: A typed frame class with its columns, for example
                `TypedDataFrame[X]`.

        Returns:
            A `pl.DataFrame`, or a `pl.LazyFrame` if `frame_type` is a
            lazy typed frame.

        Example: see class docstring.
        """
        schema = frame_type.schema
        data = {name: [self._dtype_sample_factory.built(dtype)] for name, dtype in schema.items()}
        real_schema = {name: self._real_dtype(dtype) for name, dtype in schema.items()}
        built = pl.DataFrame(data, schema=real_schema)
        if issubclass(frame_type, TypedLazyFrame):
            return built.lazy()
        return built

    def _real_dtype(self, dtype: DTypeLike) -> DTypeLike:
        if dtype == Unchecked:
            return pl.Int64
        return dtype() if isinstance(dtype, type) else dtype


class RevealedTypeReference:
    """Turn the type text in a `# N: Revealed type is "..."` comment into the Python class.

    mypy shows a type by the full names of its classes, for example
    `astrolabe.frame.TypedDataFrame[...]`. This class evaluates that text
    in a namespace that holds `polars` and `astrolabe`, and gives the
    class that it names. The cross-check then compares a real value with
    that class.

    Example:
        >>> ref = RevealedTypeReference()
        >>> ref.resolved("polars.dataframe.frame.DataFrame")
        <class 'polars.dataframe.frame.DataFrame'>
        >>> ref.resolved("Any") is None
        True
    """

    _SKIPPED: ClassVar[frozenset[str]] = frozenset({"Any", "None"})
    _EVAL_NAMESPACE: ClassVar[dict[str, object]] = {
        "Literal": Literal,
        "polars": polars,
        "astrolabe": astrolabe,
    }

    def resolved(self, note_text: str) -> type | None:
        """Give the class that the text of one revealed type names.

        Args:
            note_text: The text between the quotes of the comment.

        Returns:
            The class. None if the text is `Any` or `None`, or if it does
            not evaluate to a class.

        Example: see class docstring.
        """
        if note_text in self._SKIPPED:
            return None
        succeeded, value = _safely(lambda: eval(note_text, dict(self._EVAL_NAMESPACE)))  # ruff: ignore[suspicious-eval-usage]
        return value if succeeded and isinstance(value, type) else None

    # A revealed typed frame: its class, the dict of its declared columns,
    # and its remainder. mypy prints the columns as a `TypedDict`, with the
    # name of a named class first.
    _FRAME: ClassVar[re.Pattern[str]] = re.compile(
        r"astrolabe\.frame\.(TypedDataFrame|TypedLazyFrame)"
        r"\[TypedDict\((?:[\w.]+, )?(\{.*\}), closed=True\), (.+)\]"
    )

    @propagates(None)
    def frame(self, note_text: str) -> RevealedFrame | None:
        """Read the text of a revealed typed frame into a frame class and its remainder.

        Args:
            note_text: The text between the quotes of the comment.

        Returns:
            The class of a typed frame over the revealed columns, and
            whether the frame is closed. None for any other type, or for
            columns that the namespace cannot evaluate.

        Example:
            >>> text = (
            ...     "astrolabe.frame.TypedDataFrame[TypedDict({'x': "
            ...     "polars.datatypes.classes.Int64}, closed=True), None]"
            ... )
            >>> revealed = RevealedTypeReference().frame(text)
            >>> revealed.frame.schema, revealed.closed
            ({'x': Int64}, True)
        """
        found = not_none(self._FRAME.fullmatch(note_text))
        kind, listed, remainder = found.groups()
        source = self._untupled(listed)
        succeeded, columns = _safely(lambda: eval(source, dict(self._EVAL_NAMESPACE)))  # ruff: ignore[suspicious-eval-usage]
        not_none(columns if succeeded and columns else None)
        frame_class = TypedDataFrame if kind == "TypedDataFrame" else TypedLazyFrame
        typed = cast("type[_TypedFrameBase]", frame_class[astrolabe.columns("Revealed", columns)])
        not_none(typed.schema)
        return RevealedFrame(typed, closed=remainder == "None")

    def _untupled(self, text: str) -> str:
        """Take the `tuple[...]` wrapper off the arguments of each `Enum` and `Struct` marker.

        mypy prints `Enum[tuple[Literal['a'], ...]]`, because it stores the
        arguments of a marker with a variadic parameter in one tuple. The
        marker reads its arguments with no wrapper at run time.

        Args:
            text: The revealed text.

        Returns:
            The text with no such wrapper.

        Example:
            >>> RevealedTypeReference()._untupled("Enum[tuple[Literal['a'], X[Y]]]")
            "Enum[Literal['a'], X[Y]]"
        """
        while "[tuple[" in text:
            start = text.index("[tuple[") + 1
            end = self._closing(text, start + len("tuple"))
            text = text[:start] + text[start + len("tuple[") : end] + text[end + 1 :]
        return text

    def _closing(self, text: str, opening: int) -> int:
        """Give the place of the bracket that closes the one at a position.

        Args:
            text: The text.
            opening: The position of an opening bracket.

        Returns:
            The position of its closing bracket.

        Example:
            >>> RevealedTypeReference()._closing("a[b[c]]", 1)
            6
        """
        depth = 0
        for at in range(opening, len(text)):
            depth += {"[": 1, "]": -1}.get(text[at], 0)
            if depth == 0:
                return at
        return len(text)


@dataclass(frozen=True)
class RevealedFrame:
    """A typed frame that a `reveal_type` note states.

    Attributes:
        frame: A typed frame class over the revealed columns.
        closed: Whether the frame has no remainder. A closed frame states
            every column, so the real frame must hold no other column.
    """

    frame: type[_TypedFrameBase]
    closed: bool


class SchemaExpectation:
    """Compare the schema that a type declares with the schema of a real frame.

    A declared column of the data type `Unchecked` matches any real data
    type, because the checker states only its name. That rule also holds
    for an `Unchecked` field of a `Struct` or the element of a `List`.
    The real frame can hold more columns than an open frame declares, and
    no other column than a closed frame declares.

    Example:
        >>> expectation = SchemaExpectation()
        >>> expectation.matches({"x": pl.Int64}, {"x": pl.Int64()})
        True
        >>> expectation.matches({"x": Unchecked}, {"x": pl.String()})
        True
        >>> expectation.matches({"x": pl.Int64}, {"x": pl.String()})
        False
        >>> expectation.matches({"x": pl.Int64}, {"x": pl.Int64(), "y": pl.Int8()}, closed=True)
        False
    """

    def matches(
        self,
        expected: Mapping[str, DTypeLike],
        actual: Mapping[str, Any],
        *,
        closed: bool = False,
    ) -> bool:
        """Answer whether the real frame holds every declared column, with its data type.

        Args:
            expected: The schema of the typed frame class.
            actual: The schema of the real frame.
            closed: Whether the type is a closed frame, which states every
                column.

        Returns:
            True if each declared name is in `actual` with the same data
            type, and, for a closed frame, if `actual` holds no other name.
            An `Unchecked` column matches any data type.

        Example: see class docstring.
        """
        if closed and set(expected) != set(actual):
            return False
        for name, dtype in expected.items():
            if name not in actual or not self._dtype_matches(dtype, actual[name]):
                return False
        return True

    def _dtype_matches(self, expected: Any, actual: Any) -> bool:
        """Compare one declared data type with a real one, with `Unchecked` at any depth.

        A field of a `Struct`, or the element of a `List` or an `Array`,
        can be `Unchecked` on its own. For example, the checker can know
        the field `a` of a struct and not the field `z`. This method
        therefore goes into each nested data type.

        Two structs match if they have the same number of fields, and if
        the fields at each position have the same name and matching data
        types. Two lists, or two arrays, match if their elements have
        matching data types.

        Args:
            expected: The declared data type.
            actual: The real data type.

        Returns:
            True if `expected` is `Unchecked`, if both are containers of
            one kind whose inner data types match, or if both are equal.

        Example:
            >>> expectation = SchemaExpectation()
            >>> nested = pl.Struct({"a": pl.Int64, "z": Unchecked})
            >>> expectation._dtype_matches(nested, pl.Struct({"a": pl.Int64, "z": pl.Int32}))
            True
            >>> expectation._dtype_matches(nested, pl.Struct({"a": pl.String, "z": pl.Int32}))
            False
            >>> expectation._dtype_matches(pl.List(Unchecked), pl.List(pl.String))
            True
        """
        if expected == Unchecked:
            return True
        if isinstance(expected, pl.Struct) and isinstance(actual, pl.Struct):
            if len(expected.fields) != len(actual.fields):
                return False
            pairs = zip(expected.fields, actual.fields, strict=True)
            return all(
                declared.name == real.name and self._dtype_matches(declared.dtype, real.dtype)
                for declared, real in pairs
            )
        if isinstance(expected, pl.List | pl.Array) and isinstance(actual, type(expected)):
            return self._dtype_matches(expected.inner, actual.inner)
        return bool(expected == actual)


@dataclass(frozen=True)
class CheckResult:
    """The result of one comparison between a stated type and a real value.

    Attributes:
        label: A short text that says which statement was checked.
        expected: What the case states: a schema or a class.
        actual: What the real run gave.
        matched: Whether the real value agrees with the stated type.
    """

    label: str
    expected: object
    actual: object
    matched: bool


@dataclass(frozen=True)
class CaseSource:
    """The parts of one `.yml` case that the cross-check reads.

    Attributes:
        file: The `.yml` file that holds the case.
        case: The name of the case, from its `case:` key.
        main: The Python code of the case, from its `main:` key.
    """

    file: Path
    case: str
    main: str


class YamlCorpus:
    """Read every case from the `.yml` files under a directory.

    Example:
        >>> from pathlib import Path
        >>> cases = list(YamlCorpus(Path("tests/typing")).cases())
        >>> len(cases) > 0
        True
    """

    def __init__(self, directory: Path) -> None:
        """Keep the directory to search for `test_*.yml` files.

        Args:
            directory: The directory. The search goes into each directory
                below it, for example `tests/typing/real_world/`.
        """
        self._directory = directory

    def cases(self) -> Iterator[CaseSource]:
        """Give each case of each `test_*.yml` file, by file name and then in file order.

        Each file holds a list of cases at the top of the document. An
        empty file holds no case, because `yaml.safe_load` gives None for
        it. An entry with no `case:` key is not a case.

        Yields:
            Each case.

        Example:
            >>> from pathlib import Path
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as name:
            ...     directory = Path(name)
            ...     _ = (directory / "test_a.yml").write_text('[{case: one, main: "x = 1"}]')
            ...     _ = (directory / "test_b.yml").write_text("")
            ...     [(source.case, source.main) for source in YamlCorpus(directory).cases()]
            [('one', 'x = 1')]
        """
        for file in sorted(self._directory.glob("**/test_*.yml")):
            entries = yaml.safe_load(file.read_text(encoding="utf-8")) or []
            for entry in entries:
                if "case" in entry and not self._skipped(entry):
                    yield CaseSource(file, entry["case"], entry["main"])

    def _skipped(self, entry: dict[str, object]) -> bool:
        """Say whether the `skip:` key of a case leaves it out, as pytest-mypy-plugins reads it.

        The key is a Python expression, which pytest-mypy-plugins evaluates
        with the modules `sys`, `os`, `pytest` and `platform`. A case whose
        answer depends on the release of polars names the releases that it
        leaves out (`tests.releases`).

        Args:
            entry: The case, as `yaml.safe_load` gives it.

        Returns:
            Whether the case is left out.

        Example:
            >>> from pathlib import Path
            >>> corpus = YamlCorpus(Path("tests/typing"))
            >>> corpus._skipped({"case": "a", "skip": "True"}), corpus._skipped({"case": "b"})
            (True, False)
        """
        names = {"sys": sys, "os": os, "pytest": pytest, "platform": platform}
        return bool(eval(str(entry.get("skip", "False")), names))  # ruff: ignore[suspicious-eval-usage]


class RuntimeCrossCheck:
    """Run the code of one `.yml` case with real polars, and check each type that it states.

    The code runs one statement at a time in one namespace. A bare
    declaration of a typed frame, such as `df: Frame`, gets a real
    one-row frame. After each statement, the check compares the real
    value with the type that the statement states, if any. The run stops
    at the first statement that fails, because a case can fail on
    purpose to show an error.

    Example:
        >>> check = RuntimeCrossCheck(
        ...     SampleFrameBuilder(DTypeSampleFactory()),
        ...     SchemaExpectation(),
        ...     RevealedTypeReference(),
        ... )
        >>> main = '''
        ... import polars as pl
        ... from astrolabe import Columns, TypedDataFrame
        ... class A(Columns):
        ...     a: pl.Int64
        ... Frame = TypedDataFrame[A]
        ... df: Frame
        ... narrowed: TypedDataFrame[A] = df.select("a")
        ... '''
        >>> [result.matched for result in check.checked(main)]
        [True]
    """

    _REVEAL_NOTE: ClassVar[re.Pattern[str]] = re.compile(r'#\s*N:\s*Revealed type is "([^"]*)"')
    # lf.collect(background=True) gives an InProcessQuery, which runs a real
    # thread in the background. If the handle is dropped with no .fetch() or
    # .cancel(), a Rust panic on that thread ends the whole process, and Python
    # cannot catch it. The check skips a statement that gives one. It loses
    # nothing, because an InProcessQuery has no schema to compare.
    _UNSAFE_ANNOTATIONS: ClassVar[frozenset[object]] = frozenset({InProcessQuery})

    def __init__(
        self,
        sample_frame_builder: SampleFrameBuilder,
        schema_expectation: SchemaExpectation,
        revealed_type_reference: RevealedTypeReference,
    ) -> None:
        """Keep the classes that build the sample frames and compare the results.

        Args:
            sample_frame_builder: Builds a real frame for a bare
                declaration such as `df: Frame`.
            schema_expectation: Compares a declared schema with a real one.
            revealed_type_reference: Turns the text of a revealed type into
                a class.
        """
        self._sample_frame_builder = sample_frame_builder
        self._schema_expectation = schema_expectation
        self._revealed_type_reference = revealed_type_reference

    def checked(self, main_source: str) -> list[CheckResult]:
        """Run the code of one case, and compare each stated type with its real value.

        A statement that raises, or that starts a background thread, ends
        the run with no failure of this method. Such a statement is
        normal in a case that shows an error on purpose. A statement
        with an annotation in `_UNSAFE_ANNOTATIONS` does not run, because
        it can end the process.

        After each statement, the check compares the result of an
        annotated assignment, and the value of a `reveal_type(...)` call,
        with the type that the case states.

        Args:
            main_source: The Python code of the case.

        Returns:
            One result for each stated type that the run reached and
            compared.
        """
        namespace: dict[str, object] = {"reveal_type": lambda value: value}
        results: list[CheckResult] = []
        for statement in ast.parse(main_source).body:
            if self._bare_receiver_handled(statement, namespace):
                continue
            if self._unsafe_to_execute(statement, namespace):
                break
            if not self._executed(statement, namespace):
                break
            results.extend(
                self._annotated_assignment_check(statement, main_source, namespace) or []
            )
            results.extend(self._reveal_type_check(statement, main_source, namespace) or [])
        return results

    def _unsafe_to_execute(self, statement: ast.stmt, namespace: dict[str, object]) -> bool:
        if not isinstance(statement, ast.AnnAssign):
            return False
        annotation = self._evaluated(statement.annotation, namespace)
        return annotation in self._UNSAFE_ANNOTATIONS

    def _bare_receiver_handled(self, statement: ast.stmt, namespace: dict[str, object]) -> bool:
        """Give a real one-row frame to a bare declaration of a typed frame.

        A bare declaration, such as `df: Frame`, has no value. The name
        gets a sample frame with the declared columns, so that the next
        statements can use it.

        Args:
            statement: One statement of the code of a case.
            namespace: The namespace that the earlier statements filled.

        Returns:
            True if the statement is such a declaration and the sample
            frame was built. False otherwise, and the statement then runs
            as usual.

        Example:
            >>> from astrolabe import columns
            >>> check = RuntimeCrossCheck(
            ...     SampleFrameBuilder(DTypeSampleFactory()),
            ...     SchemaExpectation(),
            ...     RevealedTypeReference(),
            ... )
            >>> namespace = {"Frame": TypedDataFrame[columns("X", {"x": pl.Int64})]}
            >>> check._bare_receiver_handled(ast.parse("df: Frame").body[0], namespace)
            True
            >>> namespace["df"].columns
            ['x']
            >>> check._bare_receiver_handled(ast.parse("n: Frame = 1").body[0], namespace)
            False
        """
        if not isinstance(statement, ast.AnnAssign) or statement.value is not None:
            return False
        name = self._name_target(statement.target)
        if name is None:
            return False
        annotation = self._evaluated(statement.annotation, namespace)
        concrete = self._concrete_frame_type(annotation)
        if concrete is None:
            return False
        built, sample = _safely(lambda: self._sample_frame_builder.built(concrete))
        if built:
            namespace[name] = sample
        return built

    def _name_target(self, node: ast.expr) -> str | None:
        return node.id if isinstance(node, ast.Name) else None

    def _executed(self, statement: ast.stmt, namespace: dict[str, object]) -> bool:
        module = ast.Module(body=[statement], type_ignores=[])
        ast.fix_missing_locations(module)

        def run() -> None:
            code = compile(module, "<main>", "exec", dont_inherit=True)
            exec(code, namespace)  # ruff: ignore[exec-builtin]

        succeeded, _ = _safely(run)
        return succeeded

    @propagates(None)
    def _annotated_assignment_check(
        self,
        statement: ast.stmt,
        main_source: str,
        namespace: dict[str, object],
    ) -> list[CheckResult] | None:
        """Compare the value of an annotated assignment with the typed frame that it declares.

        The check reads only an assignment such as `x: Frame = value` to a
        plain name. A line with a `# E:` comment is a wrong assignment on
        purpose, which the case uses to show an error. Its real value does
        not have the stated type, so the check skips it.

        Args:
            statement: One statement that has run.
            main_source: The whole code of the case, which holds the
                comments of the statement.
            namespace: The namespace that the statement ran in.

        Returns:
            One result for the assigned frame. None for a bare declaration
            such as `x: T`, another kind of statement, a target that is not
            a plain name, a line with a `# E:` comment, or an annotation
            that is not a typed frame with columns.

        Example:
            >>> from astrolabe import columns
            >>> check = RuntimeCrossCheck(
            ...     SampleFrameBuilder(DTypeSampleFactory()),
            ...     SchemaExpectation(),
            ...     RevealedTypeReference(),
            ... )
            >>> Frame = TypedDataFrame[columns("X", {"x": pl.Int64})]
            >>> namespace = {"Frame": Frame, "df": pl.DataFrame({"x": [1]})}
            >>> source = "df: Frame = df"
            >>> statement = ast.parse(source).body[0]
            >>> results = check._annotated_assignment_check(statement, source, namespace)
            >>> [result.matched for result in results]
            [True]
            >>> source = "df: Frame"
            >>> statement = ast.parse(source).body[0]
            >>> check._annotated_assignment_check(statement, source, namespace) is None
            True
            >>> source = "df: Frame = df  # E: error"
            >>> statement = ast.parse(source).body[0]
            >>> check._annotated_assignment_check(statement, source, namespace) is None
            True
        """
        if not isinstance(statement, ast.AnnAssign) or statement.value is None:
            return None
        target = not_none(self._name_target(statement.target))
        if self._has_error_comment(statement, main_source):
            return None
        annotation = self._evaluated(statement.annotation, namespace)
        declared = not_none(self._concrete_frame_type(annotation))
        return self._schema_check(
            f"assignment to {target}",
            declared,
            namespace[target],
            closed=True,
        )

    @propagates(None)
    def _reveal_type_check(
        self,
        statement: ast.stmt,
        main_source: str,
        namespace: dict[str, object],
    ) -> list[CheckResult] | None:
        """Compare the value in a `reveal_type(...)` call with the type that its note states.

        The statement must be a call of `reveal_type` with an argument, and
        its line must hold a `# N: Revealed type is "..."` comment. A
        revealed typed frame is compared by its schema, and any other
        revealed type by its class.

        Args:
            statement: One statement that has run.
            main_source: The whole code of the case, which holds the
                comments of the statement.
            namespace: The namespace that the statement ran in.

        Returns:
            The results. None for another kind of statement, for a line
            with no note, or for a type that the check cannot compare.

        Example:
            >>> check = RuntimeCrossCheck(
            ...     SampleFrameBuilder(DTypeSampleFactory()),
            ...     SchemaExpectation(),
            ...     RevealedTypeReference(),
            ... )
            >>> namespace = {"df": pl.DataFrame({"x": [1]})}
            >>> source = 'reveal_type(df)  # N: Revealed type is "polars.dataframe.frame.DataFrame"'
            >>> results = check._reveal_type_check(ast.parse(source).body[0], source, namespace)
            >>> [result.matched for result in results]
            [True]
            >>> other = "print(df)"
            >>> check._reveal_type_check(ast.parse(other).body[0], other, namespace) is None
            True
        """
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)):
            return None
        call = statement.value
        is_reveal_type = isinstance(call.func, ast.Name) and call.func.id == "reveal_type"
        not_falsy(is_reveal_type and call.args)
        line = main_source.splitlines()[statement.lineno - 1]
        note = not_none(self._REVEAL_NOTE.search(line))
        actual = not_none(self._evaluated(call.args[0], namespace))
        label = f"reveal_type at line {statement.lineno}"
        revealed = self._revealed_type_reference.frame(note.group(1))
        if revealed is not None:
            return self._schema_check(label, revealed.frame, actual, closed=revealed.closed)
        return self._class_check(label, note.group(1), actual)

    @propagates(None)
    def _class_check(self, label: str, note_text: str, actual: object) -> list[CheckResult] | None:
        """Compare a real value with a revealed type that is not a frame of `TypedDict` columns.

        Args:
            label: The label of the check.
            note_text: The text of the revealed type.
            actual: The real value.

        Returns:
            The results, or None where the text names no class that the
            check can compare.
        """
        resolved = not_none(self._revealed_type_reference.resolved(note_text))
        concrete = self._concrete_frame_type(resolved)
        if concrete is not None:
            return self._schema_check(label, concrete, actual, closed=True)
        if issubclass(resolved, _TypedFrameBase):
            # A typed frame class with no columns, for example the type that the
            # checker gives an untyped result, is only a static marker. No
            # polars method returns an instance of it, so an isinstance check
            # always fails, and the failure says nothing about the plugin.
            return None
        return [CheckResult(label, resolved, type(actual), isinstance(actual, resolved))]

    def _schema_check(
        self,
        label: str,
        declared: type[_TypedFrameBase],
        actual: object,
        *,
        closed: bool,
    ) -> list[CheckResult]:
        succeeded, actual_schema = _safely(lambda: self._real_schema(actual))
        if not succeeded:
            return []
        matched = self._schema_expectation.matches(declared.schema, actual_schema, closed=closed)
        return [CheckResult(label, declared.schema, actual_schema, matched)]

    def _real_schema(self, value: object) -> Mapping[str, Any]:
        if isinstance(value, pl.LazyFrame):
            return dict(value.collect_schema())
        if isinstance(value, pl.DataFrame):
            return dict(value.schema)
        message = f"{value!r} is not a real polars frame"
        raise TypeError(message)

    def _has_error_comment(self, statement: ast.stmt, main_source: str) -> bool:
        lines = main_source.splitlines()
        end = getattr(statement, "end_lineno", statement.lineno)
        return any("# E:" in lines[index] for index in range(statement.lineno - 1, end))

    def _concrete_frame_type(
        self,
        candidate: object,
    ) -> type[_TypedFrameBase] | None:
        """Keep a value only if it is a typed frame class that declares at least one column.

        A typed frame class with no columns, such as `TypedDataFrame`
        itself, has nothing to compare.

        Args:
            candidate: Any value, for example an evaluated annotation.

        Returns:
            The class, or None.

        Example:
            >>> from astrolabe import columns
            >>> Frame = TypedDataFrame[columns("X", {"x": pl.Int64})]
            >>> check = RuntimeCrossCheck(
            ...     SampleFrameBuilder(DTypeSampleFactory()),
            ...     SchemaExpectation(),
            ...     RevealedTypeReference(),
            ... )
            >>> check._concrete_frame_type(Frame) is Frame
            True
            >>> check._concrete_frame_type(TypedDataFrame) is None
            True
            >>> check._concrete_frame_type(int) is None
            True
        """
        if not isinstance(candidate, type):
            return None
        if not issubclass(candidate, (TypedDataFrame, TypedLazyFrame)):
            return None
        if not candidate.schema:
            return None
        return candidate

    def _evaluated(self, expression: ast.expr, namespace: dict[str, object]) -> object:
        succeeded, value = _safely(
            lambda: eval(compile(ast.Expression(expression), "<main>", "eval"), namespace),  # ruff: ignore[suspicious-eval-usage]
        )
        return value if succeeded else None
