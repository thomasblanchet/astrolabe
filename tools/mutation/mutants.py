"""List every mutant of the source tree, except the ones that no test can ever kill.

The changes come from the operators of cosmic-ray, a mutation testing
library. An operator works on a parso tree, a syntax tree that keeps
the formatting of the source, so a mutated file differs from the
original in the one change only.

This module chooses the operators, and filters the mutants that they
find. A filter drops a mutant that survives every test whatever the
tests check, for example a change inside a type annotation. Such a
mutant adds noise to the report, and no evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from cosmic_ray.ast import ast_nodes, get_ast_from_path
from cosmic_ray.plugins import get_operator

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path


@dataclass(frozen=True)
class Mutant:
    """Name one change to one file, in a form that the next run can find again.

    The occurrence is the index of the change among the changes that one
    operator finds in one file. The cosmic-ray call that applies a
    change takes that index. It counts every position that the operator
    finds, so the index counts them too, including the positions that a
    filter drops. The line is only for a reader of the report.

    Example:
        >>> from pathlib import Path
        >>> Mutant(Path("a.py"), "core/AddNot", 2, 41).line
        41
    """

    path: Path
    operator: str
    occurrence: int
    line: int

    def label(self) -> str:
        """Give a short name for the mutant, for a line of the report.

        Returns:
            The path, the line and the name of the operator without its
            provider.

        Example:
            >>> from pathlib import Path
            >>> Mutant(Path("a.py"), "core/AddNot", 2, 41).label()
            'a.py:41 AddNot'
        """
        return f"{self.path}:{self.line} {self.operator.split('/')[-1]}"


class OperatorRoster:
    """Choose the cosmic-ray operators that the harness runs, one for each kind of change.

    cosmic-ray ships 213 operators. Most of them pair one operator of
    the source with each possible replacement. Eleven mutants for one
    `|` check the same test eleven times, so the roster keeps one
    replacement for each operator of the source. It keeps the
    replacement that inverts the meaning, because a test is most likely
    to miss an inverted operator.

    `VariableReplacer` and `VariableInserter` need a pair of variable
    names as input, so a run over the whole scope cannot build them.

    Example:
        >>> "core/AddNot" in OperatorRoster().names()
        True
        >>> "core/VariableReplacer" in OperatorRoster().names()
        False
    """

    BINARY: ClassVar[tuple[str, ...]] = (
        "Add_Sub",
        "Sub_Add",
        "Mul_Div",
        "Div_Mul",
        "FloorDiv_Div",
        "Mod_Mul",
        "Pow_Mul",
        "RShift_LShift",
        "LShift_RShift",
        "BitOr_BitAnd",
        "BitAnd_BitOr",
        "BitXor_BitAnd",
    )
    COMPARISON: ClassVar[tuple[str, ...]] = (
        "Eq_NotEq",
        "NotEq_Eq",
        "Lt_LtE",
        "LtE_Lt",
        "Gt_GtE",
        "GtE_Gt",
        "Is_IsNot",
        "IsNot_Is",
    )
    UNARY: ClassVar[tuple[str, ...]] = (
        "Delete_Not",
        "Delete_USub",
        "Delete_Invert",
        "USub_UAdd",
    )
    PLAIN: ClassVar[tuple[str, ...]] = (
        "AddNot",
        "ReplaceTrueWithFalse",
        "ReplaceFalseWithTrue",
        "ReplaceAndWithOr",
        "ReplaceOrWithAnd",
        "ReplaceBreakWithContinue",
        "ReplaceContinueWithBreak",
        "ExceptionReplacer",
        "NumberReplacer",
        "RemoveDecorator",
        "ZeroIterationForLoop",
    )

    def names(self) -> tuple[str, ...]:
        """Give the name of every operator in the roster.

        The roster holds four families: the binary, the comparison, the
        unary and the plain operators.

        Returns:
            The names, each in the `provider/name` form of cosmic-ray.

        Example:
            >>> len(OperatorRoster().names())
            35
            >>> "core/ReplaceComparisonOperator_Eq_NotEq" in OperatorRoster().names()
            True
        """
        binary = self._family("ReplaceBinaryOperator_", self.BINARY)
        comparison = self._family("ReplaceComparisonOperator_", self.COMPARISON)
        unary = self._family("ReplaceUnaryOperator_", self.UNARY)
        return binary + comparison + unary + self._family("", self.PLAIN)

    def _family(self, prefix: str, suffixes: Sequence[str]) -> tuple[str, ...]:
        """Build the full names of one family of operators.

        Args:
            prefix: The start of the class name of the family, or empty
                for an operator with a plain name.
            suffixes: The part of each name that differs in the family.

        Returns:
            The full names, with the `core/` provider.

        Example:
            >>> OperatorRoster()._family("X_", ("a", "b"))
            ('core/X_a', 'core/X_b')
        """
        return tuple(f"core/{prefix}{suffix}" for suffix in suffixes)

    def operators(self) -> tuple[tuple[str, Any], ...]:
        """Build the operator object for each name, so that it can find and apply changes.

        Returns:
            Each name beside its operator.

        Example:
            >>> name, operator = OperatorRoster().operators()[0]
            >>> name
            'core/ReplaceBinaryOperator_Add_Sub'
        """
        return tuple((name, get_operator(name)()) for name in self.names())


@dataclass(frozen=True)
class Site:
    """Hold one position that an operator can change, with the facts that a filter reads.

    A filter needs more than the node. It can read the lines around it,
    the operator that found it, and which of the changes of that operator
    this one is. One value for all of these gives every filter the same
    call.

    Attributes:
        node: The parso node that the operator found.
        lines: Every line of the file.
        operator: The name of the operator.
        occurrence: The index of this change among the changes of the
            operator in the file.
        start: The first line of the change, counted from one.
        end: The last line of the change.

    Example:
        >>> Site(None, ("x = 1",), "core/AddNot", 0, 1, 1).start
        1
    """

    node: Any
    lines: Sequence[str]
    operator: str
    occurrence: int
    start: int
    end: int


class AnnotationSite:
    """Drop a mutant inside a type annotation, which no code evaluates.

    Every module of the project starts with
    `from __future__ import annotations`, so Python keeps each annotation
    as a string and never evaluates it. A mutant there cannot change what
    the code does, and it survives every test. Most `|` in the source
    tree joins two types, so without this filter the report is mostly
    noise.

    Example:
        >>> AnnotationSite().covers(Site(None, (), "x", 0, 1, 1))
        False
    """

    ANNOTATED: ClassVar[frozenset[str]] = frozenset({"tname", "tfpdef"})
    HOLDERS: ClassVar[frozenset[str]] = frozenset({"annassign", "funcdef"})

    def covers(self, site: Site) -> bool:
        """Answer whether the position is inside an annotation.

        The walk goes from the node up to the root, and stops at the
        first node that settles the answer.

        Args:
            site: The position, with the facts around it.

        Returns:
            True if the position is inside an annotation.

        Example:
            >>> AnnotationSite().covers(Site(None, (), "x", 0, 1, 1))
            False
        """
        for child, parent in self._ancestry(site.node):
            verdict = self._verdict(child, parent)
            if verdict is not None:
                return verdict
        return False

    def _ancestry(self, node: Any) -> Iterator[tuple[Any, Any]]:
        """Walk from a node up to the root of the tree.

        Args:
            node: The node to start at.

        Yields:
            Each node beside its parent, from the node upward.

        Example:
            >>> list(AnnotationSite()._ancestry(None))
            []
        """
        child, parent = node, getattr(node, "parent", None)
        while parent is not None:
            yield child, parent
            child, parent = parent, parent.parent

    def _verdict(self, child: Any, parent: Any) -> bool | None:
        """Settle the answer at one step of the walk, if this step can.

        A `tname` or `tfpdef` node settles it as True. An `annassign`
        node holds both the annotation and the assigned value of an
        annotated assignment, so there the answer depends on whether the
        walk came up through the annotation. At a function, the answer
        depends on whether the walk came up through the return
        annotation. At a parameter with an annotation, the answer is
        False if the walk came up through the default value, else True.
        The default value runs when the function is defined, so a mutant
        in it is not inside the annotation.

        Args:
            child: The node that the walk came from.
            parent: The node that the walk reached.

        Returns:
            The answer, or None if the walk must go on.

        Example:
            >>> AnnotationSite()._verdict(None, None) is None
            True
        """
        kind = getattr(parent, "type", "")
        if kind in self.ANNOTATED:
            return True
        if kind in self.HOLDERS:
            return child is self._annotation_of(parent, kind)
        if kind != "param" or getattr(parent, "annotation", None) is None:
            return None
        return child is not parent.default

    def _annotation_of(self, parent: Any, kind: str) -> Any:
        """Give the annotation of an annotated assignment or of a function.

        The children of an `annassign` node are the colon, the
        annotation, and then the equals sign and the value, if the
        assignment has one.

        Args:
            parent: The node that holds the annotation.
            kind: Its parso type, `annassign` or `funcdef`.

        Returns:
            The node of the annotation, or None for a function with no
            return annotation.

        Example:
            >>> import parso
            >>> tree = parso.parse("x: int = 1")
            >>> held = tree.children[0].children[1]
            >>> AnnotationSite()._annotation_of(held, "annassign").get_code()
            ' int'
        """
        if kind == "annassign":
            return parent.children[1]
        return getattr(parent, "annotation", None)


class PragmaSite:
    """Drop a mutant on a line that carries `# pragma: no cover`.

    The project marks a branch that cannot run with this comment, and
    writes the reason beside it. Such a line never runs, so a mutant on
    it always survives and says nothing about the tests.

    The filter of cosmic-ray reads the last line of a span alone. This
    one reads every line of the span, because a span can start on a line
    with no pragma and end on a line with one.

    Example:
        >>> marked = ("x = 1  # pragma: no cover",)
        >>> PragmaSite().covers(Site(None, marked, "x", 0, 1, 1))
        True
    """

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"#\s*pragma:\s*no cover")

    def covers(self, site: Site) -> bool:
        """Answer whether any line of the span carries the pragma.

        Args:
            site: The position, with the facts around it.

        Returns:
            True if a line of the span carries it.

        Example:
            >>> lines = ("a", "b  # pragma: no cover")
            >>> PragmaSite().covers(Site(None, lines, "x", 0, 1, 2))
            True
        """
        span = site.lines[site.start - 1 : site.end]
        return any(self.PATTERN.search(line) for line in span)


class TypeCheckingSite:
    """Drop a mutant on an `if TYPE_CHECKING:` guard.

    `typing.TYPE_CHECKING` is False while the code runs, and in this
    project the guarded block holds imports alone. A negated guard only
    runs those imports, which succeed, so nothing that a test can read
    changes. Each such mutant survives every test, and every module that
    imports a name for its annotations alone has one.

    Example:
        >>> TypeCheckingSite().covers(Site(None, ("if TYPE_CHECKING:",), "x", 0, 1, 1))
        True
    """

    PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*if\s+(typing\.)?TYPE_CHECKING\s*:",
    )

    def covers(self, site: Site) -> bool:
        """Answer whether the first line of the span is the guard.

        Args:
            site: The position, with the facts around it.

        Returns:
            True if the mutant changes the guard.

        Example:
            >>> lines = ("x = 1", "if TYPE_CHECKING:")
            >>> TypeCheckingSite().covers(Site(None, lines, "x", 0, 2, 2))
            True
        """
        return bool(self.PATTERN.match(site.lines[site.start - 1]))


class IndexOffset:
    """Drop the downward change of a number that indexes a sequence.

    `NumberReplacer` changes a number twice, once up and once down. On a
    subscript, `[0]` becomes `[1]` and `[-1]`. If a guard above holds the
    sequence to one member, `[-1]` reads the same member as `[0]`, and
    no test can tell the two apart. The upward change still mutates the
    same subscript, so the position keeps one mutant.

    Example:
        >>> IndexOffset().covers(Site(None, (), "core/AddNot", 1, 1, 1))
        False
    """

    OPERATOR: ClassVar[str] = "core/NumberReplacer"

    def covers(self, site: Site) -> bool:
        """Answer whether the position is the downward change of a number in a subscript.

        The occurrences of `NumberReplacer` alternate up and down for
        each number, so an odd occurrence is the downward change. parso
        puts a subscript in a `trailer` node, whose code opens with a
        bracket. A number in a sum sits under an arithmetic node instead.

        Args:
            site: The position, with the facts around it.

        Returns:
            True for the downward change of an index.

        Example:
            >>> IndexOffset().covers(Site(None, (), "core/NumberReplacer", 0, 1, 1))
            False
            >>> IndexOffset().covers(Site(None, (), "core/NumberReplacer", 1, 1, 1))
            False
        """
        if site.operator != self.OPERATOR or not site.occurrence % 2:
            return False
        parent = getattr(site.node, "parent", None)
        if parent is None or parent.type != "trailer":
            return False
        code: str = parent.get_code().strip()
        return code.startswith("[")


class FrozenFlag:
    """Drop the change that turns `frozen=True` into `frozen=False` on a dataclass.

    Of the frozen dataclasses in the scope, the code hashes one, and a
    doctest already checks that one. No code relies on the others being
    immutable. A test for each of them checks a property that no caller
    uses, and the rules of the project for tests do not allow such a
    test.

    Example:
        >>> FrozenFlag().covers(Site(None, ("x = 1",), "core/AddNot", 0, 1, 1))
        False
    """

    OPERATOR: ClassVar[str] = "core/ReplaceTrueWithFalse"
    PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"@dataclass\(.*frozen=True")

    def covers(self, site: Site) -> bool:
        """Answer whether the position is the `frozen=True` flag of a dataclass.

        Args:
            site: The position, with the facts around it.

        Returns:
            True if the change only makes a dataclass mutable.

        Example:
            >>> marked = ("@dataclass(frozen=True)",)
            >>> FrozenFlag().covers(Site(None, marked, "core/ReplaceTrueWithFalse", 0, 1, 1))
            True
        """
        if site.operator != self.OPERATOR:
            return False
        return bool(self.PATTERN.search(site.lines[site.start - 1]))


class StrictZip:
    """Drop the change that turns `strict=True` into `strict=False` in a call of `zip`.

    A strict `zip` differs from a plain one only for two sequences of
    different lengths, where it raises. Each strict `zip` in the scope
    pairs two sequences of one length: two lists that mypy builds in
    parallel, for example the nodes and the types of the arguments, or two
    lists that a guard above has already compared. The two forms then give
    the same pairs, and no test can tell them apart.

    Example:
        >>> StrictZip().covers(Site(None, (), "core/AddNot", 0, 1, 1))
        False
    """

    OPERATOR: ClassVar[str] = "core/ReplaceTrueWithFalse"

    def covers(self, site: Site) -> bool:
        """Answer whether the position is the `strict=True` of a call of `zip`.

        Args:
            site: The position, with the facts around it.

        Returns:
            True for the flag of a strict `zip`.

        Example:
            >>> import parso
            >>> flag = parso.parse("zip(a, b, strict=True)").get_first_leaf()
            >>> while flag.value != "True": flag = flag.get_next_leaf()
            >>> StrictZip().covers(Site(flag, (), "core/ReplaceTrueWithFalse", 0, 1, 1))
            True
        """
        if site.operator != self.OPERATOR:
            return False
        return CallShape().keyword(site.node) == "strict" and CallShape().callee(site.node) == "zip"


class SplitBound:
    """Drop the upward change of `maxsplit=1` where the call keeps one end of the text.

    `s.split(".", maxsplit=1)[0]` keeps the text before the first cut,
    and `s.rsplit(".", maxsplit=1)[-1]` keeps the text after the last
    cut. More cuts do not change that piece, so `maxsplit=2` gives the
    same answer for every text. The downward change, `maxsplit=0`, keeps
    the whole text, so that mutant stays.

    Example:
        >>> SplitBound().covers(Site(None, (), "core/AddNot", 0, 1, 1))
        False
    """

    OPERATOR: ClassVar[str] = "core/NumberReplacer"
    ENDS: ClassVar[dict[str, str]] = {"split": "[0]", "rsplit": "[-1]"}

    def covers(self, site: Site) -> bool:
        """Answer whether the position is a `maxsplit` of one that raises to two, with no effect.

        The occurrences of `NumberReplacer` alternate up and down for
        each number, so an even occurrence is the upward change. The
        call must be `split(...)[0]` or `rsplit(...)[-1]`, and the number
        must be its last argument.

        Args:
            site: The position, with the facts around it.

        Returns:
            True for the upward change of such a bound.

        Example:
            >>> import parso
            >>> one = parso.parse('s.split(".", maxsplit=1)[0]').get_first_leaf()
            >>> while one.value != "1": one = one.get_next_leaf()
            >>> SplitBound().covers(Site(one, (), "core/NumberReplacer", 0, 1, 1))
            True
        """
        if site.operator != self.OPERATOR or site.occurrence % 2:
            return False
        if getattr(site.node, "value", "") != "1":
            return False
        shape = CallShape()
        call = shape.call_trailer(site.node)
        if call is None or not shape.last_argument(call, site.node):
            return False
        return shape.next_code(call) == self.ENDS.get(shape.callee(site.node), "")


class KeywordMarker:
    """Drop the change of the bare `*` in a list of parameters into `/`.

    A bare `*` makes every parameter after it keyword-only. A `/` in its
    place makes the parameters before it positional-only, and lets the
    parameters after it arrive by position or by keyword. Every call in
    the scope passes those parameters by keyword, as the `*` demands, so
    each call binds the same values after the change.

    Example:
        >>> KeywordMarker().covers(Site(None, (), "core/AddNot", 0, 1, 1))
        False
    """

    OPERATOR: ClassVar[str] = "core/ReplaceBinaryOperator_Mul_Div"

    def covers(self, site: Site) -> bool:
        """Answer whether the position is a bare `*` in a list of parameters.

        Args:
            site: The position, with the facts around it.

        Returns:
            True for the marker of the keyword-only parameters.

        Example:
            >>> import parso
            >>> star = parso.parse("def f(a, *, b): pass").get_first_leaf()
            >>> while star.value != "*": star = star.get_next_leaf()
            >>> operator = "core/ReplaceBinaryOperator_Mul_Div"
            >>> KeywordMarker().covers(Site(star, (), operator, 0, 1, 1))
            True
        """
        if site.operator != self.OPERATOR:
            return False
        parent = getattr(site.node, "parent", None)
        return getattr(parent, "type", "") == "parameters"


class CallShape:
    """Read the call around a node of a parso tree, for the filters that need the callee.

    parso holds `f(a, b=1)[0]` as a name followed by trailers: `(a, b=1)`
    and then `[0]`. A method call `s.split(...)` has one more trailer,
    `.split`, before the parentheses.

    Example:
        >>> CallShape().call_trailer(None) is None
        True
    """

    def call_trailer(self, node: Any) -> Any:
        """Give the parentheses of the nearest call around a node.

        Args:
            node: A node inside the arguments of a call.

        Returns:
            The trailer that holds the arguments, or None if the node is
            not inside a call.

        Example:
            >>> import parso
            >>> name = parso.parse("f(a)").get_first_leaf().get_next_leaf().get_next_leaf()
            >>> CallShape().call_trailer(name).get_code()
            '(a)'
        """
        parent = getattr(node, "parent", None)
        while parent is not None and not self._is_call(parent):
            parent = parent.parent
        return parent

    def callee(self, node: Any) -> str:
        """Give the name of the function or method that the nearest call around a node calls.

        Args:
            node: A node inside the arguments of a call.

        Returns:
            The name, or an empty string if there is none.

        Example:
            >>> import parso
            >>> name = parso.parse("s.split(a)").get_first_leaf()
            >>> while name.value != "a": name = name.get_next_leaf()
            >>> CallShape().callee(name)
            'split'
        """
        call = self.call_trailer(node)
        before = call.get_previous_sibling() if call is not None else None
        return getattr(before, "get_code", lambda: "")().strip().lstrip(".")

    def keyword(self, node: Any) -> str:
        """Give the keyword that a value is passed under, if any.

        Args:
            node: The value.

        Returns:
            The keyword, or an empty string for a value passed by
            position.

        Example:
            >>> CallShape().keyword(None)
            ''
        """
        parent: Any = getattr(node, "parent", None)
        if getattr(parent, "type", "") != "argument":
            return ""
        return str(parent.children[0].get_code().strip())

    def last_argument(self, call: Any, node: Any) -> bool:
        """Answer whether a node is the last argument of a call, or its value.

        Args:
            call: The trailer that holds the arguments.
            node: The node.

        Returns:
            True if the node ends the argument list.

        Example:
            >>> import parso
            >>> one = parso.parse("f(a, 1)").get_first_leaf()
            >>> while one.value != "1": one = one.get_next_leaf()
            >>> CallShape().last_argument(CallShape().call_trailer(one), one)
            True
        """
        return bool(node.end_pos == call.children[-2].end_pos)

    def next_code(self, call: Any) -> str:
        """Give the code of the trailer after a call, for example `[0]`.

        Args:
            call: The trailer that holds the arguments.

        Returns:
            The code, or an empty string if nothing follows the call.

        Example:
            >>> import parso
            >>> name = parso.parse("f(a)[0]").get_first_leaf().get_next_leaf().get_next_leaf()
            >>> CallShape().next_code(CallShape().call_trailer(name))
            '[0]'
        """
        after = call.get_next_sibling()
        return str(after.get_code().strip()) if after is not None else ""

    def _is_call(self, node: Any) -> bool:
        """Answer whether a node is the parentheses of a call.

        Args:
            node: A node of the tree.

        Returns:
            True for a trailer that opens with a parenthesis.

        Example:
            >>> CallShape()._is_call(None)
            False
        """
        if getattr(node, "type", "") != "trailer":
            return False
        return bool(node.children[0].get_code().strip() == "(")


class MutantCatalog:
    """List every mutant of a source tree that the filters keep.

    The catalog parses each file once and walks its nodes once. It then
    asks every operator about each node. A parse for each operator costs
    five minutes on one file of 19,000 lines, the generated expression
    catalog.

    The catalog leaves out `frame.py` at the root of the scope, that is
    `src/astrolabe/frame.py`. That module holds the classes that a user
    annotates with, and their behavior at run time. A yml case checks
    how the checker reads them, and never runs them, so a mutant there
    says nothing about the yml cases. A file of the same name deeper in
    the tree stays in the catalog.

    Example:
        >>> isinstance(MutantCatalog().roster, OperatorRoster)
        True
    """

    SKIP: ClassVar[frozenset[str]] = frozenset({"frame.py"})

    def __init__(self, roster: OperatorRoster | None = None) -> None:
        """Set the operators and the filters of the catalog.

        Args:
            roster: The operators to run. The default roster if None.

        Example:
            >>> len(MutantCatalog().filters)
            8
        """
        self.roster = roster or OperatorRoster()
        self.filters = (
            AnnotationSite(),
            TypeCheckingSite(),
            IndexOffset(),
            FrozenFlag(),
            PragmaSite(),
            StrictZip(),
            SplitBound(),
            KeywordMarker(),
        )

    def mutants(self, root: Path) -> list[Mutant]:
        """List every mutant under a directory.

        Args:
            root: The directory to walk.

        Returns:
            Every mutant that the filters keep, sorted by file.

        Example:
            >>> from pathlib import Path
            >>> MutantCatalog().mutants(Path("tools/mutation")) != []
            True
        """
        found: list[Mutant] = []
        for path in self.files(root):
            found.extend(self.for_file(path))
        return found

    def files(self, root: Path) -> list[Path]:
        """List the files to mutate.

        Args:
            root: The directory to walk.

        Returns:
            Every Python file under the directory that the catalog does
            not skip, sorted.

        Example:
            >>> from pathlib import Path
            >>> MutantCatalog().files(Path("tools/mutation"))[0].name
            '__init__.py'
        """
        found = root.rglob("*.py")
        return sorted(p for p in found if p.relative_to(root).as_posix() not in self.SKIP)

    def for_file(self, path: Path) -> list[Mutant]:
        """List the mutants of one file.

        Args:
            path: The file to read.

        The position of each change and the facts around it go into one
        value, a `Site`, that every filter reads.

        Returns:
            The mutants of the file that every filter keeps.

        Example:
            >>> from pathlib import Path
            >>> MutantCatalog().for_file(Path("tools/mutation/__init__.py"))
            []
        """
        nodes = list(ast_nodes(get_ast_from_path(path)))
        lines = path.read_text(encoding="utf-8").splitlines()
        found: list[Mutant] = []
        for name, operator in self.roster.operators():
            spans = self._spans(nodes, operator)
            for occurrence, (node, (start, end)) in enumerate(spans):
                site = Site(node, lines, name, occurrence, start[0], end[0])
                if any(drop.covers(site) for drop in self.filters):
                    continue
                found.append(Mutant(path, name, occurrence, site.start))
        return found

    def _spans(self, nodes: Sequence[Any], operator: Any) -> list[tuple[Any, Any]]:
        """Find every position that one operator can change.

        The order is the order in which the cosmic-ray call that applies
        a change counts. The index of a span in this list is therefore
        its occurrence.

        Args:
            nodes: The nodes of the file.
            operator: The operator to ask.

        Returns:
            Each node beside one of its spans.

        Example:
            >>> MutantCatalog()._spans([], None)
            []
        """
        found: list[tuple[Any, Any]] = []
        for node in nodes:
            found.extend((node, span) for span in operator.mutation_positions(node))
        return found
