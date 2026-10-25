"""Read a deferred argument of a frame operation into plain values, and evaluate it there.

In the deferred model, the type of an expression stores its code. A frame
operation such as `select` must turn such an argument into the column that it
gives on the receiver. This module reads the parts of the mypy type that the
evaluation needs: the code of each member of a union, the name and the data
type of each declared column that the code reads, and the data types of the
receiver. It then evaluates the code (`DeferredEvaluation`). It reports
nothing: the code that resolves the argument turns the answer into a column or
an error.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

import polars as pl
from mypy.types import Instance, UnionType, get_proper_type

from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.deferred.code import Code
from astrolabe._hooks.deferred.code_reader import HoleValues
from astrolabe._hooks.deferred.open_types import OpenType
from astrolabe._hooks.encoding.expr_evaluation import ExpressionArgument, ExpressionEvaluation
from astrolabe._hooks.expression.materialize import Materializer
from astrolabe._hooks.expression.selection import SelectionType
from astrolabe._hooks.expression.fit import SliceDecl
from astrolabe._hooks.expression.rendering import Rendered
from astrolabe._rules.options import RowForm
from astrolabe._hooks.deferred.evaluation import DeclaredLeaf, DeferredEvaluation
from astrolabe._hooks.deferred.open_types import LeafChoices
from astrolabe._hooks.deferred.polars_selector import NameRequirement, PolarsSelector
from astrolabe._hooks.deferred.selection_fit import CodeSelector, DeferredSelectionFit
from astrolabe._hooks.deferred.remainder import DeferredRemainder
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.marker_minter import ProperReads
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeReads
from astrolabe._hooks.encoding.selection_codec import SelectionReader
from astrolabe._hooks.expression.lifting import UNCHECKED
from astrolabe._hooks.recording.deferred_types import DeferredReader
from astrolabe.expr import Blank, TypedColumn

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from mypy.types import Type as MypyType

    from astrolabe._hooks.deferred.evaluation import DeferredAnswer
    from astrolabe._hooks.domain.column import ColumnRef
    from collections.abc import Mapping

    from astrolabe._hooks.expression.fit import Row
    from astrolabe._hooks.expression.selection import RemainderEffect
    from astrolabe._hooks.remainder.record import Record
    from astrolabe._hooks.remainder.selector import Selector


class DeferredArgument:
    """Give the answer of a deferred argument at the declared columns of a frame.

    Example:
        >>> DeferredArgument("row").codes(new_type_fixture().anyt) is None
        True
    """

    COLUMN: ClassVar[Target] = Target.of(TypedColumn)
    FAMILY: ClassVar[Target] = Target.of(pl.DataType)
    BLANK: ClassVar[Target] = Target.of(Blank)
    TEXT: ClassVar[Target] = Target.of(str)

    def __init__(self, context: str) -> None:
        """Keep the context of the frame operation.

        Args:
            context: `row` for a context such as `select`, or `agg`.

        Example:
            >>> DeferredArgument("agg")._context
            'agg'
        """
        self._context = context

    def codes(self, typ: MypyType) -> list[Code[MypyType]] | None:
        """Give the code of each member of the type of an argument.

        Args:
            typ: The type of the argument.

        Returns:
            One code for each member of a union, or `None` where a member
            holds no code.

        Example:
            >>> DeferredArgument("row").codes(new_type_fixture().anyt) is None
            True
        """
        codes: list[Code[MypyType]] = []
        reader = DeferredReader()
        for member in reader.members(typ):
            code = reader.code(member)
            if code is None:
                return None
            codes.append(code)
        return codes

    def answer(self, code: Code[MypyType], columns: Sequence[ColumnRef]) -> DeferredAnswer:
        """Give the answer of one code at the declared columns of a frame.

        Args:
            code: The code.
            columns: The declared columns of the frame.

        Returns:
            The answer.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = typed_column_instance(fixture, "price", int64_dtype(fixture))
            >>> answer = DeferredArgument("row").answer(Code("$0 * 2", (price,)), [])
            >>> answer.name, answer.rendering
            ('price', 'Int64')
        """
        renderings = self.renderings(columns)
        declared = self.declared(code)
        answer = self._evaluation(columns).answer(declared, renderings)
        stored = self._stored_evaluation(columns)
        if stored is None:
            return answer
        return self._with_stored(answer, stored.answer(declared, renderings))

    def planned(self, columns: Sequence[ColumnRef]) -> dict[str, str]:
        """Give the stored type of each column of a lazy frame that has one of its own.

        A lazy frame plans each step at the type that it stores for a column.
        Where that type differs from the type of the data, the column holds
        `Planned[run, planner]`.

        Args:
            columns: The declared columns of the frame.

        Returns:
            The rendering of the stored type of each such column, by its name.

        Example:
            >>> fixture = new_type_fixture()
            >>> small = column_ref(fixture, "a", int8_dtype(fixture))
            >>> DeferredArgument("lazy row").planned([small])
            {}
        """
        reads = ConstDtypeReads()
        planned: dict[str, str] = {}
        for column in columns:
            stored = reads.planner(column.dtype)
            if stored is not None and column.name is not None:
                planned[column.name] = stored
        return planned

    def _evaluation(self, columns: Sequence[ColumnRef]) -> DeferredEvaluation:
        """Give the evaluation of the context at the columns of a frame, with their stored types.

        Args:
            columns: The declared columns of the frame.

        Returns:
            The evaluation.

        Example:
            >>> DeferredArgument("lazy row")._evaluation([])._planned
            {}
        """
        return DeferredEvaluation(self._context, self.planned(columns))

    def _stored_evaluation(self, columns: Sequence[ColumnRef]) -> DeferredEvaluation | None:
        """Give the evaluation of the type that a lazy frame stores, or `None` for an eager frame.

        Args:
            columns: The declared columns of the frame.

        Returns:
            The evaluation, or `None`.

        Example:
            >>> DeferredArgument("row")._stored_evaluation([]) is None
            True
        """
        if not self._context.startswith(DeferredEvaluation.LAZY):
            return None
        return DeferredEvaluation(self._context + DeferredEvaluation.STORED, self.planned(columns))

    def _with_stored(self, answer: DeferredAnswer, stored: DeferredAnswer) -> DeferredAnswer:
        """Give an answer the type that a lazy frame stores, where both types are known and differ.

        Args:
            answer: The answer of the data.
            stored: The answer of the stored type.

        Returns:
            The answer.

        Example:
            >>> from astrolabe._hooks.deferred.evaluation import DeferredAnswer
            >>> run, planned = DeferredAnswer("d", "Float64"), DeferredAnswer("d", "Float32")
            >>> DeferredArgument("lazy row")._with_stored(run, planned).stored
            'Float32'
        """
        same = answer.refused or stored.refused or stored.rendering == answer.rendering
        choices = LeafChoices()
        known = choices.closed(answer.rendering) and choices.closed(stored.rendering)
        return answer if same or not known else replace(answer, stored=stored.rendering)

    def selector(self, code: Code[MypyType]) -> object | None:
        """Give the one selector of a code, serialized.

        Args:
            code: The code.

        Returns:
            The selector, or `None` where the code holds no selector or more than one.

        Example:
            >>> DeferredArgument("row").selector(Code("cs.numeric()"))
            {'Selector': {'ByDType': 'Numeric'}}
        """
        return DeferredEvaluation(self._context).selector(self.declared(code))

    def selected(
        self,
        code: Code[MypyType],
        columns: Sequence[ColumnRef],
        selector: object,
        selectable: Collection[str],
    ) -> list[DeferredAnswer] | None:
        """Give one answer for each declared column that the selector of a code picks.

        Args:
            code: The code.
            columns: The declared columns of the frame.
            selector: The serialized selector of the code.
            selectable: The names of the columns that the selector can pick.
                `agg` keeps its group keys out of every selector, and the
                expression can still read them.

        Returns:
            The answers, or `None` where the checker cannot know which
            columns the selector picks.

        Example:
            >>> fixture = new_type_fixture()
            >>> small = column_ref(fixture, "a", int8_dtype(fixture))
            >>> numeric = {"Selector": {"ByDType": "Numeric"}}
            >>> argument = DeferredArgument("row")
            >>> answers = argument.selected(Code("cs.numeric()"), [small], numeric, ["a"])
            >>> [(answer.name, answer.rendering) for answer in answers]
            [('a', 'Int8')]
        """
        renderings = self.renderings(columns)
        declared = self.declared(code)
        answers = self._evaluation(columns).selected(declared, renderings, selector, selectable)
        stored = self._stored_evaluation(columns)
        if answers is None or stored is None:
            return answers
        stored_answers = stored.selected(declared, renderings, selector, selectable)
        return self._all_stored(answers, stored_answers)

    def _all_stored(
        self,
        answers: list[DeferredAnswer],
        stored: list[DeferredAnswer] | None,
    ) -> list[DeferredAnswer]:
        """Give each answer of an expansion the type that a lazy frame stores, in order.

        Args:
            answers: The answers of the data.
            stored: The answers of the stored types, in the same order, or
                `None` where the evaluation gives none.

        Returns:
            The answers.

        Example:
            >>> from astrolabe._hooks.deferred.evaluation import DeferredAnswer
            >>> run = [DeferredAnswer("d", "Float64")]
            >>> DeferredArgument("lazy row")._all_stored(run, None) == run
            True
        """
        if (  # pragma: no cover -- both evaluations pick the same columns
            stored is None or len(stored) != len(answers)
        ):
            return answers
        pairs = zip(answers, stored, strict=True)
        return list(itertools.starmap(self._with_stored, pairs))

    def picked(self, code: Code[MypyType], columns: Sequence[ColumnRef]) -> PickedColumns | None:
        """Give the declared columns that the selector of a code picks, as polars expands it.

        A column whose data type has holes is picked where the selector picks
        a data type of its sample. Where it does not pick it at each of them,
        the data decide whether it picks the column.

        Args:
            code: The code.
            columns: The declared columns of the frame.

        Returns:
            The picked columns. `None` where the code holds no selector, or
            where the checker cannot know which columns it picks.

        Example:
            >>> fixture = new_type_fixture()
            >>> small = column_ref(fixture, "a", int8_dtype(fixture))
            >>> text = column_ref(fixture, "t", str_dtype(fixture))
            >>> DeferredArgument("row").picked(Code("cs.numeric()"), [small, text]).names
            ['a']
            >>> loose = column_ref(fixture, "u", unchecked_instance(fixture))
            >>> DeferredArgument("row").picked(Code("cs.numeric()"), [loose]).partial
            ['u']
            >>> DeferredArgument("row").picked(Code("pl.col('a')"), [small]) is None
            True
            >>> DeferredArgument("row").picked(Code("cs.matches(hole('str'))"), [small]) is None
            True
        """
        evaluation = self._evaluation(columns)
        selector = evaluation.selector(self.declared(code))
        if selector is None:
            return None
        renderings = self.renderings(columns)
        picked = evaluation.picked_columns(renderings, selector)
        if picked is None:
            return None
        partial = [
            name
            for name, samples in picked.items()
            if samples and len(samples) < len(OpenType().samples(renderings[name]))
        ]
        return PickedColumns(list(picked), partial)

    def missing(self, selector: object, columns: Sequence[ColumnRef]) -> list[str]:
        """Give each name that a selector requires and the declared columns do not hold.

        Polars raises `ColumnNotFoundError` for such a name, unless the
        remainder of an open frame holds it.

        Args:
            selector: The serialized selector.
            columns: The declared columns of the frame.

        Returns:
            The names, in the order of the selector.

        Example:
            >>> fixture = new_type_fixture()
            >>> small = column_ref(fixture, "a", int8_dtype(fixture))
            >>> names = {"Selector": {"ByName": {"names": ["a", "zz"], "strict": True}}}
            >>> DeferredArgument("row").missing(names, [small])
            ['zz']
        """
        declared = {column.name for column in columns}
        return [name for name in NameRequirement().required(selector) if name not in declared]

    def selection_verdict(self, code: Code[MypyType], decl: SelectionType) -> str | None:
        """Give the first reason that the code of an argument does not fit a declared selection.

        Args:
            code: The code of the argument.
            decl: The declared selection.

        Returns:
            The reason, `DeferredSelectionFit.NOT_SELECTION` where the code
            holds no selector, or `None` where the argument fits.

        Example:
            >>> from astrolabe._hooks.expression.selection import SelectionType
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> decl = SelectionType.identity(OfType(Family.numeric()))
            >>> DeferredArgument("row").selection_verdict(Code("pl.col('a')"), decl)
            'this value is not a selection of columns'
        """
        selector = self._code_selector(code)
        if selector is None:
            return DeferredSelectionFit.NOT_SELECTION
        return DeferredSelectionFit().verdict(self.declared(code), selector, decl)

    def selector_rule(self, code: Code[MypyType]) -> Selector | None:
        """Give the rule of the columns that the selector of a code picks.

        Args:
            code: The code.

        Returns:
            The rule, or `None` where the code holds no selector that the
            checker reads.

        Example:
            >>> DeferredArgument("row").selector_rule(Code("cs.numeric() - cs.float()"))
            SelAnd(left=OfType(family=NumericType), right=SelNot(inner=OfType(family=FloatType)))
            >>> DeferredArgument("row").selector_rule(Code("cs.first()")) is None
            True
        """
        selector = self._code_selector(code)
        return None if selector is None else selector.rule

    def _code_selector(self, code: Code[MypyType]) -> CodeSelector | None:
        """Give the selector of a code: a selector of polars, or one leaf that declares a selection.

        Args:
            code: The code.

        Returns:
            The selector, or `None` where the code holds no selector that the
            checker reads.

        Example:
            >>> DeferredArgument("row")._code_selector(Code("cs.numeric().abs()")).rule
            OfType(family=NumericType)
        """
        node = DeferredEvaluation(self._context).selector(self.declared(code))
        if node is not None:
            rule = PolarsSelector().of(node.get("Selector") if isinstance(node, dict) else None)
            return None if rule is None else CodeSelector(node, rule)
        return self.leaf_selector(code)

    def leaf_selector(self, code: Code[MypyType]) -> CodeSelector | None:
        """Give the one leaf of a code that declares a selection, with the rule of its picks.

        Args:
            code: The code.

        Returns:
            The leaf as a selector, or `None` where the code reads no
            declared selection, or more than one.

        Example:
            >>> DeferredArgument("row").leaf_selector(Code("pl.col('a')")) is None
            True
        """
        found: list[CodeSelector] = []
        for place, leaf in enumerate(code.leaves):
            selection = SelectionReader().decoded(leaf)
            if selection is not None:
                node = {"Column": DeferredEvaluation.LEAF.format(place)}
                found.append(CodeSelector(node, selection.selector))
        return found[0] if len(found) == 1 else None

    def leaf_selected(
        self,
        code: Code[MypyType],
        columns: Sequence[ColumnRef],
        selector: CodeSelector,
        selectable: Collection[str],
    ) -> list[DeferredAnswer]:
        """Give one answer for each declared column that the declared selection of a code picks.

        Args:
            code: The code.
            columns: The declared columns of the frame.
            selector: The leaf of the declared selection, with its rule.
            selectable: The names of the columns that the selection can pick.

        Returns:
            The answers, in the order of the frame.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> leaf = CodeSelector({"Column": "__leaf_0__"}, OfType(Family.numeric()))
            >>> DeferredArgument("row").leaf_selected(Code("pl.col('a')"), [], leaf, [])
            []
            >>> DeferredArgument("lazy row").leaf_selected(Code("pl.col('a')"), [], leaf, [])
            []
        """
        renderings = self.renderings(columns)
        declared = self.declared(code)
        rule = selector.rule
        node = selector.node
        answers = self._evaluation(columns).leaf_selected(
            declared,
            renderings,
            node,
            rule,
            selectable,
        )
        stored = self._stored_evaluation(columns)
        if stored is None:
            return answers
        kept = stored.leaf_selected(declared, renderings, node, rule, selectable)
        return self._all_stored(answers, kept)

    def aliased(self, code: Code[MypyType]) -> bool:
        """Say whether the outermost call of a code is `.alias()`.

        Args:
            code: The code.

        Returns:
            Whether it is.

        Example:
            >>> DeferredArgument("row").aliased(Code("pl.col('a')"))
            False
        """
        return DeferredEvaluation(self._context).aliased(self.declared(code))

    def effect(
        self,
        code: Code[MypyType],
        columns: Sequence[ColumnRef],
        record: Record,
    ) -> RemainderEffect | str | None:
        """Give what the code of a selection does to the remainder of an open frame.

        Args:
            code: The code.
            columns: The declared columns of the frame.
            record: What the checker knows about the remainder.

        Returns:
            The effect, a message where polars refuses the code at a kind
            that the remainder can hold, or `None` where the checker cannot
            read the selector or the rename.

        Example:
            >>> from astrolabe._hooks.remainder.record import Record
            >>> argument = DeferredArgument("row")
            >>> effect = argument.effect(Code("cs.numeric()"), [], Record.unconstrained())
            >>> effect.retype is None
            True
        """
        remainder = DeferredRemainder(self._context)
        return remainder.effect(self.declared(code), self.renderings(columns), record)

    def remainder_cause(
        self,
        code: Code[MypyType],
        columns: Sequence[ColumnRef],
        record: Record,
    ) -> str:
        """Give why the code of a selection leaves a picked column of the remainder `Unchecked`.

        Args:
            code: The code.
            columns: The declared columns of the frame.
            record: What the checker knows about the remainder.

        Returns:
            The cause, or the empty text where no answer has one.

        Example:
            >>> from astrolabe._hooks.remainder.record import Record
            >>> argument = DeferredArgument("row")
            >>> argument.remainder_cause(Code("cs.numeric()"), [], Record.unconstrained())
            ''
        """
        remainder = DeferredRemainder(self._context)
        return remainder.cause(self.declared(code), self.renderings(columns), record)

    def bare_column(self, typ: MypyType) -> str | None:
        """Give the name of the column of a deferred argument that is `pl.col(name)` alone.

        Args:
            typ: The type of the argument.

        Returns:
            The name, or `None` where the type holds no code of that form.

        Example:
            >>> DeferredArgument("row").bare_column(new_type_fixture().anyt) is None
            True
        """
        codes = self.codes(typ)
        if codes is None or len(codes) != 1:
            return None
        return DeferredEvaluation(self._context).bare_column(self.declared(codes[0]))

    def reads_of(self, code: Code[MypyType]) -> tuple[str, ...]:
        """Give the columns of a frame that one code reads.

        Args:
            code: The code.

        Returns:
            The names, in the order of polars.

        Example:
            >>> DeferredArgument("row").reads_of(Code("pl.col('b') + pl.col('a')"))
            ('b', 'a')
        """
        return DeferredEvaluation(self._context).reads(self.declared(code))

    def declared(self, code: Code[MypyType]) -> Code[DeclaredLeaf | None]:
        """Give a code with the declared column of each leaf in the place of its type.

        The evaluation reads plain values and no type of mypy, so code that
        checks a deferred argument with it first turns each leaf of the code,
        the type of a declared parameter, into the column that it declares.

        Args:
            code: The code.

        Returns:
            The code.

        Example:
            >>> DeferredArgument("row").declared(Code("pl.col('a')")).text
            "pl.col('a')"
        """
        return Code(code.text, tuple(self._leaf(leaf) for leaf in code.leaves))

    def reads(self, typ: MypyType) -> list[str] | None:
        """Give the names of the columns of a frame that a deferred argument reads.

        A frame with no schema, such as a plain frame, cannot answer which
        columns it holds. Code that types an operation on such a frame reads
        these names to stand in for its columns.

        Args:
            typ: The type of the argument.

        Returns:
            The names, sorted, or `None` where the argument holds no code.

        Example:
            >>> fixture = new_type_fixture()
            >>> DeferredArgument("row").reads(fixture.anyt) is None
            True
        """
        codes = self.codes(typ)
        if codes is None:
            return None
        evaluation = DeferredEvaluation(self._context)
        names: set[str] = set()
        for code in codes:
            names.update(evaluation.reads(self.declared(code)))
        return sorted(names)

    def _leaf(self, typ: MypyType) -> DeclaredLeaf | None:
        """Give the name, the data type and the declaration of a declared column that a code reads.

        The declaration gives the column back at each data type that it
        allows, and at no other. So the evaluation refuses a frame where the
        column has another data type, as the frame refuses the declared
        column itself.

        Args:
            typ: The declared type of the leaf.

        Returns:
            The column, or `None` where the leaf is not a declared column
            whose name and data type the checker reads.

        Example:
            >>> fixture = new_type_fixture()
            >>> price = typed_column_instance(fixture, "price", int64_dtype(fixture))
            >>> leaf = DeferredArgument("row")._leaf(price)
            >>> leaf.name, leaf.rendering
            ('price', 'Int64')
            >>> leaf.outcome.at({"price": "Int64"}), leaf.outcome.at({"price": "Int8"})
            ('Int64', None)
            >>> DeferredArgument("row")._leaf(fixture.anyt) is None
            True

            A name that is a plain `str`, or a union of literals, keeps the data
            type under the name of a hole of text, which no code knows:

            >>> named = price.copy_modified(args=[fixture.str_type, price.args[1]])
            >>> DeferredArgument("row")._leaf(named).rendering
            'Int64'
        """
        proper = get_proper_type(typ)
        if not isinstance(proper, Instance):
            return None
        if not self.COLUMN.matches(proper):
            return self._map_leaf(proper)
        name = ProperReads().string_literal(proper.args[0])
        choices = self._choices(proper.args[0])
        if name is None and self._blank(proper.args[0]):
            name = ""
        if name is None and (self._text(proper.args[0]) or choices):
            name = HoleValues.STR
        if name is None:
            return None
        declared = ExpressionArgument().decoded(proper)
        outcome = None if declared is None else DeclaredOutcome(SliceDecl.from_expression(declared))
        rendering = self._rendering(proper.args[1])
        return DeclaredLeaf(name, rendering, outcome=outcome, choices=choices)

    def _map_leaf(self, typ: MypyType) -> DeclaredLeaf | None:
        """Give the leaf of a declared map, as `typed_agg(...)`, whose type its declaration gives.

        A map that aggregates and a literal give one value. Any other map
        can give one value for each row, or rows of another length, so the
        code reads it as rows of another length. A declared selection is a
        map at each column that it picks.

        Args:
            typ: The declared type.

        Returns:
            The leaf, or `None` where the type declares no map.

        Example:
            >>> DeferredArgument("row")._map_leaf(new_type_fixture().anyt) is None
            True
        """
        selection = SelectionReader().decoded(typ)
        if selection is not None:
            return DeclaredLeaf("", UNCHECKED, (), selection=DeclaredSelection(selection))
        declared = ExpressionArgument().decoded(typ)
        if declared is None:
            return None
        decl = SliceDecl.from_expression(declared)
        name = decl.output_name() or ("literal" if decl.character == "literal" else "")
        reads = tuple(read.name or "" for read in decl.reads)
        form = RowForm.OTHER_LENGTH if decl.character == "elementwise" else RowForm.ONE_VALUE
        return DeclaredLeaf(name, UNCHECKED, reads, form, DeclaredOutcome(decl))

    def _blank(self, typ: MypyType) -> bool:
        """Say whether the name of a declared column is a blank, which each call binds.

        Args:
            typ: The type of the name.

        Returns:
            Whether the type is `Blank`.

        Example:
            >>> DeferredArgument("row")._blank(new_type_fixture().anyt)
            False
        """
        instance = ProperReads().instance(typ)
        return instance is not None and self.BLANK.matches(instance)

    def _choices(self, typ: MypyType) -> tuple[str, ...]:
        """Give the names that a declared name can take, where it is a union of literals.

        Args:
            typ: The type of the name, as `Literal["a", "b"]`.

        Returns:
            The names, or no name where the type is not a union of string
            literals.

        Example:
            >>> from mypy.types import UnionType
            >>> fixture = new_type_fixture()
            >>> a, b = literal_str_type(fixture, "a"), literal_str_type(fixture, "b")
            >>> names = UnionType([a, b])
            >>> argument = DeferredArgument("row")
            >>> argument._choices(names), argument._choices(fixture.str_type)
            (('a', 'b'), ())
        """
        proper = get_proper_type(typ)
        if not isinstance(proper, UnionType):
            return ()
        names = [ProperReads().string_literal(item) for item in proper.items]
        return tuple(name for name in names if name is not None) if None not in names else ()

    def _text(self, typ: MypyType) -> bool:
        """Say whether the name of a declared column is a plain `str`, whose value no code knows.

        Args:
            typ: The type of the name.

        Returns:
            Whether the type is `str`.

        Example:
            >>> DeferredArgument("row")._text(new_type_fixture().str_type)
            True
        """
        instance = ProperReads().instance(typ)
        return instance is not None and self.TEXT.matches(instance)

    def renderings(self, columns: Sequence[ColumnRef]) -> dict[str, str]:
        """Give the rendering of the data type of each declared column, by its name.

        Args:
            columns: The declared columns of the frame, each with a name.

        Returns:
            The renderings.

        Example:
            >>> fixture = new_type_fixture()
            >>> DeferredArgument("row").renderings([column_ref(fixture, "a", int8_dtype(fixture))])
            {'a': 'Int8'}
        """
        renderings: dict[str, str] = {}
        for column in columns:
            renderings[checked_not_none(column.name)] = self._rendering(column.dtype)
        return renderings

    def _rendering(self, typ: MypyType) -> str:
        """Give the rendering of a declared data type, where a family of polars keeps its name.

        A family such as `NumericType` is not a data type. Its name stays in
        the rendering as a hole, which the evaluation reads as each data type
        of the family. Any other type that the checker does not read is
        `Unchecked`.

        Args:
            typ: The declared data type.

        Returns:
            The rendering.

        Example:
            >>> fixture = new_type_fixture()
            >>> DeferredArgument("row")._rendering(int8_dtype(fixture))
            'Int8'
            >>> DeferredArgument("row")._rendering(fixture.anyt)
            'Unchecked'
        """
        rendering = ConstDtypeReads().read_type(typ)
        if rendering is not None:
            return rendering
        instance = ProperReads().instance(typ)
        if instance is not None and instance.type.has_base(self.FAMILY.fullname):
            return instance.type.name
        return UNCHECKED


@dataclass(frozen=True)
class PickedColumns:
    """Hold the declared columns that a selector picks, and those that the data decide.

    `names` gives each picked column, in the order of the frame. `partial`
    gives each of them whose data type the frame does not fully declare, and
    that the selector picks for some of its data types only.

    Example:
        >>> PickedColumns(["a", "u"], ["u"]).partial
        ['u']
    """

    names: list[str]
    partial: list[str]


class DeferredFields:
    """Read a deferred argument that reads several fields of a struct, as `.struct.unnest()`.

    Polars expands such a read into one read of each field that it picks,
    as it expands a selector of columns. Code that types a frame operation
    reads the fields of the struct, and the faults of the read, through
    this class.

    Example:
        >>> DeferredFields("row")._argument._context
        'row'
    """

    def __init__(self, context: str) -> None:
        """Keep the reader of the deferred argument in the context of the frame operation.

        Args:
            context: The context of the frame operation, as `row`.
        """
        self._context = context
        self._argument = DeferredArgument(context)

    def fields(
        self,
        code: Code[MypyType],
        columns: Sequence[ColumnRef],
    ) -> list[DeferredAnswer] | None:
        """Give one answer for each field that a read of several fields of a struct picks.

        Args:
            code: The code.
            columns: The declared columns of the frame.

        Returns:
            The answers, `None` where the code reads no several fields, or no
            answer where the checker cannot pick the fields.

        Example:
            >>> DeferredFields("row").fields(Code("pl.col('s')"), []) is None
            True
        """
        evaluation = DeferredEvaluation(self._context)
        return evaluation.fields(self._argument.declared(code), self._argument.renderings(columns))

    def picks_no_field(self, code: Code[MypyType], columns: Sequence[ColumnRef]) -> bool:
        """Say whether a read of several fields of a struct of known fields picks none of them.

        Args:
            code: The code.
            columns: The declared columns of the frame.

        Returns:
            Whether it picks none.

        Example:
            >>> DeferredFields("row").picks_no_field(Code("pl.col('s')"), [])
            False
        """
        evaluation = DeferredEvaluation(self._context)
        return evaluation.picks_no_field(
            self._argument.declared(code), self._argument.renderings(columns)
        )

    def struct_column(self, code: Code[MypyType]) -> str | None:
        """Give the column of the struct that a read of several fields reads, where it is a column.

        Args:
            code: The code.

        Returns:
            The name, or `None`.

        Example:
            >>> DeferredFields("row").struct_column(Code("pl.col('s').struct.unnest()"))
            's'
        """
        return DeferredEvaluation(self._context).struct_column(self._argument.declared(code))

    def missing_fields(self, code: Code[MypyType], columns: Sequence[ColumnRef]) -> list[str]:
        """Give each field that a read of several fields requires and the struct does not hold.

        Args:
            code: The code.
            columns: The declared columns of the frame.

        Returns:
            The names, in the order of the code.

        Example:
            >>> DeferredFields("row").missing_fields(Code("pl.col('a')"), [])
            []
        """
        evaluation = DeferredEvaluation(self._context)
        return evaluation.missing_fields(
            self._argument.declared(code), self._argument.renderings(columns)
        )


class DeclaredSelection:
    """Give the leaf of a declared selection at each column that it picks.

    A declared selection, as `typed_selection(cs.numeric(), gives=same)`,
    gives one column for each column that its selector picks: the declared
    type at the data type of that column, under the name that its rename
    gives. A code that reads such a parameter is a selection. At one picked
    column, the parameter is a declared map that reads that column.

    Example:
        >>> from astrolabe._hooks.expression.selection import SelectionType
        >>> from astrolabe._hooks.remainder.selector import OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> numeric = SelectionType.identity(OfType(Family.numeric()))
        >>> leaf = DeclaredSelection(numeric).picked("a")
        >>> leaf.name, leaf.reads, leaf.outcome.at({"a": "Int8"})
        ('a', ('a',), 'Int8')
    """

    def __init__(self, selection: SelectionType) -> None:
        """Keep the declaration.

        Args:
            selection: The declared selection.
        """
        self._selection = selection

    @property
    def rule(self) -> Selector:
        """Give the rule of the columns that the selection picks.

        Returns:
            The rule.

        Example:
            >>> from astrolabe._hooks.expression.selection import SelectionType
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> DeclaredSelection(SelectionType.identity(OfType(Family.numeric()))).rule
            OfType(family=NumericType)
        """
        return self._selection.selector

    def picked(self, column: str) -> DeclaredLeaf:
        """Give the declared map that the selection is at one picked column.

        An elementwise selection can give rows of another length, as any
        declared map that works row by row, and an aggregating one gives
        one value.

        Args:
            column: The name of the picked column.

        Returns:
            The leaf.

        Example:
            >>> from astrolabe._hooks.expression.selection import SelectionType
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = SelectionType.identity(OfType(Family.numeric()))
            >>> DeclaredSelection(numeric).picked("a").form
            <RowForm.OTHER_LENGTH: 'other length'>
        """
        elementwise = self._selection.character.label == "elementwise"
        form = RowForm.OTHER_LENGTH if elementwise else RowForm.ONE_VALUE
        name = self._selection.affix.applied(column)
        outcome = PickedOutcome(self._selection, column)
        return DeclaredLeaf(name, UNCHECKED, (column,), form, outcome)


class PickedOutcome:
    """Give the type that a declared selection promises at one picked column of a frame.

    Example:
        >>> from astrolabe._hooks.expression.selection import SelectionType
        >>> from astrolabe._hooks.remainder.selector import OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> numeric = SelectionType.identity(OfType(Family.numeric()))
        >>> PickedOutcome(numeric, "a").at({"a": "Float32"})
        'Float32'
    """

    def __init__(self, selection: SelectionType, column: str) -> None:
        """Keep the declaration and the picked column.

        Args:
            selection: The declared selection.
            column: The name of the picked column.
        """
        self._selection = selection
        self._column = column

    def at(self, columns: Mapping[str, str]) -> str | None:
        """Give the rendering of the type that the selection gives at the picked column.

        Args:
            columns: The rendering of the data type of each column of the frame, by its name.

        Returns:
            The rendering, or `None` where the frame has no such column, or
            where the declaration gives no type at its data type.

        Example:
            >>> from astrolabe._hooks.expression.selection import SelectionType
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = SelectionType.identity(OfType(Family.numeric()))
            >>> PickedOutcome(numeric, "a").at({"a": "String"}) is None
            True
            >>> PickedOutcome(numeric, "a").at({}) is None
            True
        """
        rendering = columns.get(self._column)
        if rendering is None:  # pragma: no cover -- the selection picked this column of the frame
            return None
        at = {SelectionType.AXIS: rendering}
        return Materializer(ExpressionEvaluation.UNIVERSE).read(at, self._selection.table)


class DeclaredOutcome:
    """Give the type that a declared map promises at the data types of the columns of a frame.

    The declaration splits the kinds of its reads into rows, and each row
    promises an outcome. The type at a frame is the outcome of the row that
    holds the kind of each read there, with the full data type of a read
    where the outcome is that read.

    Example:
        >>> from astrolabe._hooks.expression.fit import Read, Row
        >>> from astrolabe._hooks.expression.outcome import Axis
        >>> decl = SliceDecl(
        ...     (Read("a", frozenset({"Datetime"})),),
        ...     (Row((frozenset({"Datetime"}),), Axis("a")),),
        ...     None, "elementwise",
        ... )
        >>> DeclaredOutcome(decl).at({"a": "Datetime(time_unit='ms', time_zone=None)"})
        "Datetime(time_unit='ms', time_zone=None)"
        >>> DeclaredOutcome(decl).at({"a": "Int8"}) is None
        True
    """

    def __init__(self, decl: SliceDecl) -> None:
        """Keep the declaration.

        Args:
            decl: The declaration.
        """
        self._decl = decl

    def at(self, columns: Mapping[str, str]) -> str | None:
        """Give the rendering of the type that the declaration promises at a frame.

        Args:
            columns: The rendering of the data type of each column of the frame, by its name.

        Returns:
            The rendering, or `None` where the frame is outside the declaration.
        """
        for row in self._decl.rows:
            rendered = self._rendered(row, columns)
            if rendered is not None:
                return row.outcome.evaluated(rendered)
        return None

    def _rendered(self, row: Row, columns: Mapping[str, str]) -> dict[str, Rendered] | None:
        """Give the data type of each read at a frame, where each one is in the region of a row.

        Args:
            row: The row.
            columns: The rendering of the data type of each column of the frame.

        Returns:
            The data types, by the name of the read, or `None` where a read
            is missing or outside the row.

        Example:
            >>> from astrolabe._hooks.expression.fit import Read, Row
            >>> from astrolabe._hooks.expression.outcome import Axis
            >>> decl = SliceDecl(
            ...     (Read("a", frozenset({"Int8"})),),
            ...     (Row((frozenset({"Int8"}),), Axis("a")),),
            ...     None, "elementwise",
            ... )
            >>> DeclaredOutcome(decl)._rendered(decl.rows[0], {"a": "Int8"})
            {'a': Rendered(kind='Int8', text='Int8')}
        """
        rendered: dict[str, Rendered] = {}
        for read, region in zip(self._decl.reads, row.regions, strict=True):
            name = read.name or ""
            text = columns.get(name)
            if text is None:
                return None
            kind = text.partition("(")[0]
            if not read.unchecked and kind not in region:
                return None
            rendered[name] = Rendered(kind, text)
        return rendered
