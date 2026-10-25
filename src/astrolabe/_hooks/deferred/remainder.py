"""Give what the code of a deferred selection does to the remainder of an open frame.

An open frame has a remainder: the columns that its type does not name. A
selection such as `cs.numeric().abs()` can pick columns there too, so the
result of `select` keeps a remainder that the selection changed. The checker
cannot list those columns. It knows the kinds that they can have, that is,
their data types with the parameters taken away. So it evaluates the code at
a column of each kind that the selector can pick, and keeps three facts: the
rule of the selector, the one data type that the picked columns take, if
any, and the rename that the code applies to their names.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.deferred.code_reader import CodeReader, CodeReadError
from astrolabe._hooks.deferred.evaluation import DeferredEvaluation, Substitution
from astrolabe._hooks.deferred.open_types import Generalization, OpenType
from astrolabe._hooks.deferred.polars_selector import PolarsSelector
from astrolabe._hooks.expression.lifting import UNCHECKED
from astrolabe._hooks.expression.selection import Affix, RemainderEffect
from astrolabe._rules.dtypes import Rendering

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from astrolabe._hooks.deferred.code import Code
    from astrolabe._hooks.deferred.evaluation import DeclaredLeaf
    from astrolabe._hooks.remainder.record import Record
    from astrolabe._hooks.remainder.selector import Selector
    from astrolabe._hooks.remainder.type_set import TypeSet


class DeferredRemainder:
    """Give the effect of the code of a deferred selection on the remainder of an open frame.

    The code stands for each picked column in turn. So the evaluation puts a
    probe column, a column with a reserved name, in the place of the
    selector, and gives the probe each data type of each kind in turn. A kind
    where every data type gives itself back keeps its type. A kind where
    polars refuses every data type is an error, because the remainder can hold
    a column of that kind.

    Example:
        >>> from astrolabe._hooks.deferred.code import Code
        >>> from astrolabe._hooks.remainder.record import Record
        >>> code = Code("cs.numeric().abs()")
        >>> effect = DeferredRemainder("row").effect(code, {}, Record.unconstrained())
        >>> effect.selector, effect.retype
        (OfType(family=NumericType), None)
    """

    PROBES: ClassVar[tuple[str, str]] = ("__remainder_a__", "__Remainder_B__")
    REFUSED: ClassVar[str] = "the selection has no result type at a type the remainder can hold"
    RENAMED: ClassVar[str] = "the rename of the picked columns is not a prefix or a suffix"
    SPLIT: ClassVar[str] = "polars refuses some data types of a kind that the remainder can hold"

    def __init__(self, context: str) -> None:
        """Keep the context of the frame operation.

        Args:
            context: `row` for a context such as `select`, or `agg`.

        Example:
            >>> DeferredRemainder("agg")._context
            'agg'
        """
        self._context = context

    def effect(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        record: Record,
    ) -> RemainderEffect | str | None:
        """Give what the code of a selection does to the remainder.

        A selector alone gives each picked column back in a `select`, and a
        list of its values for each group in `agg`, at every kind.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            columns: The rendering of the data type of each declared column.
            record: What the checker knows about the remainder: the names
                that it cannot hold and the kinds that its columns can have.

        Returns:
            The effect. `REFUSED` where polars refuses the code at a kind
            that the remainder can hold, and `RENAMED` where the code renames
            the picked columns with a rewrite of the whole name, which can
            give two columns of the remainder one name. `None` where the
            checker cannot read the selector or the names.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.remainder.record import Record
            >>> code = Code("cs.all().abs()")
            >>> DeferredRemainder("row").effect(code, {}, Record.unconstrained())
            'the selection has no result type at a type the remainder can hold'
            >>> upper = Code("cs.numeric().name.to_uppercase()")
            >>> DeferredRemainder("row").effect(upper, {}, Record.unconstrained())
            'the rename of the picked columns is not a prefix or a suffix'
            >>> DeferredRemainder("row").effect(Code("pl.col('a')"), {}, Record.unconstrained())
            >>> first = Code("cs.by_index(0).abs()")
            >>> DeferredRemainder("row").effect(first, {}, Record.unconstrained()) is None
            True
        """
        node = self._node(code)
        expression = self._expression(code)
        if node is None or expression is None:
            return None
        selector = self._rule(code, node)
        if selector is None:
            return None
        affix = self.affix(code, node)
        if affix is None:
            return self.RENAMED if self._named(code, node) else None
        if self._bare(expression, node):
            listed = f"List({UNCHECKED})" if "agg" in self._context.split() else None
            return RemainderEffect(selector, listed, affix)
        kinds = sorted(selector.surviving_types(record.types).kinds())
        answers = {kind: self.kind(code, columns, node, kind, record.types) for kind in kinds}
        return self._answered(selector, affix, answers)

    def _node(self, code: Code[DeclaredLeaf | None]) -> object | None:
        """Give the selector of a code: a selector of polars, or the leaf of a declared selection.

        Args:
            code: The code.

        Returns:
            The serialized selector, the leaf as a column of the code, or
            `None` where the code holds neither.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredRemainder("row")._node(Code("pl.col('a')")) is None
            True
        """
        node = DeferredEvaluation(self._context).selector(code)
        if node is not None:
            return node
        places = [place for place, leaf in enumerate(code.leaves) if leaf and leaf.selection]
        if len(places) != 1:
            return None
        return {"Column": DeferredEvaluation.LEAF.format(places[0])}

    def _rule(self, code: Code[DeclaredLeaf | None], node: object) -> Selector | None:
        """Give the rule of the columns that the selector of a code picks.

        Args:
            code: The code.
            node: The selector of the code.

        Returns:
            The rule, or `None` where the checker does not read the selector.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> numeric = {"Selector": {"ByDType": "Numeric"}}
            >>> DeferredRemainder("row")._rule(Code("cs.numeric()"), numeric)
            OfType(family=NumericType)
            >>> from astrolabe._hooks.deferred.evaluation import DeclaredLeaf
            >>> price = Code("$0", (DeclaredLeaf("price", "Float64"),))
            >>> DeferredRemainder("row")._rule(price, numeric)
            OfType(family=NumericType)
        """
        name = node.get("Column") if isinstance(node, dict) else None
        for place, leaf in enumerate(code.leaves):
            if leaf and leaf.selection and name == DeferredEvaluation.LEAF.format(place):
                return leaf.selection.rule
        return PolarsSelector().of(self._body(node))

    def _answered(
        self,
        selector: Selector,
        affix: Affix,
        answers: Mapping[str, str | None],
    ) -> RemainderEffect | str:
        """Give the effect of a selection from what it does to the columns of each kind.

        The effect gives the answer of most kinds once, and the answer of each
        other kind beside it. That answer is the empty text, where the columns
        keep their own data type, unless every kind takes a data type.

        Args:
            selector: The rule of the columns that the selection picks.
            affix: The rename of the selection.
            answers: What the selection does to the columns of each kind: the
                empty text where they keep their own data type, and `None`
                where polars refuses every data type of the kind.

        Returns:
            The effect, or a message where polars refuses a kind, or some data
            types of a kind (`SPLIT`).

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = OfType(Family.numeric())
            >>> DeferredRemainder("row")._answered(numeric, Affix(), {"Int8": "Int64"}).retype
            'Int64'
            >>> mixed = {"Int8": "Int64", "Float32": ""}
            >>> DeferredRemainder("row")._answered(numeric, Affix(), mixed).retypes
            (('Int64', frozenset({'Int8'})),)
            >>> DeferredRemainder("row")._answered(numeric, Affix(), {"Int8": None})
            'the selection has no result type at a type the remainder can hold'
        """
        if self.SPLIT in answers.values():
            return self.SPLIT
        grouped = self._grouped(answers)
        if grouped is None:
            return self.REFUSED
        default = self._default(grouped)
        others = tuple(sorted(item for item in grouped.items() if item[0] != default))
        return RemainderEffect(selector, default or None, affix, others)

    def _grouped(self, answers: Mapping[str, str | None]) -> dict[str, frozenset[str]] | None:
        """Group the kinds by what a selection does to their columns.

        Where every kind whose columns keep their own data type has one data
        type, and other kinds take that data type, as `UInt32` under
        `n_unique`, the kind joins their answer. Every kind then takes a data
        type, which the effect can give once.

        Args:
            answers: What the selection does to the columns of each kind.

        Returns:
            The kinds of each answer, the empty text for the kinds that keep
            their data type, or `None` where polars refuses a kind.

        Example:
            >>> answers = {"Int8": "UInt32", "UInt32": ""}
            >>> grouped = DeferredRemainder("row")._grouped(answers)
            >>> {answer: sorted(kinds) for answer, kinds in grouped.items()}
            {'UInt32': ['Int8', 'UInt32']}
            >>> mixed = DeferredRemainder("row")._grouped({**answers, "String": ""})
            >>> {answer: sorted(kinds) for answer, kinds in mixed.items()}
            {'UInt32': ['Int8'], '': ['String', 'UInt32']}
            >>> DeferredRemainder("row")._grouped({"Int8": None}) is None
            True
        """
        if None in answers.values():
            return None
        texts = set(answers.values())
        kept = [kind for kind, answer in answers.items() if not answer]
        joins = all(kind in texts and self._single(kind) for kind in kept)
        grouped: dict[str, frozenset[str]] = {}
        for kind, answer in answers.items():
            joined = kind if joins and not answer else checked_not_none(answer)
            grouped[joined] = grouped.get(joined, frozenset()) | {kind}
        return grouped

    def _single(self, kind: str) -> bool:
        """Say whether a kind has one data type, as `UInt32` has and `Datetime` has not.

        Args:
            kind: The kind.

        Returns:
            Whether each sample of the kind is the kind itself.

        Example:
            >>> DeferredRemainder("row")._single("UInt32"), DeferredRemainder("row")._single("List")
            (True, False)
        """
        samples = OpenType().samples(kind)
        return bool(samples) and all(sample.render() == kind for sample in samples)

    def _default(self, grouped: Mapping[str, frozenset[str]]) -> str:
        """Give the answer that the effect gives once: the answer of the most kinds.

        A kind that keeps its own data type cannot be an exception to another
        answer, so where one does, the default is to keep.

        Args:
            grouped: The kinds of each answer.

        Returns:
            The answer, or the empty text to keep.

        Example:
            >>> DeferredRemainder("row")._default({"Int64": frozenset({"Int8", "Int16"})})
            'Int64'
            >>> kept = {"Int64": frozenset({"Int8"}), "": frozenset({"String"})}
            >>> DeferredRemainder("row")._default(kept)
            ''
        """
        if "" in grouped or not grouped:
            return ""
        return max(sorted(grouped), key=lambda answer: len(grouped[answer]))

    def cause(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        record: Record,
    ) -> str:
        """Give why the code of a selection leaves a picked column of the remainder `Unchecked`.

        The rules give `Unchecked` for a reason that the checker reports,
        for example a value of the type `Any` in the code, or for a reason
        that it does not, for example answers that differ across the data
        types of a kind. The method gives the cause of the first answer
        `Unchecked` at the data types of the kinds that the selector picks.

        Args:
            code: The code, with a declared column or `None` for each leaf.
            columns: The rendering of the data type of each declared column.
            record: What the checker knows about the remainder.

        Returns:
            The cause, or the empty text where no answer has one.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.remainder.record import Record
            >>> code = Code("cs.float().fill_nan(hole('any'))")
            >>> DeferredRemainder("row").cause(code, {}, Record.unconstrained())
            'untyped'
            >>> DeferredRemainder("row").cause(Code("pl.col('a')"), {}, Record.unconstrained())
            ''
        """
        evaluation = DeferredEvaluation(self._context)
        node = evaluation.selector(code)
        selector = None if node is None else PolarsSelector().of(self._body(node))
        if selector is None:
            return ""
        probe = self.PROBES[0]
        for kind in sorted(selector.surviving_types(record.types).kinds()):
            for sample in self._held_samples(kind, record.types, evaluation.fields_of(code)):
                renderings = {**columns, probe: sample}
                answer = evaluation.at_column(code, renderings, node, probe)
                if answer.rendering == UNCHECKED and answer.cause:
                    return answer.cause
        return ""

    def rule(self, code: Code[DeclaredLeaf | None]) -> Selector | None:
        """Give the rule of a code that is a selector alone, with no operation on it.

        Code that reads a selector as a set of columns, and not as an
        expression, such as the columns that `map_columns` maps, needs only
        this rule.

        Args:
            code: The code.

        Returns:
            The rule, or `None` where the code is not a selector alone, or
            where the checker cannot read the selector.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredRemainder("row").rule(Code("cs.numeric() - cs.integer()"))
            SelAnd(left=OfType(family=NumericType), right=SelNot(inner=OfType(family=IntegerType)))
            >>> DeferredRemainder("row").rule(Code("cs.numeric().abs()")) is None
            True
        """
        node = DeferredEvaluation(self._context).selector(code)
        expression = self._expression(code)
        if node is None or expression is None or not self._bare(expression, node):
            return None
        return PolarsSelector().of(self._body(node))

    def _expression(self, code: Code[DeclaredLeaf | None]) -> pl.Expr | None:
        """Build the expression of a code again.

        Args:
            code: The code.

        Returns:
            The expression, or `None` where the code does not give one.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> DeferredRemainder("row")._expression(Code("pl.col(")) is None
            True
        """
        try:
            expression = CodeReader().read(code).expression
        except CodeReadError:
            return None
        return expression if isinstance(expression, pl.Expr) else None

    def _body(self, node: object) -> object:
        """Give the body of a serialized selector node.

        Args:
            node: The node, a map of `Selector` to its body.

        Returns:
            The body.

        Example:
            >>> DeferredRemainder("row")._body({"Selector": "Wildcard"})
            'Wildcard'
        """
        return node.get("Selector") if isinstance(node, dict) else None

    def affix(self, code: Code[DeclaredLeaf | None], node: object) -> Affix | None:
        """Give the rename that the code applies to the name of each picked column.

        The output names of two probes of other names show the rename. A
        rename that is not a prefix, a suffix or a fixed name, such as
        `.name.to_uppercase()`, gives two names that no such rename explains.

        Args:
            code: The code.
            node: The serialized selector.

        Returns:
            The rename, or `None` where it is not a prefix, a suffix or a
            fixed name.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> selector = {"Selector": {"ByDType": "Numeric"}}
            >>> remainder = DeferredRemainder("row")
            >>> remainder.affix(Code("cs.numeric().name.suffix('_x')"), selector)
            Affix(prefix='', suffix='_x', fixed=None)
            >>> remainder.affix(Code("cs.numeric().name.to_uppercase()"), selector) is None
            True
            >>> remainder.affix(Code("pl.col("), selector) is None
            True
        """
        evaluation = DeferredEvaluation(self._context)
        first = evaluation.name_at(code, node, self.PROBES[0])
        second = evaluation.name_at(code, node, self.PROBES[1])
        if first is None or second is None:
            return None
        probe = self.PROBES[0]
        if first == probe:
            affix = Affix()
        elif probe in first:
            place = first.index(probe)
            affix = Affix(first[:place], first[place + len(probe) :])
        else:
            affix = Affix(fixed=first)
        return affix if affix.applied(self.PROBES[1]) == second else None

    def _named(self, code: Code[DeclaredLeaf | None], node: object) -> bool:
        """Say whether the checker knows the output name of a picked column, whatever its name.

        A rewrite of the whole name, such as `.name.to_uppercase()`, gives
        names that the checker computes. A function of the user, such as
        `.name.map(f)`, gives names that nobody knows.

        Args:
            code: The code.
            node: The serialized selector.

        Returns:
            Whether the output name of a probe column is known.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> selector = {"Selector": {"ByDType": "Numeric"}}
            >>> DeferredRemainder("row")._named(Code("cs.numeric().name.to_uppercase()"), selector)
            True
        """
        return DeferredEvaluation(self._context).name_at(code, node, self.PROBES[0]) is not None

    def renamer(self, text: str) -> str:
        """Give the method of the `name` namespace that renames the columns that a code picks.

        Args:
            text: The text of the code.

        Returns:
            The last call of the namespace in the code, as `name.<method>`,
            or `name` where the code holds none.

        Example:
            >>> DeferredRemainder("row").renamer("cs.numeric().name.to_uppercase().abs()")
            'name.to_uppercase'
        """
        members = re.findall(r"\.name\.(\w+)\(", text)
        return f"name.{members[-1]}" if members else "name"

    def kind(
        self,
        code: Code[DeclaredLeaf | None],
        columns: Mapping[str, str],
        node: object,
        kind: str,
        held: TypeSet | None = None,
    ) -> str | None:
        """Give what the code does to a picked column of one kind.

        Each caller brings its own columns of the kind, so the answer holds
        for every data type of the kind that the remainder can hold. Where
        polars refuses some of them and runs at others, the result depends
        on the caller.

        Args:
            code: The code.
            columns: The rendering of the data type of each declared column.
            node: The serialized selector.
            kind: The kind.
            held: The data types that the remainder can hold, or `None` for
                every data type. The evaluation tries the data types of the
                sample of the kind that it holds, or the whole sample where
                it holds none of them.

        Returns:
            The empty text where each data type of the kind gives itself
            back, the data type that every picked column takes, `Unchecked`
            where the answers differ or the rules do not know one, `SPLIT`
            where polars refuses some data types and not others, or `None`
            where polars refuses every data type of the kind.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> numeric = {"Selector": {"ByDType": "Numeric"}}
            >>> remainder = DeferredRemainder("row")
            >>> remainder.kind(Code("cs.numeric().abs()"), {}, numeric, "Int8")
            ''
            >>> remainder.kind(Code("cs.numeric().cast(pl.String)"), {}, numeric, "Datetime")
            'String'
            >>> remainder.kind(Code("cs.numeric().str.len_bytes()"), {}, numeric, "Int8") is None
            True
            >>> remainder.kind(Code("cs.numeric().abs()"), {}, numeric, "Extension")
            'Unchecked'
            >>> stamps = DeferredEvaluation("row").selector(Code("cs.datetime()"))
            >>> later = Code("cs.datetime() > datetime(2020, 1, 1, tzinfo=UTC)")
            >>> remainder.kind(later, {}, stamps, "Datetime") == DeferredRemainder.SPLIT
            True
        """
        probe = self.PROBES[0]
        samples = self._held_samples(kind, held, DeferredEvaluation(self._context).fields_of(code))
        if not samples:
            return UNCHECKED
        evaluation = DeferredEvaluation(self._context)
        pairs: list[tuple[str, str]] = []
        for sample in samples:
            answer = evaluation.at_column(code, {**columns, probe: sample}, node, probe)
            if not answer.refused:
                pairs.append((sample, answer.rendering))
        if not pairs:
            return None
        if len(pairs) != len(samples):
            return self.SPLIT
        if all(own == rendering for own, rendering in pairs):
            return ""
        return self._shared([rendering for _, rendering in pairs])

    def _held_samples(
        self,
        kind: str,
        held: TypeSet | None,
        fields: Sequence[str] = (),
    ) -> list[str]:
        """Give the data types of the sample of a kind that the remainder can hold.

        A struct of the sample holds a field that no code names. Where the
        code reads fields by name, the sample also holds a struct of exactly
        those fields, which a caller can bring.

        Args:
            kind: The kind.
            held: The data types that the remainder can hold, or `None` for every data type.
            fields: The fields that the code reads by name.

        Returns:
            The renderings of the held data types of the sample, or of the
            whole sample where the remainder holds none of them.

        Example:
            >>> from astrolabe._hooks.remainder.type_set import TypeSet
            >>> zoned = TypeSet(frozenset({"Datetime(time_unit='us', time_zone='UTC')"}))
            >>> DeferredRemainder("row")._held_samples("Datetime", zoned)
            ["Datetime(time_unit='us', time_zone='UTC')"]
            >>> len(DeferredRemainder("row")._held_samples("Datetime", None))
            3
            >>> numbers = TypeSet(frozenset({"Int8"}))
            >>> len(DeferredRemainder("row")._held_samples("Datetime", numbers))
            3
        """
        samples = [sample.render() for sample in OpenType(tuple(fields)).samples(kind)]
        kept = [sample for sample in samples if held is None or held.holds(sample)]
        return kept or samples

    def _bare(self, expression: pl.Expr, node: object) -> bool:
        """Say whether a code is the selector alone, with no operation on it.

        A selector alone gives each picked column back in a `select`, at
        every kind, also at a kind that the rules have no data type for.

        Args:
            expression: The expression of the code.
            node: The serialized selector.

        Returns:
            Whether the code is the selector alone.

        Example:
            >>> import polars.selectors as cs
            >>> selector = {"Selector": {"ByDType": "Numeric"}}
            >>> DeferredRemainder("row")._bare(cs.numeric(), selector)
            True
            >>> DeferredRemainder("row")._bare(cs.numeric().abs(), selector)
            False
        """
        probe = self.PROBES[0]
        probed = Substitution(node, probe)(expression)
        return bool(json.loads(probed.meta.serialize(format="json")) == {"Column": probe})

    def _shared(self, renderings: Sequence[str]) -> str:
        """Give what the answers at the data types of a kind share.

        Args:
            renderings: The answers.

        Returns:
            The shared data type, or `Unchecked`. Answers that differ, where
            one holds `Unchecked`, share only `Unchecked`.

        Example:
            >>> DeferredRemainder("row")._shared(["List(Int8)", "List(Date)"])
            'List(Unchecked)'
            >>> DeferredRemainder("row")._shared(["Int8", "Unchecked"])
            'Unchecked'
            >>> DeferredRemainder("row")._shared(["List(Unchecked)", "List(Unchecked)"])
            'List(Unchecked)'
            >>> DeferredRemainder("row")._shared(["List(Unchecked)", "List(Int8)"])
            'Unchecked'
        """
        distinct = set(renderings)
        if len(distinct) == 1:
            return renderings[0]
        if any(UNCHECKED in text for text in distinct):
            return UNCHECKED
        return Generalization().of([Rendering().parse(text) for text in renderings]).render()
