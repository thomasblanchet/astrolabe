"""Check that the code of a deferred argument fits the selection that its parameter declares.

A user can declare a parameter as a selection, for example with
`typed_selection(cs.numeric(), gives=same)`, and pass `cs.numeric().abs()`
to it. In the deferred model, the type of the argument holds its code, so
the check reads the selector of the code and evaluates the code at a probe
column of each kind that the selector can pick. The argument fits where four
things hold, as for a selection that the checker builds. Its selector picks
no column that the declared selector rejects. Its rename is the declared
rename. It counts rows in the declared way. And at each kind that it can
pick, it gives the declared answer.

The selector of a code is a selector of polars, as `cs.numeric()`, or a
parameter that the user declares as a selection, as in the body
`return sel.round(2)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.deferred.evaluation import DeferredEvaluation
from astrolabe._hooks.deferred.open_types import OpenType
from astrolabe._hooks.deferred.remainder import DeferredRemainder
from astrolabe._hooks.expression.character import CharacterWording
from astrolabe._hooks.expression.fit import CellRefusal, Fit
from astrolabe._hooks.expression.lifting import UNCHECKED
from astrolabe._hooks.expression.rendering import Rendered
from astrolabe._hooks.expression.selection import SelectionType
from astrolabe._hooks.expression.selection_fit import AffixWording
from astrolabe._hooks.remainder.selector import SelectorEntailment

if TYPE_CHECKING:
    from astrolabe._hooks.deferred.code import Code
    from astrolabe._hooks.deferred.evaluation import DeclaredLeaf
    from astrolabe._hooks.remainder.selector import Selector


@dataclass(frozen=True)
class CodeSelector:
    """Hold the selector of a code: the node that a probe column takes the place of, and its rule.

    Example:
        >>> from astrolabe._hooks.remainder.selector import OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> CodeSelector({"Column": "__leaf_0__"}, OfType(Family.numeric())).node
        {'Column': '__leaf_0__'}
    """

    node: object
    rule: Selector


class DeferredSelectionFit:
    """Give the first reason that the code of a deferred argument does not fit a declared selection.

    Example:
        >>> from astrolabe._hooks.deferred.code import Code
        >>> from astrolabe._hooks.remainder.selector import OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> numeric = OfType(Family.numeric())
        >>> selector = CodeSelector({"Selector": {"ByDType": "Numeric"}}, numeric)
        >>> decl = SelectionType.identity(numeric)
        >>> DeferredSelectionFit().verdict(Code("cs.numeric().abs()"), selector, decl) is None
        True
        >>> DeferredSelectionFit().verdict(Code("cs.numeric().sum()"), selector, decl)
        'it combines rows into one value, but the declared type works row by row'
    """

    NOT_SELECTION: ClassVar[str] = "this value is not a selection of columns"
    PROBE: ClassVar[str] = DeferredRemainder.PROBES[0]

    def verdict(
        self,
        code: Code[DeclaredLeaf | None],
        selector: CodeSelector,
        decl: SelectionType,
    ) -> str | None:
        """Check the code of one argument against one declared selection.

        Args:
            code: The code of the argument.
            selector: The selector of the code.
            decl: The declared selection.

        Returns:
            None where the argument fits. Otherwise the first reason that it
            does not, as the end of a sentence about the argument.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = OfType(Family.numeric())
            >>> selector = CodeSelector({"Selector": {"ByDType": "Numeric"}}, numeric)
            >>> decl = SelectionType.identity(numeric)
            >>> renamed = Code("cs.numeric().name.suffix('_r')")
            >>> DeferredSelectionFit().verdict(renamed, selector, decl)
            'it adds the suffix "_r", but the declared type adds nothing to the column names'
        """
        checks = (self._selector, self._affix, self._character, self._cells)
        for check in checks:
            refusal = check(code, selector, decl)
            if refusal is not None:
                return refusal
        return None

    def _selector(
        self,
        code: Code[DeclaredLeaf | None],
        selector: CodeSelector,
        decl: SelectionType,
    ) -> str | None:
        """Check that the argument picks no column that the declaration rejects.

        Args:
            code: The code of the argument.
            selector: The selector of the code.
            decl: The declared selection.

        Returns:
            The reason, or None.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.remainder.absence import Always
            >>> from astrolabe._hooks.remainder.selector import NameTest, OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> everything = CodeSelector({"Selector": "Wildcard"}, NameTest(Always()))
            >>> decl = SelectionType.identity(OfType(Family.numeric()))
            >>> DeferredSelectionFit()._selector(Code("pl.all()"), everything, decl)
            'it selects columns the declared type does not allow'
        """
        del code
        if SelectorEntailment()(selector.rule, decl.selector):
            return None
        return "it selects columns the declared type does not allow"

    def _affix(
        self,
        code: Code[DeclaredLeaf | None],
        selector: CodeSelector,
        decl: SelectionType,
    ) -> str | None:
        """Check that the argument renames the columns as the declaration does.

        Args:
            code: The code of the argument.
            selector: The selector of the code.
            decl: The declared selection.

        Returns:
            The reason, or None.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = OfType(Family.numeric())
            >>> selector = CodeSelector({"Selector": {"ByDType": "Numeric"}}, numeric)
            >>> upper = Code("cs.numeric().name.to_uppercase()")
            >>> decl = SelectionType.identity(numeric)
            >>> reason = DeferredSelectionFit()._affix(upper, selector, decl)
            >>> reason.startswith("it renames the columns in another way")
            True
        """
        affix = DeferredRemainder("row").affix(code, selector.node)
        if affix == decl.affix:
            return None
        wording = AffixWording()
        made = "renames the columns in another way" if affix is None else wording.phrase(affix)
        return f"it {made}, but the declared type {wording.phrase(decl.affix)}"

    def _character(
        self,
        code: Code[DeclaredLeaf | None],
        selector: CodeSelector,
        decl: SelectionType,
    ) -> str | None:
        """Check that the argument counts rows as the declaration does.

        Args:
            code: The code of the argument.
            selector: The selector of the code.
            decl: The declared selection.

        Returns:
            The reason, or None.
        """
        character = self._counted(code, selector, decl)
        if character is None or character == decl.character.label:
            return None
        wording = CharacterWording()
        made, declared = wording.phrase(character), wording.phrase(decl.character.label)
        return f"it {made}, but the declared type {declared}"

    def _counted(
        self,
        code: Code[DeclaredLeaf | None],
        selector: CodeSelector,
        decl: SelectionType,
    ) -> str | None:
        """Give the way that the argument counts rows, at the first declared kind that it picks.

        Args:
            code: The code of the argument.
            selector: The selector of the code.
            decl: The declared selection.

        Returns:
            `elementwise` or `aggregating`, or None where the answers in a
            `select` and in `agg` do not show it.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = OfType(Family.numeric())
            >>> selector = CodeSelector({"Selector": {"ByDType": "Numeric"}}, numeric)
            >>> decl = SelectionType.identity(numeric)
            >>> DeferredSelectionFit()._counted(Code("cs.numeric().max()"), selector, decl)
            'aggregating'

            Polars refuses this code at the first numeric kind, so its answers
            show nothing:

            >>> refused = Code("cs.numeric().mean().str.len_chars()")
            >>> DeferredSelectionFit()._counted(refused, selector, decl) is None
            True

            A declaration that holds no kind that the selector picks gives no kind to try:

            >>> temporal = OfType(Family.temporal())
            >>> dated = CodeSelector({"Selector": {"ByDType": "Temporal"}}, temporal)
            >>> DeferredSelectionFit()._counted(Code("cs.temporal().max()"), dated, decl) is None
            True
        """
        matched = SelectionType.matched_kinds(selector.rule)
        kinds = [kind for (kind,), _ in decl.table.cells if kind in matched]
        samples = OpenType().samples(kinds[0]) if kinds else ()
        if not samples:
            return None
        columns = {self.PROBE: samples[0].render()}
        row = DeferredEvaluation("row").at_column(code, columns, selector.node, self.PROBE)
        grouped = DeferredEvaluation("agg").at_column(code, columns, selector.node, self.PROBE)
        if grouped.rendering == f"List({row.rendering})":
            return "elementwise"
        if grouped.rendering == row.rendering and UNCHECKED not in row.rendering:
            return "aggregating"
        return None

    def _cells(
        self,
        code: Code[DeclaredLeaf | None],
        selector: CodeSelector,
        decl: SelectionType,
    ) -> str | None:
        """Check that the argument gives the declared answer at each data type that it can pick.

        A declared kind that the selector of the argument never picks does
        not count, because no picked column can have that kind. A kind such
        as `Datetime` stands for each of its units and zones, and the
        argument must give the declared answer at each of them.

        Args:
            code: The code of the argument.
            selector: The selector of the code.
            decl: The declared selection.

        Returns:
            The reason at the first kind that does not fit, or None.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = OfType(Family.numeric())
            >>> selector = CodeSelector({"Selector": {"ByDType": "Numeric"}}, numeric)
            >>> decl = SelectionType.identity(numeric)
            >>> DeferredSelectionFit()._cells(Code("cs.numeric().cast(pl.String)"), selector, decl)
            'it makes a String column, but the declared type is Decimal(precision=10, scale=2)'
        """
        matched = SelectionType.matched_kinds(selector.rule)
        for (kind,), outcome in decl.table.cells:
            if kind not in matched:
                continue
            for rendering in OpenType().renderings(kind):
                answered = self._answered(code, selector, rendering)
                declared = outcome.evaluated({SelectionType.AXIS: Rendered(kind, rendering)})
                if Fit().disagree(answered, declared):
                    return CellRefusal().wording(answered, declared)
        return None

    def _answered(
        self,
        code: Code[DeclaredLeaf | None],
        selector: CodeSelector,
        rendering: str,
    ) -> str | None:
        """Give what the argument makes of a picked column of one data type.

        Args:
            code: The code of the argument.
            selector: The selector of the code.
            rendering: The data type of the picked column.

        Returns:
            The data type of the result, or `None` where polars refuses it.

        Example:
            >>> from astrolabe._hooks.deferred.code import Code
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = OfType(Family.numeric())
            >>> selector = CodeSelector({"Selector": {"ByDType": "Numeric"}}, numeric)
            >>> DeferredSelectionFit()._answered(Code("cs.numeric().abs()"), selector, "Int8")
            'Int8'
        """
        probe = DeferredRemainder.PROBES[0]
        evaluation = DeferredEvaluation("row")
        answer = evaluation.at_column(code, {probe: rendering}, selector.node, probe)
        return None if answer.refused else answer.rendering
