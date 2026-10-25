"""Remainder types as expressions, what they mean, and one canonical form for each.

A remainder is the columns of a frame that its type does not name.
A function writes it as a variable, for example
`Rest = other_columns()` in `df: TypedDataFrame[Trades, Rest]`, and
each call binds `Rest` to the columns that the caller brings beyond
`Trades`. A return type can then state what the function did to
those columns, for example `Minus[Rest, Literal["notional"]]`, which
is `Rest` with `notional` taken out. Each such change, for example a
filter or a suffix on every name, is an effect. So a remainder type
is an expression: a variable, effects applied to one, or two
expressions joined.

Code asks three questions about an expression, and each has its own
method. `denote` gives its meaning: bind every variable to a real
column set, and the expression gives one column set. `know` gives
what the checker can prove: bind every variable to a record, that
is, the names proved absent and the data types allowed, and each
effect's record rule gives the record of the whole. `normalise`
decides whether two expressions are the same type, because it gives
each one a canonical form.

Two spellings of one type must give one canonical form. If not,
mypy, which compares types by their structure, treats them as
different types. Every form has the same shape, read from the outside in:

    data type?  over  prefix?  over  suffix?  over  filter?  over  base

The base is a variable, nothing, or a union of canonical forms. Each
effect of an expression folds into one of the four slots. A
recursive function over the expression computes the form, so every
expression has exactly one. `Applied` is an effect before it folds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from astrolabe._hooks.remainder.absence import Always, Never
from astrolabe._hooks.remainder.effects import (
    Both,
    Effect,
    Emptied,
    PrefixedBy,
    Retyped,
    SuffixedBy,
    Unchanged,
    Where,
)
from astrolabe._hooks.remainder.record import ColumnSet, Record
from astrolabe._hooks.remainder.rendering import StableText
from astrolabe._hooks.remainder.selector import (
    Conjunction,
    NameTest,
    Selector,
    SelectorEntailment,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class Base:
    """The innermost part of a canonical form, which every effect applies to.

    A base is a variable that a call binds, a union of canonical
    forms, or `Nothing`, the remainder with no columns.

    Example:
        >>> Variable("R").denote({"R": ColumnSet.of(a="Int64")}).names()
        ('a',)
    """

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the real columns of this base, once each variable is bound.

        Args:
            bindings: The column set of each variable, by name.

        Returns:
            The columns of the base.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Nothing().denote({}).names()
            ()
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Variable(Base):
    """A remainder variable, which each call binds to the columns that its caller brings.

    Example:
        >>> Variable("R").name
        'R'
    """

    name: str

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the column set that the call binds to this variable.

        Args:
            bindings: The column set of each variable, by name.

        Returns:
            The column set bound to this variable's name.

        Example:
            >>> Variable("R").denote({"R": ColumnSet.of(b="String")}).names()
            ('b',)
        """
        return bindings[self.name]


@dataclass(frozen=True)
class Nothing(Base):
    """The remainder with no columns.

    An operation that deletes every undeclared column leaves this,
    for example a `select` of fixed names. A frame type spells such a
    remainder as no remainder at all, that is, as a closed frame.

    Example:
        >>> Nothing().denote({}).entries
        ()
    """

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the empty column set.

        Args:
            bindings: Not read.

        Returns:
            The column set with no columns.

        Example:
            >>> Nothing().denote({"R": ColumnSet.of(a="Int64")}).names()
            ()
        """
        del bindings
        return ColumnSet(())


@dataclass(frozen=True)
class Union(Base):
    """Several canonical forms joined into one remainder.

    A frame that joins the undeclared columns of two frames, for
    example after a horizontal concat, has this base. The parts are
    flat and sorted, so neither the order nor the grouping in which
    the user wrote them can tell two equal types apart. `Unioned`
    builds it.

    Example:
        >>> u = Union(
        ...     (Canonical.of(Variable("A")), Canonical.of(Variable("B")))
        ... )
        >>> len(u.parts)
        2
    """

    parts: tuple[Canonical, ...]

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give every column of every part.

        Args:
            bindings: The column set of each variable, by name.

        Returns:
            The columns of all the parts together.

        Example:
            >>> u = Union(
            ...     (Canonical.of(Variable("A")), Canonical.of(Variable("B")))
            ... )
            >>> u.denote({
            ...     "A": ColumnSet.of(a="Int64"),
            ...     "B": ColumnSet.of(b="String"),
            ... }).names()
            ('a', 'b')
        """
        entries: tuple[tuple[str, str], ...] = ()
        for part in self.parts:
            entries += part.denote(bindings).entries
        return ColumnSet.holding(entries)


@dataclass(frozen=True)
class Canonical:
    """The one normal spelling of a remainder expression.

    The checker compares two types by their structure and knows
    nothing of what they mean. So each remainder type is encoded from
    its canonical form, and two expressions that mean the same thing
    must give equal forms. Read from the outside in, a form holds a
    new data type, a prefix, a suffix, a filter and a base. An empty
    slot changes nothing: `None`, or the empty text for an affix.
    Each effect of an expression folds into one of these slots.

    Example:
        >>> Canonical.of(Variable("R")).suffix
        ''
    """

    base: Base
    where: Selector | None = None
    suffix: str = ""
    prefix: str = ""
    retype: str | None = None

    @classmethod
    def of(cls, base: Base) -> Canonical:
        """Build the form of a base with every slot empty.

        Args:
            base: The innermost part.

        Returns:
            The form of the base alone.

        Example:
            >>> Canonical.of(Nothing()).where is None
            True
        """
        return cls(base)

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the real columns of this form, once each variable is bound.

        The slots apply from the inside out: the filter, the suffix,
        the prefix, then the new data type.

        Args:
            bindings: The column set of each variable, by name.

        Returns:
            The columns of the form.

        Example:
            >>> k = Canonical(Variable("R"), suffix="_p")
            >>> k.denote({"R": ColumnSet.of(price="Int64")}).names()
            ('price_p',)
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> kept = Canonical(Variable("R"), where=NameTest(StartsWith("p")))
            >>> kept.denote({"R": ColumnSet.of(price="Int64", venue="String")}).names()
            ('price',)
            >>> k = Canonical(Variable("R"), prefix="s_", retype="UInt32")
            >>> k.denote({"R": ColumnSet.of(a="Int64")}).entries
            (('s_a', 'UInt32'),)
        """
        columns = self.base.denote(bindings)
        if self.where is not None:
            columns = Where(self.where).apply(columns)
        columns = SuffixedBy(self.suffix).apply(columns)
        columns = PrefixedBy(self.prefix).apply(columns)
        if self.retype is not None:
            columns = Retyped(self.retype).apply(columns)
        return columns

    def sort_key(self) -> str:
        """Give a text key that puts canonical forms in one fixed order.

        A union sorts its parts by this key, so the order in which the
        user wrote them cannot tell two equal types apart. The key is
        the same in every run of mypy, and it reads only the form
        itself (see `StableText`).

        Returns:
            The text of the form.

        Example:
            >>> Canonical.of(Variable("A")).sort_key()
            "Canonical(Variable('A'),None,'','',None)"
        """
        return StableText()(self)


class Expr:
    """A remainder expression: a variable, effects applied to one, or two joined.

    An expression is syntax, the tree that a type spells, for
    example `Minus[Rest, Literal["notional"]]`. Its meaning is known
    only once each call binds its variables to real columns. So a
    type carries the expression, and the engine answers questions
    about it for every possible binding at once.

    Example:
        >>> Ref("R").denote({"R": ColumnSet.of(a="Int64")}).names()
        ('a',)
    """

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the real columns of this expression, once each variable is bound.

        Args:
            bindings: The column set of each variable, by name.

        Returns:
            The columns of the expression.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Ref("R").denote({"R": ColumnSet(())}).entries
            ()
        """
        raise NotImplementedError

    def know(self, records: Mapping[str, Record]) -> Record:
        """Give what the checker can prove about this expression.

        A record holds the names proved absent and the data types
        allowed. The record of a variable is what its declaration
        promises. The record of an effect is its record rule applied
        to the record of the inner expression.

        Args:
            records: The record of each variable, by name.

        Returns:
            The record of this expression.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> known = Ref("R").know({"R": Record.unconstrained()})
            >>> known.types.members == Record.unconstrained().types.members
            True
        """
        raise NotImplementedError

    def normalise(self) -> Canonical:
        """Give the canonical form of this expression.

        Returns:
            The one form that every equal expression also gives.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Ref("R").normalise().base
            Variable(name='R')
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Ref(Expr):
    """A remainder variable in expression position, for example `Rest`.

    Example:
        >>> Ref("R").name
        'R'
    """

    name: str

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the column set that the call binds to this variable.

        Args:
            bindings: The column set of each variable, by name.

        Returns:
            The column set bound to this name.

        Example:
            >>> Ref("R").denote({"R": ColumnSet.of(b="String")}).names()
            ('b',)
        """
        return bindings[self.name]

    def know(self, records: Mapping[str, Record]) -> Record:
        """Give what the declaration of this variable promises.

        Args:
            records: The record of each variable, by name.

        Returns:
            The record bound to this name.

        Example:
            >>> Ref("R").know({"R": Record.unconstrained()}).absence
            Never()
        """
        return records[self.name]

    def normalise(self) -> Canonical:
        """Give the form of this variable with every slot empty.

        Returns:
            The variable, with all four slots empty.

        Example:
            >>> Ref("R").normalise().retype is None
            True
        """
        return Canonical.of(Variable(self.name))


@dataclass(frozen=True)
class Combined(Expr):
    """Two expressions whose columns join into one remainder.

    This is the expression form of `Both[A, B]`.

    Example:
        >>> base = Combined(Ref("B"), Ref("A")).normalise().base
        >>> [p.base for p in base.parts]
        [Variable(name='A'), Variable(name='B')]
    """

    left: Expr
    right: Expr

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give every column of both operands.

        Args:
            bindings: The column set of each variable, by name.

        Returns:
            The columns of both operands.

        Example:
            >>> e = Combined(Ref("A"), Ref("B"))
            >>> e.denote({
            ...     "A": ColumnSet.of(a="Int64"),
            ...     "B": ColumnSet.of(b="String"),
            ... }).names()
            ('a', 'b')
        """
        return Both().apply(self.left.denote(bindings), self.right.denote(bindings))

    def know(self, records: Mapping[str, Record]) -> Record:
        """Join the records of the two operands.

        Args:
            records: The record of each variable, by name.

        Returns:
            The record of the joined remainder, from `Both`.

        Example:
            >>> e = Combined(Ref("A"), Ref("B"))
            >>> e.know({
            ...     "A": Record.unconstrained(),
            ...     "B": Record.unconstrained(),
            ... }).absence.holds_of("x")
            False
        """
        return Both().transform(self.left.know(records), self.right.know(records))

    def normalise(self) -> Canonical:
        """Give the form of the union of the two operands.

        Returns:
            The union of the forms of the operands, built by
            `Unioned`, so that grouping and order do not count.

        Example:
            >>> a = Combined(Combined(Ref("A"), Ref("B")), Ref("C")).normalise()
            >>> b = Combined(Ref("A"), Combined(Ref("B"), Ref("C"))).normalise()
            >>> a == b
            True
        """
        return Unioned()(side.normalise() for side in (self.left, self.right))


class Unioned:
    """Build the canonical form of a union from the forms of its operands.

    The meaning of a union does not depend on how its operands are
    grouped, in which order they come, or how often one repeats. A form
    that keeps any of these gives one type two spellings. So this
    class flattens nested unions, sorts the parts and keeps
    each part once. An operand with no columns adds nothing for any
    binding, so it is left out. A union of one operand is that
    operand.

    A union form never has an effect in its slots, because `Applied`
    folds each effect into the parts. So a nested union can always
    flatten into the outer one.

    Example:
        >>> forms = [Canonical.of(Variable("B")), Canonical.of(Variable("A"))]
        >>> [part.base for part in Unioned()(forms).base.parts]
        [Variable(name='A'), Variable(name='B')]
        >>> Unioned()([Canonical.of(Nothing()), Canonical.of(Variable("S"))]).base
        Variable(name='S')
        >>> Unioned()([Canonical.of(Nothing())]).base
        Nothing()
    """

    def __call__(self, forms: Iterable[Canonical]) -> Canonical:
        """Join the forms of the operands into one canonical form.

        Args:
            forms: The canonical form of each operand.

        Returns:
            The canonical form of the union: `Nothing` if no operand
            has columns, the one operand if only one is left, or a
            `Union` base.

        Example:
            >>> pair = Combined(Ref("A"), Ref("B")).normalise()
            >>> len(Unioned()([pair, Canonical.of(Variable("C"))]).base.parts)
            3
        """
        empty = Canonical.of(Nothing())
        kept: list[Canonical] = []
        for form in forms:
            operands = form.base.parts if isinstance(form.base, Union) else (form,)
            kept.extend(operand for operand in operands if operand != empty)
        parts = tuple(sorted(dict.fromkeys(kept), key=Canonical.sort_key))
        if not parts:
            return empty
        if len(parts) == 1:
            return parts[0]
        return Canonical.of(Union(parts))


@dataclass(frozen=True)
class Applied(Expr):
    """One effect applied to an inner expression, before it folds into a canonical form.

    An effect is what one operation does to a remainder, for example
    a filter or a rename (see `Effect`). Each subclass states two
    things. `effect` names the effect, so the meaning and the record
    of the node are the effect's own rules. `fold` states how the
    effect merges into the canonical form of the inner expression.

    Example:
        >>> Suffix(Ref("R"), "_p").normalise().suffix
        '_p'
    """

    inner: Expr

    def effect(self) -> Effect:
        """Give the effect that this node applies.

        Returns:
            The effect, whose `apply` and `transform` give the meaning
            and the record of this node.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Suffix(Ref("R"), "_p").effect()
            SuffixedBy(suffix='_p')
        """
        raise NotImplementedError

    def fold(self, form: Canonical) -> Canonical:
        """Merge this effect into the canonical form of the inner expression.

        Args:
            form: The canonical form of the inner expression. Its base
                is not a union.

        Returns:
            The form of the whole expression, before `Simplified`
            deletes the slots that change nothing.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Suffix(Ref("R"), "_p").fold(Canonical.of(Variable("R"))).suffix
            '_p'
        """
        raise NotImplementedError

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Apply the effect to the real columns of the inner expression.

        Args:
            bindings: The column set of each variable, by name.

        Returns:
            The columns after the effect.

        Example:
            >>> named = Suffix(Ref("R"), "_p")
            >>> named.denote({"R": ColumnSet.of(a="Int64")}).names()
            ('a_p',)
        """
        return self.effect().apply(self.inner.denote(bindings))

    def know(self, records: Mapping[str, Record]) -> Record:
        """Apply the record rule of the effect to the record of the inner expression.

        Args:
            records: The record of each variable, by name.

        Returns:
            The record after the effect.

        Example:
            >>> named = Suffix(Ref("R"), "_p")
            >>> named.know({"R": Record.unconstrained()}).absence.holds_of("a")
            True
        """
        return self.effect().transform(self.inner.know(records))

    def normalise(self) -> Canonical:
        """Give the canonical form of this expression.

        Every effect acts on each column alone. So an effect over a
        union is the union of the effect over each part: the effect
        folds into each part, and the parts join again. A union form
        therefore never carries an effect of its own.

        Returns:
            This expression's canonical form.

        Example:
            >>> Retype(Suffix(Ref("R"), "_p"), "UInt32").normalise().retype
            'UInt32'
            >>> over = Suffix(Combined(Ref("A"), Ref("B")), "_x").normalise()
            >>> each = Combined(Suffix(Ref("A"), "_x"), Suffix(Ref("B"), "_x")).normalise()
            >>> over == each
            True
        """
        inner = self.inner.normalise()
        if isinstance(inner.base, Union):
            return Unioned()(self._folded(part) for part in inner.base.parts)
        return self._folded(inner)

    def _folded(self, form: Canonical) -> Canonical:
        """Fold this effect into one form that is not a union, then simplify.

        Args:
            form: A canonical form whose base is a variable or
                `Nothing`.

        Returns:
            The canonical form of this effect over that form.

        Example:
            >>> Suffix(Ref("R"), "_p")._folded(Canonical.of(Variable("R"))).suffix
            '_p'
        """
        return Simplified()(self.fold(form))


@dataclass(frozen=True)
class Carried(Expr):
    """A canonical form used as an expression, so that more effects can apply to it.

    A hook for a frame operation decodes the remainder of the frame
    as a canonical form, and then applies the operation's effect to
    it. The effect nodes take expressions, so this node wraps the form
    as one. The canonical form of this node is the wrapped form itself,
    so no work is repeated.

    Example:
        >>> Carried(Canonical.of(Variable("R"))).normalise().base
        Variable(name='R')
    """

    form: Canonical

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the real columns of the wrapped form.

        Args:
            bindings: The column set of each variable, by name.

        Returns:
            The columns of the form.

        Example:
            >>> named = Carried(Canonical.of(Variable("R")))
            >>> named.denote({"R": ColumnSet.of(a="Int64")}).names()
            ('a',)
        """
        return self.form.denote(bindings)

    def know(self, records: Mapping[str, Record]) -> Record:
        """Give what the checker can prove about the wrapped form.

        Each slot reads back as the effect that folded into it. The
        record rules of the filter, the suffix, the prefix and the new
        data type apply, in that order, to the record of the base.

        Args:
            records: The record of each variable, by name.

        Returns:
            The record of the wrapped form.

        Example:
            >>> named = Carried(Canonical(Variable("R"), suffix="_p"))
            >>> named.know({"R": Record.unconstrained()}).absence.holds_of("a")
            True
            >>> fronted = Carried(Canonical(Variable("R"), prefix="s_"))
            >>> fronted.know({"R": Record.unconstrained()}).absence.holds_of("x")
            True
            >>> retyped = Carried(Canonical(Variable("R"), retype="UInt32"))
            >>> sorted(retyped.know({"R": Record.unconstrained()}).types.members)
            ['UInt32']
        """
        form = self.form
        record = self._base_record(records)
        if form.where is not None:
            record = Where(form.where).transform(record)
        if form.suffix:
            record = SuffixedBy(form.suffix).transform(record)
        if form.prefix:
            record = PrefixedBy(form.prefix).transform(record)
        if form.retype is not None:
            record = Retyped(form.retype).transform(record)
        return record

    def _base_record(self, records: Mapping[str, Record]) -> Record:
        """Give the record of the base.

        The record of a union base joins the records of its parts with
        `Both`.

        Args:
            records: The record of each variable, by name.

        Returns:
            The record of the variable, the joined record of the
            union, or the record of the remainder with no columns.

        Example:
            >>> named = Carried(Canonical.of(Variable("R")))
            >>> named._base_record({"R": Record.unconstrained()}).absence
            Never()
            >>> parts = (Canonical.of(Variable("R")), Canonical.of(Variable("S")))
            >>> joined = Carried(Canonical.of(Union(parts)))
            >>> records = {"R": Record.unconstrained(), "S": Record.unconstrained()}
            >>> joined._base_record(records).absence
            And(left=Never(), right=Never())
        """
        base = self.form.base
        if isinstance(base, Variable):
            return records[base.name]
        if isinstance(base, Nothing):
            return Emptied().transform(Record.unconstrained())
        union = checked_union_base(base)
        parts = [Carried(part).know(records) for part in union.parts]
        record = parts[0]
        for part in parts[1:]:
            record = Both().transform(record, part)
        return record

    def normalise(self) -> Canonical:
        """Give the canonical form of the wrapped form.

        A form that is not a union is already canonical. A union can
        come from a type that a user wrote. Such a union is not sorted,
        and it can apply an effect to the whole union. So each part
        gets the effects of the union, and the parts join again.

        Returns:
            The canonical form.

        Example:
            >>> form = Canonical(Variable("R"), suffix="_p")
            >>> Carried(form).normalise() is form
            True
            >>> parts = (Canonical.of(Variable("B")), Canonical.of(Variable("A")))
            >>> written = Canonical(Union(parts), suffix="_x")
            >>> [(p.base.name, p.suffix) for p in Carried(written).normalise().base.parts]
            [('A', '_x'), ('B', '_x')]
        """
        if not isinstance(self.form.base, Union):
            return self.form
        parts = self.form.base.parts
        return Unioned()(self._pushed(part) for part in parts)

    def _pushed(self, part: Canonical) -> Canonical:
        """Apply the effects of the wrapped union to one of its parts.

        The effects apply in the order of the canonical form: the
        filter, the suffix, the prefix, then the new data type. A slot
        that is empty adds no effect.

        Args:
            part: One part of the wrapped union.

        Returns:
            The canonical form of the part under those effects.

        Example:
            >>> written = Canonical(Union(()), prefix="p_")
            >>> Carried(written)._pushed(Canonical.of(Variable("A"))).prefix
            'p_'
            >>> written = Canonical(Union(()), suffix="_s")
            >>> Carried(written)._pushed(Canonical.of(Variable("A"))).suffix
            '_s'
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> written = Canonical(Union(()), where=NameTest(StartsWith("p")))
            >>> filtered = Carried(written)._pushed(Canonical.of(Variable("A")))
            >>> type(filtered.where).__name__
            'NameTest'
        """
        form = self.form
        expression: Expr = Carried(part)
        if form.where is not None:
            expression = Filter(expression, form.where)
        if form.suffix:
            expression = Suffix(expression, form.suffix)
        if form.prefix:
            expression = Prefix(expression, form.prefix)
        if form.retype is not None:
            expression = Retype(expression, form.retype)
        return expression.normalise()


def checked_union_base(base: Base) -> Union:
    """Narrow a base to `Union` for mypy, after the caller ruled out the other bases.

    A failure means a bug in this module, not a bad input.

    Args:
        base: A base that the caller knows is neither a `Variable`
            nor `Nothing`.

    Returns:
        The same base, typed as a union.

    Raises:
        TypeError: The base is not a union.

    Example:
        >>> parts = (Canonical.of(Variable("R")),)
        >>> checked_union_base(Union(parts)).parts is parts
        True
    """
    if not isinstance(base, Union):  # pragma: no cover
        # Each caller tests for a variable and for `Nothing` first,
        # so only a union reaches this line.
        problem = f"not a union base: {base!r}"
        raise TypeError(problem)
    return base


@dataclass(frozen=True)
class Keep(Applied):
    """The remainder stays as it is, as after a sort or a row filter.

    Example:
        >>> Keep(Ref("R")).normalise() == Ref("R").normalise()
        True
    """

    def effect(self) -> Effect:
        """Give the effect that changes nothing.

        Returns:
            The `Unchanged` effect.

        Example:
            >>> Keep(Ref("R")).effect()
            Unchanged()
        """
        return Unchanged()

    def fold(self, form: Canonical) -> Canonical:
        """Give the inner form back, since this effect changes nothing.

        Args:
            form: The canonical form of the inner expression.

        Returns:
            The same form.

        Example:
            >>> Keep(Ref("R")).fold(Canonical.of(Variable("R"))).base
            Variable(name='R')
        """
        return form


@dataclass(frozen=True)
class Empty(Applied):
    """Every column of the remainder is deleted, as by a `select` of fixed names.

    Example:
        >>> Empty(Ref("R")).normalise().base
        Nothing()
    """

    def effect(self) -> Effect:
        """Give the effect that deletes every column.

        Returns:
            The `Emptied` effect.

        Example:
            >>> Empty(Ref("R")).effect()
            Emptied()
        """
        return Emptied()

    def fold(self, form: Canonical) -> Canonical:
        """Give the form of no columns, whatever the inner form was.

        Once every column is gone, nothing that the inner expression
        did can matter. So this effect replaces every slot and the
        base.

        Args:
            form: Not read.

        Returns:
            The form of `Nothing` with every slot empty.

        Example:
            >>> Empty(Ref("R")).fold(
            ...     Canonical(Variable("R"), suffix="_p")
            ... ).suffix
            ''
        """
        del form
        return Canonical.of(Nothing())


@dataclass(frozen=True)
class Retype(Applied):
    """Every column of the remainder keeps its name and gets one new data type.

    Example:
        >>> Retype(Ref("R"), "UInt32").normalise().retype
        'UInt32'
    """

    dtype: str

    def effect(self) -> Effect:
        """Give the effect that sets one data type on every column.

        Returns:
            The `Retyped` effect for this data type.

        Example:
            >>> Retype(Ref("R"), "UInt32").effect()
            Retyped(dtype='UInt32')
        """
        return Retyped(self.dtype)

    def fold(self, form: Canonical) -> Canonical:
        """Set this data type in the form, over any data type that the inner form set.

        A second new data type overwrites the first one, so only the
        outermost one counts.

        Args:
            form: The canonical form of the inner expression.

        Returns:
            The same form, with this data type.

        Example:
            >>> inner = Retype(Ref("R"), "Int64")
            >>> Retype(inner, "UInt32").normalise().retype
            'UInt32'
        """
        return replace(form, retype=self.dtype)


@dataclass(frozen=True)
class Suffix(Applied):
    """Every column name of the remainder gets a fixed suffix.

    Example:
        >>> Suffix(Ref("R"), "_p").normalise().suffix
        '_p'
    """

    affix: str

    def effect(self) -> Effect:
        """Give the effect that adds the suffix to every name.

        Returns:
            The `SuffixedBy` effect for this suffix.

        Example:
            >>> Suffix(Ref("R"), "_p").effect()
            SuffixedBy(suffix='_p')
        """
        return SuffixedBy(self.affix)

    def fold(self, form: Canonical) -> Canonical:
        """Add this suffix after the suffix that the inner form already adds.

        Two suffixes in a row are one longer suffix. A prefix changes
        the other end of the name, so it does not interfere. The
        suffix slot therefore takes this suffix, whatever the inner
        form holds.

        Args:
            form: The canonical form of the inner expression.

        Returns:
            The same form, with the longer suffix.

        Example:
            >>> Suffix(Suffix(Ref("R"), "_a"), "_b").normalise().suffix
            '_a_b'
        """
        return replace(form, suffix=form.suffix + self.affix)


@dataclass(frozen=True)
class Prefix(Applied):
    """Every column name of the remainder gets a fixed prefix.

    Example:
        >>> Prefix(Ref("R"), "s_").normalise().prefix
        's_'
    """

    affix: str

    def effect(self) -> Effect:
        """Give the effect that adds the prefix to every name.

        Returns:
            The `PrefixedBy` effect for this prefix.

        Example:
            >>> Prefix(Ref("R"), "s_").effect()
            PrefixedBy(prefix='s_')
        """
        return PrefixedBy(self.affix)

    def fold(self, form: Canonical) -> Canonical:
        """Add this prefix before the prefix that the inner form already adds.

        Args:
            form: The canonical form of the inner expression.

        Returns:
            The same form, with the longer prefix.

        Example:
            >>> Prefix(Prefix(Ref("R"), "b_"), "a_").normalise().prefix
            'a_b_'
        """
        return replace(form, prefix=self.affix + form.prefix)


@dataclass(frozen=True)
class Filter(Applied):
    """Only the columns of the remainder that a selector picks stay.

    A selector is a test on a column's name and data type, for
    example `cs.numeric()`. This is the one effect whose fold must
    compute. The selector runs after everything the inner form did.
    So it tests names that the inner renames changed, and data types
    that an inner new data type set. The fold rewrites the selector
    to test the base column instead, and it can then sit in the
    filter slot, inside the renames.

    Example:
        >>> from astrolabe._hooks.remainder.absence import StartsWith
        >>> named = Filter(Ref("R"), NameTest(StartsWith("a")))
        >>> named.normalise().where is None
        False
    """

    selector: Selector

    def effect(self) -> Effect:
        """Give the effect that keeps the picked columns.

        Returns:
            The `Where` effect for this selector.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> Filter(Ref("R"), OfType(Family.numeric())).effect()
            Where(selector=OfType(family=NumericType))
        """
        return Where(self.selector)

    def fold(self, form: Canonical) -> Canonical:
        """Move the selector inside the renames, and merge it with the inner filter.

        The rewrite undoes the three outer slots, the outermost first.
        A new data type decides every test on data types, because
        every column has that data type. Then the prefix and the suffix
        each take their added text into account. The rewritten
        selector picks a base column exactly if this one picks the
        column that the base column becomes.

        Two filters in a row keep a column only if both pick it, so
        one filter with both tests means the same. If one test picks
        no more than the other, the pair means the narrower test
        alone, and `Conjunction` keeps only that one.

        Args:
            form: The canonical form of the inner expression.

        Returns:
            The same form, with one filter that makes both tests.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> renamed = Suffix(Ref("R"), "b_")
            >>> folded = Filter(renamed, NameTest(StartsWith("ab"))).normalise()
            >>> folded.where.picks("a", "Int64")
            True
            >>> f = Filter(Ref("R"), NameTest(StartsWith("a")))
            >>> type(f.fold(Canonical.of(Variable("R"))).where).__name__
            'NameTest'
        """
        selector = self.selector
        if form.retype is not None:
            selector = selector.after_retype(form.retype)
        selector = selector.through_prefix(form.prefix).through_suffix(form.suffix)
        if form.where is not None:
            selector = Conjunction()(form.where, selector)
        return replace(form, where=selector)


class Simplified:
    """Delete the slots of a canonical form that change nothing, after each fold.

    Some slots do not change the meaning, and a form that keeps them
    gives one type two spellings. A filter that picks every
    column changes nothing, so it goes. A filter that picks no column
    leaves no columns, so the whole form becomes `Nothing`. Once no
    column is left, no other slot can change anything. A filter that
    stays is rewritten to its canonical selector.

    `SelectorEntailment` decides whether a filter picks every column
    or none, and its answer is exact.

    Example:
        >>> form = Canonical(Variable("R"), where=NameTest(Always()))
        >>> Simplified()(form).where is None
        True
    """

    def __call__(self, form: Canonical) -> Canonical:
        """Give the simplest form with the same meaning.

        Args:
            form: The form that a fold gave.

        Returns:
            The same form, with every slot that changes nothing
            deleted.

        Example:
            >>> Simplified()(Canonical.of(Variable("R"))).base
            Variable(name='R')
            >>> Simplified()(Canonical(Variable("R"), where=NameTest(Never()))).base
            Nothing()
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> f = Canonical(Variable("R"), where=NameTest(StartsWith("a")))
            >>> Simplified()(f).where is None
            False
        """
        where = form.where
        if isinstance(form.base, Nothing):
            return Canonical.of(Nothing())
        if where is None:
            return form
        entails = SelectorEntailment()
        if entails(where, NameTest(Never())):
            return Canonical.of(Nothing())
        if entails(NameTest(Always()), where):
            return replace(form, where=None)
        return replace(form, where=where.canonical())


class Grafted:
    """Put the caller's remainder in place of each variable in a declared return type.

    A function declares its return type over its own remainder
    variables, for example `Minus[Rest, Literal["notional"]]`. At a
    call, each variable stands for the remainder of the caller's
    argument, which is itself a canonical form with its own effects.
    This class replaces each variable with that form. Then it applies
    the slots of the return form again on top, as effects. So a
    filter is rewritten through the renames of the caller's form, in
    the same way as in any fold, and the result is canonical.

    Example:
        >>> from astrolabe._hooks.remainder.absence import OneOf
        >>> from astrolabe._hooks.remainder.selector import SelNot
        >>> gone = SelNot(NameTest(OneOf(frozenset({"x"}))))
        >>> form = Canonical(Variable("R"), where=gone)
        >>> bound = {"R": Canonical(Variable("C"), suffix="_r")}
        >>> Grafted()(form, bound).base
        Variable(name='C')
    """

    def __call__(self, form: Canonical, parts: Mapping[str, Canonical]) -> Canonical | None:
        """Give the form with each variable replaced, in canonical form.

        The slots of the form apply again over the new base, as
        effects: the filter, the suffix, the prefix, then the new data
        type. The fold rewrites the filter through the renames of the
        bound forms.

        Args:
            form: The form whose variables are replaced.
            parts: The form bound to each variable, by name.

        Returns:
            The new canonical form, or None if a variable has no
            binding.

        Example:
            >>> Grafted()(Canonical.of(Variable("R")), {}) is None
            True
            >>> Grafted()(Canonical.of(Nothing()), {}).base
            Nothing()
            >>> renamed = Canonical(
            ...     Variable("R"), suffix="_s", prefix="p_", retype="UInt32"
            ... )
            >>> bound = {"R": Canonical.of(Variable("C"))}
            >>> grafted = Grafted()(renamed, bound)
            >>> grafted.suffix, grafted.prefix, grafted.retype
            ('_s', 'p_', 'UInt32')
        """
        expr = self._base_expr(form.base, parts)
        if expr is None:
            return None
        if form.where is not None:
            expr = Filter(expr, form.where)
        if form.suffix:
            expr = Suffix(expr, form.suffix)
        if form.prefix:
            expr = Prefix(expr, form.prefix)
        if form.retype is not None:
            expr = Retype(expr, form.retype)
        return expr.normalise()

    def _base_expr(self, base: Base, parts: Mapping[str, Canonical]) -> Expr | None:
        """Rebuild the base with each variable replaced.

        Args:
            base: The base of the form.
            parts: The form bound to each variable, by name.

        Returns:
            The base as an expression over the bound forms, or None
            if a variable has no binding.
        """
        if isinstance(base, Variable):
            bound = parts.get(base.name)
            return None if bound is None else Carried(bound)
        if isinstance(base, Nothing):
            return Carried(Canonical.of(Nothing()))
        return self._union_expr(checked_union_base(base), parts)

    def _union_expr(self, union: Union, parts: Mapping[str, Canonical]) -> Expr | None:
        """Rebuild a union base with each variable replaced.

        Args:
            union: The union base.
            parts: The form bound to each variable, by name.

        Returns:
            The union as a chain of `Combined` nodes, or None if a
            variable of any part has no binding.

        Example:
            >>> pair = Union((Canonical.of(Variable("R")), Canonical.of(Variable("S"))))
            >>> bound = {"R": Canonical.of(Variable("C")), "S": Canonical.of(Variable("D"))}
            >>> joined = Grafted()._union_expr(pair, bound)
            >>> sorted(p.base.name for p in joined.normalise().base.parts)
            ['C', 'D']
            >>> Grafted()._union_expr(pair, {"R": bound["R"]}) is None
            True
        """
        grafted: Expr | None = None
        for operand in union.parts:
            piece = self(operand, parts)
            if piece is None:
                return None
            carried = Carried(piece)
            grafted = carried if grafted is None else Combined(grafted, carried)
        return grafted
