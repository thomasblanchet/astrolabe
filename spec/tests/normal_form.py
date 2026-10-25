"""Remainder expressions, their meaning, and their canonical form.

The specification asks two different questions about an expression, and
an earlier draft answered them with one mechanism.

The first is what the expression means. `Expr.denote` answers it: give
every variable a column set, and the expression denotes one column set.

The second is when two expressions are the same type. `normalise`
answers it by computing a canonical form. A canonical form is not a
fixed point of a rewriting system here. It is the value of a function
defined by structural induction, so it exists and is unique with no
termination or confluence argument needed.

The canonical form fixes one shape for every expression:

    retype?  over  prefix?  over  suffix?  over  filter?  over  base

Reading inward, the base is a variable or a union of canonical forms.
Every effect an expression applies is folded into one of the four
slots. `Applied` is what an effect looks like before folding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product, starmap
from typing import TYPE_CHECKING

from spec.tests.model import (
    Always,
    Both,
    ColumnSet,
    Conjunction,
    Effect,
    Emptied,
    NameTest,
    Never,
    PrefixedBy,
    Retyped,
    Selector,
    SelectorEntailment,
    StableText,
    SuffixedBy,
    Unchanged,
    Where,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from spec.tests.universe import Universe


class Base:
    """What sits innermost in a canonical form.

    A base is a variable, a union of canonical forms, or nothing at all.

    Example:
        >>> Variable("R").denote({"R": ColumnSet.of(a="Int64")}).names()
        ('a',)
    """

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the column set this base stands for under one binding.

        Args:
            bindings: A column set for each variable.

        Returns:
            The column set the base denotes.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Nothing().denote({}).names()
            ()
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Variable(Base):
    """A remainder the caller supplies.

    Example:
        >>> Variable("R").name
        'R'
    """

    name: str

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Look this variable up in the binding.

        Args:
            bindings: A column set for each variable.

        Returns:
            The column set bound to this variable's name.

        Example:
            >>> Variable("R").denote({"R": ColumnSet.of(b="String")}).names()
            ('b',)
        """
        return bindings[self.name]


@dataclass(frozen=True)
class Nothing(Base):
    """No columns at all.

    This is what an emptying effect leaves. A remainder proving every
    name absent carries no information, so the specification writes it
    as no remainder.

    Example:
        >>> Nothing().denote({}).entries
        ()
    """

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the empty column set.

        Args:
            bindings: Ignored.

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
    """Several canonical forms held together.

    The operands are sorted, so neither the order they were written in
    nor the way they were grouped can distinguish two equal types.

    Example:
        >>> u = Union((Canonical.of(Variable("A")), Canonical.of(Variable("B"))))
        >>> len(u.parts)
        2
    """

    parts: tuple[Canonical, ...]

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Hold every column of every operand.

        Args:
            bindings: A column set for each variable.

        Returns:
            The column set holding every operand's columns.

        Example:
            >>> u = Union((Canonical.of(Variable("A")), Canonical.of(Variable("B"))))
            >>> u.denote({"A": ColumnSet.of(a="Int64"), "B": ColumnSet.of(b="String")}).names()
            ('a', 'b')
        """
        entries: tuple[tuple[str, str], ...] = ()
        for part in self.parts:
            entries += part.denote(bindings).entries
        return ColumnSet.holding(entries)


@dataclass(frozen=True)
class Canonical:
    """One expression, written in the single shape every expression has.

    Reading inward: a data type replacement, a prefix, a suffix, a
    filter, and a base. Every effect an expression applies folds into
    one of these slots, so two expressions are the same type exactly
    when their canonical forms are equal.

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
        """Build the canonical form that applies no effect to a base.

        Args:
            base: What sits innermost.

        Returns:
            The canonical form of the base alone.

        Example:
            >>> Canonical.of(Nothing()).where is None
            True
        """
        return cls(base)

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the column set this canonical form stands for.

        The four slots are applied inward to outward, in the order the
        shape names them: the filter, the suffix, the prefix, and the
        data type replacement.

        Args:
            bindings: A column set for each variable.

        Returns:
            The column set the form denotes.

        Example:
            >>> k = Canonical(Variable("R"), suffix="_p")
            >>> k.denote({"R": ColumnSet.of(price="Int64")}).names()
            ('price_p',)
            >>> k = Canonical(Variable("R"), prefix="s_", retype="UInt32")
            >>> k.denote({"R": ColumnSet.of(a="Int64")}).entries
            (('s_a', 'UInt32'),)
        """
        columns = self.base.denote(bindings)
        if self.where is not None:
            columns = Where(self.where).apply(columns)
        columns = SuffixedBy(self.suffix).apply(columns)
        columns = PrefixedBy(self.prefix).apply(columns)
        if self.retype is None:
            return columns
        return Retyped(self.retype).apply(columns)

    def sort_key(self) -> str:
        """Give a text key that orders two canonical forms.

        The key is total, is the same in every run, and reads only the
        form's own content. A union sorts its operands by it, so the
        order two operands were written in cannot distinguish two equal
        types.

        Returns:
            Text derived only from this form's meaning.

        Example:
            >>> Canonical.of(Variable("A")).sort_key()
            "Canonical(Variable('A'),None,'','',None)"
        """
        return StableText()(self)


class Expr:
    """A remainder expression: a variable, or an effect applied to one.

    An expression is syntax. Its meaning is fixed only once its
    variables are given column sets, which is why it is carried in a
    type and evaluated later.

    Example:
        >>> Ref("R").denote({"R": ColumnSet.of(a="Int64")}).names()
        ('a',)
    """

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Give the column set this expression stands for.

        Args:
            bindings: A column set for each variable.

        Returns:
            The column set the expression denotes.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Ref("R").denote({"R": ColumnSet(())}).entries
            ()
        """
        raise NotImplementedError

    def normalise(self) -> Canonical:
        """Compute this expression's canonical form.

        Returns:
            The one canonical form of this expression.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Ref("R").normalise().base
            Variable(name='R')
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Ref(Expr):
    """A reference to a remainder the caller supplies.

    Example:
        >>> Ref("R").name
        'R'
    """

    name: str

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Look this variable up in the binding.

        Args:
            bindings: A column set for each variable.

        Returns:
            The column set bound to this name.

        Example:
            >>> Ref("R").denote({"R": ColumnSet.of(b="String")}).names()
            ('b',)
        """
        return bindings[self.name]

    def normalise(self) -> Canonical:
        """Give the canonical form that applies no effect to this variable.

        Returns:
            This variable with all four slots empty.

        Example:
            >>> Ref("R").normalise().retype is None
            True
        """
        return Canonical.of(Variable(self.name))


@dataclass(frozen=True)
class Combined(Expr):
    """Two expressions held together in one remainder.

    Example:
        >>> base = Combined(Ref("B"), Ref("A")).normalise().base
        >>> [p.base for p in base.parts]
        [Variable(name='A'), Variable(name='B')]
    """

    left: Expr
    right: Expr

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Hold every column of both operands.

        Args:
            bindings: A column set for each variable.

        Returns:
            The column set holding both operands' columns.

        Example:
            >>> e = Combined(Ref("A"), Ref("B"))
            >>> e.denote({"A": ColumnSet.of(a="Int64"), "B": ColumnSet.of(b="String")}).names()
            ('a', 'b')
        """
        return Both().apply(self.left.denote(bindings), self.right.denote(bindings))

    def normalise(self) -> Canonical:
        """Give the canonical form of the union of the two operands.

        Returns:
            The union of the operands' canonical forms, by `Unioned`.

        Example:
            >>> a = Combined(Combined(Ref("A"), Ref("B")), Ref("C")).normalise()
            >>> b = Combined(Ref("A"), Combined(Ref("B"), Ref("C"))).normalise()
            >>> a == b
            True
        """
        return Unioned()(side.normalise() for side in (self.left, self.right))


class Unioned:
    """Build the canonical form of a union from the forms of its operands.

    Flattening removes the grouping, sorting removes the order, and
    dropping a repeat removes the count. A union's meaning has none of
    the three, so a node that kept them would distinguish two types
    that are the same. A provably empty operand contributes no columns
    under any binding, so it is dropped, and a union of one operand is
    that operand.

    A union form carries no effect: `Applied` folds every effect into
    the parts. So a union nested in another always flattens into it.

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
        """Unite the operands' forms into one canonical form.

        A nested union gives its parts in its place, and the bare form
        of no columns at all is dropped. One remaining operand needs no
        union around it.

        Args:
            forms: The canonical form of each operand.

        Returns:
            The union's canonical form.

        Example:
            >>> pair = Combined(Ref("A"), Ref("B")).normalise()
            >>> united = Unioned()([pair, Canonical.of(Variable("C"))])
            >>> [part.base for part in united.base.parts]
            [Variable(name='A'), Variable(name='B'), Variable(name='C')]
            >>> Unioned()([Canonical.of(Variable("R"))]).base
            Variable(name='R')
        """
        empty = Canonical.of(Nothing())
        kept: list[Canonical] = []
        for form in forms:
            parts = form.base.parts if isinstance(form.base, Union) else (form,)
            kept.extend(part for part in parts if part != empty)
        unique = tuple(sorted(dict.fromkeys(kept), key=Canonical.sort_key))
        if not unique:
            return empty
        if len(unique) == 1:
            return unique[0]
        return Canonical.of(Union(unique))


@dataclass(frozen=True)
class Applied(Expr):
    """One effect applied to an inner expression.

    Every subclass says two things: which effect it applies, so its
    meaning is the model's own rule, and how that effect folds into a
    canonical form.

    Example:
        >>> Suffix(Ref("R"), "_p").normalise().suffix
        '_p'
    """

    inner: Expr

    def effect(self) -> Effect:
        """Give the model effect this node applies.

        Returns:
            The effect whose `apply` is this node's meaning.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Suffix(Ref("R"), "_p").effect()
            SuffixedBy(suffix='_p')
        """
        raise NotImplementedError

    def fold(self, form: Canonical) -> Canonical:
        """Fold this effect into an inner canonical form.

        Args:
            form: The canonical form of the inner expression.

        Returns:
            The canonical form of this whole expression.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Suffix(Ref("R"), "_p").fold(Canonical.of(Variable("R"))).suffix
            '_p'
        """
        raise NotImplementedError

    def denote(self, bindings: Mapping[str, ColumnSet]) -> ColumnSet:
        """Apply this node's effect to what the inner expression denotes.

        Args:
            bindings: A column set for each variable.

        Returns:
            The column set the effect produces.

        Example:
            >>> Suffix(Ref("R"), "_p").denote({"R": ColumnSet.of(a="Int64")}).names()
            ('a_p',)
        """
        return self.effect().apply(self.inner.denote(bindings))

    def normalise(self) -> Canonical:
        """Fold this effect into the inner expression's canonical form.

        Every effect acts on each column separately, so it distributes
        over a union: the effect folds into each part, and the parts
        unite again. A union form therefore carries no effect.

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
        """Fold this effect into one form that is no union, and simplify.

        Args:
            form: A canonical form whose base is a variable or empty.

        Returns:
            The simplified canonical form of this effect over it.

        Example:
            >>> Suffix(Ref("R"), "_p")._folded(Canonical.of(Variable("R"))).suffix
            '_p'
        """
        return Simplified()(self.fold(form))


@dataclass(frozen=True)
class Keep(Applied):
    """The remainder passes through untouched.

    Example:
        >>> Keep(Ref("R")).normalise() == Ref("R").normalise()
        True
    """

    def effect(self) -> Effect:
        """Give the identity effect.

        Returns:
            The effect that changes nothing.

        Example:
            >>> Keep(Ref("R")).effect()
            Unchanged()
        """
        return Unchanged()

    def fold(self, form: Canonical) -> Canonical:
        """Return the inner form unchanged.

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
    """The remainder loses every column.

    Example:
        >>> Empty(Ref("R")).normalise().base
        Nothing()
    """

    def effect(self) -> Effect:
        """Give the emptying effect.

        Returns:
            The effect that keeps no column.

        Example:
            >>> Empty(Ref("R")).effect()
            Emptied()
        """
        return Emptied()

    def fold(self, form: Canonical) -> Canonical:
        """Discard the inner form entirely.

        Nothing an inner expression did can matter once every column is
        gone, so an emptying effect absorbs everything below it.

        Args:
            form: Ignored.

        Returns:
            The canonical form of no columns at all.

        Example:
            >>> Empty(Ref("R")).fold(Canonical(Variable("R"), suffix="_p")).suffix
            ''
        """
        del form
        return Canonical.of(Nothing())


@dataclass(frozen=True)
class Retype(Applied):
    """Every column keeps its name and takes one new data type.

    Example:
        >>> Retype(Ref("R"), "UInt32").normalise().retype
        'UInt32'
    """

    dtype: str

    def effect(self) -> Effect:
        """Give the retyping effect.

        Returns:
            The effect that replaces every data type.

        Example:
            >>> Retype(Ref("R"), "UInt32").effect()
            Retyped(dtype='UInt32')
        """
        return Retyped(self.dtype)

    def fold(self, form: Canonical) -> Canonical:
        """Replace whatever data type the inner form set.

        A later replacement overwrites an earlier one, so only the
        outermost survives.

        Args:
            form: The canonical form of the inner expression.

        Returns:
            The same form with this data type.

        Example:
            >>> inner = Retype(Ref("R"), "Int64")
            >>> Retype(inner, "UInt32").normalise().retype
            'UInt32'
        """
        return replace(form, retype=self.dtype)


@dataclass(frozen=True)
class Suffix(Applied):
    """Every column name gains a fixed suffix.

    Example:
        >>> Suffix(Ref("R"), "_p").normalise().suffix
        '_p'
    """

    affix: str

    def effect(self) -> Effect:
        """Give the suffixing effect.

        Returns:
            The effect that appends to every name.

        Example:
            >>> Suffix(Ref("R"), "_p").effect()
            SuffixedBy(suffix='_p')
        """
        return SuffixedBy(self.affix)

    def fold(self, form: Canonical) -> Canonical:
        """Append to whatever suffix the inner form already adds.

        Two suffixes in a row are one suffix, and a prefix outside a
        suffix does not interact with it, so this slot absorbs the
        affix wherever it came from.

        Args:
            form: The canonical form of the inner expression.

        Returns:
            The same form with the lengthened suffix.

        Example:
            >>> Suffix(Suffix(Ref("R"), "_a"), "_b").normalise().suffix
            '_a_b'
        """
        return replace(form, suffix=form.suffix + self.affix)


@dataclass(frozen=True)
class Prefix(Applied):
    """Every column name gains a fixed prefix.

    Example:
        >>> Prefix(Ref("R"), "s_").normalise().prefix
        's_'
    """

    affix: str

    def effect(self) -> Effect:
        """Give the prefixing effect.

        Returns:
            The effect that prepends to every name.

        Example:
            >>> Prefix(Ref("R"), "s_").effect()
            PrefixedBy(prefix='s_')
        """
        return PrefixedBy(self.affix)

    def fold(self, form: Canonical) -> Canonical:
        """Prepend to whatever prefix the inner form already adds.

        Args:
            form: The canonical form of the inner expression.

        Returns:
            The same form with the lengthened prefix.

        Example:
            >>> Prefix(Prefix(Ref("R"), "b_"), "a_").normalise().prefix
            'a_b_'
        """
        return replace(form, prefix=self.affix + form.prefix)


@dataclass(frozen=True)
class Filter(Applied):
    """Only the columns a selector picks survive.

    Folding this one is the only case that computes rather than copies.
    The selector runs after whatever the inner form already did, so it
    tests names the renames have already changed and data types a
    replacement may have already set. It must be translated to test the
    base column instead.

    Example:
        >>> from spec.tests.model import NameTest, StartsWith
        >>> Filter(Ref("R"), NameTest(StartsWith("a"))).normalise().where is None
        False
    """

    selector: Selector

    def effect(self) -> Effect:
        """Give the filtering effect.

        Returns:
            The effect that keeps the picked columns.

        Example:
            >>> from spec.tests.model import OfType, NUMERIC
            >>> Filter(Ref("R"), OfType(NUMERIC)).effect()
            Where(selector=OfType(family=NumericType))
        """
        return Where(self.selector)

    def fold(self, form: Canonical) -> Canonical:
        """Translate the selector inward and conjoin it with the inner filter.

        The translation rewrites the selector to test the base column
        rather than the result. Its three steps undo the three outer
        slots, outermost first: a data type replacement settles every
        type test, then the prefix and the suffix each restore the part
        of the name they added.

        Two filters in a row keep a column only when both accept it, so
        one filter with the conjunction says the same thing. When one
        of the two is contained in the other, the conjunction is the
        narrower one alone, and `Conjunction` drops the wider.

        Args:
            form: The canonical form of the inner expression.

        Returns:
            The same form with one filter covering both tests.

        Example:
            >>> from spec.tests.model import NameTest, StartsWith
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
        translated = selector.through_prefix(form.prefix).through_suffix(form.suffix)
        if form.where is None:
            return replace(form, where=translated)
        return replace(form, where=Conjunction()(form.where, translated))


class Simplified:
    """Apply the identity rules a canonical form must not record.

    Three effects do nothing, and a form that recorded them would
    distinguish two types that are the same. A filter picking every
    column is the identity. A filter picking none empties the
    remainder. Once a remainder is empty, no later effect can change
    it, so every other slot is inert.

    Each rule is decided by `SelectorEntailment`, which is exact for
    the constructors a user can write.

    Example:
        >>> from spec.tests.model import Always, NameTest
        >>> form = Canonical(Variable("R"), where=NameTest(Always()))
        >>> Simplified()(form).where is None
        True
    """

    def __call__(self, form: Canonical) -> Canonical:
        """Give the simplest form denoting the same thing.

        A filter that provably picks no column empties the form. A
        filter that provably picks every column is dropped. Any other
        filter is kept in its canonical spelling.

        Args:
            form: The form a fold produced.

        Returns:
            That form with every inert slot removed.

        Example:
            >>> Simplified()(Canonical.of(Variable("R"))).base
            Variable(name='R')
            >>> from spec.tests.model import NameTest, Never, StartsWith
            >>> f = Canonical(Variable("R"), where=NameTest(Never()))
            >>> Simplified()(f).base
            Nothing()
            >>> f = Canonical(Variable("R"), where=NameTest(StartsWith("a")))
            >>> Simplified()(f).where is None
            False
        """
        if isinstance(form.base, Nothing):
            return Canonical.of(Nothing())
        if form.where is None:
            return form
        entails = SelectorEntailment()
        if entails(form.where, NameTest(Never())):
            return Canonical.of(Nothing())
        if entails(NameTest(Always()), form.where):
            return replace(form, where=None)
        return replace(form, where=form.where.canonical())


class ExpressionSpace:
    """Every expression up to a nesting depth, over a fixed set of variables.

    The space is what makes an exhaustive check meaningful. It is
    bounded, so it can be enumerated, and it holds at least one
    expression per effect and per pairing of two effects.
    """

    def __init__(
        self,
        variables: tuple[str, ...],
        selectors: tuple[Selector, ...],
        unions_over_effects: bool = False,
    ) -> None:
        """Fix the variables and selectors every expression is built from.

        Args:
            variables: The remainder names expressions may reference.
            selectors: The selectors a filter may use.
            unions_over_effects: Whether the space also holds each union
                of two expressions that apply at most one effect to a
                variable, so that a union can sit above an effect.
        """
        self.variables = variables
        self.selectors = selectors
        self.unions_over_effects = unions_over_effects

    def wrappings(self, inner: Expr) -> Iterator[Expr]:
        """Give every one-step extension of an expression.

        Args:
            inner: The expression to wrap.

        Yields:
            One expression per effect this module models.
        """
        yield Keep(inner)
        yield Empty(inner)
        yield Retype(inner, "UInt32")
        yield Suffix(inner, "b_")
        yield Prefix(inner, "a")
        for selector in self.selectors:
            yield Filter(inner, selector)

    def next_layer(self, layer: Sequence[Expr]) -> list[Expr]:
        """Wrap every expression of one layer in every effect.

        Args:
            layer: The expressions to extend.

        Returns:
            Every one-step extension of every one of them.
        """
        extended: list[Expr] = []
        for expression in layer:
            extended.extend(self.wrappings(expression))
        return extended

    def up_to_depth(self, depth: int) -> list[Expr]:
        """List every expression applying at most `depth` effects.

        The starting layer holds each variable and each union of two of
        them, so a union appears both bare and under every effect. That
        matters: an effect applied to a union folds into its parts, and
        a union nested in another flattens into it.

        Args:
            depth: How many effects may be applied, at most.

        Returns:
            Each expression exactly once, shallowest first.
        """
        layer: list[Expr] = self.starting_layer()
        found = list(layer)
        for _ in range(depth):
            layer = self.next_layer(layer)
            found.extend(layer)
        return found + self.unions_of_effects()

    def unions_of_effects(self) -> list[Expr]:
        """List each union of two expressions that apply at most one effect.

        Returns:
            The unions, when the space holds unions over effects, else
            none.
        """
        if not self.unions_over_effects:
            return []
        refs: list[Expr] = [Ref(name) for name in self.variables]
        shallow = refs + self.next_layer(refs)
        return list(starmap(Combined, product(shallow, repeat=2)))

    def starting_layer(self) -> list[Expr]:
        """Give the expressions every other one is built from.

        Returns:
            Each variable, plus each union of two variables.
        """
        refs: list[Expr] = [Ref(name) for name in self.variables]
        pairs: list[Expr] = list(starmap(Combined, product(refs, refs)))
        return refs + pairs

    def bindings(self, universe: Universe) -> list[Mapping[str, ColumnSet]]:
        """List every assignment of the universe's column sets to the variables.

        Args:
            universe: The column sets a variable may bind to.

        Returns:
            One binding per combination.
        """
        sets = list(universe.column_sets())
        combinations = product(sets, repeat=len(self.variables))
        return [dict(zip(self.variables, chosen, strict=True)) for chosen in combinations]
