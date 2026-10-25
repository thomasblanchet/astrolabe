"""Sets of column names as finite automata, so that two name rules can be compared exactly.

The checker proves facts about the names of a remainder, that is,
the columns of a frame that its type does not name. A rule such as
"every name that begins with `tmp_`" is a set of names with no end.
Code must still decide whether one such set is inside another, for
example whether what a call proves covers what a declaration
demands. Every name rule here gives a regular set: a set that a
finite automaton decides. For regular sets, that question always
has an exact answer.

The `greenery` library builds and compares the automata. `NameSet`
is the only class that touches it, and every other module works with
`NameSet` values. This module also reads a polars regular
expression as a name set (`PatternReader`), writes an automaton as
text and reads it back (`MachineSignature`, `SignatureParser`), and
writes a small automaton as a regular expression (`MachineExpression`).
"""

from __future__ import annotations

import re
from functools import reduce
from itertools import pairwise
from typing import TYPE_CHECKING, ClassVar

from greenery import Charclass, Fsm, parse, rxelems
from greenery.parse import NoMatch

from astrolabe._hooks.propagate import not_falsy, not_none, propagates

if TYPE_CHECKING:
    from collections.abc import Iterable


class NameSet:
    """A set of column names that a finite automaton decides, with exact set operations.

    It wraps a `greenery` automaton and keeps it private. Code that
    compares name rules builds these values, joins them with union,
    intersection, complement and concatenation, and then asks whether
    one set is inside another. Each answer is exact, never a guess.
    Two sets with the same names give the same `signature`, which is
    how a type spells a set that has no simpler shape.

    A set never changes after it is built, so the sets of every name
    and of no name are built once and shared.

    Example:
        >>> NameSet.literal("ab").contained_in(NameSet.anything())
        True
    """

    # The automata of every name and of no name, built at the first use.
    _CONSTANTS: ClassVar[dict[str, Fsm]] = {}
    # The spelling of each signature, built at the first use.
    _SPELLINGS: ClassVar[dict[str, str]] = {}

    def __init__(self, machine: Fsm) -> None:
        """Keep one automaton.

        Args:
            machine: The `greenery` automaton that decides which names
                are in the set.
        """
        self._machine = machine

    @classmethod
    def from_signature(cls, text: str) -> NameSet:
        """Rebuild the set from the text that `signature` gave.

        A type holds a name set as its signature text alone. When mypy
        loads a frame type from its incremental cache, only that text
        is left, so the automaton must come back from it. The text
        lists every state and every transition, which is the whole
        automaton.

        Args:
            text: A text that `signature` gave.

        Returns:
            The set that the text describes.

        Example:
            >>> prefixed = NameSet.literal("ab").concatenate(NameSet.anything())
            >>> rebuilt = NameSet.from_signature(prefixed.signature())
            >>> rebuilt.accepts("abc"), rebuilt.accepts("ax")
            (True, False)
            >>> rebuilt.signature() == prefixed.signature()
            True
        """
        return cls(SignatureParser(text).machine())

    @classmethod
    def anything(cls) -> NameSet:
        """Build the set of every name.

        Returns:
            The set that holds every name, the empty name included.

        Example:
            >>> NameSet.anything().accepts("whatever")
            True
        """
        return cls(cls._constant(".*"))

    @classmethod
    def nothing(cls) -> NameSet:
        """Build the set of no name.

        Returns:
            The empty set.

        Example:
            >>> NameSet.nothing().accepts("")
            False
        """
        return cls(cls._constant("[]"))

    @classmethod
    def _constant(cls, pattern: str) -> Fsm:
        """Give the automaton of one fixed pattern, built once in each process.

        Args:
            pattern: The pattern, `.*` or `[]`.

        Returns:
            The automaton.

        Example:
            >>> NameSet._constant("[]") is NameSet._constant("[]")
            True
        """
        if pattern not in cls._CONSTANTS:
            cls._CONSTANTS[pattern] = parse(pattern).to_fsm()
        return cls._CONSTANTS[pattern]

    @classmethod
    def literal(cls, text: str) -> NameSet:
        """Build the set of exactly one name.

        Each character becomes its own character class, so a
        character such as `.` or `*` means only itself.

        Args:
            text: The name.

        Returns:
            The set that holds only that name.

        Example:
            >>> NameSet.literal("a.b").accepts("a.b")
            True
            >>> NameSet.literal("a.b").accepts("axb")
            False
        """
        return cls.of_names([text])

    @classmethod
    def of_names(cls, names: Iterable[str]) -> NameSet:
        """Build the set of exactly the given names.

        Args:
            names: The names.

        Returns:
            The set that holds those names and no other.

        Example:
            >>> both = NameSet.of_names(["ab", "a"])
            >>> both.accepts("a"), both.accepts("ab"), both.accepts("b")
            (True, True, False)
        """
        return cls(NameTrie(names).machine())

    def accepts(self, name: str) -> bool:
        """Answer whether the set holds one name.

        Args:
            name: The name to test.

        Returns:
            True if the automaton accepts the name.

        Example:
            >>> NameSet.literal("a").accepts("b")
            False
        """
        return bool(self._machine.accepts(name))

    def union(self, other: NameSet) -> NameSet:
        """Build the set of the names that either set holds.

        Args:
            other: The other set.

        Returns:
            The union.

        Example:
            >>> NameSet.literal("a").union(NameSet.literal("b")).accepts("b")
            True
        """
        return NameSet(self._machine.union(other._machine))

    def intersection(self, other: NameSet) -> NameSet:
        """Build the set of the names that both sets hold.

        Args:
            other: The other set.

        Returns:
            The intersection.

        Example:
            >>> both = NameSet.anything().intersection(NameSet.literal("a"))
            >>> both.accepts("a"), both.accepts("b")
            (True, False)
        """
        return NameSet(self._machine.intersection(other._machine))

    def complement(self) -> NameSet:
        """Build the set of every name that this set does not hold.

        Returns:
            The complement.

        Example:
            >>> NameSet.nothing().complement().accepts("x")
            True
        """
        return NameSet(self._machine.everythingbut())

    def concatenate(self, other: NameSet) -> NameSet:
        """Build the set of each name of this set followed by a name of the other set.

        A prefix rule is the prefix followed by any name, and a
        suffix rule is any name followed by the suffix.

        Args:
            other: The set of the second halves.

        Returns:
            The concatenation.

        Example:
            >>> ab = NameSet.literal("a").concatenate(NameSet.literal("b"))
            >>> ab.accepts("ab")
            True
        """
        return NameSet(self._machine.concatenate(other._machine))

    def contained_in(self, other: NameSet) -> bool:
        """Decide whether the other set holds every name of this set.

        This answer is exact. A False means that some name of this set
        is not in the other, not that the question had no answer.

        Args:
            other: The set that must hold at least as much.

        Returns:
            True if the containment holds.

        Example:
            >>> NameSet.literal("ab").contained_in(NameSet.literal("ab"))
            True
        """
        return bool(self._machine.difference(other._machine).empty())

    def empty(self) -> bool:
        """Answer whether the set holds no name.

        Code asks this of the complement of a rule, for example, to
        know whether the rule holds of every name.

        Returns:
            True if the set holds no name.

        Example:
            >>> NameSet.nothing().empty()
            True
        """
        return bool(self._machine.empty())

    def after_appending(self, suffix: str) -> NameSet:
        """Give the names that are in this set once a suffix is added to them.

        A filter that moves inside a rename needs this. After the
        rename, it tests `x_prev`. Before the rename, it must test
        `x`. An automaton reads a name from its start. So the method
        reverses the automaton, reads the reversed suffix, and
        reverses the result back.

        Args:
            suffix: The text that the rename adds.

        Returns:
            The set of each name that is in this set with the suffix
            added.

        Example:
            >>> prefixed = NameSet.literal("ab").concatenate(NameSet.anything())
            >>> shifted = prefixed.after_appending("b_")
            >>> shifted.accepts("a"), shifted.accepts("ax")
            (True, False)
        """
        derived = self._derived(self._machine.reversed(), suffix[::-1])
        return NameSet(derived.reversed())

    def after_prepending(self, prefix: str) -> NameSet:
        """Give the names that are in this set once a prefix is added to them.

        Args:
            prefix: The text that the rename adds.

        Returns:
            The set of each name that is in this set with the prefix
            added.

        Example:
            >>> starts = NameSet.literal("a").concatenate(NameSet.anything())
            >>> starts.after_prepending("a").accepts("anything")
            True
        """
        return NameSet(self._derived(self._machine, prefix))

    def _derived(self, machine: Fsm, text: str) -> Fsm:
        """Give the automaton that remains after it reads one text.

        Args:
            machine: The automaton that reads the text.
            text: The text to read first.

        Returns:
            The automaton that decides the rest of each name that
            begins with the text.

        Example:
            >>> one = NameSet.literal("ab")
            >>> one._derived(one._machine, "a").accepts("b")
            True
        """
        for character in text:
            machine = machine.derive(character)
        return machine

    def finite_names(self) -> frozenset[str] | None:
        """List the names of the set, if the set is finite and small.

        A finite set has a simple spelling in a type: a list of names,
        or the negation of a list. Code that spells a set asks this
        first, and uses the signature only if there is no list.

        Returns:
            Every name of the set, or None for a set with no end or
            one of more than 128 names. A wildcard such as `.` is
            finite over the real characters, but it is no list of
            names, and the limit keeps it out.

        Example:
            >>> two = NameSet.literal("a").union(NameSet.literal("bc"))
            >>> sorted(two.finite_names())
            ['a', 'bc']
            >>> NameSet.anything().finite_names() is None
            True
            >>> wide = NameSet.nothing()
            >>> for i in range(129):
            ...     wide = wide.union(NameSet.literal(f"c{i}"))
            >>> wide.finite_names() is None
            True
        """
        machine = self._machine.reduce()
        try:
            if machine.cardinality() > 128:
                return None
        except OverflowError:
            return None
        return frozenset(machine.strings(otherchars=[]))

    def signature(self) -> str:
        """Give a text that is equal for two sets exactly if they hold the same names.

        A type spells a `Matching` rule by this text, so two rules that
        hold the same names give one type. The text comes from the
        minimal automaton, whose states get numbers in the order of a
        breadth-first walk. Characters that every state treats alike
        form one group. So neither the order in which a rule was built
        nor the characters it names can change the text.

        Returns:
            The text of the minimal automaton.

        Example:
            >>> a = NameSet.literal("a").concatenate(NameSet.anything())
            >>> ab = NameSet.literal("ab").concatenate(NameSet.anything())
            >>> a.union(ab).signature() == a.signature()
            True
        """
        return MachineSignature(self._machine.reduce()).text()

    def spelling(self) -> str:
        """Give the text that a type holds for the set: a regular expression where one reads well.

        A regular expression is easier to read than a signature, and
        `MachineExpression` writes one for the common sets, such as the
        names that begin with a prefix: `ab.*`. It builds the expression
        from the automaton of the signature, so two sets with the same
        names give one text. Every other set keeps its signature. An
        expression that reads as a signature also keeps the signature,
        so that `from_spelling` reads each text in one way.

        Returns:
            The expression or the signature.

        Example:
            >>> NameSet.literal("ab").concatenate(NameSet.anything()).spelling()
            'ab.*'
            >>> NameSet.anything().concatenate(NameSet.literal("z")).spelling()
            '0[[^z]>0;z>1]|1F[[^z]>0;z>1]'
        """
        signature = self.signature()
        if signature not in self._SPELLINGS:
            expression = MachineExpression(signature).text()
            spelled = expression is not None and self._spells(expression, signature)
            self._SPELLINGS[signature] = expression if spelled and expression else signature
        return self._SPELLINGS[signature]

    def _spells(self, expression: str, signature: str) -> bool:
        """Say whether an expression reads back as the set, and not as a signature.

        Args:
            expression: The regular expression.
            signature: The signature of the set.

        Returns:
            Whether the expression reads back as the same names, and does
            not read as a signature.

        Example:
            >>> starts = NameSet.literal("a").concatenate(NameSet.anything())
            >>> starts._spells("a.*", starts.signature())
            True
        """
        back = NameSet(parse(expression).to_fsm()).signature() == signature
        return back and self._signed(expression) is None

    @classmethod
    def from_spelling(cls, text: str) -> NameSet:
        """Rebuild the set from the text that `spelling` gave, or from its signature.

        A type loaded from the incremental cache of mypy keeps only this
        text. A text that `spelling` does not give for its names, for
        example a regular expression that a user writes in another way,
        is not one.

        Args:
            text: The text.

        Returns:
            The set that the text spells.

        Raises:
            ValueError: The text is neither a signature nor the expression
                that `spelling` gives.

        Example:
            >>> NameSet.from_spelling("ab.*").accepts("abc")
            True
            >>> NameSet.from_spelling("0F[.>0]").accepts("x")
            True
            >>> NameSet.from_spelling("[a")
            Traceback (most recent call last):
            ...
            ValueError: not the spelling of a set of names: '[a'
            >>> NameSet.from_spelling("a(b)?.*")
            Traceback (most recent call last):
            ...
            ValueError: not the spelling of a set of names: 'a(b)?.*'
        """
        signed = cls._signed(text)
        if signed is not None:
            return signed
        problem = f"not the spelling of a set of names: {text!r}"
        try:
            names = cls(parse(text).to_fsm())
        except (NoMatch, IndexError, ValueError) as error:
            raise ValueError(problem) from error
        if names.spelling() != text:
            raise ValueError(problem)
        return names

    @classmethod
    def _signed(cls, text: str) -> NameSet | None:
        """Read a text as the signature of a set, where it is one.

        Args:
            text: The text.

        Returns:
            The set, or `None` where the text is not the signature of its set.

        Example:
            >>> NameSet._signed("0F[.>0]").accepts("x")
            True
            >>> NameSet._signed("ab.*") is None
            True
        """
        try:
            names = cls.from_signature(text)
        except (ValueError, IndexError):
            return None
        return names if names.signature() == text else None


class NameTrie:
    """Build the automaton of a finite list of names, as the tree of their prefixes.

    A rule that lists names, for example `other_columns(without=["a",
    "b"])`, needs the set of those names as an automaton. To parse one
    pattern for each name and join the results costs a full automaton
    construction for each name. The tree of prefixes is already a
    deterministic automaton: each state is one prefix, and each name
    ends in an accepting state. So this class writes the states and the
    moves directly. Each character that a name uses gets its own
    character class, one more class holds every other character, and a
    dead state takes every move that leaves the tree. One instance
    builds one automaton.

    Example:
        >>> machine = NameTrie(["a", "ab"]).machine()
        >>> machine.accepts("ab"), machine.accepts("b")
        (True, False)
    """

    def __init__(self, names: Iterable[str]) -> None:
        """Keep the names, and give each character that they use its class.

        Args:
            names: The names.
        """
        self._names = sorted(set(names))
        characters = sorted({char for name in self._names for char in name})
        self._symbols = {char: Charclass(((char, char),)) for char in characters}
        self._rest = Charclass(tuple((char, char) for char in characters)).negate()
        self._moves: list[dict[str, int]] = [{}]
        self._finals: set[int] = set()

    def machine(self) -> Fsm:
        """Build the automaton.

        The first pass walks each name from the root of the tree. It
        adds a state for each prefix that is new, and marks the state
        where the name ends as accepting. The second pass writes every
        move of each state. A character that leaves the tree goes to
        the dead state.

        Returns:
            The automaton that accepts exactly the names.

        Example:
            >>> NameTrie([]).machine().accepts("")
            False
            >>> NameTrie(["ab"]).machine().accepts("a")
            False
        """
        for name in self._names:
            state = 0
            for character in name:
                moves = self._moves[state]
                if character not in moves:
                    moves[character] = len(self._moves)
                    self._moves.append({})
                state = moves[character]
            self._finals.add(state)
        dead = len(self._moves)
        table: dict[int, dict[Charclass, int]] = {}
        for state, moves in enumerate(self._moves):
            symbols = self._symbols.items()
            row = {symbol: moves.get(character, dead) for character, symbol in symbols}
            row[self._rest] = dead
            table[state] = row
        table[dead] = dict.fromkeys([*self._symbols.values(), self._rest], dead)
        return Fsm(
            alphabet=[*self._symbols.values(), self._rest],
            states=range(dead + 1),
            initial=0,
            finals=self._finals,
            map=table,
        )


class PatternReader:
    r"""Read the regular expression of `cs.matches(...)` as the set of names it picks.

    The checker must know which undeclared columns such a selector
    picks, so it needs the pattern as a name set. Polars searches each
    column name for the pattern. So a branch with no `^` picks a name
    that holds a match at any position. The reader takes only the
    part of the syntax that means the same in polars and in
    `greenery`: literal characters, escaped punctuation, `.`,
    character classes, groups, alternation and repeats. Polars reads a
    `]` outside a class as the character itself, and `greenery` needs it
    escaped, so the reader escapes it. `^` and `$`
    can stand only at the two ends of a top-level branch. For any
    other construct the reader gives None, because polars can read it
    in another way. For example, `\d` in polars holds every Unicode
    digit.

    A search can build an automaton whose size grows exponentially
    with the pattern. So the reader also gives None for a pattern of
    more than `MOST_TOKENS` tokens, for a counted repeat such as
    `a{3}`, and for more than `MOST_FIXED` single-character wildcards
    without a `*` or a `+` after them. Code that gets None keeps the
    selector plain.

    Example:
        >>> names = PatternReader().language("^px_")
        >>> names.accepts("px_close"), names.accepts("a_px_close")
        (True, False)
        >>> PatternReader().language(r"^\d$") is None
        True
    """

    TOKEN: ClassVar[re.Pattern[str]] = re.compile(r"\\.|\[\^?\]?(?:\\.|[^\]\\])*\]|.", re.DOTALL)
    ESCAPE: ClassVar[re.Pattern[str]] = re.compile(r"\\.", re.DOTALL)
    UNSAFE_ESCAPE: ClassVar[re.Pattern[str]] = re.compile(r"\\\w")
    UNSAFE_CLASS: ClassVar[re.Pattern[str]] = re.compile(r"^\[\^?\]|.\[|&&|--|~~")
    DEPTH: ClassVar[dict[str, int]] = {"(": 1, ")": -1}
    ANCHORS: ClassVar[frozenset[str]] = frozenset({"^", "$"})
    TRANSLATED: ClassVar[dict[str, str]] = {".": "[^\\n]"}
    ESCAPED: ClassVar[dict[str, str]] = {"]": "\\]"}
    REPEATED: ClassVar[frozenset[str]] = frozenset({"*", "+"})
    MOST_TOKENS: ClassVar[int] = 32
    MOST_FIXED: ClassVar[int] = 5

    @propagates(None)
    def language(self, pattern: str) -> NameSet | None:
        r"""Give the set of names that the pattern picks.

        Args:
            pattern: The regular expression.

        Returns:
            The name set, or None for a pattern that the reader does
            not take.

        Example:
            >>> PatternReader().language("^a|b$").accepts("xb")
            True
            >>> PatternReader().language("(?i)a") is None
            True
            >>> PatternReader().language(r"\[0]$").accepts("x[0]")
            True
        """
        tokens = self.TOKEN.findall(pattern)
        not_falsy(self._safe(tokens))
        branches = [not_none(self._branch(branch)) for branch in self._branches(tokens)]
        return reduce(NameSet.union, branches)

    def _safe(self, tokens: list[str]) -> bool:
        """Answer whether polars and `greenery` read the whole pattern the same way.

        Args:
            tokens: The tokens of the pattern.

        Returns:
            False for a flag group such as `(?i)`, an escaped letter
            or digit, a character class that the two can read in
            different ways, or a pattern that is too large.

        Example:
            >>> PatternReader()._safe(["(", "?", "i", ")"])
            False
        """
        flagged = any(pair == ("(", "?") for pair in pairwise(tokens))
        return not flagged and self._small(tokens) and all(map(self._safe_token, tokens))

    def _small(self, tokens: list[str]) -> bool:
        """Answer whether the automaton of the pattern stays small.

        A fixed wildcard is a `.` or a character class with no `*` or
        `+` after it. Each one can multiply the size of the automaton,
        so the reader counts them.

        Args:
            tokens: The tokens of the pattern.

        Returns:
            False for too many tokens, a counted repeat, or too many
            fixed wildcards.

        Example:
            >>> PatternReader()._small(list("a.*b")), PatternReader()._small(list("a{3}"))
            (True, False)
            >>> PatternReader()._small(list("......")), PatternReader()._small(["[ab]", "*"] * 6)
            (False, True)
            >>> PatternReader()._small([])
            True
        """
        fixed = 0
        for token, following in zip(tokens, [*tokens[1:], ""], strict=False):
            wildcard = token in self.TRANSLATED or (token.startswith("[") and len(token) > 1)
            if wildcard and following not in self.REPEATED:
                fixed += 1
        short = len(tokens) <= self.MOST_TOKENS
        return short and "{" not in tokens and fixed <= self.MOST_FIXED

    def _safe_token(self, token: str) -> bool:
        r"""Answer whether polars and `greenery` read one token the same way.

        Args:
            token: One token.

        Returns:
            False for an escaped letter or digit, inside or outside a
            character class. Also False for a nested or named class, a
            set operation such as `&&`, or a `]` first in the class.

        Example:
            >>> PatternReader()._safe_token("[a-z]"), PatternReader()._safe_token(r"\w")
            (True, False)
            >>> PatternReader()._safe_token("[[:alpha:]]"), PatternReader()._safe_token(r"[\d]")
            (False, False)
        """
        if not token.startswith("[") or len(token) == 1:
            return self.UNSAFE_ESCAPE.fullmatch(token) is None
        escapes = self.ESCAPE.findall(token)
        unsafe = any(self.UNSAFE_ESCAPE.fullmatch(escape) for escape in escapes)
        return not unsafe and self.UNSAFE_CLASS.search(token) is None

    def _branches(self, tokens: list[str]) -> list[list[str]]:
        """Split the tokens at each `|` that is outside every group.

        Each top-level branch gets its own anchors, so the reader
        reads each one alone.

        Args:
            tokens: The tokens of the pattern.

        Returns:
            The tokens of each top-level branch.

        Example:
            >>> PatternReader()._branches(["(", "a", "|", "b", ")", "|", "c"])
            [['(', 'a', '|', 'b', ')'], ['c']]
        """
        branches: list[list[str]] = [[]]
        depth = 0
        for token in tokens:
            depth += self.DEPTH.get(token, 0)
            if depth == 0 and token == "|":  # ruff: ignore[hardcoded-password-string] - a regex token, not a password
                branches.append([])
            else:
                branches[-1].append(token)
        return branches

    @propagates(None)
    def _branch(self, tokens: list[str]) -> NameSet | None:
        """Give the set of names that one top-level branch picks.

        An end with no anchor allows any text on that side of the
        match, because polars searches the whole name. `greenery`
        raises `IndexError` for a character class that the text does
        not close, for example `a[`. So that error counts as a text
        that it cannot parse too.

        Args:
            tokens: The tokens of the branch.

        Returns:
            The name set, or None for an anchor that is not at an end
            of the branch, or for a body that `greenery` cannot parse.

        Example:
            >>> PatternReader()._branch(["^", "a", "$"]).finite_names()
            frozenset({'a'})
            >>> PatternReader()._branch(["a", "*", "*"]) is None
            True
            >>> PatternReader()._branch(["a", "["]) is None
            True
        """
        start = tokens[:1] == ["^"]
        end = tokens[-1:] == ["$"] and len(tokens) > start
        body = tokens[int(start) : len(tokens) - int(end)]
        not_falsy(self.ANCHORS.isdisjoint(body))
        written = {**self.TRANSLATED, **self.ESCAPED}
        text = "".join(written.get(token, token) for token in body)
        try:
            found = NameSet(parse(text).to_fsm())
        except (NoMatch, ValueError, IndexError):
            return None
        return self._edge(anchored=start).concatenate(found).concatenate(self._edge(anchored=end))

    def _edge(self, *, anchored: bool) -> NameSet:
        """Give the text that a branch allows beside its match, at one end.

        Args:
            anchored: True if the branch has an anchor at that end.

        Returns:
            Only the empty text for an end with an anchor, and any
            text for an end with no anchor.

        Example:
            >>> PatternReader()._edge(anchored=True).finite_names()
            frozenset({''})
        """
        return NameSet.literal("") if anchored else NameSet.anything()


class MachineSignature:
    """Write a minimal automaton as a text that depends only on the names it accepts.

    A type spells a name set with no simpler shape by this text, so
    two automata that accept the same names must give the same text.
    The constructor numbers the states in the order of a
    breadth-first walk and puts alike characters in one group, once.
    `text` then writes one line for each state. Only `NameSet` uses
    this class, because no other module holds an automaton.

    Example:
        >>> MachineSignature(parse("[]").to_fsm().reduce()).text()
        '0[.>0]'
    """

    def __init__(self, machine: Fsm) -> None:
        """Number the states and group the characters, once.

        Args:
            machine: A minimal automaton.
        """
        self.machine = machine
        self.order = self._numbered()
        self.states = sorted(self.order, key=lambda s: self.order[s])
        self.behaviour = self._grouped()

    def _numbered(self) -> dict[object, int]:
        """Number every state that the start can reach, in the order of a breadth-first walk.

        The walk follows the transitions of each state sorted by the
        text of their character class. So the numbers do not depend on
        how `greenery` named the states.

        Returns:
            Each state, mapped to its number.

        Example:
            >>> len(MachineSignature(parse("a").to_fsm().reduce())._numbered()) >= 2
            True
        """
        order, queue = {self.machine.initial: 0}, [self.machine.initial]
        while queue:
            row = self.machine.map[queue.pop(0)]
            for symbol in sorted(row, key=str):
                target = row[symbol]
                if target not in order:
                    order[target] = len(order)
                    queue.append(target)
        return order

    def _grouped(self) -> dict[tuple[int, ...], str]:
        """Put in one group the characters that every state treats alike.

        Two automata can accept the same names and split the
        characters in different ways. One rule can name `a` and `b`
        and another only `a`, although both treat `a` and `b` alike.
        One group for each behavior removes that difference. The
        behavior of a character class is the number of its target from
        each state, in order. A group is written as the union of its
        classes, in its simplest form.

        Returns:
            Each behavior, mapped to the text of the character class
            of its group.

        Example:
            >>> len(MachineSignature(parse(".*").to_fsm().reduce())._grouped())
            1
        """
        rows: dict[tuple[int, ...], list[Charclass]] = {}
        for symbol in self.machine.alphabet:
            key = tuple(self.order[self.machine.map[state][symbol]] for state in self.states)
            rows.setdefault(key, []).append(symbol)
        grouped: dict[tuple[int, ...], str] = {}
        for key, symbols in rows.items():
            union = reduce(lambda a, b: a.union(b), symbols)
            grouped[key] = str(union.reduce())
        return grouped

    def text(self) -> str:
        """Give the text of the automaton.

        Each state gives one entry: its number, `F` if it accepts, and
        its transitions in brackets. Each transition is a character
        class, `>` and a target.

        Returns:
            One entry for each state, in the order of the walk,
            joined by `|`.

        Example:
            >>> MachineSignature(parse(".*").to_fsm().reduce()).text()
            '0F[.>0]'
        """
        keys = sorted(self.behaviour, key=lambda k: self.behaviour[k])
        lines: list[str] = []
        for state in self.states:
            number = self.order[state]
            moves = ";".join(f"{self.behaviour[k]}>{k[number]}" for k in keys)
            accepts = "F" if state in self.machine.finals else ""
            lines.append(f"{number}{accepts}[{moves}]")
        return "|".join(lines)


class MachineExpression:
    """Write a small minimal automaton as a regular expression, where that reads well.

    A type spells a set of names with no simpler shape as a text that
    depends only on the names. A signature lists every state and every
    move, so it is long. A regular expression is short for the common
    sets: the names that begin with a prefix, a few names, and the
    names that a rename leaves, as `a(b.*)?`. The conversion of
    `greenery` grows fast with the loops of an automaton, so this class
    writes an expression only where the one loop of each state keeps
    every character, as after a prefix. It reads the automaton back
    from the signature, so two equal sets give one expression.

    Example:
        >>> starts = NameSet.literal("ab").concatenate(NameSet.anything())
        >>> MachineExpression(starts.signature()).text()
        'ab.*'
    """

    # The largest automaton that the class writes as an expression.
    LIMIT: ClassVar[int] = 16

    def __init__(self, signature: str) -> None:
        """Read the automaton of a signature.

        Args:
            signature: The signature of a set of names.
        """
        self._signature = signature
        self._machine = SignatureParser(signature).machine()

    def text(self) -> str | None:
        """Give the regular expression of the automaton.

        Returns:
            The expression, or `None` where the automaton is large, has a
            loop that does not keep every character, or gives a longer text
            than its signature.

        Example:
            >>> ends = NameSet.anything().concatenate(NameSet.literal("z"))
            >>> MachineExpression(ends.signature()).text() is None
            True
        """
        if len(self._machine.states) > self.LIMIT or self._loops():
            return None
        text = str(rxelems.from_fsm(self._machine).reduce())
        return text if len(text) <= len(self._signature) else None

    def _loops(self) -> bool:
        """Say whether the automaton has a loop other than a state that keeps every character.

        Returns:
            Whether it has.

        Example:
            >>> MachineExpression("0F[.>0]")._loops()
            False
            >>> MachineExpression("0[[^z]>0;z>1]|1F[[^z]>0;z>1]")._loops()
            True
        """
        moves = {state: set(self._machine.map[state].values()) for state in self._machine.states}
        kept = {state for state, targets in moves.items() if targets == {state}}
        edges = {state: targets - {state} for state, targets in moves.items() if state not in kept}
        if any(state in moves[state] for state in edges):
            return True
        return self._cyclic(edges)

    def _cyclic(self, edges: dict[object, set[object]]) -> bool:
        """Say whether a graph of states has a cycle.

        Args:
            edges: The targets of each state, with no state that keeps every character.

        Returns:
            Whether a walk can come back to a state.

        Example:
            >>> MachineExpression("0F[.>0]")._cyclic({0: {1}, 1: {0}})
            True
        """
        remaining = {state: targets & set(edges) for state, targets in edges.items()}
        while remaining:
            free = {state for state, targets in remaining.items() if not targets}
            if not free:
                return True
            remaining = {
                state: targets - free for state, targets in remaining.items() if state not in free
            }
        return False


class SignatureParser:
    """Rebuild an automaton from the text that `MachineSignature` writes.

    A type loaded from the incremental cache of mypy keeps a name set
    only as this text, so the automaton must come back from it. The
    text is a list of states joined by `|`. Each state is a number, an
    `F` if it accepts, and its moves in brackets, joined by `;`. Each
    move is a character class, `>` and the number of the target. A
    class is a bracket group, or one character that can have an
    escape. So one pass from left to right reads the text, with no
    doubt about where each part ends.

    Example:
        >>> SignatureParser("0F[.>0]").machine().accepts("xyz")
        True
    """

    def __init__(self, text: str) -> None:
        """Keep the text, and start to read it at its first character.

        Args:
            text: A text that `MachineSignature.text` gave.
        """
        self._text = text
        self._at = 0

    def machine(self) -> Fsm:
        """Read the whole text into an automaton.

        Returns:
            The automaton that the text describes, with state 0 as
            its start.

        Example:
            >>> SignatureParser("0[.>0]").machine().accepts("")
            False
        """
        finals: set[int] = set()
        table: dict[int, dict[Charclass, int]] = {}
        while self._at < len(self._text):
            number = self._number()
            if self._took("F"):
                finals.add(number)
            self._expect("[")
            moves: dict[Charclass, int] = {}
            while not self._took("]"):
                symbol = self._symbol()
                self._expect(">")
                moves[symbol] = self._number()
                self._took(";")
            table[number] = moves
            self._took("|")
        alphabet = {symbol for row in table.values() for symbol in row}
        return Fsm(
            alphabet=alphabet,
            states=set(table),
            initial=0,
            finals=finals,
            map=table,
        )

    def _symbol(self) -> Charclass:
        """Read one character class, a bracket group or one character.

        Returns:
            The class. The parser of `greenery` rebuilds it, so each
            escape and each range reads back exactly as it was
            written.
        """
        rendering = self._bracket_group() if self._text[self._at] == "[" else self._single()
        pattern = parse(rendering)
        (conc,) = pattern.concs
        multiplicand = conc.mults[0].multiplicand
        return checked_charclass(multiplicand)

    def _bracket_group(self) -> str:
        """Read past one bracket group, and skip each escaped character in it.

        Returns:
            The text of the group, with its brackets.
        """
        start = self._at
        self._at += 1
        while self._text[self._at] != "]":
            self._at += 2 if self._text[self._at] == "\\" else 1
        self._at += 1
        return self._text[start : self._at]

    def _single(self) -> str:
        """Read past one character, with its escape if it has one.

        Returns:
            The text of the character.
        """
        start = self._at
        self._at += 2 if self._text[self._at] == "\\" else 1
        return self._text[start : self._at]

    def _number(self) -> int:
        """Read past one run of digits.

        Returns:
            The number that the digits give.
        """
        start = self._at
        while self._at < len(self._text) and self._text[self._at].isdigit():
            self._at += 1
        return int(self._text[start : self._at])

    def _expect(self, character: str) -> None:
        """Read past one character that the text must have at this point.

        Args:
            character: The character that must come next.

        Raises:
            ValueError: The text does not continue with it. The text
                did not come from `MachineSignature`.

        Example:
            >>> SignatureParser("0F.>0]").machine()
            Traceback (most recent call last):
            ...
            ValueError: malformed signature at 2: '0F.>0]'
        """
        if not self._took(character):
            problem = f"malformed signature at {self._at}: {self._text!r}"
            raise ValueError(problem)

    def _took(self, character: str) -> bool:
        """Read past one character, if it comes next.

        Args:
            character: The character to look for.

        Returns:
            True if the character came next and was read.

        Example:
            >>> SignatureParser("ab")._took("a")
            True
        """
        if self._text[self._at : self._at + 1] == character:
            self._at += 1
            return True
        return False


def checked_charclass(value: object) -> Charclass:
    """Narrow one parsed value to `Charclass` for mypy.

    The value comes from a text that `MachineSignature` wrote from a
    `Charclass`, so the parser of `greenery` always gives a
    `Charclass` back. A failure means a bug in this module, not a bad
    input.

    Args:
        value: The value that the parser of `greenery` gave.

    Returns:
        The same value, typed as a character class.

    Raises:
        TypeError: The value is not a character class.

    Example:
        >>> checked_charclass(Charclass((("a", "a"),)))
        Charclass((('a', 'a'),))
    """
    if not isinstance(value, Charclass):  # pragma: no cover
        # MachineSignature writes each class from a real Charclass,
        # so its text always parses back into one.
        problem = f"not a character class: {value!r}"
        raise TypeError(problem)
    return value
