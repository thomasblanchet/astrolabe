"""The release of polars that the rules follow, and the methods whose rule changes between releases.

The rules of polars change between releases. For example, from 1.43.2 the
cast pass casts each input of `concat_str` that is not text to `String`, and
1.42.1 does not. The checker runs in the environment of the project that it
checks, so it imports the polars of that project, and its rules must be the
rules of that release.

A rule that changes is a method with one variant for each range of releases.
`polars_releases` makes a method into such a rule, and the decorator `before`
of that rule adds the variant of the older releases. Code that applies the
rule calls the plain method, and the method runs the variant of the current
release. `PolarsRelease` gives the current release: the installed one, or the
one that a test sets with `PolarsRelease.pretend`.
"""

from __future__ import annotations

import bisect
import contextlib
import contextvars
import functools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Concatenate, Self, overload

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(frozen=True, order=True)
class PolarsRelease:
    """A release of polars, ordered by its version numbers.

    The rules compare releases to pick the variant of a rule that changes
    between them. The current release is the installed release of polars,
    unless code runs inside `PolarsRelease.pretend`. A test uses `pretend` to
    run the variant of a release that is not installed.

    Example:
        >>> PolarsRelease.parse("1.42.1") < PolarsRelease.parse("1.43.2")
        True
        >>> with PolarsRelease.pretend("1.40.1"):
        ...     print(PolarsRelease.current())
        1.40.1
    """

    VERSION: ClassVar[re.Pattern[str]] = re.compile(r"(\d+)\.(\d+)\.(\d+)")
    PRETENDED: ClassVar[contextvars.ContextVar[PolarsRelease | None]] = contextvars.ContextVar(
        "pretended_polars_release",
        default=None,
    )

    numbers: tuple[int, int, int]

    def __str__(self) -> str:
        """Write the release as polars writes its version.

        Returns:
            The three numbers, joined by dots.

        Example:
            >>> str(PolarsRelease.parse("2.0.0"))
            '2.0.0'
        """
        return ".".join(map(str, self.numbers))

    @classmethod
    def parse(cls, version: str) -> PolarsRelease:
        """Read a release from a version of polars.

        A version can carry a suffix after its three numbers, for example a
        candidate release. The suffix does not change the place of the
        release in the order.

        Args:
            version: The version, for example `"1.43.2"`.

        Returns:
            The release.

        Raises:
            ValueError: The version does not start with three numbers.

        Example:
            >>> PolarsRelease.parse("2.0.0rc1")
            PolarsRelease(numbers=(2, 0, 0))
            >>> PolarsRelease.parse("main")
            Traceback (most recent call last):
            ...
            ValueError: 'main' is not a version of polars
        """
        match = cls.VERSION.match(version)
        if match is None:
            message = f"{version!r} is not a version of polars"
            raise ValueError(message)
        major, minor, patch = (int(number) for number in match.groups())
        return cls((major, minor, patch))

    @classmethod
    @functools.cache
    def installed(cls) -> PolarsRelease:
        """Give the release of polars that the environment holds.

        Returns:
            The release.

        Example:
            >>> PolarsRelease.installed() >= PolarsRelease.parse("1.39.3")
            True
        """
        return cls.parse(pl.__version__)

    @classmethod
    def current(cls) -> PolarsRelease:
        """Give the release whose rules apply: the one that `pretend` sets, or the installed one.

        Returns:
            The release.

        Example:
            >>> PolarsRelease.current() == PolarsRelease.installed()
            True
        """
        pretended = cls.PRETENDED.get()
        return cls.installed() if pretended is None else pretended

    @classmethod
    @contextlib.contextmanager
    def pretend(cls, version: str) -> Iterator[None]:
        """Apply the rules of another release to the code inside the block.

        Args:
            version: The release, for example `"1.42.1"`.

        Yields:
            Nothing. The release applies until the block ends.

        Example:
            >>> with PolarsRelease.pretend("1.42.1"):
            ...     with PolarsRelease.pretend("1.39.3"):
            ...         print(PolarsRelease.current())
            ...     print(PolarsRelease.current())
            1.39.3
            1.42.1
        """
        token = cls.PRETENDED.set(cls.parse(version))
        try:
            yield
        finally:
            cls.PRETENDED.reset(token)


class polars_releases[S, **P, R]:
    """Make a method into a rule that changes between releases of polars.

    The decorated method is the rule of the newest releases. The decorator
    `before(version)` of the rule adds a variant: a method that gives the rule
    of each release older than that version. A rule can have several
    variants, and each release runs the variant with the lowest version above
    it. A release that no variant covers runs the decorated method.

    Code that applies the rule calls the decorated method by its name, and the
    method picks its variant at each call, from `PolarsRelease.current`. So a
    test can run each variant under any installed release. The type checker
    types each call from the decorated method, and it checks that each variant
    takes the same arguments and gives the same type.

    Each variant has a name of its own, which says what it does, for example
    `_no_cast_beside_float`. The type checker does not accept two methods of
    one name under a decorator that it does not know.

    The rule takes the name, the docstring and the module of the decorated
    method, so the doctests of that method run. Code that reads the rule on
    the class calls it with an instance first, as it calls a plain function.

    Example:
        >>> class Join:
        ...     @polars_releases
        ...     def name(self, left: str) -> str:
        ...         return left + "_right"
        ...
        ...     @name.before("1.40.0")
        ...     def _with_old_suffix(self, left: str) -> str:
        ...         return left + "_r"
        >>> with PolarsRelease.pretend("1.39.3"):
        ...     print(Join().name("a"))
        a_r
        >>> with PolarsRelease.pretend("1.43.2"):
        ...     print(Join().name("a"))
        a_right
    """

    def __init__(self, newest: Callable[Concatenate[S, P], R]) -> None:
        """Hold the rule of the newest releases.

        Args:
            newest: The method that gives the rule of the newest releases.

        Example:
            >>> rule = polars_releases(str.upper)
            >>> rule.variant(PolarsRelease.parse("1.43.2")) is str.upper
            True
        """
        self._newest = newest
        self._bounds: list[PolarsRelease] = []
        self._variants: list[Callable[Concatenate[S, P], R]] = []
        functools.update_wrapper(self, newest)

    def __call__(self, instance: S, /, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run the variant of the current release, for code that reads the rule on the class.

        Args:
            instance: The object that the variant runs on.
            *args: The arguments of the rule.
            **kwargs: The keyword arguments of the rule.

        Returns:
            What the variant gives.

        Example:
            >>> rule = polars_releases(str.upper)
            >>> rule.add(PolarsRelease.parse("99.0.0"), str.lower)
            >>> rule("aB"), rule.__name__
            ('ab', 'upper')
        """
        variant = self.variant(PolarsRelease.current())
        return variant(instance, *args, **kwargs)

    @overload
    def __get__(self, instance: None, owner: type[S]) -> Self: ...

    @overload
    def __get__(self, instance: S, owner: type[S]) -> Callable[P, R]: ...

    def __get__(self, instance: S | None, owner: type[S]) -> Self | Callable[P, R]:
        """Give the variant of the current release, bound to the instance.

        Args:
            instance: The object whose method the code reads, or `None` on the class.
            owner: The class that defines the method.

        Returns:
            The bound variant, or the rule itself where the code reads it on the class.

        Example:
            >>> class Rule:
            ...     @polars_releases
            ...     def length(self, text: str) -> int:
            ...         return len(text)
            >>> Rule().length("abc")
            3
            >>> type(Rule.length).__name__
            'polars_releases'
        """
        if instance is None:
            return self
        variant = self.variant(PolarsRelease.current())
        return functools.partial(variant, instance)

    def before(self, version: str) -> OlderVariant[S, P, R]:
        """Give the decorator that adds the variant of the releases older than a version.

        Args:
            version: The first release that does not run the variant, for example `"1.43.2"`.

        Returns:
            The decorator. It gives back the variant unchanged, so the variant
            stays a method under its own name.

        Example:
            >>> rule = polars_releases(str.upper)
            >>> rule.before("1.43.2")(str.lower) is str.lower
            True
        """
        return OlderVariant(self, PolarsRelease.parse(version))

    def add(self, bound: PolarsRelease, variant: Callable[Concatenate[S, P], R]) -> None:
        """Add the variant of the releases older than a bound.

        Args:
            bound: The first release that does not run the variant.
            variant: The method that gives the rule of those releases.

        Raises:
            ValueError: Another variant has the same bound.

        Example:
            >>> rule = polars_releases(str.upper)
            >>> rule.add(PolarsRelease.parse("1.43.2"), str.lower)
            >>> rule.add(PolarsRelease.parse("1.43.2"), str.title)
            Traceback (most recent call last):
            ...
            ValueError: two variants of upper apply before polars 1.43.2
        """
        if bound in self._bounds:
            message = f"two variants of {self._newest.__name__} apply before polars {bound}"
            raise ValueError(message)
        index = bisect.bisect(self._bounds, bound)
        self._bounds.insert(index, bound)
        self._variants.insert(index, variant)

    def variant(self, release: PolarsRelease) -> Callable[Concatenate[S, P], R]:
        """Give the variant that a release runs: the one with the lowest bound above it.

        Args:
            release: The release.

        Returns:
            The variant, or the rule of the newest releases where no bound is above the release.

        Example:
            >>> rule = polars_releases(str.upper)
            >>> rule.add(PolarsRelease.parse("1.43.2"), str.lower)
            >>> rule.add(PolarsRelease.parse("1.41.0"), str.title)
            >>> for version in ("1.40.1", "1.41.0", "1.43.2"):
            ...     print(rule.variant(PolarsRelease.parse(version))("aB"))
            Ab
            ab
            AB
        """
        index = bisect.bisect(self._bounds, release)
        if index == len(self._bounds):
            return self._newest
        return self._variants[index]


class OlderVariant[S, **P, R]:
    """Decorate a variant of a rule that changes between releases, and add it to the rule.

    It holds the rule and the bound of the variant: the first release that does
    not run it. Its call takes a method with the arguments and the result of
    the rule, so the type checker reports a variant that differs.

    Example:
        >>> rule = polars_releases(str.upper)
        >>> OlderVariant(rule, PolarsRelease.parse("1.43.2"))(str.lower) is str.lower
        True
    """

    def __init__(self, rule: polars_releases[S, P, R], bound: PolarsRelease) -> None:
        """Hold the rule and the bound of the variant.

        Args:
            rule: The rule that the variant belongs to.
            bound: The first release that does not run the variant.

        Example:
            >>> rule = polars_releases(str.upper)
            >>> type(OlderVariant(rule, PolarsRelease.parse("1.43.2"))).__name__
            'OlderVariant'
        """
        self._rule = rule
        self._bound = bound

    def __call__(self, variant: Callable[Concatenate[S, P], R]) -> Callable[Concatenate[S, P], R]:
        """Add the variant to the rule.

        Args:
            variant: The method that gives the rule of the releases older than the bound.

        Returns:
            The variant, unchanged.

        Example:
            >>> rule = polars_releases(str.upper)
            >>> add = OlderVariant(rule, PolarsRelease.parse("1.43.2"))
            >>> add(str.lower)("aB")
            'ab'
        """
        self._rule.add(self._bound, variant)
        return variant
