"""Score a mutation run, and list each mutant that survived.

A score alone does not tell a maintainer what to do next. The report
therefore gives a score for each file, worst first, and names each
surviving mutant by its file, its line and its operator. Each survivor
points at a rule that no test checks. The file of results also holds the
test that killed each killed mutant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from tools.mutation.runner import Outcome

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import TextIO

    from tools.mutation.runner import MutantResult


@dataclass(frozen=True)
class Score:
    """Count the killed and the surviving mutants of a run or of one file.

    Only these two outcomes measure the tests. The other outcomes say
    nothing about them: no test reaches the mutant, the operator made no
    change or raised an exception, or the tests ran out of time. The
    score leaves those out.

    Example:
        >>> Score(3, 1).percent()
        75.0
    """

    killed: int
    survived: int

    def total(self) -> int:
        """Count the mutants that the score is out of.

        Returns:
            The killed and the surviving mutants together.

        Example:
            >>> Score(3, 1).total()
            4
        """
        return self.killed + self.survived

    def percent(self) -> float:
        """Give the share of the counted mutants that a test killed.

        Returns:
            The share as a percentage, to one decimal place. 100.0 if
            no mutant counts, because no test then missed anything.

        Example:
            >>> Score(0, 0).percent()
            100.0
        """
        if self.total() == 0:
            return 100.0
        return round(100.0 * self.killed / self.total(), 1)


class MutationReport:
    """Turn the results of a run into the text report that a maintainer triages.

    The report opens with the score of the whole run and a count of
    every outcome. It then gives the score of each file, worst first,
    and ends with one line for each surviving mutant.

    Example:
        >>> MutationReport([]).score()
        Score(killed=0, survived=0)
    """

    WIDTH: ClassVar[int] = 6

    def __init__(self, results: Sequence[MutantResult]) -> None:
        """Keep the results of one run.

        Args:
            results: Every result of the run, the results that an
                earlier run finished included.

        Example:
            >>> MutationReport([]).survivors()
            []
        """
        self.results = results

    def score(self) -> Score:
        """Score the whole run.

        Returns:
            The count of killed and of surviving mutants.

        Example:
            >>> MutationReport([]).score().percent()
            100.0
        """
        return self._score(self.results)

    def survivors(self) -> list[MutantResult]:
        """List the mutants that no test killed, which are the findings of the audit.

        Returns:
            Each surviving result, sorted by file and line.

        Example:
            >>> MutationReport([]).survivors()
            []
        """
        found = [r for r in self.results if r.outcome is Outcome.SURVIVED]
        return sorted(found, key=self._place)

    def by_module(self) -> list[tuple[str, Score]]:
        """Score each file on its own, so that a maintainer sees where the gaps gather.

        A result that carries no mutant goes under a dash.

        Returns:
            Each file beside its score, worst first.

        Example:
            >>> MutationReport([]).by_module()
            []
        """
        groups: dict[str, list[MutantResult]] = {}
        for result in self.results:
            name = "-" if result.mutant is None else str(result.mutant.path)
            groups.setdefault(name, []).append(result)
        scored = [(name, self._score(group)) for name, group in groups.items()]
        return sorted(scored, key=lambda pair: (pair[1].percent(), pair[0]))

    def counts(self) -> dict[str, int]:
        """Count each outcome, those that the score leaves out included.

        Returns:
            One count for each outcome, by its name.

        Example:
            >>> MutationReport([]).counts()  # doctest: +NORMALIZE_WHITESPACE
            {'killed': 0, 'survived': 0, 'timed out': 0, 'hung': 0,
             'unselected': 0, 'unreached': 0, 'unchanged': 0, 'errored': 0}
        """
        found = {outcome.value: 0 for outcome in Outcome}
        for result in self.results:
            found[result.outcome.value] += 1
        return found

    def write(self, out: TextIO) -> None:
        """Write the whole report.

        After the score and the count of each outcome, the report gives
        one line for each file: its score, its kills out of its total,
        and its path. It ends with one line for each surviving mutant,
        or a dash for a result that carries no mutant.

        Args:
            out: The stream to write to.

        Example:
            >>> import sys
            >>> MutationReport([]).write(sys.stdout)  # doctest: +NORMALIZE_WHITESPACE
            mutation score 100.0% (0 killed, 0 survived)
            outcomes: killed 0, survived 0, timed out 0, hung 0, unselected 0,
            unreached 0, unchanged 0, errored 0
        """
        score = self.score()
        print(
            f"mutation score {score.percent()}% ({score.killed} killed, {score.survived} survived)",
            file=out,
        )
        print("outcomes: " + ", ".join(f"{k} {v}" for k, v in self.counts().items()), file=out)
        for name, file_score in self.by_module():
            percent = f"{file_score.percent():>{self.WIDTH}}%"
            line = f"{percent}  {file_score.killed:>4}/{file_score.total():<4} {name}"
            print(line, file=out)
        for result in self.survivors():
            label = "-" if result.mutant is None else result.mutant.label()
            print(f"survived  {label}", file=out)

    def _place(self, result: MutantResult) -> tuple[str, int]:
        """Give the sort key of one result: where its mutant sits.

        Args:
            result: The result to place.

        Returns:
            The file and the line, or a dash and zero if the result has
            no mutant.

        Example:
            >>> from tools.mutation.runner import MutantResult, Outcome
            >>> MutationReport([])._place(MutantResult(None, Outcome.KILLED, None, 0.0))
            ('-', 0)
        """
        if result.mutant is None:
            return ("-", 0)
        return (str(result.mutant.path), result.mutant.line)

    def _score(self, results: Iterable[MutantResult]) -> Score:
        """Score one group of results.

        The score counts only the killed and the surviving results.

        Args:
            results: The results to count.

        Returns:
            The count of killed and of surviving mutants.

        Example:
            >>> from tools.mutation.runner import MutantResult
            >>> killed = MutantResult(None, Outcome.KILLED, None, 0.0)
            >>> unchanged = MutantResult(None, Outcome.UNCHANGED, None, 0.0)
            >>> MutationReport([])._score([killed, unchanged])
            Score(killed=1, survived=0)
        """
        counted = [result for result in results if result.scored()]
        killed = sum(1 for result in counted if result.outcome is Outcome.KILLED)
        return Score(killed, len(counted) - killed)
