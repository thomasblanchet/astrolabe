"""Check the evaluation of whole trees against polars, on random trees of the census nodes.

The tests of the definitions check the rules of each node against polars,
node by node. The evaluation of a tree assumes that a node reads only the
states of its inputs, as polars does. This module measures that assumption:
it builds random trees of the nodes that have a definition (`TreeMaker`), runs
polars on each tree over the frames of the census, and compares the four
answers of polars with the four answers of `TreeEvaluation`.

Each tree runs in a worker process, because polars crashes or hangs on some
trees. A tree that crashes or hangs its worker counts as lost.

Run it with `uv run python -m spec.tools.rules.tree_check <count> [depth] [seed]`.
"""

from __future__ import annotations

import json
import multiprocessing
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, ClassVar

from spec.tools.node_census.kinds import CensusKinds
from spec.tools.rules.census_text import CensusText
from spec.tools.rules.random_trees import TreeMaker, TreeSpace
from astrolabe._rules.dtypes import DType
from astrolabe._rules.execution import GroupBy
from astrolabe._rules.node_evaluation import ColumnLeaf
from astrolabe._rules.options import RowForm
from astrolabe._rules.tree import ExpressionTree
from astrolabe._rules.tree_evaluation import TreeEvaluation

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class TreeResult:
    """Hold the answers of polars and of the rules for one tree.

    Example:
        >>> result = TreeResult("t", ("Int8",) * 4, ("Int8",) * 4)
        >>> result.differences()
        []
    """

    NAMES: ClassVar[tuple[str, ...]] = ("planner select", "planner agg", "run select", "run agg")
    UNCHECKED: ClassVar[frozenset[str]] = frozenset({"unchecked", "hangs"})

    text: str
    polars: tuple[str, ...]
    rules: tuple[str, ...]

    def differences(self) -> list[str]:
        """Give the name of each answer where the rules and polars differ.

        An answer that the rules leave `Unchecked` is not a difference. Nor is
        a run that the rules refuse as a crash (`hangs`): such a run reads past
        the end of an input, and polars can give a value of no meaning or raise
        where the read happens to stay in its memory.

        Returns:
            The names.

        Example:
            >>> TreeResult("t", ("Int8", "raises"), ("Int8", "Int16")).differences()
            ['planner agg']
            >>> TreeResult("t", ("Int8",), ("hangs",)).differences()
            []
        """
        pairs = zip(self.NAMES, self.polars, self.rules, strict=False)
        return [name for name, truth, rule in pairs if rule not in {*self.UNCHECKED, truth}]


class TreeWorld:
    """Hold what one worker needs: the census nodes with a definition, the frames and the leaves.

    Example:
        >>> world = TreeWorld.shared()
        >>> all(template.inputs() <= 3 for template in world.space.templates)
        True
    """

    ONE_GROUP: ClassVar[GroupBy] = GroupBy((3,))

    def __init__(self) -> None:
        """Read the census nodes with a definition, and build the leaves of the columns.

        Example:
            >>> TreeWorld().columns["k0"].dtype
            Int8
        """
        self.space = TreeSpace.census()
        self.columns = self._columns()

    @classmethod
    @cache
    def shared(cls) -> TreeWorld:
        """Give the world of this process, built once.

        Returns:
            The world.

        Example:
            >>> TreeWorld.shared() is TreeWorld.shared()
            True
        """
        return cls()

    def _columns(self) -> dict[str, ColumnLeaf]:
        """Give the leaf of each column of the census frames: each kind, and its twin of nulls.

        Returns:
            The leaves, by the name of the column.

        Example:
            >>> TreeWorld.shared()._columns()["n0"].nulls
            True
        """
        columns: dict[str, ColumnLeaf] = {}
        for index, dtype in enumerate(CensusKinds().dtypes()):
            kind = DType.of(dtype)
            columns[f"k{index}"] = ColumnLeaf(kind, RowForm.PER_ROW)
            columns[f"n{index}"] = ColumnLeaf(kind, RowForm.PER_ROW, nulls=True)
        return columns

    def result(self, seed: int, depth: int) -> TreeResult:
        """Build the tree of a seed, and give the answers of polars and of the rules.

        Args:
            seed: The seed of the random source.
            depth: The depth of the tree.

        Returns:
            The result.

        Example:
            >>> result = TreeWorld.shared().result(1, 1)
            >>> len(result.polars), len(result.rules)
            (4, 4)
        """
        source = random.Random(seed)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
        tree = TreeMaker(self.space, source).tree(depth)
        truth = self.space.truth.tree(tree)
        polars = (str(truth[0]), str(truth[3]), str(truth[1]), str(truth[4]))
        evaluation = TreeEvaluation(self.columns, self.ONE_GROUP)
        answers = evaluation.answers(ExpressionTree().of(tree.expression()))
        rules = tuple(CensusText().answers(answers))
        return TreeResult(tree.text(), polars, rules)


def check_one(seed: int, depth: int) -> TreeResult:
    """Give the result of one tree, in a worker process.

    Args:
        seed: The seed of the tree.
        depth: The depth of the tree.

    Returns:
        The result.

    Example:
        >>> check_one(1, 1).text != ""
        True
    """
    return TreeWorld.shared().result(seed, depth)


@dataclass
class TreeTally:
    """Count the answers that agree, and keep examples of the trees that differ.

    Example:
        >>> tally = TreeTally()
        >>> tally.add(TreeResult("t", ("Int8",) * 4, ("Int8",) * 4))
        >>> tally.agreed
        Counter({'planner select': 1, 'planner agg': 1, 'run select': 1, 'run agg': 1})
    """

    EXAMPLES: ClassVar[int] = 200

    agreed: Counter[str] = field(default_factory=Counter)
    differed: Counter[str] = field(default_factory=Counter)
    unchecked: Counter[str] = field(default_factory=Counter)
    lost: int = 0
    examples: list[dict[str, object]] = field(default_factory=list)

    def add(self, result: TreeResult) -> None:
        """Count the answers of one tree.

        Args:
            result: The result of the tree.

        Example:
            >>> tally = TreeTally()
            >>> tally.add(TreeResult("t", ("Int8",) * 4, ("unchecked",) * 4))
            >>> tally.unchecked["run agg"]
            1
        """
        differences = result.differences()
        for name, rule in zip(TreeResult.NAMES, result.rules, strict=False):
            if rule in TreeResult.UNCHECKED:
                self.unchecked[name] += 1
            elif name in differences:
                self.differed[name] += 1
            else:
                self.agreed[name] += 1
        if differences and len(self.examples) < self.EXAMPLES:
            self.examples.append({
                "tree": result.text,
                "polars": result.polars,
                "rules": result.rules,
            })

    def summary(self) -> dict[str, object]:
        """Give the counts and the examples.

        Returns:
            The summary.

        Example:
            >>> TreeTally().summary()["lost"]
            0
        """
        return {
            "agreed": dict(self.agreed),
            "differed": dict(self.differed),
            "unchecked": dict(self.unchecked),
            "lost": self.lost,
            "examples": self.examples,
        }


class TreeCheck:
    """Run the check on many trees, in worker processes that a crash or a hang does not stop.

    Each worker starts a fresh interpreter (`spawn`): a fork copies the
    threads of polars in the parent, and the child can then hang.

    Example:
        >>> TreeCheck(workers=1).run([1], depth=1).agreed.total() + TreeTally().lost >= 0
        True
    """

    TIMEOUT: ClassVar[float] = 60.0

    def __init__(self, workers: int = 2) -> None:
        """Keep the number of workers.

        Args:
            workers: The number of worker processes.

        Example:
            >>> TreeCheck(workers=3)._workers
            3
        """
        self._workers = workers

    def run(self, seeds: Sequence[int], depth: int) -> TreeTally:
        """Check the tree of each seed, and count the answers.

        Args:
            seeds: The seeds of the trees.
            depth: The depth of the trees.

        Returns:
            The tally.

        Example:
            >>> TreeCheck(workers=1).run([2], depth=1).lost
            0
        """
        tally = TreeTally()
        pool = self._pool()
        for seed in seeds:
            try:
                tally.add(pool.submit(check_one, seed, depth).result(timeout=self.TIMEOUT))
            except (BrokenProcessPool, FutureTimeout):
                tally.lost += 1
                pool = self._restart(pool)
        pool.shutdown(cancel_futures=True)
        return tally

    def _restart(self, pool: ProcessPoolExecutor) -> ProcessPoolExecutor:
        """Stop a pool whose worker crashed or hangs, and give a new one.

        Args:
            pool: The pool.

        Returns:
            The new pool.

        Example:
            >>> pool = ProcessPoolExecutor(1)
            >>> TreeCheck(workers=1)._restart(pool) is not pool
            True
        """
        for process in list(getattr(pool, "_processes", {}).values()):
            process.kill()
        pool.shutdown(wait=False, cancel_futures=True)
        return self._pool()

    def _pool(self) -> ProcessPoolExecutor:
        """Give a pool of workers that each start a fresh interpreter.

        Returns:
            The pool.

        Example:
            >>> TreeCheck(workers=1)._pool().shutdown()
        """
        return ProcessPoolExecutor(self._workers, mp_context=multiprocessing.get_context("spawn"))

    @classmethod
    def main(cls) -> int:
        """Run the check from the command line, and print the summary as JSON.

        Returns:
            The exit code.
        """
        count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        tally = cls().run(range(start, start + count), depth)
        sys.stdout.write(json.dumps(tally.summary(), indent=1) + "\n")
        return 0


if __name__ == "__main__":
    sys.exit(TreeCheck.main())
