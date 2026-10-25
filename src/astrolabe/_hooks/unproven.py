"""Decide which results the checker accepts without a proof, one reason at a time.

Sometimes a hook cannot prove the result of an operation. For example, a
column name held in a variable gives no literal name to check. The hook
then gives up on the proof and falls back: it gives a less precise type,
such as a plain frame or a column of unknown data type. Each place that
gives up names a reason, which says what the user wrote that stopped the
proof.

Users differ in how much of this they accept. So each reason has a level
on one ordered scale. At the low end are constructs that state their
intent, such as a name in a variable. At the high end are constructs
that make the checker take a declared type on trust. The
`allow_unproven` setting picks one level, for all operations or for one
operation. A reason passes silently if its level is at that level or
lower. Otherwise the checker reports it.

Two reasons are off the scale. A checker gap is a case that the plugin
does not model, where the user wrote nothing to correct, and a separate
setting controls it. The reason `REPORTED` marks a path where an error
is already reported, so the checker reports nothing more there.

This module is data only and imports nothing from the rest of the
plugin. So the code for steps that can fail and the error classes can
both use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import ClassVar


class Level(IntEnum):
    """How much the checker accepts without a proof, from least to most.

    A user writes a level as the value of the `allow_unproven` setting.
    `NEVER` accepts nothing, checker gaps included. `NONE` accepts no
    reason on the scale, and reports checker gaps only if the
    `report_checker_gaps` setting is on. Each higher level accepts every
    reason that the lower levels accept, and more.

    Example:
        >>> Level.STATED < Level.OMITTED
        True
        >>> Level.POSSIBLE.spelling
        'possible'
    """

    NEVER = 0
    NONE = 1
    STATED = 2
    OMITTED = 3
    POSSIBLE = 4
    ASSERTED = 5

    @property
    def spelling(self) -> str:
        """Give the name that a user writes for this level under `[tool.astrolabe]`.

        Returns:
            The name of the level in lowercase.

        Example:
            >>> Level.NEVER.spelling
            'never'
        """
        return self.name.lower()

    @classmethod
    def named(cls, name: str) -> Level | None:
        """Read a level from the name that a user writes in a setting.

        The match is exact, so a name with a capital letter is not a
        level. Code that reads the settings uses the None answer to warn
        about a value it does not know.

        Args:
            name: The value of the setting.

        Returns:
            The level, or None if `name` is not the lowercase name of a
            level.

        Example:
            >>> Level.named("omitted")
            <Level.OMITTED: 3>
            >>> Level.named("Omitted") is None
            True
        """
        for level in cls:
            if level.spelling == name:
                return level
        return None


class Reason(Enum):
    """Say why one place in a hook gave up on a proof.

    Each reason carries two values. The first is its level, which decides
    whether the `allow_unproven` setting lets it pass. It is None for the
    two reasons off the scale. The second is a plain phrase that says what
    stopped the proof. An error message uses the phrase to complete a
    sentence about the operation.

    Example:
        >>> Reason.DYNAMIC_NAME.level
        <Level.STATED: 2>
        >>> Reason.DYNAMIC_NAME.wording
        'a column name is not a literal string'
        >>> Reason.CHECKER_GAP.level is None
        True
    """

    DYNAMIC_NAME = (Level.STATED, "a column name is not a literal string")
    DYNAMIC_OPTION = (Level.STATED, "an option of the call is not a literal")
    PLAIN_EXPRESSION = (Level.STATED, "an expression is not typed")
    NO_SCHEMA = (Level.STATED, "no schema gives the column types")
    PARTIAL_PICK = (
        Level.STATED,
        "the data type of a column decides whether a selector picks it, and no type declares it",
    )
    OPAQUE_NO_KEYWORD = (Level.STATED, "a Python function decides the type")
    OPAQUE_KEYWORD_OMITTED = (
        Level.OMITTED,
        "a Python function decides the type, and no return_dtype= names it",
    )
    SERIES_NO_DTYPE = (Level.OMITTED, "no dtype= names the type")
    BARE_INT = (Level.OMITTED, "a number variable has no fixed width")
    NON_LITERAL_DTYPE = (Level.OMITTED, "a data type is not written as a literal")
    REMAINDER_MAY_HOLD = (
        Level.POSSIBLE,
        "the undeclared columns of the frame can hold the name, or not",
    )
    CALLER_DEPENDENT = (
        Level.POSSIBLE,
        "the result depends on the undeclared columns the caller passes",
    )
    CALLER_KEY = (
        Level.POSSIBLE,
        "the caller can pass a join key of another form, which changes the join",
    )
    UNPROVEN_REQUIREMENT = (
        Level.ASSERTED,
        "the undeclared columns of the caller do not prove the requirement",
    )
    UNTYPED_ARGUMENT = (
        Level.ASSERTED,
        "an argument is not typed, so its columns and types are unknown",
    )
    CHECKER_GAP = (None, "the checker does not model this case")
    REPORTED = (None, "an error is already reported here")

    def __init__(self, level: Level | None, wording: str) -> None:
        """Keep the level and the phrase of one reason.

        Args:
            level: The level of the reason, or None for a reason off the
                scale.
            wording: The plain phrase that says what stopped the proof.
        """
        self.level = level
        self.wording = wording


@dataclass(frozen=True)
class UnprovenPolicy:
    """Answer whether one unproven result passes, from the settings of the user.

    It holds the three settings that decide this: the global level, a
    level for each operation that overrides it, and the switch for
    checker gaps. Code that reports an unproven result asks it two
    questions. Does this reason pass in this operation? If not, which
    setting must the user change to let it pass?

    `KNOWN_OPERATIONS` lists the operation names that an override can
    use.

    Attributes:
        level: The global level, for each operation with no override.
        overrides: A level for each operation that has its own, keyed by
            a name from `KNOWN_OPERATIONS`.
        report_gaps: The value of the `report_checker_gaps` setting.

    Example:
        >>> policy = UnprovenPolicy(Level.OMITTED, {"select": Level.NONE})
        >>> policy.allows(Reason.BARE_INT, "with_columns")
        True
        >>> policy.allows(Reason.BARE_INT, "select")
        False
    """

    KNOWN_OPERATIONS: ClassVar[frozenset[str]] = frozenset({
        "select",
        "with_columns",
        "agg",
        "group_by",
        "rolling",
        "group_by_dynamic",
        "filter",
        "remove",
        "drop",
        "rename",
        "join",
        "join_where",
        "join_asof",
        "explode",
        "unnest",
        "hstack",
        "pivot",
        "corr",
        "unpivot",
        "melt",
        "with_row_index",
        "update",
        "map_columns",
        "concat",
        "Series",
        "read_csv",
        "scan_csv",
        "read_parquet",
        "scan_parquet",
        "read_json",
        "read_ndjson",
        "scan_ndjson",
        "from_numpy",
        "from_dict",
        "from_dicts",
        "from_records",
        "map_elements",
        "map_batches",
        "rolling_map",
        "fit",
        "get_column",
        "fill_null",
        "fill_nan",
    })

    level: Level = Level.OMITTED
    overrides: dict[str, Level] = field(default_factory=dict)
    report_gaps: bool = False

    def level_for(self, operation: str) -> Level:
        """Give the level that applies to one operation.

        Args:
            operation: The name of the operation.

        Returns:
            The override of the operation if it has one, else the global
            level.

        Example:
            >>> UnprovenPolicy(Level.STATED, {"join": Level.NEVER}).level_for("join")
            <Level.NEVER: 0>
        """
        return self.overrides.get(operation, self.level)

    def reports_gaps(self, operation: str) -> bool:
        """Answer whether the checker reports a checker gap in one operation.

        Args:
            operation: The name of the operation.

        Returns:
            True if the `report_checker_gaps` setting is on, or if the
            level of the operation is `NEVER`.

        Example:
            >>> UnprovenPolicy(Level.NEVER).reports_gaps("select")
            True
            >>> UnprovenPolicy(Level.NONE).reports_gaps("select")
            False
        """
        return self.report_gaps or self.level_for(operation) is Level.NEVER

    def allows(self, reason: Reason, operation: str) -> bool:
        """Answer whether one reason passes in one operation, with no error.

        Args:
            reason: Why the operation gave up on a proof.
            operation: The name of the operation.

        Returns:
            For a reason on the scale, True if its level is at the level
            of the operation or lower. For a checker gap, True unless the
            operation reports gaps. For `REPORTED`, always True.

        Example:
            >>> policy = UnprovenPolicy(Level.STATED)
            >>> policy.allows(Reason.DYNAMIC_NAME, "select")
            True
            >>> policy.allows(Reason.REMAINDER_MAY_HOLD, "select")
            False
            >>> UnprovenPolicy(Level.NEVER).allows(Reason.REPORTED, "select")
            True
            >>> UnprovenPolicy(Level.NEVER).allows(Reason.CHECKER_GAP, "select")
            False
        """
        if reason.level is None:
            return reason is Reason.REPORTED or not self.reports_gaps(operation)
        return reason.level <= self.level_for(operation)

    def allowance(self, reason: Reason, operation: str) -> str:
        """Tell the user which setting to change so that one reported reason passes.

        The error for an unproven result ends with a note that holds this
        instruction.

        Args:
            reason: The reason that the operation reported.
            operation: The name of the operation.

        Returns:
            The setting to change, as one instruction. For a checker gap
            with the gap switch on, an instruction to turn the switch off.
            For any other checker gap, an instruction to set the lowest
            level that does not report gaps.

        Example:
            >>> hint = UnprovenPolicy(Level.NONE).allowance(Reason.BARE_INT, "select")
            >>> hint.startswith('set allow_unproven = "omitted"')
            True
            >>> UnprovenPolicy(report_gaps=True).allowance(Reason.CHECKER_GAP, "select")
            'set report_checker_gaps = false in [tool.astrolabe] to allow this'
            >>> hint = UnprovenPolicy(Level.NEVER).allowance(Reason.CHECKER_GAP, "select")
            >>> hint.startswith('set allow_unproven = "none"')
            True
        """
        if reason.level is not None:
            return self._level_allowance(reason.level, operation)
        if self.report_gaps:
            return "set report_checker_gaps = false in [tool.astrolabe] to allow this"
        return self._level_allowance(Level.NONE, operation)

    def cache_key(self) -> object:
        """Give a stable value that mypy can store in its incremental cache.

        mypy reuses the cached result of a module only if the plugin data
        is unchanged. The value must encode as JSON and must be the same
        for equal settings, so the overrides are sorted.

        Returns:
            The level, the overrides sorted by operation, and the gap
            switch, with each level as its name.

        Example:
            >>> UnprovenPolicy(Level.NONE, {"b": Level.STATED, "a": Level.NEVER}).cache_key()
            {'level': 'none', 'overrides': {'a': 'never', 'b': 'stated'}, 'report_gaps': False}
        """
        entries = sorted(self.overrides.items())
        overrides = {operation: level.spelling for operation, level in entries}
        return {
            "level": self.level.spelling,
            "overrides": overrides,
            "report_gaps": self.report_gaps,
        }

    def _level_allowance(self, level: Level, operation: str) -> str:
        """Tell the user how to raise one operation to a level.

        An override wins over the global level. So if the operation has
        an override, the instruction names only the override. Else it
        offers both the global level and a new override.

        Args:
            level: The lowest level that lets the reason pass.
            operation: The name of the operation.

        Returns:
            The setting to change, as one instruction.

        Example:
            >>> policy = UnprovenPolicy(Level.NONE, {"select": Level.NONE})
            >>> policy._level_allowance(Level.STATED, "select")
            'set [tool.astrolabe.allow_unproven_overrides] select = "stated" to allow this'
        """
        override = f'set [tool.astrolabe.allow_unproven_overrides] {operation} = "{level.spelling}"'
        if operation in self.overrides:
            return f"{override} to allow this"
        return (
            f'set allow_unproven = "{level.spelling}" in [tool.astrolabe] to allow this,'
            f" or {override}"
        )
