"""Read the settings of the plugin from the `[tool.astrolabe]` table of `pyproject.toml`.

The plugin reads its settings once, when mypy loads it. A wrong value
never stops the run: the reader logs a warning that names the key and
the value it expected, and it uses the default. Each setting is read on
its own, so one wrong key does not hide the others.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from astrolabe._hooks.unproven import Level, UnprovenPolicy

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.options import Options


@dataclass(frozen=True)
class ImplicitInferredFallbackPolicy:
    """Decide whether an inferred return type that falls back to the plain type is an error.

    A function can declare its return type as `InferredExpr`,
    `InferredDataFrame` or `InferredLazyFrame`, and mark each value that it
    returns with `infer(...)`. The checker then gives the function the union
    of the types of those values. If one of them is a plain type that the
    checker does not track, the whole union falls back to the plain type,
    and every typed column is lost. This policy decides whether the checker
    reports an error when that happens.

    The setting `disallow_implicit_inferred_fallback` gives the default. A
    sub-table of overrides can set a different value for each of the three
    kinds: `expr`, `dataframe` and `lazyframe`.

    Attributes:
        default: The value of `disallow_implicit_inferred_fallback`, which
            applies to each kind with no override.
        overrides: The value for each kind that has an override, by one of
            the names in `KNOWN_OPERATIONS`.
    """

    KNOWN_OPERATIONS: ClassVar[frozenset[str]] = frozenset({"expr", "dataframe", "lazyframe"})

    default: bool = False
    overrides: dict[str, bool] = field(default_factory=dict)

    def disallows(self, operation: str) -> bool:
        """Tell whether falling back is an error for one kind of inferred return type.

        Args:
            operation: The name of the kind, one of `KNOWN_OPERATIONS`.

        Returns:
            The override for that kind if it has one. Otherwise the default.

        Example:
            >>> policy = ImplicitInferredFallbackPolicy(default=True, overrides={"expr": False})
            >>> policy.disallows("expr")
            False
            >>> policy.disallows("dataframe")
            True
        """
        return self.overrides.get(operation, self.default)

    def cache_key(self) -> object:
        """Give a value that JSON can encode and that changes when the policy changes.

        mypy compares this value between runs. When it changes, mypy checks the
        cached modules again.

        Returns:
            The default and the overrides, with the overrides sorted by key so
            that the same policy always gives the same value.

        Example:
            >>> ImplicitInferredFallbackPolicy(default=True).cache_key()["default"]
            True
        """
        return {"default": self.default, "overrides": dict(sorted(self.overrides.items()))}


@dataclass(frozen=True)
class AstrolabeConfig:
    """The settings of the plugin, after the reader applied each default.

    Every hook that has a setting reads it from this one object. The
    reader in `AstrolabeConfigLoader` builds it, so a hook never reads the
    file itself.

    Attributes:
        allow_unproven: The level of proof that the checker needs before it
            accepts a result, for all operations and for each one, and
            whether it reports a gap in the checker. See `UnprovenPolicy`.
            The default level is `omitted`, and gaps are not reported.
        disallow_vertical_concat: Whether each vertical `pl.concat` is an
            error. This covers `how="vertical"` and the default `how`. The
            default is False. The checker never types the result of a
            vertical concat in any case. Polars stacks the columns by
            position, and the checker cannot prove the order of the columns
            that a remainder holds. With False, the checker still reports a
            sure mismatch: two frames whose column names or data types
            differ, whatever their order.
        elevate_untyped_select: Whether `.select()` on a plain frame gives a
            typed frame. The default is True. Each requested column gets its
            name from the literal arguments of the call. Its data type comes
            from a `.cast(...)`, or it is `Unchecked`, the data type of a
            column whose name is known and whose data type is not. The
            checker cannot tell a frame that was never typed from one that
            fell back to the plain type after an earlier operation, because
            both have the same type. With False, `.select()` on a plain
            frame keeps the plain type.
        disallow_implicit_inferred_fallback: Whether it is an error when an
            inferred return type falls back to the plain type, for each
            kind. See `ImplicitInferredFallbackPolicy`. The default allows
            it for every kind.
    """

    KNOWN_KEYS: ClassVar[frozenset[str]] = frozenset({
        "allow_unproven",
        "allow_unproven_overrides",
        "report_checker_gaps",
        "disallow_vertical_concat",
        "elevate_untyped_select",
        "disallow_implicit_inferred_fallback",
        "disallow_implicit_inferred_fallback_overrides",
    })

    allow_unproven: UnprovenPolicy = field(default_factory=UnprovenPolicy)
    disallow_vertical_concat: bool = False
    elevate_untyped_select: bool = True
    disallow_implicit_inferred_fallback: ImplicitInferredFallbackPolicy = field(
        default_factory=ImplicitInferredFallbackPolicy,
    )

    def cache_key(self) -> object:
        """Give a value that JSON can encode and that changes when a setting changes.

        The plugin gives it to mypy through `Plugin.report_config_data`. When
        the value changes between runs, mypy checks the cached modules again,
        so a new setting takes effect without a manual cache reset.

        Returns:
            A dictionary with every setting.

        Example:
            >>> key = AstrolabeConfig(disallow_vertical_concat=True).cache_key()
            >>> key["disallow_vertical_concat"]
            True
        """
        inferred_fallback = self.disallow_implicit_inferred_fallback.cache_key()
        return {
            "allow_unproven": self.allow_unproven.cache_key(),
            "disallow_vertical_concat": self.disallow_vertical_concat,
            "elevate_untyped_select": self.elevate_untyped_select,
            "disallow_implicit_inferred_fallback": inferred_fallback,
        }


class ConfigWarner:
    """Log one warning for each kind of wrong entry in `[tool.astrolabe]`.

    Each warning that the settings reader can give has its own method
    here, so the wording of all of them is in one place. A warning names
    the key, the value it found, and what applies in its place.
    """

    def __init__(self) -> None:
        """Get the logger that the warnings go to."""
        self._logger = logging.getLogger(__name__)

    def redefinition_not_allowed(self) -> None:
        """Warn that mypy runs without the setting `allow_redefinition`.

        A loop that assigns a new expression to the same variable, for example
        `x = x + 1`, gives a new expression type on each pass. The types reach
        a fixed point only if mypy lets the variable take a new type. Without
        `allow_redefinition`, mypy keeps the type of the first assignment, and
        the new assignment is a type error.
        """
        self._logger.warning(
            "astrolabe: [tool.mypy] allow_redefinition is off; set it to true, "
            "the expression types need it to converge in a loop that "
            "rebinds its variable"
        )

    def could_not_read(self, path: Path, exc: OSError | ValueError) -> None:
        """Warn that a TOML file cannot be read or is not valid TOML.

        Args:
            path: The TOML file.
            exc: The exception from the read or the parse.
        """
        self._logger.warning("astrolabe: could not read %s: %s", path, exc)

    def invalid_bool_setting(self, key: str, value: object, default: bool) -> None:
        """Warn that a setting that takes true or false has another value.

        Args:
            key: The key of the setting.
            value: The wrong value from `[tool.astrolabe]`.
            default: The default, which applies in its place.
        """
        self._logger.warning(
            "astrolabe: [tool.astrolabe] %s must be true or false, got %r. "
            "The default (%s) applies",
            key,
            value,
            default,
        )

    def invalid_level_setting(self, key: str, value: object, default: Level) -> None:
        """Warn that the value of a level setting is not the name of a level.

        Args:
            key: The key of the setting.
            value: The wrong value from `[tool.astrolabe]`.
            default: The level that applies in its place.
        """
        self._logger.warning(
            "astrolabe: [tool.astrolabe] %s must be %s, got %r. The default (%s) applies",
            key,
            LevelOverride.EXPECTED,
            value,
            default.spelling,
        )

    def invalid_overrides_table(self, overrides_key: str, value: object) -> None:
        """Warn that a sub-table of overrides is not a table.

        Args:
            overrides_key: The key of the sub-table.
            value: The wrong value from `[tool.astrolabe]`.
        """
        self._logger.warning(
            "astrolabe: [tool.astrolabe] %s must be a table, got %r. It is ignored",
            overrides_key,
            value,
        )

    def invalid_override_value(
        self,
        overrides_key: str,
        operation: str,
        value: object,
        expected: str,
    ) -> None:
        """Warn that the override of one operation has the wrong form.

        Args:
            overrides_key: The key of the sub-table.
            operation: The operation whose override is wrong.
            value: The wrong value from the sub-table.
            expected: A description of a correct value.
        """
        self._logger.warning(
            "astrolabe: [tool.astrolabe.%s] %s must be %s, got %r. It is ignored",
            overrides_key,
            operation,
            expected,
            value,
        )

    def unknown_override_key(
        self,
        overrides_key: str,
        key: str,
        known_operations: frozenset[str],
    ) -> None:
        """Warn that a key of a sub-table of overrides names no operation that it can override.

        Args:
            overrides_key: The key of the sub-table.
            key: The unknown key.
            known_operations: Each operation that the sub-table can override.
        """
        known = ", ".join(sorted(known_operations))
        self._logger.warning(
            "astrolabe: [tool.astrolabe.%s] unknown operation %r is ignored. Known operations: %s",
            overrides_key,
            key,
            known,
        )

    def unknown_setting(self, key: str, known_keys: frozenset[str]) -> None:
        """Warn that a key of `[tool.astrolabe]` names no setting.

        A key with a spelling mistake, or the key of a setting that a release
        deleted, gets this warning. The user then sees it, and the key does not
        do nothing in silence.

        Args:
            key: The unknown key.
            known_keys: Each key that `[tool.astrolabe]` accepts.
        """
        known = ", ".join(sorted(known_keys))
        self._logger.warning(
            "astrolabe: [tool.astrolabe] unknown setting %r is ignored. Known settings: %s",
            key,
            known,
        )


class BoolSetting:
    """Read one setting that takes true or false, and warn if it has another value."""

    def __init__(self, warner: ConfigWarner) -> None:
        """Keep the warner that reports a wrong value.

        Args:
            warner: The warner that logs a wrong value.
        """
        self._warner = warner

    def read(self, section: dict[str, Any], key: str, default: bool) -> bool:
        """Read one setting that takes true or false.

        Args:
            section: The `[tool.astrolabe]` table.
            key: The key of the setting.
            default: The value that applies if the key is missing or its value
                is not true or false.

        Returns:
            The value of the key, or the default. If the key has a value that
            is not true or false, the method logs a warning and gives the
            default.

        Example:
            >>> setting = BoolSetting(ConfigWarner())
            >>> setting.read({"strict": False}, "strict", True)
            False
            >>> setting.read({}, "strict", True)
            True
        """
        if key not in section:
            return default
        value = section[key]
        if not isinstance(value, bool):
            self._warner.invalid_bool_setting(key, value, default)
            return default
        return value


class OverrideValue[V]:
    """Read one value of a sub-table of overrides, where each sub-table holds one kind of value.

    A setting can have a sub-table that sets a different value for each
    operation. The level setting holds level names there, and the inferred
    fallback setting holds true or false. Each subclass reads one of those
    kinds, so the code that walks the sub-table is the same for both.

    Attributes:
        EXPECTED: A description of a correct value, for the warning.
    """

    EXPECTED: ClassVar[str]

    def parsed(self, raw: object) -> V | None:
        """Read one value from the TOML data.

        Args:
            raw: The value from the sub-table.

        Returns:
            The value, or None if it has the wrong form.

        Raises:
            NotImplementedError: Always. Each subclass reads its own kind of
                value.
        """
        raise NotImplementedError


class BoolOverride(OverrideValue[bool]):
    """Read an override that takes true or false.

    Example:
        >>> BoolOverride().parsed(False)
        False
        >>> BoolOverride().parsed("yes") is None
        True
    """

    EXPECTED: ClassVar[str] = "true or false"

    def parsed(self, raw: object) -> bool | None:
        """Read one value from the TOML data as true or false.

        Args:
            raw: The value from the sub-table.

        Returns:
            The value, or None for any value that is not true or false.
        """
        if isinstance(raw, bool):
            return raw
        return None


class LevelOverride(OverrideValue[Level]):
    """Read an override that names a level, by the lowercase name of the level.

    Example:
        >>> LevelOverride().parsed("stated")
        <Level.STATED: 2>
        >>> LevelOverride().parsed(True) is None
        True
    """

    EXPECTED: ClassVar[str] = "one of " + ", ".join(level.spelling for level in Level)

    def parsed(self, raw: object) -> Level | None:
        """Read one value from the TOML data as a level.

        Args:
            raw: The value from the sub-table.

        Returns:
            The level, or None for a value that is not the exact name of a
            level.
        """
        if isinstance(raw, str):
            return Level.named(raw)
        return None


class OperationOverrides[V]:
    """Read one sub-table of overrides for each operation, and warn about each wrong entry.

    For one setting, the key of the sub-table, the operations that it can
    override and the kind of its values never change. The class keeps them
    from its constructor, so the methods that read the table need only the
    table.
    """

    def __init__(
        self,
        warner: ConfigWarner,
        overrides_key: str,
        known_operations: frozenset[str],
        value: OverrideValue[V],
    ) -> None:
        """Keep what identifies the sub-table.

        Args:
            warner: The warner that logs a wrong entry.
            overrides_key: The key of the sub-table, for example
                `"allow_unproven_overrides"`.
            known_operations: Each operation that the sub-table can override.
            value: The reader for one value of the sub-table.
        """
        self._warner = warner
        self._overrides_key = overrides_key
        self._known_operations = known_operations
        self._value = value

    def resolved(self, section: dict[str, Any]) -> dict[str, V]:
        """Read the override of each operation from the sub-table.

        A missing sub-table gives no override. A sub-table that is not a
        table gives no override and a warning. For each key, a value with the
        wrong form gives a warning, and the operation keeps no override. A key
        that names no operation that the sub-table can override also gives a
        warning.

        Args:
            section: The `[tool.astrolabe]` table.

        Returns:
            The override of each operation that has a correct one.
        """
        table = section.get(self._overrides_key, {})
        if not isinstance(table, dict):
            self._warner.invalid_overrides_table(self._overrides_key, table)
            return {}
        overrides: dict[str, V] = {}
        for operation in self._known_operations:
            if operation not in table:
                continue
            value = self._value.parsed(table[operation])
            if value is None:
                self._warner.invalid_override_value(
                    self._overrides_key,
                    operation,
                    table[operation],
                    self._value.EXPECTED,
                )
                continue
            overrides[operation] = value
        for key in set(table) - self._known_operations:
            self._warner.unknown_override_key(self._overrides_key, key, self._known_operations)
        return overrides


@dataclass(frozen=True)
class PolicySettingSpec:
    """The keys and the default of one true-or-false setting that has overrides.

    Attributes:
        key: The key of the general setting, for example
            `"disallow_implicit_inferred_fallback"`.
        overrides_key: The key of the sub-table of overrides.
        default: The default of the general setting.
        known_operations: Each operation that the sub-table can override.
    """

    key: str
    overrides_key: str
    default: bool
    known_operations: frozenset[str]


class PolicySetting[P]:
    """Read one true-or-false setting and its overrides into a policy object.

    The reader reads the general value first, then the overrides. It then
    builds the policy object from the two. The class is generic over that
    object, and it keeps every fact about the setting from its constructor.
    """

    def __init__(
        self,
        warner: ConfigWarner,
        build: Callable[[bool, dict[str, bool]], P],
        spec: PolicySettingSpec,
    ) -> None:
        """Keep what identifies the setting, and build the readers for its parts.

        Args:
            warner: The warner that logs a wrong entry.
            build: The policy class. The method calls it with the default and
                the overrides, in that order.
            spec: The keys and the default of the setting.
        """
        self._bool_setting = BoolSetting(warner)
        self._build = build
        self._spec = spec
        self._overrides = OperationOverrides(
            warner,
            spec.overrides_key,
            spec.known_operations,
            BoolOverride(),
        )

    def resolved(self, section: dict[str, Any]) -> P:
        """Build the policy from the `[tool.astrolabe]` table.

        Args:
            section: The `[tool.astrolabe]` table.

        Returns:
            The policy: the general value from `spec.key`, with each override
            from the sub-table that `spec.overrides_key` names.
        """
        default = self._bool_setting.read(section, self._spec.key, self._spec.default)
        overrides = self._overrides.resolved(section)
        return self._build(default, overrides)


class LevelSetting:
    """Read one setting that names a level, and warn if its value names none."""

    def __init__(self, warner: ConfigWarner) -> None:
        """Keep the warner that reports a wrong value.

        Args:
            warner: The warner that logs a wrong value.
        """
        self._warner = warner

    def read(self, section: dict[str, Any], key: str, default: Level) -> Level:
        """Read one level setting.

        Args:
            section: The `[tool.astrolabe]` table.
            key: The key of the setting.
            default: The level that applies if the key is missing or its value
                names no level.

        Returns:
            The level that the value names, or the default. If the value names
            no level, the method logs a warning and gives the default.

        Example:
            >>> setting = LevelSetting(ConfigWarner())
            >>> setting.read({"allow_unproven": "none"}, "allow_unproven", Level.OMITTED)
            <Level.NONE: 1>
            >>> setting.read({}, "allow_unproven", Level.OMITTED)
            <Level.OMITTED: 3>
        """
        if key not in section:
            return default
        level = LevelOverride().parsed(section[key])
        if level is None:
            self._warner.invalid_level_setting(key, section[key], default)
            return default
        return level


class UnprovenSetting:
    """Read `allow_unproven`, its overrides and `report_checker_gaps` into one `UnprovenPolicy`.

    These three keys decide which unproven results the checker lets pass.
    An unproven result is one where an operation gave up on a proof, for
    example because a column name is not a literal. `allow_unproven` gives
    the level for all operations, and its sub-table sets a level for one
    operation. `report_checker_gaps` decides whether the checker reports a
    case that the plugin does not model.

    Example:
        >>> section = {"allow_unproven": "stated", "report_checker_gaps": True}
        >>> UnprovenSetting(ConfigWarner()).resolved(section)
        UnprovenPolicy(level=<Level.STATED: 2>, overrides={}, report_gaps=True)
    """

    KEY: ClassVar[str] = "allow_unproven"
    OVERRIDES_KEY: ClassVar[str] = "allow_unproven_overrides"
    GAPS_KEY: ClassVar[str] = "report_checker_gaps"

    def __init__(self, warner: ConfigWarner) -> None:
        """Build the readers for the three keys.

        Args:
            warner: The warner that logs a wrong entry.
        """
        self._level = LevelSetting(warner)
        self._gaps = BoolSetting(warner)
        self._overrides = OperationOverrides(
            warner,
            self.OVERRIDES_KEY,
            UnprovenPolicy.KNOWN_OPERATIONS,
            LevelOverride(),
        )

    def resolved(self, section: dict[str, Any]) -> UnprovenPolicy:
        """Build the policy from the `[tool.astrolabe]` table.

        Args:
            section: The `[tool.astrolabe]` table.

        Returns:
            The policy: the global level, the level of each operation that has
            an override, and whether to report gaps. Each part has its default
            if it is missing or wrong.
        """
        default = UnprovenPolicy()
        return UnprovenPolicy(
            self._level.read(section, self.KEY, default.level),
            self._overrides.resolved(section),
            self._gaps.read(section, self.GAPS_KEY, default.report_gaps),
        )


class ConfigPathResolver:
    """Find the `pyproject.toml` file that holds `[tool.astrolabe]`."""

    def resolve(self, options: Options) -> Path | None:
        """Find the `pyproject.toml` file to read.

        The plugin uses the same file as mypy when mypy reads its own settings
        from a `pyproject.toml`. Otherwise it looks in the current directory.

        Args:
            options: The options of the mypy run.

        Returns:
            The settings file of mypy, if it is a `pyproject.toml`. Otherwise
            the `pyproject.toml` in the current directory, or None if there is
            none.
        """
        config_file = options.config_file
        if config_file and config_file.endswith("pyproject.toml"):
            return Path(config_file)
        candidate = Path.cwd() / "pyproject.toml"
        return candidate if candidate.is_file() else None


class AstrolabeConfigLoader:
    """Build the `AstrolabeConfig` for a mypy run, from a TOML file or from the options of mypy.

    The loader finds the file, parses it, takes the `[tool.astrolabe]`
    table and reads each setting. A missing file, a file that is not valid
    TOML, or a missing table gives the defaults.
    """

    def __init__(self) -> None:
        """Build the warner and the file finder that the loader needs."""
        self._warner = ConfigWarner()
        self._path_resolver = ConfigPathResolver()

    def load(self, path: Path | None) -> AstrolabeConfig:
        r"""Read `[tool.astrolabe]` from one TOML file.

        Args:
            path: The TOML file, or None.

        Returns:
            The settings. The defaults, `AstrolabeConfig()`, if the path is
            None, if the file is missing or cannot be read, if it is not valid
            TOML, or if it has no `[tool.astrolabe]` table.

        Example:
            >>> import tempfile
            >>> content = "[tool.astrolabe]\ndisallow_vertical_concat = true\n"
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = Path(tmp) / "pyproject.toml"
            ...     _ = path.write_text(content)
            ...     AstrolabeConfigLoader().load(path).disallow_vertical_concat
            True
        """
        if path is None or not path.is_file():
            return AstrolabeConfig()
        data = self._load_toml(path)
        if data is None:
            return AstrolabeConfig()
        return self._config_from_section(self._astrolabe_section(data))

    def load_for_options(self, options: Options) -> AstrolabeConfig:
        """Read the settings for one mypy run.

        Args:
            options: The options of the mypy run.

        Returns:
            The settings.
        """
        return self.load(self._path_resolver.resolve(options))

    def _load_toml(self, path: Path) -> dict[str, Any] | None:
        """Parse a TOML file.

        Args:
            path: The TOML file.

        Returns:
            The parsed data, or None if the file cannot be read, is not valid
            UTF-8, or is not valid TOML. In that case, the method logs a warning.
        """
        try:
            return tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            self._warner.could_not_read(path, exc)
            return None

    def _astrolabe_section(self, data: dict[str, Any]) -> dict[str, Any]:
        """Get the `[tool.astrolabe]` table from the parsed data.

        Args:
            data: The parsed TOML data.

        Returns:
            The `[tool.astrolabe]` table, or an empty dictionary if it is
            missing or is not a table.
        """
        tool = data.get("tool", {})
        if not isinstance(tool, dict):
            return {}
        astrolabe = tool.get("astrolabe", {})
        if not isinstance(astrolabe, dict):
            return {}
        return astrolabe

    def _config_from_section(self, section: dict[str, Any]) -> AstrolabeConfig:
        """Build the settings from the `[tool.astrolabe]` table.

        The method reads each setting on its own, and uses its default if it
        is missing or wrong. It logs a warning for each key that names no
        setting. The policy of `disallow_implicit_inferred_fallback` takes its
        general value from that key, with each override from
        `[tool.astrolabe.disallow_implicit_inferred_fallback_overrides]`.

        Args:
            section: The `[tool.astrolabe]` table.

        Returns:
            The settings.
        """
        for key in sorted(set(section) - AstrolabeConfig.KNOWN_KEYS):
            self._warner.unknown_setting(key, AstrolabeConfig.KNOWN_KEYS)
        bool_setting = BoolSetting(self._warner)
        fallback_spec = PolicySettingSpec(
            "disallow_implicit_inferred_fallback",
            "disallow_implicit_inferred_fallback_overrides",
            ImplicitInferredFallbackPolicy.default,
            ImplicitInferredFallbackPolicy.KNOWN_OPERATIONS,
        )
        fallback_setting = PolicySetting(
            self._warner,
            ImplicitInferredFallbackPolicy,
            fallback_spec,
        )
        return AstrolabeConfig(
            allow_unproven=UnprovenSetting(self._warner).resolved(section),
            disallow_vertical_concat=bool_setting.read(
                section,
                "disallow_vertical_concat",
                AstrolabeConfig.disallow_vertical_concat,
            ),
            elevate_untyped_select=bool_setting.read(
                section,
                "elevate_untyped_select",
                AstrolabeConfig.elevate_untyped_select,
            ),
            disallow_implicit_inferred_fallback=fallback_setting.resolved(section),
        )
