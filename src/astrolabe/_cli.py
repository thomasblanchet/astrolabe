"""The `astrolabe init` command, which sets up the `pyproject.toml` of a project for the plugin.

A project needs three things before the checker can run: the plugin in the
`plugins` list of `[tool.mypy]`, `allow_redefinition = true`, and the
`[tool.astrolabe]` settings. This command writes all three, so that a user
does not have to copy them by hand. The `[tool.astrolabe]` table lists every
setting at its default, each with a comment that says what it does.

The command edits the file with tomlkit, a TOML library that keeps the
comments, the order and the format of a file as they are. A second run finds
nothing to change. If mypy reads its configuration from another file, the
command changes nothing and shows the lines to add there.
"""

from __future__ import annotations

import argparse
import configparser
import tomllib
import sys
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TextIO

import tomlkit
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import Array, Table

from astrolabe import plugin as plugin_module
from astrolabe._hooks.config import AstrolabeConfig
from astrolabe._hooks.unproven import Level

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomlkit.toml_document import TOMLDocument


class InitRefusal(Exception):
    """The base of each reason why the command stops and changes nothing.

    The command catches this class, shows the message, and exits with 1.
    Each subclass builds its message from its own arguments, so the text of
    each refusal stays in one place.
    """


class MissingDirectoryRefusal(InitRefusal):
    """The project directory does not exist."""

    def __init__(self, root: Path) -> None:
        """Build the message that names the directory.

        Args:
            root: The directory that the command line gives.

        Example:
            >>> str(MissingDirectoryRefusal(Path("app")))
            'the directory app does not exist'
        """
        super().__init__(f"the directory {root} does not exist")


class InvalidTomlRefusal(InitRefusal):
    """The `pyproject.toml` file is not valid TOML, so the command cannot edit it."""

    def __init__(self, path: Path, error: TOMLKitError) -> None:
        """Build the message that names the file and the parse error.

        Args:
            path: The file.
            error: The error from tomlkit.
        """
        super().__init__(f"{path} is not valid TOML: {error}")


class NotATableRefusal(InitRefusal):
    """A key that the command must write into holds a value that is not a table.

    For example, `tool = 1` leaves no place for `[tool.mypy]`. The command does
    not replace the value, because the user wrote it.
    """

    def __init__(self, dotted: str) -> None:
        """Build the message that names the key.

        Args:
            dotted: The full dotted name of the key, for example `tool.mypy`.

        Example:
            >>> str(NotATableRefusal("tool.mypy"))
            '"tool.mypy" in pyproject.toml is not a table'
        """
        super().__init__(f'"{dotted}" in pyproject.toml is not a table')


class CompetingConfigRefusal(InitRefusal):
    """mypy reads its configuration from a file other than `pyproject.toml`.

    A `[tool.mypy]` table then has no effect, or it takes the place of the
    configuration that the user already has. So the command changes nothing,
    and the message gives the lines to add to the other file by hand.
    """

    def __init__(self, competing: Path, plugin: str) -> None:
        """Build the message, with the lines to add to the other file.

        Args:
            competing: The file that mypy reads its configuration from.
            plugin: The module name of the plugin.

        Example:
            >>> print(CompetingConfigRefusal(Path("mypy.ini"), "astrolabe.plugin"))
            mypy reads its configuration from mypy.ini, so it ignores
            [tool.mypy] in pyproject.toml. Nothing is changed.
            Add these lines to the [mypy] section of mypy.ini:
                plugins = astrolabe.plugin
                allow_redefinition = True
            The plugin reads its own [tool.astrolabe] settings from pyproject.toml.
        """
        super().__init__(
            f"mypy reads its configuration from {competing.name}, so it ignores\n"
            f"[tool.mypy] in pyproject.toml. Nothing is changed.\n"
            f"Add these lines to the [mypy] section of {competing.name}:\n"
            f"    plugins = {plugin}\n"
            f"    allow_redefinition = True\n"
            f"The plugin reads its own [tool.astrolabe] settings from pyproject.toml."
        )


class Tables:
    """Find a table inside another table of a tomlkit document, or add it where it is missing.

    The command writes into `[tool.mypy]` and `[tool.astrolabe]`, and a file
    can have both, one or neither of them. This class gives each table,
    whether it exists or not, so that the code that writes the settings does
    not need to know.

    A table that this class adds inside an existing table needs a blank line
    at its end, to keep it apart from the next table in the file. The class
    adds that line in `close`, once the table holds its settings. tomlkit
    already keeps apart the tables inside a table that this class added.
    """

    def __init__(self) -> None:
        """Start with no added table."""
        self._made_tables: list[Table] = []
        self._to_close: list[Table] = []

    def child(
        self,
        parent: MutableMapping[str, object],
        key: str,
        *,
        dotted: str,
    ) -> MutableMapping[str, object]:
        r"""Give the table under `key`, and add an empty one where it is missing.

        The `tool` table is a super table: it holds only other tables. So the
        file shows `[tool.mypy]`, and not an empty `[tool]` header above it. A
        new table inside a table that this object added needs no blank line of
        its own, because tomlkit keeps it apart.

        Args:
            parent: The table that holds it.
            key: The key of the table.
            dotted: The full dotted name of the table, for the message of a
                refusal.

        Returns:
            The table.

        Raises:
            NotATableRefusal: If `key` holds a value that is not a table.

        Example:
            >>> document = tomlkit.parse("[tool.ruff]\nline-length = 100\n")
            >>> tool = Tables().child(document, "tool", dotted="tool")
            >>> mypy = Tables().child(tool, "mypy", dotted="tool.mypy")
            >>> sorted(tool)
            ['mypy', 'ruff']
            >>> Tables().child(tomlkit.document(), "tool", dotted="tool").is_super_table()
            True
            >>> Tables().child(tomlkit.parse("tool = 1\n"), "tool", dotted="tool")
            Traceback (most recent call last):
            ...
            astrolabe._cli.NotATableRefusal: "tool" in pyproject.toml is not a table
        """
        value = parent.get(key)
        if isinstance(value, MutableMapping):
            return value
        if value is not None:
            raise NotATableRefusal(dotted)
        table = tomlkit.table(is_super_table=key == "tool")
        if not any(parent is made for made in self._made_tables):
            self._to_close.append(table)
        parent[key] = table
        self._made_tables.append(table)
        return table

    def close(self) -> None:
        r"""End each added table with a blank line, once the table holds its settings.

        Example:
            >>> tables = Tables()
            >>> document = tomlkit.parse('[project]\nname = "x"\n')
            >>> tables.child(document, "mypy", dotted="mypy")["strict"] = True
            >>> tables.close()
            >>> tomlkit.dumps(document).endswith("strict = true\n\n")
            True
        """
        for table in self._to_close:
            table.add(tomlkit.nl())


class MypySection:
    """Add the plugin and `allow_redefinition = true` to `[tool.mypy]`.

    mypy loads the plugin only when `plugins` names it. The checker needs
    `allow_redefinition`: without it, mypy reports a loop that rebinds a
    variable to a new expression type. The class keeps every other plugin and
    setting that the table holds.
    """

    PLUGIN: ClassVar[str] = plugin_module.__name__

    def applied(self, table: MutableMapping[str, object]) -> list[str]:
        r"""Add the plugin and the setting where the table does not have them yet.

        Args:
            table: The `[tool.mypy]` table.

        Returns:
            One line for each change. The list is empty if the table already has
            both.

        Example:
            >>> table = tomlkit.parse('plugins = ["other.plugin"]\n')
            >>> for change in MypySection().applied(table):
            ...     print(change)
            add "astrolabe.plugin" to tool.mypy.plugins
            set tool.mypy.allow_redefinition = true
            >>> table["plugins"]
            ['other.plugin', 'astrolabe.plugin']
            >>> MypySection().applied(table)
            []
        """
        return [*self._plugins(table), *self._redefinition(table)]

    def _plugins(self, table: MutableMapping[str, object]) -> list[str]:
        r"""Add the plugin to the end of the `plugins` list, if the list does not name it.

        If the value is already an array, the method adds to it in place, so the
        array keeps its format and its comments. A string becomes an array.

        Args:
            table: The `[tool.mypy]` table.

        Returns:
            The line for the change, or an empty list.

        Example:
            >>> table = tomlkit.parse('plugins = "a.plugin"\n')
            >>> MypySection()._plugins(table)
            ['add "astrolabe.plugin" to tool.mypy.plugins']
            >>> table["plugins"].as_string()
            '["a.plugin", "astrolabe.plugin"]'
        """
        value = table.get("plugins")
        if self.PLUGIN in self._listed(value):
            return []
        if isinstance(value, Array):
            value.append(self.PLUGIN)
            array = value
        else:
            array = tomlkit.array()
            array.extend([*self._listed(value), self.PLUGIN])
        table["plugins"] = array
        return [f'add "{self.PLUGIN}" to tool.mypy.plugins']

    def _listed(self, value: object) -> list[str]:
        """Read the `plugins` value as mypy reads it.

        mypy accepts a TOML array of names, or one string with the names
        separated by commas.

        Args:
            value: The `plugins` value, or None if the table has none.

        Returns:
            The names of the plugins, with no empty name.

        Example:
            >>> MypySection()._listed("a.plugin, b.plugin,")
            ['a.plugin', 'b.plugin']
            >>> MypySection()._listed(["a.plugin"])
            ['a.plugin']
            >>> MypySection()._listed(None)
            []
        """
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    def _redefinition(self, table: MutableMapping[str, object]) -> list[str]:
        r"""Set `allow_redefinition = true`, if it is not true already.

        The line for the change gives the old value, if the table had one.

        Args:
            table: The `[tool.mypy]` table.

        Returns:
            The line for the change, or an empty list.

        Example:
            >>> MypySection()._redefinition(tomlkit.parse("allow_redefinition = false\n"))
            ['set tool.mypy.allow_redefinition = true (it was false)']
            >>> MypySection()._redefinition(tomlkit.document())
            ['set tool.mypy.allow_redefinition = true']
        """
        current = table.get("allow_redefinition")
        if current is True:
            return []
        table["allow_redefinition"] = True
        was = "" if current is None else f" (it was {tomlkit.item(current).as_string()})"
        return [f"set tool.mypy.allow_redefinition = true{was}"]


@dataclass(frozen=True)
class AstrolabeEntry:
    """One setting of `[tool.astrolabe]`, and the comment that the command writes above it.

    Attributes:
        key: The key of the setting.
        value: The value to write.
        comment: What the setting does, in one line.
    """

    key: str
    value: str | bool
    comment: str

    def spelled(self) -> str:
        """Give the value as the TOML file shows it.

        Returns:
            The TOML text of the value, for the line that reports the change.

        Example:
            >>> AstrolabeEntry("allow_unproven", "omitted", "").spelled()
            '"omitted"'
            >>> AstrolabeEntry("report_checker_gaps", False, "").spelled()
            'false'
        """
        return tomlkit.item(self.value).as_string()


class AstrolabeSection:
    """Write each setting of `[tool.astrolabe]` that the table does not have yet.

    A user who opens the file sees every setting, its default and a comment,
    so no setting stays hidden. The class never changes a value that the user
    wrote, with one exception: `--allow-unproven` on the command line
    replaces the level in the file.
    """

    def __init__(self, level: Level | None) -> None:
        """Keep the level that the command line asks for.

        Args:
            level: The level of `--allow-unproven`, or None if the command line
                does not give one.
        """
        self._level = level

    def entries(self) -> list[AstrolabeEntry]:
        """List every setting, in the order that the file shows them.

        Each value is the default of `AstrolabeConfig`, the class that the plugin
        reads its settings with. So the file and the plugin always agree on the
        defaults. The level from the command line, if it gives one, replaces
        the default level.

        Returns:
            The settings.

        Example:
            >>> [entry.key for entry in AstrolabeSection(None).entries()][:2]
            ['allow_unproven', 'report_checker_gaps']
            >>> AstrolabeSection(Level.NEVER).entries()[0].value
            'never'
        """
        defaults = AstrolabeConfig()
        levels = " | ".join(level.spelling for level in Level)
        level = defaults.allow_unproven.level if self._level is None else self._level
        return [
            AstrolabeEntry(
                "allow_unproven",
                level.spelling,
                f"What passes without a proof: {levels}",
            ),
            AstrolabeEntry(
                "report_checker_gaps",
                defaults.allow_unproven.report_gaps,
                "Report the cases that the checker does not model, at every level",
            ),
            AstrolabeEntry(
                "disallow_vertical_concat",
                defaults.disallow_vertical_concat,
                'Report pl.concat(how="vertical"), which the plugin does not narrow',
            ),
            AstrolabeEntry(
                "elevate_untyped_select",
                defaults.elevate_untyped_select,
                "Type .select() on a plain frame, each name under Unchecked",
            ),
            AstrolabeEntry(
                "disallow_implicit_inferred_fallback",
                defaults.disallow_implicit_inferred_fallback.default,
                "Report an inferred function whose type falls back to the plain type",
            ),
        ]

    def applied(self, table: MutableMapping[str, object]) -> list[str]:
        r"""Add each missing setting, and set the level from the command line.

        A missing setting gets a comment line above it.

        Args:
            table: The `[tool.astrolabe]` table.

        Returns:
            One line for each change.

        Example:
            >>> table = tomlkit.parse("disallow_vertical_concat = true\n")
            >>> AstrolabeSection(None).applied(table)[0]
            'set tool.astrolabe.allow_unproven = "omitted"'
            >>> table["disallow_vertical_concat"]
            True
            >>> fresh = tomlkit.table()
            >>> AstrolabeSection(None).applied(fresh)[1]
            'set tool.astrolabe.report_checker_gaps = false'
            >>> print(fresh.as_string().splitlines()[2])
            # Report the cases that the checker does not model, at every level
        """
        changes = self._level_change(table)
        for entry in self.entries():
            if entry.key in table:
                continue
            if isinstance(table, Table):
                table.add(tomlkit.comment(entry.comment))
            table[entry.key] = entry.value
            changes.append(f"set tool.astrolabe.{entry.key} = {entry.spelled()}")
        return changes

    def _level_change(self, table: MutableMapping[str, object]) -> list[str]:
        r"""Replace the level in the file with the level from the command line.

        If the file has no level, this method does nothing: `applied` then writes
        the level from the command line, with its comment.

        Args:
            table: The `[tool.astrolabe]` table.

        Returns:
            The line for the change, or an empty list.

        Example:
            >>> table = tomlkit.parse('allow_unproven = "none"\n')
            >>> AstrolabeSection(Level.NEVER)._level_change(table)
            ['set tool.astrolabe.allow_unproven = "never" (it was "none")']
            >>> AstrolabeSection(None)._level_change(table)
            []
        """
        current = table.get("allow_unproven")
        if self._level is None or current is None or current == self._level.spelling:
            return []
        table["allow_unproven"] = self._level.spelling
        return [
            f'set tool.astrolabe.allow_unproven = "{self._level.spelling}" (it was "{current}")',
        ]


class MypyConfigLocator:
    """Find a mypy configuration file that stops the command from using `pyproject.toml`.

    mypy uses the first file that configures it, in this order: `mypy.ini`,
    `.mypy.ini`, `pyproject.toml` with a `[tool.mypy]` table, and `setup.cfg`
    with a `[mypy]` section. If an ini file exists, mypy never reads
    `[tool.mypy]`, so the changes of the command have no effect. If
    `setup.cfg` configures mypy and `pyproject.toml` does not, a new
    `[tool.mypy]` table takes its place, and mypy stops reading the settings
    of the user. The command refuses in both cases. If `pyproject.toml`
    already has a `[tool.mypy]` table, mypy reads it and ignores `setup.cfg`,
    so the command goes on.
    """

    INI_FILES: ClassVar[tuple[str, ...]] = ("mypy.ini", ".mypy.ini")
    SETUP_CFG: ClassVar[str] = "setup.cfg"

    def __init__(self, root: Path) -> None:
        """Keep the project directory.

        Args:
            root: The project directory.
        """
        self._root = root

    def refuse_competing(self) -> None:
        """Stop the command if another file configures mypy.

        Raises:
            CompetingConfigRefusal: If `mypy.ini` or `.mypy.ini` exists, or
                `setup.cfg` has a `[mypy]` section and `pyproject.toml` has
                no `[tool.mypy]` table.

        Example:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     MypyConfigLocator(Path(tmp)).refuse_competing()
        """
        competing = self._ini_file()
        if competing is None and not self._pyproject_configures_mypy():
            competing = self._setup_cfg()
        if competing is not None:
            raise CompetingConfigRefusal(competing, MypySection.PLUGIN)

    def _ini_file(self) -> Path | None:
        """Find `mypy.ini` or `.mypy.ini` in the project directory.

        Returns:
            The first of the two files that exists, or None.
        """
        for name in self.INI_FILES:
            path = self._root / name
            if path.is_file():
                return path
        return None

    def _pyproject_configures_mypy(self) -> bool:
        """Tell whether `pyproject.toml` already has a `[tool.mypy]` table.

        A file that is not valid TOML has no table here. The command reads
        the file again later and reports it then.

        Returns:
            True if the file exists and configures mypy.
        """
        path = self._root / ProjectFile.NAME
        if not path.is_file():
            return False
        try:
            tool = tomllib.loads(path.read_text(encoding="utf-8")).get("tool", {})
        except tomllib.TOMLDecodeError:
            return False
        return isinstance(tool, dict) and "mypy" in tool

    def _setup_cfg(self) -> Path | None:
        """Find a `setup.cfg` that configures mypy.

        Returns:
            The file, if it has a `[mypy]` section. Else None.
        """
        path = self._root / self.SETUP_CFG
        if not path.is_file():
            return None
        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        if parser.has_section("mypy"):
            return path
        return None


class ProjectFile:
    """Read and write the `pyproject.toml` of one project.

    If the file does not exist, the command starts from an empty document and
    writes a new file.
    """

    NAME: ClassVar[str] = "pyproject.toml"

    def __init__(self, root: Path) -> None:
        """Keep the path of the file.

        Args:
            root: The project directory.
        """
        self._path = root / self.NAME

    @property
    def exists(self) -> bool:
        """Tell whether the file exists.

        Returns:
            True if the file exists.
        """
        return self._path.is_file()

    def read(self) -> TOMLDocument:
        """Parse the file, or give an empty document if the file does not exist.

        Returns:
            The document.

        Raises:
            InvalidTomlRefusal: If the file is not valid TOML.
        """
        if not self.exists:
            return tomlkit.document()
        try:
            return tomlkit.parse(self._path.read_text(encoding="utf-8"))
        except TOMLKitError as error:
            raise InvalidTomlRefusal(self._path, error) from error

    def write(self, document: TOMLDocument) -> None:
        """Write the document, with exactly one newline at the end of the file.

        Args:
            document: The edited document.
        """
        text = tomlkit.dumps(document).strip("\n") + "\n"
        self._path.write_text(text, encoding="utf-8")


class ProjectSetup:
    """Set up one project: read the file, edit it, and write it unless the run is a dry run.

    The class runs the steps in an order that keeps the file safe. Each check
    that can refuse runs before the command writes anything, so a refusal
    leaves the file as it was.
    """

    def __init__(self, root: Path, level: Level | None, *, dry_run: bool) -> None:
        """Keep what the command line asks for.

        Args:
            root: The project directory.
            level: The level of `--allow-unproven`, or None.
            dry_run: True to report the changes and write nothing.
        """
        self._root = root
        self._level = level
        self._dry_run = dry_run
        self._tables = Tables()

    def changes(self) -> list[str]:
        """Set up the project, and give the list of changes.

        The command writes the file only if something changed and the run is not
        a dry run. An `InitRefusal` from a step passes to the caller, and the
        command then writes nothing. A step refuses if the directory does not
        exist, if mypy reads another configuration file, or if `pyproject.toml`
        cannot be edited.

        Returns:
            One line for each change. The list is empty if the project is already
            set up.

        Raises:
            MissingDirectoryRefusal: If the project directory does not exist.
        """
        if not self._root.is_dir():
            raise MissingDirectoryRefusal(self._root)
        MypyConfigLocator(self._root).refuse_competing()
        project = ProjectFile(self._root)
        document = project.read()
        changes = [] if project.exists else [f"create {ProjectFile.NAME}"]
        changes.extend(self._edited(document))
        if changes and not self._dry_run:
            project.write(document)
        return changes

    def _edited(self, document: TOMLDocument) -> list[str]:
        """Edit `[tool.mypy]` and `[tool.astrolabe]` in the document.

        Args:
            document: The parsed file.

        Returns:
            One line for each change.
        """
        tool = self._tables.child(document, "tool", dotted="tool")
        mypy = self._tables.child(tool, "mypy", dotted="tool.mypy")
        changes = MypySection().applied(mypy)
        astrolabe = self._tables.child(tool, "astrolabe", dotted="tool.astrolabe")
        changes.extend(AstrolabeSection(self._level).applied(astrolabe))
        self._tables.close()
        return changes


class InitCommand:
    """The `astrolabe` console script, which has one subcommand: `init`.

    It reads the command line, runs the setup, and shows each change. A
    refusal goes to the error stream with the exit code 1. The two streams
    are arguments, so a test can read what the command shows.
    """

    PROGRAM: ClassVar[str] = "astrolabe"

    def __init__(self, out: TextIO, err: TextIO) -> None:
        """Keep the two output streams.

        Args:
            out: The stream for the lines of the changes.
            err: The stream for a refusal.
        """
        self._out = out
        self._err = err

    @classmethod
    def main(cls) -> int:
        """Run the command with the arguments of the process.

        The console script calls this method.

        Returns:
            The exit code.
        """
        return cls(sys.stdout, sys.stderr).run(sys.argv[1:])

    def run(self, argv: Sequence[str]) -> int:
        """Run the command.

        Args:
            argv: The arguments after the name of the program.

        Returns:
            0 if the project is set up, or 1 if the command refuses. For a bad
            argument, `argparse` exits with the code 2.
        """
        args = self._parser().parse_args(argv)
        level = None if args.allow_unproven is None else Level.named(args.allow_unproven)
        setup = ProjectSetup(args.path, level, dry_run=args.dry_run)
        try:
            changes = setup.changes()
        except InitRefusal as refusal:
            print(f"astrolabe init: {refusal}", file=self._err)
            return 1
        self._report(changes, dry_run=args.dry_run)
        return 0

    def _parser(self) -> argparse.ArgumentParser:
        """Build the parser of the command line.

        Returns:
            The parser, with the `init` subcommand and its options.

        Example:
            >>> args = InitCommand(sys.stdout, sys.stderr)._parser().parse_args(["init"])
            >>> args.path, args.allow_unproven, args.dry_run
            (PosixPath('.'), None, False)
        """
        parser = argparse.ArgumentParser(prog=self.PROGRAM)
        commands = parser.add_subparsers(dest="command", required=True)
        init = commands.add_parser(
            "init",
            help="Set up pyproject.toml for the astrolabe mypy plugin.",
        )
        init.add_argument(
            "--path",
            type=Path,
            default=Path(),
            help="The project directory. The default is the current directory.",
        )
        init.add_argument(
            "--allow-unproven",
            choices=[level.spelling for level in Level],
            help="The allow_unproven level to write. The default is omitted.",
        )
        init.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the changes and write nothing.",
        )
        return parser

    def _report(self, changes: list[str], *, dry_run: bool) -> None:
        """Show the line of each change, or say that the project is already set up.

        After a dry run, one more line says that the file is not changed.

        Args:
            changes: The lines for the changes.
            dry_run: True if the command wrote nothing.

        Example:
            >>> InitCommand(sys.stdout, sys.stderr)._report(["create x"], dry_run=True)
            create x
            Dry run: pyproject.toml is not changed.
            >>> InitCommand(sys.stdout, sys.stderr)._report([], dry_run=False)
            pyproject.toml is already set up.
        """
        if not changes:
            print(f"{ProjectFile.NAME} is already set up.", file=self._out)
            return
        for change in changes:
            print(change, file=self._out)
        if dry_run:
            print(f"Dry run: {ProjectFile.NAME} is not changed.", file=self._out)
