"""The specification documents agree with each other, with the tests, and with polars.

Writing several documents that agree once is achievable by care. Keeping them
agreeing is not, and the defects this suite exists to prevent were all of that
second kind: a rule cited by a name no section defined, a count of "61" where
there were 28, and two measurements whose values summed to more than the space
they counted.

Each test below states one invariant. A failure means a document is wrong, not
that a test needs relaxing.
"""

from __future__ import annotations

import re
from pathlib import Path

import inspect

import polars as pl
import polars.selectors as cs
import pytest

from astrolabe.remainder import Prefixed, Suffixed
from astrolabe._rules.releases import PolarsRelease
from spec.tests.documents import ROOT, Document, Glossary, LeanSources, Ledger

from spec.tests.test_measurements import MIXED
from spec.tools.node_census.calls import CensusCalls

FRAMES = Document("DATAFRAME_SPECIFICATION.md")
EXPRESSIONS = Document("EXPRESSION_SPECIFICATION.md")
OPERATIONS = Document("OPERATIONS.md")
CATALOGUE = Document("OPERATION_CATALOG.md")
ALL = (FRAMES, EXPRESSIONS, OPERATIONS, CATALOGUE)

GLOSSARIES = (
    Glossary("DATAFRAME_SPECIFICATION.md", "## 13. Glossary"),
    Glossary("EXPRESSION_SPECIFICATION.md", "## 12. Glossary"),
)


class PolarsApi:
    """The public surface of polars, as the census has to cover it.

    Example:
        >>> "join" in PolarsApi().frame_methods()
        True
    """

    GENERIC_FRAME_FUNCTIONS = frozenset({"align_frames", "concat"})

    HELPERS = frozenset({
        "expand_selector",
        "is_column",
        "is_polars_dtype",
        "is_selector",
        "or_",
        "overload",
        "re_escape",
        "reduce",
        "unstable",
        "NoReturn",
    })

    def frame_methods(self) -> set[str]:
        """Give every public method of `DataFrame` and `LazyFrame`.

        Returns:
            The union of the two classes' public callable attributes.

        Example:
            >>> len(PolarsApi().frame_methods()) > 100
            True
        """
        return self._public(pl.DataFrame) | self._public(pl.LazyFrame)

    def _public(self, cls: type) -> set[str]:
        """Give one class's public callable attributes.

        Args:
            cls: The class to read.

        Returns:
            Each public attribute name that is callable.

        Example:
            >>> "select" in PolarsApi()._public(pl.DataFrame)
            True
            >>> "_df" in PolarsApi()._public(pl.DataFrame)
            False
        """
        found = set()
        for name in dir(cls):
            if not name.startswith("_") and callable(getattr(cls, name, None)):
                found.add(name)
        return found

    def frame_functions(self) -> set[str]:
        """Give every `pl.*` function that returns a frame.

        The return annotation decides, a `Series` first in a union
        excluded. `concat` and `align_frames` return a type variable
        over frames, so the annotation does not show them, and they are
        added by name.

        Returns:
            The function names.

        Example:
            >>> {"read_csv", "concat"} <= PolarsApi().frame_functions()
            True
        """
        annotated = {name for name in dir(pl) if self._returns_frame(name)}
        return annotated | self.GENERIC_FRAME_FUNCTIONS

    def _returns_frame(self, name: str) -> bool:
        """Answer whether one `pl.*` function is annotated to return a frame.

        A function with no readable signature has no annotation, so it
        does not count.

        Args:
            name: The attribute's name.

        Returns:
            True for a public function whose return annotation names a
            frame before any `Series`.

        Example:
            >>> PolarsApi()._returns_frame("read_csv")
            True
            >>> PolarsApi()._returns_frame("col")
            False
        """
        function = getattr(pl, name)
        if name.startswith("_") or inspect.isclass(function) or not callable(function):
            return False
        try:
            annotation = inspect.signature(function).return_annotation
        except (TypeError, ValueError):
            return False
        head = str(annotation).split("|", maxsplit=1)[0]
        return re.search(r"\b(DataFrame|LazyFrame)\b", head) is not None

    def selectors(self) -> set[str]:
        """Give every selector constructor `polars.selectors` offers.

        A selector constructor is a public callable that is neither a
        type nor one of the module's helpers.

        Returns:
            Each name that builds a selector, without the helpers.

        Example:
            >>> "numeric" in PolarsApi().selectors()
            True
            >>> "Expr" in PolarsApi().selectors()
            False
        """
        found = set()
        for name in dir(cs):
            if name.startswith("_") or name in self.HELPERS:
                continue
            value = getattr(cs, name)
            if callable(value) and not isinstance(value, type):
                found.add(name)
        return found

    NAMESPACES = (
        "arr",
        "bin",
        "cat",
        "dt",
        "ext",
        "list",
        "meta",
        "name",
        "str",
        "struct",
    )

    def expression_methods(self) -> set[str]:
        """Give every public method of `pl.Expr`, namespaces excluded.

        Returns:
            Each public method name.

        Example:
            >>> "rolling_mean" in PolarsApi().expression_methods()
            True
        """
        found = {m for m in dir(pl.Expr) if not m.startswith("_")}
        return found - set(self.NAMESPACES)

    def namespace_members(self) -> set[str]:
        """Give every member of every `pl.Expr` namespace.

        Returns:
            Each public member as `namespace.member`.

        Example:
            >>> "str.len_chars" in PolarsApi().namespace_members()
            True
            >>> "struct.field" in PolarsApi().namespace_members()
            True
        """
        found = set()
        for namespace in self.NAMESPACES:
            holder = getattr(pl.col("x"), namespace)
            for member in dir(holder):
                if not member.startswith("_"):
                    found.add(f"{namespace}.{member}")
        return found

    def expression_constructors(self) -> set[str]:
        """Give every `pl` callable that builds an expression.

        Detection reads the return annotation of each public callable
        that is not a type, plus the two callable instances the
        annotation scan cannot see, `pl.col` and `pl.when`. A callable
        with no readable annotation does not count.

        Returns:
            Each constructor's name.

        Example:
            >>> "sum_horizontal" in PolarsApi().expression_constructors()
            True
            >>> "DataFrame" in PolarsApi().expression_constructors()
            False
        """
        found = {"col", "when"}
        for name in dir(pl):
            if name.startswith("_"):
                continue
            value = getattr(pl, name)
            if not callable(value) or isinstance(value, type):
                continue
            try:
                annotation = str(inspect.signature(value).return_annotation)
            except (ValueError, TypeError):
                continue
            if "Expr" in annotation:
                found.add(name)
        return found

    def group_by_methods(self) -> set[str]:
        """Give every public method of `GroupBy`.

        Returns:
            Each public attribute name.

        Example:
            >>> "agg" in PolarsApi().group_by_methods()
            True
        """
        return {m for m in dir(pl.dataframe.group_by.GroupBy) if not m.startswith("_")}


class TestIdentifiersResolve:
    """Every cross-reference names something that exists."""

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    def test_every_identifier_used_is_registered(self, document: Document) -> None:
        """A document may cite only identifiers the claim ledger registers.

        This is what makes a dangling reference impossible. An earlier
        draft cited "the `survivors_L` rule of 6.4", and section 6.4
        defined no such name.
        """
        unregistered = document.identifiers - Ledger().registered
        assert not unregistered, (
            f"{document.name} cites identifiers the ledger does not register: "
            f"{sorted(unregistered)}"
        )

    def test_every_registered_identifier_is_used(self) -> None:
        """The ledger holds no entry nothing refers to."""
        used: set[str] = set()
        for document in ALL:
            used |= document.identifiers
        unused = Ledger().registered - used
        assert not unused, f"the ledger registers unused identifiers: {sorted(unused)}"

    def test_the_ledger_covers_every_kind(self) -> None:
        """All six kinds of identifier are registered, so none was forgotten."""
        kinds = {identifier.split("-")[0] for identifier in Ledger().registered}
        assert kinds == {"D", "R", "E", "T", "M", "B"}


class TestCitedEvidenceExists:
    """Every test a document names as evidence is a test that exists."""

    @staticmethod
    def spec_test_names() -> set[str]:
        """Give every test function and test class defined under `spec/tests`.

        Returns:
            Each `test_...` function name and `Test...` class name.
        """
        names: set[str] = set()
        for path in Path(ROOT / "spec" / "tests").glob("*.py"):
            source = path.read_text()
            names |= set(re.findall(r"^\s*def (test_[a-z0-9_]+)", source, re.MULTILINE))
            names |= set(re.findall(r"^class (Test[A-Za-z0-9]+)", source, re.MULTILINE))
        return names

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    def test_every_cited_test_exists(self, document: Document) -> None:
        """A document that cites evidence must cite evidence that is there."""
        missing = document.cited_tests() - self.spec_test_names()
        assert not missing, f"{document.name} cites tests that do not exist: {sorted(missing)}"

    def test_some_evidence_is_actually_cited(self) -> None:
        """The check above is only meaningful if citations exist at all."""
        cited: set[str] = set()
        for document in ALL:
            cited |= document.cited_tests()
        assert len(cited) >= 10


class TestLeanCitationsResolve:
    """Every Lean theorem a document cites exists, is audited, and uses no escape hatch."""

    LEAN = LeanSources()

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    def test_every_cited_lean_theorem_is_declared(self, document: Document) -> None:
        """A cited proof must be a proof that is there."""
        missing = document.cited_lean() - self.LEAN.declared
        assert not missing, f"{document.name} cites undeclared Lean names: {sorted(missing)}"

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    def test_every_cited_file_declares_its_theorem(self, document: Document) -> None:
        """A citation that names a file names the file the theorem is in."""
        declared_in = self.LEAN.declared_in
        wrong = {
            name: path
            for name, path in document.cited_lean_files().items()
            if declared_in.get(name) != path
        }
        assert not wrong, f"{document.name} cites the wrong file: {wrong}"

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    def test_every_cited_lean_theorem_is_audited(self, document: Document) -> None:
        """A cited theorem is checked for non-standard axioms by the Lean build."""
        unaudited = document.cited_lean() - self.LEAN.audited
        assert not unaudited, f"{document.name} cites unaudited theorems: {sorted(unaudited)}"

    def test_every_audited_name_is_declared(self) -> None:
        """The audit list names only theorems that exist."""
        assert not self.LEAN.audited - self.LEAN.declared

    def test_no_proof_skips_its_obligation(self) -> None:
        """No `sorry`, `admit`, `native_decide` or added `axiom` in the Lean sources."""
        assert not self.LEAN.forbidden()

    def test_no_proof_is_left_in_prose(self) -> None:
        """Every proof is in Lean, so no prose proof is left to check by hand."""
        assert "*Proof.*" not in OPERATIONS.text
        assert "∎" not in OPERATIONS.text

    def test_every_proof_is_cited(self) -> None:
        """The eighteen proofs of the document cite at least eighteen theorems."""
        assert len(OPERATIONS.cited_lean()) >= 18

    def test_every_proved_ledger_row_names_a_lean_theorem(self) -> None:
        """A claim the ledger tags proved names its Lean theorem in the ledger."""
        rows = Ledger().proved_rows
        unnamed = [row[0] for row in rows if "`Spec." not in row[3]]
        assert rows
        assert not unnamed, f"proved without a Lean theorem: {unnamed}"


class TestSelfCounts:
    """A document's claims about its own size are true."""

    def test_the_catalogue_states_the_real_frame_method_count(self) -> None:
        """The census says how large the API is, and the API says so too."""
        assert "152 methods" in CATALOGUE.text
        assert len(PolarsApi().frame_methods()) == 152

    def test_the_catalogue_covers_every_frame_method(self) -> None:
        """Every public frame method appears somewhere in the census.

        A polars release that adds a method fails here, rather than
        leaving the census quietly incomplete.
        """
        text = CATALOGUE.text
        methods = PolarsApi().frame_methods()
        missing = {m for m in methods if not re.search(rf"\b{re.escape(m)}\b", text)}
        assert not missing, f"the census never names: {sorted(missing)}"

    def test_the_part_one_and_part_two_split_is_exact(self) -> None:
        """Every frame method is in Part I or Part II, and the counts say so.

        This is the check that keeps the census complete. A polars
        release adding a method puts it in neither part and fails here.
        """
        part_one, part_two = self.parts()
        methods = PolarsApi().frame_methods()
        neither = methods - part_one - part_two
        assert not neither, f"in neither part: {sorted(neither)}"
        assert len(methods & part_one) == 75
        assert len(methods & part_two) == 80
        assert sorted(methods & part_one & part_two) == [
            "clear",
            "n_unique",
            "sql",
        ]
        assert "| 75 | Part I |" in CATALOGUE.text
        assert "| 80 | Part II |" in CATALOGUE.text

    def test_the_catalogue_names_every_frame_function(self) -> None:
        """Every `pl.*` function that returns a frame appears in the catalog."""
        functions = PolarsApi().frame_functions()
        missing = {name for name in functions if f"`pl.{name}" not in CATALOGUE.text}
        assert not missing, f"the census never names: {sorted(missing)}"

    def test_the_frame_function_count_is_exact(self) -> None:
        """The scope table counts every frame function polars offers."""
        count = len(PolarsApi().frame_functions())
        assert f"| frame-level `pl.*` functions | {count} |" in CATALOGUE.text

    @staticmethod
    def parts() -> tuple[set[str], set[str]]:
        """Give the frame methods each part of the census names.

        Returns:
            The Part I set, read as calls, and the Part II set, read as
            the inline names its exclusion lists hold.
        """
        text = CATALOGUE.text
        boundary = text.index("# Part II.")
        closing = text.index("# Part III.")
        one = set(re.findall(r"\.([a-z_][a-z0-9_]*)\(", text[:boundary]))
        names = re.findall(r"`([a-z_][a-z0-9_.]*)`", text[boundary:closing])
        two = {n.removeprefix("pl.") for n in names}
        return one, two

    def test_the_catalogue_covers_every_selector(self) -> None:
        """Every selector constructor appears somewhere in the census."""
        constructors = PolarsApi().selectors()
        missing = {c for c in constructors if f"cs.{c}" not in CATALOGUE.text}
        assert not missing, f"the census never names: {sorted(missing)}"

    def test_the_catalogue_covers_every_group_by_method(self) -> None:
        """Every `GroupBy` method appears somewhere in the census."""
        methods = PolarsApi().group_by_methods()
        missing = {m for m in methods if f".{m}(" not in CATALOGUE.text}
        assert not missing, f"the census never names: {sorted(missing)}"


class TestExpressionCensus:
    """Part III covers the whole expression API, and its counts are true."""

    @staticmethod
    def part_three() -> str:
        """Give the catalogue's Part III text.

        Returns:
            Everything from the Part III heading onward.
        """
        return CATALOGUE.text[CATALOGUE.text.index("# Part III.") :]

    def test_every_expression_method_is_named(self) -> None:
        """Each `pl.Expr` method appears in Part III.

        A polars release that adds a method fails here rather than
        leaving the census quietly incomplete.
        """
        text = self.part_three()
        methods = PolarsApi().expression_methods()
        missing = {m for m in methods if not re.search(rf"\b{re.escape(m)}\b", text)}
        assert not missing, f"Part III never names: {sorted(missing)}"

    def test_every_namespace_member_is_named(self) -> None:
        """Each namespace member appears in Part III, dotted."""
        text = self.part_three()
        members = PolarsApi().namespace_members()
        missing = {m for m in members if f"{m}" not in text}
        assert not missing, f"Part III never names: {sorted(missing)}"

    def test_every_constructor_is_named(self) -> None:
        """Each expression-building `pl` callable appears in Part III."""
        text = self.part_three()
        constructors = PolarsApi().expression_constructors()
        missing = {c for c in constructors if f"pl.{c}" not in text}
        assert not missing, f"Part III never names: {sorted(missing)}"

    def test_the_census_calls_match_the_installed_polars(self) -> None:
        """The calls of the census cover exactly the library's methods and functions.

        The called methods plus the seven opaque ones are exactly
        `pl.Expr`'s methods, and the called namespace members cover
        every namespace but the three that Part III handles in prose. The
        called `pl` functions, with the ones that the census leaves out
        for a reason, are exactly the `pl` callables that build an
        expression. So a new method or a new function of polars fails
        this test.
        """
        calls = CensusCalls()
        assert calls.method_keys() == {f"root.{m}" for m in PolarsApi().expression_methods()}
        assert set(calls.members()) == self.swept_namespace_members()
        assert self.accounted_constructors() == PolarsApi().expression_constructors()

    def test_each_function_left_out_has_one_reason(self) -> None:
        """Check that each `pl` function that the census leaves out has one reason, and no call."""
        reasons = [
            set(CensusCalls.CONTEXT_BOUND),
            set(CensusCalls.OPAQUE_CONSTRUCTORS),
            set(CensusCalls.TYPE_LEVEL),
        ]
        called = set(CensusCalls.CONSTRUCTORS)
        for index, names in enumerate(reasons):
            assert not names & called, names & called
            for other in reasons[index + 1 :]:
                assert not names & other, names & other

    def accounted_constructors(self) -> set[str]:
        """Give the `pl` functions that the census accounts for.

        A function has a call, or it is in one of three lists of the
        functions that the census leaves out: those that only work
        inside the scope of another expression, those that run a
        function of the user, and those that build a data type and not
        an expression.

        Returns:
            The names of the functions.
        """
        found = set(CensusCalls.CONSTRUCTORS)
        found |= set(CensusCalls.CONTEXT_BOUND)
        found |= set(CensusCalls.OPAQUE_CONSTRUCTORS)
        return found | set(CensusCalls.TYPE_LEVEL)

    def swept_namespace_members(self) -> set[str]:
        """Give the namespace members the census must measure.

        Returns:
            Every namespace member outside the three prose sections.
        """
        found = set()
        for member in PolarsApi().namespace_members():
            if member.split(".", maxsplit=1)[0] not in {"name", "meta", "ext"}:
                found.add(member)
        return found

    def test_the_counts_the_catalogue_quotes(self) -> None:
        """Every count in section 17.3 and section 18 is computed, not remembered.

        The table of section 17.3 has one row for each release of polars, and
        the row of the installed release gives its counts.
        """
        api = PolarsApi()
        release = str(PolarsRelease.installed())
        rows = [row for row in CATALOGUE.table_rows if row[0] == release]
        assert rows, f"section 17.3 has no row for polars {release}"
        methods, members, total = (int(cell) for cell in rows[0][1:4])
        assert len(api.expression_methods()) == methods
        assert len(api.namespace_members()) == members
        assert methods + members == total
        assert len(api.expression_constructors()) == 82
        assert len(CensusCalls.ROOT_CALLS) == 205
        assert len(CensusCalls.OPAQUE) == 7
        assert len(CensusCalls.CONSTRUCTORS) == 73
        for quoted in ("205 that the census calls", "Eighty-two callables"):
            assert quoted in CATALOGUE.text, quoted

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    def test_the_provisional_prefixes_are_retired(self, document: Document) -> None:
        """The design draft's `XD-`/`XL-`/`XM-` identifiers appear nowhere."""
        for prefix in ("XD-", "XL-", "XM-"):
            assert prefix not in document.text


class TestMeasurementsAgree:
    """Every count a document quotes is a count a test computes."""

    def test_the_split_precision_numbers_are_consistent(self) -> None:
        """The two halves of the 32 caller shapes add up.

        An earlier draft quoted 11 in one section and 28 in another.
        Both count parts of the same 32, so 11 plus 28 being 39 was
        proof that one of them was wrong.
        """
        assert MIXED.exact() == 11
        assert MIXED.imprecise() == 21
        assert MIXED.exact() + MIXED.imprecise() == 32

        for document in (FRAMES, OPERATIONS):
            assert "11" in document.text
            assert "28 of 32" not in document.text

    def test_no_document_quotes_a_retired_number(self) -> None:
        """The superseded figures appear nowhere."""
        for document in ALL:
            assert "28 of 32" not in document.text
            assert "There are 61 of them" not in document.text


class TestVocabularyIsShared:
    """The documents use one vocabulary, not one each."""

    RETIRED = (
        "Merged",
        "Joinable",
        "AllOfFamily",
        "NoneOfFamily",
        "Lacks",
        "residue",
        "subsumption",
    )

    @pytest.mark.parametrize("term", RETIRED)
    def test_a_retired_term_appears_nowhere(self, term: str) -> None:
        """A word from a superseded vocabulary is not left standing."""
        for document in ALL:
            message = f"{document.name} still uses the retired term {term!r}"
            assert not re.search(rf"\b{term}\b", document.text), message

    def test_the_split_has_one_spelling(self) -> None:
        """`SplitColumns[...]` is written one way in every document."""
        for document in ALL:
            assert "split_columns(" not in document.text

    def test_the_catalogue_no_longer_carries_implementation_status(self) -> None:
        """The census describes the target system, so it grades no code."""
        for marker in ("**status**:", "UNSPECIFIED", "astrolabe returns"):
            assert marker not in CATALOGUE.text

    def test_the_catalogue_points_at_the_current_model(self) -> None:
        """The census cites the two documents that define its vocabulary."""
        assert "OPERATIONS.md" in CATALOGUE.text
        assert "DATAFRAME_SPECIFICATION.md" in CATALOGUE.text
        assert "GENERIC_SCHEMA_DESIGN.md" not in CATALOGUE.text


class TestGlossary:
    """Every term a specification uses with a technical meaning is defined."""

    @pytest.mark.parametrize("glossary", GLOSSARIES, ids=lambda g: g.document.name)
    def test_the_glossary_is_not_empty(self, glossary: Glossary) -> None:
        """A glossary that parsed to nothing would make the checks below vacuous."""
        assert len(glossary.entries) >= 15

    @pytest.mark.parametrize("glossary", GLOSSARIES, ids=lambda g: g.document.name)
    def test_every_glossary_entry_resolves(self, glossary: Glossary) -> None:
        """Each term names an identifier the ledger registers, or says otherwise."""
        registered = Ledger().registered
        for term, target in glossary.entries.items():
            reference = target.strip("`")
            if not re.fullmatch(r"[DRETMB]-\d+", reference):
                continue
            assert reference in registered, f"{term!r} names an unregistered {reference}"

    @pytest.mark.parametrize("glossary", GLOSSARIES, ids=lambda g: g.document.name)
    def test_every_glossary_term_is_used_in_the_body(self, glossary: Glossary) -> None:
        """A glossary entry for a term the document never uses is dead weight."""
        body = glossary.body()
        for term in glossary.entries:
            plain = term.strip("`")
            assert plain.lower() in body.lower(), f"{term!r} is defined and never used"


class TestTablesAreWellFormed:
    """Every table row has as many cells as its header."""

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    def test_no_row_is_ragged(self, document: Document) -> None:
        """A row with the wrong number of cells renders wrongly and hides content."""
        widths: dict[int, int] = {}
        for row in document.table_rows:
            widths[len(row)] = widths.get(len(row), 0) + 1
        assert widths, f"{document.name} has no tables"
        assert min(widths) >= 2, f"{document.name} has a one-cell table row"


class TestAffixMarkers:
    """Each affix marker that a document writes puts its literal where its class takes it."""

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    @pytest.mark.parametrize("marker", [Prefixed, Suffixed], ids=lambda m: m.__name__)
    def test_the_literal_stands_where_the_class_takes_it(
        self,
        document: Document,
        marker: type,
    ) -> None:
        """`Prefixed` takes its prefix first, as in the name, and `Suffixed` its suffix last.

        A user who copies a marker with its arguments swapped gets two
        `marker-arg` errors, so each spelling follows the order of the class.
        """
        literal_first = marker.__type_params__[0].__name__ != "NameExpr"
        for found in re.finditer(rf"\b{marker.__name__}\[(Literal\[)?", document.text):
            assert bool(found.group(1)) == literal_first, f"{document.name}: {found.group(0)}"


class TestProseDiscipline:
    """The house style the repository applies to its own prose."""

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    def test_no_em_dash(self, document: Document) -> None:
        """The style rules of this repository exclude the em dash."""
        assert "—" not in document.text

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    def test_only_the_three_permitted_modals_appear(self, document: Document) -> None:
        """A specification states what can, will and must happen.

        "Should" and "may" each read two ways, one of them optional,
        and a reader cannot tell which is meant. "Might" hedges a fact
        that either holds or does not. The counterfactual "would" and
        "could" stay, because a passage explaining why a design was
        not chosen has no other way to say it.
        """
        banned = ("should", "may", "might")
        found = [w for w in banned if re.search(rf"\b{w}\b", document.text, re.IGNORECASE)]
        assert not found, f"{document.name} uses {found}"

    @pytest.mark.parametrize("document", ALL, ids=lambda d: d.name)
    def test_no_hedging(self, document: Document) -> None:
        """A specification asserts or it does not; it does not guess."""
        hedges = (
            "probably",
            "presumably",
            "seems",
            "arguably",
            "perhaps",
            "roughly",
            "approximately",
        )
        found = [w for w in hedges if re.search(rf"\b{w}\b", document.text, re.IGNORECASE)]
        assert not found, f"{document.name} hedges with {found}"
