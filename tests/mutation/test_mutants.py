"""Check how the harness lists the mutants of a file, and the filters that drop some.

A filter drops a mutant that no test can catch whatever the tests
say, for example a change inside a type annotation, which nothing
evaluates at run time. Such a mutant only adds noise to the report,
so dropping it removes no evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cosmic_ray.ast import ast_nodes, get_ast_from_path
from cosmic_ray.plugins import get_operator

from tools.mutation.mutants import (
    AnnotationSite,
    FrozenFlag,
    IndexOffset,
    Mutant,
    MutantCatalog,
    OperatorRoster,
    PragmaSite,
    Site,
    TypeCheckingSite,
)

if TYPE_CHECKING:
    from pathlib import Path as TmpPath


def _union_verdicts(tmp_path: TmpPath, source: str) -> list[bool]:
    """Say, for each mutant that turns a `|` into `&`, whether it sits in an annotation.

    Args:
        tmp_path: The directory for the source file.
        source: The Python source to write and read.

    Returns:
        One answer for each mutant, in order. True means that the
        annotation filter drops it.
    """
    path = tmp_path / "m.py"
    path.write_text(source, encoding="utf-8")
    operator = get_operator("core/ReplaceBinaryOperator_BitOr_BitAnd")()
    site = AnnotationSite()
    found: list[bool] = []
    for node in ast_nodes(get_ast_from_path(path)):
        for start, end in operator.mutation_positions(node):
            place = Site(node, (), "core/x", 0, start[0], end[0])
            found.append(site.covers(place))
    return found


def test_roster_keeps_one_replacement_for_each_source_operator() -> None:
    names = OperatorRoster().names()
    assert len(names) == len(set(names))
    assert "core/ReplaceBinaryOperator_BitOr_BitAnd" in names
    assert "core/ReplaceBinaryOperator_BitOr_Add" not in names


def test_roster_leaves_out_the_operators_that_need_arguments() -> None:
    names = OperatorRoster().names()
    assert "core/VariableReplacer" not in names
    assert "core/VariableInserter" not in names


def test_roster_builds_every_operator_it_names() -> None:
    built = OperatorRoster().operators()
    assert len(built) == len(OperatorRoster().names())
    assert all(operator is not None for _name, operator in built)


def test_annotation_filter_finds_a_parameter_annotation(tmp_path: TmpPath) -> None:
    assert _union_verdicts(tmp_path, "def f(a: int | str) -> None: pass\n") == [True]


def test_annotation_filter_finds_a_return_annotation(tmp_path: TmpPath) -> None:
    assert _union_verdicts(tmp_path, "def f() -> int | None: pass\n") == [True]


def test_annotation_filter_finds_a_class_variable_annotation(tmp_path: TmpPath) -> None:
    source = "from typing import ClassVar\n\n\nclass C:\n    x: ClassVar[int | None] = None\n"
    assert _union_verdicts(tmp_path, source) == [True]


def test_annotation_filter_leaves_a_real_union_alone(tmp_path: TmpPath) -> None:
    assert _union_verdicts(tmp_path, "def f(a, b):\n    return a | b\n") == [False]


def test_annotation_filter_leaves_a_default_value_alone(tmp_path: TmpPath) -> None:
    assert _union_verdicts(tmp_path, "def f(a: int = 1 | 2) -> None: pass\n") == [False]


def test_annotation_filter_answers_no_for_a_node_with_no_parent() -> None:
    assert AnnotationSite().covers(Site(None, (), "core/x", 0, 1, 1)) is False


def test_pragma_filter_finds_the_mark_on_the_first_line() -> None:
    assert PragmaSite().covers(Site(None, ("a  # pragma: no cover", "b"), "core/x", 0, 1, 1))


def test_pragma_filter_finds_the_mark_on_a_later_line_of_the_span() -> None:
    assert PragmaSite().covers(Site(None, ("a", "b  # pragma: no cover"), "core/x", 0, 1, 2))


def test_pragma_filter_leaves_an_unmarked_span_alone() -> None:
    assert not PragmaSite().covers(Site(None, ("a", "b"), "core/x", 0, 1, 2))


def test_pragma_filter_reads_extra_spacing() -> None:
    assert PragmaSite().covers(Site(None, ("a  #  pragma:  no cover",), "core/x", 0, 1, 1))


def test_mutant_labels_itself_for_a_report() -> None:
    assert Mutant(Path("a/b.py"), "core/AddNot", 0, 7).label() == "a/b.py:7 AddNot"


def test_catalog_skips_the_stub_overrides(tmp_path: TmpPath) -> None:
    (tmp_path / "frame.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("x = 1\n", encoding="utf-8")
    assert [p.name for p in MutantCatalog().files(tmp_path)] == ["other.py"]


def test_catalog_keeps_a_frame_module_below_the_root(tmp_path: TmpPath) -> None:
    (tmp_path / "frame.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "domain").mkdir()
    (tmp_path / "domain" / "frame.py").write_text("x = 1\n", encoding="utf-8")
    found = MutantCatalog().files(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in found] == ["domain/frame.py"]


def test_catalog_keeps_a_mutant_in_the_value_of_an_annotated_assignment(
    tmp_path: TmpPath,
) -> None:
    source = tmp_path / "m.py"
    source.write_text("def f(a, b):\n    x: int = a + b\n    return x\n", encoding="utf-8")
    operators = {mutant.operator for mutant in MutantCatalog().for_file(source)}
    assert "core/ReplaceBinaryOperator_Add_Sub" in operators


def test_catalog_drops_a_mutant_in_the_annotation_of_an_assignment(tmp_path: TmpPath) -> None:
    source = tmp_path / "m.py"
    source.write_text("def f(a):\n    x: int | str = a\n    return x\n", encoding="utf-8")
    operators = {mutant.operator for mutant in MutantCatalog().for_file(source)}
    assert "core/ReplaceBinaryOperator_BitOr_BitAnd" not in operators


def test_catalog_drops_a_mutant_on_a_pragma_line(tmp_path: TmpPath) -> None:
    marked = tmp_path / "m.py"
    marked.write_text("def f(a):\n    return a > 1  # pragma: no cover\n", encoding="utf-8")
    assert MutantCatalog().for_file(marked) == []


def test_catalog_keeps_the_mutants_of_a_plain_line(tmp_path: TmpPath) -> None:
    plain = tmp_path / "m.py"
    plain.write_text("def f(a):\n    return a > 1\n", encoding="utf-8")
    found = MutantCatalog().for_file(plain)
    assert {m.operator.split("/")[-1] for m in found} == {
        "NumberReplacer",
        "ReplaceComparisonOperator_Gt_GtE",
    }


def test_catalog_counts_the_occurrence_before_a_filter_drops_a_span(tmp_path: TmpPath) -> None:
    source = "def f(a, b):\n    x = a > 1  # pragma: no cover\n    return b > 2\n"
    path = tmp_path / "m.py"
    path.write_text(source, encoding="utf-8")
    found = [m for m in MutantCatalog().for_file(path) if m.operator.endswith("Gt_GtE")]
    assert [(m.line, m.occurrence) for m in found] == [(3, 1)]


def test_catalog_walks_a_whole_tree(tmp_path: TmpPath) -> None:
    package = tmp_path / "p"
    package.mkdir()
    (package / "a.py").write_text("def f(a):\n    return a > 1\n", encoding="utf-8")
    assert MutantCatalog().mutants(tmp_path) != []


def test_the_guard_filter_finds_a_type_checking_block() -> None:
    assert TypeCheckingSite().covers(Site(None, ("if TYPE_CHECKING:",), "core/x", 0, 1, 1))
    assert TypeCheckingSite().covers(Site(None, ("if typing.TYPE_CHECKING:",), "core/x", 0, 1, 1))


def test_the_guard_filter_leaves_another_condition_alone() -> None:
    assert not TypeCheckingSite().covers(Site(None, ("if ready:",), "core/x", 0, 1, 1))
    assert not TypeCheckingSite().covers(Site(None, ("x = TYPE_CHECKING",), "core/x", 0, 1, 1))


def test_the_catalog_drops_a_mutant_on_a_type_checking_guard(tmp_path: TmpPath) -> None:
    source = "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import os\n"
    path = tmp_path / "m.py"
    path.write_text(source, encoding="utf-8")
    assert MutantCatalog().for_file(path) == []


def test_the_index_filter_drops_the_downward_change_on_a_subscript(tmp_path: TmpPath) -> None:
    path = tmp_path / "m.py"
    path.write_text("def f(args):\n    return args[0]\n", encoding="utf-8")
    found = [m for m in MutantCatalog().for_file(path) if m.operator.endswith("NumberReplacer")]
    assert [m.occurrence for m in found] == [0]


def test_the_index_filter_keeps_both_changes_on_a_plain_number(tmp_path: TmpPath) -> None:
    path = tmp_path / "m.py"
    path.write_text("def f(n):\n    return n + 1\n", encoding="utf-8")
    found = [m for m in MutantCatalog().for_file(path) if m.operator.endswith("NumberReplacer")]
    assert [m.occurrence for m in found] == [0, 1]


def test_the_index_filter_leaves_another_operator_alone() -> None:
    place = Site(None, ("x = 1",), "core/AddNot", 1, 1, 1)
    assert IndexOffset().covers(place) is False


def test_the_frozen_filter_drops_the_flag_of_a_value_class() -> None:
    lines = ("@dataclass(frozen=True)",)
    assert FrozenFlag().covers(Site(None, lines, "core/ReplaceTrueWithFalse", 0, 1, 1))


def test_the_frozen_filter_leaves_another_true_alone() -> None:
    lines = ("ready = True",)
    assert not FrozenFlag().covers(Site(None, lines, "core/ReplaceTrueWithFalse", 0, 1, 1))


def test_the_frozen_filter_leaves_another_operator_alone() -> None:
    lines = ("@dataclass(frozen=True)",)
    assert not FrozenFlag().covers(Site(None, lines, "core/AddNot", 0, 1, 1))


def _operators(tmp_path: TmpPath, source: str) -> list[tuple[str, int]]:
    """List the mutants of one small source, by operator and line.

    Args:
        tmp_path: The directory for the file.
        source: The Python source.

    Returns:
        The operator without its provider, beside the line, for each mutant.
    """
    path = tmp_path / "m.py"
    path.write_text(source, encoding="utf-8")
    return [(m.operator.split("/")[-1], m.line) for m in MutantCatalog().for_file(path)]


def test_catalog_drops_the_strict_flag_of_a_zip(tmp_path: TmpPath) -> None:
    found = _operators(tmp_path, "def f(a, b):\n    return list(zip(a, b, strict=True))\n")
    assert ("ReplaceTrueWithFalse", 2) not in found


def test_catalog_keeps_a_strict_flag_of_another_call(tmp_path: TmpPath) -> None:
    found = _operators(tmp_path, "def f(a):\n    return g(a, strict=True)\n")
    assert ("ReplaceTrueWithFalse", 2) in found


def test_catalog_drops_a_raised_bound_of_a_split_that_keeps_the_first_piece(
    tmp_path: TmpPath,
) -> None:
    found = _operators(tmp_path, 'def f(s):\n    return s.split(".", maxsplit=1)[0]\n')
    # The bound keeps its downward change, and the index keeps its upward one.
    assert found.count(("NumberReplacer", 2)) == 2


def test_catalog_drops_a_raised_bound_of_an_rsplit_that_keeps_the_last_piece(
    tmp_path: TmpPath,
) -> None:
    found = _operators(tmp_path, 'def f(s):\n    return s.rsplit(".", 1)[-1]\n')
    # The bound keeps its downward change. The `1` of `-1` keeps both.
    assert found.count(("NumberReplacer", 2)) == 3


def test_catalog_keeps_the_bound_of_an_rsplit_that_keeps_the_first_piece(
    tmp_path: TmpPath,
) -> None:
    found = _operators(tmp_path, 'def f(s):\n    return s.rsplit(".", 1)[0]\n')
    assert found.count(("NumberReplacer", 2)) == 3


def test_catalog_drops_the_change_of_the_keyword_marker(tmp_path: TmpPath) -> None:
    found = _operators(tmp_path, "def f(a, *, b):\n    return a * b\n")
    assert found.count(("ReplaceBinaryOperator_Mul_Div", 1)) == 0
    assert ("ReplaceBinaryOperator_Mul_Div", 2) in found
