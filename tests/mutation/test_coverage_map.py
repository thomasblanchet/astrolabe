"""Check the map from each source line to the tests that run it.

The harness runs, for each mutant, only the tests that reach the
changed line. It finds them in a coverage data file that records,
for each line, the tests that ran it. These tests write such a file
with the API of coverage.py, so the file has the same form as the file
of a real run.
"""

from __future__ import annotations

from pathlib import Path

from coverage import CoverageData

from tools.mutation.coverage_map import (
    CoverageMap,
    YamlCases,
    TestOrder,
)


def _written(tmp_path: Path, arcs: dict[str, dict[str, list[tuple[int, int]]]]) -> Path:
    """Write a coverage data file that records which test ran which lines.

    Coverage calls the test that ran a line its context. The first full
    run of the harness writes a file of this form.

    Args:
        tmp_path: The directory for the file.
        arcs: For each test, the pairs of lines that it ran, by file name.

    Returns:
        The path of the file.
    """
    data_file = tmp_path / "d.coverage"
    data = CoverageData(basename=str(data_file))
    for context, by_file in arcs.items():
        data.set_context(context)
        data.add_arcs({str(tmp_path / name): set(pairs) for name, pairs in by_file.items()})
    data.write()
    return data_file


def test_it_reads_a_line_back_as_the_tests_that_ran_it(tmp_path: Path) -> None:
    data_file = _written(tmp_path, {"t.yml::a|run": {"m.py": [(1, 2), (2, 3)]}})
    found = CoverageMap().read(data_file, tmp_path)
    assert found.tests_for(Path("m.py"), 2) == ("t.yml::a",)


def test_it_joins_the_phases_of_one_test(tmp_path: Path) -> None:
    data_file = _written(
        tmp_path,
        {"t.yml::a|setup": {"m.py": [(1, 2)]}, "t.yml::a|run": {"m.py": [(1, 2)]}},
    )
    found = CoverageMap().read(data_file, tmp_path)
    assert found.tests_for(Path("m.py"), 2) == ("t.yml::a",)


def test_it_falls_back_to_the_whole_file_for_an_import_time_line(tmp_path: Path) -> None:
    data_file = _written(
        tmp_path,
        {"": {"m.py": [(4, 5)]}, "t.yml::a|run": {"m.py": [(1, 2)]}},
    )
    found = CoverageMap().read(data_file, tmp_path)
    assert found.tests_for(Path("m.py"), 5) == ("t.yml::a",)


def test_a_line_that_only_a_unit_test_runs_is_unreached(tmp_path: Path) -> None:
    data_file = _written(
        tmp_path,
        {"t.yml::a|run": {"m.py": [(1, 2)]}, "t.py::b|run": {"m.py": [(4, 5)]}},
    )
    found = CoverageMap().read(data_file, tmp_path)
    assert found.tests_for(Path("m.py"), 5) == ()
    assert found.unreached(Path("m.py"), 5)


def test_a_line_that_a_yml_case_runs_is_not_unreached(tmp_path: Path) -> None:
    data_file = _written(
        tmp_path,
        {"t.yml::a|run": {"m.py": [(1, 2)]}, "t.py::b|run": {"m.py": [(1, 2)]}},
    )
    found = CoverageMap().read(data_file, tmp_path)
    assert not found.unreached(Path("m.py"), 2)


def test_an_import_time_line_is_not_unreached(tmp_path: Path) -> None:
    data_file = _written(
        tmp_path,
        {"": {"m.py": [(4, 5)]}, "t.yml::a|run": {"m.py": [(1, 2)]}},
    )
    found = CoverageMap().read(data_file, tmp_path)
    assert not found.unreached(Path("m.py"), 5)


def test_it_answers_nothing_for_an_unmeasured_file(tmp_path: Path) -> None:
    data_file = _written(tmp_path, {"t.py::a|run": {"m.py": [(1, 2)]}})
    found = CoverageMap().read(data_file, tmp_path)
    assert found.tests_for(Path("other.py"), 1) == ()


def test_it_leaves_out_a_file_outside_the_root(tmp_path: Path) -> None:
    data_file = _written(tmp_path, {"t.py::a|run": {"m.py": [(1, 2)]}})
    found = CoverageMap().read(data_file, tmp_path / "inner")
    assert found.files() == ()


def test_the_order_puts_a_doctest_before_an_integration_test() -> None:
    order = TestOrder()
    assert order.rank("src/astrolabe/a.py::a.B") < order.rank("tests/typing/a.yml::c")
    assert order.rank("tests/typing/a.yml::c") < order.rank("tests/test_x_integration.py::t")


def test_the_order_ranks_an_unplaced_test_in_the_middle() -> None:
    assert TestOrder().rank("other/place.py::t") == TestOrder.DEFAULT


def test_the_order_sorts_a_selection_cheapest_first() -> None:
    found = TestOrder().arranged(
        ["tests/test_a_integration.py::t", "src/astrolabe/a.py::a.B", "tests/typing/a.yml::c"],
    )
    assert found[0] == "src/astrolabe/a.py::a.B"
    assert found[-1] == "tests/test_a_integration.py::t"


def test_a_doctest_does_not_kill_a_mutant_by_default(tmp_path: Path) -> None:
    node = "src/astrolabe/expr.py::astrolabe.expr.C"
    data_file = _written(tmp_path, {f"{node}|run": {"m.py": [(1, 2)]}})
    found = CoverageMap().read(data_file, tmp_path)
    assert found.tests_for(Path("m.py"), 2) == ()


def test_a_unit_test_does_not_kill_a_mutant_by_default(tmp_path: Path) -> None:
    node = "tests/engine/test_a.py::test_b"
    data_file = _written(tmp_path, {f"{node}|run": {"m.py": [(1, 2)]}})
    found = CoverageMap().read(data_file, tmp_path)
    assert found.tests_for(Path("m.py"), 2) == ()


def test_a_yml_case_kills_a_mutant(tmp_path: Path) -> None:
    node = "tests/typing/test_select.yml::a_case"
    data_file = _written(tmp_path, {f"{node}|run": {"m.py": [(1, 2)]}})
    found = CoverageMap().read(data_file, tmp_path)
    assert found.tests_for(Path("m.py"), 2) == (node,)


def test_any_test_kills_a_mutant_when_the_run_asks_for_it(tmp_path: Path) -> None:
    node = "src/astrolabe/expr.py::astrolabe.expr.C"
    data_file = _written(tmp_path, {f"{node}|run": {"m.py": [(1, 2)]}})
    found = CoverageMap(every_test=True).read(data_file, tmp_path)
    assert found.tests_for(Path("m.py"), 2) == (node,)


def test_only_a_yml_case_stays() -> None:
    assert YamlCases().keeps("tests/typing/a.yml::c")
    assert not YamlCases().keeps("spec/tests/test_a.py::test_b")
    assert not YamlCases().keeps("src/astrolabe/expr.py::astrolabe.expr.C")
