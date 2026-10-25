"""Check `LiveInferCallWalker`, which finds the `infer(...)` calls in the body of a function.

The walk enters each kind of block, such as `if`, `try`, `for`, `while`,
`with` and `match`. In one mode it keeps only a call that is the whole
value of a `return`. In the other mode it also finds a call in an
expression statement or on the right side of an assignment, where
`infer` is an error.
"""

from __future__ import annotations

from mypy.nodes import (
    AssignmentStmt,
    Block,
    ExpressionStmt,
    ForStmt,
    IfStmt,
    MatchStmt,
    NameExpr,
    ReturnStmt,
    TryStmt,
    WhileStmt,
    WithStmt,
)
from mypy.patterns import AsPattern

from astrolabe._hooks.inferred.walker import LiveInferCallWalker
from tests.support import infer_call_node, int_literal_node


def test_finds_call_in_bare_return() -> None:
    call = infer_call_node()
    body = Block([ReturnStmt(call)])
    assert LiveInferCallWalker(only_returned=True).find(body) == {id(call)}


def test_return_with_no_expression_is_ignored() -> None:
    body = Block([ReturnStmt(None)])
    assert LiveInferCallWalker(only_returned=True).find(body) == set()


def test_return_with_non_infer_expression_is_ignored() -> None:
    body = Block([ReturnStmt(int_literal_node(1))])
    assert LiveInferCallWalker(only_returned=True).find(body) == set()


def test_if_with_no_else_body_visits_only_the_true_branch() -> None:
    call = infer_call_node()
    if_stmt = IfStmt([NameExpr("flag")], [Block([ReturnStmt(call)])], None)
    body = Block([if_stmt])
    assert LiveInferCallWalker(only_returned=True).find(body) == {id(call)}


def test_unrecognized_statement_is_ignored() -> None:
    body = Block([ExpressionStmt(int_literal_node(1))])
    assert LiveInferCallWalker(only_returned=True).find(body) == set()


def test_finds_call_in_if_and_elif_branches() -> None:
    call_a = infer_call_node()
    call_b = infer_call_node()
    if_stmt = IfStmt(
        [NameExpr("flag")],
        [Block([ReturnStmt(call_a)])],
        Block([ReturnStmt(call_b)]),
    )
    body = Block([if_stmt])
    assert LiveInferCallWalker(only_returned=True).find(body) == {id(call_a), id(call_b)}


def test_finds_call_in_try_except_else_finally() -> None:
    try_call = infer_call_node()
    except_call = infer_call_node()
    else_call = infer_call_node()
    finally_call = infer_call_node()
    try_stmt = TryStmt(
        Block([ReturnStmt(try_call)]),
        [None],
        [None],
        [Block([ReturnStmt(except_call)])],
        Block([ReturnStmt(else_call)]),
        Block([ReturnStmt(finally_call)]),
    )
    body = Block([try_stmt])
    found = LiveInferCallWalker(only_returned=True).find(body)
    assert found == {id(try_call), id(except_call), id(else_call), id(finally_call)}


def test_finds_call_in_for_body_and_else() -> None:
    body_call = infer_call_node()
    else_call = infer_call_node()
    for_stmt = ForStmt(
        NameExpr("item"),
        NameExpr("items"),
        Block([ReturnStmt(body_call)]),
        Block([ReturnStmt(else_call)]),
    )
    body = Block([for_stmt])
    assert LiveInferCallWalker(only_returned=True).find(body) == {id(body_call), id(else_call)}


def test_finds_call_in_while_body_and_else() -> None:
    body_call = infer_call_node()
    else_call = infer_call_node()
    while_stmt = WhileStmt(
        NameExpr("flag"),
        Block([ReturnStmt(body_call)]),
        Block([ReturnStmt(else_call)]),
    )
    body = Block([while_stmt])
    assert LiveInferCallWalker(only_returned=True).find(body) == {id(body_call), id(else_call)}


def test_finds_call_in_with_body() -> None:
    call = infer_call_node()
    with_stmt = WithStmt([NameExpr("ctx")], [None], Block([ReturnStmt(call)]))
    body = Block([with_stmt])
    assert LiveInferCallWalker(only_returned=True).find(body) == {id(call)}


def test_finds_call_in_match_case_bodies() -> None:
    first_call = infer_call_node()
    second_call = infer_call_node()
    match_stmt = MatchStmt(
        NameExpr("subject"),
        [AsPattern(None, None), AsPattern(None, None)],
        [None, None],
        [Block([ReturnStmt(first_call)]), Block([ReturnStmt(second_call)])],
    )
    body = Block([match_stmt])
    found = LiveInferCallWalker(only_returned=True).find(body)
    assert found == {id(first_call), id(second_call)}


def test_only_returned_excludes_bare_expression_statement() -> None:
    call = infer_call_node()
    body = Block([ExpressionStmt(call)])
    assert LiveInferCallWalker(only_returned=True).find(body) == set()


def test_full_scan_finds_bare_expression_statement() -> None:
    call = infer_call_node()
    body = Block([ExpressionStmt(call)])
    assert LiveInferCallWalker(only_returned=False).find(body) == {id(call)}


def test_only_returned_excludes_assignment() -> None:
    call = infer_call_node()
    body = Block([AssignmentStmt([NameExpr("result")], call)])
    assert LiveInferCallWalker(only_returned=True).find(body) == set()


def test_full_scan_finds_assignment_rvalue() -> None:
    call = infer_call_node()
    body = Block([AssignmentStmt([NameExpr("result")], call)])
    assert LiveInferCallWalker(only_returned=False).find(body) == {id(call)}
