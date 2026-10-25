"""Find the `infer(...)` calls in the current body of a function.

The checker keeps the type of each `infer(...)` call from the moment
that mypy checks it. That record can hold a call that the current body
does not have, or a call in a place where `infer` does not work. The walk here
reads the body as it is now, and says which calls it holds and where.
"""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

from mypy.nodes import (
    AssignmentStmt,
    Block,
    ExpressionStmt,
    ForStmt,
    IfStmt,
    MatchStmt,
    ReturnStmt,
    TryStmt,
    WhileStmt,
    WithStmt,
)

from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.dispatch.generic import Target
from astrolabe.declare import infer

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.nodes import Expression, Statement


class LiveInferCallWalker:
    """Find the `infer(...)` calls in the body of one function, by the identity of each node.

    Two checks need this walk. The mypy daemon, `dmypy`, keeps old
    records after the user edits a function, and a walk that keeps only
    returned calls tells the live records from the old ones. A walk over
    every statement also finds a call in a wrong place, for example
    `result = infer(x)`, which is an error. `infer(x)` works only as the
    whole expression of a `return` statement.

    The walk enters the blocks of `if`, `try`, `for`, `while`, `with` and
    `match`. It finds a call that is the whole expression of a `return`,
    of an expression statement or of the right side of an assignment. It
    does not look inside a larger expression, so it misses
    `return infer(x) + 1` and `y = [infer(x)]`. A call in code that
    cannot run does no harm: mypy does not check that code, so the hook
    on `infer` never records the call.

    Example:
        >>> body = return_block(infer_call_node())
        >>> len(LiveInferCallWalker(only_returned=True).find(body))
        1
    """

    _TARGET: Final[Target] = Target.of(infer)

    def __init__(self, *, only_returned: bool) -> None:
        """Choose which calls the walk collects, and map each statement class to its visit.

        Args:
            only_returned: If True, collect only a call that is the
                expression of a `return` statement. If False, also
                collect a call that is an expression statement or the
                right side of an assignment.
        """
        self._ids: set[int] = set()
        self._stmt_visitors: dict[type, Callable[[Statement], None]] = {
            Block: self._visit_block,
            IfStmt: self._visit_if,
            TryStmt: self._visit_try,
            ForStmt: self._visit_for,
            WhileStmt: self._visit_while,
            WithStmt: self._visit_with,
            MatchStmt: self._visit_match,
            ReturnStmt: self._visit_return,
        }
        if not only_returned:
            self._stmt_visitors[ExpressionStmt] = self._visit_expression_stmt
            self._stmt_visitors[AssignmentStmt] = self._visit_assignment

    def find(self, body: Block) -> set[int]:
        """Walk one body, and give the calls that it holds.

        Args:
            body: The block that is the body of the function.

        Returns:
            The `id()` of each call node that the walk collects.

        Example: see class docstring.
        """
        self._ids = set()
        self._visit_block(body)
        return self._ids

    def _visit_block(self, stmt: Statement) -> None:
        """Send each statement of a block to the visit for its class, in order.

        A statement of a class that the walk does not visit is skipped.

        Args:
            stmt: The block.
        """
        block = checked_cast(stmt, Block)
        for inner in block.body:
            visitor = self._stmt_visitors.get(type(inner))
            if visitor is not None:
                visitor(inner)

    def _visit_if(self, stmt: Statement) -> None:
        """Visit each branch of an `if` statement, `elif` and `else` included.

        Args:
            stmt: The `if` statement.
        """
        if_stmt = checked_cast(stmt, IfStmt)
        for block in if_stmt.body:
            self._visit_block(block)
        self._visit_optional_block(if_stmt.else_body)

    def _visit_try(self, stmt: Statement) -> None:
        """Visit the body, each `except` block, and the `else` and `finally` blocks of a `try`.

        Args:
            stmt: The `try` statement.
        """
        try_stmt = checked_cast(stmt, TryStmt)
        self._visit_block(try_stmt.body)
        for handler in try_stmt.handlers:
            self._visit_block(handler)
        self._visit_optional_block(try_stmt.else_body)
        self._visit_optional_block(try_stmt.finally_body)

    def _visit_for(self, stmt: Statement) -> None:
        """Visit the body and the `else` block of a `for` loop.

        Args:
            stmt: The `for` statement.
        """
        for_stmt = checked_cast(stmt, ForStmt)
        self._visit_block(for_stmt.body)
        self._visit_optional_block(for_stmt.else_body)

    def _visit_while(self, stmt: Statement) -> None:
        """Visit the body and the `else` block of a `while` loop.

        Args:
            stmt: The `while` statement.
        """
        while_stmt = checked_cast(stmt, WhileStmt)
        self._visit_block(while_stmt.body)
        self._visit_optional_block(while_stmt.else_body)

    def _visit_with(self, stmt: Statement) -> None:
        """Visit the body of a `with` statement.

        Args:
            stmt: The `with` statement.
        """
        with_stmt = checked_cast(stmt, WithStmt)
        self._visit_block(with_stmt.body)

    def _visit_match(self, stmt: Statement) -> None:
        """Visit the block of each `case` of a `match` statement.

        Args:
            stmt: The `match` statement.
        """
        match_stmt = checked_cast(stmt, MatchStmt)
        for block in match_stmt.bodies:
            self._visit_block(block)

    def _visit_optional_block(self, block: Block | None) -> None:
        """Visit a block that a statement can leave out, for example an `else` or a `finally`.

        Args:
            block: The block, or None if the statement has no such
                clause.
        """
        if block is not None:
            self._visit_block(block)

    def _visit_return(self, stmt: Statement) -> None:
        """Collect the expression of a `return` statement, if it is an `infer(...)` call.

        Args:
            stmt: The `return` statement.
        """
        return_stmt = checked_cast(stmt, ReturnStmt)
        if return_stmt.expr is not None:
            self._record_if_infer_call(return_stmt.expr)

    def _visit_expression_stmt(self, stmt: Statement) -> None:
        """Collect an expression statement, if it is an `infer(...)` call.

        Args:
            stmt: The expression statement.
        """
        expr_stmt = checked_cast(stmt, ExpressionStmt)
        self._record_if_infer_call(expr_stmt.expr)

    def _visit_assignment(self, stmt: Statement) -> None:
        """Collect the right side of an assignment, if it is an `infer(...)` call.

        Args:
            stmt: The assignment statement.
        """
        assignment = checked_cast(stmt, AssignmentStmt)
        self._record_if_infer_call(assignment.rvalue)

    def _record_if_infer_call(self, expr: Expression) -> None:
        """Collect the `id()` of an expression, if it is a call to `infer`.

        Args:
            expr: The expression to look at.
        """
        if self._TARGET.matches_call(expr):
            self._ids.add(id(expr))
