"""The rule that relates the two contexts in which an expression runs.

A polars expression runs in one of two contexts. In the row context,
for example in `select` or `with_columns`, it gives one value for each
row, or one value for the frame. In the aggregation context, inside
`group_by(...).agg(...)`, it runs once for each group. If an expression
gives the type `t` in the row context and works row by row, it gives
`List(t)` in the aggregation context, because each group collects a list
of values. Any other expression gives `t` in both contexts.

This rule holds for every expression. The few operations that accept
different data types in the two contexts carry one table for each
context in the catalog, and the rule then applies to each table.
"""

from __future__ import annotations


from astrolabe._hooks.expression.character import ELEMENTWISE, Character


class ContextRule:
    """Give the data type of an expression in either context, from its row-context type.

    The engine records each expression by its row-context answer only.
    Code that places an expression into a frame, or checks it against a
    declaration, must know the type in the real context. This class gives
    that type from the row-context answer and the character of the
    expression, that is, whether it works row by row, combines rows into
    one value, or is a fixed value.

    Example:
        >>> ContextRule().applied("Int64", ELEMENTWISE, "agg")
        'List(Int64)'
    """

    def applied(self, answer: str, character: Character, context: str) -> str:
        """Apply the rule to one row-context answer.

        Args:
            answer: The text of the data type that the expression gives in
                the row context.
            character: How the expression counts rows.
            context: "row" for the row context, or "agg" for the
                aggregation context.

        Returns:
            The text of the data type that the expression gives in
            `context`.

        Example:
            >>> from astrolabe._hooks.expression.character import AGGREGATING
            >>> ContextRule().applied("Int64", AGGREGATING, "agg")
            'Int64'
        """
        if context == "agg" and character is ELEMENTWISE:
            return f"List({answer})"
        return answer
