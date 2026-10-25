"""Read the `gives=` argument of a declaration: what the expression gives.

The value passed as `gives=` is an outcome. It says which data type the
declared expression gives for the data types that it reads. It can be a
fixed data type, the data type of a read, or a container around another
outcome. The code here reads it into the engine value `Outcome`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.calls.dtype_constructor_readers import (
    SemanticAnalyzerLiteralTypeBuilder,
)
from astrolabe._hooks.calls.dtype_node import DtypeNodeReader
from astrolabe._hooks.declaring.site import CallShape, DeclarationCall
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeReads
from astrolabe._hooks.expression.outcome import Axis, Const, Outcome, Wrap
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe.declare import same, wrap

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import DynamicClassDefContext


class OutcomeParser:
    """Read the node passed as `gives=` into an `Outcome`.

    The user can write an outcome in four forms:

    - a data type, for example `pl.Boolean`, which is fixed whatever the
      expression reads,
    - `same`, the data type of the one column that the declaration
      reads, with its parameters,
    - `same(name)`, the data type of one named read among several,
    - `wrap(constructor, outcome)`, a container such as `pl.List` around
      another outcome.

    Both the parser of an expression declaration and the parser of a
    selection declaration read `gives=` with this class. The code reads
    the syntax during semantic analysis, the mypy phase that resolves
    names before any expression has a type.
    """

    def __init__(self, ctx: DynamicClassDefContext) -> None:
        """Keep the context of the declaration, and build the readers that this class uses.

        Args:
            ctx: The mypy context of the declaration. The data type
                reader resolves names through it.
        """
        self._ctx = ctx
        self._dtypes = DtypeNodeReader(SemanticAnalyzerLiteralTypeBuilder())
        self._shape = CallShape()

    def parsed(self, node: Expression, sole_read: str | None) -> Outcome | None:
        """Read one outcome, in whichever of the four forms the user wrote it.

        Args:
            node: The node of the outcome.
            sole_read: The name of the one read, which a bare `same`
                refers to. It is the empty name for a read whose name
                the argument fills, and None if the declaration does not
                read exactly one column.

        Returns:
            The outcome, or None if the node has none of the four forms.
        """
        readers = (self._bare_same, self._named_same, self._wrap, self._constant)
        for reader in readers:
            outcome = reader(node, sole_read)
            if outcome is not None:
                return outcome
        return None

    def _bare_same(self, node: Expression, sole_read: str | None) -> Outcome | None:
        """Read a bare `same` as the data type of the one read.

        Args:
            node: The node.
            sole_read: The name of the one read, or None if there is not
                exactly one.

        Returns:
            The outcome that gives back the data type of that read, or
            None for any other node or if there is not exactly one read.

        Example:
            >>> parser = OutcomeParser.__new__(OutcomeParser)
            >>> parser._shape = CallShape()
            >>> parser._bare_same(str_literal_node("x"), "q") is None
            True
        """
        if sole_read is None or not self._shape.refers_to(node, same, "same"):
            return None
        return Axis(sole_read)

    @propagates(None)
    def _named_same(self, node: Expression, sole_read: str | None) -> Outcome | None:
        """Read `same(name)` as the data type of the named read.

        Args:
            node: The node.
            sole_read: Not used. Every reader of a form takes the same
                arguments, so that `parsed` can try them in a loop.

        Returns:
            The outcome that gives back the data type of that read, or
            None for any other node or a name that is not a string
            literal.
        """
        del sole_read
        call = DeclarationCall(not_none(self._shape.call_of_value(node, same, "same")))
        return Axis(not_none(self._shape.literal(call.argument(0, "read"))))

    @propagates(None)
    def _wrap(self, node: Expression, sole_read: str | None) -> Outcome | None:
        """Read `wrap(constructor, outcome)` as a container around an inner outcome.

        The container is the kind of the constructor, that is, the name of the data type
        without its parameters. For example, `pl.List` gives `List`.

        Args:
            node: The node.
            sole_read: The name of the one read, which a bare `same`
                inside the wrap refers to.

        Returns:
            The container outcome, or None for any other node, or if
            the constructor or the inner outcome does not read.
        """
        call = DeclarationCall(not_none(self._shape.call(node, wrap)))
        constructor_node = not_none(call.argument(0, "constructor"))
        constructor = not_none(self._rendering(constructor_node)).split("(", maxsplit=1)[0]
        inner = not_none(self.parsed(not_none(call.argument(1, "inner")), sole_read))
        return Wrap(constructor, inner)

    @propagates(None)
    def _constant(self, node: Expression, sole_read: str | None) -> Outcome | None:
        """Read a data type as a fixed outcome.

        Args:
            node: The node of the data type.
            sole_read: Not used. Every reader of a form takes the same
                arguments, so that `parsed` can try them in a loop.

        Returns:
            The fixed outcome, or None if the node names no data type.
        """
        del sole_read
        return Const(not_none(self._rendering(node)))

    @propagates(None)
    def _rendering(self, node: Expression) -> str | None:
        """Give the text of the data type that a node names.

        Args:
            node: The node of the data type.

        Returns:
            The text, for example `Datetime(time_unit='us', time_zone=None)`,
            or None if the node names no data type.
        """
        instance = not_none(self._dtypes.instance(self._ctx, node))
        return ConstDtypeReads().read(instance)
