"""Keep the inferred return type of a function in the mypy cache, beside the function.

A function declared to return `InferredExpr`, `InferredDataFrame` or
`InferredLazyFrame` gets its return type from the values that its body marks
with `infer(...)`. mypy gives the type of each such value while it checks the
body, and the plugin keeps it in a record in memory. A later run can load the
module of the function from the mypy cache, and mypy then does not check the
body again, so that record stays empty.

So the hook on `infer` also writes the types of the marked values into the
symbol table that holds the function, under a hidden name. mypy writes that
table to its cache and loads it back on a later run. The hook on a call of the
function reads the hidden name when the record in memory holds nothing. The
name holds a character that no source file can spell, so it collides with
nothing. A change to the marked types changes the table, so mypy checks each
module that calls the function again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy import nodes
from mypy.nodes import GDEF, SymbolTableNode, Var
from mypy.types import AnyType, TupleType, TypeOfAny, get_proper_type

if TYPE_CHECKING:
    from mypy.nodes import FuncDef, SymbolTable
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup


class CachedVariants:
    """Write the variants of a function under a hidden name, and read them back.

    A variant is the type of one value that the function returns through
    `infer(...)`. This class writes the list of variants as one tuple type on a
    hidden variable, in the symbol table that holds the function: the table of
    its module, or the table of its class for a method. It reads the list back
    by the full name of that variable, which works for a function that mypy
    loads from its cache.

    Example:
        >>> hidden = CachedVariants().hidden_name("price_expr")
        >>> hidden.endswith("price_expr"), hidden.isidentifier()
        (True, False)
    """

    PREFIX: ClassVar[str] = "\uffffinfer\uffff"

    def hidden_name(self, function: str) -> str:
        """Give the hidden name that holds the variants of one function.

        Args:
            function: The short name of the function.

        Returns:
            The name, which no source file can spell.

        Example:
            >>> CachedVariants().hidden_name("f").endswith("f")
            True
        """
        return self.PREFIX + function

    def write(
        self,
        module_names: SymbolTable,
        funcdef: FuncDef,
        variants: list[MypyType],
        api: object,
    ) -> None:
        """Write the variants of one function beside it.

        The hidden name goes into the table of the class for a method,
        and into the table of the module for any other function.

        Args:
            module_names: The symbol table of the module that mypy checks.
            funcdef: The function whose body holds the `infer(...)` calls.
            variants: The type of each live variant, in the order of the
                body.
            api: The mypy API of the type-checking phase, which builds the
                fallback of the tuple type.
        """
        name = self.hidden_name(funcdef.name)
        fallback = api.named_generic_type(  # type: ignore[attr-defined]
            "builtins.tuple",
            [AnyType(TypeOfAny.special_form)],
        )
        var = Var(name, TupleType(list(variants), fallback))
        var._fullname = f"{self._owner(funcdef)}.{name}"  # ruff: ignore[private-member-access]
        is_method = funcdef.info is not nodes.FUNC_NO_INFO
        table = funcdef.info.names if is_method else module_names
        table[name] = SymbolTableNode(GDEF, var, plugin_generated=True)

    def read(self, plugin: PluginLookup, funcdef: FuncDef) -> list[MypyType]:
        """Read the variants of one function back from beside it.

        Args:
            plugin: The mypy plugin, which looks up a name.
            funcdef: The function whose variants a call needs.

        Returns:
            The type of each variant, or an empty list if nothing was
            written for the function.
        """
        found = plugin.lookup_fully_qualified(
            f"{self._owner(funcdef)}.{self.hidden_name(funcdef.name)}",
        )
        if found is None or not isinstance(found.node, Var):
            return []
        held = get_proper_type(found.node.type)
        if not isinstance(held, TupleType):  # pragma: no cover
            # The plugin writes this name, always as a tuple type.
            return []
        return list(held.items)

    def _owner(self, funcdef: FuncDef) -> str:
        """Give the full name of the scope that holds the function.

        Args:
            funcdef: The function.

        Returns:
            The full name of its class for a method, and of its module for
            any other function.
        """
        return funcdef.fullname.rsplit(".", maxsplit=1)[0]
