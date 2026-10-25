"""Compose the code of an expression from the codes of its parts, in one canonical text.

A deferred type stores the code of an expression as text, for example
`pl.col('price') * 0.9`, beside the declared types that the code reads. The
code refers to the declared type at position `n` as `$n`. A part that the code
uses more than once is written once, at its first use, as `($k := (<code>))`,
and as `$k` after that, with `k` numbered after the declared types. So the text
grows with the code that the user wrote, and not with each reuse of a
variable.

Code that records a call builds the syntax tree of the call from the trees of
its parts (`CodeTree`), and this module writes the tree as text
(`CanonicalText`). The text is canonical: two trees with the same structure
give the same text, whatever spacing, quotes or order of keyword arguments the
user wrote. Python's own parser reads the text, once each reference is a
reserved name (`CodeRefs`), so the text holds no syntax of its own besides `$`.
"""

from __future__ import annotations

import ast
import io
import itertools
import tokenize
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.checked import checked_cast, checked_not_none

if TYPE_CHECKING:
    from collections.abc import Callable

    # `typing.TypeIs` exists from Python 3.13, and the checker supports 3.12.
    from typing_extensions import TypeIs


@dataclass(frozen=True)
class Code[L]:
    """Hold the code of an expression: its canonical text, and the declared types that it reads.

    `L` is the type of a declared type, which only the plugin knows. The text
    refers to `leaves[n]` as `$n`.

    Example:
        >>> Code("$0 * 0.9", ("Price",)).leaves
        ('Price',)
    """

    text: str
    leaves: tuple[L, ...] = ()


class CodeRefs:
    """Turn the references `$n` of a code into reserved names that Python parses, and back.

    The name of a reference holds its number, as `__ref_3__`. Only a token
    `$` that comes before a number is a reference, so a `$` inside a string
    literal stays as it is.

    Example:
        >>> CodeRefs().parseable("$0 * 0.9")
        '__ref_0__ * 0.9'
        >>> CodeRefs().printable("pl.col('$1') + __ref_1__")
        "pl.col('$1') + $1"
    """

    PREFIX: ClassVar[str] = "__ref_"
    SUFFIX: ClassVar[str] = "__"

    def parseable(self, text: str) -> str:
        """Give the text with each reference as a reserved name.

        Args:
            text: The code, with references `$n`.

        Returns:
            The text that `ast.parse` reads.

        Example:
            >>> CodeRefs().parseable("($1 := pl.col('a') + 1) * $1")
            "(__ref_1__ := pl.col('a') + 1) * __ref_1__"
            >>> CodeRefs().parseable("$0.abs()")
            '__ref_0__.abs()'
        """
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        pieces: list[str] = []
        position = 0
        for token, following in itertools.pairwise(tokens):
            if token.string == "$" and following.type == tokenize.NUMBER:
                start, end = token.start[1], following.end[1]
                digits = following.string.rstrip(".")
                dot = following.string[len(digits) :]
                pieces.extend((text[position:start], self.name(int(digits)), dot))
                position = end
        pieces.append(text[position:])
        return "".join(pieces)

    def printable(self, text: str) -> str:
        """Give the text with each reserved name as a reference.

        Args:
            text: The text that `ast.unparse` wrote.

        Returns:
            The code, with references `$n`.

        Example:
            >>> CodeRefs().printable("__ref_0__ * 0.9")
            '$0 * 0.9'
        """
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        pieces: list[str] = []
        position = 0
        for token in tokens:
            number = self.number(token.string) if token.type == tokenize.NAME else None
            if number is not None:
                start, end = token.start[1], token.end[1]
                pieces.extend((text[position:start], f"${number}"))
                position = end
        pieces.append(text[position:])
        return "".join(pieces)

    def name(self, number: int) -> str:
        """Give the reserved name of a reference.

        Args:
            number: The number of the reference.

        Returns:
            The name.

        Example:
            >>> CodeRefs().name(2)
            '__ref_2__'
        """
        return f"{self.PREFIX}{number}{self.SUFFIX}"

    def number(self, name: str) -> int | None:
        """Give the number of a reserved name, or `None` for any other name.

        Args:
            name: A name of the parsed text.

        Returns:
            The number, or `None`.

        Example:
            >>> CodeRefs().number("__ref_12__"), CodeRefs().number("pl")
            (12, None)
        """
        inner = name.removeprefix(self.PREFIX).removesuffix(self.SUFFIX)
        if inner == name or not inner.isdigit():
            return None
        return int(inner)


@dataclass
class CodeTree[L]:
    """Build the syntax tree of a call from the codes of its parts, with one list of declared types.

    The tree of each part comes from its text: each shared part takes the
    place of its references, so the tree holds the whole code. A shared part
    is one node in two places, so the tree costs no more than the text. Each
    declared type takes a name that `CanonicalText` reads, and two equal
    declared types take one name.

    Example:
        >>> tree = CodeTree[str]()
        >>> part = tree.part(Code("$0 * 0.9", ("Price",)))
        >>> ast.unparse(part), tree.leaves
        ('__leaf_0__ * 0.9', ['Price'])
    """

    leaves: list[L] = field(default_factory=list)

    LEAF_PREFIX: ClassVar[str] = "__leaf_"

    def part(self, code: Code[L]) -> ast.expr:
        """Give the tree of the code of a part.

        Args:
            code: The code.

        Returns:
            The tree.

        Example:
            >>> tree = CodeTree[str]()
            >>> part = tree.part(Code("($0 := pl.col('a') + 1) * $0"))
            >>> part.left is part.right
            True
        """
        parsed = ast.parse(CodeRefs().parseable(code.text), mode="eval").body
        names = [self.leaf(leaf) for leaf in code.leaves]
        return checked_cast(_Expander(names).visit(parsed), ast.expr)

    def leaf(self, leaf: L) -> ast.Name:
        """Give the name of a declared type in the tree.

        Args:
            leaf: The declared type.

        Returns:
            The name, the same for two equal declared types.

        Example:
            >>> tree = CodeTree[str]()
            >>> tree.leaf("Price").id, tree.leaf("Qty").id, tree.leaf("Price").id
            ('__leaf_0__', '__leaf_1__', '__leaf_0__')
        """
        if leaf not in self.leaves:
            self.leaves.append(leaf)
        return ast.Name(f"{self.LEAF_PREFIX}{self.leaves.index(leaf)}__", ast.Load())

    def code(self, tree: ast.expr) -> Code[L]:
        """Give the code of a tree that this builder made.

        Args:
            tree: The tree.

        Returns:
            The code, with its declared types in the order of the text.

        Example:
            >>> tree = CodeTree[str]()
            >>> sum_ = ast.parse("pl.col('a') + pl.col('b')", mode="eval").body
            >>> tree.code(ast.BinOp(sum_, ast.Mult(), sum_)).text
            "($0 := (pl.col('a') + pl.col('b'))) * $0"
        """
        return CanonicalText[L]().code(tree, self.leaves)


class _Expander(ast.NodeTransformer):
    """Put each shared part and each declared type of a parsed code in its place, in textual order.

    Example:
        >>> parsed = ast.parse("(__ref_1__ := a) + __ref_1__", mode="eval").body
        >>> expanded = _Expander([ast.Name("__leaf_0__", ast.Load())]).visit(parsed)
        >>> ast.unparse(expanded)
        'a + a'
    """

    def __init__(self, leaves: list[ast.Name]) -> None:
        """Keep the names of the declared types of the code.

        Args:
            leaves: The name of each declared type, in the order of the code.

        Example:
            >>> len(_Expander([])._shared)
            0
        """
        self._leaves = leaves
        self._shared: dict[int, ast.expr] = {}

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.expr:
        """Keep the value of a shared part, and put it in place of its definition.

        Args:
            node: The definition `($k := value)`.

        Returns:
            The value.

        Example:
            >>> parsed = ast.parse("(__ref_0__ := a)", mode="eval").body
            >>> ast.unparse(_Expander([]).visit(parsed))
            'a'
        """
        value = checked_cast(self.visit(node.value), ast.expr)
        number = checked_not_none(CodeRefs().number(node.target.id))
        self._shared[number] = value
        return value

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Put a declared type or a shared part in place of its reference.

        Args:
            node: A name.

        Returns:
            The name of the declared type, the shared part, or the name itself.

        Example:
            >>> leaf = ast.Name("__leaf_4__", ast.Load())
            >>> _Expander([leaf]).visit(ast.Name("__ref_0__", ast.Load())).id
            '__leaf_4__'
        """
        number = CodeRefs().number(node.id)
        if number is None:
            return node
        if number < len(self._leaves):
            return self._leaves[number]
        return self._shared[number]


class _Fields:
    """Walk the fields of a syntax node: rebuild a node from its children, or list them.

    A field holds a node, a list of nodes, or a plain value. A context such as
    `Load` is part of the node, and not a child.

    Example:
        >>> call = ast.parse("f(x)", mode="eval").body
        >>> [type(child).__name__ for child in _Fields().children(call)]
        ['Name', 'Name']
    """

    def mapped(self, node: ast.AST, function: Callable[[ast.AST], ast.AST]) -> ast.AST:
        """Give a copy of a node with a function applied to each child.

        Args:
            node: The node.
            function: The function that gives the new child.

        Returns:
            The copy.

        Example:
            >>> call = ast.parse("f(x)", mode="eval").body
            >>> ast.unparse(_Fields().mapped(call, lambda child: child))
            'f(x)'
        """
        values: dict[str, object] = {}
        for name, value in ast.iter_fields(node):
            values[name] = self._mapped_field(value, function)
        return type(node)(**values)

    def children(self, node: ast.AST) -> list[ast.AST]:
        """Give the children of a node, in the order of its fields.

        Args:
            node: The node.

        Returns:
            The children.

        Example:
            >>> _Fields().children(ast.Constant(1))
            []
        """
        children: list[ast.AST] = []
        for _, value in ast.iter_fields(node):
            items = value if isinstance(value, list) else [value]
            children.extend(item for item in items if self._is_child(item))
        return children

    def _mapped_field(self, value: object, function: Callable[[ast.AST], ast.AST]) -> object:
        """Give the new value of one field.

        Args:
            value: The value of the field.
            function: The function that gives the new child.

        Returns:
            The new value.

        Example:
            >>> _Fields()._mapped_field([1, "a"], lambda child: child)
            [1, 'a']
        """
        if isinstance(value, list):
            return [function(item) if self._is_child(item) else item for item in value]
        return function(value) if self._is_child(value) else value

    def _is_child(self, value: object) -> TypeIs[ast.AST]:
        """Say whether the value of a field is a child node.

        Args:
            value: The value.

        Returns:
            Whether it is a node, and not a context such as `Load`.

        Example:
            >>> _Fields()._is_child(ast.Load()), _Fields()._is_child(ast.Constant(1))
            (False, True)
        """
        return isinstance(value, ast.AST) and not isinstance(value, ast.expr_context)


class CanonicalText[L]:
    """Write a tree of code as canonical text, with each repeated part written once.

    The text follows `ast.unparse`, so the spacing, the quotes and the
    parentheses are fixed. Keyword arguments are in the order of their names.
    The declared types take the numbers `$0`, `$1` and so on in the order of
    their first place in the text. A part that the tree holds in two places or
    more, with at least `SHARED_SIZE` nodes, is written once as
    `($k := <code>)` and then as `$k`, numbered after the declared types.

    Example:
        >>> tree = ast.parse("f(b=1, a=2)", mode="eval").body
        >>> CanonicalText[str]().code(tree, []).text
        'f(a=2, b=1)'
    """

    SHARED_SIZE: ClassVar[int] = 8

    def code(self, tree: ast.expr, leaves: list[L]) -> Code[L]:
        """Give the code of a tree.

        Args:
            tree: The tree, whose declared types are names of `CodeTree`.
            leaves: The declared types that those names number.

        Returns:
            The code.

        Example:
            >>> tree = ast.parse("__leaf_1__ + __leaf_0__", mode="eval").body
            >>> CanonicalText[str]().code(tree, ["a", "b"])
            Code(text='$0 + $1', leaves=('b', 'a'))
        """
        merged = _Merger().merged(tree)
        order = _Order(merged)
        used = [leaves[number] for number in _LeafOrder().of(merged)]
        writer = _Writer(order, leaves, used)
        text = ast.unparse(writer.visit(merged))
        return Code(CodeRefs().printable(text), tuple(used))


class _Merger:
    """Make one node of each set of equal parts of a tree, with the keyword arguments in order.

    A repeated part is then one node, and the order of the fields of each
    node is the order of the text.

    Example:
        >>> tree = ast.parse("f(1) + f(1)", mode="eval").body
        >>> merged = _Merger().merged(tree)
        >>> merged.left is merged.right
        True
    """

    def __init__(self) -> None:
        """Start with no part seen.

        Example:
            >>> _Merger()._by_key
            {}
        """
        self._by_key: dict[object, ast.AST] = {}
        self._keys: dict[int, object] = {}

    def merged(self, tree: ast.expr) -> ast.expr:
        """Give the tree with each set of equal parts as one node.

        Args:
            tree: The tree.

        Returns:
            The merged tree.

        Example:
            >>> merged = _Merger().merged(ast.parse("[g(z=1, a=2), g(a=2, z=1)]", mode="eval").body)
            >>> merged.elts[0] is merged.elts[1]
            True
        """
        return checked_cast(self._node(tree), ast.expr)

    def _node(self, node: ast.AST) -> ast.AST:
        """Give the one node of the parts equal to a node.

        Args:
            node: The node.

        Returns:
            The node that stands for its equal parts.

        Example:
            >>> merger = _Merger()
            >>> merger._node(ast.Constant(1)) is merger._node(ast.Constant(1))
            True
        """
        rebuilt = self._rebuilt(node)
        key = self._key(rebuilt)
        return self._by_key.setdefault(key, rebuilt)

    def _rebuilt(self, node: ast.AST) -> ast.AST:
        """Give a copy of a node whose children are the merged nodes.

        Args:
            node: The node.

        Returns:
            The copy, with the keyword arguments of a call in the order of their names.

        Example:
            >>> ast.unparse(_Merger()._rebuilt(ast.parse("f(x, b=1, a=2)", mode="eval").body))
            'f(x, a=2, b=1)'
        """
        rebuilt = _Fields().mapped(node, self._node)
        if isinstance(rebuilt, ast.Call):
            rebuilt.keywords = sorted(rebuilt.keywords, key=lambda keyword: keyword.arg or "")
        return rebuilt

    def _key(self, node: ast.AST) -> object:
        """Give the structure of a node, whose children are merged nodes, as a value.

        Args:
            node: The node.

        Returns:
            The key: two nodes of one structure have one key.

        Example:
            >>> _Merger()._key(ast.Constant(1)) == _Merger()._key(ast.Constant(True))
            False
        """
        parts: list[object] = [type(node).__name__]
        for name, value in ast.iter_fields(node):
            parts.append((name, self._field_key(value)))
        key = tuple(parts)
        self._keys[id(node)] = key
        return key

    def _field_key(self, value: object) -> object:
        """Give the key of one field of a node.

        Args:
            value: The value of the field: a node, a list, or a plain value.

        Returns:
            The key.

        Example:
            >>> _Merger()._field_key([1, "a"])
            (('int', 1), ('str', 'a'))
        """
        if isinstance(value, ast.expr_context):
            return None
        if isinstance(value, ast.AST):
            return self._keys.get(id(value), id(value))
        if isinstance(value, list):
            return tuple(self._field_key(item) for item in value)
        return (type(value).__name__, value)


class _Children:
    """Give the child nodes of a node in the order of the text, which a dict interleaves.

    Example:
        >>> node = ast.parse("{'a': f, 'b': g}", mode="eval").body
        >>> [ast.unparse(child) for child in _Children().of(node)]
        ["'a'", 'f', "'b'", 'g']
    """

    def of(self, node: ast.AST) -> list[ast.AST]:
        """Give the child nodes of a node.

        Args:
            node: The node.

        Returns:
            Its children, in the order of the text.

        Example:
            >>> negation = ast.parse("-a", mode="eval").body
            >>> [type(child).__name__ for child in _Children().of(negation)]
            ['USub', 'Name']
        """
        if isinstance(node, ast.Dict):
            pairs = zip(node.keys, node.values, strict=True)
            return [part for pair in pairs for part in pair if part is not None]
        return _Fields().children(node)


class _Order:
    """Count the places of each node of a merged tree, and its size as a tree.

    The places of a node are the fields that hold it, over the whole tree.

    Example:
        >>> merged = _Merger().merged(ast.parse("f(1) + f(1)", mode="eval").body)
        >>> _Order(merged).places[id(merged.left)]
        2
    """

    def __init__(self, tree: ast.AST) -> None:
        """Count the places and the sizes of the nodes of a tree.

        Args:
            tree: The merged tree.

        Example:
            >>> merged = _Merger().merged(ast.parse("f(1)", mode="eval").body)
            >>> _Order(merged).sizes[id(merged)]
            3
        """
        self.places: dict[int, int] = {}
        self.sizes: dict[int, int] = {}
        self._visit(tree)

    def _visit(self, node: ast.AST) -> int:
        """Count the places below a node seen for the first time, and give its size.

        Args:
            node: The node.

        Returns:
            The number of nodes of its tree.

        Example:
            >>> _Order(ast.Constant(1))._visit(ast.parse("-a", mode="eval").body)
            3
        """
        size = 1
        for child in _Children().of(node):
            self.places[id(child)] = self.places.get(id(child), 0) + 1
            known = self.sizes.get(id(child))
            size += self._visit(child) if known is None else known
        self.sizes[id(node)] = size
        return size


class _LeafOrder:
    """Give the numbers of the declared types of a merged tree, in the order of the text.

    A node in two places is read once, at its first place, as the text
    writes a shared part in full only there.

    Example:
        >>> tree = ast.parse("g(__leaf_2__, __leaf_0__, __leaf_2__)", mode="eval").body
        >>> _LeafOrder().of(_Merger().merged(tree))
        [2, 0]
    """

    def of(self, tree: ast.AST) -> list[int]:
        """Give the numbers of the declared types, in their order.

        Args:
            tree: The merged tree.

        Returns:
            The numbers that the names of `CodeTree` hold, each once.

        Example:
            >>> _LeafOrder().of(ast.Name("x", ast.Load()))
            []
        """
        found: list[int] = []
        seen: set[int] = set()
        pending: list[ast.AST] = [tree]
        while pending:
            node = pending.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            number = self.number(node)
            if number is not None and number not in found:
                found.append(number)
            pending.extend(reversed(_Children().of(node)))
        return found

    def number(self, node: ast.AST) -> int | None:
        """Give the number of a declared type that a node names, or `None`.

        Args:
            node: The node.

        Returns:
            The number, or `None`.

        Example:
            >>> _LeafOrder().number(ast.Name("__leaf_3__", ast.Load()))
            3
        """
        prefix = CodeTree.LEAF_PREFIX
        if not isinstance(node, ast.Name) or not node.id.startswith(prefix):
            return None
        return int(node.id.removeprefix(prefix).removesuffix("__"))


class _Writer[L](ast.NodeTransformer):
    """Copy a merged tree for `ast.unparse`, with the references of the canonical text.

    A declared type becomes `$n`, by its place among the declared types in use.
    A shared part becomes `($k := <part>)` at its first place and `$k` after
    that, numbered after the declared types.

    Example:
        >>> merged = _Merger().merged(ast.parse("__leaf_0__ + 1", mode="eval").body)
        >>> ast.unparse(_Writer(_Order(merged), ["Price"], ["Price"]).visit(merged))
        '__ref_0__ + 1'
    """

    def __init__(self, order: _Order, leaves: list[L], used: list[L]) -> None:
        """Keep the counts of the tree and its declared types.

        Args:
            order: The places and the sizes of the nodes.
            leaves: The declared types that the names of the tree number.
            used: The declared types in the order of the text.

        Example:
            >>> _Writer(_Order(ast.Constant(1)), [], [])._shared
            {}
        """
        self._order = order
        self._leaves = leaves
        self._used = used
        self._shared: dict[int, int] = {}

    def generic_visit(self, node: ast.AST) -> ast.AST:
        """Copy a node, with a shared part named at its first place and referenced after.

        Args:
            node: The node.

        Returns:
            The copy, the definition of a shared part, or its reference.

        Example:
            >>> pair = ast.parse("[f(1, 2, 3, 4, 5, 6)] * 2", mode="eval").body
            >>> merged = _Merger().merged(ast.List([pair.left.elts[0]] * 2))
            >>> ast.unparse(_Writer(_Order(merged), [], []).visit(merged))
            '[(__ref_0__ := f(1, 2, 3, 4, 5, 6)), __ref_0__]'
        """
        number = self._shared.get(id(node))
        if number is not None:
            return ast.Name(CodeRefs().name(number), ast.Load())
        copy = self._copied(node)
        if not self._shares(node):
            return copy
        number = len(self._used) + len(self._shared)
        self._shared[id(node)] = number
        target = ast.Name(CodeRefs().name(number), ast.Store())
        return ast.NamedExpr(target, checked_cast(copy, ast.expr))

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Give a declared type its number, by its place among the declared types in use.

        Args:
            node: A name.

        Returns:
            The reference of the declared type, or the name itself.

        Example:
            >>> writer = _Writer(_Order(ast.Constant(1)), ["Qty", "Price"], ["Price"])
            >>> writer.visit(ast.Name("__leaf_1__", ast.Load())).id
            '__ref_0__'
        """
        number = _LeafOrder().number(node)
        if number is None:
            return node
        return ast.Name(CodeRefs().name(self._used.index(self._leaves[number])), ast.Load())

    def _copied(self, node: ast.AST) -> ast.AST:
        """Copy a node, with each child written in the order of its fields.

        Args:
            node: The node.

        Returns:
            The copy.

        Example:
            >>> merged = _Merger().merged(ast.parse("f(x)", mode="eval").body)
            >>> ast.unparse(_Writer(_Order(merged), [], [])._copied(merged))
            'f(x)'
        """
        if isinstance(node, ast.Dict):
            return self._dict(node)
        return _Fields().mapped(node, self.visit)

    def _dict(self, node: ast.Dict) -> ast.Dict:
        """Copy a dict with its keys and values in the order of the text.

        Args:
            node: The dict.

        Returns:
            The copy.

        Example:
            >>> merged = _Merger().merged(ast.parse("{'a': 1}", mode="eval").body)
            >>> ast.unparse(_Writer(_Order(merged), [], [])._dict(merged))
            "{'a': 1}"
        """
        keys: list[ast.expr | None] = []
        values: list[ast.expr] = []
        for key, value in zip(node.keys, node.values, strict=True):
            keys.append(None if key is None else checked_cast(self.visit(key), ast.expr))
            values.append(checked_cast(self.visit(value), ast.expr))
        return ast.Dict(keys, values)

    def _shares(self, node: ast.AST) -> bool:
        """Say whether a node is a shared part: in two places or more, and large enough.

        Args:
            node: The node.

        Returns:
            Whether it is.

        Example:
            >>> _Writer(_Order(ast.Constant(1)), [], [])._shares(ast.Constant(1))
            False
        """
        if not isinstance(node, ast.expr):
            return False
        places = self._order.places.get(id(node), 1)
        return places > 1 and self._order.sizes.get(id(node), 1) >= CanonicalText.SHARED_SIZE
