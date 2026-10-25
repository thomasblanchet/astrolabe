"""Check the code of a deferred type: its canonical text, and the reader that builds it back.

A deferred type stores the code of an expression as text. `CodeTree` and
`CanonicalText` compose that text from the codes of the parts of a call, and
`CodeReader` builds the polars expression of a text again, with an
interpreter that accepts only the forms that the recorder writes. These
tests cover the forms that the reader refuses, and the parts of the text
that the recorder writes only in rare calls.
"""

from __future__ import annotations

import ast
import re

import pytest

from astrolabe._hooks.deferred.code import CanonicalText, Code
from astrolabe._hooks.deferred.code_reader import CodeReader, CodeReadError


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("pl.col(", "statement"),
        ("eval('1')", "the name 'eval' is not allowed"),
        ("(x := 1)", "the name 'x' is not a shared part"),
        ("pl._private", "the attribute '_private' is not allowed"),
        ("pl.nope", "has no attribute 'nope'"),
        ("(1)(2)", "the call of a value that is not callable"),
        ("pl.col(1)", "invalid input for `col`"),
        ("ZoneInfo('No/Zone')", "polars refuses the call"),
        ("hole('tuple')", "a hole of the kind 'tuple' has no placeholder value"),
        ("not pl.col('a')", "the operator not is not allowed"),
        ("1 < pl.col('a') < 2", "only one comparison of two values is allowed"),
        ("{**{'a': 1}}", "a dict that unpacks a dict is not allowed"),
        ("pl.col('a').alias(**[1])", "a ** argument must be a dict"),
        ("pl.col(*1)", "a * argument must be a list or a tuple"),
        ("abs(hole('function'))", "a function of the user reaches only a method of an expression"),
        (
            "'a'.upper(hole('function'))",
            "a function of the user reaches only a method of an expression",
        ),
        ("pl.col('a') + pl.col", "cannot create expression literal"),
        ("1 // 0", "polars refuses the operation"),
    ],
)
def test_the_reader_refuses_a_form_that_the_recorder_does_not_write(
    text: str, message: str
) -> None:
    with pytest.raises(CodeReadError, match=re.escape(message)):
        CodeReader().read(Code(text))


def test_a_repeated_dict_is_written_once() -> None:
    entry = "{'a': 1, 'b': 2, 'c': 3, 'd': 4}"
    tree = ast.parse(f"f({entry}, {entry})", mode="eval").body
    text = CanonicalText[str]().code(tree, []).text
    assert text == "f(($0 := {'a': 1, 'b': 2, 'c': 3, 'd': 4}), $0)"
