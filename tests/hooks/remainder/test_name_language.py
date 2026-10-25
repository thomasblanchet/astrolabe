"""Check how the checker reads a name set back from the text that its cache keeps.

A type that mypy loads from its cache holds a set of column names only as
a text, and `SignatureParser` rebuilds the automaton from it. The checker
writes that text itself, so a text of another form is a fault in the
caller, and the parser must say so at once.
"""

from __future__ import annotations

import pytest

from astrolabe._hooks.remainder.name_language import MachineSignature, SignatureParser


@pytest.mark.parametrize("text", ["zz", "0F[.>0"])
def test_a_text_that_is_not_a_signature_is_refused(text: str) -> None:
    with pytest.raises((ValueError, IndexError)):
        SignatureParser(text).machine()


def test_a_signature_reads_back_the_names_it_accepts() -> None:
    machine = SignatureParser("0F[.>0]").machine()
    assert SignatureParser(MachineSignature(machine).text()).machine().accepts("ab")
