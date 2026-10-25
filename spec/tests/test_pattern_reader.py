"""`PatternReader` finds the same columns as polars for every pattern it reads."""

from __future__ import annotations

import polars as pl
import polars.selectors as cs
import pytest

from astrolabe._hooks.remainder.name_language import PatternReader

NAMES = (
    "a",
    "ab",
    "ba",
    "xab",
    "b",
    "a\nb",
    "A",
    "a1",
    "a٣",
    "a_b",
    "é",
    "a.b",
    "a$",
    "px_bid",
    "px_",
    "venue_px",
    "a]b",
    "x[0]",
)

READ = (
    "",
    "a",
    "^a",
    "a$",
    "^a$",
    "a|b",
    "^a|b$",
    "a.b",
    "^a.b$",
    "^a[._]b$",
    "^(a|b)+$",
    r"^a\.b$",
    "^[^b]$",
    "^.$",
    "^a$|^$",
    "^px_[a-z]*$",
    "_px$",
    "(a|b)x?",
    "^a]",
    r"\[0]$",
)

REFUSED = (
    "(^a)",
    r"^a\d$",
    r"^a\w",
    "(?i)^a$",
    "a{2}",
    "[[:alpha:]]",
    "[a&&b]",
    "[]a]",
    "a......",
    "abcdefghijklmnopqrstuvwxyz0123456789",
    "a**",
    "[z-a]",
)


class TestPatternReader:
    """Each pattern the reader takes picks the columns polars picks."""

    @pytest.mark.parametrize("pattern", READ)
    def test_the_reader_finds_what_polars_finds(self, pattern: str) -> None:
        """Compare the reader's names with the names `cs.matches` picks."""
        frame = pl.DataFrame({name: [1] for name in NAMES})
        names = PatternReader().language(pattern)
        assert names is not None
        found = [name for name in NAMES if names.accepts(name)]
        assert found == frame.select(cs.matches(pattern)).columns

    @pytest.mark.parametrize(
        "selector",
        [cs.matches("^a["), cs.matches("^a(?=b)"), pl.col("^a[$"), pl.all().exclude("^a[$")],  # type: ignore[invalid-call-shape]
        ids=["matches", "look-around", "col", "exclude"],
    )
    def test_polars_refuses_a_pattern_that_it_cannot_compile_at_every_frame(
        self,
        selector: pl.Expr,
    ) -> None:
        """Polars compiles the pattern where a frame expands it, also a frame of no column."""
        for frame in (pl.DataFrame(), pl.DataFrame({name: [1] for name in NAMES})):
            with pytest.raises(pl.exceptions.InvalidOperationError, match="invalid regex"):
                frame.select(selector)

    @pytest.mark.parametrize("pattern", REFUSED)
    def test_the_reader_refuses_what_it_cannot_read_safely(self, pattern: str) -> None:
        """A construct outside the part of the dialect it takes gives None."""
        assert PatternReader().language(pattern) is None
