"""Check the names of `astrolabe._rules.failures` against the Python classes of polars.

A refusal of a rule names the class of the error that polars raises, and a test
compares that name with the class of the exception that polars raises. So each
name must be a class that polars raises: an exception of `polars.exceptions`.
A panic of polars reaches Python as a
`PanicException`, which a catch of `Exception` does not see.
"""

from __future__ import annotations

import polars as pl
import pytest

from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.dispatch import KernelFacts


class Exceptions:
    """Find the Python class of an error that polars raises, by its name.

    Example:
        >>> Exceptions().of(ErrorClass.SHAPE).__name__
        'ShapeError'
    """

    def of(self, error: ErrorClass) -> type[BaseException]:
        """Give the class.

        Args:
            error: The name of the class.

        Returns:
            The class of `polars.exceptions` with the name.

        Raises:
            TypeError: No class of polars has the name.

        Example:
            >>> Exceptions().of(ErrorClass.COMPUTE).__module__
            'polars.exceptions'
        """
        found = getattr(pl.exceptions, error.value, None)
        if not isinstance(found, type) or not issubclass(found, BaseException):
            raise TypeError(error.value)
        return found


class TestFailures:
    """Check the names of the ways that polars stops and of the classes of its errors."""

    @pytest.mark.parametrize("error", list(ErrorClass), ids=lambda error: error.value)
    def test_error_class_names_an_exception_that_polars_raises(self, error: ErrorClass) -> None:
        """Each class name of a refusal is an exception class that Python catches."""
        assert issubclass(Exceptions().of(error), Exception)

    def test_a_panic_of_polars_is_not_an_exception(self) -> None:
        """A panic reaches Python as a `PanicException`, which derives from `BaseException` only."""
        probe = KernelFacts.SHARED_NULL_LEAF.probe
        with pytest.raises(BaseException) as caught:  # ruff: ignore[pytest-raises-too-broad]
            eval(probe, {"pl": pl})  # ruff: ignore[suspicious-eval-usage]
        assert type(caught.value).__name__ == "PanicException"
        assert not isinstance(caught.value, Exception)

    def test_a_refusal_that_raises_names_its_error_class(self) -> None:
        """A refusal that raises takes its class, and the other ways take no class."""
        raised = PolarsRefusal(ErrorClass.SHAPE, "lengths")
        hung = PolarsRefusal(Stop.HANGS, "a cycle")
        assert (raised.stop, raised.error) == (Stop.RAISES, ErrorClass.SHAPE)
        assert (hung.stop, hung.error) == (Stop.HANGS, None)
        with pytest.raises(ValueError, match="needs the class"):
            PolarsRefusal(Stop.RAISES, "lengths")
