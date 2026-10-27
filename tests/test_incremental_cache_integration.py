"""Check that the types the plugin mints read back the same from mypy's cache.

mypy keeps the types of each checked module in an incremental cache.
A second run over a file that did not change reads the types from the
cache, and the plugin does not mint them again. A type that does not
survive that trip gives a different answer on the second run, with
no error at all.

The first test checks a program with the hardest forms together: a
table with explicit rows, a parameter carried from the input, a read
of list elements, an open `when`/`then` chain, a time zone set and
then taken away, a struct, an array, a read of a struct field, and a
declaration whose output name a call supplies. The second run must
show the same types as the first. The second test checks that the
classes of a schema module still read as schemas when that module
comes from the cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mypy.api

if TYPE_CHECKING:
    from pathlib import Path

_SCENARIO = """\
from typing import Literal
import polars as pl
from astrolabe import Columns, Datetime, List, TypedDataFrame, same, some_column, typed_expr
from polars.datatypes.classes import FloatType

class FrameSchema(Columns):
    k: pl.String
    t: Datetime[Literal["ns"], Literal["UTC"]]
    l: List[pl.Int64]
    b: pl.Boolean
    f: pl.Float32

Frame = TypedDataFrame[FrameSchema]

stepped = pl.col("t").diff()
total = pl.col("l").list.sum() + 1
chain = pl.when(pl.col("b")).then(pl.col("l").list.len())
stamped = pl.col("k").str.to_datetime("%Y-%m-%d", time_unit="ms", time_zone="UTC")
zoned = pl.col("t").dt.cast_time_unit("ms").dt.replace_time_zone(None).alias("m")
built = pl.struct(pl.col("k"), pl.col("f").alias("g")).alias("st")
stacked = pl.concat_arr("f", "f").alias("ar")
field = built.struct.field("g").alias("fg")
Standardised = typed_expr(some_column(FloatType), gives=same)

def zscore(e: Standardised) -> Standardised:
    return (e - e.mean()) / e.std()

def run(df: Frame) -> None:
    reveal_type(stepped)
    reveal_type(total)
    reveal_type(chain)
    reveal_type(zscore(pl.col("f")))
    reveal_type(zoned)
    reveal_type(field)
    reveal_type(built)
    reveal_type(stacked)
    z = zscore(pl.col("f"))
    reveal_type(df.select(
        stepped, total, chain.otherwise(0).alias("n"), stamped, z, zoned, built, stacked, field
    ))
"""


def _run_mypy(tmp_path: Path) -> tuple[str, int]:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.mypy]\nplugins = ["astrolabe.plugin"]\nallow_redefinition = true\n')
    main_file = tmp_path / "main.py"
    main_file.write_text(_SCENARIO)
    stdout, _stderr, exit_code = mypy.api.run([
        "--config-file",
        str(pyproject),
        "--cache-dir",
        str(tmp_path / ".mypy_cache"),
        str(main_file),
    ])
    return stdout, exit_code


def test_a_warm_cache_reads_the_same_spellings_back(tmp_path: Path) -> None:
    cold, cold_exit = _run_mypy(tmp_path)
    warm, warm_exit = _run_mypy(tmp_path)
    assert cold_exit == 0, cold
    assert warm_exit == 0, warm
    assert "Revealed type" in cold
    assert "Deferred[Literal[\"pl.col('t').diff()\"]]" in cold
    assert "TypedThen[astrolabe.expr.Deferred[" in cold
    assert "'fg': polars.datatypes.classes.Float32" in cold
    assert "'ar': astrolabe.frame.Array[polars.datatypes.classes.Float32, Literal[2]]" in cold
    assert "'m': astrolabe.frame.Datetime[Literal['ms'], None]" in cold
    assert "Literal['UTC']" in cold
    assert "Duration[Literal['ns']]" in cold
    assert "TypedColumn[Literal['f'], polars.datatypes.classes.FloatType]" in cold
    assert "'f': polars.datatypes.classes.Float32" in cold
    assert warm == cold


_SCHEMAS = """\
import polars as pl
from astrolabe import Columns, columns

class Keyed[T: pl.DataType](Columns):
    k: pl.String
    v: T

Pivoted = columns("Pivoted", {"1": pl.Int64, "v1|a": pl.Datetime("us")})
"""

_USES = """\
import polars as pl
from astrolabe import TypedDataFrame
from schemas import Keyed, Pivoted

def run(pivoted: TypedDataFrame[Pivoted], keyed: TypedDataFrame[Keyed[pl.Int64]]) -> None:
    reveal_type(pivoted.select("v1|a"))
    reveal_type(keyed.select("v"))
"""


def _run_two_modules(tmp_path: Path, uses: str) -> tuple[str, int]:
    (tmp_path / "pyproject.toml").write_text('[tool.mypy]\nplugins = ["astrolabe.plugin"]\n')
    (tmp_path / "schemas.py").write_text(_SCHEMAS)
    (tmp_path / "main.py").write_text(uses)
    stdout, _stderr, exit_code = mypy.api.run([
        "--config-file",
        str(tmp_path / "pyproject.toml"),
        "--cache-dir",
        str(tmp_path / ".mypy_cache"),
        str(tmp_path / "main.py"),
        str(tmp_path / "schemas.py"),
    ])
    return stdout, exit_code


def test_schema_classes_from_a_cached_module_stay_schema_classes(tmp_path: Path) -> None:
    """A `columns(...)` class and a generic class from the cache still read as schemas.

    The second run changes only `main.py`, so `schemas.py` comes from the
    cache. The frame annotations in `main.py` then read its classes from
    the cache.
    """
    cold, cold_exit = _run_two_modules(tmp_path, _USES)
    warm, warm_exit = _run_two_modules(tmp_path, _USES + "\nx = 1\n")
    assert cold_exit == 0, cold
    assert warm_exit == 0, warm
    assert "'v1|a': astrolabe.frame.Datetime[Literal['us'], None]" in warm
    assert "'v': polars.datatypes.classes.Int64" in warm
    assert warm == cold


_INFERRED_LIB = """\
import polars as pl
from astrolabe import InferredExpr, infer

def price_expr() -> InferredExpr:
    return infer(pl.col("price").cast(pl.Float64))
"""

_INFERRED_USES = """\
from lib import price_expr

reveal_type(price_expr())
"""

_INFERRED_METHOD_LIB = """\
import polars as pl
from astrolabe import InferredExpr, infer

class Prices:
    def price_expr(self) -> InferredExpr:
        return infer(pl.col("price").cast(pl.Float64))
"""

_INFERRED_METHOD_USES = """\
from lib import Prices

reveal_type(Prices().price_expr())
"""


def _run_inferred(
    tmp_path: Path,
    uses: str,
    lib: str = _INFERRED_LIB,
    options: tuple[str, ...] = (),
) -> tuple[str, int]:
    (tmp_path / "pyproject.toml").write_text('[tool.mypy]\nplugins = ["astrolabe.plugin"]\n')
    (tmp_path / "lib.py").write_text(lib)
    (tmp_path / "main.py").write_text(uses)
    stdout, _stderr, exit_code = mypy.api.run([
        "--config-file",
        str(tmp_path / "pyproject.toml"),
        "--cache-dir",
        str(tmp_path / ".mypy_cache"),
        *options,
        str(tmp_path / "main.py"),
        str(tmp_path / "lib.py"),
    ])
    return stdout, exit_code


def test_an_inferred_return_type_survives_a_warm_cache(tmp_path: Path) -> None:
    """A function that returns `InferredExpr` keeps its type when its module comes from the cache.

    The second run changes only `main.py`, so mypy loads `lib.py` from the
    cache and does not check the body of `price_expr` again.
    """
    cold, cold_exit = _run_inferred(tmp_path, _INFERRED_USES)
    warm, warm_exit = _run_inferred(tmp_path, _INFERRED_USES + "\nx = 1\n")
    assert cold_exit == 0, cold
    assert warm_exit == 0, warm
    assert "Deferred[Literal[\"pl.col('price').cast(pl.Float64)\"]]" in warm
    assert "astrolabe.declare.InferredExpr" not in warm
    assert warm == cold


def test_an_inferred_return_type_survives_a_warm_cache_in_parallel_mode(tmp_path: Path) -> None:
    """A function that returns `InferredExpr` keeps its type from the cache of a parallel run.

    With `--num-workers`, mypy writes the cache of a module after its first
    phase of checking. The body of `price_expr` gives its return type, so
    the plugin marks it as part of the interface of `lib.py`, and mypy
    checks it in that first phase, before it writes the cache.
    """
    workers = ("--num-workers", "2")
    cold, cold_exit = _run_inferred(tmp_path, _INFERRED_USES, options=workers)
    warm, warm_exit = _run_inferred(tmp_path, _INFERRED_USES + "\nx = 1\n", options=workers)
    assert cold_exit == 0, cold
    assert warm_exit == 0, warm
    assert "Deferred[Literal[\"pl.col('price').cast(pl.Float64)\"]]" in warm
    assert "astrolabe.declare.InferredExpr" not in warm


def test_an_inferred_return_type_of_a_method_survives_a_warm_cache(tmp_path: Path) -> None:
    """A method that returns `InferredExpr` keeps its type when its class comes from the cache.

    The type sits in the symbol table of the class, which mypy caches
    with the module.
    """
    cold, cold_exit = _run_inferred(tmp_path, _INFERRED_METHOD_USES, _INFERRED_METHOD_LIB)
    warm, warm_exit = _run_inferred(
        tmp_path,
        _INFERRED_METHOD_USES + "\nx = 1\n",
        _INFERRED_METHOD_LIB,
    )
    assert cold_exit == 0, cold
    assert warm_exit == 0, warm
    assert "Deferred[Literal[\"pl.col('price').cast(pl.Float64)\"]]" in warm
    assert "astrolabe.declare.InferredExpr" not in warm
    assert warm == cold
