"""The calls of polars of the census, one for each method, member and function.

Each call fixes the arguments of one operation of the expression API, for
example `pl.col('x').clip(0)`. The census surface builds the tree of each
call, finds the node at its root, and builds that node again with any input
at each position.
The calls run over a frame that holds a column `x` of the kind of the
namespace, and the helper columns `t` (a sorted `Datetime`), `g` (`Int64`) and
`b` (`Boolean`).

The lists of names that the census leaves out say why: a function that works
only inside the scope of another expression, a function that runs code of the
user, or a function that builds a data type and not an expression.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.releases import PolarsRelease


class CensusCalls:
    """Hold the call of each operation of the expression API in the census.

    Keys are bare method names for the methods of `pl.Expr`,
    `namespace.member` for the members of a namespace, and bare function
    names for the functions of `pl`.

    Example:
        >>> CensusCalls.ROOT_CALLS["abs"]
        "pl.col('x').abs()"
        >>> CensusCalls.MEMBERS["str.len_chars"]
        "pl.col('x').str.len_chars()"
        >>> "root.map_elements" in CensusCalls().method_keys()
        True
    """

    # The methods of `pl.Expr`, each with one call.
    ROOT_CALLS: ClassVar[dict[str, str]] = {
        "abs": "pl.col('x').abs()",
        "add": "pl.col('x').add(pl.col('x'))",
        "agg_groups": "pl.col('x').agg_groups()",
        "alias": "pl.col('x').alias('y')",
        "all": "pl.col('x').all()",
        "and_": "pl.col('x').and_()",
        "any": "pl.col('x').any()",
        "append": "pl.col('x').append(pl.col('x'))",
        "approx_n_unique": "pl.col('x').approx_n_unique()",
        "arccos": "pl.col('x').arccos()",
        "arccosh": "pl.col('x').arccosh()",
        "arcsin": "pl.col('x').arcsin()",
        "arcsinh": "pl.col('x').arcsinh()",
        "arctan": "pl.col('x').arctan()",
        "arctanh": "pl.col('x').arctanh()",
        "arg_max": "pl.col('x').arg_max()",
        "arg_min": "pl.col('x').arg_min()",
        "arg_sort": "pl.col('x').arg_sort()",
        "arg_true": "pl.col('x').arg_true()",
        "arg_unique": "pl.col('x').arg_unique()",
        "backward_fill": "pl.col('x').backward_fill()",
        "bitwise_and": "pl.col('x').bitwise_and()",
        "bitwise_count_ones": "pl.col('x').bitwise_count_ones()",
        "bitwise_count_zeros": "pl.col('x').bitwise_count_zeros()",
        "bitwise_leading_ones": "pl.col('x').bitwise_leading_ones()",
        "bitwise_leading_zeros": "pl.col('x').bitwise_leading_zeros()",
        "bitwise_or": "pl.col('x').bitwise_or()",
        "bitwise_trailing_ones": "pl.col('x').bitwise_trailing_ones()",
        "bitwise_trailing_zeros": "pl.col('x').bitwise_trailing_zeros()",
        "bitwise_xor": "pl.col('x').bitwise_xor()",
        "bottom_k": "pl.col('x').bottom_k()",
        "bottom_k_by": "pl.col('x').bottom_k_by(pl.col('x'), 1)",
        "cast": "pl.col('x').cast(pl.Int32)",
        "cbrt": "pl.col('x').cbrt()",
        "ceil": "pl.col('x').ceil()",
        "clip": "pl.col('x').clip(0)",
        "cos": "pl.col('x').cos()",
        "cosh": "pl.col('x').cosh()",
        "cot": "pl.col('x').cot()",
        "count": "pl.col('x').count()",
        "cum_count": "pl.col('x').cum_count()",
        "cum_max": "pl.col('x').cum_max()",
        "cum_min": "pl.col('x').cum_min()",
        "cum_prod": "pl.col('x').cum_prod()",
        "cum_sum": "pl.col('x').cum_sum()",
        "cumulative_eval": "pl.col('x').cumulative_eval(pl.element().sum())",
        "cut": "pl.col('x').cut([0, 2])",
        "degrees": "pl.col('x').degrees()",
        "diff": "pl.col('x').diff()",
        "dot": "pl.col('x').dot(pl.col('x'))",
        "drop_nans": "pl.col('x').drop_nans()",
        "drop_nulls": "pl.col('x').drop_nulls()",
        "entropy": "pl.col('x').entropy()",
        "eq": "pl.col('x').eq(pl.col('x'))",
        "eq_missing": "pl.col('x').eq_missing(pl.col('x'))",
        "ewm_mean": "pl.col('x').ewm_mean(alpha=0.5)",
        "ewm_mean_by": "pl.col('x').ewm_mean_by(by='t', half_life='1d')",
        "ewm_std": "pl.col('x').ewm_std(alpha=0.5)",
        "ewm_sum": "pl.col('x').ewm_sum(alpha=0.5)",
        "ewm_sum_by": "pl.col('x').ewm_sum_by(by='t', half_life='1d')",
        "ewm_var": "pl.col('x').ewm_var(alpha=0.5)",
        "exclude": "pl.col('x').exclude('y')",
        "exp": "pl.col('x').exp()",
        "explode": "pl.col('x').explode()",
        "extend_constant": "pl.col('x').extend_constant(1, 1)",
        "fill_nan": "pl.col('x').fill_nan(0)",
        "fill_null": "pl.col('x').fill_null(0)",
        "filter": "pl.col('x').filter(pl.col('b'))",
        "first": "pl.col('x').first()",
        "flatten": "pl.col('x').flatten()",
        "floor": "pl.col('x').floor()",
        "floordiv": "pl.col('x').floordiv(pl.col('x'))",
        "forward_fill": "pl.col('x').forward_fill()",
        "gather": "pl.col('x').gather([0])",
        "gather_every": "pl.col('x').gather_every(2)",
        "ge": "pl.col('x').ge(pl.col('x'))",
        "get": "pl.col('x').get(0)",
        "gt": "pl.col('x').gt(pl.col('x'))",
        "has_nulls": "pl.col('x').has_nulls()",
        "hash": "pl.col('x').hash()",
        "head": "pl.col('x').head()",
        "hist": "pl.col('x').hist()",
        "implode": "pl.col('x').implode()",
        "index_of": "pl.col('x').index_of(1)",
        "inspect": "pl.col('x').inspect()",
        "interpolate": "pl.col('x').interpolate()",
        "interpolate_by": "pl.col('x').interpolate_by(pl.col('x'))",
        "is_between": "pl.col('x').is_between(0, 2)",
        "is_close": "pl.col('x').is_close(pl.col('x'))",
        "is_duplicated": "pl.col('x').is_duplicated()",
        "is_empty": "pl.col('x').is_empty()",
        "is_finite": "pl.col('x').is_finite()",
        "is_first_distinct": "pl.col('x').is_first_distinct()",
        "is_in": "pl.col('x').is_in([1])",
        "is_infinite": "pl.col('x').is_infinite()",
        "is_last_distinct": "pl.col('x').is_last_distinct()",
        "is_nan": "pl.col('x').is_nan()",
        "is_not_nan": "pl.col('x').is_not_nan()",
        "is_not_null": "pl.col('x').is_not_null()",
        "is_null": "pl.col('x').is_null()",
        "is_sorted": "pl.col('x').is_sorted()",
        "is_unique": "pl.col('x').is_unique()",
        "item": "pl.col('x').item()",
        "kurtosis": "pl.col('x').kurtosis()",
        "last": "pl.col('x').last()",
        "le": "pl.col('x').le(pl.col('x'))",
        "len": "pl.col('x').len()",
        "limit": "pl.col('x').limit()",
        "log": "pl.col('x').log()",
        "log10": "pl.col('x').log10()",
        "log1p": "pl.col('x').log1p()",
        "lower_bound": "pl.col('x').lower_bound()",
        "lt": "pl.col('x').lt(pl.col('x'))",
        "max": "pl.col('x').max()",
        "max_by": "pl.col('x').max_by(pl.col('x'))",
        "mean": "pl.col('x').mean()",
        "median": "pl.col('x').median()",
        "min": "pl.col('x').min()",
        "min_by": "pl.col('x').min_by(pl.col('x'))",
        "mod": "pl.col('x').mod(2)",
        "mode": "pl.col('x').mode()",
        "mul": "pl.col('x').mul(pl.col('x'))",
        "n_unique": "pl.col('x').n_unique()",
        "nan_max": "pl.col('x').nan_max()",
        "nan_min": "pl.col('x').nan_min()",
        "ne": "pl.col('x').ne(pl.col('x'))",
        "ne_missing": "pl.col('x').ne_missing(pl.col('x'))",
        "neg": "pl.col('x').neg()",
        "not_": "pl.col('x').not_()",
        "null_count": "pl.col('x').null_count()",
        "or_": "pl.col('x').or_()",
        "over": "pl.col('x').over('g')",
        "pct_change": "pl.col('x').pct_change()",
        "peak_max": "pl.col('x').peak_max()",
        "peak_min": "pl.col('x').peak_min()",
        "pow": "pl.col('x').pow(2)",
        "product": "pl.col('x').product()",
        "qcut": "pl.col('x').qcut([0.5])",
        "quantile": "pl.col('x').quantile(0.5)",
        "radians": "pl.col('x').radians()",
        "rank": "pl.col('x').rank()",
        "rechunk": "pl.col('x').rechunk()",
        "reinterpret": "pl.col('x').reinterpret(signed=True)",
        "repeat_by": "pl.col('x').repeat_by(1)",
        "replace": "pl.col('x').replace({1: 2})",
        "replace_strict": "pl.col('x').replace_strict({1: 2}, default=0)",
        "reshape": "pl.col('x').reshape((2, 1))",
        "reverse": "pl.col('x').reverse()",
        "rle": "pl.col('x').rle()",
        "rle_id": "pl.col('x').rle_id()",
        "rolling": "pl.col('x').rolling(index_column='t', period='2d')",
        "rolling_kurtosis": "pl.col('x').rolling_kurtosis(window_size=2)",
        "rolling_max": "pl.col('x').rolling_max(window_size=2)",
        "rolling_max_by": "pl.col('x').rolling_max_by(by='t', window_size='1d')",
        "rolling_mean": "pl.col('x').rolling_mean(window_size=2)",
        "rolling_mean_by": "pl.col('x').rolling_mean_by(by='t', window_size='1d')",
        "rolling_median": "pl.col('x').rolling_median(window_size=2)",
        "rolling_median_by": "pl.col('x').rolling_median_by(by='t', window_size='1d')",
        "rolling_min": "pl.col('x').rolling_min(window_size=2)",
        "rolling_min_by": "pl.col('x').rolling_min_by(by='t', window_size='1d')",
        "rolling_quantile": "pl.col('x').rolling_quantile(quantile=0.5, window_size=2)",
        "rolling_quantile_by": (
            "pl.col('x').rolling_quantile_by(by='t', window_size='1d', quantile=0.5)"
        ),
        "rolling_rank": "pl.col('x').rolling_rank(window_size=2)",
        "rolling_rank_by": "pl.col('x').rolling_rank_by(by='t', window_size='1d')",
        "rolling_skew": "pl.col('x').rolling_skew(window_size=2)",
        "rolling_std": "pl.col('x').rolling_std(window_size=2)",
        "rolling_std_by": "pl.col('x').rolling_std_by(by='t', window_size='1d')",
        "rolling_sum": "pl.col('x').rolling_sum(window_size=2)",
        "rolling_sum_by": "pl.col('x').rolling_sum_by(by='t', window_size='1d')",
        "rolling_var": "pl.col('x').rolling_var(window_size=2)",
        "rolling_var_by": "pl.col('x').rolling_var_by(by='t', window_size='1d')",
        "round": "pl.col('x').round()",
        "round_sig_figs": "pl.col('x').round_sig_figs(2)",
        "sample": "pl.col('x').sample()",
        "search_sorted": "pl.col('x').search_sorted(1)",
        "set_sorted": "pl.col('x').set_sorted()",
        "shift": "pl.col('x').shift()",
        "shrink_dtype": "pl.col('x').shrink_dtype()",
        "shuffle": "pl.col('x').shuffle()",
        "sign": "pl.col('x').sign()",
        "sin": "pl.col('x').sin()",
        "sinh": "pl.col('x').sinh()",
        "skew": "pl.col('x').skew()",
        "slice": "pl.col('x').slice(0, 1)",
        "sort": "pl.col('x').sort()",
        "sort_by": "pl.col('x').sort_by(pl.col('x'))",
        "sqrt": "pl.col('x').sqrt()",
        "std": "pl.col('x').std()",
        "sub": "pl.col('x').sub(pl.col('x'))",
        "sum": "pl.col('x').sum()",
        "tail": "pl.col('x').tail()",
        "tan": "pl.col('x').tan()",
        "tanh": "pl.col('x').tanh()",
        "to_physical": "pl.col('x').to_physical()",
        "top_k": "pl.col('x').top_k()",
        "top_k_by": "pl.col('x').top_k_by(pl.col('x'), 1)",
        "truediv": "pl.col('x').truediv(pl.col('x'))",
        "truncate": "pl.col('x').truncate()",
        "unique": "pl.col('x').unique()",
        "unique_counts": "pl.col('x').unique_counts()",
        "upper_bound": "pl.col('x').upper_bound()",
        "value_counts": "pl.col('x').value_counts()",
        "var": "pl.col('x').var()",
        "where": "pl.col('x').where(pl.col('b'))",
        "xor": "pl.col('x').xor(pl.col('x'))",
    }

    # The members of the namespaces, each with one call.
    MEMBERS: ClassVar[dict[str, str]] = {
        "arr.agg": "pl.col('x').arr.agg(pl.element().sum())",
        "arr.all": "pl.col('x').arr.all()",
        "arr.any": "pl.col('x').arr.any()",
        "arr.arg_max": "pl.col('x').arr.arg_max()",
        "arr.arg_min": "pl.col('x').arr.arg_min()",
        "arr.contains": "pl.col('x').arr.contains(1)",
        "arr.count_matches": "pl.col('x').arr.count_matches(1)",
        "arr.dot": "pl.col('x').arr.dot(pl.col('x'))",
        "arr.eval": "pl.col('x').arr.eval(pl.element() * 2)",
        "arr.explode": "pl.col('x').arr.explode()",
        "arr.first": "pl.col('x').arr.first()",
        "arr.get": "pl.col('x').arr.get(0)",
        "arr.head": "pl.col('x').arr.head()",
        "arr.join": "pl.col('x').arr.join(',')",
        "arr.last": "pl.col('x').arr.last()",
        "arr.len": "pl.col('x').arr.len()",
        "arr.max": "pl.col('x').arr.max()",
        "arr.mean": "pl.col('x').arr.mean()",
        "arr.median": "pl.col('x').arr.median()",
        "arr.min": "pl.col('x').arr.min()",
        "arr.n_unique": "pl.col('x').arr.n_unique()",
        "arr.reverse": "pl.col('x').arr.reverse()",
        "arr.shift": "pl.col('x').arr.shift()",
        "arr.slice": "pl.col('x').arr.slice(0, 1)",
        "arr.sort": "pl.col('x').arr.sort()",
        "arr.std": "pl.col('x').arr.std()",
        "arr.sum": "pl.col('x').arr.sum()",
        "arr.tail": "pl.col('x').arr.tail()",
        "arr.to_list": "pl.col('x').arr.to_list()",
        "arr.to_struct": "pl.col('x').arr.to_struct()",
        "arr.unique": "pl.col('x').arr.unique()",
        "arr.var": "pl.col('x').arr.var()",
        "bin.contains": "pl.col('x').bin.contains(b'a')",
        "bin.decode": "pl.col('x').bin.decode('hex')",
        "bin.encode": "pl.col('x').bin.encode('hex')",
        "bin.ends_with": "pl.col('x').bin.ends_with(b'a')",
        "bin.get": "pl.col('x').bin.get(0)",
        "bin.head": "pl.col('x').bin.head()",
        "bin.reinterpret": "pl.col('x').bin.reinterpret(dtype=pl.UInt8)",
        "bin.size": "pl.col('x').bin.size()",
        "bin.slice": "pl.col('x').bin.slice(0, 1)",
        "bin.starts_with": "pl.col('x').bin.starts_with(b'a')",
        "bin.tail": "pl.col('x').bin.tail()",
        "cat.ends_with": "pl.col('x').cat.ends_with('a')",
        "cat.get_categories": "pl.col('x').cat.get_categories()",
        "cat.len_bytes": "pl.col('x').cat.len_bytes()",
        "cat.len_chars": "pl.col('x').cat.len_chars()",
        "cat.physical": "pl.col('x').cat.physical()",
        "cat.slice": "pl.col('x').cat.slice(0, 1)",
        "cat.starts_with": "pl.col('x').cat.starts_with('a')",
        "cat.to": "pl.col('x').cat.to(pl.String)",
        "dt.add_business_days": "pl.col('x').dt.add_business_days(1)",
        "dt.base_utc_offset": "pl.col('x').dt.base_utc_offset()",
        "dt.cast_time_unit": "pl.col('x').dt.cast_time_unit('ms')",
        "dt.century": "pl.col('x').dt.century()",
        "dt.combine": "pl.col('x').dt.combine(time(1, 2))",
        "dt.convert_time_zone": "pl.col('x').dt.convert_time_zone('UTC')",
        "dt.date": "pl.col('x').dt.date()",
        "dt.datetime": "pl.col('x').dt.datetime()",
        "dt.day": "pl.col('x').dt.day()",
        "dt.days_in_month": "pl.col('x').dt.days_in_month()",
        "dt.dst_offset": "pl.col('x').dt.dst_offset()",
        "dt.epoch": "pl.col('x').dt.epoch()",
        "dt.hour": "pl.col('x').dt.hour()",
        "dt.is_business_day": "pl.col('x').dt.is_business_day()",
        "dt.is_leap_year": "pl.col('x').dt.is_leap_year()",
        "dt.iso_year": "pl.col('x').dt.iso_year()",
        "dt.microsecond": "pl.col('x').dt.microsecond()",
        "dt.millennium": "pl.col('x').dt.millennium()",
        "dt.millisecond": "pl.col('x').dt.millisecond()",
        "dt.minute": "pl.col('x').dt.minute()",
        "dt.month": "pl.col('x').dt.month()",
        "dt.month_end": "pl.col('x').dt.month_end()",
        "dt.month_start": "pl.col('x').dt.month_start()",
        "dt.nanosecond": "pl.col('x').dt.nanosecond()",
        "dt.offset_by": "pl.col('x').dt.offset_by('1d')",
        "dt.ordinal_day": "pl.col('x').dt.ordinal_day()",
        "dt.quarter": "pl.col('x').dt.quarter()",
        "dt.replace": "pl.col('x').dt.replace()",
        "dt.replace_time_zone": "pl.col('x').dt.replace_time_zone('UTC')",
        "dt.round": "pl.col('x').dt.round('1d')",
        "dt.second": "pl.col('x').dt.second()",
        "dt.strftime": "pl.col('x').dt.strftime('%Y')",
        "dt.time": "pl.col('x').dt.time()",
        "dt.timestamp": "pl.col('x').dt.timestamp()",
        "dt.to_string": "pl.col('x').dt.to_string()",
        "dt.total_days": "pl.col('x').dt.total_days()",
        "dt.total_hours": "pl.col('x').dt.total_hours()",
        "dt.total_microseconds": "pl.col('x').dt.total_microseconds()",
        "dt.total_milliseconds": "pl.col('x').dt.total_milliseconds()",
        "dt.total_minutes": "pl.col('x').dt.total_minutes()",
        "dt.total_nanoseconds": "pl.col('x').dt.total_nanoseconds()",
        "dt.total_seconds": "pl.col('x').dt.total_seconds()",
        "dt.truncate": "pl.col('x').dt.truncate('1d')",
        "dt.week": "pl.col('x').dt.week()",
        "dt.weekday": "pl.col('x').dt.weekday()",
        "dt.with_time_unit": "pl.col('x').dt.with_time_unit('ms')",
        "dt.year": "pl.col('x').dt.year()",
        "list.agg": "pl.col('x').list.agg(pl.element().sum())",
        "list.all": "pl.col('x').list.all()",
        "list.any": "pl.col('x').list.any()",
        "list.arg_max": "pl.col('x').list.arg_max()",
        "list.arg_min": "pl.col('x').list.arg_min()",
        "list.concat": "pl.col('x').list.concat(pl.col('x'))",
        "list.contains": "pl.col('x').list.contains(1)",
        "list.count_matches": "pl.col('x').list.count_matches(1)",
        "list.diff": "pl.col('x').list.diff()",
        "list.drop_nulls": "pl.col('x').list.drop_nulls()",
        "list.eval": "pl.col('x').list.eval(pl.element().sum())",
        "list.explode": "pl.col('x').list.explode()",
        "list.filter": "pl.col('x').list.filter(pl.element() > 0)",
        "list.first": "pl.col('x').list.first()",
        "list.gather": "pl.col('x').list.gather([0])",
        "list.gather_every": "pl.col('x').list.gather_every(1)",
        "list.get": "pl.col('x').list.get(0)",
        "list.head": "pl.col('x').list.head()",
        "list.item": "pl.col('x').list.item()",
        "list.join": "pl.col('x').cast(pl.List(pl.String)).list.join(',')",
        "list.last": "pl.col('x').list.last()",
        "list.len": "pl.col('x').list.len()",
        "list.max": "pl.col('x').list.max()",
        "list.mean": "pl.col('x').list.mean()",
        "list.median": "pl.col('x').list.median()",
        "list.min": "pl.col('x').list.min()",
        "list.n_unique": "pl.col('x').list.n_unique()",
        "list.reverse": "pl.col('x').list.reverse()",
        "list.sample": "pl.col('x').list.sample()",
        "list.set_difference": "pl.col('x').list.set_difference(pl.col('x'))",
        "list.set_intersection": "pl.col('x').list.set_intersection(pl.col('x'))",
        "list.set_symmetric_difference": "pl.col('x').list.set_symmetric_difference(pl.col('x'))",
        "list.set_union": "pl.col('x').list.set_union(pl.col('x'))",
        "list.shift": "pl.col('x').list.shift()",
        "list.slice": "pl.col('x').list.slice(0, 1)",
        "list.sort": "pl.col('x').list.sort()",
        "list.std": "pl.col('x').list.std()",
        "list.sum": "pl.col('x').list.sum()",
        "list.tail": "pl.col('x').list.tail()",
        "list.to_array": "pl.col('x').list.to_array(1)",
        "list.to_struct": "pl.col('x').list.to_struct(upper_bound=1)",
        "list.unique": "pl.col('x').list.unique()",
        "list.var": "pl.col('x').list.var()",
        "str.concat": "pl.col('x').str.concat()",
        "str.contains": "pl.col('x').str.contains('a')",
        "str.contains_any": "pl.col('x').str.contains_any(['a'])",
        "str.count_matches": "pl.col('x').str.count_matches('a')",
        "str.decode": "pl.col('x').str.decode('hex')",
        "str.encode": "pl.col('x').str.encode('hex')",
        "str.ends_with": "pl.col('x').str.ends_with('a')",
        "str.escape_regex": "pl.col('x').str.escape_regex()",
        "str.explode": "pl.col('x').str.explode()",
        "str.extract": "pl.col('x').str.extract(r'(a)')",
        "str.extract_all": "pl.col('x').str.extract_all('a')",
        "str.extract_groups": "pl.col('x').str.extract_groups(r'(?<g>a)')",
        "str.extract_many": "pl.col('x').str.extract_many(['a'])",
        "str.find": "pl.col('x').str.find('a')",
        "str.find_many": "pl.col('x').str.find_many(['a'])",
        "str.head": "pl.col('x').str.head(1)",
        "str.join": "pl.col('x').str.join()",
        "str.json_decode": "pl.col('x').str.json_decode(pl.Int64)",
        "str.json_path_match": "pl.col('x').str.json_path_match('a')",
        "str.len_bytes": "pl.col('x').str.len_bytes()",
        "str.len_chars": "pl.col('x').str.len_chars()",
        "str.normalize": "pl.col('x').str.normalize()",
        "str.pad_end": "pl.col('x').str.pad_end(3)",
        "str.pad_start": "pl.col('x').str.pad_start(3)",
        "str.replace": "pl.col('x').str.replace('a', 'b')",
        "str.replace_all": "pl.col('x').str.replace_all('a', 'b')",
        "str.replace_many": "pl.col('x').str.replace_many(['a'], ['b'])",
        "str.reverse": "pl.col('x').str.reverse()",
        "str.slice": "pl.col('x').str.slice(0, 1)",
        "str.split": "pl.col('x').str.split('a')",
        "str.split_exact": "pl.col('x').str.split_exact('a', 1)",
        "str.splitn": "pl.col('x').str.splitn('a', 2)",
        "str.starts_with": "pl.col('x').str.starts_with('a')",
        "str.strip_chars": "pl.col('x').str.strip_chars()",
        "str.strip_chars_end": "pl.col('x').str.strip_chars_end()",
        "str.strip_chars_start": "pl.col('x').str.strip_chars_start()",
        "str.strip_prefix": "pl.col('x').str.strip_prefix('a')",
        "str.strip_suffix": "pl.col('x').str.strip_suffix('a')",
        "str.strptime": "pl.col('x').str.strptime(pl.Datetime('us'), format='%Y')",
        "str.tail": "pl.col('x').str.tail(1)",
        "str.to_date": "pl.col('x').str.to_date()",
        "str.to_datetime": "pl.col('x').str.to_datetime()",
        "str.to_decimal": "pl.col('x').str.to_decimal(scale=2)",
        "str.to_integer": "pl.col('x').str.to_integer()",
        "str.to_lowercase": "pl.col('x').str.to_lowercase()",
        "str.to_time": "pl.col('x').str.to_time()",
        "str.to_titlecase": "pl.col('x').str.to_titlecase()",
        "str.to_uppercase": "pl.col('x').str.to_uppercase()",
        "str.zfill": "pl.col('x').str.zfill(3)",
        "struct.drop": "pl.col('x').struct.drop(['a'])",
        "struct.field": "pl.col('x').struct.field('a')",
        "struct.json_encode": "pl.col('x').struct.json_encode()",
        "struct.rename_fields": "pl.col('x').struct.rename_fields(['b'])",
        "struct.unnest": "pl.col('x').struct.unnest()",
        "struct.with_fields": "pl.col('x').struct.with_fields()",
    }

    # The functions of `pl` that build an expression, each with one call.
    CONSTRUCTORS: ClassVar[dict[str, str]] = {
        "all": "pl.all()",
        "all_horizontal": "pl.all_horizontal(pl.col('b'), pl.col('b'))",
        "any": "pl.any('b')",
        "any_horizontal": "pl.any_horizontal(pl.col('b'), pl.col('b'))",
        "approx_n_unique": "pl.approx_n_unique('x')",
        "arange": "pl.arange(0, 3)",
        "arctan2": "pl.arctan2(pl.col('y'), pl.col('y'))",
        "arctan2d": "pl.arctan2d(pl.col('y'), pl.col('y'))",
        "arg_sort_by": "pl.arg_sort_by('x')",
        "arg_where": "pl.arg_where(pl.col('b'))",
        "business_day_count": "pl.business_day_count(date(2020, 1, 1), date(2020, 2, 1))",
        "coalesce": "pl.coalesce('x', 'y')",
        "col": "pl.col('x')",
        "concat_arr": "pl.concat_arr(pl.col('x'))",
        "concat_list": "pl.concat_list(pl.col('x'))",
        "concat_str": "pl.concat_str(pl.col('x'), pl.col('y'))",
        "corr": "pl.corr('x', 'y')",
        "count": "pl.count('x')",
        "cov": "pl.cov('x', 'y')",
        "cum_count": "pl.cum_count('x')",
        "cum_fold": "pl.cum_fold(pl.lit(0), lambda a, b: a + b, ['x', 'y'])",
        "cum_reduce": "pl.cum_reduce(lambda a, b: a + b, ['x', 'y'])",
        "cum_sum": "pl.cum_sum('x')",
        "cum_sum_horizontal": "pl.cum_sum_horizontal('x', 'y')",
        "date": "pl.date(2020, 1, 1)",
        "date_range": "pl.date_range(date(2020, 1, 1), date(2020, 1, 3))",
        "date_ranges": "pl.date_ranges(date(2020, 1, 1), date(2020, 1, 3))",
        "datetime": "pl.datetime(2020, 1, 1)",
        "datetime_range": "pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 3))",
        "datetime_ranges": "pl.datetime_ranges(datetime(2020, 1, 1), datetime(2020, 1, 3))",
        "duration": "pl.duration(days=1)",
        "exclude": "pl.exclude('g', 'b', 'y')",
        "first": "pl.first('x')",
        "fold": "pl.fold(pl.lit(0), lambda a, b: a + b, ['x', 'y'])",
        "format": "pl.format('{}', pl.col('x'))",
        "from_epoch": "pl.from_epoch(pl.col('x'))",
        "head": "pl.head('x')",
        "implode": "pl.implode('x')",
        "int_range": "pl.int_range(3)",
        "int_ranges": "pl.int_ranges(3)",
        "last": "pl.last('x')",
        "len": "pl.len()",
        "linear_space": "pl.linear_space(0, 1, 3)",
        "linear_spaces": "pl.linear_spaces(0, 1, 3)",
        "lit": "pl.lit(1)",
        "max": "pl.max('x')",
        "max_horizontal": "pl.max_horizontal('x', 'y')",
        "mean": "pl.mean('x')",
        "mean_horizontal": "pl.mean_horizontal('x', 'y')",
        "median": "pl.median('x')",
        "min": "pl.min('x')",
        "min_horizontal": "pl.min_horizontal('x', 'y')",
        "n_unique": "pl.n_unique('x')",
        "nth": "pl.nth(0)",
        "ones": "pl.ones(2)",
        "quantile": "pl.quantile('x', 0.5)",
        "reduce": "pl.reduce(lambda a, b: a + b, ['x', 'y'])",
        "repeat": "pl.repeat(1, 2)",
        "rolling_corr": "pl.rolling_corr('x', 'y', window_size=2)",
        "rolling_cov": "pl.rolling_cov('x', 'y', window_size=2)",
        "row_index": "pl.row_index()",
        "sql_expr": "pl.sql_expr('x + 1')",
        "std": "pl.std('x')",
        "struct": "pl.struct('x', 'y')",
        "sum": "pl.sum('x')",
        "sum_horizontal": "pl.sum_horizontal('x', 'y')",
        "tail": "pl.tail('x')",
        "time": "pl.time(1, 2)",
        "time_range": "pl.time_range(time(1, 0), time(5, 0))",
        "time_ranges": "pl.time_ranges(time(1, 0), time(5, 0))",
        "var": "pl.var('x')",
        "when": "pl.when(pl.col('b')).then(pl.col('x'))",
        "zeros": "pl.zeros(2)",
    }

    # The methods of `pl.Expr` that run code of the user, so no call has a type.
    OPAQUE: ClassVar[tuple[str, ...]] = (
        "root.deserialize",
        "root.from_json",
        "root.map_batches",
        "root.map_elements",
        "root.pipe",
        "root.register_plugin",
        "root.rolling_map",
    )

    # The functions that work only inside the scope of another expression.
    CONTEXT_BOUND: ClassVar[tuple[str, ...]] = (
        "element",
        "field",
        "groups",
        "list",
    )

    # The functions of `pl` that run code of the user.
    OPAQUE_CONSTRUCTORS: ClassVar[tuple[str, ...]] = (
        "map_batches",
        "map_groups",
    )

    # The functions of `pl` that build a data type and not an expression.
    TYPE_LEVEL: ClassVar[tuple[str, ...]] = (
        "dtype_of",
        "self_dtype",
        "struct_with_fields",
    )

    # The functions that a recent release of polars adds, each with that release. A
    # key is a name as the other tables of the class hold it, or the name of a
    # function of `pl` that builds a frame.
    SINCE: ClassVar[dict[str, str]] = {
        "is_sorted": "1.42.0",
        "ewm_sum": "1.43.0",
        "ewm_sum_by": "1.43.0",
        "cat.physical": "1.43.0",
        "cat.to": "1.43.0",
        "list": "1.43.0",
        "scan_arrow_c_stream": "1.43.0",
        "arr.dot": "1.44.0",
        "struct.drop": "1.44.0",
    }

    def installed(self, name: str) -> bool:
        """Say whether the installed release of polars has a function of the census.

        Args:
            name: The name of the function, as the tables of the class hold it.

        Returns:
            Whether the function is in no entry of `SINCE`, or its release is
            the installed one or an older one.

        Example:
            >>> CensusCalls().installed("abs")
            True
        """
        first = self.SINCE.get(name)
        return first is None or PolarsRelease.installed() >= PolarsRelease.parse(first)

    def root_calls(self) -> dict[str, str]:
        """Give the call of each method of `pl.Expr` that the installed release has.

        Returns:
            The calls of `ROOT_CALLS` that the installed release has.

        Example:
            >>> CensusCalls().root_calls()["abs"]
            "pl.col('x').abs()"
        """
        return {name: call for name, call in self.ROOT_CALLS.items() if self.installed(name)}

    def members(self) -> dict[str, str]:
        """Give the call of each member of a namespace that the installed release has.

        Returns:
            The calls of `MEMBERS` that the installed release has.

        Example:
            >>> CensusCalls().members()["str.len_chars"]
            "pl.col('x').str.len_chars()"
        """
        return {name: call for name, call in self.MEMBERS.items() if self.installed(name)}

    def context_bound(self) -> tuple[str, ...]:
        """Give the functions of `CONTEXT_BOUND` that the installed release has.

        Returns:
            The names of the functions.

        Example:
            >>> "element" in CensusCalls().context_bound()
            True
        """
        return tuple(name for name in self.CONTEXT_BOUND if self.installed(name))

    def method_keys(self) -> set[str]:
        """Give the key of every method of `pl.Expr` that this class accounts for.

        Returns:
            Each method as `root.<name>`, the opaque methods included.

        Example:
            >>> "root.abs" in CensusCalls().method_keys()
            True
        """
        measured = {f"root.{name}" for name in self.root_calls()}
        return measured | set(self.OPAQUE)
