from typing import Callable, TypeVar

import pandas as pd

from ..group_filter import GroupFilter

T = TypeVar("T")

# Process-local cache, keyed by (id(df), cache_key, group_name). Populated and
# consumed within a single compare_groups() call; under multiprocessing each
# worker process has its own fresh import of this module and therefore its own
# empty cache, so there is no cross-run staleness to worry about.
_cache: dict[tuple[int, str, str], object] = {}

# Sentinel group_name used by dataset_computation for values that are properties
# of the whole dataframe rather than of any group. Not a valid GroupFilter
# group_name (those never contain spaces around a bare marker like this), so it
# cannot collide with a real group's cache entry.
_DATASET_SCOPE = "<dataset>"


def shared_computation(
    cache_key: str,
    df: pd.DataFrame,
    group_filter: GroupFilter | None,
    compute_fn: Callable[[], T],
) -> T:
    """
    Compute-or-fetch a value that multiple sibling ComparisonMetric instances
    need for the same (df, group) pair, so that an expensive computation (e.g.
    a model fit) is performed once and reused rather than once per metric.

    cache_key identifies the *kind* of computation (e.g. "calibration_fit"),
    so unrelated metric families never collide even when scoped to the same
    group. The (df, group) pair is identified via `group_filter.group_name`.

    If group_filter is None, compute_fn is invoked directly and the result is
    not cached. This is not merely a fallback: it is the correct behaviour.
    The main caller of metrics with only a raw group_mask (no GroupFilter) is
    significance testing, which evaluates metrics on permuted/shuffled masks
    that represent different synthetic data on every call - there is nothing
    valid to cache across such calls even if a stable key were available, so
    always recomputing is both necessary and correct here. Metrics using this
    cache can therefore support significance testing (test=True): calls made
    during hypothesis testing simply run uncached, while calls made through
    compare_groups()'s normal per-group evaluation (which always supplies a
    GroupFilter) still benefit from caching as usual.

    :param cache_key: identifies the kind of computation being cached (e.g.
        "calibration_fit"); distinct cache_keys never collide, even for the
        same (df, group).
    :param df: the dataframe the computation was/will be performed on;
        together with group_filter.group_name, identifies the (df, group) pair.
    :param group_filter: identifies which group of df the computation is
        scoped to. If None, caching is bypassed (see above).
    :param compute_fn: a zero-argument callable that performs the actual
        computation; invoked on every call when group_filter is None, or
        on a cache miss otherwise.
    :return: the cached or freshly computed value.
    """
    if group_filter is None:
        return compute_fn()

    key = (id(df), cache_key, group_filter.group_name)

    if key not in _cache:
        _cache[key] = compute_fn()

    return _cache[key]  # type: ignore[return-value]


def dataset_computation(
    cache_key: str,
    df: pd.DataFrame,
    compute_fn: Callable[[], T],
) -> T:
    """
    Compute-or-fetch a value that is a property of the *whole dataset*, not of
    any one group - e.g. a shared reference point that every group's metric is
    evaluated against.

    Companion to shared_computation: same cache, but keyed on the dataframe
    alone (group_name is fixed to a sentinel), because the value must be
    identical for every group by construction. Being group-independent, it is
    also mask-independent, so it stays valid during significance testing, where
    metrics are re-evaluated on permuted masks of the same dataframe - and
    indeed it *must* stay fixed there, since a reference point that shifted
    per permutation would invalidate the comparison the test is making.

    :param cache_key: identifies the kind of computation being cached.
    :param df: the dataframe the value is a property of.
    :param compute_fn: zero-argument callable; only invoked on a cache miss.
    :return: the cached or freshly computed value.
    """
    key = (id(df), cache_key, _DATASET_SCOPE)

    if key not in _cache:
        _cache[key] = compute_fn()

    return _cache[key]  # type: ignore[return-value]


def clear_shared_computation_cache() -> None:
    """Drop all cached values. Called at the start of each compare_groups() run."""
    _cache.clear()
