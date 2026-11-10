"""Studentized permutation test for a difference in a metric between a group and its complement.

``studentized_permut_pval`` implements Algorithm 1 of DiCiccio et al. (2020), with a two-sided
p-value, handling of permutations on which the statistic is not computable, and conservative early
stopping for clearly non-significant results.

References
----------
- C. DiCiccio, S. Vasudevan, K. Basu, K. Kenthapadi and D. Agarwal (2020). Evaluating fairness
  using permutation tests. In *Proceedings of the 26th ACM SIGKDD International Conference on
  Knowledge Discovery & Data Mining (KDD '20)*. https://doi.org/10.1145/3394486.3403199,
  preprint: https://arxiv.org/abs/2007.05124
- B. Phipson and G. K. Smyth (2010). Permutation P-values should never be zero: calculating exact
  P-values when permutations are randomly drawn. *Statistical Applications in Genetics and
  Molecular Biology* 9(1), Article 39. https://doi.org/10.2202/1544-6115.1585,
  preprint: https://arxiv.org/abs/1603.05766
- T. A. Knijnenburg, L. F. A. Wessels, M. J. T. Reinders and I. Shmulevich (2009). Fewer
  permutations, more accurate P-values. *Bioinformatics* 25(12), i161-i168.
  https://doi.org/10.1093/bioinformatics/btp211
- C. J. Clopper and E. S. Pearson (1934). The use of confidence or fiducial limits illustrated in
  the case of the binomial. *Biometrika* 26(4), 404-413. https://doi.org/10.1093/biomet/26.4.404
"""

import enum
from typing import Literal

import numpy as np
import pandas as pd
import scipy

from .._array_types import MaskLike
from ..config import settings
from ..group_filter import GroupFilter
from ..metrics.ComparisonMetric import ComparisonMetric, MetricWithAnalyticalVar
from .bootstrap import RandomState, _mask_to_positions, bootstrap_metric
from .utils import nan_mean

try:
    from numba import njit as _numba_njit
    _NUMBA_AVAILABLE = True
except Exception:
    _numba_njit = None
    _NUMBA_AVAILABLE = False


# Shuffles of more than this many indices use the numba kernel. Below it, both paths take on the
# order of a microsecond, which is negligible next to evaluating the metric on a permuted split.
_NUMBA_SHUFFLE_SIZE_THRESHOLD = 128

# Early stopping of studentized_permut_pval; see its Notes for the rationale.
_EARLY_STOP_BATCH = 100  # permutations between checks
_EARLY_STOP_MIN_PERMUT = 100  # valid permutations required before the first check
_EARLY_STOP_MIN_M = 20  # exceedances required before the first check
_EARLY_STOP_CONFIDENCE = 0.9999  # confidence of the Clopper-Pearson lower bound


class _Default(enum.Enum):
    """Sentinel for arguments whose default is read from ``settings`` at call time."""
    FROM_SETTINGS = enum.auto()


if _NUMBA_AVAILABLE:
    @_numba_njit(cache=True)  # type: ignore[misc, operator]
    def _shuffle_copy_fisher_yates_numba(src: np.ndarray, dst: np.ndarray, rand_floats: np.ndarray) -> None:
        """Copy ``src`` into ``dst`` and shuffle ``dst`` in place with the Fisher-Yates algorithm.

        ``rand_floats`` holds ``len(src) - 1`` uniform draws from [0, 1), one per swap. All
        permutations are equally likely up to the negligible bias noted below, as with
        ``np.random.Generator.shuffle``. See https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle.
        """
        n = src.shape[0]
        for i in range(n):
            dst[i] = src[i]

        k = 0
        for i in range(n - 1, 0, -1):
            # Maps a 53-bit uniform u in [0, 1) to j in {0, ..., i}. Since u < 1, the product never
            # rounds up to i + 1. The 2**53 values of u do not split evenly into i + 1 buckets, which
            # biases the probabilities by a relative amount of order (i + 1) / 2**53.
            j = int(rand_floats[k] * (i + 1))
            k += 1
            tmp = dst[i]
            dst[i] = dst[j]
            dst[j] = tmp


def shuffle_masks_from_state(
    idces_joined: np.ndarray,
    n_a: int,
    work_buffer: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a uniformly random permutation of ``idces_joined`` into two groups.

    Parameters
    ----------
    idces_joined : np.ndarray
        Row positions of both groups.
    n_a : int
        Size of the first group; the second gets the remaining positions.
    work_buffer : np.ndarray or None
        Array of the shape and dtype of ``idces_joined`` to shuffle into, which avoids an allocation
        per permutation. If None, a new array is allocated.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    tuple of np.ndarray
        Row positions of the two groups. With a ``work_buffer``, these are views into it, which the
        next call with the same buffer overwrites; use or copy them before drawing the next split.

    Notes
    -----
    The numba and numpy paths both draw uniform permutations but consume ``rng`` differently, so
    for a fixed seed the drawn splits depend on ``settings.enable_numba_shuffle`` and on the number
    of indices.
    """
    if work_buffer is None:
        idces_permuted = rng.permutation(idces_joined)
    else:
        if settings.enable_numba_shuffle and _NUMBA_AVAILABLE and idces_joined.size > _NUMBA_SHUFFLE_SIZE_THRESHOLD:
            # Uniform floats keep the kernel free of integer-range and modulo-bias handling.
            rand_floats = rng.random(size=idces_joined.size - 1, dtype=np.float64)
            _shuffle_copy_fisher_yates_numba(idces_joined, work_buffer, rand_floats)
        else:
            np.copyto(work_buffer, idces_joined)
            rng.shuffle(work_buffer)
        idces_permuted = work_buffer

    shuffled_a_np = idces_permuted[:n_a]
    shuffled_b_np = idces_permuted[n_a:]

    if settings.debug:
        assert len(shuffled_a_np) == n_a
        assert len(np.intersect1d(shuffled_a_np, shuffled_b_np, assume_unique=False)) == 0

    return shuffled_a_np, shuffled_b_np


def shuffle_masks(
    mask_a: MaskLike | None = None,
    mask_b: MaskLike | None = None,
    *,
    idces_joined: np.ndarray | None = None,
    n_a: int | None = None,
    work_buffer: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly reassign the rows of two disjoint groups to two groups of the same sizes.

    The groups are given either as ``mask_a`` and ``mask_b`` or, to avoid converting masks on every
    draw, as their concatenated row positions ``idces_joined`` and the size ``n_a`` of the first
    group. Uses the shared generator from ``RandomState``; see ``shuffle_masks_from_state`` for
    ``work_buffer`` and the returned row positions.
    """
    rng = RandomState.get_rng()

    if idces_joined is None:
        assert mask_a is not None and mask_b is not None
        mask_a_np = _mask_to_positions(mask_a, len(mask_a))
        mask_b_np = _mask_to_positions(mask_b, len(mask_b))
        idces_a = mask_a_np
        idces_b = mask_b_np
        idces_joined = np.concatenate([idces_a, idces_b])
        n_a = idces_a.size

    assert n_a is not None

    return shuffle_masks_from_state(
        idces_joined=idces_joined,
        n_a=n_a,
        work_buffer=work_buffer,
        rng=rng,
    )


def _masks_are_disjoint(mask_a: MaskLike, mask_b: MaskLike, n_rows: int) -> bool:
    pos_a = _mask_to_positions(mask_a, n_rows)
    pos_b = _mask_to_positions(mask_b, n_rows)
    if len(pos_a) == 0 or len(pos_b) == 0:
        return True
    return len(np.intersect1d(pos_a, pos_b, assume_unique=False)) == 0


def bootstrap_variance_of_metric_diff(
        df: pd.DataFrame,
        metric: ComparisonMetric,
        group_mask_a: MaskLike,
        group_mask_b: MaskLike,
        max_num_bootstrap: int = 100
    ) -> tuple[float, float]:
    """Bootstrap estimate of the variance of ``metric(a) - metric(b)`` for two disjoint groups.

    Each group is resampled within itself, independently of the other, and the squared deviations
    of the bootstrapped difference from the observed difference are averaged, as in Eq. (1) of
    DiCiccio et al. (2020). Centering at the observed rather than the mean bootstrapped difference
    makes this the bootstrap mean squared error, which also contains the squared bootstrap bias.

    Replicates are drawn in batches of 10 until the estimate changes by less than 5 % from one
    batch to the next or ``max_num_bootstrap`` replicates are reached. Replicates on which the
    metric is not computable (e.g. a resample without positives) are left out; if they make up more
    than 30 % of all replicates, the estimate of that batch is NaN, and after three such batches the
    estimation stops.

    Returns
    -------
    var_of_metric_diff : float
        Variance estimate, NaN if not computable.
    metric_diff : float
        Observed difference ``metric(a) - metric(b)``, NaN if not computable.
    """
    metric_diff = metric(df, group_mask=group_mask_a, validate=False) - metric(df, group_mask=group_mask_b, validate=False) # type: ignore

    if np.isnan(metric_diff):
        return np.nan, np.nan

    bs_count = 0
    rel_var_change = np.inf
    var_of_metric_diff_est = np.nan
    batchsize = 10
    nan_rounds = 0

    bs_metric_samples_a = None
    bs_metric_samples_b = None

    while (bs_count < max_num_bootstrap) and rel_var_change > 0.05 and nan_rounds < 3:
        bs_metric_samples_a_batch = bootstrap_metric(df, metric, group_mask=group_mask_a, num_bootstrap=batchsize)
        bs_metric_samples_b_batch = bootstrap_metric(df, metric, group_mask=group_mask_b, num_bootstrap=batchsize)

        if bs_metric_samples_a is None or bs_metric_samples_b is None:
            bs_metric_samples_a = bs_metric_samples_a_batch
            bs_metric_samples_b = bs_metric_samples_b_batch
        else:
            bs_metric_samples_a = np.concatenate([bs_metric_samples_a, bs_metric_samples_a_batch], axis=0)
            bs_metric_samples_b = np.concatenate([bs_metric_samples_b, bs_metric_samples_b_batch], axis=0)

        assert bs_metric_samples_a.ndim == 1
        assert bs_metric_samples_b.ndim == 1

        var_of_metric_diff_est_new = nan_mean(np.square(bs_metric_samples_a - bs_metric_samples_b - metric_diff), 
                                              nan_fraction_allowed=0.3)

        bs_count += batchsize

        if np.isnan(var_of_metric_diff_est_new):
            rel_var_change = np.inf
            nan_rounds += 1

        elif not np.isfinite(var_of_metric_diff_est) or var_of_metric_diff_est == 0:
            # Nothing to compare against yet: first batch, NaN or zero previous estimate.
            rel_var_change = np.inf

        else:
            rel_var_change = np.abs((var_of_metric_diff_est_new - var_of_metric_diff_est) / var_of_metric_diff_est)

        var_of_metric_diff_est = var_of_metric_diff_est_new

    return var_of_metric_diff_est, metric_diff # type: ignore


def est_variance_of_metric_diff(
        df: pd.DataFrame,
        metric: ComparisonMetric,
        group_mask: MaskLike,
        complement_mask: MaskLike,
        max_num_bootstrap: int,
    ) -> tuple[float, float]:
    """Estimate the variance of ``metric(group) - metric(complement)``, analytically if possible.

    Uses the metric's analytical variance if it is a ``MetricWithAnalyticalVar`` and
    ``bootstrap_variance_of_metric_diff`` otherwise.

    Returns
    -------
    var_of_metric_diff : float
        Variance estimate, NaN if not computable.
    metric_diff : float
        Observed difference, NaN if it or (for analytical variances) its variance is not computable.
    """
    if isinstance(metric, MetricWithAnalyticalVar):
        # A metric whose value depends on the group's complement gets the other side as complement.
        if metric.reference_class == "complement":
            kwargs_a, kwargs_b = {"complement_mask": complement_mask}, {"complement_mask": group_mask}
        else:
            kwargs_a = kwargs_b = {}
        metric_val_a, metric_var_a = metric.get_variance(df, group_mask=group_mask, validate=False, return_val=True, **kwargs_a) # type: ignore
        metric_val_b, metric_var_b = metric.get_variance(df, group_mask=complement_mask, validate=False, return_val=True, **kwargs_b) # type: ignore
        metric_diff = metric_val_a - metric_val_b
        # Var(a - b) = Var(a) + Var(b) requires independent estimates, e.g. ones computed from disjoint
        # sets of independent rows. For reference_class 'self', this follows from the group and its
        # complement being disjoint; a metric with reference_class 'complement' must ensure it itself.
        if settings.debug:
            assert _masks_are_disjoint(group_mask, complement_mask, len(df))
        var_of_metric_diff_est = metric_var_a + metric_var_b

        # A zero analytical variance signals a degenerate plug-in estimate (e.g. a proportion of exactly
        # 0 or 1 in a small group), not a metric known without error. Studentizing with it would inflate
        # |S|, so the split counts as not computable.
        if metric_var_a == 0 or metric_var_b == 0 or not np.isfinite(var_of_metric_diff_est):
            var_of_metric_diff_est, metric_diff = np.nan, np.nan
    else:
        var_of_metric_diff_est, metric_diff = bootstrap_variance_of_metric_diff(df, metric, 
                                                                                group_mask_a=group_mask,
                                                                                group_mask_b=complement_mask, 
                                                                                max_num_bootstrap=max_num_bootstrap)
        
    if np.isfinite(var_of_metric_diff_est):
        assert var_of_metric_diff_est >= 0
        
    return var_of_metric_diff_est, metric_diff


def _studentize(metric_diff: float, var_of_metric_diff: float) -> float:
    """Return ``metric_diff / sqrt(var_of_metric_diff)``, or NaN if it is not computable.

    A zero variance estimate (typically from very small groups) counts as not computable, since it
    would make the statistic infinite.
    """
    if not (np.isfinite(metric_diff) and np.isfinite(var_of_metric_diff)) or var_of_metric_diff == 0:
        return np.nan
    return metric_diff / np.sqrt(var_of_metric_diff)


def _clopper_pearson_lower(M: int, N: int, confidence: float) -> float:
    """One-sided Clopper-Pearson lower confidence bound for a binomial proportion.

    The exact bound of Clopper and Pearson (1934): with probability at least ``confidence``, the
    true proportion lies above it. For M >= 1, it is the ``1 - confidence`` quantile of
    Beta(M, N - M + 1).

    Parameters
    ----------
    M : int
        Number of successes, here permutations at least as extreme as the observed statistic (>= 1).
    N : int
        Number of trials, here valid permutations.
    confidence : float
        Confidence level, e.g. 0.9999.

    Returns
    -------
    float
        Lower confidence bound on the success probability.
    """
    return float(scipy.stats.beta.ppf(1.0 - confidence, M, N - M + 1))


def _exceedances(valid: np.ndarray, S_base: float) -> int:
    """Count the permuted statistics at least as extreme as ``S_base`` in absolute value.

    Statistics within a relative tolerance of ``100 * eps`` below ``|S_base|`` count as ties and
    hence as exceedances. Permutations that leave the statistic mathematically unchanged (e.g. ones
    that only reorder rows within each group) can otherwise compute a few ulps below it, because
    floating-point sums such as ``np.nanvar`` depend on the order of the elements. The validity of
    the p-value relies on ties counting against ``S_base``. The tolerance is the one
    ``scipy.stats.permutation_test`` uses.
    """
    tol = 100 * np.finfo(float).eps * abs(S_base)
    return int(np.sum(np.abs(valid) >= np.abs(S_base) - tol))


def _clearly_non_significant(S_permut: np.ndarray, S_base: float, alpha: float) -> bool:
    """Early-stopping criterion of ``studentized_permut_pval``; see its Notes."""
    valid = S_permut[~np.isnan(S_permut)]
    if valid.size < _EARLY_STOP_MIN_PERMUT:
        return False
    M = _exceedances(valid, S_base)
    if M < _EARLY_STOP_MIN_M:
        return False
    return _clopper_pearson_lower(M, valid.size, _EARLY_STOP_CONFIDENCE) > alpha


def studentized_permut_pval(
    df: pd.DataFrame,
    metric: ComparisonMetric,
    group_filter: GroupFilter,
    num_permut: int | None = None,
    max_num_bootstrap: int | None = None,
    *,
    pval_early_stop_alpha: float | None | Literal[_Default.FROM_SETTINGS] = _Default.FROM_SETTINGS,
    ) -> tuple[float, float]:
    """Two-sided studentized permutation test for a metric difference between a group and its complement.

    The statistic is S = (m_a - m_b) / sqrt(V), where m_a and m_b are the metric in the group
    selected by ``group_filter`` and in its complement, and V estimates the variance of their
    difference, analytically if the metric provides one and by bootstrap otherwise. Each
    permutation reassigns the rows at random to two groups of the original sizes and recomputes S,
    including V, on that split (Algorithm 1 of DiCiccio et al., 2020). Studentizing every permuted
    statistic lets the test address the weak null hypothesis m_a = m_b, not only the strong null
    hypothesis that both groups follow the same distribution: the test is valid in finite samples
    under the strong null and, for asymptotically pivotal statistics, asymptotically valid under the
    weak null (DiCiccio et al., 2020, Sec. 2.3; Theorem 2 proves this for the AUROC difference).
    Algorithm 1 is stated for the upper tail; this test is two-sided, based on |S|.

    Parameters
    ----------
    df : pd.DataFrame
        Data with the columns ``metric`` and ``group_filter`` need.
    metric : ComparisonMetric
        Metric to compare.
    group_filter : GroupFilter
        Selects the group, which is compared with ``group_filter.complement(df)``.
    num_permut : int, optional
        Maximum number of permutations. Defaults to ``settings.N_test_permut``.
    max_num_bootstrap : int, optional
        Maximum number of bootstrap replicates per variance estimate, for metrics without an
        analytical variance. Defaults to ``settings.max_N_student_bootstrap``.
    pval_early_stop_alpha : float or None, optional
        Significance level for early stopping and for refusing tests that cannot reach it (see
        Notes); it should be the level at which the corrected p-values are read. Defaults to
        ``settings.pval_early_stop_alpha``, read at call time. If None, all ``num_permut``
        permutations are run and every computable p-value is returned.

    Returns
    -------
    pval : float
        Two-sided p-value; NaN if not computable or refused (see Notes).
    metric_diff : float
        Observed difference m_a - m_b; NaN if not computable.

    Notes
    -----
    **P-value.** Let S_0 be the observed statistic, N_valid the number of permutations on which S
    is computable, and M the number of those with |S| >= |S_0|, ties within floating-point
    tolerance included. The p-value is (M + 1) / (N_valid + 1). Under the strong null and given the
    pooled rows, the observed split and the sampled splits are independent uniform draws from all
    splits of the pooled rows into groups of the original sizes, so |S_0| is equally likely to take
    any rank among the N + 1 statistics, and P(p <= t) <= t for every t and every number of
    permutations (Phipson and Smyth, 2010). Because the observed split may be drawn again, the
    p-value is slightly conservative; this matters only if the number of distinct splits is small
    (ibid.).

    Leaving out non-computable permutations preserves this guarantee. Given that S_0 is computable,
    the observed split and the computable sampled splits are independent uniform draws from the
    computable splits, so the same argument applies with N_valid in place of N. (For bootstrapped
    variances, read "split" as the split together with its bootstrap draws.) The probability that a
    sampled split is computable does not depend on which split was observed, so N_valid carries no
    information about the rank of S_0.

    **NaN p-values.** The p-value is NaN if a group is empty, if S_0 is not computable, or if no
    permuted statistic is. With ``pval_early_stop_alpha`` set, it is also NaN if non-computable
    permutations alone keep it from reaching alpha: 1 / (N_valid + 1) > alpha although
    1 / (num_permut + 1) <= alpha. Holm, Hochberg and Benjamini-Hochberg never reject a p-value above
    alpha, so such a test cannot be significant, yet it would enlarge the family of tests corrected
    for. As NaN, it is left out of the correction, which ``compare_groups`` applies to non-NaN
    p-values only. The decision depends only on N_valid, so the remaining p-values stay valid. If
    ``num_permut`` itself is too small to reach alpha, p-values are returned; ``compare_groups``
    checks the number of permutations up front when correcting with Benjamini-Hochberg.

    **Early stopping.** With ``pval_early_stop_alpha`` set, permutations are drawn in batches of 100.
    After each batch except the last, the run stops if N_valid >= 100, M >= 20, and the one-sided
    99.99 % Clopper-Pearson lower confidence bound (Clopper and Pearson, 1934) on the exceedance
    probability exceeds alpha. The exceedance probability is the p-value an unlimited number of
    permutations would give, and M is binomial given N_valid. The bound is at most
    M / N_valid <= (M + 1) / (N_valid + 1), so a stopped run returns p > alpha. Stopping thus never
    produces a p-value <= alpha that the full run would not have produced, and validity at all
    levels up to alpha is unaffected. Above alpha, the p-value of a stopped run rests on fewer
    permutations, so results are best read at levels up to alpha. A stop costs power if the
    exceedance probability is in fact at most alpha; by a union bound over the K checks, this has
    probability at most K * 1e-4 (9e-4 for 1000 permutations). Clopper-Pearson bounds hold for any M,
    so the M >= 20 floor is only an extra margin, after the M >= 10 rule that Knijnenburg et al.
    (2009) need for their normal-approximation bounds. Stopped runs are exempt from the refusal
    above: their N_valid is small because they stopped, and their p-value exceeds alpha anyway.
    """
    alpha: float | None = (
        settings.pval_early_stop_alpha if pval_early_stop_alpha is _Default.FROM_SETTINGS else pval_early_stop_alpha
    )

    if num_permut is None:
        num_permut = settings.N_test_permut

    if max_num_bootstrap is None:
        max_num_bootstrap = settings.max_N_student_bootstrap

    group_mask = group_filter(df)
    complement_mask = group_filter.complement(df)

    # Precompute shuffle state once to avoid repeated mask->index conversion in
    # every permutation draw.
    idces_a_base = _mask_to_positions(group_mask, len(df))
    idces_b_base = _mask_to_positions(complement_mask, len(df))

    if len(idces_a_base) == 0 or len(idces_b_base) == 0:
        return np.nan, np.nan

    idces_joined_base = np.concatenate([idces_a_base, idces_b_base])
    n_a_base = idces_a_base.size
    shuffle_work = np.empty_like(idces_joined_base)

    var_of_metric_diff_est, metric_diff = est_variance_of_metric_diff(df, metric=metric, group_mask=group_mask, complement_mask=complement_mask,
                                                                      max_num_bootstrap=max_num_bootstrap)

    if not (np.isfinite(metric_diff) and np.isfinite(var_of_metric_diff_est)):
        return np.nan, np.nan

    S_base = _studentize(metric_diff, var_of_metric_diff_est)
    if np.isnan(S_base):
        # Zero variance estimate: no test, but the effect itself is well defined.
        return np.nan, metric_diff

    def get_studentized_permut() -> float:
        # Shuffling precomputed row positions into a reused buffer is much faster here than
        # off-the-shelf permutation routines, which convert masks to indices on every draw.
        # test_shuffle_distribution.py compares the distribution of the drawn splits with np.random.permutation.
        mask_a, mask_b = shuffle_masks(
            idces_joined=idces_joined_base,
            n_a=n_a_base,
            work_buffer=shuffle_work,
        )

        var_of_metric_diff_est, metric_diff = est_variance_of_metric_diff(df, metric, group_mask=mask_a, complement_mask=mask_b,
                                                                          max_num_bootstrap=max_num_bootstrap)
        # Typically NaN if a permuted group has too few positives or negatives for the metric.
        return _studentize(metric_diff, var_of_metric_diff_est)

    S_permut = np.full(num_permut, np.nan)
    early_stopped = False
    for batch_start in range(0, num_permut, _EARLY_STOP_BATCH):
        batch_end = min(batch_start + _EARLY_STOP_BATCH, num_permut)
        for i in range(batch_start, batch_end):
            S_permut[i] = get_studentized_permut()

        if (
            alpha is not None
            and batch_end < num_permut
            and _clearly_non_significant(S_permut[:batch_end], S_base, alpha)
        ):
            S_permut = S_permut[:batch_end]
            early_stopped = True
            break

    valid = S_permut[~np.isnan(S_permut)]
    if valid.size == 0:
        # No permutation gave a computable statistic, so there is no reference distribution.
        return np.nan, metric_diff

    if alpha is not None and not early_stopped:
        alpha_reachable_with_all = 1 / (num_permut + 1) <= alpha
        alpha_reachable_with_valid = 1 / (valid.size + 1) <= alpha
        if alpha_reachable_with_all and not alpha_reachable_with_valid:
            # Only non-computable permutations keep this test from reaching alpha (see Notes).
            return np.nan, metric_diff

    pval = (_exceedances(valid, S_base) + 1) / (valid.size + 1)

    if settings.debug and early_stopped:
        assert alpha is not None and pval > alpha

    return pval, metric_diff
