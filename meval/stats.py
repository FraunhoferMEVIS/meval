import numpy as np
import pandas as pd
from typing import Optional
from collections.abc import Callable
import scipy

try:
    from numba import njit as _numba_njit
    _NUMBA_AVAILABLE = True
except Exception:
    _numba_njit = None
    _NUMBA_AVAILABLE = False

from ._array_types import FloatArray, LabelArray, MaskLike, NumericArray
from .config import settings
from .group_filter import GroupFilter
from .metrics.ComparisonMetric import ComparisonMetric, MetricWithAnalyticalVar


if _NUMBA_AVAILABLE:
    @_numba_njit(cache=True)  # type: ignore[misc, operator]
    def _shuffle_copy_fisher_yates_numba(src: np.ndarray, dst: np.ndarray, rand_floats: np.ndarray) -> None:
        # Claude says: The Fisher-Yates implementation itself is correct. rand_floats size is n-1, the loop 
        # runs n-1 iterations with index k in [0, n-2] — no out-of-bounds. The output is a uniform permutation 
        # equivalent to rng.shuffle.
        # Also see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
        n = src.shape[0]
        for i in range(n):
            dst[i] = src[i]

        k = 0
        for i in range(n - 1, 0, -1):
            # rand_floats contains uniform values in [0.0, 1.0)
            # Multiplying by (i + 1) and casting to int gives an unbiased integer in [0, i]
            j = int(rand_floats[k] * (i + 1))
            k += 1
            tmp = dst[i]
            dst[i] = dst[j]
            dst[j] = tmp

class RandomState:
    _rng = None
    
    @classmethod
    def get_rng(cls):
        if cls._rng is None:
            cls._rng = np.random.default_rng(seed=settings.seed)
        return cls._rng
    
    @classmethod
    def reset(cls):
        """Reset RNG with current seed from settings."""
        cls._rng = np.random.default_rng(seed=settings.seed)


def variance_of_proportion(numerator: int, denominator: int) -> float:
    # exact variance is Var(phat) = p * (1-p) / n
    # but we don't know the true p, we only have its finite-sample estimate
    # To get a finite-sample unbiased estimate of the variance, we divide by n-1 instead
    # (See https://math.stackexchange.com/questions/3968141/should-the-unbiased-estimator-of-the-variance-of-the-sample-proportion-have-n-1
    #  ... yes, I'd love a better / more definitive reference.)
    phat = numerator / denominator
    variance = phat * (1-phat) / (denominator - 1)
    assert variance >= 0
    return variance


def decide_stratify(
    y_true: pd.Series | LabelArray,
        threshold: int = 10
        ) -> tuple[bool, dict[int, int]]:
    """
    Decide whether to stratify based on class counts.
    
    Returns:
        stratify: Whether to use stratified sampling
        class_counts: Dictionary mapping class labels to their counts
    """
    classes = np.unique(y_true)
    class_counts = {cls: (y_true == cls).sum() for cls in classes}
    
    assert sum(class_counts.values()) == len(y_true)
    
    # Stratify if any class has fewer than threshold samples
    stratify = any(count < threshold for count in class_counts.values())
    
    return stratify, class_counts


def bootstrap_metric(
        df: pd.DataFrame, 
        metric: ComparisonMetric, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[MaskLike] = None,
        num_bootstrap: Optional[int] = None
        ) -> np.ndarray:

    rng = RandomState.get_rng()
    if num_bootstrap is None:
        num_bootstrap = settings.N_bootstrap
   
    if metric.reference_class == 'self':
        if group_mask is None:
            assert group_filter is not None
            group_mask = group_filter(df)
            
        sample_positions = _mask_to_positions(group_mask, len(df))
        N_sample = len(sample_positions)

        if metric.needs_all_classes:
            # `mask` accepts either a boolean mask or integer row positions.
            y_true = ComparisonMetric.get_multiclass_y_true(df, mask=sample_positions, validate=False)
            stratify, class_counts = decide_stratify(y_true)
        else:
            stratify = False
            class_counts = {}
            y_true = None

        if stratify:
            assert y_true is not None
            y_true_np = np.asarray(y_true)
            # Stratified sampling: sample from each class separately
            bs_idces_by_class = []
            for cls, count in class_counts.items():
                cls_indices = sample_positions[y_true_np == cls]
                bs_idces_cls = rng.choice(cls_indices, (count, num_bootstrap), replace=True)
                bs_idces_by_class.append(bs_idces_cls)
            bs_idces_all = np.concatenate(bs_idces_by_class, axis=0)            

        else:
            bs_idces_all = rng.choice(sample_positions, (N_sample, num_bootstrap), replace=True)

        metric_bs = np.empty(num_bootstrap, dtype=float)
        for j in range(num_bootstrap):
            bs_idces = bs_idces_all[:, j]
            metric_bs[j] = metric(
                df,
                group_mask=bs_idces,
                validate=False,
            )

        assert len(metric_bs) == num_bootstrap
        return metric_bs
    else:
        # metric with cross-group calculations; complicates UQ by BS
        raise NotImplementedError


def bootstrap_ci(df, metric, group_filter, num_bootstrap, ci_alpha):
    metric_bs = bootstrap_metric(df, metric, group_filter, num_bootstrap=num_bootstrap)
    lower = ci_nan_quantile(metric_bs, (1 - ci_alpha) / 2)
    #med = ci_nan_quantile(metric_bs, 0.5)
    upper = ci_nan_quantile(metric_bs, ci_alpha + (1 - ci_alpha) / 2)
    return lower, upper


def _mask_to_positions(mask: MaskLike, n_rows: int) -> np.ndarray:
    # Claude says: np.asarray(mask, dtype=int) on an already-int64 array returns the same object. 
    # In bootstrap_metric, this means sample_positions aliases shuffle_work[...]. 
    # As analyzed, this is safe because rng.choice(sample_positions, ...) reads before any next 
    # shuffle occurs. Not a bug but worth being aware of.
    if isinstance(mask, pd.Series):
        mask_np = mask.to_numpy(copy=False)
    else:
        mask_np = mask

    if np.issubdtype(mask_np.dtype, np.integer):
        return np.asarray(mask_np, dtype=int)

    if mask_np.dtype == bool:
        assert len(mask_np) == n_rows
        return np.flatnonzero(mask_np)

    raise TypeError("Unsupported mask dtype; expected bool mask or integer positions.")


def _masks_are_disjoint(mask_a: MaskLike, mask_b: MaskLike, n_rows: int) -> bool:
    pos_a = _mask_to_positions(mask_a, n_rows)
    pos_b = _mask_to_positions(mask_b, n_rows)
    if len(pos_a) == 0 or len(pos_b) == 0:
        return True
    return len(np.intersect1d(pos_a, pos_b, assume_unique=False)) == 0


def bootstrap_variance_of_metric_diff(
        df: pd.DataFrame,
        metric: ComparisonMetric,
        group_mask_a: pd.Series,
        group_mask_b: pd.Series,
        max_num_bootstrap: int = 100
    ) -> tuple[float, float]:

    metric_diff = metric(df, group_mask=group_mask_a, validate=False) - metric(df, group_mask=group_mask_b, validate=False) # type: ignore

    if np.isnan(metric_diff):
        return np.nan, np.nan

    bs_count = 0
    rel_var_change = np.inf
    var_of_metric_diff_est = 0
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

        elif bs_count == batchsize:
            rel_var_change = np.inf

        elif var_of_metric_diff_est == 0:
            rel_var_change = np.inf

        else:
            rel_var_change = np.abs((var_of_metric_diff_est_new - var_of_metric_diff_est) / var_of_metric_diff_est)

        var_of_metric_diff_est = var_of_metric_diff_est_new

    return var_of_metric_diff_est, metric_diff # type: ignore


def shuffle_masks(
    mask_a: Optional[MaskLike] = None,
    mask_b: Optional[MaskLike] = None,
    *,
    idces_joined: Optional[np.ndarray] = None,
    n_a: Optional[int] = None,
    work_buffer: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:

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


def shuffle_masks_from_state(
    idces_joined: np.ndarray,
    n_a: int,
    work_buffer: Optional[np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:

    # Claude says: shuffled_a_np and shuffled_b_np are views into work_buffer. 
    # They're consumed before the next permutation overwrites the buffer, so this 
    # is safe in the current sequential code, but the contract is implicit.

    if work_buffer is None:
        idces_permuted = rng.permutation(idces_joined)
    else:
        if settings.enable_numba_shuffle and _NUMBA_AVAILABLE and idces_joined.size > 128:
            # Generate 64-bit uniform floats to avoid modulo bias and uint bounds issues
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


def est_variance_of_metric_diff(df, metric, group_mask, complement_mask, max_num_bootstrap):
    if isinstance(metric, MetricWithAnalyticalVar):
        metric_val_a, metric_var_a = metric.get_variance(df, group_mask=group_mask, validate=False, return_val=True) # type: ignore
        metric_val_b, metric_var_b = metric.get_variance(df, group_mask=complement_mask, validate=False, return_val=True) # type: ignore
        metric_diff = metric_val_a - metric_val_b
        # This is only true if the two samples are independent! That is the case in the sample vs. complement situation.
        if settings.debug:
            assert _masks_are_disjoint(group_mask, complement_mask, len(df))
        var_of_metric_diff_est = metric_var_a + metric_var_b

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
    

def _clopper_pearson_lower(M: int, N: int, confidence: float) -> float:
    """
    Clopper-Pearson lower confidence bound for a binomial proportion.

    Given M successes in N trials, returns the lower bound p_lo of the
    exact one-sided (lower) Clopper-Pearson interval at the given confidence
    level.

    This is used for early stopping of the permutation test: if p_lo > alpha,
    then with probability >= `confidence` the true P-value exceeds alpha,
    and we can safely conclude the result is non-significant.

    Args:
        M: number of permutation values exceeding the test statistic (>= 1).
        N: total number of valid (non-NaN) permutation values.
        confidence: desired confidence level (e.g. 0.9999 for 99.99%).

    Returns:
        Clopper-Pearson lower confidence bound on the true proportion.
    """
    # One-sided lower bound at confidence level c uses quantile (1-c)
    # of Beta(M, N-M+1).
    # scipy.stats.beta.ppf(q, a, b) gives the q-th quantile of Beta(a,b).
    alpha_tail = 1.0 - confidence
    return float(scipy.stats.beta.ppf(alpha_tail, M, N - M + 1))


def studentized_permut_pval(
    df: pd.DataFrame,
    metric: ComparisonMetric,
    group_filter: GroupFilter,
    num_permut: Optional[int] = None,
    max_num_bootstrap: Optional[int] = None,
    correct_zero_pvals: bool = True,
    pval_early_stop_alpha: Optional[float] = settings.pval_early_stop_alpha,
    ) -> tuple[float, float]:
    # Algo 1 in https://arxiv.org/abs/2007.05124
    # (My implementation)
    #
    # Early stopping is enabled by default using
    # settings.pval_early_stop_alpha.
    # Set pval_early_stop_alpha=None to force the legacy behaviour of always
    # running all permutations in a specific call.
    #
    # Early stopping logic (when pval_early_stop_alpha is not None):
    # Permutations are run in batches of 100.  After each batch (once at
    # least 100 valid permutations have been collected), we check whether
    # the result is clearly non-significant. This criterion is inspired by
    # the binomial-count logic in Knijnenburg et al. (2009), but is an
    # intentionally simpler method than their ECDF/GPD switch: we stop if the
    # one-sided 99.99% Clopper-Pearson lower bound on the empirical P-value
    # estimate exceeds pval_early_stop_alpha.
    # We additionally require M >= 20 exceedances (vs M >= 10 in their paper)
    # for extra conservatism.
    # We never stop early when the P-value looks small.
    # Knijnenburg et al. (2009): https://pmc.ncbi.nlm.nih.gov/articles/PMC2687965/

    # Confidence level for the Clopper-Pearson lower bound used in the
    # early-stop criterion.  We stop only when the CP lower bound on P̂
    # already exceeds pval_early_stop_alpha, meaning: with probability
    # >= _EARLY_STOP_CONFIDENCE the true P-value is above alpha.
    # 99.99% is very conservative; the expected false-early-stop rate
    # across a full analysis is negligible.
    _EARLY_STOP_CONFIDENCE = 0.9999
    # Require at least this many exceedances before trusting the CP bound.
    # Higher than Knijnenburg's M>=10 for extra conservatism.
    _EARLY_STOP_MIN_M = 20
    # Minimum permutations before any early-stop check.
    _EARLY_STOP_MIN_PERMUT = 100
    # Batch size for early-stop checks.
    _EARLY_STOP_BATCH = 100
    early_stopped = False

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
    idces_joined_base = np.concatenate([idces_a_base, idces_b_base])
    n_a_base = idces_a_base.size
    shuffle_work = np.empty_like(idces_joined_base)

    if len(idces_a_base) == 0 or len(idces_b_base) == 0:
        return np.nan, np.nan

    var_of_metric_diff_est, metric_diff = est_variance_of_metric_diff(df, metric=metric, group_mask=group_mask, complement_mask=complement_mask,
                                                                      max_num_bootstrap=max_num_bootstrap)

    if not (np.isfinite(metric_diff) and np.isfinite(var_of_metric_diff_est)):
        return np.nan, np.nan
    elif var_of_metric_diff_est == 0:
        return np.nan, metric_diff

    S_base = metric_diff / np.sqrt(var_of_metric_diff_est)

    def get_studentized_permut():
        # This is a custom reimplementation of permutation shuffling that beats off-the-shelf implementations 
        # by a large margin in our setting, because it avoids repeated mask->index conversions and because
        # we can exploit the fact that we are repeatedly shuffling the same set of indices with the same sizes. 
        # See test_shuffle_distributions.py for a comparison of the distributions of our shuffle vs. np.random.permutation.
        mask_a, mask_b = shuffle_masks(
            idces_joined=idces_joined_base,
            n_a=n_a_base,
            work_buffer=shuffle_work,
        )

        var_of_metric_diff_est, metric_diff = est_variance_of_metric_diff(df, metric, group_mask=mask_a, complement_mask=mask_b,
                                                                          max_num_bootstrap=max_num_bootstrap)
        if not (np.isfinite(metric_diff) and np.isfinite(var_of_metric_diff_est)):
            S = np.nan  # This will typically occur because some metrics cannot be calculated in samples with very few positives/negatives
        elif var_of_metric_diff_est == 0:
            S = np.nan  # Typically a result of very small sample sizes. Also cf. above in est_var_of_metric_diff, where we similarly set this case to nan
        else:
            S = metric_diff / np.sqrt(var_of_metric_diff_est)
        return S

    if pval_early_stop_alpha is None:
        # Original behaviour: run all permutations upfront.
        S_permut = np.array([get_studentized_permut() for _ in range(num_permut)])
    else:
        # Batched loop with conservative early stopping.
        S_permut_list: list[float] = []
        for batch_start in range(0, num_permut, _EARLY_STOP_BATCH):
            batch_end = min(batch_start + _EARLY_STOP_BATCH, num_permut)
            for _ in range(batch_end - batch_start):
                S_permut_list.append(get_studentized_permut())

            n_done = len(S_permut_list)
            if n_done < _EARLY_STOP_MIN_PERMUT:
                continue

            s_arr = np.array(S_permut_list)
            valid = s_arr[~np.isnan(s_arr)]
            N_valid = len(valid)
            if N_valid < _EARLY_STOP_MIN_PERMUT:
                continue

            M = int(np.sum(np.abs(valid) >= np.abs(S_base)))
            if M < _EARLY_STOP_MIN_M:
                # Too few exceedances; CP bound would be unreliable.
                continue

            # Clopper-Pearson lower bound: with probability
            # >= _EARLY_STOP_CONFIDENCE the true P-value exceeds lb.
            lb = _clopper_pearson_lower(M, N_valid, _EARLY_STOP_CONFIDENCE)
            if lb > pval_early_stop_alpha:
                early_stopped = True
                break  # Clearly non-significant; stop early.

        S_permut = np.array(S_permut_list)

    # two-sided pval; NaNs are handled by nan_mean
    pval = nan_mean(np.abs(S_permut) >= np.abs(S_base), nan_fraction_allowed=0.5)
    pval = float(pval)

    if correct_zero_pvals:
        # We cannot find pvals < 1/n_valid_permut, ever.
        # So for any pval == 0 above, the correct interpretation is "pval < 1/n_valid_permut".
        # To enable meaningful further analyses, it is often useful to set pvals==0 to a value closer to 1/n_valid_permut.
        if pval == 0:
            # This combination should be impossible, because early stopping
            # requires many exceedances by construction.
            assert not early_stopped, "Internal error: early-stopped run produced pval==0."
            valid = S_permut[~np.isnan(S_permut)]
            n_valid_permut = len(valid)
            assert n_valid_permut > 0
            pval = 0.99 / n_valid_permut

    return pval, metric_diff


def bootstrap_curve(
    target: LabelArray,
    pred_probs: FloatArray,
    curve_fun: Callable[..., FloatArray],
        num_bootstraps: int, 
        num_samples: int
    ) -> FloatArray:

    if len(np.unique(target)) >= 3:
        raise NotImplementedError("bootstrap_curve called with multiclass target but only implemented for the binary case.")

    rng = RandomState.get_rng()

    N_predictions = len(target)

    yvals_bs = np.zeros((num_bootstraps, num_samples)) * np.nan

    stratify, class_counts = decide_stratify(target)

    for bs_idx in range(num_bootstraps):
        
        if stratify:
            # Stratified sampling: sample from each class separately
            bs_idces_by_class = []
            for cls, count in class_counts.items():
                bs_idces_cls = rng.choice(np.flatnonzero(target == cls), count, replace=True)
                bs_idces_by_class.append(bs_idces_cls)
            bs_idces = np.concatenate(bs_idces_by_class, axis=0)  

        else:
            bs_idces = rng.choice(range(N_predictions), N_predictions)

        if (target[bs_idces] == 0).sum() > 0 and (target[bs_idces] == 1).sum() > 0:
            yvals_bs[bs_idx, :] = curve_fun(target=target[bs_idces], pred_probs=pred_probs[bs_idces])

    return yvals_bs


def ci_nan_quantile(
    a: NumericArray,
    q: float | FloatArray,
        axis: Optional[int] = None, 
        nan_fraction_allowed: float = 0.1
    ) -> float | FloatArray:
    a_float = np.asarray(a, dtype=float)
    assert np.sum(np.isinf(a_float[:])) == 0

    if axis is None:
        too_many_nan = np.sum(np.isnan(a_float[:])) > nan_fraction_allowed * len(a_float[:])
        return np.nan if too_many_nan else np.nanquantile(a_float, q, axis=None)
    
    else:
        too_many_nan = np.sum(np.isnan(a_float), axis=axis) > nan_fraction_allowed * a_float.shape[axis]
        quantile = np.ones_like(too_many_nan, dtype=np.float64)
        quantile[too_many_nan] = np.nan
        if axis == 0 and np.ndim(a_float) == 2:
            quantile[~too_many_nan] = np.nanquantile(a_float[:, ~too_many_nan], q, axis=axis)
        elif axis == 1 and np.ndim(a_float) == 2:
            quantile[~too_many_nan] = np.nanquantile(a_float[~too_many_nan, :], q, axis=axis)
        else:
            raise NotImplementedError
        
        return quantile


def nan_mean(
    a: NumericArray,
        axis: Optional[int] = None, 
        nan_fraction_allowed: float = 0.1
    ) -> float | FloatArray:
    a_float = np.asarray(a, dtype=float)
    assert np.sum(np.isinf(a_float[:])) == 0

    if axis is None:
        too_many_nan = np.sum(np.isnan(a_float[:])) > nan_fraction_allowed * len(a_float[:])
        return np.nan if too_many_nan else np.nanmean(a_float, axis=None) # type: ignore
    
    else:
        too_many_nan = np.sum(np.isnan(a_float), axis=axis) > nan_fraction_allowed * a_float.shape[axis]
        mean = np.ones_like(too_many_nan, dtype=np.float64)
        mean[too_many_nan] = np.nan
        if axis == 0 and np.ndim(a_float) == 2:
            mean[~too_many_nan] = np.nanmean(a_float[:, ~too_many_nan], axis=axis)
        elif axis == 1 and np.ndim(a_float) == 2:
            mean[~too_many_nan] = np.nanmean(a_float[~too_many_nan, :], axis=axis)
        else:
            raise NotImplementedError
        
        return mean
    

def hanley_var(auroc: float, y_true: pd.Series | LabelArray):
    nx = np.sum(y_true == 1)
    ny = np.sum(y_true == 0)
    assert nx+ny == len(y_true)
    nxstar = nystar = len(y_true) / 2 - 1
    var = auroc * (1-auroc) * (1 + nxstar * (1-auroc)/(2-auroc) + nystar*auroc/(1+auroc))/(nx*ny)
    return var


def newcombe_auroc_ci(auroc_val: float, y_true: pd.Series | LabelArray, ci_alpha: float):  # this wants a 'small' ci_alpha, i.e. 0.05 (and not 0.95)
    
    if np.isnan(auroc_val):
        return [np.nan, np.nan]
    
    assert isinstance(y_true, pd.Series) or isinstance(y_true, np.ndarray)
    z = scipy.stats.norm(loc=0, scale=1).ppf(1-ci_alpha/2)
    if auroc_val - 1e-4 > 0.0:
        lb_result = scipy.optimize.root_scalar(lambda auroc_lb: np.abs(auroc_lb - auroc_val) - z * np.sqrt(hanley_var(auroc_lb, y_true)), bracket=[0, auroc_val-1e-4], xtol=1e-3)
        assert lb_result.converged
        lb = lb_result.root
    else:
        lb = 0.0

    if auroc_val + 1e-4 < 1.0:
        ub_result = scipy.optimize.root_scalar(lambda auroc_ub: np.abs(auroc_ub - auroc_val) - z * np.sqrt(hanley_var(auroc_ub, y_true)), bracket=[auroc_val+1e-4, 1.0], xtol=1e-3)
        assert ub_result.converged
        ub = ub_result.root
    else:
        ub = 1.0

    return [lb, ub]