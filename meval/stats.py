import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Optional
from collections.abc import Callable
import scipy

from .config import settings
from .group_filter import GroupFilter
from .metrics.ComparisonMetric import ComparisonMetric, MetricWithAnalyticalVar


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
        y_true: pd.Series | npt.NDArray[np.bool], 
        threshold: int = 10
        ) -> tuple[bool, int, int]:

    N_pos = (y_true == 1).sum()
    N_neg = (y_true == 0).sum()
    assert N_pos + N_neg == len(y_true)

    if N_pos < threshold or N_neg < threshold:
        stratify = True
    else:
        stratify = False   

    return stratify, N_pos, N_neg


def bootstrap_metric(
        df: pd.DataFrame, 
        metric: ComparisonMetric, 
        group_filter: Optional[GroupFilter] = None,
        group_mask: Optional[pd.Series] = None, 
        num_bootstrap: Optional[int] = None
        ) -> np.ndarray:

    rng = RandomState.get_rng()
    if num_bootstrap is None:
        num_bootstrap = settings.N_bootstrap
   
    if metric.reference_class == 'self':
        if group_mask is None:
            assert group_filter is not None
            group_mask = group_filter(df)
            
        N_sample = group_mask.sum()

        if metric.needs_pos_and_neg:
            y_true = ComparisonMetric.get_binary_y_true(df, mask=group_mask, validate=False)
            stratify, N_pos, N_neg = decide_stratify(y_true)
        else:
            stratify = False

        if stratify:
            bs_idces_pos = rng.choice(df.index[group_mask][y_true == 1], (N_pos, num_bootstrap), replace=True)
            bs_idces_neg = rng.choice(df.index[group_mask][y_true == 0], (N_neg, num_bootstrap), replace=True)
            bs_idces_all = np.concatenate([bs_idces_pos, bs_idces_neg], axis=0)
        else:
            bs_idces_all = rng.choice(df.index[group_mask], (N_sample, num_bootstrap), replace=True)

        metric_bs = np.apply_along_axis(lambda bs_idces: metric(df.loc[bs_idces, :], 
                                                                group_mask=group_mask.loc[bs_idces], 
                                                                validate=False),
                                        axis=0,
                                        arr=bs_idces_all)

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
            bs_metric_samples_a = np.concat([bs_metric_samples_a, bs_metric_samples_a_batch], axis=0)
            bs_metric_samples_b = np.concat([bs_metric_samples_b, bs_metric_samples_b_batch], axis=0)

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


def shuffle_masks(mask_a: pd.Series, mask_b: pd.Series) -> tuple[pd.Series, pd.Series]:

    rng = RandomState.get_rng()

    idces_a = np.nonzero(mask_a)[0]
    idces_b = np.nonzero(mask_b)[0]
    idces_joined = np.concatenate([idces_a, idces_b])
    idces_permuted = rng.permutation(idces_joined)

    shuffled_a = pd.Series(False, index=mask_a.index)
    shuffled_a.iloc[idces_permuted[:mask_a.sum()]] = True
    shuffled_b = pd.Series(False, index=mask_b.index)
    shuffled_b.iloc[idces_permuted[mask_a.sum():]] = True

    assert mask_a.sum() == shuffled_a.sum()
    assert mask_b.sum() == shuffled_b.sum()
    assert ((mask_a | mask_b) == (shuffled_a | shuffled_b)).all()

    return shuffled_a, shuffled_b


def est_variance_of_metric_diff(df, metric, group_mask, complement_mask, max_num_bootstrap):
    if isinstance(metric, MetricWithAnalyticalVar):
        metric_val_a, metric_var_a = metric.get_variance(df, group_mask=group_mask, validate=False, return_val=True) # type: ignore
        metric_val_b, metric_var_b = metric.get_variance(df, group_mask=complement_mask, validate=False, return_val=True) # type: ignore
        metric_diff = metric_val_a - metric_val_b
        # This is only true if the two samples are independent! That is the case in the sample vs. complement situation.
        assert not (group_mask & complement_mask).any()
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
    

def studentized_permut_pval(
    df: pd.DataFrame, 
    metric: ComparisonMetric, 
    group_filter: GroupFilter, 
    num_permut: Optional[int] = None,
    max_num_bootstrap: Optional[int] = None,
    correct_zero_pvals: bool = True
    ) -> tuple[float, float]:
    # Algo 1 in https://arxiv.org/abs/2007.05124
    # (My implementation)

    if num_permut is None:
        num_permut = settings.N_test_permut

    if max_num_bootstrap is None:
        max_num_bootstrap = settings.max_N_student_bootstrap

    group_mask = group_filter(df)
    complement_mask = group_filter.complement(df)

    if group_mask.sum() == 0 or complement_mask.sum() == 0:
        return np.nan, np.nan

    var_of_metric_diff_est, metric_diff = est_variance_of_metric_diff(df, metric=metric, group_mask=group_mask, complement_mask=complement_mask,
                                                                      max_num_bootstrap=max_num_bootstrap)

    if not (np.isfinite(metric_diff) and np.isfinite(var_of_metric_diff_est)):
        return np.nan, np.nan
    elif var_of_metric_diff_est == 0:
        return np.nan, metric_diff

    S_base = metric_diff / np.sqrt(var_of_metric_diff_est)

    def get_studentized_permut():
        mask_a, mask_b = shuffle_masks(group_mask, complement_mask)
        var_of_metric_diff_est, metric_diff = est_variance_of_metric_diff(df, metric, group_mask=mask_a, complement_mask=mask_b,
                                                                          max_num_bootstrap=max_num_bootstrap)
        if not (np.isfinite(metric_diff) and np.isfinite(var_of_metric_diff_est)):
            S = np.nan  # This will typically occur because some metrics cannot be calculated in samples with very few positives/negatives
        elif var_of_metric_diff_est == 0:
            S = np.nan  # Typically a result of very small sample sizes. Also cf. above in est_var_of_metric_diff, where we similarly set this case to nan
        else:
            S = metric_diff / np.sqrt(var_of_metric_diff_est)
        return S

    S_permut = np.array([get_studentized_permut() for _ in range(num_permut)])

    # two-sided pval
    pval = nan_mean(np.abs(S_permut) > np.abs(S_base), nan_fraction_allowed=0.5)

    if correct_zero_pvals:
        # We cannot find pvals < 1/n_permut, ever. 
        # So for any pval == 0 above, the correct interpretation is "pval < 1/n_permut".
        # To enable meaningful further analyses, it is often useful to set pvals==0 to a value closer to 1/n_permut.
        if pval == 0:
            pval = 0.99 / num_permut

    return pval, metric_diff


def bootstrap_curve(
        target: npt.NDArray[np.bool], 
        pred_probs: npt.NDArray[np.floating], 
        curve_fun: Callable[..., np.ndarray], 
        num_bootstraps: int, 
        num_samples: int
        ) -> np.ndarray:

    rng = RandomState.get_rng()

    N_predictions = len(target)

    yvals_bs = np.zeros((num_bootstraps, num_samples)) * np.nan

    stratify, N_pos, N_neg = decide_stratify(target)

    for bs_idx in range(num_bootstraps):
        
        if stratify:
            bs_idces_pos = rng.choice(np.flatnonzero(target == 1), N_pos)
            bs_idces_neg = rng.choice(np.flatnonzero(target == 0), N_neg)
            bs_idces = np.concatenate([bs_idces_pos, bs_idces_neg])
        else:
            bs_idces = rng.choice(range(N_predictions), N_predictions)

        if (target[bs_idces] == 0).sum() > 0 and (target[bs_idces] == 1).sum() > 0:
            yvals_bs[bs_idx, :] = curve_fun(target=target[bs_idces], pred_probs=pred_probs[bs_idces])

    return yvals_bs


def ci_nan_quantile(
        a: npt.NDArray, 
        q: float | npt.NDArray[np.floating], 
        axis: Optional[int] = None, 
        nan_fraction_allowed: float = 0.1
        ) -> float | npt.NDArray[np.floating]:
    assert np.sum(np.isinf(a[:])) == 0

    if axis is None:
        too_many_nan = np.sum(np.isnan(a[:])) > nan_fraction_allowed * len(a[:])
        return np.nan if too_many_nan else np.nanquantile(a, q, axis=None)
    
    else:
        too_many_nan = np.sum(np.isnan(a), axis=axis) > nan_fraction_allowed * a.shape[axis]
        quantile = np.ones_like(too_many_nan, dtype=np.float64)
        quantile[too_many_nan] = np.nan
        if axis == 0 and np.ndim(a) == 2:
            quantile[~too_many_nan] = np.nanquantile(a[:, ~too_many_nan], q, axis=axis)
        elif axis == 1 and np.ndim(a) == 2:
            quantile[~too_many_nan] = np.nanquantile(a[~too_many_nan, :], q, axis=axis)
        else:
            raise NotImplementedError
        
        return quantile


def nan_mean(
        a: npt.NDArray,
        axis: Optional[int] = None, 
        nan_fraction_allowed: float = 0.1
        ) -> float | npt.NDArray[np.floating]:
    assert np.sum(np.isinf(a[:])) == 0

    if axis is None:
        too_many_nan = np.sum(np.isnan(a[:])) > nan_fraction_allowed * len(a[:])
        return np.nan if too_many_nan else np.nanmean(a, axis=None) # type: ignore
    
    else:
        too_many_nan = np.sum(np.isnan(a), axis=axis) > nan_fraction_allowed * a.shape[axis]
        mean = np.ones_like(too_many_nan, dtype=np.float64)
        mean[too_many_nan] = np.nan
        if axis == 0 and np.ndim(a) == 2:
            mean[~too_many_nan] = np.nanmean(a[:, ~too_many_nan], axis=axis)
        elif axis == 1 and np.ndim(a) == 2:
            mean[~too_many_nan] = np.nanmean(a[~too_many_nan, :], axis=axis)
        else:
            raise NotImplementedError
        
        return mean
    

def hanley_var(auroc: float, y_true: pd.Series | np.ndarray):
    nx = np.sum(y_true == 1)
    ny = np.sum(y_true == 0)
    assert nx+ny == len(y_true)
    nxstar = nystar = len(y_true) / 2 - 1
    var = auroc * (1-auroc) * (1 + nxstar * (1-auroc)/(2-auroc) + nystar*auroc/(1+auroc))/(nx*ny)
    return var


def newcombe_auroc_ci(auroc_val: float, y_true: pd.Series | np.ndarray, ci_alpha: float):  # this wants a 'small' ci_alpha, i.e. 0.05 (and not 0.95)
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
