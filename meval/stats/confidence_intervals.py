
import numpy as np
import scipy
import pandas as pd

from .bootstrap import bootstrap_metric
from .utils import ci_nan_quantile
from .._array_types import LabelArray


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


def bootstrap_ci(df, metric, group_filter, num_bootstrap, ci_alpha):
    metric_bs = bootstrap_metric(df, metric, group_filter, num_bootstrap=num_bootstrap)
    lower = ci_nan_quantile(metric_bs, (1 - ci_alpha) / 2)
    #med = ci_nan_quantile(metric_bs, 0.5)
    upper = ci_nan_quantile(metric_bs, ci_alpha + (1 - ci_alpha) / 2)
    return lower, upper


def _hanley_var(auroc: float, y_true: pd.Series | LabelArray):
    nx = np.sum(y_true == 1)
    ny = np.sum(y_true == 0)
    assert nx+ny == len(y_true)
    nxstar = nystar = len(y_true) / 2 - 1
    var = auroc * (1-auroc) * (1 + nxstar * (1-auroc)/(2-auroc) + nystar*auroc/(1+auroc))/(nx*ny)
    return var


def newcombe_auroc_ci(auroc_val: float, y_true: pd.Series | LabelArray, ci_alpha: float):  # this wants a 'small' ci_alpha, i.e. 0.05 (and not 0.95)
    # See https://journals.sagepub.com/doi/10.1177/0962280215602040
    if np.isnan(auroc_val):
        return [np.nan, np.nan]
    
    assert isinstance(y_true, pd.Series) or isinstance(y_true, np.ndarray)
    z = scipy.stats.norm(loc=0, scale=1).ppf(1-ci_alpha/2)
    if auroc_val - 1e-4 > 0.0:
        lb_result = scipy.optimize.root_scalar(
            lambda auroc_lb: np.abs(auroc_lb - auroc_val) - z * np.sqrt(_hanley_var(auroc_lb, y_true)), 
            bracket=[0, auroc_val-1e-4], xtol=1e-3)
        assert lb_result.converged
        lb = lb_result.root
    else:
        lb = 0.0

    if auroc_val + 1e-4 < 1.0:
        ub_result = scipy.optimize.root_scalar(
            lambda auroc_ub: np.abs(auroc_ub - auroc_val) - z * np.sqrt(_hanley_var(auroc_ub, y_true)), 
            bracket=[auroc_val+1e-4, 1.0], xtol=1e-3)
        assert ub_result.converged
        ub = ub_result.root
    else:
        ub = 1.0

    return [lb, ub]