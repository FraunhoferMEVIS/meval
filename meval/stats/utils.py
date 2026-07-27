import numpy as np
import pandas as pd
from typing import Optional

from .._array_types import LabelArray, NumericArray, FloatArray


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