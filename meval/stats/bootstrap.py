import numpy as np
import pandas as pd
from typing import Optional
from collections.abc import Callable

from .utils import decide_stratify
from ..config import settings
from ..metrics.ComparisonMetric import ComparisonMetric
from ..group_filter import GroupFilter
from .._array_types import MaskLike, FloatArray, LabelArray


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