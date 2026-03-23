from itertools import combinations
from typing import Optional, Sequence, Union

import numpy as np

from .._array_types import FloatArray, LabelArray, MixedLabelArray

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None


def _as_binary(y_true: LabelArray) -> FloatArray:
    return (np.asarray(y_true) == 1).astype(float)


def fast_auc(
    y_true: LabelArray,
    y_score: FloatArray,
    sample_weight: Optional[FloatArray] = None,
) -> Union[float, str]:
    y_true_bin = _as_binary(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)

    desc_score_indices = np.argsort(y_score_arr, kind="mergesort")[::-1]
    y_score_arr = y_score_arr[desc_score_indices]
    y_true_bin = y_true_bin[desc_score_indices]

    if sample_weight is not None:
        sample_weight_arr = np.asarray(sample_weight, dtype=float)[desc_score_indices]
    else:
        sample_weight_arr = None

    distinct_value_indices = np.where(np.diff(y_score_arr))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_bin.size - 1]

    if sample_weight_arr is not None:
        tps = np.cumsum(y_true_bin * sample_weight_arr)[threshold_idxs]
        fps = np.cumsum((1 - y_true_bin) * sample_weight_arr)[threshold_idxs]
    else:
        tps = np.cumsum(y_true_bin)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan

    direction = 1
    dx = np.diff(fps)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            return "error"

    area = direction * np.trapezoid(tps, fps) / (tps[-1] * fps[-1])
    return float(area)


if numba is not None:

    @numba.njit
    def _trapezoid_area(x1: float, x2: float, y1: float, y2: float) -> float:
        dx = x2 - x1
        dy = y2 - y1
        return dx * y1 + dy * dx / 2.0


    @numba.njit
    def _fast_numba_auc_nonw(
        y_true: MixedLabelArray,
        y_score: FloatArray,
    ) -> float:
        y_true = (y_true == 1)

        prev_fps = 0.0
        prev_tps = 0.0
        last_counted_fps = 0.0
        last_counted_tps = 0.0
        auc = 0.0

        for i in range(len(y_true)):
            tps = prev_tps + y_true[i]
            fps = prev_fps + (1 - y_true[i])
            if i == len(y_true) - 1 or y_score[i + 1] != y_score[i]:
                auc += _trapezoid_area(last_counted_fps, fps, last_counted_tps, tps)
                last_counted_fps = fps
                last_counted_tps = tps
            prev_tps = tps
            prev_fps = fps

        return auc / (prev_tps * prev_fps)


    @numba.njit
    def _fast_numba_auc_w(
        y_true: MixedLabelArray,
        y_score: FloatArray,
        sample_weight: FloatArray,
    ) -> float:
        y_true = (y_true == 1)

        prev_fps = 0.0
        prev_tps = 0.0
        last_counted_fps = 0.0
        last_counted_tps = 0.0
        auc = 0.0

        for i in range(len(y_true)):
            weight = sample_weight[i]
            tps = prev_tps + y_true[i] * weight
            fps = prev_fps + (1 - y_true[i]) * weight
            if i == len(y_true) - 1 or y_score[i + 1] != y_score[i]:
                auc += _trapezoid_area(last_counted_fps, fps, last_counted_tps, tps)
                last_counted_fps = fps
                last_counted_tps = tps
            prev_tps = tps
            prev_fps = fps

        return auc / (prev_tps * prev_fps)

else:
    def _trapezoid_area(x1: float, x2: float, y1: float, y2: float) -> float:  # pragma: no cover
        raise RuntimeError("numba is not available")


    def _fast_numba_auc_nonw(  # pragma: no cover
        y_true: MixedLabelArray,
        y_score: FloatArray,
    ) -> float:
        raise RuntimeError("numba is not available")


    def _fast_numba_auc_w(  # pragma: no cover
        y_true: MixedLabelArray,
        y_score: FloatArray,
        sample_weight: FloatArray,
    ) -> float:
        raise RuntimeError("numba is not available")


def fast_numba_auc(
    y_true: LabelArray,
    y_score: FloatArray,
    sample_weight: Optional[FloatArray] = None,
) -> float:
    if numba is None:
        result = fast_auc(y_true=y_true, y_score=y_score, sample_weight=sample_weight)
        return np.nan if isinstance(result, str) else float(result)

    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)

    desc_score_indices = np.argsort(y_score_arr)[::-1]
    y_score_arr = y_score_arr[desc_score_indices]
    y_true_arr = y_true_arr[desc_score_indices]
    y_bin = (y_true_arr == 1)

    if sample_weight is None:
        pos_count = y_bin.sum()
        if pos_count == 0 or pos_count == len(y_bin):
            return np.nan
        return float(_fast_numba_auc_nonw(y_true=y_true_arr, y_score=y_score_arr))

    sample_weight_arr = np.asarray(sample_weight, dtype=float)[desc_score_indices]
    pos_weight = np.sum(sample_weight_arr * y_bin)
    neg_weight = np.sum(sample_weight_arr * (1 - y_bin))
    if pos_weight <= 0 or neg_weight <= 0:
        return np.nan
    return float(_fast_numba_auc_w(y_true=y_true_arr, y_score=y_score_arr, sample_weight=sample_weight_arr))


def fast_ovo_auc(
    y_true: LabelArray,
    y_score: FloatArray,
    labels: Optional[Sequence] = None,
) -> float:
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)

    classes = np.unique(y_true_arr) if labels is None else np.asarray(labels)
    n_classes = len(classes)

    if n_classes < 3:
        return np.nan

    auc_sum = 0.0
    n_valid = 0

    for i, j in combinations(range(n_classes), 2):
        ci, cj = classes[i], classes[j]
        mask = (y_true_arr == ci) | (y_true_arr == cj)

        if mask.sum() == 0:
            continue

        y_bin_i = (y_true_arr[mask] == ci).astype(float)
        y_bin_j = (y_true_arr[mask] == cj).astype(float)
        scores_i = y_score_arr[mask, i]
        scores_j = y_score_arr[mask, j]

        auc_i = fast_auc(y_bin_i, scores_i)
        auc_j = fast_auc(y_bin_j, scores_j)

        if isinstance(auc_i, str) or isinstance(auc_j, str):
            continue
        if np.isnan(auc_i) or np.isnan(auc_j):
            continue

        auc_sum += 0.5 * (float(auc_i) + float(auc_j))
        n_valid += 1

    return auc_sum / n_valid if n_valid > 0 else np.nan


def fast_ovo_auc_numba(
    y_true: LabelArray,
    y_score: FloatArray,
    labels: Optional[Sequence] = None,
) -> float:
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)

    classes = np.unique(y_true_arr) if labels is None else np.asarray(labels)
    n_classes = len(classes)

    if n_classes < 3:
        return np.nan

    auc_sum = 0.0
    n_valid = 0

    for i, j in combinations(range(n_classes), 2):
        ci, cj = classes[i], classes[j]
        mask = (y_true_arr == ci) | (y_true_arr == cj)

        if mask.sum() == 0:
            continue

        y_bin_i = (y_true_arr[mask] == ci).astype(float)
        y_bin_j = (y_true_arr[mask] == cj).astype(float)
        scores_i = y_score_arr[mask, i]
        scores_j = y_score_arr[mask, j]

        auc_i = fast_numba_auc(y_bin_i, scores_i)
        auc_j = fast_numba_auc(y_bin_j, scores_j)

        if np.isnan(auc_i) or np.isinf(auc_i) or np.isnan(auc_j) or np.isinf(auc_j):
            continue

        auc_sum += 0.5 * (float(auc_i) + float(auc_j))
        n_valid += 1

    return auc_sum / n_valid if n_valid > 0 else np.nan
