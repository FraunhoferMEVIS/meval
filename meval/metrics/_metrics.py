import warnings
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from typing import Optional

from ..config import settings
from ..stats import bootstrap_curve


def get_confusion_matrix(target: np.ndarray, preds: np.ndarray, threshold: Optional[float] = None) -> tuple[int, int, int, int]:

    if target.ndim == 1:
        target = target.reshape(1, -1)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)

    dim = (0, 1) if preds.ndim == 2 else (1, 2)
    
    assert issubclass(target.dtype.type, np.bool)

    if not issubclass(preds.dtype.type, np.bool):
        assert threshold is not None
        preds = (preds > threshold)

    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    tp = (true_pred * pos_pred).sum(axis=dim)
    fp = (false_pred * pos_pred).sum(axis=dim)

    tn = (true_pred * neg_pred).sum(axis=dim)
    fn = (false_pred * neg_pred).sum(axis=dim)

    return tp, fp, tn, fn


def _ratio(numerator, denominator):
    if denominator == 0:
        return np.nan
    else:
        return numerator / denominator


def accuracy(target: np.ndarray, preds: np.ndarray, threshold: Optional[float] = None) -> float:
    N = len(preds)
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TP + TN, N)


def recall(target: np.ndarray, preds: np.ndarray, threshold: Optional[float] = None) -> float:
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TP, TP + FN)

sensitivity = recall
TPR = recall

def precision(target: np.ndarray, preds: np.ndarray, threshold: Optional[float] = None) -> float:
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TP, TP + FP)

PPV = precision

def f1_score(target: np.ndarray, preds: np.ndarray, threshold: Optional[float] = None) -> float:
    # harmonic mean of precision and recall
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TP, TP + 0.5 * (FP+FN))


def specificity(target: np.ndarray, preds: np.ndarray, threshold: Optional[float] = None) -> float:
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TN, TN + FP)

TNR = specificity

def FPR(target: np.ndarray, preds: np.ndarray, threshold: Optional[float] = None) -> float:

    return 1 - specificity(target, preds, threshold=threshold)


def selection_rate(target: np.ndarray, preds: np.ndarray, threshold: Optional[float] = None):
    N = len(preds)
    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    return _ratio(TP + FP, N)


def check_bootstrap_curve_args(target, pred_probs):
    assert len(target) == len(pred_probs)
    assert not isinstance(target, pd.Series)
    assert not isinstance(pred_probs, pd.Series)


def bootstrap_roc_curve(
    target: npt.NDArray[np.bool],
    pred_probs: npt.NDArray[np.floating],
    num_bootstraps: int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    
    check_bootstrap_curve_args(target, pred_probs)
    
    N_predictions = len(target)

    if target.sum() == N_predictions or target.sum() == 0:
        return np.array(np.nan), np.array(np.nan), np.zeros((num_bootstraps, 1)) * np.nan

    else:

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        fpr = np.arange(0, 1+1e-7, 0.01)

        def _get_tpr(target, pred_probs):
            fpr_loc, tpr_loc, _ = roc_curve(target, pred_probs, drop_intermediate=True)
            return np.interp(fpr, fpr_loc, tpr_loc, left=np.nan, right=np.nan)  

        # point estimate  
        tpr = _get_tpr(target, pred_probs)

        # now for the bootstrapped versions
        tpr_bs = bootstrap_curve(target, pred_probs, _get_tpr, num_bootstraps, len(fpr))
        
        warnings.filterwarnings("default", category=UndefinedMetricWarning)

        return fpr, tpr, tpr_bs


def bootstrap_prg_curve(
        target: npt.NDArray[np.bool], 
        pred_probs: npt.NDArray[np.floating], 
        num_bootstraps: int
        ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    check_bootstrap_curve_args(target, pred_probs)

    N_predictions = len(target)

    if target.sum() == N_predictions or target.sum() == 0:
        return np.array(np.nan), np.array(np.nan), np.zeros((num_bootstraps, 1)) * np.nan

    else:

        recall_gains = np.arange(0, 1+1e-7, 0.01)

        def _get_precg(target, pred_probs):
            precision_gains_loc, recall_gains_loc = precision_recall_gain_curve(target, pred_probs, pos_label=1, drop_intermediate=True)
            return np.interp(recall_gains, recall_gains_loc[::-1], precision_gains_loc[::-1], left=np.nan, right=np.nan)

        # point estimate
        precision_gains = _get_precg(target, pred_probs)

        # bootstrapped samples
        precision_gains_bs = bootstrap_curve(target, pred_probs, _get_precg, num_bootstraps, len(recall_gains))

        return recall_gains, precision_gains, precision_gains_bs


# Theorem 1 in Boyd, Eng, Page; Area Under the Precision-Recall Curve: Point Estimates and Confidence Intervals
def interpolate_pr(
        rec_target: float, 
        recs: np.ndarray, 
        precs: np.ndarray):
    assert isinstance(recs, np.ndarray)
    assert isinstance(precs, np.ndarray)
    assert np.ndim(rec_target) == 0
    assert np.ndim(recs) == 1
    assert np.ndim(precs) == 1

    if rec_target in recs:
        return precs[recs == rec_target].mean()
    
    # We don't do extrapolation
    if (rec_target < recs[0]) and (rec_target < recs[-1]):
        return np.nan
    
    # We don't do extrapolation
    if (rec_target > recs[0]) and (rec_target > recs[-1]):
        return np.nan

    # we assume recs are sorted but don't assume the directionality
    if recs[0] < recs[-1]:
        idx_prev = np.nonzero(recs < rec_target)[0][-1]
        idx_next = np.nonzero(recs > rec_target)[0][0]
    else:
        idx_prev = np.nonzero(recs < rec_target)[0][0]
        idx_next = np.nonzero(recs > rec_target)[0][-1]

    rec_prev = recs[idx_prev]
    prec_prev = precs[idx_prev]
    rec_next = recs[idx_next]
    prec_next = precs[idx_next]

    # The formula below is undefined for prec_prev == 0.
    # I tried to derive the correct limiting expression for that case but got stuck.
    # This is a hack that should also work pretty well.
    if prec_prev == 0:
        prec_prev = min(0.0001, precs[precs>0].min())

    a = 1 + ((1 - prec_next) * rec_next) / (prec_next * (rec_next - rec_prev)) - ((1 - prec_prev) * rec_prev) / (prec_prev * (rec_next - rec_prev))
    b = ((1 - prec_prev) * rec_prev) / prec_prev - ((1 - prec_next) * rec_next * rec_prev) / (prec_next * (rec_next - rec_prev)) + ((1 - prec_prev) * rec_prev**2) / (prec_prev * (rec_next - rec_prev))

    prec_target = rec_target / (a * rec_target + b)

    assert np.isfinite(prec_target)

    return prec_target


def bootstrap_pr_curve(
        target: npt.NDArray[np.bool], 
        pred_probs: npt.NDArray[np.floating], 
        num_bootstraps: int
        ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    
    check_bootstrap_curve_args(target, pred_probs)
    N_predictions = len(target)

    if target.sum() == N_predictions or target.sum() == 0:
        return np.array(np.nan), np.array(np.nan), np.zeros((num_bootstraps, 1)) * np.nan

    else:

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        recall = np.arange(1e-7, 1+1e-7, 0.01)

        def _get_prec(target, pred_probs):
            precision_loc, recall_loc, _ = precision_recall_curve(target, pred_probs, drop_intermediate=True) # type: ignore
            
            # Drop last values since these are artificial ones, and reverse order to have increasing x=recall vals
            precision_loc = precision_loc[0:-1][::-1]
            recall_loc = recall_loc[0:-1][::-1]
            assert np.all(np.diff(recall_loc) >= 0)  # must be increasing for the interpolation below to work
            # The following line is NOT correct: linear interpolation in PR space is incorrect, cf. Kull and Flach, PR Analysis Done Right.
            # return np.interp(recall, recall_loc, precision_loc, left=np.nan, right=np.nan)
            # The correct approach (tm) would, of course, be to parallelize the PR interpolation...
            precision = []
            for rec_target_val in recall:
                precision.append(interpolate_pr(rec_target_val, recall_loc, precision_loc))

            return np.array(precision)

        # point estimate
        precision = _get_prec(target, pred_probs)

        # bootstrapped samples
        precision_bs = bootstrap_curve(target, pred_probs, _get_prec, num_bootstraps, len(recall))

        warnings.filterwarnings("default", category=UndefinedMetricWarning)

        return recall, precision, precision_bs


def auroc(target, preds) -> float:
    if sum(target) > 0 and sum(target == 0) > 0:
        # There are both positive and negative examples
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        r = roc_auc_score(target, preds)
        warnings.filterwarnings("default", category=UndefinedMetricWarning)
    else: 
        r = np.nan

    return r # type: ignore


def brier_scores(
        target: npt.NDArray[np.bool | np.integer], 
        pred_probs: npt.NDArray[np.floating]
        ) -> tuple[float, float, float, float]:
    # this only works for the binary case. multi-category extensions exist, though.
    assert len(np.unique(target) <= 2)
    assert issubclass(target.dtype.type, np.integer) or issubclass(target.dtype.type, np.bool)
    assert np.issubdtype(pred_probs.dtype, np.floating)
    assert np.all(target.shape == pred_probs.shape)
    assert np.all(0 <= pred_probs) and np.all(pred_probs <= 1)
    assert (target < 0).sum() == 0
    brier_score = _ratio(sum(np.square((target - pred_probs))), len(target))
    brier_score_pos = _ratio(sum(np.square((target[target > 0] - pred_probs[target > 0]))), len(target[target > 0]))
    brier_score_neg = _ratio(sum(np.square((target[target == 0] - pred_probs[target == 0]))), len(target[target == 0]))
    brier_score_bal = 0.5 * brier_score_pos + 0.5 * brier_score_neg
    return brier_score, brier_score_pos, brier_score_neg, brier_score_bal


def area_under_precision_recall_gain_score(
    y_true, y_score, *, pos_label=1, rec_gain_min: float = 0,
) -> float:

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan

    precision_gain, recall_gain = precision_recall_gain_curve(
        y_true, y_score, pos_label=pos_label
    )

    # integral should be computed between recgain =0 and =1
    assert 1 in recall_gain

    if rec_gain_min in recall_gain:

        msk = (recall_gain >= rec_gain_min) & (recall_gain <= 1)
        area_estimate = -np.trapezoid(y=precision_gain[msk], x=recall_gain[msk])

        assert not np.isnan(area_estimate)
        assert np.isfinite(area_estimate)

    else:
        # Try to interpolate.
        # Linear interpolation is meaningful in PRG space (not in PR space), cf. Kull and Flach.
        # BUT we can only interpolate if there is a threshold at which recall_gain is finite and <= rec_gain_min.
        # 
        # For rec_gain_min=0, the default, this corresponds to recall <= base rate.
        # This is the case if either the largest score is for a negative example 
        # (in that case we have a point recall = 0, precision = 0)
        # or, if the largest score is for a positive example, if
        # n_vals_at_that_score / n_pos <= base rate.
        # (I would have to think about the case when there are positive *and* negative examples at the highest score,
        # but it should not matter for the implementation.)

        finite_msk = np.isfinite(recall_gain) & np.isfinite(precision_gain)
        if np.sum((recall_gain < rec_gain_min) & finite_msk) > 0:
            y_intercept = np.interp(rec_gain_min, recall_gain[finite_msk], precision_gain[finite_msk])
            # rec_gain is sorted in REVERSE
            insert_idx = np.nonzero(recall_gain <= rec_gain_min)[0][0]
            recall_gain = np.insert(recall_gain, insert_idx, rec_gain_min)
            precision_gain = np.insert(precision_gain, insert_idx, y_intercept)

            msk = (recall_gain >= rec_gain_min) & (recall_gain <= 1)
            area_estimate = -np.trapezoid(y=precision_gain[msk], x=recall_gain[msk])       
            assert not np.isnan(area_estimate)
            assert np.isfinite(area_estimate)                         
        else:
            # OK, so technically, AUPRG is simply undefined in this case.
            # However, it is actually only the leftmost part oft he area that is unknown, and we can calculate
            # bounds on how much it might vary. If the possible variation is sufficiently small, we can still return
            # a reasonable estimate.

            # Determine the leftmost / lowest available (finite) datapoint for rec_gain + the associated prec_gain.
            # This is determined by the highest threshold that still yields at least one true positive prediction.
            # (For higher thresholds, with no true positive predictions, recall is 0 and recall_gain is -infty.)
            proportion_of_positives = (y_true == pos_label).sum() / len(y_true)
            highest_pos_thresh = np.max(y_score[y_true == pos_label])
            pos_pred = y_score >= highest_pos_thresh
            tp = np.sum((y_true == pos_label) & pos_pred)
            fp = np.sum(~(y_true == pos_label) & pos_pred)
            fn = np.sum((y_true == pos_label) & ~pos_pred)
            rec_lowest = tp / (tp + fn)
            rec_gain_lowest = (rec_lowest - proportion_of_positives) / (1 - proportion_of_positives) / rec_lowest
            prec_lowest = tp / (tp + fp)
            prec_gain_lowest = (prec_lowest - proportion_of_positives) / (1 - proportion_of_positives) / prec_lowest

            prec_ub = 1
            prec_gain_ub = (prec_ub - proportion_of_positives) / (1 - proportion_of_positives) / prec_ub

            # formula (1) from https://pmc.ncbi.nlm.nih.gov/articles/PMC3858955/
            # characterizes the achievable points in PR space and gives a lower bound for prec at each rec value
            # first translate rec_gain_min into rec_min
            rec_min = proportion_of_positives / (1 - rec_gain_min  + rec_gain_min * proportion_of_positives)
            prec_lb = proportion_of_positives * rec_min / (1 - proportion_of_positives + proportion_of_positives * rec_min)
            prec_gain_lb = (prec_lb - proportion_of_positives) / (1 - proportion_of_positives) / prec_lb

            # interpolate towards (rec_min, prec_ub=1) to get upper bound on missing area
            max_missing_auprg_area = (prec_gain_lowest + prec_gain_ub) / 2 * (rec_gain_lowest - rec_gain_min)

            # interpolate towards (rec_min, prec_lb) in PR space to get lower bound on missing area
            min_missing_auprg_area = (prec_gain_lowest + prec_gain_lb) / 2 * (rec_gain_lowest - rec_gain_min)

            assert min_missing_auprg_area <= max_missing_auprg_area

            diff_area = max_missing_auprg_area - min_missing_auprg_area

            msk = (recall_gain >= rec_gain_min) & (recall_gain <= 1)
            area_estimate = -np.trapezoid(y=precision_gain[msk], x=recall_gain[msk]) + min_missing_auprg_area + diff_area / 2

            assert not np.isnan(area_estimate)
            assert np.isfinite(area_estimate)

            # yes these are arbitrary thresholds
            if np.abs(diff_area) > 0.05 and np.abs(diff_area / area_estimate / 2) > 0.05:
                if settings.debug:
                    print(f'Returning AUPRG nan @ uncertainty +/- {diff_area/2}, estimate {area_estimate}')
                area_estimate = np.nan
            else:
                if settings.debug:
                    print(f'Returning AUPRG value @ uncertainty +/- {diff_area/2}, estimate {area_estimate}')

    assert not np.isinf(area_estimate)

    return area_estimate # type: ignore


def precision_recall_gain(precisions, recalls, proportion_of_positives):
    """
    Converts precision and recall into precision-gain and recall-gain.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        prec_gain = (precisions - proportion_of_positives) / (
            (1 - proportion_of_positives) * precisions
        )
        rec_gain = (recalls - proportion_of_positives) / (
            (1 - proportion_of_positives) * recalls
        )

    return prec_gain, rec_gain


def precision_recall_gain_curve(y_true, probas_pred, pos_label=1, drop_intermediate=False):

    precision, recall, _ = precision_recall_curve(y_true, probas_pred, pos_label=pos_label, drop_intermediate=drop_intermediate) # type: ignore

    proportion_of_positives = (y_true == pos_label).sum() / len(y_true)

    precision_gains, recall_gains = precision_recall_gain(
        precisions=precision[:-1],  # the last value is an artificial recall=0, precision=1 point; we do not want that
        recalls=recall[:-1],
        proportion_of_positives=proportion_of_positives,
    )

    return precision_gains, recall_gains


def precision_gain(target: np.ndarray, preds: np.ndarray, threshold: Optional[float] = None) -> float:

    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    n_pos = TP + FN
    n_neg = FP + TN
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_gain = 1.0 - (n_pos / n_neg) * (FP / TP)
    if TN + FN == 0:
        prec_gain = 0
    return prec_gain


def recall_gain(target: np.ndarray, preds: np.ndarray, threshold: Optional[float] = None) -> float:

    TP, FP, TN, FN = get_confusion_matrix(target, preds, threshold=threshold)

    n_pos = TP + FN
    n_neg = FP + TN
    with np.errstate(divide="ignore", invalid="ignore"):
        recg = 1.0 - (n_pos / n_neg) * (FN / TP)
    if TN + FN == 0:
        recg = 1
    return recg