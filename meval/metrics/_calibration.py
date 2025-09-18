from typing import List, Tuple, TypeVar, Sequence
from warnings import warn

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.signal
from betacal import BetaCalibration
from scipy.interpolate import PchipInterpolator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

Data = List[Tuple[float, float]]  # List of (predicted_probability, true_label).
Bins = List[float]  # List of bin boundaries, excluding 0.0, but including 1.0.
BinnedData = List[Data]  # binned_data[i] contains the data in bin i.
T = TypeVar('T')

eps = 1e-6

VERBOSE = False

def get_equal_mass_bins(probs, num_bins, min_vals_per_bin=None, return_counts=False):
    """Get bins that contain approximately an equal number of data points."""
    # The original implementation in https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py
    # did not seem reliable to me - I encountered problems with highly repetitive values. Sometimes the method would
    # yield highly imbalanced bins, with some being empty and others containing many values.
    # This is my attempt at doing better.

    # This is basic quantile binning, a standard practical approach towards equal mass / equal frequency binning.
    # Notice, however, that it can "break" in the case of many identical values, because then successive quantiles can
    # coincide at the same value. We will try to deal with that further below.
    # (See https://blogs.sas.com/content/iml/2014/11/05/binning-quantiles-rounded-data.html for a simple example of this
    # issue, which in the present context will typically arise when using tree models.)
    # The "+1e-7" is because due to floating point inaccuracies, otherwise sometimes a final "1.0" bin boundary is
    # created and sometimes not. Here, we apparently want it. (See above.)
    # I did not investigate in detail, but the "median_unbiased" method seemed to "work better" with a lot of repetitive
    # values. (It's also the recommended method as per the documentation.)
    bin_arr = np.quantile(probs, np.arange(1/num_bins, 1+1e-7, 1/num_bins), method="median_unbiased")

    bins, bin_num_vals_counts = refine_quantile_bins(probs, bin_arr, num_bins, min_vals_per_bin)

    if return_counts:
        return bins, bin_num_vals_counts
    else:
        return bins    


def refine_quantile_bins(probs, bin_arr, num_bins, min_vals_per_bin):

    assert len(bin_arr) == num_bins

    bin_idces = np.searchsorted(bin_arr, probs)
    assert bin_idces.max() <= (num_bins - 1)
    assert bin_idces.min() >= 0

    _, bin_num_vals_counts = np.unique(bin_idces, return_counts=True)
    assert bin_num_vals_counts.sum() == len(probs)

    # eliminate duplicate bins, which can arise because of highly repetitive values covering multiple quantile values
    bin_arr = np.unique(bin_arr)

    if not len(bin_arr) == num_bins or \
            (min_vals_per_bin is not None and (bin_num_vals_counts < min_vals_per_bin).any()):
        # Because of some highly repetitive values, some quantiles coincide at the same values, or some bins contain
        # very few (or no) values. Attempt to recover.

        num_unique_probs = len(np.unique(probs))

        # find how many different unique values are in the current bins
        bin_idces = np.searchsorted(bin_arr, probs)
        _, bin_num_vals_counts = np.unique(bin_idces, return_counts=True)

        assert bin_num_vals_counts.sum() == len(probs)

        # Try to split/merge bins until all problems are sorted out: exactly the desired amount of bins, and all bins
        # of sufficient size (if min_vals_per_bin is not None).
        # The following loop could be handled a lot more efficiently. We should not go there too often anyway, however.
        while len(bin_arr) < num_bins or \
                (min_vals_per_bin is not None and (bin_num_vals_counts < min_vals_per_bin).any()):

            # Find unique values contained in each of the current bins
            bin_unique_vals = []
            for bin_idx in range(len(bin_arr)):
                bin_unique_vals.append(np.unique(probs[bin_idces == bin_idx]))
            bin_unique_vals_count = [len(unique_vals) for unique_vals in bin_unique_vals]
            assert len(bin_unique_vals_count) == len(bin_num_vals_counts)
            assert sum(bin_unique_vals_count) == num_unique_probs

            # Of the bins with at least two unique values, break up one of the largest ones, such that each of the new
            # bins will have at least min_vals_per_bin vals (if that is not None).
            multi_value_bin_idces = [idx for idx in range(len(bin_arr)) if bin_unique_vals_count[idx] >= 2]
            multi_value_bin_sizes = [bin_num_vals_counts[idx] for idx in multi_value_bin_idces]

            if min_vals_per_bin is not None and not np.any([size >= 2*min_vals_per_bin for size in multi_value_bin_sizes]):
                # It's impossible to split anything while also getting sufficiently large bins.
                raise ValueError

            # Sort by current bin size (we'll start trying to split from the largest one).
            multi_value_bin_size_and_idces_sorted = sorted(zip(multi_value_bin_sizes, multi_value_bin_idces),
                                                           key=lambda pair: pair[0], reverse=True)
            candidate_bins_for_splitting = [(bin_size, idx) for (bin_size, idx) in multi_value_bin_size_and_idces_sorted
                                            if min_vals_per_bin is None or bin_size >= 2 * min_vals_per_bin]

            found_one = False
            for bin_size, idx in candidate_bins_for_splitting:
                # We have a candidate bin to be split up.
                # How many unique values are contained in this bin?
                vals = probs[bin_idces == idx]
                local_bin_unique_vals, local_bin_unique_vals_counts = np.unique(vals, return_counts=True)
                assert local_bin_unique_vals_counts.sum() == bin_size

                # split this bin into two roughly equally sized chunks
                counts_cumsum = local_bin_unique_vals_counts.cumsum()
                optimal_splitting_idx = np.argmin(np.abs(counts_cumsum - bin_size / 2))

                if min_vals_per_bin is not None:
                    # Will the two split parts both have size > min_vals_per_bin?
                    if counts_cumsum[optimal_splitting_idx] < min_vals_per_bin or \
                            (bin_size - counts_cumsum[optimal_splitting_idx]) < min_vals_per_bin:
                        # one of the two new splits would be smaller than desired; try the next candidate for splitting
                        continue

                # Split the bin, update bin array
                bin_arr = np.delete(bin_arr, idx)
                bin_arr = np.insert(bin_arr, idx, [local_bin_unique_vals[optimal_splitting_idx],
                                                   local_bin_unique_vals[-1]])
                found_one = True
                break

            if not found_one:
                raise ValueError

            # Housekeeping
            # find how many unique values are in the remaining bins
            bin_idces = np.searchsorted(bin_arr, probs)
            _, bin_num_vals_counts = np.unique(bin_idces, return_counts=True)

            if len(bin_arr) > num_bins:
                # We came here because one bin was too small (but non-empty), not because of empty bins.
                # Now we have one bin too much. Merge two neighboring bins to eliminate one of the too-small ones.
                bin_idx_to_be_merged = np.argmin(bin_num_vals_counts)
                if 0 < bin_idx_to_be_merged < len(bin_arr) - 1:
                    # there are two neighboring bins, merge with the smaller of the two
                    if bin_num_vals_counts[bin_idx_to_be_merged - 1] < bin_num_vals_counts[bin_idx_to_be_merged + 1]:
                        bin_idx_to_be_merged = bin_idx_to_be_merged - 1
                elif bin_idx_to_be_merged == len(bin_arr) - 1:
                    bin_idx_to_be_merged = len(bin_arr) - 2
                bin_arr = np.delete(bin_arr, bin_idx_to_be_merged)

                # Re-do previous housekeeping
                bin_idces = np.searchsorted(bin_arr, probs)
                _, bin_num_vals_counts = np.unique(bin_idces, return_counts=True)

            assert bin_num_vals_counts.sum() == len(probs)
            assert len(bin_arr) <= num_bins

    bins: Bins = [bin_arr[idx] for idx in range(len(bin_arr))]

    return bins, bin_num_vals_counts


def get_discrete_bins(data: Sequence[float]) -> Bins:
    # Fully copied from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT license)
    sorted_values = sorted(np.unique(data))
    bins = []
    for i in range(len(sorted_values) - 1):
        mid = (sorted_values[i] + sorted_values[i+1]) / 2.0
        bins.append(mid)
    bins.append(1.0)
    return bins


def difference_mean(data: Data) -> float:
    """Returns average pred_prob - average label."""
    # Fully copied from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT license)
    data = np.array(data)
    ave_pred_prob = np.mean(data[:, 0])
    ave_label = np.mean(data[:, 1])
    return ave_pred_prob - ave_label


def get_bin_probs(binned_data: BinnedData) -> List[float]:
    # Fully copied from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT license)
    bin_sizes = list(map(len, binned_data))
    num_data = sum(bin_sizes)
    bin_probs = list(map(lambda b: b * 1.0 / num_data, bin_sizes))
    assert(abs(sum(bin_probs) - 1.0) < eps)
    return list(bin_probs)


def bin(data: Data, bins: Bins):
    # Fully copied from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT license)
    # bin boundaries are part of the _left_ bin.
    prob_label = np.array(data)
    bin_indices = np.searchsorted(bins, prob_label[:, 0])
    bin_sort_indices = np.argsort(bin_indices)
    sorted_bins = bin_indices[bin_sort_indices]
    splits = np.searchsorted(sorted_bins, list(range(1, len(bins))))
    binned_data = np.split(prob_label[bin_sort_indices], splits)
    return binned_data


def unbiased_l2_ce(binned_data: BinnedData, abort_if_not_monotonic=False) -> float:
    # Calibration error RMSE
    # The actual computation is copied from
    # https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT license)
    # I added the abort_if_not_monotonic feature, which is used in bin sweeps. Also changed the behavior if
    # len(data) < 2 to now raise an error instead of silently returning 0.
    def bin_error(data: Data):
        if len(data) < 2:
            #return 0.0
            raise ValueError('Too few values in bin, use fewer bins or get more data.')
        biased_estimate = abs(difference_mean(data)) ** 2
        label_values = list(map(lambda x: x[1], data))
        mean_label = np.mean(label_values)
        variance = mean_label * (1.0 - mean_label) / (len(data) - 1.0)
        return biased_estimate - variance

    if abort_if_not_monotonic:
        last_incidence = 0.0
        for bin_data in binned_data:
            if np.shape(bin_data)[0] == 0:
                pass
            bin_incidence = np.mean(bin_data[:, 1])
            if bin_incidence < last_incidence:
                raise ValueError("Bin incidence are non-monotonic")
            else:
                last_incidence = bin_incidence

    bin_probs = get_bin_probs(binned_data)
    bin_errors = list(map(bin_error, binned_data))

    return max(np.dot(bin_probs, bin_errors), 0.0) ** 0.5


def get_unbiased_calibration_rmse(labels, probs, num_bins="sweep", binning_scheme=get_equal_mass_bins):
    # Partially copied from https://github.com/p-lambda/verified_calibration/blob/master/calibration/utils.py (MIT license)
    # Bin sweep functionality added by me.
    assert probs.shape == labels.shape
    assert len(probs.shape) == 1
    if len(probs) < 10:
        return np.nan
    data = list(zip(probs, labels))
    if num_bins == 'sweep':
        assert not binning_scheme == get_discrete_bins
        # having less than ~10 samples per bin doesn't make a lot of sense, nor does having more than 100 bins
        upper_bound = min(min(round(len(probs) / 10), len(np.unique(probs))), 100)
        curr_num_bins = min(10, upper_bound)
        lower_bound = 1
        if VERBOSE:
            print('Starting bin count sweep (in Kumar)...')
        while not lower_bound == upper_bound == curr_num_bins:
            if VERBOSE:
                print(f'nbin={curr_num_bins}')
            try:
                bins = binning_scheme(probs, num_bins=curr_num_bins, min_vals_per_bin=10)
                err = unbiased_l2_ce(bin(data, bins), abort_if_not_monotonic=True)
            except ValueError:
                upper_bound = curr_num_bins - 1
                curr_num_bins = round(lower_bound + (upper_bound - lower_bound) / 2)
            else:
                lower_bound = curr_num_bins
                curr_num_bins = min(curr_num_bins * 2, round(lower_bound + (upper_bound - lower_bound) / 2))
            curr_num_bins = min(upper_bound, max(lower_bound + 1, curr_num_bins))

        if curr_num_bins == 1:
            bins = binning_scheme(probs, num_bins=curr_num_bins)
            err = unbiased_l2_ce(bin(data, bins))
        if VERBOSE:
            print(f'Final nbin={curr_num_bins}')
        return err
    else:
        if binning_scheme == get_discrete_bins:
            assert (num_bins is None)
            bins = binning_scheme(probs)
        else:
            bins = binning_scheme(probs, num_bins=num_bins)
        return unbiased_l2_ce(bin(data, bins))


class BetaCalibratedClassifier(RegressorMixin, BaseEstimator):

    def __init__(self, base_clf, beta_params="abm"):
        self.base_clf = base_clf
        self.beta_model = BetaCalibration(parameters=beta_params)

    def fit(self, X_cal, y_cal):
        y_preds = self.base_clf.predict_proba(X_cal)
        # this is an N x 2 matrix with the probabilities of the two classes
        assert (sum(abs(y_preds.sum(axis=1) - 1) < 1e-7) == len(y_preds))
        # reduce it to just the likelihood of the "1" class, as usual
        scores = y_preds[:, 1]
        self.beta_model.fit(scores, y_cal)

    def predict_proba(self, X_test):
        y_preds = self.base_clf.predict_proba(X_test)
        # this is an N x 2 matrix with the probabilities of the two classes
        assert (sum(abs(y_preds.sum(axis=1) - 1) < 1e-7) == len(y_preds))
        # reduce it to just the likelihood of the "1" class, as usual
        scores = y_preds[:, 1]
        return self.beta_model.predict(scores)

    def __repr__(self):
        return "Beta calibrated model with \n " + self.beta_model.calibrator_.lr_.__repr__() + \
            "\n and base model \n" + self.base_clf.__repr__()


def smoothen_isotonic_probs(isotonic_probs, pred_probs, method='pchip'):
    isotonic_probs_unique, isotonic_probs_counts = np.unique(isotonic_probs, return_counts=True)
    pred_prob_points = np.zeros_like(isotonic_probs_unique)
    for i, isotonic_prob in enumerate(isotonic_probs_unique):
        pred_prob_points[i] = np.mean(pred_probs[isotonic_probs == isotonic_prob])

    if method == 'pchip':
        # The first paper referenced above also uses pchip interpolation (which is monotonicity-preserving).
        # In my experiments, I observe some bias around the boundaries, not sure how to fix that.
        # In the second paper they handle the boundaries in some special way; maybe have a look there?
        isotonic_probs_smoothed = PchipInterpolator(pred_prob_points, isotonic_probs_unique, extrapolate=False)(pred_probs)
        # in the two extrapolation regions, set values equal to the nearest known value
        isotonic_probs_smoothed[np.isnan(isotonic_probs_smoothed) & (pred_probs >= pred_prob_points[-1])] = \
            isotonic_probs_unique[-1]
        isotonic_probs_smoothed[np.isnan(isotonic_probs_smoothed) & (pred_probs <= pred_prob_points[0])] = \
            isotonic_probs_unique[0]
        # apparently, probably for numeric reasons (?), PchipInterpolator sometimes returns values slightly greater
        # than 1, which then causes problems downstream
        isotonic_probs_smoothed = np.minimum(isotonic_probs_smoothed, 1)
        # haven't observed the same problem with 0, but just to be sure
        isotonic_probs_smoothed = np.maximum(isotonic_probs_smoothed, 0)
    elif method == 'parzen':
        # convolve with a Parzen window for smoothing, as is typically done, e.g., in kernel density estimation
        nsample = 200
        pred_probs_reg = np.linspace(0, 1, nsample)
        # returns NaN outside the range of pred_probs
        iso_probs_reg = scipy.interpolate.interp1d(pred_probs, isotonic_probs)(pred_probs_reg)
        # kernel width in samples: median bin width
        bin_widths = isotonic_probs_counts[isotonic_probs_counts > 1]
        kwidth = int(np.minimum(1.5*np.median(bin_widths), round(0.2*nsample)))
        parzenwin = scipy.signal.windows.parzen(kwidth)
        # normalize to area 1
        parzenwin = parzenwin / np.sum(parzenwin)
        # extend iso_probs_reg on either side for the convolution
        ext_len = int(kwidth/2)
        iso_probs_reg_ext = np.concatenate((iso_probs_reg[0] * np.ones((ext_len,)),
                                            iso_probs_reg,
                                            iso_probs_reg[-1] * np.ones((ext_len,))))

        isotonic_probs_reg_smoothed_ext = scipy.signal.convolve(iso_probs_reg_ext, parzenwin, mode='same')
        isotonic_probs_reg_smoothed = isotonic_probs_reg_smoothed_ext[ext_len:-ext_len]
        assert(isotonic_probs_reg_smoothed.shape == iso_probs_reg.shape)
        isotonic_probs_smoothed = scipy.interpolate.interp1d(pred_probs_reg, isotonic_probs_reg_smoothed)(pred_probs)
    else:
        raise NotImplementedError

    assert np.all(isotonic_probs_smoothed <= 1)

    return isotonic_probs_smoothed


def smoothed_isotonic_calibration(target, pred_probs, method='pchip', n_boot=0, rng=None):
    # Smooth the isotonic regression (preserving monotonicity) to get rid of the constant sections.
    # Inspired by these papers:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3248752/
    # https://arxiv.org/pdf/1701.05964.pdf
    # Also see the discussion here: https://github.com/scikit-learn/scikit-learn/pull/21454
    if n_boot > 0:
        isotonic_probs, isotonic_probs_boots = isotonic_calibration(target, pred_probs, n_boot, rng=rng)
        isotonic_probs_smoothed = smoothen_isotonic_probs(isotonic_probs, pred_probs, method)

        isotonic_probs_smoothed_boots = np.zeros_like(isotonic_probs_boots)
        for ii in range(0, n_boot):
            isotonic_probs_smoothed_boots[:, ii] = smoothen_isotonic_probs(isotonic_probs_boots[:, ii], pred_probs,
                                                                           method)
            assert np.all(isotonic_probs_smoothed_boots[:, ii] <= 1)

        return isotonic_probs_smoothed, isotonic_probs_smoothed_boots
    else:
        isotonic_probs = isotonic_calibration(target, pred_probs, rng=rng)
        isotonic_probs_smoothed = smoothen_isotonic_probs(isotonic_probs, pred_probs, method)

        return isotonic_probs_smoothed


def score_decompose(target, pred_probs, scoring_function, calib_probs=None):
    # See Dimitriadis, Gneiting, Jordan (2021), https://www.pnas.org/doi/full/10.1073/pnas.2016191118
    # This requires isotonic regression without any modifications / "beautifications" - otherwise, the theoretical
    # guarantees don't hold anymore (and some of the assertions below may fail).
    target = target.squeeze()
    score = lambda probs: scoring_function(target, probs)
    pred_probs = pred_probs.squeeze()

    if calib_probs is None:
        calib_probs = get_calib_probs(target, pred_probs, 'isotonic')

    base_probs = np.ones_like(target) * np.mean(target)
    # miscalibration
    mcb = score(pred_probs) - score(calib_probs)
    # discrimination
    dsc = score(base_probs) - score(calib_probs)
    # uncertainty
    unc = score(base_probs)
    # the score should decompose as SCORE = MCB - DSC + UNC
    total_score = score(pred_probs)
    assert np.abs((total_score - mcb + dsc - unc) / total_score) < 1e-5
    assert total_score >= 0
    if mcb < 0:
        warn(f'MCB is {mcb:.5f}, normally should not be < 0.')
    if dsc < 0:
        warn(f'DSC is {dsc:.5f}, normally should not be < 0.')
    assert unc >= 0

    return total_score, mcb, dsc, unc


def calibration_refinement_error(target, pred_probs, num_bins=15, adaptive=True, abort_if_not_monotonic=False):
    # adapted from on https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
    # This returns a) a calibration error / loss estimate, and b) a refinement loss estimate,
    # following the proper scoring rule decomposition presented in, e.g.,
    # Kull and Flach (2015), "Novel Decompositions of Proper Scoring Rules for Classification: Score Adjustment as Precursor to Calibration."
    # Either adaptive or static binning can be used.
    # The first term is exactly the typical calibration error. With static binning, this is called ECE; with adaptive binning ACE.
    # It is, however, NOT identical to the calibration loss term in the BS decomposition, because that uses a squared distance.
    # It is apparently also possible to do these things totally bin-less?! See above.
    
    # implemented and tested for the binary case. might also work for multi-class, not sure
    assert len(np.unique(target) <= 2)
    assert issubclass(target.dtype.type, np.integer)
    assert np.issubdtype(pred_probs.dtype, np.floating)
    assert np.all(target.shape == pred_probs.shape)
    assert np.all(0 <= pred_probs) and np.all(pred_probs <= 1)

    if num_bins == 'sweep':
        # find the number of bins automatically by searching for the maximum number of bins for which bin incidences
        # are still monotonic, starting from a default of 15.
        # Inspired by this paper (but better search procedure here): https://home.cs.colorado.edu/~mozer/Research/Selected%20Publications/reprints/Roelofsetal2022.pdf
        curr_num_bins = 15
        upper_bound = round(len(target)/2)
        lower_bound = 1
        print('Starting bin count sweep...')
        while not lower_bound == upper_bound == curr_num_bins:
            print(f'nbin={curr_num_bins}')
            try:
                ece, ref_err = calibration_refinement_error(target, pred_probs, num_bins=curr_num_bins,
                                                            adaptive=adaptive, abort_if_not_monotonic=True)
            except ValueError:
                upper_bound = curr_num_bins - 1
                curr_num_bins = round(lower_bound + (upper_bound - lower_bound)/2)
            else:
                lower_bound = curr_num_bins
                curr_num_bins = min(curr_num_bins * 2, round(lower_bound + (upper_bound - lower_bound)/2))
            curr_num_bins = min(upper_bound, max(lower_bound+1, curr_num_bins))

        if curr_num_bins == 1:
            ece, ref_err = calibration_refinement_error(target, pred_probs, num_bins=curr_num_bins, adaptive=adaptive)
        print(f'Final nbin={curr_num_bins}')
        return ece, ref_err

    else:

        b = np.linspace(start=0, stop=1.0, num=num_bins)
        if adaptive:
            # compute adaptive calibration error, ACE
            # See Nixon, M. Dusenberry et al., Measuring Calibration in Deep Learning, 2020.
            b = np.quantile(pred_probs, b)
            b = np.unique(b)
            num_bins = len(b)
        else:
            # compute expected calibration error, ECE
            pass

        bins = np.digitize(pred_probs, bins=b, right=True)
        incidences = np.zeros((num_bins, ))
        mean_abs_calib_error = 0
        refinement_error = 0
        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                mean_pred = np.mean(pred_probs[mask])
                incidences[b] = np.mean(target[mask])
                if b > 0 and incidences[b] < incidences[b-1] and abort_if_not_monotonic:
                    raise ValueError
                mean_abs_calib_error += np.sum(mask) * np.abs(mean_pred - incidences[b])
                refinement_error += np.sum(mask) * incidences[b] * (1 - incidences[b])

                # original version, using accuracy as the y axis
                # calib_error += np.abs(np.sum(correct[mask] - prob_y[mask]))
                # my version, using observed relative frequency as the y axis
                # not sure if this coincides with any standard metric?!
                #mean_abs_calib_error += np.abs(np.sum(target[mask] - pred_probs[mask]))

        return mean_abs_calib_error / pred_probs.shape[0], refinement_error / pred_probs.shape[0]
    

def ece(target, pred_probs, method, ci_alpha=None, n_samples=100, num_bins=15):
    if method == 'static':
        ece = calibration_refinement_error(target, pred_probs, num_bins=num_bins, adaptive=False)[0]
        return ece
    elif method == 'adaptive':
        ece = calibration_refinement_error(target, pred_probs, num_bins=num_bins, adaptive=True)[0]
        return ece
    if method == 'IR':
        probs = isotonic_calibration(target, pred_probs)
        ece = np.mean(np.abs(pred_probs - probs))
        return ece
    elif method == 'loess':
        probs, probs_samples = loess_calibration(target, pred_probs, n_samples=n_samples)
        ece_arr = np.nanmean(np.abs(pred_probs[:, None] - probs_samples), axis=0)
        ece = np.mean(np.abs(pred_probs - probs))
        ci_lower, ci_upper = np.nanquantile(ece_arr, q=[ci_alpha, 1-ci_alpha])
        return ece, ci_lower, ci_upper
    else:
        raise NotImplementedError


def loess_calibration(target, pred_probs, n_bootstrap_samples=None, xvals=None):

    assert isinstance(target, pd.Series)

    # it: The number of residual-based reweightings to perform.
    # frac: Between 0 and 1. The fraction of the data used when estimating each y-value.
    # delta: Distance within which to use linear-interpolation instead of weighted regression.
    if xvals is None:
        delta = 0.005
    else:
        delta = 0

    # Austin and Steyerberg (2013) say frac=0.75 (="span" in their terminology) and it=0 are good for calibration
    # analyses. https://onlinelibrary.wiley.com/doi/full/10.1002/sim.5941
    # However, this frac=0.75 smoothes very strongly and is basically incapable of returning "sharp" calibration curves.
    # Hence, let me try to choose an appropriate value for frac based on the sample size.
    # (frac: fraction of samples that is taken into account for regressing at a given x value, inversely weighted by
    # distance.)
    # I want the number of datapoint taken into account to alway be ~250.
    frac = max(0.3, min(1.0, 250/len(target)))
    calib_probs = sm_lowess(target, pred_probs, frac=frac, it=0, return_sorted=False, delta=delta, xvals=xvals)

    #plt.figure()
    #plt.scatter(pred_probs, calib_probs)

    if n_bootstrap_samples is not None:
        # Reference for bootstrapping CIs for loess-based calibration:
        # https://onlinelibrary.wiley.com/doi/10.1002/sim.6167
        N_predictions = len(target)

        if xvals is None:
            calib_probs_samples = np.zeros((N_predictions, n_bootstrap_samples))
        else:
            calib_probs_samples = np.zeros((len(xvals), n_bootstrap_samples))

        for idx in range(n_bootstrap_samples):
            bs_idces = np.random.choice(target.index, N_predictions)
            target_bs = target[bs_idces]
            pred_probs_bs = pred_probs[bs_idces]
            calib_probs_samples[:, idx] = sm_lowess(target_bs, pred_probs_bs, frac=frac, it=0,
                                                    xvals=xvals, return_sorted=False, delta=delta)
            #plt.plot(xvals, calib_probs_samples[:, idx])
        return calib_probs, calib_probs_samples
    else:
        return calib_probs


def isotonic_calibration(target, pred_probs, n_boot=0, scoring_function=None, rng=None):
    isotonic = IsotonicRegression(y_min=0, y_max=1)
    isotonic.fit(pred_probs, target)
    isotonic_probs = isotonic.predict(pred_probs)
    # The following was intended to fix the edge behavior in the small-sample regime, but it actually seems to make
    # things worse, hence disabled again.
    # Also, it breaks the theoretical guarantees of the score decomposition.
    # isotonic_probs_unique, val_counts = np.unique(isotonic_probs, return_counts=True)
    # # if there is a 0 likelihood bin, join it with the next bin
    # if isotonic_probs_unique[0] == 0:
    #     new_bin_conditional_event_probability = isotonic_probs_unique[1] * val_counts[1] / (val_counts[0] + val_counts[1])
    #     isotonic_probs[isotonic_probs == 0] = new_bin_conditional_event_probability
    #     isotonic_probs[isotonic_probs == isotonic_probs_unique[1]] = new_bin_conditional_event_probability
    # # if there is a 1 likelihood bin, join it with the previous bin
    # if isotonic_probs_unique[-1] == 1:
    #     new_bin_conditional_event_probability = \
    #         (isotonic_probs_unique[-2] * val_counts[-2] + isotonic_probs_unique[-1] * val_counts[-1]) / (
    #                 val_counts[-2] + val_counts[-1])
    #     isotonic_probs[isotonic_probs == 1] = new_bin_conditional_event_probability
    #     isotonic_probs[isotonic_probs == isotonic_probs_unique[-2]] = new_bin_conditional_event_probability
    if n_boot > 0:
        isotonic_probs_boots = np.zeros((len(isotonic_probs), n_boot))
        boot_isotonic_probs = np.zeros((len(isotonic_probs), n_boot))
        boot_pred_probs = np.zeros((len(isotonic_probs), n_boot))
        boot_target = np.zeros((len(isotonic_probs), n_boot))

        if rng is None:
            rng = np.random.RandomState()

        if scoring_function is not None:
            total_scores = np.zeros((n_boot,))
            mcbs = np.zeros((n_boot,))
            dscs = np.zeros((n_boot,))
            uncs = np.zeros((n_boot,))
        for ii in range(0, n_boot):
            # One (symptom of an underlying) problem with this resampling method is that it will assign 0 uncertainty to
            # regions where the initial isotonic regression identifies a bin with conditional event likelihood 0.
            # That seems very unreasonable. Is there any way to fix that?
            # (This is exactly the method Dimitriadis et al. (2021) describe for their UQ. Their plots also seem to
            # suffer from the same problem.
            # resample predictions
            boot_idces = rng.randint(isotonic_probs.shape[0], size=len(isotonic_probs))
            #boot_idces = range(len(isotonic_probs))
            boot_pred_probs[:, ii] = pred_probs[boot_idces]
            # resample targets assuming initial isotonic regression to be correct
            boot_target[:, ii] = rng.binomial(1, p=isotonic_probs[boot_idces])
            # fit another isotonic regression to the resampled data
            boot_isotonic_probs[:, ii] = isotonic_calibration(boot_target[:, ii], boot_pred_probs[:, ii], n_boot=0)
            isotonic_probs_boots[:, ii] = scipy.interpolate.interp1d(boot_pred_probs[:, ii], boot_isotonic_probs[:, ii],
                                                                     kind="nearest",
                                                                     fill_value="extrapolate")(pred_probs)
            if scoring_function is not None:
                total_scores[ii], mcbs[ii], dscs[ii], uncs[ii] = score_decompose(boot_target[:, ii], boot_pred_probs[:, ii], scoring_function, calib_probs=boot_isotonic_probs[:, ii])
            assert np.all(isotonic_probs_boots[:, ii] <= 1)
        if scoring_function is not None:
            return isotonic_probs, isotonic_probs_boots, total_scores, mcbs, dscs, uncs
        else:
            return isotonic_probs, isotonic_probs_boots
    else:
        return isotonic_probs


def get_calib_probs(target, pred_probs, method='isotonic', n_boot=0, scoring_function=None, rng=None):

    def func_wrap(method_handle):
        if n_boot > 0:
            if scoring_function is not None:
                calib_probs, calib_probs_boots, total_scores, mcbs, dscs, uncs = method_handle()
                return calib_probs, calib_probs_boots, total_scores, mcbs, dscs, uncs
            else:
                calib_probs, calib_probs_boots = method_handle()
                return calib_probs, calib_probs_boots
        else:
            calib_probs = method_handle()
            return calib_probs

    if method == 'isotonic-smoothed':
        assert scoring_function is None
        method_handle = lambda: smoothed_isotonic_calibration(target, pred_probs, n_boot=n_boot, rng=rng)
    elif method == 'isotonic':
        method_handle = lambda: isotonic_calibration(target, pred_probs, n_boot=n_boot,
                                                     scoring_function=scoring_function, rng=rng)
    else:
        raise NotImplementedError

    return func_wrap(method_handle)