import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import inspect
from typing import cast

import numpy as np
import pandas as pd
import pytest
import scipy

import meval.stats.studentized_permut_pval as stats_pval
from meval.group_filter import GroupFilter
from meval.metrics.ComparisonMetric import ComparisonMetric
from meval.stats.studentized_permut_pval import (
    _clopper_pearson_lower,
    studentized_permut_pval,
)


def _minimal_df_and_filter() -> tuple[pd.DataFrame, GroupFilter]:
    df = pd.DataFrame(
        {
            'grp': [1, 1, 0, 0],
            'y_true': [True, False, True, False],
            'y_pred_prob': [0.9, 0.1, 0.8, 0.2],
        }
    )
    return df, GroupFilter({'grp': 1})


def _run_with_mocked_studentized_samples(
    monkeypatch: pytest.MonkeyPatch,
    s_base: float,
    s_permut: list[float],
    num_permut: int,
    pval_early_stop_alpha: float | None,
) -> tuple[float, int]:
    df, group_filter = _minimal_df_and_filter()

    call_count = {'n': 0}

    def fake_est_variance_of_metric_diff(*args, **kwargs):
        call_count['n'] += 1
        if call_count['n'] == 1:
            return 1.0, float(s_base)

        idx = call_count['n'] - 2
        if idx >= len(s_permut):
            raise AssertionError('Requested more mocked permutations than provided.')

        return 1.0, float(s_permut[idx])

    def fake_shuffle_masks(mask_a=None, mask_b=None, *, idces_joined=None, n_a=None, work_buffer=None):
        if idces_joined is not None:
            assert n_a is not None
            return idces_joined[:n_a], idces_joined[n_a:]
        return mask_a, mask_b

    monkeypatch.setattr(stats_pval, 'est_variance_of_metric_diff', fake_est_variance_of_metric_diff)
    monkeypatch.setattr(stats_pval, 'shuffle_masks', fake_shuffle_masks)

    pval, _ = studentized_permut_pval(
        df=df,
        metric=cast(ComparisonMetric, object()),
        group_filter=group_filter,
        num_permut=num_permut,
        max_num_bootstrap=1,
        pval_early_stop_alpha=pval_early_stop_alpha,
    )

    return pval, call_count['n']


def test_clopper_pearson_lower_is_one_sided_99_99_bound():
    M = 35
    N = 200
    confidence = 0.9999

    lb = _clopper_pearson_lower(M, N, confidence)
    expected = float(scipy.stats.beta.ppf(1.0 - confidence, M, N - M + 1))

    assert lb == pytest.approx(expected)


def test_two_sided_pval_counts_ties_as_exceedances(monkeypatch):
    s_base = 1.0
    s_permut = [1.0, -1.0, 0.5, 2.0]

    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch,
        s_base=s_base,
        s_permut=s_permut,
        num_permut=4,
        pval_early_stop_alpha=None,
    )

    assert pval == pytest.approx(4.0 / 5.0)


def test_early_stop_triggers_for_clear_non_significant_case(monkeypatch):
    s_base = 0.1
    s_permut = [1.0] * 1000

    pval, n_calls = _run_with_mocked_studentized_samples(
        monkeypatch,
        s_base=s_base,
        s_permut=s_permut,
        num_permut=1000,
        pval_early_stop_alpha=0.05,
    )

    assert pval > 0.05
    assert n_calls == 101  # 1 base call + first batch of 100 permutations


def test_early_stop_does_not_trigger_for_clear_significant_case(monkeypatch):
    s_base = 10.0
    s_permut = [0.0] * 300

    pval, n_calls = _run_with_mocked_studentized_samples(
        monkeypatch,
        s_base=s_base,
        s_permut=s_permut,
        num_permut=300,
        pval_early_stop_alpha=0.05,
    )

    assert n_calls == 301  # 1 base call + all permutations
    assert pval == pytest.approx(1.0 / 301.0)


# p = (M + 1) / (N_valid + 1), over the valid permutations.

def test_counts_only_valid_permutations(monkeypatch):
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=-2.5, s_permut=[np.nan, -3.0, 1.0, 2.0, 3.0], num_permut=5, pval_early_stop_alpha=None)
    assert pval == pytest.approx(3 / 5)  # |S| >= 2.5: -3, 3, over the 4 valid


def test_pval_is_never_zero(monkeypatch):
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=[np.nan, np.nan, 1.0, 2.0, 3.0], num_permut=5, pval_early_stop_alpha=None)
    assert pval == pytest.approx(1 / 4)


def test_pval_is_monotone_in_the_exceedances_with_many_nans(monkeypatch):
    # With two denominators (N_valid for M = 0, all permutations otherwise), one permutation beyond S_0
    # would make p 33 times smaller than none.
    nans = [np.nan] * 970
    run = lambda valid: _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=valid + nans, num_permut=1000, pval_early_stop_alpha=None)[0]
    assert run([1.0] * 30) == pytest.approx(1 / 31)
    assert run([1.0] * 29 + [6.0]) == pytest.approx(2 / 31)


def test_no_valid_permutation_gives_nan_instead_of_crashing(monkeypatch):
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=[np.nan] * 10, num_permut=10, pval_early_stop_alpha=None)
    assert np.isnan(pval)


def test_statistics_equal_up_to_rounding_count_as_ties(monkeypatch):
    s_base = 2.0
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=s_base, s_permut=[np.nextafter(s_base, 0.0), -np.nextafter(s_base, 0.0), 1.0],
        num_permut=3, pval_early_stop_alpha=None)
    assert pval == pytest.approx(3 / 4)


def test_refuses_tests_that_only_nan_permutations_keep_from_alpha(monkeypatch):
    # 1 / (N_valid + 1) = 1 / 31 > alpha, while all 1000 permutations could have reached alpha.
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=[1.0] * 30 + [np.nan] * 970, num_permut=1000, pval_early_stop_alpha=0.01)
    assert np.isnan(pval)


def test_keeps_tests_whose_resolution_is_low_for_other_reasons(monkeypatch):
    # Too few permutations requested, regardless of NaNs: that is the permutation guard's job.
    few, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=[1.0] * 30 + [np.nan] * 20, num_permut=50, pval_early_stop_alpha=0.01)
    assert few == pytest.approx(1 / 31)
    # Enough valid permutations to reach alpha.
    enough, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=[1.0] * 99 + [np.nan] * 901, num_permut=1000, pval_early_stop_alpha=0.01)
    assert enough == pytest.approx(1 / 100)


def test_early_stop_reports_a_pval_above_alpha_despite_nans(monkeypatch):
    # 98% NaN: the 100 valid permutations needed for the first check are reached after 5000.
    s_permut = [np.nan if i % 50 else (1.0 if i % 100 else -1.0) for i in range(10000)]
    pval, n_calls = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=0.1, s_permut=s_permut, num_permut=10000, pval_early_stop_alpha=0.005)
    assert n_calls == 5001  # 1 / 101 > alpha, but an early-stopped run is not refused
    assert pval > 0.005


def test_early_stop_alpha_is_keyword_only():
    # It took over the positional slot of the removed correct_zero_pvals, whose True would read as alpha.
    parameters = inspect.signature(studentized_permut_pval).parameters
    assert parameters["pval_early_stop_alpha"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("values", [list(range(20)), [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 5]], ids=["distinct", "ties"])
@pytest.mark.parametrize("n_nan", [0, 17])
def test_valid_by_exact_rank_enumeration(monkeypatch, values, n_nan):
    # Under exchangeability S_0 is equally likely to be any of the N_valid + 1 values, so the share of
    # positions with p <= a must not exceed a at any attainable a. NaN permutations must not matter.
    pvals = []
    for i, s_base in enumerate(values):
        s_permut = [float(v) for v in values[:i] + values[i + 1:]] + [np.nan] * n_nan
        pvals.append(_run_with_mocked_studentized_samples(
            monkeypatch, s_base=float(s_base), s_permut=s_permut, num_permut=len(s_permut),
            pval_early_stop_alpha=None)[0])
    pvals = np.array(pvals)
    for a in np.unique(pvals):
        assert np.mean(pvals <= a) <= a + 1e-12


def _null_pvals(
    values: np.ndarray, y_true: np.ndarray, group_size: int, metric: ComparisonMetric,
    n_datasets: int, min_positives: int, rng: np.random.Generator, **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    # Random groups (full exchangeability), each tested only if its group statistic can be computed.
    from meval.stats.bootstrap import RandomState

    RandomState.reset()
    results = []
    while len(results) < n_datasets:
        grp = np.zeros(len(values), dtype=int)
        grp[rng.choice(len(values), group_size, replace=False)] = 1
        if y_true[grp == 1].sum() < min_positives:
            continue
        df = pd.DataFrame({"v": values, "y_true": y_true, "y_pred_prob": 0.5, "grp": grp})
        results.append(studentized_permut_pval(
            df, metric, GroupFilter({"grp": 1}), num_permut=199, pval_early_stop_alpha=None, **kwargs))
    pvals, effects = np.array(results).T
    return pvals, effects


def _monte_carlo_upper(rate: float, n: int) -> float:
    return rate + 3 * np.sqrt(rate * (1 - rate) / n)


def test_holds_its_level_when_most_permutations_are_nan():
    # 8 positives in 400 cases: a random 30-case group rarely holds the 2 the class-conditional variance
    # needs. Dividing by all permutations would push nearly every p-value below alpha.
    from meval.metrics.Average import Average

    rng = np.random.default_rng(2)
    y_true = np.zeros(400, dtype=bool)
    y_true[:8] = True
    n_datasets, alpha = 200, 0.2
    pvals, _ = _null_pvals(
        rng.random(400), y_true, 30, Average("v", y_class=1), n_datasets, min_positives=2, rng=rng)
    assert np.mean(pvals <= alpha) <= _monte_carlo_upper(alpha, n_datasets)
