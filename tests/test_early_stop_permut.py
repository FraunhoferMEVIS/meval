import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import pytest
import scipy
from typing import cast

import meval.stats as stats
from meval.group_filter import GroupFilter
from meval.metrics.ComparisonMetric import ComparisonMetric
from meval.stats import _clopper_pearson_lower, studentized_permut_pval


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
    correct_zero_pvals: bool = True,
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

    monkeypatch.setattr(stats, 'est_variance_of_metric_diff', fake_est_variance_of_metric_diff)
    monkeypatch.setattr(stats, 'shuffle_masks', fake_shuffle_masks)

    pval, _ = studentized_permut_pval(
        df=df,
        metric=cast(ComparisonMetric, object()),
        group_filter=group_filter,
        num_permut=num_permut,
        max_num_bootstrap=1,
        correct_zero_pvals=correct_zero_pvals,
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

    assert pval == pytest.approx(3.0 / 4.0)


def test_zero_pval_correction_uses_only_valid_permutations(monkeypatch):
    s_base = 5.0
    s_permut = [np.nan, np.nan, 1.0, 2.0, 3.0]

    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch,
        s_base=s_base,
        s_permut=s_permut,
        num_permut=5,
        pval_early_stop_alpha=None,
    )

    assert pval == pytest.approx(0.99 / 3.0)


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
    assert pval == pytest.approx(0.99 / 300.0)
