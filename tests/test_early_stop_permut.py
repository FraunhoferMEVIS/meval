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
    two_sided: str = "abs",
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
        two_sided=two_sided,
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


# Both rules: min(1, f * (M + 1) / (N_valid + 1)), with f = 2 and M the smaller tail for equal tails.

def test_equal_tails_doubles_the_smaller_tail_and_counts_ties_in_both(monkeypatch):
    # |S| >= 2 for 4 of 8 (3.0, 2.5, 2.2, -2.0); S <= -2 for 1 of 8 (the tie), S >= -2 for all 8.
    s_permut = [3.0, 2.5, 2.2, 0.1, -0.1, -2.0, 1.0, 0.5]
    run = lambda two_sided: _run_with_mocked_studentized_samples(
        monkeypatch, s_base=-2.0, s_permut=s_permut, num_permut=8, pval_early_stop_alpha=None, two_sided=two_sided)[0]

    assert run("abs") == pytest.approx(5 / 9)
    assert run("equal_tails") == pytest.approx(2 * 2 / 9)


def test_equal_tails_is_capped_at_one_and_symmetric_in_the_direction(monkeypatch):
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=0.0, s_permut=[1.0, -1.0, 0.0, 2.0], num_permut=4, pval_early_stop_alpha=None,
        two_sided="equal_tails")
    assert pval == 1.0

    up, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=1.5, s_permut=[2.0, 0.0, -1.0, -3.0], num_permut=4, pval_early_stop_alpha=None,
        two_sided="equal_tails")
    down, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=-1.5, s_permut=[-2.0, 0.0, 1.0, 3.0], num_permut=4, pval_early_stop_alpha=None,
        two_sided="equal_tails")
    assert up == down == pytest.approx(2 * 2 / 5)


def test_equal_tails_stays_at_most_one_with_a_single_valid_permutation(monkeypatch):
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=[1.0] + [np.nan] * 9, num_permut=10, pval_early_stop_alpha=None,
        two_sided="equal_tails")
    assert pval == 1.0


def test_equal_tails_uses_the_tail_even_if_the_effect_has_the_other_sign(monkeypatch):
    # For a size-biased metric S_0 can be positive and still lie below every random group of its size.
    # The p-value then reports that lower tail, so a direction flag must not be read off the effect's sign.
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=0.5, s_permut=[1.0] * 199, num_permut=199, pval_early_stop_alpha=None,
        two_sided="equal_tails")
    assert pval == pytest.approx(2 * 1 / 200)


@pytest.mark.parametrize("two_sided", ["abs", "equal_tails"])
def test_both_rules_give_a_statistic_and_its_negation_the_same_pval(monkeypatch, two_sided):
    # compare_groups copies the p-value between metrics whose statistics are negations of each other
    # (BPSN and BNSP AUROC, see shared_test), which needs this symmetry.
    s_permut = [3.0, 2.5, -2.0, 0.1, np.nan, -0.4, 1.0, np.nextafter(-2.0, 0.0)]
    run = lambda sign: _run_with_mocked_studentized_samples(
        monkeypatch, s_base=sign * -2.0, s_permut=[sign * s for s in s_permut], num_permut=8,
        pval_early_stop_alpha=None, two_sided=two_sided)[0]
    assert run(1.0) == run(-1.0)


@pytest.mark.parametrize("two_sided", ["abs", "equal_tails"])
def test_both_rules_count_only_valid_permutations(monkeypatch, two_sided):
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=-2.5, s_permut=[np.nan, -3.0, 1.0, 2.0, 3.0], num_permut=5, pval_early_stop_alpha=None,
        two_sided=two_sided)
    # Over the 4 valid: |S| >= 2.5 for -3 and 3; S <= -2.5 for -3.
    assert pval == pytest.approx({"abs": 3 / 5, "equal_tails": 2 * 2 / 5}[two_sided])


@pytest.mark.parametrize(("two_sided", "factor"), [("abs", 1), ("equal_tails", 2)])
def test_pval_is_never_zero(monkeypatch, two_sided, factor):
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=[np.nan, np.nan, 1.0, 2.0, 3.0], num_permut=5, pval_early_stop_alpha=None,
        two_sided=two_sided)
    assert pval == pytest.approx(factor * 1 / 4)


@pytest.mark.parametrize(("two_sided", "factor"), [("abs", 1), ("equal_tails", 2)])
def test_pval_is_monotone_in_the_exceedances_with_many_nans(monkeypatch, two_sided, factor):
    # With two denominators (N_valid for M = 0, all permutations otherwise), one permutation beyond S_0
    # would make p 33 times smaller than none.
    nans = [np.nan] * 970
    run = lambda valid: _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=valid + nans, num_permut=1000, pval_early_stop_alpha=None,
        two_sided=two_sided)[0]
    assert run([1.0] * 30) == pytest.approx(factor * 1 / 31)
    assert run([1.0] * 29 + [6.0]) == pytest.approx(factor * 2 / 31)


@pytest.mark.parametrize("two_sided", ["abs", "equal_tails"])
def test_no_valid_permutation_gives_nan_instead_of_crashing(monkeypatch, two_sided):
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=[np.nan] * 10, num_permut=10, pval_early_stop_alpha=None,
        two_sided=two_sided)
    assert np.isnan(pval)


@pytest.mark.parametrize(("two_sided", "expected"), [("abs", 3 / 4), ("equal_tails", 2 * 2 / 4)])
def test_statistics_equal_up_to_rounding_count_as_ties(monkeypatch, two_sided, expected):
    # Just below S_0 for |S|, and just above and below it for the two tails: all ties.
    s_base = 2.0
    s_permut = [np.nextafter(s_base, 0.0), np.nextafter(s_base, 4.0), -1.0]
    if two_sided == "abs":
        s_permut = [np.nextafter(s_base, 0.0), -np.nextafter(s_base, 0.0), 1.0]
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=s_base, s_permut=s_permut, num_permut=3, pval_early_stop_alpha=None, two_sided=two_sided)
    assert pval == pytest.approx(expected)


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


def test_refusal_uses_the_floor_of_the_rule(monkeypatch):
    # With 150 valid permutations, 1 / 151 reaches alpha = 0.01 but 2 / 151 does not.
    run = lambda two_sided: _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=[1.0] * 150 + [np.nan] * 850, num_permut=1000, pval_early_stop_alpha=0.01,
        two_sided=two_sided)[0]
    assert run("abs") == pytest.approx(1 / 151)
    assert np.isnan(run("equal_tails"))


def test_equal_tails_keeps_tests_whose_permutations_cannot_reach_alpha_anyway(monkeypatch):
    # 2 / 151 > alpha = 0.01 even without NaNs, so the permutation guard, not the refusal, is responsible.
    pval, _ = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=5.0, s_permut=[1.0] * 120 + [np.nan] * 30, num_permut=150, pval_early_stop_alpha=0.01,
        two_sided="equal_tails")
    assert pval == pytest.approx(2 / 121)


@pytest.mark.parametrize("two_sided", ["abs", "equal_tails"])
def test_early_stop_reports_a_pval_above_alpha_despite_nans(monkeypatch, two_sided):
    # 98% NaN: the 100 valid permutations needed for the first check are reached after 5000.
    s_permut = [np.nan if i % 50 else (1.0 if i % 100 else -1.0) for i in range(10000)]
    pval, n_calls = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=0.1, s_permut=s_permut, num_permut=10000, pval_early_stop_alpha=0.005,
        two_sided=two_sided)
    assert n_calls == 5001  # f / 101 > alpha, but an early-stopped run is not refused
    assert pval > 0.005


def test_equal_tails_early_stop_triggers_when_both_tails_are_large(monkeypatch):
    pval, n_calls = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=0.1, s_permut=[1.0, -1.0] * 500, num_permut=1000, pval_early_stop_alpha=0.05,
        two_sided="equal_tails")
    assert pval > 0.05
    assert n_calls == 101  # 1 base call + first batch of 100 permutations


def test_equal_tails_early_stop_compares_twice_the_bound_with_alpha(monkeypatch):
    # 20 of every 100 permutations are in the lower tail. alpha lies between the bound and twice the bound.
    lower_bound = _clopper_pearson_lower(20, 100, 0.9999)
    s_permut = ([-1.0] * 20 + [1.0] * 80) * 3
    pval, n_calls = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=-0.5, s_permut=s_permut, num_permut=300, pval_early_stop_alpha=1.5 * lower_bound,
        two_sided="equal_tails")
    assert n_calls == 101
    assert pval == pytest.approx(2 * 21 / 101)


def test_equal_tails_early_stop_never_triggers_on_an_empty_tail(monkeypatch):
    # The |S| rule would stop here: |S| >= 0.1 for every permutation. In equal tails, no permutation is
    # as low as S_0 = -0.1 (S_0 lies below all of them), so the lower tail is empty and p is small.
    pval, n_calls = _run_with_mocked_studentized_samples(
        monkeypatch, s_base=-0.1, s_permut=[1.0] * 300, num_permut=300, pval_early_stop_alpha=0.05,
        two_sided="equal_tails")
    assert n_calls == 301
    assert pval == pytest.approx(2 * 1 / 301)


def test_unknown_two_sided_method_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="two-sided"):
        _run_with_mocked_studentized_samples(
            monkeypatch, s_base=0.0, s_permut=[1.0], num_permut=1, pval_early_stop_alpha=None, two_sided="both")


def test_defaults_are_equal_tails_with_the_previous_number_of_permutations(monkeypatch):
    from meval.config import Settings

    for name in ("MEVAL_PERMUT_TWO_SIDED", "MEVAL_N_TEST_PERMUT"):
        monkeypatch.delenv(name, raising=False)
    defaults = object.__new__(Settings)
    defaults._init_defaults()
    assert defaults.permut_two_sided == "equal_tails"
    assert defaults.N_test_permut == 1000


@pytest.mark.parametrize(("two_sided", "disabled"), [("equal_tails", True), ("abs", False)])
def test_permutation_guard_counts_the_equal_tail_floor(two_sided, disabled):
    # The guard counts m = 1 here (group1=A; group1=B is A's binary complement, 'all' is never tested).
    # With N = 30 and alpha = 0.05, the |S| floor m/N = 0.033 passes; the equal-tail floor 2m/N = 0.067 does not.
    import warnings

    from meval import compare_groups
    from meval.config import settings
    from meval.metrics import Accuracy

    old_settings = settings.to_dict().copy()
    try:
        settings.update(N_test_permut=30, pval_early_stop_alpha=0.05, parallel=False, testing=False,
                        enable_bh_permut_sufficiency_guard=True, permut_two_sided=two_sided)
        rng = np.random.default_rng(0)
        test_df = pd.DataFrame({
            'y_true': rng.random(60) < 0.5,
            'y_pred': rng.random(60) < 0.5,
            'group1': np.repeat(['A', 'B'], 30),
        })
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results, _ = compare_groups(df=test_df, metrics=[Accuracy(test=True)], group_by="group1",
                                        min_subgroup_size=1, test_correction_method="fdr_bh")
        warned = any("num_permut too small and num_groups too large" in str(w.message) for w in caught)
        assert warned == disabled
        assert np.isnan(results.loc['group1=A', 'Acc pval']) == disabled
    finally:
        settings.from_dict(old_settings)


def test_early_stop_alpha_is_keyword_only():
    # It took over the positional slot of the removed correct_zero_pvals, whose True would read as alpha.
    parameters = inspect.signature(studentized_permut_pval).parameters
    assert parameters["pval_early_stop_alpha"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("values", [list(range(20)), [0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 4, 5]], ids=["distinct", "ties"])
@pytest.mark.parametrize("n_nan", [0, 17])
@pytest.mark.parametrize("two_sided", ["abs", "equal_tails"])
def test_both_rules_are_valid_by_exact_rank_enumeration(monkeypatch, values, n_nan, two_sided):
    # Under exchangeability S_0 is equally likely to be any of the N_valid + 1 values, so the share of
    # positions with p <= a must not exceed a at any attainable a. NaN permutations must not matter.
    pvals = []
    for i, s_base in enumerate(values):
        s_permut = [float(v) for v in values[:i] + values[i + 1:]] + [np.nan] * n_nan
        pvals.append(_run_with_mocked_studentized_samples(
            monkeypatch, s_base=float(s_base), s_permut=s_permut, num_permut=len(s_permut),
            pval_early_stop_alpha=None, two_sided=two_sided)[0])
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


def test_equal_tails_holds_its_level_in_both_directions_on_skewed_exchangeable_data():
    from meval.metrics.Average import Average

    rng = np.random.default_rng(1)
    values = np.where(rng.random(400) < 0.35, 1.0, rng.beta(4.0, 1.2, 400))
    n_datasets, alpha = 400, 0.1
    pvals, effects = _null_pvals(
        values, np.ones(400, dtype=bool), 11, Average("v"), n_datasets, min_positives=0, rng=rng,
        two_sided="equal_tails")
    assert np.mean((pvals <= alpha) & (effects < 0)) <= _monte_carlo_upper(alpha / 2, n_datasets)
    assert np.mean((pvals <= alpha) & (effects > 0)) <= _monte_carlo_upper(alpha / 2, n_datasets)


def test_equal_tails_finds_a_clearly_worse_small_group_that_abs_misses():
    """A small group far below the rest (no random group of its size comes close) gets a p near the
    permutation floor in equal tails, while |S| counts the many random groups of near-1 values with
    collapsed variance (huge positive S) as just as extreme."""
    from meval.metrics.Average import Average
    from meval.stats.bootstrap import RandomState

    rng = np.random.default_rng(0)
    rest = np.where(rng.random(2000) < 0.35, 1.0, rng.beta(4.0, 1.2, 2000))
    group = np.array([0.0, 0.04, 0.08, 0.19, 0.2, 0.3, 0.33, 0.61, 0.61, 0.92, 0.94])
    df = pd.DataFrame({"v": np.concatenate([group, rest]), "y_true": True, "y_pred_prob": 0.5,
                       "grp": np.r_[np.ones(len(group), int), np.zeros(len(rest), int)]})
    pvals = {}
    for two_sided in ("abs", "equal_tails"):
        RandomState.reset()
        pvals[two_sided], effect = studentized_permut_pval(
            df, Average("v"), GroupFilter({"grp": 1}), num_permut=2000, pval_early_stop_alpha=None, two_sided=two_sided)
        assert effect < -0.3
    assert pvals["equal_tails"] <= 2 * 2 / 2001
    assert pvals["abs"] > 5 * pvals["equal_tails"]


@pytest.mark.parametrize("two_sided", ["abs", "equal_tails"])
def test_holds_its_level_when_most_permutations_are_nan(two_sided):
    # 8 positives in 400 cases: a random 30-case group rarely holds the 2 the class-conditional variance
    # needs. Dividing by all permutations would push nearly every p-value below alpha.
    from meval.metrics.Average import Average

    rng = np.random.default_rng(2)
    y_true = np.zeros(400, dtype=bool)
    y_true[:8] = True
    n_datasets, alpha = 200, 0.2
    pvals, _ = _null_pvals(
        rng.random(400), y_true, 30, Average("v", y_class=1), n_datasets, min_positives=2, rng=rng,
        two_sided=two_sided)
    assert np.mean(pvals <= alpha) <= _monte_carlo_upper(alpha, n_datasets)
