"""Regression test: dataset-level cached values must never leak between dataframes.

The cache is keyed by id(df). Before the fix, a dataframe created after another one had been
garbage-collected could reuse its id and silently receive its cached reference logit L0.
"""

import gc

import numpy as np
import pandas as pd
from scipy.special import expit

from meval.metrics import CalibrationIntercept
from meval.metrics._calibration import mean_reference_logit
from meval.stats.cache import clear_shared_computation_cache


def _frame(shift: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.5, 2000).astype(bool)
    p = expit(rng.normal(0.0, 1.0, 2000) + shift)
    return pd.DataFrame({"y_true": y, "y_pred_prob": p})


def test_reference_logit_is_recomputed_for_each_new_dataframe():
    clear_shared_computation_cache()
    metric = CalibrationIntercept(reference_score="mean")
    for shift in (0.0, 1.0, 2.0, 3.0, 4.0):
        df = _frame(shift)
        expected = mean_reference_logit(df["y_pred_prob"].to_numpy())
        assert metric._get_reference_logit(df, validate=False) == expected
        del df
        gc.collect()  # make id reuse as likely as possible
    clear_shared_computation_cache()
