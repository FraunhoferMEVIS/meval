import pandas as pd
import numpy as np
from meval.stats import newcombe_auroc_ci
import pytest


def test_newcombe_ci_perfect():

    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    with pytest.raises(AssertionError):
        ci = newcombe_auroc_ci(1.0, y_true, ci_alpha=0.01)

    y_true = np.array(y_true)
    ci = newcombe_auroc_ci(1.0, y_true, ci_alpha=0.01)

    assert ci[0] < 0.99
    assert ci[1] == 1.0

    y_true = pd.Series([True, True, True, False, False, False])

    ci2 = newcombe_auroc_ci(1.0, y_true, ci_alpha=0.01)

    assert ci2 == ci