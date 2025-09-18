from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from .ComparisonMetric import ComparisonMetric


def thresh_tune(
    df: Optional[pd.DataFrame] = None,
    y_true: Optional[np.ndarray | pd.Series] = None,
    y_pred_prob: Optional[np.ndarray | pd.Series] = None,
    method: str = "base_rate"
) -> float:

    if df is not None:
        y_true = ComparisonMetric.get_binary_y_true(df)
        y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(df)
    else:
        y_true = np.asarray(y_true)
        y_pred_prob = np.asarray(y_pred_prob)

    if method == "base_rate":
        return y_true.mean()

    elif method == "gmean":
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        gmean = np.sqrt(tpr * (1 - fpr))
        best_idx = np.argmax(gmean)
        return thresholds[best_idx]

    elif method == "F1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx]
    
    else:
        raise ValueError(f"Unknown method: {method}")