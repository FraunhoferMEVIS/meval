from pathlib import Path

import pandas as pd
import pytest

from meval import compare_groups
from meval.config import settings
from meval.diags import (
    metric_plot,
    plot_metric_overview,
    pr_diag,
    prg_diag,
    rel_diag,
    roc_diag,
    volcano_plot,
)
from meval.metrics import Count, MAE


@pytest.fixture()
def configured_settings() -> None:
    original_settings = settings.to_dict()
    settings.load_testing_config(parallel=False)
    try:
        yield
    finally:
        settings.from_dict(original_settings)


@pytest.fixture()
def test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": [False, False, True, True, False, True, False, True],
            "y_pred": [0.10, 0.25, 0.70, 0.90, 0.20, 0.80, 0.35, 0.65],
            "y_pred_prob": [0.10, 0.25, 0.70, 0.90, 0.20, 0.80, 0.35, 0.65],
            "site": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )


@pytest.fixture()
def metric_results(test_df: pd.DataFrame, configured_settings: None) -> tuple[MAE, pd.DataFrame, list[str]]:
    metric = MAE(test=True)
    all_metric_results_df, plot_groups = compare_groups(
        df=test_df,
        metrics=[metric, Count()],
        group_by="site",
        min_subgroup_size=1,
    )
    return metric, all_metric_results_df, plot_groups


def test_compare_groups_accepts_path_report_file(tmp_path: Path, test_df: pd.DataFrame, configured_settings: None) -> None:
    report_path = tmp_path / "compare_groups_report.html"

    compare_groups(
        df=test_df,
        metrics=[MAE(), Count()],
        group_by="site",
        min_subgroup_size=1,
        report_file=report_path,
    )

    assert report_path.exists()
    assert "Model Evaluation Report" in report_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "plot_name",
    [
        "metric_plot",
        "rel_diag",
        "roc_diag",
        "pr_diag",
        "prg_diag",
        "plot_metric_overview",
        "volcano_plot",
    ],
)
def test_plotting_functions_accept_path_export_fig_path(
    plot_name: str,
    tmp_path: Path,
    test_df: pd.DataFrame,
    metric_results: tuple[MAE, pd.DataFrame, list[str]],
) -> None:
    metric, all_metric_results_df, plot_groups = metric_results
    export_path = tmp_path / f"{plot_name}.html"

    if plot_name == "metric_plot":
        metric_plot(metric, all_metric_results_df, export_fig_path=export_path)
    elif plot_name == "rel_diag":
        rel_diag(test_df, export_fig_path=export_path)
    elif plot_name == "roc_diag":
        roc_diag(test_df, export_fig_path=export_path)
    elif plot_name == "pr_diag":
        pr_diag(test_df, export_fig_path=export_path)
    elif plot_name == "prg_diag":
        prg_diag(test_df, export_fig_path=export_path)
    elif plot_name == "plot_metric_overview":
        plot_metric_overview(
            metrics=[metric, Count()],
            metric_results_df=all_metric_results_df,
            plot_groups=plot_groups,
            test_df=test_df,
            export_fig_path=export_path,
        )
    elif plot_name == "volcano_plot":
        volcano_plot(all_metric_results_df, metric, export_fig_path=export_path)
    else:
        raise AssertionError(f"Unhandled plot function: {plot_name}")

    assert export_path.exists()