from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

import meval.reporting as reporting
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
from meval.metrics import MAE, Count


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


def test_plot_metric_overview_uses_taller_default_height(
    test_df: pd.DataFrame,
    metric_results: tuple[MAE, pd.DataFrame, list[str]],
) -> None:
    metric, all_metric_results_df, plot_groups = metric_results

    fig = plot_metric_overview(
        metrics=[metric, Count()],
        metric_results_df=all_metric_results_df,
        plot_groups=plot_groups,
        test_df=test_df,
    )

    assert fig.layout.height == 375


def test_generate_report_file_can_skip_supporting_and_risk_plots(tmp_path: Path, test_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_plot_metric_overview(*args, **kwargs):
        captured["test_df"] = kwargs.get("test_df")
        captured["add_risk_plot"] = kwargs.get("add_risk_plot")
        captured["include_supporting_curves"] = kwargs.get("include_supporting_curves")
        captured["subplot_cols"] = kwargs.get("subplot_cols")
        captured["shared_xaxes"] = kwargs.get("shared_xaxes")
        captured["layout_width_px"] = kwargs.get("layout_width_px")
        captured["layout_height_px"] = kwargs.get("layout_height_px")
        captured["count_bar_opacity"] = kwargs.get("count_bar_opacity")
        captured["count_bar_width"] = kwargs.get("count_bar_width")
        captured["significance_text_pad_scale"] = kwargs.get("significance_text_pad_scale")
        captured["group_tick_angle"] = kwargs.get("group_tick_angle")
        captured["group_tick_font_size"] = kwargs.get("group_tick_font_size")
        captured["layout_margin_px"] = kwargs.get("layout_margin_px")
        return go.Figure()

    monkeypatch.setattr(reporting, "plot_metric_overview", fake_plot_metric_overview)

    report_path = tmp_path / "report.html"
    reporting.generate_report_file(
        report_file_path=report_path,
        df=test_df,
        all_metric_results_df=pd.DataFrame({"Count": [len(test_df)]}, index=["all"]),
        plot_groups=["all"],
        metrics=[Count()],
        include_supporting_curves=False,
        include_risk_plot=False,
        overview_subplot_cols=1,
        overview_shared_xaxes=True,
        overview_layout_width_px=600,
        overview_layout_height_px=500,
        overview_count_bar_opacity=0.45,
        overview_count_bar_width=0.55,
        overview_significance_text_pad_scale=0.22,
        overview_group_tick_angle=60,
        overview_group_tick_font_size=8,
        overview_layout_margin_px={"l": 12, "r": 24, "b": 110, "t": 8},
    )

    assert report_path.exists()
    assert captured["test_df"] is test_df
    assert captured["add_risk_plot"] is False
    assert captured["include_supporting_curves"] is False
    assert captured["subplot_cols"] == 1
    assert captured["shared_xaxes"] is True
    assert captured["layout_width_px"] == 600
    assert captured["layout_height_px"] == 500
    assert captured["count_bar_opacity"] == 0.45
    assert captured["count_bar_width"] == 0.55
    assert captured["significance_text_pad_scale"] == 0.22
    assert captured["group_tick_angle"] == 60
    assert captured["group_tick_font_size"] == 8
    assert captured["layout_margin_px"] == {"l": 12, "r": 24, "b": 110, "t": 8}


def test_plot_metric_overview_can_stack_panels_vertically_with_shared_x_axis(
    test_df: pd.DataFrame,
    metric_results: tuple[MAE, pd.DataFrame, list[str]],
) -> None:
    metric, all_metric_results_df, plot_groups = metric_results

    fig = plot_metric_overview(
        metrics=[metric, Count()],
        metric_results_df=all_metric_results_df,
        plot_groups=plot_groups,
        test_df=test_df,
        include_supporting_curves=False,
        subplot_cols=1,
        shared_xaxes=True,
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.showticklabels is False
    assert fig.layout.xaxis2.showticklabels is True
    assert tuple(fig.layout.xaxis2.ticktext) == tuple(plot_groups)


def test_plot_metric_overview_can_make_count_bars_less_dominant(
    test_df: pd.DataFrame,
    metric_results: tuple[MAE, pd.DataFrame, list[str]],
) -> None:
    metric, all_metric_results_df, plot_groups = metric_results

    fig = plot_metric_overview(
        metrics=[metric, Count()],
        metric_results_df=all_metric_results_df,
        plot_groups=plot_groups,
        test_df=test_df,
        include_supporting_curves=False,
        count_bar_opacity=0.45,
        count_bar_width=0.55,
        layout_width_px=600,
    )

    bar_traces = [trace for trace in fig.data if isinstance(trace, go.Bar)]

    assert len(bar_traces) == 1
    assert bar_traces[0].opacity == 0.45
    assert bar_traces[0].width == 0.55
    assert fig.layout.width == 600


def test_plot_metric_overview_can_override_group_tick_angle_and_font_size(
    test_df: pd.DataFrame,
    metric_results: tuple[MAE, pd.DataFrame, list[str]],
) -> None:
    metric, all_metric_results_df, plot_groups = metric_results

    fig = plot_metric_overview(
        metrics=[metric, Count()],
        metric_results_df=all_metric_results_df,
        plot_groups=plot_groups,
        test_df=test_df,
        include_supporting_curves=False,
        subplot_cols=1,
        shared_xaxes=True,
        group_tick_angle=60,
        group_tick_font_size=8,
    )

    assert isinstance(fig.layout.xaxis2.ticktext, tuple)
    assert tuple(fig.layout.xaxis2.ticktext) == tuple(plot_groups)
    assert fig.layout.xaxis2.tickangle == 60
    assert fig.layout.xaxis2.tickfont.size == 8


def test_plot_metric_overview_can_apply_layout_margin_px(
    test_df: pd.DataFrame,
    metric_results: tuple[MAE, pd.DataFrame, list[str]],
) -> None:
    metric, all_metric_results_df, plot_groups = metric_results

    fig = plot_metric_overview(
        metrics=[metric, Count()],
        metric_results_df=all_metric_results_df,
        plot_groups=plot_groups,
        test_df=test_df,
        include_supporting_curves=False,
        layout_margin_px={"l": 12, "r": 24, "b": 110, "t": 8},
    )

    assert fig.layout.margin.l == 12
    assert fig.layout.margin.r == 24
    assert fig.layout.margin.b == 110
    assert fig.layout.margin.t == 8


def test_plot_metric_overview_can_apply_layout_height_px(
    test_df: pd.DataFrame,
    metric_results: tuple[MAE, pd.DataFrame, list[str]],
) -> None:
    metric, all_metric_results_df, plot_groups = metric_results

    fig = plot_metric_overview(
        metrics=[metric, Count()],
        metric_results_df=all_metric_results_df,
        plot_groups=plot_groups,
        test_df=test_df,
        include_supporting_curves=False,
        layout_height_px=250,
    )

    assert fig.layout.height == 250


def test_plot_metric_overview_can_increase_significance_text_gap(
    test_df: pd.DataFrame,
    metric_results: tuple[MAE, pd.DataFrame, list[str]],
) -> None:
    metric, all_metric_results_df, plot_groups = metric_results

    default_fig = plot_metric_overview(
        metrics=[metric, Count()],
        metric_results_df=all_metric_results_df,
        plot_groups=plot_groups,
        test_df=test_df,
        include_supporting_curves=False,
        subplot_cols=1,
        shared_xaxes=True,
    )
    padded_fig = plot_metric_overview(
        metrics=[metric, Count()],
        metric_results_df=all_metric_results_df,
        plot_groups=plot_groups,
        test_df=test_df,
        include_supporting_curves=False,
        subplot_cols=1,
        shared_xaxes=True,
        significance_text_pad_scale=0.3,
    )

    default_marker_y = {tuple(trace.x): float(trace.y[0]) for trace in default_fig.data if getattr(trace, "mode", None) == "markers"}
    default_text_y = {tuple(trace.x): float(trace.y[0]) for trace in default_fig.data if getattr(trace, "mode", None) == "text"}
    padded_marker_y = {tuple(trace.x): float(trace.y[0]) for trace in padded_fig.data if getattr(trace, "mode", None) == "markers"}
    padded_text_y = {tuple(trace.x): float(trace.y[0]) for trace in padded_fig.data if getattr(trace, "mode", None) == "text"}

    for x_key in default_text_y:
        assert abs(padded_text_y[x_key] - padded_marker_y[x_key]) > abs(default_text_y[x_key] - default_marker_y[x_key])