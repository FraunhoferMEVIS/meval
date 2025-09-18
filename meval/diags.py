from collections.abc import Callable
from typing import Optional, Sequence
import math
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.colors as pc
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.basedatatypes import BaseTraceType
from plotly.subplots import make_subplots
from warnings import warn

from .config import settings
from .group_filter import GroupFilter
from .metrics._calibration import loess_calibration
from .metrics._metrics import (
    bootstrap_pr_curve,
    bootstrap_prg_curve,
    bootstrap_roc_curve,
    FPR, TPR, precision, recall, precision_gain, recall_gain
)
from .metrics.ComparisonMetric import ComparisonMetric, CurveBasedComparisonMetric, ThresholdedComparisonMetric
from .metrics.Count import Count
from .stats import ci_nan_quantile

default_layout = {
    "template": "plotly_white",
    "title_x": 0.5,
    "legend": dict(
        x=1.02,
        y=0.5,
        xanchor="left",
        yanchor="middle",
        font=dict(size=9),
        # bgcolor='rgba(255,255,255,0.5)',
        borderwidth=0,
    ),
}

default_layout_no_title = default_layout | {
    "margin": dict(l=0, r=0, t=0, b=0),
}

default_layout_with_title = default_layout | {
    "margin": dict(l=0, r=0, t=40, b=0),
}


def process_export_file(
        fig: go.Figure, 
        export_file: str, 
        export_fig_size_in: Optional[tuple[float | int, float | int]] = None,
        export_fig_size_cm: Optional[tuple[float | int, float | int]] = None, 
        dpi = 600
        ) -> None:
    
    if export_file.endswith(".html"):
        if export_fig_size_in is not None or export_fig_size_cm is not None:
            raise ValueError("export_fig_size_in or export_fig_size_cm is not applicable for HTML export.")
        
        try:
            html_image_content = pio.to_html(fig, full_html=False, include_mathjax='cdn') # type: ignore
            with open(export_file, "w", encoding="utf-8") as file:
                file.write(html_image_content)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    elif export_file.endswith(".png"):

        if export_fig_size_in is None and export_fig_size_cm is None:
            raise ValueError("Either export_fig_size_in or export_fig_size_cm must be provided for PNG export.")
        
        if export_fig_size_in is not None and (not isinstance(export_fig_size_in, (tuple, list)) or len(export_fig_size_in) != 2):
            raise ValueError("export_fig_size_in must be a tuple or list with exactly 2 values (width, height).")
        
        if export_fig_size_cm is not None and (not isinstance(export_fig_size_cm, (tuple, list)) or len(export_fig_size_cm) != 2):
            raise ValueError("export_fig_size_cm must be a tuple or list with exactly 2 values (width, height).")
        
        if export_fig_size_in is not None and export_fig_size_cm is not None:
            raise ValueError("Provide either export_fig_size_in or export_fig_size_cm, not both.")

        try:
            if export_fig_size_in is not None:
                width = round(export_fig_size_in[0] * 100)  # Convert inches to pixels
                height = round(export_fig_size_in[1] * 100)  # Convert inches to pixels
            
            elif export_fig_size_cm is not None:
                width = round(export_fig_size_cm[0] / 2.54 * 100)  # Convert cm to inches, then to pixels
                height = round(export_fig_size_cm[1] / 2.54 * 100)  # Convert cm to inches, then to pixels

            pio.write_image(fig, export_file, format='png', width=width, height=height, scale=dpi/100)

        except Exception as e:
            print(f"An error occurred while saving PNG: {e}")

    else:
        raise ValueError("Unsupported file format. Use .html or .png only.")


def validate_plot_args(
        export_fig_path: Optional[str], 
        export_fig_size_in: Optional[tuple[float | int, float | int]] = None,
        export_fig_size_cm: Optional[tuple[float | int, float | int]] = None
        ) -> None:
    
    if export_fig_path is not None and export_fig_path.endswith(".html"):
        if export_fig_size_in is not None or export_fig_size_cm is not None:
            raise ValueError("export_fig_size_in and export_fig_size_cm are not applicable for HTML export.")


def metric_plot(
        metric: ComparisonMetric, 
        metric_results_df: pd.DataFrame, 
        plot_groups: Optional[Sequence[str]] = None,
        fig_title: Optional[str] = None,
        cmap: Optional[Sequence[str]] = None, 
        sort_groups_by_metric: bool = True,
        figure: bool = True,
        export_fig_path: Optional[str] = None, 
        export_fig_size_cm: Optional[tuple[float | int, float | int]] = None,
        export_fig_size_in: Optional[tuple[float | int, float | int]] = None,
        return_group_color_dict: bool = False
        ) -> go.Figure | list[BaseTraceType] | tuple[go.Figure | list[BaseTraceType], dict]:

    validate_plot_args(export_fig_path, export_fig_size_in, export_fig_size_cm)
    traces = []
    fig = go.Figure()

    if plot_groups is not None:
        metric_results_df = metric_results_df.loc[plot_groups, :]
    else:
        plot_groups = metric_results_df.index.to_list()

    if cmap is None:
        cmap = get_cmap(len(metric_results_df))

    if sort_groups_by_metric:
        metric_results_df = metric_results_df.sort_values(by=[metric.metric_name, metric.metric_name_high, metric.metric_name_low], ascending=False)

    assert metric.metric_name in metric_results_df.columns and metric.metric_name_low in metric_results_df.columns and metric.metric_name_high in metric_results_df.columns

    errors = np.array([metric_results_df[metric.metric_name] - metric_results_df[metric.metric_name_low],
                        metric_results_df[metric.metric_name_high] - metric_results_df[metric.metric_name]])

    upper_errors = errors[1]
    lower_errors = errors[0]
    assert errors.shape == (2, len(metric_results_df))

    x_values = list(range(len(metric_results_df)))
    colors = [cmap[idx] for idx in range(len(metric_results_df))]
    
    group_names = metric_results_df.index.tolist()

    for idx, color in enumerate(colors):

        hover_text = (f"{group_names[idx]}, {metric_results_df[metric.metric_name].iloc[idx]:.2f}")

        metric_val = metric_results_df[metric.metric_name].iloc[idx]

        trace = go.Scatter(
            x=[x_values[idx]],  
            y=[metric_val],
            mode='markers', 
            error_y=dict(
                type='data',
                array=[upper_errors[idx]],
                arrayminus=[lower_errors[idx]],
                color=color
            ),
            marker=dict(
                color=color,
                size=10
            ),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False
        )

        traces.append(trace)

        if np.isnan(upper_errors[idx]) and not np.isnan(metric_val):
            assert np.isnan(lower_errors[idx])
            # CI could not be computed for some reason.
            # By default, no error bar would be plotted.
            # But that is confusing because it looks identical to a very small CI.
            # To clearly indicate the difference, plot a dashed line extending for the whole y range.
            # I would like to simply use fig.vline() for this, but that only works on a figure level.

            ymin = min(metric_results_df[metric.metric_name_low].min(),
                       metric_results_df[metric.metric_name].min())
            ymax = max(metric_results_df[metric.metric_name_high].max(),
                       metric_results_df[metric.metric_name].max())
            yrange = max(ymax - ymin, 0.01)

            if ymin == metric_results_df[metric.metric_name].min():
                ymin -= 0.05*yrange

            if ymax == metric_results_df[metric.metric_name].max():
                ymax += 0.05*yrange

            trace = go.Scatter(
                x=[x_values[idx], x_values[idx]],  
                y=[ymin, ymax],  # Claude suggested [None, None] but that does not work?
                mode='lines', 
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=False,
                line=dict(color=color, width=1, dash='dash')
            )

            traces.append(trace)
    
    if metric.test:
        assert metric.metric_name_pval in metric_results_df.columns

        plot_max_y = metric_results_df[metric.metric_name_high].max()
        plot_min_y = metric_results_df[metric.metric_name_low].min()
        text_pad_y = (plot_max_y - plot_min_y) * 0.15       

        for idx, color in enumerate(colors):

            pval = metric_results_df[metric.metric_name_pval].iloc[idx]

            # Figure out whether to annotate significance above or below the plotted CI
            y_top = metric_results_df[metric.metric_name_high].iloc[idx] + text_pad_y
            y_bot = metric_results_df[metric.metric_name_low].iloc[idx] - text_pad_y

            if (y_top - plot_min_y) < (plot_max_y - y_bot):
                y_val = y_top
            else:
                y_val = y_bot

            if pval <= 1e-5:
                ptext = '****'
            elif pval <= 1e-4:
                ptext = '***'
            elif pval <= 1e-3:
                ptext = '**'
            elif pval <= 1e-2:
                ptext = '*'
            elif pval > 1e-2:
                ptext = 'ns'
            else:
                ptext = 'NA'

            text_trace = go.Scatter(
                x=[x_values[idx]],
                y=[y_val],
                text=[ptext],
                mode="text",
                textposition="middle center",
                hoverinfo="skip",
                showlegend=False,
            )
            traces.append(text_trace)

    fig.add_traces(traces)

    # This won't do anything in metric_overview but will work if generating the figure standalone, because it directly modifies the fig and not the traces.
    # Is that a problem?
    fig.update_layout(**(default_layout_no_title if fig_title is None else default_layout_with_title))
    fig.update_layout(
        title=fig_title,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(metric_results_df))),
            ticktext=list(group_names),
            automargin=True),
        yaxis=dict(
            title=metric.metric_name,
            automargin=True,
        )
    )

    if export_fig_path is not None:
        process_export_file(fig, export_fig_path, export_fig_size_cm=export_fig_size_cm, export_fig_size_in=export_fig_size_in)
    
    if figure:
        if return_group_color_dict:
            return fig, {group: color for group, color in zip(group_names, colors)}
        else:
            return fig
    else:
        if return_group_color_dict:
            return traces, {group: color for group, color in zip(group_names, colors)}
        else:
            return traces


def rel_diag(
        test_df: pd.DataFrame, 
        plot_groups: Optional[Sequence[str]] = None, 
        fig_title: Optional[str] = "Reliability Diagram", 
        group_color_dict: Optional[dict[str, str]] = None, 
        add_risk_density: bool = True, 
        cmap: Optional[Sequence[str]] = None,
        legend: bool = True, 
        threshold: Optional[float] = None,
        export_fig_path: Optional[str] = None, 
        export_fig_size_cm: Optional[tuple[float | int, float | int]] = None,
        export_fig_size_in: Optional[tuple[float | int, float | int]] = None,
        log_density: bool = True
        ) -> tuple[go.Figure, list[BaseTraceType], dict]:
    
    validate_plot_args(export_fig_path, export_fig_size_in, export_fig_size_cm)
    if add_risk_density:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    else:
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    traces = []

    if plot_groups is None:
        # Global reliability diagram across all data
        y_true = ComparisonMetric.get_binary_y_true(test_df)
        y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(test_df)
        xvals = np.linspace(0, 1, 100)
        calib_probs, calib_probs_samples = loess_calibration(y_true, y_pred_prob, 
                                                             n_bootstrap_samples=settings.num_loess_calibration_samples, 
                                                             xvals=xvals)

        cmap = get_cmap(1)
        rgb_color = cmap[0]

        # Add calibration curve
        traces.append(go.Scatter(x=xvals, y=calib_probs, mode='lines', name="Calibration Curve", 
                                 line=dict(color=rgb_color), showlegend=True))

        # Add confidence interval
        traces.append(go.Scatter(
            x=np.concatenate([xvals, xvals[::-1]]),
            y=np.concatenate([ci_nan_quantile(calib_probs_samples, (1 - settings.ci_alpha) / 2, axis=1),
                              ci_nan_quantile(calib_probs_samples, 1 - (1 - settings.ci_alpha) / 2, axis=1)[::-1]]),  # type: ignore
            fill='toself',
            fillcolor=rgb_color,
            line=dict(color='rgba(255,255,255,0)'),
            opacity=settings.ci_plot_alpha,
            showlegend=False
        ))
        
        if add_risk_density:
            hist_data = [y_pred_prob.dropna().tolist()]
            colors = [rgb_color]
            # Create the distribution plot
            dist_fig = ff.create_distplot(hist_data, ['all'], show_hist=False, colors=colors).update_traces(showlegend=False)
            for trace in dist_fig.data:
                fig.add_trace(trace, row=2, col=1)
            fig.update_layout(dist_fig.layout)
            fig.update_yaxes(title_text="Log Density" if log_density else "Density", row=2, col=1, type="log" if log_density else "linear")
            #fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)

    else:
        # Grouped reliability diagrams (same as before)
        if group_color_dict is None:
            cmap = get_cmap(len(plot_groups))
        else:
            cmap = [group_color_dict[group] for group in plot_groups]

        dfs = []
        for idx, group_name in enumerate(plot_groups):
            if not group_name == 'group_med':
                group_mask = GroupFilter(group_repr_str=group_name, col_types=test_df.dtypes)(test_df)
                if group_mask.sum() > 0:
                    y_true = ComparisonMetric.get_binary_y_true(test_df, mask=group_mask)
                    y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(test_df, mask=group_mask)
                    dfs.append(pd.DataFrame({"y_pred_prob": y_pred_prob, "group": group_name}))
                    xvals = np.linspace(0, 1, 100)
                    calib_probs, calib_probs_samples = loess_calibration(y_true, y_pred_prob,
                                                                        n_bootstrap_samples=settings.num_loess_calibration_samples,
                                                                        xvals=xvals)
                    
                    rgb_color = cmap[idx]

                    # Add calibration curve
                    traces.append(go.Scatter(
                        x=xvals,
                        y=calib_probs,
                        mode='lines',
                        line=dict(color=rgb_color),
                        name=group_name,
                    ))

                    # Add confidence interval
                    traces.append(go.Scatter(
                        x=np.concatenate([xvals, xvals[::-1]]),
                        y=np.concatenate([ci_nan_quantile(calib_probs_samples, (1 - settings.ci_alpha) / 2, axis=1),
                                        ci_nan_quantile(calib_probs_samples, 1 - (1 - settings.ci_alpha) / 2, axis=1)[::-1]]), # type: ignore
                        fill='toself',
                        line=dict(color='rgba(255,255,255,0)'),
                        fillcolor=rgb_color,
                        opacity=settings.ci_plot_alpha,
                        showlegend=False
                    ))

        if add_risk_density:
            # Combine all group data for risk density
            data_df = pd.concat(dfs, ignore_index=True)
            # Prepare data for Plotly
            hist_data = [data_df[data_df['group'] == group]["y_pred_prob"].dropna().tolist() for group in plot_groups]
            colors = []
            for idx in range(len(plot_groups)):
                rgb_color = cmap[idx]
                colors.append(rgb_color)

            # Create the distribution plot
            dist_fig = ff.create_distplot(hist_data, plot_groups, show_hist=False, colors=colors).update_traces(showlegend=False)
            for trace in dist_fig.data:
                fig.add_trace(trace, row=2, col=1)
            fig.update_layout(dist_fig.layout)
            fig.update_yaxes(title_text="Log Density" if log_density else "Density", row=2, col=1, type="log" if log_density else "linear")
            #fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)

    # Add ideal line to the first row
    traces.append(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='black'),
        name="Ideal",
    ))

    if threshold is not None:
        traces.append(go.Scatter(
            x=[threshold, threshold],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name="Threshold",
        )) 

    for trace in traces:
        fig.add_trace(trace, row=1, col=1)

    layout_kwargs = {
        'title': fig_title,
        'showlegend': legend,
        'xaxis': dict(
            title='$\\text{Risk Score } R$',
            range=[-0.05, 1.05],
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1]),
        'yaxis': dict(
            title='$P(Y | R, G)$',
            range=[-0.05, 1.05],
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1]),
        'yaxis2': dict(title="Log Density" if log_density else "Density"),     
    } | (default_layout_no_title if fig_title is None else default_layout_with_title)

    fig.update_layout(**layout_kwargs)

    fig.add_trace(go.Scatter(x=[0.95], y=[0.05], text=['over-confident'], mode='text', textfont=dict(style='italic'), showlegend=False, textposition="top left"))
    fig.add_trace(go.Scatter(x=[0.05], y=[0.95], text=['under-confident'], mode='text', textfont=dict(style='italic'), showlegend=False, textposition="bottom right"))
    
    if export_fig_path is not None:
        process_export_file(fig, export_fig_path, export_fig_size_in=export_fig_size_in, export_fig_size_cm=export_fig_size_cm)
    
    return fig, traces, layout_kwargs


def curve_diag(
        test_df: pd.DataFrame, 
        bootstrap_curve_fun: Callable[[npt.NDArray[np.bool], npt.NDArray[np.floating], int], tuple[np.ndarray, np.ndarray, np.ndarray]], 
        plot_groups: Optional[Sequence[str]] = None, 
        group_color_dict: Optional[dict[str, str]] = None, 
        cmap: Optional[Sequence[str]] = None
        ) -> list[BaseTraceType]:

    traces = []

    if plot_groups is None:
        # If no plot groups, calculate for the entire dataset
        y_true = ComparisonMetric.get_binary_y_true(test_df).to_numpy()
        y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(test_df).to_numpy()

        # Bootstrap curve calculation
        x, y, y_bs = bootstrap_curve_fun(y_true, y_pred_prob, settings.N_bootstrap)

        cmap = get_cmap(1)
        rgb_color = cmap[0]

        if not (np.all(np.isnan(x)) or np.all(np.isnan(y))):

            # Add point estimate curve
            traces.append(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color=rgb_color),
                name="all",
                showlegend=True,
                visible=True
            ))

            # Add confidence interval
            traces.append(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([
                    ci_nan_quantile(y_bs, (1 - settings.ci_alpha) / 2, axis=0),
                    ci_nan_quantile(y_bs, 1 - (1 - settings.ci_alpha) / 2, axis=0)[::-1] # type: ignore
                ]),
                fill='toself',
                fillcolor=rgb_color,
                opacity=settings.ci_plot_alpha,
                line=dict(color=rgb_color),
                showlegend=False
            ))
    else:

        if cmap is None:
            if group_color_dict is None:
                cmap = get_cmap(len(plot_groups))
            else:
                cmap = [group_color_dict[group] for group in plot_groups]

        for idx, group_name in enumerate(plot_groups):
            if group_name != 'group_med':
                # Filter the data based on the group
                mask = GroupFilter(group_repr_str=group_name, col_types=test_df.dtypes)(test_df)

                if mask.sum() > 0:
                    y_true = ComparisonMetric.get_binary_y_true(test_df, mask=mask).to_numpy()
                    y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(test_df, mask=mask).to_numpy()

                    x, y, y_bs = bootstrap_curve_fun(y_true, y_pred_prob, settings.N_bootstrap)

                    if not (np.all(np.isnan(x)) or np.all(np.isnan(y))):
                        # Add the main line with legend and group
                        traces.append(go.Scatter(
                            x=x,    
                            y=y,
                            mode='lines',
                            line=dict(color=cmap[idx]),
                            name=group_name,
                            hovertext=group_name,
                            hoverinfo="text",
                            showlegend=True,              # Show the main line in the legend
                            visible=True                  # Ensure the line is visible initially
                        ))

                        # Confidence interval
                        traces.append(go.Scatter(
                            x=np.concatenate([x, x[::-1]]),
                            y=np.concatenate([
                                ci_nan_quantile(y_bs, (1 - settings.ci_alpha) / 2, axis=0),
                                ci_nan_quantile(y_bs, 1 - (1 - settings.ci_alpha) / 2, axis=0)[::-1], # type: ignore
                            ]),
                            fill='toself',
                            fillcolor=cmap[idx],
                            opacity=settings.ci_plot_alpha,
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f"{group_name} CI",
                            hovertext=group_name,
                            hoverinfo="text",
                            showlegend=False,             # Do not show a separate legend entry
                            visible=True
                        ))       
                else:
                    warn("No data found for requested plot group " + group_name)      

    return traces


def add_thresh_marker(traces, x, y, color, thresh):
    traces.append(go.Scatter(
        x=[x],
        y=[y],
        mode='markers',
        marker=dict(size=10, color=color, symbol='circle-dot'),
        marker_line_width=1,
        marker_line_color="black",
        name=f'Threshold {thresh:.2f}',
        showlegend=False
    ))


def roc_diag(
        test_df: pd.DataFrame, 
        plot_groups: Optional[Sequence[str]] = None, 
        fig_title: Optional[str] = 'ROC Diagram', 
        group_color_dict: Optional[dict[str, str]] = None,
        legend: bool = True,
        threshold: Optional[float] = None,
        export_fig_path: Optional[str] = None,
        export_fig_size_cm: Optional[tuple[float | int, float | int]] = None,
        export_fig_size_in: Optional[tuple[float | int, float | int]] = None
        ) -> tuple[go.Figure, list[BaseTraceType], dict]:

    validate_plot_args(export_fig_path, export_fig_size_in, export_fig_size_cm)

    fig = go.Figure()

    traces = curve_diag(test_df, bootstrap_roc_curve, plot_groups=plot_groups, group_color_dict=group_color_dict)

    # Add the Random Classifier diagonal line with legend group
    traces.append(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='black'),
        name="Random Classifier",
        legendgroup="Random Classifier",
        showlegend=True
    ))

    if threshold is not None:
        if plot_groups is None:
            y_true = ComparisonMetric.get_binary_y_true(test_df).to_numpy()
            y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(test_df).to_numpy()

            fpr = FPR(target=y_true, preds=y_pred_prob, threshold=threshold)
            tpr = TPR(target=y_true, preds=y_pred_prob, threshold=threshold)

            add_thresh_marker(traces, x=fpr, y=tpr, color=get_cmap(1)[0], thresh=threshold)

        else:
            if group_color_dict is None:
                cmap = get_cmap(len(plot_groups))
            else:
                cmap = [group_color_dict[group] for group in plot_groups]

            for idx, group_name in enumerate(plot_groups):
                mask = GroupFilter(group_repr_str=group_name, col_types=test_df.dtypes)(test_df)
                if mask.sum() > 0:
                    y_true = ComparisonMetric.get_binary_y_true(test_df, mask=mask).to_numpy()
                    y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(test_df, mask=mask).to_numpy()

                    fpr = FPR(target=y_true, preds=y_pred_prob, threshold=threshold)
                    tpr = TPR(target=y_true, preds=y_pred_prob, threshold=threshold)

                    add_thresh_marker(traces, x=fpr, y=tpr, color=cmap[idx], thresh=threshold)

    fig.add_traces(traces)

    layout_kwargs = {
        'xaxis': dict(
            title="False Positive Rate",
            range=[-0.05, 1.05],
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1]
        ),
        'yaxis': dict(
            title="True Positive Rate",
            range=[-0.05, 1.05],
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1]
        ),
        'title': fig_title,
        'showlegend': legend,
    } | (default_layout_no_title if fig_title is None else default_layout_with_title)

    fig.update_layout(**layout_kwargs)

    if export_fig_path is not None:
        process_export_file(fig, export_fig_path, export_fig_size_in=export_fig_size_in, export_fig_size_cm=export_fig_size_cm)

    return fig, traces, layout_kwargs


def pr_diag(
        test_df: pd.DataFrame, 
        plot_groups: Optional[Sequence[str]] = None,
        fig_title: Optional[str] = 'Precision-Recall Diagram',
        group_color_dict: Optional[dict[str, str]] = None,
        legend: bool = True,
        threshold: Optional[float] = None,
        export_fig_path: Optional[str] = None,
        export_fig_size_cm: Optional[tuple[float | int, float | int]] = None,
        export_fig_size_in: Optional[tuple[float | int, float | int]] = None
        ) -> tuple[go.Figure, list[BaseTraceType], dict]:
    
    validate_plot_args(export_fig_path, export_fig_size_in, export_fig_size_cm)

    fig = go.Figure()

    traces = curve_diag(test_df, bootstrap_pr_curve, plot_groups=plot_groups, group_color_dict=group_color_dict)

    # For PR diagrams, the classifier to beat is not the random one but the always-positive one, which has Precision = base rate and recall = 1.
    # The 'line' to beat is the F1 isometry curve of that classifier (which depends on the baseline).
    # (Cf. https://papers.nips.cc/paper_files/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf)
    # precision in the below calculation becomes negative once recall < 0.5 F1 and inf for recall = 0.5 F1
    # precision in the below calculation becomes > 1 (which makes no sense) for recall < F1 / (2 - F1)
    # The latter cutoff is always larger than the previous cutoff, so we just use that

    # Add the F1 baseline curve
    if plot_groups is None:
        y_true = ComparisonMetric.get_binary_y_true(test_df).to_numpy()
        baserate = y_true.sum() / len(y_true)

        if baserate == 0:
            rec = np.array([0., 1.])
            prec_baseline = np.array([0., 0.])   

        elif baserate == 1:
            rec = np.array([0., 1.])
            prec_baseline = np.array([1., 1.])

        else:
            F1_baseline = 2 * (baserate * 1) / (baserate + 1)
            rec = np.arange(F1_baseline / (2 - F1_baseline), 1+1e-7, 0.01) 
            prec_baseline = F1_baseline * rec / (2 * rec - F1_baseline)

        traces.append(go.Scatter(
            x=rec,
            y=prec_baseline,
            mode='lines',
            line=dict(color='black', dash='dash'),
            name="$F_1 \\text{ baseline}$"
        ))

    else:

        if group_color_dict is None:
            cmap = get_cmap(len(plot_groups))
        elif group_color_dict is not None:
            cmap = [group_color_dict[group] for group in plot_groups]

        for idx, group_name in enumerate(plot_groups):
            mask = GroupFilter(group_repr_str=group_name, col_types=test_df.dtypes)(test_df)
            if mask.sum() > 0:
                y_true = ComparisonMetric.get_binary_y_true(test_df, mask=mask).to_numpy() 
                baserate = y_true.sum() / len(y_true)

                if baserate == 0:
                    rec = np.array([0., 1.])
                    prec_baseline = np.array([0., 0.])   

                elif baserate == 1:
                    rec = np.array([0., 1.])
                    prec_baseline = np.array([1., 1.])     

                else:
                    F1_baseline = 2 * (baserate * 1) / (baserate + 1)
                    rec = np.arange(F1_baseline / (2 - F1_baseline), 1+1e-7, 0.01) 
                    prec_baseline = F1_baseline * rec / (2 * rec - F1_baseline)

                traces.append(go.Scatter(
                    x=rec,
                    y=prec_baseline,
                    mode='lines',
                    line=dict(color=cmap[idx], dash='dash'),
                    name="$F_1 \\text{ baseline " + group_name + "}$"
                ))

    if threshold is not None:
        if plot_groups is None:
            y_true = ComparisonMetric.get_binary_y_true(test_df).to_numpy()
            y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(test_df).to_numpy()

            prec = precision(target=y_true, preds=y_pred_prob, threshold=threshold)
            rec = recall(target=y_true, preds=y_pred_prob, threshold=threshold)

            add_thresh_marker(traces, x=rec, y=prec, color=get_cmap(1)[0], thresh=threshold)

        else:
            if group_color_dict is None:
                cmap = get_cmap(len(plot_groups))
            else:
                cmap = [group_color_dict[group] for group in plot_groups]

            for idx, group_name in enumerate(plot_groups):
                mask = GroupFilter(group_repr_str=group_name, col_types=test_df.dtypes)(test_df)
                y_true = ComparisonMetric.get_binary_y_true(test_df, mask=mask).to_numpy()
                y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(test_df, mask=mask).to_numpy()

                prec = precision(target=y_true, preds=y_pred_prob, threshold=threshold)
                rec = recall(target=y_true, preds=y_pred_prob, threshold=threshold)

                add_thresh_marker(traces, x=rec, y=prec, color=cmap[idx], thresh=threshold)

    fig.add_traces(traces)

    layout_kwargs = {
        'title': fig_title,
        'xaxis': dict(
            title="Recall",
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
            range=[-0.05, 1.05]
        ),
        'yaxis': dict(
            title="Precision",
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
            range=[-0.05, 1.05]
        ),
        'showlegend': legend,
    } | (default_layout_no_title if fig_title is None else default_layout_with_title)

    fig.update_layout(**layout_kwargs)

    if export_fig_path is not None:
        process_export_file(fig, export_fig_path, export_fig_size_in=export_fig_size_in, export_fig_size_cm=export_fig_size_cm)
    
    return fig, traces, layout_kwargs


def prg_diag(
        test_df: pd.DataFrame,
        plot_groups: Optional[Sequence[str]] = None,
        fig_title: Optional[str] = 'Precision-Recall-Gain Diagram',
        group_color_dict: Optional[dict[str, str]] = None,
        legend: bool = True,
        threshold: Optional[float] = None,
        rec_gain_min: float = 0,
        export_fig_path: Optional[str] = None,
        export_fig_size_in: Optional[tuple[float | int, float | int]] = None,
        export_fig_size_cm: Optional[tuple[float | int, float | int]] = None
        ) -> tuple[go.Figure, list[BaseTraceType], dict]:
    
    validate_plot_args(export_fig_path, export_fig_size_in, export_fig_size_cm)

    fig = go.Figure()

    traces = curve_diag(test_df, bootstrap_prg_curve, plot_groups=plot_groups, group_color_dict=group_color_dict)

    # Add FG1 baseline
    traces.append(go.Scatter(
        x=[0, 1],
        y=[1, 0],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name="$F_1$ baseline"
    ))

    if threshold is not None:
        if plot_groups is None:
            y_true = ComparisonMetric.get_binary_y_true(test_df).to_numpy()
            y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(test_df).to_numpy()

            precg = precision_gain(target=y_true, preds=y_pred_prob, threshold=threshold)
            recg = recall_gain(target=y_true, preds=y_pred_prob, threshold=threshold)

            add_thresh_marker(traces, x=recg, y=precg, color=get_cmap(1)[0], thresh=threshold)

        else:
            if group_color_dict is None:
                cmap = get_cmap(len(plot_groups))
            else:
                cmap = [group_color_dict[group] for group in plot_groups]

            for idx, group_name in enumerate(plot_groups):
                mask = GroupFilter(group_repr_str=group_name, col_types=test_df.dtypes)(test_df)
                if mask.sum() > 0:
                    y_true = ComparisonMetric.get_binary_y_true(test_df, mask=mask).to_numpy()
                    y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(test_df, mask=mask).to_numpy()

                    precg = precision_gain(target=y_true, preds=y_pred_prob, threshold=threshold)
                    recg = recall_gain(target=y_true, preds=y_pred_prob, threshold=threshold)

                    add_thresh_marker(traces, x=recg, y=precg, color=cmap[idx], thresh=threshold)

    fig.add_traces(traces)

    # Add vertical and horizontal reference lines
    fig.add_shape(type="line", x0=rec_gain_min, x1=rec_gain_min, y0=-0.05, y1=1,
                  line=dict(color="black", width=1),
                  xref='x', yref='y')

    fig.add_shape(type="line", x0=-0.05, x1=1, y0=0, y1=0,
                  line=dict(color="black", width=1),
                xref='x', yref='y')

    fig.add_shape(type="line", x0=1, x1=1, y0=0, y1=1,
                  line=dict(color="black", width=1),
                  xref='x', yref='y')

    fig.add_shape(type="line", x0=0, x1=1, y0=1, y1=1,
                  line=dict(color="black", width=1),
                  xref='x', yref='y')

    layout_kwargs = {
        'title': fig_title,
        'xaxis': dict(
            title='Recall Gain',
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
            range=[-0.05, 1.05]
        ),
        'yaxis': dict(
            title='Precision Gain',
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
            range=[-0.05, 1.05]
        ),
        'showlegend': legend, 
    } | (default_layout_no_title if fig_title is None else default_layout_with_title)

    fig.update_layout(**layout_kwargs)

    if export_fig_path is not None:
        process_export_file(fig, export_fig_path, export_fig_size_in=export_fig_size_in, export_fig_size_cm=export_fig_size_cm)
    
    return fig, traces, layout_kwargs


def plot_metric_overview(
        metrics: Sequence[ComparisonMetric],
        metric_results_df: pd.DataFrame, 
        plot_groups: Sequence[str],
        test_df: Optional[pd.DataFrame] = None, 
        add_risk_plot: bool = False,
        export_fig_path: Optional[str] = None, 
        export_fig_size_cm: Optional[tuple[float | int, float | int]] = None, 
        export_fig_size_in: Optional[tuple[float | int, float | int]] = None, 
        log_density: bool = True,
        threshold: Optional[float] = None
        ) -> go.Figure:
    
    validate_plot_args(export_fig_path, export_fig_size_in, export_fig_size_cm)

    if threshold is not None:
        for metric in metrics:
            if isinstance(metric, ThresholdedComparisonMetric) and metric.threshold is None:
                metric.set_threshold(threshold)

    plot_metrics = [metric for metric in metrics if not isinstance(metric, Count)]
    if test_df is None:
        num_subplots = len(metrics)
    else:
        num_subplots = len(metrics) + sum([isinstance(metric, CurveBasedComparisonMetric) for metric in metrics])

    if add_risk_plot:
        assert test_df is not None
        num_subplots = num_subplots + 1

    rows = math.ceil(num_subplots / 2)
    cols = 2
    fig = make_subplots(rows=rows, cols=cols)

    if math.ceil(num_subplots / 2) * 2 > num_subplots:
        remove_last_axis = True
    else:
        remove_last_axis = False

    cmap = get_cmap(len(plot_groups))

    group_colors_dict = {group_name: cmap[idx]
                         for idx, group_name in enumerate(plot_groups)}

    plot_idx = 0
    
    for metric in plot_metrics:
        row = (plot_idx // cols) + 1
        col = (plot_idx % cols) + 1
        
        traces = metric_plot(
            metric, 
            metric_results_df.loc[plot_groups, :], 
            plot_groups=plot_groups,
            cmap=cmap, 
            fig_title='', 
            sort_groups_by_metric=False,
            figure=False
        )

        for trace in traces:
            if trace is not None:
                fig.add_trace(trace, row=row, col=col)

        fig.update_yaxes(
            title_text=metric.metric_name, 
            title_standoff=10,  # Increase this value to control the distance
            row=row, 
            col=col,
        )

        fig.update_xaxes(visible=False, row=row, col=col)

        plot_idx += 1

        if isinstance(metric, CurveBasedComparisonMetric) and test_df is not None:
            row = (plot_idx // cols) + 1
            col = (plot_idx % cols) + 1
            
            # Generate the supporting curve traces and get the y-axis label
            _, supporting_curve_traces, layout_kwargs = metric.plot_supporting_curve(
                metric_results_df.loc[plot_groups, :],
                test_df,
                group_colors_dict=group_colors_dict,
                add_all_group='all' in plot_groups,
                threshold=threshold
            )

            # Ensure traces are valid before adding
            if supporting_curve_traces is not None:
                for trace in supporting_curve_traces:
                    fig.add_trace(trace, row=row, col=col)

                fig.update_xaxes(**layout_kwargs['xaxis'], row=row, col=col)
                fig.update_yaxes(**layout_kwargs['yaxis'], row=row, col=col)

                plot_idx += 1

    if 'Count' in metric_results_df:
        row = (plot_idx // cols) + 1
        col = (plot_idx % cols) + 1

        trace = go.Bar(
            x=metric_results_df.index.tolist(),  # Group names
            y=metric_results_df.loc[plot_groups, "Count"], 
            marker=dict(color=[cmap[idx] for idx in range(len(plot_groups))]),
            width=0.8,
            showlegend=False
        )

        fig.add_trace(trace, row=row, col=col)

        fig.update_yaxes(
            type='log', 
            title_text="Sample count",
            title_standoff=5, 
            row=row, 
            col=col,
        )

        fig.update_xaxes(
            title_text="",
            tickangle=45,
            tickmode="array",
            tickvals=list(range(len(plot_groups))),
            ticktext=plot_groups,
            row=row,
            col=col, 
            automargin=True
        )

        plot_idx += 1

    if add_risk_plot:
        assert test_df is not None
        dfs = []
        for idx, group_name in enumerate(plot_groups):
            if group_name != 'group_med':
                y_pred_prob = ComparisonMetric.get_binary_y_pred_prob(
                    test_df, 
                    GroupFilter(group_repr_str=group_name, col_types=test_df.dtypes)
                ).to_numpy()
                dfs.append(pd.DataFrame({"y_pred_prob": y_pred_prob, "group": group_name}))

        # Combine data from all groups
        data_df = pd.concat(dfs, ignore_index=True)

        row = (plot_idx // cols) + 1
        col = (plot_idx % cols) + 1

        hist_data = [data_df[data_df['group'] == group]["y_pred_prob"].dropna().tolist() for group in plot_groups]
        group_labels = plot_groups
        colors = []
        for i in range(len(data_df['group'].unique())):
            colors.append(cmap[i])

        # Create the distribution plot
        dist_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
        dist_traces = dist_fig.data
        for trace in dist_traces:
            fig.add_trace(trace, row=row, col=col)
        
        fig.update_xaxes(title_text='$\\text{Risk Score } R$', range=[0, 1], row=row, col=col)
        fig.update_yaxes(
            title_text="Log Density" if log_density else "Density", 
            row=row, 
            col=col,
            title_standoff=5,
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgray',
            type="log" if log_density else "linear"
        )

        plot_idx += 1

    if remove_last_axis:
        last_row = (plot_idx // cols) + 1
        last_col = (plot_idx % cols) + 1

        fig.update_xaxes(visible=False, row=last_row, col=last_col )
        fig.update_yaxes(visible=False, row=last_row, col=last_col )

    fig.update_layout(
        showlegend=False, 
        barmode='group',
        height=rows * 250,
        width=cols * 450,
        margin=dict(l=0, r=0, t=0, b=0),
        template="plotly_white"
    )

    if export_fig_path is not None:
        process_export_file(fig, export_fig_path, export_fig_size_in=export_fig_size_in, export_fig_size_cm=export_fig_size_cm)
    
    return fig


def get_cmap(num_groups: int) -> list[str]:

    if num_groups <= len(pc.qualitative.Plotly):
        # Use Plotly qualitative palette for small number of groups
        cmap = pc.qualitative.Plotly[:num_groups]
    elif num_groups <= len(pc.qualitative.Dark24):
        # Use Dark24 palette for moderate number of groups
        cmap = pc.qualitative.Alphabet[:num_groups]
    else:
        # For large number of groups, sample colors from a continuous scale
        cmap = [
            pc.find_intermediate_color(pc.sequential.Turbo[0], pc.sequential.Turbo[-1], frac, colortype="rgb")
            for frac in [i / (num_groups - 1) for i in range(num_groups)]
        ]
    return cmap # type: ignore


def volcano_plot(
        metric_results_df: pd.DataFrame, 
        metric: ComparisonMetric,
        fig_title: Optional[str] = None,
        figure: bool = True,
        export_fig_path: Optional[str] = None, 
        export_fig_size_cm: Optional[tuple[float | int, float | int]] = None,
        export_fig_size_in: Optional[tuple[float | int, float | int]] = None
        ) -> go.Figure | list[BaseTraceType]:

    validate_plot_args(export_fig_path, export_fig_size_in, export_fig_size_cm)
    
    assert metric.metric_name_pval in metric_results_df.columns and metric.metric_name_effect in metric_results_df.columns

    msk = metric_results_df[metric.metric_name_pval].notna() & metric_results_df[metric.metric_name_effect].notna()

    fig = go.Figure()

    metric_results_df.loc[metric_results_df[metric.metric_name_pval] == 0, metric.metric_name_pval] = 1e-7

    traces = []
    traces.append(go.Scatter(
        x=metric_results_df.loc[msk, metric.metric_name_effect],
        y=-np.log10(metric_results_df.loc[msk, metric.metric_name_pval]),
        mode='markers',
        marker=dict(
            color="black",
            size=10
        ),
        hovertext=metric_results_df[msk].index,
        hoverinfo="text",
        showlegend=False
    ))

    fig.add_traces(traces)

    fig.update_layout(**(default_layout_no_title if fig_title is None else default_layout_with_title))
    fig.update_layout(
        title=fig_title,
        xaxis=dict(
            title='Metric difference group vs. complement',
            automargin=True),
        yaxis=dict(
            title='-log10(pval)',
            automargin=True
        ),
        height=450,
        width=600,
    )

    max_effect = metric_results_df.loc[msk, metric.metric_name_effect].abs().max()
    fig.update_xaxes(range=[-1.1*max_effect, 1.1*max_effect])
    fig.add_hline(y=2)

    if export_fig_path is not None:
        process_export_file(fig, export_fig_path, export_fig_size_cm=export_fig_size_cm, export_fig_size_in=export_fig_size_in)
    
    if figure:
        return fig
    else:
        return traces