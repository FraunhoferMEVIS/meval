import plotly.io as pio
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
from typing import Optional, Sequence

from .config import settings
from .diags import plot_metric_overview, volcano_plot
from .metrics.ComparisonMetric import ComparisonMetric


def plot_to_html(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_mathjax="cdn") # type: ignore


def generate_report_file(
        report_file_path: str, 
        df: pd.DataFrame, 
        all_metric_results_df: pd.DataFrame, 
        plot_groups: Sequence[str],
        metrics: Sequence[ComparisonMetric],
        threshold: Optional[float] = None
        ) -> None:

    metric_overview_fig = plot_metric_overview(metrics, all_metric_results_df, plot_groups, test_df=df, add_risk_plot=True, threshold=threshold)
    metric_overview_html = plot_to_html(metric_overview_fig)

    table_cm = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    cols = []
    for metric in metrics:
        cols.append(metric.metric_name)
        if metric.test:
            cols.append(metric.metric_name_pval)

    table_html = all_metric_results_df[cols].style.background_gradient(cmap=table_cm).to_html(index=False, classes="table-style", na_rep='NaN', float_format='%.2f'.__mod__)

    html_content = f"""
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #003366; }}
            p {{ font-size: 16px; }}
            .graph {{ margin-bottom: 40px; }}
            .table-style {{ border-collapse: collapse; width: 100%; }}
            .table-style th, .table-style td {{ border: 1px solid black; padding: 8px; text-align: left; }}
            .table-style th {{ background-color: lightblue; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        <h2>Metric Overview</h2>
        <p>This is an automatically selected subset of all available groups (in general).</p>
        <p>Error bars are obtained using test-set percentile bootstrapping and represent {settings.ci_alpha}-confidence intervals.</p>
        {metric_overview_html}
    """
        
    first = True
    for metric in metrics:
        if metric.test:
            volcano_plot_fig = volcano_plot(all_metric_results_df, metric, fig_title=metric.metric_name)
            volcano_plot_html = plot_to_html(volcano_plot_fig)

            if first:
                html_content += """
                    <h2>Volcano plots</h2>
                    <p>See <a href="https://en.wikipedia.org/wiki/Volcano_plot_(statistics)">Wiki - Volcano plot</a>. Tests performed against the per-attribute complementary group. Groups that have both small (="significant") pvals and large effect sizes are in the top right and left corners.</p>
                    <p>The dashed line indicates a p=0.01 reference; points above this line have smaller pvals.</p>
                """
                first = False

            html_content += volcano_plot_html + "<br><br>"
        
    html_content += f"""
        <h2>Data Table</h2>
        {table_html}        
    </body>
    </html>
    """

    try:
        with open(report_file_path, "w", encoding="utf-8") as file:
            file.write(html_content)
            print("Report generated and written successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
