import pandas as pd

from meval.metrics import Specificity, ProportionOfPos, BrierScore, AUROC, AUPRG, DRMSCE, Count
from meval.metrics.ThresholdSelection import thresh_tune
from meval.diags import metric_plot, roc_diag, plot_metric_overview, rel_diag
from meval import compare_groups


def read_mimic_df():
    df = pd.read_csv('mimic_results.csv', index_col=0)

    df.loc[:, "y_true"] = df["label_No Finding"].astype(bool)
    df.loc[:, "y_pred_prob"] = df["pred_No Finding"].astype(float)
    df = df[df.y_pred_prob.notnull()] 

    return df


def mimic_demo():

    test_df = read_mimic_df()

    metrics=[Specificity(test=True), ProportionOfPos(), BrierScore(balanced=False), BrierScore(balanced=True), 
             AUROC(test=True), AUPRG(rec_gain_min=0.8), DRMSCE(), Count()]

    threshold = thresh_tune(df=test_df, method="gmean")  # ! In an actual application, threshold selection should be done on a different dataset !

    # Run overall analysis - in general, this is intended to be the first thing to run.
    # It automatically evaluates all specified metrics over all desired sub-(sub-)groups, selects the most interesting ones to show,
    # and generates an interactive HTML report.
    all_metric_results_df, plot_groups = compare_groups(
        df=test_df,
        metrics=metrics,
        group_interactions=None,  # this specifies the maximum number of group interactions to consider, i.e. 2 means feature A x feature B x feature C (=2 interactions, 3 features)
        group_by=["race"],  # view_position
        report_file="mimic_evaluation_report.html",
        threshold=threshold
    )

    all_metric_results_df.to_csv("mimic_evaluation_results.csv")

    plot_metric_overview(metrics, all_metric_results_df, plot_groups, test_df, add_risk_plot=True, export_fig_size_cm=(20, 23), export_fig_path="mimic_overview.png", threshold=threshold)
    metric_plot(Specificity(threshold=threshold, test=True), all_metric_results_df, plot_groups=plot_groups,
                export_fig_size_cm=(9, 6), export_fig_path='mimic_spec.png', sort_groups_by_metric=False)
    metric_plot(Specificity(threshold=threshold, test=True), all_metric_results_df, plot_groups=plot_groups, export_fig_path='mimic_spec.html', sort_groups_by_metric=False)
    metric_plot(AUROC(test=True), all_metric_results_df, plot_groups=plot_groups,
                export_fig_size_cm=(9, 6), export_fig_path='mimic_auroc.png', sort_groups_by_metric=False)    
    _, group_color_dict = metric_plot(AUROC(test=True), all_metric_results_df, plot_groups=plot_groups, export_fig_path='mimic_auroc.html', sort_groups_by_metric=False, return_group_color_dict=True)        
    roc_diag(test_df, plot_groups=['race=WHITE', 'race=BLACK'], group_color_dict=group_color_dict, export_fig_size_cm=(5, 5), export_fig_path='mimic_roc.png', legend=False, fig_title=None, threshold=threshold)
    roc_diag(test_df, plot_groups=['race=WHITE', 'race=BLACK'], group_color_dict=group_color_dict, export_fig_path='mimic_roc.html', legend=True, fig_title=None, threshold=threshold)
    rel_diag(test_df, plot_groups=['race=WHITE', 'race=BLACK'], group_color_dict=group_color_dict, export_fig_size_cm=(5, 5), export_fig_path='mimic_rel.png', legend=False, fig_title=None, threshold=threshold)

if __name__ == "__main__":
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None
        
    mimic_demo()