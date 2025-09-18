import pandas as pd
from meval import compare_groups
from meval.metrics import Accuracy, ProportionOfPos, BrierScore, AUROC, AUPRG, DRMSCE, Count, thresh_tune
from meval.diags import plot_metric_overview, metric_plot, roc_diag, rel_diag, prg_diag, pr_diag

def isic_demo():

    test_df = pd.read_csv('isic_test_results.csv', dtype={"label": "bool"})
    test_df.loc[:, "age_group"] = "NA"
    test_df.loc[test_df.age < 20, "age_group"] = "<20"
    test_df.loc[(test_df.age >= 20) & (test_df.age < 45), "age_group"] = "20-44"
    test_df.loc[(test_df.age >= 45) & (test_df.age < 65), "age_group"] = "45-64"
    test_df.loc[test_df.age >= 65, "age_group"] = "65+"
    test_df["age"] = test_df.age_group

    test_df.loc[test_df.site == "head/neck", "site"] = "H/N"
    test_df.loc[test_df.site == "lower extremity", "site"] = "LE"
    test_df.loc[test_df.site == "upper extremity", "site"] = "UE"
    test_df.loc[test_df.site == "oral/genital", "site"] = "O/G"
    test_df.loc[test_df.site == "torso", "site"] = "TO"
    test_df.loc[test_df.site == "palms/soles", "site"] = "P/S"

    # Select metrics to assess
    metrics=[Accuracy(test=True), ProportionOfPos(), BrierScore(balanced=False), BrierScore(balanced=True), 
             AUROC(test=True), AUPRG(rec_gain_min=0.9), DRMSCE(), Count()]

    threshold = thresh_tune(df=test_df, method="base_rate")  # ! In an actual application, threshold selection should be done on a different dataset !

    # Run overall analysis - in general, this is intended to be the first thing to run.
    # It automatically evaluates all specified metrics over all desired sub-(sub-)groups, selects the most interesting ones to show,
    # and generates an interactive HTML report.
    all_metric_results_df, plot_groups = compare_groups(
        df=test_df,
        metrics=metrics,
        group_interactions=1,  # this specifies the maximum number of group interactions to consider, i.e. 2 means feature A x feature B x feature C (=2 interactions, 3 features)
        group_by=["sex", "site", "age"],
        report_file="isic_evaluation_report.html",
        threshold=threshold
    )

    all_metric_results_df.to_csv("isic_evaluation_results.csv")

    # For publications etc., we can also generate and export individual plots (static png or dynamic html), e.g.:
    plot_metric_overview(metrics=metrics, plot_groups=plot_groups, metric_results_df=all_metric_results_df, test_df=test_df,
                        add_risk_plot=True, export_fig_path='isic_overview.png', export_fig_size_cm=(20, 23), threshold=threshold)
    metric_plot(Accuracy(threshold=threshold), all_metric_results_df, plot_groups=plot_groups, 
                export_fig_size_cm=(15, 10), export_fig_path='isic_accuracy.png')
    roc_diag(test_df, plot_groups=['site=TO', 'all', 'site=nan'], export_fig_size_cm=(12, 7), export_fig_path='isic_roc.png', legend=True, fig_title=None, threshold=threshold)
    rel_diag(test_df, plot_groups=['age=65+', 'all', 'sex=0'], export_fig_size_cm=(12, 7), export_fig_path='isic_reliability.png', legend=True, fig_title=None, threshold=threshold)
    prg_diag(test_df, plot_groups=['site=H/N', 'all', 'site=nan'], export_fig_size_cm=(12, 7), export_fig_path='isic_prg.png', legend=True, fig_title=None, threshold=threshold)
    pr_diag(test_df, plot_groups=['site=H/N', 'all', 'site=nan'], export_fig_path='isic_pr.html', legend=True, fig_title=None, threshold=threshold)

    # You can also just create standard 'global' diagrams, i.e. average over all samples as is usually done by default
    rel_diag(test_df, export_fig_size_cm=(12, 7), export_fig_path='isic_reliability_global.png', legend=True, fig_title=None, add_risk_density=True, threshold=threshold)
    roc_diag(test_df, export_fig_size_cm=(12, 7), export_fig_path='isic_roc_global.png', legend=False, fig_title=None, threshold=threshold)


if __name__ == "__main__":
    # fix for multiprocessing / pdb bug: https://github.com/python/cpython/issues/87115
    __spec__ = None

    isic_demo()