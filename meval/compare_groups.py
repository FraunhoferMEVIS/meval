import itertools
import time
import warnings
from multiprocessing import Pool, cpu_count
from typing import Optional, Sequence
import signal
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

from .config import settings
from .metrics.ComparisonMetric import ComparisonMetric, MetricWithAnalyticalCI, ThresholdedComparisonMetric
from .group_filter import GroupFilter, find_binary_complements
from .select_groups import select_extreme_tested_groups
from .stats import bootstrap_ci, studentized_permut_pval
from .reporting import generate_report_file


def calc_metric(df: pd.DataFrame, 
                metric: ComparisonMetric, 
                group_filter: GroupFilter, 
                test_groups: Optional[Sequence[str]] = None,
                ) -> pd.DataFrame:

    with warnings.catch_warnings():
        warnings.simplefilter('once', UserWarning)

        metric_val = metric(df, group_filter)

        if metric.test and not group_filter.group_name == 'all' and (test_groups is None or group_filter.group_name in test_groups):
            pval, effect = studentized_permut_pval(df, metric, group_filter)
        else:
            pval, effect = np.nan, np.nan

        if metric.is_descriptive:
            return pd.DataFrame([{
                'group': group_filter.group_name,
                metric.metric_name: metric_val,
                metric.metric_name_pval: pval
            }]).set_index('group')
        
        else:
            
            if isinstance(metric, MetricWithAnalyticalCI):

                lower, upper = metric.get_ci(df, group_filter, ci_alpha=settings.ci_alpha)

            else:
                lower, upper = bootstrap_ci(df, metric, group_filter, num_bootstrap=settings.N_bootstrap, ci_alpha=settings.ci_alpha)
            
            results = pd.DataFrame([{'group': group_filter.group_name,
                                metric.metric_name: metric_val,
                                metric.metric_name_low: lower,
                                metric.metric_name_high: upper,
                                metric.metric_name_pval: pval,
                                metric.metric_name_effect: effect
            }]).set_index('group')

            return results


def calc_metric_wrap(args):
    return calc_metric(*args)


def initialize_pool(settings_dict):
    # Ignore CTRL+C in the worker process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Prevent excessive CPU oversubscription due to numpy/blas parallelism in addition to multiprocessing.
    # Empirically, I haven't see much of a difference from this. But it seems to be commonly recommend so let's do it, anyway.
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4' 
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'

    # The settings are a singleton, which is re-initialized for each worker by default.
    # This means that without the two lines below, every worker would use the default
    # settings, disregarding any changes to the settings that have been made.
    from meval.config import settings
    settings.update(**settings_dict)


def compare_groups(df: pd.DataFrame, 
                   metrics: Sequence[ComparisonMetric], 
                   group_by: Optional[str | Sequence[str]] = None, 
                   group_interactions: None | int = None,
                   min_subgroup_size: int = 20,
                   report_file: Optional[str] = None, 
                   add_all_group: bool = True,
                   threshold: Optional[float] = None,
                   max_plot_groups: int = 12
                   ) -> tuple[pd.DataFrame, list[str]]:
    # df should contain group vars, score/performance metrics, any additional metadata that are available
    # group_interactions = (None, int - interaction levels). If None or 0, only top-level groups are compared.
    # If group_by is None, no stratification is performed, i.e., analyses are only performed over the whole test population

    # ------------------------------------------------------------------ #
    ### A: DETERMINE GROUPS FOR WHICH TO EVALUATE THE MODEL SEPARATELY ###
    # ------------------------------------------------------------------ #

    if add_all_group:
        analysis_groups = [{}]  # 'all' group
    else:
        analysis_groups = []

    if group_by is None:
        assert group_interactions is None or group_interactions == 0
        assert add_all_group
    else:
        # Are subgroups determined by the intersection of multiple attributes to be considered, and if yes, how/which?
        if isinstance(group_by, str):
            group_by = [group_by]

        for attr in group_by:
            assert len(df[attr].unique()) > 1, "Requested grouping/stratifying by a variable which only takes on a single value."
        
        if group_interactions is None or group_interactions == 0:
            for attribute in group_by:
                analysis_groups = analysis_groups + [{attribute: v} for v in df[attribute].unique()]
        elif isinstance(group_interactions, int):  # int analyze_subgroups specifies the maximum grouping variable combination depth
            assert group_interactions < len(group_by)
            analysis_groups = analysis_groups + get_group_combinations(df, group_by, group_interactions)
            analysis_groups = [grp for grp in analysis_groups if GroupFilter(grp)(df).sum() >= min_subgroup_size]
        else:
            raise NotImplementedError
        
    analysis_group_filters = [GroupFilter(group_attribute_dict) for group_attribute_dict in analysis_groups]


    # ------------------------------------------------------- #
    ### B: COMPUTE METRICS OVER THE DETERMINED (SUB-)GROUPS ###
    # ------------------------------------------------------- #
    start_time = time.time()

    if any([metric.test for metric in metrics]):
        
        assert group_by is not None, "Cannot test for differences with no stratifying ('group_by') attributes specified."

        # If we test eg the "sex=male" group, and sex is a binary attribute in the df, then we do not need
        # to repeat all calculations for the "sex=female" group - we have already done that (assuming sex to be binary).
        # The pval will be the same, the effect will be the negative.
        # Notice that this is also crucial so we don't over-correct for too many tests below!

        # All grouping attrs are at least 2-ary, as per the assert check a bit further above.
        # For our hypothesis testing purposes, we are interested in the case where one group is exactly the complement of another.
        # (This is because we're testing each group for differences w.r.t. its complement.)
        # We here define the complement as attaining different values in *all* group-defining attributes. (Cf. group_filter.py.)
        # I.e. the complement of "male and old" is "non-male *and* non-old". 
        # So one group being the exact complement of another can happen only when all group-defining attributes are binary.
        # (Even in that case, it does not *have* to happen because one of the two complementary groups could be too small.)

        test_groups, test_group_complements = find_binary_complements(df, group_by, analysis_group_filters)

    else:
        test_groups = None
        test_group_complements = {}

    if threshold is not None:
        for metric in metrics:
            if isinstance(metric, ThresholdedComparisonMetric) and metric.threshold is None:
                metric.set_threshold(threshold)

    if settings.parallel and cpu_count() > 1:

        all_inputs = []
        for metric in metrics:
            inputs = list(zip(itertools.repeat(df), itertools.repeat(metric), analysis_group_filters, itertools.repeat(test_groups)))
            all_inputs.extend(inputs)

        with Pool(max(2, cpu_count() - 1), 
                initializer=initialize_pool, 
                initargs=(settings.to_dict(),)) as pool:
            try:
                results_iter = pool.imap_unordered(calc_metric_wrap, all_inputs, chunksize=1)
                
                # Collect results with progress bar
                metric_results_lst = []
                for result in tqdm(results_iter, total=len(all_inputs), desc="Processing"):
                    metric_results_lst.append(result)
                    
            except KeyboardInterrupt:
                print("\n\nTerminating workers...")
                pool.terminate()  # Send SIGTERM to worker processes
                pool.join()       # Wait for workers to terminate
                print("All workers terminated.")
                sys.exit(1)  # Exit cleanly

        with warnings.catch_warnings():
            # pandas has a FutureWarning for concatenating DataFrames with Null entries, but I really do want these empty rows
            warnings.filterwarnings("ignore", category=FutureWarning)
            intermediate = pd.concat(metric_results_lst, axis=0)

        all_metric_results_df = intermediate.groupby(intermediate.index).first().sort_index()

    else:
        metric_results_dfs = []
        for metric in metrics:
            metric_results_lst = []

            print('Evaluating ' + metric.metric_name + '...')
            for group_filter in tqdm(analysis_group_filters):
                metric_results_lst.append(calc_metric(df, metric, group_filter, test_groups=test_groups))

            with warnings.catch_warnings():
                # pandas has a FutureWarning for concatenating DataFrames with Null entries, but I really do want these empty rows
                warnings.filterwarnings("ignore", category=FutureWarning)
                metric_results_df = pd.concat(metric_results_lst)

            metric_results_dfs.append(metric_results_df)

        all_metric_results_df = pd.concat(metric_results_dfs, axis=1).sort_index()

    for metric in metrics:
        if not metric.test:
            if metric.metric_name_pval in all_metric_results_df.columns:
                all_metric_results_df = all_metric_results_df.drop(columns=[metric.metric_name_pval])
            if metric.metric_name_effect in all_metric_results_df.columns:
                all_metric_results_df = all_metric_results_df.drop(columns=[metric.metric_name_effect])

    if any([metric.test for metric in metrics]):

        # Multiple testing correction
        pval_cols = [col for col in all_metric_results_df.columns if 'pval' in col.lower()]
        mask = all_metric_results_df[pval_cols].notna()
        pvals = all_metric_results_df[pval_cols].values[mask.values]
        if len(pvals) > 0:
            _, pvals_transformed, _, _ = multipletests(pvals, 
                                                       method="holm",
                                                       is_sorted=False, 
                                                       returnsorted=False)
            assert pvals_transformed.shape == pvals.shape
            assert np.all(pvals_transformed >= pvals)
            assert np.isnan(pvals_transformed).sum() == 0
            assert np.all(pvals_transformed >= 0) and np.all(pvals_transformed <= 1)
            assert np.all(pvals_transformed <= (len(pvals) * pvals))  # this would be Bonferroni
            
            all_metric_results_df[pval_cols].values[mask.values] = pvals_transformed

            # Fill in the complement pvals we left empty (nan) so far.
            # It is important that this happens *after* multiple testing correction!
            for target_group_name, source_group_name in test_group_complements.items():
                for pval_col in pval_cols:
                    all_metric_results_df.loc[target_group_name, pval_col] = all_metric_results_df.loc[source_group_name, pval_col]
                    effect_col = pval_col[:-4] + "effect"
                    all_metric_results_df.loc[target_group_name, effect_col] = -all_metric_results_df.loc[source_group_name, effect_col]

    print(f"Time spent evaluating all requested metrics on all subgroups: {time.time() - start_time:.1f} seconds")
    

    # ------------------------------------------------------- #
    ### C: GENERATE REPORT                                  ###
    # ------------------------------------------------------- #
    start_time = time.time()

    plot_groups = select_extreme_tested_groups(
        all_metric_results_df=all_metric_results_df, 
        metrics=[metric for metric in metrics if not metric.is_descriptive],
        n_groups=max_plot_groups,
        add_all_group=True)

    if report_file is not None:
        if report_file.endswith(".html"):
            print(f"Report will be written to {report_file}.")
            generate_report_file(report_file_path=report_file, 
                                 df=df, 
                                 all_metric_results_df=all_metric_results_df, 
                                 plot_groups=plot_groups, 
                                 metrics=metrics, 
                                 threshold=threshold)
        else:
            raise ValueError("Unsupported file format. Use .html extension only.")
             
    print(f"Time spent generating report file: {time.time() - start_time:.1f} seconds")

    return all_metric_results_df, plot_groups

    
def get_group_combinations(
        df: pd.DataFrame, 
        group_vars: Sequence[str], 
        max_combinations: int):

    assert max_combinations < len(group_vars)

    # gather values for the different group vars, e.g. gender in [male, female]
    var_vals = []
    for var in group_vars:
        var_vals.append(df[var].unique())

    # get combinations of group vars, e.g. [gender, hospital, gender x hospital]
    # technically, var_combinations is sth like [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)] where the ints specify the position in group_vars
    var_combinations = []
    for ii in range(max_combinations + 1):
        var_combinations = var_combinations + list(itertools.combinations(range(len(group_vars)), ii+1))

    # now get the actual value combinations, e.g. [male, female, hospital A, hospital B, male x hospital A, male x hospital B, ...]
    # technically, groups is a list like [{}, {'gender': 'male'}, {'hospital': 'A'}, {'gender:'male', 'hospital': 'A'}, ...]
    groups = []
    for var_combination in var_combinations:
        var_val_combinations = itertools.product(*[var_vals[idx] for idx in var_combination])
        for var_val_combination in var_val_combinations:
            # var_val_combination is a tuple like (0, 1) where var_val_combination[0] specifies the value taken by group_var[var_combination[0]]
            groups.append({group_vars[var_combination[ii]]: var_val_combination[ii] for ii in range(len(var_combination))})
            
    return groups
