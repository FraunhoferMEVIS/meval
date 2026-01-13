# meval

`meval` is a comprehensive toolbox for machine learning model performance evaluation. Notable features include:
- **Automatic handling of sub-(sub-...)groups of inputs** based on any available metadata (patient demographics such as age and sex, scanner field strength, comorbidities, ... whatever you have available). Out of the potentially hundreds of analyzed subgroups, the most "interesting" ones to be shown in overview plots are selected based on effect size and statistical significance simultaneously, using a [Volcano plot](https://en.wikipedia.org/wiki/Volcano_plot_(statistics)).
- **Uncertainty estimates** (confidence intervals) around performance metrics and relevant performance curves (ROC, PR, Reliability) are automatically provided, based on carefully selected CI construction approaches.
- **Statistical hypothesis testing methodology** specifically designed for handling arbitrary metrics and many different subgroups simultaneously, while properly correcting for multiple testing.
- **Ease of use**: from a simple csv file with model predictions or performance data and some metadata, you will get a fully featured (HTML/plotly-based) model performance report with a single line of code. Using the toolbox does *not* require any access to the model under test, only to its predictive performance on some evaluation dataset.
- **Publication-ready plots**: plot export functionality designed to produce publication-ready plots, both of broad group-stratified performance overviews but also of simple overall ROC/PR/Reliability plots, etc. Plots can also easily be further customized.
- **Extensibility**: new metrics, plot types, etc. can easily be added by users around the core functionality provided by the toolbox. These immediately benefit from the core built-in statistical machinery (uncertainty quantification/CIs, statistical hypothesis testing, etc.).

The following is an example of the kind of analyses produced by the toolbox (in this case, for a skin lesion malignancy classifier on the ISIC dataset). (Note that this is a static screenshot of an interactive HTML document.)

![meval analysis figure for ISIC classifier](demos/isic/isic_overview.png)

The toolbox is designed with two main **use cases** in mind:
1. You have trained a model that you consider reasonably close-to-final. You then use the toolbox to obtain a comprehensive assessment of model performance across many metrics and subgroups of (potential) interest, with minimal effort. If this helps you uncover important failure modes or weak spots of the model, you go back and fix them before shipping, resulting in a more robust final model.
2. You have a final model and want to generate a comprehensive subgroup performance evaluation report, e.g. for communication with collaborators, scientific publication, or other evaluation purposes. The toolbox enables you to do this, with proper confidence intervals, significance testing, and beautiful resulting graphs, with minimal time effort.

Please refer to our MICCAI 2025 FAIMI Workshop paper for details: [meval: A Statistical Toolbox for Fine-Grained Model Performance Analysis](https://doi.org/10.1007/978-3-032-05870-6_19). ([arxiv link](https://arxiv.org/abs/2512.17409))

## Installation

Install as a package using
```
    pip install meval@git+https://github.com/FraunhoferMEVIS/meval
```
or clone the repo and `pip install .` (while in the top directory of the repo).

## Getting started

The following minimal example (minimally adapted from [demos/isic/demo.py](demos/isic/demo.py)) shows how to generate a full HTML evaluation report with a single function call:
```python
import pandas as pd
from meval import compare_groups
from meval.metrics import Accuracy, ProportionOfPos, BrierScore, AUROC, AUPRG, DRMSCE, Count
from meval.diags import plot_metric_overview

test_df = pd.read_csv('isic_test_results.csv', dtype={"label": "bool"})

# Select metrics to assess
# Significance testing will only be done for AUROC here
# DRMSCE is a sample size-debiased calibration metric, refer to https://doi.org/10.1145/3593013.3594045
# AUPRG is the area under the precision-recall-*gain* curve, see our paper and
# https://papers.nips.cc/paper_files/paper/2015/file/33e8075e9970de0cfea955afd4644bb2-Paper.pdf for details.
metrics=[Accuracy(threshold=test_df.label.mean()), ProportionOfPos(), BrierScore(balanced=False), BrierScore(balanced=True), 
         AUROC(test=True), AUPRG(rec_gain_min=0.8), DRMSCE(), Count()]  

# Run overall analysis - in general, this is intended to be the first and main thing to run.
# It automatically evaluates all specified metrics over all desired sub-(sub-)groups, selects the 
# most interesting ones to show, and generates an interactive HTML report.
all_metric_results_df, plot_groups = compare_groups(
    df=test_df,
    metrics=metrics,
    group_interactions=1,  # maximum number of group interactions to consider
    group_by=["sex", "site"],
    report_file="isic_evaluation_report.html",
)
```

You can check out an example of the resulting HTML report [here](demos/isic/isic_evaluation_report.html).

Alternatively, you can also generate and export individual plots, with many customization options:
```python
from meval.diags import plot_metric_overview, metric_plot, roc_diag, rel_diag, prg_diag, pr_diag

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
```

All resulting plot files for this demo case can be found in the `demos/isic/` folder.


## Status, feature requests, future plans

The toolbox is under active development. 

If there is a _specific feature that would make this toolbox more useful for you_, or if you have any questions about using the toolbox, please do not hesitate to reach out or open an issue here.

_Bug reports and pull requests_ are, of course, always highly welcome.

_Use cases_: if you happen to use the toolbox for a scientific publication or basically any other purpose, I would love to learn about it!
(This is true if you find it useful or like it, but even more so if that is _not_ the case!)

*Commercial usage* of the toolbox is not permitted without a specific licensing agreement, cf. below. If you would like to use the toolbox (or a specifically adapted version of it) for commercial purposes, please reach out and we will find an agreement.


## Citation
If you use the toolbox in a scientific publication, please consider citing our MICCAI FAIMI paper:
```
@InProceedings{Sutariya2025,
  author    = {Dishantkumar Sutariya and Eike Petersen},
  booktitle = {MICCAI Fairness of AI in Medical Imaging (FAIMI) Workshop},
  title     = {meval: A Statistical Toolbox for Fine-Grained Model Performance Analysis},
  year      = {2025},
  doi       = {10.1007/978-3-032-05870-6_19}
}
```

## License
This project is under a non-commercial license, for details refer to [the license file](LICENSE).

## Disclaimer
The software is not qualified for use as a medical product or as part thereof. Provided 'as is' without specific verification or validation.

## Contact
Eike Petersen, Fraunhofer Institute for Digital Medicine MEVIS, Germany.
