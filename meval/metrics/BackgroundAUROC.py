"""
AUROCs that take the positives and the negatives from different sets of cases.

The ordinary AUROC of a group takes both classes from the group. The metrics here take one class
from the group and the other class from elsewhere:

    Metric                      Positives from          Negatives from
    AUROC (for comparison)      the group               the group
    BPSN_AUROC                  the group's complement  the group
    BNSP_AUROC                  the group               the group's complement
    ClassConditionalAUROC(0)    the whole dataset       the group
    ClassConditionalAUROC(1)    the group               the whole dataset

The complement C of a group G is the one compare_groups tests G against: the rows that differ from
G in every grouping attribute (for "sex=m, age=young": sex != m and age != young). For a group
defined by one attribute, that is every row outside G, as in [1], where each subgroup is defined by
one identity. A group given only as a mask has no attributes to define a complement, so BPSN and
BNSP then also need complement_mask (e.g. ~group_mask for every row outside the group).

Why this is useful: if a model shifts y_pred_prob of all cases in a group up or down, the group's
ordinary AUROC does not change, because both of its classes move together. The metrics above do
change, because only one of their two classes moves. A low value with negatives taken from the
group means that the group's negatives have too high y_pred_prob (false alarms); a low value with
positives taken from the group means that the group's positives have too low y_pred_prob (misses).

With per_class=True and multiclass labels, each metric expands into one metric per class (one
class alone: class_id). That class counts as positive, all other classes as negative, and
y_pred_prob_{class} serves as y_pred_prob.

BPSN and BNSP are defined in [1]. They are NaN for the 'all' group, which has no complement. With
G and C as the two groups, BNSP(G) is xAUC(G, C) and BPSN(G) is xAUC(C, G) as defined in [2].

Significance test of BPSN and BNSP: compare_groups compares the metric of G with the metric of C,
so both classes must be present in both G and C. The value for C takes its other class from G, so
BPSN(C) equals BNSP(G): both metrics give the same test, of BNSP(G) - BPSN(G) (the Delta xAUC of
[2]), and the two values compared use disjoint cases, as the test's variance of the difference
assumes. If both metrics are tested, compare_groups runs the test once and counts it once in the
multiple testing correction (shared_test).

[1] Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, Lucy Vasserman. Nuanced Metrics for
    Measuring Unintended Bias with Real Data for Text Classification. Companion of The 2019 World
    Wide Web Conference, pp. 491-500, 2019. https://doi.org/10.1145/3308560.3317593
[2] Nathan Kallus, Angela Zhou. The Fairness of Risk Scores Beyond Classification: Bipartite Ranking
    and the xAUC Metric. Advances in Neural Information Processing Systems 32, 2019.
    https://proceedings.neurips.cc/paper/2019/hash/73e0f7487b8e5297182c5a711d20bf26-Abstract.html
[3] EunYi Chung, Joseph P. Romano. Exact and asymptotically robust permutation tests. The Annals of
    Statistics 41(2), pp. 484-507, 2013. https://doi.org/10.1214/13-AOS1090
"""

import numpy as np
import pandas as pd

from ..config import settings
from ..group_filter import GroupFilter
from ..stats.bootstrap import _mask_to_positions
from ..stats.cache import dataset_computation
from ..stats.utils import mean_and_variance_of_mean
from ._metrics import auroc_components
from .AUROC import _AUROCBase, _raise_if_hard_binary_predictions, _reject_complement_mask
from .ComparisonMetric import ComparisonMetric, DecompositionTerm, MaskLike


def _row_mask(mask: MaskLike, n_rows: int) -> np.ndarray:
    """Boolean mask over all rows, from a boolean mask or row positions."""
    rows = np.zeros(n_rows, dtype=bool)
    rows[_mask_to_positions(mask, n_rows)] = True
    return rows


class _GroupVsReferenceAUROC(_AUROCBase):
    """
    AUROC between the group's cases of one class and a reference's cases of the other class.

    Subclasses choose the class taken from the group (group_class: True = positives) and the
    reference (_reference, by default the group's complement). Everything else is the ordinary AUROC computed on the selected cases:
    the value, the minimum case counts, and the DeLong or Newcombe CI and variance. The two classes
    are disjoint sets of cases, hence independent samples, as those methods assume.
    """

    _reference_class = "complement"
    _reference_description = "of the complementary group"

    def __init__(self, metric_name: str, group_class: bool, test: bool, per_class: bool, class_id: int | None):
        self.group_class = group_class  # defines whether pos or neg examples are taken from the in-group
        self.per_class = per_class
        self.class_id = class_id  # defines the binarization in the multiclass case
        multiclass = per_class or class_id is not None
        super().__init__(
            req_cols=[ComparisonMetric.y_true_cols,
                      ComparisonMetric.y_pred_prob_cols_mc if multiclass else ComparisonMetric.y_pred_prob_cols],
            metric_name=metric_name,
            # Not 'self': the value depends on rows outside the group, so it must not be bootstrapped
            # by resampling the group's rows. No bootstrap is needed, since CI and variance are analytical.
            # 'complement' also makes the permutation test pass each side the other side as complement_mask.
            reference_class=self._reference_class,
            needs_all_classes=False,
            is_descriptive=False,
            direction="maximize",
            test=test,
        )

    def _reference(self, df: pd.DataFrame, group_filter: GroupFilter | None, in_group: np.ndarray,
                   complement_mask: MaskLike | None) -> np.ndarray:
        """Which rows supply the other class: the group's complement."""
        if complement_mask is not None:
            return _row_mask(complement_mask, len(df))
        if group_filter is not None:
            return group_filter.complement(df).to_numpy()
        if in_group.all():  # the 'all' group has no complement
            return np.zeros_like(in_group)
        raise ValueError(
            f"{type(self).__name__} takes its other class from the group's complement, which a group_mask "
            "alone does not define. Pass group_filter, or complement_mask as well (e.g. ~group_mask)."
        )

    @staticmethod
    def _reject_arrays(y_true, y_pred_prob) -> None:
        if y_true is not None or y_pred_prob is not None:
            raise NotImplementedError(
                "These metrics also need cases from outside the group, so they must be computed from "
                "the whole dataframe; precomputed arrays cannot be used."
            )

    def _cases(self, df, group_filter, group_mask, validate, y_true=None, y_pred_prob=None,
               complement_mask=None) -> tuple[np.ndarray, np.ndarray]:
        self._reject_arrays(y_true, y_pred_prob)
        in_group = _row_mask(self.get_group_mask(df, group_filter, group_mask, validate=validate), len(df))
        all_rows = np.ones(len(df), dtype=bool)  # much cheaper than the default mask, a pd.Series
        y_true_np = self.get_binary_y_true(df, mask=all_rows, validate=validate, return_array=True, class_id=self.class_id)
        is_group_class = y_true_np == self.group_class
        reference = self._reference(df, group_filter, in_group, complement_mask)
        selected = (in_group & is_group_class) | (reference & ~is_group_class)
        y_pred_prob_np = self.get_binary_y_pred_prob(df, mask=selected, validate=validate, return_array=True, class_id=self.class_id)
        return y_true_np[selected], y_pred_prob_np

    def _for_class(self, class_id: int) -> ComparisonMetric:
        return type(self)(test=self.test, class_id=class_id)  # type: ignore[call-arg]

    @property
    def shared_test(self) -> tuple[str, int] | None:
        # BPSN(C) = BNSP(G), so BPSN and BNSP both test BNSP(G) - BPSN(G); see the module docstring.
        if self.reference_class != "complement":
            return None
        return f"BNSP - BPSN AUROC, class {self.class_id}", (1 if self.group_class else -1)

    def resolve_metrics(self, df: pd.DataFrame, validate: bool = True) -> list[ComparisonMetric]:
        if self.class_id is not None:
            return [self]
        return self._expand_per_class(df, self.per_class, self._for_class, validate=validate)

    def check_suspicious_usage(self, df: pd.DataFrame) -> None:
        _raise_if_hard_binary_predictions(self.get_binary_y_pred_prob(df, validate=False, return_array=True, class_id=self.class_id))

    @property
    def description(self) -> str | None:
        if self.class_id is None:
            pos, neg, on_prob = "positive", "negative", ""
        else:
            pos, neg, on_prob = f"class-{self.class_id} case", f"non-class-{self.class_id} case", f" on P(class {self.class_id})"
        if self.group_class:
            return (f"Probability that a random {pos} of the group scores higher{on_prob} than a random {neg} "
                    f"{self._reference_description}. Low = the group's {pos}s score low (misses).")
        return (f"Probability that a random {pos} {self._reference_description} scores higher{on_prob} than a random "
                f"{neg} of the group. Low = the group's {neg}s score high (false alarms).")


class BPSN_AUROC(_GroupVsReferenceAUROC):
    """Background-positive, subgroup-negative AUROC [1]: positives of the group's complement against
    negatives of the group. Low if the group's negatives have high y_pred_prob (false alarms)."""

    def __init__(self, test: bool = False, per_class: bool = False, class_id: int | None = None):
        name = "BPSN AUROC" if class_id is None else f"BPSN AUROC[class={class_id}]"
        super().__init__(name, group_class=False, test=test, per_class=per_class, class_id=class_id)


class BNSP_AUROC(_GroupVsReferenceAUROC):
    """Background-negative, subgroup-positive AUROC [1]: positives of the group against negatives
    of the group's complement. Low if the group's positives have low y_pred_prob (misses)."""

    def __init__(self, test: bool = False, per_class: bool = False, class_id: int | None = None):
        name = "BNSP AUROC" if class_id is None else f"BNSP AUROC[class={class_id}]"
        super().__init__(name, group_class=True, test=test, per_class=per_class, class_id=class_id)


class ClassConditionalAUROC(_GroupVsReferenceAUROC):
    """
    AUROC of the group's cases of one class against all cases of the other class in the dataset.

    y_class is the label of the cases taken from the group; the cases of the other label always come
    from the whole dataset. Labels are those of the binary task: with class_id, y_class=1 means class
    class_id and y_class=0 all other classes.

        y_class=1: probability that a random positive of the group has a higher y_pred_prob than a
                   random negative of the whole dataset. Low = the group's positives are missed.
        y_class=0: probability that a random positive of the whole dataset has a higher y_pred_prob
                   than a random negative of the group. Low = false alarms among the group's negatives.

    Ties count one half. Unlike BPSN and BNSP, this needs only cases of y_class in the group, and all
    groups are measured against the same cases of the other label. With w the share of the dataset's
    other-label cases that lie in the group:

        value = w * AUROC(group) + (1 - w) * BPSN(group)    for y_class=0
        value = w * AUROC(group) + (1 - w) * BNSP(group)    for y_class=1

    So the value is the ordinary AUROC for the 'all' group, and BPSN or BNSP for a group without
    cases of the other label.

    Uncertainty: the CI (get_ci, inherited from AUROC) is AUROC's DeLong or Newcombe CI on the cases
    used. It includes the sampling uncertainty of the other-label cases, as it should for the
    group's value. The significance test of compare_groups does not, see get_variance.
    """

    _reference_class = "dataset"
    _reference_description = "of the whole dataset"

    def __init__(self, y_class: int | bool, test: bool = False, per_class: bool = False,
                 class_id: int | None = None):
        assert y_class in (0, 1), "y_class is the class taken from the group: 0 or 1 (per class: 1 is the class itself)."
        if class_id is None:
            name = f"AUROC | y={int(y_class)}"
        else:
            name = f"AUROC[class={class_id}] | y{'=' if y_class else '!='}{class_id}"
        super().__init__(name, group_class=bool(y_class), test=test, per_class=per_class, class_id=class_id)

    def _reference(self, df: pd.DataFrame, group_filter: GroupFilter | None, in_group: np.ndarray,
                   complement_mask: MaskLike | None) -> np.ndarray:
        _reject_complement_mask(self, complement_mask)
        return np.ones_like(in_group)

    def _for_class(self, class_id: int) -> ComparisonMetric:
        return ClassConditionalAUROC(self.group_class, test=self.test, class_id=class_id)

    def _components(self, df: pd.DataFrame, validate: bool) -> np.ndarray:
        """
        Per row of df: what the row contributes to the value of any group that contains it.

        A row labelled y_class gets the share of the dataset's other-label cases that it is ranked
        correctly against, ties counting one half: for y_class=1 the share of negatives with a lower
        y_pred_prob, for y_class=0 the share of positives with a higher one (see auroc_components).
        A row with the other label gets NaN: it belongs to the reference, not to the group's mean.
        All rows get NaN if the dataset has too few other-label cases.

        Because the reference is the same for every group, the value for a group is simply the mean
        of these numbers over its rows. They are therefore computed once per dataset and reused for
        every group and every permutation of the significance test (get_variance), and by decompose.
        """
        if validate:  # the cached computation below does not validate
            self.get_binary_y_true(df, return_array=True, class_id=self.class_id)
            self.get_binary_y_pred_prob(df, return_array=True, class_id=self.class_id)

        def compute() -> np.ndarray:
            y_true_np = np.asarray(self.get_binary_y_true(df, validate=False, return_array=True, class_id=self.class_id), dtype=bool)
            y_pred_prob_np = self.get_binary_y_pred_prob(df, validate=False, return_array=True, class_id=self.class_id)
            is_other_class = y_true_np != self.group_class
            if is_other_class.sum() < settings.auroc_min_cases_per_class:
                return np.full(len(df), np.nan)
            return np.where(is_other_class, np.nan, auroc_components(y_true_np, y_pred_prob_np))

        return dataset_computation(f"ClassConditionalAUROC:{self.class_id}:{self.group_class}", df, compute)

    def get_variance(
        self,
        df: pd.DataFrame,
        group_filter: GroupFilter | None = None,
        group_mask: MaskLike | None = None,
        validate: bool = True,
        y_true: pd.Series | np.ndarray | None = None,
        y_pred: pd.Series | np.ndarray | None = None,
        y_pred_prob: pd.Series | np.ndarray | None = None,
        return_val: bool | None = False,
        method: str = "auto",
        complement_mask: MaskLike | None = None,
    ) -> float | tuple[float, float]:
        """
        Variance used by the significance test of compare_groups, the only user of get_variance.

        This is the variance of the group's value with the dataset's other-label cases held fixed:
        the variance of the mean of _components over the group's rows labelled y_class. The test
        compares a group G with its complement C, and both values are measured against these same
        other-label cases. Holding them fixed makes the test statistic Welch's t of the _components
        values of G's and C's rows labelled y_class, and the studentized permutation test is then
        asymptotically valid [3] for

            H0: G's and C's cases labelled y_class have the same expected _components value, i.e.
                rank equally well on average against this dataset's other-label cases.

        The CI (get_ci) instead uses DeLong's variance, which also includes the sampling uncertainty
        of the other-label cases. Adding the two groups' DeLong variances would count that shared
        uncertainty twice and ignore that it affects both values alike.

        Limitation: H0 concerns this dataset's other-label cases, not the population. If they are few
        and the y_pred_prob distributions of G and C differ in shape, G and C can rank differently
        against them although their values in the population are equal, and the test then rejects
        more often than its nominal level for the population question.
        """
        self._reject_arrays(y_true, y_pred_prob)
        _reject_complement_mask(self, complement_mask)
        mask = self.get_group_mask(df, group_filter, group_mask, validate=validate)
        values = self._components(df, validate)[_mask_to_positions(mask, len(df))]
        val, var = np.nan, np.nan
        if np.isfinite(values).sum() >= settings.auroc_min_cases_per_class:
            val, var = mean_and_variance_of_mean(values)
        return (val, var) if return_val else var

    def decompose(self, df: pd.DataFrame, validate: bool = True) -> list[DecompositionTerm] | None:
        if self.per_class and self.class_id is None:
            return None  # expanded into one metric per class by resolve_metrics()
        components = self._components(df, validate)
        return [DecompositionTerm(1.0, components, np.isfinite(components))]
