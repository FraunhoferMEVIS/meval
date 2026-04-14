import numpy as np
import pandas as pd
import scipy.stats
from meval.stats import studentized_permut_pval, RandomState
from meval.metrics import Average
from meval.config import settings
from meval.group_filter import GroupFilter

def get_exact_permutation_p_value(n_total, k_total, n_drawn, observed_x):
    """
    Computes the EXACT p-value for a studentized difference of means 
    permutation test using the Hypergeometric distribution.
    """
    def calc_studentized_stat(x, n_a, n_b, k_tot):
        p_a = x / n_a
        p_b = (k_tot - x) / n_b
        diff = p_b - p_a
        # Unbiased sample variance for binary data: p * (1-p) * n / (n-1)
        var_a = (p_a * (1 - p_a) * n_a / (n_a - 1)) if n_a > 1 else 0
        var_b = (p_b * (1 - p_b) * n_b / (n_b - 1)) if n_b > 1 else 0
        if var_a + var_b == 0: 
            return 0
        return diff / np.sqrt((var_a / n_a) + (var_b / n_b))

    # Calculate the observed test statistic
    s_obs = calc_studentized_stat(observed_x, n_drawn, n_total - n_drawn, k_total)
    
    # Sum the probabilities of all possible Hypergeometric draws that result 
    # in a test statistic as extreme or more extreme than what we observed.
    exact_p = 0.0
    for x in range(max(0, n_drawn - (n_total - k_total)), min(k_total, n_drawn) + 1):
        s_perm = calc_studentized_stat(x, n_drawn, n_total - n_drawn, k_total)
        if abs(s_perm) >= abs(s_obs):
            exact_p += scipy.stats.hypergeom.pmf(x, n_total, k_total, n_drawn)
            
    return exact_p

def test_permutation_hypergeometric_convergence():
    # 1. Setup Parameters (n=590 per group)
    n_each = 590
    n_total = n_each * 2
    k_a = 295  # Group A: 50.0% successes
    k_b = 339  # Group B: 57.45% successes
    k_total = k_a + k_b
    
    # 2. Compute the EXACT mathematical ground truth for this permutation test
    true_permutation_p = get_exact_permutation_p_value(n_total, k_total, n_each, k_a)
    
    # For visibility: The exact p-value is 0.01202...
    assert np.round(true_permutation_p, 4) == 0.0120

    # 3. Create the Dataset
    data = np.concatenate([
        np.array([1]*k_a + [0]*(n_each-k_a)),
        np.array([1]*k_b + [0]*(n_each-k_b))
    ])
    df = pd.DataFrame({'val': data, 'group': ['A']*n_each + ['B']*n_each})
    
    metric = Average(metric_col='val')
    g_filter = GroupFilter({"group": "A"})
    
    # 4. Comparative Runs
    results = {}
    for use_numba in [True, False]:
        settings.update(seed=42, enable_numba_shuffle=use_numba)
        RandomState.reset()
        
        pval, _ = studentized_permut_pval(
            df, metric, g_filter, num_permut=10000, pval_early_stop_alpha=None
        )
        results['numba' if use_numba else 'numpy'] = pval

    print(f"Exact Hypergeometric P: {true_permutation_p:.5f}")
    print(f"Numba Permutation P:    {results['numba']:.5f}")
    print(f"NumPy Permutation P:    {results['numpy']:.5f}")

    # 5. The Verification
    # Both methods must converge on the exact hypergeometric probability.
    assert abs(results['numba'] - true_permutation_p) < 0.0015
    assert abs(results['numpy'] - true_permutation_p) < 0.0015
    assert abs(results['numba'] - results['numpy']) < 0.001

if __name__ == "__main__":
    test_permutation_hypergeometric_convergence()