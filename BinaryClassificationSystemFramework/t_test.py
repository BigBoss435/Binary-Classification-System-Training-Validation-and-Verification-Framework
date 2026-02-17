import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# ==================================================
# 1. INSERT YOUR VALUES HERE
# --------------------------------------------------

# Example structure: replace values with your own
control = {
    "auc":       [0.805722, 0.694681, 0.838649, 0.753978, 0.848770],
    "accuracy":  [0.931388, 0.969618, 0.982495, 0.981690, 0.978068],
    "precision": [0.068259, 0.129412, 0.571429, 0.1167, 0.243902],
    "recall":    [0.227273, 0.125000, 0.045455, 0.1591, 0.113636],
    "f1":        [0.104987, 0.127168, 0.084211, 0.1000, 0.155039],
    "loss":      [0.186415, 0.170233, 0.170075, 0.166926, 0.167053]
}

enhanced = {
    "auc":       [0.867856, 0.886741, 0.868604, 0.886692, 0.881697],
    "accuracy":  [0.958350, 0.973239, 0.964789, 0.955332, 0.970423],
    "precision": [0.167598, 0.263158, 0.215686, 0.165000, 0.207921],
    "recall":    [0.340909, 0.284091, 0.375000, 0.375000, 0.238636],
    "f1":        [0.224719, 0.273224, 0.273859, 0.229167, 0.222222],
    "loss":      [0.070618, 0.063962, 0.061374, 0.058467, 0.056965]
}

# ==================================================
# 2. RUN PAIRED T-TEST (Wohlin recommended)
# --------------------------------------------------

results = []

for metric in control.keys():
    c = np.array(control[metric])
    e = np.array(enhanced[metric])

    t_stat, p_val = ttest_rel(e, c)   # paired t-test

    results.append({
        "metric": metric,
        "mean_control": np.mean(c),
        "mean_enhanced": np.mean(e),
        "t_statistic": t_stat,
        "p_value": p_val
    })

results_df = pd.DataFrame(results)

# ==================================================
# 3. PRINT RESULTS
# --------------------------------------------------

print("\n===== WOHLIN-STYLE (PAIRED) T-TEST RESULTS =====\n")
print(results_df.to_string(index=False))

print("\nInterpretation:")
print(" p < 0.05 → significant improvement (reject H0)")
print(" Positive t-statistic → Enhanced > Control")
print(" Negative t-statistic → Control > Enhanced")