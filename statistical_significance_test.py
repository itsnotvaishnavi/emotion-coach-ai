import pandas as pd
from scipy.stats import wilcoxon

# --------------------------------------
# Manually enter LOSO accuracies
# (from your saved LOSO tables)
# --------------------------------------

baseline_acc = [
    0.5493,  # S2
    0.6667,  # S3
    0.6849,  # S4
    0.6351,  # S5
    0.6301,  # S6
    0.6712,  # S7
    0.7219,  # S8
    0.8559,  # S9
    0.6653,  # S10
    0.7620,  # S11
    0.7051,  # S13
    0.4565,  # S14
    0.5500,  # S15
    0.4535,  # S16
    0.3097   # S17
]

optimized_acc = [
    0.7756,  # S2
    0.6684,  # S3
    0.6273,  # S4
    0.8155,  # S5
    0.6482,  # S6
    0.7211,  # S7
    0.7219,  # S8
    0.8559,  # S9
    0.6653,  # S10
    0.7620,  # S11
    0.7051,  # S13
    0.4565,  # S14
    0.5500,  # S15
    0.4535,  # S16
    0.3097   # S17
]

# --------------------------------------
# Wilcoxon Signed-Rank Test
# --------------------------------------

stat, p_value = wilcoxon(baseline_acc, optimized_acc)

print("\n==============================")
print("STATISTICAL SIGNIFICANCE TEST")
print("==============================")
print("Wilcoxon statistic:", round(stat, 4))
print("p-value:", round(p_value, 6))

if p_value < 0.05:
    print("❌ Statistically significant difference (p < 0.05)")
else:
    print("✅ No statistically significant difference (p ≥ 0.05)")


