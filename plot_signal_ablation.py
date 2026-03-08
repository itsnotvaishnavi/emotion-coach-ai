import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ablation_study_results.csv")

plt.figure()
plt.bar(df["Signals Used"], df["LOSO Accuracy"])
plt.xlabel("Signal Combination")
plt.ylabel("Accuracy")
plt.title("Signal Ablation Study (LOSO Evaluation)")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("fig_signal_ablation.png", dpi=300)
plt.show()
