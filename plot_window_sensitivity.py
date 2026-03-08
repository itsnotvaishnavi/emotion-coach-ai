import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("window_length_sensitivity_results.csv")

plt.figure()
plt.plot(df["Window_Seconds"], df["LOSO_Accuracy"], marker='o', label="Accuracy")
plt.plot(df["Window_Seconds"], df["LOSO_Macro_F1"], marker='s', label="Macro F1")

plt.xlabel("Window Length (seconds)")
plt.ylabel("Score")
plt.title("Window Length Sensitivity Analysis")
plt.legend()

plt.tight_layout()
plt.savefig("fig_window_sensitivity.png", dpi=300)
plt.show()
