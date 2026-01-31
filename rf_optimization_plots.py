import pandas as pd
import matplotlib.pyplot as plt


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved: {filename}")
    plt.close()


def load_and_prepare(csv_path="rf_optimization_results.csv"):
    df = pd.read_csv(csv_path)

    # Sort by gzip size (largest -> smallest)
    df = df.sort_values("Model_Size_gzip_KB", ascending=False).reset_index(drop=True)

    return df


def plot_rf_config_vs_size(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Config"], df["Model_Size_gzip_KB"], marker="o")
    plt.xticks(rotation=30, ha="right")
    plt.title("RF Optimization: Config vs Model Size (gzip KB)")
    plt.xlabel("Random Forest Configuration")
    plt.ylabel("Compressed Model Size (KB)")
    plt.grid(True, alpha=0.3)
    save_plot("rf_config_vs_size.png")


def plot_rf_config_vs_macro_f1(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Config"], df["Avg_LOSO_MacroF1"], marker="o", color="green")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("RF Optimization: Config vs Macro F1 (LOSO)")
    plt.xlabel("Random Forest Configuration")
    plt.ylabel("Macro F1 Score")
    plt.grid(True, alpha=0.3)
    save_plot("rf_config_vs_macro_f1.png")


def plot_rf_config_vs_accuracy(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Config"], df["Avg_LOSO_Accuracy"], marker="o", color="blue")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("RF Optimization: Config vs Accuracy (LOSO)")
    plt.xlabel("Random Forest Configuration")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    save_plot("rf_config_vs_accuracy.png")


def plot_rf_config_vs_inference_time(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Config"], df["Inference_ms(2000_samples)"], marker="o", color="red")
    plt.xticks(rotation=30, ha="right")
    plt.title("RF Optimization: Config vs Inference Time (2000 samples)")
    plt.xlabel("Random Forest Configuration")
    plt.ylabel("Inference Time (ms)")
    plt.grid(True, alpha=0.3)
    save_plot("rf_config_vs_inference_time.png")


def plot_rf_tradeoff_scatter(df):
    """
    BEST plot for paper:
    x = size (gzip KB)
    y = macro f1
    bubble size = speed (faster -> bigger bubble)
    """
    plt.figure(figsize=(8, 6))

    x = df["Model_Size_gzip_KB"]
    y = df["Avg_LOSO_MacroF1"]
    t = df["Inference_ms(2000_samples)"]

    bubble = (1 / t) * 50000  # faster = bigger bubble

    plt.scatter(x, y, s=bubble, alpha=0.75)

    for _, row in df.iterrows():
        plt.text(row["Model_Size_gzip_KB"], row["Avg_LOSO_MacroF1"], row["Config"], fontsize=8)

    plt.title("RF Optimization Trade-off: Size vs Macro F1 (bubble = speed)")
    plt.xlabel("Compressed Size (gzip KB) → smaller is better")
    plt.ylabel("Macro F1 (LOSO) → higher is better")
    plt.grid(True, alpha=0.3)

    save_plot("rf_tradeoff_size_vs_f1.png")


if __name__ == "__main__":
    print("📌 Generating RF optimization plots...")

    df = load_and_prepare("rf_optimization_results.csv")

    plot_rf_config_vs_size(df)
    plot_rf_config_vs_macro_f1(df)
    plot_rf_config_vs_accuracy(df)
    plot_rf_config_vs_inference_time(df)
    plot_rf_tradeoff_scatter(df)

    print("\n✅ Done! RF optimization plots generated successfully.")
