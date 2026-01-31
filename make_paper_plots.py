import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# -----------------------------
# Helper: Save plot cleanly
# -----------------------------
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved plot: {filename}")
    plt.close()


# -----------------------------
# 1) LOSO Accuracy per Subject Plot
# -----------------------------
def plot_loso_subjectwise_accuracy(csv_path="loso_results_rf_all_wrist.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("Test_Subject")

    plt.figure(figsize=(10, 5))
    plt.bar(df["Test_Subject"], df["Accuracy"])
    plt.ylim(0, 1)
    plt.title("LOSO Subject-wise Accuracy (Random Forest)")
    plt.xlabel("Test Subject")
    plt.ylabel("Accuracy")

    save_plot("plot_loso_subjectwise_accuracy.png")


# -----------------------------
# 2) LOSO Macro F1 per Subject Plot
# -----------------------------
def plot_loso_subjectwise_macro_f1(csv_path="loso_results_rf_all_wrist.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("Test_Subject")

    plt.figure(figsize=(10, 5))
    plt.bar(df["Test_Subject"], df["Macro_F1"])
    plt.ylim(0, 1)
    plt.title("LOSO Subject-wise Macro F1 (Random Forest)")
    plt.xlabel("Test Subject")
    plt.ylabel("Macro F1 Score")

    save_plot("plot_loso_subjectwise_macro_f1.png")


# -----------------------------
# 3) Model Comparison Plot (RF vs CNN)
# -----------------------------
def plot_model_comparison_rf_vs_cnn():
    # Your final results (from your outputs)
    models = ["Random Forest (LOSO)", "1D-CNN (Subject Split)"]
    accuracy = [0.6491, 0.4671]
    macro_f1 = [0.5455, 0.3797]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, accuracy, width, label="Accuracy")
    plt.bar(x + width / 2, macro_f1, width, label="Macro F1")

    plt.xticks(x, models, rotation=10)
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.legend()

    save_plot("plot_model_comparison_rf_vs_cnn.png")


# -----------------------------
# 4) Quantization Speed Plot
# -----------------------------
def plot_quantization_speed():
    # Your quantization results (from your outputs)
    models = ["CNN FP32", "CNN INT8 (Dynamic)"]
    inference_ms = [3.8855, 2.8750]

    plt.figure(figsize=(7, 5))
    plt.bar(models, inference_ms)
    plt.title("CNN Inference Time (CPU) Before vs After Quantization")
    plt.xlabel("Model")
    plt.ylabel("Avg Inference Time (ms)")

    save_plot("plot_quantization_inference_time.png")


# -----------------------------
# 5) Quantization Model Size Plot
# -----------------------------
def plot_quantization_size():
    # Your model size results (from your outputs)
    models = ["CNN FP32", "CNN INT8 (Dynamic)"]
    size_kb = [155.18, 154.67]

    plt.figure(figsize=(7, 5))
    plt.bar(models, size_kb)
    plt.title("CNN Model Size (KB) Before vs After Quantization")
    plt.xlabel("Model")
    plt.ylabel("Model Size (KB)")

    save_plot("plot_quantization_model_size.png")


# -----------------------------
# 6) CNN Confusion Matrix Plot
# -----------------------------
def plot_cnn_confusion_matrix():
    # CNN Confusion Matrix from your output
    cm = np.array([
        [798, 217, 167, 232],
        [317, 278,  23, 216],
        [331, 113,   0,   0],
        [174,  40,  96, 612]
    ])

    labels = ["Neutral", "Stress", "Happiness", "Relaxation"]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=(7, 6))
    disp.plot(values_format="d", cmap="Blues", ax=plt.gca())
    plt.title("1D-CNN Confusion Matrix (Test Split)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    save_plot("plot_cnn_confusion_matrix.png")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("📌 Generating ALL paper plots again...")

    if not os.path.exists("loso_results_rf_all_wrist.csv"):
        print("❌ ERROR: loso_results_rf_all_wrist.csv not found in this folder.")
        print("👉 Put make_paper_plots.py in the same folder as the CSV.")
    else:
        plot_loso_subjectwise_accuracy("loso_results_rf_all_wrist.csv")
        plot_loso_subjectwise_macro_f1("loso_results_rf_all_wrist.csv")

    plot_model_comparison_rf_vs_cnn()
    plot_quantization_speed()
    plot_quantization_size()
    plot_cnn_confusion_matrix()

    print("\n✅ Done! All PNG images are regenerated.")
