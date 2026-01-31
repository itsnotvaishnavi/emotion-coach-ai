import os
import time
import joblib
import gzip
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from preprocess_wesad_all_wrist_subjectwise import preprocess_wesad_dataset_subjectwise
from features_all_wrist import extract_features


# -----------------------------
# Helper: model size
# -----------------------------
def save_and_get_size(model, name_prefix="rf_model"):
    """
    Saves model in 2 formats:
    1) normal joblib
    2) joblib + gzip compressed
    Returns sizes in KB
    """
    joblib_path = f"{name_prefix}.joblib"
    gz_path = f"{name_prefix}.joblib.gz"

    # Save normal
    joblib.dump(model, joblib_path)

    # Save compressed
    with open(joblib_path, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            f_out.writelines(f_in)

    size_joblib_kb = os.path.getsize(joblib_path) / 1024
    size_gz_kb = os.path.getsize(gz_path) / 1024

    return round(size_joblib_kb, 2), round(size_gz_kb, 2)


# -----------------------------
# Helper: inference time
# -----------------------------
def measure_inference_time(model, X_test, runs=200):
    """
    Measures average prediction time in ms
    """
    # warmup
    _ = model.predict(X_test[:50])

    start = time.time()
    for _ in range(runs):
        _ = model.predict(X_test)
    end = time.time()

    avg_ms = ((end - start) / runs) * 1000
    return round(avg_ms, 4)


# -----------------------------
# LOSO evaluation (fast version)
# -----------------------------
def loso_eval_rf(X_feat, y, groups, rf_params):
    """
    Train RF with LOSO and return avg accuracy and avg macro-f1
    """
    unique_subjects = np.unique(groups)

    acc_list = []
    f1_list = []

    for test_sub in unique_subjects:
        train_mask = groups != test_sub
        test_mask = groups == test_sub

        X_train, y_train = X_feat[train_mask], y[train_mask]
        X_test, y_test = X_feat[test_mask], y[test_mask]

        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        acc_list.append(acc)
        f1_list.append(macro_f1)

    return round(float(np.mean(acc_list)), 4), round(float(np.mean(f1_list)), 4)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    WESAD_ROOT = r"C:\Users\lenovo\Desktop\minor\WESAD"
    SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S13","S14","S15","S16","S17"]

    WINDOW_SIZE = 320
    STEP_SIZE = 160

    print("Loading dataset...")
    X, y, groups = preprocess_wesad_dataset_subjectwise(WESAD_ROOT, SUBJECTS, WINDOW_SIZE, STEP_SIZE)

    print("\nExtracting features...")
    X_feat = extract_features(X)
    print("Feature shape:", X_feat.shape)

    # -----------------------------
    # Different RF configs (Optimization Search)
    # -----------------------------
    configs = [
        {"name": "RF_Base_300", "n_estimators": 300, "max_depth": None, "min_samples_split": 2},
        {"name": "RF_200_Depth20", "n_estimators": 200, "max_depth": 20, "min_samples_split": 2},
        {"name": "RF_150_Depth15", "n_estimators": 150, "max_depth": 15, "min_samples_split": 2},
        {"name": "RF_100_Depth12", "n_estimators": 100, "max_depth": 12, "min_samples_split": 2},
        {"name": "RF_80_Depth10", "n_estimators": 80, "max_depth": 10, "min_samples_split": 2},
        {"name": "RF_50_Depth8", "n_estimators": 50, "max_depth": 8, "min_samples_split": 2},
    ]

    results = []

    print("\n==============================")
    print("RF OPTIMIZATION EXPERIMENT")
    print("==============================")

    for cfg in configs:
        rf_params = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "min_samples_split": cfg["min_samples_split"],
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": -1
        }

        print(f"\nRunning LOSO for: {cfg['name']} ...")

        avg_acc, avg_f1 = loso_eval_rf(X_feat, y, groups, rf_params)

        # Train one final model on ALL data (for size/time measurement)
        final_model = RandomForestClassifier(**rf_params)
        final_model.fit(X_feat, y)

        size_joblib_kb, size_gz_kb = save_and_get_size(final_model, cfg["name"])

        # Inference time test on small batch
        # (use first 2000 samples as test load)
        X_test_sample = X_feat[:2000]
        infer_ms = measure_inference_time(final_model, X_test_sample, runs=100)

        results.append({
            "Config": cfg["name"],
            "Avg_LOSO_Accuracy": avg_acc,
            "Avg_LOSO_MacroF1": avg_f1,
            "Model_Size_joblib_KB": size_joblib_kb,
            "Model_Size_gzip_KB": size_gz_kb,
            "Inference_ms(2000_samples)": infer_ms
        })

        print("Avg LOSO Acc:", avg_acc)
        print("Avg LOSO MacroF1:", avg_f1)
        print("Size joblib (KB):", size_joblib_kb)
        print("Size gzip (KB):", size_gz_kb)
        print("Inference avg ms:", infer_ms)

    df = pd.DataFrame(results)
    print("\n==============================")
    print("FINAL OPTIMIZATION TABLE")
    print("==============================")
    print(df)

    df.to_csv("rf_optimization_results.csv", index=False)
    print("\n✅ Saved: rf_optimization_results.csv")

    # Sort by best tradeoff: high F1 + low size
    df_sorted = df.sort_values(by=["Avg_LOSO_MacroF1", "Model_Size_gzip_KB"], ascending=[False, True])
    df_sorted.to_csv("rf_optimization_results_sorted.csv", index=False)
    print("✅ Saved: rf_optimization_results_sorted.csv")

