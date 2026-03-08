import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from preprocess_wesad_all_wrist_subjectwise import preprocess_wesad_dataset_subjectwise
from features_all_wrist import extract_features


def loso_rf_eval(X_feat, y, groups):
    subjects = np.unique(groups)
    accs, f1s = [], []

    for sub in subjects:
        train_idx = groups != sub
        test_idx = groups == sub

        X_train, y_train = X_feat[train_idx], y[train_idx]
        X_test, y_test = X_feat[test_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, average="macro"))

    return np.mean(accs), np.mean(f1s)


if __name__ == "__main__":

    WESAD_ROOT = r"C:\Users\lenovo\Desktop\minor\WESAD"
    SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11",
                "S13","S14","S15","S16","S17"]

    window_configs = [
        {"samples": 160, "seconds": 5},
        {"samples": 320, "seconds": 10},
        {"samples": 640, "seconds": 20},
    ]

    results = []

    for cfg in window_configs:
        print(f"\nRunning window size: {cfg['samples']} samples ({cfg['seconds']} sec)")

        X, y, groups = preprocess_wesad_dataset_subjectwise(
            WESAD_ROOT,
            SUBJECTS,
            window_size=cfg["samples"],
            step_size=cfg["samples"] // 2
        )

        print("Total windows:", len(X))

        X_feat = extract_features(X)

        acc, f1 = loso_rf_eval(X_feat, y, groups)

        results.append({
            "Window_Samples": cfg["samples"],
            "Window_Seconds": cfg["seconds"],
            "LOSO_Accuracy": round(acc, 4),
            "LOSO_Macro_F1": round(f1, 4)
        })

        print("Accuracy:", round(acc, 4))
        print("Macro F1:", round(f1, 4))

    df = pd.DataFrame(results)
    print("\n==============================")
    print("WINDOW LENGTH SENSITIVITY RESULTS")
    print("==============================")
    print(df)

    df.to_csv("window_length_sensitivity_results.csv", index=False)
    print("\n✅ Saved: window_length_sensitivity_results.csv")
