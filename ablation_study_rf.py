import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from preprocess_wesad_all_wrist_subjectwise import preprocess_wesad_dataset_subjectwise
from features_all_wrist import extract_features


# ----------------------------------------
# Channel indices (IMPORTANT)
# Order used in preprocessing:
# [ACCx, ACCy, ACCz, BVP, EDA, TEMP]
# ----------------------------------------
CHANNEL_MAP = {
    "ACC": [0, 1, 2],
    "BVP": [3],
    "EDA": [4],
    "TEMP": [5],
    "EDA_BVP": [3, 4],
    "EDA_BVP_TEMP": [3, 4, 5],
    "ALL": [0, 1, 2, 3, 4, 5]
}


def loso_rf_eval(X, y, groups):
    subjects = np.unique(groups)
    accs, f1s = [], []

    for sub in subjects:
        train_idx = groups != sub
        test_idx = groups == sub

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

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
    SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S13","S14","S15","S16","S17"]

    print("Loading full dataset...")
    X, y, groups = preprocess_wesad_dataset_subjectwise(
        WESAD_ROOT, SUBJECTS, window_size=320, step_size=160
    )

    results = []

    for name, ch_idx in CHANNEL_MAP.items():
        print(f"\nRunning ablation: {name}")

        # Select channels
        X_sel = X[:, :, ch_idx]

        # Extract features
        X_feat = extract_features(X_sel)

        acc, f1 = loso_rf_eval(X_feat, y, groups)

        results.append({
            "Signals Used": name,
            "LOSO Accuracy": round(acc, 4),
            "LOSO Macro F1": round(f1, 4)
        })

        print("Accuracy:", round(acc, 4))
        print("Macro F1:", round(f1, 4))

    df = pd.DataFrame(results)
    print("\n==============================")
    print("ABLATION STUDY RESULTS")
    print("==============================")
    print(df)

    df.to_csv("ablation_study_results.csv", index=False)
    print("\n✅ Saved: ablation_study_results.csv")
