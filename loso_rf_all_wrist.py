import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from preprocess_wesad_all_wrist_subjectwise import preprocess_wesad_dataset_subjectwise
from features_all_wrist import extract_features


if __name__ == "__main__":

    WESAD_ROOT = r"C:\Users\lenovo\Desktop\minor\WESAD"

    # All subjects you have (based on your screenshot)
    SUBJECTS = ["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S13","S14","S15","S16","S17"]

    WINDOW_SIZE = 320
    STEP_SIZE = 160

    # Load dataset
    X, y, groups = preprocess_wesad_dataset_subjectwise(WESAD_ROOT, SUBJECTS, WINDOW_SIZE, STEP_SIZE)

    # Features
    X_feat = extract_features(X)
    print("\nFeature shape:", X_feat.shape)

    results = []
    unique_subjects = np.unique(groups)

    for test_sub in unique_subjects:
        train_mask = groups != test_sub
        test_mask = groups == test_sub

        X_train, y_train = X_feat[train_mask], y[train_mask]
        X_test, y_test = X_feat[test_mask], y[test_mask]

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        results.append({
            "Test_Subject": test_sub,
            "Accuracy": acc,
            "Macro_F1": macro_f1,
            "Test_Samples": len(y_test)
        })

        print(f"\nTest Subject: {test_sub}")
        print("Samples:", len(y_test))
        print("Accuracy:", round(acc, 4))
        print("Macro F1:", round(macro_f1, 4))

    df = pd.DataFrame(results)

    print("\n==============================")
    print("LOSO RESULTS (ALL WRIST SIGNALS)")
    print("==============================")
    print(df)

    print("\nAverage Accuracy:", round(df["Accuracy"].mean(), 4))
    print("Average Macro F1:", round(df["Macro_F1"].mean(), 4))

    df.to_csv("loso_results_rf_all_wrist.csv", index=False)
    print("\n✅ Saved: loso_results_rf_all_wrist.csv")
