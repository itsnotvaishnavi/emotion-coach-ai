import os
import joblib
import zipfile
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from preprocess_wesad import preprocess_wesad_dataset
from train_ml import extract_features  # reuse feature extraction


if __name__ == "__main__":
    WESAD_ROOT = r"C:\Users\lenovo\Desktop\minor\WESAD"
    SUBJECTS = ["S2", "S3"]

    WINDOW_SIZE = 320
    STEP_SIZE = 160

    # Load data
    X, y = preprocess_wesad_dataset(WESAD_ROOT, SUBJECTS, WINDOW_SIZE, STEP_SIZE)

    # Features
    X_feat = extract_features(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train best model
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("Random Forest Accuracy:", round(acc, 4))
    print("Random Forest Macro F1:", round(macro_f1, 4))

    # Save model
    model_path = "rf_wesad_model.pkl"
    joblib.dump(rf, model_path)

    size_bytes = os.path.getsize(model_path)
    print("\nSaved model:", model_path)
    print("Model size (bytes):", size_bytes)
    print("Model size (KB):", round(size_bytes / 1024, 2))

    # Zip compression (simple optimization)
    zip_path = "rf_wesad_model.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path)

    zip_size = os.path.getsize(zip_path)
    print("\nZipped model:", zip_path)
    print("Zipped size (bytes):", zip_size)
    print("Zipped size (KB):", round(zip_size / 1024, 2))
