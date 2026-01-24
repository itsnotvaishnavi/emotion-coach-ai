import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import your preprocessing function
from preprocess_wesad import preprocess_wesad_dataset, LABEL_NAMES


# -----------------------------
# Feature Extraction (Simple)
# -----------------------------
def extract_features(X):
    """
    X shape: (samples, window_size, channels)
    We extract very simple features:
    mean, std, min, max for each channel
    """
    feats = []

    for sample in X:
        # sample shape: (320, 2)
        channel_features = []
        for ch in range(sample.shape[1]):
            sig = sample[:, ch]
            channel_features.extend([
                np.mean(sig),
                np.std(sig),
                np.min(sig),
                np.max(sig),
            ])
        feats.append(channel_features)

    return np.array(feats)


# -----------------------------
# Main Training Script
# -----------------------------
if __name__ == "__main__":

    # Load dataset (same as preprocessing)
    WESAD_ROOT = r"C:\Users\lenovo\Desktop\minor\WESAD"
    SUBJECTS = ["S2", "S3"]   # for now (later add more subjects)

    WINDOW_SIZE = 320
    STEP_SIZE = 160

    X, y = preprocess_wesad_dataset(WESAD_ROOT, SUBJECTS, WINDOW_SIZE, STEP_SIZE)

    # Extract features
    X_feat = extract_features(X)
    print("\nFeature shape:", X_feat.shape)

    # Train-test split (simple for now)
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Models
    # -----------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM (RBF)": SVC(kernel="rbf"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    # -----------------------------
    # Train + Evaluate
    # -----------------------------
    for name, model in models.items():
        print("\n==============================")
        print("MODEL:", name)
        print("==============================")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        print("Accuracy:", round(acc, 4))
        print("Macro F1:", round(macro_f1, 4))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=list(LABEL_NAMES.values())))
