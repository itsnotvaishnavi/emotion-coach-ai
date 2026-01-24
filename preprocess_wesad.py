import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Target classes
# 1 = baseline (Neutral)
# 2 = stress
# 3 = amusement (Happiness)
# 4 = meditation (Relaxation)

LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}

LABEL_NAMES = {
    0: "Neutral",
    1: "Stress",
    2: "Happiness",
    3: "Relaxation"
}


def load_subject_pkl(subject_path):
    with open(subject_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def extract_wrist_signals(subject_data):
    wrist = subject_data["signal"]["wrist"]
    eda = wrist["EDA"].reshape(-1)
    temp = wrist["TEMP"].reshape(-1)
    signals = np.stack([eda, temp], axis=1)  # (N_wrist, 2)
    return signals


def get_labels(subject_data):
    labels = subject_data["label"].reshape(-1)
    return labels


def resample_labels_to_match_signals(labels, target_length):
    """
    Resample label array to match wrist signal length.
    This is needed because labels are aligned with chest timeline.
    """
    if len(labels) == target_length:
        return labels

    # Map indices from target length to original label length
    idx = np.linspace(0, len(labels) - 1, target_length).astype(int)
    return labels[idx]


def filter_4_emotions(signals, labels):
    mask = np.isin(labels, list(LABEL_MAP.keys()))
    signals = signals[mask]
    labels = labels[mask]

    if len(signals) == 0:
        return np.array([]), np.array([])

    mapped = np.array([LABEL_MAP[int(x)] for x in labels], dtype=np.int64)
    return signals, mapped


def normalize_signals(signals):
    scaler = StandardScaler()
    return scaler.fit_transform(signals), scaler


def create_windows(signals, labels, window_size=320, step_size=160):
    X, y = [], []
    N = len(signals)

    for start in range(0, N - window_size + 1, step_size):
        end = start + window_size

        w_sig = signals[start:end]
        w_lab = labels[start:end]

        # majority label in window
        values, counts = np.unique(w_lab, return_counts=True)
        maj = values[np.argmax(counts)]

        X.append(w_sig)
        y.append(maj)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def preprocess_subject(pkl_path, window_size=320, step_size=160):
    data = load_subject_pkl(pkl_path)

    wrist_signals = extract_wrist_signals(data)   # (N_wrist, 2)
    labels = get_labels(data)                     # (N_label,)

    print("Wrist signal length:", len(wrist_signals))
    print("Original label length:", len(labels))
    print("Unique labels ORIGINAL:", np.unique(labels))

    # Resample labels to wrist length
    labels_resampled = resample_labels_to_match_signals(labels, len(wrist_signals))

    print("Unique labels RESAMPLED:", np.unique(labels_resampled))

    # Filter 4 emotions
    signals_f, labels_f = filter_4_emotions(wrist_signals, labels_resampled)

    print("Unique labels AFTER filtering:", np.unique(labels_f) if len(labels_f) > 0 else [])

    if len(signals_f) == 0:
        return np.array([]), np.array([])

    signals_f, _ = normalize_signals(signals_f)

    X, y = create_windows(signals_f, labels_f, window_size, step_size)
    return X, y


def preprocess_wesad_dataset(wesad_root, subjects, window_size=320, step_size=160):
    X_all, y_all = [], []

    for sub in subjects:
        pkl_path = os.path.join(wesad_root, sub, f"{sub}.pkl")

        print("\n==============================")
        print(f"Processing {sub} -> {pkl_path}")
        print("==============================")

        if not os.path.exists(pkl_path):
            print("❌ File not found:", pkl_path)
            continue

        X, y = preprocess_subject(pkl_path, window_size, step_size)

        if len(X) == 0:
            print(f"⚠️ Skipping {sub} (no valid windows)")
            continue

        print(f"✅ Windows created for {sub}: {len(X)}")

        X_all.append(X)
        y_all.append(y)

    if len(X_all) == 0:
        print("\n❌ No valid data created.")
        return np.array([]), np.array([])

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    print("\n==============================")
    print("FINAL DATASET SHAPE")
    print("==============================")
    print("X shape:", X_all.shape)
    print("y shape:", y_all.shape)

    unique, counts = np.unique(y_all, return_counts=True)
    print("\nLabel Distribution:")
    for u, c in zip(unique, counts):
        print(f"{LABEL_NAMES[int(u)]}: {c}")

    return X_all, y_all


if __name__ == "__main__":
    WESAD_ROOT = r"C:\Users\lenovo\Desktop\minor\WESAD"
    SUBJECTS = ["S2", "S3"]

    WINDOW_SIZE = 320  # 10 sec at 32Hz
    STEP_SIZE = 160    # 50% overlap

    X, y = preprocess_wesad_dataset(WESAD_ROOT, SUBJECTS, WINDOW_SIZE, STEP_SIZE)
