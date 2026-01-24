import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    return np.stack([eda, temp], axis=1)


def get_labels(subject_data):
    return subject_data["label"].reshape(-1)


def resample_labels_to_match_signals(labels, target_length):
    if len(labels) == target_length:
        return labels
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
    return scaler.fit_transform(signals)


def create_windows(signals, labels, window_size=320, step_size=160):
    X, y = [], []
    N = len(signals)

    for start in range(0, N - window_size + 1, step_size):
        end = start + window_size

        w_sig = signals[start:end]
        w_lab = labels[start:end]

        values, counts = np.unique(w_lab, return_counts=True)
        maj = values[np.argmax(counts)]

        X.append(w_sig)
        y.append(maj)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def preprocess_subject(pkl_path, window_size=320, step_size=160):
    data = load_subject_pkl(pkl_path)

    wrist_signals = extract_wrist_signals(data)
    labels = get_labels(data)

    labels_resampled = resample_labels_to_match_signals(labels, len(wrist_signals))

    signals_f, labels_f = filter_4_emotions(wrist_signals, labels_resampled)
    if len(signals_f) == 0:
        return np.array([]), np.array([])

    signals_f = normalize_signals(signals_f)

    X, y = create_windows(signals_f, labels_f, window_size, step_size)
    return X, y


def preprocess_wesad_dataset_subjectwise(wesad_root, subjects, window_size=320, step_size=160):
    X_all, y_all, groups_all = [], [], []

    for sub in subjects:
        pkl_path = os.path.join(wesad_root, sub, f"{sub}.pkl")

        print(f"\nProcessing {sub} -> {pkl_path}")

        if not os.path.exists(pkl_path):
            print(f"❌ Missing: {pkl_path}")
            continue

        X, y = preprocess_subject(pkl_path, window_size, step_size)

        if len(X) == 0:
            print(f"⚠️ Skipping {sub} (no windows)")
            continue

        print(f"✅ {sub} windows: {len(X)}")

        groups = np.array([sub] * len(X))

        X_all.append(X)
        y_all.append(y)
        groups_all.append(groups)

    if len(X_all) == 0:
        return np.array([]), np.array([]), np.array([])

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    groups_all = np.concatenate(groups_all, axis=0)

    print("\nFINAL SHAPES:")
    print("X:", X_all.shape)
    print("y:", y_all.shape)
    print("groups:", groups_all.shape)

    return X_all, y_all, groups_all
