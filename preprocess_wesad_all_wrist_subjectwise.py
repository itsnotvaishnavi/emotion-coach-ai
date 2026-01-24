import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

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


def resample_1d(x, target_len):
    """Resample 1D signal to target length using index mapping."""
    if len(x) == target_len:
        return x
    idx = np.linspace(0, len(x) - 1, target_len).astype(int)
    return x[idx]


def resample_2d(x, target_len):
    """Resample 2D signal (N, C) to target length along axis 0."""
    if x.shape[0] == target_len:
        return x
    idx = np.linspace(0, x.shape[0] - 1, target_len).astype(int)
    return x[idx, :]


def extract_all_wrist_signals(subject_data):
    """
    Extract wrist signals and align them to SAME length.
    Wrist signals have different sampling rates in WESAD.
    We resample all to BVP length (highest sampling rate).
    Final shape: (N, 6)
    """
    wrist = subject_data["signal"]["wrist"]

    acc = wrist["ACC"].reshape(-1, 3)    # (Nacc, 3)
    bvp = wrist["BVP"].reshape(-1, 1)    # (Nbvp, 1)
    eda = wrist["EDA"].reshape(-1, 1)    # (Neda, 1)
    temp = wrist["TEMP"].reshape(-1, 1)  # (Ntemp, 1)

    # Choose reference length (BVP)
    target_len = bvp.shape[0]

    # Resample others to match BVP length
    acc_r = resample_2d(acc, target_len)
    eda_r = resample_2d(eda, target_len)
    temp_r = resample_2d(temp, target_len)

    signals = np.concatenate([acc_r, bvp, eda_r, temp_r], axis=1)  # (N, 6)
    return signals


def get_labels(subject_data):
    return subject_data["label"].reshape(-1)


def resample_labels_to_match_signals(labels, target_length):
    return resample_1d(labels, target_length)


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

    wrist_signals = extract_all_wrist_signals(data)  # (N, 6)
    labels = get_labels(data)                        # (N_label,)

    # Resample labels to wrist length
    labels_resampled = resample_labels_to_match_signals(labels, len(wrist_signals))

    # Filter 4 emotions
    signals_f, labels_f = filter_4_emotions(wrist_signals, labels_resampled)
    if len(signals_f) == 0:
        return np.array([]), np.array([])

    # Normalize
    signals_f = normalize_signals(signals_f)

    # Windowing
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
    print("X:", X_all.shape)  # (samples, 320, 6)
    print("y:", y_all.shape)
    print("groups:", groups_all.shape)

    return X_all, y_all, groups_all
