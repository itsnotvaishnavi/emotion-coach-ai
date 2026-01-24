import numpy as np

def extract_features(X):
    """
    X shape: (samples, window_size, channels)
    For each channel we extract:
    mean, std, min, max, median, rms, energy, peak-to-peak
    """
    feats = []

    for sample in X:
        ch_feats = []
        for ch in range(sample.shape[1]):
            sig = sample[:, ch]

            mean = np.mean(sig)
            std = np.std(sig)
            mn = np.min(sig)
            mx = np.max(sig)
            med = np.median(sig)
            rms = np.sqrt(np.mean(sig ** 2))
            energy = np.sum(sig ** 2)
            ptp = mx - mn

            ch_feats.extend([mean, std, mn, mx, med, rms, energy, ptp])

        feats.append(ch_feats)

    return np.array(feats, dtype=np.float32)
