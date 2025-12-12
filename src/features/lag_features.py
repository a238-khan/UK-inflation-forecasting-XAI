import numpy as np


def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i : i + lookback])
        y_seq.append(y[i + lookback])
    return np.array(X_seq), np.array(y_seq)
