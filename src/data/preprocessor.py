from typing import Tuple
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def temporal_splits(df: pd.DataFrame, train_end: str, val_end: str):
    train_mask = df["date"] <= train_end
    val_mask = (df["date"] > train_end) & (df["date"] <= val_end)
    test_mask = df["date"] > val_end
    return train_mask, val_mask, test_mask


def scale_features(
    X_all: pd.DataFrame,
    y_all: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_all[train_mask.values])
    X_val = scaler_X.transform(X_all[val_mask.values])
    X_test = scaler_X.transform(X_all[test_mask.values])

    y_train = scaler_y.fit_transform(y_all[train_mask.values])
    y_val = scaler_y.transform(y_all[val_mask.values])
    y_test = scaler_y.transform(y_all[test_mask.values])

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y
