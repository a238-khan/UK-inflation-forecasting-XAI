from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb

from ..data.loader import load_model_ready_dataset
from ..data.preprocessor import temporal_splits, scale_features
from ..features.lag_features import create_sequences
from .xgboost_model import train_xgb_stack, predict_xgb


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 0.001
    batch_size: int = 16
    num_epochs: int = 150
    patience: int = 15


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, cfg: LSTMConfig, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            cfg.hidden_size,
            cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
        )
        self.drop1 = nn.Dropout(cfg.dropout)
        self.fc1 = nn.Linear(cfg.hidden_size, 32)
        self.drop2 = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        x = self.drop1(last)
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


def train_hybrid_b(config: Dict, results_dir: Path) -> Dict:
    df = load_model_ready_dataset()
    feature_cols: List[str] = config["features"]["feature_cols"]
    target_cols: List[str] = config["features"]["target_cols"]

    train_end = config["data"]["train_end"]
    val_end = config["data"]["val_end"]
    lookback = config["model"]["lookback"]

    train_mask, val_mask, test_mask = temporal_splits(df, train_end, val_end)

    X_all = df[feature_cols].values
    y_all = df[target_cols].values

    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = scale_features(
        X_all, y_all, train_mask, val_mask, test_mask
    )

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback)

    X_train_tab = X_train[lookback:]
    X_val_tab = X_val[lookback:]
    X_test_tab = X_test[lookback:]

    lstm_cfg = LSTMConfig(**config["model"]["lstm"])
    input_size = X_train_seq.shape[2]
    model = LSTMForecaster(input_size=input_size, cfg=lstm_cfg)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_cfg.lr)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32)),
        batch_size=lstm_cfg.batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32)),
        batch_size=lstm_cfg.batch_size,
        shuffle=False,
    )

    best_val = float("inf")
    pat = 0
    for epoch in range(lstm_cfg.num_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict().copy()
            pat = 0
        else:
            pat += 1
        if pat >= lstm_cfg.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_train_pred_scaled = model(torch.tensor(X_train_seq, dtype=torch.float32)).cpu().numpy()
        y_val_pred_scaled = model(torch.tensor(X_val_seq, dtype=torch.float32)).cpu().numpy()
        y_test_pred_scaled = model(torch.tensor(X_test_seq, dtype=torch.float32)).cpu().numpy()

    stack_train_X = np.hstack([X_train_tab, y_train_pred_scaled])
    stack_val_X = np.hstack([X_val_tab, y_val_pred_scaled])
    stack_test_X = np.hstack([X_test_tab, y_test_pred_scaled])

    feature_names = feature_cols + ["LSTM_t1", "LSTM_t3", "LSTM_t5"]

    horizons = ["t1", "t3", "t5"]
    horizon_idx = {"t1": 0, "t3": 1, "t5": 2}
    xgb_params = config["model"]["xgb"]

    models = {}
    for h in horizons:
        idx = horizon_idx[h]
        models[h] = train_xgb_stack(
            stack_train_X,
            y_train_pred_scaled[:, idx],
            stack_val_X,
            y_val_pred_scaled[:, idx],
            {
                "learning_rate": xgb_params["learning_rate"],
                "max_depth": xgb_params["max_depth"],
                "n_estimators": xgb_params["n_estimators"],
                "subsample": xgb_params["subsample"],
                "colsample_bytree": xgb_params["colsample_bytree"],
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "num_boost_round": xgb_params.get("num_boost_round", 2000),
                "early_stopping_rounds": xgb_params.get("early_stopping_rounds", 75),
            },
        )

    # Save artifacts
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "feature_names.json").write_text(json.dumps(feature_names))
    np.save(results_dir / "stack_test_X.npy", stack_test_X)
    np.save(results_dir / "test_dates.npy", df["date"][df["date"] > val_end].values[lookback:])
    (results_dir / "lookback.json").write_text(json.dumps({"lookback": lookback}))

    for h in horizons:
        models[h].save_model(str(results_dir / f"xgb_{h}.json"))

    return {
        "models": models,
        "stack_test_X": stack_test_X,
        "feature_names": feature_names,
        "lookback": lookback,
    }
