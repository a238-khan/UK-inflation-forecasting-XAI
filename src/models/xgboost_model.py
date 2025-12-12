from typing import Dict
import xgboost as xgb
import numpy as np


def train_xgb_stack(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict,
):
    """Train an XGBoost model for stacking with clean parameter handling.

    Removes sklearn-style or train-argument parameters from `params` to avoid
    XGBoost warnings, while still honoring values like `n_estimators` by mapping
    them to `num_boost_round`.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Map/derive training loop arguments
    num_boost_round = params.get("num_boost_round", params.get("n_estimators", 2000))
    early_stopping_rounds = params.get("early_stopping_rounds", 75)

    # Sanitize params to avoid XGBoost "not used" warnings
    clean_params = {
        k: v
        for k, v in params.items()
        if k not in {"num_boost_round", "early_stopping_rounds", "n_estimators"}
    }

    model = xgb.train(
        params=clean_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return model


def predict_xgb(model: xgb.Booster, X: np.ndarray) -> np.ndarray:
    return model.predict(xgb.DMatrix(X))
