import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import shap
import xgboost as xgb


def load_artifacts(results_dir: Path):
    feature_names = json.loads((results_dir / "feature_names.json").read_text())
    stack_test_X = np.load(results_dir / "stack_test_X.npy")
    lookback = json.loads((results_dir / "lookback.json").read_text())["lookback"]
    models = {}
    for h in ["t1", "t3", "t5"]:
        m = xgb.Booster()
        m.load_model(str(results_dir / f"xgb_{h}.json"))
        models[h] = m
    return models, stack_test_X, feature_names, lookback


def compute_shap(models: Dict[str, xgb.Booster], X: np.ndarray):
    shap_values_dict = {}
    explainers_dict = {}
    for h, model in models.items():
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        shap_values_dict[h] = shap_vals
        explainers_dict[h] = explainer
    return shap_values_dict, explainers_dict
