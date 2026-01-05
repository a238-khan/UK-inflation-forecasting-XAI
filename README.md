# UK Inflation Forecasting XAI

This repo is organized as a simple, connected pipeline so that each stage feeds the next:

- Raw data (`data/raw/`) → cleaned/merged (`data/processed/`) → model-ready (`notebooks/model_ready_dataset.csv`).
- Feature engineering and baseline modeling notebooks lead into the ensemble training (`notebooks/05_ensemble_methods.ipynb`).
- Ensemble training saves Hybrid Model artifacts under `results/hybrid_b/` (models, feature names, stacked test features).
- Explainability (`notebooks/06_shap_analysis.ipynb`) loads those artifacts to compute SHAP/LIME.

## Run Order
1. `notebooks/02_feature_engineering.ipynb`: produce model-ready dataset.
2. `notebooks/05_ensemble_methods.ipynb`: run the pipeline cell to train the Hybrid Model and save artifacts.
3. `notebooks/06_shap_analysis.ipynb`: verify artifacts (top diagnostics cell) and run SHAP/LIME.

## Hybrid Model Artifacts
Saved in `results/hybrid_b/`:
- `feature_names.json`, `stack_test_X.npy`, `lookback.json`, `test_dates.npy`
- `xgb_t1.json`, `xgb_t3.json`, `xgb_t5.json`

If any are missing, re-run the pipeline in `05_ensemble_methods.ipynb`.

## Config-Driven
Model parameters and data splits are defined in `configs/hybrid_b.yaml`. Update the config, re-run the ensemble notebook, and explainability will reflect the changes automatically.
