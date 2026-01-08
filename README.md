# UK Inflation Forecasting — XAI

This repository forecasts UK CPI (YoY) across multiple horizons and applies explainable AI (XAI) techniques (SHAP and LIME) for model interpretation. The workflow is notebook-driven and artifact-based for reproducibility.

## Project Structure
- Data: `data/raw/` (source files), `data/processed/` (cleaned/merged), `notebooks/model_ready_dataset.csv` (final tabular dataset).
- Notebooks (primary):
	- `notebooks/01_eda_analysis.ipynb` — exploratory data analysis.
	- `notebooks/02_feature_engineering.ipynb` — feature creation and dataset assembly.
	- `notebooks/03_baseline_models.ipynb` — statistical baselines (e.g., ARIMA) and setup.
	- `notebooks/04_ml_models.ipynb` — Random Forest multi-output baseline, metrics, and saved artifacts.
	- `notebooks/06_evaluation.ipynb` — artifact-based evaluation and RF vs Hybrid comparison; plots and tables.

Older or deprecated notebooks (root-level `05_ensemble_methods.ipynb` and `06_shap_analysis.ipynb`) have been removed in favor of the curated notebooks above. XAI is handled within the evaluation flow using SHAP and LIME (see artifacts and analysis cells).

## Artifacts
- Random Forest baseline: `artifacts/rf_baseline_metrics.csv`, plots in `artifacts/rf_baseline_plots/` and predictions in `results/baselines/rf/`.
- Hybrid (LSTM–XGBoost): predictions per horizon in `results/hybrid/` (`xgb_cpi_t1_predictions.csv`, `xgb_cpi_t3_predictions.csv`, `xgb_cpi_t5_predictions.csv`).

## Explainable AI (XAI)
- SHAP and LIME analyses are integrated in the evaluation and reporting workflow to interpret feature contributions and local explanations. Generated explainability artifacts (if any) live under `artifacts/explainability/`.

## Data Sources
- Office for National Statistics (ONS): CPI and related macroeconomic indicators.
- Bank of England (BoE): policy rate and selected macro/market series.

Source files are stored in `data/raw/` (e.g., `ons_cpi.csv`, `boe_interest.csv`, `exchange_rates.csv`), then cleaned and merged into `data/processed/` before model-ready assembly.

## Reproducibility Notes
- All evaluation in `notebooks/06_evaluation.ipynb` is artifact-based — metrics and plots are computed from saved predictions to ensure consistent comparisons across models and runs.
- Configurations (features and targets) are defined in `configs/hybrid.yaml` and are used consistently across notebooks.
