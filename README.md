# UK Inflation Forecasting — XAI

Forecasts UK CPI (YoY) at multiple horizons and applies explainable AI (SHAP, LIME) for interpretation. The workflow is notebook-driven and produces versionable artifacts (predictions, metrics, plots) for reproducibility.

## Project Structure
- data/raw: raw source files (ONS, BoE, etc.)
- data/processed: cleaned/merged datasets used by notebooks
- notebooks: end-to-end workflow (EDA → features → baselines → ML → tuning → evaluation)
- results: saved model predictions and serialized artifacts (per model/horizon)
- artifacts: metrics and plots generated from notebooks
- configs: configuration for features/targets/horizons (hybrid.yaml)
- src: modular code for data, features, models, evaluation, explainability, and utils

## Notebooks
- notebooks/01_eda_analysis.ipynb: exploratory data analysis
- notebooks/02_feature_engineering.ipynb: feature creation and dataset assembly (writes notebooks/model_ready_dataset.csv)
- notebooks/03_baseline_models.ipynb: statistical and ML baselines setup
- notebooks/04_ml_models.ipynb: RandomForest/XGBoost baselines; saves metrics and predictions
- notebooks/05_Hypertune_hybrid_method.ipynb: tuning and training hybrid model; saves predictions
- notebooks/06_evaluation.ipynb: artifact-based comparison across models/horizons

Additional analysis notebooks at the repo root:
- 05_ensemble_methods.ipynb: ensemble/stacking experiments over saved predictions
- 06_shap_analysis.ipynb: focused SHAP analysis for feature attributions

## Artifacts and Results
- Baselines — results/baselines/
  - rf/: RF models, params, and predictions (e.g., rf_test_predictions.csv)
  - xgb/: XGBoost models, params, and predictions (e.g., xgb_test_predictions.csv)
- RF metrics and plots — artifacts/
  - rf_baseline_metrics.csv
  - rf_baseline_plots/
- Hybrid predictions — results/hybrid/
  - xgb_cpi_t1_predictions.csv, xgb_cpi_t3_predictions.csv, xgb_cpi_t5_predictions.csv (plus companion arrays/configs)
- Explainability outputs — artifacts/explainability/

## Data
- Place source CSVs in data/raw (e.g., ons_cpi.csv, boe_interest.csv, exchange_rates.csv, Unemployment_Rate.csv, GDP_growth_Rate.csv, petrol_oil_everage_price_change.csv).
- Processed datasets are written to data/processed and the model-ready table to notebooks/model_ready_dataset.csv by the feature engineering notebook.

## Configuration
- configs/hybrid.yaml centralizes features, targets, horizons, and related options used across notebooks.

## How to Run (local)
1) Create and activate a virtual environment (Windows PowerShell):

```powershell
py -3 -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

2) Install required Python packages (typical set used in notebooks):

```powershell
pip install pandas numpy scikit-learn xgboost shap lime matplotlib seaborn statsmodels joblib pyyaml jupyter ipykernel
```

3) Open and execute notebooks in order (01 → 06). Intermediate artifacts will be written under results/ and artifacts/ as described above.

## Reproducibility
- Evaluation notebooks read saved predictions/metrics to ensure consistent comparisons across runs and dates.
- Where applicable, configurations are managed via configs/hybrid.yaml for consistent feature/target definitions.
