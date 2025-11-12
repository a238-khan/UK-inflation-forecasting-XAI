# UK Inflation Forecasting using Machine Learning and XAI

MSc Data Science Project

## Project Overview

This project implements a comprehensive system for forecasting UK inflation using machine learning algorithms and Explainable AI (XAI) methods. The project demonstrates the complete data science workflow from data collection to model interpretation.

### Key Features

1. **Data Collection**: Automated collection of UK economic indicators including CPI inflation, GDP growth, unemployment rate, interest rates, and more
2. **Machine Learning Models**: Implementation of multiple algorithms including:
   - Linear Regression
   - Ridge Regression
   - Random Forest
   - XGBoost
   - LightGBM
   - LSTM Neural Networks
3. **Explainable AI (XAI)**: Model interpretability using:
   - SHAP (SHapley Additive exPlanations)
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Feature Importance Analysis
   - Partial Dependence Plots
4. **Comprehensive Visualizations**: Time series plots, model comparisons, residual analysis, and more

## Project Structure

```
UK-inflation-forecasting-XAI/
│
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Preprocessed data
│
├── src/
│   ├── __init__.py
│   ├── data_collection.py      # Data collection module
│   ├── data_preprocessing.py   # Data preprocessing and feature engineering
│   ├── models.py               # Machine learning models
│   ├── explainability.py       # XAI methods (SHAP, LIME)
│   └── visualization.py        # Visualization utilities
│
├── notebooks/                  # Jupyter notebooks for analysis
│
├── models/                     # Saved trained models
│
├── results/
│   ├── plots/                  # Generated visualizations
│   └── reports/                # Analysis reports
│
├── run_pipeline.py             # Main pipeline script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/a238-khan/UK-inflation-forecasting-XAI.git
cd UK-inflation-forecasting-XAI
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

To run the entire forecasting pipeline (data collection, preprocessing, training, evaluation, and XAI):

```bash
python run_pipeline.py
```

This will:
1. Collect and prepare the data
2. Train multiple ML models
3. Evaluate model performance
4. Generate visualizations
5. Create XAI explanations

### Running Individual Modules

You can also run individual components:

**Data Collection:**
```bash
cd src
python data_collection.py
```

**Data Preprocessing:**
```bash
cd src
python data_preprocessing.py
```

**Model Training:**
```bash
cd src
python models.py
```

**XAI Analysis:**
```bash
cd src
python explainability.py
```

**Visualizations:**
```bash
cd src
python visualization.py
```

## Results

After running the pipeline, you will find:

- **Trained Models**: Saved in `models/` directory
- **Visualizations**: All plots saved in `results/plots/`
- **Data**: Raw and processed data in `data/` directory

### Model Performance Metrics

The system evaluates models using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared Score)
- **MAPE** (Mean Absolute Percentage Error)

### XAI Outputs

The XAI module generates:
- Feature importance rankings
- SHAP summary plots
- SHAP bar plots
- LIME explanations for individual predictions
- Partial dependence plots
- Multi-model feature importance comparisons

## Methodology

### 1. Data Collection
- Collection of UK economic indicators from multiple sources
- Time series data spanning multiple years
- Features include: CPI inflation, GDP growth, unemployment, interest rates, exchange rates, oil prices, wage growth, and retail sales

### 2. Feature Engineering
- Lag features (1, 2, 3, 6, 12 months)
- Rolling statistics (mean, standard deviation)
- Time-based features (year, month, quarter)
- Cyclical encoding of temporal features

### 3. Model Training
Multiple models are trained and compared:
- **Linear Models**: For baseline performance
- **Tree-based Models**: Random Forest, XGBoost, LightGBM for non-linear relationships
- **Neural Networks**: LSTM for capturing temporal dependencies

### 4. Model Evaluation
- Time series cross-validation
- Multiple performance metrics
- Residual analysis
- Confidence intervals

### 5. Explainability
- **SHAP**: Global and local feature importance
- **LIME**: Local explanations for individual predictions
- **Feature Importance**: Tree-based model insights
- **Partial Dependence**: Understanding feature effects

## Dependencies

Key libraries used:
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm, tensorflow
- **XAI**: shap, lime
- **Visualization**: matplotlib, seaborn, plotly
- **Statistical Analysis**: statsmodels, scipy

See `requirements.txt` for complete list.

## Future Enhancements

- Integration with real-time data APIs (Bank of England, ONS)
- Additional models (Prophet, ARIMA, Transformer-based)
- Hyperparameter optimization
- Ensemble methods
- Web-based dashboard for interactive exploration
- Automated model retraining pipeline

## Project Context

This project was developed as part of an MSc Data Science program, demonstrating:
- End-to-end machine learning pipeline development
- Time series forecasting techniques
- Model interpretability and explainability
- Best practices in data science project organization
- Comprehensive documentation and code quality

## License

This project is part of academic work for MSc Data Science.

## Contact

For questions or collaboration opportunities, please open an issue in the GitHub repository.

## Acknowledgments

- UK Office for National Statistics (ONS)
- Bank of England
- Open-source ML and XAI communities 
