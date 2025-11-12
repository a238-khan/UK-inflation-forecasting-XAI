"""
Main Pipeline for UK Inflation Forecasting with XAI

This script runs the complete pipeline:
1. Data Collection
2. Data Preprocessing
3. Model Training
4. Model Evaluation
5. XAI Explanations
6. Visualization
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import DataCollector
from data_preprocessing import DataPreprocessor
from models import InflationForecaster
from explainability import MultiModelExplainer
from visualization import Visualizer
import pandas as pd
import pickle


def main():
    """
    Run the complete inflation forecasting pipeline
    """
    print("="*70)
    print("UK INFLATION FORECASTING WITH MACHINE LEARNING AND XAI")
    print("="*70)
    
    # Step 1: Data Collection
    print("\n" + "="*70)
    print("STEP 1: DATA COLLECTION")
    print("="*70)
    collector = DataCollector(output_dir='data/raw')
    df_raw = collector.collect_all_data()
    
    # Step 2: Data Preprocessing
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)
    preprocessor = DataPreprocessor(input_dir='data/raw', output_dir='data/processed')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_all()
    
    # Step 3: Model Training and Evaluation
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING AND EVALUATION")
    print("="*70)
    forecaster = InflationForecaster(model_dir='models')
    results = forecaster.train_all_models(X_train, y_train, X_test, y_test)
    
    best_model_name, best_model = forecaster.get_best_model()
    print(f"\n{'*'*70}")
    print(f"Best Performing Model: {best_model_name}")
    print(f"RMSE: {results[best_model_name]['rmse']:.4f}")
    print(f"{'*'*70}")
    
    # Step 4: Visualization
    print("\n" + "="*70)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*70)
    
    test_df = pd.read_csv('data/processed/test_data.csv')
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    visualizer = Visualizer(output_dir='results/plots')
    visualizer.create_all_visualizations(
        df_raw, 
        results, 
        test_df['date'], 
        y_test
    )
    
    # Step 5: XAI Explanations
    print("\n" + "="*70)
    print("STEP 5: GENERATING XAI EXPLANATIONS")
    print("="*70)
    
    # Load tree-based models for XAI
    xai_models = {}
    for model_name in ['random_forest', 'xgboost', 'lightgbm']:
        xai_models[model_name] = forecaster.models[model_name]
    
    multi_explainer = MultiModelExplainer(
        xai_models, 
        X_train, 
        X_test, 
        X_train.columns.tolist(),
        output_dir='results/plots'
    )
    
    multi_explainer.explain_all_models()
    multi_explainer.compare_feature_importance()
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults Summary:")
    print(f"- Raw data: data/raw/uk_inflation_data.csv")
    print(f"- Processed data: data/processed/")
    print(f"- Trained models: models/")
    print(f"- Visualizations: results/plots/")
    print(f"- Best model: {best_model_name} (RMSE: {results[best_model_name]['rmse']:.4f})")
    
    print("\n" + "="*70)
    print("Model Performance Summary:")
    print("="*70)
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  MAE: {result['mae']:.4f}")
        print(f"  R²: {result['r2']:.4f}")
        print(f"  MAPE: {result['mape']:.2f}%")
    
    print("\n" + "="*70)
    print("Thank you for using the UK Inflation Forecasting System!")
    print("="*70)


if __name__ == "__main__":
    main()
