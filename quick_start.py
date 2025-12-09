"""
Quick Start Guide for UK Inflation Forecasting Project

This script demonstrates how to quickly get started with the project.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def quick_demo():
    """
    Run a quick demonstration of the key features
    """
    from data_collection import DataCollector
    from data_preprocessing import DataPreprocessor
    
    print("="*70)
    print("UK INFLATION FORECASTING - QUICK START DEMO")
    print("="*70)
    
    # 1. Collect sample data
    print("\n1. Collecting sample data...")
    collector = DataCollector(output_dir='data/raw')
    df = collector.collect_all_data()
    print(f"   ✓ Collected {len(df)} observations")
    print(f"   ✓ Features: {', '.join(df.columns.tolist()[:5])}...")
    
    # 2. Show data summary
    print("\n2. Data Summary:")
    print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   - Average inflation: {df['cpi_inflation'].mean():.2f}%")
    print(f"   - Inflation range: {df['cpi_inflation'].min():.2f}% to {df['cpi_inflation'].max():.2f}%")
    
    # 3. Preprocess data
    print("\n3. Preprocessing data...")
    preprocessor = DataPreprocessor(input_dir='data/raw', output_dir='data/processed')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_all()
    print(f"   ✓ Train set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    print(f"   ✓ Features: {X_train.shape[1]}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run 'python run_pipeline.py' to train models and generate results")
    print("2. Check 'results/plots/' for visualizations")
    print("3. Check 'models/' for trained models")
    print("4. Open 'notebooks/exploratory_analysis.ipynb' for detailed analysis")
    

if __name__ == "__main__":
    quick_demo()
