"""
Example Usage Script for UK Inflation Forecasting Project

This script demonstrates various ways to use the modules in this project.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def example_1_data_collection():
    """Example 1: Collect and save data"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Data Collection")
    print("="*70)
    
    from data_collection import DataCollector
    
    # Create collector
    collector = DataCollector(output_dir='data/raw')
    
    # Collect inflation data
    inflation_df = collector.collect_inflation_data()
    print(f"\nInflation data collected: {len(inflation_df)} observations")
    print(inflation_df.head())
    
    # Collect economic indicators
    economic_df = collector.collect_economic_indicators()
    print(f"\nEconomic indicators collected: {len(economic_df)} observations")
    print(economic_df.head())
    
    # Merge and save all data
    merged_df = collector.collect_all_data()
    print(f"\nComplete dataset: {merged_df.shape}")


def example_2_data_preprocessing():
    """Example 2: Preprocess data for modeling"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Data Preprocessing")
    print("="*70)
    
    from data_preprocessing import DataPreprocessor
    
    # Create preprocessor
    preprocessor = DataPreprocessor(input_dir='data/raw', output_dir='data/processed')
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_all()
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"\nFeatures created: {X_train.shape[1]}")
    print(f"Feature names: {list(X_train.columns[:5])}...")


def example_3_individual_preprocessing_steps():
    """Example 3: Use individual preprocessing steps"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Individual Preprocessing Steps")
    print("="*70)
    
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor(input_dir='data/raw', output_dir='data/processed')
    
    # Load data
    df = preprocessor.load_data()
    print(f"Loaded data: {df.shape}")
    
    # Create lag features
    df_with_lags = preprocessor.create_lag_features(df, lags=[1, 3, 6])
    print(f"\nAfter adding lag features: {df_with_lags.shape}")
    print(f"New columns: {[col for col in df_with_lags.columns if 'lag' in col]}")
    
    # Create rolling features
    df_with_rolling = preprocessor.create_rolling_features(df, windows=[3, 6])
    print(f"\nAfter adding rolling features: {df_with_rolling.shape}")
    print(f"New columns: {[col for col in df_with_rolling.columns if 'rolling' in col][:3]}...")
    
    # Create time features
    df_with_time = preprocessor.create_time_features(df)
    print(f"\nAfter adding time features: {df_with_time.shape}")
    print(f"New columns: {[col for col in df_with_time.columns if col in ['year', 'month', 'quarter', 'month_sin', 'month_cos']]}")


def example_4_custom_pipeline():
    """Example 4: Create a custom pipeline"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Pipeline")
    print("="*70)
    
    from data_collection import DataCollector
    from data_preprocessing import DataPreprocessor
    import pandas as pd
    
    # Step 1: Collect data
    print("\nStep 1: Collecting data...")
    collector = DataCollector(output_dir='data/raw')
    df = collector.collect_all_data()
    
    # Step 2: Custom feature engineering
    print("\nStep 2: Custom feature engineering...")
    df['inflation_change'] = df['cpi_inflation'].diff()
    df['high_inflation'] = (df['cpi_inflation'] > 3).astype(int)
    print(f"Added custom features: inflation_change, high_inflation")
    
    # Step 3: Select features
    print("\nStep 3: Feature selection...")
    features = ['gdp_growth', 'unemployment_rate', 'interest_rate', 'wage_growth']
    X = df[features].iloc[1:]  # Skip first row due to diff
    y = df['cpi_inflation'].iloc[1:]
    
    print(f"Selected features: {features}")
    print(f"Dataset size: {len(X)} observations")
    
    # Step 4: Simple split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain set: {len(X_train)} observations")
    print(f"Test set: {len(X_test)} observations")


def example_5_data_analysis():
    """Example 5: Basic data analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Data Analysis")
    print("="*70)
    
    import pandas as pd
    import numpy as np
    
    # Load data
    df = pd.read_csv('data/raw/uk_inflation_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("\nDataset Information:")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    print("\nInflation Statistics:")
    print(f"Mean: {df['cpi_inflation'].mean():.2f}%")
    print(f"Median: {df['cpi_inflation'].median():.2f}%")
    print(f"Std Dev: {df['cpi_inflation'].std():.2f}%")
    print(f"Min: {df['cpi_inflation'].min():.2f}%")
    print(f"Max: {df['cpi_inflation'].max():.2f}%")
    
    print("\nCorrelation with Inflation:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['cpi_inflation'].sort_values(ascending=False)
    print(correlations[1:6])  # Top 5 correlations (excluding self)


def main():
    """Run all examples"""
    print("="*70)
    print("UK INFLATION FORECASTING - USAGE EXAMPLES")
    print("="*70)
    
    # Check if data exists, if not run example 1
    if not os.path.exists('data/raw/uk_inflation_data.csv'):
        print("\nNo data found. Running data collection first...")
        example_1_data_collection()
    
    # Menu
    print("\n" + "="*70)
    print("Available Examples:")
    print("="*70)
    print("1. Data Collection")
    print("2. Data Preprocessing (Complete Pipeline)")
    print("3. Individual Preprocessing Steps")
    print("4. Custom Pipeline")
    print("5. Data Analysis")
    print("0. Run All Examples")
    print()
    
    choice = input("Select an example (0-5): ").strip()
    
    if choice == '1':
        example_1_data_collection()
    elif choice == '2':
        example_2_data_preprocessing()
    elif choice == '3':
        example_3_individual_preprocessing_steps()
    elif choice == '4':
        example_4_custom_pipeline()
    elif choice == '5':
        example_5_data_analysis()
    elif choice == '0':
        example_1_data_collection()
        example_2_data_preprocessing()
        example_3_individual_preprocessing_steps()
        example_4_custom_pipeline()
        example_5_data_analysis()
    else:
        print("Invalid choice. Please run again and select 0-5.")
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)


if __name__ == "__main__":
    main()
