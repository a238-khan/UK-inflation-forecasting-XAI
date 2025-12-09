"""
Data Preprocessing Module for UK Inflation Forecasting

This module handles:
- Data cleaning
- Feature engineering
- Train-test split
- Data normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os


class DataPreprocessor:
    """Preprocess data for machine learning models"""
    
    def __init__(self, input_dir='data/raw', output_dir='data/processed'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scaler = StandardScaler()
        
    def load_data(self, filename='uk_inflation_data.csv'):
        """Load raw data"""
        filepath = os.path.join(self.input_dir, filename)
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def create_lag_features(self, df, target_col='cpi_inflation', lags=[1, 2, 3, 6, 12]):
        """
        Create lagged features for time series forecasting
        """
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df
    
    def create_rolling_features(self, df, target_col='cpi_inflation', windows=[3, 6, 12]):
        """
        Create rolling window statistics
        """
        df = df.copy()
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        return df
    
    def create_time_features(self, df):
        """
        Extract time-based features
        """
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values
        """
        df = df.copy()
        # Drop rows with missing values (due to lagging/rolling)
        df = df.dropna()
        return df
    
    def engineer_features(self, df):
        """
        Apply all feature engineering steps
        """
        print("Creating lag features...")
        df = self.create_lag_features(df)
        
        print("Creating rolling features...")
        df = self.create_rolling_features(df)
        
        print("Creating time features...")
        df = self.create_time_features(df)
        
        print("Handling missing values...")
        df = self.handle_missing_values(df)
        
        return df
    
    def split_data(self, df, target_col='cpi_inflation', test_size=0.2):
        """
        Split data into train and test sets
        """
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['date', target_col]]
        X = df[feature_cols]
        y = df[target_col]
        
        # Time series split (no shuffle)
        split_idx = int(len(df) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        train_dates = df['date'].iloc[:split_idx]
        test_dates = df['date'].iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test, train_dates, test_dates
    
    def normalize_features(self, X_train, X_test):
        """
        Normalize features using StandardScaler
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train_scaled, X_test_scaled
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, train_dates, test_dates):
        """
        Save processed data
        """
        train_df = X_train.copy()
        train_df['cpi_inflation'] = y_train.values
        train_df['date'] = train_dates.values
        
        test_df = X_test.copy()
        test_df['cpi_inflation'] = y_test.values
        test_df['date'] = test_dates.values
        
        train_path = os.path.join(self.output_dir, 'train_data.csv')
        test_path = os.path.join(self.output_dir, 'test_data.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Train data saved to {train_path}")
        print(f"Test data saved to {test_path}")
    
    def preprocess_all(self):
        """
        Run complete preprocessing pipeline
        """
        print("Loading data...")
        df = self.load_data()
        
        print("Engineering features...")
        df_processed = self.engineer_features(df)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test, train_dates, test_dates = self.split_data(df_processed)
        
        print(f"Train set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        print("Normalizing features...")
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        
        print("Saving processed data...")
        self.save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, train_dates, test_dates)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_all()
    print("\nPreprocessing complete!")
    print(f"Number of features: {X_train.shape[1]}")
