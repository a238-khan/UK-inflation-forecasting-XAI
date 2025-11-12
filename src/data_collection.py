"""
Data Collection Module for UK Inflation Data

This module handles data collection from various sources including:
- Bank of England API
- ONS (Office for National Statistics) API
- CSV file imports
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os


class DataCollector:
    """Collect UK inflation and related economic data from multiple sources"""
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def collect_inflation_data(self):
        """
        Collect UK CPI inflation data
        Returns a DataFrame with inflation data
        """
        # Simulated inflation data for demonstration
        # In production, this would call real APIs like ONS or Bank of England
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='MS')
        
        # Generate realistic inflation data with trend and seasonality
        np.random.seed(42)
        base_inflation = 2.0
        trend = np.linspace(0, 3, len(dates))
        seasonality = 0.5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12)
        noise = np.random.normal(0, 0.3, len(dates))
        
        inflation = base_inflation + trend + seasonality + noise
        inflation = np.maximum(inflation, -1)  # Ensure no extreme deflation
        
        df = pd.DataFrame({
            'date': dates,
            'cpi_inflation': inflation
        })
        
        return df
    
    def collect_economic_indicators(self):
        """
        Collect related economic indicators
        Returns a DataFrame with economic indicators
        """
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='MS')
        
        np.random.seed(42)
        n = len(dates)
        
        # Simulate various economic indicators
        data = {
            'date': dates,
            'gdp_growth': np.random.normal(2.0, 1.5, n),
            'unemployment_rate': np.random.normal(5.0, 1.0, n),
            'interest_rate': np.random.normal(1.5, 1.0, n),
            'exchange_rate_usd': np.random.normal(1.3, 0.1, n),
            'oil_price': np.random.normal(70, 20, n),
            'wage_growth': np.random.normal(2.5, 1.0, n),
            'retail_sales_growth': np.random.normal(3.0, 2.0, n),
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic ranges
        df['unemployment_rate'] = df['unemployment_rate'].clip(2, 10)
        df['interest_rate'] = df['interest_rate'].clip(0.1, 5)
        df['oil_price'] = df['oil_price'].clip(20, 150)
        
        return df
    
    def merge_datasets(self, inflation_df, economic_df):
        """
        Merge inflation data with economic indicators
        """
        merged_df = pd.merge(inflation_df, economic_df, on='date', how='inner')
        return merged_df
    
    def save_data(self, df, filename):
        """
        Save collected data to CSV
        """
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
    
    def collect_all_data(self):
        """
        Collect all data sources and merge them
        """
        print("Collecting inflation data...")
        inflation_df = self.collect_inflation_data()
        
        print("Collecting economic indicators...")
        economic_df = self.collect_economic_indicators()
        
        print("Merging datasets...")
        merged_df = self.merge_datasets(inflation_df, economic_df)
        
        print("Saving data...")
        self.save_data(merged_df, 'uk_inflation_data.csv')
        
        return merged_df


if __name__ == "__main__":
    collector = DataCollector()
    data = collector.collect_all_data()
    print(f"\nData collection complete!")
    print(f"Shape: {data.shape}")
    print(f"\nFirst few rows:")
    print(data.head())
    print(f"\nData summary:")
    print(data.describe())
