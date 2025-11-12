"""
Visualization Module for UK Inflation Forecasting

This module creates various visualizations:
- Time series plots
- Model comparison plots
- Prediction vs actual plots
- Error analysis plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Visualizer:
    """Create visualizations for the inflation forecasting project"""
    
    def __init__(self, output_dir='results/plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_time_series(self, df, target_col='cpi_inflation'):
        """
        Plot the inflation time series
        """
        plt.figure(figsize=(14, 6))
        plt.plot(df['date'], df[target_col], linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('CPI Inflation (%)')
        plt.title('UK CPI Inflation Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'inflation_time_series.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Time series plot saved to {filepath}")
    
    def plot_correlation_matrix(self, df, figsize=(12, 10)):
        """
        Plot correlation matrix of features
        """
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'correlation_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Correlation matrix saved to {filepath}")
    
    def plot_predictions(self, dates, y_true, predictions_dict):
        """
        Plot predictions from multiple models
        """
        plt.figure(figsize=(14, 7))
        
        # Plot actual values
        plt.plot(dates, y_true, label='Actual', linewidth=2, color='black')
        
        # Plot predictions from each model
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            plt.plot(dates, y_pred, label=model_name, 
                    linewidth=1.5, alpha=0.7, color=colors[i % len(colors)])
        
        plt.xlabel('Date')
        plt.ylabel('CPI Inflation (%)')
        plt.title('Model Predictions vs Actual Inflation')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'predictions_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Predictions comparison plot saved to {filepath}")
    
    def plot_model_performance(self, results_dict):
        """
        Create bar plots comparing model performance
        """
        metrics = ['rmse', 'mae', 'r2', 'mape']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            model_names = list(results_dict.keys())
            values = [results_dict[name][metric] for name in model_names]
            
            axes[i].bar(model_names, values, color='steelblue')
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'model_performance_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Model performance comparison saved to {filepath}")
    
    def plot_residuals(self, y_true, y_pred, model_name):
        """
        Plot residuals analysis
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals scatter plot
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residuals Plot - {model_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residuals Distribution - {model_name}')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'{model_name}_residuals.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Residuals plot saved to {filepath}")
    
    def plot_forecast_confidence(self, dates, y_true, y_pred, model_name):
        """
        Plot predictions with confidence intervals
        """
        # Calculate prediction errors for confidence interval estimation
        errors = y_true - y_pred
        std_error = np.std(errors)
        
        plt.figure(figsize=(14, 6))
        
        # Plot actual and predicted
        plt.plot(dates, y_true, label='Actual', linewidth=2, color='black')
        plt.plot(dates, y_pred, label='Predicted', linewidth=2, color='blue')
        
        # Add confidence intervals
        plt.fill_between(dates, y_pred - 1.96*std_error, y_pred + 1.96*std_error,
                        alpha=0.2, color='blue', label='95% Confidence Interval')
        
        plt.xlabel('Date')
        plt.ylabel('CPI Inflation (%)')
        plt.title(f'Forecast with Confidence Intervals - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f'{model_name}_forecast_confidence.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Forecast confidence plot saved to {filepath}")
    
    def create_all_visualizations(self, df_raw, results_dict, test_dates, y_test):
        """
        Generate all visualizations
        """
        print(f"\n{'='*50}")
        print("Generating Visualizations")
        print(f"{'='*50}\n")
        
        # Time series plot
        self.plot_time_series(df_raw)
        
        # Correlation matrix
        self.plot_correlation_matrix(df_raw)
        
        # Model performance comparison
        self.plot_model_performance(results_dict)
        
        # Predictions comparison
        predictions_dict = {name: results['predictions'] 
                           for name, results in results_dict.items()}
        self.plot_predictions(test_dates, y_test, predictions_dict)
        
        # Individual model residuals and confidence intervals
        for model_name, results in results_dict.items():
            y_pred = results['predictions']
            self.plot_residuals(y_test, y_pred, model_name)
            self.plot_forecast_confidence(test_dates, y_test, y_pred, model_name)
        
        print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    # Load data
    df_raw = pd.read_csv('data/raw/uk_inflation_data.csv')
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    
    test_df = pd.read_csv('data/processed/test_data.csv')
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # Create visualizer
    viz = Visualizer()
    
    # Example: Generate basic visualizations
    viz.plot_time_series(df_raw)
    viz.plot_correlation_matrix(df_raw)
