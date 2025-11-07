"""
Machine Learning Models for UK Inflation Forecasting

This module implements various ML models:
- Linear Regression
- Random Forest
- XGBoost
- LightGBM
- LSTM Neural Network
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class InflationForecaster:
    """Train and evaluate multiple ML models for inflation forecasting"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.results = {}
        
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model"""
        print("Training Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['linear_regression'] = model
        return model
    
    def train_ridge(self, X_train, y_train, alpha=1.0):
        """Train Ridge Regression model"""
        print("Training Ridge Regression...")
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        self.models['ridge'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """Train Random Forest model"""
        print("Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("Training XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        print("Training LightGBM...")
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        return model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train LSTM model"""
        print("Training LSTM...")
        
        # Reshape data for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_lstm = X_val.values.reshape((X_val.shape[0], 1, X_val.shape[1]))
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.models['lstm'] = model
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a model and store results"""
        # Handle LSTM separately
        if model_name == 'lstm':
            X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results = {
            'model': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred
        }
        
        self.results[model_name] = results
        
        print(f"\n{model_name} Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return results
    
    def save_model(self, model, model_name):
        """Save trained model"""
        filepath = os.path.join(self.model_dir, f'{model_name}.pkl')
        
        if model_name == 'lstm':
            # Save Keras model
            model.save(os.path.join(self.model_dir, f'{model_name}.h5'))
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"Model saved to {filepath}")
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models"""
        # Split train into train/val for LSTM
        val_size = int(len(X_train) * 0.2)
        X_train_sub = X_train.iloc[:-val_size]
        y_train_sub = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        
        # Train models
        self.train_linear_regression(X_train, y_train)
        self.train_ridge(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        self.train_lstm(X_train_sub, y_train_sub, X_val, y_val)
        
        # Evaluate all models
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
            self.save_model(model, model_name)
        
        return self.results
    
    def get_best_model(self):
        """Get the best performing model based on RMSE"""
        best_model_name = min(self.results.items(), key=lambda x: x[1]['rmse'])[0]
        return best_model_name, self.models[best_model_name]


if __name__ == "__main__":
    # Load processed data
    train_df = pd.read_csv('data/processed/train_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')
    
    X_train = train_df.drop(['cpi_inflation', 'date'], axis=1)
    y_train = train_df['cpi_inflation']
    X_test = test_df.drop(['cpi_inflation', 'date'], axis=1)
    y_test = test_df['cpi_inflation']
    
    # Train and evaluate models
    forecaster = InflationForecaster()
    results = forecaster.train_all_models(X_train, y_train, X_test, y_test)
    
    best_model_name, best_model = forecaster.get_best_model()
    print(f"\nBest model: {best_model_name}")
