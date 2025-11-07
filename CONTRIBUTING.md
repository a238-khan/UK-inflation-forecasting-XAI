# UK Inflation Forecasting Project - Contributing Guidelines

## Project Goals

This MSc Data Science project aims to:
1. Forecast UK inflation using multiple machine learning approaches
2. Demonstrate proper ML pipeline development
3. Provide model interpretability through XAI methods
4. Showcase best practices in data science project organization

## Code Structure

### Modules

- **data_collection.py**: Handles data collection from various sources
- **data_preprocessing.py**: Feature engineering and data preparation
- **models.py**: ML model implementations (Linear, RF, XGBoost, LSTM)
- **explainability.py**: XAI methods (SHAP, LIME, feature importance)
- **visualization.py**: Plotting and visualization utilities

### Design Principles

1. **Modularity**: Each module has a single, well-defined responsibility
2. **Reusability**: Classes and functions are designed to be reusable
3. **Documentation**: All functions include docstrings
4. **Error Handling**: Graceful handling of common errors
5. **Configurability**: Key parameters are configurable

## Adding New Features

### Adding a New ML Model

1. Add the model in `src/models.py`:
```python
def train_new_model(self, X_train, y_train):
    """Train new model"""
    print("Training New Model...")
    model = NewModel(params)
    model.fit(X_train, y_train)
    self.models['new_model'] = model
    return model
```

2. Update the `train_all_models` method to include your new model

3. Test the model:
```bash
cd src
python models.py
```

### Adding New Features

1. Add feature engineering in `src/data_preprocessing.py`:
```python
def create_new_features(self, df):
    """Create new features"""
    df = df.copy()
    # Add your feature engineering logic
    return df
```

2. Update the `engineer_features` method to include your new features

### Adding New XAI Methods

1. Add new explanation method in `src/explainability.py`:
```python
def explain_with_new_method(self, model_name='model'):
    """Generate new explanations"""
    # Add your XAI logic
    pass
```

2. Update the `generate_all_explanations` method

## Testing

Before submitting changes:

1. Test data collection:
```bash
python src/data_collection.py
```

2. Test preprocessing:
```bash
python src/data_preprocessing.py
```

3. Test models:
```bash
python src/models.py
```

4. Run complete pipeline:
```bash
python run_pipeline.py
```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add docstrings to all functions
- Keep functions focused and concise
- Add comments for complex logic

## Documentation

When adding new features, update:
- README.md with new capabilities
- Docstrings in the code
- This CONTRIBUTING.md if needed

## Questions?

For questions about the project:
1. Check the README.md
2. Review the code documentation
3. Open an issue on GitHub
