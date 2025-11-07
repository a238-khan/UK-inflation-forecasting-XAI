"""
UK Inflation Forecasting with Machine Learning and XAI
"""

__version__ = "1.0.0"
__author__ = "MSc Data Science Project"

from . import data_collection
from . import data_preprocessing
from . import models
from . import explainability
from . import visualization

__all__ = [
    'data_collection',
    'data_preprocessing',
    'models',
    'explainability',
    'visualization'
]
