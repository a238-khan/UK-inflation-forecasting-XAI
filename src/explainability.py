"""
Explainable AI (XAI) Module for Model Interpretability

This module implements various XAI techniques:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature Importance Analysis
- Partial Dependence Plots
"""

import pandas as pd
import numpy as np
import pickle
import os

# Optional visualization and XAI imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualizations will be skipped.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. SHAP explanations will be skipped.")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. LIME explanations will be skipped.")


class ModelExplainer:
    """Explain model predictions using XAI methods"""
    
    def __init__(self, model, X_train, X_test, feature_names, output_dir='results/plots'):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_feature_importance(self, model_name='model'):
        """
        Analyze and visualize feature importance for tree-based models
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping feature importance plot - matplotlib not available")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(feature_importance_df['feature'][:15], 
                    feature_importance_df['importance'][:15])
            plt.xlabel('Importance')
            plt.title(f'Top 15 Feature Importances - {model_name}')
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, f'{model_name}_feature_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Feature importance plot saved to {filepath}")
            return feature_importance_df
        else:
            print(f"Model {model_name} does not have feature_importances_ attribute")
            return None
    
    def explain_with_shap(self, model_name='model', sample_size=100):
        """
        Generate SHAP explanations
        """
        if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print(f"Skipping SHAP explanations - required libraries not available")
            return None
        
        print(f"Generating SHAP explanations for {model_name}...")
        
        try:
            # Sample data for faster computation
            X_sample = self.X_test.iloc[:sample_size]
            
            # Create SHAP explainer based on model type
            model_class_name = type(self.model).__name__
            
            if any(name in model_class_name for name in ['RandomForest', 'XGB', 'LGBM', 'GradientBoosting']):
                # Tree-based models
                explainer = shap.TreeExplainer(self.model)
            elif any(name in model_class_name for name in ['Linear', 'Ridge', 'Lasso']):
                # Linear models
                explainer = shap.LinearExplainer(self.model, self.X_train)
            else:
                # Fallback to KernelExplainer (slower but works for any model)
                explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X_train, 50))
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, 
                            show=False, max_display=15)
            filepath = os.path.join(self.output_dir, f'{model_name}_shap_summary.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot saved to {filepath}")
            
            # Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names,
                            plot_type="bar", show=False, max_display=15)
            filepath = os.path.join(self.output_dir, f'{model_name}_shap_bar.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP bar plot saved to {filepath}")
            
            return shap_values
            
        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
            return None
    
    def explain_with_lime(self, model_name='model', num_samples=5):
        """
        Generate LIME explanations for individual predictions
        """
        if not LIME_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print(f"Skipping LIME explanations - required libraries not available")
            return None
        
        print(f"Generating LIME explanations for {model_name}...")
        
        try:
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                self.X_train.values,
                feature_names=self.feature_names,
                mode='regression',
                verbose=False
            )
            
            # Explain random samples
            sample_indices = np.random.choice(len(self.X_test), 
                                             min(num_samples, len(self.X_test)), 
                                             replace=False)
            
            for idx, sample_idx in enumerate(sample_indices):
                instance = self.X_test.iloc[sample_idx].values
                
                # Generate explanation
                exp = explainer.explain_instance(
                    instance,
                    self.model.predict,
                    num_features=10
                )
                
                # Save explanation
                fig = exp.as_pyplot_figure()
                filepath = os.path.join(self.output_dir, 
                                       f'{model_name}_lime_sample_{idx}.png')
                plt.tight_layout()
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
            print(f"LIME explanations saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Error generating LIME explanations: {e}")
    
    def plot_partial_dependence(self, feature_idx=0, model_name='model'):
        """
        Create partial dependence plot for a specific feature
        """
        from sklearn.inspection import partial_dependence
        
        try:
            # Calculate partial dependence
            pd_result = partial_dependence(
                self.model, 
                self.X_test, 
                [feature_idx]
            )
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(pd_result['values'][0], pd_result['average'][0])
            plt.xlabel(self.feature_names[feature_idx])
            plt.ylabel('Partial Dependence')
            plt.title(f'Partial Dependence Plot - {self.feature_names[feature_idx]}')
            plt.grid(True, alpha=0.3)
            
            filepath = os.path.join(self.output_dir, 
                                   f'{model_name}_pdp_{self.feature_names[feature_idx]}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Partial dependence plot saved to {filepath}")
            
        except Exception as e:
            print(f"Error generating partial dependence plot: {e}")
    
    def generate_all_explanations(self, model_name='model'):
        """
        Generate all XAI explanations
        """
        print(f"\n{'='*50}")
        print(f"Generating XAI Explanations for {model_name}")
        print(f"{'='*50}\n")
        
        # Feature importance
        self.analyze_feature_importance(model_name)
        
        # SHAP explanations
        self.explain_with_shap(model_name)
        
        # LIME explanations
        self.explain_with_lime(model_name)
        
        print(f"\nAll XAI explanations generated successfully!")


class MultiModelExplainer:
    """Compare explanations across multiple models"""
    
    def __init__(self, models_dict, X_train, X_test, feature_names, output_dir='results/plots'):
        self.models_dict = models_dict
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def compare_feature_importance(self):
        """
        Compare feature importance across all tree-based models
        """
        plt.figure(figsize=(14, 8))
        
        importance_data = []
        
        for model_name, model in self.models_dict.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for feature, importance in zip(self.feature_names, importances):
                    importance_data.append({
                        'model': model_name,
                        'feature': feature,
                        'importance': importance
                    })
        
        if importance_data:
            df = pd.DataFrame(importance_data)
            
            # Get top features
            top_features = (df.groupby('feature')['importance'].mean()
                           .sort_values(ascending=False)
                           .head(10).index)
            
            df_top = df[df['feature'].isin(top_features)]
            
            # Plot
            pivot_df = df_top.pivot(index='feature', columns='model', values='importance')
            pivot_df.plot(kind='barh', figsize=(12, 8))
            plt.xlabel('Importance')
            plt.title('Feature Importance Comparison Across Models')
            plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, 'multi_model_feature_importance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Multi-model feature importance comparison saved to {filepath}")
    
    def explain_all_models(self):
        """
        Generate explanations for all models
        """
        for model_name, model in self.models_dict.items():
            if model_name != 'lstm':  # Skip LSTM for XAI (needs special handling)
                explainer = ModelExplainer(model, self.X_train, self.X_test, 
                                          self.feature_names, self.output_dir)
                explainer.generate_all_explanations(model_name)


if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv('data/processed/test_data.csv')
    train_df = pd.read_csv('data/processed/train_data.csv')
    
    X_train = train_df.drop(['cpi_inflation', 'date'], axis=1)
    X_test = test_df.drop(['cpi_inflation', 'date'], axis=1)
    
    # Load models
    models = {}
    model_names = ['random_forest', 'xgboost', 'lightgbm']
    
    for name in model_names:
        with open(f'models/{name}.pkl', 'rb') as f:
            models[name] = pickle.load(f)
    
    # Generate explanations
    multi_explainer = MultiModelExplainer(models, X_train, X_test, 
                                         X_train.columns.tolist())
    multi_explainer.explain_all_models()
    multi_explainer.compare_feature_importance()
