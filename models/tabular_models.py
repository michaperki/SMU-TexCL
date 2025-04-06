"""
Models for tabular data from the SMU-Textron Cognitive Load dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    classification_report, 
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.inspection import permutation_importance

from models.base_model import BaseModel
from utils.visualization import save_or_show_plot


class TabularModel(BaseModel):
    """Base class for tabular models."""
    
    def __init__(self, model_type: str = 'rf', params: Optional[Dict[str, Any]] = None):
        """Initialize tabular model.
        
        Args:
            model_type: Type of model to train ('rf', 'gb', 'svm', 'mlp', 'elastic')
            params: Parameters for the model
        """
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.is_regression = None
        self.feature_names = None
        self.training_history = {}
    
    def train(self, X_train, y_train) -> None:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Determine if classification or regression
        self.is_regression = pd.api.types.is_numeric_dtype(y_train) and len(np.unique(y_train)) > 10
        
        # Save feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        # Create model based on type
        if self.model_type == 'rf':
            if self.is_regression:
                default_params = {'n_estimators': 100, 'random_state': 42}
                self.model = RandomForestRegressor(**{**default_params, **self.params})
            else:
                default_params = {'n_estimators': 100, 'random_state': 42}
                self.model = RandomForestClassifier(**{**default_params, **self.params})
        elif self.model_type == 'gb':
            if self.is_regression:
                default_params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
                self.model = GradientBoostingRegressor(**{**default_params, **self.params})
            else:
                default_params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
                self.model = GradientBoostingClassifier(**{**default_params, **self.params})
        elif self.model_type == 'svm':
            if self.is_regression:
                default_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
                self.model = SVR(**{**default_params, **self.params})
            else:
                default_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True}
                self.model = SVC(**{**default_params, **self.params})
        elif self.model_type == 'mlp':
            if self.is_regression:
                default_params = {'hidden_layer_sizes': (100,), 'activation': 'relu', 'random_state': 42, 'max_iter': 500}
                self.model = MLPRegressor(**{**default_params, **self.params})
            else:
                default_params = {'hidden_layer_sizes': (100,), 'activation': 'relu', 'random_state': 42, 'max_iter': 500}
                self.model = MLPClassifier(**{**default_params, **self.params})
        elif self.model_type == 'elastic':
            if self.is_regression:
                default_params = {'alpha': 1.0, 'l1_ratio': 0.5, 'random_state': 42}
                self.model = ElasticNet(**{**default_params, **self.params})
            else:
                default_params = {'C': 1.0, 'l1_ratio': 0.5, 'random_state': 42, 'solver': 'saga'}
                self.model = LogisticRegression(**{**default_params, **self.params})
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported.")
        
        # Train model
        self.model.fit(X_train, y_train)
        print(f"Trained {self.model_type} model for {'regression' if self.is_regression else 'classification'}")
    
    def predict(self, X) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """Make probability predictions for classification.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        if self.is_regression:
            raise ValueError("Probability predictions not available for regression models")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save evaluation plot
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Compute metrics
        metrics = {}
        
        # Classification metrics if categorical
        if not self.is_regression:
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics.update({f"classification_{k}": v for k, v in report.items()})
            print(classification_report(y_test, y_pred))
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = mean_squared_error(y_test, y_pred, squared=False)
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
        
        if self.is_regression:
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"R²: {metrics['r2']:.4f}")
        
        # Plot predicted vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel("Actual TLX")
        plt.ylabel("Predicted TLX")
        plt.title("Actual vs Predicted TLX Values")
        
        # Save or show
        save_or_show_plot(save_path, "Evaluation plot saved")
        
        return metrics
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame of feature importances
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature importances")
        
        # Create feature importance DataFrame
        importances = self.model.feature_importances_
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importances))]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            feature_importance = feature_importance.head(top_n)
            
        return feature_importance
    
    def calculate_permutation_importance(
        self, 
        X_test, 
        y_test, 
        n_repeats: int = 10, 
        random_state: int = 42,
        scoring: Optional[Union[str, Callable]] = None
    ) -> pd.DataFrame:
        """Calculate permutation feature importance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            n_repeats: Number of permutation repeats
            random_state: Random state for reproducibility
            scoring: Scoring method
            
        Returns:
            DataFrame of permutation importances
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Use default scoring if not specified
        if scoring is None:
            scoring = 'r2' if self.is_regression else 'accuracy'
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring
        )
        
        # Create DataFrame with results
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(perm_importance.importances_mean))]
        perm_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return perm_importance_df
    
    def optimize_hyperparameters(
        self,
        X_train,
        y_train,
        param_grid: Dict[str, Any],
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        method: str = 'grid',
        n_iter: int = 10,
        verbose: int = 1
    ) -> 'TabularModel':
        """Optimize hyperparameters using grid search or random search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid to search
            cv: Number of cross-validation folds
            scoring: Scoring method
            n_jobs: Number of parallel jobs
            method: Search method ('grid' or 'random')
            n_iter: Number of iterations for random search
            verbose: Verbosity level
            
        Returns:
            Self with optimized model
        """
        # Determine if classification or regression
        self.is_regression = pd.api.types.is_numeric_dtype(y_train) and len(np.unique(y_train)) > 10
        
        # Save feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        # Create base model
        if self.model_type == 'rf':
            if self.is_regression:
                base_model = RandomForestRegressor(random_state=42)
            else:
                base_model = RandomForestClassifier(random_state=42)
        elif self.model_type == 'gb':
            if self.is_regression:
                base_model = GradientBoostingRegressor(random_state=42)
            else:
                base_model = GradientBoostingClassifier(random_state=42)
        elif self.model_type == 'svm':
            if self.is_regression:
                base_model = SVR()
            else:
                base_model = SVC(probability=True)
        elif self.model_type == 'mlp':
            if self.is_regression:
                base_model = MLPRegressor(random_state=42, max_iter=500)
            else:
                base_model = MLPClassifier(random_state=42, max_iter=500)
        elif self.model_type == 'elastic':
            if self.is_regression:
                base_model = ElasticNet(random_state=42)
            else:
                base_model = LogisticRegression(random_state=42, solver='saga')
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported.")
        
        # Choose default scoring if not specified
        if scoring is None:
            scoring = 'neg_mean_squared_error' if self.is_regression else 'accuracy'
        
        # Create search object
        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=True
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=42,
                return_train_score=True
            )
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Save best model
        self.model = search.best_estimator_
        self.params = search.best_params_
        
        # Save search results
        self.training_history = {
            'search_results': search.cv_results_,
            'best_score': search.best_score_,
            'best_params': search.best_params_
        }
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best score: {search.best_score_:.4f}")
        
        return self
    
    def plot_hyperparameter_search(self, save_path: Optional[str] = None) -> None:
        """Plot hyperparameter search results.
        
        Args:
            save_path: Path to save the plot
        """
        if 'search_results' not in self.training_history:
            raise ValueError("No hyperparameter search results to plot")
        
        # Get search results
        results = self.training_history['search_results']
        
        # Plot mean test score and mean train score
        plt.figure(figsize=(12, 6))
        plt.plot(results['mean_test_score'], 'o-', label='Test Score')
        plt.plot(results['mean_train_score'], 'o-', label='Train Score')
        plt.xlabel('Parameter Combination')
        plt.ylabel('Score')
        plt.title('Hyperparameter Search Results')
        plt.legend()
        plt.grid(True)
        
        # Save or show
        save_or_show_plot(save_path, "Hyperparameter search plot saved")
    
    def save_model(self, file_path: str) -> None:
        """Save model to disk.
        
        Args:
            file_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model and metadata
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'is_regression': self.is_regression,
            'feature_names': self.feature_names,
            'params': self.params,
            'training_history': self.training_history
        }, file_path)
        
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """Load model from disk.
        
        Args:
            file_path: Path to load the model from
        """
        # Load model and metadata
        data = joblib.load(file_path)
        
        self.model = data['model']
        self.model_type = data['model_type']
        self.is_regression = data['is_regression']
        self.feature_names = data['feature_names']
        self.params = data['params']
        self.training_history = data.get('training_history', {})
        
        print(f"Model loaded from {file_path}")
    
    def fit(self, X, y):
        """Wrapper for the train method to match scikit-learn API.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self: The fitted model
        """
        self.train(X, y)
        return self


class TabularPipeline(TabularModel):
    """Tabular model with preprocessing pipeline."""
    
    def __init__(
        self, 
        model_type: str = 'rf', 
        params: Optional[Dict[str, Any]] = None,
        scale: bool = True
    ):
        """Initialize tabular pipeline.
        
        Args:
            model_type: Type of model to train
            params: Parameters for the model
            scale: Whether to scale features
        """
        super().__init__(model_type, params)
        self.scale = scale
        self.pipeline = None
    
    def train(self, X_train, y_train) -> None:
        """Train the pipeline.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Determine if classification or regression
        self.is_regression = pd.api.types.is_numeric_dtype(y_train) and len(np.unique(y_train)) > 10
        
        # Save feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        # Create base model
        if self.model_type == 'rf':
            if self.is_regression:
                default_params = {'n_estimators': 100, 'random_state': 42}
                base_model = RandomForestRegressor(**{**default_params, **self.params})
            else:
                default_params = {'n_estimators': 100, 'random_state': 42}
                base_model = RandomForestClassifier(**{**default_params, **self.params})
        elif self.model_type == 'gb':
            if self.is_regression:
                default_params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
                base_model = GradientBoostingRegressor(**{**default_params, **self.params})
            else:
                default_params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
                base_model = GradientBoostingClassifier(**{**default_params, **self.params})
        elif self.model_type == 'svm':
            if self.is_regression:
                default_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
                base_model = SVR(**{**default_params, **self.params})
            else:
                default_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale', 'probability': True}
                base_model = SVC(**{**default_params, **self.params})
        elif self.model_type == 'mlp':
            if self.is_regression:
                default_params = {'hidden_layer_sizes': (100,), 'activation': 'relu', 'random_state': 42, 'max_iter': 500}
                base_model = MLPRegressor(**{**default_params, **self.params})
            else:
                default_params = {'hidden_layer_sizes': (100,), 'activation': 'relu', 'random_state': 42, 'max_iter': 500}
                base_model = MLPClassifier(**{**default_params, **self.params})
        elif self.model_type == 'elastic':
            if self.is_regression:
                default_params = {'alpha': 1.0, 'l1_ratio': 0.5, 'random_state': 42}
                base_model = ElasticNet(**{**default_params, **self.params})
            else:
                default_params = {'C': 1.0, 'l1_ratio': 0.5, 'random_state': 42, 'solver': 'saga'}
                base_model = LogisticRegression(**{**default_params, **self.params})
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported.")
        
        # Create pipeline
        steps = []
        if self.scale:
            steps.append(('scaler', StandardScaler()))
        steps.append(('model', base_model))
        
        self.pipeline = Pipeline(steps)
        self.model = self.pipeline  # For compatibility with parent class methods
        
        # Train pipeline
        self.pipeline.fit(X_train, y_train)
        print(f"Trained {self.model_type} pipeline for {'regression' if self.is_regression else 'classification'}")
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get feature importance from the pipeline model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame of feature importances
        """
        if self.pipeline is None:
            raise ValueError("Pipeline has not been trained yet")
        
        # Get the model from the pipeline
        model = self.pipeline.named_steps['model']
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature importances")
        
        # Create feature importance DataFrame
        importances = model.feature_importances_
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importances))]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            feature_importance = feature_importance.head(top_n)
            
        return feature_importance
    
    def optimize_hyperparameters(
        self,
        X_train,
        y_train,
        param_grid: Dict[str, Any],
        cv: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        method: str = 'grid',
        n_iter: int = 10,
        verbose: int = 1
    ) -> 'TabularPipeline':
        """Optimize hyperparameters for the pipeline.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid to search (use 'model__param_name' for model parameters)
            cv: Number of cross-validation folds
            scoring: Scoring method
            n_jobs: Number of parallel jobs
            method: Search method ('grid' or 'random')
            n_iter: Number of iterations for random search
            verbose: Verbosity level
            
        Returns:
            Self with optimized pipeline
        """
        # Determine if classification or regression
        self.is_regression = pd.api.types.is_numeric_dtype(y_train) and len(np.unique(y_train)) > 10
        
        # Save feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        # Create base model
        if self.model_type == 'rf':
            if self.is_regression:
                base_model = RandomForestRegressor(random_state=42)
            else:
                base_model = RandomForestClassifier(random_state=42)
        elif self.model_type == 'gb':
            if self.is_regression:
                base_model = GradientBoostingRegressor(random_state=42)
            else:
                base_model = GradientBoostingClassifier(random_state=42)
        elif self.model_type == 'svm':
            if self.is_regression:
                base_model = SVR()
            else:
                base_model = SVC(probability=True)
        elif self.model_type == 'mlp':
            if self.is_regression:
                base_model = MLPRegressor(random_state=42, max_iter=500)
            else:
                base_model = MLPClassifier(random_state=42, max_iter=500)
        elif self.model_type == 'elastic':
            if self.is_regression:
                base_model = ElasticNet(random_state=42)
            else:
                base_model = LogisticRegression(random_state=42, solver='saga')
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported.")
        
        # Create pipeline
        steps = []
        if self.scale:
            steps.append(('scaler', StandardScaler()))
        steps.append(('model', base_model))
        
        pipeline = Pipeline(steps)
        
        # Choose default scoring if not specified
        if scoring is None:
            scoring = 'neg_mean_squared_error' if self.is_regression else 'accuracy'
        
        # Create search object
        if method == 'grid':
            search = GridSearchCV(
                pipeline,
                param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=True
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                pipeline,
                param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=42,
                return_train_score=True
            )
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Save best pipeline
        self.pipeline = search.best_estimator_
        self.model = self.pipeline
        
        # Extract model parameters
        model_params = {}
        for key, value in search.best_params_.items():
            if key.startswith('model__'):
                model_params[key.replace('model__', '')] = value
        
        self.params = model_params
        
        # Save search results
        self.training_history = {
            'search_results': search.cv_results_,
            'best_score': search.best_score_,
            'best_params': search.best_params_
        }
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best score: {search.best_score_:.4f}")
        
        return self

def create_turbulence_response_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create turbulence-specific response features.
    
    Args:
        df: DataFrame with features and turbulence levels
        
    Returns:
        DataFrame with added features
    """
    # Create a copy to avoid modifying the original
    enhanced_df = df.copy()
    
    # Ensure turbulence column exists
    if 'turbulence' not in enhanced_df.columns:
        print("Turbulence column not found, skipping feature creation")
        return enhanced_df
    
    # SCR response based on turbulence level
    if 'scr_mean' in enhanced_df.columns and 'turbulence' in enhanced_df.columns:
        # Group by turbulence level to get baselines
        turb_baselines = enhanced_df.groupby('turbulence')['scr_mean'].mean()
        
        # Create response feature relative to turbulence baseline
        enhanced_df['scr_turb_response'] = enhanced_df.apply(
            lambda row: row['scr_mean'] - turb_baselines[row['turbulence']], 
            axis=1
        )
    
    # Heart rate response based on turbulence
    if 'hr_mean' in enhanced_df.columns and 'turbulence' in enhanced_df.columns:
        # Group by turbulence level to get baselines
        hr_baselines = enhanced_df.groupby('turbulence')['hr_mean'].mean()
        
        # Create response feature relative to turbulence baseline
        enhanced_df['hr_turb_response'] = enhanced_df.apply(
            lambda row: row['hr_mean'] - hr_baselines[row['turbulence']], 
            axis=1
        )
    
    # Ratio of SCR to HRV by turbulence (stress response indicator)
    if all(col in enhanced_df.columns for col in ['scr_mean', 'sdrr', 'turbulence']):
        enhanced_df['scr_hrv_ratio'] = enhanced_df['scr_mean'] / (enhanced_df['sdrr'] + 1e-6)
        
        # Normalize by turbulence level
        ratio_baselines = enhanced_df.groupby('turbulence')['scr_hrv_ratio'].median()
        enhanced_df['scr_hrv_turb_norm'] = enhanced_df.apply(
            lambda row: row['scr_hrv_ratio'] / (ratio_baselines[row['turbulence']] + 1e-6),
            axis=1
        )
    
    return enhanced_df

class OrdinalClassifier(BaseModel):
    """Ordinal classifier for cognitive load that respects order relationships."""
    
    def __init__(self, n_classes: int = 3, alpha: float = 0.1):
        """Initialize ordinal classifier.
        
        Args:
            n_classes: Number of ordinal classes
            alpha: Regularization strength
        """
        self.n_classes = n_classes
        self.alpha = alpha
        self.models = []
        self.thresholds = []
        self.class_names = None
        
    def train(self, X_train, y_train) -> None:
        """Train the ordinal classifier with multiple binary classifiers.
        
        Args:
            X_train: Training features
            y_train: Training labels (ordinal categories)
        """
        # Convert categorical labels to ordinal integers if needed
        if not pd.api.types.is_numeric_dtype(y_train):
            self.class_names = np.unique(y_train)
            y_ordinal = pd.Categorical(y_train, categories=self.class_names, ordered=True).codes
        else:
            y_ordinal = y_train
            
        # Train binary classifiers for each threshold
        for k in range(self.n_classes - 1):
            # Create binary target: 1 if class >= k+1, else 0
            binary_target = (y_ordinal > k).astype(int)
            
            # Train logistic regression model with class weighting
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                C=1/self.alpha, 
                class_weight='balanced',
                random_state=42
            )
            model.fit(X_train, binary_target)
            
            # Store model
            self.models.append(model)
            
            # Calculate optimal threshold (default 0.5, can be calibrated)
            self.thresholds.append(0.5)
    
    def predict(self, X) -> np.ndarray:
        """Make ordinal predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Ordinal class predictions
        """
        if not self.models:
            raise ValueError("Model has not been trained")
        
        # Get probabilities from each binary classifier
        probs = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            probs[:, i] = model.predict_proba(X)[:, 1]
            
        # Apply thresholds to get binary decisions
        binary_decisions = probs >= np.array(self.thresholds).reshape(1, -1)
        
        # Convert to ordinal predictions (count number of 1s)
        ordinal_preds = np.sum(binary_decisions, axis=1)
        
        # Convert back to original class names if available
        if self.class_names is not None:
            return np.array([self.class_names[min(pred, len(self.class_names)-1)] for pred in ordinal_preds])
        else:
            return ordinal_preds
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict probability for each ordinal class.
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        if not self.models:
            raise ValueError("Model has not been trained")
            
        # Get binary probabilities
        binary_probs = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            binary_probs[:, i] = model.predict_proba(X)[:, 1]
            
        # Convert to class probabilities
        class_probs = np.zeros((X.shape[0], self.n_classes))
        
        # First class: probability of not exceeding first threshold
        class_probs[:, 0] = 1 - binary_probs[:, 0]
        
        # Middle classes: probability of exceeding k-1 threshold but not k threshold
        for k in range(1, self.n_classes - 1):
            class_probs[:, k] = binary_probs[:, k-1] - binary_probs[:, k]
            
        # Last class: probability of exceeding last threshold
        class_probs[:, -1] = binary_probs[:, -1]
        
        return class_probs
    
    def evaluate(self, X_test, y_test, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the ordinal classifier.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save evaluation plot
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert categorical labels to ordinal integers if needed
        if not pd.api.types.is_numeric_dtype(y_test) and self.class_names is not None:
            y_ordinal = pd.Categorical(y_test, categories=self.class_names, ordered=True).codes
        else:
            y_ordinal = y_test
            
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Convert predictions to ordinal if necessary
        if self.class_names is not None:
            y_pred_ordinal = pd.Categorical(y_pred, categories=self.class_names, ordered=True).codes
        else:
            y_pred_ordinal = y_pred
            
        # Calculate ordinal-specific metrics
        from sklearn.metrics import mean_absolute_error, accuracy_score
        
        # MAE as a measure of ordinal distance
        mae = mean_absolute_error(y_ordinal, y_pred_ordinal)
        
        # Standard classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate adjacent accuracy (predictions off by at most 1 class)
        adjacent_correct = np.abs(y_ordinal - y_pred_ordinal) <= 1
        adjacent_accuracy = np.mean(adjacent_correct)
        
        metrics = {
            'accuracy': accuracy,
            'adjacent_accuracy': adjacent_accuracy,
            'mae': mae
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Adjacent Accuracy: {adjacent_accuracy:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        
        # Plot results if save_path is provided
        if save_path:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names if self.class_names is not None else 'auto',
                       yticklabels=self.class_names if self.class_names is not None else 'auto')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Ordinal Classifier Confusion Matrix')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            
        return metrics

class EnsembleModel(BaseModel):
    """Ensemble of multiple tabular models."""
    
    def __init__(self, models: List[TabularModel] = None):
        """Initialize ensemble model.
        
        Args:
            models: List of models to ensemble
        """
        self.models = models or []
        self.is_regression = None
        self.weights = None
    
    def add_model(self, model: TabularModel) -> None:
        """Add a model to the ensemble.
        
        Args:
            model: Model to add
        """
        self.models.append(model)
    
    def set_weights(self, weights: List[float]) -> None:
        """Set weights for the ensemble.
        
        Args:
            weights: List of weights for each model
        """
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
    
    def train(self, X_train, y_train) -> None:
        """Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Determine if classification or regression
        self.is_regression = pd.api.types.is_numeric_dtype(y_train) and len(np.unique(y_train)) > 10
        
        # Train each model
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}...")
            model.train(X_train, y_train)
        
        # Set equal weights if not set
        if self.weights is None:
            self.weights = [1 / len(self.models)] * len(self.models)
    
    def predict(self, X) -> np.ndarray:
        """Make predictions with the ensemble.
        
        Args:
            X: Features to predict on
            
        Returns:
            Weighted average predictions
        """
        if not self.models:
            raise ValueError("No models in the ensemble")
        
        # Get predictions from each model
        predictions = [model.predict(X) for model in self.models]
        
        # Combine predictions
        if self.is_regression:
            # Weighted average for regression
            ensemble_pred = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                ensemble_pred += pred * weight
        else:
            # Weighted voting for classification
            if hasattr(self.models[0].model, 'predict_proba'):
                # Use probability predictions if available
                proba_predictions = [model.predict_proba(X) for model in self.models]
                ensemble_proba = np.zeros_like(proba_predictions[0])
                
                # Weighted average of probabilities
                for proba, weight in zip(proba_predictions, self.weights):
                    ensemble_proba += proba * weight
                
                # Get class with highest probability
                ensemble_pred = np.argmax(ensemble_proba, axis=1)
            else:
                # Use voting approach
                from scipy import stats
                ensemble_pred = stats.mode([pred for pred in predictions], axis=0)[0][0]
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_test, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the ensemble model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save evaluation plot
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.models:
            raise ValueError("No models in the ensemble")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Compute metrics
        metrics = {}
        
        # Classification metrics if categorical
        if not self.is_regression:
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics.update({f"classification_{k}": v for k, v in report.items()})
            print(classification_report(y_test, y_pred))
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = mean_squared_error(y_test, y_pred, squared=False)
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
        
        if self.is_regression:
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"R²: {metrics['r2']:.4f}")
        
        # Plot predicted vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel("Actual TLX")
        plt.ylabel("Predicted TLX")
        plt.title("Actual vs Predicted TLX Values (Ensemble)")
        
        # Save or show
        save_or_show_plot(save_path, "Ensemble evaluation plot saved")
        
        return metrics
    
    def compare_models(self, X_test, y_test, save_path: Optional[str] = None) -> pd.DataFrame:
        """Compare individual models with the ensemble.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save comparison plot
            
        Returns:
            DataFrame with comparison results
        """
        if not self.models:
            raise ValueError("No models in the ensemble")
        
        # Initialize results
        results = []
        
        # Evaluate each model
        for i, model in enumerate(self.models):
            y_pred = model.predict(X_test)
            
            if self.is_regression:
                mse = mean_squared_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'Model': f"Model {i+1} ({model.model_type})",
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                })
            else:
                accuracy = (y_pred == y_test).mean()
                results.append({
                    'Model': f"Model {i+1} ({model.model_type})",
                    'Accuracy': accuracy
                })
        
        # Evaluate ensemble
        y_pred = self.predict(X_test)
        
        if self.is_regression:
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': 'Ensemble',
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
        else:
            accuracy = (y_pred == y_test).mean()
            results.append({
                'Model': 'Ensemble',
                'Accuracy': accuracy
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Create comparison plot
        if save_path:
            plt.figure(figsize=(12, 6))
            
            if self.is_regression:
                # Plot regression metrics
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # MSE (lower is better)
                axes[0].bar(results_df['Model'], results_df['MSE'])
                axes[0].set_title('MSE (lower is better)')
                axes[0].set_ylabel('MSE')
                axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
                
                # MAE (lower is better)
                axes[1].bar(results_df['Model'], results_df['MAE'])
                axes[1].set_title('MAE (lower is better)')
                axes[1].set_ylabel('MAE')
                axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
                
                # R2 (higher is better)
                axes[2].bar(results_df['Model'], results_df['R2'])
                axes[2].set_title('R² (higher is better)')
                axes[2].set_ylabel('R²')
                axes[2].set_xticklabels(results_df['Model'], rotation=45, ha='right')
            else:
                # Plot classification metrics
                plt.bar(results_df['Model'], results_df['Accuracy'])
                plt.title('Accuracy Comparison')
                plt.ylabel('Accuracy')
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        return results_df
    
    def fit(self, X, y):
        """Wrapper for the train method to match scikit-learn API.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self: The fitted model
        """
        self.train(X, y)
        return self


class StackingModel(BaseModel):
    """Stacking ensemble model."""
    
    def __init__(
        self, 
        base_models: List[TabularModel],
        meta_model: Optional[TabularModel] = None
    ):
        """Initialize stacking model.
        
        Args:
            base_models: List of base models
            meta_model: Meta-model for stacking
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.is_regression = None
        self.feature_names = None
    
    def train(self, X_train, y_train, cv: int = 5) -> None:
        """Train stacking model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # Determine if classification or regression
        self.is_regression = pd.api.types.is_numeric_dtype(y_train) and len(np.unique(y_train)) > 10
        
        # Save feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        # Convert to numpy arrays
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
        
        # Train base models and generate meta-features
        n_models = len(self.base_models)
        n_samples = X_train.shape[0]
        
        # Choose CV strategy
        if self.is_regression:
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Initialize meta-features array
        if self.is_regression:
            meta_features = np.zeros((n_samples, n_models))
        else:
            n_classes = len(np.unique(y_train))
            meta_features = np.zeros((n_samples, n_models * n_classes))
        
        # Generate meta-features using cross-validation
        for i, model in enumerate(self.base_models):
            print(f"Training base model {i+1}/{n_models}...")
            
            # Generate predictions for each fold
            for train_idx, val_idx in kf.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train = y_train[train_idx]
                
                # Train model on fold
                model.train(X_fold_train, y_fold_train)
                
                # Generate predictions for validation fold
                if self.is_regression:
                    preds = model.predict(X_fold_val)
                    meta_features[val_idx, i] = preds
                else:
                    if hasattr(model.model, 'predict_proba'):
                        preds = model.predict_proba(X_fold_val)
                        meta_features[val_idx, (i*n_classes):((i+1)*n_classes)] = preds
                    else:
                        preds = model.predict(X_fold_val)
                        meta_features[val_idx, i] = preds
            
            # Retrain model on full dataset
            model.train(X_train, y_train)
        
        # Create and train meta-model
        if self.meta_model is None:
            # Default meta-model
            if self.is_regression:
                self.meta_model = TabularModel(model_type='rf')
            else:
                self.meta_model = TabularModel(model_type='rf')
        
        print("Training meta-model...")
        self.meta_model.train(meta_features, y_train)
    
    def predict(self, X) -> np.ndarray:
        """Make predictions with stacking model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions
        """
        # Save feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Convert to numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Generate meta-features
        n_models = len(self.base_models)
        n_samples = X.shape[0]
        
        if self.is_regression:
            meta_features = np.zeros((n_samples, n_models))
            
            # Generate predictions from base models
            for i, model in enumerate(self.base_models):
                preds = model.predict(X)
                meta_features[:, i] = preds
        else:
            n_classes = len(np.unique(self.meta_model.predict(np.zeros((1, n_models)))))
            meta_features = np.zeros((n_samples, n_models * n_classes))
            
            # Generate predictions from base models
            for i, model in enumerate(self.base_models):
                if hasattr(model.model, 'predict_proba'):
                    preds = model.predict_proba(X)
                    meta_features[:, (i*n_classes):((i+1)*n_classes)] = preds
                else:
                    preds = model.predict(X)
                    meta_features[:, i] = preds
        
        # Make predictions with meta-model
        return self.meta_model.predict(meta_features)
    
    def evaluate(self, X_test, y_test, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate stacking model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save evaluation plot
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Compute metrics
        metrics = {}
        
        # Classification metrics if categorical
        if not self.is_regression:
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics.update({f"classification_{k}": v for k, v in report.items()})
            print(classification_report(y_test, y_pred))
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = mean_squared_error(y_test, y_pred, squared=False)
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
        
        if self.is_regression:
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"R²: {metrics['r2']:.4f}")
        
        # Plot predicted vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel("Actual TLX")
        plt.ylabel("Predicted TLX")
        plt.title("Actual vs Predicted TLX Values (Stacking)")
        
        # Save or show
        save_or_show_plot(save_path, "Stacking evaluation plot saved")
        
        return metrics
    
    def compare_with_base_models(
        self, 
        X_test, 
        y_test, 
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare stacking model with base models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save comparison plot
            
        Returns:
            DataFrame with comparison results
        """
        # Initialize results
        results = []
        
        # Evaluate each base model
        for i, model in enumerate(self.base_models):
            y_pred = model.predict(X_test)
            
            if self.is_regression:
                mse = mean_squared_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'Model': f"Base {i+1} ({model.model_type})",
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                })
            else:
                accuracy = (y_pred == y_test).mean()
                results.append({
                    'Model': f"Base {i+1} ({model.model_type})",
                    'Accuracy': accuracy
                })
        
        # Evaluate stacking model
        y_pred = self.predict(X_test)
        
        if self.is_regression:
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': 'Stacking',
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
        else:
            accuracy = (y_pred == y_test).mean()
            results.append({
                'Model': 'Stacking',
                'Accuracy': accuracy
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Create comparison plot
        if save_path:
            plt.figure(figsize=(12, 6))
            
            if self.is_regression:
                # Plot regression metrics
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # MSE (lower is better)
                axes[0].bar(results_df['Model'], results_df['MSE'])
                axes[0].set_title('MSE (lower is better)')
                axes[0].set_ylabel('MSE')
                axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
                
                # MAE (lower is better)
                axes[1].bar(results_df['Model'], results_df['MAE'])
                axes[1].set_title('MAE (lower is better)')
                axes[1].set_ylabel('MAE')
                axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
                
                # R2 (higher is better)
                axes[2].bar(results_df['Model'], results_df['R2'])
                axes[2].set_title('R² (higher is better)')
                axes[2].set_ylabel('R²')
                axes[2].set_xticklabels(results_df['Model'], rotation=45, ha='right')
            else:
                # Plot classification metrics
                plt.bar(results_df['Model'], results_df['Accuracy'])
                plt.title('Accuracy Comparison')
                plt.ylabel('Accuracy')
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        return results_df
    
    def fit(self, X, y, cv: int = 5):
        """Wrapper for the train method to match scikit-learn API.
        
        Args:
            X: Training features
            y: Training labels
            cv: Number of cross-validation folds
            
        Returns:
            self: The fitted model
        """
        self.train(X, y, cv=cv)
        return self


"""
Implemented improvements:
1. Added support for more model types (gradient boosting, SVM, neural networks, elastic net)
2. Implemented hyperparameter optimization using grid search and random search
3. Added model serialization for saving and loading trained models
4. Added ensemble models with weighted averaging and stacking
5. Implemented permutation feature importance for model interpretability
6. Added visualization of hyperparameter search results
7. Implemented model comparison utilities for ensemble models

Improvements still to implement:
8. Add support for imbalanced data handling
9. Implement custom scoring functions for domain-specific metrics
10. Add interpretable model alternatives (decision trees, rule-based models)
11. Implement feature interaction analysis
12. Add confidence intervals for predictions
13. Add anomaly detection capabilities
14. Implement incremental learning for large datasets
15. Add support for multi-output regression/classification
16. Implement domain-specific model evaluation metrics for cognitive load assessment
17. Add learning curve analysis to diagnose overfitting/underfitting
18. Implement model calibration for probability outputs
19. Add support for ensemble models with different feature sets
20. Implement online learning for streaming data
"""
