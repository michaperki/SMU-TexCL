"""
Class for handling the tabular data format of the SMU-Textron Cognitive Load dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from data.data_loader import BaseDataset, categorize_pilot_id
from utils.visualization import save_or_show_plot


class TableDataset(BaseDataset):
    """Class to handle loading and processing of the Table Data format."""
    
    def __init__(self, file_path: str = 'pre_proccessed_table_data.parquet'):
        """Initialize and load the dataset.
        
        Args:
            file_path: Path to the parquet file containing the table data
        """
        self.file_path = file_path
        self.df = None
        self.meta_cols = []
        self.label_cols = []
        self.feature_cols = []
        self.label_col = 'avg_tlx_quantile'  # Default label column
        
        # Load the data
        self.load_data()
    
    def load_data(self) -> None:
        """Load the dataset from the parquet file."""
        self.df = pd.read_parquet(self.file_path)
        
        # Define column categories
        self.meta_cols = [
            'pilot_id', 'trial', 'turbulence', 'date', 'start_order', 
            'start_time', 'end_time', 'duration', 'turbulence_check', 'augmented'
        ]
        
        self.label_cols = [
            'avg_tlx', 'mental_effort', 'rating', 'avg_tlx_zscore', 
            'avg_mental_effort_zscore', 'avg_tlx_quantile', 'tlx_dynamic_range'
        ]
        
        # All features that are not metadata or labels
        self.feature_cols = [col for col in self.df.columns 
                            if col not in self.meta_cols + self.label_cols]
        
        # Extract pilot categories
        self.df['pilot_category'] = self.df['pilot_id'].apply(categorize_pilot_id)
        
        print(f"Loaded table dataset with {len(self.df)} rows and {len(self.feature_cols)} features.")
    
    def get_features(self, selected_features: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature dataframe.
        
        Args:
            selected_features: List of specific features to select
            
        Returns:
            Features dataframe
        """
        if selected_features is not None:
            return self.df[selected_features]
        return self.df[self.feature_cols]
    
    def get_labels(self, label_column: Optional[str] = None) -> pd.Series:
        """Get labels.
        
        Args:
            label_column: Specific label column to use
            
        Returns:
            Label series
        """
        label_col = label_column if label_column else self.label_col
        return self.df[label_col]
    
    def explore_data(self) -> None:
        """Print basic statistics and information about the dataset."""
        print("Dataset Overview:")
        print(f"Number of pilots: {self.df['pilot_id'].nunique()}")
        print(f"Number of trials: {len(self.df)}")
        print(f"Turbulence levels: {sorted(self.df['turbulence'].unique())}")
        print(f"Pilot categories: {self.df['pilot_category'].value_counts().to_dict()}")
        
        # Print basic statistics of the labels
        print("\nLabel Statistics:")
        for label in self.label_cols:
            if pd.api.types.is_numeric_dtype(self.df[label]):
                print(f"{label}: mean={self.df[label].mean():.2f}, std={self.df[label].std():.2f}, "
                     f"min={self.df[label].min():.2f}, max={self.df[label].max():.2f}")
            else:
                print(f"{label}: {self.df[label].value_counts().to_dict()}")
    
    def visualize_turbulence_vs_load(self, save_path: Optional[str] = None) -> None:
        """Visualize relationship between turbulence and cognitive load.
        
        Args:
            save_path: Path to save the figure
        """
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plot of TLX scores by turbulence level
        sns.boxplot(x='turbulence', y='avg_tlx', data=self.df, ax=axes[0, 0])
        sns.stripplot(x='turbulence', y='avg_tlx', data=self.df, 
                     size=4, color=".3", linewidth=0, alpha=0.7, ax=axes[0, 0])
        axes[0, 0].set_title('Turbulence Level vs Average TLX')
        
        # Box plot by pilot category
        sns.boxplot(x='pilot_category', y='avg_tlx', data=self.df, ax=axes[0, 1])
        sns.stripplot(x='pilot_category', y='avg_tlx', data=self.df, 
                     size=4, color=".3", linewidth=0, alpha=0.7, ax=axes[0, 1])
        axes[0, 1].set_title('Pilot Experience vs Average TLX')
        
        # Heatmap of turbulence vs pilot category
        turb_by_cat = pd.crosstab(
            self.df['turbulence'], 
            self.df['pilot_category'], 
            values=self.df['avg_tlx'], 
            aggfunc='mean'
        )
        sns.heatmap(turb_by_cat, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[1, 0])
        axes[1, 0].set_title('Mean TLX by Turbulence and Pilot Category')
        
        # Distribution of avg_tlx scores
        sns.histplot(self.df['avg_tlx'], kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution of Average TLX Scores')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        save_or_show_plot(save_path, "Figure saved")
    
    def select_important_features(
        self, 
        label_column: Optional[str] = None,
        threshold: str = "mean", 
        model = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select important features based on feature importance.
        
        Args:
            label_column: Label column to use
            threshold: Threshold for feature selection
            model: Model to use for feature selection
            
        Returns:
            Tuple of (Selected features dataframe, list of selected feature names)
        """
        # Get features and labels
        X = self.get_features()
        y = self.get_labels(label_column)
        
        # Default model for feature selection
        if model is None:
            if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create feature selector
        selector = SelectFromModel(model, threshold=threshold)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected {len(selected_features)} out of {X.shape[1]} features")
        
        # Return selected features
        return X[selected_features], selected_features

    def visualize_feature_importance(
        self,
        label_column: Optional[str] = None,
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> Any:
        """Visualize feature importance.
        
        Args:
            label_column: Label column to use
            top_n: Number of top features to show
            save_path: Path to save the figure
        
        Returns:
            Trained model
        """
        # Get features and labels
        X = self.get_features()
        y = self.get_labels(label_column)
        
        # Train a Random Forest model for feature importance
        if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        # Get feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Important Features")
        plt.tight_layout()
        
        # Save or show
        save_or_show_plot(save_path, f"Feature importance plot saved")
        
        return model


"""
TODO Improvements:
1. Implement feature engineering for derived features
2. Add support for different feature selection methods (mutual information, chi-squared, etc.)
3. Implement feature importance stability analysis across multiple runs
4. Add feature correlation analysis to identify redundant features
5. Implement automated feature transformation (log, power, etc.)
6. Add support for categorical feature encoding (one-hot, target, etc.)
7. Implement cross-validation for feature selection
8. Add support for time-based feature selection for handling temporal dependencies
9. Implement feature drift detection across different data collections
10. Add pilot-specific feature analysis to understand individual differences
11. Implement feature standardization and normalization
12. Add support for missing value imputation with different strategies
13. Implement outlier detection and handling
14. Add visualization of feature distributions by class
15. Implement interactive feature exploration
"""
