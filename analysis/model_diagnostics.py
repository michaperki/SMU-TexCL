"""
Diagnostic script for investigating model performance issues in the SMU-TexCL dataset.
This script performs a series of analyses to understand why models are not performing well.

Usage:
    python analysis/model_diagnostics.py --tabular_data_path ../pre_proccessed_table_data.parquet --windowed_data_path ../
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import torch
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# Add parent directory to path to import project modules
sys.path.append(os.path.abspath(".."))
from data.table_dataset import TableDataset
from data.windowed_dataset import WindowedDataset
from models.base_model import BaseModel
from models.tabular_models import TabularModel
from models.sequence_models import SequenceModel, PilotDataset, collate_fn
from utils.visualization import set_plot_style
from utils.preprocessing import assess_signal_quality
from utils.evaluation import evaluate_regression_model

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
        """Initialize ordinal classifier."""
        self.n_classes = n_classes
        self.alpha = alpha
        self.models = []
        self.thresholds = []
        self.class_names = None
        
    def train(self, X_train, y_train) -> None:
        """Train the ordinal classifier with multiple binary classifiers."""
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
        """Make ordinal predictions."""
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
    
    def evaluate(self, X_test, y_test, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the ordinal classifier."""
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



# Set up plotting style
set_plot_style()

OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='SMU-TexCL Model Diagnostics')
    
    parser.add_argument('--tabular_data_path', type=str, required=True,
                       help='Path to tabular data')
    parser.add_argument('--windowed_data_path', type=str, required=True,
                       help='Path to windowed data JSON files')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='Directory to save analysis outputs')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def analyze_feature_importance(table_data: TableDataset) -> None:
    """Analyze feature importance in more detail."""
    print("\n======= Feature Importance Analysis =======")
    
    # Select important features and visualize
    model = TabularModel(model_type='rf')
    X = table_data.get_features()
    y = table_data.get_labels()
    
    # Train model
    model.train(X, y)
    
    # Get feature importance
    importance_df = model.get_feature_importance()
    
    # Plot feature importance
    plt.figure(figsize=(14, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20), color='skyblue')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'top_features.png'), dpi=300)
    
    # Calculate feature correlation with target
    corr_with_target = []
    target_col = table_data.label_col
    
    for col in X.columns:
        corr = X[col].corr(y)
        corr_with_target.append({'feature': col, 'correlation': corr})
    
    corr_df = pd.DataFrame(corr_with_target).sort_values('correlation', key=abs, ascending=False)
    
    # Plot correlation with target
    plt.figure(figsize=(14, 8))
    sns.barplot(x='correlation', y='feature', data=corr_df.head(20), palette='coolwarm')
    plt.title(f'Top 20 Feature Correlations with {target_col}')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlations.png'), dpi=300)
    
    # Print top correlated features
    print("\nTop 10 features by importance:")
    print(importance_df.head(10)[['feature', 'importance']])
    
    print("\nTop 10 features by correlation with target:")
    print(corr_df.head(10)[['feature', 'correlation']])


def analyze_signal_quality(windowed_data: WindowedDataset) -> None:
    """Analyze physiological signal quality."""
    print("\n======= Signal Quality Analysis =======")
    
    signal_quality_scores = {
        'ppg': [],
        'eda': [],
        'accel': [],
        'temp': []
    }
    
    pilot_count = 0
    max_pilots = 5  # Limit to 5 pilots for speed
    
    # Sample some pilots
    for pilot_id, trials in windowed_data.pilot_data.items():
        if pilot_count >= max_pilots:
            break
            
        for trial_idx, trial in enumerate(trials[:2]):  # Analyze first 2 trials per pilot
            windows = trial.get('windowed_features', [])
            
            if windows:
                # Sample 5 windows per trial
                sample_windows = windows[:5] if len(windows) >= 5 else windows
                
                for window in sample_windows:
                    # Assess quality of each signal type
                    if 'ppg_input' in window and window['ppg_input']:
                        ppg_signal = np.array(window['ppg_input'][0])
                        quality = assess_signal_quality(ppg_signal, 64.0, 'ppg')
                        signal_quality_scores['ppg'].append(quality.get('quality_score', 0))
                    
                    if 'eda_input' in window and window['eda_input']:
                        eda_signal = np.array(window['eda_input'][0])
                        quality = assess_signal_quality(eda_signal, 4.0, 'eda')
                        signal_quality_scores['eda'].append(quality.get('quality_score', 0))
                    
                    if 'accel_input' in window and window['accel_input']:
                        accel_signal = np.array(window['accel_input'][0])
                        quality = assess_signal_quality(accel_signal, 32.0, 'accel')
                        signal_quality_scores['accel'].append(quality.get('quality_score', 0))
                    
                    if 'temp_input' in window and window['temp_input']:
                        temp_signal = np.array(window['temp_input'][0])
                        quality = assess_signal_quality(temp_signal, 4.0, 'temp')
                        signal_quality_scores['temp'].append(quality.get('quality_score', 0))
        
        pilot_count += 1
    
    # Plot quality distribution
    plt.figure(figsize=(12, 8))
    for signal_type, scores in signal_quality_scores.items():
        if scores:
            sns.kdeplot(scores, label=signal_type)
    
    plt.title('Signal Quality Distribution')
    plt.xlabel('Quality Score (0-1)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'signal_quality.png'), dpi=300)
    
    # Print summary statistics
    print("\nSignal Quality Statistics:")
    for signal_type, scores in signal_quality_scores.items():
        if scores:
            print(f"{signal_type.upper()}: Mean={np.mean(scores):.3f}, Median={np.median(scores):.3f}, Min={np.min(scores):.3f}, Max={np.max(scores):.3f}")

def analyze_signal_quality_by_condition(windowed_data: WindowedDataset) -> None:
    """Analyze physiological signal quality across different conditions."""
    print("\n======= Signal Quality Analysis by Condition =======")
    
    # Initialize dictionaries to store signal quality metrics
    quality_by_turbulence = {}
    quality_by_pilot_category = {}
    
    # Get a sample of pilots for faster analysis
    sample_pilots = list(windowed_data.pilot_data.keys())[:10]
    
    # Analyze signal quality
    for pilot_id in sample_pilots:
        pilot_trials = windowed_data.pilot_data.get(pilot_id, [])
        for trial in pilot_trials:
            # Get metadata
            turbulence = trial.get('turbulence', 'unknown')
            
            # Get pilot category
            if pilot_id.startswith('8'):
                pilot_category = 'minimal_exp'
            elif pilot_id.startswith('9'):
                pilot_category = 'commercial'
            else:
                pilot_category = 'air_force'
            
            # Initialize containers for the categories if not present
            if turbulence not in quality_by_turbulence:
                quality_by_turbulence[turbulence] = {
                    'ppg': [], 'eda': [], 'accel': [], 'temp': []
                }
            
            if pilot_category not in quality_by_pilot_category:
                quality_by_pilot_category[pilot_category] = {
                    'ppg': [], 'eda': [], 'accel': [], 'temp': []
                }
            
            # Analyze signal quality for each window
            windows = trial.get('windowed_features', [])
            for window in windows[:5]:  # Limit to first 5 windows per trial
                # PPG quality
                if 'ppg_input' in window and window['ppg_input']:
                    ppg_signal = np.array(window['ppg_input'][0])
                    quality = assess_signal_quality(ppg_signal, 64.0, 'ppg')
                    quality_by_turbulence[turbulence]['ppg'].append(quality.get('quality_score', 0))
                    quality_by_pilot_category[pilot_category]['ppg'].append(quality.get('quality_score', 0))
                
                # EDA quality
                if 'eda_input' in window and window['eda_input']:
                    eda_signal = np.array(window['eda_input'][0])
                    quality = assess_signal_quality(eda_signal, 4.0, 'eda')
                    quality_by_turbulence[turbulence]['eda'].append(quality.get('quality_score', 0))
                    quality_by_pilot_category[pilot_category]['eda'].append(quality.get('quality_score', 0))
    
    # Plot quality by turbulence
    plt.figure(figsize=(12, 6))
    for signal_type in ['ppg', 'eda']:
        means = []
        stds = []
        labels = []
        
        for turbulence, metrics in sorted(quality_by_turbulence.items()):
            if metrics[signal_type]:
                means.append(np.mean(metrics[signal_type]))
                stds.append(np.std(metrics[signal_type]))
                labels.append(str(turbulence))
        
        if means:
            x = range(len(means))
            plt.errorbar([i + (0.2 if signal_type == 'eda' else 0) for i in x], 
                        means, yerr=stds, label=signal_type.upper(), fmt='o-')
    
    plt.title('Signal Quality by Turbulence Level')
    plt.xlabel('Turbulence Level')
    plt.xticks(range(len(labels)), labels)
    plt.ylabel('Quality Score (0-1)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'signal_quality_by_turbulence.png'), dpi=300)
    
    # Plot quality by pilot category
    plt.figure(figsize=(12, 6))
    for signal_type in ['ppg', 'eda']:
        means = []
        stds = []
        labels = []
        
        for category, metrics in quality_by_pilot_category.items():
            if metrics[signal_type]:
                means.append(np.mean(metrics[signal_type]))
                stds.append(np.std(metrics[signal_type]))
                labels.append(category)
        
        if means:
            x = range(len(means))
            plt.errorbar([i + (0.2 if signal_type == 'eda' else 0) for i in x], 
                        means, yerr=stds, label=signal_type.upper(), fmt='o-')
    
    plt.title('Signal Quality by Pilot Category')
    plt.xlabel('Pilot Category')
    plt.xticks(range(len(labels)), labels)
    plt.ylabel('Quality Score (0-1)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'signal_quality_by_category.png'), dpi=300)
    
    print("Signal quality analysis by condition completed!")

def analyze_prediction_errors(table_data: TableDataset) -> None:
    """Analyze patterns in prediction errors."""
    print("\n======= Prediction Error Analysis =======")
    
    # Split data
    X = table_data.get_features()  # This should only get numeric features
    y = table_data.get_labels()
    
    # Create a copy for metadata that we'll use for analysis later
    metadata_df = pd.DataFrame(index=X.index)
    
    # Add metadata columns if they exist in the original dataframe
    metadata_cols = ['pilot_id', 'turbulence', 'pilot_category']
    for col in metadata_cols:
        if col in table_data.df.columns:
            metadata_df[col] = table_data.df[col]
    
    # Split data for training
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, metadata_df, test_size=0.2, random_state=42
    )
    
    # Train a simple model
    model = TabularModel(model_type='rf')
    model.train(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate errors
    errors = np.abs(y_test - y_pred)
    
    # Add metadata and errors to a dataframe
    error_df = pd.DataFrame(index=X_test.index)
    # Add metadata
    for col in metadata_df.columns:
        error_df[col] = meta_test[col]
    # Add prediction data
    error_df['true_tlx'] = y_test
    error_df['pred_tlx'] = y_pred
    error_df['abs_error'] = errors
    error_df['rel_error'] = errors / (y_test + 1e-10)  # Avoid division by zero
    
    # Analyze errors by turbulence level if column exists
    if 'turbulence' in error_df.columns:
        error_by_turbulence = error_df.groupby('turbulence')['abs_error'].agg(['mean', 'std', 'count'])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=error_by_turbulence.index, y=error_by_turbulence['mean'])
        plt.title('Mean Absolute Error by Turbulence Level')
        plt.xlabel('Turbulence Level')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'error_by_turbulence.png'), dpi=300)
        
        print("\nError by Turbulence Level:")
        print(error_by_turbulence)
    
    # Analyze errors by pilot category if column exists
    if 'pilot_category' in error_df.columns:
        error_by_category = error_df.groupby('pilot_category')['abs_error'].agg(['mean', 'std', 'count'])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=error_by_category.index, y=error_by_category['mean'])
        plt.title('Mean Absolute Error by Pilot Category')
        plt.xlabel('Pilot Category')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'error_by_category.png'), dpi=300)
        
        print("\nError by Pilot Category:")
        print(error_by_category)
    
    # Analyze errors by true TLX value
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='true_tlx', y='abs_error', data=error_df, alpha=0.7)
    plt.title('Error vs. True TLX')
    plt.xlabel('True TLX')
    plt.ylabel('Absolute Error')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_vs_tlx.png'), dpi=300)
    
    # Top 10 worst predicted samples - with safe column checking
    display_cols = ['true_tlx', 'pred_tlx', 'abs_error']
    # Add metadata columns if available
    for col in metadata_cols:
        if col in error_df.columns:
            display_cols.insert(0, col)
    
    worst_predictions = error_df.sort_values('abs_error', ascending=False).head(10)
    print("\nTop 10 Worst Predictions:")
    print(worst_predictions[display_cols])


def perform_pilot_wise_cv(table_data: TableDataset) -> None:
    """Perform pilot-wise cross-validation to check generalization."""
    print("\n======= Pilot-wise Cross-Validation =======")
    
    # Get data
    X = table_data.get_features()
    y = table_data.get_labels()
    
    # Extract pilot IDs
    if 'pilot_id' not in X.columns:
        print("Pilot ID column not found, skipping pilot-wise CV")
        return
    
    pilot_ids = X['pilot_id'].values
    unique_pilots = np.unique(pilot_ids)
    print(f"Number of unique pilots: {len(unique_pilots)}")
    
    # Define cross-validation
    group_kfold = GroupKFold(n_splits=5)
    
    # Prepare for storing results
    cv_results = []
    
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=pilot_ids)):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model = TabularModel(model_type='rf')
        model.train(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store results
        cv_results.append({
            'fold': fold,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'test_pilots': np.unique(X_test['pilot_id']).tolist()
        })
        
        print(f"Fold {fold+1}: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    
    # Calculate overall metrics
    mean_r2 = np.mean([r['r2'] for r in cv_results])
    mean_rmse = np.mean([r['rmse'] for r in cv_results])
    mean_mae = np.mean([r['mae'] for r in cv_results])
    
    print(f"\nPilot-wise CV - Mean R²: {mean_r2:.4f}, Mean RMSE: {mean_rmse:.4f}, Mean MAE: {mean_mae:.4f}")
    
    # Compare to standard CV
    print("\nComparing to standard random-split CV:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = TabularModel(model_type='rf')
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Standard CV - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

def analyze_classification_approach(table_data: TableDataset) -> None:
    """Try classification instead of regression with improved ordinal approach."""
    print("\n======= Classification Approach Analysis =======")
    
    # Get features and labels
    X = table_data.get_features()
    y = table_data.get_labels()
    
    # Add metadata if available
    if 'pilot_id' in table_data.df.columns and 'pilot_id' not in X.columns:
        X['pilot_id'] = table_data.df['pilot_id']
    if 'turbulence' in table_data.df.columns and 'turbulence' not in X.columns:
        X['turbulence'] = table_data.df['turbulence']
    
    # Add turbulence-specific features
    X = create_turbulence_response_features(X)
    
    # Convert the continuous target to categorical
    # Create 3 equal-sized bins for Low, Medium, High cognitive load
    y_binned = pd.qcut(y, 3, labels=['Low', 'Medium', 'High'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binned, test_size=0.2, random_state=42, stratify=y_binned
    )
    
    # Remove metadata columns for training
    meta_cols = ['pilot_id', 'turbulence', 'pilot_category']
    X_train_model = X_train.drop(columns=[col for col in meta_cols if col in X_train.columns])
    X_test_model = X_test.drop(columns=[col for col in meta_cols if col in X_test.columns])
    
    # Train ordinal classifier
    print("Training ordinal classifier...")
    ordinal_model = OrdinalClassifier(n_classes=3)
    ordinal_model.train(X_train_model, y_train)
    
    # Evaluate ordinal model
    print("\nOrdinal Classifier Results:")
    ordinal_metrics = ordinal_model.evaluate(
        X_test_model, y_test, 
        save_path=os.path.join(OUTPUT_DIR, 'ordinal_confusion_matrix.png')
    )
    
    # Train a standard classification model for comparison
    print("\nTraining standard classifier for comparison...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_model, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test_model)
    
    # Evaluate model
    classification_scores = classification_report(y_test, y_pred, output_dict=True)
    print("\nStandard Classifier Results:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Standard Classifier Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
    
    # Analyze per-class accuracy by conditions
    if 'turbulence' in X_test.columns:
        class_acc_by_turbulence = {}
        for turbulence in X_test['turbulence'].unique():
            mask = X_test['turbulence'] == turbulence
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            class_acc_by_turbulence[turbulence] = class_acc
        
        plt.figure(figsize=(10, 6))
        plt.bar(class_acc_by_turbulence.keys(), class_acc_by_turbulence.values())
        plt.title('Classification Accuracy by Turbulence Level')
        plt.xlabel('Turbulence Level')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'class_acc_by_turbulence.png'), dpi=300)
        
        print("\nClassification Accuracy by Turbulence Level:")
        for turb, acc in class_acc_by_turbulence.items():
            print(f"Turbulence {turb}: {acc:.4f}")
    
    print("Classification approach analysis completed!")

def analyze_pilot_normalized_features(table_data: TableDataset) -> None:
    """Analyze features normalized on a per-pilot basis."""
    print("\n======= Pilot-Normalized Features Analysis =======")
    
    # Get features and labels
    X = table_data.get_features()
    y = table_data.get_labels()
    
    # Check if pilot_id is available
    if 'pilot_id' not in table_data.df.columns:
        print("Pilot ID not available, skipping pilot normalization analysis")
        return
    
    pilot_ids = table_data.df['pilot_id']
    
    # Create normalized features
    X_norm = X.copy()
    
    # For each pilot, normalize their features
    for pilot in pilot_ids.unique():
        pilot_mask = pilot_ids == pilot
        
        # Skip if not enough data
        if pilot_mask.sum() < 3:
            continue
            
        # Normalize each feature by pilot
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Get pilot's values for this feature
                pilot_values = X.loc[pilot_mask, col]
                
                # Calculate z-score within pilot
                mean = pilot_values.mean()
                std = pilot_values.std()
                
                # Avoid division by zero
                if std > 0:
                    X_norm.loc[pilot_mask, f"{col}_pilot_norm"] = (pilot_values - mean) / std
                else:
                    X_norm.loc[pilot_mask, f"{col}_pilot_norm"] = 0
    
    # Use only the normalized features for analysis
    norm_cols = [col for col in X_norm.columns if col.endswith('_pilot_norm')]
    if not norm_cols:
        print("No normalized features created, skipping analysis")
        return
        
    X_pilot_norm = X_norm[norm_cols]
    
    # Split data for model training
    X_train, X_test, y_train, y_test = train_test_split(
        X_pilot_norm, y, test_size=0.2, random_state=42
    )
    
    # Train model on normalized features
    model = TabularModel(model_type='rf')
    model.train(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nResults with pilot-normalized features:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title('Actual vs Predicted (Pilot-Normalized Features)')
    plt.xlabel('Actual TLX')
    plt.ylabel('Predicted TLX')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'pilot_normalized_predictions.png'), dpi=300)
    
    # Get top normalized features
    feature_importance = model.get_feature_importance()
    
    plt.figure(figsize=(12, 10))
    top_n = min(20, len(feature_importance))
    sns.barplot(y='feature', x='importance', data=feature_importance.head(top_n))
    plt.title('Top Pilot-Normalized Features')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pilot_normalized_features.png'), dpi=300)
    
    print("Pilot-normalized features analysis completed!")

def analyze_target_distribution(table_data: TableDataset) -> None:
    """Analyze the distribution of target variables."""
    print("\n======= Target Distribution Analysis =======")
    
    # Get target data
    target_cols = [
        'avg_tlx', 'mental_effort', 'avg_tlx_zscore', 
        'avg_mental_effort_zscore', 'avg_tlx_quantile'
    ]
    
    targets = {col: table_data.df[col] for col in target_cols if col in table_data.df.columns}
    
    # Plot distributions
    fig, axes = plt.subplots(len(targets), 1, figsize=(12, 4 * len(targets)))
    
    if len(targets) == 1:
        axes = [axes]
    
    for i, (col, values) in enumerate(targets.items()):
        sns.histplot(values, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].grid(True)
        
        # Add statistics
        if pd.api.types.is_numeric_dtype(values):
            mean = values.mean()
            median = values.median()
            std = values.std()
            skew = values.skew()
            kurt = values.kurtosis()
            
            stats_text = f"Mean: {mean:.2f}, Median: {median:.2f}, Std: {std:.2f}\nSkew: {skew:.2f}, Kurtosis: {kurt:.2f}"
            axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_distributions.png'), dpi=300)
    
    # Check consistency between metrics
    for col1 in targets:
        for col2 in targets:
            if col1 != col2:
                corr = targets[col1].corr(targets[col2])
                print(f"Correlation between {col1} and {col2}: {corr:.4f}")
    
    # Visualize relationships between metrics
    if len(targets) > 1:
        plt.figure(figsize=(12, 10))
        df_targets = pd.DataFrame(targets)
        sns.pairplot(df_targets)
        plt.suptitle('Relationships Between Target Variables', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'target_relationships.png'), dpi=300)


def analyze_simple_sequence_baselines(windowed_data: WindowedDataset) -> None:
    """Analyze performance of simple baselines on sequence data."""
    print("\n======= Simple Sequence Model Baselines =======")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split train/test
    train_trials, test_trials = train_test_split(
        windowed_data.all_trials, 
        test_size=0.2, 
        random_state=42
    )
    
    # Prepare datasets
    train_dataset = PilotDataset(train_trials, feature_type='eng_features')
    test_dataset = PilotDataset(test_trials, feature_type='eng_features')
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Dummy baseline: predict mean of training set
    all_y_train = []
    for _, _, labels, _, _ in train_loader:
        all_y_train.extend(labels.numpy())
    
    mean_target = np.mean(all_y_train)
    print(f"Mean target value: {mean_target:.4f}")
    
    # Evaluate dummy baseline
    all_y_test = []
    for _, _, labels, _, _ in test_loader:
        all_y_test.extend(labels.numpy())
    
    dummy_predictions = np.ones_like(all_y_test) * mean_target
    dummy_metrics = evaluate_regression_model(
        np.array(all_y_test), 
        dummy_predictions,
        verbose=True
    )
    
    # Test a very simple LSTM (1 layer, small hidden size)
    simple_lstm = SequenceModel(
        model_type='lstm',
        input_dim=22,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        batch_size=16,
        learning_rate=0.001,
        epochs=5,
        device=device
    )
    
    print("\nTraining simple LSTM...")
    simple_lstm.train(train_loader, test_loader)
    
    # Evaluate simple LSTM
    print("\nEvaluating simple LSTM:")
    lstm_metrics = simple_lstm.evaluate(test_loader)
    
    # Try a different feature type
    print("\nTrying with raw engineered features...")
    train_dataset_raw = PilotDataset(train_trials, feature_type='raw_eng_features')
    test_dataset_raw = PilotDataset(test_trials, feature_type='raw_eng_features')
    
    train_loader_raw = torch.utils.data.DataLoader(
        train_dataset_raw, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader_raw = torch.utils.data.DataLoader(
        test_dataset_raw, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    simple_lstm_raw = SequenceModel(
        model_type='lstm',
        input_dim=22,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        batch_size=16,
        learning_rate=0.001,
        epochs=5,
        device=device
    )
    
    print("Training simple LSTM with raw features...")
    simple_lstm_raw.train(train_loader_raw, test_loader_raw)
    
    print("\nEvaluating simple LSTM with raw features:")
    lstm_raw_metrics = simple_lstm_raw.evaluate(test_loader_raw)


def main():
    """Main function to run all analyses."""
    args = parse_args()
    
    print(f"Running model diagnostics on the SMU-TexCL dataset")
    print(f"Output directory: {args.output_dir}")
    
    # Ensure output directory is created
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Override OUTPUT_DIR with command-line specified directory
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    
    # Load tabular data
    print("\nLoading tabular data...")
    table_data = TableDataset(args.tabular_data_path)
    table_data.explore_data()
    
    # Load windowed data
    print("\nLoading windowed data...")
    windowed_data = WindowedDataset(args.windowed_data_path)
    windowed_data.load_all_pilots()
    windowed_data.explore_data()
    
    # Run diagnostic analyses
    analyze_target_distribution(table_data)
    analyze_feature_importance(table_data)
    analyze_prediction_errors(table_data)
    perform_pilot_wise_cv(table_data)
    analyze_signal_quality(windowed_data)
    
    # Run additional analyses
    analyze_signal_quality_by_condition(windowed_data)
    analyze_classification_approach(table_data)
    analyze_pilot_normalized_features(table_data)
    analyze_simple_sequence_baselines(windowed_data)
    
    print(f"\nDiagnostic analyses complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
