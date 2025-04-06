"""
Model evaluation functions for the SMU-Textron Cognitive Load dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    cohen_kappa_score,
    explained_variance_score,
    mean_absolute_percentage_error,
    balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, TimeSeriesSplit

from utils.visualization import save_or_show_plot


def evaluate_regression_model(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    verbose: bool = True,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate regression model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        verbose: Whether to print results
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metrics
    """
    # Default metrics to calculate
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'r2', 'explained_variance', 'mape']
    
    # Dictionary to store results
    results = {}
    
    # Calculate metrics
    if 'mse' in metrics:
        results['mse'] = mean_squared_error(y_true, y_pred)
    
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true, y_pred)
    
    if 'r2' in metrics:
        results['r2'] = r2_score(y_true, y_pred)
    
    if 'explained_variance' in metrics:
        results['explained_variance'] = explained_variance_score(y_true, y_pred)
    
    if 'mape' in metrics:
        # Handle zero values in y_true
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            results['mape'] = mean_absolute_percentage_error(
                y_true[non_zero_mask], y_pred[non_zero_mask]
            )
        else:
            results['mape'] = np.nan
    
    if 'median_ae' in metrics:
        results['median_ae'] = np.median(np.abs(y_true - y_pred))
    
    if 'max_error' in metrics:
        results['max_error'] = np.max(np.abs(y_true - y_pred))
    
    if verbose:
        print("Regression Metrics:")
        for metric, value in results.items():
            print(f"{metric.upper()}: {value:.4f}")
    
    return results


def evaluate_classification_model(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    average: str = 'weighted',
    verbose: bool = True,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate classification model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_score: Predicted probabilities (for ROC AUC)
        average: Averaging method for multi-class metrics
        verbose: Whether to print results
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metrics
    """
    # Default metrics to calculate
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'kappa']
    
    # Dictionary to store results
    results = {}
    
    # Calculate metrics
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_true, y_pred)
    
    if 'balanced_accuracy' in metrics:
        results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    if 'precision' in metrics:
        results['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    
    if 'recall' in metrics:
        results['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    
    if 'f1' in metrics:
        results['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    if 'kappa' in metrics:
        results['kappa'] = cohen_kappa_score(y_true, y_pred)
    
    if 'roc_auc' in metrics and y_score is not None:
        try:
            if y_score.ndim == 1:
                # Binary classification
                results['roc_auc'] = roc_auc_score(y_true, y_score)
            else:
                # Multi-class
                results['roc_auc'] = roc_auc_score(y_true, y_score, average=average, multi_class='ovr')
        except ValueError:
            results['roc_auc'] = np.nan
    
    if verbose:
        print("Classification Metrics:")
        for metric, value in results.items():
            print(f"{metric.upper()}: {value:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
    
    return results


def plot_regression_results(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Values",
    save_path: Optional[str] = None,
    show_metrics: bool = True,
    residuals: bool = False
) -> None:
    """Plot regression results.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the figure
        show_metrics: Whether to show metrics in the plot
        residuals: Whether to include residual plot
    """
    if residuals:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of actual vs predicted
    ax1.scatter(y_true, y_pred, alpha=0.5)
    
    # Add diagonal line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add regression metrics to the plot
    if show_metrics:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        ax1.text(
            0.05, 0.95, 
            f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}",
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', alpha=0.1)
        )
    
    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title(title)
    ax1.grid(True)
    
    # Add residuals plot if requested
    if residuals:
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals Plot")
        ax2.grid(True)
        
        # Add trend line to residuals
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, residuals)
            x_line = np.linspace(min(y_pred), max(y_pred), 100)
            y_line = slope * x_line + intercept
            ax2.plot(x_line, y_line, 'g-', alpha=0.7, 
                    label=f'Trend: y={slope:.4f}x+{intercept:.4f}')
            ax2.legend()
        except:
            pass
    
    plt.tight_layout()
    save_or_show_plot(save_path, "Regression results plot saved")


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = False
) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        class_names: Names of classes
        title: Plot title
        save_path: Path to save the figure
        normalize: Whether to normalize the confusion matrix
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Get class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add class names
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_or_show_plot(save_path, "Confusion matrix plot saved")


def plot_roc_curve(
    y_true: np.ndarray, 
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_score: Predicted probabilities or decision scores
        class_names: Names of classes
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Dictionary with AUC values
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    plt.figure(figsize=(10, 8))
    
    # Dictionary to store AUC values
    auc_values = {}
    
    # Handle multi-class case
    if y_score.ndim > 1 and y_score.shape[1] > 1:
        # One-hot encode true labels
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(classes))]
        
        # Plot ROC curve for each class
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            auc_values[class_names[i]] = roc_auc
            plt.plot(
                fpr, tpr,
                lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})'
            )
    else:
        # Binary case
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        auc_values["ROC"] = roc_auc
        plt.plot(
            fpr, tpr,
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})'
        )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    save_or_show_plot(save_path, "ROC curve plot saved")
    
    return auc_values


def plot_precision_recall_curve(
    y_true: np.ndarray, 
    y_score: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """Plot precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_score: Predicted probabilities or decision scores
        class_names: Names of classes
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Dictionary with AP values
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    plt.figure(figsize=(10, 8))
    
    # Dictionary to store AP values
    ap_values = {}
    
    # Handle multi-class case
    if y_score.ndim > 1 and y_score.shape[1] > 1:
        # One-hot encode true labels
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(classes))]
        
        # Plot PR curve for each class
        for i in range(len(classes)):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
            ap_values[class_names[i]] = ap
            plt.plot(
                recall, precision,
                lw=2,
                label=f'{class_names[i]} (AP = {ap:.2f})'
            )
    else:
        # Binary case
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ap_values["PR"] = ap
        plt.plot(
            recall, precision,
            lw=2,
            label=f'PR curve (AP = {ap:.2f})'
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    
    save_or_show_plot(save_path, "Precision-recall curve plot saved")
    
    return ap_values


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None
) -> Tuple[float, float]:
    """Plot calibration curve.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve
        title: Plot title
        save_path: Path to save the figure
        
    Returns:
        Tuple of (Expected Calibration Error, Maximum Calibration Error)
    """
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Compute calibration errors
    ece = np.sum(np.abs(prob_true - prob_pred) * np.histogram(y_prob, bins=n_bins)[0] / len(y_prob))
    mce = np.max(np.abs(prob_true - prob_pred))
    
    # Plot calibration curve
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, "s-", label=f"Model (ECE={ece:.3f}, MCE={mce:.3f})")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    
    save_or_show_plot(save_path, "Calibration curve plot saved")
    
    return ece, mce


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, Any] = 5,
    scoring: Union[str, List[str]] = 'r2',
    is_classification: bool = False,
    is_time_series: bool = False,
    verbose: bool = True,
    return_train_score: bool = False
) -> Dict[str, np.ndarray]:
    """Perform cross-validation.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv: Number of folds or cross-validation strategy
        scoring: Scoring metric(s)
        is_classification: Whether it's a classification task
        is_time_series: Whether data is time series
        verbose: Whether to print results
        return_train_score: Whether to return training scores
        
    Returns:
        Dictionary of scores
    """
    # Choose cross-validation strategy if not provided
    if isinstance(cv, int):
        if is_time_series:
            cv_strategy = TimeSeriesSplit(n_splits=cv)
        elif is_classification:
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_strategy = cv
    
    # Handle multiple scoring metrics
    if isinstance(scoring, list):
        from sklearn.model_selection import cross_validate
        cv_results = cross_validate(
            model, X, y, 
            cv=cv_strategy, 
            scoring=scoring,
            return_train_score=return_train_score
        )
        scores = {}
        for metric in scoring:
            test_key = f'test_{metric}'
            scores[metric] = cv_results[test_key]
            
            if return_train_score:
                train_key = f'train_{metric}'
                scores[f'train_{metric}'] = cv_results[train_key]
    else:
        scores = {scoring: cross_val_score(
            model, X, y, cv=cv_strategy, scoring=scoring
        )}
    
    if verbose:
        print("Cross-Validation Results:")
        for metric, values in scores.items():
            if metric.startswith('train_'):
                continue
            print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return scores


def plot_learning_curve(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Learning Curve",
    cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
    scoring: str = 'r2',
    is_classification: bool = False,
    is_time_series: bool = False,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot learning curve.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        title: Plot title
        cv: Number of folds
        train_sizes: Training set sizes to evaluate
        scoring: Scoring metric
        is_classification: Whether it's a classification task
        is_time_series: Whether data is time series
        save_path: Path to save the figure
        
    Returns:
        Tuple of (train_sizes, train_scores, test_scores)
    """
    from sklearn.model_selection import learning_curve
    
    # Choose cross-validation strategy
    if is_time_series:
        cv_strategy = TimeSeriesSplit(n_splits=cv)
    elif is_classification:
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate learning curve
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv_strategy, scoring=scoring
    )
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel(f"Score ({scoring})")
    plt.grid(True)
    
    plt.fill_between(
        train_sizes_abs, train_mean - train_std, train_mean + train_std,
        alpha=0.1, color="r"
    )
    plt.fill_between(
        train_sizes_abs, test_mean - test_std, test_mean + test_std,
        alpha=0.1, color="g"
    )
    plt.plot(train_sizes_abs, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes_abs, test_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    
    save_or_show_plot(save_path, "Learning curve plot saved")
    
    return train_sizes_abs, train_scores, test_scores


def compare_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_classification: bool = False,
    save_path: Optional[str] = None,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compare multiple models.
    
    Args:
        models: Dictionary of models to compare
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        is_classification: Whether it's a classification task
        save_path: Path to save the comparison plot
        metrics: List of metrics to compute
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    # Default metrics
    if metrics is None:
        if is_classification:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']
        else:
            metrics = ['mse', 'rmse', 'mae', 'r2']
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available (for classification)
        y_prob = None
        if is_classification and hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)
            except:
                pass
        
        # Evaluate the model
        if is_classification:
            metrics_dict = evaluate_classification_model(
                y_test, y_pred, y_score=y_prob, 
                verbose=False, metrics=metrics
            )
            
            result = {'Model': name}
            result.update(metrics_dict)
        else:
            metrics_dict = evaluate_regression_model(
                y_test, y_pred, verbose=False, metrics=metrics
            )
            
            result = {'Model': name}
            result.update(metrics_dict)
        
        # Add training time if available
        if hasattr(model, 'fit_time_'):
            result['Training Time'] = model.fit_time_
        
        results.append(result)
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    # Plot comparison
    if save_path:
        plt.figure(figsize=(12, 6))
        
        # Metrics to plot
        if is_classification:
            metrics_to_plot = [m for m in metrics if m in results_df.columns]
        else:
            # For regression, we plot negative MSE and MAE since lower is better
            metrics_to_plot = []
            
            if 'mse' in results_df.columns:
                results_df['Neg MSE'] = -results_df['mse']
                metrics_to_plot.append('Neg MSE')
                
            if 'rmse' in results_df.columns:
                results_df['Neg RMSE'] = -results_df['rmse']
                metrics_to_plot.append('Neg RMSE')
                
            if 'mae' in results_df.columns:
                results_df['Neg MAE'] = -results_df['mae']
                metrics_to_plot.append('Neg MAE')
                
            if 'r2' in results_df.columns:
                metrics_to_plot.append('r2')
        
        # Prepare data for plotting
        models = results_df['Model']
        metrics_data = []
        
        for metric in metrics_to_plot:
            values = results_df[metric]
            metrics_data.append((metric, values))
        
        # Create subplots
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, (metric, values) in enumerate(metrics_data):
            axes[i].bar(models, values)
            axes[i].set_title(metric)
            axes[i].set_ylabel(metric)
            axes[i].set_xlabel('Model')
            axes[i].set_xticklabels(models, rotation=45, ha='right')
            
            # Add values on top of bars
            for j, v in enumerate(values):
                axes[i].text(j, v, f"{v:.4f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    return results_df


def calculate_confidence_intervals(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    metric_func: Callable,
    confidence: float = 0.95,
    n_bootstraps: int = 1000,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence intervals for a metric.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metric_func: Function to calculate metric
        confidence: Confidence level
        n_bootstraps: Number of bootstrap samples
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (mean, lower bound, upper bound)
    """
    np.random.seed(random_state)
    
    # Calculate base metric value
    base_metric = metric_func(y_true, y_pred)
    
    # Perform bootstrap sampling
    bootstrap_metrics = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        
        # Calculate metric on bootstrap sample
        bootstrap_metric = metric_func(y_true[indices], y_pred[indices])
        bootstrap_metrics.append(bootstrap_metric)
    
    # Calculate confidence interval
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(bootstrap_metrics, 100 * alpha)
    upper_bound = np.percentile(bootstrap_metrics, 100 * (1 - alpha))
    
    return base_metric, lower_bound, upper_bound


def evaluate_time_series_forecast(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    time_steps: Optional[np.ndarray] = None,
    verbose: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate time series forecast.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        time_steps: Time steps for plotting
        verbose: Whether to print results
        save_path: Path to save the figure
        
    Returns:
        Dictionary of metrics
    """
    # Calculate metrics
    metrics = evaluate_regression_model(y_true, y_pred, verbose=False)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    with np.errstate(divide='ignore', invalid='ignore'):
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.nan
    
    # Add MAPE to metrics
    metrics['mape'] = mape
    
    if verbose:
        print("Time Series Forecast Metrics:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MAPE: {mape:.4f}%")
        print(f"R²: {metrics['r2']:.4f}")
    
    # Plot forecast vs. actual
    if save_path:
        plt.figure(figsize=(12, 6))
        
        if time_steps is None:
            time_steps = np.arange(len(y_true))
        
        plt.plot(time_steps, y_true, 'b-', label='Actual')
        plt.plot(time_steps, y_pred, 'r--', label='Forecast')
        
        plt.title('Time Series Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Forecast plot saved to {save_path}")
    
    return metrics


def evaluate_pilot_level_performance(
    pilot_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_classification: bool = False,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Evaluate model performance for each pilot.
    
    Args:
        pilot_ids: List of pilot IDs
        y_true: True values
        y_pred: Predicted values
        is_classification: Whether it's a classification task
        save_path: Path to save the figure
        
    Returns:
        DataFrame with per-pilot metrics
    """
    # Create DataFrame with predictions and pilot IDs
    results_df = pd.DataFrame({
        'pilot_id': pilot_ids,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    # Calculate metrics for each pilot
    pilot_metrics = []
    
    for pilot_id in np.unique(pilot_ids):
        # Get data for this pilot
        pilot_data = results_df[results_df['pilot_id'] == pilot_id]
        
        if is_classification:
            # Classification metrics
            accuracy = accuracy_score(pilot_data['y_true'], pilot_data['y_pred'])
            precision = precision_score(
                pilot_data['y_true'], pilot_data['y_pred'], 
                average='weighted', zero_division=0
            )
            recall = recall_score(
                pilot_data['y_true'], pilot_data['y_pred'], 
                average='weighted', zero_division=0
            )
            f1 = f1_score(
                pilot_data['y_true'], pilot_data['y_pred'], 
                average='weighted', zero_division=0
            )
            
            pilot_metrics.append({
                'pilot_id': pilot_id,
                'n_samples': len(pilot_data),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        else:
            # Regression metrics
            mse = mean_squared_error(pilot_data['y_true'], pilot_data['y_pred'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(pilot_data['y_true'], pilot_data['y_pred'])
            r2 = r2_score(pilot_data['y_true'], pilot_data['y_pred'])
            
            pilot_metrics.append({
                'pilot_id': pilot_id,
                'n_samples': len(pilot_data),
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
    
    # Create DataFrame with metrics
    metrics_df = pd.DataFrame(pilot_metrics)
    
    # Plot pilot-level metrics
    if save_path:
        plt.figure(figsize=(12, 6))
        
        # Sort by pilot ID
        metrics_df = metrics_df.sort_values('pilot_id')
        
        # Select metrics to plot
        if is_classification:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            title = 'Classification Metrics by Pilot'
        else:
            metrics_to_plot = ['rmse', 'mae', 'r2']
            title = 'Regression Metrics by Pilot'
        
        # Create subplots
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            axes[i].bar(metrics_df['pilot_id'], metrics_df[metric])
            axes[i].set_title(metric.upper())
            axes[i].set_ylabel(metric)
            axes[i].set_xlabel('Pilot ID')
            
            # Handle tick labels for large number of pilots
            if len(metrics_df) > 10:
                step = max(1, len(metrics_df) // 10)
                tick_indices = np.arange(0, len(metrics_df), step)
                axes[i].set_xticks(tick_indices)
                axes[i].set_xticklabels(metrics_df['pilot_id'].iloc[tick_indices], rotation=45, ha='right')
            else:
                axes[i].set_xticklabels(metrics_df['pilot_id'], rotation=45, ha='right')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pilot-level metrics plot saved to {save_path}")
    
    return metrics_df


def feature_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
    feature_names: Optional[List[str]] = None,
    plot: bool = True,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Calculate feature importance by permutation.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        n_repeats: Number of times to permute each feature
        random_state: Random state for reproducibility
        feature_names: Names of features
        plot: Whether to plot importance
        save_path: Path to save plot
        
    Returns:
        DataFrame with feature importances
    """
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    result = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats,
        random_state=random_state
    )
    
    # Get feature names if not provided
    if feature_names is None:
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrame with importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Plot importance if requested
    if plot:
        plt.figure(figsize=(12, 8))
        
        # Select top features to plot
        top_n = min(20, len(importance_df))
        top_features = importance_df.head(top_n)
        
        # Create barplot
        plt.barh(range(top_n), top_features['importance_mean'], align='center')
        plt.yticks(range(top_n), top_features['feature'])
        plt.xlabel('Mean Decrease in Metric')
        plt.ylabel('Feature')
        plt.title('Permutation Feature Importance')
        plt.grid(True, axis='x')
        plt.tight_layout()
        
        # Add error bars
        plt.errorbar(
            top_features['importance_mean'], 
            range(top_n),
            xerr=top_features['importance_std'],
            fmt='o', 
            color='black',
            ecolor='lightgray', 
            elinewidth=2, 
            capsize=5
        )
        
        save_or_show_plot(save_path, "Feature importance plot saved")
    
    return importance_df


def evaluate_model_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attributes: np.ndarray,
    attribute_values: Optional[List[Any]] = None,
    is_classification: bool = True,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Evaluate model fairness across different groups.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        protected_attributes: Array of protected attribute values for each sample
        attribute_values: List of attribute values to evaluate
        is_classification: Whether it's a classification task
        save_path: Path to save figure
        
    Returns:
        DataFrame with fairness metrics
    """
    # Determine attribute values if not provided
    if attribute_values is None:
        attribute_values = np.unique(protected_attributes)
    
    fairness_metrics = []
    
    for attr_value in attribute_values:
        # Get indices for this group
        group_indices = (protected_attributes == attr_value)
        
        # Skip if no samples in this group
        if not np.any(group_indices):
            continue
        
        # Calculate metrics for this group
        group_y_true = y_true[group_indices]
        group_y_pred = y_pred[group_indices]
        
        if is_classification:
            group_metrics = evaluate_classification_model(
                group_y_true, group_y_pred, verbose=False
            )
        else:
            group_metrics = evaluate_regression_model(
                group_y_true, group_y_pred, verbose=False
            )
        
        # Add group info
        group_metrics['group'] = attr_value
        group_metrics['count'] = np.sum(group_indices)
        
        fairness_metrics.append(group_metrics)
    
    # Create DataFrame
    fairness_df = pd.DataFrame(fairness_metrics)
    
    # Plot fairness metrics
    if save_path:
        plt.figure(figsize=(12, 6))
        
        # Select metrics to plot
        if is_classification:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            title = 'Classification Metrics by Group'
        else:
            metrics_to_plot = ['rmse', 'mae', 'r2']
            title = 'Regression Metrics by Group'
        
        # Create subplots
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_plot):
            if metric in fairness_df.columns:
                axes[i].bar(fairness_df['group'], fairness_df[metric])
                axes[i].set_title(metric.upper())
                axes[i].set_ylabel(metric)
                axes[i].set_xlabel('Group')
                
                # Add count labels
                for j, (group, count) in enumerate(zip(fairness_df['group'], fairness_df['count'])):
                    axes[i].text(j, 0.02, f"n={count}", ha='center')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fairness metrics plot saved to {save_path}")
    
    return fairness_df


"""
Implemented improvements:
1. Enhanced regression metrics with explained_variance, MAPE, median_ae
2. Added support for balanced_accuracy and Cohen's kappa in classification metrics
3. Added customizable metrics parameter to both regression and classification evaluation
4. Added residuals plot option to regression visualization
5. Added normalization option to confusion matrix visualization
6. Implemented calibration curve plotting with calibration errors
7. Enhanced cross-validation with time series support and return_train_score option
8. Improved model comparison with customizable metrics
9. Added feature permutation importance calculation and visualization
10. Added model fairness evaluation across protected attributes
11. Implemented detailed evaluation for time series forecasting
12. Added pilot-level performance evaluation

Not yet implemented:
13. Specialized metrics for imbalanced datasets
14. Support for statistical significance testing
15. Model explanation (SHAP, LIME) integration
16. Model drift detection
17. Model robustness evaluation
18. Multi-modal evaluation metrics
19. Evaluation by turbulence level
20. Reliability diagrams
"""
