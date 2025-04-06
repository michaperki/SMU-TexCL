"""
Data preprocessing functions for the SMU-Textron Cognitive Load dataset.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from scipy import signal
from scipy.stats import skew
import pywt


def create_preprocessing_pipeline(
    scaling: str = 'standard',
    impute_strategy: str = 'mean',
    feature_transform: Optional[str] = None,
    outlier_removal: Optional[str] = None,
    outlier_threshold: float = 3.0
) -> Pipeline:
    """Create preprocessing pipeline.
    
    Args:
        scaling: Scaling method ('standard', 'minmax', 'robust', 'none')
        impute_strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant', 'knn')
        feature_transform: Feature transformation method ('log', 'sqrt', 'boxcox', 'yeo-johnson', 'none')
        outlier_removal: Outlier removal method ('zscore', 'iqr', 'none')
        outlier_threshold: Threshold for outlier detection
        
    Returns:
        Preprocessing pipeline
    """
    steps = []
    
    # Add outlier removal
    if outlier_removal and outlier_removal != 'none':
        steps.append(('outlier_remover', OutlierRemover(
            method=outlier_removal, 
            threshold=outlier_threshold
        )))
    
    # Add imputation
    if impute_strategy != 'none':
        if impute_strategy == 'knn':
            steps.append(('imputer', KNNImputer(n_neighbors=5)))
        else:
            steps.append(('imputer', SimpleImputer(strategy=impute_strategy)))
    
    # Add feature transformation
    if feature_transform and feature_transform != 'none':
        if feature_transform in ['boxcox', 'yeo-johnson']:
            steps.append(('transformer', PowerTransformer(method=feature_transform)))
        elif feature_transform == 'log':
            steps.append(('transformer', FunctionTransformer(np.log1p)))
        elif feature_transform == 'sqrt':
            steps.append(('transformer', FunctionTransformer(np.sqrt)))
    
    # Add scaling
    if scaling == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaling == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif scaling == 'robust':
        steps.append(('scaler', RobustScaler()))
    elif scaling != 'none':
        raise ValueError(f"Unknown scaling method: {scaling}")
    
    return Pipeline(steps)


def preprocess_tabular_data(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    scaling: str = 'standard',
    impute_strategy: str = 'mean',
    feature_transform: Optional[str] = None,
    outlier_removal: Optional[str] = None,
    outlier_threshold: float = 3.0
) -> Tuple[pd.DataFrame, Pipeline]:
    """Preprocess tabular data.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature columns to preprocess
        scaling: Scaling method ('standard', 'minmax', 'robust', 'none')
        impute_strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant', 'knn')
        feature_transform: Feature transformation method ('log', 'sqrt', 'boxcox', 'yeo-johnson', 'none')
        outlier_removal: Outlier removal method ('zscore', 'iqr', 'none')
        outlier_threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (Preprocessed DataFrame, Preprocessing pipeline)
    """
    # Select features
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[feature_columns].copy()
    
    # Create preprocessing pipeline
    pipeline = create_preprocessing_pipeline(
        scaling=scaling, 
        impute_strategy=impute_strategy,
        feature_transform=feature_transform,
        outlier_removal=outlier_removal,
        outlier_threshold=outlier_threshold
    )
    
    # Apply preprocessing
    X_preprocessed = pipeline.fit_transform(X)
    
    # Convert back to DataFrame
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_columns, index=df.index)
    
    return X_preprocessed_df, pipeline


class OutlierRemover:
    """Custom transformer for removing outliers."""
    
    def __init__(self, method: str = 'zscore', threshold: float = 3.0):
        """Initialize OutlierRemover.
        
        Args:
            method: Method for outlier detection ('zscore', 'iqr')
            threshold: Threshold for outlier detection
        """
        self.method = method
        self.threshold = threshold
        self.mask_ = None
        self.feature_stats_ = {}
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OutlierRemover':
        """Fit outlier remover to data.
        
        Args:
            X: Input features
            y: Target values (unused)
            
        Returns:
            Self
        """
        # Convert to DataFrame for easier handling
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Calculate statistics for each feature
        for col in X.columns:
            self.feature_stats_[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'q1': X[col].quantile(0.25),
                'q3': X[col].quantile(0.75),
                'iqr': X[col].quantile(0.75) - X[col].quantile(0.25)
            }
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by removing outliers.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        # Convert to DataFrame for easier handling
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        # Create mask for rows to keep
        mask = np.ones(len(X_df), dtype=bool)
        
        for col in X_df.columns:
            if col not in self.feature_stats_:
                continue
            
            stats = self.feature_stats_[col]
            
            if self.method == 'zscore':
                # Z-score method
                z_scores = np.abs((X_df[col] - stats['mean']) / stats['std'])
                mask = mask & (z_scores < self.threshold)
            elif self.method == 'iqr':
                # IQR method
                lower_bound = stats['q1'] - self.threshold * stats['iqr']
                upper_bound = stats['q3'] + self.threshold * stats['iqr']
                mask = mask & (X_df[col] >= lower_bound) & (X_df[col] <= upper_bound)
        
        # Save mask for later inspection
        self.mask_ = mask
        
        # Apply mask and return transformed data
        return X_df[mask].values
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform data.
        
        Args:
            X: Input features
            y: Target values (unused)
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)


class FunctionTransformer:
    """Custom transformer for applying a function to data."""
    
    def __init__(self, func: Callable, inverse_func: Optional[Callable] = None):
        """Initialize FunctionTransformer.
        
        Args:
            func: Function to apply
            inverse_func: Inverse function for inverse_transform
        """
        self.func = func
        self.inverse_func = inverse_func
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FunctionTransformer':
        """Fit transformer to data (no-op).
        
        Args:
            X: Input features
            y: Target values (unused)
            
        Returns:
            Self
        """
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by applying function.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        return self.func(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform data by applying inverse function.
        
        Args:
            X: Input features
            
        Returns:
            Inverse-transformed features
        """
        if self.inverse_func is None:
            raise ValueError("Inverse function not provided")
        return self.inverse_func(X)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform data.
        
        Args:
            X: Input features
            y: Target values (unused)
            
        Returns:
            Transformed features
        """
        return self.transform(X)


def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = 'mean', 
    categorical_strategy: str = 'most_frequent',
    knn_n_neighbors: int = 5
) -> pd.DataFrame:
    """Handle missing values in a DataFrame.
    
    Args:
        df: DataFrame with missing values
        strategy: Strategy for numerical columns ('mean', 'median', 'most_frequent', 'constant', 'knn')
        categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')
        knn_n_neighbors: Number of neighbors for KNN imputation
        
    Returns:
        DataFrame with imputed values
    """
    # Make a copy to avoid modifying the original
    df_imputed = df.copy()
    
    # Handle numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        if strategy == 'knn':
            imputer = KNNImputer(n_neighbors=knn_n_neighbors)
            df_imputed[num_cols] = imputer.fit_transform(df[num_cols])
        else:
            imputer = SimpleImputer(strategy=strategy)
            df_imputed[num_cols] = imputer.fit_transform(df[num_cols])
    
    # Handle categorical columns
    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy=categorical_strategy)
        df_imputed[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    return df_imputed


def remove_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.DataFrame:
    """Remove outliers from DataFrame.
    
    Args:
        df: DataFrame with outliers
        columns: List of columns to check for outliers
        method: Method for outlier detection ('zscore', 'iqr', 'isolation_forest')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Select columns to check
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create mask for rows to keep
    mask = np.ones(len(df), dtype=bool)
    
    if method == 'isolation_forest':
        # Use Isolation Forest for outlier detection
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=threshold, random_state=42)
        outlier_labels = iso_forest.fit_predict(df[columns])
        mask = outlier_labels == 1  # 1 for inliers, -1 for outliers
    else:
        # Use traditional methods
        for col in columns:
            if method == 'zscore':
                # Z-score method
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask = mask & (z_scores < threshold)
            elif method == 'iqr':
                # IQR method
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                mask = mask & (df[col] >= q1 - threshold * iqr) & (df[col] <= q3 + threshold * iqr)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Apply mask
    df_clean = df_clean[mask]
    
    print(f"Removed {len(df) - len(df_clean)} outliers out of {len(df)} rows")
    
    return df_clean


def filter_signal(
    signal_data: np.ndarray,
    fs: float,
    filter_type: str = 'lowpass',
    cutoff_freq: Union[float, Tuple[float, float]] = 5.0,
    order: int = 4,
    window: Optional[str] = None
) -> np.ndarray:
    """Apply filter to signal.
    
    Args:
        signal_data: Signal data
        fs: Sampling frequency
        filter_type: Filter type ('lowpass', 'highpass', 'bandpass', 'bandstop')
        cutoff_freq: Cutoff frequency (or tuple of frequencies for bandpass/bandstop)
        order: Filter order
        window: Window function to apply ('hamming', 'hanning', 'blackman', etc.)
        
    Returns:
        Filtered signal
    """
    nyquist = 0.5 * fs
    
    # Apply window if specified
    if window is not None:
        if window == 'hamming':
            window_func = np.hamming(len(signal_data))
        elif window == 'hanning':
            window_func = np.hanning(len(signal_data))
        elif window == 'blackman':
            window_func = np.blackman(len(signal_data))
        else:
            raise ValueError(f"Unknown window function: {window}")
        
        signal_data = signal_data * window_func
    
    # Design filter
    if filter_type == 'lowpass':
        b, a = signal.butter(order, cutoff_freq / nyquist, btype='lowpass')
    elif filter_type == 'highpass':
        b, a = signal.butter(order, cutoff_freq / nyquist, btype='highpass')
    elif filter_type in ['bandpass', 'bandstop']:
        if not isinstance(cutoff_freq, tuple) or len(cutoff_freq) != 2:
            raise ValueError(f"{filter_type} filter requires a tuple of (low, high) cutoff frequencies")
        b, a = signal.butter(order, [f / nyquist for f in cutoff_freq], btype=filter_type)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply filter
    return signal.filtfilt(b, a, signal_data)


def extract_features_from_window(window: Dict[str, Any], include_frequency_domain: bool = False) -> np.ndarray:
    """Extract features from a window of time series data.
    
    Args:
        window: Window dictionary with time series data
        include_frequency_domain: Whether to include frequency domain features
        
    Returns:
        Feature vector
    """
    features = []
    
    # Add PPG features
    if 'ppg_input' in window and window['ppg_input']:
        ppg = np.array(window['ppg_input'][0])
        
        # Time domain features
        features.extend([
            np.mean(ppg),
            np.std(ppg),
            np.max(ppg),
            np.min(ppg),
            np.median(ppg),
            skew(ppg),  # Skewness
            np.percentile(ppg, 25),  # 25th percentile
            np.percentile(ppg, 75),  # 75th percentile
        ])
        
        # Frequency domain features
        if include_frequency_domain:
            ppg_freq = np.abs(np.fft.rfft(ppg))
            ppg_freq = ppg_freq / len(ppg)  # Normalize
            
            # Extract features from frequency domain
            features.extend([
                np.sum(ppg_freq),  # Total power
                np.max(ppg_freq),  # Peak power
                np.mean(ppg_freq),  # Mean power
                np.std(ppg_freq),  # Std of power
            ])
    
    # Add EDA features
    if 'eda_input' in window and window['eda_input']:
        eda = np.array(window['eda_input'][0])
        
        # Time domain features
        features.extend([
            np.mean(eda),
            np.std(eda),
            np.max(eda),
            np.min(eda),
            np.median(eda),
            skew(eda),  # Skewness
            np.percentile(eda, 25),  # 25th percentile
            np.percentile(eda, 75),  # 75th percentile
        ])
        
        # Frequency domain features
        if include_frequency_domain:
            eda_freq = np.abs(np.fft.rfft(eda))
            eda_freq = eda_freq / len(eda)  # Normalize
            
            # Extract features from frequency domain
            features.extend([
                np.sum(eda_freq),  # Total power
                np.max(eda_freq),  # Peak power
                np.mean(eda_freq),  # Mean power
                np.std(eda_freq),  # Std of power
            ])
    
    # Add accelerometer features
    if 'accel_input' in window and window['accel_input']:
        accel = np.array(window['accel_input'][0])
        features.extend([
            np.mean(accel),
            np.std(accel),
            np.max(accel),
            np.min(accel),
            np.median(accel)
        ])
    
    # Add temperature features
    if 'temp_input' in window and window['temp_input']:
        temp = np.array(window['temp_input'][0])
        features.extend([
            np.mean(temp),
            np.std(temp),
            np.max(temp),
            np.min(temp),
            np.median(temp)
        ])
    
    # Add engineering features if available
    if 'eng_features_input' in window and window['eng_features_input']:
        eng_features = np.array(window['eng_features_input'][:22])  # Only first 22 are valid
        features.extend(eng_features)
    
    return np.array(features)


def extract_wavelet_features(signal_data: np.ndarray, wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    """Extract wavelet transform features from signal.
    
    Args:
        signal_data: Signal data
        wavelet: Wavelet type
        level: Decomposition level
        
    Returns:
        Wavelet features
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    
    # Extract features from each level
    features = []
    
    # Process approximation coefficients
    cA = coeffs[0]
    features.extend([
        np.mean(cA),
        np.std(cA),
        np.max(cA),
        np.min(cA),
        np.median(cA)
    ])
    
    # Process detail coefficients
    for i, cD in enumerate(coeffs[1:], 1):
        features.extend([
            np.mean(cD),
            np.std(cD),
            np.max(cD),
            np.min(cD),
            np.median(cD)
        ])
    
    return np.array(features)


def assess_signal_quality(
    signal_data: np.ndarray,
    fs: float,
    signal_type: str = 'ppg'
) -> Dict[str, float]:
    """Assess quality of physiological signal.
    
    Args:
        signal_data: Signal data
        fs: Sampling frequency
        signal_type: Type of signal ('ppg', 'eda', 'accel', 'temp')
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Calculate signal-to-noise ratio
    signal_mean = np.mean(signal_data)
    signal_std = np.std(signal_data)
    metrics['snr'] = signal_mean / signal_std if signal_std > 0 else 0
    
    # Calculate power in different frequency bands
    n = len(signal_data)
    freqs = np.fft.rfftfreq(n, 1/fs)
    fft_vals = np.abs(np.fft.rfft(signal_data))
    
    # Normalized power spectrum
    power = fft_vals**2 / n
    
    # Quality metrics based on signal type
    if signal_type == 'ppg':
        # PPG typically has heart rate information in 0.5-3.5 Hz
        hr_band = (freqs >= 0.5) & (freqs <= 3.5)
        noise_band = ~hr_band
        
        hr_power = np.sum(power[hr_band])
        noise_power = np.sum(power[noise_band])
        
        metrics['hr_snr'] = hr_power / noise_power if noise_power > 0 else 0
        metrics['hr_power_ratio'] = hr_power / np.sum(power) if np.sum(power) > 0 else 0
        
        # Detect flatlines or extreme values
        metrics['flatline_ratio'] = np.sum(np.diff(signal_data) == 0) / (n - 1)
        metrics['extreme_ratio'] = np.sum(np.abs(signal_data - signal_mean) > 3*signal_std) / n
    
    elif signal_type == 'eda':
        # EDA typically has information below 0.5 Hz
        eda_band = freqs <= 0.5
        noise_band = ~eda_band
        
        eda_power = np.sum(power[eda_band])
        noise_power = np.sum(power[noise_band])
        
        metrics['eda_snr'] = eda_power / noise_power if noise_power > 0 else 0
        metrics['eda_power_ratio'] = eda_power / np.sum(power) if np.sum(power) > 0 else 0
        
        # Detect flatlines
        metrics['flatline_ratio'] = np.sum(np.diff(signal_data) == 0) / (n - 1)
    
    # Calculate overall quality score (0-1)
    if signal_type == 'ppg':
        metrics['quality_score'] = (
            (1 - metrics['flatline_ratio']) * 0.3 + 
            (1 - metrics['extreme_ratio']) * 0.3 + 
            min(metrics['hr_power_ratio'] * 2, 1) * 0.4
        )
    elif signal_type == 'eda':
        metrics['quality_score'] = (
            (1 - metrics['flatline_ratio']) * 0.4 + 
            min(metrics['eda_power_ratio'] * 2, 1) * 0.6
        )
    else:
        metrics['quality_score'] = 1 - (np.std(signal_data) / np.max(np.abs(signal_data)) if np.max(np.abs(signal_data)) > 0 else 0)
    
    return metrics


def normalize_label(
    label: float,
    method: str = 'zscore',
    stats: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """Normalize a label value.
    
    Args:
        label: Label value
        method: Normalization method ('zscore', 'minmax', 'quantile')
        stats: Statistics for normalization (mean, std, min, max)
        
    Returns:
        Tuple of (Normalized label, Statistics)
    """
    if stats is None:
        stats = {}
    
    if method == 'zscore':
        if 'mean' not in stats or 'std' not in stats:
            return label, stats
        return (label - stats['mean']) / stats['std'], stats
    
    elif method == 'minmax':
        if 'min' not in stats or 'max' not in stats:
            return label, stats
        if stats['max'] - stats['min'] == 0:
            return 0.5, stats
        return (label - stats['min']) / (stats['max'] - stats['min']), stats
    
    elif method == 'quantile':
        if 'quantile_func' not in stats:
            return label, stats
        return stats['quantile_func'](label), stats
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_pilot_statistics(
    pilot_data: List[Dict[str, Any]],
    label_key: str = 'label'
) -> Dict[str, float]:
    """Compute statistics for a pilot's data.
    
    Args:
        pilot_data: List of trial data for a pilot
        label_key: Key for the label value
        
    Returns:
        Dictionary of statistics
    """
    labels = [trial[label_key] for trial in pilot_data if label_key in trial]
    
    if not labels:
        return {}
    
    stats = {
        'mean': np.mean(labels),
        'std': np.std(labels),
        'min': np.min(labels),
        'max': np.max(labels),
        'median': np.median(labels)
    }
    
    # Create quantile function
    from scipy import stats as scipy_stats
    labels_array = np.array(labels)
    stats['quantile_func'] = lambda x: scipy_stats.percentileofscore(labels_array, x) / 100
    
    return stats


"""
Implemented improvements:
1. Added advanced feature transformation options (log, sqrt, boxcox, yeo-johnson)
2. Implemented outlier handling with multiple methods (zscore, iqr, isolation_forest)
3. Added KNN imputation for missing values
4. Implemented signal quality assessment for physiological signals
5. Added frequency domain feature extraction
6. Implemented wavelet transform feature extraction
7. Added window function options for signal filtering
8. Created custom transformers for outlier removal and function transformation
9. Enhanced feature extraction with additional statistical measures
10. Improved signal filtering with additional options
"""
