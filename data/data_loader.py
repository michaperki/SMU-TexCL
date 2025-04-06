"""
Base data loading functionality for the SMU-Textron Cognitive Load dataset.

This module provides common functionality for data loading that is used by
both the TableDataset and WindowedDataset classes. It includes support for
various file formats, caching, parallel loading, and data validation.
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Set
from abc import ABC, abstractmethod
import hashlib
import pickle
import concurrent.futures
from pathlib import Path
import glob
import re
import time
from functools import lru_cache
from tqdm import tqdm


# Set up logging
logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, cache_dir: Optional[str] = '.cache'):
        """Initialize the dataset.
        
        Args:
            cache_dir: Directory for caching loaded data
        """
        self.cache_dir = cache_dir
        self._create_cache_dir()
        self.data_validator = DataValidator()
        
    def _create_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset."""
        pass
    
    @abstractmethod
    def explore_data(self) -> None:
        """Print basic statistics and information about the dataset."""
        pass
    
    def cache_key(self, identifier: str) -> str:
        """Generate cache key for a dataset.
        
        Args:
            identifier: Unique identifier for the dataset
            
        Returns:
            Cache key string
        """
        return hashlib.md5(identifier.encode()).hexdigest()
    
    def cache_path(self, identifier: str) -> str:
        """Generate cache file path for a dataset.
        
        Args:
            identifier: Unique identifier for the dataset
            
        Returns:
            Cache file path
        """
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{self.cache_key(identifier)}.pkl")
    
    def save_to_cache(self, data: Any, identifier: str) -> bool:
        """Save data to cache.
        
        Args:
            data: Data to cache
            identifier: Unique identifier for the dataset
            
        Returns:
            True if successful, False otherwise
        """
        if not self.cache_dir:
            return False
            
        try:
            cache_file = self.cache_path(identifier)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Data cached to {cache_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache data: {str(e)}")
            return False
    
    def load_from_cache(self, identifier: str) -> Optional[Any]:
        """Load data from cache.
        
        Args:
            identifier: Unique identifier for the dataset
            
        Returns:
            Cached data if available, None otherwise
        """
        if not self.cache_dir:
            return None
            
        cache_file = self.cache_path(identifier)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Data loaded from cache: {cache_file}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached data: {str(e)}")
                return None
        return None
    
    def clear_cache(self, identifier: Optional[str] = None) -> None:
        """Clear cached data.
        
        Args:
            identifier: Optional specific identifier to clear, or None to clear all
        """
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return
            
        if identifier:
            cache_file = self.cache_path(identifier)
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.debug(f"Cleared cache for {identifier}")
        else:
            # Clear all cache files
            for cache_file in glob.glob(os.path.join(self.cache_dir, "*.pkl")):
                os.remove(cache_file)
            logger.debug("Cleared all cache files")
    
    def validate_data(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate data using the data validator.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        return self.data_validator.validate(data)


class DataValidator:
    """Class for validating dataset integrity."""
    
    def __init__(self):
        """Initialize the data validator."""
        self.validation_rules = []
        
    def add_rule(self, rule_func: Callable, error_message: str) -> None:
        """Add a validation rule.
        
        Args:
            rule_func: Function that returns True if data is valid
            error_message: Error message to show if validation fails
        """
        self.validation_rules.append((rule_func, error_message))
    
    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate data against all rules.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        for rule_func, error_message in self.validation_rules:
            try:
                if not rule_func(data):
                    errors.append(error_message)
            except Exception as e:
                errors.append(f"Validation error: {str(e)}")
        
        return len(errors) == 0, errors


class FileFormatHandler:
    """Class for handling different file formats."""
    
    @staticmethod
    def detect_format(file_path: str) -> str:
        """Detect file format from extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            File format string
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.json':
            return 'json'
        elif ext in ['.csv', '.tsv']:
            return 'csv'
        elif ext == '.parquet':
            return 'parquet'
        elif ext in ['.xls', '.xlsx']:
            return 'excel'
        elif ext == '.hdf5':
            return 'hdf5'
        elif ext in ['.yaml', '.yml']:
            return 'yaml'
        else:
            return 'unknown'
    
    @staticmethod
    def load_file(file_path: str, **kwargs) -> Any:
        """Load data from a file based on its format.
        
        Args:
            file_path: Path to file
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            Loaded data
        """
        format_type = FileFormatHandler.detect_format(file_path)
        
        try:
            if format_type == 'json':
                return FileFormatHandler.load_json(file_path, **kwargs)
            elif format_type == 'csv':
                return FileFormatHandler.load_csv(file_path, **kwargs)
            elif format_type == 'parquet':
                return FileFormatHandler.load_parquet(file_path, **kwargs)
            elif format_type == 'excel':
                return FileFormatHandler.load_excel(file_path, **kwargs)
            elif format_type == 'hdf5':
                return FileFormatHandler.load_hdf5(file_path, **kwargs)
            elif format_type == 'yaml':
                return FileFormatHandler.load_yaml(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {format_type}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_json(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """Load data from a JSON file.
        
        Args:
            file_path: Path to JSON file
            encoding: File encoding
            
        Returns:
            Loaded JSON data as dictionary
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from a CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded CSV data as DataFrame
        """
        try:
            # Default settings for CSV loading
            defaults = {
                'delimiter': None,  # Auto-detect
                'encoding': 'utf-8',
                'skip_blank_lines': True,
                'low_memory': False
            }
            
            # Update defaults with provided kwargs
            params = {**defaults, **kwargs}
            
            # Detect if TSV based on extension
            if file_path.lower().endswith('.tsv'):
                params['delimiter'] = '\t'
            
            return pd.read_csv(file_path, **params)
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_parquet(file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from a Parquet file.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional arguments for pd.read_parquet
            
        Returns:
            Loaded Parquet data as DataFrame
        """
        try:
            return pd.read_parquet(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading Parquet {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_excel(file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from an Excel file.
        
        Args:
            file_path: Path to Excel file
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            Loaded Excel data as DataFrame
        """
        try:
            return pd.read_excel(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading Excel {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_hdf5(file_path: str, key: str = 'data', **kwargs) -> pd.DataFrame:
        """Load data from an HDF5 file.
        
        Args:
            file_path: Path to HDF5 file
            key: Group key in HDF5 file
            **kwargs: Additional arguments for pd.read_hdf
            
        Returns:
            Loaded HDF5 data as DataFrame
        """
        try:
            return pd.read_hdf(file_path, key=key, **kwargs)
        except Exception as e:
            logger.error(f"Error loading HDF5 {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_yaml(file_path: str, **kwargs) -> Dict[str, Any]:
        """Load data from a YAML file.
        
        Args:
            file_path: Path to YAML file
            **kwargs: Additional arguments for yaml.safe_load
            
        Returns:
            Loaded YAML data as dictionary
        """
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading YAML {file_path}: {str(e)}")
            raise


def list_json_files(directory: str, pattern: Optional[str] = None) -> List[str]:
    """List all JSON files in a directory.
    
    Args:
        directory: Path to directory containing JSON files
        pattern: Optional regex pattern to filter filenames
        
    Returns:
        List of JSON filenames
    """
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if pattern:
        regex = re.compile(pattern)
        files = [f for f in files if regex.search(f)]
    
    return files


def load_json_file(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """Load data from a JSON file.
    
    Args:
        file_path: Path to JSON file
        encoding: File encoding
        
    Returns:
        Loaded JSON data as dictionary
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return {}


def load_json_files_parallel(file_paths: List[str], max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load multiple JSON files in parallel.
    
    Args:
        file_paths: List of paths to JSON files
        max_workers: Maximum number of worker threads (None for auto)
        
    Returns:
        List of loaded JSON data
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start loading tasks
        future_to_file = {executor.submit(load_json_file, file_path): file_path 
                         for file_path in file_paths}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                          total=len(file_paths),
                          desc="Loading files in parallel"):
            file_path = future_to_file[future]
            try:
                data = future.result()
                results.append(data)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    return results


@lru_cache(maxsize=128)
def categorize_pilot_id(pilot_id: str) -> str:
    """Categorize pilot ID based on its prefix.
    
    Args:
        pilot_id: Pilot identifier
        
    Returns:
        Category string ('minimal_exp', 'commercial', or 'air_force')
    """
    pilot_id_str = str(pilot_id)
    if pilot_id_str.startswith('8'):
        return 'minimal_exp'
    elif pilot_id_str.startswith('9'):
        return 'commercial'
    else:
        return 'air_force'


class DatasetVersion:
    """Class for tracking dataset versions."""
    
    def __init__(self, base_dir: str):
        """Initialize dataset version tracker.
        
        Args:
            base_dir: Base directory containing dataset files
        """
        self.base_dir = base_dir
        self.version_file = os.path.join(base_dir, 'version.json')
        self.version_info = self._load_version_info()
    
    def _load_version_info(self) -> Dict[str, Any]:
        """Load version information from version file.
        
        Returns:
            Version information dictionary
        """
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except:
                return {'version': 'unknown', 'timestamp': 0}
        return {'version': 'unknown', 'timestamp': 0}
    
    def get_version(self) -> str:
        """Get dataset version.
        
        Returns:
            Version string
        """
        return self.version_info.get('version', 'unknown')
    
    def get_timestamp(self) -> int:
        """Get dataset timestamp.
        
        Returns:
            Timestamp as integer
        """
        return self.version_info.get('timestamp', 0)
    
    def compute_hash(self) -> str:
        """Compute hash of dataset files.
        
        Returns:
            Hash string
        """
        hasher = hashlib.md5()
        
        # Get list of data files
        data_files = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.json') or file.endswith('.parquet'):
                    data_files.append(os.path.join(root, file))
        
        # Sort files for consistency
        data_files.sort()
        
        # Compute hash of file names and modification times
        for file_path in data_files:
            rel_path = os.path.relpath(file_path, self.base_dir)
            mtime = os.path.getmtime(file_path)
            hasher.update(f"{rel_path}:{mtime}".encode())
        
        return hasher.hexdigest()
    
    def update_version(self, version: str) -> None:
        """Update dataset version information.
        
        Args:
            version: New version string
        """
        self.version_info = {
            'version': version,
            'timestamp': int(time.time()),
            'hash': self.compute_hash()
        }
        
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.version_info, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update version file: {str(e)}")


class RemoteDataLoader:
    """Class for loading data from remote sources."""
    
    def __init__(self, cache_dir: str = '.remote_cache'):
        """Initialize remote data loader.
        
        Args:
            cache_dir: Directory for caching remote data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_from_url(self, url: str, force_reload: bool = False) -> str:
        """Load data from a URL.
        
        Args:
            url: URL to load data from
            force_reload: Whether to force reload even if cached
            
        Returns:
            Path to local file
        """
        import requests
        
        # Generate cache file path
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, url_hash)
        cache_meta = f"{cache_file}.meta"
        
        # Check if cached version exists and is not forced to reload
        if os.path.exists(cache_file) and os.path.exists(cache_meta) and not force_reload:
            # Load metadata
            with open(cache_meta, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache is still valid
            if 'url' in metadata and metadata['url'] == url:
                logger.debug(f"Using cached version of {url}")
                return cache_file
        
        # Download file
        logger.info(f"Downloading {url}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress
            with open(cache_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Save metadata
            metadata = {
                'url': url,
                'timestamp': int(time.time()),
                'content_type': response.headers.get('content-type', 'unknown')
            }
            with open(cache_meta, 'w') as f:
                json.dump(metadata, f)
            
            return cache_file
        
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            
            # If download fails but cache exists, use cache
            if os.path.exists(cache_file):
                logger.warning(f"Using cached version due to download error")
                return cache_file
            
            raise


class DatasetStatistics:
    """Class for computing statistics on datasets."""
    
    @staticmethod
    def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic statistics for a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isna().sum().sum(),
            'missing_percent': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'column_stats': {}
        }
        
        # Compute stats for each column
        for col in df.columns:
            col_stats = {}
            
            # Numerical columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'type': 'numeric',
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'missing': df[col].isna().sum(),
                    'missing_percent': (df[col].isna().sum() / len(df)) * 100
                })
            
            # Categorical columns
            elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                value_counts = df[col].value_counts()
                col_stats.update({
                    'type': 'categorical',
                    'unique_values': len(value_counts),
                    'most_common': value_counts.index[0] if not value_counts.empty else None,
                    'missing': df[col].isna().sum(),
                    'missing_percent': (df[col].isna().sum() / len(df)) * 100
                })
            
            # Datetime columns
            elif pd.api.types.is_datetime64_dtype(df[col]):
                col_stats.update({
                    'type': 'datetime',
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'missing': df[col].isna().sum(),
                    'missing_percent': (df[col].isna().sum() / len(df)) * 100
                })
            
            stats['column_stats'][col] = col_stats
        
        return stats
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, List[int]]:
        """Detect anomalies in a DataFrame using Z-score method.
        
        Args:
            df: Input DataFrame
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Dictionary mapping column names to lists of anomaly indices
        """
        anomalies = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            # Calculate Z-scores
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            
            # Find indices of anomalies
            anomaly_indices = np.where(z_scores > threshold)[0].tolist()
            
            if anomaly_indices:
                anomalies[col] = anomaly_indices
        
        return anomalies
    
    @staticmethod
    def compute_correlations(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """Compute correlation matrix for numerical columns.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            Correlation matrix
        """
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)
    
    @staticmethod
    def detect_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect duplicates in a DataFrame.
        
        Args:
            df: Input DataFrame
            subset: Optional list of columns to check for duplicates
            
        Returns:
            Dictionary with duplicate information
        """
        # Find duplicates
        duplicates = df.duplicated(subset=subset, keep=False)
        duplicate_df = df[duplicates]
        
        return {
            'total_duplicates': len(duplicate_df),
            'duplicate_percent': (len(duplicate_df) / len(df)) * 100,
            'duplicate_indices': duplicate_df.index.tolist(),
            'duplicate_groups': duplicate_df.groupby(duplicate_df.columns.tolist()).size().to_dict()
        }


# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
