"""
Configuration settings for the SMU-Textron Cognitive Load Dataset Analysis.

This module provides a centralized configuration system with support for
command-line arguments, environment variables, and YAML/JSON configuration files.
"""

import os
import argparse
import yaml
import json
import logging
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    # Data paths
    "data": {
        "tabular_data_path": '../pre_proccessed_table_data.parquet',
        "windowed_data_path": '../',  # Directory containing JSON files
        "output_dir": 'output'
    },
    
    # Analysis settings
    "analysis": {
        "do_windowed_analysis": True,
        "do_tabular_analysis": True,
        "max_pilots": None  # Set to a number to limit the number of pilots to analyze
    },
    
    # Feature selection settings
    "feature_selection": {
        "threshold": 'mean',
        "top_n_features": 15,  # Number of top features to visualize
        "feature_importance_method": "permutation"  # 'permutation', 'mdi', or 'shap'
    },
    
    # Model training settings
    "training": {
        "test_size": 0.2,
        "random_seed": 42,
        "cv_folds": 5,
        "stratify": True,
        "early_stopping": True,
        "patience": 5
    },
    
    # Tabular model settings
    "tabular_model": {
        "model_type": 'rf',  # 'rf' for Random Forest, 'gb' for Gradient Boosting, 'svm' for Support Vector Machine
        "rf_n_estimators": 100,
        "rf_max_depth": None,
        "rf_min_samples_split": 2,
        "gb_n_estimators": 100,
        "gb_learning_rate": 0.1,
        "svm_kernel": 'rbf',
        "optimize_hyperparams": False
    },
    
    # Sequence model settings
    "sequence_model": {
        "batch_size": 16,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 1e-3,
        "epochs": 10,
        "model_types": ['lstm'],  # Models to train: 'lstm', 'attention', 'transformer', 'tcn', 'gru'
        "bidirectional": False,
        "gradient_clip": 1.0
    },
    
    # Visualization settings
    "visualization": {
        "figsize_large": (14, 12),
        "figsize_medium": (12, 8),
        "figsize_small": (10, 6),
        "style": "whitegrid",  # 'whitegrid', 'darkgrid', 'white', 'dark'
        "color_palette": "viridis",  # 'viridis', 'magma', 'plasma', 'inferno', 'cividis'
        "dpi": 300
    },
    
    # Preprocessing settings
    "preprocessing": {
        "scaling": "standard",  # 'standard', 'minmax', 'robust', 'none'
        "imputation": "mean",  # 'mean', 'median', 'knn', 'none'
        "outlier_removal": None,  # 'zscore', 'iqr', 'isolation_forest', None
        "outlier_threshold": 3.0,
        "feature_transform": None  # 'log', 'sqrt', 'boxcox', 'yeo-johnson', None
    },
    
    # Logging settings
    "logging": {
        "level": "INFO",  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        "log_file": "analysis.log",
        "console_output": True,
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    
    # Experiment tracking settings
    "experiment_tracking": {
        "enabled": False,
        "tracking_uri": None,
        "experiment_name": "cognitive-load-analysis",
        "tags": {}
    }
}

# Global configuration
_CONFIG = {}


def load_config_from_yaml(file_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        file_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading YAML config: {str(e)}")
        return {}


def load_config_from_json(file_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        file_path: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config
    except Exception as e:
        print(f"Error loading JSON config: {str(e)}")
        return {}


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables.
    
    Environment variables should be prefixed with 'TEXCL_'.
    Nested configuration is supported with double underscore notation, e.g.:
    TEXCL_DATA__TABULAR_DATA_PATH
    
    Returns:
        Configuration dictionary
    """
    config = {}
    
    # Process environment variables
    for key, value in os.environ.items():
        if key.startswith('TEXCL_'):
            # Remove prefix and split by double underscore
            key_parts = key[6:].lower().split('__')
            
            # Convert string to appropriate type
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none':
                value = None
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
                value = float(value)
            
            # Build nested dictionary
            current = config
            for i, part in enumerate(key_parts):
                if i == len(key_parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
    
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='SMU-Textron Cognitive Load Analysis')
    
    # Configuration files
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML or JSON)')
    
    # Data settings
    parser.add_argument('--tabular_data_path', type=str, help='Path to tabular data')
    parser.add_argument('--windowed_data_path', type=str, help='Path to windowed data JSON files')
    parser.add_argument('--output_dir', type=str, help='Directory for output files')
    
    # Analysis settings
    parser.add_argument('--analyze_tabular', action='store_true', help='Perform tabular data analysis')
    parser.add_argument('--analyze_windowed', action='store_true', help='Perform windowed data analysis')
    parser.add_argument('--max_pilots', type=int, help='Maximum number of pilots to load')
    
    # Model settings
    parser.add_argument('--tabular_model_type', type=str, choices=['rf', 'gb', 'svm'], help='Type of tabular model')
    parser.add_argument('--sequence_model_type', type=str, choices=['lstm', 'gru', 'attention', 'transformer', 'tcn'], help='Type of sequence model')
    parser.add_argument('--label_column', type=str, help='Label column to predict')
    
    # Training settings
    parser.add_argument('--test_size', type=float, help='Test set size')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--optimize_hyperparams', action='store_true', help='Optimize hyperparameters')
    
    # Logging settings
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level')
    parser.add_argument('--log_file', type=str, help='Log file path')
    
    # Experiment tracking
    parser.add_argument('--tracking_uri', type=str, help='MLflow tracking URI')
    parser.add_argument('--experiment_name', type=str, help='MLflow experiment name')
    
    return parser.parse_args()


def _update_nested_dict(base_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Update a nested dictionary with another nested dictionary.
    
    Args:
        base_dict: Base dictionary to update
        new_dict: Dictionary with new values
        
    Returns:
        Updated dictionary
    """
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _update_nested_dict(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to nested dictionary.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Nested dictionary
    """
    args_dict = vars(args)
    config_dict = {}
    
    # Process special cases first
    if 'config' in args_dict and args_dict['config'] is not None:
        # Skip, handled separately
        del args_dict['config']
    
    if 'analyze_tabular' in args_dict and args_dict['analyze_tabular'] is not None:
        if 'analysis' not in config_dict:
            config_dict['analysis'] = {}
        config_dict['analysis']['do_tabular_analysis'] = args_dict['analyze_tabular']
        del args_dict['analyze_tabular']
    
    if 'analyze_windowed' in args_dict and args_dict['analyze_windowed'] is not None:
        if 'analysis' not in config_dict:
            config_dict['analysis'] = {}
        config_dict['analysis']['do_windowed_analysis'] = args_dict['analyze_windowed']
        del args_dict['analyze_windowed']
    
    # Map remaining arguments to configuration structure
    mapping = {
        'tabular_data_path': ('data', 'tabular_data_path'),
        'windowed_data_path': ('data', 'windowed_data_path'),
        'output_dir': ('data', 'output_dir'),
        'max_pilots': ('analysis', 'max_pilots'),
        'tabular_model_type': ('tabular_model', 'model_type'),
        'sequence_model_type': ('sequence_model', 'model_types'),  # Note: converts to list
        'label_column': ('training', 'label_column'),
        'test_size': ('training', 'test_size'),
        'random_seed': ('training', 'random_seed'),
        'batch_size': ('sequence_model', 'batch_size'),
        'epochs': ('sequence_model', 'epochs'),
        'optimize_hyperparams': ('tabular_model', 'optimize_hyperparams'),
        'log_level': ('logging', 'level'),
        'log_file': ('logging', 'log_file'),
        'tracking_uri': ('experiment_tracking', 'tracking_uri'),
        'experiment_name': ('experiment_tracking', 'experiment_name')
    }
    
    for arg_name, value in args_dict.items():
        if value is not None and arg_name in mapping:
            section, key = mapping[arg_name]
            if section not in config_dict:
                config_dict[section] = {}
            
            # Special case for sequence_model_type
            if arg_name == 'sequence_model_type':
                config_dict[section][key] = [value]
            else:
                config_dict[section][key] = value
    
    return config_dict


def initialize_config() -> Dict[str, Any]:
    """Initialize configuration from various sources.
    
    Priority order (highest to lowest):
    1. Command-line arguments
    2. Environment variables
    3. Configuration file
    4. Default configuration
    
    Returns:
        Initialized configuration dictionary
    """
    global _CONFIG
    config = DEFAULT_CONFIG.copy()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration from file if specified
    if args.config:
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            file_config = load_config_from_yaml(args.config)
        elif args.config.endswith('.json'):
            file_config = load_config_from_json(args.config)
        else:
            print(f"Unsupported configuration file format: {args.config}")
            file_config = {}
        
        config = _update_nested_dict(config, file_config)
    
    # Load configuration from environment variables
    env_config = load_config_from_env()
    config = _update_nested_dict(config, env_config)
    
    # Update with command-line arguments
    args_config = args_to_dict(args)
    config = _update_nested_dict(config, args_config)
    
    # Store configuration
    _CONFIG = config
    
    return config


def get_config() -> Dict[str, Any]:
    """Get current configuration.
    
    Returns:
        Current configuration dictionary
    """
    global _CONFIG
    if not _CONFIG:
        return initialize_config()
    return _CONFIG


def save_config(file_path: str, format_type: str = 'yaml') -> None:
    """Save current configuration to a file.
    
    Args:
        file_path: Path to save configuration
        format_type: File format ('yaml' or 'json')
    """
    config = get_config()
    
    try:
        if format_type.lower() == 'yaml':
            with open(file_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
        elif format_type.lower() == 'json':
            with open(file_path, 'w') as file:
                json.dump(config, file, indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        print(f"Configuration saved to {file_path}")
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")


def setup_logging() -> logging.Logger:
    """Set up logging based on configuration.
    
    Returns:
        Configured logger
    """
    config = get_config()
    log_config = config['logging']
    
    # Create logger
    logger = logging.getLogger('texcl')
    logger.setLevel(getattr(logging, log_config['level']))
    
    # Create formatter
    formatter = logging.Formatter(log_config['log_format'])
    
    # Create file handler if log file specified
    if log_config.get('log_file'):
        file_handler = logging.FileHandler(log_config['log_file'])
        file_handler.setLevel(getattr(logging, log_config['level']))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Create console handler if console output enabled
    if log_config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_config['level']))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def setup_experiment_tracking() -> Optional[Any]:
    """Set up experiment tracking with MLflow.
    
    Returns:
        MLflow experiment object or None if tracking is disabled
    """
    config = get_config()
    tracking_config = config['experiment_tracking']
    
    if not tracking_config.get('enabled', False):
        return None
    
    try:
        import mlflow
        
        # Set tracking URI if specified
        if tracking_config.get('tracking_uri'):
            mlflow.set_tracking_uri(tracking_config['tracking_uri'])
        
        # Set experiment
        experiment_name = tracking_config.get('experiment_name', 'cognitive-load-analysis')
        mlflow.set_experiment(experiment_name)
        
        # Start a new run
        run = mlflow.start_run()
        
        # Log configuration
        mlflow.log_params({
            "tabular_model_type": config['tabular_model']['model_type'],
            "sequence_model_types": config['sequence_model']['model_types'],
            "test_size": config['training']['test_size'],
            "random_seed": config['training']['random_seed'],
            "batch_size": config['sequence_model']['batch_size'],
            "epochs": config['sequence_model']['epochs']
        })
        
        # Set tags
        for key, value in tracking_config.get('tags', {}).items():
            mlflow.set_tag(key, value)
        
        return run
    
    except ImportError:
        print("MLflow not installed. Experiment tracking disabled.")
        return None
    except Exception as e:
        print(f"Error setting up experiment tracking: {str(e)}")
        return None


# Convenience accessors for configuration sections
def get_data_config() -> Dict[str, Any]:
    """Get data configuration section.
    
    Returns:
        Data configuration dictionary
    """
    return get_config()['data']


def get_analysis_config() -> Dict[str, Any]:
    """Get analysis configuration section.
    
    Returns:
        Analysis configuration dictionary
    """
    return get_config()['analysis']


def get_tabular_model_config() -> Dict[str, Any]:
    """Get tabular model configuration section.
    
    Returns:
        Tabular model configuration dictionary
    """
    return get_config()['tabular_model']


def get_sequence_model_config() -> Dict[str, Any]:
    """Get sequence model configuration section.
    
    Returns:
        Sequence model configuration dictionary
    """
    return get_config()['sequence_model']


def get_training_config() -> Dict[str, Any]:
    """Get training configuration section.
    
    Returns:
        Training configuration dictionary
    """
    return get_config()['training']


def get_visualization_config() -> Dict[str, Any]:
    """Get visualization configuration section.
    
    Returns:
        Visualization configuration dictionary
    """
    return get_config()['visualization']


def get_preprocessing_config() -> Dict[str, Any]:
    """Get preprocessing configuration section.
    
    Returns:
        Preprocessing configuration dictionary
    """
    return get_config()['preprocessing']


def get_validation_config() -> Dict[str, Any]:
    """Validate configuration and return valid configuration.
    
    This function checks configuration values and ensures they meet the required criteria.
    
    Returns:
        Validated configuration dictionary
    """
    config = get_config()
    
    # Validate tabular model type
    valid_tabular_models = ['rf', 'gb', 'svm', 'mlp', 'en']
    if config['tabular_model']['model_type'] not in valid_tabular_models:
        print(f"Warning: Invalid tabular model type '{config['tabular_model']['model_type']}'. Using 'rf'.")
        config['tabular_model']['model_type'] = 'rf'
    
    # Validate sequence model types
    valid_sequence_models = ['lstm', 'gru', 'attention', 'transformer', 'tcn']
    for model_type in config['sequence_model']['model_types']:
        if model_type not in valid_sequence_models:
            print(f"Warning: Invalid sequence model type '{model_type}'. Removing from list.")
            config['sequence_model']['model_types'].remove(model_type)
    
    if not config['sequence_model']['model_types']:
        print("Warning: No valid sequence model types specified. Using 'lstm'.")
        config['sequence_model']['model_types'] = ['lstm']
    
    # Validate test size
    if not (0 < config['training']['test_size'] < 1):
        print(f"Warning: Invalid test size {config['training']['test_size']}. Using 0.2.")
        config['training']['test_size'] = 0.2
    
    # Validate batch size
    if config['sequence_model']['batch_size'] <= 0:
        print(f"Warning: Invalid batch size {config['sequence_model']['batch_size']}. Using 16.")
        config['sequence_model']['batch_size'] = 16
    
    # Validate epochs
    if config['sequence_model']['epochs'] <= 0:
        print(f"Warning: Invalid epochs {config['sequence_model']['epochs']}. Using 10.")
        config['sequence_model']['epochs'] = 10
    
    return config


# Initialize configuration on module import
config = initialize_config()

# Export variables for backward compatibility
TABULAR_DATA_PATH = config['data']['tabular_data_path']
WINDOWED_DATA_PATH = config['data']['windowed_data_path']
OUTPUT_DIR = config['data']['output_dir']

DO_WINDOWED_ANALYSIS = config['analysis']['do_windowed_analysis']
MAX_PILOTS = config['analysis']['max_pilots']

FEATURE_SELECTION_THRESHOLD = config['feature_selection']['threshold']
TOP_N_FEATURES = config['feature_selection']['top_n_features']

TEST_SIZE = config['training']['test_size']
RANDOM_SEED = config['training']['random_seed']

TABULAR_MODEL_TYPE = config['tabular_model']['model_type']
RF_N_ESTIMATORS = config['tabular_model']['rf_n_estimators']

BATCH_SIZE = config['sequence_model']['batch_size']
HIDDEN_DIM = config['sequence_model']['hidden_dim']
NUM_LAYERS = config['sequence_model']['num_layers']
DROPOUT = config['sequence_model']['dropout']
LEARNING_RATE = config['sequence_model']['learning_rate']
EPOCHS = config['sequence_model']['epochs']
SEQUENCE_MODEL_TYPES = config['sequence_model']['model_types']

FIGSIZE_LARGE = config['visualization']['figsize_large']
FIGSIZE_MEDIUM = config['visualization']['figsize_medium']
FIGSIZE_SMALL = config['visualization']['figsize_small']
