"""
Main script for running the SMU-Textron Cognitive Load dataset analysis.

This module serves as the entry point for the analysis pipeline, handling argument
parsing, data loading, model training, evaluation, and visualization.
"""

import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import threading
import multiprocessing
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Import local modules
import config
from data.table_dataset import TableDataset
from data.windowed_dataset import WindowedDataset
from models.tabular_models import TabularModel, TabularPipeline
from models.sequence_models import SequenceModel, PilotDataset, collate_fn
from utils.visualization import set_plot_style
from utils.preprocessing import preprocess_tabular_data
from utils.evaluation import evaluate_regression_model, plot_regression_results, compare_models


# Set up logger
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging initialized at {log_level} level")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='SMU-Textron Cognitive Load Analysis')
    
    # Data settings
    data_group = parser.add_argument_group('Data Settings')
    data_group.add_argument('--tabular_data_path', type=str, default=config.TABULAR_DATA_PATH,
                        help='Path to tabular data')
    data_group.add_argument('--windowed_data_path', type=str, default=config.WINDOWED_DATA_PATH,
                        help='Path to windowed data JSON files')
    data_group.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help='Directory for output files')
    
    # Analysis settings
    analysis_group = parser.add_argument_group('Analysis Settings')
    analysis_group.add_argument('--analyze_tabular', action='store_true',
                        help='Perform tabular data analysis')
    analysis_group.add_argument('--analyze_windowed', action='store_true',
                        help='Perform windowed data analysis')
    analysis_group.add_argument('--max_pilots', type=int, default=config.MAX_PILOTS,
                        help='Maximum number of pilots to load')
    analysis_group.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment (used for output subdirectory)')
    
    # Model settings
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument('--tabular_model_type', type=str, default=config.TABULAR_MODEL_TYPE,
                        choices=['rf', 'gb', 'svm', 'mlp', 'en'],
                        help='Type of tabular model')
    model_group.add_argument('--sequence_model_type', type=str, default=config.SEQUENCE_MODEL_TYPES[0],
                        choices=['lstm', 'gru', 'attention', 'transformer', 'tcn'],
                        help='Type of sequence model')
    model_group.add_argument('--label_column', type=str, default='avg_tlx_quantile',
                        help='Label column to predict')
    
    # Training settings
    training_group = parser.add_argument_group('Training Settings')
    training_group.add_argument('--test_size', type=float, default=config.TEST_SIZE,
                        help='Test set size')
    training_group.add_argument('--random_seed', type=int, default=config.RANDOM_SEED,
                        help='Random seed')
    training_group.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size')
    training_group.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Number of epochs')
    training_group.add_argument('--optimize_hyperparams', action='store_true',
                        help='Optimize hyperparameters using grid search')
    
    # System settings
    system_group = parser.add_argument_group('System Settings')
    system_group.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    system_group.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes/threads')
    system_group.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to use for model training')
    system_group.add_argument('--profile', action='store_true',
                        help='Profile code execution')
    
    return parser.parse_args()


def setup_output_dir(output_dir: str, experiment_name: Optional[str] = None) -> str:
    """Set up output directory.
    
    Args:
        output_dir: Base output directory
        experiment_name: Optional experiment name for subdirectory
        
    Returns:
        Path to the output directory
    """
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory path with experiment name if provided
    if experiment_name:
        dir_path = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    else:
        dir_path = os.path.join(output_dir, timestamp)
    
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Output directory set up at: {dir_path}")
    
    return dir_path


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Set CUDA deterministic behavior
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seeds set to {seed}")


def select_device(device_preference: Optional[str] = None) -> torch.device:
    """Select the appropriate device for model training.
    
    Args:
        device_preference: Preferred device ('cpu', 'cuda', or 'mps')
        
    Returns:
        PyTorch device object
    """
    # Default to CUDA if available
    if device_preference is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        # Use specified device if available
        if device_preference == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif device_preference == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            if device_preference != 'cpu':
                logger.warning(f"Requested device {device_preference} is not available. Using CPU instead.")
            device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    return device


class PerformanceProfiler:
    """Utility class for profiling execution time of code blocks."""
    
    def __init__(self):
        """Initialize the profiler."""
        self.times = {}
        self.start_times = {}
    
    def start(self, section_name: str) -> None:
        """Start timing a section.
        
        Args:
            section_name: Name of the section to time
        """
        self.start_times[section_name] = time.time()
        logger.debug(f"Started timing section: {section_name}")
    
    def end(self, section_name: str) -> float:
        """End timing a section and record the duration.
        
        Args:
            section_name: Name of the section to end timing
            
        Returns:
            Time elapsed in seconds
        """
        if section_name not in self.start_times:
            logger.warning(f"No start time found for section: {section_name}")
            return 0.0
        
        elapsed = time.time() - self.start_times[section_name]
        
        if section_name in self.times:
            self.times[section_name].append(elapsed)
        else:
            self.times[section_name] = [elapsed]
        
        logger.debug(f"Section {section_name} took {elapsed:.4f} seconds")
        return elapsed
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all timed sections.
        
        Returns:
            Dictionary with timing statistics
        """
        summary = {}
        
        for section, times in self.times.items():
            summary[section] = {
                'total': sum(times),
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'count': len(times)
            }
        
        return summary
    
    def print_summary(self) -> None:
        """Print a summary of all timed sections."""
        summary = self.get_summary()
        
        logger.info("Performance Profile Summary:")
        total_time = sum(stats['total'] for stats in summary.values())
        
        for section, stats in sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True):
            percentage = (stats['total'] / total_time) * 100
            logger.info(f"  {section}: {stats['total']:.2f}s total, {stats['mean']:.4f}s avg, {stats['count']} calls ({percentage:.1f}%)")
        
        logger.info(f"Total time: {total_time:.2f}s")


def analyze_tabular_data(args: argparse.Namespace, profiler: Optional[PerformanceProfiler] = None) -> Tuple[Any, Any, Any]:
    """Analyze tabular data.
    
    Args:
        args: Command-line arguments
        profiler: Optional performance profiler
        
    Returns:
        Tuple of (model, test features, test labels)
    """
    logger.info("\n" + "="*80)
    logger.info("Tabular Data Analysis")
    logger.info("="*80)
    
    # Load data
    if profiler: profiler.start("tabular_data_loading")
    logger.info("\n[1] Loading and exploring table data...")
    table_data = TableDataset(args.tabular_data_path)
    table_data.explore_data()
    if profiler: profiler.end("tabular_data_loading")
    
    # Visualize data
    if profiler: profiler.start("tabular_data_visualization")
    logger.info("\n[2] Visualizing relationships in the data...")
    table_data.visualize_turbulence_vs_load(
        save_path=os.path.join(args.output_dir, 'turbulence_vs_load.png')
    )
    if profiler: profiler.end("tabular_data_visualization")
    
    # Identify important features
    if profiler: profiler.start("feature_importance")
    logger.info("\n[3] Identifying important features...")
    table_data.visualize_feature_importance(
        label_column=args.label_column,
        save_path=os.path.join(args.output_dir, 'feature_importance.png')
    )
    if profiler: profiler.end("feature_importance")
    
    # Select important features
    if profiler: profiler.start("feature_selection")
    logger.info("\n[4] Selecting important features for modeling...")
    X_selected, selected_features = table_data.select_important_features(
        label_column=args.label_column,
        threshold=config.FEATURE_SELECTION_THRESHOLD
    )
    if profiler: profiler.end("feature_selection")
    
    # Split data for training
    if profiler: profiler.start("data_preparation")
    logger.info("\n[5] Preparing data for modeling...")
    X = table_data.get_features(selected_features)
    y = table_data.get_labels(args.label_column)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, shuffle=True
    )
    
    # Preprocess data
    logger.info("\n[6] Preprocessing data...")
    X_train_processed, preprocess_pipeline = preprocess_tabular_data(
        X_train, scaling='standard', impute_strategy='mean'
    )
    X_test_processed = preprocess_pipeline.transform(X_test)
    if profiler: profiler.end("data_preparation")
    
    # Train models
    if profiler: profiler.start("tabular_model_training")
    logger.info("\n[7] Training models...")
    
    # Standard model
    standard_model = TabularModel(model_type=args.tabular_model_type)
    standard_model.train(X_train_processed, y_train)
    
    # Pipeline model with preprocessing
    pipeline_model = TabularPipeline(model_type=args.tabular_model_type, scale=True)
    pipeline_model.train(X_train, y_train)
    if profiler: profiler.end("tabular_model_training")
    
    # Hyperparameter optimization if requested
    if args.optimize_hyperparams:
        if profiler: profiler.start("hyperparameter_optimization")
        logger.info("\n[7.1] Optimizing hyperparameters...")
        opt_model = TabularModel(model_type=args.tabular_model_type)
        
        # Define default parameter grid based on model type
        if args.tabular_model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif args.tabular_model_type == 'gb':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        # Add other model types as needed
        else:
            param_grid = {}

        opt_model.optimize_hyperparameters(X_train_processed, y_train, param_grid, cv=5)
        logger.info(f"Best hyperparameters: {opt_model.params}")
        standard_model = opt_model  # Use optimized model
        if profiler: profiler.end("hyperparameter_optimization")
    
    # Evaluate models
    if profiler: profiler.start("tabular_model_evaluation")
    logger.info("\n[8] Evaluating models...")
    
    # Compare models
    models = {
        'Standard': standard_model,
        'Pipeline': pipeline_model
    }
    
    # Use preprocessed data for standard model, original data for pipeline
    results_df = compare_models(
        {'Standard': standard_model},
        X_train_processed, y_train,
        X_test_processed, y_test,
        is_classification=False,
        save_path=os.path.join(args.output_dir, 'tabular_model_comparison.png')
    )
    
    logger.info("\nModel Comparison:")
    logger.info(results_df)
    if profiler: profiler.end("tabular_model_evaluation")
    
    # Save feature importance
    if profiler: profiler.start("save_results")
    logger.info("\n[9] Saving feature importance...")
    feature_importance = standard_model.get_feature_importance(top_n=15)
    feature_importance.to_csv(
        os.path.join(args.output_dir, 'feature_importance.csv'),
        index=False
    )
    if profiler: profiler.end("save_results")
    
    logger.info("\nTabular data analysis complete!")
    return standard_model, X_test_processed, y_test


def analyze_windowed_data(
    args: argparse.Namespace, 
    device: torch.device,
    profiler: Optional[PerformanceProfiler] = None
) -> Tuple[Any, Any, Any]:
    """Analyze windowed data.
    
    Args:
        args: Command-line arguments
        device: PyTorch device to use
        profiler: Optional performance profiler
        
    Returns:
        Tuple of (model, test data loader, metrics)
    """
    logger.info("\n" + "="*80)
    logger.info("Windowed Data Analysis")
    logger.info("="*80)
    
    # Load data
    if profiler: profiler.start("windowed_data_loading")
    logger.info("\n[1] Loading and exploring windowed data...")
    windowed_data = WindowedDataset(args.windowed_data_path)
    windowed_data.load_all_pilots(max_pilots=args.max_pilots)
    windowed_data.explore_data()
    if profiler: profiler.end("windowed_data_loading")
    
    # Visualize window features
    if profiler: profiler.start("window_visualization")
    logger.info("\n[2] Visualizing window features...")
    windowed_data.visualize_window_features(
        save_path=os.path.join(args.output_dir, 'window_features.png')
    )
    if profiler: profiler.end("window_visualization")
    
    # Prepare data for deep learning
    if profiler: profiler.start("sequence_data_preparation")
    logger.info("\n[3] Preparing data for sequence modeling...")
    train_trials, test_trials = train_test_split(
        windowed_data.all_trials, 
        test_size=args.test_size, 
        random_state=args.random_seed
    )
    
    train_dataset = PilotDataset(train_trials, feature_type='eng_features')
    test_dataset = PilotDataset(test_trials, feature_type='eng_features')
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        logger.warning("Not enough data in datasets. Skipping sequence modeling.")
        return None, None, None
    
    num_workers = args.num_workers if args.num_workers is not None else min(4, multiprocessing.cpu_count())
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    if profiler: profiler.end("sequence_data_preparation")
    
    # Train sequence model
    if profiler: profiler.start("sequence_model_training")
    logger.info(f"\n[4] Training {args.sequence_model_type.upper()} model...")
    sequence_model = SequenceModel(
        model_type=args.sequence_model_type,
        input_dim=22,  # Default for eng_features
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        batch_size=args.batch_size,
        learning_rate=config.LEARNING_RATE,
        epochs=args.epochs,
        device=device
    )
    
    # Train model
    sequence_model.train(train_loader, test_loader)
    
    # Plot training history
    sequence_model.plot_training_history(
        save_path=os.path.join(args.output_dir, f'{args.sequence_model_type}_training_history.png')
    )
    if profiler: profiler.end("sequence_model_training")
    
    # Evaluate model
    if profiler: profiler.start("sequence_model_evaluation")
    logger.info("\n[5] Evaluating sequence model...")
    metrics = sequence_model.evaluate(
        test_loader,
        save_path=os.path.join(args.output_dir, f'{args.sequence_model_type}_evaluation.png')
    )
    if profiler: profiler.end("sequence_model_evaluation")
    
    # Save model
    if profiler: profiler.start("save_sequence_model")
    logger.info("\n[6] Saving model...")
    model_path = os.path.join(args.output_dir, f'{args.sequence_model_type}_model.pt')
    sequence_model.save_model(model_path)
    if profiler: profiler.end("save_sequence_model")
    
    logger.info("\nWindowed data analysis complete!")
    return sequence_model, test_loader, metrics


def save_experiment_metadata(args: argparse.Namespace, output_dir: str, results: Dict[str, Any] = None) -> None:
    """Save experiment metadata for reproducibility.
    
    Args:
        args: Command-line arguments
        output_dir: Output directory
        results: Optional results dictionary
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'results': results,
        'system_info': {
            'python_version': __import__('sys').version,
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        }
    }
    
    # Save as JSON
    import json
    with open(os.path.join(output_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Experiment metadata saved to {os.path.join(output_dir, 'experiment_metadata.json')}")


def run_analysis_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the complete analysis pipeline.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary of results
    """
    # Set up logging
    setup_logging(args.log_level)
    
    # Initialize profiler if requested
    profiler = PerformanceProfiler() if args.profile else None
    
    # Start profiling overall execution time
    if profiler: profiler.start("total_execution")
    
    # Set up output directory
    args.output_dir = setup_output_dir(args.output_dir, args.experiment_name)
    
    # Set random seeds for reproducibility
    set_random_seeds(args.random_seed)
    
    # Set up device for PyTorch
    device = select_device(args.device)
    
    # Set plotting style
    set_plot_style()
    
    logger.info("\n" + "="*80)
    logger.info("SMU-Textron Cognitive Load Dataset Analysis")
    logger.info("="*80)
    
    # Run analyses
    results = {}
    tabular_model = None
    sequence_model = None
    
    if args.analyze_tabular or not args.analyze_windowed:  # Default to tabular if none specified
        if profiler: profiler.start("tabular_analysis")
        tabular_model, X_test, y_test = analyze_tabular_data(args, profiler)
        results['tabular'] = {
            'model': tabular_model,
            'test_data': (X_test, y_test)
        }
        if profiler: profiler.end("tabular_analysis")
    
    if args.analyze_windowed:
        if profiler: profiler.start("windowed_analysis")
        sequence_model, test_loader, sequence_metrics = analyze_windowed_data(args, device, profiler)
        results['windowed'] = {
            'model': sequence_model,
            'test_loader': test_loader,
            'metrics': sequence_metrics
        }
        if profiler: profiler.end("windowed_analysis")
    
    # Save experiment metadata
    save_experiment_metadata(args, args.output_dir, results)
    
    # End overall profiling
    if profiler:
        profiler.end("total_execution")
        profiler.print_summary()
        
        # Save profiling results
        import json
        with open(os.path.join(args.output_dir, 'performance_profile.json'), 'w') as f:
            json.dump(profiler.get_summary(), f, indent=2, default=str)
    
    logger.info("\nAnalysis Complete!")
    
    return {
        'tabular_model': tabular_model,
        'sequence_model': sequence_model,
        'output_dir': args.output_dir
    }


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Run analysis pipeline
        results = run_analysis_pipeline(args)
        
        # Return results for potential further processing
        return results
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return None
    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        return None


if __name__ == "__main__":
    main()
