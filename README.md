# SMU-TexCL: Cognitive Load Analysis

## Project Overview

This project implements a comprehensive analysis pipeline for the SMU-Textron Cognitive Load (SMU-TexCL) dataset, featuring both tabular and windowed data formats. The pipeline includes data preprocessing, feature engineering, model training, evaluation, and visualization for predicting cognitive load from physiological signals.

## Features

- **Dual Data Format Support**: Handles both tabular data and physiological signal windows
- **Advanced Machine Learning Models**:
  - Tabular models (Random Forest, Gradient Boosting, SVM, MLP, Elastic Net)
  - Sequence models (LSTM, GRU, Attention mechanisms, Transformer, TCN)
- **Comprehensive Visualization Tools**: Feature importance, learning curves, PCA/t-SNE, signal visualization
- **Robust Preprocessing Pipeline**: Scaling, imputation, outlier removal, feature transformation
- **Model Evaluation**: Cross-validation, performance metrics, model comparison
- **Signal Processing**: Filtering, feature extraction, quality assessment

## Project Structure

```
v2/
├── config.py                 # Configuration settings and parameter management
├── data/                     # Data loading and processing
│   ├── data_loader.py        # Base functionality for data loading
│   ├── table_dataset.py      # Tabular data handling
│   ├── windowed_dataset.py   # Windowed/time series data handling
├── main.py                   # Main entry point for analysis pipeline
├── models/                   # Machine learning models
│   ├── base_model.py         # Base model interface
│   ├── sequence_models.py    # Models for sequential/time series data
│   ├── tabular_models.py     # Models for tabular data
├── output/                   # Output files and visualizations
├── utils/                    # Utility functions
│   ├── evaluation.py         # Model evaluation functions
│   ├── preprocessing.py      # Data preprocessing functions
│   ├── visualization.py      # Visualization utilities
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main analysis pipeline:

```bash
python main.py
```

### Advanced Usage

Customize the analysis with command-line arguments:

```bash
python main.py --tabular_data_path path/to/table_data.parquet \
               --windowed_data_path path/to/windowed_data/ \
               --output_dir results \
               --analyze_tabular \
               --analyze_windowed \
               --tabular_model_type rf \
               --sequence_model_type lstm \
               --label_column avg_tlx_quantile
```

### Configuration

Configure the analysis through:
- Command-line arguments
- Environment variables (prefixed with `TEXCL_`)
- YAML or JSON configuration files
- Default settings in `config.py`

Example configuration file:
```yaml
data:
  tabular_data_path: data/tabular_data.parquet
  windowed_data_path: data/windowed_data/
  output_dir: results

analysis:
  do_windowed_analysis: true
  do_tabular_analysis: true
  max_pilots: 20

tabular_model:
  model_type: rf
  rf_n_estimators: 100
  optimize_hyperparams: true

sequence_model:
  batch_size: 16
  hidden_dim: 64
  num_layers: 2
  model_types: 
    - lstm
  learning_rate: 0.001
  epochs: 10
```

## Key Components

### Data Loading

The project supports two data formats:

1. **Tabular Data**: Each row represents a trial with various features and a cognitive load label
   ```python
   from data.table_dataset import TableDataset
   
   # Load tabular data
   table_data = TableDataset('data/tabular_data.parquet')
   
   # Explore data
   table_data.explore_data()
   ```

2. **Windowed Data**: Time series physiological signals (PPG, EDA, etc.) in fixed-length windows
   ```python
   from data.windowed_dataset import WindowedDataset
   
   # Load windowed data
   windowed_data = WindowedDataset('data/windowed_data/')
   windowed_data.load_all_pilots()
   
   # Visualize a window
   windowed_data.visualize_window_features(pilot_id='123', trial_idx=0)
   ```

### Preprocessing

```python
from utils.preprocessing import preprocess_tabular_data

# Preprocess tabular data
X_processed, pipeline = preprocess_tabular_data(
    df, 
    scaling='standard',
    impute_strategy='mean',
    feature_transform='boxcox',
    outlier_removal='zscore'
)
```

### Model Training

Tabular Models:
```python
from models.tabular_models import TabularModel

# Create and train model
model = TabularModel(model_type='rf')
model.train(X_train, y_train)

# Evaluate model
metrics = model.evaluate(X_test, y_test)
feature_importance = model.get_feature_importance(top_n=15)
```

Sequence Models:
```python
from models.sequence_models import SequenceModel

# Create and train model
model = SequenceModel(
    model_type='lstm',
    input_dim=22,
    hidden_dim=64,
    num_layers=2
)
model.train(train_loader, val_loader)

# Evaluate model
metrics = model.evaluate(test_loader)
```

### Visualization

```python
from utils.visualization import plot_feature_importance, plot_learning_curves

# Plot feature importance
plot_feature_importance(
    feature_names, 
    importances, 
    title="Feature Importances",
    save_path="output/feature_importance.png"
)

# Plot learning curves
plot_learning_curves(
    history,
    title="Training History",
    save_path="output/learning_curves.png"
)
```

## Example Workflow

Here's a complete example of a typical workflow:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load data
from data.table_dataset import TableDataset
table_data = TableDataset('data/table_data.parquet')

# 2. Preprocess data
from utils.preprocessing import preprocess_tabular_data
X = table_data.get_features()
y = table_data.get_labels('avg_tlx')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_processed, pipeline = preprocess_tabular_data(X_train)
X_test_processed = pipeline.transform(X_test)

# 3. Train model
from models.tabular_models import TabularModel
model = TabularModel(model_type='rf')
model.train(X_train_processed, y_train)

# 4. Evaluate model
from utils.evaluation import evaluate_regression_model, plot_regression_results
y_pred = model.predict(X_test_processed)
metrics = evaluate_regression_model(y_test, y_pred)
plot_regression_results(y_test, y_pred, save_path='output/regression_results.png')

# 5. Visualize feature importance
from utils.visualization import plot_feature_importance
feature_importance = model.get_feature_importance()
plot_feature_importance(
    feature_importance['feature'],
    feature_importance['importance'],
    save_path='output/feature_importance.png'
)
```

## Advanced Features

### Ensemble Models

```python
from models.tabular_models import TabularModel, EnsembleModel

# Create base models
rf_model = TabularModel(model_type='rf')
gb_model = TabularModel(model_type='gb')
svm_model = TabularModel(model_type='svm')

# Create ensemble
ensemble = EnsembleModel([rf_model, gb_model, svm_model])
ensemble.train(X_train, y_train)
```

### Signal Processing

```python
from utils.preprocessing import filter_signal, extract_wavelet_features

# Filter signal
filtered_signal = filter_signal(
    signal_data,
    fs=64.0,
    filter_type='bandpass',
    cutoff_freq=(0.5, 8.0)
)

# Extract wavelet features
wavelet_features = extract_wavelet_features(
    signal_data,
    wavelet='db4',
    level=3
)
```

### Advanced Visualization

```python
from utils.visualization import plot_physiological_signals, plot_attention_weights

# Plot physiological signals
plot_physiological_signals(
    data_dict={'PPG': ppg_data, 'EDA': eda_data},
    sampling_rates={'PPG': 64.0, 'EDA': 4.0},
    events=[{'time': 10.5, 'label': 'Event 1'}],
    save_path='output/physiological_signals.png'
)

# Plot attention weights
plot_attention_weights(
    time_points,
    attention_weights,
    save_path='output/attention_weights.png'
)
```

## Future Improvements

- [ ] Add support for real-time processing
- [ ] Implement online learning capabilities
- [ ] Add more sophisticated feature extraction from physiological signals
- [ ] Implement multi-modal fusion techniques
- [ ] Add explainable AI features for better interpretability

## License

[MIT License]

## Contact

For questions or collaboration, please contact [Project Maintainer].
