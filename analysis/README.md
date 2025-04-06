# SMU-TexCL Model Diagnostics

## Overview
This script performs comprehensive diagnostic analysis on the SMU-TexCL (Situation Monitoring Under Turbulence - Cognitive Load) dataset. It aims to investigate model performance, feature importance, signal quality, and various modeling approaches.

## Prerequisites
- Python 3.7+
- Required libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - torch
  - tqdm

## Usage
```bash
python model_diagnostics.py --tabular_data_path /path/to/pre_proccessed_table_data.parquet --windowed_data_path /path/to/windowed/data/directory --output ./analysis_output
```

### Arguments
- `--tabular_data_path`: Path to the preprocessed tabular data (Parquet file)
- `--windowed_data_path`: Directory containing windowed data JSON files
- `--output_dir`: Directory to save analysis outputs (default: `analysis_output`)
- `--random_seed`: Random seed for reproducibility (default: 42)

## Analyses Performed
1. **Target Distribution Analysis**
   - Examine distribution of target variables
   - Calculate correlations between different metrics

2. **Feature Importance Analysis**
   - Identify top influential features
   - Visualize feature importances and correlations

3. **Prediction Error Analysis**
   - Investigate error patterns
   - Analyze errors by turbulence level and pilot category

4. **Signal Quality Analysis**
   - Assess quality of physiological signals
   - Analyze signal quality across different conditions

5. **Classification Approach**
   - Compare ordinal and standard classification approaches
   - Evaluate performance across different turbulence levels

6. **Pilot-Normalized Features**
   - Create features normalized by pilot
   - Assess impact on model performance

7. **Sequence Model Baselines**
   - Test simple sequence models (LSTM)
   - Compare performance with different feature types

## Output
The script generates several visualization files in the specified output directory:
- `target_distributions.png`
- `top_features.png`
- `feature_correlations.png`
- `error_by_turbulence.png`
- `error_by_category.png`
- `error_vs_tlx.png`
- `signal_quality.png`
- `signal_quality_by_turbulence.png`
- `signal_quality_by_category.png`
- `ordinal_confusion_matrix.png`
- `confusion_matrix.png`
- `class_acc_by_turbulence.png`
- `pilot_normalized_features.png`
- `pilot_normalized_predictions.png`

## Troubleshooting
- Ensure all required libraries are installed
- Check file paths are correct
- Verify data file formats
- Ensure sufficient permissions for writing output files

## Notes
- The analysis is computationally intensive
- Results may vary depending on dataset and random seed
