### Basic Usage Commands

1. **Run the full pipeline with default settings:**
   ```bash
   python main.py
   ```

2. **Run only tabular data analysis:**
   ```bash
   python main.py --analyze_tabular --tabular_model_type rf
   ```

3. **Run only windowed data analysis:**
   ```bash
   python main.py --analyze_windowed --sequence_model_type lstm
   ```

### Advanced Usage Commands

4. **Try different tabular models with custom parameters:**
   ```bash
   python main.py --tabular_model_type gb --optimize_hyperparams
   ```

5. **Compare multiple sequence models:**
   ```bash
   python main.py --sequence_model_type transformer --hidden_dim 128 --num_layers 3
   ```

6. **Limit the number of pilots for faster testing:**
   ```bash
   python main.py --max_pilots 5
   ```

7. **Use a custom configuration file:**
   ```bash
   python main.py --config my_custom_config.yaml
   ```

8. **Profile the code execution:**
   ```bash
   python main.py --profile
   ```

9. **Use a specific output directory with an experiment name:**
   ```bash
   python main.py --output_dir results --experiment_name cognitive_load_exp1
   ```

10. **Run with specific preprocessing parameters:**
    ```bash
    export TEXCL_PREPROCESSING__SCALING=robust
    export TEXCL_PREPROCESSING__IMPUTATION=knn
    python main.py
    ```

### Feature-Focused Commands

11. **Focus on feature importance analysis:**
    ```bash
    python main.py --tabular_model_type rf --feature_selection_method permutation --top_n_features 20
    ```

12. **Analyze a specific label column:**
    ```bash
    python main.py --label_column mental_effort
    ```

13. **Run the full pipeline with all logs and save everything:**
    ```bash
    python main.py --log_level DEBUG --analyze_tabular --analyze_windowed --optimize_hyperparams
    ```

