# Results Directory

This directory contains all visualization outputs, data tables, and plotting scripts for the OptScale research project. The results showcase performance comparisons across multiple inference-time scaling methods on five mathematical reasoning datasets.

## 📊 Datasets

All results are evaluated on the following mathematical reasoning datasets:
- **MATH500**: 500 samples from the MATH dataset
- **GSM8K**: Grade School Math 8K dataset
- **AIME24**: 2024 American Invitational Mathematics Examination
- **AIME25**: 2025 American Invitational Mathematics Examination  
- **AMC23**: 2023 American Mathematics Competitions

## 📈 Visualization Files

### Comparison Plots (PNG Files)

#### Individual Dataset Comparisons
For each dataset, three comparison plots are generated showing different performance metrics:

| File | Description |
|------|-------------|
| `{DATASET}_accuracy_comparison.png` | Accuracy (%) comparison across all methods for given dataset |
| `{DATASET}_tokens_comparison.png` | Token consumption comparison for achieving different accuracy levels |
| `{DATASET}_efficiency_comparison.png` | Efficiency ratio (accuracy per token) across different N values |

**Datasets**: MATH500, GSM8K, AIME24, AIME25, AMC23

#### Combined Multi-Dataset Plots
| File | Description |
|------|-------------|
| `all_datasets_accuracy_comparison.png` | Aggregate accuracy comparison across all 5 datasets |
| `all_datasets_tokens_comparison.png` | Aggregate token consumption across all 5 datasets |
| `all_datasets_efficiency_comparison.png` | Aggregate efficiency comparison across all 5 datasets |

#### Special Plots
| File | Description |
|------|-------------|
| `accuracy_token_plots.png` | Multi-panel plot showing accuracy vs token consumption curves for all 5 datasets |

### Methods Compared

The visualizations compare the following inference-time scaling approaches:

1. **Best-of-N (BoN)**: Baseline that selects the best solution from N samples using an oracle verifier
2. **Self-Consistency (SC)**: Majority voting across N solutions  
3. **MR-Thinking**: Sequential reasoning approach with N-step generation
4. **Early-stopping SC (ESC)**: Self-consistency with early termination based on confidence
5. **Adaptive SC (ASC)**: Self-consistency with adaptive sampling based on solution agreement
6. **Difficulty-Adaptive SC (DSC)**: Self-consistency adapted to problem difficulty
7. **OptScale⁰** (OptScale by MLE): Our method using Maximum Likelihood Estimation for scaling optimization
8. **OptScale^t** (OptScale w/ Predictor): Our method using a trained predictor for accuracy prediction

## 📄 Data Tables

### LaTeX Tables (TXT Files)

| File | Description |
|------|-------------|
| `latex_table_simple_output.txt` | LaTeX formatted table with core baseline methods (BoN, SC, MR-Thinking, OptScale⁰, OptScale^t) |
| `latex_table_final_output.txt` | Comprehensive LaTeX table including all baseline methods and advanced variants (ESC, ASC, DSC) |
| `deepseek_qwen7b_latex_table_output.txt` | Alternative table output for reference |

These tables present accuracy and token consumption metrics across all datasets at N values of 8, 16, 32, and 64.

### CSV Data

| File | Description |
|------|-------------|
| `results_table.csv` | Raw CSV format of all results used as input for LaTeX generation and plotting scripts |
| `comparison_data_summary.json` | JSON summary of all comparison data across datasets and methods |

## 🔧 Python Scripts

### Main Plotting and Data Generation Scripts

#### 1. **`comparison_plot.py`**
   - **Purpose**: Generates accuracy, token consumption, and efficiency comparison plots for individual datasets and combined views
   - **Outputs**:
     - `{DATASET}_accuracy_comparison.png` (5 files)
     - `{DATASET}_tokens_comparison.png` (5 files)
     - `{DATASET}_efficiency_comparison.png` (5 files)
     - `all_datasets_accuracy_comparison.png`
     - `all_datasets_tokens_comparison.png`
     - `all_datasets_efficiency_comparison.png`
     - `comparison_data_summary.json`
   - **Input**: Result JSON files from `../BoN_OptScale/`, `../SC/`, and `../Seq/` directories

#### 2. **`gen_csv_table.py`**
   - **Purpose**: Aggregates results from multiple baseline methods and datasets into a single CSV table
   - **Outputs**: `results_table.csv`
   - **Input**: Result JSON files from `../BoN_OptScale/`, `../SC/`, and `../Seq/` directories
   - **Usage**: 
     ```bash
     python gen_csv_table.py
     ```

#### 3. **`gen_graph.py`**
   - **Purpose**: Creates multi-panel accuracy vs token consumption plots
   - **Outputs**: `accuracy_token_plots.png`
   - **Input**: `results_table.csv`
   - **Usage**:
     ```bash
     python gen_graph.py
     ```

#### 4. **`gen_latex_simple.py`**
   - **Purpose**: Generates LaTeX table with core methods (excludes ESC, ASC, DSC variants)
   - **Outputs**: `latex_table_simple_output.txt`
   - **Input**: `results_table.csv`
   - **Usage**:
     ```bash
     python gen_latex_simple.py
     ```

#### 5. **`gen_latex_final.py`**
   - **Purpose**: Generates comprehensive LaTeX table with all baseline methods
   - **Outputs**: `latex_table_final_output.txt`
   - **Input**: `results_table.csv`
   - **Usage**:
     ```bash
     python gen_latex_final.py
     ```

## 📊 Typical Workflow

To regenerate all results:

1. **Generate CSV table** from baseline result files:
   ```bash
   python gen_csv_table.py
   ```

2. **Create LaTeX tables** for publication:
   ```bash
   python gen_latex_simple.py
   python gen_latex_final.py
   ```

3. **Generate visualization plots**:
   ```bash
   python comparison_plot.py
   python gen_graph.py
   ```

## 📁 Input Data Dependencies

The scripts in this directory depend on result files generated by other modules:

- **BoN_OptScale results** (`../BoN_OptScale/*.json`): Best-of-N and OptScale results
  - `*_BoN_results.json`: Best-of-N baseline results
  - `*_OptScale_MLE_results_*.json`: OptScale⁰ results
  - `*_OptScale_MAP_results_*.json`: OptScale^t predictor results

- **Self-Consistency results** (`../SC/*.json`): SC and variant results
  - `*_maj_vote_results.json`: Majority voting results
  - `*_esc_asc_results.json`: ESC and ASC results  
  - `*_dsc_results.json`: DSC results

- **Sequential results** (`../Seq/*.json`): MR-Thinking baseline
  - `*_MR-Thinking.json`: MR-Thinking results for each dataset (MATH500, GSM8K, AIME24, AIME25, AMC23)

## 📈 Performance Summary

OptScale (both variants) significantly outperforms baseline methods in token efficiency:

- **OptScale⁰** (MLE-based): 40-60% token reduction compared to BoN while maintaining comparable accuracy
- **OptScale^t** (Predictor-based): Flexible accuracy-efficiency trade-offs with adaptive token budgets

Key improvements visible across all datasets at various N values (8, 16, 32, 64).

## 📝 Notes

- All plots are saved as high-resolution PNG files (suitable for publication)
- LaTeX tables use `booktabs` package styling for professional appearance
- Token counts represent average tokens generated per problem
- Accuracy values are in percentage (0-100 scale)
- Results use Qwen 7B as the base language model


