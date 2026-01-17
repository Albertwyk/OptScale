# OptScale: Probabilistic Optimality for Inference-time Scaling

<div align="center">

</div>


[![Paper](https://img.shields.io/badge/Paper-AAAI%202026-blue)](https://arxiv.org/abs/2506.22376)
[![Data](https://img.shields.io/badge/Data-Google%20Drive-green)](https://drive.google.com/drive/folders/1clbEyLvQWc3DPtBh7Jb-BTb4z7jYWWkr?usp=sharing)

Official implementation of **OptScale** (OptScale<sup>0</sup> and OptScale<sup>t</sup>), a probabilistic framework for optimal inference-time scaling, accepted at **AAAI 2026**.

> ⭐ **NEW**: We release **free pre-generated completion pools** to enable researchers to reproduce results and develop new algorithms without GPU costs! [Jump to Data Resources →](#-reproducibility--data-resources)

---

<div align="center">

## 🔄 **Reproducibility & Data Resources** ⭐

**Free Compute Resources for Inference-time Scaling Research**

</div>

### 📦 **What We Provide**

We release **pre-generated completion pools** to enable researchers with limited compute resources to:
- ✅ **Reproduce our results** without running expensive model inference
- ✅ **Develop new algorithms** by sampling from our completion pools
- ✅ **Conduct experiments** on inference-time scaling without GPU costs

> 💡 **Why This Matters**: Inference-time scaling is compute-intensive. Our data pools democratize access to this research area, making it accessible to all researchers regardless of compute resources.

### 🔗 **Download Links**

<div align="center">

**[📥 Download Completion Pools & Training Data](https://drive.google.com/drive/folders/1clbEyLvQWc3DPtBh7Jb-BTb4z7jYWWkr?usp=sharing)**

</div>

### 📊 **What's Included**

The Google Drive folder contains:

1. **`data/`** folder:
   - **Parallel scaling completions**: Pre-generated completions for Best-of-N (BoN) style experiments
   - **Sequential scaling completions**: Pre-generated completions for MR-Thinking style experiments
   - Contains all completions up to maximum budget (N=64) for all 5 datasets

2. **`train_predictor/`** folder:
   - **`train_completion.json`**: Training completions for OptScale<sup>t</sup> predictor
   - **`val_completion.json`**: Validation completions for OptScale<sup>t</sup> predictor

### 🚀 **Quick Setup for Reproducibility**

**Step 1:** Clone this repository
```bash
git clone https://github.com/Albertwyk/OptScale.git
cd OptScale
```

**Step 2:** Download and place the data files
```bash
# Download from Google Drive and extract to repository root
# Place the following folders/files:
data/completions/r1_distill_qwen7b/parallel/
data/completions/r1_distill_qwen7b/sequential/
train_predictor/train_completion.json
train_predictor/val_completion.json
```

**Step 3:** Run experiments
```bash
# Now you can run all experiments without GPU inference!
cd BoN_OptScale
python bon.py --dataset MATH500 --max_N 64
```

### 🧪 **Using the Data for New Research**

You can use our completion pools to:
- **Sample completions** randomly to simulate scaling processes
- **Test new algorithms** without generating new completions
- **Compare methods** fairly using the same completion pool
- **Develop stopping criteria** and adaptive sampling strategies

The pools are structured to allow equivalent scaling processes while eliminating compute costs.

### 📝 **Citation**

If you use our data pools in your research, please cite:

```bibtex
@article{wang2026optscale,
  title={OptScale: Probabilistic Optimality for Inference-time Scaling},
  author={Wang, Youkang and Wang, Jian and Chen, Rubing and Wei, Xiao-Yong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---

<div align="center">

**🎯 Ready to get started?** [Download the data →](https://drive.google.com/drive/folders/1clbEyLvQWc3DPtBh7Jb-BTb4z7jYWWkr?usp=sharing)

</div>

---

## 📋 Table of Contents

- [🔄 Reproducibility & Data Resources](#-reproducibility--data-resources) ⭐ **NEW**
- [Overview](#overview)
- [Key Features](#key-features)
- [Method](#method)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Overview

OptScale addresses the fundamental challenge of **inference-time scaling**: determining the optimal number of samples $N$ to generate for a given problem to maximize accuracy while minimizing computational cost. Unlike existing methods that rely on heuristics or fixed stopping criteria, OptScale provides a principled probabilistic framework that:

- **Optimizes sample size** based on problem difficulty and model competence
- **Dynamically adapts** to different problem characteristics
- **Achieves superior efficiency** compared to existing test-time scaling methods

## Key Features

- 🎯 **Two Variants**:
  - **OptScale<sup>0</sup>**: Maximum Likelihood Estimation (MLE) based approach
  - **OptScale<sup>t</sup>**: Uses a learned predictor for problem-specific parameters

- 📊 **Comprehensive Baselines**: Full implementations of:
  - Best-of-N (BoN)
  - Self-Consistency (SC)
  - Early-stopping SC (ESC)
  - Adaptive SC (ASC)
  - Difficulty-Adaptive SC (DSC)
  - MR-Thinking

- 🔬 **Multiple Datasets**: Evaluated on 5 mathematical reasoning datasets:
  - MATH500
  - GSM8K
  - AIME24
  - AIME25
  - AMC23

- 🛠️ **Complete Pipeline**: Includes data generation, model training, evaluation, and visualization tools

## Method

OptScale formulates inference-time scaling as a probabilistic optimization problem. Given a problem $x$ and model $M$, we aim to find the optimal sample size $N^*$ that maximizes the expected accuracy while considering computational cost.

### OptScale<sup>0</sup> (MLE-based)
Uses Maximum Likelihood Estimation to infer problem-specific parameters ($\mu$, $\sigma$) from a small number of observed samples (typically 10), then optimizes $N$ based on these estimates. This approach adapts to each problem's difficulty without requiring pre-training.

### OptScale<sup>t</sup> (MAP-based with Predictor)
Employs a learned neural predictor (based on Qwen-1.5B) to estimate problem difficulty parameters ($\mu$, $\sigma$) before generation. These predictions serve as priors for Maximum A Posteriori (MAP) estimation, enabling more efficient scaling decisions with fewer initial samples.

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.12+

### Setup

```bash
# Clone the repository
git clone https://github.com/Albertwyk/OptScale.git
cd OptScale

# Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` includes all necessary packages: numpy, torch, transformers, accelerate, vllm, math-verify, scipy, tqdm, matplotlib, scikit-learn, and pytz.

## Quick Start

### Running Best-of-N (BoN) Baseline

```bash
cd BoN_OptScale
# Run for a specific dataset (AIME24, AIME25, AMC23, GSM8K, or MATH500)
python bon.py --dataset MATH500 --max_N 64
```

### Running OptScale<sup>t</sup> (Predictor-based with MAP)

OptScale<sup>t</sup> uses a trained predictor to estimate problem difficulty parameters. Run the Jupyter notebooks for each dataset:

```bash
cd BoN_OptScale
# Example for MATH500 dataset
jupyter notebook pred_bon_optscale_math500.ipynb
```

Available notebooks:
- `pred_bon_optscale_math500.ipynb`
- `pred_bon_optscale_gsm8k.ipynb`
- `pred_bon_optscale_aime24.ipynb`
- `pred_bon_optscale_aime25.ipynb`
- `pred_bon_optscale_amc23.ipynb`

These notebooks implement both OptScale<sup>0</sup> (MLE-based) and OptScale<sup>t</sup> (MAP-based with predictor) methods.

### Training the Predictor

Before running OptScale<sup>t</sup>, you need to train the predictor model:

```bash
cd train_predictor
python train_predictor_real.py
```

The trained predictor will be saved in `checkpoints_direct_qwen_real/` directory.

### Running Baselines

**Best-of-N (BoN):**
```bash
cd BoN_OptScale
python bon.py --dataset DATASET_NAME --max_N 64
```

**Self-Consistency variants:**

Each SC method has both a general script (with hardcoded dataset) and dataset-specific versions:

```bash
cd SC
# Difficulty-Adaptive SC (DSC)
# Edit DATASET variable in dsc.py, or use dataset-specific scripts:
python dsc_gsm8k.py
python dsc_aime24.py
python dsc_aime25.py
python dsc_amc23.py
python dsc.py  # Default: MATH500 (edit DATASET variable to change)

# Early-stopping SC (ESC) and Adaptive SC (ASC)
# Edit DATASET variable in esc_and_asc.py, or use dataset-specific scripts:
python esc_and_asc_gsm8k.py
python esc_and_asc_aime24.py
python esc_and_asc_aime25.py
python esc_and_asc_amc23.py
python esc_and_asc.py  # Default: MATH500 (edit DATASET variable to change)

# Majority voting (supports --dataset argument)
python maj_vote.py --dataset MATH500
```

**Sequential baseline (MR-Thinking):**
```bash
cd Seq
python mr_thinking.py --dataset MATH500
```

**Note:** All baseline scripts use `path.json` (or `seq_path.json` for sequential) to locate dataset files. Make sure the paths in these JSON files are correct for your setup.

### Configuration Files

Each directory contains a `path.json` file that specifies the locations of dataset files:

- `BoN_OptScale/path.json`: Paths for BoN and OptScale experiments
- `SC/path.json`: Paths for Self-Consistency methods
- `Seq/seq_path.json`: Paths for sequential baseline

These files map dataset names to their prompt and completion file paths. Update them according to your data directory structure.

### Generation and Scoring Utilities

The `generation_and_scoring/` directory contains utilities for:

- **Model Loading**: `llama8b_series_model.py` and `llama8b_correct_model.py` provide model loading and generation functions
- **PRM Scoring**: `prm_scoring.py` implements Process Reward Model (PRM) scoring for evaluating solution quality

These utilities are used internally by the evaluation scripts and can be adapted for custom generation pipelines.

## Repository Structure

```
OptScale/
├── BoN_OptScale/              # OptScale implementations and BoN baseline
│   ├── bon.py                 # Best-of-N baseline script
│   ├── pred_bon_optscale_*.ipynb  # OptScale^t notebooks (MLE & MAP) for each dataset
│   ├── utils.py               # Utility functions (answer extraction, verification)
│   ├── path.json              # Dataset path configuration
│   ├── *_BoN_results.json     # Best-of-N baseline results
│   ├── *_OptScale_MLE_results_*.json  # OptScale^0 (MLE) results
│   ├── *_OptScale_MAP_results_*.json  # OptScale^t (MAP) results
│   └── *_train_mu_sigma.json  # Training data statistics for predictor
│
├── SC/                        # Self-Consistency baselines
│   ├── dsc.py                 # Difficulty-Adaptive SC (general)
│   ├── dsc_*.py               # DSC for specific datasets
│   ├── esc_and_asc.py         # Early-stopping & Adaptive SC (general)
│   ├── esc_and_asc_*.py       # ESC/ASC for specific datasets
│   ├── maj_vote.py            # Majority voting baseline
│   ├── path.json               # Dataset path configuration
│   ├── utils.py                # Utility functions
│   ├── *_dsc_results.json      # DSC results
│   ├── *_esc_asc_results.json  # ESC/ASC results
│   └── *_maj_vote_results.json # Majority voting results
│
├── Seq/                       # Sequential baselines
│   ├── mr_thinking.py         # MR-Thinking sequential baseline
│   ├── seq_path.json          # Dataset path configuration
│   ├── utils.py               # Utility functions
│   └── *_MR-Thinking.json     # MR-Thinking results for each dataset
│
├── data/                      # Dataset files
│   ├── test_prompts/          # Test problem prompts
│   │   ├── math500_test.json
│   │   ├── gsm8k_test.json
│   │   ├── aime24.json
│   │   ├── aime25.json
│   │   └── amc23.json
│   └── completions/           # Pre-generated model completions
│       └── r1_distill_qwen7b/
│           ├── parallel/      # Parallel generation completions
│           └── sequential/    # Sequential generation completions
│
├── generation_and_scoring/    # Generation and PRM scoring utilities
│   ├── llama8b_series_model.py    # Generate Sequential Rollout Completions (Data seen in our Google Drive)
│   ├── llama8b_correct_model.py    # Generate Parallel Rollout Completions (Data seen in our Google Drive)
│   └── prm_scoring.py              # PRM (Process Reward Model) scoring
│
├── train_predictor/           # Predictor training for OptScale^t
│   ├── train_predictor_real.py    # Main training script
│   ├── construct_training_data_mu_sigma.py  # Training data construction
│   ├── train_data.json            # Training dataset
│   ├── val_data.json              # Validation dataset
│   ├── train_completion.json      # Training completions
│   ├── val_completion.json        # Validation completions
│   ├── train_mu_sigma.json        # Training statistics
│   ├── val_mu_sigma.json          # Validation statistics
│   └── checkpoints_direct_qwen_real/  # Trained model checkpoints
│
├── results/                   # Results visualization and analysis
│   ├── comparison_plot.py     # Generate comparison plots
│   ├── gen_csv_table.py       # Generate CSV results table
│   ├── gen_graph.py           # Generate accuracy-token plots
│   ├── gen_latex_simple.py    # Generate simple LaTeX table
│   ├── gen_latex_final.py     # Generate comprehensive LaTeX table
│   ├── results_table.csv      # Comprehensive results table
│   ├── comparison_data_summary.json  # Results summary
│   ├── *_accuracy_comparison.png     # Accuracy comparison plots
│   ├── *_tokens_comparison.png       # Token consumption plots
│   ├── *_efficiency_comparison.png   # Efficiency comparison plots
│   ├── accuracy_token_plots.png      # Multi-panel accuracy-token plots
│   ├── latex_table_*.txt             # LaTeX formatted tables
│   └── README.md                     # Results directory documentation
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Experiments

### Datasets

The repository includes test prompts and pre-generated completions for:

- **MATH500**: 500 challenging math problems from the MATH dataset
- **GSM8K**: Grade school math word problems (test set)
- **AIME24**: 2024 American Invitational Mathematics Examination problems
- **AIME25**: 2025 American Invitational Mathematics Examination problems
- **AMC23**: 2023 American Mathematics Competition problems

**Data Format:**
- Test prompts are stored in `data/test_prompts/` as JSON files
- Pre-generated completions (with PRM scores) are in `data/completions/r1_distill_qwen7b/`
- Completions are available in both `parallel/` (parallel generation) and `sequential/` (sequential generation) formats
- Each completion file contains completions, token counts, and PRM scores for up to 64 samples per problem

### Evaluation Metrics

- **Accuracy**: Percentage of correctly solved problems
- **Token Efficiency**: Tokens consumed per problem
- **Efficiency Ratio**: Accuracy per 1000 tokens

### Running Experiments

Each method can be run independently for different datasets. Results are saved as JSON files in their respective directories.

**Data Preparation:**
- Test prompts should be placed in `data/test_prompts/`
- Pre-generated completions should be in `data/completions/r1_distill_qwen7b/parallel/` or `sequential/`
- Update `path.json` files in each directory to point to your data paths

**Generating Results and Visualizations:**

After running experiments, aggregate and visualize results:

```bash
cd results
# Generate CSV table from all result JSON files
python gen_csv_table.py

# Generate comparison plots (accuracy, tokens, efficiency)
python comparison_plot.py

# Generate accuracy-token consumption plots
python gen_graph.py

# Generate LaTeX tables for publication
python gen_latex_simple.py    # Core methods only
python gen_latex_final.py     # All methods including ESC, ASC, DSC
```

All visualization outputs are saved in the `results/` directory. See `results/README.md` for detailed documentation.

## Results

OptScale achieves state-of-the-art efficiency across all evaluated datasets. Comprehensive results are available in the `results/` directory.

### Performance Summary

OptScale (both variants) significantly outperforms baseline methods in token efficiency:

- **OptScale<sup>0</sup> (MLE-based)**: 40-60% token reduction compared to BoN while maintaining comparable accuracy
- **OptScale<sup>t</sup> (MAP-based)**: Flexible accuracy-efficiency trade-offs with adaptive token budgets

Key improvements are visible across all datasets (MATH500, GSM8K, AIME24, AIME25, AMC23) at various N values (8, 16, 32, 64).

### Results Files

**Data Tables:**
- `results/results_table.csv`: Comprehensive CSV table with all methods and datasets
- `results/comparison_data_summary.json`: JSON summary of comparison data
- `results/latex_table_simple_output.txt`: LaTeX table with core methods
- `results/latex_table_final_output.txt`: Comprehensive LaTeX table with all methods

**Visualization Plots:**

Individual dataset comparisons (available for MATH500, GSM8K, AIME24, AIME25, AMC23):
- `results/{DATASET}_accuracy_comparison.png`: Accuracy (%) comparison across all methods
- `results/{DATASET}_tokens_comparison.png`: Token consumption comparison
- `results/{DATASET}_efficiency_comparison.png`: Efficiency ratio (accuracy per token)

Combined multi-dataset plots:
- `results/all_datasets_accuracy_comparison.png`: Aggregate accuracy comparison
- `results/all_datasets_tokens_comparison.png`: Aggregate token consumption
- `results/all_datasets_efficiency_comparison.png`: Aggregate efficiency comparison
- `results/accuracy_token_plots.png`: Multi-panel accuracy vs token consumption curves

For detailed information about results and visualization scripts, see `results/README.md`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026optscale,
  title={OptScale: Probabilistic Optimality for Inference-time Scaling},
  author={Wang, Youkang and Wang, Jian and Chen, Rubing and Wei, Xiao-Yong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is provided for research purposes. If you use this code or data in your research, please cite our paper as described in the [Citation](#citation) section.

