import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel
import re
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
class QwenMuSigmaPredictor(nn.Module):
    def __init__(self, model_name, freeze_base=True):
        super().__init__()
        self.qwen = AutoModel.from_pretrained(model_name)
        print(self.qwen)
        # This is the hidden size of the Qwen model's output embeddings
        hidden_size = self.qwen.config.hidden_size  # For Qwen-1.5B, this is typically 2048
        
        # Freeze base model parameters if specified
        if freeze_base:
            # Freeze all base model params except the last 2 layers
            for name, param in self.qwen.named_parameters():
                # Only unfreeze the last 2 transformer layers (26 and 27)
                if "layers.26" not in name and "layers.27" not in name:
                    param.requires_grad = False
        
        # Improved prediction head for mu and sigma
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Separate predictor for mu
        self.mu_predictor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # 1 output: mu
        )
        
        # Separate predictor for sigma
        self.sigma_predictor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # 1 output: sigma
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use attention pooling to get a weighted sum of token representations
        attention_weights = self.attention_pooling(hidden_states)
        pooled_output = torch.sum(attention_weights * hidden_states, dim=1)
        
        # Get predictions from separate networks
        mu = self.mu_predictor(pooled_output)
        sigma = self.sigma_predictor(pooled_output)
        
        # Concatenate predictions
        predictions = torch.cat([mu, sigma], dim=1)
        return predictions
    
class LlamaMuSigmaPredictor(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", freeze_base=True):
        super().__init__()
        self.llama = AutoModel.from_pretrained(model_name)
        hidden_size = self.llama.config.hidden_size
        
        # Freeze base model parameters if specified
        if freeze_base:
            for name, param in self.llama.named_parameters():
                if "layers.14" not in name and "layers.15" not in name:
                    param.requires_grad = False
        
        # Improved prediction head for mu and sigma
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Separate predictor for mu
        self.mu_predictor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Separate predictor for sigma
        self.sigma_predictor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use attention pooling to get a weighted sum of token representations
        attention_weights = self.attention_pooling(hidden_states)
        pooled_output = torch.sum(attention_weights * hidden_states, dim=1)
        
        # Get predictions from separate networks
        mu = self.mu_predictor(pooled_output)
        sigma = self.sigma_predictor(pooled_output)
        
        # Concatenate predictions
        predictions = torch.cat([mu, sigma], dim=1)
        return predictions


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=768): # Previously was 768
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            
            'labels': torch.tensor(label, dtype=torch.float32)
        }
        

class TextDatasetNoLabels(Dataset):
    def __init__(self, texts, tokenizer, max_length=768): # Previously was 768
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
import regex
from typing import Optional
from math_verify import parse, verify
import re

BOXED_ANSWER_REGEX = regex.compile(r"\\boxed{((?:[^{}]|{(?1)})*)}", regex.DOTALL)

def extract_last_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the content of the last \boxed{...} in the text.
    
    Args:
        text: The text to extract from
        
    Returns:
        The content of the last boxed expression, or None if not found
    """
    matches = BOXED_ANSWER_REGEX.findall(text)
    if matches:
        return matches[-1]  # Return the last match
    return None

def get_solution_part(generation_text: str) -> Optional[str]:
    """
    Extract the solution part from a generation (text after </think> tag).
    
    Args:
        generation_text: The full generation text
        
    Returns:
        The solution part or None if not found
    """
    if '</think>' in generation_text:
        parts = generation_text.split('</think>', 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
    return None

def verify_extracted_answer(gold_answer_raw, extracted_answer_content: Optional[str]) -> Optional[bool]:
    """
    Verify the extracted answer against the gold answer using math_verify.
    
    Args:
        gold_answer_raw: The raw gold answer string
        extracted_answer_content: The extracted answer content (from inside boxed)
        
    Returns:
        True if correct, False if incorrect, None if verification failed
    """
    if extracted_answer_content is None:
        return None
    
    if not isinstance(gold_answer_raw, str):
        gold_answer_raw = str(gold_answer_raw)

    # Format both answers with \boxed{...}
    if "\\boxed{" in gold_answer_raw:
        gold_boxed = gold_answer_raw
    else:
        gold_boxed = f"\\boxed{{{gold_answer_raw}}}"
    if "\\boxed{" in extracted_answer_content:
        extracted_boxed = extracted_answer_content
    else:
        extracted_boxed = f"\\boxed{{{extracted_answer_content}}}"
    
    try:
        # Suppress stderr to avoid printing "Error during comparison" messages
        import sys
        import os
        from contextlib import redirect_stderr
        
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                # Parse and verify the answers
                parsed_gold = parse(gold_boxed)
                parsed_extracted = parse(extracted_boxed)
                is_correct = verify(parsed_gold, parsed_extracted)
        return is_correct
    except Exception as e:
        # Silently handle verification errors without printing
        return None


def get_answer(completion):
    extracted_answer = extract_last_boxed_answer(completion)
    # print(extracted_answer)
    return extracted_answer

# def get_answer(completion):
#     extracted_answer = extract_last_boxed_answer(completion)
#     print(extracted_answer)
#     return extracted_answer


def truncated_normal_pdf(x, mu, sigma):
    """Compute PDF of truncated normal distribution (0 ≤ x ≤ 1)."""
    a = (0 - mu) / sigma
    b = (1 - mu) / sigma
    Z = norm.cdf(b) - norm.cdf(a)  # Normalization factor
    return norm.pdf(x, mu, sigma) / Z

def truncated_normal_percentile(q, mu, sigma):
    """Compute percentile for truncated normal distribution."""
    a = (0 - mu) / sigma
    b = (1 - mu) / sigma
    prob = norm.cdf(a) + q * (norm.cdf(b) - norm.cdf(a))
    percentile = mu + sigma * norm.ppf(prob)
    return np.clip(percentile, 0, 1)

def truncated_normal_cdf(x, mu, sigma):
    """Compute CDF of truncated normal distribution (0 ≤ x ≤ 1)."""
    a = (0 - mu)/sigma
    b = (1 - mu)/sigma
    trunc_Z = norm.cdf(b) - norm.cdf(a)
    return (norm.cdf((x - mu)/sigma) - norm.cdf(a)) / trunc_Z

def max_score_distribution_pdf(x, mu, sigma, N):
    """
    Compute PDF of maximum score after N samples from truncated normal.
    
    The PDF of the maximum of N independent samples is:
    f_max(x) = N * f(x) * [F(x)]^(N-1)
    
    Where:
    - f(x) is the original PDF
    - F(x) is the original CDF
    - N is the number of samples
    """
    pdf = truncated_normal_pdf(x, mu, sigma)
    cdf = truncated_normal_cdf(x, mu, sigma)
    
    # PDF of maximum is: N * original_pdf * original_cdf^(N-1)
    max_pdf = N * pdf * np.power(cdf, N-1)
    
    return max_pdf


def calculate_percentile_from_pdf(pdf, x_values, percentile):
    """Calculate percentile from PDF by first converting to CDF"""
    # Convert PDF to CDF through numerical integration
    dx = x_values[1] - x_values[0]  # Step size
    cdf = np.cumsum(pdf) * dx  # Numerical integration
    cdf = cdf / cdf[-1]  # Normalize to ensure max is 1.0
    
    # Find the x value where CDF crosses the percentile threshold
    idx = np.searchsorted(cdf, percentile)
    if idx >= len(x_values):
        return x_values[-1]  # Return max if percentile is beyond range
    return x_values[idx]

x = np.linspace(0, 1, 1000)  # Score range

def find_min_N_for_threshold(mu, sigma, target_score, percentile, max_N):
    """Find minimum N where percentile of max distribution exceeds target_score"""
    for N in range(1, max_N+1):
        # Calculate the PDF for this N
        if N == 1:
            pdf = truncated_normal_pdf(x, mu, sigma)
        else:
            pdf = max_score_distribution_pdf(x, mu, sigma, N)
            
        # Calculate the percentile
        score = calculate_percentile_from_pdf(pdf, x, percentile)
        
        # Check if we've reached the threshold
        if score >= target_score:
            return N
    
    return max_N # None  # Couldn't find N within max_N


def neg_log_likelihood(params, data_array):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    a = (0 - mu) / sigma
    b = (1 - mu) / sigma
    Z = norm.cdf(b) - norm.cdf(a)
    if Z <= 0:
        return np.inf
    log_p = np.sum(norm.logpdf((data_array - mu)/sigma) - np.log(sigma)) - len(data_array)*np.log(Z)
    return -log_p


def neg_log_posterior(params, data_array, prior_mu, prior_sigma, prior_mu_std=0.1, prior_sigma_std=0.1):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
        
    # Log likelihood (same as before)
    a = (0 - mu) / sigma
    b = (1 - mu) / sigma
    Z = norm.cdf(b) - norm.cdf(a)
    if Z <= 0:
        return np.inf
    log_likelihood = np.sum(norm.logpdf((data_array - mu)/sigma) - np.log(sigma)) - len(data_array)*np.log(Z)
    
    # Log prior (assuming Gaussian priors for both mu and sigma)
    log_prior_mu = norm.logpdf(mu, loc=prior_mu, scale=prior_mu_std)
    log_prior_sigma = norm.logpdf(sigma, loc=prior_sigma, scale=prior_sigma_std)
    
    # Negative log posterior (we minimize this)
    return -(log_likelihood + log_prior_mu + log_prior_sigma)



def plot_optscale(grid_search_results, baseline_average_token_counts, baseline_accuracy_values, max_N_panel):
    # Plot results
    plt.figure(figsize=(12, 8))

    # Extract data for plotting
    x_values = [result['average_token_count'] for result in grid_search_results]
    y_values = [result['accuracy'] for result in grid_search_results]
    labels = [f"TS:{result['target_score']}, P:{result['percentile']}" for result in grid_search_results]

    # Create scatter plot
    scatter = plt.scatter(x_values, y_values, c=range(len(grid_search_results)), 
                        cmap='viridis', s=100, alpha=0.7)

    # Add annotations for each point
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        target = grid_search_results[i]['target_score']
        perc = grid_search_results[i]['percentile']
        plt.annotate(f"({target}, {perc})", 
                    xy=(x, y), 
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=6)

    plt.plot(baseline_average_token_counts[:max_N_panel], baseline_accuracy_values[:max_N_panel], 'k-', 
            linewidth=1.5, alpha=0.8, label='BoN Baseline')

    for i in range(max_N_panel):
        n_value = i + 1  # Convert index to N value
        plt.annotate(f"N={n_value}", 
                    xy=(baseline_average_token_counts[i], baseline_accuracy_values[i]), 
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=6)

    # Add labels and title
    plt.xlabel('Average Token Count')
    plt.ylabel('Accuracy')
    plt.title('Grid Search Results: Accuracy vs. Token Count')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Parameter Combination Index')

    # Highlight Pareto frontier
    pareto_indices = []
    sorted_by_x = sorted(enumerate(zip(x_values, y_values)), key=lambda i_xy: i_xy[1][0])
    max_y = -float('inf')
    for i, (x, y) in sorted_by_x:
        if y > max_y:
            pareto_indices.append(i)
            max_y = y

    pareto_x = [x_values[i] for i in pareto_indices]
    pareto_y = [y_values[i] for i in pareto_indices]
    plt.plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Frontier')

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print the Pareto-optimal configurations
    print("\nPareto-optimal configurations:")
    for i in pareto_indices:
        result = grid_search_results[i]
        print(f"Target Score: {result['target_score']}, Percentile: {result['percentile']}, "
            f"Accuracy: {result['accuracy']:.4f}, Avg Token Count: {result['average_token_count']:.2f}")
        
def plot_optscale_start(grid_search_results, baseline_average_token_counts, baseline_accuracy_values, max_N_panel, start_N):
    # Plot results
    plt.figure(figsize=(12, 8))

    # Extract data for plotting
    x_values = [result['average_token_count'] for result in grid_search_results]
    y_values = [result['accuracy'] for result in grid_search_results]
    labels = [f"TS:{result['target_score']}, P:{result['percentile']}" for result in grid_search_results]

    # Create scatter plot
    scatter = plt.scatter(x_values, y_values, c=range(len(grid_search_results)), 
                        cmap='viridis', s=100, alpha=0.7)

    # Add annotations for each point
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        target = grid_search_results[i]['target_score']
        perc = grid_search_results[i]['percentile']
        plt.annotate(f"({target}, {perc})", 
                    xy=(x, y), 
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=6)

    plt.plot(baseline_average_token_counts[start_N:max_N_panel], baseline_accuracy_values[start_N:max_N_panel], 'k-', 
            linewidth=1.5, alpha=0.8, label='BoN Baseline')

    for i in range(start_N, max_N_panel):
        n_value = i + 1  # Convert index to N value
        plt.annotate(f"N={n_value}", 
                    xy=(baseline_average_token_counts[i], baseline_accuracy_values[i]), 
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=6)

    # Add labels and title
    plt.xlabel('Average Token Count')
    plt.ylabel('Accuracy')
    plt.title('Grid Search Results: Accuracy vs. Token Count')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Parameter Combination Index')

    # Highlight Pareto frontier
    pareto_indices = []
    sorted_by_x = sorted(enumerate(zip(x_values, y_values)), key=lambda i_xy: i_xy[1][0])
    max_y = -float('inf')
    for i, (x, y) in sorted_by_x:
        if y > max_y:
            pareto_indices.append(i)
            max_y = y

    pareto_x = [x_values[i] for i in pareto_indices]
    pareto_y = [y_values[i] for i in pareto_indices]
    plt.plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Frontier')

    plt.legend()
    plt.tight_layout()
    plt.show()
    
    

def analyze_estimation_performance(predictor_params, mle_estimated_params, map_estimated_params, original_params_compare):
    """Analyze and compare performance of different parameter estimation methods
    
    Args:
        predictor_params: List of tuples containing predictor's (mu, sigma) estimates
        mle_estimated_params: List of tuples containing MLE (mu, sigma) estimates
        map_estimated_params: List of tuples containing MAP (mu, sigma) estimates
        original_params_compare: List of tuples containing ground truth (mu, sigma) values
    """
    # Convert to numpy arrays for analysis
    predictor_params = np.array(predictor_params)
    mle_estimated_params = np.array(mle_estimated_params)
    map_estimated_params = np.array(map_estimated_params)
    original_params = np.array(original_params_compare)

    # Calculate MSE for each method compared to ground truth
    predictor_mse_mu = np.mean((predictor_params[:, 0] - original_params[:, 0]) ** 2)
    predictor_mse_sigma = np.mean((predictor_params[:, 1] - original_params[:, 1]) ** 2)

    mle_mse_mu = np.mean((mle_estimated_params[:, 0] - original_params[:, 0]) ** 2)
    mle_mse_sigma = np.mean((mle_estimated_params[:, 1] - original_params[:, 1]) ** 2)

    map_mse_mu = np.mean((map_estimated_params[:, 0] - original_params[:, 0]) ** 2)
    map_mse_sigma = np.mean((map_estimated_params[:, 1] - original_params[:, 1]) ** 2)

    print("\nPerformance Comparison (MSE against ground truth):")
    print(f"Predictor MSE - μ: {predictor_mse_mu:.6f}, σ: {predictor_mse_sigma:.6f}")
    print(f"MLE MSE      - μ: {mle_mse_mu:.6f}, σ: {mle_mse_sigma:.6f}")
    print(f"MAP MSE      - μ: {map_mse_mu:.6f}, σ: {map_mse_sigma:.6f}")

    # Calculate statistics for each method
    print("\nStatistics of Different Estimation Methods:")
    print("Predictor Estimates:")
    print(f"Mean μ: {np.mean(predictor_params[:, 0]):.4f} ± {np.std(predictor_params[:, 0]):.4f}")
    print(f"Mean σ: {np.mean(predictor_params[:, 1]):.4f} ± {np.std(predictor_params[:, 1]):.4f}")

    print("\nMLE Estimates (10 samples):")
    print(f"Mean μ: {np.mean(mle_estimated_params[:, 0]):.4f} ± {np.std(mle_estimated_params[:, 0]):.4f}")
    print(f"Mean σ: {np.mean(mle_estimated_params[:, 1]):.4f} ± {np.std(mle_estimated_params[:, 1]):.4f}")

    print("\nMAP Estimates (10 samples):")
    print(f"Mean μ: {np.mean(map_estimated_params[:, 0]):.4f} ± {np.std(map_estimated_params[:, 0]):.4f}")
    print(f"Mean σ: {np.mean(map_estimated_params[:, 1]):.4f} ± {np.std(map_estimated_params[:, 1]):.4f}")

    print("\nGround Truth (100 samples):")
    print(f"Mean μ: {np.mean(original_params[:, 0]):.4f} ± {np.std(original_params[:, 0]):.4f}")
    print(f"Mean σ: {np.mean(original_params[:, 1]):.4f} ± {np.std(original_params[:, 1]):.4f}")
    

def plot_mle_map_optscale(mle_grid_search_results, map_grid_search_results, baseline_average_token_counts, baseline_accuracy_values, max_N_panel):
    # Plot results
    plt.figure(figsize=(12, 8))

    # Extract data for plotting - MLE
    mle_x_values = [result['average_token_count'] for result in mle_grid_search_results]
    mle_y_values = [result['accuracy'] for result in mle_grid_search_results]

    # Extract data for plotting - MAP
    map_x_values = [result['average_token_count'] for result in map_grid_search_results]
    map_y_values = [result['accuracy'] for result in map_grid_search_results]

    # Create scatter plots with different colors
    mle_scatter = plt.scatter(mle_x_values, mle_y_values, c='blue', s=100, alpha=0.7, label='MLE')
    map_scatter = plt.scatter(map_x_values, map_y_values, c='green', s=100, alpha=0.7, label='MAP')

    # Add annotations for MLE points
    for i, (x, y) in enumerate(zip(mle_x_values, mle_y_values)):
        target = mle_grid_search_results[i]['target_score']
        perc = mle_grid_search_results[i]['percentile']
        plt.annotate(f"MLE({target}, {perc})", 
                    xy=(x, y), 
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=6)

    # Add annotations for MAP points
    for i, (x, y) in enumerate(zip(map_x_values, map_y_values)):
        target = map_grid_search_results[i]['target_score']
        perc = map_grid_search_results[i]['percentile']
        plt.annotate(f"MAP({target}, {perc})", 
                    xy=(x, y), 
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=6)

    # Add baseline plot (N=10 to N=60)    
    plt.plot(baseline_average_token_counts[:max_N_panel], baseline_accuracy_values[:max_N_panel], 'k-', 
            linewidth=1.5, alpha=0.8, label='Baseline (N=10 to N=60)')

    # Add labels for each baseline point
    for i in range(max_N_panel):
        n_value = i + 1  # Convert index to N value
        plt.annotate(f"N={n_value}", 
                    xy=(baseline_average_token_counts[i], baseline_accuracy_values[i]), 
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=6)

    # Add labels and title
    plt.xlabel('Average Token Count')
    plt.ylabel('Accuracy')
    plt.title('Grid Search Results: Accuracy vs. Token Count (MLE vs MAP)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Highlight Pareto frontier for MLE
    mle_pareto_indices = []
    sorted_by_x = sorted(enumerate(zip(mle_x_values, mle_y_values)), key=lambda i_xy: i_xy[1][0])
    max_y = -float('inf')
    for i, (x, y) in sorted_by_x:
        if y > max_y:
            mle_pareto_indices.append(i)
            max_y = y

    mle_pareto_x = [mle_x_values[i] for i in mle_pareto_indices]
    mle_pareto_y = [mle_y_values[i] for i in mle_pareto_indices]
    plt.plot(mle_pareto_x, mle_pareto_y, 'b--', linewidth=2, label='MLE Pareto Frontier')

    # Highlight Pareto frontier for MAP
    map_pareto_indices = []
    sorted_by_x = sorted(enumerate(zip(map_x_values, map_y_values)), key=lambda i_xy: i_xy[1][0])
    max_y = -float('inf')
    for i, (x, y) in sorted_by_x:
        if y > max_y:
            map_pareto_indices.append(i)
            max_y = y

    map_pareto_x = [map_x_values[i] for i in map_pareto_indices]
    map_pareto_y = [map_y_values[i] for i in map_pareto_indices]
    plt.plot(map_pareto_x, map_pareto_y, 'g--', linewidth=2, label='MAP Pareto Frontier')

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print the Pareto-optimal configurations for MLE
    print("\nMLE Pareto-optimal configurations:")
    for i in mle_pareto_indices:
        result = mle_grid_search_results[i]
        print(f"Target Score: {result['target_score']}, Percentile: {result['percentile']}, "
            f"Accuracy: {result['accuracy']:.4f}, Avg Token Count: {result['average_token_count']:.2f}")

    # Print the Pareto-optimal configurations for MAP
    print("\nMAP Pareto-optimal configurations:")
    for i in map_pareto_indices:
        result = map_grid_search_results[i]
        print(f"Target Score: {result['target_score']}, Percentile: {result['percentile']}, "
            f"Accuracy: {result['accuracy']:.4f}, Avg Token Count: {result['average_token_count']:.2f}")


def plot_optscale_early_stop(grid_search_results, baseline_average_token_counts, baseline_accuracy_values, early_stop_grid, max_N_panel):
    """
    Plot OptScale results including early stopping grid results.
    
    Args:
        grid_search_results: Results from main grid search
        baseline_average_token_counts: BoN baseline token counts
        baseline_accuracy_values: BoN baseline accuracy values
        early_stop_grid: Results from early stopping experiments
        max_N_panel: Maximum N value to show in baseline
    """
    # Plot results
    plt.figure(figsize=(14, 10))

    # Extract data for plotting - Main grid search
    x_values = [result['average_token_count'] for result in grid_search_results]
    y_values = [result['accuracy'] for result in grid_search_results]

    # Extract data for plotting - Early stop grid
    early_stop_x_values = [result['average_token_count'] for result in early_stop_grid]
    early_stop_y_values = [result['accuracy'] for result in early_stop_grid]

    # Create scatter plots with different colors and markers
    main_scatter = plt.scatter(x_values, y_values, c='blue', s=100, alpha=0.7, 
                              marker='o', label='Main OptScale')
    early_stop_scatter = plt.scatter(early_stop_x_values, early_stop_y_values, c='red', s=100, alpha=0.7, 
                                    marker='s', label='Early Stop OptScale')

    # Add annotations for main grid search points
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        target = grid_search_results[i]['target_score']
        perc = grid_search_results[i]['percentile']
        plt.annotate(f"({target}, {perc})", 
                    xy=(x, y), 
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=6,
                    color='blue')

    # Add annotations for early stop points
    for i, (x, y) in enumerate(zip(early_stop_x_values, early_stop_y_values)):
        target = early_stop_grid[i]['target_score']
        plt.annotate(f"ES({target})", 
                    xy=(x, y), 
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=6,
                    color='red')

    # Plot baseline
    plt.plot(baseline_average_token_counts[:max_N_panel], baseline_accuracy_values[:max_N_panel], 'k-', 
            linewidth=1.5, alpha=0.8, label='BoN Baseline')

    # Add annotations for baseline points
    for i in range(max_N_panel):
        n_value = i + 1  # Convert index to N value
        plt.annotate(f"N={n_value}", 
                    xy=(baseline_average_token_counts[i], baseline_accuracy_values[i]), 
                    xytext=(1, 1),
                    textcoords='offset points',
                    fontsize=6,
                    color='black')

    # Add labels and title
    plt.xlabel('Average Token Count')
    plt.ylabel('Accuracy')
    plt.title('OptScale Results: Accuracy vs. Token Count (Including Early Stop)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Highlight Pareto frontier for main grid search
    main_pareto_indices = []
    sorted_by_x = sorted(enumerate(zip(x_values, y_values)), key=lambda i_xy: i_xy[1][0])
    max_y = -float('inf')
    for i, (x, y) in sorted_by_x:
        if y > max_y:
            main_pareto_indices.append(i)
            max_y = y

    main_pareto_x = [x_values[i] for i in main_pareto_indices]
    main_pareto_y = [y_values[i] for i in main_pareto_indices]
    plt.plot(main_pareto_x, main_pareto_y, 'b--', linewidth=2, label='Main OptScale Pareto')

    # Highlight Pareto frontier for early stop grid
    early_stop_pareto_indices = []
    sorted_by_x = sorted(enumerate(zip(early_stop_x_values, early_stop_y_values)), key=lambda i_xy: i_xy[1][0])
    max_y = -float('inf')
    for i, (x, y) in sorted_by_x:
        if y > max_y:
            early_stop_pareto_indices.append(i)
            max_y = y

    early_stop_pareto_x = [early_stop_x_values[i] for i in early_stop_pareto_indices]
    early_stop_pareto_y = [early_stop_y_values[i] for i in early_stop_pareto_indices]
    plt.plot(early_stop_pareto_x, early_stop_pareto_y, 'r--', linewidth=2, label='Early Stop Pareto')

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print the Pareto-optimal configurations for main grid search
    print("\nMain OptScale Pareto-optimal configurations:")
    for i in main_pareto_indices:
        result = grid_search_results[i]
        print(f"Target Score: {result['target_score']}, Percentile: {result['percentile']}, "
              f"Accuracy: {result['accuracy']:.4f}, Avg Token Count: {result['average_token_count']:.2f}")

    # Print the Pareto-optimal configurations for early stop
    print("\nEarly Stop Pareto-optimal configurations:")
    for i in early_stop_pareto_indices:
        result = early_stop_grid[i]
        print(f"Target Score: {result['target_score']}, "
              f"Accuracy: {result['accuracy']:.4f}, Avg Token Count: {result['average_token_count']:.2f}")