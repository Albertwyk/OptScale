import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def load_bon_results_by_n(dataset: str) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Load BoN results for all N values."""
    file_path = f"../BoN_OptScale/{dataset}_BoN_results.json"
    accuracy_results = {}
    token_results = {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            n = item["N"]
            accuracy = item["accuracy"] * 100  # Convert to percentage
            tokens = int(round(item["token_count"]))
            accuracy_results[n] = accuracy
            token_results[n] = tokens
        
        return accuracy_results, token_results
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return {}, {}

def load_sc_results_by_n(dataset: str) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Load Self-Consistency results for all N values."""
    file_path = f"../SC/{dataset}_maj_vote_results.json"
    accuracy_results = {}
    token_results = {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # N-th result is at index N-1
        for i, item in enumerate(data):
            n = i + 1  # N starts from 1
            accuracy = item["accuracy"] * 100  # Convert to percentage
            tokens = int(round(item["average_token_count"]))
            accuracy_results[n] = accuracy
            token_results[n] = tokens
        
        return accuracy_results, token_results
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return {}, {}

def load_seq_results_by_n(dataset: str) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Load MR-Thinking results for all N values."""
    file_path = f"../Seq/{dataset}_MR-Thinking.json"
    accuracy_results = {}
    token_results = {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            n = item["max_N"]
            accuracy = item["accuracy"] * 100  # Convert to percentage
            tokens = int(round(item["avg_tokens"]))
            accuracy_results[n] = accuracy
            token_results[n] = tokens
        
        return accuracy_results, token_results
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return {}, {}

def load_optscale_results_by_n(dataset: str, method: str) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Load OptScale results (MLE or MAP) for available N values."""
    accuracy_results = {}
    token_results = {}
    n_values = [8, 16, 32, 64]  # Only these N values are available
    
    for n in n_values:
        file_path = f"../BoN_OptScale/{dataset}_OptScale_{method}_results_{n}.json"
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not data:
                continue
            
            # Find the highest accuracy
            max_accuracy = max(item["accuracy"] for item in data)
            
            # Among items with highest accuracy, find the one with lowest token count
            best_items = [item for item in data if item["accuracy"] == max_accuracy]
            best_item = min(best_items, key=lambda x: x["average_token_count"])
            
            accuracy = best_item["accuracy"] * 100  # Convert to percentage
            tokens = int(round(best_item["average_token_count"]))
            accuracy_results[n] = accuracy
            token_results[n] = tokens
            
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
            continue
    
    return accuracy_results, token_results

def get_method_data(dataset: str, method: str, n_values: List[int]) -> Tuple[List[int], List[float], List[int]]:
    """Get data for a specific method across all N values."""
    if method == "BoN":
        acc_data, token_data = load_bon_results_by_n(dataset)
    elif method == "Self-Consistency":
        acc_data, token_data = load_sc_results_by_n(dataset)
    elif method == "Sequential":
        acc_data, token_data = load_seq_results_by_n(dataset)
    elif method == "OptScale_MLE":
        acc_data, token_data = load_optscale_results_by_n(dataset, "MLE")
        # For OptScale methods, use BoN N=1 result only for N=1
        bon_acc, bon_tokens = load_bon_results_by_n(dataset)
        if 1 in bon_acc:
            acc_data[1] = bon_acc[1]
            token_data[1] = bon_tokens[1]
        # Don't fill N=2 and N=4, let the plot connect N=1 directly to N=8
    elif method == "OptScale_MAP":
        acc_data, token_data = load_optscale_results_by_n(dataset, "MAP")
        # For OptScale methods, use BoN N=1 result only for N=1
        bon_acc, bon_tokens = load_bon_results_by_n(dataset)
        if 1 in bon_acc:
            acc_data[1] = bon_acc[1]
            token_data[1] = bon_tokens[1]
        # Don't fill N=2 and N=4, let the plot connect N=1 directly to N=8
    else:
        return [], [], []
    
    # Extract values for the specified N values
    x_vals = []
    y_acc_vals = []
    y_token_vals = []
    for n in n_values:
        if n in acc_data and n in token_data:
            x_vals.append(n)
            y_acc_vals.append(acc_data[n])
            y_token_vals.append(token_data[n])
    
    return x_vals, y_acc_vals, y_token_vals

def plot_comparison(dataset: str, plot_type: str, save_path: str = None):
    """Plot comparison for a single dataset."""
    n_values = [1, 2, 4, 8, 16, 32, 64]  # 2^0 to 2^6
    methods = ["BoN", "Self-Consistency", "Sequential", "OptScale_MLE", "OptScale_MAP"]
    method_labels = {
        "BoN": "Best-of-N (BoN)",
        "Self-Consistency": "Self-Consistency",
        "Sequential": "MR-Thinking",
        "OptScale_MLE": "OptScale$^0$ (MLE)",
        "OptScale_MAP": "OptScale$^t$ (MAP)"
    }
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, method in enumerate(methods):
        x_vals, y_acc_vals, y_token_vals = get_method_data(dataset, method, n_values)
        if x_vals and y_acc_vals and y_token_vals:
            if plot_type == "accuracy":
                y_vals = y_acc_vals
                ylabel = "Accuracy (%)"
                title_suffix = "Accuracy"
            elif plot_type == "tokens":
                y_vals = y_token_vals
                ylabel = "Token Count"
                title_suffix = "Token Consumption"
            elif plot_type == "efficiency":
                y_vals = [acc / tokens * 1000 for acc, tokens in zip(y_acc_vals, y_token_vals)]  # Scale by 1000 for readability
                ylabel = "Efficiency (Accuracy % / Token Count × 1000)"
                title_suffix = "Efficiency"
            else:
                continue
                
            plt.plot(x_vals, y_vals, marker=markers[i], color=colors[i], 
                    label=method_labels[method], linewidth=2, markersize=6)
    
    plt.xlabel('Max N', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{dataset} - {title_suffix} Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to log scale and specify ticks
    plt.xscale('log', base=2)
    plt.xticks(n_values, labels=[str(n) for n in n_values])
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_dataset_comparison(dataset: str, save_path: str = None):
    """Plot comparison for a single dataset."""
    n_values = [1, 2, 4, 8, 16, 32, 64]  # 2^0 to 2^6
    methods = ["BoN", "Self-Consistency", "Sequential", "OptScale_MLE", "OptScale_MAP"]
    method_labels = {
        "BoN": "Best-of-N (BoN)",
        "Self-Consistency": "Self-Consistency",
        "Sequential": "MR-Thinking",
        "OptScale_MLE": "OptScale$^0$ (MLE)",
        "OptScale_MAP": "OptScale$^t$ (MAP)"
    }
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, method in enumerate(methods):
        x_vals, y_vals, _ = get_method_data(dataset, method, n_values)
        if x_vals and y_vals:
            plt.plot(x_vals, y_vals, marker=markers[i], color=colors[i], 
                    label=method_labels[method], linewidth=2, markersize=6)
    
    plt.xlabel('Max N', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'{dataset} - Test-time Scaling Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to log scale and specify ticks
    plt.xscale('log', base=2)
    plt.xticks(n_values, labels=[str(n) for n in n_values])
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def collect_all_data():
    """Collect all data from different methods and datasets."""
    datasets = ["MATH500", "GSM8K", "AIME24", "AIME25", "AMC23"]
    n_values = [1, 2, 4, 8, 16, 32, 64]
    methods = ["BoN", "Self-Consistency", "Sequential", "OptScale_MLE", "OptScale_MAP"]
    
    all_data = {}
    
    for dataset in datasets:
        print(f"Collecting data for {dataset}...")
        all_data[dataset] = {}
        
        for method in methods:
            x_vals, y_acc_vals, y_token_vals = get_method_data(dataset, method, n_values)
            
            # Create efficiency values
            efficiency_vals = []
            for acc, tokens in zip(y_acc_vals, y_token_vals):
                efficiency = acc / tokens * 1000  # Scale by 1000 for readability
                efficiency_vals.append(efficiency)
            
            # Store data for this method
            method_data = {}
            for i, n in enumerate(x_vals):
                method_data[str(n)] = {
                    "accuracy": y_acc_vals[i],
                    "tokens": y_token_vals[i], 
                    "efficiency": efficiency_vals[i]
                }
            
            all_data[dataset][method] = method_data
    
    return all_data

def save_data_summary(filename: str = "comparison_data_summary.json"):
    """Save all collected data to a JSON file."""
    print("Collecting and saving data summary...")
    all_data = collect_all_data()
    
    # Add metadata
    summary = {
        "metadata": {
            "description": "Test-time scaling comparison data",
            "datasets": ["MATH500", "GSM8K", "AIME24", "AIME25", "AMC23"],
            "methods": ["BoN", "Self-Consistency", "Sequential", "OptScale_MLE", "OptScale_MAP"],
            "n_values": [1, 2, 4, 8, 16, 32, 64],
            "metrics": {
                "accuracy": "Accuracy percentage (%)",
                "tokens": "Token count",
                "efficiency": "Accuracy/token ratio × 1000"
            },
            "notes": {
                "OptScale_methods": "For N=1, uses BoN N=1 result. For N=8,16,32,64, uses best accuracy with lowest token count.",
                "missing_data": "OptScale methods have no data for N=2,4 (direct line from N=1 to N=8)"
            }
        },
        "data": all_data
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Data summary saved to {filename}")
    return summary

def plot_all_datasets():
    """Plot comparison for all datasets."""
    datasets = ["MATH500", "GSM8K", "AIME24", "AIME25", "AMC23"]
    plot_types = ["accuracy", "tokens", "efficiency"]
    plot_type_names = {
        "accuracy": "Accuracy",
        "tokens": "Token Consumption", 
        "efficiency": "Efficiency"
    }
    
    # Save data summary first
    save_data_summary()
    
    # Create individual plots for each dataset and plot type
    for dataset in datasets:
        for plot_type in plot_types:
            print(f"Generating {plot_type} plot for {dataset}...")
            save_path = f"{dataset}_{plot_type}_comparison.png"
            plot_comparison(dataset, plot_type, save_path)
    
    # Create combined plots with subplots for each plot type
    for plot_type in plot_types:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        n_values = [1, 2, 4, 8, 16, 32, 64]
        methods = ["BoN", "Self-Consistency", "Sequential", "OptScale_MLE", "OptScale_MAP"]
        method_labels = {
            "BoN": "Best-of-N (BoN)",
            "Self-Consistency": "Self-Consistency", 
            "Sequential": "MR-Thinking",
            "OptScale_MLE": "OptScale$^0$ (MLE)",
            "OptScale_MAP": "OptScale$^t$ (MAP)"
        }
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']
        
        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            
            for i, method in enumerate(methods):
                x_vals, y_acc_vals, y_token_vals = get_method_data(dataset, method, n_values)
                if x_vals and y_acc_vals and y_token_vals:
                    if plot_type == "accuracy":
                        y_vals = y_acc_vals
                        ylabel = "Accuracy (%)"
                    elif plot_type == "tokens":
                        y_vals = y_token_vals
                        ylabel = "Token Count"
                    elif plot_type == "efficiency":
                        y_vals = [acc / tokens * 1000 for acc, tokens in zip(y_acc_vals, y_token_vals)]
                        ylabel = "Efficiency (×1000)"
                    else:
                        continue
                        
                    ax.plot(x_vals, y_vals, marker=markers[i], color=colors[i], 
                           label=method_labels[method], linewidth=2, markersize=4)
            
            ax.set_xlabel('Max N', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.set_xticks(n_values)
            ax.set_xticklabels([str(n) for n in n_values])
            
            if idx == 0:  # Add legend only to the first subplot
                ax.legend(fontsize=8, loc='best')
        
        # Hide the last subplot (since we have 5 datasets but 6 subplots)
        axes[5].set_visible(False)
        
        plt.suptitle(f'All Datasets - {plot_type_names[plot_type]} Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"all_datasets_{plot_type}_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Combined {plot_type} plot saved to all_datasets_{plot_type}_comparison.png")
        plt.show()

if __name__ == "__main__":
    print("Generating test-time scaling comparison plots and data summary...")
    plot_all_datasets()
