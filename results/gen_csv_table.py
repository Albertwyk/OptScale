import json
import os
import csv
from typing import Dict, List, Tuple

def load_bon_results(dataset: str, n: int) -> Tuple[float, int]:
    """Load BoN results from BoN_OptScale folder."""
    file_path = f"../BoN_OptScale/{dataset}_BoN_results.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            if item["N"] == n:
                accuracy = item["accuracy"] * 100  # Convert to percentage
                token_count = int(round(item["token_count"]))
                return accuracy, token_count
        
        print(f"Warning: N={n} not found in {file_path}")
        return 0.0, 0
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return 0.0, 0

def load_sc_results(dataset: str, n: int) -> Tuple[float, int]:
    """Load Self-Consistency results from SC folder."""
    file_path = f"../SC/{dataset}_maj_vote_results.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # N-th result is at index N-1
        if n <= len(data):
            item = data[n-1]
            accuracy = item["accuracy"] * 100  # Convert to percentage
            token_count = int(round(item["average_token_count"]))
            return accuracy, token_count
        
        print(f"Warning: N={n} not found in {file_path}")
        return 0.0, 0
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return 0.0, 0

def load_seq_results(dataset: str, n: int) -> Tuple[float, int]:
    """Load MR-Thinking results from Seq folder."""
    file_path = f"../Seq/{dataset}_MR-Thinking.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            if item["max_N"] == n:
                accuracy = item["accuracy"] * 100  # Convert to percentage
                token_count = int(round(item["avg_tokens"]))
                return accuracy, token_count
        
        print(f"Warning: N={n} not found in {file_path}")
        return 0.0, 0
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return 0.0, 0


def load_optscale_map_results(dataset: str, n: int) -> Tuple[float, int]:
    """Load OptScale w/ Predictor results from BoN_OptScale folder.
    
    Args:
        dataset: Dataset name
        n: N value
    
    Returns:
        Tuple of (accuracy, token_count) where accuracy is the highest found,
        and token_count is the lowest among items with the highest accuracy.
    """
    file_path = f"../BoN_OptScale/{dataset}_OptScale_MAP_results_{n}.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            print(f"Warning: No data found in {file_path}")
            return 0.0, 0
        
        # Find the highest accuracy
        max_accuracy = max(item["accuracy"] for item in data)
        
        # Among items with highest accuracy, find the one with lowest token count
        best_items = [item for item in data if item["accuracy"] == max_accuracy]
        best_item = min(best_items, key=lambda x: x["average_token_count"])
        
        accuracy = best_item["accuracy"] * 100  # Convert to percentage
        token_count = int(round(best_item["average_token_count"]))
        
        return accuracy, token_count
        
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return 0.0, 0

def load_optscale_mle_results(dataset: str, n: int) -> Tuple[float, int]:
    """Load OptScale by MLE results from BoN_OptScale folder.
    
    Args:
        dataset: Dataset name
        n: N value
    
    Returns:
        Tuple of (accuracy, token_count) where accuracy is the highest found,
        and token_count is the lowest among items with the highest accuracy.
    """
    file_path = f"../BoN_OptScale/{dataset}_OptScale_MLE_results_{n}.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            print(f"Warning: No data found in {file_path}")
            return 0.0, 0
        
        # Find the highest accuracy
        max_accuracy = max(item["accuracy"] for item in data)
        
        # Among items with highest accuracy, find the one with lowest token count
        best_items = [item for item in data if item["accuracy"] == max_accuracy]
        best_item = min(best_items, key=lambda x: x["average_token_count"])
        
        accuracy = best_item["accuracy"] * 100  # Convert to percentage
        token_count = int(round(best_item["average_token_count"]))
        
        return accuracy, token_count
        
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return 0.0, 0

def load_esc_results(dataset: str, n: int) -> Tuple[float, int]:
    """Load Early-stopping Self-Consistency (ESC) results from SC folder."""
    file_path = f"../SC/{dataset}_esc_asc_results.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        esc_sweep = data.get("esc_max_n_sweep", [])
        for item in esc_sweep:
            if item["max_n"] == n:
                accuracy = item["accuracy"] * 100  # Convert to percentage
                token_count = int(round(item["average_token_count"]))
                return accuracy, token_count
        
        print(f"Warning: max_n={n} not found in ESC results for {dataset}")
        return 0.0, 0
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return 0.0, 0

def load_asc_results(dataset: str, n: int) -> Tuple[float, int]:
    """Load Adaptive Self-Consistency (ASC) results from SC folder."""
    file_path = f"../SC/{dataset}_esc_asc_results.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        asc_sweep = data.get("asc_max_n_sweep", [])
        for item in asc_sweep:
            if item["max_n"] == n:
                accuracy = item["accuracy"] * 100  # Convert to percentage
                token_count = int(round(item["average_token_count"]))
                return accuracy, token_count
        
        print(f"Warning: max_n={n} not found in ASC results for {dataset}")
        return 0.0, 0
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return 0.0, 0

def load_dsc_results(dataset: str, n: int) -> Tuple[float, int]:
    """Load Difficulty-Adaptive Self-Consistency (DSC) results from SC folder."""
    file_path = f"../SC/{dataset}_dsc_results.json"
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        dsc_sweep = data.get("dsc_max_sample_size_sweep", [])
        for item in dsc_sweep:
            if item["max_sample_size"] == n:
                accuracy = item["accuracy"] * 100  # Convert to percentage
                token_count = int(round(item["average_token_count"]))
                return accuracy, token_count
        
        print(f"Warning: max_sample_size={n} not found in DSC results for {dataset}")
        return 0.0, 0
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return 0.0, 0

def generate_csv_table():
    """Generate the CSV table for test-time scaling results."""
    datasets = ["MATH500", "GSM8K", "AIME24", "AIME25", "AMC23"]
    n_values = [8, 16, 32, 64]
    
    # Prepare the header
    header = ["Baseline"]
    for dataset in datasets:
        header.extend([f"{dataset} acc.", f"{dataset} token."])
    
    rows = []
    rows.append(header)
    
    for n in n_values:
        # Best-of-N (BoN)
        bon_row = [f"Best-of-N (BoN) ($N={n}$)"]
        for dataset in datasets:
            acc, tokens = load_bon_results(dataset, n)
            bon_row.extend([f"{acc:.1f}", str(tokens)])
        rows.append(bon_row)
        
        # Self-Consistency
        sc_row = [f"Self-Consistency ($N={n}$)"]
        for dataset in datasets:
            acc, tokens = load_sc_results(dataset, n)
            sc_row.extend([f"{acc:.1f}", str(tokens)])
        rows.append(sc_row)
        
        # MR-Thinking
        seq_row = [f"MR-Thinking ($N={n}$)"]
        for dataset in datasets:
            acc, tokens = load_seq_results(dataset, n)
            seq_row.extend([f"{acc:.1f}", str(tokens)])
        rows.append(seq_row)
        
        # Early-stopping Self-Consistency (ESC)
        esc_row = [f"Early-stopping SC (ESC) ($N={n}$)"]
        for dataset in datasets:
            acc, tokens = load_esc_results(dataset, n)
            esc_row.extend([f"{acc:.1f}", str(tokens)])
        rows.append(esc_row)
        
        # Adaptive Self-Consistency (ASC)
        asc_row = [f"Adaptive SC (ASC) ($N={n}$)"]
        for dataset in datasets:
            acc, tokens = load_asc_results(dataset, n)
            asc_row.extend([f"{acc:.1f}", str(tokens)])
        rows.append(asc_row)
        
        # Difficulty-Adaptive Self-Consistency (DSC)
        dsc_row = [f"Difficulty-Adaptive SC (DSC) ($N={n}$)"]
        for dataset in datasets:
            acc, tokens = load_dsc_results(dataset, n)
            dsc_row.extend([f"{acc:.1f}", str(tokens)])
        rows.append(dsc_row)
        
        # OptScale^0 (Ours) - OptScale by MLE
        optscale_0_row = [f"OptScale$^0$ (MLE) (Ours) ($N={n}$)"]
        for dataset in datasets:
            acc, tokens = load_optscale_mle_results(dataset, n)
            optscale_0_row.extend([f"{acc:.1f}", str(tokens)])
        rows.append(optscale_0_row)
        
        # OptScale^t (Ours) - OptScale w/ Predictor
        optscale_t_row = [f"OptScale$^t$ (MAP) (Ours) ($N={n}$)"]
        for dataset in datasets:
            acc, tokens = load_optscale_map_results(dataset, n)
            optscale_t_row.extend([f"{acc:.1f}", str(tokens)])
        rows.append(optscale_t_row)
        
        # Add empty row between different N values (except after the last one)
        if n != n_values[-1]:
            rows.append([""])
    
    return rows

def save_to_csv(rows: List[List[str]], filename: str = "results_table.csv"):
    """Save the results to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)
    print(f"Results saved to {filename}")

def print_table(rows: List[List[str]]):
    """Print the table in a formatted way."""
    for row in rows:
        if row and row[0]:  # Skip empty rows for printing
            print('\t'.join(row))
        else:
            print()  # Print empty line

if __name__ == "__main__":
    print("Generating CSV table for test-time scaling results...")
    rows = generate_csv_table()
    
    print("\nGenerated table:")
    print_table(rows)
    
    save_to_csv(rows)
    print(f"\nTable saved to results_table.csv")
