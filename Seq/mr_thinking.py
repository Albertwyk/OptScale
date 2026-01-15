import sys
import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from utils import get_answer, verify_extracted_answer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run sequential baseline evaluation')
parser.add_argument('--dataset', type=str, required=True, 
                    choices=['AIME24', 'AIME25', 'MATH500', 'AMC23', 'GSM8K'],
                    help='Dataset to evaluate on')
args = parser.parse_args()

DATASET = args.dataset

# Load paths from seq_path.json
with open('seq_path.json', 'r') as f:
    path_config = json.load(f)

if DATASET not in path_config:
    raise ValueError(f"Dataset {DATASET} not found in seq_path.json")

dataset_config = path_config[DATASET]
prompts_path = dataset_config['prompts']
completions_path = dataset_config['completions']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Dataset: {DATASET}")
print(f"Prompts path: {prompts_path}")
print(f"Completions path: {completions_path}")

with open(prompts_path, 'r') as f:
    val_dataset = json.load(f)
    
with open(completions_path, 'r') as f:
    val_dataset_completion = json.load(f)

# Store results for each max_N
baseline_results = []

# Test different max_N values
for max_N in range(1, 65):  # Test max_N from 1 to 8
    print(f"\nTesting with max_N = {max_N}")
    correct = 0
    entire_token_count = 0

    for i in range(len(val_dataset)):
        gt_answer = val_dataset[i]['answer']
        
        # Get completions up to max_N
        completions = val_dataset_completion[i]['score']["completions"][0][:max_N]
        completion_tokens = val_dataset_completion[i]['score']["completion_tokens"][0][:max_N]
        total_tokens = sum(completion_tokens)
        
        # no prm
        output = completions[-1]
        completion_answer = get_answer(output)
        
        if verify_extracted_answer(gt_answer, completion_answer):
            correct += 1
        entire_token_count += total_tokens

    accuracy = correct/len(val_dataset)
    avg_tokens = entire_token_count/len(val_dataset)
    
    print(f"Results for max_N = {max_N}:")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average Tokens: {avg_tokens:.2f}")
    
    # Store results
    baseline_results.append({
        "max_N": max_N,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "total_correct": correct
    })
    

# Save results to file
with open(f'{DATASET}_MR-Thinking.json', 'w') as f:
    json.dump(baseline_results, f, indent=4) 
    