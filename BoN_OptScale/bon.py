#!/usr/bin/env python3
"""
BoN (Best of N) evaluation script for multiple datasets.
Consolidates the logic from bon_*.ipynb notebooks into a single script.
"""

import json
import argparse
import torch
import numpy as np
from utils import get_answer, verify_extracted_answer


def load_validation_data(prompts_path, completions_path, prompt_key):
    """
    Load validation data from prompts and completions files.
    
    Args:
        prompts_path: Path to the prompts JSON file
        completions_path: Path to the completions JSON file
        prompt_key: Key to use for extracting prompts ('problem' or 'question')
    
    Returns:
        Tuple of (val_texts, val_gt_answers, val_completions, val_completion_tokens, val_scores)
    """
    with open(prompts_path, 'r') as f:
        dataset = json.load(f)
        
    with open(completions_path, 'r') as f:
        completion_data = json.load(f)
    
    # Prepare data - extract using the appropriate prompt key
    texts = [item[prompt_key] for item in dataset]
    gt_answers = [item['answer'] for item in dataset]
    completions = [item['score']['completions'] for item in completion_data]
    completion_tokens = [item['score']['completion_tokens'] for item in completion_data]
    scores = [item['score']['scores'] for item in completion_data]
    
    print(f"Total dataset size: {len(texts)}")
    print(f"Validation size: {len(texts)}")
    
    return texts, gt_answers, completions, completion_tokens, scores


def evaluate_bon(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_scores, max_N=64):
    """
    Evaluate Best of N (BoN) baseline for different values of N.
    
    Args:
        val_texts: List of problem/question texts
        val_gt_answers: List of ground truth answers
        val_completions: List of completion lists
        val_completion_tokens: List of token count lists
        val_scores: List of score lists
        max_N: Maximum N value to evaluate (default: 64)
    
    Returns:
        Tuple of (baseline_accuracy_values, baseline_average_token_counts)
    """
    N_values = list(range(1, max_N + 1))
    baseline_accuracy_values = []
    baseline_average_token_counts = []
    
    for N in N_values:
        entire_token_count = 0
        correct = 0
        
        print(f"\n\n***************N={N}***************")
        for idx in range(len(val_texts)):
            completions = val_completions[idx][0][:N]
            completion_tokens = val_completion_tokens[idx][0][:N]
            scores = val_scores[idx][0][:N]
            total_tokens = sum(completion_tokens)
            
            highest_scores_idx = scores.index(max(scores))
            while completions[highest_scores_idx] == "":
                scores[highest_scores_idx] = 0
                if max(scores) == 0:
                    print("Warning: All completions are empty!")
                    break
                highest_scores_idx = scores.index(max(scores))
            
            output = completions[highest_scores_idx]
            answer = get_answer(output)
            
            if verify_extracted_answer(answer, val_gt_answers[idx]):
                correct += 1
            entire_token_count += total_tokens
        
        print(f"When N={N}, Correct {correct}")
        print(f"When N={N}, Total Token Count {entire_token_count}")
        accuracy = correct / len(val_texts)
        baseline_accuracy_values.append(accuracy)
        print(f"When N={N}, Accuracy {accuracy}")
        average_token_count = entire_token_count / len(val_texts)
        baseline_average_token_counts.append(average_token_count)
        print(f"When N={N}, Average Token Count {average_token_count}")
    
    return baseline_accuracy_values, baseline_average_token_counts


def main():
    parser = argparse.ArgumentParser(description='Evaluate Best of N (BoN) baseline for different datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['AIME24', 'AIME25', 'AMC23', 'GSM8K', 'MATH500'],
                       help='Dataset to evaluate (AIME24, AIME25, AMC23, GSM8K, or MATH500)')
    parser.add_argument('--max_N', type=int, default=64,
                       help='Maximum N value to evaluate (default: 64)')
    parser.add_argument('--path_json', type=str, default='path.json',
                       help='Path to path.json file (default: path.json)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset configuration from path.json
    with open(args.path_json, 'r') as f:
        path_config = json.load(f)
    
    if args.dataset not in path_config:
        raise ValueError(f"Dataset '{args.dataset}' not found in {args.path_json}")
    
    dataset_config = path_config[args.dataset]
    prompts_path = dataset_config['prompts']
    completions_path = dataset_config['completions']
    prompt_key = dataset_config['prompt_key']
    
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Dataset: {args.dataset}")
    print(f"Prompts path: {prompts_path}")
    print(f"Completions path: {completions_path}")
    print(f"Prompt key: {prompt_key}")
    
    # Load validation data
    val_texts, val_gt_answers, val_completions, val_completion_tokens, val_scores = load_validation_data(
        prompts_path, completions_path, prompt_key
    )
    
    # Evaluate BoN baseline
    baseline_accuracy_values, baseline_average_token_counts = evaluate_bon(
        val_texts, val_gt_answers, val_completions, val_completion_tokens, val_scores, args.max_N
    )
    
    # Save results
    output_file = f'{args.dataset}_BoN_results.json'
    with open(output_file, 'w') as f:
        results = [
            {'N': n+1, 'accuracy': acc, 'token_count': tokens}
            for n, (acc, tokens) in enumerate(zip(baseline_accuracy_values, baseline_average_token_counts))
        ]
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()

