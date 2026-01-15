import json
import torch
import numpy as np
import math
import argparse
from collections import Counter

# Set random seed for reproducibility (same as in train_predictor_initial.py)
torch.manual_seed(42)
np.random.seed(42)

from utils import get_answer, verify_extracted_answer

def load_config(config_path='path.json'):
    """Load configuration from path.json file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_validation_data(dataset_name, config):
    """Load validation data for a specific dataset using configuration"""
    dataset_config = config[dataset_name]
    prompts_path = dataset_config['prompts']
    completions_path = dataset_config['completions']
    prompt_key = dataset_config['prompt_key']
    
    with open(prompts_path, 'r') as f:
        dataset = json.load(f)
    
    with open(completions_path, 'r') as f:
        completion_data = json.load(f)
    
    # Prepare data - same as in train_predictor_initial.py
    texts = [item[prompt_key] for item in dataset]
    gt_answers = [item['answer'] for item in dataset]
    # print("GT Answers", gt_answers)
    completions = [item['score']['completions'] for item in completion_data]
    completion_tokens = [item['score']['completion_tokens'] for item in completion_data]
    # for item in completions[0]:
    #     print(len(item))
    model_answers = [[get_answer(item) for item in answer_set[0]] for answer_set in completions]
    print(len(model_answers[0]))
    
    val_texts = texts
    val_gt_answers = gt_answers
    val_completions = completions
    val_completion_tokens = completion_tokens
    val_model_answers = model_answers
    
    print(f"Total dataset size: {len(texts)}")
    print(f"Validation size: {len(val_texts)}")
    
    return val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run majority voting on test-time scaling')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['AIME24', 'AIME25', 'MATH500', 'AMC23', 'GSM8K'],
                        help='Dataset to evaluate on')
    args = parser.parse_args()
    
    DATASET = args.dataset
    
    # Load configuration
    config = load_config('path.json')
    
    # Load validation data
    val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers = load_validation_data(DATASET, config)

    def get_majority_answer(model_answers, N):
        answer_counts = {}
        for answer in model_answers[:N]:
            if answer in answer_counts:
                answer_counts[answer] += 1
            else:
                answer_counts[answer] = 1
        
        # print(f"Most Common Answer: {max(answer_counts, key=answer_counts.get)}, Count: {answer_counts[max(answer_counts, key=answer_counts.get)]}")
        # print(f"All Answers: {answer_counts}")
        return max(answer_counts, key=answer_counts.get)

    N_values = list(range(1, 65))
    baseline_accuracy_values = []
    baseline_average_token_counts = []

    for N in N_values:    
        entire_token_count = 0    
        correct = 0

        print(f"\n\n***************N={N}***************")
        for idx in range(len(val_texts)):
            # print(len(val_completions[0][0]))
            completions = val_completions[idx][0][:N]
            completion_tokens = val_completion_tokens[idx][0][:N]
            total_tokens = sum(completion_tokens)
            
            majority_answer = get_majority_answer(val_model_answers[idx], N)

            if verify_extracted_answer(val_gt_answers[idx], majority_answer):
                correct += 1
            entire_token_count += total_tokens

        print(f"When N={N}, Correct", correct)
        print(f"When N={N}, Total Token Count", entire_token_count)

        accuracy = correct / len(val_texts)
        baseline_accuracy_values.append(accuracy)
        print(f"When N={N}, Accuracy", accuracy)

        average_token_count = entire_token_count / len(val_texts)
        baseline_average_token_counts.append(average_token_count)
        print(f"When N={N}, Average Token Count", average_token_count)

    results = []
    for i in range(len(baseline_accuracy_values)):
        results.append({
            "accuracy": baseline_accuracy_values[i],
            "average_token_count": baseline_average_token_counts[i]
        })

    # Save model performance as a list of dictionaries into json file
    with open(f"{DATASET}_maj_vote_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()