import json
import torch
import numpy as np
from collections import Counter
from typing import List, Dict
import scipy.integrate as integrate

# Set random seed for reproducibility (same as in train_predictor_initial.py)
torch.manual_seed(42)
np.random.seed(42)

from utils import get_answer, verify_extracted_answer

# Configuration
DATASET = "AIME25"
ESC_WINDOW_SIZE = 5
ASC_THRESHOLD = 0.95
MAX_N = 64  # Maximum number of samples to consider

class BetaStoppingCriteria:
    """
    Beta-based stopping criteria for Adaptive Self-Consistency (ASC)
    Uses Beta distribution to model confidence in the most common answer
    """
    
    def __init__(self, threshold: float = 0.95):
        """
        Initialize stopping criteria
        
        Args:
            threshold: Confidence threshold for stopping (default: 0.95)
        """
        self.threshold = threshold
    
    def should_stop(self, answers: List) -> Dict:
        """
        Determine if sampling should stop based on answer distribution
        
        Args:
            answers: List of answers from multiple samples
            
        Returns:
            Dictionary with stopping decision and confidence probability
        """
        if len(answers) < 2:
            return {
                'stop': False,
                'prob': 0.0,
                'most_common': answers[0] if answers else None
            }
        
        # Count answer frequencies
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(2)
        
        if len(most_common) == 1:
            a, b = most_common[0][1], 0  # All answers are the same
        else:
            a, b = most_common[0][1], most_common[1][1]  # Top 2 answer counts
        
        a = float(a)
        b = float(b)
        
        return_dict = {
            'most_common': most_common[0][0],
            'prob': -1,
            'stop': False,
        }
        
        try:
            # Calculate probability that most common answer is correct using Beta distribution
            # P(correct) = ∫[0.5,1] x^a * (1-x)^b dx / ∫[0,1] x^a * (1-x)^b dx
            numerator = integrate.quad(lambda x: x ** a * (1 - x) ** b, 0.5, 1)[0]
            denominator = integrate.quad(lambda x: x ** a * (1 - x) ** b, 0, 1)[0]
            prob = numerator / denominator
            
            return_dict['prob'] = prob
            return_dict['stop'] = prob >= self.threshold
            
        except Exception as e:
            print(f"Error during numerical integration: {e}")
            return_dict['stop'] = False
            return_dict['prob'] = -1
        
        return return_dict


def ESC(input_list, tokens, window_size=5, max_n=64):
    """
    Early-stopping Self-Consistency algorithm
    Processes answers in windows of fixed size and stops when all answers in a window are identical
    """
    # Limit input to max_n samples
    input_list = input_list[:max_n]
    tokens_output = tokens["output"][:max_n]
    
    if len(input_list) < window_size:
        return input_list, {"input": tokens["input"], "output": tokens_output}
    
    esc_num = 0
    
    while esc_num + window_size <= len(input_list):
        # Check if all answers in current window are identical
        window_answers = input_list[esc_num:esc_num + window_size]
        if len(set(window_answers)) == 1:  # All answers in window are the same
            stop_position = esc_num + window_size
            out = {}
            out["input"] = tokens["input"]
            out["output"] = tokens_output[:stop_position]
            return input_list[:stop_position], out
        
        esc_num += window_size
    
    # If no early stopping occurred, return all answers (up to max_n)
    return input_list, {"input": tokens["input"], "output": tokens_output}


def ASC(input_list, tokens, threshold=0.95, max_n=64):
    """
    Adaptive Self-Consistency algorithm
    Uses statistical confidence based on Beta distribution
    """
    # Limit input to max_n samples
    input_list = input_list[:max_n]
    tokens_output = tokens["output"][:max_n]
    
    stop_judge = BetaStoppingCriteria(threshold)
    stop_position = len(input_list)
    
    for i in range(len(input_list)):
        judge_result = stop_judge.should_stop(input_list[:i+1])
        if judge_result["stop"]:
            stop_position = i+1
            break

    out = {}
    out["input"] = tokens["input"]
    out["output"] = tokens_output[:stop_position]
    return input_list[:stop_position], out


def get_majority_answer(model_answers, N):
    """Get majority answer from the first N answers"""
    answer_counts = {}
    for answer in model_answers[:N]:
        if answer in answer_counts:
            answer_counts[answer] += 1
        else:
            answer_counts[answer] = 1
    
    return max(answer_counts, key=answer_counts.get)


def load_validation_data():
    """Load validation data in the same format as majority voting notebook"""
    with open('../data/test_prompts/aime25.json', 'r') as f:
        dataset = json.load(f)

    with open('../data/completions/r1_distill_qwen7b/parallel/scored_qwen7b_par_aime25_64.json', 'r') as f:
        completion_data = json.load(f)
    
    # Prepare data - same as in train_predictor_initial.py
    texts = [item['question'] for item in dataset]
    gt_answers = [item['answer'] for item in dataset]
    completions = [item['score']['completions'] for item in completion_data]
    completion_tokens = [item['score']['completion_tokens'] for item in completion_data]
    model_answers = [[get_answer(item) for item in answer_set[0]] for answer_set in completions]
    
    print(f"Total dataset size: {len(texts)}")
    print(f"Number of completions per problem: {len(model_answers[0])}")
    
    return texts, gt_answers, completions, completion_tokens, model_answers


def evaluate_esc(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, window_size=5, max_n=MAX_N):
    """Evaluate ESC algorithm"""
    print(f"\n=== Evaluating ESC with window_size={window_size}, max_n={max_n} ===")
    
    correct = 0
    total_tokens = 0
    early_stops = 0
    
    for idx in range(len(val_texts)):
        # Create tokens structure for ESC
        tokens = {
            "input": 0,  # Placeholder for input tokens
            "output": val_completion_tokens[idx][0]
        }
        
        # Apply ESC algorithm
        selected_answers, used_tokens = ESC(val_model_answers[idx], tokens, window_size, max_n)
        
        # Get majority answer from selected answers
        majority_answer = get_majority_answer(selected_answers, len(selected_answers))
        
        # Check correctness
        if verify_extracted_answer(val_gt_answers[idx], majority_answer):
            correct += 1
        
        # Count tokens used
        tokens_used = sum(used_tokens["output"])
        total_tokens += tokens_used
        
        # Count early stops
        if len(selected_answers) < min(max_n, len(val_model_answers[idx])):
            early_stops += 1
    
    accuracy = correct / len(val_texts)
    average_tokens = total_tokens / len(val_texts)
    early_stop_rate = early_stops / len(val_texts)
    
    print(f"ESC Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Average tokens: {average_tokens:.2f}")
    print(f"  Early stop rate: {early_stop_rate:.4f}")
    
    return {
        "accuracy": accuracy,
        "average_token_count": average_tokens,
        "early_stop_rate": early_stop_rate,
        "correct": correct,
        "total_tokens": total_tokens
    }


def evaluate_asc(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, threshold=0.95, max_n=MAX_N):
    """Evaluate ASC algorithm"""
    print(f"\n=== Evaluating ASC with threshold={threshold}, max_n={max_n} ===")
    
    correct = 0
    total_tokens = 0
    early_stops = 0
    
    for idx in range(len(val_texts)):
        # Create tokens structure for ASC
        tokens = {
            "input": 0,  # Placeholder for input tokens
            "output": val_completion_tokens[idx][0]
        }
        
        # Apply ASC algorithm
        selected_answers, used_tokens = ASC(val_model_answers[idx], tokens, threshold, max_n)
        
        # Get majority answer from selected answers
        majority_answer = get_majority_answer(selected_answers, len(selected_answers))
        
        # Check correctness
        if verify_extracted_answer(val_gt_answers[idx], majority_answer):
            correct += 1
        
        # Count tokens used
        tokens_used = sum(used_tokens["output"])
        total_tokens += tokens_used
        
        # Count early stops
        if len(selected_answers) < min(max_n, len(val_model_answers[idx])):
            early_stops += 1
    
    accuracy = correct / len(val_texts)
    average_tokens = total_tokens / len(val_texts)
    early_stop_rate = early_stops / len(val_texts)
    
    print(f"ASC Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Average tokens: {average_tokens:.2f}")
    print(f"  Early stop rate: {early_stop_rate:.4f}")
    
    return {
        "accuracy": accuracy,
        "average_token_count": average_tokens,
        "early_stop_rate": early_stop_rate,
        "correct": correct,
        "total_tokens": total_tokens
    }


def evaluate_different_hyperparameters(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers):
    """Evaluate ESC and ASC with different hyperparameters"""
    
    # ESC with different window sizes
    esc_results = {}
    window_sizes = [3, 5, 7, 10]
    
    print("\n" + "="*50)
    print("ESC EVALUATION WITH DIFFERENT WINDOW SIZES")
    print("="*50)
    
    for window_size in window_sizes:
        result = evaluate_esc(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, window_size, MAX_N)
        esc_results[f"window_{window_size}"] = result
    
    # ASC with different thresholds
    asc_results = {}
    thresholds = [0.90, 0.95, 0.99]
    
    print("\n" + "="*50)
    print("ASC EVALUATION WITH DIFFERENT THRESHOLDS")
    print("="*50)
    
    for threshold in thresholds:
        result = evaluate_asc(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, threshold, MAX_N)
        asc_results[f"threshold_{threshold}"] = result
    
    return esc_results, asc_results


def evaluate_max_n_sweep(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers):
    """Evaluate ESC and ASC with different max_n values from 1 to 64"""
    
    max_n_values = list(range(1, 65))
    esc_results = []
    asc_results = []
    
    print("\n" + "="*60)
    print("EVALUATING ESC AND ASC WITH DIFFERENT MAX_N VALUES")
    print("="*60)
    
    for max_n in max_n_values:
        print(f"\n***************max_n={max_n}***************")
        
        # Evaluate ESC with current max_n
        esc_correct = 0
        esc_total_tokens = 0
        esc_early_stops = 0
        
        for idx in range(len(val_texts)):
            tokens = {
                "input": 0,
                "output": val_completion_tokens[idx][0]
            }
            
            selected_answers, used_tokens = ESC(val_model_answers[idx], tokens, ESC_WINDOW_SIZE, max_n)
            majority_answer = get_majority_answer(selected_answers, len(selected_answers))
            
            if verify_extracted_answer(val_gt_answers[idx], majority_answer):
                esc_correct += 1
            
            tokens_used = sum(used_tokens["output"])
            esc_total_tokens += tokens_used
            
            if len(selected_answers) < min(max_n, len(val_model_answers[idx])):
                esc_early_stops += 1
        
        esc_accuracy = esc_correct / len(val_texts)
        esc_average_tokens = esc_total_tokens / len(val_texts)
        esc_early_stop_rate = esc_early_stops / len(val_texts)
        
        print(f"ESC max_n={max_n}, Correct: {esc_correct}, Accuracy: {esc_accuracy:.4f}, Avg Tokens: {esc_average_tokens:.2f}, Early Stop Rate: {esc_early_stop_rate:.4f}")
        
        # Evaluate ASC with current max_n
        asc_correct = 0
        asc_total_tokens = 0
        asc_early_stops = 0
        
        for idx in range(len(val_texts)):
            tokens = {
                "input": 0,
                "output": val_completion_tokens[idx][0]
            }
            
            selected_answers, used_tokens = ASC(val_model_answers[idx], tokens, ASC_THRESHOLD, max_n)
            majority_answer = get_majority_answer(selected_answers, len(selected_answers))
            
            if verify_extracted_answer(val_gt_answers[idx], majority_answer):
                asc_correct += 1
            
            tokens_used = sum(used_tokens["output"])
            asc_total_tokens += tokens_used
            
            if len(selected_answers) < min(max_n, len(val_model_answers[idx])):
                asc_early_stops += 1
        
        asc_accuracy = asc_correct / len(val_texts)
        asc_average_tokens = asc_total_tokens / len(val_texts)
        asc_early_stop_rate = asc_early_stops / len(val_texts)
        
        print(f"ASC max_n={max_n}, Correct: {asc_correct}, Accuracy: {asc_accuracy:.4f}, Avg Tokens: {asc_average_tokens:.2f}, Early Stop Rate: {asc_early_stop_rate:.4f}")
        
        # Store results
        esc_results.append({
            "max_n": max_n,
            "accuracy": esc_accuracy,
            "average_token_count": esc_average_tokens,
            "early_stop_rate": esc_early_stop_rate,
            "correct": esc_correct,
            "total_tokens": esc_total_tokens
        })
        
        asc_results.append({
            "max_n": max_n,
            "accuracy": asc_accuracy,
            "average_token_count": asc_average_tokens,
            "early_stop_rate": asc_early_stop_rate,
            "correct": asc_correct,
            "total_tokens": asc_total_tokens
        })
    
    return esc_results, asc_results


def main():
    """Main evaluation function"""
    print("Loading validation data...")
    val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers = load_validation_data()
    
    # Evaluate with default hyperparameters
    print("\n" + "="*60)
    print("EVALUATION WITH DEFAULT HYPERPARAMETERS")
    print("="*60)
    
    esc_result = evaluate_esc(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, ESC_WINDOW_SIZE, MAX_N)
    asc_result = evaluate_asc(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, ASC_THRESHOLD, MAX_N)
    
    # Evaluate with different max_n values (1 to 64)
    esc_max_n_results, asc_max_n_results = evaluate_max_n_sweep(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers)
    
    # Save results
    all_results = {
        "esc_default": esc_result,
        "asc_default": asc_result,
        "esc_max_n_sweep": esc_max_n_results,
        "asc_max_n_sweep": asc_max_n_results
    }
    
    with open(f"{DATASET}_esc_asc_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {DATASET}_esc_asc_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"ESC (window_size={ESC_WINDOW_SIZE}, max_n={MAX_N}):")
    print(f"  Accuracy: {esc_result['accuracy']:.4f}")
    print(f"  Average tokens: {esc_result['average_token_count']:.2f}")
    print(f"  Early stop rate: {esc_result['early_stop_rate']:.4f}")
    
    print(f"\nASC (threshold={ASC_THRESHOLD}, max_n={MAX_N}):")
    print(f"  Accuracy: {asc_result['accuracy']:.4f}")
    print(f"  Average tokens: {asc_result['average_token_count']:.2f}")
    print(f"  Early stop rate: {asc_result['early_stop_rate']:.4f}")
    
    # Print best results from max_n sweep
    best_esc = max(esc_max_n_results, key=lambda x: x['accuracy'])
    best_asc = max(asc_max_n_results, key=lambda x: x['accuracy'])
    
    print(f"\nBest ESC result: max_n={best_esc['max_n']}, accuracy={best_esc['accuracy']:.4f}, tokens={best_esc['average_token_count']:.2f}")
    print(f"Best ASC result: max_n={best_asc['max_n']}, accuracy={best_asc['accuracy']:.4f}, tokens={best_asc['average_token_count']:.2f}")


if __name__ == "__main__":
    main()
