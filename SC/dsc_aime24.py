import json
import torch
import numpy as np
import random
from collections import Counter
from typing import List, Dict
import scipy.integrate as integrate

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from utils import get_answer, verify_extracted_answer

# Configuration
DATASET = "AIME24"
JUDGE_WINDOW_SIZE = 3
SAMPLE_WINDOW_SIZE = 5
MAX_SAMPLE_SIZE = 64
INTERPLOT_SIZE = 5


class MyStoppingCriteria:
    """
    Stopping criteria for DSC with both easy and hard thresholds
    """
    
    def __init__(self, easy_conf_thresh: float = 0.95, hard_conf_thresh: float = 0.50):
        self.easy_conf_thresh = easy_conf_thresh
        self.hard_conf_thresh = hard_conf_thresh
    
    def should_stop(self, answers: List, easy_conf_thresh: float = None, hard_conf_thresh: float = None, verbose: bool = False) -> Dict:
        if easy_conf_thresh is None:
            easy_conf_thresh = self.easy_conf_thresh
        if hard_conf_thresh is None:
            hard_conf_thresh = self.hard_conf_thresh
        
        if len(answers) < 2:
            return {
                'most_common': answers[0] if answers else None,
                'prob': -1,
                'easy_stop': False,
                'hard_stop': False,
            }
        
        most_common = Counter(answers).most_common(2)
        if len(most_common) == 1:
            a, b = most_common[0][1], 0
        else:
            a, b = most_common[0][1], most_common[1][1]
        
        a = float(a)
        b = float(b)
        
        return_dict = {
            'most_common': most_common[0][0],
            'prob': -1,
            'easy_stop': False,
            'hard_stop': False,
        }
        
        try:
            prob = integrate.quad(lambda x: x ** (a) * (1 - x) ** (b), 0.5, 1)[0] / \
                   integrate.quad(lambda x: x ** (a) * (1 - x) ** (b), 0, 1)[0]
        except Exception as e:
            print(f"Error during numerical integration: {e}")
            return_dict['easy_stop'] = False
            return_dict['prob'] = -1
            return return_dict
        
        return_dict['prob'] = prob
        return_dict['easy_stop'] = prob >= easy_conf_thresh
        return_dict['hard_stop'] = prob <= hard_conf_thresh
        return return_dict


def DSC_judge(input_list, max_sample_size, easy_threshold=0.95, hard_threshold=0.50, window_size=5):
    """Judge function for DSC binary search"""
    easy_stop_judge = MyStoppingCriteria(easy_threshold, hard_threshold)
    Flag = False
    
    for i in range(max_sample_size // window_size):
        if (i+1) * window_size > len(input_list):
            break
        judge_result = easy_stop_judge.should_stop(input_list[:(i+1)*window_size])
        if (i+1) * window_size >= max_sample_size:
            break
        if judge_result["easy_stop"]:
            Flag = True
            allocate = (i+1) * window_size
            break
        if judge_result["hard_stop"]:
            Flag = True
            allocate = (i+1) * window_size
            break
    
    if Flag:
        return allocate
    else:
        return max_sample_size


def dsc_allocate(pre_list, judge_window_size, sample_window_size, max_sample_size):
    """Binary search to allocate easy vs hard questions"""
    length = len(pre_list)
    allocate = [0 for i in range(length)]
    easy_sample_flag = [False for i in range(length)]
    hard_sample_flag = [False for i in range(length)]
    
    # Track binary search statistics
    easy_midpoints_count = 0
    total_midpoints_checked = 0
    
    # Binary search for easy questions
    left = 0
    right = length
    while True:
        if right <= left:
            break
        idx = (right + left) // 2
        total_midpoints_checked += 1
        
        judge_left = max(0, idx - judge_window_size // 2)
        judge_right = min(idx + judge_window_size // 2, length)
        
        midpoint_is_easy = False
        for j in range(judge_left, judge_right):
            if len(pre_list[j]) >= max_sample_size:
                random_choice = random.sample(pre_list[j], max_sample_size)
            else:
                random_choice = pre_list[j]
            sub_allocate = DSC_judge(random_choice, max_sample_size=max_sample_size, 
                                   easy_threshold=0.95, hard_threshold=0, window_size=sample_window_size)
            if sub_allocate == sample_window_size:
                easy_sample_flag[j] = True
                if j == idx:  # This is the actual midpoint
                    midpoint_is_easy = True
        
        if midpoint_is_easy:
            easy_midpoints_count += 1
        
        if all(tem == True for tem in easy_sample_flag[judge_left:judge_right]):
            for j in range(left, idx + 1):
                allocate[j] = 1
            left = idx + 1
        else:
            right = idx
    
    # Binary search for hard questions
    left = 0
    right = length
    while True:
        if right <= left:
            break
        idx = (right + left) // 2
        judge_left = max(0, idx - judge_window_size // 2)
        judge_right = min(idx + judge_window_size // 2, length)
        
        for j in range(judge_left, judge_right):
            sample_size = min(max_sample_size, len(pre_list[j]))
            sub_allocate = DSC_judge(pre_list[j][:sample_size], max_sample_size=sample_size, 
                                   easy_threshold=0.95, hard_threshold=0)
            if sub_allocate >= (sample_size // sample_window_size) * sample_window_size / 2:
                hard_sample_flag[j] = True
        
        if sum(hard_sample_flag[judge_left:judge_right]) >= judge_window_size:
            for j in range(idx, right):
                allocate[j] = max_sample_size
            right = idx
        else:
            left = idx + 1
    
    # Set remaining to -1 (difficult)
    for i in range(len(allocate)):
        if allocate[i] == 0:
            allocate[i] = -1
    
    # Print binary search statistics
    print(f"Binary search statistics:")
    print(f"  Total midpoints checked: {total_midpoints_checked}")
    print(f"  Easy midpoints (converge within {sample_window_size}): {easy_midpoints_count}")
    print(f"  Easy midpoint rate: {easy_midpoints_count/total_midpoints_checked:.2f}" if total_midpoints_checked > 0 else "  Easy midpoint rate: N/A")
    
    return allocate, easy_midpoints_count, total_midpoints_checked


def split_easy_hard(eval_result, pre_batch, judge_window_size, window_size, max_sample_size):
    """Split questions into easy and hard based on difficulty scores"""
    question = eval_result["questions"]
    eval_scores = eval_result["eval"]
    index = list(range(len(question)))
    
    zipped = sorted(zip(eval_scores, question, pre_batch, index), reverse=False)
    eval_scores, question, pre_batch, index = zip(*zipped)
    
    allocate_list, easy_midpoints_count, total_midpoints_checked = dsc_allocate(pre_batch, judge_window_size, window_size, max_sample_size)
    
    # Restore original order
    zipped = sorted(zip(index, eval_scores, question, pre_batch, allocate_list), reverse=False)
    index, eval_scores, question, pre_batch, allocate_list = zip(*zipped)
    
    print("easy count={}".format(sum([sub == 1 for sub in allocate_list])))
    print("medium count={}".format(sum([sub == -1 for sub in allocate_list])))
    print("hard count={}".format(sum([sub == max_sample_size for sub in allocate_list])))
    
    return index, eval_scores, question, pre_batch, allocate_list, easy_midpoints_count, total_midpoints_checked


def get_min_allocate_size(easy_stop_judge, input_list, max_sample_size):
    """Get minimum allocation size based on stopping criteria"""
    out = max_sample_size
    for i in range(min(max_sample_size, len(input_list))):
        judge_result = easy_stop_judge.should_stop(input_list[:i+1])
        if judge_result["easy_stop"]:
            out = i + 1
            break
    return out


def load_validation_data():
    """Load validation data including difficulty scores"""
    with open('../data/test_prompts/aime24.json', 'r') as f:
        dataset = json.load(f)

    with open('../data/completions/r1_distill_qwen7b/parallel/scored_qwen7b_par_aime24_64.json', 'r') as f:
        completion_data = json.load(f)
    
    # Prepare data
    texts = [item['problem'] for item in dataset]
    gt_answers = [item['answer'] for item in dataset]
    completions = [item['score']['completions'] for item in completion_data]
    completion_tokens = [item['score']['completion_tokens'] for item in completion_data]
    scores = [item['score']['scores'] for item in completion_data]
    model_answers = [[get_answer(item) for item in answer_set[0]] for answer_set in completions]
    
    # Calculate difficulty scores (1 - average_score)
    difficulty_scores = []
    for score_set in scores:
        avg_score = np.mean(score_set[0])  # Take average of scores
        difficulty_score = 1 - avg_score  # Higher difficulty when avg score is lower
        difficulty_scores.append(difficulty_score)
    
    print(f"Total dataset size: {len(texts)}")
    print(f"Number of completions per problem: {len(model_answers[0])}")
    print(f"Average difficulty score: {np.mean(difficulty_scores):.4f}")
    
    return texts, gt_answers, completions, completion_tokens, model_answers, difficulty_scores


def evaluate_dsc(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, difficulty_scores):
    """Evaluate DSC algorithm"""
    print(f"\n=== Evaluating DSC ===")
    
    # Prepare evaluation structure
    eval_result = {
        "questions": val_texts,
        "eval": difficulty_scores
    }
    
    # Split easy and hard questions
    index, eval_scores, question, pre_batch, allocate_list, easy_midpoints_count, total_midpoints_checked = split_easy_hard(
        eval_result, val_model_answers, JUDGE_WINDOW_SIZE, SAMPLE_WINDOW_SIZE, MAX_SAMPLE_SIZE
    )
    
    correct = 0
    total_tokens = 0
    easy_stops = 0
    hard_adaptive_stops = 0
    binary_search_tokens = 0
    
    # Calculate binary search overhead tokens using actual easy midpoints count
    questions_sampled_for_search = easy_midpoints_count * JUDGE_WINDOW_SIZE
    binary_search_tokens = questions_sampled_for_search * 4 * np.mean([np.mean(tokens[0][:SAMPLE_WINDOW_SIZE]) for tokens in val_completion_tokens])
    
    easy_stop_judge = MyStoppingCriteria(0.95, 0.50)
    
    for idx in range(len(val_texts)):
        allocation = allocate_list[idx]
        
        if allocation == 1:  # Easy question - use only 1 sample
            majority_answer = val_model_answers[idx][0]
            tokens_used = val_completion_tokens[idx][0][0]
            easy_stops += 1
        elif allocation == -1:  # Hard question - use adaptive stopping
            # Apply adaptive stopping criteria
            stop_position = MAX_SAMPLE_SIZE
            for i in range(min(MAX_SAMPLE_SIZE, len(val_model_answers[idx]))):
                judge_result = easy_stop_judge.should_stop(val_model_answers[idx][:i+1])
                if judge_result["easy_stop"]:
                    stop_position = i + 1
                    hard_adaptive_stops += 1
                    break
            
            # Get majority answer from selected samples
            selected_answers = val_model_answers[idx][:stop_position]
            answer_counts = Counter(selected_answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            tokens_used = sum(val_completion_tokens[idx][0][:stop_position])
        else:  # allocation == MAX_SAMPLE_SIZE (hard question with max samples)
            # Use all available samples
            selected_answers = val_model_answers[idx][:MAX_SAMPLE_SIZE]
            answer_counts = Counter(selected_answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            tokens_used = sum(val_completion_tokens[idx][0][:MAX_SAMPLE_SIZE])
        
        # Check correctness
        if verify_extracted_answer(val_gt_answers[idx], majority_answer):
            correct += 1
        
        total_tokens += tokens_used
    
    # Add binary search overhead
    total_tokens += binary_search_tokens
    
    accuracy = correct / len(val_texts)
    average_tokens = total_tokens / len(val_texts)
    easy_rate = easy_stops / len(val_texts)
    adaptive_stop_rate = hard_adaptive_stops / len(val_texts)
    
    print(f"DSC Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Average tokens: {average_tokens:.2f}")
    print(f"  Easy questions rate: {easy_rate:.4f}")
    print(f"  Hard questions with adaptive stopping: {adaptive_stop_rate:.4f}")
    print(f"  Binary search overhead tokens: {binary_search_tokens:.0f}")
    
    return {
        "accuracy": accuracy,
        "average_token_count": average_tokens,
        "easy_rate": easy_rate,
        "adaptive_stop_rate": adaptive_stop_rate,
        "correct": correct,
        "total_tokens": total_tokens,
        "binary_search_overhead": binary_search_tokens
    }


def evaluate_dsc_with_max_sample_size(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, difficulty_scores, max_sample_size):
    """Evaluate DSC algorithm with specific max_sample_size"""
    print(f"\n=== Evaluating DSC with max_sample_size={max_sample_size} ===")
    
    # Prepare evaluation structure
    eval_result = {
        "questions": val_texts,
        "eval": difficulty_scores
    }
    
    # Split easy and hard questions
    index, eval_scores, question, pre_batch, allocate_list, easy_midpoints_count, total_midpoints_checked = split_easy_hard(
        eval_result, val_model_answers, JUDGE_WINDOW_SIZE, SAMPLE_WINDOW_SIZE, max_sample_size
    )
    
    correct = 0
    total_tokens = 0
    easy_stops = 0
    hard_adaptive_stops = 0
    binary_search_tokens = 0
    
    # Calculate binary search overhead tokens using actual easy midpoints count
    questions_sampled_for_search = easy_midpoints_count * JUDGE_WINDOW_SIZE
    binary_search_tokens = questions_sampled_for_search * 4 * np.mean([np.mean(tokens[0][:4]) for tokens in val_completion_tokens])
    
    easy_stop_judge = MyStoppingCriteria(0.95, 0.50)
    
    for idx in range(len(val_texts)):
        allocation = allocate_list[idx]
        
        if allocation == 1:  # Easy question - use only 1 sample
            majority_answer = val_model_answers[idx][0]
            tokens_used = val_completion_tokens[idx][0][0]
            easy_stops += 1
        elif allocation == -1:  # Hard question - use adaptive stopping
            # Apply adaptive stopping criteria
            stop_position = max_sample_size
            for i in range(min(max_sample_size, len(val_model_answers[idx]))):
                judge_result = easy_stop_judge.should_stop(val_model_answers[idx][:i+1])
                if judge_result["easy_stop"]:
                    stop_position = i + 1
                    hard_adaptive_stops += 1
                    break
            
            # Get majority answer from selected samples
            selected_answers = val_model_answers[idx][:stop_position]
            answer_counts = Counter(selected_answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            tokens_used = sum(val_completion_tokens[idx][0][:stop_position])
        else:  # allocation == max_sample_size (hard question with max samples)
            # Use all available samples up to max_sample_size
            selected_answers = val_model_answers[idx][:max_sample_size]
            answer_counts = Counter(selected_answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            tokens_used = sum(val_completion_tokens[idx][0][:max_sample_size])
        
        # Check correctness
        if verify_extracted_answer(val_gt_answers[idx], majority_answer):
            correct += 1
        
        total_tokens += tokens_used
    
    # Add binary search overhead
    total_tokens += binary_search_tokens
    
    accuracy = correct / len(val_texts)
    average_tokens = total_tokens / len(val_texts)
    easy_rate = easy_stops / len(val_texts)
    adaptive_stop_rate = hard_adaptive_stops / len(val_texts)
    
    return {
        "max_sample_size": max_sample_size,
        "accuracy": accuracy,
        "average_token_count": average_tokens,
        "easy_rate": easy_rate,
        "adaptive_stop_rate": adaptive_stop_rate,
        "correct": correct,
        "total_tokens": total_tokens,
        "binary_search_overhead": binary_search_tokens
    }


def evaluate_max_sample_size_sweep(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, difficulty_scores):
    """Evaluate DSC with different max_sample_size values from 1 to 64"""
    
    max_sample_size_values = list(range(1, 65))
    dsc_results = []
    
    print("\n" + "="*60)
    print("EVALUATING DSC WITH DIFFERENT MAX_SAMPLE_SIZE VALUES")
    print("="*60)
    
    for max_sample_size in max_sample_size_values:
        print(f"\n***************max_sample_size={max_sample_size}***************")
        
        result = evaluate_dsc_with_max_sample_size(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, difficulty_scores, max_sample_size)
        
        print(f"DSC max_sample_size={max_sample_size}, Correct: {result['correct']}, Accuracy: {result['accuracy']:.4f}, Avg Tokens: {result['average_token_count']:.2f}")
        print(f"Easy Rate: {result['easy_rate']:.4f}, Adaptive Stop Rate: {result['adaptive_stop_rate']:.4f}")
        
        dsc_results.append(result)
    
    return dsc_results


def main():
    """Main evaluation function"""
    print("Loading validation data...")
    val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, difficulty_scores = load_validation_data()
    
    # Evaluate with default hyperparameters
    print("\n" + "="*60)
    print("EVALUATION WITH DEFAULT HYPERPARAMETERS")
    print("="*60)
    
    dsc_result = evaluate_dsc(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, difficulty_scores)
    
    # Evaluate with different max_sample_size values (1 to 64)
    dsc_max_sample_size_results = evaluate_max_sample_size_sweep(val_texts, val_gt_answers, val_completions, val_completion_tokens, val_model_answers, difficulty_scores)
    
    # Save results
    all_results = {
        "dsc_default": dsc_result,
        "dsc_max_sample_size_sweep": dsc_max_sample_size_results
    }
    
    with open(f"{DATASET}_dsc_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {DATASET}_dsc_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"DSC (default MAX_SAMPLE_SIZE={MAX_SAMPLE_SIZE}):")
    print(f"  Accuracy: {dsc_result['accuracy']:.4f}")
    print(f"  Average tokens: {dsc_result['average_token_count']:.2f}")
    print(f"  Easy questions rate: {dsc_result['easy_rate']:.4f}")
    print(f"  Adaptive stopping rate: {dsc_result['adaptive_stop_rate']:.4f}")
    
    # Print best result from max_sample_size sweep
    best_dsc = max(dsc_max_sample_size_results, key=lambda x: x['accuracy'])
    print(f"\nBest DSC result: max_sample_size={best_dsc['max_sample_size']}, accuracy={best_dsc['accuracy']:.4f}, tokens={best_dsc['average_token_count']:.2f}")


if __name__ == "__main__":
    main()
