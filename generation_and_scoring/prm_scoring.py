import torch
import json
import argparse
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
import os
import gc

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()  # Move to CPU
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset

def append_result(result, output_path):
    # Create file with header if it doesn't exist
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            json.dump([], f)
    
    # Read existing data
    with open(output_path, 'r') as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
    
    # Append new result
    existing_data.append(result)
    
    # Write back to file
    with open(output_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

def run_prm_scoring(args):
    # Load PRM model
    print("Loading PRM model...")
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_model, trust_remote_code=True)
    prm_model = AutoModel.from_pretrained(
        args.prm_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    
    # Load Llama-3.1-8B-Instruct tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
    
    # Load generated completions
    print(f"Loading generated completions from {args.input_path}")
    generated_data = load_dataset(args.input_path)
    
    # Get list of already processed IDs
    processed_ids = set()
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, 'r') as f:
                existing_data = json.load(f)
                processed_ids = {item['id'] for item in existing_data}
        except json.JSONDecodeError:
            pass
    
    # Process each problem
    for idx, item in enumerate(tqdm(generated_data)):
        problem_id = idx+1  # Assign incremental ID starting from 1
        
        # Skip if this problem has already been processed
        if problem_id in processed_ids:
            continue
            
        problem = item['problem']
        completions = item['outputs']  # Changed from 'completions' to 'outputs'
        
        # Calculate completion tokens for each output using Llama tokenizer
        completion_tokens = []
        for completion in completions:
            tokens = tokenizer.encode(completion)
            completion_tokens.append(len(tokens))
        
        # Prepare system prompt
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        
        # Score each completion
        completion_scores = []
        completion_last_scores = []
        
        for completion in completions:
            # Make sure steps are separated
            steps = completion.split("\n\n")
            response_with_steps = "<extra_0>".join(steps) + "<extra_0>"
            
            # Evaluate with PRM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": response_with_steps},
            ]
            
            conversation_str = prm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            with torch.no_grad():
                input_ids = prm_tokenizer.encode(
                    conversation_str,
                    return_tensors="pt",
                ).to(prm_model.device)
                
                outputs = prm_model(input_ids=input_ids)
                
                step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
                token_masks = (input_ids == step_sep_id)
                step_reward = make_step_rewards(outputs[0], token_masks)
                
                # Calculate average reward
                avg_reward = np.mean(step_reward[0]) if step_reward[0] else 0
                last_reward = step_reward[0][-1] if step_reward[0] else 0
                
                completion_scores.append(avg_reward)
                completion_last_scores.append(last_reward)
            
            # Clear memory
            del input_ids, outputs, token_masks
            torch.cuda.empty_cache()
        
        # Find best completion based on PRM score
        best_idx = np.argmax(completion_scores)
        best_completion = completions[best_idx]
        
        # Create result for this question
        result = {
            "id": problem_id,
            "output": best_completion,
            "n": len(completions),
            "score": {
                "problem": [problem],
                "completions": [completions],
                "scores": [completion_scores],
                "last_scores": [completion_last_scores],
                "pred": [best_completion],
                "completion_tokens": [completion_tokens]
            }
        }
        
        # Append result to file immediately
        append_result(result, args.output_path)
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRM scoring for generated completions using Llama tokenizer")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the generated completions JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the scored results")
    parser.add_argument("--prm_model", type=str, default="Qwen/Qwen2.5-Math-PRM-7B", help="PRM model name")
    
    args = parser.parse_args()
    
    # Set environmental variable for expandable segments to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    run_prm_scoring(args)