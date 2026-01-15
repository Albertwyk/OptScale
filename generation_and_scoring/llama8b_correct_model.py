import json
import os
import re
from tqdm import tqdm
import numpy as np
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map = 'auto', low_cpu_mem_usage=True, torch_dtype=torch.float16, trust_remote_code=True)    
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    return model, tokenizer

def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    return tokenizer

def generate(prompt, model, tokenizer, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    output = output[:, inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if response.startswith("assistant\n\n"):
        response = response[11:]
    return response

def generate_batch(prompts, model, tokenizer, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=1.0, top_p=0.95)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True) 
    for i in range(len(responses)):
        if responses[i].startswith("assistant\n\n"):
            responses[i] = responses[i][11:]
    return responses

def generate_batch_vllm(prompts, model, sampling_params):
    outputs = model.generate(prompts, sampling_params, use_tqdm=True)
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    return responses

def prompt_R1(prompt, tokenizer):
    messages = [
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def clean_output(output, return_tag=False):
    if "\\boxed{" in output:
        o = output.split("\\boxed{")[1]
        if "}" in o:
            ans = ""
            while "}" in o:
                if o[0] == "\n" or (o[0]=='}' and "}" not in o[1:]):
                    break
                ans += o[0]
                o = o[1:]
            return ans
        else:
            return ""
    else:
        return ""

def main(args):
    benchmark = json.load(open(args.benchmark_path))
    
    model = LLM(
        model=args.model_path,
        max_model_len=10000,  # AIME models usually set larger to 16000
        gpu_memory_utilization=0.9
    )
    tokenizer = load_tokenizer(args.model_path)
    print(f"[INFO] Model and tokenizer loaded from {args.model_path}")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.max_new_tokens)

    print(f"[INFO] Running initial round 0 of {args.max_n-1}")
    problems = [b['problem'] for b in benchmark]

    outputs = []
    # Create template for all rounds
    template = """Answer the following questions. You should think step-by-step and put your final answer within $\\boxed{{answer}}$, where [answer] is just the final number or expression that solves the problem.\n Question: {problem}"""
    template_prompts = [prompt_R1(template.format(problem=problem), tokenizer) for problem in problems]
    responses = generate_batch_vllm(template_prompts, model, sampling_params)
    for j, response in enumerate(responses):
        outputs.append({
            'problem': problems[j],
            'outputs': [response],
        })

    # Use the same template for all rounds
    for round in range(1, args.max_n):
        prompts = []
        for j, item in enumerate(outputs):
            prompts.append(prompt_R1(template.format(problem=item['problem']), tokenizer))
        responses = generate_batch_vllm(prompts, model, sampling_params)
        for j, response in enumerate(responses):
            outputs[j]['outputs'].append(response)

    with open(args.output_file, 'w') as f:
        json.dump(outputs, f, indent=4)

    print(f"Outputs saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with think series')
    parser.add_argument('--model_path', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help='Path to the model')
    parser.add_argument('--max_n', type=int, required=True,
                        help='Maximum number of iterations')
    parser.add_argument('--max_new_tokens', type=int, default=10000,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--benchmark_path', type=str, required=True,
                        help='Path to benchmark file')
    parser.add_argument('--output_file', type=str, required=True, 
                        help='Path to output file')
    args = parser.parse_args()
    main(args)