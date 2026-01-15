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
