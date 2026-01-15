import json
import numpy as np
from tqdm import tqdm
import os

def process_completion_file(input_file, data_file, prefix):
    """
    Process a completion JSON file and calculate statistics,
    ensuring alignment with the corresponding data file.
    
    Args:
        input_file (str): Path to the completion JSON file
        data_file (str): Path to the corresponding data file (for alignment)
        prefix (str): Prefix for output files
        
    Returns:
        dict: The results dictionary with statistics
    """
    # Load the JSON data
    print(f"Loading {input_file} file...")
    with open(input_file, 'r') as f:
        completion_data = json.load(f)
    
    print(f"Loading {data_file} file for alignment...")
    with open(data_file, 'r') as f:
        problem_data = json.load(f)
    
    # Verify counts match
    if len(completion_data) != len(problem_data):
        print(f"WARNING: {input_file} ({len(completion_data)} items) and {data_file} ({len(problem_data)} items) have different item counts!")
        print("This will likely cause alignment issues.")
    
    # Dictionary to store results
    results = {}
    
    # List to store parameters in the same order as problem_data
    params_list = []
    
    # Process each question
    print(f"Processing scores and calculating statistics for {input_file}...")
    missed_items = 0
    
    # Make sure we process in order matching the problem_data
    for i, problem_item in enumerate(tqdm(problem_data)):
        if i >= len(completion_data):
            print(f"Error: Not enough items in {input_file}, missing item at index {i}")
            raise ValueError(f"Not enough items in {input_file}, missing item at index {i}")
            # Add a default value to maintain alignment
            params_list.append([0.5, 0.2])  # Default values
            missed_items += 1
            continue
            
        completion_item = completion_data[i]
        
        # Needed: Verify alignment if there's a consistent field
        if 'unique_id' in problem_item and 'id' in completion_item:
            if problem_item['unique_id'] != completion_item['id']:
                print(f"Warning: Item at index {i} has mismatched IDs: {problem_item['unique_id']} vs {completion_item['id']}")
            # else:
            #     print(f"Item at index {i} has matching IDs: {problem_item['unique_id']} vs {completion_item['id']}")
        
        # Extract the scores list
        try:
            scores_list = completion_item["score"]["scores"][0]
            
            # Calculate mean and standard deviation
            mean = np.mean(scores_list)
            std = np.std(scores_list)
            
            # Add to params_list to maintain same order as problem_data
            params_list.append([float(mean), float(std)])
            
            # Store in results dictionary
            key = str(i)  # Use index as key for consistency
            results[key] = {
                "mean": float(mean),
                "std": float(std)
            }
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error processing item at index {i}: {e}")
            # Add a default value to maintain alignment
            raise ValueError(f"Error processing item at index {i}: {e}")
            params_list.append([0.5, 0.2])  # Default values
            missed_items += 1
    
    # Verify we have the right number of parameters
    if len(params_list) != len(problem_data):
        print(f"ERROR: Generated {len(params_list)} parameter pairs but need {len(problem_data)} to match {data_file}")
    else:
        print(f"Successfully generated {len(params_list)} parameter pairs, matching {data_file}")
    
    if missed_items > 0:
        print(f"WARNING: {missed_items} items had processing errors and were given default values")
    
    # Save results to a JSON file
    output_file = f'{prefix}_mu_sigma_dict.json'
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save the list format for compatibility with train_predictor.py
    list_output_file = f'{prefix}_mu_sigma.json'
    print(f"Saving list format to {list_output_file}...")
    with open(list_output_file, 'w') as f:
        json.dump(params_list, f)
    
    return results, params_list

def main():
    # Verify required files exist
    required_files = ['train_data.json', 'train_completion.json', 'val_data.json', 'val_completion.json']
    for file in required_files:
        if not os.path.exists(file):
            print(f"ERROR: Required file {file} not found!")
            return
    
    # Process training completion data
    train_results, train_params = process_completion_file('train_completion.json', 'train_data.json', 'train')
    
    # Process validation completion data
    val_results, val_params = process_completion_file('val_completion.json', 'val_data.json', 'val')
    
    print("\nAll processing complete!")
    print(f"Training data: processed {len(train_params)} items")
    print(f"Validation data: processed {len(val_params)} items")
    
    # Final verification
    with open('train_data.json', 'r') as f:
        train_data_count = len(json.load(f))
    
    with open('val_data.json', 'r') as f:
        val_data_count = len(json.load(f))
    
    print("\nVerification:")
    print(f"train_data.json: {train_data_count} items")
    print(f"train_mu_sigma.json: {len(train_params)} items")
    print(f"val_data.json: {val_data_count} items")
    print(f"val_mu_sigma.json: {len(val_params)} items")
    
    # Check for alignment issues
    if train_data_count != len(train_params):
        print(f"ERROR: Misalignment between train_data.json and train_mu_sigma.json!")
    if val_data_count != len(val_params):
        print(f"ERROR: Misalignment between val_data.json and val_mu_sigma.json!")

if __name__ == "__main__":
    main() 