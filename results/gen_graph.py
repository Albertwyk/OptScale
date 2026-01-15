import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from typing import Dict, List, Tuple

def load_and_process_data(csv_file: str) -> Dict:
    """
    Load CSV data and organize it by method and dataset.
    Returns a dictionary with method names as keys and their data points.
    """
    df = pd.read_csv(csv_file)
    
    # Remove empty rows and rows with "OptScale (Ours)" (but keep OptScale^0 and OptScale^t variants)
    df = df.dropna(subset=['Baseline'])
    df = df[~df['Baseline'].str.contains('OptScale \\(Ours\\)(?!.*\\(MLE\\)|.*\\(MAP\\))', na=False, regex=True)]
    
    # Define datasets and N values
    datasets = ['MATH500', 'GSM8K', 'AIME24', 'AIME25', 'AMC23']
    n_values = [8, 16, 32, 64]
    
    # Organize data by method
    methods_data = {}
    
    for _, row in df.iterrows():
        baseline = row['Baseline']
        
        # Extract method name and N value
        n_match = re.search(r'\$N=(\d+)\$', baseline)
        if not n_match:
            continue
        
        n = int(n_match.group(1))
        if n not in n_values:
            continue
        
        # Clean method name (remove N specification)
        method_name = re.sub(r'\s*\(\$N=\d+\$\)', '', baseline)
        
        # Rename methods for cleaner display
        if 'OptScale$^0$ (MLE) (Ours)' in method_name:
            method_name = 'OptScale$^0$'
        elif 'OptScale$^t$ (MAP) (Ours)' in method_name:
            method_name = 'OptScale$^t$'
        elif 'Early-stopping SC (ESC)' in method_name:
            method_name = 'Early-stopping SC'
        elif 'Adaptive SC (ASC)' in method_name:
            method_name = 'Adaptive SC'
        elif 'Difficulty-Adaptive SC (DSC)' in method_name:
            method_name = 'Difficulty-Adaptive SC'
        
        if method_name not in methods_data:
            methods_data[method_name] = {dataset: {'accuracy': [], 'tokens': [], 'n_values': []} 
                                       for dataset in datasets}
        
        # Extract data for each dataset
        for dataset in datasets:
            acc_col = f'{dataset} acc.'
            token_col = f'{dataset} token.'
            
            if acc_col in row and token_col in row:
                accuracy = float(row[acc_col])
                tokens = int(row[token_col])
                
                methods_data[method_name][dataset]['accuracy'].append(accuracy)
                methods_data[method_name][dataset]['tokens'].append(tokens)
                methods_data[method_name][dataset]['n_values'].append(n)
    
    return methods_data

def create_plots(methods_data: Dict, output_file: str = 'accuracy_token_plots.png'):
    """
    Create 5 subplots showing accuracy vs token consumption for each dataset.
    Layout: 3 subplots in first row, 2 in second row.
    """
    datasets = ['MATH500', 'GSM8K', 'AIME24', 'AIME25', 'AMC23']
    
    # Set up the figure with 2 rows: 3 subplots in first row, 2 in second row
    fig = plt.figure(figsize=(18, 10))
    
    # Create subplots manually for better control
    axes = []
    for i in range(3):  # First row: 3 subplots
        axes.append(plt.subplot(2, 3, i + 1))
    for i in range(2):  # Second row: 2 subplots
        axes.append(plt.subplot(2, 3, i + 4))
    
    fig.suptitle('Accuracy vs Token Consumption by Dataset', fontsize=16, fontweight='bold')
    
    # Define 7 rainbow colors + black for 8 methods
    rainbow_colors = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']  # ROYGBIV
    colors = rainbow_colors + ['#000000']  # Add black as 8th color
    
    # Use simple, distinct markers
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    method_styles = {}
    method_names = list(methods_data.keys())
    
    # Assign specific colors for key methods
    for i, method in enumerate(method_names):
        if 'Best-of-N' in method or 'BoN' in method:
            color = '#000000'  # Black for BoN
        elif method == 'OptScale$^0$':
            color = '#FF0000'  # Red for OptScale^0
        elif method == 'OptScale$^t$':
            color = '#FF69B4'  # Pink for OptScale^t
        else:
            # Use rainbow colors for other methods, avoiding red which is used for OptScale^0
            available_colors = ['#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']  # Orange, Yellow, Green, Blue, Indigo, Violet
            color = available_colors[i % len(available_colors)]
        
        method_styles[method] = {
            'color': color,
            'marker': markers[i % len(markers)],
            'linewidth': 1.5,  # Thinner lines
            'markersize': 5    # Smaller points
        }
    
    # Create subplot for each dataset
    for dataset_idx, dataset in enumerate(datasets):
        ax = axes[dataset_idx]
        
        # Plot each method
        for method_name, method_data in methods_data.items():
            dataset_data = method_data[dataset]
            
            if len(dataset_data['accuracy']) > 0:
                # Sort by N values for proper line connection
                sorted_data = sorted(zip(dataset_data['n_values'], 
                                       dataset_data['accuracy'], 
                                       dataset_data['tokens']))
                
                n_vals, accuracies, tokens = zip(*sorted_data)
                
                # Convert tokens to thousands for better readability
                tokens_k = [t/1000 for t in tokens]
                
                ax.plot(tokens_k, accuracies, 
                       label=method_name,
                       color=method_styles[method_name]['color'],
                       marker=method_styles[method_name]['marker'],
                       linewidth=method_styles[method_name]['linewidth'],
                       markersize=method_styles[method_name]['markersize'],
                       alpha=0.8)
                
                # Annotate points with N values (smaller font)
                for n_val, acc, tok_k in zip(n_vals, accuracies, tokens_k):
                    ax.annotate(f'N={n_val}', 
                              (tok_k, acc), 
                              xytext=(3, 3),  # Smaller offset
                              textcoords='offset points', 
                              fontsize=7,     # Smaller font
                              alpha=0.6)      # More transparent
        
        # Customize subplot
        ax.set_xlabel('Token Consumption (×1000)', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(f'{dataset}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        # Set adaptive y-axis limits based on actual data range
        all_accuracies = []
        for method_name, method_data in methods_data.items():
            dataset_data = method_data[dataset]
            if len(dataset_data['accuracy']) > 0:
                all_accuracies.extend(dataset_data['accuracy'])
        
        if all_accuracies:
            min_acc = min(all_accuracies)
            max_acc = max(all_accuracies)
            acc_range = max_acc - min_acc
            # Add 5% padding on both sides, but ensure we don't go below 0 or above 100
            padding = max(acc_range * 0.05, 2)  # At least 2% padding
            y_min = max(0, min_acc - padding)
            y_max = min(100, max_acc + padding)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 100)  # Fallback if no data
    
    # Add legend to the bottom right subplot (AMC23)
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.93)  # Make room for legend and title
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show the plot
    plt.show()

def generate_summary_stats(methods_data: Dict):
    """
    Generate summary statistics for the data.
    """
    datasets = ['MATH500', 'GSM8K', 'AIME24', 'AIME25', 'AMC23']
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for method_name, method_data in methods_data.items():
        print(f"\n{method_name}:")
        for dataset in datasets:
            data = method_data[dataset]
            if len(data['accuracy']) > 0:
                avg_acc = np.mean(data['accuracy'])
                avg_tokens = np.mean(data['tokens'])
                efficiency = avg_acc / (avg_tokens / 1000)  # Accuracy per 1K tokens
                print(f"  {dataset:8}: Avg Acc={avg_acc:5.1f}%, Avg Tokens={avg_tokens:8.0f}, Efficiency={efficiency:6.2f}")

def main():
    """
    Main function to generate accuracy-token plots.
    """
    csv_file = 'results_table.csv'
    
    try:
        # Load and process data
        print("Loading data from CSV file...")
        methods_data = load_and_process_data(csv_file)
        
        print(f"Found {len(methods_data)} methods")
        for method in methods_data.keys():
            print(f"  - {method}")
        
        # Generate plots
        print("\nGenerating plots...")
        create_plots(methods_data)
        
        # Generate summary statistics
        generate_summary_stats(methods_data)
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        print("Please make sure 'results_table.csv' exists in the current directory.")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()
