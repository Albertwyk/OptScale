import pandas as pd
import numpy as np

def convert_csv_to_latex(csv_file):
    """
    Convert CSV results table to LaTeX format (Simplified version).
    
    Includes only core baseline methods:
    - Best-of-N (BoN)
    - Self-Consistency
    - MR-Thinking
    
    Excludes:
    - Early-stopping SC (ESC)
    - Adaptive SC (ASC)
    - Difficulty-Adaptive SC (DSC)
    
    Operations:
    - Exclude "OptScale (Ours)" rows
    - Exclude ESC, ASC, DSC methods
    - Rename "OptScale$^0$ (MLE) (Ours)" to "\\textsc{OptScale}$^0$"
    - Rename "OptScale$^t$ (MAP) (Ours)" to "\\textsc{OptScale}$^t$"
    - Bold highest accuracy and lowest token count in each group
    - Add \\midrule between groups
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Remove empty rows and unwanted methods
    df = df.dropna(subset=['Baseline'])
    
    # Exclude OptScale (Ours) rows - note: keep OptScale^0 (MLE) and OptScale^t (MAP)
    # Only exclude the generic "OptScale (Ours)" if it exists
    df = df[~df['Baseline'].str.contains('OptScale \\(Ours\\)(?!.*\\(MLE\\)|.*\\(MAP\\))', na=False, regex=True)]
    df = df[~df['Baseline'].str.contains('Early-stopping SC \\(ESC\\)', na=False)]
    df = df[~df['Baseline'].str.contains('Adaptive SC \\(ASC\\)', na=False)]
    df = df[~df['Baseline'].str.contains('Difficulty-Adaptive SC \\(DSC\\)', na=False)]
    
    # Define the N values and corresponding groups
    n_values = [8, 16, 32, 64]
    
    latex_output = []
    
    for n in n_values:
        # Filter rows for current N value
        group_df = df[df['Baseline'].str.contains(f'\\$N={n}\\$', na=False)].copy()
        
        if group_df.empty:
            continue
            
        # Get accuracy and token columns (alternating pattern)
        acc_cols = ['MATH500 acc.', 'GSM8K acc.', 'AIME24 acc.', 'AIME25 acc.', 'AMC23 acc.']
        token_cols = ['MATH500 token.', 'GSM8K token.', 'AIME24 token.', 'AIME25 token.', 'AMC23 token.']
        
        # Find max accuracy and min tokens for each dataset in this group
        max_acc = {}
        min_tokens = {}
        
        for acc_col, token_col in zip(acc_cols, token_cols):
            if acc_col in group_df.columns and token_col in group_df.columns:
                max_acc[acc_col] = group_df[acc_col].max()
                min_tokens[token_col] = group_df[token_col].min()
        
        # Generate LaTeX rows for this group
        for _, row in group_df.iterrows():
            baseline = row['Baseline']
            
            # Rename methods according to requirements
            if 'OptScale$^0$ (MLE) (Ours)' in baseline:
                baseline = baseline.replace('OptScale$^0$ (MLE) (Ours)', '\\textsc{OptScale}$^0$')
            elif 'OptScale$^t$ (MAP) (Ours)' in baseline:
                baseline = baseline.replace('OptScale$^t$ (MAP) (Ours)', '\\textsc{OptScale}$^t$')
            
            latex_row = [baseline]
            
            # Add alternating accuracy and token values
            for acc_col, token_col in zip(acc_cols, token_cols):
                if acc_col in row and token_col in row:
                    acc_val = row[acc_col]
                    token_val = int(row[token_col])
                    
                    # Format accuracy (bold if maximum in group)
                    if acc_val == max_acc.get(acc_col, -1):
                        acc_str = f'\\textbf{{{acc_val}}}'
                    else:
                        acc_str = str(acc_val)
                    
                    # Format tokens (bold if minimum in group)
                    if token_val == min_tokens.get(token_col, float('inf')):
                        token_str = f'\\textbf{{{token_val}}}'
                    else:
                        token_str = str(token_val)
                    
                    latex_row.extend([acc_str, token_str])
            
            # Join with & and add line ending
            latex_output.append('& '.join(latex_row) + '\\\\')
        
        # Add midrule after each group (except the last one)
        if n != n_values[-1]:
            latex_output.append('\\midrule')
    
    return '\n'.join(latex_output)

def main():
    """Main function to convert CSV to LaTeX and save/print results."""
    csv_file = 'results_table.csv'
    
    try:
        latex_table = convert_csv_to_latex(csv_file)
        
        # Print the LaTeX table
        print("Generated LaTeX table (Simplified - without ESC/ASC/DSC):")
        print("=" * 60)
        print(latex_table)
        print("=" * 60)
        
        # Optionally save to file
        output_file = 'latex_table_simple_output.txt'
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"\nSimplified LaTeX table saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main() 