import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import re

def load_results(filepath='outputs/results_all.csv'):
    """Load results from the CSV file."""
    df = pd.read_csv(filepath)
    return df

def perform_ttest(df, model1, model2, backbone1=None, backbone2=None, shots=None, metric='ACC'):
    """Perform a t-test comparing two models with specific backbones."""
    # Filter for model1 with backbone1
    if backbone1 is None:
        model1_df = df[df['MODEL'] == model1]
    else:
        model1_df = df[(df['MODEL'] == model1) & (df['BACKBONE'] == backbone1)]
    
    # Filter for model2 with backbone2
    if backbone2 is None:
        model2_df = df[df['MODEL'] == model2]
    else:
        model2_df = df[(df['MODEL'] == model2) & (df['BACKBONE'] == backbone2)]
    
    # Apply shots filter if provided
    if shots is not None:
        model1_df = model1_df[model1_df['SHOTS'] == shots]
        model2_df = model2_df[model2_df['SHOTS'] == shots]
    
    model1_data = model1_df[metric].values
    model2_data = model2_df[metric].values
    
    if len(model1_data) == 0 or len(model2_data) == 0:
        return None  # Return None if no data found for either model/backbone
    
    t_stat, p_value = stats.ttest_rel(model1_data, model2_data)
    
    result = {
        'Model 1': f"{model1}_{backbone1}" if backbone1 else model1,
        'Model 2': f"{model2}_{backbone2}" if backbone2 else model2,
        'Shots': shots if shots is not None else 'All',
        'Metric': metric,
        'Model 1 Mean': model1_data.mean(),
        'Model 2 Mean': model2_data.mean(),
        'Difference': model1_data.mean() - model2_data.mean(),
        't-statistic': t_stat,
        'p-value': p_value,
        'Significant (p<0.05)': p_value < 0.05
    }
    
    return result


def comprehensive_model_comparison(df, models=None, backbones=None, shots_list=[1, 2, 4, 8, 16], metrics=['ACC', 'F1_MACRO']):
    """Run comprehensive t-tests for all model-backbone pairs across shots."""
    results = []
    
    # If no models or backbones specified, get all unique combinations
    if models is None or backbones is None:
        model_backbone_pairs = []
        unique_combinations = df[['MODEL', 'BACKBONE']].drop_duplicates()
        
        for _, row in unique_combinations.iterrows():
            model_backbone_pairs.append((row['MODEL'], row['BACKBONE']))
    else:
        model_backbone_pairs = [(model, backbone) for model in models for backbone in backbones]
    
    for shots in shots_list:
        for metric in metrics:
            for i, (model1, backbone1) in enumerate(model_backbone_pairs):
                for j, (model2, backbone2) in enumerate(model_backbone_pairs):
                    if i < j:  # Compare each pair only once
                        result = perform_ttest(df, model1, model2, backbone1, backbone2, shots, metric)
                        if result:  # Only add if result is not None
                            results.append(result)
    
    return pd.DataFrame(results)



def create_comparison_matrix(comparison_df, shots=None, metric='ACC'):
    """Create a matrix showing comparison results between all model pairs."""
    # Filter data
    filtered_df = comparison_df[comparison_df['Metric'] == metric]
    
    if shots is not None:
        filtered_df = filtered_df[filtered_df['Shots'] == shots]
    
    # Get unique model-backbone combinations
    models = list(set(filtered_df['Model 1'].unique()) | set(filtered_df['Model 2'].unique()))
    models.sort()
    
    # Escape underscore for LaTeX compatibility
    latex_models = [model.replace('_', '\\_') for model in models]
    
    # Create empty matrices
    n_models = len(models)
    diff_matrix = np.zeros((n_models, n_models))
    p_matrix = np.ones((n_models, n_models))
    significance_matrix = np.zeros((n_models, n_models), dtype=bool)
    
    # Fill matrices
    for _, row in filtered_df.iterrows():
        if row['Model 1'] in models and row['Model 2'] in models:
            i = models.index(row['Model 1'])
            j = models.index(row['Model 2'])
            diff_matrix[i, j] = row['Difference']
            diff_matrix[j, i] = -row['Difference']
            p_matrix[i, j] = row['p-value']
            p_matrix[j, i] = row['p-value']
            significance_matrix[i, j] = row['Significant (p<0.05)']
            significance_matrix[j, i] = row['Significant (p<0.05)']
    
    # Create DataFrame representation
    result = {
        'model_names': models,
        'difference_matrix': diff_matrix,
        'p_value_matrix': p_matrix,
        'significance_matrix': significance_matrix
    }
    
    # Optional: Create a more readable DataFrame
    readable_matrix = pd.DataFrame(diff_matrix, index=latex_models, columns=latex_models)
    
    # Mark significant differences
    significant_readable = pd.DataFrame(
        np.where(significance_matrix, '*', ''), 
        index=latex_models, 
        columns=latex_models
    )
    
    # Combine difference values with significance markers
    formatted_matrix = pd.DataFrame(index=latex_models, columns=latex_models, dtype=object)
    
    for i in range(n_models):
        for j in range(n_models):
            if significance_matrix[i, j]:
                formatted_matrix.iloc[i, j] = f"{diff_matrix[i, j]:.2f}*"
            else:
                formatted_matrix.iloc[i, j] = f"{diff_matrix[i, j]:.2f}"
    
    result = {
        'model_names': models,
        'latex_model_names': latex_models,
        'difference_matrix': diff_matrix,
        'p_value_matrix': p_matrix,
        'significance_matrix': significance_matrix,
        'readable_matrix': formatted_matrix
    }
    
    return result


def generate_latex_tables(comparison_df, shots_list=[1, 2, 4, 8, 16], metric='ACC'):
    latex_output_dir = 'outputs/latex'
    os.makedirs(latex_output_dir, exist_ok=True)
    
    generated_files_paths = []

    safe_metric_for_label = re.sub(r'[^\w]', '', metric.lower())
    safe_metric_for_filename = re.sub(r'[^\w-]', '', metric.lower().replace('_', '-'))

    for shots in shots_list:
        current_file_description = f"shots={shots}, metric='{metric}'"
        try:
            matrix_info = create_comparison_matrix(comparison_df, shots=shots, metric=metric)

            if not isinstance(matrix_info, dict) or 'readable_matrix' not in matrix_info:
                print(f"Skipping LaTeX for {current_file_description}: 'readable_matrix' missing or invalid.")
                continue
            matrix_df = matrix_info['readable_matrix']
            if not isinstance(matrix_df, pd.DataFrame) or matrix_df.empty:
                print(f"Skipping LaTeX for {current_file_description}: 'readable_matrix' is not a valid DataFrame or is empty.")
                continue
        except Exception as e:
            print(f"Error during create_comparison_matrix for {current_file_description}: {e}")
            continue

        try:
            n_data_cols = len(matrix_df.columns)
            n_total_latex_cols = n_data_cols + 1
            

            column_format_str = 'l|' + 'c' * n_data_cols
            

            rotated_column_headers = []
            for col_label in matrix_df.columns:  
                escaped_col_label = str(col_label) 
                rotated_column_headers.append(f"\\rotatebox{{70}}{{\\scriptsize\\bfseries\\strut {escaped_col_label}}}")

            header_labels = [""] + rotated_column_headers 
            header_tex_string = " & ".join(header_labels) + " \\\\"

            caption_metric_name = metric.replace('_', '\\_')
            label_shots_str = str(shots)

            caption_template_1 = (
                "\\caption{{Matrice di confronto ({0}) per {1} shot(s). "
                "Valori positivi indicano che il modello di riga è migliore del modello di colonna. "
                "L'asterisco (*) indica p<0.05.}}"
                "\\label{{tab:matrix-{1}shots-{2}}}\\\\"
            )
            caption_line_1 = caption_template_1.format(
                caption_metric_name, label_shots_str, safe_metric_for_label
            )

            caption_line_continued = (
                "\\caption[]{{(Segue) Matrice di confronto ({0}) per {1} shot(s)}}\\\\"
                .format(caption_metric_name, label_shots_str)
            )

            latex_content = [
                "% Questo file è un frammento LaTeX generato automaticamente.",
                "% Assicurati che il tuo documento principale includa i pacchetti:",
                "% \\usepackage{longtable}, \\usepackage{booktabs}, \\usepackage{caption}, \\usepackage{array}",
                "% !!! AGGIUNGI ANCHE: \\usepackage{graphicx} per \\rotatebox !!!",
                "\n",
                "\\begin{scriptsize}", 
                f"\\begin{{longtable}}{{{column_format_str}}}",
                
                caption_line_1,

                "\\toprule",
                header_tex_string,
                "\\midrule",
                "\\endfirsthead",
                
                caption_line_continued,

                "\\toprule",
                header_tex_string,
                "\\midrule",
                "\\endhead",
                
                "\\midrule",
                f"\\multicolumn{{{n_total_latex_cols}}}{{r}}{{\\textit{{Segue nella pagina successiva}}}} \\\\",
                "\\endfoot",
                
                "\\bottomrule",
                "\\endlastfoot",
            ]
            
            for row_idx, row_data_series in matrix_df.iterrows():
                # row_idx (etichetta di riga) dovrebbe essere già LaTeX-escaped
                str_row_label = str(row_idx) 
                str_row_values = [str(val) for val in row_data_series.values]
                row_tex_string = " & ".join([str_row_label] + str_row_values) + " \\\\"
                latex_content.append(row_tex_string)
            
            latex_content.append("\\end{longtable}")
            latex_content.append("\\end{scriptsize}")
            
            latex_matrix_string = "\n".join(latex_content)
            
            file_name = f'table_shots{label_shots_str}_{safe_metric_for_filename}.tex'
            file_path = os.path.join(latex_output_dir, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(latex_matrix_string)
            generated_files_paths.append(file_path)
        
        except Exception as e:
            print(f"Error during LaTeX string generation or file writing for {current_file_description}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if generated_files_paths:
        print(f"\nMatrici di confronto LaTeX generate con successo in '{latex_output_dir}':")
        for fp in generated_files_paths:
            print(f" - {fp}")
        return f"File LaTeX generati in '{latex_output_dir}'."
    else:
        print(f"\nNessuna matrice LaTeX è stata generata per la metrica '{metric}'. Controllare i messaggi precedenti per dettagli.")
        return f"Nessuna matrice LaTeX è stata generata per la metrica '{metric}'."




if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Load results
    results_df = load_results()
    
    # Models and backbones to compare
    models = ['TIMO', 'TIMO_S', 'Tip_Adapter', 'APE', 'GDA_CLIP']
    backbones = ["Custom_FSL1_RN50","Custom_FSL4_RN50","RN50"]
    shots_list = [1, 2, 4, 8, 16]
    
    # Run comprehensive comparison
    print("Running comprehensive model comparison...")
    comparison_df = comprehensive_model_comparison(results_df, models, backbones)
    comparison_df.to_csv('outputs/model_comparison_stats.csv', index=False)
    
    # Create comparison matrices for different shot counts
    print("Creating comparison matrices...")
    comparison_matrices = {}
    for shots in shots_list:
        comparison_matrices[shots] = create_comparison_matrix(comparison_df, shots=shots, metric='ACC')
        # Save readable matrix to CSV
        filename = f'outputs/comparison_matrix_shots{shots}_ACC.csv'
        comparison_matrices[shots]['readable_matrix'].to_csv(filename)
        print(f"\nComparison Matrix for {shots} shots saved to {filename}")
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    latex_result = generate_latex_tables(comparison_df, shots_list=shots_list)
    print(latex_result)
    
    print("\nAnalysis completed successfully!")