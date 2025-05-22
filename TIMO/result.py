import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from collections import defaultdict

def load_results(filepath='outputs/results_all.csv'):
    """Load results from the CSV file."""
    df = pd.read_csv(filepath)
    return df

def compute_summary(df):
    """Compute summary statistics across seeds for each model, backbone, and shots configuration."""
    # Group by relevant columns and compute statistics
    summary = df.groupby(['SHOTS', 'DATASET', 'MODEL', 'BACKBONE']).agg({
        'ACC': ['mean', 'std'],
        'F1_MACRO': ['mean', 'std']
    }).reset_index()
    
    # Flatten the column multiindex
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    
    return summary

def save_summary(summary, filepath='outputs/results_summary.csv'):
    """Save summary statistics to a CSV file."""
    summary.to_csv(filepath, index=False)
    print(f"Summary saved to {filepath}")

def plot_model_performance(df, dataset='artgraph', metric='ACC', save_path=None):
    """Plot performance of all models across shots for each backbone."""
    # Get all unique models and backbones
    models = df['MODEL'].unique()
    backbones = df['BACKBONE'].unique()
    
    # Filter by dataset
    filtered_df = df[df['DATASET'] == dataset]
    
    # Create a multi-plot figure
    fig, axes = plt.subplots(len(backbones), 1, figsize=(12, 5*len(backbones)), sharex=True)
    
    # For single backbone, convert axes to array for consistent indexing
    if len(backbones) == 1:
        axes = [axes]
    
    for i, backbone in enumerate(backbones):
        backbone_df = filtered_df[filtered_df['BACKBONE'] == backbone]
        
        # Group by model and shots, compute mean
        grouped = backbone_df.groupby(['MODEL', 'SHOTS'])[metric].mean().reset_index()
        
        # Pivot data for easier plotting
        if not grouped.empty:
            pivot_df = grouped.pivot(index='SHOTS', columns='MODEL', values=metric)
            
            # Plot for this backbone
            for model in pivot_df.columns:
                axes[i].plot(pivot_df.index, pivot_df[model], marker='o', label=model)
            
            axes[i].set_title(f'Performance on {backbone} backbone')
            axes[i].set_ylabel(f'{metric} (%)')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].legend()
    
    plt.xlabel('Number of Shots')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_backbone_comparison(df, models, dataset='artgraph', metric='ACC', shots=16, save_path=None):
    """Create a bar plot comparing backbones for each model at a specific shot setting."""
    # Filter data
    filtered_df = df[(df['DATASET'] == dataset) & 
                      (df['MODEL'].isin(models)) & 
                      (df['SHOTS'] == shots)]
    
    # Group by MODEL and BACKBONE
    grouped = filtered_df.groupby(['MODEL', 'BACKBONE'])[metric].mean().reset_index()
    
    # Pivot data for easier plotting
    pivot_df = grouped.pivot(index='MODEL', columns='BACKBONE', values=metric)
    
    # Create plot
    plt.figure(figsize=(14, 8))
    pivot_df.plot(kind='bar', figsize=(14, 8), width=0.8)
    
    plt.xlabel('Model')
    plt.ylabel(f'{metric} (%)')
    plt.title(f'Backbone Comparison ({shots} shots)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Backbone')
    
    # Add value labels on top of bars
    for container in plt.gca().containers:
        plt.bar_label(container, fmt='%.1f', padding=3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def create_latex_table(df, models, backbones, shots, dataset='artgraph', metric='ACC'):
    """Create a LaTeX formatted table with results."""
    # Filter data
    filtered_df = df[(df['DATASET'] == dataset) & 
                      (df['MODEL'].isin(models)) &
                      (df['BACKBONE'].isin(backbones)) &
                      (df['SHOTS'].isin(shots))]
    
    # Group by MODEL, BACKBONE, and SHOTS
    grouped = filtered_df.groupby(['MODEL', 'BACKBONE', 'SHOTS'])[metric].agg(['mean', 'std']).reset_index()
    
    # Format values as mean±std
    grouped['value'] = grouped.apply(lambda x: f"{x['mean']:.2f} $\pm$ {x['std']:.2f}", axis=1)
    
    # Create a pivot table
    table = pd.pivot_table(grouped, 
                          values='value', 
                          index=['MODEL', 'BACKBONE'], 
                          columns=['SHOTS'], 
                          aggfunc='first').reset_index()
    
    # Convert to LaTeX
    latex = table.to_latex(index=False, escape=False)
    
    return latex

def highlight_best_results(df, dataset='artgraph', metric='ACC'):
    """Identify and highlight the best performing configurations."""
    # Filter by dataset
    filtered_df = df[df['DATASET'] == dataset]
    
    # Group by relevant columns and compute mean
    grouped = filtered_df.groupby(['MODEL', 'BACKBONE', 'SHOTS'])[metric].mean().reset_index()
    
    # Find best model-backbone combination for each shot setting
    best_results = grouped.loc[grouped.groupby('SHOTS')[metric].idxmax()]
    best_results = best_results.sort_values('SHOTS')
    
    print(f"Best {metric} Results for {dataset}:")
    for _, row in best_results.iterrows():
        print(f"Shots: {row['SHOTS']}, Model: {row['MODEL']}, Backbone: {row['BACKBONE']}, {metric}: {row[metric]:.2f}%")
    
    return best_results

def analyze_seed_variance(df, dataset='artgraph', metric='ACC'):
    """Analyze variance across different seeds."""
    # Filter by dataset
    filtered_df = df[df['DATASET'] == dataset]
    
    # Group by all factors except SEED
    grouped = filtered_df.groupby(['MODEL', 'BACKBONE', 'SHOTS'])[metric].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate coefficient of variation (CV)
    grouped['cv'] = grouped['std'] / grouped['mean'] * 100
    
    # Sort by CV to find most/least consistent configurations
    sorted_by_cv = grouped.sort_values('cv')
    
    print(f"Most consistent configurations ({metric}):")
    print(sorted_by_cv[['MODEL', 'BACKBONE', 'SHOTS', 'mean', 'std', 'cv']].head(5))
    
    print(f"\nLeast consistent configurations ({metric}):")
    print(sorted_by_cv[['MODEL', 'BACKBONE', 'SHOTS', 'mean', 'std', 'cv']].tail(5))
    
    return grouped

def plot_performance_scaling(df, models, backbones, dataset='artgraph', metric='ACC', save_path=None):
    """Plot how performance scales with the number of shots for each model-backbone combination."""
    # Filter data
    filtered_df = df[(df['DATASET'] == dataset) & 
                      (df['MODEL'].isin(models)) &
                      (df['BACKBONE'].isin(backbones))]
    
    # Group by MODEL, BACKBONE, and SHOTS
    grouped = filtered_df.groupby(['MODEL', 'BACKBONE', 'SHOTS'])[metric].mean().reset_index()
    
    # Create a unique identifier for each model-backbone combination
    grouped['model_backbone'] = grouped['MODEL'] + '_' + grouped['BACKBONE']
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    # Get unique model-backbone combinations
    model_backbones = grouped['model_backbone'].unique()
    
    # Define a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_backbones)))
    
    # Plot each model-backbone combination
    for i, mb in enumerate(model_backbones):
        mb_data = grouped[grouped['model_backbone'] == mb]
        plt.plot(mb_data['SHOTS'], mb_data[metric], marker='o', label=mb, color=colors[i], linewidth=2)
    
    plt.xlabel('Number of Shots')
    plt.ylabel(f'{metric} (%)')
    plt.title(f'Performance Scaling with Number of Shots')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Make x-axis logarithmic to better show differences at low shot counts
    plt.xscale('log')
    plt.xticks([1, 2, 4, 8, 16], labels=['1', '2', '4', '8', '16'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Load results
    results_df = load_results()
    
    # Compute and save summary
    summary_df = compute_summary(results_df)
    save_summary(summary_df)
    
    # Models and backbones to analyze
    models = ['TIMO', 'TIMO_S', 'Tip_Adapter', 'APE', 'GDA_CLIP']
    backbones = ['RN50', 'RN101', 'ViT-B/16', 'ViT-B/32', 'CustomRN50']
    
    # Plot model performance across shots for each backbone
    plot_model_performance(results_df, save_path='outputs/model_performance_all_backbones.png')
    
    # Plot backbone comparison for high-shot scenario
    plot_backbone_comparison(results_df, models, shots=16, save_path='outputs/backbone_comparison_16shots.png')
    
    # Create LaTeX table
    latex_table = create_latex_table(results_df, models, backbones, shots=[1, 4, 16])
    with open('outputs/results_latex_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Highlight best results
    best_results = highlight_best_results(results_df)
    
    # Analyze seed variance
    variance_results = analyze_seed_variance(results_df)
    
    # Plot performance scaling
    plot_performance_scaling(results_df, models, backbones, save_path='outputs/performance_scaling.png')
    
    print("\nAnalysis completed successfully!")