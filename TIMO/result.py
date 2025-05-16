import pandas as pd

# Carica il dataset
df = pd.read_csv('outputs/results_all.csv')

# Raggruppa e calcola la media per ogni metrica
grouped_results = df.groupby(['BACKBONE', 'SHOTS', 'MODEL']).agg({
    'ACC': 'mean',
    'PRECISION_MACRO': 'mean',
    'RECALL_MACRO': 'mean',
    'F1_MACRO': 'mean'
}).reset_index()

# Crea pivot tables per ogni metrica
metrics = ['ACC', 'PRECISION_MACRO', 'RECALL_MACRO', 'F1_MACRO']
result_dfs = {}

for metric in metrics:
    pivoted = grouped_results.pivot_table(
        index=['BACKBONE', 'SHOTS'],
        columns='MODEL',
        values=metric
    ).reset_index()
    result_dfs[metric] = pivoted

# Salva i risultati in un unico file CSV con sezioni separate per ogni metrica
with open('outputs/results_summary.csv', 'w') as f:
    # Prima sezione: Accuracy
    f.write("ACCURACY\n")
    result_dfs['ACC'].to_csv(f, index=False)
    
    # Seconda sezione: Precision
    f.write("\nPRECISION_MACRO\n")
    result_dfs['PRECISION_MACRO'].to_csv(f, index=False)
    
    # Terza sezione: Recall
    f.write("\nRECALL_MACRO\n")
    result_dfs['RECALL_MACRO'].to_csv(f, index=False)
    
    # Quarta sezione: F1
    f.write("\nF1_MACRO\n")
    result_dfs['F1_MACRO'].to_csv(f, index=False)

print("Risultati salvati in 'outputs/results_summary.csv'")