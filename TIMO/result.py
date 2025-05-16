import pandas as pd
df= pd.read_csv('outputs/results_all.csv')

averaged_results = df.groupby(['BACKBONE', 'SHOTS', 'MODEL'])['AVG ACC'].mean().reset_index()

# Ora potremmo fare un pivot per avere i modelli come colonne
pivoted_results = averaged_results.pivot_table(
index=['BACKBONE', 'SHOTS'],
columns='MODEL',
values='AVG ACC'
).reset_index()
print(pivoted_results.to_string()) # Per visualizzare e poi trasferire in LaTeX