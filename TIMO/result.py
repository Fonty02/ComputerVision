import pandas as pd
import os

if __name__ == "__main__":
    backbones = ["Custom_FSL1_RN50", "Custom_FSL4_RN50", "RN50"]
    
    # Leggi il CSV con i risultati
    results_path = "/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO/outputs/results_all.csv"
    df = pd.read_csv(results_path)
    
    # Filtra per le backbones specificate
    df_filtered = df[df['BACKBONE'].isin(backbones)]
    
    # Raggruppa per DATASET, MODEL, BACKBONE, SHOTS e calcola la media delle metriche
    grouped = df_filtered.groupby(['DATASET', 'MODEL', 'BACKBONE', 'SHOTS']).agg({
        'ACC': 'mean',
        'PRECISION_MACRO': 'mean', 
        'RECALL_MACRO': 'mean',
        'F1_MACRO': 'mean'
    }).reset_index()
    
    # Crea una tabella separata per ogni numero di shot
    unique_shots = sorted(grouped['SHOTS'].unique())
    
    for shot in unique_shots:
        # Filtra per il numero di shot specifico
        shot_data = grouped[grouped['SHOTS'] == shot].copy()
        shot_data = shot_data.drop('SHOTS', axis=1)  # Rimuovi la colonna SHOTS
        
        # Salva il risultato per ogni shot
        output_path = f"/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO/outputs/results_{shot}shot.csv"
        shot_data.to_csv(output_path, index=False)
        
        print(f"Risultati {shot}-shot salvati in: {output_path}")
        print(f"Configurazioni elaborate per {shot}-shot: {len(shot_data)}")
    
    print(f"\nTotale configurazioni elaborate: {len(grouped)}")
    print(f"Numeri di shot trovati: {unique_shots}")