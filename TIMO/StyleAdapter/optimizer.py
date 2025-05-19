import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import clip
import optuna
from optuna.trial import TrialState
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from StyleAdapter import CLIPWithStyleAdapter, find_artgraph_path, ArtgraphDataset, train_epoch, validate_model


def define_model_and_train(trial, args, device, train_loader, val_loader, text_tokens):
    """
    Definisce e addestra il modello con i parametri suggeriti dal trial Optuna.
    Restituisce la metrica di valutazione (accuratezza di validazione).
    """
    # Parametri da ottimizzare
    fusion_bottleneck_dim = trial.suggest_categorical("fusion_bottleneck_dim", [64, 128, 256, 512])
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    dropout_rate_adapter = trial.suggest_float("dropout_rate_adapter", 0.0, 0.3, step=0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    
    # Creazione del modello
    try:
        model = CLIPWithStyleAdapter(
            clip_model_name=args.clip_model_name,
            fusion_bottleneck_dim=fusion_bottleneck_dim,
            gram_style_projection_dim=args.gram_style_projection_dim,
            layers_for_gram_rn50=args.layers_for_gram_rn50,
            dropout_rate=dropout_rate_adapter,
            use_layernorm_adapter=args.use_layernorm_adapter,
            device=device
        ).to(device)
    except Exception as e:
        print(f"Errore nella creazione del modello: {e}")
        raise optuna.exceptions.TrialPruned()

    # Raccolta parametri addestrabili
    trainable_params = []
    if model.gram_layer_projections:
        trainable_params.extend(list(model.gram_layer_projections.parameters()))
    if hasattr(model, 'fusion_adapter'):
        trainable_params.extend(list(model.fusion_adapter.parameters()))
    
    if not trainable_params:
        print("Nessun parametro addestrabile trovato nel modello.")
        raise optuna.exceptions.TrialPruned()
    
    # Optimizer e criterion
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    def lr_lambda(current_step): 
        return float(current_step) / float(max(1, args.warmup_steps)) if current_step < args.warmup_steps else 1.0
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_acc = 0.0
    no_improvement_count = 0
    
    for epoch in range(args.epochs):
        # Training
        model.gram_layer_projections.train() if model.gram_layer_projections else None
        model.fusion_adapter.train()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, text_tokens, criterion, 
            optimizer, scheduler, device, epoch, args.epochs
        )
        
        # Validation
        model.gram_layer_projections.eval() if model.gram_layer_projections else None
        model.fusion_adapter.eval()
        
        val_acc = validate_model(model, val_loader, text_tokens, device, epoch, args.epochs)
        
        # Reporting al trial Optuna
        trial.report(val_acc, epoch)
        
        # Early stopping basato sui risultati del trial
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Tracking per early stopping
        if val_acc > best_val_acc + args.early_stopping_min_delta:
            best_val_acc = val_acc
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= args.early_stopping_patience:
            print(f'Early stopping attivato per il trial dopo {epoch+1} epoche.')
            break
    
    # Cleanup hooks
    model._remove_gram_hooks()
    del model
    torch.cuda.empty_cache()
    
    return best_val_acc


def objective(trial, args, device, train_loader, val_loader, text_tokens, train_dataset):
    """
    Funzione obiettivo per l'ottimizzazione con Optuna.
    """
    try:
        val_acc = define_model_and_train(trial, args, device, train_loader, val_loader, text_tokens)
        
        # Se il trial è stato completato con successo, salva i parametri del miglior modello
        if val_acc > args.best_val_acc:
            args.best_val_acc = val_acc
            best_params = {
                'fusion_bottleneck_dim': trial.params['fusion_bottleneck_dim'],
                'lr': trial.params['lr'],
                'dropout_rate_adapter': trial.params['dropout_rate_adapter'],
                'weight_decay': trial.params['weight_decay'],
                'val_accuracy': val_acc
            }
            
            # Salva i migliori parametri in ogni caso
            with open(os.path.join(args.study_dir, "best_params.json"), 'w') as f:
                json.dump(best_params, f, indent=4)
                
            print(f"Nuovi migliori parametri trovati con accuratezza {val_acc:.4f}: {best_params}")
            
            # Opzionalmente, potresti voler addestrare un modello completo con i migliori parametri
            # e salvarlo qui. Evito per ora per risparmiare tempo computazionale durante la ricerca.
            
        return val_acc
        
    except Exception as e:
        print(f"Errore durante il trial: {e}")
        return float('-inf')  # In caso di errore, il trial è considerato fallito


def save_optuna_visualizations(study, study_dir):
    """
    Salva visualizzazioni utili dello studio Optuna.
    """
    # Crea directory per le visualizzazioni
    viz_dir = os.path.join(study_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Plot dell'importanza dei parametri
    try:
        param_importances = optuna.visualization.plot_param_importances(study)
        param_importances.write_image(os.path.join(viz_dir, "param_importances.png"))
    except Exception as e:
        print(f"Errore nel salvataggio del plot delle importanze dei parametri: {e}")
    
    # 2. Plot della storia dell'ottimizzazione
    try:
        optimization_history = optuna.visualization.plot_optimization_history(study)
        optimization_history.write_image(os.path.join(viz_dir, "optimization_history.png"))
    except Exception as e:
        print(f"Errore nel salvataggio del plot della storia di ottimizzazione: {e}")
        
    # 3. Plot delle relazioni tra parametri e valore obiettivo
    try:
        params = ["fusion_bottleneck_dim", "lr", "dropout_rate_adapter", "weight_decay"]
        for param in params:
            param_plot = optuna.visualization.plot_param_importances(study, target=lambda t: t.params[param])
            param_plot.write_image(os.path.join(viz_dir, f"{param}_importance.png"))
    except Exception as e:
        print(f"Errore nel salvataggio dei plot dei singoli parametri: {e}")
    
    # 4. Plot di scatter matrix
    try:
        fig = plt.figure(figsize=(16, 16))
        param_values = {param: [trial.params.get(param) for trial in study.trials if trial.state == TrialState.COMPLETE] 
                         for param in ["fusion_bottleneck_dim", "lr", "dropout_rate_adapter", "weight_decay"]}
        values = [trial.value for trial in study.trials if trial.state == TrialState.COMPLETE]
        
        sns.pairplot(pd.DataFrame({**param_values, "accuracy": values}))
        plt.savefig(os.path.join(viz_dir, "parameter_relationships.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Errore nel salvataggio dello scatter matrix: {e}")


def main():
    parser = argparse.ArgumentParser(description='Ottimizzazione iperparametri per StyleAdapter con Optuna')
    
    # Parametri per la ricerca degli iperparametri
    parser.add_argument('--n_trials', type=int, default=20, help='Numero di trial per la ricerca')
    parser.add_argument('--study_name', type=str, default=f"styleadapter_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--study_dir', type=str, default=None, help='Directory per salvare i risultati dello studio')
    parser.add_argument('--storage', type=str, default=None, help='URL di storage per Optuna (es. sqlite:///example.db)')
    parser.add_argument('--pruning', action='store_true', help='Abilita pruning automatico con Optuna')
    
    # Parametri del modello (fissi)
    parser.add_argument('--clip_model_name', type=str, default='RN50', help='Nome del modello CLIP di base')
    parser.add_argument('--gram_style_projection_dim', type=int, default=256, help='Dimensione proiezione delle feature Gram')
    parser.add_argument('--layers_for_gram_rn50', type=str, nargs='+', default=['layer2', 'layer3'], help='Layer da usare per Gram')
    parser.add_argument('--use_layernorm_adapter', type=bool, default=True, help='Uso di LayerNorm negli adapter')
    
    # Parametri di training
    parser.add_argument('--epochs', type=int, default=5, help='Numero massimo di epoche per trial')
    parser.add_argument('--batch_size', type=int, default=16, help='Dimensione del batch')
    parser.add_argument('--warmup_steps', type=int, default=200, help='Step di warmup per lo scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=2, help='Patience per early stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help='Minimo miglioramento per early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Seed per riproducibilità')
    
    args = parser.parse_args()
    
    # Directory per i risultati dello studio
    if args.study_dir is None:
        args.study_dir = f"optuna_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.study_dir, exist_ok=True)
    
    # Salva configurazione
    with open(os.path.join(args.study_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # Aggiungi tracciamento del miglior modello
    args.best_val_acc = 0.0
    
    # Setup ambiente
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"Utilizzo device: {device}")
    
    # Prepara dataset
    try:
        dataset_path = find_artgraph_path()
        # Usa il modello CLIP di base per il preprocessamento
        _, preprocess = clip.load(args.clip_model_name, device=device)
        
        train_dataset = ArtgraphDataset(dataset_path, split='train', transform=preprocess, seed=args.seed)
        val_dataset = ArtgraphDataset(dataset_path, split='val', transform=preprocess, seed=args.seed)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        
        # Tokenizzazione dei testi
        text_prompts = [f"a painting by {c.replace('-', ' ').replace('_', ' ')}" for c in train_dataset.classnames]
        text_tokens = clip.tokenize(text_prompts).to(device)
        
    except Exception as e:
        print(f"Errore nella preparazione del dataset: {e}")
        return
    
    # Configura storage Optuna
    storage = args.storage if args.storage else None
    
    # Configurazione pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30) if args.pruning else None
    
    # Crea studio Optuna
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage, 
        direction="maximize",  # Massimizzare l'accuratezza
        pruner=pruner,
        load_if_exists=True
    )
    
    # Esegui l'ottimizzazione
    try:
        study.optimize(
            lambda trial: objective(trial, args, device, train_loader, val_loader, text_tokens, train_dataset),
            n_trials=args.n_trials,
            timeout=None  # Nessun timeout
        )
    except KeyboardInterrupt:
        print("Ottimizzazione interrotta dall'utente.")
    
    # Salva risultati completi
    joblib.dump(study, os.path.join(args.study_dir, "study.pkl"))
    
    # Estrai e salva i migliori parametri
    best_trial = study.best_trial
    best_params = {
        "fusion_bottleneck_dim": best_trial.params["fusion_bottleneck_dim"],
        "lr": best_trial.params["lr"],
        "dropout_rate_adapter": best_trial.params["dropout_rate_adapter"],
        "weight_decay": best_trial.params["weight_decay"],
        "val_accuracy": best_trial.value
    }
    
    with open(os.path.join(args.study_dir, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Stampa risultati
    print("Ottimizzazione completata!")
    print(f"Miglior accuratezza: {best_trial.value:.4f}")
    print(f"Migliori parametri: {best_params}")
    print(f"Risultati salvati in: {args.study_dir}")
    
    # Genera e salva visualizzazioni
    try:
        import pandas as pd
        save_optuna_visualizations(study, args.study_dir)
    except ImportError:
        print("pandas o plotly non installati, impossibile generare visualizzazioni avanzate")
    except Exception as e:
        print(f"Errore nella generazione delle visualizzazioni: {e}")

if __name__ == "__main__":
    main()