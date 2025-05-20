import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import clip
import optuna
# from optuna.trial import TrialState # Rimosso
import joblib
import json
from datetime import datetime
# import matplotlib.pyplot as plt # Rimosso
# import seaborn as sns # Rimosso
# import pandas as pd # Rimosso
from StyleAdapter import CLIPWithStyleAdapter, find_artgraph_path, ArtgraphDataset, train_epoch, validate_model


def define_model_and_train(trial, args, device, train_loader, val_loader, text_tokens):
    """
    Definisce e addestra il modello con i parametri suggeriti dal trial Optuna.
    Restituisce l'accuratezza di validazione, la loss di training e la loss di validazione.
    """
    # Parametri da ottimizzare
    fusion_bottleneck_dim = trial.suggest_categorical("fusion_bottleneck_dim", [32, 64, 128, 256]) 
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True) 
    dropout_rate_adapter = trial.suggest_float("dropout_rate_adapter", 0.0, 0.5, step=0.05) 
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True) 
    
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
        print(f"Errore nella creazione del modello durante il trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned()

    # Raccolta parametri addestrabili
    trainable_params = []
    if model.gram_layer_projections:
        trainable_params.extend(list(model.gram_layer_projections.parameters()))
    if hasattr(model, 'fusion_adapter') and model.fusion_adapter is not None:
        trainable_params.extend(list(model.fusion_adapter.parameters()))
    
    if not trainable_params:
        print(f"Nessun parametro addestrabile trovato nel modello per il trial {trial.number}.")
        raise optuna.exceptions.TrialPruned()
    
    # Optimizer e criterion
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    def lr_lambda(current_step): 
        if args.warmup_steps > 0:
            return float(current_step) / float(max(1, args.warmup_steps)) if current_step < args.warmup_steps else 1.0
        return 1.0 
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_acc_trial = 0.0
    best_train_loss_trial = float('inf') # Aggiunto
    best_val_loss_trial = float('inf')   # Aggiunto
    no_improvement_count = 0
    
    for epoch in range(args.epochs):
        # Training
        if model.gram_layer_projections: model.gram_layer_projections.train()
        if hasattr(model, 'fusion_adapter') and model.fusion_adapter is not None: model.fusion_adapter.train()
        
        current_train_loss, current_train_acc = train_epoch( # Rinominato per chiarezza
            model, train_loader, text_tokens, criterion, 
            optimizer, scheduler, device, epoch, args.epochs
        )
        
        # Validation
        if model.gram_layer_projections: model.gram_layer_projections.eval()
        if hasattr(model, 'fusion_adapter') and model.fusion_adapter is not None: model.fusion_adapter.eval()
        
        current_val_loss, current_val_acc = validate_model(model, val_loader, text_tokens, criterion, device, epoch, args.epochs) # Rinominato
        
        trial.report(current_val_acc, epoch)
        
        if trial.should_prune():
            model._remove_gram_hooks()
            del model, optimizer, criterion, scheduler
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        
        if current_val_acc > best_val_acc_trial + args.early_stopping_min_delta:
            best_val_acc_trial = current_val_acc
            best_train_loss_trial = current_train_loss # Aggiorna best train loss
            best_val_loss_trial = current_val_loss     # Aggiorna best val loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= args.early_stopping_patience:
            print(f'Early stopping attivato per il trial {trial.number} dopo {epoch+1} epoche.')
            break
    
    model._remove_gram_hooks()
    del model, optimizer, criterion, scheduler
    torch.cuda.empty_cache()
    
    return best_val_acc_trial, best_train_loss_trial, best_val_loss_trial # Restituisce anche le loss


def objective(trial, args, device, train_loader, val_loader, text_tokens, train_dataset):
    """
    Funzione obiettivo per l'ottimizzazione con Optuna.
    """
    try:
        val_acc, train_loss, val_loss = define_model_and_train(trial, args, device, train_loader, val_loader, text_tokens) # Riceve anche le loss
        
        if val_acc > args.best_val_acc:
            args.best_val_acc = val_acc 
            current_best_params = {
                'fusion_bottleneck_dim': trial.params['fusion_bottleneck_dim'],
                'lr': trial.params['lr'],
                'dropout_rate_adapter': trial.params['dropout_rate_adapter'],
                'weight_decay': trial.params['weight_decay'],
                'val_accuracy': val_acc,
                'train_loss': train_loss, # Aggiunto train_loss
                'val_loss': val_loss,     # Aggiunto val_loss
                'trial_number': trial.number
            }
            with open(os.path.join(args.study_dir, "best_hyperparams_intermediate.json"), 'w') as f:
                json.dump(current_best_params, f, indent=4)
            print(f"Nuovi migliori iperparametri (intermedi) trovati dal trial {trial.number} con accuratezza {val_acc:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return val_acc # Optuna ottimizza ancora solo su val_acc
        
    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} potato.")
        raise


def main():
    parser = argparse.ArgumentParser(description='Ottimizzazione iperparametri per StyleAdapter con Optuna')
    
    # Parametri per la ricerca degli iperparametri
    parser.add_argument('--n_trials', type=int, default=10, help='Numero di trial per la ricerca')
    parser.add_argument('--study_name', type=str, default=f"styleadapter_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--study_dir', type=str, default=None, help='Directory per salvare i risultati dello studio')
    parser.add_argument('--storage', type=str, default=None, help='URL di storage per Optuna (es. sqlite:///example.db)')
    parser.add_argument('--pruning', action='store_true', help='Abilita pruning automatico con Optuna')
    
    # Parametri del modello (fissi)
    parser.add_argument('--clip_model_name', type=str, default='RN50', help='Nome del modello CLIP di base')
    parser.add_argument('--gram_style_projection_dim', type=int, default=256, help='Dimensione proiezione delle feature Gram') # Modificato default
    parser.add_argument('--layers_for_gram_rn50', type=str, nargs='+', default=['layer2', 'layer3'], help='Layer da usare per Gram')
    parser.add_argument('--use_layernorm_adapter', type=bool, default=True, help='Uso di LayerNorm negli adapter')
    
    # Parametri di training
    parser.add_argument('--epochs', type=int, default=50, help='Numero massimo di epoche per trial e per il training finale') # Modificato default
    parser.add_argument('--batch_size', type=int, default=512, help='Dimensione del batch') # Modificato default
    parser.add_argument('--warmup_steps', type=int, default=200, help='Step di warmup per lo scheduler') 
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Patience per early stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help='Minimo miglioramento per early stopping') 
    parser.add_argument('--seed', type=int, default=42, help='Seed per riproducibilità')
    
    args = parser.parse_args()
    
    if args.study_dir is None:
        args.study_dir = args.study_name 
    os.makedirs(args.study_dir, exist_ok=True)
    
    with open(os.path.join(args.study_dir, "config_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    args.best_val_acc = 0.0 
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"Utilizzo device: {device}")
    
    try:
        dataset_path = find_artgraph_path()
        _, preprocess = clip.load(args.clip_model_name, device="cpu") 
        
        train_dataset = ArtgraphDataset(dataset_path, split='train', transform=preprocess, seed=args.seed)
        val_dataset = ArtgraphDataset(dataset_path, split='val', transform=preprocess, seed=args.seed)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=2, pin_memory=True if device=="cuda" else False 
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=2, pin_memory=True if device=="cuda" else False 
        )
        text_prompts = [f"a painting by {c.replace('-', ' ').replace('_', ' ')}" for c in train_dataset.classnames]
        text_tokens = clip.tokenize(text_prompts).to(device)
    except Exception as e:
        print(f"Errore nella preparazione del dataset: {e}")
        return
    
    storage = args.storage if args.storage else None
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1, interval_steps=1) if args.pruning else None 
    
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, 
        direction="maximize", pruner=pruner, load_if_exists=True
    )
    
    try:
        study.optimize(
            lambda trial: objective(trial, args, device, train_loader, val_loader, text_tokens, train_dataset),
            n_trials=args.n_trials, timeout=None
        )
    except KeyboardInterrupt:
        print("Ottimizzazione interrotta dall'utente.")
    except Exception as e:
        print(f"Errore durante study.optimize: {e}")
    
    joblib.dump(study, os.path.join(args.study_dir, "study.pkl"))
    
    if not study.trials:
        print("Nessun trial completato. Impossibile determinare i migliori parametri o addestrare il modello finale.")
        return

    # completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE] # Rimosso optuna.trial.TrialState
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE] # Assicurati che optuna.trial.TrialState sia disponibile
    if not completed_trials:
        print("Nessun trial completato con successo. Impossibile determinare i migliori parametri o addestrare il modello finale.")
        return
        
    best_trial = study.best_trial
    
    if best_trial is None:
        print("Nessun best trial trovato (possibilmente tutti falliti o potati). Impossibile addestrare il modello finale.")
        return

    best_hyperparams = best_trial.params
    best_value = best_trial.value

    print("\nOttimizzazione completata!")
    print(f"Miglior accuratezza di validazione durante la ricerca: {best_value:.4f}")
    print(f"Migliori iperparametri trovati: {best_hyperparams}")
    
    final_best_params_path = os.path.join(args.study_dir, "best_hyperparams_final.json")
    with open(final_best_params_path, 'w') as f:
        json.dump({**best_hyperparams, "val_accuracy_from_search": best_value, "best_trial_number": best_trial.number}, f, indent=4)
    print(f"Migliori iperparametri finali salvati in: {final_best_params_path}")

    print("\nInizio addestramento del modello finale con i migliori iperparametri...")
    
    final_model = CLIPWithStyleAdapter(
        clip_model_name=args.clip_model_name,
        fusion_bottleneck_dim=best_hyperparams["fusion_bottleneck_dim"],
        gram_style_projection_dim=args.gram_style_projection_dim,
        layers_for_gram_rn50=args.layers_for_gram_rn50,
        dropout_rate=best_hyperparams["dropout_rate_adapter"],
        use_layernorm_adapter=args.use_layernorm_adapter,
        device=device
    ).to(device)

    trainable_params_final = []
    if final_model.gram_layer_projections:
        trainable_params_final.extend(list(final_model.gram_layer_projections.parameters()))
    if hasattr(final_model, 'fusion_adapter') and final_model.fusion_adapter is not None:
        trainable_params_final.extend(list(final_model.fusion_adapter.parameters()))

    if not trainable_params_final:
        print("Nessun parametro addestrabile nel modello finale. Impossibile procedere.")
        return

    optimizer_final = optim.AdamW(trainable_params_final, lr=best_hyperparams["lr"], weight_decay=best_hyperparams["weight_decay"])
    criterion_final = nn.CrossEntropyLoss()
    
    def lr_lambda_final(current_step):
        if args.warmup_steps > 0:
            return float(current_step) / float(max(1, args.warmup_steps)) if current_step < args.warmup_steps else 1.0
        return 1.0
    scheduler_final = LambdaLR(optimizer_final, lr_lambda_final)

    best_val_acc_final_model = 0.0
    best_train_loss_final_model = float('inf')
    best_val_loss_final_model = float('inf')
    no_improvement_count_final = 0
    final_model_actual_epochs = 0

    for epoch in range(args.epochs):
        final_model_actual_epochs = epoch + 1
        if final_model.gram_layer_projections: final_model.gram_layer_projections.train()
        if hasattr(final_model, 'fusion_adapter') and final_model.fusion_adapter is not None: final_model.fusion_adapter.train()
        
        train_loss_f, train_acc_f = train_epoch(
            final_model, train_loader, text_tokens, criterion_final,
            optimizer_final, scheduler_final, device, epoch, args.epochs
        )
        
        if final_model.gram_layer_projections: final_model.gram_layer_projections.eval()
        if hasattr(final_model, 'fusion_adapter') and final_model.fusion_adapter is not None: final_model.fusion_adapter.eval()
        
        # Assumendo che validate_model ora restituisca (val_loss, val_acc) e accetti criterion
        val_loss_f, val_acc_f = validate_model(final_model, val_loader, text_tokens, criterion_final, device, epoch, args.epochs)
        
        print(f"Addestramento finale - Epoch {epoch+1}/{args.epochs} -> Train Loss: {train_loss_f:.4f}, Train Acc: {train_acc_f:.4f}, Val Loss: {val_loss_f:.4f}, Val Acc: {val_acc_f:.4f}")

        if val_acc_f > best_val_acc_final_model + args.early_stopping_min_delta:
            best_val_acc_final_model = val_acc_f
            best_train_loss_final_model = train_loss_f
            best_val_loss_final_model = val_loss_f
            no_improvement_count_final = 0
            
            output_model_filename = os.path.join(args.study_dir, f"best_trained_model_e{epoch+1}_acc{best_val_acc_final_model:.4f}.pt")
            save_dict = {
                'epoch': epoch + 1,
                'train_loss': best_train_loss_final_model,
                'val_loss': best_val_loss_final_model,
                'val_acc': best_val_acc_final_model,
                'args_script': vars(args),
                'hyperparams_optuna': best_hyperparams,
                'classnames': train_dataset.classnames,
                'gram_projections_state_dict': final_model.gram_layer_projections.state_dict() if final_model.gram_layer_projections else None,
                'fusion_adapter_state_dict': final_model.fusion_adapter.state_dict() if hasattr(final_model, 'fusion_adapter') and final_model.fusion_adapter is not None else None
            }
            torch.save(save_dict, output_model_filename)
            print(f"Modello finale migliorato salvato in: {output_model_filename}")

        else:
            no_improvement_count_final += 1
        
        if no_improvement_count_final >= args.early_stopping_patience:
            print(f'Early stopping durante l\'addestramento finale all\'epoca {epoch+1}.')
            break
            
    print(f"\nAddestramento finale completato dopo {final_model_actual_epochs} epoche.")
    print(f"Migliore accuratezza di validazione del modello finale addestrato: {best_val_acc_final_model:.4f}")
    print(f"Corrispondente Train Loss: {best_train_loss_final_model:.4f}, Val Loss: {best_val_loss_final_model:.4f}")


    final_model._remove_gram_hooks()
    del final_model, optimizer_final, criterion_final, scheduler_final
    torch.cuda.empty_cache()

    print(f"Risultati completi dello studio e modello salvati in: {args.study_dir}")

if __name__ == "__main__":
    main()