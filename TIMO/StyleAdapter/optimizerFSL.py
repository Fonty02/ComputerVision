import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import joblib
import json
from datetime import datetime
import random
import sys

# Assumendo che StyleAdapterFLS.py sia nella stessa directory o PYTHONPATH
try:
    from StyleAdapterFLS import (
        CLIPWithStyleAdapter,
        MetaArtgraphDataset,
        meta_train_epoch,
        meta_validate_epoch,
        save_meta_model,
        find_artgraph_path
    )
    # Se CLIP è usato direttamente qui, altrimenti è gestito da StyleAdapterFLS
    import clip 
except ImportError as e:
    print(f"Errore nell'importare da StyleAdapterFLS: {e}")
    print("Assicurati che StyleAdapterFLS.py sia accessibile e che tutte le sue dipendenze (CLIP, higher) siano installate.")
    sys.exit(1)


def define_model_and_meta_train(trial, args, device, meta_train_loader, meta_val_loader):
    """
    Definisce, meta-addestra e meta-valida il modello con i parametri suggeriti da Optuna.
    Restituisce l'accuratezza di meta-validazione, la loss di meta-training e la loss di meta-validazione.
    """
    # Parametri da ottimizzare
    fusion_bottleneck_dim = trial.suggest_categorical("fusion_bottleneck_dim", [32, 64, 128, 256])
    meta_lr = trial.suggest_float("meta_lr", 1e-5, 1e-3, log=True)
    inner_lr = trial.suggest_float("inner_lr", 1e-3, 1e-1, log=True) # Inner loop LR
    dropout_rate_adapter = trial.suggest_float("dropout_rate_adapter", 0.0, 0.5, step=0.05)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    
    # Creazione del modello
    try:
        meta_model = CLIPWithStyleAdapter(
            clip_model_name=args.clip_model_name,
            fusion_bottleneck_dim=fusion_bottleneck_dim,
            gram_style_projection_dim=args.gram_style_projection_dim,
            layers_for_gram_rn50=args.layers_for_gram_rn50,
            dropout_rate=dropout_rate_adapter,
            use_layernorm_adapter=args.use_layernorm_adapter,
            device=device 
        ).to(device)
        # Registra gli hook Gram dopo aver spostato il modello sul device
        if hasattr(meta_model, '_register_gram_hooks'):
             meta_model._register_gram_hooks(meta_model.visual)

    except Exception as e:
        print(f"Errore nella creazione del modello durante il trial {trial.number}: {e}")
        if hasattr(meta_model, '_remove_gram_hooks'): meta_model._remove_gram_hooks()
        raise optuna.exceptions.TrialPruned()

    # Raccolta parametri addestrabili per il meta-optimizer
    meta_trainable_params = []
    if hasattr(meta_model, 'gram_layer_projections') and meta_model.gram_layer_projections:
        meta_trainable_params.extend(list(meta_model.gram_layer_projections.parameters()))
    if hasattr(meta_model, 'fusion_adapter') and meta_model.fusion_adapter is not None:
        meta_trainable_params.extend(list(meta_model.fusion_adapter.parameters()))
    
    if not meta_trainable_params:
        print(f"Nessun parametro addestrabile trovato nel modello per il trial {trial.number}.")
        if hasattr(meta_model, '_remove_gram_hooks'): meta_model._remove_gram_hooks()
        raise optuna.exceptions.TrialPruned()
    
    for p in meta_trainable_params:
        p.requires_grad = True

    meta_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, meta_trainable_params), lr=meta_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_meta_val_acc_trial = 0.0
    best_meta_train_loss_trial = float('inf')
    best_meta_val_loss_trial = float('inf')
    no_improvement_count = 0
    
    for epoch in range(args.epochs_optuna): # Usare epochs_optuna per i trial
        # Meta-Training
        # Assicurati che le parti del modello che devono essere addestrabili lo siano
        if hasattr(meta_model, 'gram_layer_projections'): meta_model.gram_layer_projections.train()
        if hasattr(meta_model, 'fusion_adapter'): meta_model.fusion_adapter.train()
        
        # La funzione meta_train_epoch dovrebbe gestire il loop interno con 'higher'
        current_meta_train_loss = meta_train_epoch(
            meta_model, meta_train_loader, meta_optimizer,
            inner_lr, args.inner_steps, criterion, device, epoch, args.epochs_optuna
        )
        
        # Meta-Validation
        # Il modello base (adapter) è in modalità valutazione per l'adattamento nel meta-val
        if hasattr(meta_model, 'gram_layer_projections'): meta_model.gram_layer_projections.eval()
        if hasattr(meta_model, 'fusion_adapter'): meta_model.fusion_adapter.eval()

        current_meta_val_loss, current_meta_val_acc = meta_validate_epoch(
            meta_model, meta_val_loader, criterion, device, epoch, args.epochs_optuna,
            inner_lr, args.inner_steps # Passa inner_lr e inner_steps
        )
        
        trial.report(current_meta_val_acc, epoch)
        
        if trial.should_prune():
            if hasattr(meta_model, '_remove_gram_hooks'): meta_model._remove_gram_hooks()
            del meta_model, meta_optimizer, criterion
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        
        if current_meta_val_acc > best_meta_val_acc_trial + args.early_stopping_min_delta:
            best_meta_val_acc_trial = current_meta_val_acc
            best_meta_train_loss_trial = current_meta_train_loss 
            best_meta_val_loss_trial = current_meta_val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= args.early_stopping_patience:
            print(f'Early stopping attivato per il trial {trial.number} dopo {epoch+1} epoche.')
            break
    
    if hasattr(meta_model, '_remove_gram_hooks'): meta_model._remove_gram_hooks()
    del meta_model, meta_optimizer, criterion
    torch.cuda.empty_cache()
    
    return best_meta_val_acc_trial, best_meta_train_loss_trial, best_meta_val_loss_trial


def objective(trial, args, device, meta_train_loader, meta_val_loader, meta_train_dataset_obj):
    try:
        meta_val_acc, meta_train_loss, meta_val_loss = define_model_and_meta_train(
            trial, args, device, meta_train_loader, meta_val_loader
        )
        
        # Salva i parametri del trial corrente se migliorano la best_meta_val_acc globale
        # args.best_meta_val_acc è condiviso tra i trial tramite l'oggetto args
        if meta_val_acc > args.best_meta_val_acc:
            args.best_meta_val_acc = meta_val_acc 
            current_best_params = {
                'fusion_bottleneck_dim': trial.params['fusion_bottleneck_dim'],
                'meta_lr': trial.params['meta_lr'],
                'inner_lr': trial.params['inner_lr'],
                'dropout_rate_adapter': trial.params['dropout_rate_adapter'],
                'weight_decay': trial.params['weight_decay'],
                'meta_val_accuracy': meta_val_acc,
                'meta_train_loss': meta_train_loss,
                'meta_val_loss': meta_val_loss,
                'trial_number': trial.number
            }
            with open(os.path.join(args.study_dir, "best_hyperparams_intermediate.json"), 'w') as f:
                json.dump(current_best_params, f, indent=4)
            print(f"Nuovi migliori iperparametri (intermedi) trovati dal trial {trial.number} con Meta-Val Acc: {meta_val_acc:.4f}, Meta-Train Loss: {meta_train_loss:.4f}, Meta-Val Loss: {meta_val_loss:.4f}")

        return meta_val_acc 
        
    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} potato.")
        raise
    except Exception as e:
        print(f"Errore imprevisto nel trial {trial.number}: {e}")
        # Potrebbe essere utile loggare l'eccezione completa per debug
        import traceback
        traceback.print_exc()
        # Per Optuna, è meglio rilanciare come TrialPruned o un'eccezione che Optuna gestisce
        # o semplicemente lasciare che fallisca se è un errore di configurazione grave.
        # Se l'errore è recuperabile o specifico del trial, TrialPruned è appropriato.
        raise optuna.exceptions.TrialPruned()


def main():
    parser = argparse.ArgumentParser(description='Ottimizzazione iperparametri per StyleAdapterFSL con Optuna')
    
    # Optuna
    parser.add_argument('--n_trials', type=int, default=25, help='Numero di trial per la ricerca Optuna')
    parser.add_argument('--study_name', type=str, default=f"styleadapter_fsl_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--study_dir', type=str, default=None, help='Directory per salvare i risultati dello studio')
    parser.add_argument('--storage', type=str, default=None, help='URL di storage per Optuna (es. sqlite:///study_fsl.db)')
    parser.add_argument('--pruning', action='store_true', help='Abilita pruning automatico con Optuna MedianPruner')
    
    # Modello CLIP e Adapter (fissi durante HPO, ma configurabili)
    parser.add_argument('--clip_model_name', type=str, default='RN50', help='Nome del modello CLIP (es. RN50, ViT-B/32)')
    parser.add_argument('--gram_style_projection_dim', type=int, default=256, help='Dimensione totale Gram features proiettate')
    parser.add_argument('--layers_for_gram_rn50', type=str, nargs='+', default=['layer2', 'layer3'], help='Layer RN50 per Gram')
    parser.add_argument('--use_layernorm_adapter', type=lambda x: (str(x).lower() == 'true'), default=True, help="Usa LayerNorm nel fusion adapter")

    # Dataset e Meta-Learning (fissi durante HPO, ma configurabili)
    parser.add_argument('--n_way', type=int, default=5, help="N-way per task FSL")
    parser.add_argument('--k_shot_support', type=int, default=1, help="K-shot (support set) per task FSL")
    parser.add_argument('--k_shot_query', type=int, default=5, help="Numero di campioni query per classe per task FSL")
    parser.add_argument('--num_tasks_per_epoch', type=int, default=100, help="Numero di task (episodi) per meta-epoca (per Optuna e training finale)")
    
    # Ottimizzazione Meta (alcuni ottimizzati da Optuna, altri fissi)
    parser.add_argument('--epochs_optuna', type=int, default=10, help="Numero di meta-epoche per ogni trial Optuna")
    parser.add_argument('--epochs_final', type=int, default=30, help="Numero di meta-epoche per il training finale")
    # meta_lr, inner_lr saranno suggeriti da Optuna
    parser.add_argument('--inner_steps', type=int, default=5, help="Numero di step di adattamento nell'inner loop (fisso)")
    # weight_decay sarà suggerito da Optuna
    
    # Altri
    parser.add_argument('--batch_size_dataloader', type=int, default=1, help="Batch size per DataLoader (1 per task-based meta-learning)")
    parser.add_argument('--num_workers_dataloader', type=int, default=0, help="Numero di workers per DataLoader (0 per debug su Windows)")
    parser.add_argument('--output_model_dir', type=str, default='trained_meta_models', help="Directory per salvare il miglior meta-modello finale")
    parser.add_argument('--seed', type=int, default=42, help="Seed per la riproducibilità")
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Patience per early stopping nei trial e training finale')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help='Minimo miglioramento per early stopping')
    
    args = parser.parse_args()
    
    if args.study_dir is None:
        args.study_dir = os.path.join("studies", args.study_name) 
    os.makedirs(args.study_dir, exist_ok=True)
    os.makedirs(args.output_model_dir, exist_ok=True)
    
    with open(os.path.join(args.study_dir, "config_args_hpo.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    args.best_meta_val_acc = 0.0 # Per tracciare la migliore acc globale tra i trial
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.benchmark = False # Per riproducibilità, può rallentare
        # torch.backends.cudnn.deterministic = True
    print(f"Utilizzo device: {device}")
    
    try:
        artgraph_path = find_artgraph_path()
        # Il preprocess viene caricato qui solo per passarlo al dataset, 
        # il modello CLIP effettivo è istanziato dentro CLIPWithStyleAdapter
        _, preprocess = clip.load(args.clip_model_name, device="cpu") # Preprocess su CPU
        
        meta_train_dataset = MetaArtgraphDataset(
            root_dir=artgraph_path, num_tasks=args.num_tasks_per_epoch, # num_tasks è per epoca
            n_way=args.n_way, k_shot_support=args.k_shot_support, k_shot_query=args.k_shot_query,
            transform=preprocess, split='meta_train', seed=args.seed
        )
        meta_val_dataset = MetaArtgraphDataset(
            root_dir=artgraph_path, num_tasks=args.num_tasks_per_epoch, # num_tasks per valutazione
            n_way=args.n_way, k_shot_support=args.k_shot_support, k_shot_query=args.k_shot_query,
            transform=preprocess, split='meta_val', seed=args.seed # Seed diverso per val o stessa divisione? Usiamo lo stesso per ora.
        )
        
        # Collate_fn per gestire l'output del dataset (che è già un task)
        collate_fn_task = lambda x: x[0] if isinstance(x, list) and len(x)==1 and isinstance(x[0], tuple) else x

        meta_train_loader = torch.utils.data.DataLoader(
            meta_train_dataset, batch_size=args.batch_size_dataloader, shuffle=True, 
            num_workers=args.num_workers_dataloader, collate_fn=collate_fn_task
        )
        meta_val_loader = torch.utils.data.DataLoader(
            meta_val_dataset, batch_size=args.batch_size_dataloader, shuffle=False, 
            num_workers=args.num_workers_dataloader, collate_fn=collate_fn_task
        )
    except FileNotFoundError as e:
        print(f"Errore nella preparazione del dataset: {e}")
        return
    except Exception as e:
        print(f"Errore generico nella preparazione del dataset o caricamento CLIP: {e}")
        import traceback
        traceback.print_exc()
        return

    storage_path = args.storage if args.storage else f"sqlite:///{os.path.join(args.study_dir, 'optuna_fsl_study.db')}"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(1,args.n_trials // 5), n_warmup_steps=max(1,args.epochs_optuna // 4), interval_steps=1) if args.pruning else None
    
    study = optuna.create_study(
        study_name=args.study_name, storage=storage_path, 
        direction="maximize", pruner=pruner, load_if_exists=True # load_if_exists=True per riprendere
    )
    
    try:
        study.optimize(
            lambda trial: objective(trial, args, device, meta_train_loader, meta_val_loader, meta_train_dataset),
            n_trials=args.n_trials, timeout=None # timeout in secondi se necessario
        )
    except KeyboardInterrupt:
        print("Ottimizzazione Optuna interrotta dall'utente.")
    except Exception as e:
        print(f"Errore durante study.optimize: {e}")
        import traceback
        traceback.print_exc()
    
    joblib.dump(study, os.path.join(args.study_dir, "study_fsl.pkl"))
    
    if not study.trials:
        print("Nessun trial completato. Impossibile determinare i migliori parametri o addestrare il modello finale.")
        return

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("Nessun trial completato con successo. Impossibile determinare i migliori parametri o addestrare il modello finale.")
        return
        
    best_trial = study.best_trial
    
    if best_trial is None: # Può accadere se tutti i trial falliscono o vengono potati prima di riportare un valore
        print("Nessun best trial trovato (possibilmente tutti falliti o potati). Impossibile addestrare il modello finale.")
        # Prova a prendere il migliore dai completed trials se best_trial è None ma ci sono trial completati
        if completed_trials:
            best_trial = max(completed_trials, key=lambda t: t.value)
            if best_trial.value is None: # Ancora nessun valore
                 print("Anche il miglior trial tra quelli completati non ha un valore valido.")
                 return
            print(f"Best_trial era None, ma è stato selezionato il migliore tra i {len(completed_trials)} completati: Trial {best_trial.number}")
        else:
            return


    best_hyperparams = best_trial.params
    best_value_hpo = best_trial.value # Migliore meta_val_acc dalla ricerca HPO

    print("\nOttimizzazione Optuna completata!")
    print(f"Migliore Meta-Val Acc durante la ricerca HPO: {best_value_hpo:.4f}")
    print(f"Migliori iperparametri trovati: {best_hyperparams}")
    
    final_best_params_path = os.path.join(args.study_dir, "best_hyperparams_fsl_final.json")
    with open(final_best_params_path, 'w') as f:
        json.dump({**best_hyperparams, "meta_val_accuracy_from_hpo": best_value_hpo, "best_trial_number": best_trial.number}, f, indent=4)
    print(f"Migliori iperparametri finali salvati in: {final_best_params_path}")

    print("\nInizio addestramento del modello finale con i migliori iperparametri...")
    
    final_meta_model = CLIPWithStyleAdapter(
        clip_model_name=args.clip_model_name,
        fusion_bottleneck_dim=best_hyperparams["fusion_bottleneck_dim"],
        gram_style_projection_dim=args.gram_style_projection_dim,
        layers_for_gram_rn50=args.layers_for_gram_rn50,
        dropout_rate=best_hyperparams["dropout_rate_adapter"],
        use_layernorm_adapter=args.use_layernorm_adapter,
        device=device
    ).to(device)
    if hasattr(final_meta_model, '_register_gram_hooks'):
        final_meta_model._register_gram_hooks(final_meta_model.visual)

    final_meta_trainable_params = []
    if hasattr(final_meta_model, 'gram_layer_projections') and final_meta_model.gram_layer_projections:
        final_meta_trainable_params.extend(list(final_meta_model.gram_layer_projections.parameters()))
    if hasattr(final_meta_model, 'fusion_adapter') and final_meta_model.fusion_adapter is not None:
        final_meta_trainable_params.extend(list(final_meta_model.fusion_adapter.parameters()))

    if not final_meta_trainable_params:
        print("Nessun parametro addestrabile nel modello finale. Impossibile procedere.")
        if hasattr(final_meta_model, '_remove_gram_hooks'): final_meta_model._remove_gram_hooks()
        return

    for p in final_meta_trainable_params:
        p.requires_grad = True

    optimizer_final = optim.AdamW(
        filter(lambda p: p.requires_grad, final_meta_trainable_params), 
        lr=best_hyperparams["meta_lr"], 
        weight_decay=best_hyperparams["weight_decay"]
    )
    criterion_final = nn.CrossEntropyLoss()

    best_meta_val_acc_final = 0.0
    # best_meta_train_loss_final = float('inf') # Non tracciato esplicitamente per il salvataggio del modello qui
    # best_meta_val_loss_final = float('inf')   # Non tracciato esplicitamente per il salvataggio del modello qui
    no_improvement_count_final = 0
    final_model_actual_epochs = 0

    for epoch in range(args.epochs_final): # Usare epochs_final
        final_model_actual_epochs = epoch + 1
        if hasattr(final_meta_model, 'gram_layer_projections'): final_meta_model.gram_layer_projections.train()
        if hasattr(final_meta_model, 'fusion_adapter'): final_meta_model.fusion_adapter.train()
        
        train_loss_f = meta_train_epoch(
            final_meta_model, meta_train_loader, optimizer_final,
            best_hyperparams["inner_lr"], args.inner_steps, criterion_final, device, 
            epoch, args.epochs_final
        )
        
        if hasattr(final_meta_model, 'gram_layer_projections'): final_meta_model.gram_layer_projections.eval()
        if hasattr(final_meta_model, 'fusion_adapter'): final_meta_model.fusion_adapter.eval()
        
        val_loss_f, val_acc_f = meta_validate_epoch(
            final_meta_model, meta_val_loader, criterion_final, device, epoch, args.epochs_final,
            best_hyperparams["inner_lr"], args.inner_steps
        )
        
        print(f"Addestramento finale - Epoch {epoch+1}/{args.epochs_final} -> Meta-Train Loss: {train_loss_f:.4f}, Meta-Val Loss: {val_loss_f:.4f}, Meta-Val Acc: {val_acc_f:.4f}")

        if val_acc_f > best_meta_val_acc_final + args.early_stopping_min_delta:
            best_meta_val_acc_final = val_acc_f
            no_improvement_count_final = 0
            
            # Sovrascrive il modello migliore se ne trova uno nuovo
            output_model_filename = os.path.join(args.output_model_dir, f"best_meta_style_adapter_fsl.pt")
            # Passa una porzione dei classnames del meta-train dataset per riferimento
            meta_train_classnames_sample = meta_train_dataset.classnames[:20] if hasattr(meta_train_dataset, 'classnames') else None
            
            # Aggiorna args con i migliori iperparametri per il salvataggio
            args_for_saving = vars(args).copy()
            args_for_saving.update(best_hyperparams) # Aggiunge/sovrascrive con i migliori iperparametri di Optuna

            save_meta_model(final_meta_model, epoch, best_meta_val_acc_final, argparse.Namespace(**args_for_saving), classnames_meta_train=meta_train_classnames_sample)
            print(f"Modello finale migliorato salvato in: {output_model_filename} (Meta-Val Acc: {best_meta_val_acc_final:.4f})")
        else:
            no_improvement_count_final += 1
        
        if no_improvement_count_final >= args.early_stopping_patience:
            print(f'Early stopping durante l\'addestramento finale all\'epoca {epoch+1}.')
            break
            
    print(f"\nAddestramento finale completato dopo {final_model_actual_epochs} epoche.")
    print(f"Migliore Meta-Val Acc del modello finale addestrato: {best_meta_val_acc_final:.4f}")

    if hasattr(final_meta_model, '_remove_gram_hooks'): final_meta_model._remove_gram_hooks()
    del final_meta_model, optimizer_final, criterion_final
    torch.cuda.empty_cache()

    print(f"Risultati completi dello studio Optuna salvati in: {args.study_dir}")
    print(f"Miglior modello meta-addestrato (se trovato) salvato in: {args.output_model_dir}")

if __name__ == "__main__":
    main()