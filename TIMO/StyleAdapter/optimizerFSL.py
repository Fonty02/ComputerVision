import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import clip
import optuna
from optuna.trial import TrialState
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Aggiunto per save_optuna_visualizations
from tqdm import tqdm
from typing import List, Tuple, Dict, Iterator
import random
# Importa dal file StyleAdapter.py
from StyleAdapterFLS import CLIPWithStyleAdapter, find_artgraph_path, ArtgraphDataset

# --- Parametri Fissi per Meta-Learning ---
# Questi potrebbero essere resi argomenti del parser o iperparametri di Optuna
N_WAY = 10  # Numero di classi per task
K_SHOT = 1  # Numero di esempi di supporto per classe per N_WAY > 1
Q_QUERIES = 3 # Numero di esempi di query per classe



# --- Sampler Episodico ---
class EpisodicBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset_labels_by_class: Dict[int, List[int]],
                 n_way: int, k_shot: int, q_queries: int, num_episodes: int):
        """
        Args:
            dataset_labels_by_class: Dizionario {class_idx_globale: [idx_campione_1, idx_campione_2, ...]}
                                     gli idx_campione sono indici nel dataset piatto.
            n_way: Numero di classi per episodio.
            k_shot: Numero di campioni di supporto per classe.
            q_queries: Numero di campioni di query per classe.
            num_episodes: Numero di episodi da generare per epoca.
        """
        super().__init__(None) # data_source non è usato direttamente
        self.dataset_labels_by_class = dataset_labels_by_class
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.num_episodes = num_episodes

        self.available_classes = list(self.dataset_labels_by_class.keys())
        
        # Filtra classi con campioni insufficienti
        self.valid_classes = [
            cls_idx for cls_idx, samples in self.dataset_labels_by_class.items()
            if len(samples) >= self.k_shot + self.q_queries
        ]
        if len(self.valid_classes) < self.n_way:
            raise ValueError(
                f"Non ci sono abbastanza classi ({len(self.valid_classes)}) con campioni sufficienti "
                f"({self.k_shot + self.q_queries} richiesti) per formare episodi {self.n_way}-way. "
                "Riduci N_WAY, K_SHOT, Q_QUERIES o aumenta il dataset."
            )
        print(f"EpisodicSampler: {len(self.valid_classes)} classi valide per episodi {self.n_way}-way {self.k_shot}-shot {self.q_queries}-query.")


    def __len__(self) -> int:
        return self.num_episodes

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_episodes):
            # Campiona N classi senza rimpiazzo dalle classi valide
            selected_class_indices_global = random.sample(self.valid_classes, self.n_way)
            
            episode_sample_indices: List[int] = []
            # Le etichette relative al task (0...N-1) sono implicite nell'ordine

            for class_idx_global in selected_class_indices_global:
                samples_for_class = self.dataset_labels_by_class[class_idx_global]
                # Campiona K+Q campioni senza rimpiazzo da questa classe
                selected_samples_for_class_indices = random.sample(
                    samples_for_class, self.k_shot + self.q_queries
                )
                episode_sample_indices.extend(selected_samples_for_class_indices)
            
            yield episode_sample_indices


# --- Collate Function Episodica ---
def episodic_collate_fn(batch_samples: List[Tuple[torch.Tensor, int]],
                        n_way: int, k_shot: int, q_queries: int
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Organizza un batch di campioni (da un episodio) in support e query set.
    Args:
        batch_samples: Lista di (immagine_tensor, label_globale) fornita da DataLoader.
                       La lunghezza di questa lista è N_WAY * (K_SHOT + Q_QUERIES).
                       L'ordine è determinato dal EpisodicBatchSampler.
    Returns:
        support_images, support_labels_relative, query_images, query_labels_relative, global_class_ids_in_episode
    """
    all_images = torch.stack([s[0] for s in batch_samples])
    all_global_labels = torch.tensor([s[1] for s in batch_samples]) # Etichette globali

    support_images_list: List[torch.Tensor] = []
    query_images_list: List[torch.Tensor] = []
    support_labels_relative_list: List[torch.Tensor] = []
    query_labels_relative_list: List[torch.Tensor] = []
    
    global_class_ids_in_episode: List[int] = [] # Mantiene traccia delle classi globali nell'episodio
    
    # Il batch_samples è già ordinato per classe dal sampler:
    # [class0_s1..sk, class0_q1..qq, class1_s1..sk, class1_q1..qq, ...]
    # No, il sampler fornisce solo una lista piatta di indici. Il collate_fn riceve i campioni in quell'ordine.
    # Dobbiamo ricostruire. Assumiamo che il sampler dia N*(K+Q) indici, raggruppati per classe.

    current_pos = 0
    for i_way in range(n_way): # Per ogni classe nell'episodio
        # Estrai le immagini per la classe corrente
        class_support_imgs = all_images[current_pos : current_pos + k_shot]
        class_query_imgs = all_images[current_pos + k_shot : current_pos + k_shot + q_queries]
        
        # Etichetta relativa al task (0 to N-1)
        relative_label = torch.tensor(i_way, dtype=torch.long)
        
        support_images_list.append(class_support_imgs)
        query_images_list.append(class_query_imgs)
        support_labels_relative_list.extend([relative_label] * k_shot)
        query_labels_relative_list.extend([relative_label] * q_queries)
        
        # Prendi l'ID globale della classe (assumendo che tutti i campioni di supporto per una classe abbiano la stessa label globale)
        global_class_ids_in_episode.append(all_global_labels[current_pos].item())

        current_pos += (k_shot + q_queries)

    # Concatena tutte le immagini di supporto e query
    # Support: (N*K, C, H, W), Query: (N*Q, C, H, W)
    s_images = torch.cat(support_images_list, dim=0)
    q_images = torch.cat(query_images_list, dim=0)
    s_labels_rel = torch.stack(support_labels_relative_list)
    q_labels_rel = torch.stack(query_labels_relative_list)

    return s_images, s_labels_rel, q_images, q_labels_rel, global_class_ids_in_episode


# --- Funzioni di Meta-Training e Validazione per Prototypical Networks ---
# 1. Modifica il calcolo dei logits in meta_train_epoch e meta_validate_epoch
# Cerca il calcolo delle distanze e sostituisci con:

def meta_train_epoch(model: CLIPWithStyleAdapter,
                     train_loader_episodic: data.DataLoader,
                     optimizer: optim.Optimizer,
                     scheduler: LambdaLR,
                     criterion: nn.Module,
                     device: torch.device,
                     epoch_num: int, total_epochs: int,
                     n_way: int, k_shot: int, q_queries: int
                     ) -> Tuple[float, float]:
    model.train()
    
    total_epoch_loss = 0.0
    total_epoch_correct_preds = 0
    total_epoch_query_samples = 0

    for batch_idx, (support_images, support_labels_rel, query_images, query_labels_rel, _) in \
            enumerate(tqdm(train_loader_episodic, desc=f"Meta Epoch {epoch_num+1}/{total_epochs} [Train]")):

        support_images = support_images.to(device)
        support_labels_rel = support_labels_rel.to(device)
        query_images = query_images.to(device)
        query_labels_rel = query_labels_rel.to(device)

        # Stampa diagnostica per verificare il flusso dei gradienti
        if batch_idx == 0 and epoch_num == 0:
            print("Verifica parametri trainabili:")
            total_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"Parametro trainabile: {name}, shape: {param.shape}")
                    total_params += param.numel()
            print(f"Totale parametri trainabili: {total_params}")

        optimizer.zero_grad()

        # Estrazione feature
        support_embeddings = model.encode_image_with_style_adapter(support_images)
        support_embeddings_reshaped = support_embeddings.view(n_way, k_shot, -1)
        prototypes = support_embeddings_reshaped.mean(dim=1)
        query_embeddings = model.encode_image_with_style_adapter(query_images)
        
        # MODIFICA: Calcolo di similarità coseno invece di distanze euclidee
        # Normalizzazione L2 per prototipi e query embeddings
        prototypes = nn.functional.normalize(prototypes, p=2, dim=-1)
        query_embeddings = nn.functional.normalize(query_embeddings, p=2, dim=-1)
        
        # Similarità coseno (valori più alti = più simili)
        similarity = torch.mm(query_embeddings, prototypes.t())
        
        # Usa direttamente la similarità coseno come logits
        logits = similarity * 20.0  # Scala la similarità (temperatura)
        
        # Stampa diagnostica
        if batch_idx == 0:
            print(f"Epoch {epoch_num+1} - Batch 0 logits stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
            print(f"Logits range: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        
        loss = criterion(logits, query_labels_rel)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"AVVISO: NaN/Inf nella loss, episodio {batch_idx} saltato")
            continue
            
        loss.backward()
        
        # Verifica i gradienti nel primo batch della prima epoca
        if batch_idx == 0 and epoch_num == 0:
            print("Verifica gradienti:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        print(f"PROBLEMA: {name} ha grad=None!")
                    elif param.grad.abs().sum().item() == 0:
                        print(f"PROBLEMA: {name} ha gradienti tutti zero!")
                    else:
                        print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
        
        # Clip gradienti per stabilità
        torch.nn.utils.clip_grad_norm_((p for group in optimizer.param_groups for p in group['params'] if p.requires_grad), max_norm=1.0)
        optimizer.step()
        if scheduler:
             scheduler.step()

        # Statistiche
        total_epoch_loss += loss.item()
        
        preds = logits.argmax(dim=1)
        total_epoch_correct_preds += (preds == query_labels_rel).sum().item()
        total_epoch_query_samples += query_labels_rel.size(0)

    avg_loss = total_epoch_loss / len(train_loader_episodic) if len(train_loader_episodic) > 0 else 0
    avg_acc = total_epoch_correct_preds / total_epoch_query_samples if total_epoch_query_samples > 0 else 0
    
    return avg_loss, avg_acc


# 2. Modifica anche la funzione di validazione in modo analogo
@torch.no_grad()
def meta_validate_epoch(model: CLIPWithStyleAdapter,
                        val_loader_episodic: data.DataLoader,
                        criterion: nn.Module,
                        device: torch.device,
                        epoch_num: int, total_epochs: int,
                        n_way: int, k_shot: int, q_queries: int
                        ) -> float:
    model.eval()
    
    total_epoch_correct_preds = 0
    total_epoch_query_samples = 0

    for batch_idx, (support_images, support_labels_rel, query_images, query_labels_rel, _) in \
            enumerate(tqdm(val_loader_episodic, desc=f"Meta Epoch {epoch_num+1}/{total_epochs} [Val]")):
        
        support_images = support_images.to(device)
        query_images = query_images.to(device)
        query_labels_rel = query_labels_rel.to(device)

        support_embeddings = model.encode_image_with_style_adapter(support_images)
        support_embeddings_reshaped = support_embeddings.view(n_way, k_shot, -1)
        prototypes = support_embeddings_reshaped.mean(dim=1)
        query_embeddings = model.encode_image_with_style_adapter(query_images)
        
        # MODIFICA: Usa similarità coseno come nella funzione di training
        prototypes = nn.functional.normalize(prototypes, p=2, dim=-1)
        query_embeddings = nn.functional.normalize(query_embeddings, p=2, dim=-1)
        
        similarity = torch.mm(query_embeddings, prototypes.t())
        logits = similarity * 20.0  # Stessa temperatura usata nel training
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"AVVISO: NaN/Inf nei logits di validazione, episodio {batch_idx} saltato")
            continue

        preds = logits.argmax(dim=1)
        total_epoch_correct_preds += (preds == query_labels_rel).sum().item()
        total_epoch_query_samples += query_labels_rel.size(0)

    avg_acc = total_epoch_correct_preds / total_epoch_query_samples if total_epoch_query_samples > 0 else 0
    return avg_acc



# --- Funzione per Optuna ---
def define_model_and_train_meta(trial: optuna.Trial, args: argparse.Namespace, device: torch.device,
                                meta_train_loader: data.DataLoader, meta_val_loader: data.DataLoader):
    # Parametri da ottimizzare
    fusion_bottleneck_dim = trial.suggest_categorical("fusion_bottleneck_dim", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) # Aumentato limite superiore per meta-learning
    dropout_rate_adapter = trial.suggest_float("dropout_rate_adapter", 0.0, 0.5, step=0.1) # Aumentato limite
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) # Aumentato limite
    
    # K_SHOT per il training (può essere un iperparametro, ma per ora fisso)
    current_k_shot = args.k_shot_train 
    if args.n_way == 1 and args.k_shot_train > 1 : # Per 1-way, K-shot è più come # campioni per batch
        print("N_WAY è 1, k_shot nel sampler sarà 1 per il prototipo, il resto per la query.")
    
    print(f"Trial {trial.number}: fusion_bottleneck={fusion_bottleneck_dim}, lr={lr:.2e}, dropout={dropout_rate_adapter:.2f}, wd={weight_decay:.2e}")

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
        print(f"Errore nella creazione del modello nel trial {trial.number}: {e}")
        model._remove_gram_hooks() # Assicurati che gli hook siano rimossi
        raise optuna.exceptions.TrialPruned()

    trainable_params = []
    if hasattr(model, 'gram_layer_projections') and model.gram_layer_projections:
        trainable_params.extend(list(model.gram_layer_projections.parameters()))
    if hasattr(model, 'fusion_adapter'):
        trainable_params.extend(list(model.fusion_adapter.parameters()))
    
    if not trainable_params:
        print("Nessun parametro addestrabile trovato nel modello.")
        model._remove_gram_hooks()
        raise optuna.exceptions.TrialPruned()
    
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler LR (opzionale per meta-learning, ma può aiutare)
    num_train_steps_per_epoch = len(meta_train_loader)
    total_train_steps = args.epochs * num_train_steps_per_epoch
    
    def lr_lambda(current_step):
        if args.warmup_steps > 0 and current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        # Opzionale: decadimento dopo warmup
        # return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - args.warmup_steps) / (total_train_steps - args.warmup_steps))))
        return 1.0 # Nessun decadimento dopo warmup per ora
    
    scheduler = LambdaLR(optimizer, lr_lambda) if args.warmup_steps > 0 else None

    best_val_acc_trial = 0.0
    no_improvement_count = 0
    
    # Salvataggio dei moduli addestrabili migliori per questo trial
    best_gram_projections_state_dict = None
    best_fusion_adapter_state_dict = None
    
    for epoch in range(args.epochs):
        train_loss, train_acc = meta_train_epoch(
            model, meta_train_loader, optimizer, scheduler, criterion, device,
            epoch, args.epochs, args.n_way, current_k_shot, args.q_queries
        )
        
        val_acc = meta_validate_epoch(
            model, meta_val_loader, criterion, device,
            epoch, args.epochs, args.n_way, args.k_shot_val, args.q_queries # Usa k_shot_val per la validazione
        )
        
        print(f"Trial {trial.number} - Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        trial.report(val_acc, epoch)
        
        if trial.should_prune():
            model._remove_gram_hooks()
            del model
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        
        if val_acc > best_val_acc_trial + args.early_stopping_min_delta:
            best_val_acc_trial = val_acc
            no_improvement_count = 0
            
            # Salva lo stato dei moduli addestrabili del modello
            if hasattr(model, 'gram_layer_projections') and model.gram_layer_projections:
                best_gram_projections_state_dict = {
                    k: v.cpu().clone() for k, v in model.gram_layer_projections.state_dict().items()
                }
            if hasattr(model, 'fusion_adapter'):
                best_fusion_adapter_state_dict = {
                    k: v.cpu().clone() for k, v in model.fusion_adapter.state_dict().items()
                }
            
            # Salva i checkpoint del trial corrente
            trial_checkpoint = {
                'trial': trial.number,
                'epoch': epoch,
                'val_acc': val_acc,
                'fusion_bottleneck_dim': fusion_bottleneck_dim,
                'dropout_rate': dropout_rate_adapter,
                'gram_layer_projections_state_dict': best_gram_projections_state_dict,
                'fusion_adapter_state_dict': best_fusion_adapter_state_dict,
                'hyperparams': trial.params
            }
            
            # Salva il checkpoint del miglior modello per questo trial
            torch.save(trial_checkpoint, os.path.join(args.study_dir, f"best_model_trial_{trial.number}.pt"))
            print(f"Salvato miglior modello per trial {trial.number} con val_acc {val_acc:.4f}")
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= args.early_stopping_patience:
            print(f'Early stopping attivato per il trial {trial.number} dopo {epoch+1} epoche.')
            break
    
    # Aggiorna lo studio con informazioni aggiuntive sul miglior trial
    trial.set_user_attr('best_val_acc', best_val_acc_trial)
    
    model._remove_gram_hooks() # Rimuovi hook prima di eliminare il modello
    del model
    torch.cuda.empty_cache() # Libera memoria GPU
    
    return best_val_acc_trial


def objective(trial: optuna.Trial, args: argparse.Namespace, device: torch.device,
              meta_train_dataset_full: ArtgraphDataset, meta_val_dataset_full: ArtgraphDataset,
              preprocess_fn # Funzione di preprocessamento da CLIP
             ) -> float:
    try:
        # Creazione dei DataLoader episodici all'interno di ogni trial per gestire k_shot variabile
        # (se k_shot fosse un iperparametro di optuna)
        # Per ora k_shot è fisso dagli args

        # --- Sampler e DataLoader per Meta-Training ---
        # Mappa: class_idx_globale -> lista di indici di campioni per quella classe nel dataset piatto
        train_labels_by_class = {
            cls_idx: [i for i, (_, lab) in enumerate(meta_train_dataset_full.flat_samples) if lab == cls_idx]
            for cls_idx in meta_train_dataset_full.class_to_idx.values()
        }
        train_sampler = EpisodicBatchSampler(
            train_labels_by_class, args.n_way, args.k_shot_train, args.q_queries, args.num_episodes_train_epoch
        )
        # Il DataLoader ora prende il dataset completo. Il sampler sceglie gli indici.
        meta_train_loader = data.DataLoader(
            meta_train_dataset_full,
            batch_sampler=train_sampler, # NOTA: batch_sampler, non sampler. shuffle, batch_size, drop_last sono ignorati.
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: episodic_collate_fn(batch, args.n_way, args.k_shot_train, args.q_queries)
        )

        # --- Sampler e DataLoader per Meta-Validazione ---
        val_labels_by_class = {
            cls_idx: [i for i, (_, lab) in enumerate(meta_val_dataset_full.flat_samples) if lab == cls_idx]
            for cls_idx in meta_val_dataset_full.class_to_idx.values()
        }
        val_sampler = EpisodicBatchSampler(
            val_labels_by_class, args.n_way, args.k_shot_val, args.q_queries, args.num_episodes_val
        )
        meta_val_loader = data.DataLoader(
            meta_val_dataset_full,
            batch_sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: episodic_collate_fn(batch, args.n_way, args.k_shot_val, args.q_queries)
        )

        val_acc = define_model_and_train_meta(trial, args, device, meta_train_loader, meta_val_loader)
        
        current_best_acc_overall = trial.study.user_attrs.get("best_val_acc_overall", float('-inf'))
        if val_acc > current_best_acc_overall:
            trial.study.set_user_attr("best_val_acc_overall", val_acc) # Salva nello studio
            best_params_trial = {
                'trial_number': trial.number,
                'fusion_bottleneck_dim': trial.params['fusion_bottleneck_dim'],
                'lr': trial.params['lr'],
                'dropout_rate_adapter': trial.params['dropout_rate_adapter'],
                'weight_decay': trial.params['weight_decay'],
                'val_accuracy': val_acc
            }
            with open(os.path.join(args.study_dir, "best_params_overall.json"), 'w') as f:
                json.dump(best_params_trial, f, indent=4)
            print(f"Nuovi migliori parametri GLOBALI trovati con accuratezza {val_acc:.4f}: {best_params_trial}")
            
        return val_acc
        
    except optuna.exceptions.TrialPruned:
        raise # Rialza per Optuna
    except Exception as e:
        print(f"Errore grave durante il trial {trial.number if trial else 'N/A'}: {e}")
        import traceback
        traceback.print_exc()
        return float('-inf')


def save_optuna_visualizations(study: optuna.Study, study_dir: str):
    viz_dir = os.path.join(study_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    plot_functions = {
        "param_importances": optuna.visualization.plot_param_importances,
        "optimization_history": optuna.visualization.plot_optimization_history,
        "slice": optuna.visualization.plot_slice,
        "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
        "contour": optuna.visualization.plot_contour 
    }
    
    for name, plot_func in plot_functions.items():
        try:
            if name == "contour" and len(study.best_params) < 2: # Contour plot needs at least 2 params
                print("Skipping contour plot, not enough parameters.")
                continue
            fig = plot_func(study)
            fig.write_image(os.path.join(viz_dir, f"{name}.png"))
        except (ValueError, RuntimeError, TypeError) as e: # Catch more specific errors
            print(f"Errore nel salvataggio del plot '{name}': {e}")
        except Exception as e:
            print(f"Errore generico nel salvataggio del plot '{name}': {e}")
            
    # Scatter matrix personalizzato
    try:
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
        if completed_trials:
            param_names = list(completed_trials[0].params.keys())
            data_for_df = {param: [t.params.get(param) for t in completed_trials] for param in param_names}
            data_for_df["accuracy"] = [t.value for t in completed_trials]
            
            df = pd.DataFrame(data_for_df)
            if not df.empty:
                plt.figure(figsize=(12, 12)) # Aumentato figsize per leggibilità
                sns.pairplot(df, diag_kind='kde', corner=True) # Aggiunto corner=True
                plt.suptitle("Scatter Matrix dei Parametri vs Accuratezza", y=1.02)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, "parameter_relationships_scatter_matrix.png"))
                plt.close()
            else:
                print("DataFrame vuoto per scatter matrix, skipping.")
        else:
            print("Nessun trial completato per scatter matrix.")
            
    except ImportError:
        print("Seaborn o Pandas non installati, impossibile generare scatter matrix.")
    except Exception as e:
        print(f"Errore nel salvataggio dello scatter matrix: {e}")


def main():
    parser = argparse.ArgumentParser(description='Ottimizzazione iperparametri Meta-Learning Prototipico per StyleAdapter')
    
    # Optuna HPO
    parser.add_argument('--n_trials', type=int, default=15, help='Numero di trial per Optuna')
    parser.add_argument('--study_name_prefix', type=str, default="styleadapter_proto_hpo")
    parser.add_argument('--study_dir_root', type=str, default="optuna_studies_meta", help='Directory radice per salvare i risultati degli studi')
    parser.add_argument('--storage', type=str, default=None, help='URL di storage Optuna (es. sqlite:///mystudy.db)')
    parser.add_argument('--pruning', action='store_true', help='Abilita pruning Optuna')
    
    # Modello
    parser.add_argument('--clip_model_name', type=str, default='RN50')
    parser.add_argument('--gram_style_projection_dim', type=int, default=256)
    parser.add_argument('--layers_for_gram_rn50', type=str, nargs='+', default=['layer2', 'layer3'])
    parser.add_argument('--use_layernorm_adapter', type=bool, default=True)
    
    # Meta-Learning Training
    parser.add_argument('--epochs', type=int, default=100, help='Max epoche per trial Optuna') # Ridotto per trial veloci
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100) # Relativo agli step totali del trial
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)

    # Parametri Episodici (Fissi per ora, potrebbero diventare iperparametri)
    parser.add_argument('--n_way', type=int, default=N_WAY)
    parser.add_argument('--k_shot_train', type=int, default=K_SHOT)
    parser.add_argument('--k_shot_val', type=int, default=4) # Più shot per una validazione più stabile
    parser.add_argument('--q_queries', type=int, default=Q_QUERIES)
    parser.add_argument('--num_episodes_train_epoch', type=int, default=50, help="Numero di episodi per epoca di meta-training")
    parser.add_argument('--num_episodes_val', type=int, default=25, help="Numero di episodi per la meta-validazione")
    parser.add_argument('--meta_val_split_ratio', type=float, default=0.3, help="Percentuale di artisti per meta-validazione")
    
    args = parser.parse_args()

    # Configurazione dinamica nome studio e directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.study_name = f"{args.study_name_prefix}_{timestamp}"
    args.study_dir = os.path.join(args.study_dir_root, args.study_name)
    os.makedirs(args.study_dir, exist_ok=True)

    with open(os.path.join(args.study_dir, "config_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False # Per riproducibilità, può rallentare
        torch.backends.cudnn.deterministic = True
    print(f"Utilizzo device: {device}")
    
    # --- Preparazione Dataset per Meta-Learning ---
    try:
        dataset_root_path = find_artgraph_path()
        # Carica tutti gli artisti disponibili per dividerli
        temp_images_dir = os.path.join(dataset_root_path, 'images')
        all_artists = sorted([d for d in os.listdir(temp_images_dir) if os.path.isdir(os.path.join(temp_images_dir, d))])
        random.shuffle(all_artists) # Shuffle per divisione casuale
        
        num_val_artists = int(len(all_artists) * args.meta_val_split_ratio)
        meta_val_artists = all_artists[:num_val_artists]
        meta_train_artists = all_artists[num_val_artists:]

        if not meta_train_artists or not meta_val_artists:
            raise ValueError("Divisione artisti in meta-train/meta-val ha prodotto set vuoti. Controlla meta_val_split_ratio e il numero di artisti.")

        print(f"Artisti totali: {len(all_artists)}")
        print(f"Artisti Meta-Train ({len(meta_train_artists)}): {meta_train_artists[:5]}...")
        print(f"Artisti Meta-Val ({len(meta_val_artists)}): {meta_val_artists[:5]}...")

        # Preprocessing da un modello CLIP (non serve il modello intero qui, solo preprocess)
        # Carica il preprocess associato al modello CLIP specificato
        model_name_to_load_clip = "RN50" if args.clip_model_name == "CustomRN50" else args.clip_model_name
        _, preprocess_fn = clip.load(model_name_to_load_clip, device=device) # Carica su CPU o GPU, non importa per solo preprocess
        
        meta_train_dataset_full = ArtgraphDataset(
            dataset_root_path, transform=preprocess_fn, seed=args.seed, artist_subset=meta_train_artists
        )
        meta_val_dataset_full = ArtgraphDataset(
            dataset_root_path, transform=preprocess_fn, seed=args.seed, artist_subset=meta_val_artists
        )
    except Exception as e:
        print(f"Errore nella preparazione del dataset per meta-learning: {e}")
        import traceback
        traceback.print_exc()
        return

    storage = args.storage if args.storage else f"sqlite:///{os.path.join(args.study_dir, 'optuna_study.db')}"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=args.epochs // 3, interval_steps=1) if args.pruning else None
    
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="maximize",
        pruner=pruner, load_if_exists=True # Permette di riprendere studi interrotti
    )
    study.set_user_attr("best_val_acc_overall", float('-inf')) # Inizializza attributo utente

    try:
        study.optimize(
            lambda trial: objective(trial, args, device, meta_train_dataset_full, meta_val_dataset_full, preprocess_fn),
            n_trials=args.n_trials, timeout=None # Nessun timeout globale
        )
    except KeyboardInterrupt:
        print("Ottimizzazione interrotta dall'utente.")
    except Exception as e:
        print(f"Errore imprevisto durante study.optimize: {e}")
        import traceback
        traceback.print_exc()
    
    joblib.dump(study, os.path.join(args.study_dir, "study_results.pkl"))
    
    print("\nOttimizzazione Completata!")
    if study.trials: # Controlla se ci sono trials prima di accedere a best_trial
        best_trial_overall = None
        try: # Prova a trovare il miglior trial in base al valore effettivo, non solo l'ultimo
            best_trial_overall = study.best_trial
            print(f"Miglior accuratezza di meta-validazione nel trial #{best_trial_overall.number}: {best_trial_overall.value:.4f}")
            print(f"Migliori parametri del trial: {best_trial_overall.params}")
            
            # Salva i parametri del miglior trial dello studio
            with open(os.path.join(args.study_dir, "final_best_trial_params.json"), 'w') as f:
                json.dump({**best_trial_overall.params, "val_accuracy": best_trial_overall.value}, f, indent=4)

            # Salva il miglior modello complessivo per poterlo utilizzare in altri file
            best_model_path = os.path.join(args.study_dir, f"best_model_trial_{best_trial_overall.number}.pt")
            if os.path.exists(best_model_path):
                # Carica il checkpoint del miglior trial
                best_checkpoint = torch.load(best_model_path, map_location="cpu")
                
                # Salva il miglior modello nella directory principale per facile accesso
                main_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                best_model_final_path = os.path.join(main_project_dir, "best_style_adapted_clip_artgraph.pt")
                
                # Utilizza una copia del checkpoint per evitare modifiche accidentali
                final_checkpoint = {
                    'epoch': best_checkpoint.get('epoch', 0),
                    'val_acc': best_checkpoint.get('val_acc', 0.0),
                    'fusion_bottleneck_dim': best_checkpoint.get('fusion_bottleneck_dim', 128),
                    'gram_layer_projections_state_dict': best_checkpoint.get('gram_layer_projections_state_dict', None),
                    'fusion_adapter_state_dict': best_checkpoint.get('fusion_adapter_state_dict', None),
                    'hyperparams': best_checkpoint.get('hyperparams', {}),
                    'clip_model_name': args.clip_model_name,
                    'gram_style_projection_dim': args.gram_style_projection_dim,
                    'layers_for_gram_rn50': args.layers_for_gram_rn50,
                    'use_layernorm_adapter': args.use_layernorm_adapter,
                    'args': args
                }
                
                # Salva il checkpoint finale
                torch.save(final_checkpoint, best_model_final_path)
                print(f"\nMiglior modello salvato in: {best_model_final_path}")
                print("Questo modello può essere utilizzato in altri script tramite il nome 'CustomRN50' in clip.load()")
                
                # Crea una copia anche nella directory del codice per essere sicuri
                alternative_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_adapted_clip_artgraph.pt")
                torch.save(final_checkpoint, alternative_path)
                print(f"Copia di backup salvata in: {alternative_path}")
            else:
                print(f"ATTENZIONE: File del miglior modello {best_model_path} non trovato!")

        except ValueError: # Può succedere se nessun trial è stato completato con successo
            print("Nessun trial completato con successo trovato nello studio.")
        except Exception as e:
            print(f"Errore nell'ottenere il miglior trial: {e}")

        # Controlla anche i parametri salvati in best_params_overall.json (aggiornati durante i trial)
        best_params_file = os.path.join(args.study_dir, "best_params_overall.json")
        if os.path.exists(best_params_file):
            with open(best_params_file, 'r') as f:
                overall_best = json.load(f)
            print(f"\nMigliori parametri registrati durante la ricerca (da best_params_overall.json):")
            print(f"Accuratezza: {overall_best.get('val_accuracy', 'N/A'):.4f}")
            print(f"Parametri: {overall_best}")

    else:
        print("Nessun trial eseguito nello studio.")
        
    print(f"Risultati dettagliati e log salvati in: {args.study_dir}")
    
    try:
        save_optuna_visualizations(study, args.study_dir)
    except Exception as e:
        print(f"Errore nella generazione delle visualizzazioni finali: {e}")

if __name__ == "__main__":
    main()