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
import pandas as pd # Added for save_optuna_visualizations
from tqdm import tqdm
from typing import List, Tuple, Dict, Iterator
import random
# Import from StyleAdapter.py file
from StyleAdapterFLS import CLIPWithStyleAdapter, find_artgraph_path, ArtgraphDataset

# --- Fixed Parameters for Meta-Learning ---
# These could be made parser arguments or Optuna hyperparameters
N_WAY = 10  # Number of classes per task
K_SHOT = 1  # Number of support examples per class for N_WAY > 1
Q_QUERIES = 3 # Number of query examples per class



# --- Episodic Sampler ---
class EpisodicBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset_labels_by_class: Dict[int, List[int]],
                 n_way: int, k_shot: int, q_queries: int, num_episodes: int):
        """
        Args:
            dataset_labels_by_class: Dictionary {global_class_idx: [sample_idx_1, sample_idx_2, ...]}
                                     the sample_idx are indices in the flat dataset.
            n_way: Number of classes per episode.
            k_shot: Number of support samples per class.
            q_queries: Number of query samples per class.
            num_episodes: Number of episodes to generate per epoch.
        """
        super().__init__(None) # data_source is not used directly
        self.dataset_labels_by_class = dataset_labels_by_class
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.num_episodes = num_episodes

        self.available_classes = list(self.dataset_labels_by_class.keys())
        
        # Filter classes with insufficient samples
        self.valid_classes = [
            cls_idx for cls_idx, samples in self.dataset_labels_by_class.items()
            if len(samples) >= self.k_shot + self.q_queries
        ]
        if len(self.valid_classes) < self.n_way:
            raise ValueError(
                f"Not enough classes ({len(self.valid_classes)}) with sufficient samples "
                f"({self.k_shot + self.q_queries} required) to form {self.n_way}-way episodes. "
                "Reduce N_WAY, K_SHOT, Q_QUERIES or increase the dataset."
            )
        print(f"EpisodicSampler: {len(self.valid_classes)} valid classes for {self.n_way}-way {self.k_shot}-shot {self.q_queries}-query episodes.")


    def __len__(self) -> int:
        return self.num_episodes

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_episodes):
            # Sample N classes without replacement from valid classes
            selected_class_indices_global = random.sample(self.valid_classes, self.n_way)
            
            episode_sample_indices: List[int] = []
            # Task-relative labels (0...N-1) are implicit in the order

            for class_idx_global in selected_class_indices_global:
                samples_for_class = self.dataset_labels_by_class[class_idx_global]
                # Sample K+Q samples without replacement from this class
                selected_samples_for_class_indices = random.sample(
                    samples_for_class, self.k_shot + self.q_queries
                )
                episode_sample_indices.extend(selected_samples_for_class_indices)
            
            yield episode_sample_indices


# --- Episodic Collate Function ---
def episodic_collate_fn(batch_samples: List[Tuple[torch.Tensor, int]],
                        n_way: int, k_shot: int, q_queries: int
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Organizes a batch of samples (from an episode) into support and query sets.
    Args:
        batch_samples: List of (image_tensor, global_label) provided by DataLoader.
                       The length of this list is N_WAY * (K_SHOT + Q_QUERIES).
                       The order is determined by EpisodicBatchSampler.
    Returns:
        support_images, support_labels_relative, query_images, query_labels_relative, global_class_ids_in_episode
    """
    all_images = torch.stack([s[0] for s in batch_samples])
    all_global_labels = torch.tensor([s[1] for s in batch_samples]) # Global labels

    support_images_list: List[torch.Tensor] = []
    query_images_list: List[torch.Tensor] = []
    support_labels_relative_list: List[torch.Tensor] = []
    query_labels_relative_list: List[torch.Tensor] = []
    
    global_class_ids_in_episode: List[int] = [] # Keeps track of global classes in the episode
    


    current_pos = 0
    for i_way in range(n_way): # For each class in the episode
        # Extract images for current class
        class_support_imgs = all_images[current_pos : current_pos + k_shot]
        class_query_imgs = all_images[current_pos + k_shot : current_pos + k_shot + q_queries]
        
        # Task-relative label (0 to N-1)
        relative_label = torch.tensor(i_way, dtype=torch.long)
        
        support_images_list.append(class_support_imgs)
        query_images_list.append(class_query_imgs)
        support_labels_relative_list.extend([relative_label] * k_shot)
        query_labels_relative_list.extend([relative_label] * q_queries)
        
        # Take global class ID (assuming all support samples for a class have the same global label)
        global_class_ids_in_episode.append(all_global_labels[current_pos].item())

        current_pos += (k_shot + q_queries)

    # Concatenate all support and query images
    # Support: (N*K, C, H, W), Query: (N*Q, C, H, W)
    s_images = torch.cat(support_images_list, dim=0)
    q_images = torch.cat(query_images_list, dim=0)
    s_labels_rel = torch.stack(support_labels_relative_list)
    q_labels_rel = torch.stack(query_labels_relative_list)

    return s_images, s_labels_rel, q_images, q_labels_rel, global_class_ids_in_episode




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

        # Diagnostic print to verify gradient flow
        if batch_idx == 0 and epoch_num == 0:
            print("Verifying trainable parameters:")
            total_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"Trainable parameter: {name}, shape: {param.shape}")
                    total_params += param.numel()
            print(f"Total trainable parameters: {total_params}")

        optimizer.zero_grad()

        # Feature extraction
        support_embeddings = model.encode_image_with_style_adapter(support_images)
        support_embeddings_reshaped = support_embeddings.view(n_way, k_shot, -1)
        prototypes = support_embeddings_reshaped.mean(dim=1)
        query_embeddings = model.encode_image_with_style_adapter(query_images)
        
        
        prototypes = nn.functional.normalize(prototypes, p=2, dim=-1)
        query_embeddings = nn.functional.normalize(query_embeddings, p=2, dim=-1)
        
       
        similarity = torch.mm(query_embeddings, prototypes.t())
        

        logits = similarity * 20.0  # Scale similarity (temperature)
        
        # Diagnostic print
        if batch_idx == 0:
            print(f"Epoch {epoch_num+1} - Batch 0 logits stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
            print(f"Logits range: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        
        loss = criterion(logits, query_labels_rel)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf in loss, episode {batch_idx} skipped")
            continue
            
        loss.backward()
        

        if batch_idx == 0 and epoch_num == 0:
            print("Verifying gradients:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        print(f"PROBLEM: {name} has grad=None!")
                    elif param.grad.abs().sum().item() == 0:
                        print(f"PROBLEM: {name} has all zero gradients!")
                    else:
                        print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
        

        torch.nn.utils.clip_grad_norm_((p for group in optimizer.param_groups for p in group['params'] if p.requires_grad), max_norm=1.0)
        optimizer.step()
        if scheduler:
             scheduler.step()


        total_epoch_loss += loss.item()
        
        preds = logits.argmax(dim=1)
        total_epoch_correct_preds += (preds == query_labels_rel).sum().item()
        total_epoch_query_samples += query_labels_rel.size(0)

    avg_loss = total_epoch_loss / len(train_loader_episodic) if len(train_loader_episodic) > 0 else 0
    avg_acc = total_epoch_correct_preds / total_epoch_query_samples if total_epoch_query_samples > 0 else 0
    
    return avg_loss, avg_acc



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
        

        prototypes = nn.functional.normalize(prototypes, p=2, dim=-1)
        query_embeddings = nn.functional.normalize(query_embeddings, p=2, dim=-1)
        
        similarity = torch.mm(query_embeddings, prototypes.t())
        logits = similarity * 20.0  
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"WARNING: NaN/Inf in validation logits, episode {batch_idx} skipped")
            continue

        preds = logits.argmax(dim=1)
        total_epoch_correct_preds += (preds == query_labels_rel).sum().item()
        total_epoch_query_samples += query_labels_rel.size(0)

    avg_acc = total_epoch_correct_preds / total_epoch_query_samples if total_epoch_query_samples > 0 else 0
    return avg_acc




def define_model_and_train_meta(trial: optuna.Trial, args: argparse.Namespace, device: torch.device,
                                meta_train_loader: data.DataLoader, meta_val_loader: data.DataLoader):
    # Parameters to optimize
    fusion_bottleneck_dim = trial.suggest_categorical("fusion_bottleneck_dim", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) # Increased upper limit for meta-learning
    dropout_rate_adapter = trial.suggest_float("dropout_rate_adapter", 0.0, 0.5, step=0.1) # Increased limit
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) # Increased limit
    
    # K_SHOT for training (can be a hyperparameter, but fixed for now)
    current_k_shot = args.k_shot_train 
    if args.n_way == 1 and args.k_shot_train > 1 : # For 1-way, K-shot is more like # samples per batch
        print("N_WAY is 1, k_shot in sampler will be 1 for prototype, rest for query.")
    
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
        print(f"Error creating model in trial {trial.number}: {e}")
        model._remove_gram_hooks() # Ensure hooks are removed
        raise optuna.exceptions.TrialPruned()

    trainable_params = []
    if hasattr(model, 'gram_layer_projections') and model.gram_layer_projections:
        trainable_params.extend(list(model.gram_layer_projections.parameters()))
    if hasattr(model, 'fusion_adapter'):
        trainable_params.extend(list(model.fusion_adapter.parameters()))
    
    if not trainable_params:
        print("No trainable parameters found in the model.")
        model._remove_gram_hooks()
        raise optuna.exceptions.TrialPruned()
    
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # LR Scheduler (optional for meta-learning, but can help)
    num_train_steps_per_epoch = len(meta_train_loader)
    total_train_steps = args.epochs * num_train_steps_per_epoch
    
    def lr_lambda(current_step):
        if args.warmup_steps > 0 and current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return 1.0 # No decay after warmup for now
    
    scheduler = LambdaLR(optimizer, lr_lambda) if args.warmup_steps > 0 else None

    best_val_acc_trial = 0.0
    no_improvement_count = 0
    

    best_gram_projections_state_dict = None
    best_fusion_adapter_state_dict = None
    
    for epoch in range(args.epochs):
        train_loss, train_acc = meta_train_epoch(
            model, meta_train_loader, optimizer, scheduler, criterion, device,
            epoch, args.epochs, args.n_way, current_k_shot, args.q_queries
        )
        
        val_acc = meta_validate_epoch(
            model, meta_val_loader, criterion, device,
            epoch, args.epochs, args.n_way, args.k_shot_val, args.q_queries # Use k_shot_val for validation
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
            
            # Save state of trainable model modules
            if hasattr(model, 'gram_layer_projections') and model.gram_layer_projections:
                best_gram_projections_state_dict = {
                    k: v.cpu().clone() for k, v in model.gram_layer_projections.state_dict().items()
                }
            if hasattr(model, 'fusion_adapter'):
                best_fusion_adapter_state_dict = {
                    k: v.cpu().clone() for k, v in model.fusion_adapter.state_dict().items()
                }
            
            # Save current trial checkpoints
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
            
            # Save best model checkpoint for this trial
            torch.save(trial_checkpoint, os.path.join(args.study_dir, f"best_model_trial_{trial.number}.pt"))
            print(f"Saved best model for trial {trial.number} with val_acc {val_acc:.4f}")
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= args.early_stopping_patience:
            print(f'Early stopping activated for trial {trial.number} after {epoch+1} epochs.')
            break
    
    # Update study with additional information about best trial
    trial.set_user_attr('best_val_acc', best_val_acc_trial)
    
    model._remove_gram_hooks() # Remove hooks before deleting model
    del model
    torch.cuda.empty_cache() # Free GPU memory
    
    return best_val_acc_trial


def objective(trial: optuna.Trial, args: argparse.Namespace, device: torch.device,
              meta_train_dataset_full: ArtgraphDataset, meta_val_dataset_full: ArtgraphDataset,
              preprocess_fn # Preprocessing function from CLIP
             ) -> float:
    try:
       
        train_labels_by_class = {
            cls_idx: [i for i, (_, lab) in enumerate(meta_train_dataset_full.flat_samples) if lab == cls_idx]
            for cls_idx in meta_train_dataset_full.class_to_idx.values()
        }
        train_sampler = EpisodicBatchSampler(
            train_labels_by_class, args.n_way, args.k_shot_train, args.q_queries, args.num_episodes_train_epoch
        )
        meta_train_loader = data.DataLoader(
            meta_train_dataset_full,
            batch_sampler=train_sampler, # NOTE: batch_sampler, not sampler. shuffle, batch_size, drop_last are ignored.
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: episodic_collate_fn(batch, args.n_way, args.k_shot_train, args.q_queries)
        )

        # --- Sampler and DataLoader for Meta-Validation ---
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
            trial.study.set_user_attr("best_val_acc_overall", val_acc) # Save in study
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
            print(f"New GLOBAL best parameters found with accuracy {val_acc:.4f}: {best_params_trial}")
            
        return val_acc
        
    except optuna.exceptions.TrialPruned:
        raise # Re-raise for Optuna
    except Exception as e:
        print(f"Serious error during trial {trial.number if trial else 'N/A'}: {e}")
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
            print(f"Error saving plot '{name}': {e}")
        except Exception as e:
            print(f"Generic error saving plot '{name}': {e}")
            
    # Custom scatter matrix
    try:
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
        if completed_trials:
            param_names = list(completed_trials[0].params.keys())
            data_for_df = {param: [t.params.get(param) for t in completed_trials] for param in param_names}
            data_for_df["accuracy"] = [t.value for t in completed_trials]
            
            df = pd.DataFrame(data_for_df)
            if not df.empty:
                plt.figure(figsize=(12, 12)) # Increased figsize for readability
                sns.pairplot(df, diag_kind='kde', corner=True) # Added corner=True
                plt.suptitle("Parameters vs Accuracy Scatter Matrix", y=1.02)
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, "parameter_relationships_scatter_matrix.png"))
                plt.close()
            else:
                print("Empty DataFrame for scatter matrix, skipping.")
        else:
            print("No completed trials for scatter matrix.")
            
    except ImportError:
        print("Seaborn or Pandas not installed, cannot generate scatter matrix.")
    except Exception as e:
        print(f"Error saving scatter matrix: {e}")


def main():
    parser = argparse.ArgumentParser(description='Prototypical Meta-Learning Hyperparameter Optimization for StyleAdapter')
    
    # Optuna HPO
    parser.add_argument('--n_trials', type=int, default=15, help='Number of trials for Optuna')
    parser.add_argument('--study_name_prefix', type=str, default="styleadapter_proto_hpo")
    parser.add_argument('--study_dir_root', type=str, default="optuna_studies_meta", help='Root directory to save study results')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL (e.g. sqlite:///mystudy.db)')
    parser.add_argument('--pruning', action='store_true', help='Enable Optuna pruning')
    
    # Model
    parser.add_argument('--clip_model_name', type=str, default='RN50')
    parser.add_argument('--gram_style_projection_dim', type=int, default=256)
    parser.add_argument('--layers_for_gram_rn50', type=str, nargs='+', default=['layer2', 'layer3'])
    parser.add_argument('--use_layernorm_adapter', type=bool, default=True)
    
    # Meta-Learning Training
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs per Optuna trial') # Reduced for fast trials
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100) # Relative to total trial steps
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)

    # Episodic Parameters (Fixed for now, could become hyperparameters)
    parser.add_argument('--n_way', type=int, default=N_WAY)
    parser.add_argument('--k_shot_train', type=int, default=K_SHOT)
    parser.add_argument('--k_shot_val', type=int, default=4) # More shots for more stable validation
    parser.add_argument('--q_queries', type=int, default=Q_QUERIES)
    parser.add_argument('--num_episodes_train_epoch', type=int, default=50, help="Number of episodes per meta-training epoch")
    parser.add_argument('--num_episodes_val', type=int, default=25, help="Number of episodes for meta-validation")
    parser.add_argument('--meta_val_split_ratio', type=float, default=0.3, help="Percentage of artists for meta-validation")
    
    args = parser.parse_args()

    # Dynamic study name and directory configuration
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
        torch.backends.cudnn.benchmark = False # For reproducibility, may slow down
        torch.backends.cudnn.deterministic = True
    print(f"Using device: {device}")
    
    # --- Dataset Preparation for Meta-Learning ---
    try:
        dataset_root_path = find_artgraph_path()
        # Load all available artists to split them
        temp_images_dir = os.path.join(dataset_root_path, 'images')
        all_artists = sorted([d for d in os.listdir(temp_images_dir) if os.path.isdir(os.path.join(temp_images_dir, d))])
        random.shuffle(all_artists) # Shuffle for random split
        
        num_val_artists = int(len(all_artists) * args.meta_val_split_ratio)
        meta_val_artists = all_artists[:num_val_artists]
        meta_train_artists = all_artists[num_val_artists:]

        if not meta_train_artists or not meta_val_artists:
            raise ValueError("Artist split into meta-train/meta-val produced empty sets. Check meta_val_split_ratio and number of artists.")

        print(f"Total artists: {len(all_artists)}")
        print(f"Meta-Train artists ({len(meta_train_artists)}): {meta_train_artists[:5]}...")
        print(f"Meta-Val artists ({len(meta_val_artists)}): {meta_val_artists[:5]}...")

       
        model_name_to_load_clip = "RN50" if args.clip_model_name == "CustomRN50" else args.clip_model_name
        _, preprocess_fn = clip.load(model_name_to_load_clip, device=device) # Load on CPU or GPU, doesn't matter for preprocess only
        
        meta_train_dataset_full = ArtgraphDataset(
            dataset_root_path, transform=preprocess_fn, seed=args.seed, artist_subset=meta_train_artists
        )
        meta_val_dataset_full = ArtgraphDataset(
            dataset_root_path, transform=preprocess_fn, seed=args.seed, artist_subset=meta_val_artists
        )
    except Exception as e:
        print(f"Error in dataset preparation for meta-learning: {e}")
        import traceback
        traceback.print_exc()
        return

    storage = args.storage if args.storage else f"sqlite:///{os.path.join(args.study_dir, 'optuna_study.db')}"
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=args.epochs // 3, interval_steps=1) if args.pruning else None
    
    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="maximize",
        pruner=pruner, load_if_exists=True # Allows resuming interrupted studies
    )
    study.set_user_attr("best_val_acc_overall", float('-inf')) # Initialize user attribute

    try:
        study.optimize(
            lambda trial: objective(trial, args, device, meta_train_dataset_full, meta_val_dataset_full, preprocess_fn),
            n_trials=args.n_trials, timeout=None # No global timeout
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    except Exception as e:
        print(f"Unexpected error during study.optimize: {e}")
        import traceback
        traceback.print_exc()
    
    joblib.dump(study, os.path.join(args.study_dir, "study_results.pkl"))
    
    print("\nOptimization Completed!")
    if study.trials: # Check if there are trials before accessing best_trial
        best_trial_overall = None
        try: # Try to find best trial based on actual value, not just the last one
            best_trial_overall = study.best_trial
            print(f"Best meta-validation accuracy in trial #{best_trial_overall.number}: {best_trial_overall.value:.4f}")
            print(f"Best trial parameters: {best_trial_overall.params}")
            
            # Save best trial parameters from study
            with open(os.path.join(args.study_dir, "final_best_trial_params.json"), 'w') as f:
                json.dump({**best_trial_overall.params, "val_accuracy": best_trial_overall.value}, f, indent=4)

            # Save overall best model for use in other files
            best_model_path = os.path.join(args.study_dir, f"best_model_trial_{best_trial_overall.number}.pt")
            if os.path.exists(best_model_path):
                # Load best trial checkpoint
                best_checkpoint = torch.load(best_model_path, map_location="cpu")
                
                # Save best model in main directory for easy access
                main_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                best_model_final_path = os.path.join(main_project_dir, "best_style_adapted_clip_artgraph.pt")
                
                # Use a copy of checkpoint to avoid accidental modifications
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
                
                # Save final checkpoint
                torch.save(final_checkpoint, best_model_final_path)
                print(f"\nBest model saved in: {best_model_final_path}")
                print("This model can be used in other scripts via the name 'CustomRN50' in clip.load()")
                
                # Create a copy also in code directory to be sure
                alternative_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_adapted_clip_artgraph.pt")
                torch.save(final_checkpoint, alternative_path)
                print(f"Backup copy saved in: {alternative_path}")
            else:
                print(f"WARNING: Best model file {best_model_path} not found!")

        except ValueError: # Can happen if no trial was completed successfully
            print("No successfully completed trials found in study.")
        except Exception as e:
            print(f"Error getting best trial: {e}")

        # Also check parameters saved in best_params_overall.json (updated during trials)
        best_params_file = os.path.join(args.study_dir, "best_params_overall.json")
        if os.path.exists(best_params_file):
            with open(best_params_file, 'r') as f:
                overall_best = json.load(f)
            print(f"\nBest parameters recorded during search (from best_params_overall.json):")
            print(f"Accuracy: {overall_best.get('val_accuracy', 'N/A'):.4f}")
            print(f"Parameters: {overall_best}")

    else:
        print("No trials executed in study.")
        
    print(f"Detailed results and logs saved in: {args.study_dir}")
    
    try:
        save_optuna_visualizations(study, args.study_dir)
    except Exception as e:
        print(f"Error generating final visualizations: {e}")

if __name__ == "__main__":
    main()