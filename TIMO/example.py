import os
import random
import yaml
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader
from utils import *
from models import *
from extract_features_all import *

def extract_or_load_features(cfg, backbone, seed, k_shots, dataset_name):
    """Extracts or loads existing features from cache"""
    # Setup cache directory (same path as run_all_fewshot.py)
    cache_dir = os.path.join(f'./caches/{backbone}/{seed}', dataset_name)
    cfg['cache_dir'] = cache_dir
    
    # Required files
    keys_file = os.path.join(cache_dir, f'keys_{k_shots}shots.pt')
    values_file = os.path.join(cache_dir, f'values_{k_shots}shots.pt')
    vecs_file = os.path.join(cache_dir, f'{k_shots}_vecs_f.pt')
    val_features_file = os.path.join(cache_dir, 'val_f.pt')
    test_features_file = os.path.join(cache_dir, 'test_f.pt')
    text_weights_all_file = os.path.join(cache_dir, 'text_weights_cupl_t_all.pt')
      # Check if all files exist
    required_files = [keys_file, values_file, vecs_file, val_features_file, 
                     test_features_file, text_weights_all_file]
    
    files_exist = all(os.path.exists(f) for f in required_files)
    
    if files_exist:
        print(f"✓ Already extracted features found in: {cache_dir}")
        print("Loading existing features...")
        
        # Load only dataset to get class information
        data_path = os.path.join(os.getcwd(), '$DATA/')
        dataset = build_dataset(dataset_name, data_path, k_shots)
        
        return dataset, True
    else:
        print(f"✗ Features not found. Extraction needed...")
        print(f"Missing files:")
        for f in required_files:
            if not os.path.exists(f):
                print(f"  - {f}")
        
        # Load CLIP model for extraction
        clip_model, preprocess = clip.load(backbone)
        clip_model.eval()
        
        # Load dataset
        data_path = os.path.join(os.getcwd(), '$DATA/')
        dataset = build_dataset(dataset_name, data_path, k_shots)
        
        # Create directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Prepare data loaders
        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_transform, is_train=True, shuffle=False)
        
        # Extract features
        print("Extracting few-shot features...")
        extract_few_shot_feature(cfg, clip_model, train_loader_cache)
        
        print("Extracting few-shot features all...")
        extract_few_shot_feature_all(cfg, clip_model, train_loader_cache, norm=True)
        
        print("Extracting val features...")
        extract_val_test_feature(cfg, "val", clip_model, val_loader, norm=True)
        
        print("Extracting test features...")
        extract_val_test_feature(cfg, "test", clip_model, test_loader, norm=True)
        
        print("Extracting text features...")
        extract_text_feature_all(cfg, dataset.classnames, [dataset.cupl_path], clip_model, dataset.template, norm=True)
        
        return dataset, False

def create_episode_from_cache(cfg, selected_classes, n_shots=2, n_queries=1):
    """Creates an episode by selecting from cached features"""
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load all features from cache
    clip_weights_cupl_all = torch.load(os.path.join(cfg['cache_dir'], 'text_weights_cupl_t_all.pt'), weights_only=False).to(device)
    cache_keys, cache_values = load_few_shot_feature(cfg)
    cache_keys = cache_keys.to(device)
    cache_values = cache_values.to(device)
    
    test_features, test_labels = loda_val_test_feature(cfg, "test")
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    # Filter text features for selected classes
    episode_clip_weights_cupl_all = clip_weights_cupl_all[selected_classes]
    
    # Create global -> local mapping for classes
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(selected_classes)}
    
    # Select support features for episode classes
    episode_cache_keys = []
    episode_cache_values = []
    episode_support_indices = []
    
    for local_idx, global_class_idx in enumerate(selected_classes):
        # Find support features for this class
        class_mask = torch.argmax(cache_values, dim=1) == global_class_idx
        class_features = cache_keys.t()[class_mask]
        class_indices = torch.where(class_mask)[0]
        
        if len(class_features) >= n_shots:
            # Take the first n_shots features for this class
            selected_features = class_features[:n_shots]
            selected_indices = class_indices[:n_shots]
            
            episode_cache_keys.append(selected_features.t())
            episode_support_indices.extend(selected_indices.cpu().tolist())  # Convert to CPU for indices
            
            # Create local one-hot encoding
            local_one_hot = torch.zeros(n_shots, len(selected_classes), device=device)
            local_one_hot[:, local_idx] = 1.0
            episode_cache_values.append(local_one_hot)
        else:
            raise ValueError(f"Class {global_class_idx} has only {len(class_features)} samples, but need {n_shots}")
    
    # Combine episode features
    episode_cache_keys = torch.cat(episode_cache_keys, dim=1)
    episode_cache_values = torch.cat(episode_cache_values, dim=0)
      # Select query features for episode classes
    # Distribute n_queries among selected classes
    episode_query_features = []
    episode_query_labels = []
    episode_query_global_labels = []
    
    # Calculate how many queries per class (equal distribution)
    queries_per_class = max(1, n_queries // len(selected_classes))
    extra_queries = n_queries % len(selected_classes)
    
    print(f"Query distribution: {queries_per_class} per class + {extra_queries} extra")
    
    total_queries = 0
    for local_idx, global_class_idx in enumerate(selected_classes):
        # Calculate how many queries to take from this class
        class_queries = queries_per_class
        if local_idx < extra_queries:
            class_queries += 1
            
        if total_queries + class_queries > n_queries:
            class_queries = n_queries - total_queries
            
        if class_queries <= 0:
            break
            
        class_mask = test_labels == global_class_idx
        class_test_features = test_features[class_mask]
        
        if len(class_test_features) >= class_queries:
            selected_query = class_test_features[:class_queries]
            episode_query_features.append(selected_query)
            episode_query_labels.extend([local_idx] * class_queries)  # Local label
            episode_query_global_labels.extend([global_class_idx] * class_queries)  # Global label
            total_queries += class_queries
        else:
            raise ValueError(f"Class {global_class_idx} has only {len(class_test_features)} test samples, but need {class_queries}")
            
        # If we've reached the total queries, exit the loop
        if total_queries >= n_queries:
            break
    
    episode_query_features = torch.cat(episode_query_features, dim=0)
    episode_query_labels = torch.tensor(episode_query_labels, device=device)
    episode_query_global_labels = torch.tensor(episode_query_global_labels, device=device)
    
    return (episode_cache_keys, episode_cache_values, episode_query_features, 
            episode_query_labels, episode_query_global_labels, episode_clip_weights_cupl_all,
            episode_support_indices)

def load_episode_images(dataset, episode_support_indices, episode_query_global_labels, selected_classes):
    """Loads images corresponding to episode indices"""
    
    # Load support images
    support_images = []
    support_labels = []
    support_names = []
    
    for idx in episode_support_indices:
        img_path = dataset.train_x[idx].impath
        support_images.append(Image.open(img_path).convert('RGB'))
        support_labels.append(dataset.train_x[idx].label)
        # Extract painting name from path
        support_names.append(os.path.basename(img_path).split('.')[0])
    
    # Load query images (take the first for each class)
    query_images = []
    query_labels = []
    query_names = []
    used_paths = set()  # Keep track of already used images
    
    for global_class_idx in episode_query_global_labels:
        # Find a test image for this class (avoiding duplicates)
        found = False
        for item in dataset.test:
            if item.label == global_class_idx.item() and item.impath not in used_paths:
                query_images.append(Image.open(item.impath).convert('RGB'))
                query_labels.append(item.label)
                query_names.append(os.path.basename(item.impath).split('.')[0])
                used_paths.add(item.impath)
                found = True
                break
        
        # If we haven't found a unique image, we can use one already seen
        if not found:
            for item in dataset.test:
                if item.label == global_class_idx.item():
                    query_images.append(Image.open(item.impath).convert('RGB'))
                    query_labels.append(item.label)
                    query_names.append(os.path.basename(item.impath).split('.')[0] + " (duplicate)")
                    break
    
    return support_images, support_labels, support_names, query_images, query_labels, query_names

def visualize_episode_results(support_images, support_labels, support_names, query_images, query_labels, query_names,
                            predictions, probabilities, selected_classes, classnames):
    """Visualize an example of a few-shot episode with support and query images"""
    
    # Handle flexible number of images
    n_support = len(support_images)  # Should be n_classes * n_shots
    n_query = min(10, len(query_images))  # Limit to 10 queries for visualization
    n_classes = len(selected_classes)
    n_shots_per_class = n_support // n_classes
    
    # Create layout with first column for queries and others for support
    n_rows = max(n_query, n_classes * n_shots_per_class)
    n_cols = 1 + n_classes  # 1 column for queries + 1 column for each class
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 20))
    
    # Ensure axes is always a 2D array
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Organize support set by class
    support_by_class = {}
    for i in range(n_support):
        class_label = support_labels[i]
        if class_label not in support_by_class:
            support_by_class[class_label] = []
        support_by_class[class_label].append((support_images[i], class_label))
      # 1. Show query images in first column
    for i in range(n_query):
        axes[i, 0].imshow(query_images[i])
        
        true_class = classnames[query_labels[i]]
        pred_class = classnames[selected_classes[predictions[i]]]
        painting_name = query_names[i]
        
        # Calculate probability for true class
        true_class_local_idx = selected_classes.index(query_labels[i]) if query_labels[i] in selected_classes else -1
        prob_true = probabilities[i, true_class_local_idx] if true_class_local_idx >= 0 else 0.0
        
        title = f'Query {i+1}: {painting_name}\nTrue: {true_class}\nPredicted: {pred_class}'
        subtitle = f'Prob: {prob_true:.3f}'
        axes[i, 0].set_title(f'{title}\n{subtitle}', fontsize=8)
        axes[i, 0].axis('off')
    
    # Hide empty cells in query column
    for i in range(n_query, n_rows):
        axes[i, 0].axis('off')
    
    # 2. Show support images by class (one class per column)
    for col, class_idx in enumerate(selected_classes, start=1):
        # Header for class
        if col < n_cols:
            axes[0, col].text(0.5, 0.5, f'Class: {classnames[class_idx]}', 
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=12, fontweight='bold')
            axes[0, col].axis('off')
          # Support images for this class
        support_items = support_by_class.get(class_idx, [])
        for j, (img, _) in enumerate(support_items):
            row = j + 1  # +1 because row 0 is header
            if row < n_rows and col < n_cols:
                axes[row, col].imshow(img)
                # Find corresponding index in original array to get name
                orig_idx = support_labels.index(class_idx) + j
                if orig_idx < len(support_names):
                    name = support_names[orig_idx]
                    axes[row, col].set_title(f'Support {j+1}: {name}', fontsize=8)
                else:
                    axes[row, col].set_title(f'Support {j+1}', fontsize=9)
                axes[row, col].axis('off')
    
    # Hide all remaining empty cells
    for row in range(n_rows):
        for col in range(1, n_cols):
            # Skip first row of each column (used for header)
            if row > 0 and row <= n_shots_per_class:
                continue
            # For the rest, hide them
            if not axes[row, col].has_data():
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('episode_results.png', dpi=300, bbox_inches='tight')
    plt.show()
def main():
    # Configuration
    seed = 42
    n_classes = 5  # 5 classes
    n_shots = 4   # 4 elements per class in support set
    n_queries = 10  # 10 queries in total
    backbone = "RN50"
    dataset_name = "artgraph"
    
    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    cfg = yaml.load(open(f'configs/{dataset_name}.yaml', 'r'), Loader=yaml.Loader)
    cfg['shots'] = n_shots
    cfg['backbone'] = backbone
    cfg['seed'] = seed
    
    print(f"=== FEW-SHOT EPISODE ===")
    print(f"Dataset: {dataset_name}")
    print(f"Backbone: {backbone}")
    print(f"Seed: {seed}")
    print(f"Task: {n_classes}-way {n_shots}-shot with {n_queries} queries")
    
    # Extract or load features
    dataset, features_existed = extract_or_load_features(cfg, backbone, seed, n_shots, dataset_name)
    
    # Select classes for episode
    selected_classes = random.sample(range(len(dataset.classnames)), n_classes)
    print(f"\nSelected classes: {[dataset.classnames[i] for i in selected_classes]}")
    
    # Create episode from cached features
    print("Creating episode from cached features...")
    (episode_cache_keys, episode_cache_values, episode_query_features, 
     episode_query_labels, episode_query_global_labels, episode_clip_weights_cupl_all,
     episode_support_indices) = create_episode_from_cache(cfg, selected_classes, n_shots, n_queries)
      # Load images for visualization
    support_images, support_labels, support_names, query_images, query_labels, query_names = load_episode_images(
        dataset, episode_support_indices, episode_query_global_labels, selected_classes)
    
    # Prepare data for TIMO
    cate_num = len(selected_classes)
    
    # Fusion for IGT - ensure everything is on same device
    image_weights_all = torch.stack([episode_cache_keys.t()[torch.argmax(episode_cache_values, dim=1)==i] for i in range(cate_num)])
    image_weights = image_weights_all.mean(dim=1)
    image_weights = image_weights / image_weights.norm(dim=1, keepdim=True)
    
    # Run TIMO-S
    print("Running TIMO-S...")
    clip_weights_IGT, matching_score = image_guide_text_search(
        cfg, episode_clip_weights_cupl_all, episode_query_features, episode_query_labels, image_weights)
    
    results = TIMO(cfg, episode_query_features, episode_query_labels, episode_query_features, episode_query_labels, 
                   clip_weights_IGT, episode_clip_weights_cupl_all, matching_score, 
                   grid_search=True, n_quick_search=10, is_print=True)
    
    print(f"\nTIMO-S results: {results}")
    
    # Calculate predictions and probabilities for visualization
    with torch.no_grad():
        logits = episode_query_features @ clip_weights_IGT
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
      # Visualize results
    print("Creating visualization...")
    visualize_episode_results(
        support_images, support_labels, support_names, query_images, query_labels, query_names,
        predictions.cpu().numpy(), probabilities.cpu().numpy(), selected_classes, dataset.classnames)
    
    print("Experiment completed!")

if __name__ == '__main__':
    main()