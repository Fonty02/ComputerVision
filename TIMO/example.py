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
    """Estrae o carica le feature esistenti dalla cache"""
    # Setup cache directory (stesso path di run_all_fewshot.py)
    cache_dir = os.path.join(f'./caches/{backbone}/{seed}', dataset_name)
    cfg['cache_dir'] = cache_dir
    
    # File necessari
    keys_file = os.path.join(cache_dir, f'keys_{k_shots}shots.pt')
    values_file = os.path.join(cache_dir, f'values_{k_shots}shots.pt')
    vecs_file = os.path.join(cache_dir, f'{k_shots}_vecs_f.pt')
    val_features_file = os.path.join(cache_dir, 'val_f.pt')
    test_features_file = os.path.join(cache_dir, 'test_f.pt')
    text_weights_all_file = os.path.join(cache_dir, 'text_weights_cupl_t_all.pt')
    
    # Controlla se tutti i file esistono
    required_files = [keys_file, values_file, vecs_file, val_features_file, 
                     test_features_file, text_weights_all_file]
    
    files_exist = all(os.path.exists(f) for f in required_files)
    
    if files_exist:
        print(f"✓ Feature già estratte trovate in: {cache_dir}")
        print("Caricamento feature esistenti...")
        
        # Carica solo il dataset per ottenere le informazioni sulle classi
        data_path = os.path.join(os.getcwd(), '$DATA/')
        dataset = build_dataset(dataset_name, data_path, k_shots)
        
        return dataset, True
    else:
        print(f"✗ Feature non trovate. Estrazione necessaria...")
        print(f"File mancanti:")
        for f in required_files:
            if not os.path.exists(f):
                print(f"  - {f}")
        
        # Carica modello CLIP per estrazione
        clip_model, preprocess = clip.load(backbone)
        clip_model.eval()
        
        # Carica dataset
        data_path = os.path.join(os.getcwd(), '$DATA/')
        dataset = build_dataset(dataset_name, data_path, k_shots)
        
        # Crea directory se non esiste
        os.makedirs(cache_dir, exist_ok=True)
        
        # Prepara data loaders
        val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_transform, is_train=True, shuffle=False)
        
        # Estrai feature
        print("Estrazione feature few-shot...")
        extract_few_shot_feature(cfg, clip_model, train_loader_cache)
        
        print("Estrazione feature few-shot all...")
        extract_few_shot_feature_all(cfg, clip_model, train_loader_cache, norm=True)
        
        print("Estrazione feature val...")
        extract_val_test_feature(cfg, "val", clip_model, val_loader, norm=True)
        
        print("Estrazione feature test...")
        extract_val_test_feature(cfg, "test", clip_model, test_loader, norm=True)
        
        print("Estrazione feature testuali...")
        extract_text_feature_all(cfg, dataset.classnames, [dataset.cupl_path], clip_model, dataset.template, norm=True)
        
        return dataset, False

def create_episode_from_cache(cfg, selected_classes, n_shots=2, n_queries=1):
    """Crea un episodio selezionando dalle feature in cache"""
    
    # Determina il device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carica tutte le feature dalla cache
    clip_weights_cupl_all = torch.load(os.path.join(cfg['cache_dir'], 'text_weights_cupl_t_all.pt'), weights_only=False).to(device)
    cache_keys, cache_values = load_few_shot_feature(cfg)
    cache_keys = cache_keys.to(device)
    cache_values = cache_values.to(device)
    
    test_features, test_labels = loda_val_test_feature(cfg, "test")
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    # Filtra le feature testuali per le classi selezionate
    episode_clip_weights_cupl_all = clip_weights_cupl_all[selected_classes]
    
    # Crea mapping globale -> locale per le classi
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(selected_classes)}
    
    # Seleziona feature di supporto per le classi dell'episodio
    episode_cache_keys = []
    episode_cache_values = []
    episode_support_indices = []
    
    for local_idx, global_class_idx in enumerate(selected_classes):
        # Trova le feature di supporto per questa classe
        class_mask = torch.argmax(cache_values, dim=1) == global_class_idx
        class_features = cache_keys.t()[class_mask]
        class_indices = torch.where(class_mask)[0]
        
        if len(class_features) >= n_shots:
            # Prendi le prime n_shots feature per questa classe
            selected_features = class_features[:n_shots]
            selected_indices = class_indices[:n_shots]
            
            episode_cache_keys.append(selected_features.t())
            episode_support_indices.extend(selected_indices.cpu().tolist())  # Converti in CPU per gli indici
            
            # Crea one-hot encoding locale
            local_one_hot = torch.zeros(n_shots, len(selected_classes), device=device)
            local_one_hot[:, local_idx] = 1.0
            episode_cache_values.append(local_one_hot)
        else:
            raise ValueError(f"Classe {global_class_idx} ha solo {len(class_features)} campioni, ma servono {n_shots}")
    
    # Combina le feature dell'episodio
    episode_cache_keys = torch.cat(episode_cache_keys, dim=1)
    episode_cache_values = torch.cat(episode_cache_values, dim=0)
    
    # Seleziona feature di query per le classi dell'episodio
    episode_query_features = []
    episode_query_labels = []
    episode_query_global_labels = []
    
    for local_idx, global_class_idx in enumerate(selected_classes):
        class_mask = test_labels == global_class_idx
        class_test_features = test_features[class_mask]
        
        if len(class_test_features) >= n_queries:
            selected_query = class_test_features[:n_queries]
            episode_query_features.append(selected_query)
            episode_query_labels.extend([local_idx] * n_queries)  # Label locale
            episode_query_global_labels.extend([global_class_idx] * n_queries)  # Label globale
        else:
            raise ValueError(f"Classe {global_class_idx} ha solo {len(class_test_features)} campioni di test, ma servono {n_queries}")
    
    episode_query_features = torch.cat(episode_query_features, dim=0)
    episode_query_labels = torch.tensor(episode_query_labels, device=device)
    episode_query_global_labels = torch.tensor(episode_query_global_labels, device=device)
    
    return (episode_cache_keys, episode_cache_values, episode_query_features, 
            episode_query_labels, episode_query_global_labels, episode_clip_weights_cupl_all,
            episode_support_indices)

def load_episode_images(dataset, episode_support_indices, episode_query_global_labels, selected_classes):
    """Carica le immagini corrispondenti agli indici dell'episodio"""
    
    # Carica immagini di supporto
    support_images = []
    support_labels = []
    
    for idx in episode_support_indices:
        support_images.append(Image.open(dataset.train_x[idx].impath).convert('RGB'))
        support_labels.append(dataset.train_x[idx].label)
    
    # Carica immagini di query (prendi la prima per ogni classe)
    query_images = []
    query_labels = []
    
    for global_class_idx in selected_classes:
        # Trova la prima immagine di test per questa classe
        for item in dataset.test:
            if item.label == global_class_idx:
                query_images.append(Image.open(item.impath).convert('RGB'))
                query_labels.append(item.label)
                break
    
    return support_images, support_labels, query_images, query_labels

def visualize_episode_results(support_images, support_labels, query_images, query_labels, 
                            predictions, probabilities, selected_classes, classnames):
    """Visualize an example of a few-shot episode with support and query images"""
    
    # Crea una riga con 5 subplot (4 support + 1 query)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Mostra le immagini di support
    for i in range(4):
        axes[i].imshow(support_images[i])
        class_name = classnames[support_labels[i]]
        axes[i].set_title(f'Support {i+1}\nClass: {class_name}', fontsize=12)
        axes[i].axis('off')
    
    # Mostra l'immagine query
    axes[4].imshow(query_images[0])
    true_class = classnames[query_labels[0]]
    pred_class = classnames[selected_classes[predictions[0]]]
    
    # Calcola probabilità per la classe vera
    true_class_local_idx = selected_classes.index(query_labels[0])
    prob_true = probabilities[0, true_class_local_idx]
    
    title = f'Query\nReal: {true_class}\nEstimated: {pred_class}\nProb. real class {prob_true:.3f}'
    axes[4].set_title(title, fontsize=12)
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig('episode_results.png', dpi=300, bbox_inches='tight')
    plt.show()
def main():
    # Configurazione
    seed = 42
    n_classes = 2
    n_shots = 2
    n_queries = 1
    backbone = "RN50"
    dataset_name = "artgraph"
    
    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Determina il device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    # Carica configurazione
    cfg = yaml.load(open(f'configs/{dataset_name}.yaml', 'r'), Loader=yaml.Loader)
    cfg['shots'] = n_shots
    cfg['backbone'] = backbone
    cfg['seed'] = seed
    
    print(f"=== EPISODIO FEW-SHOT ===")
    print(f"Dataset: {dataset_name}")
    print(f"Backbone: {backbone}")
    print(f"Seed: {seed}")
    print(f"Task: {n_classes}-way {n_shots}-shot con {n_queries} query")
    
    # Estrai o carica feature
    dataset, features_existed = extract_or_load_features(cfg, backbone, seed, n_shots, dataset_name)
    
    # Seleziona classi per l'episodio
    selected_classes = random.sample(range(len(dataset.classnames)), n_classes)
    print(f"\nClassi selezionate: {[dataset.classnames[i] for i in selected_classes]}")
    
    # Crea episodio dalle feature in cache
    print("Creazione episodio dalle feature in cache...")
    (episode_cache_keys, episode_cache_values, episode_query_features, 
     episode_query_labels, episode_query_global_labels, episode_clip_weights_cupl_all,
     episode_support_indices) = create_episode_from_cache(cfg, selected_classes, n_shots, n_queries)
    
    # Carica immagini per visualizzazione
    support_images, support_labels, query_images, query_labels = load_episode_images(
        dataset, episode_support_indices, episode_query_global_labels, selected_classes)
    
    # Prepara i dati per TIMO
    cate_num = len(selected_classes)
    
    # Fusion per IGT - assicurati che tutto sia sullo stesso device
    image_weights_all = torch.stack([episode_cache_keys.t()[torch.argmax(episode_cache_values, dim=1)==i] for i in range(cate_num)])
    image_weights = image_weights_all.mean(dim=1)
    image_weights = image_weights / image_weights.norm(dim=1, keepdim=True)
    
    # Esegui TIMO-S
    print("Esecuzione TIMO-S...")
    clip_weights_IGT, matching_score = image_guide_text_search(
        cfg, episode_clip_weights_cupl_all, episode_query_features, episode_query_labels, image_weights)
    
    results = TIMO(cfg, episode_query_features, episode_query_labels, episode_query_features, episode_query_labels, 
                   clip_weights_IGT, episode_clip_weights_cupl_all, matching_score, 
                   grid_search=True, n_quick_search=10, is_print=True)
    
    print(f"\nRisultati TIMO-S: {results}")
    
    # Calcola predizioni e probabilità per visualizzazione
    with torch.no_grad():
        logits = episode_query_features @ clip_weights_IGT
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    # Visualizza risultati
    print("Creazione visualizzazione...")
    visualize_episode_results(
        support_images, support_labels, query_images, query_labels,
        predictions.cpu().numpy(), probabilities.cpu().numpy(), selected_classes, dataset.classnames)
    
    print("Esperimento completato!")

if __name__ == '__main__':
    main()