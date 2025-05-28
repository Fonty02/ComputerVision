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
    # Distribuisci le n_queries tra le classi selezionate
    episode_query_features = []
    episode_query_labels = []
    episode_query_global_labels = []
    
    # Calcola quante query per classe (distribuzione equa)
    queries_per_class = max(1, n_queries // len(selected_classes))
    extra_queries = n_queries % len(selected_classes)
    
    print(f"Distribuzione query: {queries_per_class} per classe + {extra_queries} extra")
    
    total_queries = 0
    for local_idx, global_class_idx in enumerate(selected_classes):
        # Calcola quante query prendere da questa classe
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
            episode_query_labels.extend([local_idx] * class_queries)  # Label locale
            episode_query_global_labels.extend([global_class_idx] * class_queries)  # Label globale
            total_queries += class_queries
        else:
            raise ValueError(f"Classe {global_class_idx} ha solo {len(class_test_features)} campioni di test, ma servono {class_queries}")
            
        # Se abbiamo raggiunto il totale di query, usciamo dal ciclo
        if total_queries >= n_queries:
            break
    
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
    support_names = []
    
    for idx in episode_support_indices:
        img_path = dataset.train_x[idx].impath
        support_images.append(Image.open(img_path).convert('RGB'))
        support_labels.append(dataset.train_x[idx].label)
        # Estrai il nome del dipinto dal percorso
        support_names.append(os.path.basename(img_path).split('.')[0])
    
    # Carica immagini di query (prendi la prima per ogni classe)
    query_images = []
    query_labels = []
    query_names = []
    used_paths = set()  # Tieni traccia delle immagini già utilizzate
    
    for global_class_idx in episode_query_global_labels:
        # Trova un'immagine di test per questa classe (evitando duplicati)
        found = False
        for item in dataset.test:
            if item.label == global_class_idx.item() and item.impath not in used_paths:
                query_images.append(Image.open(item.impath).convert('RGB'))
                query_labels.append(item.label)
                query_names.append(os.path.basename(item.impath).split('.')[0])
                used_paths.add(item.impath)
                found = True
                break
        
        # Se non abbiamo trovato un'immagine unica, possiamo usarne una già vista
        if not found:
            for item in dataset.test:
                if item.label == global_class_idx.item():
                    query_images.append(Image.open(item.impath).convert('RGB'))
                    query_labels.append(item.label)
                    query_names.append(os.path.basename(item.impath).split('.')[0] + " (duplicata)")
                    break
    
    return support_images, support_labels, support_names, query_images, query_labels, query_names

def visualize_episode_results(support_images, support_labels, support_names, query_images, query_labels, query_names,
                            predictions, probabilities, selected_classes, classnames):
    """Visualize an example of a few-shot episode with support and query images"""
    
    # Gestisce il numero flessibile di immagini
    n_support = len(support_images)  # Dovrebbe essere n_classes * n_shots
    n_query = min(10, len(query_images))  # Limita a 10 query per la visualizzazione
    n_classes = len(selected_classes)
    n_shots_per_class = n_support // n_classes
    
    # Crea un layout con prima colonna per le query e le altre per i support
    n_rows = max(n_query, n_classes * n_shots_per_class)
    n_cols = 1 + n_classes  # 1 colonna per query + 1 colonna per ogni classe
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 20))
    
    # Assicura che axes sia sempre un array 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Organizza support set per classe
    support_by_class = {}
    for i in range(n_support):
        class_label = support_labels[i]
        if class_label not in support_by_class:
            support_by_class[class_label] = []
        support_by_class[class_label].append((support_images[i], class_label))
      # 1. Mostra le immagini di query nella prima colonna
    for i in range(n_query):
        axes[i, 0].imshow(query_images[i])
        
        true_class = classnames[query_labels[i]]
        pred_class = classnames[selected_classes[predictions[i]]]
        painting_name = query_names[i]
        
        # Calcola probabilità per la classe vera
        true_class_local_idx = selected_classes.index(query_labels[i]) if query_labels[i] in selected_classes else -1
        prob_true = probabilities[i, true_class_local_idx] if true_class_local_idx >= 0 else 0.0
        
        title = f'Query {i+1}: {painting_name}\nReale: {true_class}\nStimata: {pred_class}'
        subtitle = f'Prob: {prob_true:.3f}'
        axes[i, 0].set_title(f'{title}\n{subtitle}', fontsize=8)
        axes[i, 0].axis('off')
    
    # Nascondi le celle vuote nella colonna delle query
    for i in range(n_query, n_rows):
        axes[i, 0].axis('off')
    
    # 2. Mostra le immagini di support per classe (una classe per colonna)
    for col, class_idx in enumerate(selected_classes, start=1):
        # Intestazione per la classe
        if col < n_cols:
            axes[0, col].text(0.5, 0.5, f'Classe: {classnames[class_idx]}', 
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=12, fontweight='bold')
            axes[0, col].axis('off')
          # Immagini di support per questa classe
        support_items = support_by_class.get(class_idx, [])
        for j, (img, _) in enumerate(support_items):
            row = j + 1  # +1 perché la riga 0 è l'intestazione
            if row < n_rows and col < n_cols:
                axes[row, col].imshow(img)
                # Trova l'indice corrispondente nell'array originale per ottenere il nome
                orig_idx = support_labels.index(class_idx) + j
                if orig_idx < len(support_names):
                    name = support_names[orig_idx]
                    axes[row, col].set_title(f'Support {j+1}: {name}', fontsize=8)
                else:
                    axes[row, col].set_title(f'Support {j+1}', fontsize=9)
                axes[row, col].axis('off')
    
    # Nascondi tutte le celle vuote rimanenti
    for row in range(n_rows):
        for col in range(1, n_cols):
            # Saltiamo la prima riga di ogni colonna (usata per l'intestazione)
            if row > 0 and row <= n_shots_per_class:
                continue
            # Per il resto, nascondiamo
            if not axes[row, col].has_data():
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('episode_results.png', dpi=300, bbox_inches='tight')
    plt.show()
def main():
    # Configurazione
    seed = 42
    n_classes = 5  # 5 classi
    n_shots = 4   # 4 elementi per classe nel support set
    n_queries = 10  # 10 query in totale
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
    support_images, support_labels, support_names, query_images, query_labels, query_names = load_episode_images(
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
        support_images, support_labels, support_names, query_images, query_labels, query_names,
        predictions.cpu().numpy(), probabilities.cpu().numpy(), selected_classes, dataset.classnames)
    
    print("Esperimento completato!")

if __name__ == '__main__':
    main()