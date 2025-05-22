import os
import random
import yaml
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import csv

from datasets import build_dataset
from datasets.imagenet import ImageNet
from datasets.utils import build_data_loader
import clip
from utils import *
from models import *
from extract_features_all import *

# Parametri da iterare
DATASETS = ['artgraph']  # aggiungi altri dataset se vuoi
#BACKBONES = ["RN50","RN101", "ViT-B/32", "ViT-B/16","CustomRN50"]
BACKBONES=["Custom_FSL1_RN50","Custom_FSL_RN50","RN50"]
SEEDS = [1,2,3,4,5,6,7,8,9,42]             #42
K_SHOTS = [1, 2, 4, 8, 16]

def estrai_e_carica_feature(cfg, backbone, seed, k, preprocess):
    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)
    cfg['shots'] = k
    cfg['backbone'] = backbone
    cfg['seed'] = seed

    # Prepara cartelle
    cache_dir = os.path.join(f'./caches/{backbone}/{seed}', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    # Verifica se le feature sono già state estratte
    keys_file = os.path.join(cache_dir, f'keys_{k}shots.pt')
    values_file = os.path.join(cache_dir, f'values_{k}shots.pt')
    vecs_file = os.path.join(cache_dir, f'{k}_vecs_f.pt')
    val_features_file = os.path.join(cache_dir, 'val_f.pt')
    test_features_file = os.path.join(cache_dir, 'test_f.pt') if cfg['dataset'] != 'imagenet' else None
    text_weights_file = os.path.join(cache_dir, 'text_weights_t.pt')
    text_weights_gpt_file = os.path.join(cache_dir, 'text_weights_gpt_t.pt')
    text_weights_all_file = os.path.join(cache_dir, 'text_weights_cupl_t_all.pt')
    
    # Carica modello CLIP solo se necessario
    files_to_extract = False
    required_files = [keys_file, values_file, vecs_file, val_features_file, text_weights_file, 
                     text_weights_gpt_file, text_weights_all_file]
    if cfg['dataset'] != 'imagenet':
        required_files.append(test_features_file)
    
    for file in required_files:
        if file and not os.path.exists(file):
            files_to_extract = True
            break
    
    if files_to_extract:
        print(f"\n[Estrazione necessaria] {cfg['dataset']} | {backbone} | seed={seed} | shots={k}")
        
        # Carica modello CLIP
        clip_model, _ = clip.load(backbone)
        clip_model.eval()
    
        # Prepara dataset e loader
        if cfg['dataset'] == 'imagenet':
            dataset = ImageNet(cfg['root_path'], cfg['shots'], preprocess)
            val_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=0, shuffle=False)
            train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=0, shuffle=False)
        else:
            data_path = os.path.join(os.getcwd(), '$DATA/')
            dataset = build_dataset(cfg['dataset'], data_path, k)
            val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
            test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
            train_tranform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
            train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    
        # Estrazione feature (controlla file per file)
        if not os.path.exists(keys_file) or not os.path.exists(values_file):
            print(f"Estraendo feature few-shot...")
            extract_few_shot_feature(cfg, clip_model, train_loader_cache)
        
        if not os.path.exists(vecs_file):
            print(f"Estraendo feature few-shot all...")
            extract_few_shot_feature_all(cfg, clip_model, train_loader_cache, norm=True)
        
        if not os.path.exists(val_features_file):
            print(f"Estraendo feature val...")
            extract_val_test_feature(cfg, "val", clip_model, val_loader, norm=True)
        
        if cfg['dataset'] != 'imagenet' and not os.path.exists(test_features_file):
            print(f"Estraendo feature test...")
            extract_val_test_feature(cfg, "test", clip_model, test_loader, norm=True)
        
        if not os.path.exists(text_weights_file):
            print(f"Estraendo feature text (no GPT)...")
            extract_text_feature(cfg, dataset.classnames, [dataset.cupl_path], clip_model, dataset.template, use_gpt_prompt=False)
        
        if not os.path.exists(text_weights_gpt_file):
            print(f"Estraendo feature text (GPT)...")
            extract_text_feature(cfg, dataset.classnames, [dataset.cupl_path], clip_model, dataset.template)
        
        if not os.path.exists(text_weights_all_file):
            print(f"Estraendo feature text all...")
            extract_text_feature_all(cfg, dataset.classnames, [dataset.cupl_path], clip_model, dataset.template, norm=True)
    else:
        print(f"\n[Feature già estratte] {cfg['dataset']} | {backbone} | seed={seed} | shots={k}")
        # Carica il dataset solo per ottenere le classi (necessario per il resto del codice)
        if cfg['dataset'] == 'imagenet':
            dataset = ImageNet(cfg['root_path'], cfg['shots'], preprocess)
        else:
            data_path = os.path.join(os.getcwd(), '$DATA/')
            dataset = build_dataset(cfg['dataset'], data_path, k)

    # Caricamento feature per il task few-shot (come in main.py)
    print("Caricamento feature per elaborazione...")
    clip_weights_cupl_all = torch.load(cfg['cache_dir'] + "/text_weights_cupl_t_all.pt", weights_only=False)
    cate_num, prompt_cupl_num, dim = clip_weights_cupl_all.shape
    clip_weights_cupl = clip_weights_cupl_all.mean(dim=1).t()
    clip_weights_cupl = clip_weights_cupl / clip_weights_cupl.norm(dim=0, keepdim=True)
    cache_keys, cache_values = load_few_shot_feature(cfg)
    val_features, val_labels = loda_val_test_feature(cfg, "val")
    if cfg['dataset'] == 'imagenet':
        test_features, test_labels = loda_val_test_feature(cfg, "val")
    else:
        test_features, test_labels = loda_val_test_feature(cfg, "test")

    # Fusion
    image_weights_all = torch.stack([cache_keys.t()[torch.argmax(cache_values, dim=1)==i] for i in range(cate_num)])
    image_weights = image_weights_all.mean(dim=1)
    image_weights = image_weights / image_weights.norm(dim=1, keepdim=True)
    clip_weights_IGT, matching_score = image_guide_text(cfg, clip_weights_cupl_all, image_weights, return_matching=True)
    clip_weights_IGT = clip_weights_IGT.t()
    metric = {}
    if backbone in ["Custom_FSL_RN50", "Custom_FSL1_RN50"]:
        cache_keys, cache_values, val_features, test_features, clip_weights_cupl = ensure_float32_tensors(
        cache_keys, cache_values, val_features, test_features, clip_weights_cupl)

    # Baseline e metodi
    metric['Tip_Adapter'] = run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights_cupl)
    metric['APE'] = APE(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights_cupl)
    metric['GDA_CLIP'] = GDA_CLIP(cfg, val_features, val_labels, test_features, test_labels, clip_weights_cupl)
    metric['TIMO'] = TIMO(cfg, val_features, val_labels, test_features, test_labels, clip_weights_IGT, clip_weights_cupl_all, matching_score, grid_search=False, is_print=True)
    clip_weights_IGT, matching_score = image_guide_text_search(cfg, clip_weights_cupl_all, val_features, val_labels, image_weights)
    metric['TIMO_S'] = TIMO(cfg, val_features, val_labels, test_features, test_labels, clip_weights_IGT, clip_weights_cupl_all, matching_score, grid_search=True, n_quick_search=10, is_print=True)

    return metric

if __name__ == '__main__':
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, 'results_all.csv')
    csv_header = ['SEED', 'SHOTS', 'DATASET', 'MODEL', 'BACKBONE', 'ACC', 'PRECISION_MACRO', 'RECALL_MACRO', 'F1_MACRO']

    # Controlla se il file esiste e se è vuoto per decidere se scrivere l'header
    file_exists = os.path.isfile(csv_file_path)
    write_header = not file_exists

    # Se il file esiste ma è vuoto, dobbiamo comunque scrivere l'header
    if file_exists:
        with open(csv_file_path, 'r') as csvfile:
            write_header = len(csvfile.read().strip()) == 0

    for backbone in BACKBONES:
        for seed in SEEDS:
            torch.manual_seed(seed)
            random.seed(seed)
            for dataset_name in DATASETS:
                cfg = yaml.load(open(f'configs/{dataset_name}.yaml', 'r'), Loader=yaml.Loader)
                for k in K_SHOTS:
                    clip_model, preprocess = clip.load(backbone)
                    clip_model.eval()
                    metrics_dict = estrai_e_carica_feature(cfg, backbone, seed, k, preprocess)
                    
                    # Apri e chiudi il file per ogni esperimento
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        
                        # Scrivi l'header solo se necessario
                        if write_header:
                            writer.writerow(csv_header)
                            write_header = False  # Imposta a False per non scrivere più l'header
                        
                        # Scrivi i risultati per ogni metodo
                        for model_name, model_metrics in metrics_dict.items():
                            row = [
                                seed, 
                                k, 
                                dataset_name, 
                                model_name, 
                                backbone,
                                model_metrics['accuracy'], 
                                model_metrics['precision_macro'], 
                                model_metrics['recall_macro'], 
                                model_metrics['f1_macro']
                            ]
                            writer.writerow(row)
                    
                    print(f"Risultati salvati per {dataset_name} | {backbone} | seed={seed} | shots={k}")