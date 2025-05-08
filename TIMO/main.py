import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from models import *


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', dest='shot', type=int, default=1, help='shots number')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    parser.add_argument('--dbg', dest='dbg', type=float, default=0, help='debug mode')
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()
    return args


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['shots'] = args.shot
    cfg['seed'] = args.seed
    cfg['dbg'] = args.dbg
    print("shots", cfg['shots'])
    print("seed", cfg['seed'])
    print("dbg", cfg['dbg'])
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    cache_dir = os.path.join(f'./caches/{cfg["backbone"]}/{cfg["seed"]}/{cfg["dataset"]}')
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    print(cfg)

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights_cupl_all = torch.load(cfg['cache_dir'] + "/text_weights_cupl_t_all.pt", weights_only=False)
    cate_num, prompt_cupl_num, dim = clip_weights_cupl_all.shape
    clip_weights_cupl = clip_weights_cupl_all.mean(dim=1).t()
    clip_weights_cupl = clip_weights_cupl / clip_weights_cupl.norm(dim=0, keepdim=True)
    
    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = load_few_shot_feature(cfg)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = loda_val_test_feature(cfg, "val")
    
    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    if cfg['dataset'] == 'imagenet':
        test_features, test_labels = loda_val_test_feature(cfg, "val")
    else:
        test_features, test_labels = loda_val_test_feature(cfg, "test")


    # ------------------------------------------ Fusion ------------------------------------------
    
    image_weights_all = torch.stack([cache_keys.t()[torch.argmax(cache_values, dim=1)==i] for i in range(cate_num)])
    image_weights = image_weights_all.mean(dim=1)
    image_weights = image_weights / image_weights.norm(dim=1, keepdim=True) 
    
    clip_weights_IGT, matching_score = image_guide_text(cfg, 
        clip_weights_cupl_all, image_weights, return_matching=True)
    clip_weights_IGT = clip_weights_IGT.t()
    metric = {}
    
    # ------------------------------------------ Baseline ------------------------------------------
    # Tip-Adapter
    acc_free = run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, 
        test_features, test_labels, clip_weights_cupl)
    metric['Tip_Adapter'] = acc_free
    
    # APE
    acc_free = APE(cfg, cache_keys, cache_values, val_features, val_labels,  
        test_features, test_labels, clip_weights_cupl)
    metric['APE'] = acc_free
    
    # GDA-CLIP
    acc_free = GDA_CLIP(cfg, val_features, val_labels, test_features, test_labels, clip_weights_cupl)
    metric['GDA_CLIP'] = acc_free
    
    # ------------------------------------------ Ours ------------------------------------------
    # TIMO   
    acc_free = TIMO(cfg, val_features, val_labels, test_features, test_labels, 
        clip_weights_IGT, clip_weights_cupl_all, matching_score,
        grid_search=False, is_print=True)
    metric['TIMO'] = acc_free

    # TIMO-S
    clip_weights_IGT, matching_score = image_guide_text_search(cfg, 
        clip_weights_cupl_all, val_features, val_labels, image_weights)
    acc_free = TIMO(cfg, val_features, val_labels, test_features, test_labels, 
        clip_weights_IGT, clip_weights_cupl_all, matching_score, 
        grid_search=True, n_quick_search=10, is_print=True)
    metric['TIMO_S'] = acc_free

    save_log(cfg, metric)
    
    
    
if __name__ == '__main__':
    main()