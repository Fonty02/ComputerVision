import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def cal_criterion(cfg, clip_weights, cache_keys, only_use_txt=True, training_free=True, force=False, with_IE=False):
    
    feat_dim, cate_num = clip_weights.shape
    text_feat = clip_weights.t().unsqueeze(1)
    cache_feat = cache_keys.reshape(cate_num, -1, feat_dim)
    
    save_path = f'./caches/{cfg["backbone"]}/{cfg["seed"]}/{cfg["dataset"]}'
    if with_IE:
        save_file = '{}/criterion_{}_{}shot_IE.pt'.format(save_path, cfg['backbone'].replace('/', ''), cfg['shots'])
    else:
        save_file = '{}/criterion_{}_{}shot_Pure.pt'.format(save_path, cfg['backbone'].replace('/', ''), cfg['shots'])
        
    if os.path.exists(save_file) and not force:
        print('Loading criterion...')
        sim = torch.load(save_file, weights_only=False)
    elif only_use_txt:
        print('Calculating criterion...')
        
        feats = text_feat.squeeze()
        
        sim_sum = torch.zeros((feat_dim)).cuda()
        count = 0
        for i in range(cate_num):
            for j in range(cate_num):
                if i != j:
                    sim_sum += feats[i, :] * feats[j, :]
                    count += 1
        sim = sim_sum / count
        torch.save(sim, save_file)
    else:
        print('Calculating criterion...')
        
        feats = torch.cat([text_feat, cache_feat], dim=1)
        samp_num = feats.shape[1]
        
        sim_sum = torch.zeros((feat_dim)).cuda()
        count = 0
        for i in range(cate_num):
            for j in range(cate_num):
                for m in range(samp_num):
                    for n in range(samp_num):
                        if i != j:
                            sim_sum += feats[i, m, :] * feats[j, n, :]
                            count += 1
        sim = sim_sum / count
        torch.save(sim, save_file)

    criterion = (-1) * cfg['w'][0] * sim + cfg['w'][1] * torch.var(clip_weights, dim=1)
    
    ratio = 1024 / feat_dim
    if training_free:
        _, indices = torch.topk(criterion, k=int(cfg['training_free_feat_num'] // ratio))
    else: 
        _, indices = torch.topk(criterion, k=int(cfg['training_feat_num'] // ratio))
    return indices


def load_text_feature(cfg):
    save_path = cfg['cache_dir'] + "/text_weights_gpt_t.pt"
    clip_weights = torch.load(save_path, weights_only=False)
    return clip_weights


def load_few_shot_feature(cfg, norm=True):
    if norm:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt", weights_only=False)
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt", weights_only=False)
    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots_unnormed.pt", weights_only=False)
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots_unnormed.pt", weights_only=False)
    return cache_keys, cache_values


def loda_val_test_feature(cfg, split, norm=True):
    if norm:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt", weights_only=False)
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt", weights_only=False)
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f_unnormed.pt", weights_only=False)
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l_unnormed.pt", weights_only=False)
    return features, labels


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def accuracy(shot_logits, cache_values, topk=(1,)):
    target = cache_values.topk(max(topk), 1, True, True)[1].squeeze()
    pred = shot_logits.topk(max(topk), 1, True, True)[1].squeeze()
    idx = (target != pred)
    return idx


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


# t_features [c,p,d]
# s_features [c,n,d] or [c,d]
def image_guide_text(cfg, t_features, s_features, gamma=-1, 
        return_weights=False, return_matching=False):
    t_features = t_features / t_features.norm(dim=-1, keepdim=True)
    
    if gamma == -1:
        if cfg['dataset'] == "imagenet":
            gamma = 1
        elif cfg['dataset'] == "oxford_flowers":
            gamma = 100
        else:
            gamma = 50
        
    cate_num, prompt_num, feat_dim = t_features.shape # c, p, d
    if len(s_features.shape) == 3:
        s_features = s_features.mean(dim=1) # c,d
        s_features = s_features / s_features.norm(dim=-1, keepdim=True)
    weights = torch.ones(cate_num, prompt_num).to(t_features.dtype).to(t_features.device) # c, p
    s_features = s_features.to(t_features.dtype)    
    t_features = t_features / t_features.norm(dim=-1, keepdim=True)
    
    matching_score = []
    for c in range(cate_num):
        # weights[c:c+1] # 1, p
        # t_features[c] # p, d
        # s_features[c:c+1] # 1, d
        weights[c] = (s_features[c:c+1] @ t_features[c].t()).squeeze(dim=0)
        matching_score.append(weights[c].clone())
        weights[c] = weights[c] / weights[c].norm() 
        weights[c] = F.softmax(weights[c] * gamma, dim=0)
    matching_score = torch.stack(matching_score, dim=0) # N, P
    
    for weights in [weights]:
        normed_weights = weights
        normed_clip_weights = torch.einsum("cp, cpd-> cd", normed_weights, t_features)
        normed_clip_weights = normed_clip_weights / normed_clip_weights.norm(dim=-1, keepdim=True)

    
    if return_matching:
        return normed_clip_weights, matching_score
    elif return_weights:
        return normed_clip_weights, normed_weights
    else:
        return normed_clip_weights


def image_guide_text_search(cfg, clip_weights_cupl_all, val_features, val_labels, image_weights):
    best_acc = 0
    best_gamma = 0
    for gamma in range(5, 101, 5):
        clip_weights_cupl_IGT, matching_score = image_guide_text(cfg, 
            clip_weights_cupl_all, image_weights, return_matching=True, gamma=gamma)
        clip_weights_cupl_IGT = clip_weights_cupl_IGT.t() # D, C
        
        val_logits = val_features @ clip_weights_cupl_IGT # N, C
        acc = (val_logits.argmax(-1) == val_labels).sum() / len(val_labels)
        
        if acc > best_acc:
            best_acc = acc
            best_gamma = gamma
    print("best_gamma:", best_gamma)
    clip_weights_cupl_IGT, matching_score = image_guide_text(cfg, 
        clip_weights_cupl_all, image_weights, return_matching=True, gamma=best_gamma)
    clip_weights_cupl_IGT = clip_weights_cupl_IGT.t()
    return clip_weights_cupl_IGT,matching_score


# anchor K,D
def vec_sort(vecs_t, matching_score):
    cate_num, prompt_num, dim = vecs_t.shape # N,P,D
    
    weights, sorted_idx = torch.topk(matching_score, k=prompt_num, dim=-1)
    sort_vecs_t = []
    for c in range(cate_num):
        sort_vecs_t.append(vecs_t[c][sorted_idx[c]].clone())
    sort_vecs_t = torch.stack(sort_vecs_t, dim=0)
    
    if len(sort_vecs_t.shape) == 2:
        sort_vecs_t = sort_vecs_t.unsqueeze(1)
        
    return sort_vecs_t, weights


def save_log(cfg, metric:dict):
        for key in metric.keys():
            with open(f'outputs/{key}.txt', 'a') as f:
                f.write(f"{cfg['dataset']}_{cfg['shots']}_{cfg['seed']}: {metric[key]}\n")
                