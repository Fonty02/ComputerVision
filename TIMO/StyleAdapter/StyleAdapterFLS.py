import os
import random
import numpy as np
from PIL import Image
import sys

import torch
import torch.nn as nn
import torch.utils.data as data
from typing import List, Dict, Tuple

import clip

# --- Function find_artgraph_path ---
def find_artgraph_path():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_script_dir, "..", "$DATA", "artgraph"),
        os.path.join(os.path.dirname(os.path.dirname(current_script_dir)), "$DATA", "artgraph_complementary"),
        os.path.abspath("/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO/$DATA/artgraph_complementary")
    ]
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            print(f"Artgraph dataset found in: {path}")
            return path
    raise FileNotFoundError("Artgraph dataset not found. Check paths or create directory.")

# --- Class ArtgraphDataset---
class ArtgraphDataset(data.Dataset):
    def __init__(self, root_dir: str, transform=None, seed: int = 42,
                 artist_subset: List[str] = None): 
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        
        random.seed(seed)
        np.random.seed(seed)

        all_available_artists = sorted([d for d in os.listdir(self.images_dir)
                                   if os.path.isdir(os.path.join(self.images_dir, d))])
        if not all_available_artists:
            raise FileNotFoundError(f"No subfolders (class/artist) found in {self.images_dir}")

        if artist_subset:
            self.classnames = [name for name in artist_subset if name in all_available_artists]
            if len(self.classnames) != len(artist_subset):
                print("Warning: Some artists in the provided subset were not found in the dataset.")
        else:
            self.classnames = all_available_artists
        
        if not self.classnames:
            raise ValueError("No artists selected or found for the dataset.")

        self.class_to_idx = {cls_name: i+150 for i, cls_name in enumerate(self.classnames)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        self.samples_by_class: Dict[int, List[str]] = {label: [] for label in self.class_to_idx.values()}
        self.flat_samples: List[Tuple[str, int]] = [] # List of (image_path, global_label)

        for artist_name in self.classnames:
            artist_dir = os.path.join(self.images_dir, artist_name)
            artist_label = self.class_to_idx[artist_name]
            if os.path.isdir(artist_dir):
                img_names_for_artist = [
                    img_name for img_name in os.listdir(artist_dir)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
                ]
                random.shuffle(img_names_for_artist) # Shuffle for K-shot and Q-query
                for img_name in img_names_for_artist:
                    img_path = os.path.join(artist_dir, img_name)
                    self.samples_by_class[artist_label].append(img_path)
                    self.flat_samples.append((img_path, artist_label))
        


        print(f"Loaded dataset from '{root_dir}'. Selected artists: {len(self.classnames)}. Total images for these artists: {len(self.flat_samples)}")
        if len(self.flat_samples) == 0:
            print(f"WARNING: No samples loaded. Check paths and artist_subset.")
    
    def __len__(self):
        return len(self.flat_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.flat_samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Critical error loading image {img_path} at index {idx}: {e}. This could interrupt episodic training.")
            raise IOError(f"Unable to load {img_path}")


# --- Adapter Class ---
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim_override=None, dropout_rate=0.1, use_layernorm=True):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.output_dim = output_dim_override if output_dim_override is not None else input_dim
        self.up_project = nn.Linear(bottleneck_dim, self.output_dim)
        

        nn.init.xavier_uniform_(self.down_project.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.down_project.bias)
        nn.init.xavier_uniform_(self.up_project.weight, gain=0.001)
        nn.init.zeros_(self.up_project.bias)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(bottleneck_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):        
        original_x = x
        x = self.down_project(x)
        if self.use_layernorm:
            x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        if x.shape[-1] == original_x.shape[-1]:
            alpha = 0.2 
            x = alpha * x + original_x
        return x


# --- CLASS: CLIPWithStyleAdapter---
class CLIPWithStyleAdapter(nn.Module):
    def __init__(self, clip_model_name, fusion_bottleneck_dim, gram_style_projection_dim, device, 
                 layers_for_gram_rn50=None, 
                 dropout_rate=0.1, use_layernorm_adapter=True):
        super(CLIPWithStyleAdapter, self).__init__()
        
        if layers_for_gram_rn50 is None:
            layers_for_gram_rn50 = ['layer2', 'layer3']
            
        model_name_to_load = "RN50" if clip_model_name == "CustomRN50" else clip_model_name
        try:
            self.clip_model, self.preprocess = clip.load(model_name_to_load, device=device)
        except Exception as e:
            raise RuntimeError(f"Error loading CLIP model '{model_name_to_load}': {e}") from e

        self.visual = self.clip_model.visual
        self.feature_extractor_hooks = []
        self.extracted_gram_feature_maps = {}
        self.device = device
        self.clip_model_name = clip_model_name 
        self.layers_for_gram_config = layers_for_gram_rn50

        if not self.clip_model_name.startswith("RN") and self.layers_for_gram_config:
            print(f"WARNING: Gram extraction is optimized for RN architectures. Model {self.clip_model_name} might not have the specified layers ('layer1', 'layer2', etc.).")

        self.semantic_feature_dim = self._get_semantic_feature_dim()
        print(f"Semantic feature dimension (encode_image output): {self.semantic_feature_dim}")

        if self.layers_for_gram_config:
            self._register_gram_hooks(self.clip_model.visual)
            self.total_gram_projected_dim, self.gram_layer_projections = self._setup_gram_projections(gram_style_projection_dim)
            print(f"Total projected Gram features dimension: {self.total_gram_projected_dim}")
        else:
            self.total_gram_projected_dim = 0
            self.gram_layer_projections = nn.ModuleDict()
            print("No Gram features will be used (layers_for_gram_config empty or not RN)")

        fusion_input_dim = self.semantic_feature_dim + self.total_gram_projected_dim
        if fusion_input_dim <=0:
            raise ValueError(f"Fusion input dimension is {fusion_input_dim}. Check semantic_feature_dim and total_gram_projected_dim.")

        self.fusion_adapter = Adapter(
            input_dim=fusion_input_dim,
            bottleneck_dim=fusion_bottleneck_dim,
            output_dim_override=self.semantic_feature_dim,
            dropout_rate=dropout_rate,
            use_layernorm=use_layernorm_adapter
        ).to(device)
        
        print(f"Fusion Adapter input dim: {fusion_input_dim}, bottleneck: {fusion_bottleneck_dim}, output dim: {self.semantic_feature_dim}")
        
    def _get_semantic_feature_dim(self):
        try:
            if hasattr(self.clip_model.visual, 'output_dim'):
                return self.clip_model.visual.output_dim
            elif hasattr(self.clip_model, 'visual') and hasattr(self.clip_model.visual, 'proj') and self.clip_model.visual.proj is not None:
                 # For some ViT models, output dimension is after projection
                if isinstance(self.clip_model.visual.proj, torch.Tensor): # ViT-L/14@336px
                     return self.clip_model.visual.proj.shape[1]
                else: # Other ViTs
                     return self.clip_model.visual.proj.out_features
            else: # Fallback for RN and others
                resolution = self.clip_model.visual.input_resolution
                dummy_image = torch.randn(1, 3, resolution, resolution).to(self.device)
                with torch.no_grad():
                    return self.clip_model.encode_image(dummy_image).shape[-1]
        except Exception as e:
            print(f"Error determining semantic_feature_dim: {e}. Using 1024 as fallback.")
            return 1024 # Common value for RN50

    def _setup_gram_projections(self, gram_style_projection_dim):
        if not self.layers_for_gram_config or not self.clip_model_name.startswith("RN"):
            return 0, nn.ModuleDict()
            
        # Ensure there's at least one layer to avoid division by zero
        num_gram_layers = len(self.layers_for_gram_config)
        if num_gram_layers == 0:
            return 0, nn.ModuleDict()
        per_gram_vector_projection_dim = gram_style_projection_dim // num_gram_layers
        
        gram_layer_projections = nn.ModuleDict()
        total_gram_dim = 0
        
        try:
            # Run a dummy forward pass to capture feature map dimensions
            # This is necessary because dimensions can vary with the specific CLIP model
            if hasattr(self.clip_model.visual, 'input_resolution'):
                 resolution = self.clip_model.visual.input_resolution
            elif hasattr(self.clip_model, 'input_resolution'): # Some wrappers might have it here
                 resolution = self.clip_model.input_resolution
            else: # Fallback to 224 if not found, common for RN50
                 print("Warning: input_resolution not found, using 224 as fallback for Gram dim.")
                 resolution = 224

            dummy_image = torch.randn(1, 3, resolution, resolution).to(self.device)
            with torch.no_grad():
                self.clip_model.visual(dummy_image) # Activate hooks
            
            for layer_name in self.layers_for_gram_config:
                if layer_name in self.extracted_gram_feature_maps:
                    C = self.extracted_gram_feature_maps[layer_name].shape[1]
                    gram_vector_dim = C * (C + 1) // 2
                    dict_key = layer_name.replace('.', '_') # For ModuleDict compatibility
                    gram_layer_projections[dict_key] = nn.Linear(gram_vector_dim, per_gram_vector_projection_dim).to(self.device)
                    nn.init.xavier_uniform_(gram_layer_projections[dict_key].weight)
                    nn.init.zeros_(gram_layer_projections[dict_key].bias)
                    total_gram_dim += per_gram_vector_projection_dim
                    print(f"Gram Layer '{layer_name}': C={C}, Gram Vector Dim={gram_vector_dim}, Projected to {per_gram_vector_projection_dim}")
            
            self.extracted_gram_feature_maps.clear() # Clean after dummy forward
            
        except Exception as e:
            print(f"Error during Gram projections initialization: {e}. Gram features might not work.")
            return 0, nn.ModuleDict()
            
        return total_gram_dim, gram_layer_projections

    def _get_gram_vector(self, feature_map_batch):
        B, C, H, W = feature_map_batch.size()
        features = feature_map_batch.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (H * W) # Normalization
        indices = torch.triu_indices(C, C, offset=0, device=gram.device)
        return gram[:, indices[0], indices[1]]

    def _hook_fn_gram(self, layer_name):
        def hook(module, input, output):
            self.extracted_gram_feature_maps[layer_name] = output
        return hook

    def _register_gram_hooks(self, visual_model):
        if not self.clip_model_name.startswith("RN"):
            print(f"Gram extraction supported mainly for RN models. Current model: {self.clip_model_name}. No hooks registered if not RN.")
            self.layers_for_gram_config = [] # Disable if not RN
            return
            
        # Generic mapping for ResNet-like models
        target_modules_rn = {
            'layer1': getattr(visual_model, 'layer1', None),
            'layer2': getattr(visual_model, 'layer2', None),
            'layer3': getattr(visual_model, 'layer3', None),
            'layer4': getattr(visual_model, 'layer4', None)
        }
        
        for layer_key in self.layers_for_gram_config:
            module_to_hook = target_modules_rn.get(layer_key)
            if module_to_hook is not None:
                try:
                    hook = module_to_hook.register_forward_hook(self._hook_fn_gram(layer_key))
                    self.feature_extractor_hooks.append(hook)
                    print(f"Hook registered for Gram on: {layer_key}")
                except Exception as e:
                    print(f"Error registering hook for {layer_key} on {self.clip_model_name}: {e}")
            else:
                print(f"WARNING: layer '{layer_key}' not found or not supported in ResNet-like model {self.clip_model_name}")
        
        # If no hooks were successfully registered, clean the configuration
        if not self.feature_extractor_hooks:
            print("No Gram hooks successfully registered. Gram features will be disabled.")
            self.layers_for_gram_config = []


    def encode_image_with_style_adapter(self, image_input: torch.Tensor) -> torch.Tensor:
        self.extracted_gram_feature_maps.clear()

        with torch.no_grad(): # CLIP backbone is frozen
            # This activates hooks for Gram features
            _ = self.clip_model.visual(image_input) 
            # Now extract final semantic features
            semantic_features = self.clip_model.encode_image(image_input).float()

        if self.total_gram_projected_dim > 0 and self.gram_layer_projections:
            projected_gram_vectors = self._process_gram_features(image_input.size(0))
            if projected_gram_vectors.shape[0] != semantic_features.shape[0]:
                 print(f"Warning: Batch size mismatch between semantic ({semantic_features.shape[0]}) and gram ({projected_gram_vectors.shape[0]})")
                 # Take minimum batch size to avoid concatenation errors
                 min_batch_size = min(semantic_features.shape[0], projected_gram_vectors.shape[0])
                 semantic_features = semantic_features[:min_batch_size]
                 projected_gram_vectors = projected_gram_vectors[:min_batch_size]

            if projected_gram_vectors.numel() > 0 : # Make sure it's not empty
                features_for_fusion = torch.cat([semantic_features, projected_gram_vectors], dim=1)
            else:
                features_for_fusion = semantic_features
        else:
            features_for_fusion = semantic_features
        
        adapted_features = self.fusion_adapter(features_for_fusion)
        return torch.nn.functional.normalize(adapted_features, p=2, dim=-1)

    def _process_gram_features(self, batch_size: int) -> torch.Tensor:
        projected_gram_vectors_list = []
        
        # Fallback dimension if necessary (e.g., first projected layer)
        fallback_dim_per_layer = 0
        if self.gram_layer_projections:
            first_proj_layer_key = next(iter(self.gram_layer_projections), None)
            if first_proj_layer_key:
                fallback_dim_per_layer = self.gram_layer_projections[first_proj_layer_key].out_features

        for layer_name in self.layers_for_gram_config:
            dict_key = layer_name.replace('.', '_')
            if layer_name in self.extracted_gram_feature_maps and \
               dict_key in self.gram_layer_projections and \
               self.gram_layer_projections[dict_key] is not None:
                
                feature_map = self.extracted_gram_feature_maps[layer_name]
                if feature_map.shape[0] != batch_size: #
                    feature_map = feature_map[:batch_size]


                gram_vector = self._get_gram_vector(feature_map.float())
                projected_gram = self.gram_layer_projections[dict_key](gram_vector)
                projected_gram_vectors_list.append(projected_gram)
            elif fallback_dim_per_layer > 0 : # Fallback if a layer wasn't extracted but others were
                print(f"Warning: Feature map for '{layer_name}' not found or projection not defined. Using zero fallback.")
                projected_gram_vectors_list.append(torch.zeros(batch_size, fallback_dim_per_layer, device=self.device))
                
        if projected_gram_vectors_list:
            try:
                return torch.cat(projected_gram_vectors_list, dim=1)
            except RuntimeError as e:
                print(f"Error concatenating Gram vectors: {e}")
                # Print dimensions for debug
                for i, p_vec in enumerate(projected_gram_vectors_list):
                    print(f"Vector {i}: {p_vec.shape}")
                # Return zero tensor of expected dimension to not block training
                return torch.zeros(batch_size, self.total_gram_projected_dim, device=self.device)

        else: # No gram features extracted or projected
            return torch.zeros(batch_size, 0, device=self.device) # Empty tensor with correct dim


    def encode_text(self, text_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input).float()
        return torch.nn.functional.normalize(text_features, p=2, dim=-1)


    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.encode_image_with_style_adapter(image)


    def forward(self, image_input: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image_with_style_adapter(image_input)
        text_features = self.encode_text(text_tokens)
        text_features = text_features.type_as(image_features)
        
        # logit_scale might not always be present or trainable
        if hasattr(self.clip_model, 'logit_scale'):
            logit_scale = self.clip_model.logit_scale.exp().float()
        else:
            logit_scale = torch.tensor(1.0, device=self.device).float() # Fixed value if not present
            
        return logit_scale * (image_features @ text_features.t())

    def _remove_gram_hooks(self):
        for hook in self.feature_extractor_hooks:
            hook.remove()
        self.feature_extractor_hooks = []

    def __del__(self):
        self._remove_gram_hooks()


