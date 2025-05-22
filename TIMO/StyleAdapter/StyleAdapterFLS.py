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

# --- Funzione find_artgraph_path (invariata) ---
def find_artgraph_path():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_script_dir, "..", "$DATA", "artgraph"),
        os.path.join(os.path.dirname(os.path.dirname(current_script_dir)), "$DATA", "artgraph_complementary"),
        os.path.abspath("/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO/$DATA/artgraph_complementary")
    ]
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            print(f"Dataset artgraph trovato in: {path}")
            return path
    raise FileNotFoundError("Dataset artgraph non trovato. Controlla i percorsi o crea la directory.")

# --- Classe ArtgraphDataset (Modificata per Meta-Learning) ---
class ArtgraphDataset(data.Dataset):
    def __init__(self, root_dir: str, transform=None, seed: int = 42,
                 artist_subset: List[str] = None): # Nuovo: per selezionare un sottoinsieme di artisti
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        
        random.seed(seed)
        np.random.seed(seed)

        all_available_artists = sorted([d for d in os.listdir(self.images_dir)
                                   if os.path.isdir(os.path.join(self.images_dir, d))])
        if not all_available_artists:
            raise FileNotFoundError(f"Nessuna sottocartella (classe/artista) trovata in {self.images_dir}")

        if artist_subset:
            self.classnames = [name for name in artist_subset if name in all_available_artists]
            if len(self.classnames) != len(artist_subset):
                print("Attenzione: Alcuni artisti nel subset fornito non sono stati trovati nel dataset.")
        else:
            self.classnames = all_available_artists
        
        if not self.classnames:
            raise ValueError("Nessun artista selezionato o trovato per il dataset.")

        self.class_to_idx = {cls_name: i+150 for i, cls_name in enumerate(self.classnames)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        self.samples_by_class: Dict[int, List[str]] = {label: [] for label in self.class_to_idx.values()}
        self.flat_samples: List[Tuple[str, int]] = [] # Lista di (path_immagine, label_globale)

        for artist_name in self.classnames:
            artist_dir = os.path.join(self.images_dir, artist_name)
            artist_label = self.class_to_idx[artist_name]
            if os.path.isdir(artist_dir):
                img_names_for_artist = [
                    img_name for img_name in os.listdir(artist_dir)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
                ]
                random.shuffle(img_names_for_artist) # Shuffle per K-shot e Q-query
                for img_name in img_names_for_artist:
                    img_path = os.path.join(artist_dir, img_name)
                    self.samples_by_class[artist_label].append(img_path)
                    self.flat_samples.append((img_path, artist_label))
        
        # Per __getitem__ e __len__ sul dataset piatto (usato da DataLoader standard)
        # Il campionamento episodico avverrà esternamente tramite un Sampler.
        # random.shuffle(self.flat_samples) # Lo fa già EpisodicBatchSampler se necessario

        print(f"Caricato dataset da '{root_dir}'. Artisti selezionati: {len(self.classnames)}. Immagini totali per questi artisti: {len(self.flat_samples)}")
        if len(self.flat_samples) == 0:
            print(f"ATTENZIONE: Nessun campione caricato. Controlla i percorsi e artist_subset.")

    def __len__(self):
        # Lunghezza basata sul numero di episodi (gestita dal sampler) o sul totale immagini?
        # Per un DataLoader standard, questa è la lunghezza totale delle immagini.
        return len(self.flat_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Questo __getitem__ restituisce una singola immagine e la sua etichetta GLOBALE.
        # Il sampler episodico fornirà gli indici corretti.
        img_path, label = self.flat_samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Gestione errore migliorata: se un'immagine specifica fallisce, potrebbe essere un problema
            # per il campionamento episodico. Per ora, stampiamo e solleviamo errore o ritorniamo placeholder.
            print(f"Errore grave nel caricamento dell'immagine {img_path} all'indice {idx}: {e}. Questo potrebbe interrompere il training episodico.")
            # Potrebbe essere necessario restituire un'immagine placeholder o gestire l'errore nel sampler/collate_fn
            # Per semplicità, proviamo a prendere un'altra immagine casuale, ma è subottimale
            # random_idx = random.randint(0, len(self.flat_samples) - 1)
            # return self.__getitem__(random_idx)
            raise IOError(f"Impossibile caricare {img_path}")


# --- Classe Adapter (invariata) ---
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim_override=None, dropout_rate=0.1, use_layernorm=True):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.output_dim = output_dim_override if output_dim_override is not None else input_dim
        self.up_project = nn.Linear(bottleneck_dim, self.output_dim)
        
        # MODIFICA: Migliore inizializzazione per evitare il blocco del gradiente
        nn.init.xavier_uniform_(self.down_project.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.down_project.bias)
        # Inizializzazione non-zero per up_project (piccola ma non zero)
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
            # MODIFICA: Aumentato fattore di skip-connection per migliorare il flusso del gradiente
            alpha = 0.2  # Aumentato da 0.1
            x = alpha * x + original_x
        return x


# --- CLASSE: CLIPWithStyleAdapter (invariata nella struttura interna, ma il forward non sarà usato direttamente) ---
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
            raise RuntimeError(f"Errore durante il caricamento del modello CLIP '{model_name_to_load}': {e}") from e

        self.visual = self.clip_model.visual
        self.feature_extractor_hooks = []
        self.extracted_gram_feature_maps = {}
        self.device = device
        self.clip_model_name = clip_model_name # Usare il nome originale per logica RN
        self.layers_for_gram_config = layers_for_gram_rn50

        if not self.clip_model_name.startswith("RN") and self.layers_for_gram_config:
            print(f"ATTENZIONE: L'estrazione Gram è ottimizzata per architetture RN. Il modello {self.clip_model_name} potrebbe non avere i layer specificati ('layer1', 'layer2', ecc.).")

        self.semantic_feature_dim = self._get_semantic_feature_dim()
        print(f"Dimensione feature semantica (output di encode_image): {self.semantic_feature_dim}")

        if self.layers_for_gram_config:
            self._register_gram_hooks(self.clip_model.visual)
            self.total_gram_projected_dim, self.gram_layer_projections = self._setup_gram_projections(gram_style_projection_dim)
            print(f"Dimensione totale Gram features proiettate: {self.total_gram_projected_dim}")
        else:
            self.total_gram_projected_dim = 0
            self.gram_layer_projections = nn.ModuleDict()
            print("Nessuna feature Gram sarà utilizzata (layers_for_gram_config vuoto o non RN)")

        fusion_input_dim = self.semantic_feature_dim + self.total_gram_projected_dim
        if fusion_input_dim <=0:
            raise ValueError(f"Fusion input dimension è {fusion_input_dim}. Controlla semantic_feature_dim e total_gram_projected_dim.")

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
                 # Per alcuni modelli ViT, la dimensione dell'output è dopo la proiezione
                if isinstance(self.clip_model.visual.proj, torch.Tensor): # ViT-L/14@336px
                     return self.clip_model.visual.proj.shape[1]
                else: # Altri ViT
                     return self.clip_model.visual.proj.out_features
            else: # Fallback per RN e altri
                resolution = self.clip_model.visual.input_resolution
                dummy_image = torch.randn(1, 3, resolution, resolution).to(self.device)
                with torch.no_grad():
                    return self.clip_model.encode_image(dummy_image).shape[-1]
        except Exception as e:
            print(f"Errore nel determinare semantic_feature_dim: {e}. Usando 1024 come fallback.")
            return 1024 # Valore comune per RN50

    def _setup_gram_projections(self, gram_style_projection_dim):
        if not self.layers_for_gram_config or not self.clip_model_name.startswith("RN"):
            return 0, nn.ModuleDict()
            
        # Assicurati che ci sia almeno un layer per evitare divisione per zero
        num_gram_layers = len(self.layers_for_gram_config)
        if num_gram_layers == 0:
            return 0, nn.ModuleDict()
        per_gram_vector_projection_dim = gram_style_projection_dim // num_gram_layers
        
        gram_layer_projections = nn.ModuleDict()
        total_gram_dim = 0
        
        try:
            # Esegui una forward pass fittizia per catturare le dimensioni delle feature map
            # Questo è necessario perché le dimensioni possono variare con il modello CLIP specifico
            if hasattr(self.clip_model.visual, 'input_resolution'):
                 resolution = self.clip_model.visual.input_resolution
            elif hasattr(self.clip_model, 'input_resolution'): # Alcuni wrapper potrebbero averlo qui
                 resolution = self.clip_model.input_resolution
            else: # Fallback a 224 se non trovato, comune per RN50
                 print("Attenzione: input_resolution non trovato, usando 224 come fallback per Gram dim.")
                 resolution = 224

            dummy_image = torch.randn(1, 3, resolution, resolution).to(self.device)
            with torch.no_grad():
                self.clip_model.visual(dummy_image) # Attiva gli hook
            
            for layer_name in self.layers_for_gram_config:
                if layer_name in self.extracted_gram_feature_maps:
                    C = self.extracted_gram_feature_maps[layer_name].shape[1]
                    gram_vector_dim = C * (C + 1) // 2
                    dict_key = layer_name.replace('.', '_') # Per compatibilità con ModuleDict
                    gram_layer_projections[dict_key] = nn.Linear(gram_vector_dim, per_gram_vector_projection_dim).to(self.device)
                    nn.init.xavier_uniform_(gram_layer_projections[dict_key].weight)
                    nn.init.zeros_(gram_layer_projections[dict_key].bias)
                    total_gram_dim += per_gram_vector_projection_dim
                    print(f"Layer Gram '{layer_name}': C={C}, Dim Vettore Gram={gram_vector_dim}, Proiettato a {per_gram_vector_projection_dim}")
            
            self.extracted_gram_feature_maps.clear() # Pulisci dopo la fwd fittizia
            
        except Exception as e:
            print(f"Errore durante l'inizializzazione delle proiezioni Gram: {e}. Le feature Gram potrebbero non funzionare.")
            # Ritorna 0 e un ModuleDict vuoto per disabilitare le feature Gram in caso di errore
            return 0, nn.ModuleDict()
            
        return total_gram_dim, gram_layer_projections

    def _get_gram_vector(self, feature_map_batch):
        B, C, H, W = feature_map_batch.size()
        features = feature_map_batch.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (H * W) # Normalizzazione
        indices = torch.triu_indices(C, C, offset=0, device=gram.device)
        return gram[:, indices[0], indices[1]]

    def _hook_fn_gram(self, layer_name):
        def hook(module, input, output):
            self.extracted_gram_feature_maps[layer_name] = output
        return hook

    def _register_gram_hooks(self, visual_model):
        if not self.clip_model_name.startswith("RN"):
            print(f"Estrazione Gram supportata principalmente per modelli RN. Modello attuale: {self.clip_model_name}. Nessun hook registrato se non RN.")
            self.layers_for_gram_config = [] # Disabilita se non RN
            return
            
        # Mappatura generica per modelli ResNet-like
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
                    print(f"Hook registrato per Gram su: {layer_key}")
                except Exception as e:
                    print(f"Errore nella registrazione dell'hook per {layer_key} su {self.clip_model_name}: {e}")
            else:
                print(f"AVVISO: layer '{layer_key}' non trovato o non supportato nel modello ResNet-like {self.clip_model_name}")
        
        # Se nessun hook è stato registrato con successo, pulisci la configurazione
        if not self.feature_extractor_hooks:
            print("Nessun hook Gram registrato con successo. Le feature Gram saranno disabilitate.")
            self.layers_for_gram_config = []


    def encode_image_with_style_adapter(self, image_input: torch.Tensor) -> torch.Tensor:
        self.extracted_gram_feature_maps.clear()

        with torch.no_grad(): # Il backbone CLIP è congelato
            # Questo attiva gli hook per le feature Gram
            _ = self.clip_model.visual(image_input) 
            # Ora estrai le feature semantiche finali
            semantic_features = self.clip_model.encode_image(image_input).float()

        if self.total_gram_projected_dim > 0 and self.gram_layer_projections:
            projected_gram_vectors = self._process_gram_features(image_input.size(0))
            if projected_gram_vectors.shape[0] != semantic_features.shape[0]:
                 print(f"Attenzione: Mismatch batch size tra semantic ({semantic_features.shape[0]}) e gram ({projected_gram_vectors.shape[0]})")
                 # Prendi il minimo batch size per evitare errori di concatenazione
                 min_batch_size = min(semantic_features.shape[0], projected_gram_vectors.shape[0])
                 semantic_features = semantic_features[:min_batch_size]
                 projected_gram_vectors = projected_gram_vectors[:min_batch_size]

            if projected_gram_vectors.numel() > 0 : # Assicurati che non sia vuoto
                features_for_fusion = torch.cat([semantic_features, projected_gram_vectors], dim=1)
            else:
                features_for_fusion = semantic_features
        else:
            features_for_fusion = semantic_features
        
        adapted_features = self.fusion_adapter(features_for_fusion)
        return torch.nn.functional.normalize(adapted_features, p=2, dim=-1)

    def _process_gram_features(self, batch_size: int) -> torch.Tensor:
        projected_gram_vectors_list = []
        
        # Fallback dimension se necessario (es. primo layer proiettato)
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
                if feature_map.shape[0] != batch_size: # Può succedere se i batch precedenti avevano dimensioni diverse e le mappe non sono state pulite
                    # print(f"Warning: feature_map batch size {feature_map.shape[0]} != input batch_size {batch_size} for layer {layer_name}. Using input batch_size portion.")
                    feature_map = feature_map[:batch_size]


                gram_vector = self._get_gram_vector(feature_map.float())
                projected_gram = self.gram_layer_projections[dict_key](gram_vector)
                projected_gram_vectors_list.append(projected_gram)
            elif fallback_dim_per_layer > 0 : # Fallback se un layer non è stato estratto ma altri sì
                print(f"Attenzione: Feature map per '{layer_name}' non trovata o proiezione non definita. Uso zeri di fallback.")
                projected_gram_vectors_list.append(torch.zeros(batch_size, fallback_dim_per_layer, device=self.device))
                
        if projected_gram_vectors_list:
            try:
                return torch.cat(projected_gram_vectors_list, dim=1)
            except RuntimeError as e:
                print(f"Errore nella concatenazione dei vettori Gram: {e}")
                # Stampa dimensioni per debug
                for i, p_vec in enumerate(projected_gram_vectors_list):
                    print(f"Vettore {i}: {p_vec.shape}")
                # Ritorna un tensore di zeri della dimensione attesa per non bloccare il training
                return torch.zeros(batch_size, self.total_gram_projected_dim, device=self.device)

        else: # Nessuna feature gram estratta o proiettata
            return torch.zeros(batch_size, 0, device=self.device) # Tensore vuoto con dim corretta


    def encode_text(self, text_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input).float()
        return torch.nn.functional.normalize(text_features, p=2, dim=-1)

    # encode_image è usato per compatibilità, ma per Prototypical Networks
    # useremo direttamente encode_image_with_style_adapter
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.encode_image_with_style_adapter(image)

    # Il metodo forward originale (per classificazione standard con prompt testuali)
    # non è usato direttamente da Prototypical Networks, ma lo lasciamo per potenziale uso futuro.
    def forward(self, image_input: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_image_with_style_adapter(image_input)
        text_features = self.encode_text(text_tokens)
        text_features = text_features.type_as(image_features)
        
        # logit_scale potrebbe non essere sempre presente o addestrabile
        if hasattr(self.clip_model, 'logit_scale'):
            logit_scale = self.clip_model.logit_scale.exp().float()
        else:
            logit_scale = torch.tensor(1.0, device=self.device).float() # Valore fisso se non presente
            
        return logit_scale * (image_features @ text_features.t())

    def _remove_gram_hooks(self):
        for hook in self.feature_extractor_hooks:
            hook.remove()
        self.feature_extractor_hooks = []

    def __del__(self):
        self._remove_gram_hooks()

# Le funzioni train_epoch, validate_model, save_model, main sono state spostate
# e adattate in optimizer.py per il meta-training.

