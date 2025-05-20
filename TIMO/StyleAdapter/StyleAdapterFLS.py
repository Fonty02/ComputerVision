import os
import random
import sys
import argparse
from collections import defaultdict
import copy # For deepcopy in validation if needed

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR # Used in original, might be adapted for meta

try:
    import clip
except ImportError:
    print("CLIP library not found. Please install it with 'pip install git+https://github.com/openai/CLIP.git'")
    sys.exit(1)

from tqdm import tqdm

try:
    import higher
except ImportError:
    print("Higher library not found. Please install it with 'pip install higher'")
    sys.exit(1)


# --- Funzione find_artgraph_path (dall'originale) ---
def find_artgraph_path():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_script_dir, "..", "$DATA", "artgraph_complementary"),
        os.path.join(os.path.dirname(os.path.dirname(current_script_dir)), "$DATA", "artgraph_complementary"),
        # Aggiungi altri percorsi se necessario, o rendi questo un argomento
    ]
    for path in possible_paths:
        # Sostituisci $DATA con il nome effettivo della cartella se è una variabile d'ambiente o un placeholder
        # Per ora, assumiamo che sia un nome di cartella letterale o che venga gestito esternamente.
        # Se $DATA è un placeholder per una variabile d'ambiente:
        # path = os.path.expandvars(path) 
        if os.path.exists(path) and os.path.isdir(path):
            print(f"Dataset Artgraph trovato in: {path}")
            return path
            
    # Fallback se non trovato nei percorsi comuni, per coerenza con l'originale
    fallback_path = os.path.abspath("/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO/$DATA/artgraph_complementary")
    if os.path.exists(fallback_path) and os.path.isdir(fallback_path):
        print(f"Dataset Artgraph trovato nel percorso di fallback: {fallback_path}")
        return fallback_path
        
    raise FileNotFoundError("Dataset artgraph non trovato. Controlla i percorsi configurati in `find_artgraph_path` o crea la directory.")


# --- Classe MetaArtgraphDataset (per Few-Shot Learning) ---
class MetaArtgraphDataset(data.Dataset):
    def __init__(self, root_dir, num_tasks, n_way, k_shot_support, k_shot_query, transform=None, split='meta_train', seed=42):
        super(MetaArtgraphDataset, self).__init__()
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        self.num_tasks = num_tasks
        self.n_way = n_way
        self.k_shot_support = k_shot_support
        self.k_shot_query = k_shot_query
        self.split = split
        self._random_state = random.Random(seed) # Usa un'istanza di Random per questo dataset
        self._np_random_state = np.random.RandomState(seed)

        all_classnames = sorted([d for d in os.listdir(self.images_dir)
                                 if os.path.isdir(os.path.join(self.images_dir, d))])
        if not all_classnames:
            raise FileNotFoundError(f"Nessuna sottocartella (classe) trovata in {self.images_dir}")

        self._np_random_state.shuffle(all_classnames) # Shuffle classi per la divisione

        num_classes = len(all_classnames)
        # Divisioni indicative, da adattare in base alla dimensione del dataset
        meta_train_split_idx = int(0.7 * num_classes)
        meta_val_split_idx = int(0.85 * num_classes)

        if split == 'meta_train':
            self.classnames = all_classnames[:meta_train_split_idx]
        elif split == 'meta_val':
            self.classnames = all_classnames[meta_train_split_idx:meta_val_split_idx]
        elif split == 'meta_test':
            self.classnames = all_classnames[meta_val_split_idx:]
        else:
            raise ValueError("Split non valido. Scegli tra 'meta_train', 'meta_val', 'meta_test'.")

        if len(self.classnames) < self.n_way:
            raise ValueError(f"Split '{split}' non ha abbastanza classi ({len(self.classnames)}) per creare task {self.n_way}-way. Minimo richiesto: {self.n_way}.")

        self.class_to_images = defaultdict(list)
        for classname in self.classnames:
            class_dir = os.path.join(self.images_dir, classname)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    self.class_to_images[classname].append(os.path.join(class_dir, img_name))
            if not self.class_to_images[classname]:
                print(f"Attenzione: nessuna immagine trovata per la classe {classname} in {class_dir}")
        
        # Filtra classi senza abbastanza immagini
        self.classnames = [cn for cn in self.classnames if len(self.class_to_images[cn]) >= self.k_shot_support + self.k_shot_query]
        if len(self.classnames) < self.n_way:
             raise ValueError(f"Dopo il filtraggio, lo split '{split}' non ha abbastanza classi ({len(self.classnames)}) con campioni sufficienti ({self.k_shot_support + self.k_shot_query} richiesti per classe) per creare task {self.n_way}-way.")


        print(f"MetaArtgraphDataset ({split}): {len(self.classnames)} classi utilizzabili, {self.num_tasks} tasks, {n_way}-way, {k_shot_support}-shot support, {k_shot_query}-shot query.")

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        selected_classes_names = self._random_state.sample(self.classnames, self.n_way)

        support_images_list = []
        support_labels_list = []
        query_images_list = []
        query_labels_list = []
        
        class_to_episode_label = {classname: i for i, classname in enumerate(selected_classes_names)}

        for classname in selected_classes_names:
            all_images_for_class = list(self.class_to_images[classname]) # Copia per fare shuffle
            self._random_state.shuffle(all_images_for_class)
            
            num_needed = self.k_shot_support + self.k_shot_query
            
            # Se non ci sono abbastanza immagini uniche, campiona con rimpiazzo (o duplica)
            if len(all_images_for_class) < num_needed:
                selected_image_paths = self._np_random_state.choice(all_images_for_class, size=num_needed, replace=True)
            else:
                selected_image_paths = all_images_for_class[:num_needed]

            episode_label = class_to_episode_label[classname]

            for i, img_path in enumerate(selected_image_paths):
                try:
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                except Exception as e:
                    print(f"Errore caricamento immagine {img_path}: {e}. Tento di usare un placeholder o salto.")
                    # Potresti voler gestire questo in modo più robusto, es. saltando l'episodio
                    # o usando un'immagine placeholder se il transform lo permette.
                    # Per ora, se un'immagine fallisce, l'episodio potrebbe essere più piccolo.
                    continue 

                if i < self.k_shot_support:
                    support_images_list.append(image)
                    support_labels_list.append(episode_label)
                else:
                    query_images_list.append(image)
                    query_labels_list.append(episode_label)
        
        # Se una delle liste è vuota, l'episodio non è valido.
        # Questo può accadere se tutte le immagini per una classe campionata falliscono il caricamento.
        if not support_images_list or not query_images_list or \
           len(support_images_list) < self.n_way * self.k_shot_support or \
           len(query_images_list) < self.n_way * self.k_shot_query : # Controllo più stretto
            # print(f"Attenzione: episodio {index} incompleto a causa di errori di caricamento o campionamento. Rigenero...")
            # return self.__getitem__(self._random_state.randint(0, self.num_tasks - 1)) # Rigenera un altro task
            # Per evitare ricorsione infinita in casi estremi, potremmo restituire un flag o dati vuoti gestiti a monte.
            # Per ora, assumiamo che il dataloader possa gestire batch con 0 elementi se collate_fn è robusto.
            # Oppure, si solleva un errore o si logga e si salta nel training loop.
             return (torch.empty(0), torch.empty(0, dtype=torch.long), 
                    torch.empty(0), torch.empty(0, dtype=torch.long), [])


        s_images_tensor = torch.stack(support_images_list)
        s_labels_tensor = torch.tensor(support_labels_list, dtype=torch.long)
        q_images_tensor = torch.stack(query_images_list)
        q_labels_tensor = torch.tensor(query_labels_list, dtype=torch.long)

        return s_images_tensor, s_labels_tensor, q_images_tensor, q_labels_tensor, selected_classes_names


# --- Classe Adapter (dall'originale) ---
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim_override=None, dropout_rate=0.1, use_layernorm=True):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.output_dim = output_dim_override if output_dim_override is not None else input_dim
        self.up_project = nn.Linear(bottleneck_dim, self.output_dim)
        
        nn.init.xavier_uniform_(self.down_project.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.weight) # Inizializzazione a zero per l'up_project è comune per gli adapter
        nn.init.zeros_(self.up_project.bias)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = nn.LayerNorm(bottleneck_dim)
        self.activation = nn.ReLU() # o nn.GELU()
        
    def forward(self, x):        
        original_x = x 
        
        x_down = self.down_project(x)
        if self.use_layernorm:
            x_down = self.layernorm(x_down)
        x_activated = self.activation(x_down)
        x_dropout = self.dropout(x_activated)
        x_up = self.up_project(x_dropout)
        
        if x_up.shape[-1] == original_x.shape[-1]:
            return original_x + x_up # Connessione residuale
        else:
            # Se le dimensioni non corrispondono, non si può fare la somma residuale diretta.
            # Questo può accadere se output_dim_override è diverso da input_dim.
            # In tal caso, l'adapter agisce più come un trasformatore di feature.
            return x_up


# --- CLASSE: CLIPWithStyleAdapter (dall'originale, con piccole modifiche per chiarezza) ---
class CLIPWithStyleAdapter(nn.Module):
    def __init__(self, clip_model_name, fusion_bottleneck_dim, gram_style_projection_dim, device, 
                 layers_for_gram_rn50=None, 
                 dropout_rate=0.1, use_layernorm_adapter=True):
        super(CLIPWithStyleAdapter, self).__init__()
        
        if layers_for_gram_rn50 is None:
            layers_for_gram_rn50 = ['layer2', 'layer3'] # Default
            
        model_name_to_load = "RN50" if clip_model_name == "CustomRN50" else clip_model_name
        try:
            self.clip_model, self.preprocess = clip.load(model_name_to_load, device=device)
        except Exception as e:
            print(f"Errore durante il caricamento del modello CLIP '{model_name_to_load}': {e}")
            print("Modelli disponibili:", clip.available_models())
            raise
        
        self.visual = self.clip_model.visual
        self.feature_extractor_hooks = []
        self.extracted_gram_feature_maps = {}
        
        self.device = device
        self.clip_model_name = clip_model_name # Potrebbe essere "CustomRN50" o un nome CLIP standard
        self.layers_for_gram_config = layers_for_gram_rn50

        if not self.clip_model_name.startswith("RN") and self.layers_for_gram_config:
            print(f"Attenzione: layers_for_gram_rn50 specificati ({self.layers_for_gram_config}) ma il modello base è {self.clip_model_name}. Gli hook potrebbero non funzionare come previsto.")

        self.semantic_feature_dim = self._get_semantic_feature_dim()
        print(f"Dimensione feature semantica (output di encode_image del CLIP base): {self.semantic_feature_dim}")

        self.gram_layer_projections = nn.ModuleDict() # Inizializza sempre
        self.total_gram_projected_dim = 0
        if self.layers_for_gram_config and gram_style_projection_dim > 0:
            self._setup_gram_projections(gram_style_projection_dim)
            if self.clip_model_name.startswith("RN"): # Registra hook solo per ResNet e se ci sono layer configurati
                 self._register_gram_hooks(self.visual)
            else:
                print("Skipping Gram hooks registration for non-ResNet model or zero gram_style_projection_dim.")
        else:
            print("Nessun layer per Gram specificato o gram_style_projection_dim è zero. L'adapter di stile Gram non sarà usato.")


        fusion_input_dim = self.semantic_feature_dim + self.total_gram_projected_dim
        self.fusion_adapter = Adapter(
            input_dim=fusion_input_dim,
            bottleneck_dim=fusion_bottleneck_dim,
            output_dim_override=self.semantic_feature_dim, # L'output deve corrispondere alla dim delle feature di testo
            dropout_rate=dropout_rate,
            use_layernorm=use_layernorm_adapter
        ).to(device)
        
        print(f"Fusion Adapter: input_dim={fusion_input_dim}, bottleneck_dim={fusion_bottleneck_dim}, output_dim={self.semantic_feature_dim}")
        
    def _get_semantic_feature_dim(self):
        try:
            # Prova a ottenere la dimensione dall'output del modello visuale
            # o dalla proiezione se esiste (es. per ViT)
            if hasattr(self.visual, 'output_dim'):
                 return self.visual.output_dim
            elif hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
                 # Assumiamo che la dimensione delle feature visuali e testuali proiettate sia la stessa
                 return self.clip_model.text_projection.shape[-1]
            else: # Fallback: esegui un forward fittizio
                dummy_image = torch.randn(1, 3, self.clip_model.visual.input_resolution, self.clip_model.visual.input_resolution).to(self.device)
                with torch.no_grad():
                    features = self.clip_model.encode_image(dummy_image)
                return features.shape[-1]
        except Exception as e:
            print(f"Errore nel determinare semantic_feature_dim: {e}. Uso fallback 512 per ViT, 1024 per RN50.")
            if "ViT" in self.clip_model_name: return 512 # Valore comune per ViT-B/32
            if "RN50" in self.clip_model_name: return 1024 # Valore comune per RN50
            return 512 # Default generico

    def _setup_gram_projections(self, gram_style_projection_dim):
        if not self.layers_for_gram_config:
            self.total_gram_projected_dim = 0
            return

        # Determina la dimensione di output per ogni Gram vector proiettato
        # Assicura che sia divisibile o gestisci il resto
        num_gram_layers = len(self.layers_for_gram_config)
        if num_gram_layers == 0:
            self.total_gram_projected_dim = 0
            return
            
        per_gram_vector_projection_dim = gram_style_projection_dim // num_gram_layers
        
        # Forward fittizia per determinare le dimensioni delle feature map dei layer target
        # Questo è cruciale e può essere complesso per modelli generici.
        # Per RN50, le dimensioni dei canali sono note, ma è meglio verificarle.
        dummy_image = torch.randn(1, 3, self.visual.input_resolution, self.visual.input_resolution).to(self.device)
        temp_hooks = []
        temp_feature_maps = {}

        def temp_hook_fn(layer_name):
            def hook(module, input, output):
                temp_feature_maps[layer_name] = output
            return hook

        target_modules_for_dim_check = {
            'layer1': self.visual.layer1 if hasattr(self.visual, 'layer1') else None,
            'layer2': self.visual.layer2 if hasattr(self.visual, 'layer2') else None,
            'layer3': self.visual.layer3 if hasattr(self.visual, 'layer3') else None,
            'layer4': self.visual.layer4 if hasattr(self.visual, 'layer4') else None,
        }

        for layer_key in self.layers_for_gram_config:
            module = target_modules_for_dim_check.get(layer_key)
            if module:
                temp_hooks.append(module.register_forward_hook(temp_hook_fn(layer_key)))
        
        with torch.no_grad():
            self.visual(dummy_image) # Esegui forward per attivare gli hook
        
        for h in temp_hooks: h.remove() # Rimuovi hook temporanei

        current_total_projected_dim = 0
        for layer_name in self.layers_for_gram_config:
            if layer_name not in temp_feature_maps:
                print(f"Attenzione: feature map per '{layer_name}' non catturata durante il setup. Salto la proiezione per questo layer.")
                continue
            
            feature_map_shape = temp_feature_maps[layer_name].shape # B, C, H, W
            C = feature_map_shape[1] # Numero di canali
            gram_vector_dim = C * (C + 1) // 2 # Dimensione del vettore di Gram vettorizzato (triangolo superiore)
            
            self.gram_layer_projections[layer_name] = nn.Linear(gram_vector_dim, per_gram_vector_projection_dim).to(self.device)
            nn.init.xavier_uniform_(self.gram_layer_projections[layer_name].weight)
            nn.init.zeros_(self.gram_layer_projections[layer_name].bias)
            current_total_projected_dim += per_gram_vector_projection_dim
            print(f"Proiezione Gram per '{layer_name}': input_dim={gram_vector_dim}, output_dim={per_gram_vector_projection_dim}")

        self.total_gram_projected_dim = current_total_projected_dim
        if self.total_gram_projected_dim != gram_style_projection_dim and num_gram_layers > 0 :
            print(f"Attenzione: total_gram_projected_dim ({self.total_gram_projected_dim}) è diverso da gram_style_projection_dim ({gram_style_projection_dim}) a causa dell'arrotondamento.")


    def _get_gram_vector(self, feature_map_batch):
        B, C, H, W = feature_map_batch.size()
        features = feature_map_batch.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (H * W * C) # Normalizzazione più robusta (dividi anche per C)
        
        # Vettorizzazione triangolare superiore (inclusa la diagonale)
        indices = torch.triu_indices(C, C, offset=0, device=gram.device)
        return gram[:, indices[0], indices[1]]

    def _hook_fn_gram(self, layer_name):
        def hook(module, input, output):
            self.extracted_gram_feature_maps[layer_name] = output
        return hook

    def _register_gram_hooks(self, visual_model):
        if not self.clip_model_name.startswith("RN"): # Solo per ResNet
            print("Skipping Gram hooks registration for non-ResNet model.")
            return
            
        target_modules = {
            'layer1': visual_model.layer1 if hasattr(visual_model, 'layer1') else None,
            'layer2': visual_model.layer2 if hasattr(visual_model, 'layer2') else None,
            'layer3': visual_model.layer3 if hasattr(visual_model, 'layer3') else None,
            'layer4': visual_model.layer4 if hasattr(visual_model, 'layer4') else None
        }
        
        for layer_key in self.layers_for_gram_config:
            module_to_hook = target_modules.get(layer_key)
            if module_to_hook:
                print(f"Registrazione hook Gram per: {layer_key}")
                self.feature_extractor_hooks.append(
                    module_to_hook.register_forward_hook(self._hook_fn_gram(layer_key))
                )
            else:
                print(f"Attenzione: layer '{layer_key}' non trovato in visual_model per la registrazione dell'hook Gram.")


    def _process_gram_features(self, batch_size):
        projected_gram_vectors_list = []
        
        for layer_name in self.layers_for_gram_config:
            if layer_name in self.extracted_gram_feature_maps:
                feature_map = self.extracted_gram_feature_maps[layer_name]
                gram_vector = self._get_gram_vector(feature_map)
                
                if layer_name in self.gram_layer_projections:
                    projected_gram_vector = self.gram_layer_projections[layer_name](gram_vector)
                    projected_gram_vectors_list.append(projected_gram_vector)
                else:
                    # Questo non dovrebbe accadere se _setup_gram_projections è corretto
                    print(f"Attenzione: nessuna proiezione definita per il layer Gram '{layer_name}'.")
            # else:
                # print(f"Attenzione: feature map per '{layer_name}' non trovata durante l'elaborazione Gram.")
                
        if projected_gram_vectors_list:
            # Concatena lungo la dimensione delle feature
            return torch.cat(projected_gram_vectors_list, dim=-1)
        else:
            # Ritorna un tensore vuoto con la dimensione del batch corretta se non ci sono feature Gram
            return torch.empty(batch_size, 0, device=self.device)


    def encode_image_with_style_adapter(self, image_input):
        self.extracted_gram_feature_maps.clear() # Pulisci le feature map precedenti

        # 1. Ottieni feature semantiche dal modello CLIP base (congelato)
        with torch.no_grad(): # Assicura che il modello CLIP base non venga addestrato
            # Il forward pass attraverso self.visual attiverà gli hook se registrati
            # e popolerà self.extracted_gram_feature_maps
            semantic_features = self.clip_model.encode_image(image_input) 
            # encode_image di solito fa già la normalizzazione, ma riverifichiamo
            semantic_features = F.normalize(semantic_features.float(), p=2, dim=-1)


        # 2. Elabora le feature Gram se ci sono proiezioni definite
        if self.total_gram_projected_dim > 0 and self.gram_layer_projections:
            batch_size = image_input.shape[0]
            gram_style_features = self._process_gram_features(batch_size)
            if gram_style_features.numel() == 0 and batch_size > 0 : # Se process_gram_features ritorna vuoto ma dovrebbe esserci qualcosa
                 # Fallback a tensore di zeri se le feature Gram non sono state estratte correttamente
                 # ma erano attese. Questo evita errori di concatenazione.
                 print("Attenzione: gram_style_features vuote nonostante total_gram_projected_dim > 0. Uso zeri.")
                 gram_style_features = torch.zeros(batch_size, self.total_gram_projected_dim, device=self.device, dtype=semantic_features.dtype)

        else:
            gram_style_features = torch.empty(image_input.shape[0], 0, device=self.device, dtype=semantic_features.dtype)

        # 3. Concatena feature semantiche e di stile (Gram)
        # Assicurati che semantic_features sia float per la concatenazione e l'adapter
        features_for_fusion = torch.cat([semantic_features.float(), gram_style_features.float()], dim=-1)
        
        # 4. Applica Fusion Adapter
        adapted_features = self.fusion_adapter(features_for_fusion)
        
        # 5. Normalizza l'output finale (pratica comune per le feature CLIP)
        return F.normalize(adapted_features, p=2, dim=-1)

    def encode_text(self, text_input):
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
        return F.normalize(text_features.float(), p=2, dim=-1)

    def encode_image(self, image): # Wrapper per compatibilità
        return self.encode_image_with_style_adapter(image)

    def forward(self, image_input, text_tokens):
        image_features = self.encode_image_with_style_adapter(image_input)
        text_features = self.encode_text(text_tokens)
        
        # Logit scale dal modello CLIP base
        # Converti text_features allo stesso tipo di image_features se necessario
        # (dovrebbero già essere float dopo la normalizzazione)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.t())
        return logits

    def _remove_gram_hooks(self):
        for hook in self.feature_extractor_hooks:
            hook.remove()
        self.feature_extractor_hooks = []
        # print("Hook Gram rimossi.")

    def __del__(self):
        self._remove_gram_hooks()


def meta_train_epoch(meta_model, meta_train_loader, meta_optimizer, 
                     inner_lr, inner_steps, criterion, device, epoch, total_epochs):
    meta_model.train() 
    total_meta_loss_accumulator = 0.0
    tasks_processed = 0
    
    for task_idx, (s_images, s_labels, q_images, q_labels, task_classnames) in enumerate(
            tqdm(meta_train_loader, desc=f'Meta-Epoca {epoch+1}/{total_epochs} [Meta-Train]')):

        if s_images.nelement() == 0 or q_images.nelement() == 0 or not task_classnames:
            continue

        s_images, s_labels = s_images.squeeze(0).to(device), s_labels.squeeze(0).to(device) 
        q_images, q_labels = q_images.squeeze(0).to(device), q_labels.squeeze(0).to(device)
        
        if s_images.nelement() == 0 or q_images.nelement() == 0: 
            continue

        meta_optimizer.zero_grad()
        
        task_text_prompts = [f"a painting by {c.replace('-', ' ').replace('_', ' ')}" for c in task_classnames]
        task_text_tokens = clip.tokenize(task_text_prompts).to(device)

        # Inner Loop con `higher` per MAML
        with higher.innerloop_ctx(meta_model, meta_optimizer, copy_initial_weights=True, track_higher_grads=True) as (fmodel, diffopt):
            # Imposta il learning rate per l'ottimizzatore dell'inner loop
            for group in diffopt.param_groups:
                group['lr'] = inner_lr

            # Adattamento al task (inner loop)
            for _ in range(inner_steps):
                support_logits = fmodel(s_images, task_text_tokens)
                support_loss = criterion(support_logits, s_labels)
                diffopt.step(support_loss) # Aggiorna i pesi del modello funzionale (fmodel)

            # Calcolo della loss sul query set usando i pesi adattati
            query_logits = fmodel(q_images, task_text_tokens) 
            query_loss_for_task = criterion(query_logits, q_labels)
            
            # Calcola i gradienti della loss del query set rispetto ai parametri originali del meta_model
            # (meta-gradienti), propagando attraverso gli step dell'inner loop.
            query_loss_for_task.backward() 
            
            total_meta_loss_accumulator += query_loss_for_task.item()
            tasks_processed += 1

        # Aggiorna i parametri del meta_model usando i meta-gradienti accumulati (o calcolati per questo task)
        meta_optimizer.step() 
    
    avg_meta_loss = total_meta_loss_accumulator / tasks_processed if tasks_processed > 0 else 0.0
    return avg_meta_loss

def meta_validate_epoch(meta_model, meta_val_loader, criterion, device, epoch, total_epochs, inner_lr, inner_steps):
    meta_model.eval() # Il modello base (adapter) è in modalità valutazione per l'adattamento
    total_query_loss_accumulator = 0.0
    total_query_acc_accumulator = 0.0
    tasks_processed = 0

    for task_idx, (s_images, s_labels, q_images, q_labels, task_classnames) in enumerate(
            tqdm(meta_val_loader, desc=f'Meta-Epoca {epoch+1}/{total_epochs} [Meta-Val]')):

        if s_images.nelement() == 0 or q_images.nelement() == 0 or not task_classnames:
            continue
        
        s_images, s_labels = s_images.squeeze(0).to(device), s_labels.squeeze(0).to(device)
        q_images, q_labels = q_images.squeeze(0).to(device), q_labels.squeeze(0).to(device)

        if s_images.nelement() == 0 or q_images.nelement() == 0:
            continue

        task_text_prompts = [f"a painting by {c.replace('-', ' ').replace('_', ' ')}" for c in task_classnames]
        task_text_tokens = clip.tokenize(task_text_prompts).to(device)
        
        # Approccio: Salvare e ripristinare lo stato degli adapter (usato qui per semplicità)
        original_gram_state = copy.deepcopy(meta_model.gram_layer_projections.state_dict()) if hasattr(meta_model, 'gram_layer_projections') and meta_model.gram_layer_projections else None
        original_fusion_state = copy.deepcopy(meta_model.fusion_adapter.state_dict()) if hasattr(meta_model, 'fusion_adapter') else None

        # Metti gli adapter in modalità train per l'adattamento
        if hasattr(meta_model, 'gram_layer_projections') and meta_model.gram_layer_projections: meta_model.gram_layer_projections.train()
        if hasattr(meta_model, 'fusion_adapter'): meta_model.fusion_adapter.train()
        
        adapter_params_to_tune = []
        if hasattr(meta_model, 'gram_layer_projections') and meta_model.gram_layer_projections:
            adapter_params_to_tune.extend(list(meta_model.gram_layer_projections.parameters()))
        if hasattr(meta_model, 'fusion_adapter'):
            adapter_params_to_tune.extend(list(meta_model.fusion_adapter.parameters()))

        if not adapter_params_to_tune: # Se non ci sono adapter, valuta direttamente
            with torch.no_grad():
                meta_model.eval() # Assicura che tutto il modello sia in eval
                query_logits = meta_model(q_images, task_text_tokens)
                query_loss = criterion(query_logits, q_labels)
                preds = query_logits.argmax(dim=-1)
                total_query_acc_accumulator += (preds == q_labels).float().sum().item()
                total_query_loss_accumulator += query_loss.item() * q_images.size(0) # Loss totale per il task
                tasks_processed += q_images.size(0) # Numero di campioni query
            continue

        temp_optimizer = optim.SGD(adapter_params_to_tune, lr=inner_lr) # Ottimizzatore per l'adattamento

        for _ in range(inner_steps):
            support_logits = meta_model(s_images, task_text_tokens) # Usa il modello originale con pesi che vengono adattati
            support_loss = criterion(support_logits, s_labels)
            temp_optimizer.zero_grad()
            support_loss.backward() # Calcola gradienti sui parametri dell'adapter
            temp_optimizer.step()   # Aggiorna i parametri dell'adapter
        
        meta_model.eval() # Riporta il modello (e i suoi adapter) in modalità valutazione per il query set
        
        with torch.no_grad():
            query_logits = meta_model(q_images, task_text_tokens) # Valuta con i pesi adattati
            query_loss = criterion(query_logits, q_labels)
            
            preds = query_logits.argmax(dim=-1)
            total_query_acc_accumulator += (preds == q_labels).float().sum().item() # Numero di corretti
            total_query_loss_accumulator += query_loss.item() * q_images.size(0) # Loss totale per il task
            tasks_processed += q_images.size(0) # Numero di campioni query

        # Ripristina lo stato originale degli adapter
        if original_gram_state and hasattr(meta_model, 'gram_layer_projections') and meta_model.gram_layer_projections:
            meta_model.gram_layer_projections.load_state_dict(original_gram_state)
        if original_fusion_state and hasattr(meta_model, 'fusion_adapter'):
            meta_model.fusion_adapter.load_state_dict(original_fusion_state)
        meta_model.eval() # Assicurati che sia in eval mode dopo il ripristino

    avg_query_loss = total_query_loss_accumulator / tasks_processed if tasks_processed > 0 else 0.0
    avg_query_acc = total_query_acc_accumulator / tasks_processed if tasks_processed > 0 else 0.0
    
    return avg_query_loss, avg_query_acc


# --- Funzione di Salvataggio Modello (adattata per meta-parametri) ---
def save_meta_model(model, epoch, val_acc, args, classnames_meta_train=None): # classnames_meta_train opzionale
    """Salvataggio del modello meta-addestrato."""
    save_path = args.output_model
    save_dict = {
        'epoch': epoch + 1, 
        'meta_val_acc': val_acc, 
        'args': vars(args), # Salva gli argomenti come dizionario
        'meta_train_classnames_sample': classnames_meta_train[:20] if classnames_meta_train else None, # Esempio
        'model_state_dict': model.state_dict() # Salva l'intero state_dict del modello
    }
    
    torch.save(save_dict, save_path)
    print(f"Meta-modello (intero state_dict) salvato in {save_path}")



# --- Main per Meta-Learning ---
def main_meta():
    parser = argparse.ArgumentParser(description="Meta-train CLIPWithStyleAdapter for Few-Shot Learning.")
    # Argomenti del modello CLIP e Adapter
    parser.add_argument('--clip_model_name', type=str, default='RN50', help="Nome del modello CLIP (es. RN50, ViT-B/32)")
    parser.add_argument('--fusion_bottleneck_dim', type=int, default=128, help="Dimensione bottleneck del Fusion Adapter")
    parser.add_argument('--gram_style_projection_dim', type=int, default=256, help="Dimensione totale delle Gram features proiettate")
    parser.add_argument('--layers_for_gram_rn50', type=str, nargs='+', default=['layer2', 'layer3'], help="Layer RN50 per Gram (es. layer1 layer2 layer3 layer4)")
    parser.add_argument('--dropout_rate_adapter', type=float, default=0.1, help="Dropout rate per il fusion adapter")
    parser.add_argument('--use_layernorm_adapter', type=lambda x: (str(x).lower() == 'true'), default=True, help="Usa LayerNorm nel fusion adapter")

    # Argomenti del Dataset e Meta-Learning
    parser.add_argument('--n_way', type=int, default=5, help="N-way per task FSL")
    parser.add_argument('--k_shot_support', type=int, default=1, help="K-shot (support set) per task FSL")
    parser.add_argument('--k_shot_query', type=int, default=5, help="Numero di campioni query per classe per task FSL") # Query per classe
    parser.add_argument('--num_tasks_per_epoch', type=int, default=100, help="Numero di task (episodi) per meta-epoca")
    
    # Argomenti di Ottimizzazione Meta
    parser.add_argument('--epochs', type=int, default=50, help="Numero di meta-epoche")
    parser.add_argument('--meta_lr', type=float, default=1e-4, help="Learning rate per il meta-optimizer (outer loop)")
    parser.add_argument('--inner_lr', type=float, default=0.01, help="Learning rate per l'inner loop (adattamento al task)")
    parser.add_argument('--inner_steps', type=int, default=5, help="Numero di step di adattamento nell'inner loop")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay per il meta-optimizer")
    
    # Altri argomenti
    parser.add_argument('--batch_size_dataloader', type=int, default=1, help="Batch size per il DataLoader (solitamente 1 per task-based meta-learning)")
    parser.add_argument('--num_workers_dataloader', type=int, default=0, help="Numero di workers per DataLoader (0 per debug su Windows)")
    parser.add_argument('--output_model', type=str, default='best_meta_style_adapter.pt', help="Path per salvare il miglior meta-modello")
    parser.add_argument('--seed', type=int, default=42, help="Seed per la riproducibilità")
    # Early stopping (opzionale, da implementare)
    # parser.add_argument('--early_stopping_patience', type=int, default=5)
    
    args = parser.parse_args()
    print(f"Argomenti Meta-Learning: {vars(args)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.benchmark = False # Per riproducibilità, può rallentare
        # torch.backends.cudnn.deterministic = True
    print(f"Utilizzo del device: {device}")

    try:
        dataset_path = find_artgraph_path()
    except FileNotFoundError as e:
        print(f"Errore critico: {e}")
        sys.exit(1)

    print(f"Creazione del modello CLIP ({args.clip_model_name}) con Style Adapter per Meta-Learning...")
    try:
        meta_model = CLIPWithStyleAdapter(
            clip_model_name=args.clip_model_name,
            fusion_bottleneck_dim=args.fusion_bottleneck_dim,
            gram_style_projection_dim=args.gram_style_projection_dim,
            layers_for_gram_rn50=args.layers_for_gram_rn50,
            dropout_rate=args.dropout_rate_adapter,
            use_layernorm_adapter=args.use_layernorm_adapter,
            device=device
        ).to(device)
    except Exception as e:
        print(f"Errore durante la creazione del modello CLIPWithStyleAdapter: {e}")
        sys.exit(1)

    # Il meta-optimizer aggiorna i parametri "base" degli adapter
    meta_trainable_params = []
    if hasattr(meta_model, 'gram_layer_projections') and meta_model.gram_layer_projections:
        meta_trainable_params.extend(list(meta_model.gram_layer_projections.parameters()))
    if hasattr(meta_model, 'fusion_adapter'):
        meta_trainable_params.extend(list(meta_model.fusion_adapter.parameters()))
    
    if not meta_trainable_params:
        print("ERRORE: Nessun parametro addestrabile (adapter) trovato nel modello per il meta-learning.")
        # sys.exit(1) # Potrebbe essere intenzionale se gram_style_projection_dim è 0 e si vuole addestrare solo fusion_adapter senza gram
        # O se fusion_adapter è l'unico componente. Controllare la logica.
        # Per ora, continuiamo, ma è un avviso importante.
        print("Attenzione: Nessun parametro specifico per gram_layer_projections. Solo fusion_adapter sarà addestrato (se presente).")
        if not hasattr(meta_model, 'fusion_adapter'):
             print("ERRORE CRITICO: Né gram_layer_projections né fusion_adapter hanno parametri. Nulla da addestrare.")
             sys.exit(1)


    meta_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, meta_trainable_params), 
                                 lr=args.meta_lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print("Caricamento dei Meta-Dataset Artgraph...")
    try:
        meta_train_dataset = MetaArtgraphDataset(dataset_path, 
                                                 num_tasks=args.num_tasks_per_epoch, 
                                                 n_way=args.n_way, 
                                                 k_shot_support=args.k_shot_support, 
                                                 k_shot_query=args.k_shot_query, 
                                                 transform=meta_model.preprocess, 
                                                 split='meta_train', seed=args.seed)
        meta_val_dataset = MetaArtgraphDataset(dataset_path, 
                                               num_tasks=max(10, args.num_tasks_per_epoch // 5), # Meno task per la validazione
                                               n_way=args.n_way, 
                                               k_shot_support=args.k_shot_support, 
                                               k_shot_query=args.k_shot_query, 
                                               transform=meta_model.preprocess, 
                                               split='meta_val', seed=args.seed + 1) 
    except ValueError as e:
        print(f"Errore durante la creazione del dataset: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Errore File Not Found durante la creazione del dataset: {e}")
        sys.exit(1)


    # DataLoader per i task (batch_size=1 perché ogni "campione" è un intero task)
    meta_train_loader = torch.utils.data.DataLoader(meta_train_dataset, batch_size=args.batch_size_dataloader, 
                                                    shuffle=True, num_workers=args.num_workers_dataloader,
                                                    collate_fn=lambda x: x[0]) # Prende il primo (e unico) task dal batch
    meta_val_loader = torch.utils.data.DataLoader(meta_val_dataset, batch_size=args.batch_size_dataloader, 
                                                  shuffle=False, num_workers=args.num_workers_dataloader,
                                                  collate_fn=lambda x: x[0])


    print(f"Inizio Meta-Addestramento per {args.epochs} epoche...")
    best_meta_val_acc = 0.0
    # patience_counter = 0 # Per early stopping
    
    for epoch in range(args.epochs):
        meta_train_loss, _ = meta_train_epoch(
            meta_model, meta_train_loader, meta_optimizer,
            args.inner_lr, args.inner_steps, criterion, device, epoch, args.epochs
        )
        
        meta_val_loss, meta_val_acc = meta_validate_epoch(
            meta_model, meta_val_loader, criterion, device, epoch, args.epochs, 
            args.inner_lr, args.inner_steps
        )
        
        print(f'Meta-Epoca {epoch+1}/{args.epochs}: Meta-Train Loss={meta_train_loss:.4f}, '
              f'Meta-Val Loss={meta_val_loss:.4f}, Meta-Val Acc={meta_val_acc:.4f}')
        sys.stdout.flush() # Forza la stampa immediata

        if meta_val_acc > best_meta_val_acc:
            best_meta_val_acc = meta_val_acc
            save_meta_model(meta_model, epoch, best_meta_val_acc, args, 
                            classnames_meta_train=meta_train_dataset.classnames if meta_train_dataset else None)
            # patience_counter = 0
        # else:
        #     patience_counter += 1

        # if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
        #     print(f"Early stopping dopo {epoch+1} epoche.")
        #     break
            
    print(f'Meta-Addestramento completato. Miglior Meta-Val Acc: {best_meta_val_acc:.4f}')
    if hasattr(meta_model, '_remove_gram_hooks'): # Assicura la pulizia degli hook
        meta_model._remove_gram_hooks()

if __name__ == '__main__':
    main_meta()