# Adapter.py (Modificato)

import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR

import clip

# --- Funzione find_artgraph_path (invariata) ---
def find_artgraph_path():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # PROVA QUESTO PERCORSO PER IL TUO CASO SPECIFICO SE NON FUNZIONA
    # base_path = "/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO"
    base_path = current_script_dir # O specifica il percorso base di TIMO
    
    # Percorsi da controllare, relativi allo script o a una directory base TIMO
    # Assicurati che la struttura $DATA/artgraph esista all'interno di uno di questi
    data_root="/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO/$DATA"
    artgraph_path = os.path.join(data_root, "artgraph")
    abs_artgraph_path = os.path.abspath(artgraph_path)
    if os.path.exists(abs_artgraph_path) and os.path.isdir(abs_artgraph_path):
        print(f"Dataset artgraph trovato in: {abs_artgraph_path}")
        return abs_artgraph_path

# --- Classe ArtgraphDataset (invariata) ---
class ArtgraphDataset(data.Dataset):
    def __init__(self, root_dir, split='train', transform=None, train_ratio=0.7, val_ratio=0.15, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.split = split
        
        random.seed(seed)
        
        self.classnames = sorted([d for d in os.listdir(self.images_dir) 
                                 if os.path.isdir(os.path.join(self.images_dir, d))])
        if not self.classnames:
            raise FileNotFoundError(f"Nessuna sottocartella (classe/artista) trovata in {self.images_dir}")
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classnames)}
        
        all_samples_by_class = {label: [] for label in self.class_to_idx.values()}
        
        for artist in self.classnames:
            artist_dir = os.path.join(self.images_dir, artist)
            artist_label = self.class_to_idx[artist]
            if os.path.isdir(artist_dir):
                for img_name in os.listdir(artist_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        all_samples_by_class[artist_label].append(os.path.join(artist_dir, img_name))
        
        self.samples = []
        for label, img_paths in all_samples_by_class.items():
            random.shuffle(img_paths)
            num_samples_class = len(img_paths)
            
            train_end = int(train_ratio * num_samples_class)
            val_end = train_end + int(val_ratio * num_samples_class)
            
            if self.split == 'train':
                split_paths = img_paths[:train_end]
            elif self.split == 'val':
                split_paths = img_paths[train_end:val_end]
            else:
                split_paths = img_paths[val_end:]
            
            for img_path in split_paths:
                self.samples.append((img_path, label))
        
        random.shuffle(self.samples)
        
        print(f"Caricato {len(self.samples)} immagini per lo split '{split}' dal dataset in '{root_dir}'. Classi: {len(self.classnames)}")
        if len(self.samples) == 0:
            print(f"ATTENZIONE: Nessun campione caricato per lo split '{split}'. Controlla i percorsi e i ratio.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Errore nel caricamento dell'immagine {img_path}: {e}. Salto e provo un'altra.")
            random_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_idx)

# --- Classe Adapter (invariata) ---
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim_override=None, dropout_rate=0.1, use_layernorm=True):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        
        # Se output_dim_override è specificato, up_project proietta a quella dimensione
        # Altrimenti, proietta di nuovo a input_dim (comportamento originale)
        actual_output_dim = output_dim_override if output_dim_override is not None else input_dim
        self.up_project = nn.Linear(bottleneck_dim, actual_output_dim)
        
        nn.init.xavier_uniform_(self.down_project.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.down_project.bias)
        
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(bottleneck_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x # La residual connection è sull'input originale x
                     # Se l'output di up_project ha dimensione diversa, la residual additiva non è banale
                     # Per ora, assumiamo che l'Adapter sia usato in un modo che gestisce questo
                     # o che output_dim_override == input_dim
        
        original_x_for_residual = x # Conserva l'input per la connessione residuale

        x_down = self.down_project(x)
        
        if self.use_layernorm:
            x_activated = self.layer_norm(x_down)
        else:
            x_activated = x_down

        x_activated = self.activation(x_activated)
        x_dropped = self.dropout(x_activated)
        
        x_up = self.up_project(x_dropped)
        
        # Gestione della connessione residuale se le dimensioni cambiano
        if x_up.shape[-1] == original_x_for_residual.shape[-1]:
            alpha = 0.1 
            combined = alpha * x_up + original_x_for_residual
            return combined
        else:
            # Se le dimensioni sono diverse, non possiamo fare una somma residuale diretta.
            # Restituiamo solo l'output dell'up_project.
            # L'utente del modulo dovrà gestire la fusione se necessario.
            # print(f"Attenzione: Dimensioni input ({original_x_for_residual.shape[-1]}) e output ({x_up.shape[-1]}) dell'Adapter diverse. Nessuna connessione residuale applicata.")
            return x_up


# --- NUOVA CLASSE: CLIPWithStyleAdapter ---
class CLIPWithStyleAdapter(nn.Module):
    def __init__(self, clip_model_name, fusion_bottleneck_dim, gram_style_projection_dim, device, 
                 layers_for_gram_rn50=['layer1', 'layer2', 'layer3', 'layer4'], 
                 dropout_rate=0.1, use_layernorm_adapter=True):
        super(CLIPWithStyleAdapter, self).__init__()
        
        # Carica il modello CLIP base
        self.clip_model, self.preprocess = clip.load(clip_model_name if clip_model_name != "CustomRN50" else "RN50", device=device)
        
        # Aggiungi l'attributo visual per mantenere compatibilità con l'interfaccia CLIP
        self.visual = self.clip_model.visual
        
        # Inizializza gli attributi necessari per gli hook
        self.feature_extractor_hooks = []
        self.extracted_gram_feature_maps = {}
        
        # Parametri per l'adapter e le feature gram
        self.device = device
        self.clip_model_name = clip_model_name
        self.layers_for_gram_config = layers_for_gram_rn50  # Ora è definito correttamente

        if not self.clip_model_name.startswith("RN"):
            print(f"ATTENZIONE: L'estrazione Gram è implementata specificamente per RN50. Il modello {self.clip_model_name} potrebbe non funzionare correttamente.")

        # Dimensione feature semantica
        if hasattr(self.clip_model.visual, 'output_dim'): # Es. ViT
            self.semantic_feature_dim = self.clip_model.visual.output_dim
        else: # Es. RN50, output dell'attnpool prima della proiezione finale se esiste
            try: # Prova ad accedere all'output dell'attention pool per RN50
                 # L'output di encode_image è già quello proiettato, quindi usiamo la sua dimensione
                dummy_image = torch.randn(1, 3, self.clip_model.visual.input_resolution, self.clip_model.visual.input_resolution).to(device)
                self.semantic_feature_dim = self.clip_model.encode_image(dummy_image).shape[-1]
            except Exception:
                print("Impossibile determinare semantic_feature_dim automaticamente. Usando 1024 come fallback per RN50.")
                self.semantic_feature_dim = 1024 # Fallback comune per RN50 feature pre-proiezione

        print(f"Dimensione feature semantica (output di encode_image): {self.semantic_feature_dim}")

        self.feature_extractor_hooks = []
        self.extracted_gram_feature_maps = {} # Nome del layer -> feature map

        # Registra gli hook per estrarre le feature map per le Gram
        self._register_gram_hooks(self.clip_model.visual)
        
        # Calcola la dimensione delle Gram features combinate (dopo vettorizzazione)
        # Questo è cruciale e dipende dai layer scelti e dalla loro dimensionalità
        # Per RN50: layer1=256, layer2=512, layer3=1024, layer4=2048 canali (output del blocco)
        # Ma le feature map *all'interno* dei blocchi (prima del downsampling) hanno C diverse.
        # Es. RN50:
        # - self.clip_model.visual.layer1[-1].conv3.out_channels -> 256 (per RN50 CLIP)
        # - self.clip_model.visual.layer2[-1].conv3.out_channels -> 512
        # - self.clip_model.visual.layer3[-1].conv3.out_channels -> 1024
        # - self.clip_model.visual.layer4[-1].conv3.out_channels -> 2048
        # Se vettorizziamo la Gram (C*C), la dimensione esplode.
        # Useremo una strategia di appiattimento della triangolare superiore della Gram.
        # E proietteremo ogni Gram vettorizzata a una dimensione fissa prima di concatenare.

        self.gram_layer_projections = nn.ModuleDict()
        current_total_gram_projected_dim = 0
        
        # Dimensioni canali per RN50 (output dei blocchi) - Adattare se si usano sub-layer
        # Questi sono gli output *dopo* la bottleneck expansion e la residual.
        # Per le Gram, è meglio usare le feature map *prima* di un forte downsampling o dell'ultimo ReLU.
        # I nomi usati in _register_gram_hooks devono corrispondere a questi.
        # Queste C sono l'output del blocco, non necessariamente il C della Gram matrix desiderata.
        # Le C per le Gram dovrebbero essere quelle delle feature map estratte.
        # Placeholder: otteniamo C dinamicamente dopo il primo forward pass fittizio.
        # Per ora, assumiamo di poterle proiettare individualmente.
        self.per_gram_vector_projection_dim = gram_style_projection_dim // len(self.layers_for_gram_config) if self.layers_for_gram_config else gram_style_projection_dim


        # Per calcolare dinamicamente C e la dim delle Gram vettorizzate, facciamo un forward fittizio
        if self.feature_extractor_hooks: # Solo se gli hook sono stati registrati
            print("Esecuzione forward pass fittizio per determinare dimensioni Gram...")
            try:
                with torch.no_grad():
                    dummy_image = torch.randn(1, 3, self.clip_model.visual.input_resolution, self.clip_model.visual.input_resolution).to(device)
                    self.clip_model.visual(dummy_image) # Attiva gli hook
                
                for layer_name in self.layers_for_gram_config:
                    if layer_name in self.extracted_gram_feature_maps:
                        C = self.extracted_gram_feature_maps[layer_name].shape[1] # B,C,H,W
                        gram_vector_dim = C * (C + 1) // 2 # Triangolare superiore appiattita
                        self.gram_layer_projections[layer_name.replace('.','_')] = nn.Linear(gram_vector_dim, self.per_gram_vector_projection_dim).to(device)
                        current_total_gram_projected_dim += self.per_gram_vector_projection_dim
                        print(f"Layer Gram '{layer_name}': C={C}, Dim Vettore Gram={gram_vector_dim}, Proiettato a {self.per_gram_vector_projection_dim}")
                    else:
                        print(f"ATTENZIONE: Feature map per {layer_name} non trovata dopo il forward fittizio.")
                self.extracted_gram_feature_maps = {} # Pulisci dopo il pass fittizio
            except Exception as e:
                print(f"Errore durante il forward pass fittizio per le Gram: {e}. Le proiezioni Gram potrebbero non essere inizializzate.")
                current_total_gram_projected_dim = gram_style_projection_dim # Fallback

        else: # Nessun hook, usa dimensione fallback
            current_total_gram_projected_dim = gram_style_projection_dim


        self.total_gram_projected_dim = current_total_gram_projected_dim
        if self.total_gram_projected_dim == 0 and self.layers_for_gram_config:
            print(f"ATTENZIONE: total_gram_projected_dim è 0 ma layers_for_gram è configurato. Forzando a gram_style_projection_dim.")
            self.total_gram_projected_dim = gram_style_projection_dim


        # Fusion Adapter (usa la tua classe Adapter)
        # Input: feature semantiche + Gram features proiettate e concatenate
        fusion_input_dim = self.semantic_feature_dim + self.total_gram_projected_dim
        
        # L'Adapter deve produrre un output della stessa dimensione di semantic_feature_dim
        # Quindi, output_dim_override = self.semantic_feature_dim
        self.fusion_adapter = Adapter(
            input_dim=fusion_input_dim,
            bottleneck_dim=fusion_bottleneck_dim,
            output_dim_override=self.semantic_feature_dim, # Cruciale!
            dropout_rate=dropout_rate,
            use_layernorm=use_layernorm_adapter
        ).to(device)
        
        print(f"Fusion Adapter input dim: {fusion_input_dim}, bottleneck: {fusion_bottleneck_dim}, output dim: {self.semantic_feature_dim}")
        if self.total_gram_projected_dim > 0:
             print(f"Dimensione totale Gram features proiettate (input per fusione): {self.total_gram_projected_dim}")
        else:
            print("Nessuna feature Gram sarà usata nella fusione.")


    def _get_gram_vector(self, feature_map_batch): # feature_map_batch: B x C x H x W
        B, C, H, W = feature_map_batch.size()
        features = feature_map_batch.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2)) # B x C x C
        gram.div_(H * W) # Normalizzazione opzionale
        
        # Vettorizzazione: appiattire la triangolare superiore (inclusa la diagonale)
        # Questo è un modo comune per vettorizzare una matrice simmetrica
        indices = torch.triu_indices(C, C, offset=0, device=gram.device)
        gram_vectors = gram[:, indices[0], indices[1]] # B x (C*(C+1)//2)
        return gram_vectors

    def _hook_fn_gram(self, layer_name):
        def hook(module, input, output):
            self.extracted_gram_feature_maps[layer_name] = output # .detach() se non serve il grad qui
        return hook

    def _register_gram_hooks(self, visual_model):
        # Specifica per RN50 di CLIP. Adattare se il modello è diverso!
        # I nomi dei layer devono corrispondere a moduli esistenti.
        # Per RN50 di CLIP, i layer sono accessibili tramite `visual_model.layerX`
        if self.clip_model_name.startswith("RN"):
            target_modules = {
                'layer1': visual_model.layer1, # Output del blocco layer1
                'layer2': visual_model.layer2, # Output del blocco layer2
                'layer3': visual_model.layer3, # Output del blocco layer3
                'layer4': visual_model.layer4  # Output del blocco layer4
            }
            for layer_key_config in self.layers_for_gram_config: # es. 'layer2'
                if layer_key_config in target_modules:
                    module_to_hook = target_modules[layer_key_config]
                    # Potresti voler agganciare un sottomodulo specifico, es. l'ultima conv del blocco
                    # Per semplicità, agganciamo l'output del blocco intero.
                    hook = module_to_hook.register_forward_hook(self._hook_fn_gram(layer_key_config))
                    self.feature_extractor_hooks.append(hook)
                    print(f"Registrato hook per Gram su: {layer_key_config}")
                else:
                    print(f"Attenzione: layer '{layer_key_config}' non trovato per Gram in {self.clip_model_name}.")
        else:
            print(f"Estrazione Gram non configurata per modello: {self.clip_model_name}")

    def _remove_gram_hooks(self):
        for hook in self.feature_extractor_hooks:
            hook.remove()
        self.feature_extractor_hooks = []

    def encode_image_with_style_adapter(self, image_input):
        self.extracted_gram_feature_maps = {} # Pulisci prima di ogni forward

        # 1. Ottieni feature semantiche (output finale di encode_image)
        # Questo passaggio attiverà anche gli hook per popolare extracted_gram_feature_maps
        with torch.no_grad(): # L'encoder CLIP è congelato
            semantic_features = self.clip_model.encode_image(image_input).float() # Converti esplicitamente in float

        # 2. Estrai, vettorizza e proietta le Gram features
        projected_gram_vectors_list = []
        if self.total_gram_projected_dim > 0: # Solo se le Gram sono configurate e inizializzate
            for layer_name_config in self.layers_for_gram_config:
                layer_name_safe_dict_key = layer_name_config.replace('.', '_')
                if layer_name_config in self.extracted_gram_feature_maps and \
                   layer_name_safe_dict_key in self.gram_layer_projections:
                    feature_map = self.extracted_gram_feature_maps[layer_name_config]
                    gram_vector_raw = self._get_gram_vector(feature_map.float()) # Converti esplicitamente in float
                    
                    # Proietta il vettore Gram di questo layer
                    projected_gram = self.gram_layer_projections[layer_name_safe_dict_key](gram_vector_raw)
                    projected_gram_vectors_list.append(projected_gram)
                else:
                    # Fallback: aggiungi zeri se un layer non ha prodotto output o non ha proiezione
                    projected_gram_vectors_list.append(
                        torch.zeros(image_input.size(0), self.per_gram_vector_projection_dim, device=self.device)
                    )
            
            if projected_gram_vectors_list:
                combined_projected_gram_features = torch.cat(projected_gram_vectors_list, dim=1) # B x total_gram_projected_dim
            else: # Se la lista è vuota per qualche motivo (nessun layer valido)
                 combined_projected_gram_features = torch.zeros(image_input.size(0), self.total_gram_projected_dim, device=self.device)
        else: # Nessuna feature Gram da usare
            combined_projected_gram_features = torch.zeros(image_input.size(0), 0, device=self.device) # Tensore vuoto sulla dim 1 se non ci sono gram

        # 3. Concatena feature semantiche e Gram features processate
        if combined_projected_gram_features.numel() > 0 : # Se ci sono gram features
            features_for_fusion = torch.cat([semantic_features.float(), combined_projected_gram_features.float()], dim=1)
        else: # Solo semantiche se le gram non sono disponibili
            features_for_fusion = semantic_features.float()

        # 4. Applica il Fusion Adapter
        adapted_features = self.fusion_adapter(features_for_fusion) # B x D_semantic
            
        final_features = torch.nn.functional.normalize(adapted_features, p=2, dim=-1)
        return final_features.float()  # Assicura che output sia float

    def encode_text(self, text_input):
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        return text_features.float()

    def forward(self, image_input, text_tokens):
        image_features_styled = self.encode_image_with_style_adapter(image_input)
        text_features_encoded = self.encode_text(text_tokens)
        
        logit_scale = self.clip_model.logit_scale.exp().float()
        logits = logit_scale * (image_features_styled @ text_features_encoded.t())
        
        return logits

    def encode_image(self, image):
        """
        Codifica un'immagine usando la pipeline completa (CLIP + StyleAdapter)
        """
        # Estrai feature base e applica l'adapter
        return self.encode_image_with_style_adapter(image)

    def __del__(self): # Pulisce gli hook quando l'oggetto viene distrutto
        self._remove_gram_hooks()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_name', type=str, default='RN50', help="Nome del modello CLIP (es. RN50, ViT-B/32)")
    # Parametri per il Fusion Adapter
    parser.add_argument('--fusion_bottleneck_dim', type=int, default=128, help="Dimensione bottleneck del Fusion Adapter")
    # Parametri per le Gram Style Features
    parser.add_argument('--gram_style_projection_dim', type=int, default=256, help="Dimensione totale a cui proiettare le Gram features combinate")
    parser.add_argument('--layers_for_gram_rn50', type=str, nargs='+', default=['layer2', 'layer3'], help="Layer di RN50 da cui estrarre Gram (es. layer1 layer2 layer3 layer4)")

    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate") # Ridotto un po'
    parser.add_argument('--epochs', type=int, default=5, help="Numero di epoche")
    parser.add_argument('--batch_size', type=int, default=16, help="Dimensione del batch")
    parser.add_argument('--output_model', type=str, default='best_style_adapted_clip_artgraph.pt')
    parser.add_argument('--seed', type=int, default=42, help="Seed per la riproducibilità")
    parser.add_argument('--dropout_rate_adapter', type=float, default=0.1, help="Dropout rate per il fusion adapter")
    parser.add_argument('--use_layernorm_adapter', type=bool, default=True, help="Usa LayerNorm nel fusion adapter")
    parser.add_argument('--warmup_steps', type=int, default=200, help="Numero di warmup steps") # Ridotto
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay") # Ridotto

    args = parser.parse_args()
    print(f"Argomenti: {args}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"Utilizzo del device: {device}")

    dataset_path = find_artgraph_path()
    
    print(f"Creazione del modello CLIP ({args.clip_model_name}) con Style Adapter...")
    model = CLIPWithStyleAdapter(
        clip_model_name=args.clip_model_name,
        fusion_bottleneck_dim=args.fusion_bottleneck_dim,
        gram_style_projection_dim=args.gram_style_projection_dim,
        layers_for_gram_rn50=args.layers_for_gram_rn50,
        dropout_rate=args.dropout_rate_adapter,
        use_layernorm_adapter=args.use_layernorm_adapter,
        device=device
    ).to(device)
    
    # Parametri da addestrare: gram_layer_projections e fusion_adapter
    trainable_params = []
    if hasattr(model, 'gram_layer_projections') and model.gram_layer_projections:
        trainable_params.extend(list(model.gram_layer_projections.parameters()))
    if hasattr(model, 'fusion_adapter'):
        trainable_params.extend(list(model.fusion_adapter.parameters()))
    
    if not trainable_params:
        print("ATTENZIONE: Nessun parametro addestrabile trovato nel modello. Controllare l'inizializzazione.")
        return
    else:
        print(f"Numero di gruppi di parametri addestrabili: {len(trainable_params)} (non il numero totale di tensori)")


    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return 1.0 
    scheduler = LambdaLR(optimizer, lr_lambda)

    print("Caricamento del dataset Artgraph...")
    train_dataset = ArtgraphDataset(dataset_path, split='train', transform=model.preprocess, seed=args.seed)
    val_dataset = ArtgraphDataset(dataset_path, split='val', transform=model.preprocess, seed=args.seed)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Errore: Dataset vuoto.")
        return

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Preparazione dei token di classe...")
    text_prompts = [f"a painting by {c.replace('-', ' ').replace('_', ' ')}" for c in train_dataset.classnames]
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    print(f"Inizio addestramento per {args.epochs} epoche...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Metti in modalità training i moduli addestrabili
        if hasattr(model, 'gram_layer_projections'): model.gram_layer_projections.train()
        if hasattr(model, 'fusion_adapter'): model.fusion_adapter.train()
        
        total_loss = 0
        correct_train = 0
        total_train_samples = 0
        
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoca {epoch+1}/{args.epochs} [Train]')):
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images, text_tokens)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"!!! ATTENZIONE: NaN/Inf logits train batch {i} epoca {epoch+1} !!!")
                continue
            loss = criterion(logits, labels)
            if torch.isnan(loss):
                print(f"!!! ATTENZIONE: NaN loss train batch {i} epoca {epoch+1} !!!")
                continue

            optimizer.zero_grad()
            loss.backward()
            # Clip grad norm per i parametri addestrabili
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train_samples += images.size(0)
        
        avg_train_loss = total_loss / total_train_samples if total_train_samples > 0 else 0
        avg_train_acc = correct_train / total_train_samples if total_train_samples > 0 else 0
        
        # Validazione
        if hasattr(model, 'gram_layer_projections'): model.gram_layer_projections.eval()
        if hasattr(model, 'fusion_adapter'): model.fusion_adapter.eval()
        correct_val = 0
        total_val_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoca {epoch+1}/{args.epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                logits = model(images, text_tokens)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"!!! ATTENZIONE: NaN/Inf logits val !!!")
                    continue
                preds = logits.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val_samples += images.size(0)
        
        avg_val_acc = correct_val / total_val_samples if total_val_samples > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoca {epoch+1}: LR={current_lr:.2e}, Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}, Val Acc={avg_val_acc:.4f}')
        sys.stdout.flush()

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            # Salva solo i moduli addestrabili
            save_dict = {'epoch': epoch + 1, 'val_acc': avg_val_acc, 'args': args, 'classnames': train_dataset.classnames}
            if hasattr(model, 'gram_layer_projections') and model.gram_layer_projections:
                 save_dict['gram_layer_projections_state_dict'] = model.gram_layer_projections.state_dict()
            if hasattr(model, 'fusion_adapter'):
                 save_dict['fusion_adapter_state_dict'] = model.fusion_adapter.state_dict()
            
            torch.save(save_dict, args.output_model)
            print(f'Miglior modello (moduli add.) salvato in {args.output_model} con Val Acc: {best_val_acc:.4f}')
            sys.stdout.flush()
    
    print(f'Addestramento completato. Miglior Val Acc: {best_val_acc:.4f}')
    # Rimuovi gli hook alla fine
    model._remove_gram_hooks()

if __name__ == '__main__':
    main()