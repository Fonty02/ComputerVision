# Adapter.py (Ottimizzato)

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

# --- Funzione find_artgraph_path (migliorata) ---
def find_artgraph_path():
    # Cerca in più posizioni possibili
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lista di possibili percorsi relativi/assoluti da provare
    possible_paths = [
        # Percorso relativo al script
        os.path.join(current_script_dir, "..", "$DATA", "artgraph_complementary"),
        # Percorso assoluto Windows (dal percorso del progetto)
        os.path.join(os.path.dirname(os.path.dirname(current_script_dir)), "$DATA", "artgraph_complementary"),
        # Fallback sul path Linux precedente
        os.path.abspath("/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO/$DATA/artgraph_complementary")
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            print(f"Dataset artgraph trovato in: {path}")
            return path
            
    # Se arriviamo qui, non abbiamo trovato il dataset
    raise FileNotFoundError("Dataset artgraph non trovato. Controlla i percorsi o crea la directory.")

# --- Classe ArtgraphDataset (invariata) ---
class ArtgraphDataset(data.Dataset):
    def __init__(self, root_dir, split='train', transform=None, train_ratio=0.7, val_ratio=0.3, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.split = split
        
        random.seed(seed)
        
        self.classnames = sorted([d for d in os.listdir(self.images_dir) 
                                 if os.path.isdir(os.path.join(self.images_dir, d))])
        if not self.classnames:
            raise FileNotFoundError(f"Nessuna sottocartella (classe/artista) trovata in {self.images_dir}")
            
        self.class_to_idx = {cls_name: i + 150 for i, cls_name in enumerate(self.classnames)} # MODIFICATO: le etichette partono da 150
        
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
            else:  # 'test'
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

# --- Classe Adapter (ottimizzata) ---
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim_override=None, dropout_rate=0.1, use_layernorm=True):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.output_dim = output_dim_override if output_dim_override is not None else input_dim
        self.up_project = nn.Linear(bottleneck_dim, self.output_dim)
        
        # Inizializzazione pesi
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
        original_x = x  # Per la connessione residuale
        
        # Pipeline principale dell'adapter
        x = self.down_project(x)
        if self.use_layernorm:
            x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        
        # Connessione residuale condizionale
        if x.shape[-1] == original_x.shape[-1]:
            alpha = 0.1 
            x = alpha * x + original_x
            
        return x


# --- CLASSE: CLIPWithStyleAdapter (ottimizzata) ---
class CLIPWithStyleAdapter(nn.Module):
    def __init__(self, clip_model_name, fusion_bottleneck_dim, gram_style_projection_dim, device, 
                 layers_for_gram_rn50=None, 
                 dropout_rate=0.1, use_layernorm_adapter=True):
        super(CLIPWithStyleAdapter, self).__init__()
        
        # Valori predefiniti sicuri
        if layers_for_gram_rn50 is None:
            layers_for_gram_rn50 = ['layer2', 'layer3']
            
        # Carica il modello CLIP base
        model_name = "RN50" if clip_model_name == "CustomRN50" else clip_model_name
        self.clip_model, self.preprocess = clip.load(model_name, device=device)
        
        # Mantieni compatibilità con l'interfaccia CLIP
        self.visual = self.clip_model.visual
        
        # Inizializza per gli hook feature
        self.feature_extractor_hooks = []
        self.extracted_gram_feature_maps = {}
        
        # Parametri base
        self.device = device
        self.clip_model_name = clip_model_name
        self.layers_for_gram_config = layers_for_gram_rn50

        # Avviso per modelli non RN
        if not self.clip_model_name.startswith("RN"):
            print(f"ATTENZIONE: L'estrazione Gram è ottimizzata per RN50. Il modello {self.clip_model_name} potrebbe non funzionare correttamente.")

        # Determina dimensione feature semantica
        self.semantic_feature_dim = self._get_semantic_feature_dim()
        print(f"Dimensione feature semantica (output di encode_image): {self.semantic_feature_dim}")

        # Registra hook per feature Gram 
        if layers_for_gram_rn50:
            self._register_gram_hooks(self.clip_model.visual)
            # Inizializza proiezioni Gram
            self.total_gram_projected_dim, self.gram_layer_projections = self._setup_gram_projections(gram_style_projection_dim)
            print(f"Dimensione totale Gram features proiettate: {self.total_gram_projected_dim}")
        else:
            self.total_gram_projected_dim = 0
            self.gram_layer_projections = nn.ModuleDict()
            print("Nessuna feature Gram sarà utilizzata (layers_for_gram_rn50 vuoto)")

        # Fusion Adapter
        fusion_input_dim = self.semantic_feature_dim + self.total_gram_projected_dim
        self.fusion_adapter = Adapter(
            input_dim=fusion_input_dim,
            bottleneck_dim=fusion_bottleneck_dim,
            output_dim_override=self.semantic_feature_dim,
            dropout_rate=dropout_rate,
            use_layernorm=use_layernorm_adapter
        ).to(device)
        
        print(f"Fusion Adapter input dim: {fusion_input_dim}, bottleneck: {fusion_bottleneck_dim}, output dim: {self.semantic_feature_dim}")
        
    def _get_semantic_feature_dim(self):
        """Determina la dimensione della feature semantica in modo robusto"""
        try:
            if hasattr(self.clip_model.visual, 'output_dim'):
                return self.clip_model.visual.output_dim
            else:
                # Test con immagine fittizia per RN50
                resolution = self.clip_model.visual.input_resolution
                dummy_image = torch.randn(1, 3, resolution, resolution).to(self.device)
                with torch.no_grad():
                    return self.clip_model.encode_image(dummy_image).shape[-1]
        except Exception as e:
            print(f"Errore nel determinare semantic_feature_dim: {e}. Usando 1024 come fallback.")
            return 1024

    def _setup_gram_projections(self, gram_style_projection_dim):
        """Configura le proiezioni per le feature Gram"""
        if not self.layers_for_gram_config:
            return 0, nn.ModuleDict()
            
        per_gram_vector_projection_dim = gram_style_projection_dim // len(self.layers_for_gram_config)
        gram_layer_projections = nn.ModuleDict()
        total_gram_dim = 0
        
        # Forward fittizia per determinare dimensioni
        try:
            with torch.no_grad():
                resolution = self.clip_model.visual.input_resolution
                dummy_image = torch.randn(1, 3, resolution, resolution).to(self.device)
                self.clip_model.visual(dummy_image)
            
            for layer_name in self.layers_for_gram_config:
                if layer_name in self.extracted_gram_feature_maps:
                    C = self.extracted_gram_feature_maps[layer_name].shape[1]
                    gram_vector_dim = C * (C + 1) // 2  # Triangolare superiore
                    dict_key = layer_name.replace('.', '_')
                    gram_layer_projections[dict_key] = nn.Linear(gram_vector_dim, per_gram_vector_projection_dim).to(self.device)
                    total_gram_dim += per_gram_vector_projection_dim
                    print(f"Layer Gram '{layer_name}': C={C}, Dim Vettore={gram_vector_dim}, Proiettato a {per_gram_vector_projection_dim}")
            
            # Pulizia dopo forward fittizia
            self.extracted_gram_feature_maps.clear()
            
        except Exception as e:
            print(f"Errore inizializzazione Gram projections: {e}. Usando dimensione di fallback.")
            total_gram_dim = gram_style_projection_dim
            
        return total_gram_dim, gram_layer_projections

    def _get_gram_vector(self, feature_map_batch):
        """Calcola e vettorizza la matrice di Gram per una feature map"""
        B, C, H, W = feature_map_batch.size()
        features = feature_map_batch.view(B, C, H * W)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram.div_(H * W)  # Normalizzazione
        
        # Vettorizzazione triangolare superiore (più efficiente)
        indices = torch.triu_indices(C, C, offset=0, device=gram.device)
        return gram[:, indices[0], indices[1]]

    def _hook_fn_gram(self, layer_name):
        """Funzione hook per catturare le feature maps"""
        def hook(module, input, output):
            self.extracted_gram_feature_maps[layer_name] = output
        return hook

    def _register_gram_hooks(self, visual_model):
        """Registra gli hook per estrarre le feature maps"""
        if not self.clip_model_name.startswith("RN"):
            print(f"Estrazione Gram non configurata per modello: {self.clip_model_name}")
            return
            
        target_modules = {
            'layer1': visual_model.layer1,
            'layer2': visual_model.layer2,
            'layer3': visual_model.layer3,
            'layer4': visual_model.layer4
        }
        
        for layer_key in self.layers_for_gram_config:
            if layer_key in target_modules:
                module = target_modules[layer_key]
                hook = module.register_forward_hook(self._hook_fn_gram(layer_key))
                self.feature_extractor_hooks.append(hook)
                print(f"Hook registrato per Gram su: {layer_key}")
            else:
                print(f"AVVISO: layer '{layer_key}' non trovato nel modello {self.clip_model_name}")

    def encode_image_with_style_adapter(self, image_input):
        """Codifica un'immagine con l'adapter di stile"""
        self.extracted_gram_feature_maps.clear()

        # 1. Ottieni feature semantiche
        with torch.no_grad():
            semantic_features = self.clip_model.encode_image(image_input).float()

        # 2. Elabora le feature Gram se presenti
        if self.total_gram_projected_dim > 0:
            projected_gram_vectors = self._process_gram_features(image_input.size(0))
            
            # 3. Concatena feature semantiche e Gram
            features_for_fusion = torch.cat([semantic_features, projected_gram_vectors], dim=1)
        else:
            features_for_fusion = semantic_features

        # 4. Applica Fusion Adapter e normalizza
        adapted_features = self.fusion_adapter(features_for_fusion)
        return torch.nn.functional.normalize(adapted_features, p=2, dim=-1)

    def _process_gram_features(self, batch_size):
        """Processa le feature Gram estratte"""
        projected_gram_vectors_list = []
        
        for layer_name in self.layers_for_gram_config:
            dict_key = layer_name.replace('.', '_')
            if layer_name in self.extracted_gram_feature_maps and dict_key in self.gram_layer_projections:
                feature_map = self.extracted_gram_feature_maps[layer_name]
                gram_vector = self._get_gram_vector(feature_map.float())
                projected_gram = self.gram_layer_projections[dict_key](gram_vector)
                projected_gram_vectors_list.append(projected_gram)
            else:
                # Fallback: zeri per layer mancanti
                dim = list(self.gram_layer_projections.values())[0].out_features if self.gram_layer_projections else 0
                projected_gram_vectors_list.append(torch.zeros(batch_size, dim, device=self.device))
                
        if projected_gram_vectors_list:
            return torch.cat(projected_gram_vectors_list, dim=1)
        else:
            return torch.zeros(batch_size, self.total_gram_projected_dim, device=self.device)

    def encode_text(self, text_input):
        """Codifica testo usando il modello CLIP base"""
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input).float()  # Aggiungi .float() qui
        return torch.nn.functional.normalize(text_features, p=2, dim=-1)

    def encode_image(self, image):
        """Wrapper per compatibilità con interfaccia CLIP"""
        return self.encode_image_with_style_adapter(image)

    def forward(self, image_input, text_tokens):
        """Forward pass completo: immagine→adapter, testo→base, calcolo similarità"""
        image_features = self.encode_image_with_style_adapter(image_input)
        text_features = self.encode_text(text_tokens)
        
        # Converti text_features allo stesso tipo di image_features
        text_features = text_features.type_as(image_features)
        
        logit_scale = self.clip_model.logit_scale.exp().float()
        return logit_scale * (image_features @ text_features.t())

    def _remove_gram_hooks(self):
        """Rimuove gli hook feature extractor"""
        for hook in self.feature_extractor_hooks:
            hook.remove()
        self.feature_extractor_hooks = []

    def __del__(self):
        """Cleanup degli hook quando l'istanza viene distrutta"""
        self._remove_gram_hooks()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_name', type=str, default='RN50', help="Nome del modello CLIP (es. RN50, ViT-B/32)")
    parser.add_argument('--fusion_bottleneck_dim', type=int, default=256, help="Dimensione bottleneck del Fusion Adapter")
    parser.add_argument('--gram_style_projection_dim', type=int, default=256, help="Dimensione totale delle Gram features proiettate")
    parser.add_argument('--layers_for_gram_rn50', type=str, nargs='+', default=['layer2', 'layer3'], help="Layer RN50 per Gram")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=5, help="Numero di epoche")
    parser.add_argument('--batch_size', type=int, default=16, help="Dimensione del batch")
    parser.add_argument('--output_model', type=str, default='best_style_adapted_clip_artgraph.pt')
    parser.add_argument('--seed', type=int, default=42, help="Seed per la riproducibilità")
    parser.add_argument('--dropout_rate_adapter', type=float, default=0.1, help="Dropout rate per il fusion adapter")
    parser.add_argument('--use_layernorm_adapter', type=bool, default=True, help="Usa LayerNorm nel fusion adapter")
    parser.add_argument('--warmup_steps', type=int, default=200, help="Numero di warmup steps")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay")
    parser.add_argument('--early_stopping_patience', type=int, default=3, 
                      help="Numero di epoche da attendere prima di terminare se non c'è miglioramento")
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, 
                      help="Miglioramento minimo da considerare significativo per l'early stopping")
    
    args = parser.parse_args()
    print(f"Argomenti: {args}")

    # Setup semi casuali per riproducibilità
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"Utilizzo del device: {device}")

    try:
        dataset_path = find_artgraph_path()
    except FileNotFoundError as e:
        print(f"Errore: {e}")
        sys.exit(1)
    
    print(f"Creazione del modello CLIP ({args.clip_model_name}) con Style Adapter...")
    try:
        model = CLIPWithStyleAdapter(
            clip_model_name=args.clip_model_name,
            fusion_bottleneck_dim=args.fusion_bottleneck_dim,
            gram_style_projection_dim=args.gram_style_projection_dim,
            layers_for_gram_rn50=args.layers_for_gram_rn50,
            dropout_rate=args.dropout_rate_adapter,
            use_layernorm_adapter=args.use_layernorm_adapter,
            device=device
        ).to(device)
    except Exception as e:
        print(f"Errore creazione modello: {e}")
        sys.exit(1)
    
    # Raccogli i parametri addestrabili
    trainable_params = []
    if model.gram_layer_projections:
        trainable_params.extend(list(model.gram_layer_projections.parameters()))
    if hasattr(model, 'fusion_adapter'):
        trainable_params.extend(list(model.fusion_adapter.parameters()))
    
    if not trainable_params:
        print("ERRORE: Nessun parametro addestrabile trovato nel modello.")
        sys.exit(1)
    
    print(f"Parametri addestrabili: {sum(p.numel() for p in trainable_params)}")

    # Optimizer, criterion, scheduler
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    def lr_lambda(current_step): 
        return float(current_step) / float(max(1, args.warmup_steps)) if current_step < args.warmup_steps else 1.0
    scheduler = LambdaLR(optimizer, lr_lambda)

    print("Caricamento del dataset Artgraph...")
    try:
        train_dataset = ArtgraphDataset(dataset_path, split='train', transform=model.preprocess, seed=args.seed)
        val_dataset = ArtgraphDataset(dataset_path, split='val', transform=model.preprocess, seed=args.seed)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Dataset vuoto. Controlla il percorso e la struttura del dataset.")

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )
    except Exception as e:
        print(f"Errore nel caricamento dataset: {e}")
        sys.exit(1)
    
    # Tokenizzazione del testo
    print("Preparazione dei token di classe...")
    text_prompts = [f"a painting by {c.replace('-', ' ').replace('_', ' ')}" for c in train_dataset.classnames]
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    # Addestramento
    print(f"Inizio addestramento per {args.epochs} epoche...")
    best_val_acc = 0.0
    no_improvement_count = 0
    
    for epoch in range(args.epochs):
        # Training
        model.gram_layer_projections.train() if model.gram_layer_projections else None
        model.fusion_adapter.train()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, text_tokens, criterion, 
            optimizer, scheduler, device, epoch, args.epochs
        )
        
        # Validation
        model.gram_layer_projections.eval() if model.gram_layer_projections else None
        model.fusion_adapter.eval()
        
        val_loss, val_acc = validate_model(model, val_loader, text_tokens, criterion, device, epoch, args.epochs) # MODIFICATA QUESTA RIGA
        
        # Logging e salvataggio modello
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoca {epoch+1}: LR={current_lr:.2e}, Train Loss={train_loss:.4f}, '
              f'Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}') # Aggiunto Val Loss al log
        sys.stdout.flush()

        # Salvataggio modello e controllo early stopping
        if val_acc - best_val_acc > args.early_stopping_min_delta:
            best_val_acc = val_acc
            save_model(model, epoch, val_acc, args, train_dataset.classnames)
            print(f'Miglior modello salvato con Val Acc: {best_val_acc:.4f}')
            no_improvement_count = 0  # Reset del contatore
        else:
            no_improvement_count += 1
            print(f'Nessun miglioramento per {no_improvement_count} epoche...')
            
            if no_improvement_count >= args.early_stopping_patience:
                print(f'Early stopping attivato dopo {epoch+1} epoche.')
                break
    
    print(f'Addestramento completato. Miglior Val Acc: {best_val_acc:.4f}')
    model._remove_gram_hooks()


def train_epoch(model, train_loader, text_tokens, criterion, optimizer, scheduler, device, epoch, total_epochs):
    """Funzione di addestramento per una singola epoca"""
    total_loss = 0
    correct = 0
    total_samples = 0
    
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoca {epoch+1}/{total_epochs} [Train]')):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        logits = model(images, text_tokens)
        
        # Controllo NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"AVVISO: NaN/Inf nei logits, batch {i} saltato")
            continue
            
        # Loss e backward
        adjusted_labels = labels - 150 # MODIFICATO: Aggiusta le etichette per la loss
        loss = criterion(logits, adjusted_labels)
        if torch.isnan(loss):
            print(f"AVVISO: NaN nella loss, batch {i} saltato")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Statistiche
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == adjusted_labels).sum().item() # MODIFICATO: Confronta con etichette aggiustate
        total_samples += images.size(0)
    
    # Metriche medie
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc = correct / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc


def validate_model(model, val_loader, text_tokens, criterion, device, epoch, total_epochs): # Aggiunto criterion
    """Funzione di validazione"""
    total_val_loss = 0 # Aggiunto per accumulare la loss
    correct = 0
    total_samples = 0
    
    model.eval() # Assicurati che il modello sia in modalità valutazione
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoca {epoch+1}/{total_epochs} [Val]'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            logits = model(images, text_tokens)
            
            # Controllo NaN/Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"AVVISO: NaN/Inf nei logits di validazione, batch saltato")
                # Potresti voler gestire diversamente, es. assegnando una loss alta o saltando l'aggiornamento delle metriche per questo batch
                continue
                
            # Calcolo loss
            adjusted_labels = labels - 150 
            loss = criterion(logits, adjusted_labels)
            if torch.isnan(loss):
                print(f"AVVISO: NaN nella loss di validazione, batch saltato")
                continue
            
            total_val_loss += loss.item() * images.size(0)
            
            # Calcolo accuratezza
            preds = logits.argmax(dim=1)
            correct += (preds == adjusted_labels).sum().item() 
            total_samples += images.size(0)
    
    avg_val_loss = total_val_loss / total_samples if total_samples > 0 else 0
    accuracy = correct / total_samples if total_samples > 0 else 0
    
    return avg_val_loss, accuracy 


def save_model(model, epoch, val_acc, args, classnames):
    """Salvataggio del modello addestrato"""
    save_dict = {
        'epoch': epoch + 1, 
        'val_acc': val_acc, 
        'args': args,
        'classnames': classnames
    }
    
    if model.gram_layer_projections:
        save_dict['gram_layer_projections_state_dict'] = model.gram_layer_projections.state_dict()
        
    if hasattr(model, 'fusion_adapter'):
        save_dict['fusion_adapter_state_dict'] = model.fusion_adapter.state_dict()
    
    torch.save(save_dict, args.output_model)
    print(f"Modello salvato in {args.output_model}")


if __name__ == '__main__':
    main()