import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys # Per flushare l'output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR


import clip
# from utils import cls_acc # Se hai una funzione cls_acc, assicurati che sia disponibile

# Funzione per trovare automaticamente il percorso del dataset artgraph
def find_artgraph_path():
    # Verifica il percorso esatto dalla struttura fornita
    # MODIFICA QUESTO SE NECESSARIO
    # base_path = "/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO"
    # Prova a trovare la directory TIMO relativa allo script corrente
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = "/home/fonty/Scrivania/UniRepo/ComputerVision/TIMO"
    data_path = os.path.join(base_path, "$DATA", "artgraph")
    # Controlla se esiste
    if os.path.exists(data_path) and os.path.isdir(data_path):
        print(f"Dataset artgraph trovato in: {data_path}")
        return data_path

    # Se non trova il percorso esatto, cerca nelle directory vicine
    alternative_paths = [
        os.path.join(base_path, "DATA", "artgraph"),
        os.path.join(base_path, "..", "DATA", "artgraph"),
        os.path.join(base_path, "data", "artgraph"),
        os.path.join(base_path, "..", "data", "artgraph")
    ]
    
    # Se non trova il percorso esatto, cerca nelle directory vicine
    alternative_paths = [
        os.path.join(base_path, "DATA", "artgraph"),
        os.path.join(base_path, "data", "artgraph"),
        os.path.join(current_script_dir, "DATA", "artgraph"),
        os.path.join(current_script_dir, "data", "artgraph"),
        os.path.join(current_script_dir, "..","DATA", "artgraph"),
        os.path.join(current_script_dir, "..","data", "artgraph"),
    ]
    
    for path in alternative_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            print(f"Dataset artgraph trovato in: {abs_path}")
            return abs_path
    
    raise FileNotFoundError(
        "Non è stato possibile trovare il dataset artgraph. "
        "Verifica che sia presente nel percorso $DATA/artgraph o in una sottocartella 'data' o 'DATA' "
        "relativa alla directory TIMO o alla directory dello script."
    )

# Definizione della classe per il dataset Artgraph con divisione manuale
class ArtgraphDataset(data.Dataset):
    def __init__(self, root_dir, split='train', transform=None, train_ratio=0.7, val_ratio=0.15, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.split = split
        
        random.seed(seed) # Set seed per la riproducibilità dello split
        
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
            random.shuffle(img_paths) # Mescola immagini per ogni classe
            num_samples_class = len(img_paths)
            
            train_end = int(train_ratio * num_samples_class)
            val_end = train_end + int(val_ratio * num_samples_class)
            
            if self.split == 'train':
                split_paths = img_paths[:train_end]
            elif self.split == 'val':
                split_paths = img_paths[train_end:val_end]
            else:  # test
                split_paths = img_paths[val_end:]
            
            for img_path in split_paths:
                self.samples.append((img_path, label))
        
        random.shuffle(self.samples) # Mescola tutti i campioni dello split
        
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
            print(f"Errore nel caricamento dell'immagine {img_path}: {e}. Salto questa immagine e ne provo un'altra.")
            # Prova a caricare un'immagine casuale diversa per evitare loop infiniti se molte sono corrotte
            random_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_idx)


# Definizione della classe Adapter migliorata
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, dropout_rate=0.1, use_layernorm=True):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        
        # Inizializzazione
        nn.init.xavier_uniform_(self.down_project.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.down_project.bias)
        
        nn.init.zeros_(self.up_project.weight) # Inizializza up_project a zero per stabilità
        nn.init.zeros_(self.up_project.bias)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(bottleneck_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # L'input x dovrebbe essere già in float32 se il modello non è in .half()
        residual = x 
        
        x = self.down_project(x)
        
        if self.use_layernorm:
            x = self.layer_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.up_project(x)
        
        # Connessione residuale
        # Alpha basso per dare più peso alle feature originali all'inizio
        alpha = 0.1 # Può essere un iperparametro
        combined = alpha * x + residual # Mantenuto (1-alpha) implicito nella residuale originale
                                       # Se volessi (1-alpha) * residual, sarebbe:
                                       # combined = alpha * x + (1 - alpha) * residual
        
        return combined

# Modifica alla classe CLIPWithAdapter
class CLIPWithAdapter(nn.Module):
    def __init__(self, clip_model, bottleneck_dim=256, dropout_rate=0.1, use_layernorm=True, device="cuda"):
        super(CLIPWithAdapter, self).__init__()
        
        # Usa il modello CLIP già caricato invece di caricarne uno nuovo
        self.clip_model = clip_model
        self.clip_model.eval()  # Metti il modello CLIP in modalità valutazione

        # Congela i parametri di CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Determina la dimensione delle feature
        if hasattr(self.clip_model.visual, 'output_dim'):
            self.feature_dim = self.clip_model.visual.output_dim
        elif hasattr(self.clip_model, 'visual_projection'):
            # Per alcuni modelli ViT
            self.feature_dim = self.clip_model.visual_projection.shape[0]
        else:
            # Default per RN50 e altri simili
            self.feature_dim = 1024

        print(f"Dimensione feature del modello CLIP: {self.feature_dim}")
        
        self.image_adapter = Adapter(
            self.feature_dim, 
            bottleneck_dim,
            dropout_rate=dropout_rate,
            use_layernorm=use_layernorm
        ).to(device)
        
        self.device = device
        
    def encode_image_with_adapter(self, image_input):
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
            
            # Applica l'adapter alle feature (assicura float32 per i calcoli dell'adapter)
            adapted_features = self.image_adapter(image_features.float()) 
            
            adapted_features = torch.nn.functional.normalize(adapted_features, p=2, dim=-1)
            return adapted_features.float() # Assicura output in float32

    def encode_text(self, text_input): # text_input sono i token
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
        
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        return text_features.float() # Assicura output in float32

    def forward(self, image_input, text_tokens):
        image_features_adapted = self.encode_image_with_adapter(image_input) # Ora dovrebbe essere float32
        text_features_encoded = self.encode_text(text_tokens)              # Ora dovrebbe essere float32
        
        # Debug: stampa i dtypes
        # print(f"Dtype image_features_adapted: {image_features_adapted.dtype}")
        # print(f"Dtype text_features_encoded: {text_features_encoded.dtype}")

        # logit_scale dovrebbe anche essere float32 se le feature lo sono.
        # Se clip_model è in half, logit_scale potrebbe essere half. Convertiamolo.
        logit_scale = self.clip_model.logit_scale.exp().float() 
        
        # Ora entrambi i tensori principali per matmul dovrebbero essere float32
        logits = logit_scale * (image_features_adapted @ text_features_encoded.t())
        
        return logits
    
    def encode_image(self, image_input):
        # Reindirizza a encode_image_with_adapter per compatibilità
        return self.encode_image_with_adapter(image_input)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_name', type=str, default='RN50', help="Nome del modello CLIP (es. RN50, ViT-B/32)")
    parser.add_argument('--bottleneck_dim', type=int, default=64, help="Dimensione del bottleneck dell'adapter") # Ridotto
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate per l'adapter") # Aumentato leggermente, con warmup
    parser.add_argument('--epochs', type=int, default=20, help="Numero di epoche") # Aumentato
    parser.add_argument('--batch_size', type=int, default=16, help="Dimensione del batch") # Aumentato
    parser.add_argument('--output_model', type=str, default='best_adapted_clip_artgraph.pt')
    parser.add_argument('--seed', type=int, default=42, help="Seed per la riproducibilità")
    parser.add_argument('--dropout_rate', type=float, default=0.2, help="Dropout rate per l'adapter") # Aumentato
    parser.add_argument('--use_layernorm', type=bool, default=True, help="Usa LayerNorm nell'adapter")
    parser.add_argument('--warmup_steps', type=int, default=500, help="Numero di warmup steps per il LR scheduler")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay per l'optimizer")

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
    
    # Crea il modello con adapter (carica CLIP al suo interno)
    print(f"Creazione del modello CLIP ({args.clip_model_name}) con adapter (bottleneck_dim={args.bottleneck_dim})...")
    model = CLIPWithAdapter(
        clip_model_name=args.clip_model_name,
        bottleneck_dim=args.bottleneck_dim,
        dropout_rate=args.dropout_rate,
        use_layernorm=args.use_layernorm,
        device=device
    ).to(device)
    
    # Solo i parametri dell'adapter devono essere addestrati
    optimizer = optim.AdamW(model.image_adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Scheduler del Learning Rate con Warmup
    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        # Potresti aggiungere un decadimento dopo il warmup, es. decadimento lineare o coseno
        # Per ora, manteniamolo costante dopo il warmup
        return 1.0 
    
    scheduler = LambdaLR(optimizer, lr_lambda)

    print("Caricamento del dataset Artgraph con divisione manuale...")
    # Usa il preprocess del modello caricato
    train_dataset = ArtgraphDataset(dataset_path, split='train', transform=model.preprocess, seed=args.seed, train_ratio=0.7, val_ratio=0.15)
    val_dataset = ArtgraphDataset(dataset_path, split='val', transform=model.preprocess, seed=args.seed, train_ratio=0.7, val_ratio=0.15)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Errore: Uno dei dataset è vuoto. Controlla la logica di divisione e i percorsi.")
        return

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Preparazione dei token di classe...")
    # Crea i prompt una volta sola
    text_prompts = [f"a painting by {c.replace('-', ' ').replace('_', ' ')}" for c in train_dataset.classnames]
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    print(f"Inizio addestramento per {args.epochs} epoche...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.image_adapter.train() # Metti solo l'adapter in modalità training
        total_loss = 0
        correct_train = 0
        total_train_samples = 0
        
        # pbar_train = tqdm(train_loader, desc=f'Epoca {epoch+1}/{args.epochs} [Train]')
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoca {epoch+1}/{args.epochs} [Train]')):
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images, text_tokens) # La forward di CLIPWithAdapter ora gestisce tutto
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"!!! ATTENZIONE: NaN o Inf nei logits all'iterazione {i} dell'epoca {epoch+1} !!!")
                print(f"Logits: {logits}")
                # Potrebbe essere utile salvare lo stato o fare un break
                # sys.exit("Training interrotto a causa di logits instabili.")
                continue # Salta questo batch

            loss = criterion(logits, labels)
            
            if torch.isnan(loss):
                print(f"!!! ATTENZIONE: Loss è NaN all'iterazione {i} dell'epoca {epoch+1} !!!")
                print(f"Logits che hanno causato NaN loss: {logits}")
                print(f"Labels: {labels}")
                # sys.exit("Training interrotto a causa di loss NaN.")
                continue # Salta questo batch

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(), max_norm=1.0) # Clip solo i gradienti dell'adapter
            optimizer.step()
            scheduler.step() # Aggiorna il LR ad ogni step
            
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train_samples += images.size(0)
        
        avg_train_loss = total_loss / total_train_samples if total_train_samples > 0 else 0
        avg_train_acc = correct_train / total_train_samples if total_train_samples > 0 else 0
        
        # Validazione
        model.image_adapter.eval() # Metti solo l'adapter in modalità valutazione
        correct_val = 0
        total_val_samples = 0
        # pbar_val = tqdm(val_loader, desc=f'Epoca {epoch+1}/{args.epochs} [Val]')
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoca {epoch+1}/{args.epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                logits = model(images, text_tokens)
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"!!! ATTENZIONE: NaN o Inf nei logits durante la validazione !!!")
                    continue

                preds = logits.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val_samples += images.size(0)
        
        avg_val_acc = correct_val / total_val_samples if total_val_samples > 0 else 0
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoca {epoch+1}: LR={current_lr:.2e}, Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}, Val Acc={avg_val_acc:.4f}')
        sys.stdout.flush() # Forza la stampa immediata


        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.image_adapter.state_dict(), # Salva solo lo stato dell'adapter
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': avg_val_acc,
                'classnames': train_dataset.classnames,
                'args': args
            }, args.output_model)
            print(f'Miglior modello (solo adapter) salvato in {args.output_model} con Val Acc: {best_val_acc:.4f}')
            sys.stdout.flush()
    
    print(f'Addestramento completato. Miglior Val Acc dell\'adapter: {best_val_acc:.4f}')

if __name__ == '__main__':
    main()