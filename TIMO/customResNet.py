import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification, AutoProcessor
from PIL import Image

# Importazioni dal workspace TIMO
import clip 
from datasets.artgraph import Artgraph
from clip.model import AttentionPool2d # Assumendo che AttentionPool2d sia accessibile qui


# --- Configurazione ---
# !!! MODIFICA QUESTO PERCORSO !!!
# DATA_ROOT deve essere la directory che CONTIENE la cartella 'artgraph'
# Esempio: se le immagini sono in '$DATA/artgraph/images', DATA_ROOT dovrebbe essere '$DATA'
DATA_ROOT = "$DATA/"  # Esempio: "/path/to/your/data_parent_folder" o "./$DATA"

WIKIART_MODEL_NAME = "prithivMLmods/WikiArt-Style"
CLIP_MODEL_NAME = "RN50"  # Usato per l'encoder testuale e la dim. di embedding target
NUM_EPOCHS = 10
BATCH_SIZE = 16 # Riduci se hai problemi di memoria
LEARNING_RATE = 1e-5 # Tasso di apprendimento più basso per il fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Custom Visual Encoder ---
class CustomVisualEncoder(nn.Module):
    def __init__(self, wikiart_backbone, clip_embed_dim, input_resolution=224):
        super().__init__()
        self.backbone = wikiart_backbone
        
        # Questi valori potrebbero necessitare di aggiustamenti basati sul modello WikiArt effettivo.
        # Esempio per un backbone tipo ResNet50 che produce 2048 canali di feature:
        backbone_output_channels = 2048 
        num_attention_heads = 32 # Esempio da CLIP RN50 (per embed_dim 1024)
        
        # AttentionPool2d proietta direttamente alla clip_embed_dim
        self.attnpool = AttentionPool2d(
            spacial_dim=input_resolution // 32, # Dimensione spaziale delle feature map in input
            embed_dim=backbone_output_channels, # Canali delle feature map in input
            num_heads=num_attention_heads,
            output_dim=clip_embed_dim # Dimensione dell'embedding di output (target CLIP)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passa attraverso il backbone WikiArt.
        # Ci aspettiamo che l'output sia una feature map [B, C, H, W].
        features = self.backbone(x) 
        
        # Gestione dell'output del backbone (comune per i modelli HuggingFace)
        if isinstance(features, tuple): # Alcuni modelli restituiscono tuple
            features = features[0]
        # Se l'output è un oggetto con attributi specifici (es. BaseModelOutput)
        if hasattr(features, 'last_hidden_state'): # Tipico per ViT o modelli basati su Transformer
            # Potrebbe essere necessario un reshape se last_hidden_state è [B, SeqLen, Dim]
            # e AttentionPool2d si aspetta [B, C, H, W].
            # Questa parte è critica e dipende dalla natura del backbone (CNN vs ViT).
            # Per ora, assumiamo che se ha last_hidden_state, sia già stato processato
            # in modo compatibile o che il backbone stesso sia un CNN.
            # Se è un ViT, AttentionPool2d potrebbe non essere l'approccio corretto senza adattamenti.
            features = features.last_hidden_state 
            # Se features è [B, N, D] (es. output patch di ViT) e N = H*W, D = C
            # Esempio di reshape (ipotetico, verifica le dimensioni):
            # B, N, D = features.shape
            # H = W = int(N**0.5)
            # if H * W == N:
            #    features = features.permute(0, 2, 1).reshape(B, D, H, W)
            # else:
            #    print("Attenzione: output del backbone non facilmente convertibile in feature map 2D per AttentionPool2d")


        # Se features non è nella forma [B, C, H, W], potrebbero essere necessarie ulteriori manipolazioni.
        # Ad esempio, se il backbone è un ViT che restituisce [B, NumPatches+1, DimEmb],
        # dovresti probabilmente prendere l'output del token CLS e proiettarlo,
        # oppure rimuovere il token CLS, riorganizzare le patch in una griglia e poi usare AttentionPool2d.
        # Per un backbone CNN, l'output dovrebbe essere già [B, C, H, W].

        pooled_features = self.attnpool(features)
        return pooled_features

# --- Helper: Estrae il Backbone da WikiArt ---
def get_wikiart_backbone(model_name_hf: str):
    full_model = AutoModelForImageClassification.from_pretrained(model_name_hf)
    
    # Tentativi comuni per estrarre il backbone
    if hasattr(full_model, 'base_model'): # Molti modelli HF hanno questo
        backbone = full_model.base_model
        print("Estratto backbone come 'base_model'.")
    elif hasattr(full_model, 'resnet'): # Specifico per ResNet
        backbone = full_model.resnet
        # Potrebbe essere necessario rimuovere il pooler e fc del resnet originale
        if hasattr(backbone, 'fc'): backbone.fc = nn.Identity()
        if hasattr(backbone, 'avgpool'): backbone.avgpool = nn.Identity()
        print("Estratto backbone come 'resnet' (fc e avgpool rimossi se presenti).")
    elif hasattr(full_model, 'features'): # Altra convenzione comune
        backbone = full_model.features
        print("Estratto backbone come 'features'.")
    elif hasattr(full_model, 'vit'): # Per modelli basati su Vision Transformer
        # Attenzione: l'output di un ViT è tipicamente [B, NumPatches+1, DimEmb]
        # AttentionPool2d si aspetta [B, C, H, W]. Potrebbe essere necessario un adattatore.
        backbone = full_model.vit
        print("Estratto backbone come 'vit'. Attenzione all'output per AttentionPool2d.")
    else:
        # Fallback: rimuove l'ultimo layer (spesso il classificatore)
        children = list(full_model.children())
        if len(children) > 1 and isinstance(children[-1], (nn.Linear, nn.modules.container.Sequential)):
            # Se l'ultimo è Linear o un blocco sequenziale (es. pooler+classifier)
            backbone = nn.Sequential(*children[:-1])
            print("Estratto backbone rimuovendo l'ultimo modulo figlio.")
        elif len(children) == 1: # Se il modello è un singolo blocco (es. un Sequential che è il backbone)
             backbone = children[0] # Potrebbe essere già il backbone
             print("Estratto backbone assumendo che il modello sia un singolo modulo Sequential.")
        else:
            raise ValueError(f"Impossibile estrarre automaticamente il backbone da {model_name_hf}. Ispeziona la sua struttura.")
    
    # Assicurati che il backbone sia in modalità valutazione se ha dropout/batchnorm
    backbone.eval() 
    return backbone

# --- Contrastive Loss (da CLIP) ---
def contrastive_loss_fn(logits: torch.Tensor, device: str) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=device)
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2.0

# --- Main Training Script ---
def main():
    
    
    artgraph_dir = os.path.join(DATA_ROOT, "artgraph")
    if not os.path.isdir(artgraph_dir):
        print(f"ERRORE: La directory '{artgraph_dir}' non esiste.")
        print(f"Assicurati che DATA_ROOT ('{DATA_ROOT}') sia la directory genitore di 'artgraph'.")
        return

    print(f"Utilizzo del dispositivo: {DEVICE}")
    print("Caricamento modelli...")
    wikiart_processor =AutoProcessor.from_pretrained("prithivMLmods/WikiArt-Style")
    
    clip_model_loaded, _ = clip.load(CLIP_MODEL_NAME, device="cpu") 
    clip_embed_dim = clip_model_loaded.visual.output_dim 
    text_encoder = clip_model_loaded.transformer
    # logit_scale è un parametro addestrabile in CLIP. Lo usiamo preaddestrato e fisso.
    logit_scale = clip_model_loaded.logit_scale.exp().detach().to(DEVICE)

    wikiart_raw_backbone = get_wikiart_backbone(WIKIART_MODEL_NAME)
    
    try:
        # Determina input_resolution dal processore WikiArt
        if isinstance(wikiart_processor.size, dict):
            # Es. {'shortest_edge': 224} o {'height': 224, 'width': 224}
            if 'shortest_edge' in wikiart_processor.size:
                input_resolution = wikiart_processor.size['shortest_edge']
            elif 'height' in wikiart_processor.size: # Prendi height o width se disponibili
                input_resolution = wikiart_processor.size['height']
            else: # Fallback se la struttura di size non è riconosciuta
                input_resolution = 224 
                print(f"Avviso: Chiavi 'shortest_edge' o 'height' non trovate in wikiart_processor.size. Assumo {input_resolution}.")
        elif isinstance(wikiart_processor.size, (int, float)):
            input_resolution = int(wikiart_processor.size)
        else: # Fallback generico
            input_resolution = 224
            print(f"Avviso: Impossibile determinare input_resolution da wikiart_processor.size. Assumo {input_resolution}.")
        print(f"Risoluzione di input determinata/assunta: {input_resolution}")

    except Exception as e:
        input_resolution = 224
        print(f"Avviso: Errore nel determinare input_resolution da processor ({e}). Assumo {input_resolution}.")

    custom_visual_model = CustomVisualEncoder(wikiart_raw_backbone, clip_embed_dim, input_resolution=input_resolution)
    
    custom_visual_model = custom_visual_model.to(DEVICE)
    text_encoder = text_encoder.to(DEVICE)

    print("Impostazione ottimizzatore...")
    for param in custom_visual_model.backbone.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    # Solo i parametri di attnpool saranno addestrati
    trainable_parameters = list(custom_visual_model.attnpool.parameters())
    
    if not trainable_parameters:
        print("ERRORE: Nessun parametro addestrabile trovato in custom_visual_model.attnpool.")
        return
    print(f"Numero di parametri addestrabili in attnpool: {sum(p.numel() for p in trainable_parameters if p.requires_grad)}")

    optimizer = optim.AdamW(trainable_parameters, lr=LEARNING_RATE, weight_decay=0.01)
    
    print("Caricamento dataset ArtGraph...")
    
    def artgraph_image_preprocess_pil(pil_image): # Prende un'immagine PIL
        # Il processore WikiArt si aspetta immagini PIL o un batch
        return wikiart_processor(images=pil_image, return_tensors="pt").pixel_values.squeeze(0)

    # Artgraph class in `datasets/artgraph.py` usa num_shots.
    # Per usare tutti i dati di training, potremmo aver bisogno di un num_shots elevato
    # o di modificare come Artgraph carica i dati.
    # `max_support_perclass: 16` in `configs/artgraph.yaml`
    artgraph_full_dataset = Artgraph(root=DATA_ROOT, num_shots=16) 
    
    if not artgraph_full_dataset.train:
        print("ERRORE: Il set di training di ArtGraph è vuoto. Controlla i percorsi e il file di split.")
        return

    class ArtgraphPytorchTrainDataset(torch.utils.data.Dataset):
        def __init__(self, datum_list, image_transform_fn, text_tokenizer_fn, 
                     text_template="Un dipinto di {}.", input_res=224):
            self.datum_list = datum_list
            self.image_transform = image_transform_fn
            self.text_tokenizer = text_tokenizer_fn
            self.text_template = text_template
            self.input_res = input_res


        def __len__(self):
            return len(self.datum_list)

        def __getitem__(self, idx):
            datum = self.datum_list[idx]
            try:
                image = Image.open(datum.impath).convert("RGB")
                processed_image = self.image_transform(image)
            except Exception as e:
                print(f"Errore nel caricare l'immagine {datum.impath}: {e}. Restituisco un tensore dummy.")
                processed_image = torch.zeros(3, self.input_res, self.input_res) # Tensore dummy

            # Usa il classname per generare il prompt testuale
            text_description = self.text_template.format(datum.classname)
            # clip.tokenize si aspetta una stringa o una lista di stringhe
            tokenized_text = self.text_tokenizer(text_description).squeeze(0) # Rimuove la dimensione batch
            
            return processed_image, tokenized_text

    train_dataset = ArtgraphPytorchTrainDataset(
        artgraph_full_dataset.train, 
        artgraph_image_preprocess_pil, 
        clip.tokenize,
        input_res=input_resolution
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    print(f"Inizio addestramento per {NUM_EPOCHS} epoche...")
    custom_visual_model.attnpool.train() # Solo attnpool in modalità addestramento
    custom_visual_model.backbone.eval()
    text_encoder.eval()

    for epoch in range(NUM_EPOCHS):
        total_epoch_loss = 0
        num_batches = len(train_loader)
        for batch_idx, (images, texts) in enumerate(train_loader):
            images = images.to(DEVICE)
            texts = texts.to(DEVICE)

            optimizer.zero_grad()

            image_features = custom_visual_model(images) # Output da CustomVisualEncoder
            text_features = text_encoder(texts)     # Output dall'encoder testuale di CLIP

            # Normalizzazione L2 delle feature
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            logits_per_image = logit_scale * image_features @ text_features.T
            
            loss = contrastive_loss_fn(logits_per_image, DEVICE)
            
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches: # Log ogni 20 batch o all'ultimo batch
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = total_epoch_loss / num_batches
        print(f"Epoch {epoch+1} completata. Loss media: {avg_epoch_loss:.4f}")

    print("Addestramento completato.")
    
    output_path = "trained_wikiart_attnpool_on_artgraph.pth"
    torch.save(custom_visual_model.attnpool.state_dict(), output_path)
    print(f"Pesi del layer AttnPool addestrato salvati in: {output_path}")

if __name__ == "__main__":
    main()