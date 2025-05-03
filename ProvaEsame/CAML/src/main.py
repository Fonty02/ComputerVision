import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Dataset, Subset
import random
import numpy as np
from tqdm import tqdm
import math
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import copy 
# --- Funzione per generare ELMES ---
# NOTA: Questa è un'implementazione pratica per generare vettori fissi
# equiangolari-ish. Il paper fa riferimento a costrutti teorici più formali.
# Basata su #file:src/models/TransformerEncoder.py (line 139) del codice originale del paper
def get_elmes(p: int, C: int, device: torch.device) -> torch.Tensor:
    """Genera vettori ELMES deterministici per C classi in dimensione p."""
    if C <= 1:
        print(f"Warning: get_elmes called with C={C} <= 1. Returning zeros.")
        return torch.zeros((C, p), device=device)

    # Usa una seed fissa per la riproducibilità, come nel codice originale del paper
    # Questo assicura che i vettori ELMES siano fissi durante l'addestramento.
    rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    torch.manual_seed(50)
    np.random.seed(50)

    # Calcola la matrice M* per Simplex ETF
    ones = torch.ones((C, 1), dtype=torch.float32, device=device)
    M_star = torch.sqrt(torch.tensor(C / (C - 1), device=device)) * (
            torch.eye(C, device=device) - (1 / C) * torch.matmul(ones, ones.T))

    # Genera una matrice casuale e rendila ortonormale (base casuale)
    U = np.random.rand(p, C)
    U_ortho, _ = np.linalg.qr(U) # Decomposizione QR per ottenere una base ortonormale
    # Se p < C, QR restituirà solo p vettori ortonormali. Dobbiamo gestirlo.
    if p < C:
         # Aggiungi vettori casuali e ri-ortonormalizza per ottenere C vettori in p-dim
         # Questo non è l'ideale, ma una possibile patch. Idealmente p >= C.
         print(f"Warning: ELMES embedding dim p={p} < n_way C={C}. Trying to extend basis.")
         extra_cols = C - p
         random_extra = np.random.rand(p, extra_cols)
         U_combined = np.hstack((U_ortho, random_extra))
         U_ortho, _ = np.linalg.qr(U_combined) # Ri-ortonormalizza

    # Usa solo le prime C colonne della base ortonormale
    U_tensor = torch.tensor(U_ortho[:, :C], dtype=torch.float32, device=device)

    # Calcola i vettori ELMES finali
    elmes_vectors = (U_tensor @ M_star).T # Trasponi per ottenere (C, p)

    # Ripristina lo stato del generatore di numeri casuali
    torch.set_rng_state(rng_state)
    np.random.set_state(np_rng_state)

    return elmes_vectors


# --- 1. Definizione dei Componenti ---

class FrozenImageEncoder(nn.Module):
    """Encoder di immagini pre-addestrato con parametri congelati."""
    def __init__(self):
        super().__init__()
        # NOTA DAL PAPER: Gli esperimenti principali usano CLIP (ViT-base).
        # Qui usiamo ResNet18 per semplicità e minor dipendenze.
        # Le prestazioni saranno diverse da quelle riportate nel paper.
        print("Using ResNet18 as frozen image encoder (Paper uses CLIP ViT-base).")
        weights = ResNet18_Weights.DEFAULT
        self.base_model = resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(self.base_model.children())[:-1])
        self.transform = weights.transforms()
        # Congela i parametri
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval() # Metti in modalità valutazione permanente

        # Determina la dimensione dell'output
        # Esegui un forward pass fittizio
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224) # Dimensione input standard ResNet
            dummy_output = self.feature_extractor(dummy_input)
            self.output_dim = dummy_output.view(dummy_output.size(0), -1).shape[1]
            print(f"ResNet18 feature dimension: {self.output_dim}")


    def forward(self, image_list: List) -> torch.Tensor:
        """Applica trasformazioni ed encoder a una lista di immagini PIL."""
        self.feature_extractor.eval() # Assicurati che sia in modalità eval
        with torch.no_grad():
            # Applica la trasformazione a ciascuna immagine
            # Gestisci possibile errore se image_list è vuota
            if not image_list:
                # Restituisci un tensore vuoto della forma corretta (0, output_dim)
                # Scegli un device (es. CPU o quello del modello se accessibile)
                device = next(self.feature_extractor.parameters()).device
                return torch.empty((0, self.output_dim), device=device)

            transformed_images = [self.transform(img) for img in image_list]
            x_transformed = torch.stack(transformed_images)
            device = next(self.feature_extractor.parameters()).device
            x_transformed = x_transformed.to(device)
            features = self.feature_extractor(x_transformed)
        return features.view(features.size(0), -1) # Flatten

class ELMESEncoder(nn.Module):
    """
    Encoder ELMES con parametri congelati per classi note e embedding
    'Unknown' addestrabile per la query.
    """
    def __init__(self, n_way: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.n_way = n_way # Numero di classi nel task corrente

        # Genera gli embedding ELMES *fissi* per le N classi del task
        # Forma: (n_way, embedding_dim)
        elmes_vectors = get_elmes(embedding_dim, n_way, device)
        # Registra come parametro non addestrabile
        self.register_parameter('label_elmes', nn.Parameter(elmes_vectors, requires_grad=False))

        # Crea l'embedding *addestrabile* per la classe 'Unknown' (query)
        # Inizializzato casualmente (piccoli valori)
        # Forma: (1, embedding_dim)
        unk_tensor = torch.randn(1, embedding_dim, device=device) * 0.01
        self.unk_emb = nn.Parameter(unk_tensor, requires_grad=True)
        print(f"ELMES Encoder: {n_way} fixed class embeddings, 1 trainable unknown embedding.")

    def forward(self, class_indices: torch.Tensor, is_query: bool) -> torch.Tensor:
        """
        Restituisce l'embedding ELMES appropriato.
        Args:
            class_indices (Tensor): Indici delle classi (per support, 0 a n_way-1)
                                     o placeholder (per query).
            is_query (bool): True se l'input è per la query.
        Returns:
            Tensor: Embedding ELMES.
        """
        if is_query:
            # Restituisce l'embedding 'Unknown' addestrabile
            # Si assume batch size 1 per la query in questo contesto
            return self.unk_emb # Shape (1, embedding_dim)
        else:
            # Usa gli indici (0..n_way-1) per recuperare gli embedding ELMES fissi
            # class_indices ha shape (N*K)
            # F.embedding cerca nella matrice `self.label_elmes` (n_way, embedding_dim)
            return F.embedding(class_indices, self.label_elmes)

class TrainableSequenceModel(nn.Module):
    """Modello di sequenza non causale addestrabile (Transformer Encoder)."""
    def __init__(self, input_dim: int, num_heads: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        # Assicurati che input_dim sia divisibile per num_heads
        if input_dim % num_heads != 0:
             # Trova il num_heads più vicino che divide input_dim o aggiusta input_dim
             # Semplice fallback: usa 1 head se non divisibile
             print(f"Warning: input_dim ({input_dim}) not divisible by num_heads ({num_heads}). Falling back to num_heads=1.")
             num_heads = 1 # O potresti arrotondare input_dim o num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True, # Assicura formato (batch, seq, feature)
            activation=F.gelu # Usa GELU come attivazione comune nei Transformer
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        return self.transformer_encoder(x)

class TrainableMLP(nn.Module):
    """MLP addestrabile per la classificazione finale."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(), # Usa GELU
            nn.Dropout(0.1), # Dropout leggermente ridotto
            nn.Linear(input_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prende l'output del sequence model corrispondente alla query.
        # Assumiamo che la query sia il primo elemento della sequenza.
        # x ha shape (batch_size, sequence_length, input_dim)
        query_output = x[:, 0, :] # Estrai il primo token (query)
        return self.mlp(query_output)

# --- 2. Modello Completo CAML ---

class CAMLNet(nn.Module):
    def __init__(self, n_way: int, device: torch.device, elmes_embedding_dim: int = 128, seq_model_heads: int = 4, seq_model_layers: int = 2):
        super().__init__()
        self.device = device
        self.n_way = n_way

        self.image_encoder = FrozenImageEncoder()
        image_feature_dim = self.image_encoder.output_dim

        self.elmes_encoder = ELMESEncoder(n_way=n_way,
                                          embedding_dim=elmes_embedding_dim,
                                          device=self.device)

        combined_embedding_dim = image_feature_dim + elmes_embedding_dim
        print(f"Combined embedding dimension: {combined_embedding_dim}")

        self.sequence_model = TrainableSequenceModel(input_dim=combined_embedding_dim,
                                                     num_heads=seq_model_heads,
                                                     num_layers=seq_model_layers)
        self.classifier_mlp = TrainableMLP(input_dim=combined_embedding_dim,
                                           num_classes=n_way)

    def forward(self, query_image: List, support_images: List, support_labels: torch.Tensor) -> torch.Tensor:
        # (Forward pass invariato rispetto alla versione corretta precedente)
        # ... (Incolla qui il corpo della funzione forward) ...
        # 1. Estrarre features dalle immagini (Congelato)
        self.image_encoder.to(self.device)
        query_feat = self.image_encoder(query_image)
        support_feats = self.image_encoder(support_images)

        # 2. Ottenere embedding delle classi
        query_placeholder = torch.empty(1, dtype=torch.long, device=self.device) # Dummy
        query_class_embedding = self.elmes_encoder(query_placeholder, is_query=True)
        support_class_embeddings = self.elmes_encoder(support_labels.long(), is_query=False)

        # 3. Concatenare features immagine e classe
        query_combined = torch.cat([query_feat, query_class_embedding], dim=1)
        support_combined = torch.cat([support_feats, support_class_embeddings], dim=1)

        # 4. Formare la sequenza per il Transformer
        sequence_items = torch.cat([query_combined, support_combined], dim=0)
        sequence = sequence_items.unsqueeze(0)

        # 5. Passare attraverso il Sequence Model
        sequence_output = self.sequence_model(sequence)

        # 6. Classificare con MLP
        logits = self.classifier_mlp(sequence_output)

        return logits

# --- Data Loading Few-Shot Generico ---
class FewShotDataset(Dataset):
    """Crea episodi few-shot da un dataset generico torchvision."""
    def __init__(self, dataset, n_way, k_shot, q_query, num_episodes, dataset_name="Dataset"):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query_per_class = q_query
        self.num_episodes_target = num_episodes
        self.dataset_name = dataset_name

        # Trova le etichette uniche e organizza i dati per classe
        self.data_by_class: Dict[int, List[int]] = {}
        print(f"[{self.dataset_name}] Indexing dataset by class...")
        try:
            # Prova ad accedere alle etichette (comune per CIFAR, MNIST, etc.)
            labels = getattr(self.dataset, 'targets', getattr(self.dataset, 'labels', None))
            if labels is None:
                 # Se le etichette non sono attributi diretti, itera
                 print(f"[{self.dataset_name}] Labels attribute not found, iterating...")
                 labels = [label for _, label in tqdm(self.dataset)]

            unique_labels = sorted(list(set(labels)))
            self.num_total_classes = len(unique_labels)
            print(f"[{self.dataset_name}] Found {self.num_total_classes} unique classes.")

            for i, label in enumerate(tqdm(labels)):
                if label not in self.data_by_class:
                    self.data_by_class[label] = []
                self.data_by_class[label].append(i) # Salva l'indice nel dataset originale

        except Exception as e:
            print(f"Error indexing dataset {self.dataset_name}: {e}")
            raise

        if self.num_total_classes < self.n_way:
             raise ValueError(f"[{self.dataset_name}] Requested n_way={self.n_way}, but only {self.num_total_classes} classes available.")

        self.episode_list = self._generate_episodes()
        print(f"[{self.dataset_name}] Generated {len(self.episode_list)} episodes.")

    def _generate_episodes(self):
        episodes = []
        available_classes = list(self.data_by_class.keys())

        print(f"[{self.dataset_name}] Generating {self.num_episodes_target} episodes...")
        attempts = 0
        max_attempts = self.num_episodes_target * 5 # Limita i tentativi

        required_samples_per_class = self.k_shot + self.q_query_per_class

        while len(episodes) < self.num_episodes_target and attempts < max_attempts:
            attempts += 1
            episode_classes = random.sample(available_classes, self.n_way)

            support_indices = []
            query_indices = []
            # Mappa le etichette originali (potrebbero essere sparse) a 0..N-1
            episode_labels_map = {original_label: new_label for new_label, original_label in enumerate(episode_classes)}
            possible = True

            for original_label in episode_classes:
                class_indices = self.data_by_class[original_label]
                if len(class_indices) < required_samples_per_class:
                    possible = False
                    break

                chosen_indices = random.sample(class_indices, required_samples_per_class)
                support_indices.extend(chosen_indices[:self.k_shot])
                query_indices.extend(chosen_indices[self.k_shot:])

            if possible:
                 episodes.append({
                     "support_indices": support_indices,
                     "query_indices": query_indices,
                     "episode_labels_map": episode_labels_map
                 })

        if len(episodes) < self.num_episodes_target:
            print(f"Warning: [{self.dataset_name}] Could only generate {len(episodes)}/{self.num_episodes_target} episodes.")
        if not episodes:
            print(f"Error: [{self.dataset_name}] No episodes generated.")

        return episodes


    def __len__(self):
        queries_per_episode = self.n_way * self.q_query_per_class
        if queries_per_episode == 0: return 0
        return len(self.episode_list) * queries_per_episode

    def __getitem__(self, index: int) -> Optional[Tuple[object, torch.Tensor, List, torch.Tensor]]:
        queries_per_episode = self.n_way * self.q_query_per_class
        if queries_per_episode == 0 or not self.episode_list:
            return None

        episode_idx = index // queries_per_episode
        query_in_episode_idx = index % queries_per_episode

        if episode_idx >= len(self.episode_list):
             # print(f"Error: [{self.dataset_name}] Index {index} out of bounds.")
             return None # DataLoader gestirà questo

        episode = self.episode_list[episode_idx]
        support_indices = episode["support_indices"]
        query_indices = episode["query_indices"]
        labels_map = episode["episode_labels_map"]

        # Estrai l'indice della query specifica
        if query_in_episode_idx >= len(query_indices):
            # Questo non dovrebbe succedere se len è calcolato correttamente
            print(f"Error: [{self.dataset_name}] Query index {query_in_episode_idx} out of bounds for episode {episode_idx}.")
            return None
        query_idx = query_indices[query_in_episode_idx]

        # Ottieni dati della query
        try:
            query_img, query_original_label = self.dataset[query_idx]
        except IndexError:
             print(f"Error: [{self.dataset_name}] Index {query_idx} out of bounds for underlying dataset.")
             return None
        query_label = labels_map[query_original_label]

        # Ottieni dati del support set
        support_images = []
        support_labels = []
        for idx in support_indices:
             try:
                 img, original_label = self.dataset[idx]
                 support_images.append(img)
                 support_labels.append(labels_map[original_label])
             except IndexError:
                  print(f"Error: [{self.dataset_name}] Index {idx} out of bounds for underlying dataset (support).")
                  return None # Episodio invalido

        return query_img, torch.tensor(query_label), support_images, torch.tensor(support_labels)

# Funzione collate (Invariata)
def identity_collate(batch: List) -> Optional[Tuple]:
    # ... (Incolla qui identity_collate dalla risposta precedente) ...
    if not batch or batch[0] is None:
        return None
    return batch[0]


# --- Funzioni di Training e Test (Leggermente modificate per chiarezza) ---

def run_epoch(model: CAMLNet, dataloader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: Optional[optim.Optimizer] = None, phase: str = "Test") -> Tuple[float, float]:
    """Esegue un'epoca di addestramento o test."""
    is_train = optimizer is not None
    if is_train:
        model.train()
        model.image_encoder.eval() # L'encoder di immagini resta sempre in eval mode
        print(f"\n--- Starting Training Epoch ---")
    else:
        model.eval() # Modalità valutazione per tutti i componenti (Transformer, MLP, etc.)
        print(f"\n--- Starting Evaluation Phase ({phase}) ---")

    total_loss = 0
    correct = 0
    total = 0
    processed_batches = 0

    pbar = tqdm(dataloader, desc=f"{phase}")
    for batch in pbar:
        if batch is None:
            continue

        query_img, query_label, support_imgs, support_labels = batch

        query_label = query_label.to(device)
        support_labels = support_labels.to(device)

        # Esegui forward pass
        # Se non stiamo addestrando, usa torch.no_grad()
        with torch.set_grad_enabled(is_train):
            logits = model([query_img], support_imgs, support_labels)
            loss = criterion(logits, query_label.unsqueeze(0))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss ({loss.item()}) in {phase}. Skipping batch.")
            continue

        # Se stiamo addestrando, esegui backward pass e ottimizza
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            # Opzionale: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()

        # Statistiche
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += 1
        correct += (predicted == query_label).sum().item()
        processed_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / total:.2f}%" if total > 0 else "0.00%")

    if processed_batches == 0:
        print(f"Warning: No batches processed in {phase}.")
        return 0.0, 0.0

    avg_loss = total_loss / processed_batches
    accuracy = 100 * correct / total
    print(f"{phase} Summary: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# --- Script Principale Modificato ---

if __name__ == "__main__":
    # --- Parametri ---
    N_WAY = 5
    K_SHOT = 1
    Q_QUERY_PER_CLASS = 5 # Aumenta le query per episodio per un test più stabile
    # Dataset per simulare pre-addestramento e test unseen
    PRETRAIN_DATASET_NAME = "CIFAR-100"
    TEST_UNSEEN_DATASET_NAME = "CIFAR-10"
    # Numero di episodi
    # NOTA: Il pre-training reale richiederebbe molti più episodi e dataset
    NUM_EPISODES_PRETRAIN = 10000 # Aumentato significativamente
    NUM_EPISODES_TEST_FINAL = 5000  # Numero robusto per la valutazione finale
    # Parametri di addestramento
    LEARNING_RATE = 5e-5
    NUM_EPOCHS_PRETRAIN = 5 # Potrebbe servire di più
    WEIGHT_DECAY = 1e-4
    # Architettura
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ELMES_EMBEDDING_DIM = 64
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 2
    NUM_WORKERS = 2

    print("--- CAML Simulation: Pre-training & Unseen Task Evaluation ---")
    print(f"Device: {DEVICE}")
    print(f"Task: {N_WAY}-way {K_SHOT}-shot, Queries per class: {Q_QUERY_PER_CLASS}")
    print(f"Pre-training on: {PRETRAIN_DATASET_NAME} ({NUM_EPISODES_PRETRAIN} episodes x {NUM_EPOCHS_PRETRAIN} epochs)")
    print(f"Final evaluation on: {TEST_UNSEEN_DATASET_NAME} ({NUM_EPISODES_TEST_FINAL} episodes)")

    # --- Carica i Dataset ---
    print("\nLoading datasets...")
    # Pre-training dataset (CIFAR-100)
    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
    # Test dataset (CIFAR-10) - Usiamo il set di test di CIFAR-10
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

    # --- Crea Dataset Few-Shot ---
    pretrain_fewshot_dataset = FewShotDataset(cifar100_train, N_WAY, K_SHOT, Q_QUERY_PER_CLASS, NUM_EPISODES_PRETRAIN, dataset_name=PRETRAIN_DATASET_NAME)
    test_unseen_fewshot_dataset = FewShotDataset(cifar10_test, N_WAY, K_SHOT, Q_QUERY_PER_CLASS, NUM_EPISODES_TEST_FINAL, dataset_name=TEST_UNSEEN_DATASET_NAME)

    if len(pretrain_fewshot_dataset) == 0 or len(test_unseen_fewshot_dataset) == 0:
        print("Error generating episodes. Exiting.")
        exit()

    # --- Crea DataLoaders ---
    pretrain_loader = DataLoader(pretrain_fewshot_dataset, batch_size=1, shuffle=True, collate_fn=identity_collate, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda', persistent_workers=NUM_WORKERS > 0)
    test_unseen_loader = DataLoader(test_unseen_fewshot_dataset, batch_size=1, shuffle=False, collate_fn=identity_collate, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda', persistent_workers=NUM_WORKERS > 0)

    # --- Inizializza Modello, Loss, Optimizer ---
    model = CAMLNet(n_way=N_WAY,
                    device=DEVICE,
                    elmes_embedding_dim=ELMES_EMBEDDING_DIM,
                    seq_model_heads=TRANSFORMER_HEADS,
                    seq_model_layers=TRANSFORMER_LAYERS).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Fase di Pre-Addestramento ---
    print("\n--- Starting Pre-training Phase ---")
    print(f"Training on {PRETRAIN_DATASET_NAME} for {NUM_EPOCHS_PRETRAIN} epochs...")
    for epoch in range(NUM_EPOCHS_PRETRAIN):
        print(f"\nPre-training Epoch {epoch+1}/{NUM_EPOCHS_PRETRAIN}")
        train_loss, train_acc = run_epoch(model, pretrain_loader, criterion, DEVICE, optimizer=optimizer, phase="Pre-Train")
        # Opzionale: si potrebbe valutare su un piccolo set di validazione CIFAR-100 qui
        # per monitorare l'overfitting durante il pre-training.

    print("\n--- Pre-training Finished ---")
    # Salva il modello pre-addestrato (opzionale ma consigliato)
    # torch.save(model.state_dict(), "caml_pretrained_cifar100.pth")
    # print("Pre-trained model state saved.")

    # --- Fase di Valutazione Finale (a pesi congelati) ---
    print("\n--- Starting Final Evaluation on Unseen Task ---")
    print(f"Evaluating on {TEST_UNSEEN_DATASET_NAME} with ALL weights frozen...")

    # Assicurati che il modello sia in modalità eval
    model.eval()

    # NON resettiamo l'ottimizzatore, semplicemente non lo usiamo.
    # run_epoch con optimizer=None eseguirà il test a pesi congelati.
    final_test_loss, final_test_acc = run_epoch(model, test_unseen_loader, criterion, DEVICE, optimizer=None, phase="Final Test (Frozen)")

    print("\n--- Evaluation Finished ---")
    print(f"Final Performance on {TEST_UNSEEN_DATASET_NAME} (Unseen):")
    print(f"  Accuracy: {final_test_acc:.2f}%")
    print("\nNOTE: This simulates the *flow* (pre-train -> freeze -> test unseen).")
    print("Actual CAML paper results depend on large-scale, multi-domain pre-training and likely a CLIP encoder.")