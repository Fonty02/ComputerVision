import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import time

# --- Configurazione ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {device}")

# Dimensione desiderata (ridurre se si esaurisce la memoria)
imsize = 512 if torch.cuda.is_available() else 128

# --- Funzioni Utili ---

# Caricamento e trasformazione immagine
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])

def image_loader(image_name):
    try:
        image = Image.open(image_name).convert('RGB')
        image = loader(image).unsqueeze(0) # con questa riga aggiungiamo una dimensione batch, inizialmente è (3,imsize,imsize) mentre ora è (1,3,imsize,imsize)
        return image.to(device, torch.float)
    except FileNotFoundError:
        print(f"Errore: Immagine '{image_name}' non trovata.")
        exit()
    except Exception as e:
        print(f"Errore durante il caricamento dell'immagine '{image_name}': {e}")
        exit()

# Visualizzazione/Salvataggio immagine
unloader = transforms.ToPILImage()

def imshow(tensor, title=None, save_path=None):
    image = tensor.cpu().clone().squeeze(0) # Rimuovi dimensione batch
    image = unloader(image)
    if save_path:
        try:
            image.save(save_path)
            print(f"Immagine salvata in: {save_path}")
        except Exception as e:
            print(f"Errore durante il salvataggio dell'immagine: {e}")
    else:
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

# Funzione per calcolare la matrice di Gram
def gram_matrix(input):
    batch_size, n_feature_maps, height, width = input.size()
    features = input.view(batch_size * n_feature_maps, height * width) # (N, C*H*W)
    G = torch.mm(features, features.t())
    return G.div(batch_size * n_feature_maps * height * width)

# Modulo di Normalizzazione (come prima)
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# --- Classe Principale del Modello di Style Transfer ---

class StyleTransferModel(nn.Module):
    def __init__(self, cnn, normalization_mean, normalization_std,
                 style_img, content_img,
                 content_layers, style_layers):
        super().__init__()
        self.cnn = copy.deepcopy(cnn) # Usiamo una copia del VGG per non alterare l'originale

        # Modulo di normalizzazione
        self.normalization = Normalization(normalization_mean, normalization_std).to(device)

        # Nomi dei layer per le loss
        self.content_layers = content_layers
        self.style_layers = style_layers

        # Variabili per memorizzare le loss (verranno aggiornate nel forward)
        self.content_loss = torch.tensor(0., device=device)
        self.style_loss = torch.tensor(0., device=device)

        # Calcoliamo *una volta sola* i target per le loss
        self.content_targets = {}
        self.style_targets = {}

        # Costruiamo un modello "sonda" temporaneo per calcolare i target
        # senza necessità di un forward completo ogni volta. Tanto le immagini style e content sono fisse, le passiamo solo una volta alla rete all'inizio e salviamo i risultati in self.content_targets e self.style_targets
        model = nn.Sequential(self.normalization)
        i = 0
        # Iteriamo attraverso i layer del modello VGG
        # e aggiungiamo i layer al nostro modello "sonda"
        # e calcoliamo i target per le loss
        # Questo ci permette di definire i layer di contenuto e stile
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False) # Usiamo ReLU out-of-place
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}' # Anche se VGG19 non ha BatchNorm di default
            else:
                 raise RuntimeError(f'Layer non riconosciuto: {layer.__class__.__name__}')

            model.add_module(name, layer)

            # Se è un layer di stile, calcola e salva la Gram matrix target
            if name in self.style_layers:
                with torch.no_grad(): # Non serve calcolare gradienti qui
                    target_feature = model(style_img).detach() #questa riga di codice calcola le feature del layer corrente per l'immagine di stile
                self.style_targets[name] = gram_matrix(target_feature)
                # print(f"Salvato target di stile per: {name}")

            # Se è un layer di contenuto, salva le feature target
            if name in self.content_layers:
                 with torch.no_grad():
                    target_feature = model(content_img).detach()
                 self.content_targets[name] = target_feature
                 # print(f"Salvato target di contenuto per: {name}")

        # Ora che abbiamo i target, non ci serve più il modello "sonda"
        # Teniamo solo i layer VGG originali e la normalizzazione
        # Rimuoviamo i layer dopo l'ultimo layer di loss richiesto per efficienza
        self.model_layers = nn.Sequential(self.normalization)
        last_layer_index = 0
        current_index = 0
        i = 0
        for layer in self.cnn.children():
            current_index += 1 # Indice del layer nel modello completo
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                 raise RuntimeError(f'Layer non riconosciuto: {layer.__class__.__name__}')

            self.model_layers.add_module(name, layer)

            if name in self.content_layers or name in self.style_layers:
                last_layer_index = current_index + 1 # +1 perché c'è anche la normalizzazione all'inizio

        self.model_layers = self.model_layers[:last_layer_index]
        # print("Modello finale per il forward:", self.model_layers)


    def forward(self, input_img):
        """
        Esegue il forward pass, calcolando le loss di contenuto e stile
        man mano che l'input attraversa i layer.
        """
        self.content_loss = torch.tensor(0., device=device, requires_grad=True) # Resettiamo a zero con grad
        self.style_loss = torch.tensor(0., device=device, requires_grad=True)   # Resettiamo a zero con grad
        x = input_img

        # Iteriamo attraverso i layer del nostro modello preparato
        for name, layer in self.model_layers.named_children():
            x = layer(x) # Passa l'input attraverso il layer

            # Calcola Content Loss se è un content layer
            if name in self.content_targets:
                # loss = F.mse_loss(x, self.content_targets[name])
                # self.content_loss = self.content_loss + loss # Somma la loss
                # Usare un clone per permettere la somma in-place senza errori di gradiente
                current_content_loss = F.mse_loss(x, self.content_targets[name])
                # print(f"Content Loss ({name}): {current_content_loss.item()}")
                self.content_loss = self.content_loss.clone() + current_content_loss


            # Calcola Style Loss se è uno style layer
            if name in self.style_targets:
                input_gram = gram_matrix(x)
                # loss = F.mse_loss(input_gram, self.style_targets[name])
                # self.style_loss = self.style_loss + loss # Somma la loss
                # Usare un clone per permettere la somma in-place senza errori di gradiente
                current_style_loss = F.mse_loss(input_gram, self.style_targets[name])
                # print(f"Style Loss ({name}): {current_style_loss.item()}")
                self.style_loss = self.style_loss.clone() + current_style_loss


        # Il forward pass non ha bisogno di restituire l'immagine,
        # il suo scopo è calcolare e accumulare le loss negli attributi
        # self.content_loss e self.style_loss.
        # Restituire le loss può essere utile per debug o logging esterno.
        # return self.content_loss, self.style_loss
        # Non ritorniamo nulla, le loss sono accessibili come attributi dell'istanza


# --- Funzione di Ottimizzazione (leggermente modificata) ---

def get_input_optimizer(input_img):
    optimizer = optim.AdamW([input_img.requires_grad_()]) # L'immagine input è il parametro
    return optimizer

def run_style_transfer(model, input_img, optimizer, num_steps=2000,
                       style_weight=10000, content_weight=100):
    """Esegue lo Style Transfer usando il modello fornito."""
    print('Ottimizzazione..')
    start_time = time.time()
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # Correggi i valori dell'immagine input (0-1)
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad() # Resettiamo i gradienti dell'ottimizzatore

            # Esegui il forward pass nel modello. Questo aggiornerà
            # model.content_loss e model.style_loss internamente.
            model(input_img)

            # Accedi alle loss calcolate dal modello
            style_score = model.style_loss * style_weight
            content_score = model.content_loss * content_weight

            # Loss totale
            loss = style_score + content_score

            # Calcola i gradienti per input_img rispetto alla loss totale
            # Verifica che la loss richieda gradiente prima di chiamare backward
            if loss.requires_grad:
                 loss.backward()
            else:
                 print("Attenzione: la loss non richiede gradienti. Qualcosa potrebbe essere errato.")


            run[0] += 1
            if run[0] % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Iterazione {run[0]}:")
                # Usiamo .item() per ottenere il valore numerico senza gradiente
                print(f'  Style Loss : {model.style_loss.item():4f} (Pesata: {style_score.item():4f})')
                print(f'  Content Loss: {model.content_loss.item():4f} (Pesata: {content_score.item():4f})')
                print(f'  Loss Totale : {loss.item():4f}')
                print(f'  Tempo trascorso: {elapsed:.2f}s')

            return loss # LBFGS ha bisogno che closure restituisca la loss

        optimizer.step(closure)

    # Ultima correzione
    with torch.no_grad():
        input_img.clamp_(0, 1)

    end_time = time.time()
    print(f"\nOttimizzazione completata in {end_time - start_time:.2f} secondi.")

    return input_img

# --- Esecuzione Principale ---

if __name__ == "__main__":
    # Percorsi immagini
    style_img_path = "style.jpg"
    content_img_path = "content.jpg"
    output_img_path = "output_oop.jpg" # Nome output diverso

    # Carica VGG19 pre-addestrato (solo features)
    cnn_base = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    # Disabilita gradienti per VGG, non lo addestriamo
    for param in cnn_base.parameters():
        param.requires_grad_(False)

    # Normalizzazione richiesta per VGG
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Layer desiderati per content e style loss
    content_layers_default = ['conv_4'] # Prova 'relu_4' o altri
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] # Prova 'relu_X'

    print("Caricamento immagini...")
    style_img = image_loader(style_img_path)
    content_img = image_loader(content_img_path)

    assert style_img.size() == content_img.size(), \
        "Le immagini di stile e contenuto devono avere la stessa dimensione"

    # Immagine iniziale (copia del contenuto o rumore)
    input_img = torch.randn(content_img.data.size(), device=device)

    # --- Crea l'istanza del modello ---
    print("Costruzione del modello di Style Transfer...")
    style_transfer_model = StyleTransferModel(
        cnn=cnn_base,
        normalization_mean=cnn_normalization_mean,
        normalization_std=cnn_normalization_std,
        style_img=style_img,
        content_img=content_img,
        content_layers=content_layers_default,
        style_layers=style_layers_default
    ).to(device).eval() # Metti il modello in modalità eval

    # Ottieni l'ottimizzatore per l'immagine di input
    optimizer = get_input_optimizer(input_img)

    # Esegui lo style transfer
    output = run_style_transfer(model=style_transfer_model,
                                input_img=input_img,
                                optimizer=optimizer)

    # Visualizza e salva il risultato
    plt.figure()
    imshow(output, title='Output Image (OOP)', save_path=output_img_path)
    # plt.ioff()
    # plt.show()